"""CLI to compute the CKA workspace-geometry plot for a decoder-only model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel

from jspace import JSpaceError
from jspace.discovery import (
    centered_kernel_alignment,
    compute_discovery_metrics,
    infer_workspace_boundaries,
)
from jspace.jacobian_lens import train_jacobian_lens
from jspace.model_adapter import get_unembedding_matrix, layer_indices, load_model, normalize_fn
from jspace.utils import get_cache_dir, hash_file, model_fingerprint
from jspace.validation import validate_path, validate_workspace
from jspace.viz import (
    config_table,
    get_console,
    header_panel,
    jl_track,
    metrics_table,
    plot_cka_block,
    plot_layer_metrics,
    render_html_report,
)

matplotlib.use("Agg")  # headless rendering for CLI use

err_console = Console(stderr=True, highlight=False)


def _fail(message: str) -> SystemExit:
    """Print a styled error to stderr and return an exit-1 SystemExit."""
    err_console.print(Panel(f"[bold red]{message}[/]", border_style="red", title="Error"))
    return SystemExit(1)


DEFAULT_PROBE_COUNT = 4096


def _parse_probe_ids(probe_ids_str: str, vocab_size: int) -> list[int]:
    """Parse a comma-separated list of probe token ids."""
    try:
        ids = [int(x.strip()) for x in probe_ids_str.split(",")]
    except ValueError as exc:
        raise JSpaceError(
            f"invalid --probe-ids {probe_ids_str!r}: expected comma-separated integers"
        ) from exc
    for token_id in ids:
        if not 0 <= token_id < vocab_size:
            raise JSpaceError(f"probe token_id {token_id} out of range (vocab_size={vocab_size})")
    return ids


def _default_probe_ids(tokenizer, n: int) -> list[int]:
    """Return the first n non-special token ids as a basic probe set."""
    special = set(tokenizer.all_special_ids)
    ids = [i for i in range(tokenizer.vocab_size) if i not in special]
    return ids[:n]


def _load_corpus(path: Path, tokenizer, max_positions: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a JSON corpus and tokenize it; returns (input_ids, attention_mask)."""
    with path.open() as f:
        prompts = json.load(f)
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_positions,
    )
    return enc["input_ids"], enc["attention_mask"]


def build_V_l(
    J_l: np.ndarray,
    W_U: torch.Tensor,
    probe_ids: torch.Tensor,
    gamma: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build centered token geometry V_l = W_U[probe_ids] @ J_l.

    Following the article notation, the J-lens vectors are the rows of
    W_U @ J_l: V_l = (W_U[probe_ids] * gamma) @ J_l.
    """
    device = W_U.device
    W_probe = W_U[probe_ids].to(device, dtype)
    J = torch.from_numpy(J_l).to(device, dtype)
    V = gamma * torch.matmul(W_probe, J)
    return V - V.mean(dim=0, keepdim=True)


def compute_cka_block(V_by_layer: dict[int, torch.Tensor]) -> np.ndarray:
    """Compute layer-by-layer CKA similarity matrix."""
    layers = sorted(V_by_layer.keys())
    n = len(layers)
    cka = np.zeros((n, n), dtype=np.float64)
    for i, li in enumerate(jl_track(layers, "CKA block")):
        for j, lj in enumerate(layers):
            if j < i:
                cka[i, j] = cka[j, i]
            else:
                cka[i, j] = centered_kernel_alignment(
                    V_by_layer[li].float(), V_by_layer[lj].float()
                )
    return cka


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CKA workspace geometry for a model")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--model-revision", default=None, help="Pinned revision")
    parser.add_argument(
        "--allow-unlisted-model",
        action="store_true",
        help="Opt-in to loading a model not in the allowlist",
    )
    parser.add_argument("--corpus", required=True, help="Path to JSON corpus file")
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument("--cache-dir", default="lens_cache", help="Directory to cache J_l")
    parser.add_argument("--output-dir", default="workspace_out", help="Directory for outputs")
    parser.add_argument(
        "--workspace",
        default=".",
        help="Root directory that --corpus, --cache-dir, and --output-dir must stay inside",
    )
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-positions", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dim-chunk", type=int, default=16)
    parser.add_argument("--probe-ids", default=None, help="Comma-separated token ids to probe")
    parser.add_argument(
        "--n-probes", type=int, default=DEFAULT_PROBE_COUNT, help="Number of default probe tokens"
    )
    parser.add_argument("--frozen-qk", action="store_true")
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing the self-contained HTML report",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        workspace = validate_workspace(args.workspace)
        corpus_path = validate_path(args.corpus, workspace, must_exist=True, must_be_file=True)
        cache_base = validate_path(args.cache_dir, workspace)
        output_dir = validate_path(args.output_dir, workspace)
        output_dir.mkdir(parents=True, exist_ok=True)
    except JSpaceError as exc:
        raise _fail(f"Invalid path: {exc}") from exc

    model, tokenizer = load_model(
        args.model,
        device,
        dtype,
        revision=args.model_revision,
        allow_unlisted=args.allow_unlisted_model,
    )
    corpus, attn_mask = _load_corpus(corpus_path, tokenizer, args.max_positions)
    layers = layer_indices(model)
    if args.target_layer is None and len(layers) < 2:
        raise _fail(
            f"{args.model} has {len(layers)} layer(s); "
            "need at least 2 to pick a penultimate target layer"
        )
    target_layer = args.target_layer if args.target_layer is not None else layers[-2]

    cache_dir = get_cache_dir(
        cache_base,
        model_fingerprint(
            args.model,
            target_layer,
            args.frozen_qk,
            max_positions=args.max_positions,
            dtype=args.dtype,
            output_dim_chunk=args.output_dim_chunk,
            revision=args.model_revision,
            corpus_hash=hash_file(corpus_path),
        ),
    )

    console = get_console()
    console.print(header_panel("Workspace Geometry", "CKA layer geometry and workspace discovery"))
    console.print(
        config_table(
            {
                "model": args.model,
                "target layer": target_layer,
                "dtype": args.dtype,
                "device": device,
                "max positions": args.max_positions,
            }
        )
    )
    J = train_jacobian_lens(
        model,
        corpus,
        target_layer=target_layer,
        cache_dir=cache_dir,
        dtype=dtype,
        max_positions=args.max_positions,
        batch_size=args.batch_size,
        output_dim_chunk=args.output_dim_chunk,
        frozen_qk=args.frozen_qk,
        attention_mask=attn_mask,
    )

    W_U = get_unembedding_matrix(model)
    vocab_size = W_U.shape[0]
    try:
        if args.probe_ids is not None:
            probe_ids_list = _parse_probe_ids(args.probe_ids, vocab_size)
        else:
            probe_ids_list = _default_probe_ids(tokenizer, args.n_probes)
    except JSpaceError as exc:
        raise _fail(str(exc)) from exc
    probe_ids = torch.tensor(probe_ids_list, dtype=torch.long)

    console.print(f"[bold]Building token geometry for[/] {len(probe_ids_list)} probe tokens")
    V_by_layer: dict[int, torch.Tensor] = {}
    for layer in jl_track(sorted(J.keys()), "Building V_l"):
        V_by_layer[layer] = build_V_l(
            J[layer],
            W_U,
            probe_ids,
            gamma=1.0,
            dtype=torch.float32,
        )

    cka = compute_cka_block(V_by_layer)
    layers_with_v = sorted(V_by_layer.keys())

    discovery_metrics = compute_discovery_metrics(
        model,
        tokenizer,
        J,
        corpus,
        W_U,
        normalize_fn(model),
        layers=layers_with_v,
        probe_ids=probe_ids,
    )
    start, end = infer_workspace_boundaries(discovery_metrics)

    console.print(metrics_table(layers_with_v, discovery_metrics, (start, end)))
    console.print(
        Panel(
            f"[bold #f0b72f]Workspace band: layers {start} – {end}[/]",
            border_style="#f0b72f",
            expand=False,
        )
    )

    png_path = output_dir / "cka_block.png"
    fig = plot_cka_block(cka, (start, end), layers_with_v, args.model, png_path)
    plt.close(fig)

    metrics_png_path = output_dir / "layer_metrics.png"
    fig = plot_layer_metrics(discovery_metrics, (start, end), layers_with_v, metrics_png_path)
    plt.close(fig)

    metrics = {
        "model": args.model,
        "target_layer": target_layer,
        "n_layers": len(layers_with_v),
        "workspace_start": int(start),
        "workspace_end": int(end),
        "n_probe_tokens": len(probe_ids_list),
        "mean_cka": float(cka.mean()),
        "max_cka": float(cka.max()),
        "min_cka": float(cka.min()),
        "cka_block": cka.tolist(),
        "kurtosis": discovery_metrics["kurtosis"].tolist(),
        "accuracy": discovery_metrics["accuracy"].tolist(),
        "autocorr": discovery_metrics["autocorr"].tolist(),
    }
    json_path = output_dir / "metrics.json"
    with json_path.open("w") as f:
        json.dump(metrics, f)

    artifacts = [png_path, metrics_png_path, json_path]
    if not args.no_report:
        report_path = output_dir / "report.html"
        render_html_report(
            args.model,
            {
                "model": args.model,
                "target_layer": target_layer,
                "dtype": args.dtype,
                "max_positions": args.max_positions,
                "frozen_qk": args.frozen_qk,
                "n_probe_tokens": len(probe_ids_list),
            },
            metrics,
            (start, end),
            layers_with_v,
            {
                "CKA layer-by-layer similarity": png_path,
                "Per-layer discovery metrics": metrics_png_path,
            },
            report_path,
        )
        artifacts.append(report_path)

    listing = "\n".join(f"[bold]{p}[/]" for p in artifacts)
    console.print(
        Panel(
            f"[bold green]✓ Analysis complete[/]\n\n{listing}",
            border_style="green",
            title="Artifacts",
        )
    )


if __name__ == "__main__":
    main()
