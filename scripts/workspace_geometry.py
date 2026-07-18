"""CLI to compute the CKA workspace-geometry plot for a decoder-only model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from jspace import JSpaceError
from jspace.discovery import (
    centered_kernel_alignment,
    compute_discovery_metrics,
    infer_workspace_boundaries,
)
from jspace.jacobian_lens import train_jacobian_lens
from jspace.model_adapter import get_unembedding_matrix, layer_indices, load_model, normalize_fn
from jspace.utils import get_cache_dir, model_fingerprint

DEFAULT_PROBE_COUNT = 4096


def _parse_probe_ids(probe_ids_str: str, vocab_size: int) -> list[int]:
    """Parse a comma-separated list of probe token ids."""
    ids = [int(x.strip()) for x in probe_ids_str.split(",")]
    for token_id in ids:
        if not 0 <= token_id < vocab_size:
            raise JSpaceError(f"probe token_id {token_id} out of range (vocab_size={vocab_size})")
    return ids


def _default_probe_ids(tokenizer, n: int) -> list[int]:
    """Return the first n non-special token ids as a basic probe set."""
    special = set(tokenizer.all_special_ids)
    ids = [i for i in range(tokenizer.vocab_size) if i not in special]
    return ids[:n]


def _load_corpus(path: Path, tokenizer, max_positions: int) -> torch.Tensor:
    """Load a JSON corpus and tokenize it."""
    with path.open() as f:
        prompts = json.load(f)
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_positions,
    )
    return enc["input_ids"]


def build_V_l(
    J_l: np.ndarray,
    W_U: torch.Tensor,
    probe_ids: torch.Tensor,
    gamma: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build centered token geometry V_l = W_U[probe_ids] @ J_l.T.

    Following the article notation: V_l = (W_U[probe_ids] * gamma) @ J_l.T.
    """
    device = W_U.device
    W_probe = W_U[probe_ids].to(device, dtype)
    J = torch.from_numpy(J_l).to(device, dtype)
    V = gamma * torch.matmul(W_probe, J.t())
    return V - V.mean(dim=0, keepdim=True)


def compute_cka_block(V_by_layer: dict[int, torch.Tensor]) -> np.ndarray:
    """Compute layer-by-layer CKA similarity matrix."""
    layers = sorted(V_by_layer.keys())
    n = len(layers)
    cka = np.zeros((n, n), dtype=np.float64)
    for i, li in enumerate(tqdm(layers, desc="CKA block")):
        for j, lj in enumerate(layers):
            if j < i:
                cka[i, j] = cka[j, i]
            else:
                cka[i, j] = centered_kernel_alignment(
                    V_by_layer[li].float(), V_by_layer[lj].float()
                )
    return cka


def _plot_cka_block(
    cka: np.ndarray,
    boundaries: tuple[int, int],
    layers: list[int],
    model_name: str,
    out_path: Path,
) -> None:
    """Render a PNG heatmap of the CKA block with workspace overlay."""
    matplotlib.use("Agg")
    start, end = boundaries
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cka, cmap="viridis", vmin=0.0, vmax=1.0)
    if 0 <= start < len(layers) and 0 <= end < len(layers):
        rect = plt.Rectangle(
            (start - 0.5, start - 0.5),
            end - start + 1,
            end - start + 1,
            linewidth=3,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title(f"J-Lens workspace geometry (CKA) — {model_name}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CKA workspace geometry for a model")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--corpus", required=True, help="Path to JSON corpus file")
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument("--cache-dir", default="lens_cache", help="Directory to cache J_l")
    parser.add_argument("--output-dir", default="workspace_out", help="Directory for outputs")
    parser.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-positions", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dim-chunk", type=int, default=16)
    parser.add_argument("--probe-ids", default=None, help="Comma-separated token ids to probe")
    parser.add_argument(
        "--n-probes", type=int, default=DEFAULT_PROBE_COUNT, help="Number of default probe tokens"
    )
    parser.add_argument("--frozen-qk", action="store_true")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model, device, dtype, token=args.hf_token)
    corpus = _load_corpus(Path(args.corpus), tokenizer, args.max_positions)
    layers = layer_indices(model)
    target_layer = args.target_layer if args.target_layer is not None else layers[-2]

    cache_dir = get_cache_dir(
        args.cache_dir,
        model_fingerprint(
            args.model,
            target_layer,
            args.frozen_qk,
            max_positions=args.max_positions,
            dtype=args.dtype,
            output_dim_chunk=args.output_dim_chunk,
        ),
    )

    print(f"Training/caching J-Lens for {args.model} target_layer={target_layer}")
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
    )

    W_U = get_unembedding_matrix(model)
    vocab_size = W_U.shape[0]
    if args.probe_ids is not None:
        probe_ids_list = _parse_probe_ids(args.probe_ids, vocab_size)
    else:
        probe_ids_list = _default_probe_ids(tokenizer, args.n_probes)
    probe_ids = torch.tensor(probe_ids_list, dtype=torch.long)

    print(f"Building token geometry for {len(probe_ids_list)} probe tokens")
    V_by_layer: dict[int, torch.Tensor] = {}
    for layer in tqdm(sorted(J.keys()), desc="Building V_l"):
        V_by_layer[layer] = build_V_l(
            J[layer],
            W_U,
            probe_ids,
            gamma=1.0,
            dtype=torch.float32,
        )

    print("Computing CKA block matrix")
    cka = compute_cka_block(V_by_layer)
    layers_with_v = sorted(V_by_layer.keys())

    print("Computing discovery metrics for workspace boundary inference")
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

    png_path = output_dir / "cka_block.png"
    _plot_cka_block(cka, (start, end), layers_with_v, args.model, png_path)

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

    print(f"Wrote {png_path}")
    print(f"Wrote {json_path}")
    print(f"Workspace boundaries: layer {start} to {end}")


if __name__ == "__main__":
    main()
