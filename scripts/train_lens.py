"""CLI to train and cache J-Lens matrices."""

import argparse
import json

import torch
from rich.console import Console
from rich.panel import Panel

from jspace import JSpaceError
from jspace.jacobian_lens import train_jacobian_lens
from jspace.model_adapter import layer_indices, load_model
from jspace.utils import get_cache_dir, model_fingerprint
from jspace.validation import validate_path, validate_workspace
from jspace.viz import config_table, get_console, header_panel

err_console = Console(stderr=True, highlight=False)


def _fail(message: str) -> SystemExit:
    """Print a styled error to stderr and return an exit-1 SystemExit."""
    err_console.print(Panel(f"[bold red]{message}[/]", border_style="red", title="Error"))
    return SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Train Jacobian Lens matrices")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--model-revision", default=None, help="Pinned revision")
    parser.add_argument(
        "--allow-unlisted-model",
        action="store_true",
        help="Opt-in to loading a model not in the allowlist",
    )
    parser.add_argument("--corpus", required=True, help="Path to JSON corpus file")
    parser.add_argument(
        "--target-layer",
        type=int,
        default=None,
        help="Target layer; default = penultimate (retained for API compatibility)",
    )
    parser.add_argument(
        "--frozen-qk",
        action="store_true",
        help="Stop gradient through Q/K projections",
    )
    parser.add_argument("--cache-dir", default="lens_cache", help="Directory to cache J_l")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--max-positions", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dim-chunk", type=int, default=16)
    parser.add_argument(
        "--workspace",
        default=".",
        help="Root directory that --corpus and --cache-dir must stay inside",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        workspace = validate_workspace(args.workspace)
        corpus_path = validate_path(args.corpus, workspace, must_exist=True, must_be_file=True)
        cache_base = validate_path(args.cache_dir, workspace)
    except JSpaceError as exc:
        raise _fail(f"Invalid path: {exc}") from exc

    try:
        with corpus_path.open() as f:
            prompts = json.load(f)
    except FileNotFoundError as exc:
        raise _fail(f"Corpus file not found: {args.corpus}") from exc
    except json.JSONDecodeError as exc:
        raise _fail(f"Invalid JSON corpus at {args.corpus}: {exc}") from exc

    model, tokenizer = load_model(
        args.model,
        device,
        dtype,
        revision=args.model_revision,
        allow_unlisted=args.allow_unlisted_model,
    )

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_positions,
    )
    corpus_ids = enc["input_ids"]

    layers = layer_indices(model)
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
        ),
    )

    console = get_console()
    console.print(header_panel("J-Lens Training", "Jacobian Lens matrices for mid-layer readout"))
    console.print(
        config_table(
            {
                "model": args.model,
                "target layer": target_layer,
                "frozen qk": args.frozen_qk,
                "dtype": args.dtype,
                "device": device,
                "prompts": len(prompts),
                "max positions": args.max_positions,
            }
        )
    )
    try:
        train_jacobian_lens(
            model,
            corpus_ids,
            target_layer=target_layer,
            cache_dir=cache_dir,
            dtype=dtype,
            max_positions=args.max_positions,
            batch_size=args.batch_size,
            output_dim_chunk=args.output_dim_chunk,
            frozen_qk=args.frozen_qk,
        )
    except JSpaceError as exc:
        raise _fail(f"J-Space training failed: {exc}") from exc
    console.print(
        Panel(
            f"[bold green]✓ Training complete[/]\n\nSaved J_l to [bold]{cache_dir}[/]",
            border_style="green",
            title="Done",
        )
    )


if __name__ == "__main__":
    main()
