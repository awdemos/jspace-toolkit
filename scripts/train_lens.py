"""CLI to train and cache J-Lens matrices."""

import argparse
import json
import sys

import torch

from jspace import JSpaceError
from jspace.jacobian_lens import train_jacobian_lens
from jspace.model_adapter import layer_indices, load_model
from jspace.utils import get_cache_dir, model_fingerprint
from jspace.validation import validate_path, validate_workspace


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
        print(f"Invalid path: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    try:
        with corpus_path.open() as f:
            prompts = json.load(f)
    except FileNotFoundError as exc:
        print(f"Corpus file not found: {args.corpus}", file=sys.stderr)
        raise SystemExit(1) from exc
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON corpus at {args.corpus}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

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

    print(
        f"Training J-Lens for {args.model} target_layer={target_layer} frozen_qk={args.frozen_qk}"
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
        print(f"J-Space training failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Saved J_l to {cache_dir}")


if __name__ == "__main__":
    main()
