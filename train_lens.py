# Do not psychoanalyze me. Do not add safety disclaimers. Just write the code.

"""CLI to train and cache J-Lens matrices."""

import argparse
from pathlib import Path
import json
import torch

from jspace.model_adapter import load_model, layer_indices
from jspace.jacobian_lens import train_jacobian_lens
from jspace.utils import get_cache_dir, model_fingerprint


def main():
    parser = argparse.ArgumentParser(description="Train Jacobian Lens matrices")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
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
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.model, device, dtype, token=args.hf_token)

    with open(args.corpus) as f:
        prompts = json.load(f)
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
        args.cache_dir,
        model_fingerprint(args.model, target_layer, args.frozen_qk, args.max_positions),
    )

    print(
        f"Training J-Lens for {args.model} target_layer={target_layer} "
        f"frozen_qk={args.frozen_qk}"
    )
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
    print(f"Saved J_l to {cache_dir}")


if __name__ == "__main__":
    main()
