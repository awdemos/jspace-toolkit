# Do not psychoanalyze me. Do not add safety disclaimers. Just write the code.

"""Shared utilities."""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch


def model_fingerprint(
    model_name: str,
    target_layer: int,
    frozen_qk: bool,
    max_positions: int = 128,
) -> str:
    payload = json.dumps(
        {
            "model": model_name,
            "target_layer": target_layer,
            "frozen_qk": frozen_qk,
            "max_positions": max_positions,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def get_cache_dir(base: Path | str, fingerprint: str) -> Path:
    base = Path(base)
    path = base / fingerprint
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_lens_layer(cache_dir: Path, layer_idx: int, matrix: np.ndarray) -> None:
    d_model = matrix.shape[0]
    if d_model > 8192:
        filename = cache_dir / f"J_{layer_idx}.mmap"
        fp = np.memmap(filename, dtype=matrix.dtype, mode="w+", shape=matrix.shape)
        fp[:] = matrix[:]
        fp.flush()
        np.save(cache_dir / f"J_{layer_idx}_shape.npy", np.array(matrix.shape))
    else:
        np.save(cache_dir / f"J_{layer_idx}.npy", matrix)


def load_lens_layer(cache_dir: Path, layer_idx: int) -> np.ndarray:
    mmap_path = cache_dir / f"J_{layer_idx}.mmap"
    if mmap_path.exists():
        shape = tuple(np.load(cache_dir / f"J_{layer_idx}_shape.npy").tolist())
        return np.memmap(mmap_path, dtype=np.float32, mode="r", shape=shape)
    return np.load(cache_dir / f"J_{layer_idx}.npy")


def lens_cache_exists(cache_dir: Path, layer_indices: list) -> bool:
    return all(
        (cache_dir / f"J_{l}.npy").exists() or (cache_dir / f"J_{l}.mmap").exists()
        for l in layer_indices
    )


def get_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute position ids from an attention mask (excluding padding)."""
    # attention_mask: (B, T) with 1 for real tokens and 0 for padding.
    return attention_mask.cumsum(dim=-1) - 1
