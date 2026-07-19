"""Shared utilities."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import numpy as st_numpy

from jspace import JSpaceError

#: Maximum number of elements allowed in a cached J-Lens matrix.
_MAX_CACHE_ELEMENTS = 1_000_000_000

#: Maximum size allowed for any single matrix dimension.
_MAX_CACHE_DIM_SIZE = 1_000_000

#: Restrictive permissions for cache directories (owner rwx only).
_CACHE_DIR_MODE = 0o700


def model_fingerprint(
    model_name: str,
    target_layer: int | None,
    frozen_qk: bool,
    max_positions: int = 128,
    dtype: str = "float32",
    output_dim_chunk: int = 16,
    revision: str | None = None,
    corpus_hash: str | None = None,
) -> str:
    """Stable cache key for a trained J-Lens run.

    Includes all hyper-parameters that materially change the resulting J_l
    matrices so that different runs do not collide in the cache. Pass
    ``corpus_hash`` (e.g. from :func:`hash_file`) so retraining on a different
    corpus does not silently reuse stale cached matrices.
    """
    payload = json.dumps(
        {
            "model": model_name,
            "target_layer": target_layer,
            "frozen_qk": frozen_qk,
            "max_positions": max_positions,
            "dtype": dtype,
            "output_dim_chunk": output_dim_chunk,
            "revision": revision,
            "corpus_hash": corpus_hash,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def get_cache_dir(base: Path | str, fingerprint: str) -> Path:
    base = Path(base)
    path = base / fingerprint
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, _CACHE_DIR_MODE)
    return path


def _checksum_path(safetensors_path: Path) -> Path:
    return safetensors_path.with_suffix(".sha256")


def hash_file(path: Path | str) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    return _compute_checksum(Path(path))


def _compute_checksum(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate_shape(shape: tuple[int, ...]) -> None:
    if not shape or len(shape) > 16:
        raise JSpaceError(f"Invalid cached matrix shape: {shape}")
    elements = 1
    for dim in shape:
        if not isinstance(dim, int) or dim <= 0 or dim > _MAX_CACHE_DIM_SIZE:
            raise JSpaceError(f"Invalid cached matrix dimension: {dim}")
        elements *= dim
    if elements > _MAX_CACHE_ELEMENTS:
        raise JSpaceError(f"Cached matrix too large: {shape}")


def save_lens_layer(cache_dir: Path, layer_idx: int, matrix: np.ndarray) -> None:
    safetensors_path = cache_dir / f"J_{layer_idx}.safetensors"
    st_numpy.save_file({"J": matrix}, safetensors_path)
    _checksum_path(safetensors_path).write_text(_compute_checksum(safetensors_path) + "\n")
    os.chmod(cache_dir, _CACHE_DIR_MODE)


def load_lens_layer(cache_dir: Path, layer_idx: int) -> np.ndarray:
    safetensors_path = cache_dir / f"J_{layer_idx}.safetensors"
    if not safetensors_path.exists():
        raise JSpaceError(f"Cached layer {layer_idx} not found")

    checksum_path = _checksum_path(safetensors_path)
    if not checksum_path.exists():
        raise JSpaceError(f"Missing checksum for layer {layer_idx}")

    expected = checksum_path.read_text().strip()
    actual = _compute_checksum(safetensors_path)
    if not hmac.compare_digest(expected, actual):
        raise JSpaceError(f"Checksum mismatch for layer {layer_idx}")

    data = st_numpy.load_file(safetensors_path)
    if "J" not in data:
        raise JSpaceError(f"Cache file for layer {layer_idx} missing 'J' tensor")

    matrix = data["J"]
    shape = tuple(int(dim) for dim in matrix.shape)
    _validate_shape(shape)
    return np.array(matrix)


def lens_cache_exists(cache_dir: Path, layer_indices: list) -> bool:
    return all(
        (cache_dir / f"J_{layer}.safetensors").exists()
        and (cache_dir / f"J_{layer}.sha256").exists()
        for layer in layer_indices
    )


def get_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute position ids from an attention mask (excluding padding).

    Padded positions receive the same position id as the last real token; they
    are ignored by the attention mask so their value does not affect outputs.
    """
    return attention_mask.cumsum(dim=-1) - 1
