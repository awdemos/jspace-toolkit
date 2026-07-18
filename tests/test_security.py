"""Security-focused tests for path containment and cache integrity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jspace import JSpaceError
from jspace.utils import _validate_shape, get_cache_dir, load_lens_layer, save_lens_layer
from jspace.validation import validate_path, validate_workspace


class TestPathContainment:
    def test_validate_path_inside_workspace(self, tmp_path: Path) -> None:
        file = tmp_path / "corpus.json"
        file.write_text("{}")
        resolved = validate_path(file, tmp_path, must_exist=True, must_be_file=True)
        assert resolved == file.resolve()

    def test_validate_path_rejects_escape(self, tmp_path: Path) -> None:
        with pytest.raises(JSpaceError, match="outside"):
            validate_path("../escape.json", tmp_path)

    def test_validate_path_rejects_absolute_outside_workspace(self, tmp_path: Path) -> None:
        with pytest.raises(JSpaceError, match="outside"):
            validate_path("/etc/hosts", tmp_path)

    def test_validate_path_rejects_symlink_escape(self, tmp_path: Path) -> None:
        outside = tmp_path / ".." / "real_target.json"
        outside.write_text("{}")
        link = tmp_path / "symlink.json"
        link.symlink_to(outside)
        with pytest.raises(JSpaceError, match="outside"):
            validate_path(link, tmp_path, must_exist=True, must_be_file=True)

    def test_validate_workspace_rejects_missing(self, tmp_path: Path) -> None:
        with pytest.raises(JSpaceError):
            validate_workspace(tmp_path / "does-not-exist")


class TestLensCacheIntegrity:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        cache_dir = get_cache_dir(tmp_path, "fingerprint")
        matrix = np.random.rand(16, 16).astype(np.float32)
        save_lens_layer(cache_dir, 0, matrix)
        loaded = load_lens_layer(cache_dir, 0)
        np.testing.assert_array_equal(loaded, matrix)

    def test_tampered_checksum_fails(self, tmp_path: Path) -> None:
        cache_dir = get_cache_dir(tmp_path, "fingerprint")
        matrix = np.random.rand(8, 8).astype(np.float32)
        save_lens_layer(cache_dir, 0, matrix)
        safetensors_path = cache_dir / "J_0.safetensors"
        # Append a byte to corrupt the file; the SHA-256 check must fail.
        with safetensors_path.open("ab") as fh:
            fh.write(b"x")
        with pytest.raises(JSpaceError, match="Checksum mismatch"):
            load_lens_layer(cache_dir, 0)

    def test_missing_checksum_fails(self, tmp_path: Path) -> None:
        cache_dir = get_cache_dir(tmp_path, "fingerprint")
        matrix = np.random.rand(8, 8).astype(np.float32)
        save_lens_layer(cache_dir, 0, matrix)
        (cache_dir / "J_0.sha256").unlink()
        with pytest.raises(JSpaceError, match="Missing checksum"):
            load_lens_layer(cache_dir, 0)

    def test_invalid_shape_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(JSpaceError):
            _validate_shape((1_000_000_001,))
