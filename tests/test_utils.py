"""Tests for jspace.utils cache IO, fingerprints, and revision resolution."""

import numpy as np
import pytest

from jspace.utils import (
    load_lens_layer,
    resolve_model_revision,
    save_lens_layer,
)


def test_load_lens_layer_preserves_saved_dtype(tmp_path):
    matrix = np.random.rand(8, 8).astype(np.float64)
    save_lens_layer(tmp_path, 0, matrix)
    loaded = load_lens_layer(tmp_path, 0)
    assert isinstance(loaded, np.ndarray)
    assert loaded.dtype == np.float64
    assert np.array_equal(loaded, matrix)


def test_load_lens_layer_returns_defensive_copy(tmp_path):
    matrix = np.random.rand(8, 8).astype(np.float32)
    save_lens_layer(tmp_path, 0, matrix)
    loaded = load_lens_layer(tmp_path, 0)
    loaded[0, 0] = -123.0
    reloaded = load_lens_layer(tmp_path, 0)
    assert reloaded[0, 0] == pytest.approx(matrix[0, 0])


def test_resolve_model_revision_passes_through_explicit_revision():
    assert resolve_model_revision("gpt2", "abc123") == "abc123"


def test_resolve_model_revision_returns_none_for_local_dir(tmp_path, monkeypatch):
    def _boom(repo_id):  # pragma: no cover - must never be called
        raise AssertionError("model_info must not be hit for local paths")

    monkeypatch.setattr("huggingface_hub.model_info", _boom)
    assert resolve_model_revision(str(tmp_path), None) is None


def test_resolve_model_revision_returns_none_on_hub_failure(monkeypatch):
    def _raise(repo_id):
        raise RuntimeError("network is unreachable")

    monkeypatch.setattr("huggingface_hub.model_info", _raise)
    assert resolve_model_revision("bogus-repo-that-cannot-resolve", None) is None


def test_resolve_model_revision_returns_hub_sha(monkeypatch):
    class _Info:
        sha = "cafe0123beef"

    def _fake_model_info(repo_id):
        assert repo_id == "gpt2"
        return _Info()

    monkeypatch.setattr("huggingface_hub.model_info", _fake_model_info)
    assert resolve_model_revision("gpt2", None) == "cafe0123beef"
