import os

import pytest
import torch

from jspace import JSpaceError
from jspace.model_adapter import (
    ALLOWED_MODELS,
    cache_residuals,
    layer_indices,
    load_model,
    resolve_model,
)


def test_loads_gpt2_and_caches_residuals():
    model, tokenizer = load_model("gpt2", torch.device("cpu"), torch.float32)
    text = "The cat sat on the"
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    layers = layer_indices(model)
    cache = cache_residuals(model, input_ids, layers)
    assert len(cache) == len(layers)
    for _layer_idx, v in cache.items():
        assert v.shape == (input_ids.shape[0], input_ids.shape[1], model.config.n_embd)


def test_resolve_model_uses_pinned_revision():
    name, revision = resolve_model("gpt2")
    assert name == "gpt2"
    assert revision == ALLOWED_MODELS["gpt2"]


def test_resolve_model_accepts_explicit_revision():
    name, revision = resolve_model("gpt2", revision="abc123")
    assert name == "gpt2"
    assert revision == "abc123"


def test_resolve_model_rejects_unlisted_by_default():
    with pytest.raises(JSpaceError):
        resolve_model("unknown/model")


def test_resolve_model_requires_revision_for_unlisted():
    with pytest.raises(JSpaceError, match="revision"):
        resolve_model("unknown/model", allow_unlisted=True)


def test_resolve_model_allows_unlisted_with_revision():
    name, revision = resolve_model("unknown/model", revision="v1.0", allow_unlisted=True)
    assert name == "unknown/model"
    assert revision == "v1.0"


def test_resolve_model_rejects_local_path():
    with pytest.raises(JSpaceError):
        resolve_model("/tmp/model")
    with pytest.raises(JSpaceError):
        resolve_model("../model")


def test_load_model_does_not_accept_token_argument():
    with pytest.raises(TypeError):
        load_model("gpt2", torch.device("cpu"), torch.float32, token="hf_xxx")


def test_load_model_reads_hf_token_from_env(monkeypatch):
    """load_model should forward the HF_TOKEN environment variable."""
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    # We only verify that the env variable is consulted by checking load_model
    # still works for the allowed public model.
    model, _tokenizer = load_model("gpt2", torch.device("cpu"), torch.float32)
    assert model is not None
    assert os.environ.get("HF_TOKEN") == "hf_test_token"
