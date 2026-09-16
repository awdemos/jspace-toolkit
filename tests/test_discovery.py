"""Tests for workspace discovery: CKA/boundaries and attention-mask handling."""

import numpy as np
import pytest
import torch

from jspace import JSpaceError
from jspace.discovery import (
    centered_kernel_alignment,
    compute_discovery_metrics,
    infer_workspace_boundaries,
)
from jspace.model_adapter import (
    get_unembedding_matrix,
    layer_indices,
    load_model,
    normalize_fn,
)


def test_cka_and_boundary_inference():
    X = torch.randn(10, 64)
    Y = torch.randn(10, 64)
    cka = centered_kernel_alignment(X, Y)
    assert 0.0 <= cka <= 1.0
    metrics = {
        "cka_block": np.random.rand(10, 10),
        "kurtosis": np.random.rand(10),
        "accuracy": np.linspace(0.1, 0.95, 10),
        "autocorr": np.random.rand(10),
    }
    start, end = infer_workspace_boundaries(metrics)
    assert 0 <= start <= end < 10


def _tiny_jacobians(model) -> dict[int, np.ndarray]:
    """Small random J_l matrices, one per layer (metric values are irrelevant)."""
    d = model.config.n_embd
    rng = np.random.default_rng(0)
    return {
        layer: (0.01 * np.eye(d) + rng.normal(scale=0.01, size=(d, d))).astype(np.float32)
        for layer in layer_indices(model)
    }


@pytest.fixture()
def tiny_gpt2():
    return load_model("sshleifer/tiny-gpt2", torch.device("cpu"), torch.float32)


def _pad(ids: torch.Tensor, tokenizer, n_pads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Append ``n_pads`` trailing pad columns plus the matching attention mask."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    padded = torch.cat(
        [ids, torch.full((ids.shape[0], n_pads), pad_id, dtype=ids.dtype)], dim=1
    )
    mask = torch.cat(
        [torch.ones_like(ids), torch.zeros(ids.shape[0], n_pads, dtype=ids.dtype)], dim=1
    )
    return padded, mask


def _metrics(model, tokenizer, corpus, attention_mask=None):
    return compute_discovery_metrics(
        model,
        tokenizer,
        _tiny_jacobians(model),
        corpus,
        get_unembedding_matrix(model),
        normalize_fn(model),
        attention_mask=attention_mask,
    )


def test_masked_metrics_ignore_trailing_pads(tiny_gpt2):
    """Metrics with a mask must be identical for a row with any number of pads."""
    model, tokenizer = tiny_gpt2
    ids = tokenizer("The cat sat on the mat.", return_tensors="pt")["input_ids"]

    reference = _metrics(model, tokenizer, ids)
    padded2, mask2 = _pad(ids, tokenizer, 2)
    padded5, mask5 = _pad(ids, tokenizer, 5)
    m2 = _metrics(model, tokenizer, padded2, attention_mask=mask2)
    m5 = _metrics(model, tokenizer, padded5, attention_mask=mask5)

    for key in ("kurtosis", "accuracy", "autocorr"):
        assert np.allclose(reference[key], m2[key], atol=1e-5), key
        assert np.allclose(reference[key], m5[key], atol=1e-5), key

    # Without the mask the pad positions pollute at least one metric.
    unmasked = _metrics(model, tokenizer, padded2)
    assert any(
        not np.allclose(reference[key], unmasked[key], atol=1e-5)
        for key in ("kurtosis", "accuracy", "autocorr")
    )


def test_discovery_metrics_forwards_attention_mask(tiny_gpt2, monkeypatch):
    model, tokenizer = tiny_gpt2
    ids = tokenizer("The cat sat.", return_tensors="pt")["input_ids"]
    mask = torch.ones_like(ids)
    captured = {}
    orig_forward = model.forward

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return orig_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", spy)
    _metrics(model, tokenizer, ids, attention_mask=mask)

    assert "attention_mask" in captured
    assert torch.equal(captured["attention_mask"], mask)


def test_discovery_metrics_rejects_mismatched_mask(tiny_gpt2):
    model, tokenizer = tiny_gpt2
    ids = tokenizer("The cat sat.", return_tensors="pt")["input_ids"]
    bad_mask = torch.ones(1, ids.shape[1] + 1, dtype=torch.long)
    with pytest.raises(JSpaceError, match="attention_mask"):
        _metrics(model, tokenizer, ids, attention_mask=bad_mask)
