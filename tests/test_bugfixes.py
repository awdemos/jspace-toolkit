"""Regression tests for bug fixes (see commit history)."""

import numpy as np
import pytest
import torch

from jspace import JSpaceError
from jspace.decomposition import _resolve_V, jspace_occupancy
from jspace.interventions import apply_intervention
from jspace.jacobian_lens import _attach_frozen_qk_hooks, train_jacobian_lens
from jspace.model_adapter import layer_indices, load_model
from jspace.utils import model_fingerprint


def test_resolve_V_uses_j_lens_vector_geometry():
    """V must be W_U @ J_l (rows = J-lens vectors), not W_U @ J_l.T."""
    d = 8
    vocab = 5
    J_l = np.arange(d * d, dtype=np.float32).reshape(d, d)  # non-symmetric
    W_U = torch.randn(vocab, d)
    V = _resolve_V(None, J_l, W_U, torch.device("cpu"), torch.float32)
    expected = W_U @ torch.from_numpy(J_l)
    assert torch.allclose(V, expected)


def test_occupancy_accumulates_selected_vectors():
    """A 2-sparse signal must need at least 2 vectors (was 1 before the fix)."""
    d = 32
    V = torch.randn(50, d)
    V = V / (V.norm(dim=-1, keepdim=True) + 1e-9)
    h = (2.0 * V[3] + 1.5 * V[17]).to(torch.float32)
    k = jspace_occupancy(h, V=V, max_k=10, threshold=1e-3, random_seed=0)
    assert 2 <= k <= 10


def test_fingerprint_changes_with_corpus_and_revision():
    base = model_fingerprint("gpt2", 10, False)
    assert base != model_fingerprint("gpt2", 10, False, corpus_hash="abc")
    assert base != model_fingerprint("gpt2", 10, False, revision="v1")
    assert model_fingerprint("gpt2", 10, False) == model_fingerprint("gpt2", 10, False)


def test_train_jacobian_lens_rejects_empty_corpus():
    corpus = torch.zeros(0, 4, dtype=torch.long)
    with pytest.raises(JSpaceError, match="empty"):
        train_jacobian_lens(None, corpus)


def test_train_jacobian_lens_rejects_mismatched_mask():
    corpus = torch.ones(2, 4, dtype=torch.long)
    mask = torch.ones(2, 5, dtype=torch.long)
    with pytest.raises(JSpaceError, match="attention_mask"):
        train_jacobian_lens(None, corpus, attention_mask=mask)


def test_frozen_qk_hooks_match_gpt2_c_attn():
    """GPT-2's fused c_attn must be hooked (0 hooks before the fix)."""
    model, _ = load_model("sshleifer/tiny-gpt2", torch.device("cpu"), torch.float32)
    handles = _attach_frozen_qk_hooks(model)
    try:
        assert len(handles) == len(layer_indices(model))
    finally:
        for h in handles:
            h.remove()


def test_apply_intervention_accepts_batched_inputs():
    """apply_intervention must not crash for batch size > 1."""
    model, tokenizer = load_model("sshleifer/tiny-gpt2", torch.device("cpu"), torch.float32)
    enc = tokenizer(["hello world", "hi"], return_tensors="pt", padding=True)
    logits = apply_intervention(
        model,
        lambda h, layer_idx: h,
        (0, 1),
        enc["input_ids"],
        attention_mask=enc["attention_mask"],
    )
    assert logits.shape[0] == 2
    assert logits.shape[1] == enc["input_ids"].shape[1]
