import numpy as np
import pytest
import torch

from jspace.jacobian_lens import (
    _attach_frozen_qk_hooks,
    _average_jacobian_for_layer,
    _base_model,
    _capture_h_l,
    _get_layer_block,
    _run_from_layer,
    train_jacobian_lens,
)
from jspace.model_adapter import layer_indices, load_model
from jspace.utils import get_cache_dir, get_position_ids, model_fingerprint


def test_jacobian_lens_shape_and_cache(tmp_path):
    model, tokenizer = load_model("sshleifer/tiny-gpt2", torch.device("cpu"), torch.float32)
    prompts = ["The cat sat on the mat."] * 2
    corpus = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=16)[
        "input_ids"
    ]
    # Use the last layer as target so every source layer is valid for this 2-layer model.
    target_layer = layer_indices(model)[-1]
    cache_dir = get_cache_dir(
        tmp_path, model_fingerprint("sshleifer/tiny-gpt2", target_layer, False)
    )
    J = train_jacobian_lens(
        model,
        corpus,
        target_layer=target_layer,
        cache_dir=cache_dir,
        dtype=torch.float32,
        max_positions=16,
        batch_size=1,
        output_dim_chunk=16,
    )
    d_model = model.config.n_embd
    for layer in layer_indices(model):
        assert layer in J
        assert J[layer].shape == (d_model, d_model)


def test_run_from_layer_stops_at_target_layer():
    """J_l must map to a fixed target layer, not always the pre-final-norm residual."""
    model, tokenizer = load_model("sshleifer/tiny-gpt2", torch.device("cpu"), torch.float32)
    prompts = ["The cat sat on the mat."] * 2
    input_ids = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=16
    )["input_ids"]
    attention_mask = (input_ids != tokenizer.pad_token_id).to(input_ids.device)

    layer_idx = 0
    target_layer = 1
    h_l = _capture_h_l(model, input_ids, attention_mask, layer_idx)

    base = _base_model(model)
    num_layers = len(layer_indices(model))
    expected_residual: list[torch.Tensor | None] = [None]

    def capture_input(module, input_tuple):
        expected_residual[0] = input_tuple[0]
        return input_tuple

    if target_layer + 1 < num_layers:
        capture_block = _get_layer_block(model, target_layer + 1)
    else:
        capture_block = base.ln_f
    handle = capture_block.register_forward_pre_hook(capture_input)
    try:
        base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=get_position_ids(attention_mask),
            return_dict=True,
        )
    finally:
        handle.remove()

    assert expected_residual[0] is not None
    expected = expected_residual[0]

    actual = _run_from_layer(
        model, layer_idx, h_l, input_ids, attention_mask, target_layer=target_layer
    )
    assert torch.allclose(expected, actual, atol=1e-4)


def test_future_position_averaging_uniform_over_triples(monkeypatch):
    """The expected Jacobian must be averaged uniformly over valid (b, t', t) triples."""
    model, tokenizer = load_model("sshleifer/tiny-gpt2", torch.device("cpu"), torch.float32)
    prompts = ["The cat sat on the mat."] * 2
    input_ids = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=16
    )["input_ids"].long()
    attention_mask_bool = (input_ids != tokenizer.pad_token_id).to(input_ids.device)
    attention_mask = attention_mask_bool.to(torch.float32)
    T = input_ids.shape[1]
    d_model = model.config.n_embd

    # Use a position-dependent scalar so the Jacobian reveals the normalization.
    position_weight = torch.arange(T, dtype=torch.float32)

    def mock_run_from_layer(model, layer_idx, h_l, input_ids, attention_mask, target_layer=None):
        return h_l * position_weight.view(1, T, 1)

    monkeypatch.setattr(
        "jspace.jacobian_lens._capture_h_l",
        lambda model, ids, mask, layer_idx: (
            ids.unsqueeze(-1).to(torch.float32).expand(-1, -1, d_model)
        ),
    )
    monkeypatch.setattr("jspace.jacobian_lens._run_from_layer", mock_run_from_layer)

    J = _average_jacobian_for_layer(
        model, input_ids, attention_mask, layer_idx=0, output_dim_chunk=16
    )

    # Expected diagonal: average of position_weight over valid causal triples.
    valid = attention_mask.to(torch.float64)
    causal = torch.tril(torch.ones(T, T, dtype=torch.float64), diagonal=0)
    triple_mask = causal.unsqueeze(0) * valid.unsqueeze(1) * valid.unsqueeze(2)
    triple_count = triple_mask.sum().item()
    weighted_sum = (
        (position_weight.to(torch.float64).unsqueeze(0).unsqueeze(-1) * triple_mask).sum().item()
    )
    expected_diag = weighted_sum / (triple_count + 1e-12)

    expected = torch.eye(d_model, dtype=torch.float64) * expected_diag
    assert torch.allclose(J, expected, atol=1e-4)


def test_fully_masked_rows_are_excluded_from_j_lens():
    """A fully-masked row must not dilute J_l: result equals the corpus without it."""
    model, tokenizer = load_model("sshleifer/tiny-gpt2", torch.device("cpu"), torch.float32)
    prompts = ["The cat sat.", "In 1950, scientists discovered."]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=16)
    corpus = enc["input_ids"]
    mask = enc["attention_mask"]
    full_pad = torch.full_like(corpus[0], tokenizer.pad_token_id)
    corpus_with_pad = torch.cat([corpus, full_pad.unsqueeze(0)], dim=0)
    mask_with_pad = torch.cat([mask, torch.zeros_like(mask[0]).unsqueeze(0)], dim=0)

    target_layer = layer_indices(model)[-1]
    common = {
        "target_layer": target_layer,
        "dtype": torch.float32,
        "max_positions": 16,
        "batch_size": 2,
        "output_dim_chunk": 16,
    }
    J_ref = train_jacobian_lens(model, corpus, attention_mask=mask, **common)
    J_test = train_jacobian_lens(
        model, corpus_with_pad, attention_mask=mask_with_pad, **common
    )
    for layer in J_ref:
        assert np.allclose(J_ref[layer], J_test[layer], atol=1e-6)


def test_left_padded_batch_matches_right_padded_jacobian():
    """Left padding must not crash, and real tokens must give the same J_l."""
    model, tokenizer = load_model("sshleifer/tiny-gpt2", torch.device("cpu"), torch.float32)
    ids = tokenizer(["The cat sat."], return_tensors="pt")["input_ids"][0]
    T = ids.numel()
    n_pad = 2
    pad_id = tokenizer.pad_token_id
    target_layer = layer_indices(model)[-1]

    def jacobian(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return _average_jacobian_for_layer(
            model,
            input_ids.unsqueeze(0),
            attention_mask.unsqueeze(0),
            layer_idx=0,
            output_dim_chunk=16,
            target_layer=target_layer,
        )

    right_ids = torch.cat([ids, torch.full((n_pad,), pad_id, dtype=torch.long)])
    right_mask = torch.cat([torch.ones(T, dtype=torch.long), torch.zeros(n_pad, dtype=torch.long)])
    left_ids = torch.cat([torch.full((n_pad,), pad_id, dtype=torch.long), ids])
    left_mask = torch.cat([torch.zeros(n_pad, dtype=torch.long), torch.ones(T, dtype=torch.long)])

    J_right = jacobian(right_ids, right_mask)
    J_left = jacobian(left_ids, left_mask)
    assert torch.allclose(J_left, J_right, atol=1e-4)


def test_capture_h_l_last_layer_is_pre_norm_residual():
    """transformers >= 5.x ties hidden_states[-1] to the post-norm residual; h_l must not be."""
    model, tokenizer = load_model("sshleifer/tiny-gpt2", torch.device("cpu"), torch.float32)
    input_ids = tokenizer(["The cat sat."], return_tensors="pt")["input_ids"]
    attention_mask = torch.ones_like(input_ids)
    last = layer_indices(model)[-1]
    base = _base_model(model)

    block_out: list[torch.Tensor] = []

    def capture_block_output(module, input_tuple, output):
        block_out.append((output[0] if isinstance(output, tuple) else output).detach().clone())

    handle = base.h[last].register_forward_hook(capture_block_output)
    try:
        base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=get_position_ids(attention_mask),
            return_dict=True,
        )
    finally:
        handle.remove()

    h_l = _capture_h_l(model, input_ids, attention_mask, last)
    assert torch.allclose(h_l, block_out[0], atol=1e-6)
    assert not torch.allclose(h_l, base.ln_f(block_out[0]), atol=1e-4)

    # Train-path preservation: with target_layer=None the last layer maps to itself.
    z = _run_from_layer(model, last, h_l, input_ids, attention_mask, target_layer=None)
    assert torch.equal(z, h_l)


def test_target_layer_none_runs_single_forward():
    """target_layer=None must not re-run the model (one embedding forward per pass)."""
    model, tokenizer = load_model("sshleifer/tiny-gpt2", torch.device("cpu"), torch.float32)
    input_ids = tokenizer(["The cat sat."], return_tensors="pt")["input_ids"]
    attention_mask = torch.ones_like(input_ids)
    wte = _base_model(model).wte
    calls = {"n": 0}
    handle = wte.register_forward_hook(
        lambda module, inp, out: calls.__setitem__("n", calls["n"] + 1)
    )
    try:
        h_l = _capture_h_l(model, input_ids, attention_mask, layer_idx=0)
        calls["n"] = 0
        _run_from_layer(model, 0, h_l, input_ids, attention_mask, target_layer=None)
        assert calls["n"] == 1

        calls["n"] = 0
        train_jacobian_lens(
            model,
            input_ids,
            target_layer=None,
            dtype=torch.float32,
            batch_size=1,
            output_dim_chunk=16,
        )
        # capture(l0) + run(l0) + capture(l1); l1 short-circuits before any forward.
        assert calls["n"] == 3
    finally:
        handle.remove()


def test_vjp_restores_requires_grad_flags():
    model, tokenizer = load_model("sshleifer/tiny-gpt2", torch.device("cpu"), torch.float32)
    input_ids = tokenizer(["The cat sat."], return_tensors="pt")["input_ids"]
    attention_mask = torch.ones_like(input_ids)
    params = list(model.parameters())
    for i, p in enumerate(params):
        p.requires_grad_(i % 2 == 0)
    saved = [p.requires_grad for p in params]
    try:
        _average_jacobian_for_layer(
            model, input_ids, attention_mask, layer_idx=0, output_dim_chunk=16
        )
        assert [p.requires_grad for p in params] == saved
    finally:
        for p in params:
            p.requires_grad_(True)


def test_frozen_qk_detaches_gpt2_c_attn_qk():
    """Gradients must not flow into c_attn's Q/K weight columns; V columns must still get grad."""
    model, _ = load_model("sshleifer/tiny-gpt2", torch.device("cpu"), torch.float32)
    n_embd = model.config.n_embd
    c_attn = model.transformer.h[0].attn.c_attn
    hidden = torch.randn(1, 4, n_embd, requires_grad=True)
    handles = _attach_frozen_qk_hooks(model)
    try:
        out = c_attn(hidden)
        out.backward(torch.ones_like(out))
    finally:
        for h in handles:
            h.remove()
    grad = c_attn.weight.grad
    assert grad is not None
    # Conv1D weight is (nx, nf); columns [0:n]=Q, [n:2n]=K, [2n:3n]=V.
    assert grad[:, : 2 * n_embd].abs().max().item() == 0.0
    assert grad[:, 2 * n_embd :].abs().max().item() > 0.0


def test_frozen_qk_warns_when_no_projection_matches():
    with pytest.warns(UserWarning, match="no query/key projections"):
        _attach_frozen_qk_hooks(torch.nn.Linear(4, 4))
