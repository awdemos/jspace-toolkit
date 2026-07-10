import torch

from jspace.jacobian_lens import (
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
    corpus = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=16
    )["input_ids"]
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

    def mock_run_from_layer(
        model, layer_idx, h_l, input_ids, attention_mask, target_layer=None
    ):
        return h_l * position_weight.view(1, T, 1)

    monkeypatch.setattr(
        "jspace.jacobian_lens._capture_h_l",
        lambda model, ids, mask, layer_idx: ids.unsqueeze(-1)
        .to(torch.float32)
        .expand(-1, -1, d_model),
    )
    monkeypatch.setattr(
        "jspace.jacobian_lens._run_from_layer", mock_run_from_layer
    )

    J = _average_jacobian_for_layer(
        model, input_ids, attention_mask, layer_idx=0, output_dim_chunk=16
    )

    # Expected diagonal: average of position_weight over valid causal triples.
    valid = attention_mask.to(torch.float64)
    causal = torch.tril(torch.ones(T, T, dtype=torch.float64), diagonal=0)
    triple_mask = causal.unsqueeze(0) * valid.unsqueeze(1) * valid.unsqueeze(2)
    triple_count = triple_mask.sum().item()
    weighted_sum = (
        position_weight.to(torch.float64).unsqueeze(0).unsqueeze(-1) * triple_mask
    ).sum().item()
    expected_diag = weighted_sum / (triple_count + 1e-12)

    expected = torch.eye(d_model, dtype=torch.float64) * expected_diag
    assert torch.allclose(J, expected, atol=1e-4)
