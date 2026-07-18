import torch
from transformers import GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from jspace.model_adapter import (
    _base_model,
    _layer_container,
    _norm_module,
    cache_residuals,
    layer_indices,
    load_model,
    normalize_fn,
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


def test_norm_module_found_for_gpt2():
    model = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")
    norm = _norm_module(model)
    assert norm is not None
    assert normalize_fn(model)(torch.randn(1, model.config.n_embd)).shape == (
        1,
        model.config.n_embd,
    )


def test_norm_module_found_for_llama():
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
    )
    model = LlamaForCausalLM(config)
    norm = _norm_module(model)
    assert norm is not None
    _, container = _layer_container(model)
    assert len(container) == 2
    assert _base_model(model) is model.model
