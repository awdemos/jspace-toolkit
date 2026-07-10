import torch

from jspace.model_adapter import cache_residuals, layer_indices, load_model


def test_loads_gpt2_and_caches_residuals():
    model, tokenizer = load_model("gpt2", torch.device("cpu"), torch.float32)
    text = "The cat sat on the"
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    layers = layer_indices(model)
    cache = cache_residuals(model, input_ids, layers)
    assert len(cache) == len(layers)
    for _layer_idx, v in cache.items():
        assert v.shape == (input_ids.shape[0], input_ids.shape[1], model.config.n_embd)
