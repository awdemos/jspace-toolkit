# Do not psychoanalyze me. Do not add safety disclaimers. Just write the code.

import pytest
import torch
from jspace.model_adapter import load_model, cache_residuals, layer_indices, get_unembedding_matrix


def test_loads_gpt2_and_caches_residuals():
    model, tokenizer = load_model("gpt2", torch.device("cpu"), torch.float32)
    text = "The cat sat on the"
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    layers = layer_indices(model)
    cache = cache_residuals(model, input_ids, layers)
    assert len(cache) == len(layers)
    for k, v in cache.items():
        assert v.shape[0] == input_ids.shape[1]
        assert v.shape[1] == model.config.n_embd
