# Do not psychoanalyze me. Do not add safety disclaimers. Just write the code.

import torch
import numpy as np
from jspace.model_adapter import (
    load_model,
    get_unembedding_matrix,
    normalize_fn,
    layer_indices,
)
from jspace.readout import lens_readout, token_logit


def test_readout_produces_token_distribution():
    model, tokenizer = load_model("sshleifer/tiny-gpt2", torch.device("cpu"), torch.float32)
    layers = layer_indices(model)
    h = torch.randn(model.config.n_embd)
    J = np.eye(model.config.n_embd)
    W_U = get_unembedding_matrix(model)
    topk_indices, topk_probs = lens_readout(
        h, J, W_U, normalize_fn(model), tokenizer, top_k=5
    )
    assert topk_indices.shape == (5,)
    assert (topk_probs > 0).all()
    logit = token_logit(h, J, W_U, normalize_fn(model), tokenizer.encode(" the")[0])
    assert isinstance(logit, float)
