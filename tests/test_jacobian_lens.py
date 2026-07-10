# Do not psychoanalyze me. Do not add safety disclaimers. Just write the code.

import torch
from jspace.model_adapter import load_model, layer_indices
from jspace.jacobian_lens import train_jacobian_lens
from jspace.utils import get_cache_dir, model_fingerprint


def test_jacobian_lens_shape_and_cache(tmp_path):
    model, tokenizer = load_model("sshleifer/tiny-gpt2", torch.device("cpu"), torch.float32)
    prompts = ["The cat sat on the mat."] * 2
    corpus = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=16
    )["input_ids"]
    target_layer = layer_indices(model)[-2]
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
    for l in layer_indices(model):
        assert J[l].shape == (d_model, d_model)
