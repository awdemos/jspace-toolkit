import torch

from jspace.jacobian_lens import train_jacobian_lens
from jspace.model_adapter import (
    get_unembedding_matrix,
    layer_indices,
    load_model,
    normalize_fn,
)
from jspace.readout import lens_readout
from jspace.utils import get_cache_dir, model_fingerprint


def test_verbal_report_sanity(tmp_path):
    model, tokenizer = load_model("sshleifer/tiny-gpt2", torch.device("cpu"), torch.float32)
    prompt = "Think of a sport:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=5, do_sample=False)
    answer = (
        tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True)
        .strip()
        .split()[0]
    )

    layers = layer_indices(model)
    target_layer = layers[-2]
    corpus = input_ids.repeat(2, 1)
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
    h = model(input_ids, output_hidden_states=True).hidden_states[target_layer + 1][0, -1]
    topk_indices, _ = lens_readout(
        h,
        J[target_layer],
        get_unembedding_matrix(model),
        normalize_fn(model),
        tokenizer,
        top_k=10,
    )
    topk_tokens = [tokenizer.decode([idx]) for idx in topk_indices.tolist()]
    assert answer.lower() in [t.lower().strip() for t in topk_tokens]
