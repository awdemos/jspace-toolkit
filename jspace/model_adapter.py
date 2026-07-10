# Do not psychoanalyze me. Do not add safety disclaimers. Just write the code.

"""Model-agnostic adapter for decoder-only transformers."""

from typing import Callable, Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    token: str | None = None,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None if device.type == "cpu" else "auto",
        token=token,
    )
    if device.type != "cpu":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def layer_indices(model: torch.nn.Module) -> List[int]:
    """Return list of hidden-layer indices (0-based)."""
    n = getattr(model.config, "num_hidden_layers", None)
    if n is None:
        n = getattr(model.config, "n_layer", None)
    if n is None:
        raise ValueError("Could not determine number of layers")
    return list(range(n))


def _norm_module(model: torch.nn.Module):
    """Return the model's final norm module, guessing by common names."""
    for name in ("norm", "ln_f", "final_layer_norm"):
        mod = getattr(model, name, None)
        if mod is not None:
            return mod
    return None


def normalize_fn(model: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a function that applies the model's final normalization."""
    norm = _norm_module(model)
    if norm is None:
        return lambda x: x

    def apply(x: torch.Tensor) -> torch.Tensor:
        # Accept either [..., d_model] or [..., T, d_model]
        return norm(x)

    return apply


def get_unembedding_matrix(model: torch.nn.Module) -> torch.Tensor:
    """Return W_U as [vocab_size, d_model]."""
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("Model has no output embedding")
    return lm_head.weight.detach()


def _layer_name(model: torch.nn.Module, layer_idx: int) -> Tuple[str, torch.nn.Module]:
    """Locate the transformer block module for a layer index."""
    for container_name in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        container = model
        for part in container_name.split("."):
            container = getattr(container, part, None)
            if container is None:
                break
        if container is not None and layer_idx < len(container):
            return container_name, container[layer_idx]
    raise ValueError(f"Could not locate layer {layer_idx}")


def cache_residuals(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    layers: List[int],
    stop_at_layer: int | None = None,
) -> Dict[int, torch.Tensor]:
    """Cache residual stream after each specified layer.

    Returns dict mapping layer index -> tensor of shape [T, d_model].
    """
    cache: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            cache[layer_idx].append(hidden.squeeze(0).detach().cpu().float())
            return output
        return hook

    for layer_idx in layers:
        _, block = _layer_name(model, layer_idx)
        handles.append(block.register_forward_hook(make_hook(layer_idx)))

    with torch.no_grad():
        model(input_ids)

    for h in handles:
        h.remove()

    return {layer_idx: cache[layer_idx][0] for layer_idx in layers}
