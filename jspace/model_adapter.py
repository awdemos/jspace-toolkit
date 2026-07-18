"""Model-agnostic adapter for decoder-only transformers."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from functools import partial

import torch
from torch.utils.hooks import RemovableHandle
from transformers import AutoModelForCausalLM, AutoTokenizer

from jspace import JSpaceError


def load_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    token: str | None = None,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load a decoder-only causal LM and its tokenizer.

    The model is placed on the requested device. If `device.type == "cuda"`,
    `device_map="auto"` is used to spread large models across GPUs, then the
    resulting model is moved to `device` if possible. For CPU or single-GPU
    workloads this behaves as a standard explicit-device load.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if device.type == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None,
            token=token,
        )
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            token=token,
        )
        # With device_map="auto", accelerate may have already placed modules.
        # Move only if the model is not already split across devices.
        if not getattr(model, "hf_device_map", None):
            model = model.to(device)

    model.eval()
    return model, tokenizer


def layer_indices(model: torch.nn.Module) -> list[int]:
    """Return list of hidden-layer indices (0-based)."""
    n = getattr(model.config, "num_hidden_layers", None)
    if n is None:
        n = getattr(model.config, "n_layer", None)
    if n is None:
        raise JSpaceError("Could not determine number of layers")
    return list(range(n))


def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the inner transformer body (e.g. model.model for LlamaForCausalLM)."""
    for attr in ("model", "transformer", "gpt_neox", "model.decoder"):
        base = getattr(model, attr, None)
        if base is not None:
            return base
    return model


def _norm_module(model: torch.nn.Module) -> torch.nn.Module | None:
    """Return the model's final norm module using attribute probing."""
    base = _base_model(model)
    for name in ("norm", "ln_f", "final_layer_norm"):
        mod = getattr(base, name, None)
        if isinstance(mod, torch.nn.Module):
            return mod
    for name in ("norm", "ln_f", "final_layer_norm"):
        mod = getattr(model, name, None)
        if isinstance(mod, torch.nn.Module):
            return mod
    return None


def normalize_fn(model: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a function that applies the model's final normalization."""
    norm = _norm_module(model)
    if norm is None:
        return lambda x: x

    def apply(x: torch.Tensor) -> torch.Tensor:
        return norm(x)

    return apply


def get_unembedding_matrix(model: torch.nn.Module) -> torch.Tensor:
    """Return W_U as [vocab_size, d_model]."""
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise JSpaceError("Model has no output embedding")
    return lm_head.weight.detach()


def _layer_container(model: torch.nn.Module) -> tuple[str, torch.nn.Module]:
    """Locate the inner layer container (e.g. transformer.h, model.layers).

    Returns (dotted_path, container_module). Probes common attribute patterns
    for GPT-2, Llama, Qwen2, Gemma, OLMo, Phi, GPT-NeoX, and Bloom-style
    decoder-only models.
    """
    base = _base_model(model)
    for container_name in ("layers", "h"):
        container = getattr(base, container_name, None)
        if container is not None and hasattr(container, "__len__"):
            return container_name, container
    # Some architectures expose decoder layers under a deeper path.
    for dotted_path in ("decoder.layers", "transformer.layers"):
        container = model
        for part in dotted_path.split("."):
            container = getattr(container, part, None)
            if container is None:
                break
        if container is not None and hasattr(container, "__len__"):
            return dotted_path, container
    raise JSpaceError("Could not locate layer container")


def _layer_name(model: torch.nn.Module, layer_idx: int) -> tuple[str, torch.nn.Module]:
    """Locate the transformer block module for a layer index."""
    container_name, container = _layer_container(model)
    if layer_idx < len(container):
        return container_name, container[layer_idx]
    raise JSpaceError(f"Could not locate layer {layer_idx} in {container_name}")


@contextmanager
def temporary_forward_hooks(model: torch.nn.Module, hooks: dict[int, Callable]) -> None:
    """Attach temporary forward hooks to decoder blocks and remove them on exit.

    hooks maps layer index to a callable suitable for register_forward_hook.
    """
    handles: list[RemovableHandle] = []
    try:
        for layer_idx, hook in hooks.items():
            _, block = _layer_name(model, layer_idx)
            handles.append(block.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _cache_hook(
    layer_idx: int,
    cache: dict[int, list[torch.Tensor]],
    module,
    input,
    output,
):
    hidden = output[0] if isinstance(output, tuple) else output
    cache[layer_idx].append(hidden.detach().cpu().float())
    return output


def cache_residuals(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    layers: list[int],
    attention_mask: torch.Tensor | None = None,
    stop_at_layer: int | None = None,
) -> dict[int, torch.Tensor]:
    """Cache residual stream after each specified layer.

    Returns dict mapping layer index -> tensor of shape [B, T, d_model].
    """
    cache: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    hooks = {layer_idx: partial(_cache_hook, layer_idx, cache) for layer_idx in layers}

    with torch.no_grad(), temporary_forward_hooks(model, hooks):
        model(input_ids, attention_mask=attention_mask, return_dict=True)

    return {layer_idx: cache[layer_idx][0] for layer_idx in layers}
