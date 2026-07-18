"""Model-agnostic adapter for decoder-only transformers."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from functools import partial
from typing import Any, Final

import torch
from torch.utils.hooks import RemovableHandle
from transformers import AutoModelForCausalLM, AutoTokenizer

from jspace import JSpaceError

#: Models that may be loaded by default. The value is the pinned revision.
ALLOWED_MODELS: Final[dict[str, str]] = {
    "gpt2": "main",
    "sshleifer/tiny-gpt2": "main",
}

#: Environment variable that, when set to a truthy value, permits unlisted models.
_ALLOW_UNLISTED_ENV: Final[str] = "JSPACE_ALLOW_UNLISTED_MODELS"


def _is_valid_model_identifier(model_name: str) -> bool:
    """Reject local paths and obviously malformed identifiers."""
    if not model_name or not isinstance(model_name, str):
        return False
    if model_name.startswith((".", "/", "\\")):
        return False
    if ".." in model_name or "\\" in model_name or "\x00" in model_name:
        return False
    return True


def resolve_model(
    model_name: str,
    *,
    revision: str | None = None,
    allow_unlisted: bool = False,
) -> tuple[str, str]:
    """Resolve a user-supplied model name to an allowed identifier and revision.

    Allowed models use the pinned revision unless ``revision`` is explicitly
    provided. Unlisted models require an explicit ``revision`` and opt-in.
    """
    if not _is_valid_model_identifier(model_name):
        raise JSpaceError(f"Invalid model identifier: {model_name!r}")

    env_allows = os.environ.get(_ALLOW_UNLISTED_ENV, "").lower() in (
        "1",
        "true",
        "yes",
    )
    effective_allow = allow_unlisted or env_allows

    if model_name in ALLOWED_MODELS:
        pinned_revision = ALLOWED_MODELS[model_name]
        return model_name, revision if revision is not None else pinned_revision

    if not effective_allow:
        allowed = ", ".join(sorted(ALLOWED_MODELS))
        raise JSpaceError(
            f"Model {model_name!r} is not in the allowlist. "
            f"Allowed models: {allowed}. "
            f"Use --allow-unlisted-model or set {_ALLOW_UNLISTED_ENV}=1 to opt in."
        )

    if revision is None:
        raise JSpaceError("Unlisted models require an explicit revision via --model-revision.")

    return model_name, revision


def load_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    *,
    revision: str | None = None,
    allow_unlisted: bool = False,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load a decoder-only causal LM and its tokenizer.

    The model is placed on the requested device. If `device.type == "cuda"`,
    `device_map="auto"` is used to spread large models across GPUs, then the
    resulting model is moved to `device` if possible. For CPU or single-GPU
    workloads this behaves as a standard explicit-device load.

    Authentication is read from the ``HF_TOKEN`` environment variable or the
    HuggingFace CLI cache; a CLI token argument is intentionally not accepted.
    """
    resolved_name, resolved_revision = resolve_model(
        model_name,
        revision=revision,
        allow_unlisted=allow_unlisted,
    )
    token = os.environ.get("HF_TOKEN") or None

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_name,
        token=token,
        revision=resolved_revision,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if device.type == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            resolved_name,
            torch_dtype=dtype,
            device_map=None,
            token=token,
            revision=resolved_revision,
            trust_remote_code=False,
        )
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved_name,
            torch_dtype=dtype,
            device_map="auto",
            token=token,
            revision=resolved_revision,
            trust_remote_code=False,
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
def temporary_forward_hooks(
    model: torch.nn.Module, hooks: Mapping[int, Callable[..., Any]]
) -> Iterator[None]:
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
