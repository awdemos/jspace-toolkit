"""Train Jacobian Lens matrices J_l."""

import contextlib
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from jspace import JSpaceError
from jspace.model_adapter import _base_model, _layer_container, _norm_module, layer_indices
from jspace.utils import get_position_ids, lens_cache_exists, load_lens_layer, save_lens_layer
from jspace.viz import jl_track


def _d_model(model: nn.Module) -> int:
    d_model = getattr(model.config, "hidden_size", getattr(model.config, "n_embd", None))
    if d_model is None:
        raise JSpaceError("Could not determine model hidden size from config")
    return int(d_model)


def _get_layer_block(model: nn.Module, layer_idx: int) -> nn.Module:
    """Return decoder block at layer_idx from the inner transformer body."""
    _, container = _layer_container(model)
    if layer_idx < len(container):
        return container[layer_idx]
    raise JSpaceError(f"Could not locate layer {layer_idx}")


@contextlib.contextmanager
def _model_dtype(model: nn.Module, dtype: torch.dtype):
    """Temporarily cast all parameters and buffers to dtype and restore them exactly.

    Stores both dtype and device so that models split via accelerate return to
    their original per-tensor placements after the training VJP pass.
    """
    original: dict[str, tuple[torch.dtype, torch.device]] = {}
    with torch.no_grad():
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            original[name] = (tensor.dtype, tensor.device)
        model.to(dtype)
    try:
        yield
    finally:
        with torch.no_grad():
            for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
                if name in original:
                    orig_dtype, orig_device = original[name]
                    tensor.data = tensor.data.to(orig_dtype)
                    if tensor.device != orig_device:
                        tensor.data = tensor.data.to(orig_device)


def _attach_frozen_qk_hooks(model: nn.Module) -> list:
    """Detach query/key projection outputs so gradients do not flow through them.

    Handles Llama-style separate q/k projections and GPT-2's fused ``c_attn``
    Conv1D (whose output's first two thirds are Q and K). Warns when nothing
    matched so the flag is never a silent no-op.
    """
    handles = []

    def detach_hook(module, inp, out):
        return out.detach()

    def make_fused_hook(n_embd: int):
        def hook(module, inp, out):
            q, k, v = out.split(n_embd, dim=-1)
            return torch.cat([q.detach(), k.detach(), v], dim=-1)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            x in name for x in ("q_proj", "query", "k_proj", "key")
        ):
            handles.append(module.register_forward_hook(detach_hook))
        elif name.endswith("c_attn"):
            # GPT-2-style fused QKV projection (transformers Conv1D, weight
            # shape [d_model, 3 * d_model]).
            weight = getattr(module, "weight", None)
            if weight is None or weight.dim() != 2 or weight.shape[1] % 3 != 0:
                continue
            n_embd = weight.shape[1] // 3
            handles.append(module.register_forward_hook(make_fused_hook(n_embd)))
    if not handles:
        warnings.warn(
            "frozen_qk=True but no query/key projections matched this "
            "architecture; the flag has no effect",
            stacklevel=2,
        )
    return handles


def _capture_h_l(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Capture the residual stream after layer_idx (post-layer residual)."""
    position_ids = get_position_ids(attention_mask)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )
    # hidden_states[0] = embedding output, hidden_states[layer_idx+1] = output after layer_idx.
    return out.hidden_states[layer_idx + 1]


def _run_from_layer(
    model: nn.Module,
    layer_idx: int,
    h_l: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_layer: int | None = None,
) -> torch.Tensor:
    """Run forward from layer_idx with h_l injected, returning target residual.

    If target_layer is None, return the pre-final-norm residual (the original
    behavior). Otherwise return the residual after `target_layer`, i.e. the
    output of transformer block `target_layer`. This lets every source layer
    map to a fixed penultimate layer as described in the paper.
    """
    base = _base_model(model)
    num_layers = len(layer_indices(model))
    position_ids = get_position_ids(attention_mask)

    if target_layer is None:
        capture_layer = num_layers - 1
    else:
        if not 0 <= target_layer < num_layers:
            raise JSpaceError(f"target_layer {target_layer} out of range [0, {num_layers})")
        capture_layer = target_layer

    if layer_idx > capture_layer:
        raise JSpaceError(f"source layer {layer_idx} cannot be after target layer {capture_layer}")

    if layer_idx == capture_layer:
        return h_l

    target_h: list[torch.Tensor | None] = [None]

    def capture_block_input(module, input_tuple):
        target_h[0] = input_tuple[0]
        return input_tuple

    # Residual after target_layer is the input to the following block or norm.
    if capture_layer + 1 < num_layers:
        residual_after_target = _get_layer_block(model, capture_layer + 1)
    else:
        final_norm = _norm_module(model)
        if final_norm is None:
            raise JSpaceError("Could not locate final normalization module")
        residual_after_target = final_norm
    capture_handle = residual_after_target.register_forward_pre_hook(capture_block_input)

    def replace_source_input(module, input_tuple):
        return (h_l,) + input_tuple[1:]

    inject_block = _get_layer_block(model, layer_idx + 1)
    inject_handle = inject_block.register_forward_pre_hook(replace_source_input)

    try:
        base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
    finally:
        capture_handle.remove()
        inject_handle.remove()

    if target_h[0] is None:
        raise JSpaceError("Failed to capture target-layer residual")

    # When target_layer is None the capture hook sits on the final norm, so
    # target_h[0] is already the pre-final-norm residual — no second forward
    # pass is needed.
    return target_h[0]


def _average_jacobian_for_layer(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
    output_dim_chunk: int = 16,
    target_layer: int | None = None,
) -> torch.Tensor:
    """Compute averaged Jacobian for one prompt across all source positions."""
    device = next(model.parameters()).device
    B, T = input_ids.shape
    d_model = _d_model(model)

    # Capture residual after layer_idx.
    h_l = _capture_h_l(model, input_ids, attention_mask, layer_idx).to(device, torch.float32)
    h_l.requires_grad_(True)

    # Causal + padding mask. pair_mask[b, t_prime, t] = 1 if t_prime >= t and both valid.
    valid = attention_mask.to(torch.float32)
    causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.float32), diagonal=0)
    pair_mask = causal.unsqueeze(0) * valid.unsqueeze(1) * valid.unsqueeze(2)

    accum = torch.zeros(d_model, d_model, device="cpu", dtype=torch.float64)

    # Disable parameter gradients; we only need gradients w.r.t. h_l.
    prev_requires_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    try:
        for i_start in range(0, d_model, output_dim_chunk):
            i_end = min(d_model, i_start + output_dim_chunk)
            C = i_end - i_start

            z = _run_from_layer(
                model, layer_idx, h_l, input_ids, attention_mask, target_layer=target_layer
            ).to(torch.float32)
            # z: (B, T, d_model) target-layer residual
            z_chunk = z[..., i_start:i_end]

            # s_c = sum over batch, target positions, source positions of z[b, t_prime, c]
            s_chunk = torch.einsum("btc,bts->c", z_chunk, pair_mask.to(torch.float32))

            eye = torch.eye(C, device=device, dtype=torch.float32)
            grads = torch.autograd.grad(
                outputs=s_chunk,
                inputs=h_l,
                grad_outputs=eye,
                is_grads_batched=True,
                retain_graph=False,
                create_graph=False,
            )[0]
            # grads: (C, B, T, d_model)
            grads = grads.to(torch.float64)
            # Uniform average over valid (batch, source, target) triples.
            per_i = grads.sum(dim=(1, 2))  # (C, d_model)
            triple_count = pair_mask.sum().item()
            per_i = per_i / (triple_count + 1e-9)
            accum[i_start:i_end] += per_i.cpu()

            del z, z_chunk, s_chunk, grads, per_i
    finally:
        for p, flag in zip(model.parameters(), prev_requires_grad, strict=True):
            p.requires_grad_(flag)

    return accum


def train_jacobian_lens(
    model: nn.Module,
    corpus: torch.Tensor,
    target_layer: int | None = None,
    cache_dir: Path | None = None,
    dtype: torch.dtype = torch.float32,
    max_positions: int = 128,
    batch_size: int = 1,
    output_dim_chunk: int = 16,
    frozen_qk: bool = False,
    attention_mask: torch.Tensor | None = None,
) -> dict[int, np.ndarray]:
    """Train and cache J_l matrices. Returns dict layer_idx -> numpy matrix.

    Pass the tokenizer's ``attention_mask`` alongside ``corpus`` so padded
    positions are excluded from the averaged Jacobian; without it the mask is
    reconstructed from ``pad_token_id``, which not all models define.
    """
    if corpus.shape[0] == 0:
        raise JSpaceError("corpus is empty; nothing to train on")
    if attention_mask is not None and attention_mask.shape != corpus.shape:
        raise JSpaceError(
            f"attention_mask shape {tuple(attention_mask.shape)} does not match "
            f"corpus shape {tuple(corpus.shape)}"
        )

    layers = layer_indices(model)
    d_model = _d_model(model)

    if cache_dir is not None and lens_cache_exists(cache_dir, layers):
        return {layer: load_lens_layer(cache_dir, layer) for layer in layers}

    # The paper defines J_l for source layers l <= target_layer.
    effective_target = target_layer if target_layer is not None else len(layers) - 1
    source_layers = [layer for layer in layers if layer <= effective_target]

    qk_handles = _attach_frozen_qk_hooks(model) if frozen_qk else []
    try:
        with _model_dtype(model, torch.float32):
            J: dict[int, torch.Tensor] = {
                layer: torch.zeros(d_model, d_model, device="cpu", dtype=torch.float64)
                for layer in source_layers
            }
            count = 0

            model.eval()
            batches = range(0, corpus.shape[0], batch_size)
            for start in jl_track(batches, "Training J-Lens"):
                batch_ids = corpus[start : start + batch_size]
                B, T = batch_ids.shape
                if attention_mask is not None:
                    batch_mask = attention_mask[start : start + batch_size]
                else:
                    pad_id = getattr(model.config, "pad_token_id", None)
                    if pad_id is None:
                        pad_id = -1
                    batch_mask = batch_ids != pad_id
                device = next(model.parameters()).device
                batch_ids = batch_ids.to(device)
                batch_mask = batch_mask.to(device)
                for layer_idx in source_layers:
                    grad_mat = _average_jacobian_for_layer(
                        model,
                        batch_ids,
                        batch_mask,
                        layer_idx,
                        output_dim_chunk=output_dim_chunk,
                        target_layer=target_layer,
                    )
                    J[layer_idx] += grad_mat
                count += B

            for layer in source_layers:
                J[layer] = J[layer] / count

            result = {layer: J[layer].numpy().astype(np.float32) for layer in source_layers}
            if cache_dir is not None:
                for layer, mat in result.items():
                    save_lens_layer(cache_dir, layer, mat)
            return result
    finally:
        for h in qk_handles:
            h.remove()
