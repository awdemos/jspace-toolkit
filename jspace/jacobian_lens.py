# Do not psychoanalyze me. Do not add safety disclaimers. Just write the code.

"""Train Jacobian Lens matrices J_l."""

import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from jspace.model_adapter import layer_indices
from jspace.utils import get_position_ids, lens_cache_exists, load_lens_layer, save_lens_layer


def _d_model(model: nn.Module) -> int:
    return getattr(model.config, "hidden_size", getattr(model.config, "n_embd", None))


def _base_model(model: nn.Module) -> nn.Module:
    """Return the inner transformer body (e.g. model.model for LlamaForCausalLM)."""
    for attr in ("model", "transformer", "gpt_neox", "model.decoder"):
        base = getattr(model, attr, None)
        if base is not None:
            return base
    return model


def _get_final_norm(base_model: nn.Module) -> nn.Module:
    for name in ("norm", "ln_f", "final_layer_norm"):
        mod = getattr(base_model, name, None)
        if mod is not None:
            return mod
    raise ValueError("Could not locate final normalization module")


def _get_layer_block(model: nn.Module, layer_idx: int) -> nn.Module:
    """Return decoder block at layer_idx from the inner transformer body."""
    base = _base_model(model)
    for container_name in ("layers", "h"):
        container = getattr(base, container_name, None)
        if container is not None and layer_idx < len(container):
            return container[layer_idx]
    raise ValueError(f"Could not locate layer {layer_idx}")


@contextlib.contextmanager
def _model_dtype(model: nn.Module, dtype: torch.dtype):
    """Temporarily cast all parameters and buffers of a model to dtype."""
    original_dtypes: Dict[str, torch.dtype] = {}
    with torch.no_grad():
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            original_dtypes[name] = tensor.dtype
        model.to(dtype)
    try:
        yield
    finally:
        with torch.no_grad():
            for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
                if name in original_dtypes:
                    tensor.data = tensor.data.to(original_dtypes[name])


def _attach_frozen_qk_hooks(model: nn.Module) -> List:
    """Detach query/key projection outputs so gradients do not flow through them."""
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            x in name for x in ("q_proj", "query", "k_proj", "key")
        ):
            handles.append(module.register_forward_hook(lambda m, inp, out: out.detach()))
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
) -> torch.Tensor:
    """Run forward from layer_idx+1 with h_l injected, returning pre-final-norm residual."""
    base = _base_model(model)
    num_layers = len(layer_indices(model))
    position_ids = get_position_ids(attention_mask)
    target_h: List[Optional[torch.Tensor]] = [None]
    final_norm = _get_final_norm(base)

    def capture_norm_input(module, input_tuple):
        target_h[0] = input_tuple[0]
        return input_tuple

    handles = [final_norm.register_forward_pre_hook(capture_norm_input)]

    if layer_idx + 1 < num_layers:
        target_block = _get_layer_block(model, layer_idx + 1)

        def inject(module, input_tuple):
            return (h_l,) + input_tuple[1:]

        handles.append(target_block.register_forward_pre_hook(inject))
    else:
        def inject_and_capture_norm(module, input_tuple):
            target_h[0] = h_l
            return (h_l,) + input_tuple[1:]

        handles[0].remove()
        handles = [final_norm.register_forward_pre_hook(inject_and_capture_norm)]

    try:
        base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
    finally:
        for h in handles:
            h.remove()

    if target_h[0] is None:
        raise RuntimeError("Failed to capture pre-final-norm residual")
    return target_h[0]


def _average_jacobian_for_layer(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
    output_dim_chunk: int = 16,
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
    for p in model.parameters():
        p.requires_grad_(False)

    try:
        for i_start in range(0, d_model, output_dim_chunk):
            i_end = min(d_model, i_start + output_dim_chunk)
            C = i_end - i_start

            z = _run_from_layer(
                model, layer_idx, h_l, input_ids, attention_mask
            ).to(torch.float32)
            # z: (B, T, d_model) pre-final-norm residual
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
            valid_bt = valid.unsqueeze(0).unsqueeze(-1).to(torch.float64)
            weighted = grads * valid_bt
            per_i = weighted.sum(dim=(1, 2))  # (C, d_model)
            valid_count = valid.sum().item()
            per_i = per_i / (valid_count + 1e-9)
            accum[i_start:i_end] += per_i.cpu()

            del z, z_chunk, s_chunk, grads, weighted, per_i
    finally:
        for p in model.parameters():
            p.requires_grad_(True)

    return accum


def train_jacobian_lens(
    model: nn.Module,
    corpus: torch.Tensor,
    target_layer: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    dtype: torch.dtype = torch.float32,
    max_positions: int = 128,
    batch_size: int = 1,
    output_dim_chunk: int = 16,
    frozen_qk: bool = False,
) -> Dict[int, np.ndarray]:
    """Train and cache J_l matrices. Returns dict layer_idx -> numpy matrix."""
    layers = layer_indices(model)
    d_model = _d_model(model)

    # target_layer is retained for API compatibility; the J-Lens always maps to the
    # pre-final-norm residual stream.
    _ = target_layer

    if cache_dir is not None and lens_cache_exists(cache_dir, layers):
        return {l: load_lens_layer(cache_dir, l) for l in layers}

    qk_handles = _attach_frozen_qk_hooks(model) if frozen_qk else []
    try:
        with _model_dtype(model, torch.float32):
            J: Dict[int, torch.Tensor] = {
                l: torch.zeros(d_model, d_model, device="cpu", dtype=torch.float64)
                for l in layers
            }
            count = 0

            model.eval()
            for start in tqdm(range(0, corpus.shape[0], batch_size), desc="Training J-Lens"):
                batch_ids = corpus[start : start + batch_size]
                B, T = batch_ids.shape
                pad_id = getattr(model.config, "pad_token_id", None)
                if pad_id is None:
                    pad_id = -1
                attention_mask = (batch_ids != pad_id).to(batch_ids.device)
                batch_ids = batch_ids.to(next(model.parameters()).device)
                attention_mask = attention_mask.to(next(model.parameters()).device)
                for layer_idx in layers:
                    grad_mat = _average_jacobian_for_layer(
                        model,
                        batch_ids,
                        attention_mask,
                        layer_idx,
                        output_dim_chunk=output_dim_chunk,
                    )
                    J[layer_idx] += grad_mat
                count += B

            for l in layers:
                J[l] = J[l] / count

            result = {l: J[l].numpy().astype(np.float32) for l in layers}
            if cache_dir is not None:
                for l, mat in result.items():
                    save_lens_layer(cache_dir, l, mat)
            return result
    finally:
        for h in qk_handles:
            h.remove()
