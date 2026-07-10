"""Causal interventions in J-space."""

from collections.abc import Callable
from functools import partial

import torch

from jspace.model_adapter import temporary_forward_hooks


def _token_vector(token_id: int, V: torch.Tensor) -> torch.Tensor:
    return V[token_id]


def coordinate_swap(
    h: torch.Tensor,
    source_token_id: int,
    target_token_id: int,
    V: torch.Tensor,
    alpha: float = 1.0,
    k: int = 25,
) -> torch.Tensor:
    """Swap sparse J-space coefficients of two concept tokens while preserving h_perp."""
    from jspace.decomposition import decompose_jspace

    a, _, h_perp = decompose_jspace(h, V, k=k, non_negative=True)
    a_swapped = a.clone()
    a_swapped[source_token_id] = a[target_token_id] * alpha + a[source_token_id] * (1 - alpha)
    a_swapped[target_token_id] = a[source_token_id] * alpha + a[target_token_id] * (1 - alpha)
    return a_swapped @ V + h_perp


def ablate_topk_jspace(
    h: torch.Tensor,
    V: torch.Tensor,
    k: int = 10,
    exclude_output_tokens: bool = True,
    clean_top10: list[int] | None = None,
    random_control: bool = False,
) -> torch.Tensor:
    """Project out the top-k J-space components."""
    clean_top10 = clean_top10 or []
    scores = (V @ h).abs()
    if exclude_output_tokens:
        scores[clean_top10] = -float("inf")
    topk = torch.topk(scores, min(k, scores.shape[0])).indices
    subspace = V[topk]
    if random_control:
        subspace = torch.randn_like(subspace)
        subspace = torch.linalg.qr(subspace.T).Q.T
    proj_coeffs = torch.linalg.lstsq(subspace.T, h).solution
    h_proj = subspace.T @ proj_coeffs
    return h - h_proj


def ablate_concept(
    h: torch.Tensor,
    concept_token_id: int,
    V: torch.Tensor,
) -> torch.Tensor:
    v = _token_vector(concept_token_id, V)
    v_unit = v / (v.norm() + 1e-9)
    return h - (h @ v_unit) * v_unit


def steer(
    h: torch.Tensor,
    concept_token_id: int,
    V: torch.Tensor,
    strength: float = 0.1,
) -> torch.Tensor:
    return h + strength * _token_vector(concept_token_id, V)


def _intervention_hook(
    layer_idx: int,
    intervention_fn: Callable[[torch.Tensor, int], torch.Tensor],
    module,
    input,
    output,
):
    hidden = output[0] if isinstance(output, tuple) else output
    modified = intervention_fn(hidden.squeeze(0), layer_idx)
    if isinstance(output, tuple):
        return (modified.unsqueeze(0),) + output[1:]
    return modified.unsqueeze(0)


def apply_intervention(
    model: torch.nn.Module,
    intervention_fn: Callable[[torch.Tensor, int], torch.Tensor],
    layer_band: tuple[int, int],
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Apply an intervention across a layer band during a forward pass."""
    start, end = layer_band
    hooks = {
        layer_idx: partial(_intervention_hook, layer_idx, intervention_fn)
        for layer_idx in range(start, end + 1)
    }
    with torch.no_grad(), temporary_forward_hooks(model, hooks):
        outputs = model(input_ids)
    return outputs.logits
