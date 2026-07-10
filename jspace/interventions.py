# Do not psychoanalyze me. Do not add safety disclaimers. Just write the code.

"""Causal interventions in J-space."""

from typing import Callable, List, Optional, Tuple
import torch
import torch.nn.functional as F


def _token_vector(token_id: int, V: torch.Tensor) -> torch.Tensor:
    return V[token_id]


def coordinate_swap(
    h: torch.Tensor,
    source_token_id: int,
    target_token_id: int,
    V: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Swap lens coordinates between two concept tokens while preserving h_perp."""
    v_source = _token_vector(source_token_id, V)
    v_target = _token_vector(target_token_id, V)
    M = torch.stack([v_source, v_target], dim=1)
    c = torch.linalg.lstsq(M, h).solution
    c_swapped = torch.stack([c[1], c[0]]) * alpha + c * (1 - alpha)
    delta = M @ (c_swapped - c)
    return h + delta


def ablate_topk_jspace(
    h: torch.Tensor,
    V: torch.Tensor,
    k: int = 10,
    exclude_output_tokens: bool = True,
    clean_top10: Optional[List[int]] = None,
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


def apply_intervention(
    model: torch.nn.Module,
    intervention_fn: Callable[[torch.Tensor, int], torch.Tensor],
    layer_band: Tuple[int, int],
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Apply an intervention across a layer band during a forward pass."""
    from jspace.model_adapter import _layer_name

    handles = []
    start, end = layer_band

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            modified = intervention_fn(hidden.squeeze(0), layer_idx)
            if isinstance(output, tuple):
                return (modified.unsqueeze(0),) + output[1:]
            return modified.unsqueeze(0)

        return hook

    for layer_idx in range(start, end + 1):
        _, block = _layer_name(model, layer_idx)
        handles.append(block.register_forward_hook(make_hook(layer_idx)))

    with torch.no_grad():
        outputs = model(input_ids)

    for h in handles:
        h.remove()

    return outputs.logits
