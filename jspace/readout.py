"""J-Lens readout functions."""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from jspace import JSpaceError


def _project(
    h_l: torch.Tensor,
    J_l: np.ndarray,
    W_U: torch.Tensor,
    norm_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Apply J_l to h_l, normalize, and project onto the unembedding matrix."""
    J = torch.from_numpy(J_l).to(h_l.device, h_l.dtype)
    y = J @ h_l
    y_norm = norm_fn(y)
    return F.linear(y_norm, W_U.to(h_l.device, h_l.dtype))


def _check_token_id(token_id: int, vocab_size: int) -> None:
    if not 0 <= token_id < vocab_size:
        raise JSpaceError(
            f"token_id {token_id} out of range (vocab_size={vocab_size})"
        )


def lens_readout(
    h_l: torch.Tensor,
    J_l: np.ndarray,
    W_U: torch.Tensor,
    norm_fn: Callable[[torch.Tensor], torch.Tensor],
    tokenizer: PreTrainedTokenizer,
    top_k: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode an intermediate residual stream into a token distribution.

    Returns (top_k_token_ids, top_k_probabilities).  If top_k exceeds the
    vocabulary size it is clamped silently.
    """
    logits = _project(h_l, J_l, W_U, norm_fn)
    probs = F.softmax(logits, dim=-1)
    k = min(top_k, probs.shape[-1])
    topk = torch.topk(probs, k)
    return topk.indices, topk.values


def token_logit(
    h_l: torch.Tensor,
    J_l: np.ndarray,
    W_U: torch.Tensor,
    norm_fn: Callable[[torch.Tensor], torch.Tensor],
    token_id: int,
) -> float:
    """Return the logit for a single token_id."""
    logits = _project(h_l, J_l, W_U, norm_fn)
    _check_token_id(token_id, logits.shape[-1])
    return float(logits[token_id].item())


def token_similarity(
    h_l: torch.Tensor,
    J_l: np.ndarray,
    W_U: torch.Tensor,
    token_id: int,
) -> float:
    """Cosine similarity <v_token, h_l> where v_token = W_U[token_id] @ J_l."""
    _check_token_id(token_id, W_U.shape[0])
    J = torch.from_numpy(J_l).to(h_l.device, h_l.dtype)
    v = F.linear(W_U[token_id].to(h_l.device, h_l.dtype), J.t())
    return float(F.cosine_similarity(v.unsqueeze(0), h_l.unsqueeze(0)).item())
