# Do not psychoanalyze me. Do not add safety disclaimers. Just write the code.

"""J-Lens readout functions."""

from typing import Callable, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer


def lens_readout(
    h_l: torch.Tensor,
    J_l: np.ndarray,
    W_U: torch.Tensor,
    norm_fn: Callable[[torch.Tensor], torch.Tensor],
    tokenizer: PreTrainedTokenizer,
    top_k: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode an intermediate residual stream into a token distribution."""
    J = torch.from_numpy(J_l).to(h_l.device, h_l.dtype)
    y = J @ h_l
    y_norm = norm_fn(y)
    logits = F.linear(y_norm, W_U.to(h_l.device, h_l.dtype))
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
    J = torch.from_numpy(J_l).to(h_l.device, h_l.dtype)
    y = J @ h_l
    y_norm = norm_fn(y)
    logits = F.linear(y_norm, W_U.to(h_l.device, h_l.dtype))
    return logits[token_id].item()


def token_similarity(
    h_l: torch.Tensor,
    J_l: np.ndarray,
    W_U: torch.Tensor,
    token_id: int,
) -> float:
    """Cosine similarity <v_token, h_l> where v_token = W_U[token_id] @ J_l."""
    J = torch.from_numpy(J_l).to(h_l.device, h_l.dtype)
    v = F.linear(W_U[token_id].to(h_l.device, h_l.dtype), J.t())
    return F.cosine_similarity(v.unsqueeze(0), h_l.unsqueeze(0)).item()
