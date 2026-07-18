"""Sparse J-space decomposition."""

from __future__ import annotations

import numpy as np
import scipy.optimize
import torch


def _build_dictionary(V: torch.Tensor) -> np.ndarray:
    return V.detach().cpu().numpy()


def _resolve_V(
    V: torch.Tensor | None,
    J_l: np.ndarray | None,
    W_U: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Resolve a V dictionary either provided directly or via J_l and W_U."""
    if V is not None:
        return V.to(device, dtype)
    if J_l is None or W_U is None:
        raise ValueError("Either V or both J_l and W_U must be provided")
    J = torch.from_numpy(J_l).to(device, dtype)
    W = W_U.to(device, dtype)
    return torch.matmul(W, J.t())


def decompose_jspace(
    h: torch.Tensor,
    V: torch.Tensor | None = None,
    k: int = 25,
    non_negative: bool = True,
    random_seed: int | None = None,
    J_l: np.ndarray | None = None,
    W_U: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve min ||h - sum_j a_j v_j||_2 s.t. a_j >= 0, ||a||_0 <= k.

    V: [vocab, d_model] dictionary of J-lens vectors. Alternatively pass
       J_l [d_model, d_model] and W_U [vocab, d_model]; V is computed as
       W_U @ J_l.T.
    Returns: coefficients a [vocab], h_J [d_model], h_perp [d_model].
    """
    V_tensor = _resolve_V(V, J_l, W_U, h.device, h.dtype)
    h_np = h.detach().cpu().numpy()
    V_np = _build_dictionary(V_tensor)

    if non_negative:
        a = np.zeros(V_np.shape[0], dtype=np.float32)
        residual = h_np.copy()
        selected: list[int] = []
        for _ in range(k):
            corr = V_np @ residual
            for s in selected:
                corr[s] = -np.inf
            best = int(np.argmax(corr))
            if corr[best] <= 0:
                break
            selected.append(best)
            coeffs, _ = scipy.optimize.nnls(V_np[selected].T, h_np)
            residual = h_np - V_np[selected].T @ coeffs
            a[:] = 0.0
            a[selected] = coeffs
    else:
        from sklearn.linear_model import OrthogonalMatchingPursuit

        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=False)
        omp.fit(V_np.T, h_np)
        a = omp.coef_.astype(np.float32)

    a_t = torch.from_numpy(a).to(h.device, h.dtype)
    h_J = (a_t @ V_tensor).to(h.device, h.dtype)
    h_perp = h - h_J
    return a_t, h_J, h_perp


def jspace_occupancy(
    h: torch.Tensor,
    V: torch.Tensor | None = None,
    max_k: int = 100,
    threshold: float = 0.01,
    random_seed: int | None = None,
    J_l: np.ndarray | None = None,
    W_U: torch.Tensor | None = None,
) -> int:
    """Return smallest k where adding a (k+1)-th vector does not improve
    reconstruction more than a random control direction.

    V or the (J_l, W_U) pair are accepted as in decompose_jspace.
    """
    V_tensor = _resolve_V(V, J_l, W_U, h.device, h.dtype)
    rng = np.random.default_rng(random_seed)
    h_np = h.detach().cpu().numpy()
    V_np = _build_dictionary(V_tensor)
    residual = h_np.copy()
    prev_error = float(np.linalg.norm(residual))
    for k in range(max_k):
        corr = V_np @ residual
        best = int(np.argmax(np.abs(corr)))
        selected = [best]
        coeffs, _ = scipy.optimize.nnls(V_np[selected].T, h_np)
        new_residual = h_np - V_np[selected].T @ coeffs
        new_error = float(np.linalg.norm(new_residual))
        rand_dir = rng.standard_normal(V_np.shape[1]).astype(np.float32)
        rand_dir /= np.linalg.norm(rand_dir) + 1e-9
        rand_improve = max(
            0.0,
            prev_error - np.linalg.norm(h_np - (h_np @ rand_dir) * rand_dir),
        )
        rel_improve = (prev_error - new_error) / (prev_error + 1e-9)
        if rel_improve < threshold * (rand_improve / (prev_error + 1e-9)):
            return k
        prev_error = new_error
        residual = new_residual
    return max_k
