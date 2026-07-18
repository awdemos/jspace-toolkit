import numpy as np
import torch

from jspace.decomposition import decompose_jspace, jspace_occupancy


def test_decomposition_reconstructs_vector():
    d = 64
    vocab = 100
    h = torch.randn(d)
    V = torch.randn(vocab, d)
    a, h_J, h_perp = decompose_jspace(h, V=V, k=10)
    assert a.shape == (vocab,)
    assert torch.allclose(h, h_J + h_perp, atol=1e-4)
    k = jspace_occupancy(h, V=V, max_k=20, threshold=1e-3)
    assert 0 <= k <= 20


def test_decomposition_accepts_j_l_and_w_u():
    d = 64
    vocab = 100
    h = torch.randn(d)
    J_l = np.eye(d).astype(np.float32)
    W_U = torch.randn(vocab, d)
    a, h_J, h_perp = decompose_jspace(h, J_l=J_l, W_U=W_U, k=10)
    assert a.shape == (vocab,)
    assert torch.allclose(h, h_J + h_perp, atol=1e-4)


def test_non_negative_decomposition_returns_non_negative_coefficients():
    """Default non_negative=True must yield a non-negative coefficient vector."""
    d = 64
    vocab = 100
    V = torch.randn(vocab, d)
    V = V / (V.norm(dim=-1, keepdim=True) + 1e-9)
    target = torch.arange(5, dtype=torch.float32)
    h = (target @ V[:5]).to(torch.float32)
    a, h_J, h_perp = decompose_jspace(h, V=V, k=5)
    assert torch.all(a >= -1e-6)
    assert torch.allclose(h, h_J + h_perp, atol=1e-3)


def test_non_negative_decomposition_matches_omp_support():
    """Non-negative OMP should reconstruct a sparse non-negative target."""
    d = 32
    V = torch.randn(50, d)
    V = V / (V.norm(dim=-1, keepdim=True) + 1e-9)
    coeffs = torch.abs(torch.randn(5))
    support = [3, 7, 12, 20, 41]
    h = sum(c * V[i] for c, i in zip(coeffs, support, strict=True))
    a, h_J, h_perp = decompose_jspace(h, V=V, k=5)
    assert torch.all(a >= -1e-6)
    assert h_perp.norm().item() < 1e-2


def test_non_negative_decomposition_rejects_negative_atoms():
    """Non-negative OMP must not activate atoms whose correlation is non-positive."""
    d = 32
    V = torch.randn(50, d)
    V = V / (V.norm(dim=-1, keepdim=True) + 1e-9)
    # h points opposite to atom 0, so atom 0 should never be selected.
    h = -2.0 * V[0] + 1.5 * V[3]
    a, h_J, h_perp = decompose_jspace(h, V=V, k=5)
    assert a[0].item() <= 1e-6
    assert torch.all(a >= -1e-6)
    residual_norm = h_perp.norm().item()
    assert residual_norm < h.norm().item()


def test_non_negative_coeffs_match_nnls_refit():
    """Selected coefficients should equal a non-negative least-squares refit over the support."""
    import scipy.optimize

    d = 32
    V = torch.randn(50, d)
    V = V / (V.norm(dim=-1, keepdim=True) + 1e-9)
    coeffs = torch.tensor([2.0, 0.5, 1.5, 0.3, 1.0], dtype=torch.float32)
    support = [4, 11, 22, 33, 44]
    h = sum(c * V[i] for c, i in zip(coeffs, support, strict=True))

    a, _, _ = decompose_jspace(h, V=V, k=5)
    recovered_support = [i for i, val in enumerate(a) if val > 1e-3]
    V_np = V.detach().cpu().numpy()
    h_np = h.detach().cpu().numpy()
    expected, _ = scipy.optimize.nnls(V_np[recovered_support].T, h_np)
    assert torch.allclose(
        a[recovered_support].float(),
        torch.from_numpy(expected).float(),
        atol=1e-4,
    )
