import torch

from jspace.interventions import ablate_topk_jspace, coordinate_swap, steer


def test_swap_changes_vector_in_expected_direction():
    d = 64
    V = torch.randn(100, d)
    # Build h so it has non-zero coefficients on the swapped atoms.
    h = 2.0 * V[0] + 3.0 * V[1] + 0.5 * V[7]
    h_swapped = coordinate_swap(h, 0, 1, V, alpha=1.0)
    assert h_swapped.shape == h.shape
    assert not torch.allclose(h, h_swapped)
    expected = 3.0 * V[0] + 2.0 * V[1] + 0.5 * V[7]
    assert torch.allclose(h_swapped, expected, atol=1e-3)
    h_ablated = ablate_topk_jspace(h, V, k=5)
    assert h_ablated.shape == h.shape
    h_steer = steer(h, 0, V, strength=0.1)
    assert h_steer.shape == h.shape


def test_coordinate_swap_swaps_sparse_coefficients():
    """coordinate_swap must swap the J-space coefficients of h under the two token vectors."""
    d = 32
    V = torch.randn(50, d)
    V = V / (V.norm(dim=-1, keepdim=True) + 1e-9)
    h = 2.0 * V[3] + 5.0 * V[7] + 0.1 * V[12]
    source_token_id, target_token_id = 3, 7
    h_swapped = coordinate_swap(h, source_token_id, target_token_id, V, alpha=1.0)
    # After full swap, coefficient of source vector becomes 5.0 and target becomes 2.0.
    expected = 5.0 * V[source_token_id] + 2.0 * V[target_token_id] + 0.1 * V[12]
    assert torch.allclose(h_swapped, expected, atol=1e-4)
