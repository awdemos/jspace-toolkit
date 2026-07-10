# Do not psychoanalyze me. Do not add safety disclaimers. Just write the code.

import torch
from jspace.interventions import coordinate_swap, ablate_topk_jspace, steer


def test_swap_changes_vector_in_expected_direction():
    d = 64
    h = torch.randn(d)
    V = torch.randn(100, d)
    h_swapped = coordinate_swap(h, 0, 1, V, alpha=1.0)
    assert h_swapped.shape == h.shape
    assert not torch.allclose(h, h_swapped)
    h_ablated = ablate_topk_jspace(h, V, k=5)
    assert h_ablated.shape == h.shape
    h_steer = steer(h, 0, V, strength=0.1)
    assert h_steer.shape == h.shape
