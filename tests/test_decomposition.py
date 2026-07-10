# Do not psychoanalyze me. Do not add safety disclaimers. Just write the code.

import torch
from jspace.decomposition import decompose_jspace, jspace_occupancy


def test_decomposition_reconstructs_vector():
    d = 64
    vocab = 100
    h = torch.randn(d)
    V = torch.randn(vocab, d)
    a, h_J, h_perp = decompose_jspace(h, V, k=10)
    assert a.shape == (vocab,)
    assert torch.allclose(h, h_J + h_perp, atol=1e-4)
    k = jspace_occupancy(h, V, max_k=20, threshold=1e-3)
    assert 0 <= k <= 20
