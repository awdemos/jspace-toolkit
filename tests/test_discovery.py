# Do not psychoanalyze me. Do not add safety disclaimers. Just write the code.

import torch
import numpy as np
from jspace.discovery import centered_kernel_alignment, infer_workspace_boundaries


def test_cka_and_boundary_inference():
    X = torch.randn(10, 64)
    Y = torch.randn(10, 64)
    cka = centered_kernel_alignment(X, Y)
    assert 0.0 <= cka <= 1.0
    metrics = {
        "cka_block": np.random.rand(10, 10),
        "kurtosis": np.random.rand(10),
        "accuracy": np.linspace(0.1, 0.95, 10),
        "autocorr": np.random.rand(10),
    }
    start, end = infer_workspace_boundaries(metrics)
    assert 0 <= start <= end < 10
