"""Tests for workspace discovery: CKA/boundaries and attention-mask validation."""

import numpy as np
import pytest
import torch

from jspace import JSpaceError
from jspace.discovery import (
    centered_kernel_alignment,
    compute_discovery_metrics,
    infer_workspace_boundaries,
)


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


def test_discovery_metrics_rejects_mismatched_mask():
    model = torch.nn.Linear(4, 4)
    corpus = torch.ones(1, 3, dtype=torch.long)
    bad_mask = torch.ones(1, 4, dtype=torch.long)
    with pytest.raises(JSpaceError, match="attention_mask"):
        compute_discovery_metrics(
            model,
            object(),
            {},
            corpus,
            torch.ones(4, 4),
            lambda x: x,
            layers=[0],
            attention_mask=bad_mask,
        )
