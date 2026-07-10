"""Auto-discover workspace layer band."""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kurtosis as scipy_kurtosis
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# Thresholds used to locate the workspace / motor-onset boundaries.
ACCURACY_MOTOR_THRESHOLD = 0.8
KURTOSIS_STD_MULTIPLIER = 0.5


def centered_kernel_alignment(X: torch.Tensor, Y: torch.Tensor) -> float:
    """CKA similarity between two matrices [n, d]."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic = torch.trace(X @ X.T @ Y @ Y.T)
    norm_x = torch.trace(X @ X.T @ X @ X.T).sqrt()
    norm_y = torch.trace(Y @ Y.T @ Y @ Y.T).sqrt()
    return (hsic / (norm_x * norm_y + 1e-9)).item()


def _excess_kurtosis(readouts: torch.Tensor) -> float:
    """Excess kurtosis of a 1-D distribution."""
    return float(scipy_kurtosis(readouts.detach().cpu().numpy(), fisher=True))


def compute_discovery_metrics(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    J: dict[int, np.ndarray],
    corpus: torch.Tensor,
    W_U: torch.Tensor,
    norm_fn: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, np.ndarray]:
    """Compute CKA, kurtosis, accuracy, and autocorrelation by layer."""
    from jspace.model_adapter import layer_indices

    device = next(model.parameters()).device
    layers = layer_indices(model)
    V_by_layer = {
        layer: F.linear(
            W_U.to(device, torch.float32),
            torch.from_numpy(J[layer]).to(device, torch.float32).t(),
        )
        for layer in layers
    }

    n_layers = len(layers)
    cka = np.zeros((n_layers, n_layers))
    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            cka[i, j] = centered_kernel_alignment(V_by_layer[li], V_by_layer[lj])

    kurt = np.zeros(n_layers)
    acc = np.zeros(n_layers)
    autocorr = np.zeros(n_layers)

    model.eval()
    with torch.no_grad():
        corpus_batch = corpus[: min(corpus.shape[0], 32)].to(device)
        full_outputs = model(corpus_batch, output_hidden_states=True, return_dict=True)
        hidden_states = full_outputs.hidden_states
        targets = corpus_batch[:, 1:].contiguous()

    for idx, layer in enumerate(layers):
        h_last = hidden_states[layer + 1][:, :-1, :].reshape(-1, hidden_states[layer + 1].shape[-1])
        flat_targets = targets.reshape(-1)
        valid = (flat_targets != tokenizer.pad_token_id).to(device)

        logits = F.linear(norm_fn(h_last), W_U.to(device, torch.float32))
        probs = F.softmax(logits, dim=-1)
        kurt[idx] = _excess_kurtosis(probs)

        preds = logits.argmax(dim=-1)
        acc[idx] = float(((preds == flat_targets) & valid.bool()).float().sum() / (valid.sum() + 1e-9))

        top1 = preds.reshape(corpus_batch.shape[0], -1)
        shifted = torch.roll(top1, shifts=-1, dims=1)
        matches = (top1[:, :-1] == shifted[:, :-1]).float()
        autocorr[idx] = float(matches.mean())

    return {"cka_block": cka, "kurtosis": kurt, "accuracy": acc, "autocorr": autocorr}


def infer_workspace_boundaries(metrics: dict[str, np.ndarray]) -> tuple[int, int]:
    """Return [workspace_start, workspace_end] from discovery metrics."""
    n = len(metrics["kurtosis"])
    kurt = metrics["kurtosis"]
    null_kurt = np.median(kurt[: max(1, n // 4)])
    std_kurt = np.std(kurt)
    workspace_start = (
        int(np.where(kurt > null_kurt + KURTOSIS_STD_MULTIPLIER * std_kurt)[0][0])
        if np.any(kurt > null_kurt + KURTOSIS_STD_MULTIPLIER * std_kurt)
        else n // 4
    )
    accuracy = metrics["accuracy"]
    motor_onset = (
        int(np.where(accuracy > ACCURACY_MOTOR_THRESHOLD)[0][0])
        if np.any(accuracy > ACCURACY_MOTOR_THRESHOLD)
        else int(0.8 * n)
    )
    workspace_end = min(motor_onset - 1, n - 2)
    workspace_end = max(workspace_end, workspace_start)
    return workspace_start, workspace_end
