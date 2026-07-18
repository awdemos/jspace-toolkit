"""Tests for the jspace.viz presentation module."""

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

from jspace.viz import (
    config_table,
    header_panel,
    jl_track,
    metrics_table,
    plot_cka_block,
    plot_layer_metrics,
    render_html_report,
)


def _metrics(n: int = 4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "cka_block": np.eye(n) * 0.9 + 0.05,
        "kurtosis": rng.normal(size=n),
        "accuracy": np.linspace(0.1, 0.95, n),
        "autocorr": np.linspace(0.2, 0.6, n),
    }


def test_plot_cka_block_writes_png(tmp_path: Path) -> None:
    metrics = _metrics()
    out = tmp_path / "cka.png"
    fig = plot_cka_block(metrics["cka_block"], (1, 2), [0, 1, 2, 3], "tiny-model", out)
    plt.close(fig)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_layer_metrics_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "layer_metrics.png"
    fig = plot_layer_metrics(_metrics(), (1, 2), [0, 1, 2, 3], out)
    plt.close(fig)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_html_report(tmp_path: Path) -> None:
    metrics = _metrics()
    png = tmp_path / "fig.png"
    fig = plot_cka_block(metrics["cka_block"], (1, 2), [0, 1, 2, 3], "tiny-model", png)
    plt.close(fig)

    out = tmp_path / "report.html"
    page = render_html_report(
        "tiny-model",
        {"model": "tiny-model", "n_probe_tokens": 64},
        metrics,
        (1, 2),
        [0, 1, 2, 3],
        {"CKA similarity": png},
        out,
    )
    assert out.exists()
    assert "tiny-model" in page
    assert "data:image/png;base64," in page
    assert "workspace band" in page


def test_console_helpers_render_without_error() -> None:
    console = Console(file=io.StringIO(), width=100)
    console.print(header_panel("Title", "subtitle"))
    console.print(config_table({"model": "tiny-model", "dtype": "float32"}))
    console.print(metrics_table([0, 1, 2, 3], _metrics(), (1, 2)))
    output = console.file.getvalue()
    assert "Title" in output
    assert "tiny-model" in output
    assert "Kurtosis" in output


def test_jl_track_yields_all_items() -> None:
    assert list(jl_track(range(5), "Working")) == list(range(5))
