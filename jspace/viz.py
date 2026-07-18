"""Presentation layer: rich terminal output, matplotlib figures, HTML report.

All visual output of the toolkit lives here so the CLI scripts and the demo
notebook share one consistent visual language.
"""

from __future__ import annotations

import base64
import html
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Shared visual theme.
BG = "#0d1117"
PANEL = "#161b22"
FG = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#58e6d9"
ACCENT2 = "#f0b72f"
CMAP = "magma"

_console: Console | None = None


def get_console() -> Console:
    """Return the shared rich console."""
    global _console
    if _console is None:
        _console = Console(highlight=False)
    return _console


def header_panel(title: str, subtitle: str = "") -> Panel:
    """Banner panel shown at the start of a CLI run."""
    body = f"[bold {ACCENT}]{title}[/]"
    if subtitle:
        body += f"\n[{MUTED}]{subtitle}[/]"
    return Panel(body, border_style=ACCENT, padding=(1, 3))


def config_table(config: Mapping[str, object]) -> Table:
    """Two-column key/value table describing a run configuration."""
    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2))
    table.add_column(style=f"bold {ACCENT}", justify="right")
    table.add_column(style=FG)
    for key, value in config.items():
        table.add_row(str(key), str(value))
    return table


def metrics_table(
    layers: Sequence[int],
    metrics: Mapping[str, np.ndarray],
    band: tuple[int, int],
) -> Table:
    """Per-layer discovery metrics with the workspace band highlighted."""
    start, end = band
    table = Table(title="Per-layer discovery metrics", title_style=f"bold {ACCENT}")
    table.add_column("Layer", justify="right", style=f"bold {FG}")
    table.add_column("Kurtosis", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Autocorr", justify="right")
    table.add_column("Workspace", justify="center")
    for idx, layer in enumerate(layers):
        in_band = start <= idx <= end
        acc = float(metrics["accuracy"][idx])
        acc_style = _scale_style(acc)
        row = [
            str(layer),
            f"{float(metrics['kurtosis'][idx]):.2f}",
            f"[{acc_style}]{acc:.3f}[/]",
            f"{float(metrics['autocorr'][idx]):.3f}",
            f"[bold {ACCENT2}]◆[/]" if in_band else "",
        ]
        table.add_row(*row, style="on #161b22" if in_band else None)
    return table


def _scale_style(value: float) -> str:
    """Map a [0, 1] value onto a red -> amber -> green style."""
    if value >= 0.8:
        return "bold green"
    if value >= 0.5:
        return ACCENT2
    return "red"


def jl_track(iterable: Iterable, description: str) -> Iterable:
    """Wrap an iterable in a rich progress bar (replaces tqdm)."""
    progress = Progress(
        SpinnerColumn(style=ACCENT),
        TextColumn(f"[bold {ACCENT}]{description}"),
        BarColumn(bar_width=None, complete_style=ACCENT, finished_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=get_console(),
        transient=True,
    )
    total = len(iterable) if hasattr(iterable, "__len__") else None
    with progress:
        task = progress.add_task(description, total=total)
        for item in iterable:
            yield item
            progress.advance(task)


def _style_ax(ax) -> None:
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def plot_cka_block(
    cka: np.ndarray,
    boundaries: tuple[int, int],
    layers: Sequence[int],
    model_name: str,
    out_path: Path | None = None,
) -> Figure:
    """Render the CKA layer-by-layer heatmap with a workspace overlay."""
    start, end = boundaries
    fig, ax = plt.subplots(figsize=(8.5, 7.5), facecolor=BG, layout="constrained")
    _style_ax(ax)

    im = ax.imshow(cka, cmap=CMAP, vmin=0.0, vmax=1.0)

    if 0 <= start < len(layers) and 0 <= end < len(layers):
        rect = plt.Rectangle(
            (start - 0.5, start - 0.5),
            end - start + 1,
            end - start + 1,
            linewidth=2.5,
            edgecolor=ACCENT2,
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)

    tick_step = max(1, len(layers) // 8)
    ticks = np.arange(0, len(layers), tick_step)
    ax.set_xticks(ticks, [str(layers[i]) for i in ticks])
    ax.set_yticks(ticks, [str(layers[i]) for i in ticks])
    ax.set_xlabel("Source layer")
    ax.set_ylabel("Target layer")
    ax.set_title(
        f"J-Lens workspace geometry — {model_name}\nCKA similarity between layer outputs",
        fontsize=13,
        pad=12,
    )

    if len(layers) <= 16:
        for i in range(len(layers)):
            for j in range(len(layers)):
                color = "#1a1a1a" if cka[i, j] > 0.55 else FG
                ax.text(
                    j,
                    i,
                    f"{cka[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="CKA similarity")
    cbar.ax.yaxis.label.set_color(FG)
    cbar.ax.tick_params(colors=MUTED)
    cbar.outline.set_color(MUTED)

    fig.text(
        0.99,
        0.005,
        f"workspace band: layers {start}–{end}",
        ha="right",
        va="bottom",
        color=ACCENT2,
        fontsize=9,
    )

    if out_path is not None:
        fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    return fig


def plot_layer_metrics(
    metrics: Mapping[str, np.ndarray],
    boundaries: tuple[int, int],
    layers: Sequence[int] | None = None,
    out_path: Path | None = None,
) -> Figure:
    """Render per-layer kurtosis / accuracy / autocorrelation with band shading."""
    start, end = boundaries
    n = len(metrics["kurtosis"])
    x = list(layers) if layers is not None else list(range(n))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=BG, layout="constrained")
    panels = [
        ("kurtosis", "Excess kurtosis", ACCENT),
        ("accuracy", "Next-token accuracy proxy", "#7ee787"),
        ("autocorr", "Top-1 token autocorrelation", ACCENT2),
    ]
    for ax, (key, title, color) in zip(axes, panels, strict=True):
        _style_ax(ax)
        ax.axvspan(start - 0.5, end + 0.5, color=ACCENT2, alpha=0.12, lw=0)
        ax.plot(x, metrics[key], color=color, marker="o", markersize=4, linewidth=1.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Layer")
        ax.grid(True, color=MUTED, alpha=0.15, linewidth=0.6)
        ax.margins(x=0.02)

    fig.suptitle("Workspace discovery metrics", color=FG, fontsize=13)
    fig.text(
        0.995,
        0.985,
        f"shaded band: inferred workspace (layers {start}–{end})",
        ha="right",
        va="top",
        color=ACCENT2,
        fontsize=9,
    )

    if out_path is not None:
        fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    return fig


_CSS = (
    """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem clamp(1rem, 6vw, 4rem);
  background: __BG__; color: __FG__;
  font-family: "Inter", "Segoe UI", system-ui, sans-serif;
  line-height: 1.55;
}
header { border-bottom: 1px solid #30363d; padding-bottom: 1rem; margin-bottom: 1.5rem; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; color: __ACCENT__; }
h1 small { color: __MUTED__; font-weight: 400; font-size: .95rem; }
h2 { font-size: 1.05rem; color: __ACCENT__; margin-top: 2rem; }
.cards { display: flex; flex-wrap: wrap; gap: .75rem; margin: 1rem 0; }
.card {
  background: __PANEL__; border: 1px solid #30363d; border-radius: 10px;
  padding: .8rem 1.2rem; min-width: 9rem;
}
.card .value { font-size: 1.4rem; font-weight: 700; color: __ACCENT2__; }
.card .label { font-size: .75rem; color: __MUTED__; text-transform: uppercase; letter-spacing: .06em; }
table { border-collapse: collapse; width: 100%; font-size: .85rem; }
th, td { padding: .35rem .7rem; border-bottom: 1px solid #21262d; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { color: __ACCENT__; font-weight: 600; }
tr.band td { background: rgba(240, 183, 47, .08); }
tr.band td:first-child { color: __ACCENT2__; font-weight: 700; }
.figures { display: flex; flex-wrap: wrap; gap: 1.25rem; align-items: flex-start; }
.figures figure { margin: 0; flex: 1 1 22rem; }
.figures img { width: 100%; border: 1px solid #30363d; border-radius: 10px; }
figcaption { color: __MUTED__; font-size: .8rem; margin-top: .4rem; text-align: center; }
footer { margin-top: 2.5rem; color: __MUTED__; font-size: .75rem; border-top: 1px solid #30363d; padding-top: .8rem; }
""".replace("__BG__", BG)
    .replace("__PANEL__", PANEL)
    .replace("__FG__", FG)
    .replace("__MUTED__", MUTED)
    .replace("__ACCENT2__", ACCENT2)
    .replace("__ACCENT__", ACCENT)
)


def _embed_png(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def render_html_report(
    model_name: str,
    config: Mapping[str, object],
    metrics: Mapping[str, np.ndarray],
    boundaries: tuple[int, int],
    layers: Sequence[int],
    images: Mapping[str, Path],
    out_path: Path,
) -> str:
    """Render a self-contained HTML report (inline CSS + base64 PNGs).

    Returns the HTML string and writes it to ``out_path``.
    """
    start, end = boundaries
    name = html.escape(model_name)
    cka = np.asarray(metrics["cka_block"])

    cards = [
        (f"{start} – {end}", "workspace band"),
        (f"{float(cka.mean()):.3f}", "mean CKA"),
        (f"{float(cka.max()):.3f}", "max CKA"),
        (str(len(layers)), "layers"),
    ]
    if "n_probe_tokens" in config:
        cards.append((str(config["n_probe_tokens"]), "probe tokens"))
    cards_html = "".join(
        f'<div class="card"><div class="value">{v}</div><div class="label">{k}</div></div>'
        for v, k in cards
    )

    config_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in config.items()
    )

    figures_html = "".join(
        f'<figure><img src="{_embed_png(path)}" alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        for caption, path in images.items()
    )

    table_rows = []
    for idx, layer in enumerate(layers):
        in_band = start <= idx <= end
        cls = ' class="band"' if in_band else ""
        table_rows.append(
            f"<tr{cls}><td>Layer {layer}</td>"
            f"<td>{float(metrics['kurtosis'][idx]):.2f}</td>"
            f"<td>{float(metrics['accuracy'][idx]):.3f}</td>"
            f"<td>{float(metrics['autocorr'][idx]):.3f}</td>"
            f"<td>{'◆' if in_band else ''}</td></tr>"
        )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>J-Space workspace report — {name}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>J-Space workspace report <small>— {name}</small></h1>
</header>

<section class="cards">{cards_html}</section>

<h2>Configuration</h2>
<table><tbody>{config_rows}</tbody></table>

<h2>Figures</h2>
<div class="figures">{figures_html}</div>

<h2>Per-layer metrics</h2>
<table>
<thead><tr><th>Layer</th><th>Kurtosis</th><th>Accuracy</th><th>Autocorr</th><th>Workspace</th></tr></thead>
<tbody>{"".join(table_rows)}</tbody>
</table>

<footer>Generated by jspace-toolkit · workspace band layers {start}–{end}</footer>
</body>
</html>
"""
    out_path.write_text(page)
    return page
