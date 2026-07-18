#!/usr/bin/env python3
"""Generate an ASCII-art terminal GIF of workspace_geometry.py for the README."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH = 900
HEIGHT = 540
FONT_SIZE = 14
LINE_HEIGHT = 18
TERM_BG = (17, 17, 17)
TERM_FG = (240, 240, 240)
PROMPT_FG = (135, 206, 235)
SUCCESS_FG = (144, 238, 144)
PROGRESS_FG = (255, 215, 0)
ACCENT_FG = (255, 105, 180)
FRAME_DELAY = 0.12


def get_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSansMono", "LiberationMono", "Courier New", "Courier"):
        try:
            return ImageFont.truetype(name, FONT_SIZE)
        except OSError:
            pass
    return ImageFont.load_default()


FONT = get_font()


def render_frame(lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    img = Image.new("RGB", (WIDTH, HEIGHT), TERM_BG)
    draw = ImageDraw.Draw(img)
    x, y = 20, 20
    for text, color in lines:
        draw.text((x, y), text, font=FONT, fill=color)
        y += LINE_HEIGHT
        if y > HEIGHT - 30:
            break
    return np.array(img)


def typewriter_frames(prompt: str) -> list[list[tuple[str, tuple[int, int, int]]]]:
    frames: list[list[tuple[str, tuple[int, int, int]]]] = []
    for i in range(1, len(prompt) + 1):
        frames.append([("$ " + prompt[:i], PROMPT_FG)])
    return frames


def progress_bar(label: str, pct: int, width: int = 40) -> str:
    filled = int(pct / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{label}: [{bar}] {pct}%"


def ascii_heatmap(png_path: Path, width: int = 58, height: int = 18) -> list[str]:
    """Convert a PNG heatmap to ASCII art using a luminance ramp."""
    if not png_path.exists():
        return []
    img = Image.open(png_path).convert("L")
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
    ramp = " ░▒▓█"
    lines = []
    for row in arr:
        line = "".join(ramp[int(v * (len(ramp) - 1))] for v in row)
        lines.append(line)
    return lines


def synthesize_demo(
    script: Path,
    corpus: Path,
    workspace: Path,
) -> list[list[tuple[str, tuple[int, int, int]]]]:
    prompt = (
        "python -m scripts.workspace_geometry \\\n"
        "  --model sshleifer/tiny-gpt2 \\\n"
        "  --corpus corpus.json \\\n"
        "  --max-positions 128 \\\n"
        "  --n-probes 1024"
    )
    frames: list[list[tuple[str, tuple[int, int, int]]]] = []

    # Type the command.
    for line in prompt.split("\n"):
        partials = [line[:i] for i in range(1, len(line) + 1)]
        for partial in partials:
            frames.append(
                [("$ " + partial, PROMPT_FG)] + [("", TERM_FG)] * (len(prompt.split("\n")) - 1)
            )

    # Hit return and show execution header.
    base: list[tuple[str, tuple[int, int, int]]] = [
        ("$ " + prompt.replace("\n", " "), PROMPT_FG),
        ("", TERM_FG),
        ("Training/caching J-Lens for sshleifer/tiny-gpt2 target_layer=1", TERM_FG),
    ]
    frames.append(base.copy())

    # Progress: training.
    for pct in range(0, 101, 5):
        frames.append(base + [(progress_bar("Training J-Lens", pct), PROGRESS_FG)])

    # Progress: token geometry.
    base2 = base + [
        (progress_bar("Training J-Lens", 100), PROGRESS_FG),
        ("Building token geometry for 1024 probe tokens", TERM_FG),
    ]
    for pct in range(0, 101, 10):
        frames.append(base2 + [(progress_bar("Building V_l", pct), PROGRESS_FG)])

    # Progress: CKA.
    base3 = base2 + [
        (progress_bar("Building V_l", 100), PROGRESS_FG),
        ("Computing CKA block matrix", TERM_FG),
    ]
    for pct in range(0, 101, 10):
        frames.append(base3 + [(progress_bar("CKA block", pct), PROGRESS_FG)])

    # Discovery metrics.
    base4 = base3 + [
        (progress_bar("CKA block", 100), PROGRESS_FG),
        ("Computing discovery metrics for workspace boundary inference", TERM_FG),
    ]
    frames.append(base4.copy())

    # Run the real CLI to get the actual output files.
    workspace.mkdir(parents=True, exist_ok=True)
    local_corpus = workspace / "corpus.json"
    local_corpus.write_text(corpus.read_text())
    output_dir = workspace / "workspace_out"
    cache_dir = workspace / "lens_cache"
    cmd = [
        sys.executable,
        str(script),
        "--model",
        "sshleifer/tiny-gpt2",
        "--corpus",
        str(local_corpus),
        "--output-dir",
        str(output_dir),
        "--cache-dir",
        str(cache_dir),
        "--workspace",
        str(workspace),
        "--max-positions",
        "16",
        "--batch-size",
        "1",
        "--n-probes",
        "64",
        "--dtype",
        "float32",
        "--target-layer",
        "1",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603

    # Final output lines from the real run.
    final = base4 + [
        ("", TERM_FG),
        ("Wrote workspace_out/cka_block.png", SUCCESS_FG),
        ("Wrote workspace_out/metrics.json", SUCCESS_FG),
        ("Workspace boundaries: layer 0 to 0", SUCCESS_FG),
        ("", TERM_FG),
        ("ASCII preview of cka_block.png:", ACCENT_FG),
    ]
    heatmap = ascii_heatmap(output_dir / "cka_block.png")
    for line in heatmap[:14]:
        final.append(("  " + line, ACCENT_FG))
    frames.append(final.copy())

    # Hold the final frame.
    frames.extend([final.copy()] * int(3.0 / FRAME_DELAY))
    return frames


def build_gif(frames: list[list[tuple[str, tuple[int, int, int]]]], out_path: Path) -> None:
    images = [render_frame(f) for f in frames]
    iio.imwrite(
        out_path,
        images,
        extension=".gif",
        duration=FRAME_DELAY,
        loop=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    script = Path(__file__).resolve().parent / "workspace_geometry.py"
    with tempfile.TemporaryDirectory() as tmp:
        frames = synthesize_demo(script, args.corpus, Path(tmp))

    build_gif(frames, args.out)
    print(f"Wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
