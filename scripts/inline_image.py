#!/usr/bin/env python3
"""Render a PNG as inline ANSI art for terminal demos."""

from __future__ import annotations

import argparse

from PIL import Image

RAMP = " ░▒▓█"


def render(path: str, width: int = 70) -> str:
    img = Image.open(path).convert("L")
    aspect = img.height / img.width
    height = max(1, int(width * aspect * 0.5))
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    lines = []
    for y in range(img.height):
        row = ""
        for x in range(img.width):
            val = img.getpixel((x, y)) / 255.0
            idx = int(val * (len(RAMP) - 1))
            row += RAMP[idx]
        lines.append(row)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--width", type=int, default=70)
    args = parser.parse_args()
    print(render(args.image, args.width))


if __name__ == "__main__":
    main()
