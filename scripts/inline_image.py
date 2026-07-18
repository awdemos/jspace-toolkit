#!/usr/bin/env python3
"""Render a PNG as inline ANSI art for terminal demos."""

from __future__ import annotations

import argparse

from PIL import Image

RAMP = " ░▒▓█"


def _rgb_to_256(r: int, g: int, b: int) -> int:
    if r == g == b:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return 232 + ((r - 8) // 10)
    return 16 + (36 * (r // 51)) + (6 * (g // 51)) + (b // 51)


def render(path: str, width: int = 70) -> str:
    img = Image.open(path).convert("RGB")
    aspect = img.height / img.width
    height = max(1, int(width * aspect * 0.5))
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    lines = []
    for y in range(img.height):
        row = ""
        for x in range(img.width):
            r, g, b = img.getpixel((x, y))
            color = _rgb_to_256(r, g, b)
            row += f"\033[38;5;{color}m█\033[0m"
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
