"""Bar-histogram-to-concentric-ring transformation for ARC-style puzzles."""

from __future__ import annotations

import random
from pathlib import Path
import sys

import numpy as np

try:
    from arc_utils import Color, get_bg
except Exception:
    _ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers")
    if _ARC_UTILS_PATH.exists():
        sys.path.insert(0, str(_ARC_UTILS_PATH))
        from arc_utils import Color, get_bg
    else:
        class Color:
            BLACK = 0

        def get_bg(grid: np.ndarray) -> int:
            values, counts = np.unique(grid.astype(int), return_counts=True)
            return int(values[np.argmax(counts)])


MAX_SIZE = 30
MAX_TOTAL_HEIGHT = 14
MAX_GENERATE_ATTEMPTS = 300


def _extract_bottom_bars(grid: np.ndarray, bg: int) -> tuple[list[int], list[int], list[int]]:
    height, width = grid.shape
    bar_cols = np.where(grid[height - 1] != bg)[0]
    if bar_cols.size == 0:
        raise AssertionError("expected at least one bar touching the bottom row")
    if np.any(np.diff(bar_cols) == 1):
        raise AssertionError("adjacent foreground columns on the bottom row are not allowed")

    cols: list[int] = []
    heights: list[int] = []
    colors: list[int] = []
    for col in map(int, bar_cols):
        col_vals = grid[:, col]
        fg_rows = np.where(col_vals != bg)[0]
        if fg_rows.size == 0:
            raise AssertionError("bar column must contain foreground")

        if fg_rows[-1] != height - 1:
            raise AssertionError("bars must stand on the bottom row")
        if not np.array_equal(fg_rows, np.arange(fg_rows[0], fg_rows[-1] + 1, dtype=int)):
            raise AssertionError("bars must be solid vertical segments")

        color = int(col_vals[fg_rows[0]])
        if color == bg:
            raise AssertionError("bar color must be non-background")
        if not np.all((col_vals[fg_rows] == color)):
            raise AssertionError("a bar must be monochrome")

        cols.append(col)
        heights.append(int(fg_rows.size))
        colors.append(color)

    if len(set(colors)) != len(colors):
        raise AssertionError("bar colors must be distinct")
    return cols, heights, colors


def _draw_concentric_rings(size: int, bg: int, heights: list[int], colors: list[int]) -> np.ndarray:
    out = np.full((size, size), int(bg), dtype=int)
    if size == 0:
        return out

    rows = np.arange(size, dtype=int).reshape(-1, 1)
    cols = np.arange(size, dtype=int).reshape(1, -1)
    dist = np.minimum(np.minimum(rows, cols), np.minimum(size - 1 - rows, size - 1 - cols))

    start = 0
    for thickness, color in zip(heights, colors):
        stop = start + int(thickness)
        band = (dist >= start) & (dist < stop)
        out[band] = int(color)
        start = stop
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert a bottom-anchored vertical bar histogram into concentric square rings.

    Input structure:
    - The background is the dominant color.
    - The bottom row contains one or more width-1 vertical solid bars.
    - Each bar has its own color, bars are separated by at least one background column, and no other foreground exists.

    Rule:
    - Let bar heights from left to right be h1, h2, ..., hk.
    - Create an output square of size N = 2 * (h1 + h2 + ... + hk).
    - Fill concentric square layers from outside to inside.
    - Layer i has thickness hi and color of the i-th bar from left to right.

    Key insight:
    - The bottom-side bars are a radial codebook: bar order gives outer-to-inner ring order, and bar heights give ring thicknesses.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")
    if max(g.shape) > MAX_SIZE:
        raise AssertionError("input grid exceeds ARC size limit")

    bg = get_bg(g)
    _, heights, colors = _extract_bottom_bars(g, bg)

    total = int(sum(heights))
    if total <= 0:
        raise AssertionError("bar heights must sum to a positive value")
    size = 2 * total
    if size > MAX_SIZE:
        raise AssertionError("output size exceeds ARC size limit")

    return _draw_concentric_rings(size, bg, heights, colors)


def _split_height(total: int, parts: int) -> list[int]:
    if parts <= 1:
        return [total]
    if total < parts:
        raise AssertionError("total height too small for requested bar count")

    cuts = sorted(random.sample(range(1, total), parts - 1))
    segments = [cuts[0]]
    segments.extend(int(cuts[i] - cuts[i - 1]) for i in range(1, len(cuts)))
    segments.append(total - cuts[-1])
    return segments


def _build_bar_columns(num_bars: int) -> tuple[list[int], int]:
    min_width = 2 * num_bars - 1
    if min_width > MAX_SIZE:
        raise AssertionError("cannot fit bars in the canvas")

    extra = random.randint(0, MAX_SIZE - min_width)
    left_pad = random.randint(0, extra)
    extra -= left_pad
    right_pad = random.randint(0, extra)
    extra -= right_pad

    gaps = [1] * max(0, num_bars - 1)
    if num_bars > 1:
        for _ in range(extra):
            idx = random.randrange(num_bars - 1)
            gaps[idx] += 1

    cols = [left_pad]
    for gap in gaps:
        cols.append(cols[-1] + gap + 1)

    width = cols[-1] + right_pad + 1
    if width < 2:
        width = 2

    return cols, width


def _valid_bg_height(total_height: int, width: int) -> int:
    min_height = max(1, (2 * total_height) // width + 1)
    return min_height


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        try:
            num_bars = random.randint(1, min(5, len(Color.NOT_BLACK), MAX_SIZE // 2))
            total_height = random.randint(num_bars, MAX_TOTAL_HEIGHT)
            heights = _split_height(total_height, num_bars)

            bg = int(random.choice(Color.ALL_COLORS))
            palette = [int(c) for c in Color.ALL_COLORS if int(c) != bg]
            bar_colors = random.sample(palette, num_bars)

            cols, width = _build_bar_columns(num_bars)
            min_height = _valid_bg_height(total_height, width)
            if min_height > MAX_SIZE:
                raise AssertionError("canvas too small for dominant background")

            height = random.randint(min_height, MAX_SIZE)
            grid = np.full((height, width), bg, dtype=int)

            for col, h, color in zip(cols, heights, bar_colors):
                grid[height - h :, col] = color

            # Ensure the sample strictly follows the contract before returning it.
            _ = main(grid)
            return grid
        except Exception:
            continue

    raise RuntimeError("failed to generate valid input")
