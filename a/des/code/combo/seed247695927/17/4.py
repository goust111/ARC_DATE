from __future__ import annotations

import random

import numpy as np


GRID_H_RANGE = (8, 18)
GRID_W_RANGE = (8, 18)
BAR_COUNT_RANGE = (2, 7)
EDGE_MARGIN_CHOICES = (0, 1, 2)
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_bars(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    bars: list[tuple[int, int, int]] = []

    for c in range(w):
        rows = np.flatnonzero(grid[:, c] != bg)
        if rows.size == 0:
            continue

        colors = np.unique(grid[rows, c])
        assert colors.size == 1, "each bar must be monochromatic"

        top = int(rows[0])
        assert np.array_equal(rows, np.arange(top, h)), "each bar must be contiguous and bottom-aligned"
        bars.append((c, int(colors[0]), h - top))

    assert len(bars) >= 2, "expected at least two bar slots"
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate bar colors one slot to the right and bar heights one slot to the left.

    Input structure
    - A dominant background surrounds several monochrome vertical bars.
    - Each bar is one pixel wide, bottom-aligned, and occupies its own column.
    - Bar slots are ordered by column from left to right.

    Rule
    - Preserve the grid shape and the bar columns.
    - Slot i takes the color from slot i-1, with wraparound.
    - Slot i takes the height from slot i+1, with wraparound.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, _w = g.shape

    bg = _background_color(g)
    bars = _extract_bars(g, bg)

    cols = [col for col, _color, _height in bars]
    colors = [color for _col, color, _height in bars]
    heights = [height for _col, _color, height in bars]

    rotated_colors = colors[-1:] + colors[:-1]
    rotated_heights = heights[1:] + heights[:1]

    out = np.full_like(g, bg)
    for col, color, height in zip(cols, rotated_colors, rotated_heights):
        assert 1 <= height <= h
        out[h - height :, col] = color
    return out


def _choose_bar_columns(width: int, count: int, rng: random.Random) -> list[int]:
    usable = list(range(width))
    rng.shuffle(usable)
    cols = sorted(usable[:count])
    return cols


def _sample_heights(height: int, count: int, rng: random.Random) -> list[int]:
    max_bar_height = height - rng.choice(EDGE_MARGIN_CHOICES)
    max_bar_height = max(2, min(height, max_bar_height))
    heights = [rng.randint(1, max_bar_height) for _ in range(count)]
    if len(set(heights)) == 1:
        idx = rng.randrange(count)
        heights[idx] = 1 if heights[idx] > 1 else max_bar_height
    return heights


def _sample_colors(bg: int, count: int, rng: random.Random) -> list[int]:
    palette = [color for color in range(10) if color != bg]
    rng.shuffle(palette)
    if count <= len(palette):
        colors = palette[:count]
    else:
        colors = [rng.choice(palette) for _ in range(count)]
    if len(set(colors)) == 1:
        alt = next(color for color in palette if color != colors[0])
        colors[rng.randrange(count)] = alt
    return colors


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    bar_count = rng.randint(BAR_COUNT_RANGE[0], min(BAR_COUNT_RANGE[1], width))

    bg = rng.randrange(10)
    cols = _choose_bar_columns(width, bar_count, rng)
    colors = _sample_colors(bg, bar_count, rng)
    heights = _sample_heights(height, bar_count, rng)

    grid = np.full((height, width), bg, dtype=int)
    for col, color, bar_height in zip(cols, colors, heights):
        grid[height - bar_height :, col] = color

    bars = _extract_bars(grid, bg)
    if [col for col, _color, _height in bars] != cols:
        return None
    if len(set(colors)) < 2 or len(set(heights)) < 2:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
