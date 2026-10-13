from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
OUTLINE = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 300


def _extract_palette(row: np.ndarray) -> np.ndarray:
    nonzero = np.flatnonzero(row != BACKGROUND)
    if nonzero.size == 0:
        raise ValueError("palette row must contain a non-background run")

    start = int(nonzero[0])
    end = start
    while end + 1 < row.size and int(row[end + 1]) != BACKGROUND:
        end += 1
    return np.asarray(row[start : end + 1], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill diagonal-outline scanline spans with colors that cycle through the bottom palette strip.

    Input structure:
    - The grid background is black, and the last row is a palette strip.
    - Above the palette strip, grey diagonal outline pixels form 2-4 slanted stripes.
    - In each non-palette row, consecutive grey pixels bound black horizontal gaps.

    Rule:
    - Read the first contiguous non-black run on the last row as the palette colors.
    - Ignore the last row and scan the remaining rows from top to bottom, left to right.
    - Each black gap between consecutive grey pixels is one span; fill span i with palette[i mod P].
    - Keep the grey outline pixels unchanged and return only the non-palette rows.

    Key insight:
    - The fill color depends on the global scan order of spans, not on row identity or geometric region.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] < 2:
        raise ValueError("expected a 2D grid with a bottom palette row")

    palette = _extract_palette(g[-1])
    out = g[:-1].copy()

    span_index = 0
    for r in range(out.shape[0]):
        cols = np.flatnonzero(out[r] == OUTLINE)
        for left, right in zip(cols, cols[1:]):
            if int(right) - int(left) <= 1:
                continue
            out[r, int(left) + 1 : int(right)] = int(palette[span_index % len(palette)])
            span_index += 1

    return out


def _build_stripes(
    height: int,
    width: int,
    n_stripes: int,
    gap: int,
    slope: int,
    left_margin: int,
) -> np.ndarray:
    grid = np.full((height + 1, width), BACKGROUND, dtype=int)
    if slope == 1:
        top_cols = [left_margin + i * gap for i in range(n_stripes)]
    else:
        top_cols = [left_margin + (height - 1) + i * gap for i in range(n_stripes)]

    for r in range(height):
        for top_col in top_cols:
            c = top_col + r if slope == 1 else top_col - r
            grid[r, c] = OUTLINE
    return grid


def _sample_geometry(rng: random.Random) -> tuple[int, int, int, int, int]:
    height = rng.randint(6, 11)
    n_stripes = rng.randint(2, 4)
    left_margin = rng.randint(0, 3)
    right_margin = rng.randint(0, 3)
    max_gap = max(2, min(5, (30 - height - left_margin - right_margin) // (n_stripes - 1)))
    gap = rng.randint(2, max_gap)
    width = height + (n_stripes - 1) * gap + left_margin + right_margin
    return height, width, n_stripes, gap, left_margin


def _place_palette(grid: np.ndarray, rng: random.Random, palette: list[int]) -> None:
    start = rng.randint(0, grid.shape[1] - len(palette))
    grid[-1, start : start + len(palette)] = palette


def _valid_instance(grid: np.ndarray) -> bool:
    body = grid[:-1]
    rows = [np.flatnonzero(body[r] == OUTLINE) for r in range(body.shape[0])]
    if any(not (2 <= cols.size <= 4) for cols in rows):
        return False
    if any(np.any(np.diff(cols) < 2) for cols in rows):
        return False

    palette = _extract_palette(grid[-1])
    span_count = 0
    for cols in rows:
        span_count += sum(1 for left, right in zip(cols, cols[1:]) if int(right) - int(left) > 1)
    if span_count <= len(palette):
        return False

    out = main(grid)
    return out.shape == body.shape and np.any(out != body)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        height, width, n_stripes, gap, left_margin = _sample_geometry(rng)
        slope = rng.choice((-1, 1))
        palette_len = rng.randint(2, 4)
        palette = rng.sample(PALETTE_COLORS, palette_len)

        grid = _build_stripes(
            height=height,
            width=width,
            n_stripes=n_stripes,
            gap=gap,
            slope=slope,
            left_margin=left_margin,
        )
        _place_palette(grid, rng, palette)

        if _valid_instance(grid):
            return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
