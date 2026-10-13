from __future__ import annotations

import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each bottom-marked vertical column by propagating the next colored pixel below upward through background cells.

    Input structure
    - The grid has a dominant background color.
    - One or more non-background pixels in the bottom row mark the vertical columns to complete.
    - Each marked column contains colored pixels, including its bottom marker, that define stacked vertical color bands.

    Rule
    - Keep all original pixels unchanged.
    - In each marked column, scan from bottom to top while remembering the most recent non-background color.
    - Replace every background pixel in that column with the remembered color, so each colored pixel extends upward until another colored pixel is reached.

    Key insight
    - The bottom-row markers identify the only columns that should be rewritten; the scattered colors in those columns are band endpoints, not isolated objects.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    H, W = g.shape

    marked_cols = [c for c in range(W) if int(g[H - 1, c]) != bg]
    assert marked_cols, "expected at least one bottom-row column marker"

    for c in marked_cols:
        current = int(g[H - 1, c])
        assert current != bg, "marked column must have a non-background bottom pixel"
        for r in range(H - 1, -1, -1):
            color = int(g[r, c])
            if color != bg:
                current = color
            else:
                out[r, c] = current

    return out

# --------------- generator (o5x) ---------------

from itertools import combinations

# Canvas, palette, and retries.
BG_COLOR = 0
GRID_SIZE_RANGE = (7, 18)
MARKER_COLOR = 5
NON_MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_RETRIES = 300

# Marked-column and band structure.
MARKED_COL_COUNT_RANGE = (1, 4)
MARKED_COL_SPACING = 3
BAND_COUNT_RANGE = (2, 5)
BAND_HEIGHT_MIN, BAND_HEIGHT_MAX = 1, 13

# Sparse input pixels and noise.
MAX_INPUT_PIXELS_PER_MARKED_COL = 5
MIN_EXTRA_BAND_HEIGHT, EXTRA_PIXEL_PROB = 3, 0.11
MAX_NOISE_DENSITY = 0.17

def _gen_marked_columns(n: int, k: int, rng) -> list[int]:
    choices = [
        cols
        for cols in combinations(range(n), k)
        if all(b - a >= MARKED_COL_SPACING for a, b in zip(cols, cols[1:]))
    ]
    return list(choices[int(rng.integers(0, len(choices)))])


def _gen_band_heights(total: int, parts: int, rng) -> list[int]:
    choices: list[tuple[int, ...]] = []

    def rec(remaining: int, parts_left: int, picked: list[int]) -> None:
        if parts_left == 1:
            if BAND_HEIGHT_MIN <= remaining <= BAND_HEIGHT_MAX:
                choices.append(tuple(picked + [remaining]))
            return
        lo = max(BAND_HEIGHT_MIN, remaining - BAND_HEIGHT_MAX * (parts_left - 1))
        hi = min(BAND_HEIGHT_MAX, remaining - BAND_HEIGHT_MIN * (parts_left - 1))
        for h in range(lo, hi + 1):
            rec(remaining - h, parts_left - 1, picked + [h])

    rec(total, parts, [])
    return list(choices[int(rng.integers(0, len(choices)))])


def _render_marked_column(inp: np.ndarray, out: np.ndarray, c: int, rng) -> None:
    """Build one completed output column and its sparse input samples."""
    band_count = int(rng.integers(*BAND_COUNT_RANGE))
    colors = list(rng.choice(NON_MARKER_COLORS, size=band_count - 1, replace=False))
    colors.append(MARKER_COLOR)
    heights = _gen_band_heights(out.shape[0], band_count, rng)

    row = 0
    eligible_extra_rows: list[int] = []
    for color, height in zip(colors, heights):
        start = row
        end = row + height - 1
        out[start : end + 1, c] = int(color)
        inp[end, c] = int(color)
        if color != MARKER_COLOR and height >= MIN_EXTRA_BAND_HEIGHT:
            eligible_extra_rows.append(int(rng.integers(start + 1, end)))
        row = end + 1

    extra_limit = max(0, MAX_INPUT_PIXELS_PER_MARKED_COL - band_count)
    rng.shuffle(eligible_extra_rows)
    for r in eligible_extra_rows[:extra_limit]:
        if rng.random() < EXTRA_PIXEL_PROB:
            inp[r, c] = out[r, c]


def _scatter_noise(inp: np.ndarray, out: np.ndarray, marked_cols: list[int], rng) -> None:
    """Add same-color noise to input and expected output in unmarked columns."""
    n = inp.shape[0]
    noise_cells = [(r, c) for c in range(n) if c not in marked_cols for r in range(n - 1)]
    max_noise = int(MAX_NOISE_DENSITY * len(noise_cells))
    noise_count = int(rng.integers(0, max_noise + 1)) if max_noise else 0
    if not noise_count:
        return
    picks = rng.choice(len(noise_cells), size=noise_count, replace=False)
    for idx in picks:
        r, c = noise_cells[int(idx)]
        color = int(rng.choice(NON_MARKER_COLORS))
        inp[r, c] = out[r, c] = color


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(*GRID_SIZE_RANGE))
    out = np.full((n, n), BG_COLOR, dtype=int)
    inp = np.full((n, n), BG_COLOR, dtype=int)

    k = int(rng.integers(*MARKED_COL_COUNT_RANGE))
    marked_cols = _gen_marked_columns(n, k, rng)
    for c in marked_cols:
        _render_marked_column(inp, out, c, rng)
    _scatter_noise(inp, out, marked_cols, rng)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        if (grid := _gen_attempt()) is not None:
            return grid
    raise RuntimeError("failed to generate a valid 17b80ad2 input")
