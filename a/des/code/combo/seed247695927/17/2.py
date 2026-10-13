import random

import numpy as np


GRID_H_RANGE = (8, 18)
GRID_W_RANGE = (8, 18)
MAX_GENERATION_ATTEMPTS = 300
BRANCH_OFFSETS = (-1, 0, 1)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill rows with top-row palette columns, centered on the palette gap.

    Input structure:
    - The top row is a palette row with exactly one background gap.
    - The only foreground below the palette is a monochrome vertical marker bar.
    - The bar lies in the gap column or an adjacent column.

    Rule:
    - Clear the grid to background.
    - Starting at the bar's top row, copy the top palette row downward as a wedge:
      row k below the start copies columns [gap-k, gap+k], clipped to the grid.
    - The gap column is filled with the marker color.
    - Preserve the original vertical marker bar.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = _background_color(g)
    top = g[0].copy()
    gap_cols = np.flatnonzero(top == bg)
    assert len(gap_cols) == 1, "expected exactly one palette gap"
    gap_col = int(gap_cols[0])

    lower_fg = np.argwhere(g[1:] != bg)
    assert lower_fg.size > 0, "expected a marker bar below the palette"
    lower_fg[:, 0] += 1

    marker_colors = {int(g[r, c]) for r, c in lower_fg}
    assert len(marker_colors) == 1, "expected a monochrome marker bar"
    marker_color = marker_colors.pop()

    marker_cols = {int(c) for _, c in lower_fg}
    assert len(marker_cols) == 1, "expected a vertical marker bar"
    marker_col = marker_cols.pop()
    assert abs(marker_col - gap_col) <= 1, "marker bar must be at or next to the gap"

    marker_rows = sorted(int(r) for r, _ in lower_fg)
    start_row = marker_rows[0]
    assert marker_rows == list(range(start_row, marker_rows[-1] + 1)), "marker bar must be contiguous"

    out = np.full_like(g, bg)
    for r in range(start_row, h):
        radius = r - start_row
        c0 = max(0, gap_col - radius)
        c1 = min(w - 1, gap_col + radius)
        out[r, c0 : c1 + 1] = top[c0 : c1 + 1]
        out[r, gap_col] = marker_color

    out[marker_rows, marker_col] = marker_color
    return out


def _make_palette(width: int, bg: int, marker_color: int, gap_col: int, rng: random.Random) -> np.ndarray:
    palette = np.empty(width, dtype=int)
    usable = [c for c in range(10) if c not in (bg, marker_color)]
    rng.shuffle(usable)
    motif_len = rng.randint(2, min(5, len(usable)))
    motif = usable[:motif_len]
    for c in range(width):
        palette[c] = motif[c % motif_len]
    if width >= 6 and rng.random() < 0.5:
        perm = list(range(width))
        rng.shuffle(perm)
        palette = palette[perm]
    palette[gap_col] = bg
    return palette


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)

    bg = rng.randrange(10)
    marker_color_choices = [c for c in range(10) if c != bg]
    marker_color = rng.choice(marker_color_choices)

    offset = rng.choice(BRANCH_OFFSETS)
    if offset == -1:
        gap_col = rng.randint(1, w - 1)
    elif offset == 1:
        gap_col = rng.randint(0, w - 2)
    else:
        gap_col = rng.randint(0, w - 1)
    marker_col = gap_col + offset

    start_row = rng.randint(1, h - 3)
    bar_len = rng.randint(2, h - start_row)

    grid = np.full((h, w), bg, dtype=int)
    grid[0] = _make_palette(w, bg, marker_color, gap_col, rng)
    grid[start_row : start_row + bar_len, marker_col] = marker_color

    if len(np.flatnonzero(grid[0] == bg)) != 1:
        return None
    below = np.argwhere(grid[1:] != bg)
    if below.shape[0] != bar_len:
        return None

    expected = main(grid)
    if np.array_equal(expected, np.full_like(grid, bg)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
