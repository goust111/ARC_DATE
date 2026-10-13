import random
from typing import Optional

import numpy as np


BG = 0
MARKER = 5
STRIPE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MIN_STRIPES = 3
MAX_STRIPES = 5
MIN_GAP = 3
MAX_GAP = 6
MIN_WIDTH = 6
MAX_WIDTH = 30
GEN_ATTEMPTS = 300


def _dominant_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _stripe_rows(grid: np.ndarray, bg: int) -> list[int]:
    rows: list[int] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if int(row[0]) != bg and bool(np.all(row == row[0])):
            rows.append(int(r))
    return rows


def _marker_positions(grid: np.ndarray, bg: int, stripe_rows: list[int]) -> list[tuple[int, int]]:
    stripe_set = set(stripe_rows)
    markers: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        if r in stripe_set:
            continue
        cols = np.flatnonzero(grid[r] != bg)
        for c in cols:
            markers.append((int(r), int(c)))
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cross-propagate the colors of adjacent horizontal bands through each grey marker.

    Input structure:
    - The background is black.
    - Several distinct 1-row horizontal color bands span the full grid width.
    - Consecutive bands are separated by black gaps that are at least 3 rows tall.
    - Each gap contains exactly one grey marker pixel, and grey is different from every band color.

    Rule:
    - For each grey marker, find the nearest full-width band above it and the nearest full-width band below it.
    - Paint the cells strictly above the marker in that column, up to the row just below the upper band, with the lower band's color.
    - Paint the marker itself and the cells strictly below it in that column, down to the row just above the lower band, with the upper band's color.
    - Keep the band rows unchanged and leave all other black cells unchanged.

    Key insight:
    - The grey marker is a swap hub: the upper band's color travels downward while the lower band's color travels upward, and they meet at the marker column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    stripe_rows = _stripe_rows(g, bg)
    assert len(stripe_rows) >= 2

    stripe_colors = {r: int(g[r, 0]) for r in stripe_rows}
    markers = _marker_positions(g, bg, stripe_rows)

    out = g.copy()
    for r, c in markers:
        above = [sr for sr in stripe_rows if sr < r]
        below = [sr for sr in stripe_rows if sr > r]
        assert above and below
        upper_row = max(above)
        lower_row = min(below)
        upper_color = stripe_colors[upper_row]
        lower_color = stripe_colors[lower_row]

        if upper_row + 1 < r:
            out[upper_row + 1 : r, c] = lower_color
        out[r, c] = upper_color
        if r + 1 < lower_row:
            out[r + 1 : lower_row, c] = upper_color

    return out


def _build_grid(
    width: int,
    stripe_colors: list[int],
    gap_heights: list[int],
    marker_rows_in_gap: list[int],
    marker_cols: list[int],
) -> np.ndarray:
    height = len(stripe_colors) + sum(gap_heights)
    grid = np.full((height, width), BG, dtype=int)

    row = 0
    grid[row, :] = stripe_colors[0]
    for gap_idx, gap_height in enumerate(gap_heights):
        gap_top = row + 1
        marker_row = gap_top + marker_rows_in_gap[gap_idx]
        grid[marker_row, marker_cols[gap_idx]] = MARKER
        row = gap_top + gap_height
        grid[row, :] = stripe_colors[gap_idx + 1]

    return grid


def _is_valid_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    bg = _dominant_color(grid)
    stripe_rows = _stripe_rows(grid, bg)
    if len(stripe_rows) < 2:
        return False

    stripe_colors = [int(grid[r, 0]) for r in stripe_rows]
    if len(set(stripe_colors)) != len(stripe_colors):
        return False

    marker_color = None
    for upper_row, lower_row in zip(stripe_rows, stripe_rows[1:]):
        gap = grid[upper_row + 1 : lower_row]
        if gap.shape[0] < MIN_GAP:
            return False
        coords = np.argwhere(gap != bg)
        if coords.shape[0] != 1:
            return False
        local_r, local_c = map(int, coords[0])
        color = int(gap[local_r, local_c])
        if color == bg or color in stripe_colors:
            return False
        if marker_color is None:
            marker_color = color
        elif color != marker_color:
            return False

    if marker_color is None:
        return False

    occupied = set(stripe_rows)
    for r in range(grid.shape[0]):
        if r in occupied:
            continue
        non_bg = np.flatnonzero(grid[r] != bg)
        if non_bg.size > 1:
            return False
        if non_bg.size == 1:
            if int(grid[r, int(non_bg[0])]) != marker_color:
                return False

    return True


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    stripe_count = rng.randint(MIN_STRIPES, MAX_STRIPES)
    width = rng.randint(MIN_WIDTH, MAX_WIDTH)
    stripe_colors = rng.sample(list(STRIPE_COLORS), stripe_count)
    gap_heights = [rng.randint(MIN_GAP, MAX_GAP) for _ in range(stripe_count - 1)]
    marker_rows_in_gap = [rng.randrange(gap_height) for gap_height in gap_heights]
    marker_cols = [rng.randrange(width) for _ in gap_heights]

    grid = _build_grid(width, stripe_colors, gap_heights, marker_rows_in_gap, marker_cols)
    if not _is_valid_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
