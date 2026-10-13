from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
MARKER_COLOR = 5
SEED_COLOR = 6
LEGEND_SIZE = 2
LEGEND_COLORS = (1, 2, 3, 4, 7, 8, 9)
GRID_HEIGHT_RANGE = (12, 19)
GRID_WIDTH_RANGE = (12, 19)
GENERATION_RETRY_LIMIT = 300
LEGEND_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _legend_origin(corner: str, h: int, w: int) -> tuple[int, int]:
    if corner == "top_left":
        return 0, 0
    if corner == "top_right":
        return 0, w - LEGEND_SIZE
    if corner == "bottom_left":
        return h - LEGEND_SIZE, 0
    if corner == "bottom_right":
        return h - LEGEND_SIZE, w - LEGEND_SIZE
    raise ValueError(f"unknown corner: {corner}")


def _find_legend(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for corner in LEGEND_CORNERS:
        r0, c0 = _legend_origin(corner, h, w)
        patch = grid[r0 : r0 + LEGEND_SIZE, c0 : c0 + LEGEND_SIZE]
        if patch.shape != (LEGEND_SIZE, LEGEND_SIZE):
            continue
        values = [int(v) for v in patch.reshape(-1)]
        if bg in values:
            continue
        if len(set(values)) != LEGEND_SIZE * LEGEND_SIZE:
            continue
        candidates.append(((r0, c0, r0 + 1, c0 + 1), patch.copy()))
    assert len(candidates) == 1, "expected exactly one 2x2 corner legend"
    return candidates[0]


def _find_seed(grid: np.ndarray, bg: int, legend_bbox: tuple[int, int, int, int]) -> tuple[int, int, int]:
    r0, c0, r1, c1 = legend_bbox
    mask = grid != bg
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    coords = np.argwhere(mask)
    assert coords.shape[0] == 1, "expected exactly one seed pixel outside the legend"
    sr, sc = map(int, coords[0])
    return sr, sc, int(grid[sr, sc])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow rightward periodic vertical bars from the seed, then recolor those bars by the legend colors of the seed-defined quadrants.

    Input structure:
    - The grid is mostly black background and contains exactly one isolated non-black seed pixel away from the border.
    - One 2x2 legend of four distinct non-black colors sits in a corner and stays fixed in place.
    - No other painted objects are present in the input.

    Rule:
    - Use the seed column as the phase origin: draw full-height vertical bars every two columns from the seed column to the right edge, and place gray pixels in the intervening columns with top/bottom alternation starting at the top row.
    - Use the seed row and seed column as the quadrant divider: rows above the seed are top quadrants, rows at or below it are bottom quadrants; columns left of the seed are left quadrants, columns at or right of it are right quadrants.
    - Recolor every temporary seed-color bar pixel in each quadrant to the corresponding cell of the 2x2 legend, keep gray marker pixels gray, keep the legend unchanged, and recolor the seed itself with its lower-right quadrant legend color.

    Key insight:
    - The same seed pixel controls both the stripe phase and the quadrant split, while the corner legend supplies the final recoloring for the seed-origin stripe pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    legend_bbox, legend = _find_legend(g, bg)
    sr, sc, _seed_color = _find_seed(g, bg, legend_bbox)

    h, w = map(int, g.shape)
    out = np.full_like(g, bg)

    upper_right_color = int(legend[0, 1])
    lower_right_color = int(legend[1, 1])
    for c in range(sc, w, 2):
        if sr > 0:
            out[:sr, c] = upper_right_color
        out[sr:, c] = lower_right_color

    for idx, c in enumerate(range(sc + 1, w, 2)):
        marker_row = 0 if idx % 2 == 0 else h - 1
        out[marker_row, c] = MARKER_COLOR

    lr0, lc0, lr1, lc1 = legend_bbox
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = legend
    return out


def _seed_row(h: int) -> int:
    midpoint = h // 2
    if random.random() < 0.5:
        return random.randint(2, max(2, midpoint))
    return random.randint(min(midpoint, h - 3), h - 3)


def _seed_col(w: int, corner: str) -> int:
    left_bound = 3 if "left" in corner else 2
    right_bound = w - 6
    assert left_bound <= right_bound, "grid width must leave room for the legend and stripe pattern"
    return random.randint(left_bound, right_bound)


def _validate_input(grid: np.ndarray) -> None:
    bg = _get_bg(grid)
    legend_bbox, legend = _find_legend(grid, bg)
    sr, sc, seed_color = _find_seed(grid, bg, legend_bbox)
    h, w = map(int, grid.shape)

    assert bg == BACKGROUND
    assert seed_color == SEED_COLOR
    assert 1 <= sr <= h - 2 and 1 <= sc <= w - 2
    assert sr >= 2 and sr <= h - 3 and sc >= 2 and sc <= w - 6

    legend_values = [int(v) for v in legend.reshape(-1)]
    assert len(set(legend_values)) == 4
    assert seed_color not in legend_values
    assert MARKER_COLOR not in legend_values

    non_bg = np.argwhere(grid != bg)
    assert non_bg.shape[0] == 5, "input should contain only the legend and the seed"


def _gen_attempt() -> np.ndarray:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((h, w), BACKGROUND, dtype=int)

    corner = random.choice(LEGEND_CORNERS)
    lr0, lc0 = _legend_origin(corner, h, w)
    legend_colors = random.sample(list(LEGEND_COLORS), 4)
    grid[lr0 : lr0 + 2, lc0 : lc0 + 2] = np.array(legend_colors, dtype=int).reshape(2, 2)

    sr = _seed_row(h)
    sc = _seed_col(w, corner)
    if grid[sr, sc] != BACKGROUND:
        return None
    grid[sr, sc] = SEED_COLOR

    _validate_input(grid)
    out = main(grid)
    assert out.shape == grid.shape
    assert out[sr, sc] == int(grid[lr0 + 1, lc0 + 1])
    assert np.any(out[:sr, sc:] == int(grid[lr0, lc0 + 1]))
    assert np.any(out[sr:, sc:] == int(grid[lr0 + 1, lc0 + 1]))
    assert np.count_nonzero(out == MARKER_COLOR) >= 2
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
