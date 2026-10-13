from __future__ import annotations

import numpy as np

from arc_utils import colors_present
import itertools
import random


def _find_frame(g: np.ndarray) -> tuple[int, int, int, int, int]:
    height, width = map(int, g.shape)
    candidates = []
    for color in colors_present(g):
        full_rows = [r for r in range(height) if np.all(g[r, :] == color)]
        full_cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if len(full_rows) != 2 or len(full_cols) != 2:
            continue

        top, bottom = full_rows
        left, right = full_cols
        if bottom - top > 1 and right - left > 1:
            candidates.append((int(color), top, bottom, left, right))

    assert len(candidates) == 1, "expected one pair of same-color full rows and columns"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the framed mask and recolor each mask pixel with the legend color from its quadrant.

    Input structure:
    - A one-color frame is encoded by two full horizontal lines and two full vertical
      lines of the same color.
    - The four diagonal pixels outside the frame corners are the quadrant legend colors.
    - The framed interior contains one background color and one mask color.

    Rule:
    - Output only the rectangular interior of the frame.
    - Split that interior into four equal quadrants.
    - Replace each mask pixel by the corresponding outside-corner legend color and
      leave background pixels unchanged.

    Key insight:
    - The four corner pixels are not part of the output; they are a color legend
      for the mask cells in the matching quadrant of the cropped interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    frame, top, bottom, left, right = _find_frame(g)
    interior = g[top + 1 : bottom, left + 1 : right]
    out_h, out_w = map(int, interior.shape)
    assert out_h > 0 and out_w > 0, "frame must enclose a nonempty interior"
    assert out_h % 2 == 0 and out_w % 2 == 0, "interior must split into four equal quadrants"

    outside = np.ones((height, width), dtype=bool)
    outside[top + 1 : bottom, left + 1 : right] = False
    outside_colors = set(map(int, g[outside].reshape(-1))) - {frame}
    interior_colors = set(map(int, interior.reshape(-1)))
    mask_candidates = interior_colors - outside_colors
    assert len(mask_candidates) == 1, "expected one interior-only mask color"
    mask_color = mask_candidates.pop()

    bg_candidates = interior_colors - {mask_color}
    assert len(bg_candidates) == 1, "expected one interior background color"
    bg = bg_candidates.pop()

    legend_coords = [
        (top - 1, left - 1),
        (top - 1, right + 1),
        (bottom + 1, left - 1),
        (bottom + 1, right + 1),
    ]
    for r, c in legend_coords:
        assert 0 <= r < height and 0 <= c < width, "legend pixels must be outside frame corners"
    top_left, top_right, bottom_left, bottom_right = [int(g[r, c]) for r, c in legend_coords]
    assert all(
        color not in {frame, bg, mask_color}
        for color in (top_left, top_right, bottom_left, bottom_right)
    ), "legend colors must be distinct from frame, background, and mask roles"

    mask = interior == mask_color
    assert np.all(interior[~mask] == bg), "all non-mask interior pixels must be background"
    out = interior.copy()

    row_top = np.arange(out_h)[:, None] < out_h // 2
    col_left = np.arange(out_w)[None, :] < out_w // 2
    out[mask & row_top & col_left] = top_left
    out[mask & row_top & ~col_left] = top_right
    out[mask & ~row_top & col_left] = bottom_left
    out[mask & ~row_top & ~col_left] = bottom_right
    return out

# --------------- generator (o5x) ---------------

# Grid/frame geometry and fixed color roles.
INTERIOR_SIDES = [2, 4, 6]
GRID_MARGIN, FRAME_INSET, INTERIOR_INSET = 4, 1, 2
BG_COLOR, FRAME_COLOR, MASK_COLOR = 0, 1, 8
LEGEND_PALETTE = [2, 3, 4, 5, 6, 7, 9]

# Mask density constraints and retry budget.
SMALL_QUAD_TOTALS = [2, 3]
MIN_QUAD_DENSITY, MAX_QUAD_DENSITY = 0.25, 0.75
MIN_TOTAL_DENSITY, MAX_TOTAL_DENSITY = 0.45, 0.75
MAX_GENERATION_ATTEMPTS = 300


def _gen_quad_counts(side: int) -> tuple[int, int, int, int]:
    quad_side = side // 2
    quad_area = quad_side * quad_side
    if quad_area == 1:
        total = random.choice(SMALL_QUAD_TOTALS)
        counts = [0, 0, 0, 0]
        for idx in random.sample(range(4), total):
            counts[idx] = 1
        return tuple(counts)

    lo = int(np.ceil(MIN_QUAD_DENSITY * quad_area))
    hi = int(np.floor(MAX_QUAD_DENSITY * quad_area))
    total_lo = int(np.ceil(MIN_TOTAL_DENSITY * side * side))
    total_hi = int(np.floor(MAX_TOTAL_DENSITY * side * side))
    ranges = itertools.product(range(lo, hi + 1), repeat=4)
    options = [counts for counts in ranges if total_lo <= sum(counts) <= total_hi]
    return random.choice(options)


def _draw_output(side: int, legend_colors: list[int]) -> np.ndarray:
    """Draw the colored output grid that the input will reverse to mask color."""
    out = np.full((side, side), BG_COLOR, dtype=int)
    quad_side = side // 2
    quad_origins = [(0, 0), (0, quad_side), (quad_side, 0), (quad_side, quad_side)]
    for (r0, c0), color, count in zip(quad_origins, legend_colors, _gen_quad_counts(side)):
        cells = [(r0 + dr, c0 + dc) for dr in range(quad_side) for dc in range(quad_side)]
        for r, c in random.sample(cells, count):
            out[r, c] = color
    return out


def _assemble_grid(out: np.ndarray, legend_colors: list[int]) -> np.ndarray:
    """Embed the interior mask, frame, and corner legend pixels into the input grid."""
    side = int(out.shape[0])
    size = side + GRID_MARGIN
    frame_end = size - FRAME_INSET - 1
    interior = np.where(out == BG_COLOR, BG_COLOR, MASK_COLOR)
    grid = np.full((size, size), BG_COLOR, dtype=int)
    grid[FRAME_INSET, :] = FRAME_COLOR
    grid[frame_end, :] = FRAME_COLOR
    grid[:, FRAME_INSET] = FRAME_COLOR
    grid[:, frame_end] = FRAME_COLOR
    grid[INTERIOR_INSET : size - INTERIOR_INSET, INTERIOR_INSET : size - INTERIOR_INSET] = interior

    last = size - 1
    grid[0, 0] = legend_colors[0]
    grid[0, last] = legend_colors[1]
    grid[last, 0] = legend_colors[2]
    grid[last, last] = legend_colors[3]
    return grid


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(INTERIOR_SIDES)
    legend_colors = random.sample(LEGEND_PALETTE, 4)
    out = _draw_output(side, legend_colors)
    grid = _assemble_grid(out, legend_colors)

    return grid if np.array_equal(main(grid), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 77fdfe62")
