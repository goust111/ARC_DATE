from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg
import random


def _rectangular_frame_bbox(g: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(g == int(color))
    if coords.size == 0:
        return None

    r0, c0, r1, c1 = bbox_coords(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    sub = (g[r0 : r1 + 1, c0 : c1 + 1] == int(color))
    perimeter = np.zeros((h, w), dtype=bool)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True
    if np.array_equal(sub, perimeter):
        return int(r0), int(c0), int(r1), int(c1)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the cross so it is aligned with the rectangular frame's top-right corner while leaving the frame fixed.

    Input structure
    - The grid has a background, one complete one-pixel-thick rectangular frame, and one cross-shaped color.
    - The cross is visible as a single row and a single column, with frame pixels covering the places where it crosses the frame border.
    - The original cross row and column both pass through the strict interior of the rectangular frame.

    Rule
    - Erase the old cross from its original row and column.
    - Redraw that cross color across the frame's top row and down the frame's right column.
    - Restore the rectangular frame on top, so the cross is visible only outside the frame and through the frame interior is cleared.

    Key insight
    - The rectangle supplies the destination anchor: the moving cross is recentered from its interior crossing point to the frame's top-right corner.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected one frame color and one cross color"

    frame_candidates = []
    for color in colors:
        frame_bbox = _rectangular_frame_bbox(g, color)
        if frame_bbox is not None:
            frame_candidates.append((int(color), frame_bbox))
    assert len(frame_candidates) == 1, "expected exactly one rectangular frame color"

    frame_color, (r0, c0, r1, c1) = frame_candidates[0]
    cross_colors = [int(color) for color in colors if int(color) != frame_color]
    assert len(cross_colors) == 1, "expected exactly one cross color"
    cross_color = cross_colors[0]

    cross_coords = np.argwhere(g == cross_color)
    assert cross_coords.size > 0, "expected visible cross pixels"
    row_counts = Counter(map(int, cross_coords[:, 0]))
    col_counts = Counter(map(int, cross_coords[:, 1]))
    cross_rows = [r for r, count in row_counts.items() if count > 1]
    cross_cols = [c for c, count in col_counts.items() if count > 1]
    assert len(cross_rows) == 1 and len(cross_cols) == 1, "expected one cross row and one cross column"

    cross_r = int(cross_rows[0])
    cross_c = int(cross_cols[0])
    assert r0 < cross_r < r1 and c0 < cross_c < c1, "cross must pass through the frame interior"
    assert all(int(r) == cross_r or int(c) == cross_c for r, c in cross_coords), "cross pixels must be row/column aligned"

    out = g.copy()
    out[out == cross_color] = bg
    out[r0, :] = cross_color
    out[:, c1] = cross_color

    frame_region = out[r0 : r1 + 1, c0 : c1 + 1]
    frame_perimeter = np.zeros_like(frame_region, dtype=bool)
    frame_perimeter[0, :] = True
    frame_perimeter[-1, :] = True
    frame_perimeter[:, 0] = True
    frame_perimeter[:, -1] = True
    frame_region[frame_perimeter] = frame_color
    return out

# --------------- generator (o5x) ---------------

# Canvas/colors.
GRID_SIZE = 12
BACKGROUND_COLOR = 0
COLOR_POOL = range(1, 10)
# Frame geometry and retry limit.
FRAME_SIZE_RANGE = (5, 8)
FRAME_MARGIN = 2
MAX_ATTEMPTS = 300


def _gen_sample_frame_bbox() -> tuple[int, int, int, int]:
    """Return a frame bbox sampled under the fixed margin constraints."""
    h = random.randint(*FRAME_SIZE_RANGE)
    w = random.randint(*FRAME_SIZE_RANGE)
    r0 = random.randint(FRAME_MARGIN, GRID_SIZE - h - FRAME_MARGIN)
    c0 = random.randint(FRAME_MARGIN, GRID_SIZE - w - FRAME_MARGIN)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _gen_paint_frame(
    grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int
) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _gen_attempt() -> np.ndarray | None:
    frame_color, cross_color = random.sample(COLOR_POOL, 2)
    r0, c0, r1, c1 = _gen_sample_frame_bbox()

    out = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND_COLOR, dtype=int)
    out[r0, :] = cross_color
    out[:, c1] = cross_color
    _gen_paint_frame(out, r0, c0, r1, c1, frame_color)

    cross_r = random.randint(r0 + 1, r1 - 1)
    cross_c = random.randint(c0 + 1, c1 - 1)
    inp = out.copy()
    inp[inp == cross_color] = BACKGROUND_COLOR
    inp[cross_r, :] = cross_color
    inp[:, cross_c] = cross_color
    _gen_paint_frame(inp, r0, c0, r1, c1, frame_color)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input for 917bccba")
