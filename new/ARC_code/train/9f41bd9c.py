from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_coords
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the side-attached foreground object to the opposite wall, shear its lower rows inward, and paint the crossed floor cells.

    Input structure:
    - The top row is the upper background color and the bottom row is the
      floor color.
    - A single non-background, non-floor object touches exactly one vertical
      grid edge and rests immediately above the first solid floor row.

    Rule:
    - Clear the source object to the upper background color.
    - Repaint each object row on the opposite side: the first three relative
      rows keep the edge alignment, and each lower row shifts one more pixel
      inward from that side.
    - On the first floor row, repaint the traversed side-to-object segment
      maroon, stopping just before the final outer bottom support pixel.

    Key insight:
    - The side of the source object gives the horizontal direction, while the
      row index inside the object gives the one-cell-per-row diagonal shear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be two-dimensional"
    height, width = g.shape

    top_colors = np.unique(g[0])
    bottom_colors = np.unique(g[-1])
    assert len(top_colors) == 1, "top row must be a uniform background"
    assert len(bottom_colors) == 1, "bottom row must be a uniform floor"
    bg = int(top_colors[0])
    floor = int(bottom_colors[0])
    assert bg != floor, "background and floor colors must differ"

    colors = set(map(int, np.unique(g)))
    assert len(colors) == 3, "expected exactly background, floor, and one object color"
    object_colors = colors - {bg, floor}
    assert len(object_colors) == 1, "expected one moving object color"
    obj = object_colors.pop()

    full_floor_rows = np.where(np.all(g == floor, axis=1))[0]
    assert len(full_floor_rows) > 0, "expected a solid floor row"
    floor_row = int(full_floor_rows[0])
    assert 0 < floor_row < height, "floor row must be inside the grid"
    assert np.all(g[floor_row:] == floor), "floor must be a solid bottom band"

    obj_coords = np.argwhere(g == obj)
    assert len(obj_coords) > 0, "expected a moving object"
    r0, c0, r1, c1 = bbox_coords(obj_coords)
    object_width = c1 - c0 + 1
    assert r1 == floor_row - 1, "object must rest directly above the floor"
    assert (c0 == 0) ^ (c1 == width - 1), "object must touch exactly one side edge"
    assert np.all(g[:floor_row][g[:floor_row] != obj] == bg), "upper region must contain only bg and object"

    moving_right = c0 == 0
    out = g.copy()
    out[g == obj] = bg

    for r in range(r0, r1 + 1):
        rel_r = r - r0
        rel_cols = np.where(g[r, c0 : c1 + 1] == obj)[0]
        assert len(rel_cols) > 0, "every object bbox row must contain object pixels"

        inward_shift = max(0, rel_r - 2)
        if moving_right:
            start_col = width - object_width - inward_shift
        else:
            start_col = inward_shift
        dst_cols = start_col + rel_cols
        assert int(dst_cols.min()) >= 0 and int(dst_cols.max()) < width, "moved object must fit in the grid"
        out[r, dst_cols] = obj

    bottom_rel_cols = np.where(g[r1, c0 : c1 + 1] == obj)[0]
    bottom_shift = max(0, r1 - r0 - 2)
    bottom_start = width - object_width - bottom_shift if moving_right else bottom_shift
    final_bottom_cols = bottom_start + bottom_rel_cols
    trail_color = Color.MAROON
    assert int(trail_color) not in colors, "trail color must be new"
    if moving_right:
        trail_end = int(final_bottom_cols.max()) - 1
        assert trail_end >= 0, "trail must cover at least one floor pixel"
        out[floor_row, : trail_end + 1] = int(trail_color)
    else:
        trail_start = int(final_bottom_cols.min()) + 1
        assert trail_start < width, "trail must cover at least one floor pixel"
        out[floor_row, trail_start:] = int(trail_color)

    return out

# --------------- generator (att2) ---------------
# Canvas and band bounds.
GRID_ROWS = range(16, 19)
GRID_COLS = (16, 18)
FLOOR_THICKNESS = range(4, 7)
EMPTY_ROWS = range(3, 7)
# Object shape and fit constraints.
OBJECT_WIDTH = 5
OBJECT_HEIGHT = (5, 11)
SOLID_TOP_ROWS = 2
SHEAR_LOCK_ROWS = 3
CHECKER_COLS = (0, 2, 4)
# Colors and retry budget.
TRAIL_COLOR = int(Color.MAROON)
AVAILABLE_COLORS = [c for c in range(10) if c != TRAIL_COLOR]
MAX_GENERATION_ATTEMPTS = 300
SIDES = ("left", "right")

ROW_LAYOUTS = [
    (rows, floor_thick, empty_rows, rows - floor_thick - empty_rows)
    for rows in GRID_ROWS
    for floor_thick in FLOOR_THICKNESS
    for empty_rows in EMPTY_ROWS
    if OBJECT_HEIGHT[0] <= rows - floor_thick - empty_rows <= OBJECT_HEIGHT[1]
]


def _gen_object_mask(height: int) -> np.ndarray:
    mask = np.zeros((height, OBJECT_WIDTH), dtype=bool)
    mask[:SOLID_TOP_ROWS, :] = True
    mask[SOLID_TOP_ROWS:, CHECKER_COLS] = True
    return mask


def _gen_place_original(
    grid: np.ndarray, top: int, side: str, color: int, mask: np.ndarray
) -> None:
    c0 = 0 if side == "left" else grid.shape[1] - mask.shape[1]
    region = grid[top : top + mask.shape[0], c0 : c0 + mask.shape[1]]
    region[mask] = color


def _gen_place_sheared(
    grid: np.ndarray, top: int, side: str, color: int, mask: np.ndarray
) -> np.ndarray:
    width = grid.shape[1]
    bottom_cols = []
    for rel_r in range(mask.shape[0]):
        rel_cols = np.flatnonzero(mask[rel_r])
        inward_shift = max(0, rel_r - SHEAR_LOCK_ROWS + 1)
        c0 = width - mask.shape[1] - inward_shift if side == "right" else inward_shift
        dst_cols = c0 + rel_cols
        grid[top + rel_r, dst_cols] = color
        bottom_cols = dst_cols
    return bottom_cols


def _gen_attempt() -> np.ndarray | None:
    height, floor_thick, empty_rows, obj_h = random.choice(ROW_LAYOUTS)
    width = random.randint(*GRID_COLS)
    if OBJECT_WIDTH + max(0, obj_h - SHEAR_LOCK_ROWS) > width:
        return None

    bg, floor, obj = random.sample(AVAILABLE_COLORS, 3)
    floor_row = height - floor_thick
    obj_mask = _gen_object_mask(obj_h)
    dst_side = random.choice(SIDES)
    src_side = "right" if dst_side == "left" else "left"

    out = np.full((height, width), bg, dtype=int)
    out[floor_row:, :] = floor
    bottom_cols = _gen_place_sheared(out, empty_rows, dst_side, obj, obj_mask)
    if dst_side == "right":
        out[floor_row, : int(bottom_cols.max())] = TRAIL_COLOR
    else:
        out[floor_row, int(bottom_cols.min()) + 1 :] = TRAIL_COLOR

    inp = out.copy()
    inp[inp == obj] = bg
    inp[floor_row, inp[floor_row] == TRAIL_COLOR] = floor
    _gen_place_original(inp, empty_rows, src_side, obj, obj_mask)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
