from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg, in_bounds
import random


def _has_complete_square_border(g: np.ndarray, obj: dict) -> bool:
    r0, c0, r1, c1 = obj["bbox"]
    h = bbox_h(obj["bbox"])
    w = bbox_w(obj["bbox"])
    if h != w or h < 2:
        return False

    color = int(obj["color"])
    return bool(
        np.all(g[r0, c0 : c1 + 1] == color)
        and np.all(g[r1, c0 : c1 + 1] == color)
        and np.all(g[r0 : r1 + 1, c0] == color)
        and np.all(g[r0 : r1 + 1, c1] == color)
    )


def _corner_marker_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [
        (r0 - 1, c0),
        (r0, c0 - 1),
        (r0 - 1, c1),
        (r0, c1 + 1),
        (r1 + 1, c0),
        (r1, c0 - 1),
        (r1 + 1, c1),
        (r1, c1 + 1),
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark every complete square object with red pixels just outside its corners.

    Input structure:
    - A uniform background contains one or more monochrome 4-connected foreground objects.
    - The relevant objects are square frames or filled squares: their bbox is square
      and all cells on the bbox border have the object's color.
    - Non-square bars and incomplete fragments may also appear and are ignored.

    Rule:
    - Keep the input unchanged.
    - For each complete square object, paint red on the two outside background
      pixels orthogonally adjacent to each of the square's four corners.

    Key insight: The transformation is not based on size or color; it selects
    objects whose bounding box is a complete square border and labels only their
    exposed outer corner-adjacent cells.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    square_objects = [
        obj
        for obj in find_components(g, bg=bg, connectivity=4, monochromatic=True)
        if _has_complete_square_border(g, obj)
    ]
    assert square_objects, "expected at least one complete square object"

    for obj in square_objects:
        coords = _corner_marker_coords(obj["bbox"])
        for r, c in coords:
            assert in_bounds(g.shape, r, c), "square object should have one-cell outside margin"
            assert int(g[r, c]) == int(bg), "corner marker positions should be background"
            out[r, c] = Color.RED

    return out

# --------------- generator (c) ---------------

# ---- Configuration ----
GRID_H, GRID_W = 10, 10
BG_COLOR = Color.ORANGE                       # background color (7)
FG_COLORS = [Color.PINK, Color.TEAL]          # foreground color choices
SQUARE_COUNT_RANGE = [1, 2, 3, 4]             # number of squares to attempt
SQUARE_SIZE_MIN = 2                            # minimum square side length
MAX_SZ_BY_COUNT = {1: 8, 2: 6, 3: 5, 4: 4}   # max square size per count
EDGE_MARGIN = 1                                # min gap from grid edge for squares
DISTRACTOR_COUNT_MAX = 5                       # max distractor objects
DISTRACTOR_TYPES = ['pixel', 'hbar', 'vbar', 'rect']
DISTRACTOR_BAR_RANGE = (2, 4)                  # bar/rect dimension range
DISTRACTOR_PLACE_RETRIES = 80                  # retries per distractor placement
GEN_MAX_RETRIES = 300                          # outer generate_input retry limit


def _marker_coords(r0, c0, sz):
    """Return the 8 corner-adjacent cells outside a square at (r0,c0) with side `sz`."""
    r1, c1 = r0 + sz - 1, c0 + sz - 1
    return [
        (r0 - 1, c0), (r0, c0 - 1),
        (r0 - 1, c1), (r0, c1 + 1),
        (r1 + 1, c0), (r1, c0 - 1),
        (r1 + 1, c1), (r1, c1 + 1),
    ]


def _border_mask(r0, c0, sz, shape):
    """Return a boolean mask with True on the border cells of a square."""
    mask = np.zeros(shape, dtype=bool)
    r1, c1 = r0 + sz - 1, c0 + sz - 1
    mask[r0, c0:c1+1] = True
    mask[r1, c0:c1+1] = True
    mask[r0:r1+1, c0] = True
    mask[r0:r1+1, c1] = True
    return mask


def _draw_square(grid, occupied, marker_reserved, r0, c0, sz, fg):
    """Draw a square frame (or filled 2x2) onto the grid and update tracking masks."""
    r1, c1 = r0 + sz - 1, c0 + sz - 1
    if sz == 2:
        grid[r0:r1+1, c0:c1+1] = fg
    else:
        grid[r0, c0:c1+1] = fg
        grid[r1, c0:c1+1] = fg
        grid[r0:r1+1, c0] = fg
        grid[r0:r1+1, c1] = fg
    occupied |= _border_mask(r0, c0, sz, grid.shape)
    for mr, mc in _marker_coords(r0, c0, sz):
        marker_reserved[mr, mc] = True


def _distractor_dims(dtype):
    """Return (height, width) for a distractor of the given type."""
    lo, hi = DISTRACTOR_BAR_RANGE
    if dtype == 'pixel':
        return 1, 1
    elif dtype == 'hbar':
        return 1, random.randint(lo, hi)
    elif dtype == 'vbar':
        return random.randint(lo, hi), 1
    else:
        dh = random.randint(lo, hi)
        dw = random.randint(lo, hi)
        if dh == dw:
            dw = dh + 1 if dh < hi else dh - 1
        return dh, dw


def _gen_attempt() -> np.ndarray | None:
    fg = random.choice(FG_COLORS)
    grid = np.full((GRID_H, GRID_W), BG_COLOR, dtype=int)
    occupied = np.zeros((GRID_H, GRID_W), dtype=bool)
    marker_reserved = np.zeros((GRID_H, GRID_W), dtype=bool)

    num_squares = random.choice(SQUARE_COUNT_RANGE)
    max_sz = MAX_SZ_BY_COUNT[num_squares]
    sizes = sorted([random.randint(SQUARE_SIZE_MIN, max_sz)
                     for _ in range(num_squares)], reverse=True)

    placed_squares = []

    def can_place(r0, c0, sz):
        r1, c1 = r0 + sz - 1, c0 + sz - 1
        if r0 < EDGE_MARGIN or c0 < EDGE_MARGIN or r1 > GRID_H - EDGE_MARGIN - 1 or c1 > GRID_W - EDGE_MARGIN - 1:
            return False
        for mr, mc in _marker_coords(r0, c0, sz):
            if mr < 0 or mr >= GRID_H or mc < 0 or mc >= GRID_W:
                return False
            if occupied[mr, mc] or marker_reserved[mr, mc]:
                return False
        # Border cells must not overlap existing occupied cells
        if (np.any(occupied[r0, c0:c1+1]) or np.any(occupied[r1, c0:c1+1]) or
                np.any(occupied[r0:r1+1, c0]) or np.any(occupied[r0:r1+1, c1])):
            return False
        return True

    for sz in sizes:
        positions = [(r0, c0)
                     for r0 in range(EDGE_MARGIN, GRID_H - sz)
                     for c0 in range(EDGE_MARGIN, GRID_W - sz)
                     if can_place(r0, c0, sz)]
        if not positions:
            continue
        r0, c0 = random.choice(positions)
        _draw_square(grid, occupied, marker_reserved, r0, c0, sz, fg)
        placed_squares.append((r0, c0, sz))

    if not placed_squares:
        return None

    # Place distractors
    num_distractors = random.randint(0, DISTRACTOR_COUNT_MAX)
    for _ in range(num_distractors):
        dtype = random.choice(DISTRACTOR_TYPES)
        for _try in range(DISTRACTOR_PLACE_RETRIES):
            dh, dw = _distractor_dims(dtype)
            r0 = random.randint(0, GRID_H - dh)
            c0 = random.randint(0, GRID_W - dw)
            region = grid[r0:r0+dh, c0:c0+dw]
            reserved = marker_reserved[r0:r0+dh, c0:c0+dw]
            if np.all(region == BG_COLOR) and not np.any(reserved):
                grid[r0:r0+dh, c0:c0+dw] = fg
                occupied[r0:r0+dh, c0:c0+dw] = True
                break

    # Validate with solver
    try:
        out = main(grid)
    except Exception:
        return None

    expected = grid.copy()
    for r0, c0, sz in placed_squares:
        for mr, mc in _marker_coords(r0, c0, sz):
            expected[mr, mc] = Color.RED
    if not np.array_equal(out, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after %d attempts" % GEN_MAX_RETRIES)
