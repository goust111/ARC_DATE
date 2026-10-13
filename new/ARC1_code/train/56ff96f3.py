from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each two-pixel foreground color into the axis-aligned rectangle whose opposite corners those pixels mark.

    Input structure
    - The background is the most common color.
    - Every foreground color appears exactly twice, as two isolated corner markers.
    - The rectangles spanned by different colors do not overlap.

    Rule
    - For each foreground color, take the inclusive bounding box of its two pixels.
    - Recolor every pixel inside that bounding box to the same foreground color.
    - Leave all pixels outside those rectangles as background.

    Key insight
    - The two equal-colored pixels are not separate objects; together they specify the opposite corners of a filled rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert colors, "expected at least one foreground color"

    rects: list[tuple[int, tuple[int, int, int, int]]] = []
    occupied = np.zeros(g.shape, dtype=bool)
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each foreground color must have exactly two marker pixels"
        r0, c0, r1, c1 = bbox_coords(coords)

        rect_mask = np.zeros(g.shape, dtype=bool)
        rect_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        assert not np.any(occupied & rect_mask), "filled rectangles must not overlap"

        occupied |= rect_mask
        rects.append((color, (r0, c0, r1, c1)))

    out = np.full(g.shape, bg, dtype=int)
    for color, (r0, c0, r1, c1) in rects:
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out

# --------------- generator (o5x) ---------------

# Grid size and filled-coverage bounds.
GRID_H_RANGE, GRID_W_RANGE = (6, 10), (8, 11)
COVERAGE_FRAC_RANGE = (0.25, 0.36)

# Rectangle count and inclusive side-length bounds.
N_RECTS_RANGE, RECT_SIDE_RANGE = (1, 2), (2, 7)

# Palette and rejection limit.
BACKGROUND_COLOR = 0
FOREGROUND_COLORS = np.arange(1, 10)
MAX_GENERATION_ATTEMPTS = 300


def _gen_area_bounds(h: int, w: int) -> tuple[int, int]:
    area = h * w
    lo, hi = COVERAGE_FRAC_RANGE
    return int(np.ceil(area * lo)), int(np.floor(area * hi))


def _gen_packable(dims: list[tuple[int, int]], h: int, w: int) -> bool:
    if len(dims) == 1:
        rh, rw = dims[0]
        return rh <= h and rw <= w

    (rh0, rw0), (rh1, rw1) = dims
    side_by_side = max(rh0, rh1) <= h and rw0 + rw1 <= w
    stacked = rh0 + rh1 <= h and max(rw0, rw1) <= w
    return side_by_side or stacked


def _gen_dimension_sets(h: int, w: int, n_rects: int) -> list[list[tuple[int, int]]]:
    min_area, max_area = _gen_area_bounds(h, w)
    one_rect = [
        (rh, rw)
        for rh in range(RECT_SIDE_RANGE[0], min(RECT_SIDE_RANGE[1], h) + 1)
        for rw in range(RECT_SIDE_RANGE[0], min(RECT_SIDE_RANGE[1], w) + 1)
    ]

    if n_rects == 1:
        return [[d] for d in one_rect if min_area <= d[0] * d[1] <= max_area]

    out: list[list[tuple[int, int]]] = []
    for d0 in one_rect:
        for d1 in one_rect:
            total_area = d0[0] * d0[1] + d1[0] * d1[1]
            dims = [d0, d1]
            if min_area <= total_area <= max_area and _gen_packable(dims, h, w):
                out.append(dims)
    return out


def _gen_rect_positions(
    h: int, w: int, dims: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    pos0 = [(r, c) for r in range(h - dims[0][0] + 1) for c in range(w - dims[0][1] + 1)]

    if len(dims) == 1:
        r, c = pos0[np.random.randint(len(pos0))]
        rh, rw = dims[0]
        return [(r, c, r + rh - 1, c + rw - 1)]

    pos1 = [(r, c) for r in range(h - dims[1][0] + 1) for c in range(w - dims[1][1] + 1)]
    valid: list[tuple[tuple[int, int], tuple[int, int]]] = []
    rh0, rw0 = dims[0]
    rh1, rw1 = dims[1]
    for r0, c0 in pos0:
        b0 = (r0, c0, r0 + rh0 - 1, c0 + rw0 - 1)
        for r1, c1 in pos1:
            b1 = (r1, c1, r1 + rh1 - 1, c1 + rw1 - 1)
            rows_overlap = b0[0] <= b1[2] and b1[0] <= b0[2]
            cols_overlap = b0[1] <= b1[3] and b1[1] <= b0[3]
            if not (rows_overlap and cols_overlap):
                valid.append(((r0, c0), (r1, c1)))

    if not valid:
        return None

    (r0, c0), (r1, c1) = valid[np.random.randint(len(valid))]
    return [
        (r0, c0, r0 + rh0 - 1, c0 + rw0 - 1),
        (r1, c1, r1 + rh1 - 1, c1 + rw1 - 1),
    ]


def _gen_layout() -> tuple[int, int, list[tuple[int, int, int, int]]] | None:
    """Sample grid size, dimensions, and non-overlapping rectangle positions."""
    h = int(np.random.randint(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(np.random.randint(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    n_rects = int(np.random.randint(N_RECTS_RANGE[0], N_RECTS_RANGE[1] + 1))

    dimension_sets = _gen_dimension_sets(h, w, n_rects)
    if not dimension_sets:
        return None

    dims = dimension_sets[np.random.randint(len(dimension_sets))]
    rects = _gen_rect_positions(h, w, dims)
    if rects is None:
        return None

    return h, w, rects


def _gen_render(h: int, w: int, rects: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Build the sparse marker input and filled-rectangle target for a layout."""
    colors = np.random.choice(FOREGROUND_COLORS, size=len(rects), replace=False).tolist()
    out = np.full((h, w), BACKGROUND_COLOR, dtype=int)
    inp = np.full((h, w), BACKGROUND_COLOR, dtype=int)

    for color, (r0, c0, r1, c1) in zip(colors, rects):
        out[r0 : r1 + 1, c0 : c1 + 1] = color
        if np.random.randint(2) == 0:
            inp[r0, c0] = color
            inp[r1, c1] = color
        else:
            inp[r0, c1] = color
            inp[r1, c0] = color

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    layout = _gen_layout()
    if layout is None:
        return None

    h, w, rects = layout
    inp, out = _gen_render(h, w, rects)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
