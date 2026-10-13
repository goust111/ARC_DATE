import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CONTROL_OUTER_COLORS = [BLUE, RED, YELLOW, ORANGE]
SQUARE_HALF_SIDES = [2, 3, 4]  # side lengths 4, 6, 8
GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _infer_square(green_coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    assert green_coords.shape == (3, 2), "expected exactly three green corner markers"

    rows = sorted({int(r) for r, _c in green_coords})
    cols = sorted({int(c) for _r, c in green_coords})
    assert len(rows) == 2 and len(cols) == 2, "corner markers must occupy two rows and two columns"

    r0, r1 = rows
    c0, c1 = cols
    assert r1 - r0 == c1 - c0, "corner markers must define an axis-aligned square"
    assert (r0 + r1) % 2 == 0 and (c0 + c1) % 2 == 0, "square center must be a grid cell"

    corners = {(r, c) for r in rows for c in cols}
    present = {tuple(map(int, rc)) for rc in green_coords}
    assert present < corners, "green markers must be exactly three of the four square corners"

    missing = next(iter(corners - present))
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    return (r0, c0), (r1, c1), missing, center


def _find_control_bar(grid: np.ndarray, bg: int) -> tuple[int, int, list[int]]:
    grey_coords = np.argwhere(grid == GREY)
    assert grey_coords.shape == (1, 2), "expected exactly one grey control-bar center pixel"

    cr, cc = map(int, grey_coords[0])
    assert 2 <= cr < grid.shape[0] - 2, "control bar center must allow two cells above and below"

    values = [int(grid[cr + dr, cc]) for dr in (-2, -1, 0, 1, 2)]
    assert values[2] == GREY, "control bar middle pixel must be grey"
    assert all(v != bg for v in values), "control bar must be fully foreground"

    outer = values[:2] + values[3:]
    assert sorted(outer) == sorted(CONTROL_OUTER_COLORS), "control bar outer colors must be blue/red/yellow/orange"
    assert len(set(values)) == 5, "all five control-bar pixels must be distinct"
    return cr, cc, values


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a square center from three green corners, flip a five-pixel control bar, and emit color-coded rays from the center.

    Input structure:
    - The grid has a black background with exactly three green single-pixel markers at three corners of one axis-aligned square.
    - A separate 5x1 vertical control bar appears elsewhere; its middle pixel is grey and its other four pixels are blue, red, yellow, and orange in some order.
    - The square center is a single cell, and the control bar does not overlap the square markers or center.

    Rule:
    - Infer the missing fourth square corner from the three green markers and paint that corner green.
    - Infer the square center and paint it grey.
    - Flip the control bar locally by swapping the top two pixels and swapping the bottom two pixels while keeping the grey middle fixed.
    - Let the flipped top color be the diagonal-ray color and the flipped bottom color be the cardinal-ray color; paint diagonal rays first and then cardinal rays from the center to the grid edges, filling only background cells.

    Key insight:
    - The three green corners determine the ray origin, while the control bar must be flipped before its top and bottom colors can be read as the diagonal and cardinal ray colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    assert bg == BLACK, "expected black background"

    green_coords = np.argwhere(g == GREEN)
    (_r0, _c0), (_r1, _c1), missing_corner, center = _infer_square(green_coords)
    bar_r, bar_c, bar_values = _find_control_bar(g, bg)

    assert tuple(center) != (bar_r, bar_c), "square center cannot overlap the control bar"

    out = g.copy()
    out[missing_corner] = GREEN
    out[center] = GREY

    flipped = [bar_values[1], bar_values[0], bar_values[2], bar_values[4], bar_values[3]]
    for idx, value in enumerate(flipped):
        out[bar_r - 2 + idx, bar_c] = value

    diagonal_color = flipped[0]
    cardinal_color = flipped[4]
    painted_by_diagonal = np.zeros_like(out, dtype=bool)

    for dr, dc in DIAGONALS:
        r = center[0] + dr
        c = center[1] + dc
        while _in_bounds(out.shape, r, c):
            if out[r, c] == bg:
                out[r, c] = diagonal_color
                painted_by_diagonal[r, c] = True
            r += dr
            c += dc

    for dr, dc in CARDINALS:
        r = center[0] + dr
        c = center[1] + dc
        while _in_bounds(out.shape, r, c):
            if out[r, c] == bg or painted_by_diagonal[r, c]:
                out[r, c] = cardinal_color
            r += dr
            c += dc

    return out


def _bar_clear_of_square(center: tuple[int, int], corner_rows: set[int], corner_cols: set[int], bar_center: tuple[int, int]) -> bool:
    cr, cc = center
    br, bc = bar_center
    bar_rows = {br - 2, br - 1, br, br + 1, br + 2}

    if bc in corner_cols or bc == cc:
        return False
    if bar_rows & corner_rows:
        return False
    if cr in bar_rows:
        return False

    for r in bar_rows:
        if abs(r - cr) == abs(bc - cc):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 20)
    w = random.randint(15, 20)
    half = random.choice(SQUARE_HALF_SIDES)

    min_center_r = half + 1
    max_center_r = h - half - 2
    min_center_c = half + 1
    max_center_c = w - half - 2
    if min_center_r > max_center_r or min_center_c > max_center_c:
        return None

    center = (
        random.randint(min_center_r, max_center_r),
        random.randint(min_center_c, max_center_c),
    )
    cr, cc = center
    corners = [
        (cr - half, cc - half),
        (cr - half, cc + half),
        (cr + half, cc - half),
        (cr + half, cc + half),
    ]
    missing_index = random.randrange(4)
    shown_corners = [corner for idx, corner in enumerate(corners) if idx != missing_index]

    bar_candidates = []
    corner_rows = {r for r, _c in corners}
    corner_cols = {c for _r, c in corners}
    for br in range(2, h - 2):
        for bc in range(w):
            if _bar_clear_of_square(center, corner_rows, corner_cols, (br, bc)):
                bar_candidates.append((br, bc))
    if not bar_candidates:
        return None

    bar_r, bar_c = random.choice(bar_candidates)
    outer = random.sample(CONTROL_OUTER_COLORS, k=4)

    grid = np.full((h, w), BLACK, dtype=int)
    for r, c in shown_corners:
        grid[r, c] = GREEN
    for idx, value in enumerate([outer[0], outer[1], GREY, outer[2], outer[3]]):
        grid[bar_r - 2 + idx, bar_c] = value

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
