import random

import numpy as np


AZURE = 8
GREY = 5
PINK = 6
MAROON = 9

_PERIMETER = np.ones((3, 3), dtype=bool)
_PERIMETER[1, 1] = False

_DIRECTIONS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _relation(value: int, center: int) -> int:
    if value < center:
        return -1
    if value > center:
        return 1
    return 0


def _find_grey_region_center(grid: np.ndarray) -> tuple[int, int]:
    matches: list[tuple[int, int]] = []
    h, w = grid.shape
    for r in range(h - 2):
        for c in range(w - 2):
            block = grid[r : r + 3, c : c + 3]
            center = int(block[1, 1])
            if center in (AZURE, MAROON) and np.all(block[_PERIMETER] == GREY):
                matches.append((r + 1, c + 1))

    assert len(matches) == 1, "expected exactly one 3x3 grey ring"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Place the maroon answer cell in the grey 3x3 region pointed to by the pink control pixel.

    Input structure:
    - The grid has an azure background.
    - One 3x3 grey region appears as eight grey pixels around a center cell that is azure or maroon.
    - Exactly one pink control pixel sits outside that 3x3 region.

    Rule:
    - Copy the grid and restore any maroon cell inside the grey 3x3 region to grey.
    - Compare the pink pixel to the center of the grey region: above/same row/below selects the first/second/third region row.
    - Compare the pink pixel to the center of the grey region: left/same column/right selects the first/second/third region column.
    - Paint the selected grey-region cell maroon, and also change the pink control pixel to maroon.

    Key insight:
    - The pink pixel is a direction pointer whose row and column relation to the grey-region center names a position in the 3x3 target.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all(np.isin(g, [AZURE, GREY, PINK, MAROON])), "unexpected color in grid"

    pink_cells = np.argwhere(g == PINK)
    assert len(pink_cells) == 1, "expected exactly one pink control pixel"
    pr, pc = map(int, pink_cells[0])

    cr, cc = _find_grey_region_center(g)
    assert not (abs(pr - cr) <= 1 and abs(pc - cc) <= 1), "pink control must be outside the 3x3 grey region"

    row_index = _relation(pr, cr) + 1
    col_index = _relation(pc, cc) + 1

    out = g.copy()
    region = out[cr - 1 : cr + 2, cc - 1 : cc + 2]
    region[region == MAROON] = GREY
    out[cr - 1 + row_index, cc - 1 + col_index] = MAROON
    out[pr, pc] = MAROON
    return out


def _pink_candidates(h: int, w: int, cr: int, cc: int, dr: int, dc: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(h):
        if _relation(r, cr) != dr:
            continue
        for c in range(w):
            if _relation(c, cc) != dc:
                continue
            # Keep an azure moat around the grey region so the two objects stay visually separate.
            if cr - 2 <= r <= cr + 2 and cc - 2 <= c <= cc + 2:
                continue
            cells.append((r, c))
    return cells


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(7, 30)
    w = random.randint(7, 30)
    cr = random.randint(3, h - 4)
    cc = random.randint(3, w - 4)
    dr, dc = random.choice(_DIRECTIONS)

    candidates = _pink_candidates(h, w, cr, cc, dr, dc)
    if not candidates:
        return None
    pr, pc = random.choice(candidates)

    grid = np.full((h, w), AZURE, dtype=int)
    grid[cr - 1 : cr + 2, cc - 1 : cc + 2] = GREY
    grid[cr, cc] = random.choice([AZURE, MAROON])
    grid[pr, pc] = PINK

    try:
        main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
