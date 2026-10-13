from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from arc_utils import Color, neighbors8


BG = Color.ORANGE
RED = Color.RED
GREY = Color.GREY
BLUE = Color.BLUE
GREEN = Color.GREEN
YELLOW = Color.YELLOW

GRID_SIZE = 15
CENTER = GRID_SIZE // 2
SIGNAL_COLORS = {RED, GREY}
DISTRACTOR_COLORS = (BLUE, GREEN, YELLOW)
GEN_BAND_WIDTHS = (7, 9, 11)
GEN_ATTEMPTS = 300


def _swap_signal(color: int, steps: int) -> int:
    assert color in SIGNAL_COLORS
    if steps % 2 == 0:
        return int(color)
    return GREY if int(color) == RED else RED


def _band_span(grid: np.ndarray) -> tuple[int, int]:
    cols = np.flatnonzero(np.isin(grid[0], list(SIGNAL_COLORS)))
    assert cols.size > 0, "expected a top-row signal band"
    left = int(cols[0])
    right = int(cols[-1])
    assert np.array_equal(cols, np.arange(left, right + 1)), "top band must be contiguous"
    band = grid[0, left : right + 1]
    assert 5 <= band.size <= 11 and band.size % 2 == 1, "band width must be odd and between 5 and 11"
    assert np.all(np.isin(band, list(SIGNAL_COLORS))), "band must use only red/grey"
    assert np.array_equal(band, band[::-1]), "band must be symmetric about the center column"
    if band.size > 1:
        assert np.all(band[:-1] != band[1:]), "band must alternate red and grey"
    return left, right


def _find_control(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid[1:, :] == RED)
    assert len(coords) == 1, "expected exactly one extra red control pixel below the top row"
    r, c = map(int, coords[0])
    r += 1
    assert r != CENTER and c != CENTER, "control pixel must not sit on the center row or center column"
    return r, c


def _quadrant_bounds(shape: tuple[int, int], r: int, c: int) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    mid_r = h // 2
    mid_c = w // 2
    assert r != mid_r and c != mid_c
    if r < mid_r:
        r0, r1 = 0, mid_r - 1
    else:
        r0, r1 = mid_r + 1, h - 1
    if c < mid_c:
        c0, c1 = 0, mid_c - 1
    else:
        c0, c1 = mid_c + 1, w - 1
    return r0, r1, c0, c1


def _clean_input(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int).copy()
    keep = (g == BG) | (g == RED) | (g == GREY)
    g[~keep] = BG
    return g


def _render(
    grid: np.ndarray,
    *,
    restrict_to_quadrant: bool,
    use_offset: bool,
    erase_control: bool,
) -> np.ndarray:
    g = _clean_input(grid)
    h, w = map(int, g.shape)
    left, right = _band_span(g)
    control_r, control_c = _find_control(g)
    q_r0, q_r1, q_c0, q_c1 = _quadrant_bounds(g.shape, control_r, control_c)
    offset_k = control_r - q_r0

    out = np.full((h, w), BG, dtype=int)
    out[0, left : right + 1] = g[0, left : right + 1]

    for row in range(1, h):
        lo = left + row
        hi = right - row
        if lo > hi:
            break
        if use_offset and row - q_r0 < offset_k:
            continue
        if restrict_to_quadrant and not (q_r0 <= row <= q_r1):
            continue
        draw_lo = lo if not restrict_to_quadrant else max(lo, q_c0)
        draw_hi = hi if not restrict_to_quadrant else min(hi, q_c1)
        for col in range(draw_lo, draw_hi + 1):
            src = int(g[0, col - row])
            out[row, col] = _swap_signal(src, row)

    if erase_control:
        out[control_r, control_c] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase distractors, use the red control pixel to pick a quadrant and row offset, and draw only that delayed wedge slice.

    Input structure:
    - The grid is 15x15 with orange background.
    - The top row contains one centered odd-width red/grey alternating band.
    - Below the top row there is exactly one extra red control pixel plus blue/green/yellow distractor singletons.

    Rule:
    - Erase every distractor pixel to background.
    - Split the grid by the center row and center column, then find the quadrant containing the control pixel.
    - Let K be the control pixel's row depth inside that quadrant.
    - Starting from the top-row band, build the shrinking b5bb5719 wedge: row r fills columns left+r through right-r, taking the source color from r steps up-left and swapping red/grey once per row.
    - Keep only wedge pixels that lie inside the selected quadrant, skip that quadrant's first K rows of wedge, and erase the control pixel itself.

    Key insight:
    - The single red control pixel encodes two independent parameters at once: which quadrant is allowed to show the wedge, and how many top rows of that quadrant must be skipped before drawing starts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 15x15 grid"
    assert int(np.bincount(g.ravel(), minlength=10).argmax()) == BG, "expected orange background"
    return _render(g, restrict_to_quadrant=True, use_offset=True, erase_control=True)


def _build_band(width: int, start_color: int) -> np.ndarray:
    band = np.empty(width, dtype=int)
    other = GREY if int(start_color) == RED else RED
    for i in range(width):
        band[i] = start_color if i % 2 == 0 else other
    return band


def _control_row_choices(width: int) -> list[int]:
    max_row = (width - 1) // 2
    return list(range(2, max_row))


def _control_col_choices(width: int, row: int, right_side: bool) -> list[int]:
    left = CENTER - width // 2
    right = CENTER + width // 2
    wedge_lo = left + row
    wedge_hi = right - row
    if right_side:
        cols = range(CENTER + 1, GRID_SIZE)
    else:
        cols = range(0, CENTER)
    return [c for c in cols if c < wedge_lo or c > wedge_hi]


def _sample_distractors(control: tuple[int, int], count: int) -> list[tuple[int, int, int]]:
    blocked = {control}
    for nbr in neighbors8(*control):
        if 0 <= nbr[0] < GRID_SIZE and 0 <= nbr[1] < GRID_SIZE:
            blocked.add((int(nbr[0]), int(nbr[1])))

    candidates = [
        (r, c)
        for r in range(1, GRID_SIZE)
        for c in range(GRID_SIZE)
        if (r, c) not in blocked
    ]
    cells = random.sample(candidates, count)
    return [(r, c, random.choice(DISTRACTOR_COLORS)) for r, c in cells]


def _gen_attempt() -> np.ndarray | None:
    width = random.choice(GEN_BAND_WIDTHS)
    row_choices = _control_row_choices(width)
    if not row_choices:
        return None
    control_row = random.choice(row_choices)
    right_side = bool(random.getrandbits(1))
    col_choices = _control_col_choices(width, control_row, right_side)
    if not col_choices:
        return None
    control_col = random.choice(col_choices)

    grid = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    band = _build_band(width, random.choice((RED, GREY)))
    left = CENTER - width // 2
    grid[0, left : left + width] = band
    grid[control_row, control_col] = RED

    distractor_count = random.randint(5, 10)
    for r, c, color in _sample_distractors((control_row, control_col), distractor_count):
        grid[r, c] = color

    expected = _render(grid, restrict_to_quadrant=True, use_offset=True, erase_control=True)
    if not np.array_equal(main(grid), expected):
        return None

    no_offset = _render(grid, restrict_to_quadrant=True, use_offset=False, erase_control=True)
    if np.array_equal(expected, no_offset):
        return None

    no_quadrant = _render(grid, restrict_to_quadrant=False, use_offset=True, erase_control=True)
    if np.array_equal(expected, no_quadrant):
        return None

    if np.count_nonzero(expected[1:, :] != BG) == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the quadrant-gated wedge task")
