from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color


MARKER_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE)
CYCLE_COLORS = (Color.GREY, Color.TEAL, Color.MAROON)
BLACK = Color.BLACK

GRID_HEIGHT_RANGE = (11, 15)
GRID_WIDTH_RANGE = (10, 13)
MARKED_ROW_COUNT_RANGE = (5, 7)
ACTIVE_ROW_COUNT = 4
GEN_ATTEMPTS = 400

_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _grid_sig(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _reference_coords(grid: np.ndarray) -> list[tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(np.isin(grid, CYCLE_COLORS))]
    assert len(coords) == 3, "expected exactly three cycle reference pixels"
    rows = sorted(r for r, _ in coords)
    cols = {c for _, c in coords}
    assert len(cols) == 1, "cycle reference should be vertical"
    assert rows == list(range(rows[0], rows[0] + 3)), "cycle reference rows should be consecutive"
    return sorted(coords)


def _cycle_from_grid(grid: np.ndarray) -> list[int]:
    return [int(grid[r, c]) for r, c in _reference_coords(grid)]


def _marked_rows(grid: np.ndarray) -> list[int]:
    rows: list[int] = []
    for r in range(grid.shape[0]):
        left = int(grid[r, 0])
        right = int(grid[r, -1])
        if left in MARKER_COLORS and right in MARKER_COLORS:
            rows.append(int(r))
    return rows


def _active_rows(grid: np.ndarray) -> list[int]:
    rows: list[int] = []
    for r in _marked_rows(grid):
        if int(grid[r, 0]) == int(grid[r, -1]):
            rows.append(int(r))
    return rows


def _spec_output(grid: np.ndarray, *, bottom_anchored: bool = True, fill_by_marker: bool = False, activate_all_marked: bool = False) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    cycle = _cycle_from_grid(g)
    out = g.copy()

    marked = _marked_rows(g)
    active = _active_rows(g)
    inactive = [r for r in marked if r not in active]

    if activate_all_marked:
        active = marked[:]
        inactive = []

    if not activate_all_marked:
        for r in inactive:
            out[r, 0] = BLACK
            out[r, -1] = BLACK

    ordered = sorted(active, reverse=bottom_anchored)
    for k, r in enumerate(ordered):
        fill = int(g[r, 0]) if fill_by_marker else int(cycle[k % len(cycle)])
        out[r, :] = fill

    for r, c in _reference_coords(g):
        out[r, c] = int(g[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only same-color edge-paired rows using a bottom-anchored independent color cycle.

    Input structure:
    - Black is the background.
    - Several rows have one marker pixel on the left border and one on the right border,
      with marker colors drawn from a palette disjoint from the cycle-reference colors.
    - The upper-right area contains a 3-pixel vertical reference whose top-to-bottom colors
      define an independent fill cycle.

    Rule:
    - A row is active iff both border markers exist and have the same color.
    - If the two border markers on a marked row have different colors, erase just those two
      border pixels to black.
    - Order the active rows from bottom to top as indices 0, 1, 2, ...
    - Fill each active row across the full width with the cycle color at index
      `row_index mod 3`, while keeping the 3 reference pixels unchanged.

    Key insight:
    - The border-marker colors are only a gating test for which rows activate; the actual
      fill colors come from a separate cycle, and the cycle phase is anchored at the bottom.
    """
    return _spec_output(grid)


def _make_input() -> np.ndarray:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)

    ref_r0 = random.randint(0, 1)
    ref_c = random.choice((w - 3, w - 2))
    cycle = random.sample(list(CYCLE_COLORS), len(CYCLE_COLORS))
    for i, color in enumerate(cycle):
        grid[ref_r0 + i, ref_c] = int(color)

    marked_count = random.randint(*MARKED_ROW_COUNT_RANGE)
    available_rows = [r for r in range(h) if r not in range(ref_r0, ref_r0 + 3)]
    marked_rows = sorted(random.sample(available_rows, marked_count))
    active_rows = set(random.sample(marked_rows, ACTIVE_ROW_COUNT))

    for r in marked_rows:
        if r in active_rows:
            color = random.choice(MARKER_COLORS)
            grid[r, 0] = color
            grid[r, -1] = color
        else:
            left, right = random.sample(MARKER_COLORS, 2)
            grid[r, 0] = left
            grid[r, -1] = right

    return grid


def _semantic_checks(inp: np.ndarray) -> bool:
    h, w = inp.shape
    if h > 30 or w > 30:
        return False

    try:
        ref = _reference_coords(inp)
    except AssertionError:
        return False

    ref_rows = [r for r, _ in ref]
    ref_cols = [c for _, c in ref]
    if ref_rows[0] > 1 or ref_cols[0] < w - 3:
        return False

    marked = _marked_rows(inp)
    active = _active_rows(inp)
    inactive = [r for r in marked if r not in active]
    if not (MARKED_ROW_COUNT_RANGE[0] <= len(marked) <= MARKED_ROW_COUNT_RANGE[1]):
        return False
    if len(active) != ACTIVE_ROW_COUNT or len(inactive) < 1:
        return False

    cycle = _cycle_from_grid(inp)
    if len(set(cycle)) != 3:
        return False

    expected = _spec_output(inp)
    if not np.array_equal(main(inp), expected):
        return False

    top_anchored = _spec_output(inp, bottom_anchored=False)
    marker_fill = _spec_output(inp, fill_by_marker=True)
    all_marked = _spec_output(inp, activate_all_marked=True)
    if np.array_equal(expected, top_anchored):
        return False
    if np.array_equal(expected, marker_fill):
        return False
    if np.array_equal(expected, all_marked):
        return False

    for r in inactive:
        if int(expected[r, 0]) != BLACK or int(expected[r, -1]) != BLACK:
            return False
    for r in active:
        if int(expected[r, 0]) not in CYCLE_COLORS or int(expected[r, -1]) not in CYCLE_COLORS:
            return False
    for r, c in ref:
        if int(expected[r, c]) != int(inp[r, c]):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp = _make_input()
        if not _semantic_checks(inp):
            continue

        out = main(inp)
        inp_sig = _grid_sig(inp)
        out_sig = _grid_sig(out)
        if inp_sig in _SEEN_INPUT_SIGS or out_sig in _SEEN_OUTPUT_SIGS:
            continue

        _SEEN_INPUT_SIGS.add(inp_sig)
        _SEEN_OUTPUT_SIGS.add(out_sig)
        return inp
    raise RuntimeError("failed to generate a valid input for puzzle 3")
