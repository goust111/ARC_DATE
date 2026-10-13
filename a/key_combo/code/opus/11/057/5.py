from __future__ import annotations

from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
try:
    from arc_utils import Color, get_bg
except Exception:
    try:
        _ARC_UTILS = SourceFileLoader("_arc_utils_opus_11_057_5", str(_ARC_UTILS_PATH)).load_module()
        Color = _ARC_UTILS.Color
        get_bg = _ARC_UTILS.get_bg
    except Exception:
        class Color:
            BLACK = 0
            RED = 2
            GREEN = 3
            TEAL = 8

        def get_bg(grid: np.ndarray) -> int:
            return int(Counter(map(int, np.asarray(grid).flat)).most_common(1)[0][0])


BG = Color.BLACK
MARKER = Color.TEAL
PAIR = Color.RED
ENVELOPE = Color.GREEN

PITCH = 6
CELL_LAYOUTS = ((2, 2), (2, 3), (3, 2), (3, 3))
FIRST_MARKER_CHOICES = (0, 1, 2)
INNER_PAIR_OFFSETS = (2, 4)
OUTER_PAIR_OFFSETS = (1, 5)
ROW_OFFSETS = (1, 2, 3, 4, 5)
GENERATION_ATTEMPTS = 400


def _infer_lattice(marker_mask: np.ndarray) -> tuple[list[int], list[int], int, int]:
    coords = np.argwhere(marker_mask)
    assert coords.size > 0, "expected teal lattice markers"

    rows = sorted(set(int(r) for r, _ in coords))
    cols = sorted(set(int(c) for _, c in coords))
    assert len(rows) >= 2 and len(cols) >= 2, "lattice needs at least two marker rows and cols"

    row_steps = {b - a for a, b in zip(rows, rows[1:])}
    col_steps = {b - a for a, b in zip(cols, cols[1:])}
    assert len(row_steps) == 1 and len(col_steps) == 1, "marker rows/cols must be evenly spaced"

    row_pitch = row_steps.pop()
    col_pitch = col_steps.pop()
    assert row_pitch == col_pitch, "expected square lattice cells"
    assert row_pitch >= 3, "pitch must leave interior room for cell contents"

    expected = len(rows) * len(cols)
    assert int(coords.shape[0]) == expected, "teal markers must form a full cartesian lattice"
    return rows, cols, int(row_pitch), int(col_pitch)


def _cell_pairs(
    grid: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
) -> list[tuple[int, int, int, int]]:
    interior_top = int(top) + 1
    interior_bottom = int(bottom) - 1
    interior_left = int(left) + 1
    interior_right = int(right) - 1
    axis_sum = int(left) + int(right)

    pairs: list[tuple[int, int, int, int]] = []
    for r in range(interior_top, interior_bottom + 1):
        cols = np.flatnonzero(grid[r, interior_left : interior_right + 1] == PAIR) + interior_left
        col_set = {int(c) for c in cols}
        for c0 in cols:
            c1 = axis_sum - int(c0)
            if int(c0) < c1 and c1 in col_set:
                pairs.append((int(c1 - c0 - 1), int(r), int(c0), int(c1)))
    return pairs


def _paint_envelope(
    out: np.ndarray,
    row: int,
    c0: int,
    c1: int,
    top: int,
    bottom: int,
    left: int,
    right: int,
    bg: int,
) -> None:
    radius = (int(c1) - int(c0)) // 2
    interior_top = int(top) + 1
    interior_bottom = int(bottom) - 1
    interior_left = int(left) + 1
    interior_right = int(right) - 1

    for offset in range(1, radius + 1):
        fill_left = max(interior_left, int(c0) + offset)
        fill_right = min(interior_right, int(c1) - offset)
        if fill_left > fill_right:
            continue
        for rr in (int(row) - offset, int(row) + offset):
            if not (interior_top <= rr <= interior_bottom):
                continue
            view = out[rr, fill_left : fill_right + 1]
            view[view == int(bg)] = ENVELOPE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the closest mirrored pair in each lattice cell, then grow a clipped diamond envelope around that filled segment.

    Input structure:
    - The background is black, teal single pixels form a complete regular cartesian lattice, and red pixels appear only inside the lattice cells.
    - Each cell is the interior region between two consecutive teal marker rows and two consecutive teal marker columns.
    - Inside every cell, red pixels form mirrored horizontal pairs about the cell's vertical mid-axis.

    Rule:
    - Keep the teal lattice fixed and keep every original red endpoint.
    - In each cell, find all red mirrored pairs, choose the one with the smallest horizontal gap, and break any tie by the smaller row index.
    - Fill the chosen pair into a solid horizontal red segment, then paint the green diamond envelope induced by that segment, clipped to the cell interior and without overwriting red.

    Key insight:
    - The lattice is not decoration: it defines local coordinate frames, and only the nearest mirrored pair inside each frame is activated to seed the clipped diamond.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    marker_rows, marker_cols, _row_pitch, _col_pitch = _infer_lattice(g == MARKER)

    out = g.copy()
    for top, bottom in zip(marker_rows, marker_rows[1:]):
        for left, right in zip(marker_cols, marker_cols[1:]):
            pairs = _cell_pairs(g, top, bottom, left, right)
            assert pairs, "each cell must contain at least one mirrored pair"
            gap, row, c0, c1 = min(pairs, key=lambda item: (item[0], item[1]))
            assert gap >= 0, "pair gap must be non-negative"
            out[row, c0 : c1 + 1] = PAIR
            _paint_envelope(out, row, c0, c1, top, bottom, left, right, bg)

    return out


def _cell_bounds(
    marker_rows: list[int],
    marker_cols: list[int],
    cell_r: int,
    cell_c: int,
) -> tuple[int, int, int, int]:
    return (
        int(marker_rows[cell_r]),
        int(marker_rows[cell_r + 1]),
        int(marker_cols[cell_c]),
        int(marker_cols[cell_c + 1]),
    )


def _build_case(
    cell_rows: int,
    cell_cols: int,
    first_r: int,
    first_c: int,
    plans: list[dict[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    marker_rows = [int(first_r) + PITCH * i for i in range(int(cell_rows) + 1)]
    marker_cols = [int(first_c) + PITCH * j for j in range(int(cell_cols) + 1)]
    height = int(marker_rows[-1]) + 1
    width = int(marker_cols[-1]) + 1

    inp = np.full((height, width), BG, dtype=int)
    inp[np.ix_(marker_rows, marker_cols)] = MARKER

    for plan in plans:
        top, bottom, left, right = _cell_bounds(marker_rows, marker_cols, plan["cell_r"], plan["cell_c"])

        min_row = int(top) + int(plan["min_row"])
        other_row = int(top) + int(plan["other_row"])
        inner_left = int(left) + INNER_PAIR_OFFSETS[0]
        inner_right = int(left) + INNER_PAIR_OFFSETS[1]
        outer_left = int(left) + OUTER_PAIR_OFFSETS[0]
        outer_right = int(left) + OUTER_PAIR_OFFSETS[1]

        inp[min_row, [inner_left, inner_right]] = PAIR
        inp[other_row, [outer_left, outer_right]] = PAIR

    expected = inp.copy()
    for plan in plans:
        top, bottom, left, right = _cell_bounds(marker_rows, marker_cols, plan["cell_r"], plan["cell_c"])
        min_row = int(top) + int(plan["min_row"])
        inner_left = int(left) + INNER_PAIR_OFFSETS[0]
        inner_right = int(left) + INNER_PAIR_OFFSETS[1]
        expected[min_row, inner_left : inner_right + 1] = PAIR
        _paint_envelope(expected, min_row, inner_left, inner_right, top, bottom, left, right, BG)

    return inp, expected


def _sample_plans(cell_rows: int, cell_cols: int) -> list[dict[str, int]]:
    plans: list[dict[str, int]] = []
    for cell_r in range(int(cell_rows)):
        for cell_c in range(int(cell_cols)):
            min_row = random.choice(ROW_OFFSETS)
            other_choices = [row for row in ROW_OFFSETS if row != min_row]
            plans.append(
                {
                    "cell_r": int(cell_r),
                    "cell_c": int(cell_c),
                    "min_row": int(min_row),
                    "other_row": int(random.choice(other_choices)),
                }
            )
    return plans


def _plan_has_coverage(plans: list[dict[str, int]]) -> bool:
    min_rows = [plan["min_row"] for plan in plans]
    other_rows = [plan["other_row"] for plan in plans]
    has_edge = any(row in (1, 5) for row in min_rows)
    has_interior = any(row in (2, 3, 4) for row in min_rows)
    has_above = any(plan["min_row"] < plan["other_row"] for plan in plans)
    has_below = any(plan["min_row"] > plan["other_row"] for plan in plans)
    has_row_diversity = len(set(min_rows + other_rows)) >= 4
    return has_edge and has_interior and has_above and has_below and has_row_diversity


def _validate_input_structure(
    inp: np.ndarray,
    cell_rows: int,
    cell_cols: int,
    first_r: int,
    first_c: int,
) -> bool:
    marker_rows = [int(first_r) + PITCH * i for i in range(int(cell_rows) + 1)]
    marker_cols = [int(first_c) + PITCH * j for j in range(int(cell_cols) + 1)]
    expected_markers = np.zeros_like(inp, dtype=bool)
    expected_markers[np.ix_(marker_rows, marker_cols)] = True

    if not np.array_equal(inp == MARKER, expected_markers):
        return False
    for top, bottom in zip(marker_rows, marker_rows[1:]):
        for left, right in zip(marker_cols, marker_cols[1:]):
            pairs = sorted(_cell_pairs(inp, top, bottom, left, right))
            if len(pairs) != 2:
                return False
            if [pair[0] for pair in pairs] != [1, 3]:
                return False
            red_count = int(np.count_nonzero(inp[top + 1 : bottom, left + 1 : right] == PAIR))
            if red_count != 4:
                return False
    return int(np.count_nonzero(inp == ENVELOPE)) == 0


def _gen_attempt() -> np.ndarray | None:
    cell_rows, cell_cols = random.choice(CELL_LAYOUTS)
    first_r = random.choice(FIRST_MARKER_CHOICES)
    first_c = random.choice(FIRST_MARKER_CHOICES)
    plans = _sample_plans(cell_rows, cell_cols)
    if not _plan_has_coverage(plans):
        return None

    inp, expected = _build_case(cell_rows, cell_cols, first_r, first_c, plans)
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _validate_input_structure(inp, cell_rows, cell_cols, first_r, first_c):
        return None

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None

    green_count = int(np.count_nonzero(solved == ENVELOPE))
    if green_count < cell_rows * cell_cols:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 057/5")
