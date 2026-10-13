from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, get_bg


BACKGROUND = Color.BLACK
GATE = Color.GREY
MARK = Color.RED
SOURCE = Color.ORANGE
TARGET = Color.MAROON

NARROW = 7
INITIAL_H = 3
INITIAL_W = 2
CROSS_START = 2
LONG_MIN = 18
LONG_MAX = 27
GATE_COUNT_CHOICES = (1, 2, 3)
GATE_SPACING = 4
GEN_ATTEMPTS = 400


def _solid_rect_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    box = bbox(grid == color)
    assert box is not None, f"expected color {color} to appear"
    r0, c0, r1, c1 = map(int, box)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(patch == color), f"color {color} must form one solid rectangle"
    assert int(np.count_nonzero(grid == color)) == patch.size, f"color {color} must be unique"
    return r0, c0, r1, c1


def _paint_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[int(r0) : int(r0) + int(h), int(c0) : int(c0) + int(w)] = int(color)


def _gate_positions_between(
    start_after: int,
    end_before: int,
    gate_positions: list[int],
    increasing: bool,
) -> list[int]:
    lo = min(int(start_after), int(end_before))
    hi = max(int(start_after), int(end_before))
    inside = [int(p) for p in gate_positions if lo < int(p) < hi]
    inside.sort(reverse=not increasing)
    return inside


def _opening_indices(line: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(line != GATE)
    assert idx.size in (2, 3), "gate opening must be size 2 or 3"
    assert np.all(np.diff(idx) == 1), "gate opening must be contiguous"
    return idx


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the orange rectangle through the gated corridor, rotating it 90 degrees after each gate and marking every gate crossing in red.

    Input structure:
    - The background is black, with one solid orange source rectangle and one solid maroon target rectangle.
    - The source and target are axis-aligned 3x2 rectangles placed at opposite ends of a narrow horizontal or vertical corridor.
    - Between them are 1 to 3 grey one-column or one-row gates, each leaving a centered opening sized for the rectangle's current orientation.

    Rule:
    - Erase the original orange rectangle to background.
    - Keep every grey gate, and color each traversed gate opening red using the rectangle orientation that passes through that gate.
    - After the rectangle fully clears a gate, rotate it 90 degrees clockwise; after all gates, place the orange rectangle at the target end with its accumulated final orientation, overwriting the maroon marker.

    Key insight:
    - Gate order matters twice: each gate both leaves a red footprint matching the current orientation and flips the rectangle dimensions for all later gates and for the final landing shape.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == BACKGROUND, "expected black background"

    sr0, sc0, sr1, sc1 = _solid_rect_bbox(g, SOURCE)
    tr0, tc0, tr1, tc1 = _solid_rect_bbox(g, TARGET)
    sh, sw = sr1 - sr0 + 1, sc1 - sc0 + 1
    th, tw = tr1 - tr0 + 1, tc1 - tc0 + 1
    assert (sh, sw) == (INITIAL_H, INITIAL_W)
    assert (th, tw) == (INITIAL_H, INITIAL_W)

    same_rows = (sr0, sr1) == (tr0, tr1)
    same_cols = (sc0, sc1) == (tc0, tc1)
    assert same_rows != same_cols, "source and target must align on exactly one axis"

    out = g.copy()
    out[g == SOURCE] = bg
    out[g == TARGET] = bg

    cur_h, cur_w = sh, sw
    if same_rows:
        moving_right = sc0 < tc0
        gate_cols = np.flatnonzero(np.any(g == GATE, axis=0)).tolist()
        ordered_gates = _gate_positions_between(sc1, tc0, gate_cols, increasing=moving_right)
        for c in ordered_gates:
            rows = _opening_indices(g[:, c])
            assert rows.size == cur_h, "horizontal gate opening must match current rectangle height"
            out[rows, c] = MARK
            cur_h, cur_w = cur_w, cur_h

        final_r0 = sr0
        final_c0 = (tc1 - cur_w + 1) if moving_right else tc0
        _paint_rect(out, final_r0, final_c0, cur_h, cur_w, SOURCE)
    else:
        moving_down = sr0 < tr0
        gate_rows = np.flatnonzero(np.any(g == GATE, axis=1)).tolist()
        ordered_gates = _gate_positions_between(sr1, tr0, gate_rows, increasing=moving_down)
        for r in ordered_gates:
            cols = _opening_indices(g[r, :])
            assert cols.size == cur_w, "vertical gate opening must match current rectangle width"
            out[r, cols] = MARK
            cur_h, cur_w = cur_w, cur_h

        final_r0 = (tr1 - cur_h + 1) if moving_down else tr0
        final_c0 = sc0
        _paint_rect(out, final_r0, final_c0, cur_h, cur_w, SOURCE)

    return out


def _sample_gate_positions(long_side: int, gate_count: int) -> list[int] | None:
    candidates = list(range(4, long_side - 4))
    if len(candidates) < gate_count:
        return None
    for _ in range(200):
        picked = sorted(random.sample(candidates, gate_count))
        if all(b - a >= GATE_SPACING for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _build_case(
    horizontal: bool,
    source_at_low_end: bool,
    gate_positions: list[int],
    long_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    if horizontal:
        H, W = NARROW, long_side
        grid = np.full((H, W), BACKGROUND, dtype=int)
        src_c0 = 0 if source_at_low_end else W - INITIAL_W
        dst_c0 = W - INITIAL_W if source_at_low_end else 0
        _paint_rect(grid, CROSS_START, src_c0, INITIAL_H, INITIAL_W, SOURCE)
        _paint_rect(grid, CROSS_START, dst_c0, INITIAL_H, INITIAL_W, TARGET)

        traversal = gate_positions if source_at_low_end else list(reversed(gate_positions))
        cur_h, cur_w = INITIAL_H, INITIAL_W
        for c in traversal:
            grid[:, c] = GATE
            grid[CROSS_START : CROSS_START + cur_h, c] = BACKGROUND
            cur_h, cur_w = cur_w, cur_h

        expected = grid.copy()
        expected[expected == SOURCE] = BACKGROUND
        expected[expected == TARGET] = BACKGROUND
        cur_h, cur_w = INITIAL_H, INITIAL_W
        for c in traversal:
            expected[CROSS_START : CROSS_START + cur_h, c] = MARK
            cur_h, cur_w = cur_w, cur_h
        final_c0 = (W - cur_w) if source_at_low_end else 0
        _paint_rect(expected, CROSS_START, final_c0, cur_h, cur_w, SOURCE)
        return grid, expected

    H, W = long_side, NARROW
    grid = np.full((H, W), BACKGROUND, dtype=int)
    src_r0 = 0 if source_at_low_end else H - INITIAL_H
    dst_r0 = H - INITIAL_H if source_at_low_end else 0
    _paint_rect(grid, src_r0, CROSS_START, INITIAL_H, INITIAL_W, SOURCE)
    _paint_rect(grid, dst_r0, CROSS_START, INITIAL_H, INITIAL_W, TARGET)

    traversal = gate_positions if source_at_low_end else list(reversed(gate_positions))
    cur_h, cur_w = INITIAL_H, INITIAL_W
    for r in traversal:
        grid[r, :] = GATE
        grid[r, CROSS_START : CROSS_START + cur_w] = BACKGROUND
        cur_h, cur_w = cur_w, cur_h

    expected = grid.copy()
    expected[expected == SOURCE] = BACKGROUND
    expected[expected == TARGET] = BACKGROUND
    cur_h, cur_w = INITIAL_H, INITIAL_W
    for r in traversal:
        expected[r, CROSS_START : CROSS_START + cur_w] = MARK
        cur_h, cur_w = cur_w, cur_h
    final_r0 = (H - cur_h) if source_at_low_end else 0
    _paint_rect(expected, final_r0, CROSS_START, cur_h, cur_w, SOURCE)
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([False, True])
    source_at_low_end = random.choice([False, True])
    gate_count = random.choice(GATE_COUNT_CHOICES)
    long_side = random.randint(LONG_MIN, LONG_MAX)
    gate_positions = _sample_gate_positions(long_side, gate_count)
    if gate_positions is None:
        return None

    inp, expected = _build_case(horizontal, source_at_low_end, gate_positions, long_side)
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(actual, expected):
        return None
    if int(np.count_nonzero(expected == MARK)) != sum((INITIAL_H if i % 2 == 0 else INITIAL_W) for i in range(gate_count)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gated-rotation corridor sample")
