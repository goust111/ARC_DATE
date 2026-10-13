from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, get_bg
import random


def _solid_rect_bbox(g: np.ndarray, color: int) -> tuple[int, int, int, int]:
    box = bbox(g == color)
    assert box is not None, "expected color to appear"
    r0, c0, r1, c1 = box
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    assert int(np.count_nonzero(g == color)) == area, "colored object must be a solid rectangle"
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "colored object bbox must be solid"
    return box


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the orange rectangle onto the maroon rectangle and mark gated corridor crossings in red.

    Input structure
    - The dominant background contains black obstacle segments.
    - There is one solid orange rectangle and one solid maroon rectangle of the same size.
    - The two rectangles share either their row span or their column span, so the move is straight and axis-aligned.

    Rule
    - Erase the original orange rectangle to the background.
    - Replace the maroon rectangle with the orange rectangle.
    - Between the two rectangles, paint red across the moving rectangle's width or height only at corridor cells flanked by black obstacle segments perpendicular to the movement.

    Key insight
    - The black segments define gates in the straight corridor; only those gated openings, not every swept background cell, become red.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    source = _solid_rect_bbox(g, Color.ORANGE)
    target = _solid_rect_bbox(g, Color.MAROON)
    sr0, sc0, sr1, sc1 = source
    tr0, tc0, tr1, tc1 = target

    assert (sr1 - sr0, sc1 - sc0) == (tr1 - tr0, tc1 - tc0), "rectangles must have the same size"
    same_cols = (sc0, sc1) == (tc0, tc1)
    same_rows = (sr0, sr1) == (tr0, tr1)
    assert same_cols != same_rows, "rectangles must align on exactly one axis"

    out = g.copy()
    out[g == Color.ORANGE] = bg
    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = Color.ORANGE

    H, W = g.shape
    if same_cols:
        c0, c1 = sc0, sc1
        rows = range(tr1 + 1, sr0) if sr0 > tr0 else range(sr1 + 1, tr0)
        for r in rows:
            gated = (
                c0 > 0
                and c1 + 1 < W
                and g[r, c0 - 1] == Color.BLACK
                and g[r, c1 + 1] == Color.BLACK
            )
            if gated:
                assert np.all(g[r, c0 : c1 + 1] == bg), "gated opening must be background before painting"
                out[r, c0 : c1 + 1] = Color.RED
    else:
        r0, r1 = sr0, sr1
        cols = range(tc1 + 1, sc0) if sc0 > tc0 else range(sc1 + 1, tc0)
        for c in cols:
            gated = (
                r0 > 0
                and r1 + 1 < H
                and g[r0 - 1, c] == Color.BLACK
                and g[r1 + 1, c] == Color.BLACK
            )
            if gated:
                assert np.all(g[r0 : r1 + 1, c] == bg), "gated opening must be background before painting"
                out[r0 : r1 + 1, c] = Color.RED

    return out

# --------------- generator (o5x) ---------------

# Canvas and object geometry.
GRID_SIZE_MIN, GRID_SIZE_MAX = 8, 26
RECT_SIDE_MIN, RECT_SIDE_MAX = 1, 4
RECT_CANVAS_MARGIN = 4
GAP_MIN, GAP_MAX = 5, 7
CROSS_AXIS_PADDING = 2

# Gate count, spacing, and wall thickness.
BETWEEN_GATE_COUNTS = (2, 3)
GATE_TOTAL_MIN, GATE_TOTAL_MAX = 2, 8
GATE_SPACING_MIN, GATE_SPACING_MAX = 2, 4
GATE_SEGMENT_LEN = 2

# Palette and retry controls.
BACKGROUND_COLOR = Color.PINK
GATE_COLOR = Color.BLACK
RED_COLOR = Color.RED
SOURCE_COLOR = Color.ORANGE
TARGET_COLOR = Color.MAROON
BINARY_CHOICES = [False, True]
GENERATION_ATTEMPTS = 300


def _gen_extensions(anchor: int, limit: int, count: int, sign: int) -> list[list[int]]:
    if count == 0:
        return [[]]
    out: list[list[int]] = []
    for step in range(GATE_SPACING_MIN, GATE_SPACING_MAX + 1):
        pos = anchor + sign * step
        if not (0 <= pos <= limit):
            continue
        for tail in _gen_extensions(pos, limit, count - 1, sign):
            out.append(([pos] + tail) if sign > 0 else (tail + [pos]))
    return out


def _gen_between_positions(gap_cells: list[int], count: int) -> list[tuple[int, ...]]:
    """Enumerate in-gap gate positions with valid spacing, preserving nested-loop order."""
    out: list[tuple[int, ...]] = []
    if count == 2:
        for a in gap_cells:
            for b in gap_cells:
                if GATE_SPACING_MIN <= b - a <= GATE_SPACING_MAX:
                    out.append((a, b))
    else:
        for a in gap_cells:
            for b in gap_cells:
                for c in gap_cells:
                    if (
                        GATE_SPACING_MIN <= b - a <= GATE_SPACING_MAX
                        and GATE_SPACING_MIN <= c - b <= GATE_SPACING_MAX
                    ):
                        out.append((a, b, c))
    return out


def _gen_gate_positions(size: int, gap_cells: list[int]) -> tuple[list[int], list[int]] | None:
    choices: list[tuple[list[int], list[int]]] = []
    for between_count in BETWEEN_GATE_COUNTS:
        for between in _gen_between_positions(gap_cells, between_count):
            remaining = GATE_TOTAL_MAX - between_count
            max_left = min(remaining, (between[0] + 1) // GATE_SPACING_MIN)
            max_right = min(remaining, (size - between[-1]) // GATE_SPACING_MIN)
            for left_count in range(max_left + 1):
                for right_count in range(max_right + 1):
                    total = between_count + left_count + right_count
                    if not (GATE_TOTAL_MIN <= total <= GATE_TOTAL_MAX):
                        continue
                    for left in _gen_extensions(between[0], size - 1, left_count, -1):
                        for right in _gen_extensions(between[-1], size - 1, right_count, 1):
                            choices.append((left + list(between) + right, list(between)))
    if not choices:
        return None
    return random.choice(choices)


def _sample_axis_layout(size: int, rect_side: int, gap: int) -> tuple[int, int, int, int, list[int]]:
    """Sample aligned source/target spans and return the corridor cells between them."""
    source_first = random.choice(BINARY_CHOICES)
    span0 = random.randint(0, size - (2 * rect_side + gap))
    first0, second0 = span0, span0 + rect_side + gap
    src_axis0, dst_axis0 = (first0, second0) if source_first else (second0, first0)
    gap_cells = list(range(span0 + rect_side, span0 + rect_side + gap))
    return src_axis0, src_axis0 + rect_side - 1, dst_axis0, dst_axis0 + rect_side - 1, gap_cells


def _paint_rect(grid: np.ndarray, horizontal: bool, axis0: int, axis1: int, cross0: int, cross1: int, color: int) -> None:
    if horizontal:
        grid[cross0 : cross1 + 1, axis0 : axis1 + 1] = color
    else:
        grid[axis0 : axis1 + 1, cross0 : cross1 + 1] = color


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    rect_side = random.randint(RECT_SIDE_MIN, min(RECT_SIDE_MAX, size - RECT_CANVAS_MARGIN))
    gap = random.randint(GAP_MIN, GAP_MAX)
    if 2 * rect_side + gap > size:
        return None

    horizontal = random.choice(BINARY_CHOICES)
    src_axis0, src_axis1, dst_axis0, dst_axis1, gap_cells = _sample_axis_layout(size, rect_side, gap)
    gate_choice = _gen_gate_positions(size, gap_cells)
    if gate_choice is None:
        return None
    gate_positions, red_positions = gate_choice

    cross0 = random.randint(CROSS_AXIS_PADDING, size - rect_side - CROSS_AXIS_PADDING)
    cross1 = cross0 + rect_side - 1
    output = np.full((size, size), BACKGROUND_COLOR, dtype=int)

    for pos in gate_positions:
        if horizontal:
            output[cross0 - GATE_SEGMENT_LEN : cross0, pos] = GATE_COLOR
            output[cross1 + 1 : cross1 + 1 + GATE_SEGMENT_LEN, pos] = GATE_COLOR
        else:
            output[pos, cross0 - GATE_SEGMENT_LEN : cross0] = GATE_COLOR
            output[pos, cross1 + 1 : cross1 + 1 + GATE_SEGMENT_LEN] = GATE_COLOR
    _paint_rect(output, horizontal, dst_axis0, dst_axis1, cross0, cross1, SOURCE_COLOR)
    for pos in red_positions:
        if horizontal:
            output[cross0 : cross1 + 1, pos] = RED_COLOR
        else:
            output[pos, cross0 : cross1 + 1] = RED_COLOR

    inp = output.copy()
    inp[inp == RED_COLOR] = BACKGROUND_COLOR
    _paint_rect(inp, horizontal, dst_axis0, dst_axis1, cross0, cross1, TARGET_COLOR)
    _paint_rect(inp, horizontal, src_axis0, src_axis1, cross0, cross1, SOURCE_COLOR)

    if np.array_equal(main(inp), output):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fc10701f input")
