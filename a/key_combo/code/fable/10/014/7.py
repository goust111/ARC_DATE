from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg


BLACK = int(Color.BLACK)
BLUE = int(Color.BLUE)
GREEN = int(Color.GREEN)
GEN_ATTEMPTS = 400


def _has_horizontal_pair(values: np.ndarray) -> bool:
    return bool(np.any((values[:-1] == BLUE) & (values[1:] == BLUE)))


def _has_vertical_pair(values: np.ndarray) -> bool:
    return bool(np.any((values[:-1] == BLUE) & (values[1:] == BLUE)))


def _marker_box(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == GREEN)
    assert coords.shape == (4, 2), "expected four single-pixel green corner markers"
    rows = sorted(set(int(r) for r in coords[:, 0]))
    cols = sorted(set(int(c) for c in coords[:, 1]))
    assert len(rows) == 2 and len(cols) == 2, "markers must form an axis-aligned rectangle"
    expected = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}
    assert expected == {(int(r), int(c)) for r, c in coords}, "markers must occupy the four corners"
    return rows[0], cols[0], rows[1], cols[1]


def _gcd_diffs(values: list[int]) -> int:
    g = 0
    for a, b in zip(values, values[1:]):
        g = math.gcd(g, int(b - a))
    return g


def _infer_period_phase(marked_positions: list[int]) -> tuple[int, int]:
    marked = sorted(set(int(v) for v in marked_positions))
    assert len(marked) >= 2, "need at least two explicit traces to infer a period"
    pitch = _gcd_diffs(marked)
    assert pitch >= 2, "expected a non-trivial period"
    phase = marked[0] % pitch
    assert all(v % pitch == phase for v in marked), "all marked traces must share one phase"
    return pitch, phase


def _all_axis_lines(size: int, pitch: int, phase: int) -> list[int]:
    return [idx for idx in range(int(phase), int(size), int(pitch))]


def _fill_box(
    grid: np.ndarray,
    line_rows: list[int],
    line_cols: list[int],
    box: tuple[int, int, int, int],
) -> np.ndarray:
    top, left, bottom, right = box
    out = np.array(grid, copy=True)
    for r in line_rows:
        if top <= r <= bottom:
            out[r, left : right + 1] = BLUE
    for c in line_cols:
        if left <= c <= right:
            out[top : bottom + 1, c] = BLUE
    out[grid == GREEN] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden periodic grid and complete it only inside the corner-marked rectangle.

    Input structure:
    - A black background contains exactly four green single-pixel corner markers that define an axis-aligned rectangle.
    - Blue pixels form short horizontal and vertical trace fragments from one underlying infinite periodic grid.
    - Some true grid rows or columns may have no same-orientation fragment inside the marked rectangle, and fragments outside the rectangle are only clues.

    Rule:
    - Detect explicit horizontal traces by rows containing a horizontal blue adjacency, and explicit vertical traces by columns containing a vertical blue adjacency.
    - Infer one row period/phase and one column period/phase from the spacings of those explicit traces.
    - Using that periodic grid, draw every full blue horizontal and vertical line segment whose row or column lies inside the green-corner rectangle.
    - Keep all fragments outside the rectangle unchanged and keep the green markers unchanged.

    Key insight:
    - The visible blue pieces do not directly list every line that must be completed; their repeated spacing reveals the missing grid rows and columns that also have to be drawn inside the marker box.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(get_bg(g)) == BLACK, "expected black background"

    top, left, bottom, right = _marker_box(g)
    marked_rows = [r for r in range(g.shape[0]) if _has_horizontal_pair(g[r, :])]
    marked_cols = [c for c in range(g.shape[1]) if _has_vertical_pair(g[:, c])]
    assert marked_rows and marked_cols, "expected both horizontal and vertical blue traces"

    row_pitch, row_phase = _infer_period_phase(marked_rows)
    col_pitch, col_phase = _infer_period_phase(marked_cols)
    all_rows = _all_axis_lines(g.shape[0], row_pitch, row_phase)
    all_cols = _all_axis_lines(g.shape[1], col_pitch, col_phase)
    return _fill_box(g, all_rows, all_cols, (top, left, bottom, right))


def _choose_axis(size: int) -> tuple[int, int, list[int]] | None:
    choices: list[tuple[int, int, list[int]]] = []
    for pitch in range(3, 8):
        for phase in range(pitch):
            lines = list(range(phase, size, pitch))
            if not (5 <= len(lines) <= 8):
                continue
            if lines[0] < 1 or lines[-1] > size - 2:
                continue
            choices.append((pitch, phase, lines))
    return random.choice(choices) if choices else None


def _choose_box(axis_lines: list[int], size: int) -> tuple[int, int, list[int]] | None:
    line_set = set(axis_lines)
    non_lines = [idx for idx in range(1, size - 1) if idx not in line_set]
    candidates: list[tuple[int, int, list[int]]] = []
    for top in non_lines:
        for bottom in non_lines:
            if bottom <= top:
                continue
            inside = [v for v in axis_lines if top < v < bottom]
            if not (3 <= len(inside) <= 5):
                continue
            if bottom - top < 7:
                continue
            if sum(v < top for v in axis_lines) < 1 or sum(v > bottom for v in axis_lines) < 1:
                continue
            candidates.append((top, bottom, inside))
    return random.choice(candidates) if candidates else None


def _choose_marked_lines(all_lines: list[int], inside_lines: list[int], pitch: int) -> list[int] | None:
    inside = set(inside_lines)
    for _ in range(200):
        marked = [v for v in all_lines if random.random() < 0.72]
        if len(marked) < 3:
            continue
        if sum(v in inside for v in marked) < 2:
            continue
        if not any(v not in inside for v in marked):
            continue
        if not any(v in inside and v not in marked for v in inside_lines):
            continue
        if _gcd_diffs(marked) != pitch:
            continue
        return marked
    return None


def _zones(size: int, start: int, end: int) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    if start >= 2:
        spans.append(("before", 0, start - 1))
    if end - start >= 3:
        spans.append(("inside", start + 1, end - 1))
    if size - end - 1 >= 2:
        spans.append(("after", end + 1, size - 1))
    return spans


def _paint_segment(grid: np.ndarray, fixed: int, lo: int, hi: int, *, horizontal: bool) -> bool:
    span = hi - lo + 1
    if span < 2:
        return False
    max_len = min(5, span)
    if span >= 4 and random.random() < 0.7:
        max_len = min(max_len, span - 1)
    seg_len = random.randint(2, max_len)
    start = random.randint(lo, hi - seg_len + 1)
    if horizontal:
        grid[fixed, start : start + seg_len] = BLUE
    else:
        grid[start : start + seg_len, fixed] = BLUE
    return True


def _draw_horizontal_fragments(grid: np.ndarray, row: int, box: tuple[int, int, int, int]) -> None:
    top, left, bottom, right = box
    spans = _zones(grid.shape[1], left, right)
    if not spans:
        return

    chosen: list[tuple[str, int, int]] = []
    inside_span = next((span for span in spans if span[0] == "inside"), None)
    if top < row < bottom and inside_span is not None and random.random() < 0.8:
        chosen.append(inside_span)

    shuffled = spans[:]
    random.shuffle(shuffled)
    target_count = 1 + int(len(spans) > 1 and random.random() < 0.35)
    for span in shuffled:
        if span not in chosen:
            chosen.append(span)
        if len(chosen) >= target_count:
            break

    painted = False
    for _name, lo, hi in chosen:
        painted |= _paint_segment(grid, row, lo, hi, horizontal=True)
    if not painted:
        _paint_segment(grid, row, spans[0][1], spans[0][2], horizontal=True)


def _draw_vertical_fragments(grid: np.ndarray, col: int, box: tuple[int, int, int, int]) -> None:
    top, left, bottom, right = box
    spans = _zones(grid.shape[0], top, bottom)
    if not spans:
        return

    chosen: list[tuple[str, int, int]] = []
    inside_span = next((span for span in spans if span[0] == "inside"), None)
    if left < col < right and inside_span is not None and random.random() < 0.8:
        chosen.append(inside_span)

    shuffled = spans[:]
    random.shuffle(shuffled)
    target_count = 1 + int(len(spans) > 1 and random.random() < 0.35)
    for span in shuffled:
        if span not in chosen:
            chosen.append(span)
        if len(chosen) >= target_count:
            break

    painted = False
    for _name, lo, hi in chosen:
        painted |= _paint_segment(grid, col, lo, hi, horizontal=False)
    if not painted:
        _paint_segment(grid, col, spans[0][1], spans[0][2], horizontal=False)


def _all_blue_on_grid_lines(grid: np.ndarray, row_lines: list[int], col_lines: list[int]) -> bool:
    row_set = set(row_lines)
    col_set = set(col_lines)
    for r, c in np.argwhere(grid == BLUE):
        if int(r) not in row_set and int(c) not in col_set:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(18, 30), random.randint(18, 30))
    row_axis = _choose_axis(shape[0])
    col_axis = _choose_axis(shape[1])
    if row_axis is None or col_axis is None:
        return None
    row_pitch, row_phase, all_rows = row_axis
    col_pitch, col_phase, all_cols = col_axis

    row_box = _choose_box(all_rows, shape[0])
    col_box = _choose_box(all_cols, shape[1])
    if row_box is None or col_box is None:
        return None
    top, bottom, inside_rows = row_box
    left, right, inside_cols = col_box
    box = (top, left, bottom, right)

    marked_rows = _choose_marked_lines(all_rows, inside_rows, row_pitch)
    marked_cols = _choose_marked_lines(all_cols, inside_cols, col_pitch)
    if marked_rows is None or marked_cols is None:
        return None

    hidden_inside_rows = [r for r in inside_rows if r not in marked_rows]
    hidden_inside_cols = [c for c in inside_cols if c not in marked_cols]
    if not hidden_inside_rows or not hidden_inside_cols:
        return None

    grid = np.full(shape, BLACK, dtype=int)
    for r, c in ((top, left), (top, right), (bottom, left), (bottom, right)):
        grid[r, c] = GREEN

    for row in marked_rows:
        _draw_horizontal_fragments(grid, row, box)
    for col in marked_cols:
        _draw_vertical_fragments(grid, col, box)

    if not _all_blue_on_grid_lines(grid, all_rows, all_cols):
        return None
    if [r for r in range(shape[0]) if _has_horizontal_pair(grid[r, :])] != marked_rows:
        return None
    if [c for c in range(shape[1]) if _has_vertical_pair(grid[:, c])] != marked_cols:
        return None

    expected = _fill_box(grid, all_rows, all_cols, box)
    observed_only = _fill_box(grid, marked_rows, marked_cols, box)
    any_blue_rows = [int(r) for r in np.where(np.any(grid == BLUE, axis=1))[0]]
    any_blue_cols = [int(c) for c in np.where(np.any(grid == BLUE, axis=0))[0]]
    any_blue_fill = _fill_box(grid, any_blue_rows, any_blue_cols, box)
    if np.array_equal(expected, observed_only) or np.array_equal(expected, any_blue_fill):
        return None

    outside_mask = np.ones(shape, dtype=bool)
    outside_mask[top : bottom + 1, left : right + 1] = False
    if not np.array_equal(expected[outside_mask], grid[outside_mask]):
        return None

    added = expected != grid
    if not any(bool(np.any(added[r, left : right + 1])) for r in hidden_inside_rows):
        return None
    if not any(bool(np.any(added[top : bottom + 1, c])) for c in hidden_inside_cols):
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-grid restoration input")
