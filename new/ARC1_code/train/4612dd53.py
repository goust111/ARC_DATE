from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, get_bg
import itertools
import random


def _has_adjacent_pair(values: np.ndarray, color: int) -> bool:
    return bool(np.any((values[:-1] == color) & (values[1:] == color)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the broken blue grid lines by painting red into their missing background cells.

    Input structure:
    - A black background contains one blue interrupted rectilinear grid.
    - The foreground bbox contains the whole grid.
    - A horizontal line is marked by at least one horizontally adjacent pair of blue
      pixels in its row; a vertical line is marked by at least one vertically
      adjacent pair of blue pixels in its column.

    Rule:
    - Keep all input pixels unchanged.
    - Across the foreground bbox, fill background cells red on every marked
      horizontal row and every marked vertical column.

    Key insight:
    - Isolated blue pixels can be intersections of the opposite orientation, while
      adjacent blue pairs reveal which rows and columns are actual grid lines.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = set(map(int, np.unique(g)))
    line_colors = colors - {bg}
    assert bg == Color.BLACK, "expected black background"
    assert line_colors == {Color.BLUE}, "expected blue as the only foreground color"

    grid_bbox = bbox(g != bg)
    assert grid_bbox is not None, "expected a non-empty foreground grid"
    r0, c0, r1, c1 = grid_bbox

    line_rows = [
        r
        for r in range(r0, r1 + 1)
        if _has_adjacent_pair(g[r, c0 : c1 + 1], Color.BLUE)
    ]
    line_cols = [
        c
        for c in range(c0, c1 + 1)
        if _has_adjacent_pair(g[r0 : r1 + 1, c], Color.BLUE)
    ]
    assert line_rows, "expected at least one marked horizontal line"
    assert line_cols, "expected at least one marked vertical line"

    out = g.copy()
    for r in line_rows:
        span = out[r, c0 : c1 + 1]
        span[span == bg] = Color.RED
    for c in line_cols:
        span = out[r0 : r1 + 1, c]
        span[span == bg] = Color.RED

    return out

# --------------- generator (o5x) ---------------

# Colors used by the task.
_GEN_BLACK = 0
_GEN_BLUE = 1
_GEN_RED = 2
# Canvas and foreground bbox bounds.
_GEN_GRID_WIDTH = 13
_GEN_GRID_H_RANGE = (9, 13)
_GEN_BOX_SIZE_RANGE = (7, 9)
_GEN_PADDING_RANGE = (1, 4)
# Marked-line selection constraints.
_GEN_MARKED_COUNT_CHOICES = (2, 3)
_GEN_MIN_MARK_SPACING = 2
# Gap and segment rates.
_GEN_GAP_RATE_RANGE = (0.22, 0.44)
_GEN_INTERSECTION_GAP_PROB = 0.18
_GEN_VERTICAL_SEGMENT_RATE_RANGE = (0.50, 0.80)
# Candidate retry budget.
_GEN_MAX_ATTEMPTS = 300


def _gen_has_adjacent_blue(values: np.ndarray) -> bool:
    return bool(np.any((values[:-1] == _GEN_BLUE) & (values[1:] == _GEN_BLUE)))


def _gen_has_adjacent_true(values: np.ndarray) -> bool:
    return bool(np.any(values[:-1] & values[1:]))


def _gen_gap_bounds(length: int) -> tuple[int, int]:
    lo = int(np.ceil(_GEN_GAP_RATE_RANGE[0] * length))
    hi = int(np.floor(_GEN_GAP_RATE_RANGE[1] * length))
    return lo, max(lo, hi)


def _gen_marked_positions(size: int) -> list[int]:
    if random.choice(_GEN_MARKED_COUNT_CHOICES) == 2:
        return [0, size - 1]
    middle = random.randint(_GEN_MIN_MARK_SPACING, size - _GEN_MIN_MARK_SPACING - 1)
    return [0, middle, size - 1]


def _gen_choose_bbox_rows() -> tuple[int, int, int, int]:
    grid_h = random.randint(*_GEN_GRID_H_RANGE)
    choices = []
    for box_h in range(_GEN_BOX_SIZE_RANGE[0], _GEN_BOX_SIZE_RANGE[1] + 1):
        for top in range(_GEN_PADDING_RANGE[0], _GEN_PADDING_RANGE[1] + 1):
            bottom = grid_h - box_h - top
            if _GEN_PADDING_RANGE[0] <= bottom <= _GEN_PADDING_RANGE[1]:
                choices.append((box_h, top, bottom))
    box_h, top, bottom = random.choice(choices)
    return grid_h, box_h, top, bottom


def _gen_choose_bbox_cols() -> tuple[int, int, int]:
    box_w = random.randint(*_GEN_BOX_SIZE_RANGE)
    choices = []
    for left in range(_GEN_PADDING_RANGE[0], _GEN_PADDING_RANGE[1] + 1):
        right = _GEN_GRID_WIDTH - box_w - left
        if _GEN_PADDING_RANGE[0] <= right <= _GEN_PADDING_RANGE[1]:
            choices.append((left, right))
    left, right = random.choice(choices)
    return box_w, left, right


def _gen_complete_line_gaps(
    existing_gaps: np.ndarray,
    target_count: int,
    available_positions: list[int],
) -> tuple[int, ...] | None:
    needed = target_count - int(np.count_nonzero(existing_gaps))
    if needed < 0 or needed > len(available_positions):
        return None

    valid = []
    for combo in itertools.combinations(available_positions, needed):
        gaps = existing_gaps.copy()
        if combo:
            gaps[list(combo)] = True
        if _gen_has_adjacent_true(~gaps):
            valid.append(combo)
    if not valid:
        return None
    return random.choice(valid)


def _gen_seed_intersection_gaps(gap_mask, marked_rows, marked_cols, row_targets, col_targets, box_h, box_w):
    """Try optional intersection gaps without breaking any marked line."""
    intersections = [(r, c) for r in marked_rows for c in marked_cols]
    random.shuffle(intersections)
    row_inter_counts = {r: 0 for r in marked_rows}
    col_inter_counts = {c: 0 for c in marked_cols}
    for r, c in intersections:
        if random.random() >= _GEN_INTERSECTION_GAP_PROB:
            continue
        if row_inter_counts[r] >= row_targets[r] or col_inter_counts[c] >= col_targets[c]:
            continue
        gap_mask[r, c] = True
        row_blue = np.ones(box_w, dtype=bool)
        row_blue[gap_mask[r]] = False
        col_blue = np.ones(box_h, dtype=bool)
        col_blue[gap_mask[:, c]] = False
        if not (_gen_has_adjacent_true(row_blue) and _gen_has_adjacent_true(col_blue)):
            gap_mask[r, c] = False
            continue
        row_inter_counts[r] += 1
        col_inter_counts[c] += 1


def _gen_complete_axis_gaps(gap_mask, marked_rows, marked_cols, row_targets, col_targets, box_h, box_w):
    """Fill row-only then col-only gap quotas; return False if a quota is impossible."""
    row_set = set(marked_rows)
    col_set = set(marked_cols)
    row_only_cols = [c for c in range(box_w) if c not in col_set]
    for r in marked_rows:
        combo = _gen_complete_line_gaps(gap_mask[r], row_targets[r], row_only_cols)
        if combo is None:
            return False
        if combo:
            gap_mask[r, list(combo)] = True

    col_only_rows = [r for r in range(box_h) if r not in row_set]
    for c in marked_cols:
        combo = _gen_complete_line_gaps(gap_mask[:, c], col_targets[c], col_only_rows)
        if combo is None:
            return False
        if combo:
            gap_mask[list(combo), c] = True
    return True


def _gen_valid_candidate(inp, out, box_h, box_w, top_pad, left_pad, marked_rows, marked_cols, line_mask):
    """Check observable constraints and the solver oracle for a rendered candidate."""
    patch_in = inp[top_pad : top_pad + box_h, left_pad : left_pad + box_w]

    detected_rows = [r for r in range(box_h) if _gen_has_adjacent_blue(patch_in[r, :])]
    detected_cols = [c for c in range(box_w) if _gen_has_adjacent_blue(patch_in[:, c])]
    if detected_rows != marked_rows or detected_cols != marked_cols:
        return False
    if np.any(patch_in[~line_mask] != _GEN_BLACK):
        return False

    off_rows = [r for r in range(box_h) if r not in set(marked_rows)]
    off_total = len(off_rows) * len(marked_cols)
    off_blue = int(np.count_nonzero(patch_in[np.ix_(off_rows, marked_cols)] == _GEN_BLUE))
    off_blue_rate = off_blue / off_total
    lo, hi = _GEN_VERTICAL_SEGMENT_RATE_RANGE
    if not (lo <= off_blue_rate <= hi):
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, out)


def _gen_attempt() -> np.ndarray | None:
    grid_h, box_h, top_pad, _bottom_pad = _gen_choose_bbox_rows()
    box_w, left_pad, _right_pad = _gen_choose_bbox_cols()
    marked_rows = _gen_marked_positions(box_h)
    marked_cols = _gen_marked_positions(box_w)

    row_lo, row_hi = _gen_gap_bounds(box_w)
    col_lo, col_hi = _gen_gap_bounds(box_h)
    row_targets = {r: random.randint(row_lo, row_hi) for r in marked_rows}
    col_targets = {c: random.randint(col_lo, col_hi) for c in marked_cols}

    gap_mask = np.zeros((box_h, box_w), dtype=bool)
    _gen_seed_intersection_gaps(
        gap_mask, marked_rows, marked_cols, row_targets, col_targets, box_h, box_w
    )
    if not _gen_complete_axis_gaps(
        gap_mask, marked_rows, marked_cols, row_targets, col_targets, box_h, box_w
    ):
        return None

    line_mask = np.zeros((box_h, box_w), dtype=bool)
    line_mask[marked_rows, :] = True
    line_mask[:, marked_cols] = True
    blue_mask = line_mask & ~gap_mask

    out = np.zeros((grid_h, _GEN_GRID_WIDTH), dtype=int)
    patch_out = out[top_pad : top_pad + box_h, left_pad : left_pad + box_w]
    patch_out[line_mask] = _GEN_RED
    patch_out[blue_mask] = _GEN_BLUE

    inp = out.copy()
    inp[inp == _GEN_RED] = _GEN_BLACK
    if not _gen_valid_candidate(
        inp, out, box_h, box_w, top_pad, left_pad, marked_rows, marked_cols, line_mask
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_MAX_ATTEMPTS} attempts")
