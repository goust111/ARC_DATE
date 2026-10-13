from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
CYAN = 8

GEN_ATTEMPTS = 500


def _steps_from_run(run_len: int) -> int:
    return max(0, (int(run_len) - 1) // 2)


def _rounded_mask(height: int, width: int, radius: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    for r in range(int(height)):
        edge_distance = min(r, int(height) - 1 - r, int(radius))
        inset = int(radius) - edge_distance
        mask[r, inset : int(width) - inset] = True
    return mask


def _centered_span(lo: int, hi: int, length: int) -> tuple[int, int]:
    span = int(hi) - int(lo) + 1
    assert 0 < int(length) <= span
    shrink = span - int(length)
    left_trim = shrink // 2
    return int(lo) + left_trim, int(lo) + left_trim + int(length) - 1


def _component4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _marker_rows(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    rows: list[tuple[int, int, int, int]] = []
    for r in range(h):
        cols = np.flatnonzero(grid[r] == RED)
        if cols.size == 0:
            continue
        assert cols.size == 2, "each marker row must contain exactly two red pixels"
        c0, c1 = map(int, cols)
        assert c0 + c1 == w - 1, "red markers must be mirrored across the vertical center"
        rows.append((c1 - c0, r, c0, c1))
    assert rows, "expected mirrored red marker pairs"
    return rows


def _paint_shrinking_rows(
    out: np.ndarray,
    row: int,
    step: int,
    c0: int,
    c1: int,
    *,
    barrier_row: int | None = None,
) -> None:
    source_len = int(c1) - int(c0) + 1
    length = source_len - 2
    distance = 1
    while length > 0:
        rr = int(row) + int(step) * distance
        if rr < 0 or rr >= out.shape[0]:
            distance += 1
            length -= 2
            continue
        if barrier_row is not None and step < 0 and rr <= int(barrier_row):
            break
        lo, hi = _centered_span(c0, c1, length)
        for cc in range(lo, hi + 1):
            if 0 <= cc < out.shape[1] and out[rr, cc] == BLACK:
                out[rr, cc] = YELLOW
        distance += 1
        length -= 2


def _paint_shrinking_cols(out: np.ndarray, col: int, step: int, r0: int, r1: int) -> None:
    source_len = int(r1) - int(r0) + 1
    length = source_len - 2
    distance = 1
    while length > 0:
        cc = int(col) + int(step) * distance
        if cc < 0 or cc >= out.shape[1]:
            distance += 1
            length -= 2
            continue
        lo, hi = _centered_span(r0, r1, length)
        for rr in range(lo, hi + 1):
            if 0 <= rr < out.shape[0] and out[rr, cc] == BLACK:
                out[rr, cc] = YELLOW
        distance += 1
        length -= 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the closest mirrored red markers, then clip the teal object's upward shrinking cap against that new barrier.

    Input structure:
    - The grid is black except for one 4-connected teal rounded object and several red marker pairs.
    - Every red marker row contains exactly two red pixels mirrored across the grid's vertical center.
    - The teal object's top, bottom, left, and right boundary runs are contiguous side spans suitable for shrinking extensions.

    Rule:
    - Preserve all input pixels.
    - Find the red marker row with the smallest horizontal gap and fill the inclusive segment between its two markers, creating a horizontal red barrier.
    - Extend the teal object's top, bottom, left, and right side runs outward in yellow, shrinking each step by two pixels.
    - Yellow pixels are only painted onto black cells; upward yellow pixels are additionally blocked once they reach or cross the barrier row.

    Key insight:
    - The red line is not present in the input; it is produced first from the closest mirrored pair, and that derived barrier then asymmetrically truncates only the object's upward shrinking cap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    object_comps = _component4(g == CYAN)
    assert len(object_comps) == 1, "expected exactly one teal object"
    coords = object_comps[0]
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

    local = g[r0 : r1 + 1, c0 : c1 + 1] == CYAN
    top_cols = np.flatnonzero(local[0])
    bottom_cols = np.flatnonzero(local[-1])
    left_rows = np.flatnonzero(local[:, 0])
    right_rows = np.flatnonzero(local[:, -1])
    assert min(len(top_cols), len(bottom_cols), len(left_rows), len(right_rows)) >= 3

    rows = _marker_rows(g)
    min_gap = min(gap for gap, _r, _c0, _c1 in rows)
    barrier_rows = [(r, a, b) for gap, r, a, b in rows if gap == min_gap]

    out = g.copy()
    for r, a, b in barrier_rows:
        out[r, a : b + 1] = RED

    rows_above = [r for r, _a, _b in barrier_rows if r < r0]
    barrier_row = max(rows_above) if rows_above else None

    _paint_shrinking_rows(
        out,
        r0,
        -1,
        c0 + int(top_cols[0]),
        c0 + int(top_cols[-1]),
        barrier_row=barrier_row,
    )
    _paint_shrinking_rows(out, r1, 1, c0 + int(bottom_cols[0]), c0 + int(bottom_cols[-1]))
    _paint_shrinking_cols(out, c0, -1, r0 + int(left_rows[0]), r0 + int(left_rows[-1]))
    _paint_shrinking_cols(out, c1, 1, r0 + int(right_rows[0]), r0 + int(right_rows[-1]))
    return out


def _gap_to_cols(width: int, gap: int) -> tuple[int, int]:
    left = (int(width) - 1 - int(gap)) // 2
    right = int(width) - 1 - left
    return left, right


def _place_marker_pair(grid: np.ndarray, row: int, gap: int) -> None:
    c0, c1 = _gap_to_cols(grid.shape[1], gap)
    grid[int(row), int(c0)] = RED
    grid[int(row), int(c1)] = RED


def _upward_cap_rows(top_row: int, run_len: int) -> list[int]:
    rows = []
    length = int(run_len) - 2
    distance = 1
    while length > 0:
        rows.append(int(top_row) - distance)
        distance += 1
        length -= 2
    return rows


def _grid_is_valid(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    object_comps = _component4(grid == CYAN)
    if len(object_comps) != 1:
        return False
    coords = object_comps[0]
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

    rows = _marker_rows(grid)
    gaps = [gap for gap, _r, _a, _b in rows]
    min_gap = min(gaps)
    if gaps.count(min_gap) != 1:
        return False
    barrier_gap, barrier_row, barrier_c0, barrier_c1 = min(rows)
    if barrier_row >= r0:
        return False
    if barrier_row > r0 - 2:
        return False

    local = grid[r0 : r1 + 1, c0 : c1 + 1] == CYAN
    top_run = int(np.flatnonzero(local[0]).size)
    side_run = int(np.flatnonzero(local[:, 0]).size)
    up_steps = _steps_from_run(top_run)
    side_steps = _steps_from_run(side_run)
    if min(up_steps, side_steps) < 2:
        return False

    possible_up_rows = _upward_cap_rows(r0, top_run)
    if not possible_up_rows:
        return False
    if not any(rr <= barrier_row for rr in possible_up_rows):
        return False
    if not any(barrier_row < rr < r0 for rr in possible_up_rows):
        return False

    if np.any(out[: barrier_row + 1] == YELLOW):
        return False
    if np.count_nonzero(out[:r0] == YELLOW) == 0:
        return False
    if np.count_nonzero(out[r1 + 1 :] == YELLOW) == 0:
        return False
    if np.count_nonzero(out[:, :c0] == YELLOW) == 0:
        return False
    if np.count_nonzero(out[:, c1 + 1 :] == YELLOW) == 0:
        return False
    if not np.all(out[barrier_row, barrier_c0 : barrier_c1 + 1] == RED):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    radius = random.randint(1, 2)
    height = random.randint(2 * radius + 5, min(9, 2 * radius + 7))
    width = random.randint(2 * radius + 5, min(10, 2 * radius + 7))

    obj_mask = _rounded_mask(height, width, radius)
    top_run = width - 2 * radius
    side_run = height - 2 * radius
    up_steps = _steps_from_run(top_run)
    side_steps = _steps_from_run(side_run)
    if min(up_steps, side_steps) < 2:
        return None

    top_space = random.randint(max(6, up_steps + 2), 10)
    n_above = random.randint(2, 4)
    n_below = random.randint(1, 3)
    bottom_space = side_steps + 2 + n_below + random.randint(1, 3)
    grid_h = top_space + height + bottom_space
    if grid_h > 30:
        return None

    min_w = max(15, width + 2 * side_steps + 4)
    if min_w > 23:
        return None
    grid_w = random.randint(min_w, 23)

    gap_levels = sorted(
        {
            grid_w - 1 - 2 * left
            for left in range(grid_w)
            if left < grid_w - 1 - left and grid_w - 1 - 2 * left >= 3
        }
    )
    if len(gap_levels) < 2:
        return None

    min_gap_index = random.randint(0, min(2, len(gap_levels) - 2))
    min_gap = gap_levels[min_gap_index]
    larger_pool = gap_levels[min_gap_index + 1 :]
    chosen_larger = random.sample(larger_pool, random.randint(1, min(3, len(larger_pool))))

    obj_c0_min = side_steps + 1
    obj_c0_max = grid_w - width - side_steps - 1
    if obj_c0_min > obj_c0_max:
        return None
    center_c0 = (grid_w - width) // 2
    central_choices = [c for c in range(obj_c0_min, obj_c0_max + 1) if abs(c - center_c0) <= 2]
    obj_c0 = random.choice(central_choices or list(range(obj_c0_min, obj_c0_max + 1)))
    obj_r0 = top_space
    obj_r1 = obj_r0 + height - 1

    barrier_gap = random.randint(2, min(up_steps, obj_r0))
    barrier_row = obj_r0 - barrier_gap
    other_above_slots = [r for r in range(obj_r0 - 1) if r != barrier_row]
    if len(other_above_slots) < n_above - 1:
        return None
    above_rows = random.sample(other_above_slots, n_above - 1) + [barrier_row]

    below_start = obj_r1 + side_steps + 2
    below_slots = list(range(below_start, grid_h))
    if len(below_slots) < n_below:
        return None
    below_rows = random.sample(below_slots, n_below)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[obj_r0 : obj_r0 + height, obj_c0 : obj_c0 + width][obj_mask] = CYAN

    _place_marker_pair(grid, barrier_row, min_gap)
    for row in above_rows:
        if row != barrier_row:
            _place_marker_pair(grid, row, random.choice(chosen_larger))
    for row in below_rows:
        _place_marker_pair(grid, row, random.choice(chosen_larger))

    if not _grid_is_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shrinking-cap barrier input")
