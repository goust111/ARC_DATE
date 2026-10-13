from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
TEAL = 8

GEN_ATTEMPTS = 400
WALL_ATTEMPTS = 160


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _repair_wall(grid: np.ndarray) -> np.ndarray:
    repaired = np.array(grid, dtype=int, copy=True)
    h, w = repaired.shape
    center = w // 2
    for r in range(h):
        for d in range(1, center + 1):
            c1 = center - d
            c2 = center + d
            if c1 < 0 or c2 >= w:
                continue
            v1 = int(repaired[r, c1])
            v2 = int(repaired[r, c2])
            if v1 == GREY and v2 == BLACK:
                repaired[r, c2] = GREY
            elif v1 == BLACK and v2 == GREY:
                repaired[r, c1] = GREY
    return repaired


def _reachable_mask(grid: np.ndarray, start: tuple[int, int] = (0, 0)) -> np.ndarray:
    h, w = grid.shape
    reached = np.zeros((h, w), dtype=bool)
    sr, sc = start
    if not (0 <= sr < h and 0 <= sc < w) or int(grid[sr, sc]) == GREY:
        return reached

    q: deque[tuple[int, int]] = deque([(sr, sc)])
    reached[sr, sc] = True
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and not reached[rr, cc] and int(grid[rr, cc]) != GREY:
                reached[rr, cc] = True
                q.append((rr, cc))
    return reached


def _reaches_right_half(wall_mask: np.ndarray) -> bool:
    h, w = wall_mask.shape
    center = w // 2
    if wall_mask[0, 0]:
        return False

    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        if c > center:
            return True
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and not wall_mask[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return False


def _transform(grid: np.ndarray) -> np.ndarray:
    repaired = _repair_wall(grid)
    reached = _reachable_mask(repaired, (0, 0))
    out = repaired.copy()
    movable = repaired != GREY
    out[movable & reached] = GREEN
    out[movable & ~reached] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore vertical wall symmetry, then recolor space by left-corner reachability.

    Input structure:
    - Black background fills most cells, with a grey wall-like structure running roughly from top to bottom near the center column.
    - The wall is almost vertically symmetric about the center column, but a few asymmetric grey omissions or additions break that symmetry.
    - Teal distractor pixels sit on both sides of the wall and are not themselves part of the wall.

    Rule:
    - Use the grid's center column as the symmetry axis.
    - For every symmetric pair of off-axis cells, if one side is grey and the other side is black, fill the black side with grey; if the non-grey side is any non-black color, leave the pair unchanged.
    - On the repaired grid, flood-fill 4-connectedly from the upper-left corner through every non-grey cell.
    - Recolor every reached non-grey cell green, recolor every unreached non-grey cell red, and keep all grey wall cells grey.

    Key insight:
    - Symmetry repair changes the wall topology before flood-filling, so the correct red/green partition depends on the repaired wall rather than the original noisy wall.
    """
    return _transform(np.asarray(grid, dtype=int))


def _build_symmetric_wall(h: int, w: int) -> np.ndarray:
    center = w // 2
    left_min = max(1, center - 4)
    left_max = center - 1
    left_cols = [random.randint(left_min, left_max)]
    for _ in range(1, h):
        step = random.choice((-1, 0, 1))
        left_cols.append(max(left_min, min(left_max, left_cols[-1] + step)))

    wall = np.zeros((h, w), dtype=bool)
    for r, left in enumerate(left_cols):
        right = 2 * center - left
        wall[r, left] = True
        wall[r, right] = True
        if r > 0:
            prev_left = left_cols[r - 1]
            prev_right = 2 * center - prev_left
            a, b = sorted((prev_left, left))
            wall[r, a : b + 1] = True
            a, b = sorted((prev_right, right))
            wall[r, a : b + 1] = True

    if random.random() < 0.8:
        left = left_cols[0]
        right = 2 * center - left
        wall[0, left : right + 1] = True
    if random.random() < 0.8:
        left = left_cols[-1]
        right = 2 * center - left
        wall[h - 1, left : right + 1] = True

    for r in range(h):
        if random.random() < 0.2:
            wall[r, center] = True
    return wall


def _find_gap_sets(wall: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    h, w = wall.shape
    center = w // 2
    left_cells = [(r, c) for r in range(h) for c in range(center) if wall[r, c]]
    random.shuffle(left_cells)

    singles: list[tuple[int, int]] = []
    benign: list[tuple[int, int]] = []
    for cell in left_cells:
        trial = wall.copy()
        trial[cell] = False
        if _reaches_right_half(trial):
            singles.append(cell)
        else:
            benign.append(cell)

    result: dict[int, list[tuple[int, int]]] = {}
    if singles:
        result[1] = [random.choice(singles)]

    random.shuffle(benign)
    pair_candidates = benign[:18]
    random.shuffle(pair_candidates)
    for i, cell1 in enumerate(pair_candidates):
        for cell2 in pair_candidates[i + 1 :]:
            trial = wall.copy()
            trial[cell1] = False
            trial[cell2] = False
            if _reaches_right_half(trial):
                result[2] = [cell1, cell2]
                return result
    return result


def _sample_spaced(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    random.shuffle(candidates)
    picked: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(abs(r - rr) + abs(c - cc) >= 2 for rr, cc in picked):
            picked.append((r, c))
            if len(picked) == count:
                break
    return picked


def _place_distractors(grid: np.ndarray, wall: np.ndarray) -> bool:
    h, w = grid.shape
    center = w // 2
    left_candidates = [
        (r, c)
        for r in range(h)
        for c in range(max(0, center - 5))
        if not wall[r, c] and (r, c) != (0, 0)
    ]
    right_candidates = [
        (r, c)
        for r in range(h)
        for c in range(center + 2, w)
        if not wall[r, c]
    ]
    left_count = random.randint(2, 5)
    right_count = random.randint(2, 5)
    left_pick = _sample_spaced(left_candidates, left_count)
    right_pick = _sample_spaced(right_candidates, right_count)
    if len(left_pick) < left_count or len(right_pick) < right_count:
        return False
    for r, c in left_pick + right_pick:
        grid[r, c] = TEAL
    return True


def _mirror_cell(cell: tuple[int, int], w: int) -> tuple[int, int]:
    r, c = cell
    return r, w - 1 - c


def _make_input_from_wall(wall: np.ndarray, gap_cells: list[tuple[int, int]], noise_total: int) -> np.ndarray | None:
    h, w = wall.shape
    center = w // 2
    grid = np.full((h, w), BLACK, dtype=int)
    grid[wall] = GREY

    left_candidates = [(r, c) for r in range(h) for c in range(center) if wall[r, c]]
    single_critical = set()
    for cell in left_candidates:
        trial = wall.copy()
        trial[cell] = False
        if _reaches_right_half(trial):
            single_critical.add(cell)

    extra_pool = [cell for cell in left_candidates if cell not in gap_cells and cell not in single_critical]
    if len(extra_pool) < noise_total - len(gap_cells):
        return None
    random.shuffle(extra_pool)
    removed_left = list(gap_cells) + extra_pool[: noise_total - len(gap_cells)]

    if random.random() < 0.5:
        removed = removed_left
    else:
        removed = [_mirror_cell(cell, w) for cell in removed_left]

    for r, c in removed:
        grid[r, c] = BLACK

    if not _place_distractors(grid, wall):
        return None

    repaired = _repair_wall(grid)
    repaired_wall = repaired == GREY
    if not np.array_equal(repaired_wall, wall):
        return None
    if _reaches_right_half(wall):
        return None
    if not _reaches_right_half(grid == GREY):
        return None

    before = _reachable_mask(grid, (0, 0))
    after = _reachable_mask(repaired, (0, 0))
    if np.array_equal(before, after):
        return None
    if not np.any(before[:, center + 1 :]):
        return None
    if np.any(after[:, center + 1 :]):
        return None
    if not np.any((repaired != GREY) & after):
        return None
    if not np.any((repaired != GREY) & ~after):
        return None
    return grid


def _try_generate() -> np.ndarray | None:
    h = random.choice((13, 15, 17, 19))
    w = random.choice((13, 15, 17, 19))
    for _ in range(WALL_ATTEMPTS):
        wall = _build_symmetric_wall(h, w)
        if wall[0, 0]:
            continue
        if _reaches_right_half(wall):
            continue

        gap_sets = _find_gap_sets(wall)
        gap_count_choices = [k for k in (1, 2) if k in gap_sets]
        if not gap_count_choices:
            continue
        gap_count = random.choice(gap_count_choices)
        noise_total = random.randint(max(2, gap_count), 4)
        grid = _make_input_from_wall(wall, gap_sets[gap_count], noise_total)
        if grid is not None:
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _try_generate()
        if grid is None:
            continue
        out = main(grid)
        if np.max(grid.shape) <= 30 and np.max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid symmetric-wall ARC input")
