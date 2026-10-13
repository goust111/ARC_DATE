from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np


class Color:
    """ARC palette ids, kept consistent with the shared toolbox."""

    BLACK = 0
    BLUE = 1
    GREY = 5


Coord = Tuple[int, int]


def _in_bounds(shape: Tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _mirror_col(col: int, axis: int) -> int:
    return 2 * axis - col


def _blue_cells(grid: np.ndarray) -> List[Coord]:
    return [tuple(map(int, rc)) for rc in np.argwhere(grid == Color.BLUE)]


def _ray_direction(grid: np.ndarray, r: int, c: int, axis: int) -> Tuple[int, int]:
    mirror_c = _mirror_col(c, axis)
    has_mirror = _in_bounds(grid.shape, r, mirror_c) and int(grid[r, mirror_c]) == Color.BLUE
    if has_mirror:
        if c < axis:
            return (0, -1)
        if c > axis:
            return (0, 1)
        return (-1, 0)
    if c < axis:
        return (0, 1)
    if c > axis:
        return (0, -1)
    return (1, 0)


def _first_grey(grid: np.ndarray, start: Coord, step: Tuple[int, int]) -> Optional[Coord]:
    r, c = start
    dr, dc = step
    r += dr
    c += dc
    while _in_bounds(grid.shape, r, c):
        if int(grid[r, c]) == Color.GREY:
            return (r, c)
        r += dr
        c += dc
    return None


def _expected_targets(grid: np.ndarray) -> Dict[Coord, Coord]:
    axis = grid.shape[1] // 2
    targets: Dict[Coord, Coord] = {}
    for r, c in _blue_cells(grid):
        step = _ray_direction(grid, r, c, axis)
        target = _first_grey(grid, (r, c), step)
        if target is not None:
            targets[(r, c)] = target
    return targets


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Erase the first gray obstacle hit by each blue pixel, using mirror symmetry to choose the ray direction.

    Input structure:
    - Black background fills an odd-width grid.
    - Blue single-pixel markers appear on various rows; some have a blue mirror across the vertical center axis and some do not.
    - Gray obstacle pixels lie along the possible ray paths without overlapping any blue marker.

    Rule:
    - Let the vertical center axis be the middle column.
    - For each blue pixel, inspect its mirrored position on the same row.
    - If that mirrored position is also blue, cast a ray away from the center axis; left-side blues shoot left, right-side blues shoot right, and center-axis blues shoot upward.
    - Otherwise cast a ray toward the center axis; left-side blues shoot right, right-side blues shoot left, and center-axis blues shoot downward.
    - Erase only the first gray obstacle encountered by each ray, leaving all other cells unchanged.

    Key insight:
    - Reflection symmetry is not the output pattern itself; it encodes the ray direction, and that direction determines which obstacle is the removable first hit.
    """
    grid = np.asarray(input_grid, dtype=int)
    out = grid.copy()
    for r, c in set(_expected_targets(grid).values()):
        out[r, c] = Color.BLACK
    return out


def _place_grey(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != Color.BLACK:
        return False
    grid[r, c] = Color.GREY
    return True


def _pick(options: List[int], used: set[int]) -> Optional[int]:
    choices = [x for x in options if x not in used]
    if not choices:
        return None
    return random.choice(choices)


def _pair_row(
    grid: np.ndarray,
    r: int,
    axis: int,
    need_opposite: bool,
    need_farther: bool,
) -> Tuple[bool, bool]:
    w = grid.shape[1]
    left_col = random.randint(2, axis - 2)
    right_col = _mirror_col(left_col, axis)
    grid[r, left_col] = Color.BLUE
    grid[r, right_col] = Color.BLUE

    row_used = {left_col, right_col}

    left_target = random.randint(0, left_col - 1)
    right_target = random.randint(right_col + 1, w - 1)
    _place_grey(grid, r, left_target)
    _place_grey(grid, r, right_target)
    row_used.update({left_target, right_target})

    if need_opposite:
        candidates = [c for c in range(left_col + 1, axis) if c not in row_used]
        picked = _pick(candidates, row_used)
        if picked is not None:
            _place_grey(grid, r, picked)
            row_used.add(picked)
            need_opposite = False
        candidates = [c for c in range(axis + 1, right_col) if c not in row_used]
        picked = _pick(candidates, row_used)
        if picked is not None:
            _place_grey(grid, r, picked)
            row_used.add(picked)
            need_opposite = False

    if need_farther:
        candidates = [c for c in range(0, left_target) if c not in row_used]
        if not candidates:
            candidates = [c for c in range(right_target + 1, w) if c not in row_used]
        picked = _pick(candidates, row_used)
        if picked is not None:
            _place_grey(grid, r, picked)
            row_used.add(picked)
            need_farther = False

    for candidates in (
        list(range(0, left_target)),
        list(range(right_target + 1, w)),
        list(range(left_col + 1, axis)),
        list(range(axis + 1, right_col)),
    ):
        if random.random() < 0.45:
            picked = _pick(candidates, row_used)
            if picked is not None:
                _place_grey(grid, r, picked)
                row_used.add(picked)

    return need_opposite, need_farther


def _singleton_row(
    grid: np.ndarray,
    r: int,
    axis: int,
    side: str,
    need_opposite: bool,
    need_farther: bool,
) -> Tuple[bool, bool]:
    w = grid.shape[1]
    if side == "left":
        col = random.randint(2, axis - 2)
        target = random.randint(col + 1, axis - 1)
        grid[r, col] = Color.BLUE
        row_used = {col}
        _place_grey(grid, r, target)
        row_used.add(target)

        if need_opposite:
            picked = _pick(list(range(0, col)), row_used)
            if picked is not None:
                _place_grey(grid, r, picked)
                row_used.add(picked)
                need_opposite = False

        if need_farther:
            picked = _pick(list(range(target + 1, axis)), row_used)
            if picked is not None:
                _place_grey(grid, r, picked)
                row_used.add(picked)
                need_farther = False

        for candidates in (list(range(0, col)), list(range(target + 1, axis))):
            if random.random() < 0.55:
                picked = _pick(candidates, row_used)
                if picked is not None:
                    _place_grey(grid, r, picked)
                    row_used.add(picked)
        return need_opposite, need_farther

    col = random.randint(axis + 2, w - 3)
    target = random.randint(axis + 1, col - 1)
    grid[r, col] = Color.BLUE
    row_used = {col}
    _place_grey(grid, r, target)
    row_used.add(target)

    if need_opposite:
        picked = _pick(list(range(col + 1, w)), row_used)
        if picked is not None:
            _place_grey(grid, r, picked)
            row_used.add(picked)
            need_opposite = False

    if need_farther:
        picked = _pick(list(range(axis + 1, target)), row_used)
        if picked is not None:
            _place_grey(grid, r, picked)
            row_used.add(picked)
            need_farther = False

    for candidates in (list(range(axis + 1, target)), list(range(col + 1, w))):
        if random.random() < 0.55:
            picked = _pick(candidates, row_used)
            if picked is not None:
                _place_grey(grid, r, picked)
                row_used.add(picked)
    return need_opposite, need_farther


def _center_row(
    grid: np.ndarray,
    r: int,
    axis: int,
    need_opposite: bool,
    need_farther: bool,
) -> Tuple[bool, bool]:
    h = grid.shape[0]
    grid[r, axis] = Color.BLUE
    up_target = random.randint(0, r - 1)
    _place_grey(grid, up_target, axis)

    if need_farther and up_target > 0:
        extra = random.randint(0, up_target - 1)
        _place_grey(grid, extra, axis)
        need_farther = False

    if need_opposite and r < h - 2:
        down = random.randint(r + 1, h - 1)
        _place_grey(grid, down, axis)
        need_opposite = False

    if random.random() < 0.5 and up_target > 0:
        _place_grey(grid, random.randint(0, up_target - 1), axis)
    if random.random() < 0.5 and r < h - 2:
        _place_grey(grid, random.randint(r + 1, h - 1), axis)

    return need_opposite, need_farther


def _analyze_semantics(grid: np.ndarray) -> dict:
    axis = grid.shape[1] // 2
    blues = _blue_cells(grid)
    targets = {}
    symmetric = []
    isolated = []
    same_ray_extra = 0
    opposite_hits = {"symmetric": 0, "isolated": 0}

    for r, c in blues:
        mirror_c = _mirror_col(c, axis)
        has_mirror = _in_bounds(grid.shape, r, mirror_c) and int(grid[r, mirror_c]) == Color.BLUE
        role = "symmetric" if has_mirror else "isolated"
        (symmetric if has_mirror else isolated).append((r, c))

        step = _ray_direction(grid, r, c, axis)
        target = _first_grey(grid, (r, c), step)
        if target is None:
            raise AssertionError("blue pixel has no gray obstacle on its rule ray")
        targets[(r, c)] = target

        rr, cc = target[0] + step[0], target[1] + step[1]
        while _in_bounds(grid.shape, rr, cc):
            if int(grid[rr, cc]) == Color.GREY:
                same_ray_extra += 1
                break
            rr += step[0]
            cc += step[1]

        opposite = _first_grey(grid, (r, c), (-step[0], -step[1]))
        if opposite is not None:
            opposite_hits[role] += 1

    target_values = list(targets.values())
    if len(set(target_values)) != len(target_values):
        raise AssertionError("two blue pixels share the same first obstacle")

    out = main(grid)
    erased = {coord for coord in target_values if int(out[coord]) == Color.BLACK}
    if erased != set(target_values):
        raise AssertionError("output does not erase exactly the intended targets")
    if np.any(out[grid == Color.BLUE] != Color.BLUE):
        raise AssertionError("blue pixels changed in output")

    surviving_grey = int(np.sum((grid == Color.GREY) & (out == Color.GREY)))
    if surviving_grey == 0:
        raise AssertionError("all grey pixels disappeared; first-obstacle logic is not necessary")

    return {
        "blue_count": len(blues),
        "symmetric_count": len(symmetric),
        "isolated_count": len(isolated),
        "same_ray_extra": same_ray_extra,
        "opposite_symmetric": opposite_hits["symmetric"],
        "opposite_isolated": opposite_hits["isolated"],
    }


def _generate_candidate() -> np.ndarray:
    h = random.randint(9, 21)
    w = random.choice(list(range(11, 22, 2)))
    axis = w // 2

    grid = np.full((h, w), Color.BLACK, dtype=int)

    use_center = random.random() < 0.6
    max_specs = min(6, h - 3)
    non_center_max = max_specs - (1 if use_center else 0)
    num_pairs = random.randint(1, min(3, non_center_max - 1))
    num_singletons = random.randint(1, min(4, non_center_max - num_pairs))

    center_row = None
    if use_center:
        center_row = random.randint(2, h - 3)

    available_rows = [r for r in range(1, h - 1) if r != center_row]
    chosen_rows = random.sample(available_rows, num_pairs + num_singletons)
    pair_rows = chosen_rows[:num_pairs]
    singleton_rows = chosen_rows[num_pairs:]

    singleton_sides = [random.choice(["left", "right"]) for _ in singleton_rows]
    if len(singleton_sides) >= 2 and len(set(singleton_sides)) == 1:
        singleton_sides[0] = "left"
        singleton_sides[1] = "right"

    need_pair_opposite = True
    need_single_opposite = True
    need_farther = True

    random.shuffle(pair_rows)
    for r in pair_rows:
        need_pair_opposite, need_farther = _pair_row(grid, r, axis, need_pair_opposite, need_farther)

    order = list(range(len(singleton_rows)))
    random.shuffle(order)
    for idx in order:
        r = singleton_rows[idx]
        side = singleton_sides[idx]
        need_single_opposite, need_farther = _singleton_row(
            grid, r, axis, side, need_single_opposite, need_farther
        )

    if center_row is not None:
        center_need = need_pair_opposite or need_single_opposite
        center_need, need_farther = _center_row(grid, center_row, axis, center_need, need_farther)
        if need_pair_opposite:
            need_pair_opposite = center_need
        elif need_single_opposite:
            need_single_opposite = center_need

    if need_pair_opposite or need_single_opposite or need_farther:
        raise AssertionError("failed to place required semantic distractors")

    return grid


def _validate_instance(grid: np.ndarray) -> None:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        raise AssertionError("grid must be a 2D numpy array")
    h, w = grid.shape
    if max(h, w) > 30 or w % 2 != 1:
        raise AssertionError("grid must be odd-width and at most 30x30")
    stats = _analyze_semantics(grid)
    if stats["blue_count"] < 3:
        raise AssertionError("need multiple blue pixels")
    if stats["symmetric_count"] < 2:
        raise AssertionError("need at least one mirrored pair")
    if stats["isolated_count"] < 1:
        raise AssertionError("need at least one isolated blue")
    if stats["same_ray_extra"] < 1:
        raise AssertionError("need a surviving farther obstacle on a valid ray")
    if stats["opposite_symmetric"] < 1:
        raise AssertionError("need an opposite-side distractor for a symmetric blue")
    if stats["opposite_isolated"] < 1:
        raise AssertionError("need an opposite-side distractor for an isolated blue")


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            grid = _generate_candidate()
            _validate_instance(grid)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input grid")
