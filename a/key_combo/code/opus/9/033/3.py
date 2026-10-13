from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

LENGTH_TO_COLOR = {2: BLUE, 3: YELLOW, 4: PINK, 5: GREEN}

SIDE_TEMPLATES = [
    {"lengths": (5, 3, 2, 2, 4), "bounces": ("flip_r", "flip_r", "flip_r", "flip_c"), "allow_revisit": False},
    {"lengths": (5, 3, 2, 2, 4), "bounces": ("flip_r", "flip_r", "flip_c", "flip_c"), "allow_revisit": False},
    {"lengths": (5, 3, 2, 2, 4), "bounces": ("flip_r", "flip_c", "flip_r", "flip_r"), "allow_revisit": False},
    {"lengths": (5, 3, 2, 2, 4), "bounces": ("flip_r", "flip_c", "flip_c", "flip_r"), "allow_revisit": False},
    {"lengths": (5, 3, 2, 2, 4), "bounces": ("flip_r", "flip_c", "flip_c", "flip_c"), "allow_revisit": False},
    {"lengths": (5, 3, 2, 2, 4), "bounces": ("flip_c", "flip_r", "flip_r", "flip_r"), "allow_revisit": False},
]

FRONT_TEMPLATES = [
    {"lengths": (3, 4, 2, 5), "bounces": ("flip_r", "front", "flip_r"), "allow_revisit": True},
    {"lengths": (3, 4, 2, 5), "bounces": ("flip_r", "front", "flip_c"), "allow_revisit": True},
    {"lengths": (3, 4, 2, 5), "bounces": ("flip_c", "front", "flip_r"), "allow_revisit": True},
    {"lengths": (3, 4, 2, 5), "bounces": ("flip_c", "front", "flip_c"), "allow_revisit": True},
    {"lengths": (4, 5, 3, 2), "bounces": ("flip_r", "flip_r", "front"), "allow_revisit": True},
    {"lengths": (4, 5, 3, 2), "bounces": ("flip_r", "flip_c", "front"), "allow_revisit": True},
]

def _in_bounds(shape: tuple[int, int], pos: tuple[int, int]) -> bool:
    h, w = shape
    r, c = pos
    return 0 <= r < h and 0 <= c < w


def _chebyshev_adj(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(int(a[0] - b[0])), abs(int(a[1] - b[1]))) <= 1


def _arrow_cells_from_start(start: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    return {(r, c - dc), (r - dr, c), (r - dr, c - dc)}


def _find_arrow(grid: np.ndarray) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)]
    assert len(coords) == 3, "expected exactly one 3-pixel grey arrow"
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    assert len(rows) == 2 and len(cols) == 2, "arrow must occupy a 2x2 box"
    corners = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}
    missing = list(corners - set(coords))
    assert len(missing) == 1, "arrow must miss exactly one corner"
    start = missing[0]
    direction = (1 if start[0] == rows[1] else -1, 1 if start[1] == cols[1] else -1)
    return set(coords), (int(start[0]), int(start[1])), (int(direction[0]), int(direction[1]))


def _trace_ray(
    grid: np.ndarray,
) -> tuple[list[list[tuple[int, int]]], list[str]]:
    arrow_cells, start, direction = _find_arrow(grid)
    walls = {tuple(map(int, rc)) for rc in np.argwhere(grid == RED)}
    current = start
    dr, dc = direction
    segments: list[list[tuple[int, int]]] = [[start]]
    bounce_kinds: list[str] = []
    seen_states: set[tuple[int, int, int, int, int]] = set()
    step_limit = 8 * grid.shape[0] * grid.shape[1]

    for _ in range(step_limit):
        state = (int(current[0]), int(current[1]), int(dr), int(dc), int(len(segments[-1])))
        if state in seen_states:
            raise AssertionError("ray entered a loop")
        seen_states.add(state)

        nr, nc = current[0] + dr, current[1] + dc
        if not _in_bounds(grid.shape, (nr, nc)):
            assert current[0] in (0, grid.shape[0] - 1) or current[1] in (0, grid.shape[1] - 1)
            break

        if (nr, nc) in walls:
            bounce_kinds.append("front")
            dr, dc = -dr, -dc
            segments.append([])
            continue

        assert (nr, nc) not in arrow_cells, "ray should not hit the arrow body"
        assert int(grid[nr, nc]) == BG, "ray path must stay on background cells"
        current = (int(nr), int(nc))
        segments[-1].append(current)

        if current[0] in (0, grid.shape[0] - 1) or current[1] in (0, grid.shape[1] - 1):
            break

        flip_r = (current[0] + dr, current[1]) in walls
        flip_c = (current[0], current[1] + dc) in walls
        if flip_r or flip_c:
            if flip_r and flip_c:
                bounce_kinds.append("side_both")
            elif flip_r:
                bounce_kinds.append("flip_r")
            else:
                bounce_kinds.append("flip_c")
            if flip_r:
                dr = -dr
            if flip_c:
                dc = -dc
            segments.append([])
    else:
        raise AssertionError("ray did not terminate")

    assert segments and all(seg for seg in segments), "all segments must contain at least one pixel"
    return segments, bounce_kinds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire the diagonal ray from the grey arrow, split it at each bounce, and recolor each segment by its length.

    Input structure:
    - A black background contains exactly one grey 3-pixel L-arrow in a 2x2 box with one missing corner.
    - The missing corner is the ray start cell and also determines the diagonal launch direction.
    - Four to seven isolated red wall pixels are scattered away from the arrow body.

    Rule:
    - Erase the grey arrow to black, then trace a diagonal ray from the missing-corner cell.
    - If the next diagonal step is a red wall, bounce in place by reversing both axes; if a side wall is one cell
      ahead on exactly one axis after a move, reverse that axis only. Wall pixels stay red.
    - Split the traversed ray into bounce-to-bounce segments, counting the first segment from the start cell and
      assigning each bounce pixel to the segment before it.
    - Recolor each segment by its pixel count using the fixed code 2->1, 3->4, 4->6, 5->3, and stop when the ray
      reaches the boundary.

    Key insight:
    - The arrow gives the initial diagonal, the red walls determine the bounce geometry, and the resulting segment
      lengths are the only clue for the final segment colors.
    """
    g = np.asarray(grid, dtype=int)
    segments, _bounce_kinds = _trace_ray(g)
    arrow_cells, _start, _direction = _find_arrow(g)

    out = g.copy()
    for r, c in arrow_cells:
        out[r, c] = BG

    for seg in segments:
        color = LENGTH_TO_COLOR[len(seg)]
        for r, c in seg:
            out[r, c] = color

    return out


def _build_relative_plan(lengths: tuple[int, ...], bounces: tuple[str, ...], allow_revisit: bool) -> dict:
    current = (0, 0)
    dr, dc = 1, 1
    arrow_cells = _arrow_cells_from_start(current, (dr, dc))
    path_sequence = [current]
    path_seen = {current}
    segments = [[current]]
    walls: list[tuple[int, int]] = []

    for i, bounce in enumerate(bounces):
        steps = int(lengths[i] - (1 if i == 0 else 0))
        for _ in range(steps):
            nr, nc = current[0] + dr, current[1] + dc
            assert (nr, nc) not in arrow_cells
            assert (nr, nc) not in walls
            if not allow_revisit:
                assert (nr, nc) not in path_seen
            current = (nr, nc)
            path_sequence.append(current)
            path_seen.add(current)
            segments[-1].append(current)
        if bounce == "front":
            wall = (current[0] + dr, current[1] + dc)
            dr, dc = -dr, -dc
        elif bounce == "flip_r":
            wall = (current[0] + dr, current[1])
            dr = -dr
        else:
            wall = (current[0], current[1] + dc)
            dc = -dc
        assert wall not in arrow_cells
        assert wall not in walls
        walls.append(wall)
        segments.append([])

    for _ in range(int(lengths[-1])):
        nr, nc = current[0] + dr, current[1] + dc
        assert (nr, nc) not in arrow_cells
        assert (nr, nc) not in walls
        if not allow_revisit:
            assert (nr, nc) not in path_seen
        current = (nr, nc)
        path_sequence.append(current)
        path_seen.add(current)
        segments[-1].append(current)

    return {
        "segments": segments,
        "path_sequence": path_sequence,
        "walls": walls,
        "arrow_cells": arrow_cells,
        "end": current,
    }


def _transform_point(point: tuple[int, int], swap: bool, row_sign: int, col_sign: int) -> tuple[int, int]:
    r, c = point
    if swap:
        r, c = c, r
    return int(row_sign * r), int(col_sign * c)


def _transform_plan(plan: dict) -> dict:
    swap = bool(random.getrandbits(1))
    row_sign = random.choice((-1, 1))
    col_sign = random.choice((-1, 1))

    def tf(point: tuple[int, int]) -> tuple[int, int]:
        return _transform_point(point, swap, row_sign, col_sign)

    return {
        "segments": [[tf(p) for p in seg] for seg in plan["segments"]],
        "path_sequence": [tf(p) for p in plan["path_sequence"]],
        "walls": [tf(p) for p in plan["walls"]],
        "arrow_cells": {tf(p) for p in plan["arrow_cells"]},
        "end": tf(plan["end"]),
    }


def _all_plan_points(plan: dict) -> list[tuple[int, int]]:
    points = list(plan["path_sequence"])
    points.extend(plan["walls"])
    points.extend(plan["arrow_cells"])
    return points


def _span(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rows = [p[0] for p in pts]
    cols = [p[1] for p in pts]
    return min(rows), max(rows), min(cols), max(cols)


def _place_plan(plan: dict) -> tuple[np.ndarray, np.ndarray]:
    points = _all_plan_points(plan)
    min_r, max_r, min_c, max_c = _span(points)
    span_h = max_r - min_r + 1
    span_w = max_c - min_c + 1
    end_r, end_c = plan["end"]

    sides = []
    if end_r == min_r:
        sides.append("top")
    if end_r == max_r:
        sides.append("bottom")
    if end_c == min_c:
        sides.append("left")
    if end_c == max_c:
        sides.append("right")
    if not sides:
        raise ValueError("endpoint cannot be placed on a boundary without clipping another object")
    side = random.choice(sides)

    min_h = max(14, span_h)
    min_w = max(14, span_w)
    max_h = max(min_h, min(18, span_h + 4))
    max_w = max(min_w, min(18, span_w + 4))
    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)

    if side == "top":
        off_r = -end_r
    elif side == "bottom":
        off_r = h - 1 - end_r
    else:
        off_r = random.randint(-min_r, h - 1 - max_r)

    if side == "left":
        off_c = -end_c
    elif side == "right":
        off_c = w - 1 - end_c
    else:
        off_c = random.randint(-min_c, w - 1 - max_c)

    def shift(point: tuple[int, int]) -> tuple[int, int]:
        return int(point[0] + off_r), int(point[1] + off_c)

    shifted = {
        "segments": [[shift(p) for p in seg] for seg in plan["segments"]],
        "path_sequence": [shift(p) for p in plan["path_sequence"]],
        "walls": [shift(p) for p in plan["walls"]],
        "arrow_cells": {shift(p) for p in plan["arrow_cells"]},
        "end": shift(plan["end"]),
    }

    grid = np.zeros((h, w), dtype=int)
    for r, c in shifted["arrow_cells"]:
        grid[r, c] = GREY
    for r, c in shifted["walls"]:
        grid[r, c] = RED

    expected = np.zeros((h, w), dtype=int)
    for r, c in shifted["walls"]:
        expected[r, c] = RED
    for seg in shifted["segments"]:
        color = LENGTH_TO_COLOR[len(seg)]
        for r, c in seg:
            expected[r, c] = color

    return grid, expected


def _count_overlaps(plan: dict) -> int:
    seq = plan["path_sequence"]
    return len(seq) - len(set(seq))


def _validate_structure(grid: np.ndarray) -> tuple[list[list[tuple[int, int]]], list[str]]:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    arrow_cells, _start, _direction = _find_arrow(grid)
    walls = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    assert 4 <= len(walls) <= 7
    assert not np.any((grid != BG) & (grid != RED) & (grid != GREY))
    for i, wall in enumerate(walls):
        assert wall not in arrow_cells
        assert not any(_chebyshev_adj(wall, arrow) for arrow in arrow_cells)
        for other in walls[i + 1 :]:
            assert not _chebyshev_adj(wall, other)
    segments, bounce_kinds = _trace_ray(grid)
    assert len(bounce_kinds) >= 3
    seg_lengths = [len(seg) for seg in segments]
    assert all(length in LENGTH_TO_COLOR for length in seg_lengths)
    assert {2, 3, 4, 5}.issubset(set(seg_lengths))
    return segments, bounce_kinds


def _maybe_add_decoy_walls(grid: np.ndarray, expected: np.ndarray) -> np.ndarray:
    target_extra = random.randint(0, 3)
    result = grid.copy()
    if target_extra == 0:
        return result

    arrow_cells, _start, _direction = _find_arrow(result)
    segments, _bounce_kinds = _trace_ray(result)
    path_cells = {cell for seg in segments for cell in seg}

    candidates = [(r, c) for r in range(result.shape[0]) for c in range(result.shape[1])]
    random.shuffle(candidates)
    added = 0
    for cell in candidates:
        if added >= target_extra or int(np.count_nonzero(result == RED)) >= 7:
            break
        if result[cell] != BG:
            continue
        if any(_chebyshev_adj(cell, pt) for pt in arrow_cells):
            continue
        if any(_chebyshev_adj(cell, tuple(map(int, pt))) for pt in np.argwhere(result == RED)):
            continue
        if any(_chebyshev_adj(cell, pt) for pt in path_cells):
            continue
        trial = result.copy()
        trial[cell] = RED
        try:
            if np.array_equal(main(trial), expected):
                result = trial
                added += 1
        except Exception:
            continue
    return result


def _build_instance(template: dict, require_front: bool) -> np.ndarray | None:
    canonical = _build_relative_plan(template["lengths"], template["bounces"], template["allow_revisit"])
    if require_front and _count_overlaps(canonical) > 3:
        return None

    for _ in range(80):
        transformed = _transform_plan(canonical)
        try:
            grid, expected = _place_plan(transformed)
        except ValueError:
            continue
        grid = _maybe_add_decoy_walls(grid, expected)
        try:
            _segments, bounce_kinds = _validate_structure(grid)
        except AssertionError:
            continue
        if require_front and "front" not in bounce_kinds:
            continue
        if not require_front and "front" in bounce_kinds:
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid
    return None


def _pick_family() -> tuple[list[dict], bool]:
    return SIDE_TEMPLATES, False


def generate_input() -> np.ndarray:
    templates, require_front = _pick_family()
    order = templates[:]
    random.shuffle(order)
    for _ in range(300):
        template = random.choice(order)
        grid = _build_instance(template, require_front=require_front)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid ray-bounce instance")
