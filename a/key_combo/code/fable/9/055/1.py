from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
HEAD_PALETTE = (2, 3, 4, 6, 7, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

GRID_MIN = 12
GRID_MAX = 24
EMITTER_COUNT_RANGE = (2, 4)
TURN_MIN = 2
FINAL_MIN = 2
PLACEMENT_ATTEMPTS = 240
GENERATION_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _turn_ccw(dir_idx: int) -> int:
    return (dir_idx - 1) % 4


def _steps_to_boundary(shape: tuple[int, int], pos: tuple[int, int], dir_idx: int) -> int:
    h, w = shape
    r, c = pos
    dr, dc = DIRS[dir_idx]
    if dr == -1:
        return r
    if dr == 1:
        return h - 1 - r
    if dc == -1:
        return c
    return w - 1 - c


def _segment_cells(pos: tuple[int, int], dir_idx: int, steps: int) -> list[tuple[int, int]]:
    r, c = pos
    dr, dc = DIRS[dir_idx]
    return [(r + dr * k, c + dc * k) for k in range(1, steps + 1)]


def _find_emitters(grid: np.ndarray) -> list[dict[str, object]]:
    emitters: list[dict[str, object]] = []
    tail_owners: dict[tuple[int, int], tuple[int, int]] = {}

    for r, c in np.argwhere((grid != BLACK) & (grid != BLUE) & (grid != GREY)):
        rr = int(r)
        cc = int(c)
        tails = [
            (nr, nc)
            for nr, nc in _neighbors4(rr, cc)
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GREY
        ]
        assert len(tails) == 1, "each emitter head must have exactly one grey tail"
        tail = tails[0]
        if tail in tail_owners:
            raise AssertionError("grey tail cannot belong to multiple heads")
        tail_owners[tail] = (rr, cc)
        dir_vec = (rr - tail[0], cc - tail[1])
        assert dir_vec in DIRS, "tail/head must be orthogonally adjacent"
        emitters.append(
            {
                "head": (rr, cc),
                "tail": tail,
                "dir_idx": DIRS.index(dir_vec),
                "color": int(grid[rr, cc]),
            }
        )

    grey_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)}
    assert grey_cells == set(tail_owners), "all grey cells must be emitter tails"
    emitters.sort(key=lambda item: (item["head"][0], item["head"][1]))
    assert emitters, "expected at least one emitter"
    return emitters


def _trace_ray(
    grid: np.ndarray,
    emitter: dict[str, object],
    *,
    pivot_mode: str,
) -> dict[str, object]:
    head = emitter["head"]
    dir_idx = int(emitter["dir_idx"])
    color = int(emitter["color"])
    r, c = head
    traversed: list[tuple[int, int]] = []
    pivots: list[tuple[int, int]] = []
    seen_local: set[tuple[int, int]] = set()
    turned_once = False

    for _ in range(grid.shape[0] * grid.shape[1]):
        dr, dc = DIRS[dir_idx]
        r += dr
        c += dc
        if not _in_bounds(grid.shape, r, c):
            break
        cell = int(grid[r, c])
        assert cell in (BLACK, BLUE), "rays may traverse only black cells and blue pivots"
        coord = (r, c)
        assert coord not in seen_local, "ray path must not loop"
        seen_local.add(coord)
        traversed.append(coord)
        if cell == BLUE:
            pivots.append(coord)
            if pivot_mode == "all":
                dir_idx = _turn_ccw(dir_idx)
            elif pivot_mode == "first":
                if not turned_once:
                    dir_idx = _turn_ccw(dir_idx)
                    turned_once = True
            elif pivot_mode == "ignore":
                pass
            else:
                raise ValueError(f"unknown pivot mode: {pivot_mode}")

    assert traversed, "each emitter must emit at least one step"
    return {"cells": traversed, "pivots": pivots, "color": color}


def _paint_solution(grid: np.ndarray, *, pivot_mode: str) -> tuple[np.ndarray, list[dict[str, object]]]:
    out = np.asarray(grid, dtype=int).copy()
    emitters = _find_emitters(out)
    used: set[tuple[int, int]] = set()
    traced: list[dict[str, object]] = []

    for emitter in emitters:
        data = _trace_ray(out, emitter, pivot_mode=pivot_mode)
        traced.append(data)
        for coord in data["cells"]:
            if coord in used:
                raise AssertionError("different rays must not overlap")
            used.add(coord)
            r, c = coord
            if int(out[r, c]) == BLACK:
                out[r, c] = int(data["color"])

    if pivot_mode == "all":
        pivot_cells = {tuple(map(int, rc)) for rc in np.argwhere(out == BLUE)}
        visited_pivots = {pivot for data in traced for pivot in data["pivots"]}
        assert pivot_cells == visited_pivots, "every blue pivot must belong to exactly one ray"
    return out, traced


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each emitter ray, turning it 90 degrees counterclockwise at every blue pivot.

    Input structure:
    - The background is black.
    - Each emitter is a colored head pixel with one orthogonally adjacent grey tail pixel.
    - Single blue pixels are pivots, and all other traversable cells are black.

    Rule:
    - For each emitter, start at the head and move one cell at a time in the
      direction from tail to head.
    - Whenever the ray enters a blue pivot cell, keep that pivot unchanged but
      rotate the ray direction 90 degrees counterclockwise for the subsequent steps.
    - Continue until the ray leaves the grid, coloring only the traversed black
      cells with the emitter head color; emitters and pivots remain unchanged.

    Key insight:
    - The blue pixels are not obstacles: they are chained turning points, so the
      final colored trace is a polyline determined by the ordered pivot sequence.
    """
    out, _ = _paint_solution(np.asarray(grid, dtype=int), pivot_mode="all")
    return out


def _search_path(
    shape: tuple[int, int],
    occupied: np.ndarray,
    pos: tuple[int, int],
    dir_idx: int,
    turns_left: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    if turns_left == 0:
        dist = _steps_to_boundary(shape, pos, dir_idx)
        if dist < FINAL_MIN:
            return None
        cells = _segment_cells(pos, dir_idx, dist)
        if any(bool(occupied[r, c]) for r, c in cells):
            return None
        for r, c in cells:
            occupied[r, c] = True
        return cells, []

    dist = _steps_to_boundary(shape, pos, dir_idx)
    lengths = list(range(TURN_MIN, dist))
    random.shuffle(lengths)

    for steps in lengths:
        cells = _segment_cells(pos, dir_idx, steps)
        if any(bool(occupied[r, c]) for r, c in cells):
            continue
        for r, c in cells:
            occupied[r, c] = True
        pivot = cells[-1]
        result = _search_path(shape, occupied, pivot, _turn_ccw(dir_idx), turns_left - 1)
        if result is not None:
            tail_cells, tail_pivots = result
            return cells + tail_cells, [pivot] + tail_pivots
        for r, c in cells:
            occupied[r, c] = False
    return None


def _try_place_emitter(
    shape: tuple[int, int],
    occupied: np.ndarray,
    head_cells: set[tuple[int, int]],
    tail_cells: set[tuple[int, int]],
    color: int,
    turns: int,
) -> dict[str, object] | None:
    h, w = shape
    for _ in range(PLACEMENT_ATTEMPTS):
        dir_idx = random.randrange(4)
        req = TURN_MIN if turns > 0 else FINAL_MIN
        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if _steps_to_boundary(shape, (r, c), dir_idx) >= req
        ]
        if not candidates:
            return None
        head = random.choice(candidates)
        dr, dc = DIRS[dir_idx]
        tail = (head[0] - dr, head[1] - dc)
        if not _in_bounds(shape, tail[0], tail[1]):
            continue
        if bool(occupied[head]) or bool(occupied[tail]):
            continue
        if any(nei in tail_cells for nei in _neighbors4(*head)):
            continue
        if any(nei in head_cells for nei in _neighbors4(*tail)):
            continue

        occupied[head] = True
        occupied[tail] = True
        result = _search_path(shape, occupied, head, dir_idx, turns)
        if result is not None:
            cells, pivots = result
            head_cells.add(head)
            tail_cells.add(tail)
            return {
                "head": head,
                "tail": tail,
                "dir_idx": dir_idx,
                "color": color,
                "path": cells,
                "pivots": pivots,
            }
        occupied[head] = False
        occupied[tail] = False
    return None


def _turn_plan(count: int) -> list[int]:
    plan = [random.randint(2, 3), random.randint(0, 1)]
    while len(plan) < count:
        plan.append(random.randint(0, 3))
    random.shuffle(plan)
    return plan


def _build_input(shape: tuple[int, int], emitters: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for emitter in emitters:
        hr, hc = emitter["head"]
        tr, tc = emitter["tail"]
        grid[tr, tc] = GREY
        grid[hr, hc] = int(emitter["color"])
        for pr, pc in emitter["pivots"]:
            grid[pr, pc] = BLUE
    return grid


def _variant_output(grid: np.ndarray, pivot_mode: str) -> np.ndarray:
    out, _ = _paint_solution(grid, pivot_mode=pivot_mode)
    return out


def _validate_case(grid: np.ndarray) -> bool:
    try:
        emitters = _find_emitters(grid)
        true_out, traced = _paint_solution(grid, pivot_mode="all")
    except Exception:
        return False

    pivot_counts = [len(data["pivots"]) for data in traced]
    if max(pivot_counts) < 2 or min(pivot_counts) > 1:
        return False
    if len({int(item["color"]) for item in emitters}) != len(emitters):
        return False
    if len({int(item["dir_idx"]) for item in emitters}) < 2:
        return False

    try:
        ignore_out = _variant_output(grid, "ignore")
        if np.array_equal(true_out, ignore_out):
            return False
    except Exception:
        pass

    try:
        first_out = _variant_output(grid, "first")
        if np.array_equal(true_out, first_out):
            return False
    except Exception:
        pass

    for emitter, data in zip(emitters, traced):
        head = emitter["head"]
        cells = data["cells"]
        if len(cells) < FINAL_MIN:
            return False
        if cells[-1][0] not in (0, grid.shape[0] - 1) and cells[-1][1] not in (0, grid.shape[1] - 1):
            return False
        if any(int(grid[r, c]) not in (BLACK, BLUE) for r, c in cells):
            return False
        if int(true_out[head]) != int(emitter["color"]):
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    count = random.randint(*EMITTER_COUNT_RANGE)
    min_side = max(GRID_MIN, 10 + 2 * count)
    shape = (
        random.randint(min_side, GRID_MAX),
        random.randint(min_side, GRID_MAX),
    )
    colors = random.sample(HEAD_PALETTE, count)
    turns = _turn_plan(count)
    order = list(range(count))
    order.sort(key=lambda idx: turns[idx], reverse=True)

    occupied = np.zeros(shape, dtype=bool)
    head_cells: set[tuple[int, int]] = set()
    tail_cells: set[tuple[int, int]] = set()
    emitters: list[dict[str, object] | None] = [None] * count

    for idx in order:
        emitter = _try_place_emitter(shape, occupied, head_cells, tail_cells, colors[idx], turns[idx])
        if emitter is None:
            return None
        emitters[idx] = emitter

    grid = _build_input(shape, [item for item in emitters if item is not None])
    if not _validate_case(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pivot-refraction input")
