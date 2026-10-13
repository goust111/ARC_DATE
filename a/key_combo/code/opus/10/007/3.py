from __future__ import annotations

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
TEAL = 8

MIN_SIZE = 12
MAX_SIZE_EXCLUSIVE = 17
MIN_TARGETS = 4
MAX_TARGETS = 6
TIE_MODE_PROB = 0.4
MAX_GENERATE_ATTEMPTS = 800
MAX_LAYOUT_ATTEMPTS = 300


Coord = tuple[int, int]


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _extract_points(grid: np.ndarray) -> tuple[Coord, list[Coord]]:
    red = np.argwhere(grid == RED)
    teal = np.argwhere(grid == TEAL)
    assert red.shape[0] == 1, "expected exactly one red start pixel"
    assert teal.shape[0] >= 1, "expected at least one teal endpoint"
    start = (int(red[0, 0]), int(red[0, 1]))
    targets = [(int(r), int(c)) for r, c in teal]
    return start, targets


def _choose_next(current: Coord, remaining: list[Coord]) -> Coord:
    return min(remaining, key=lambda p: (_manhattan(current, p), p[0], p[1]))


def _l_path_cells(src: Coord, dst: Coord) -> list[Coord]:
    r0, c0 = src
    r1, c1 = dst
    cells: list[Coord] = []

    if c0 != c1:
        step_c = 1 if c1 > c0 else -1
        for c in range(c0 + step_c, c1 + step_c, step_c):
            if (r0, c) == (r1, c1):
                break
            cells.append((r0, c))

    if r0 != r1:
        step_r = 1 if r1 > r0 else -1
        for r in range(r0 + step_r, r1, step_r):
            cells.append((r, c1))

    return cells


def _trace_steps(start: Coord, targets: list[Coord]) -> tuple[list[tuple[Coord, Coord]], int, int]:
    remaining = list(targets)
    current = start
    steps: list[tuple[Coord, Coord]] = []
    tie_count = 0
    close_choice_count = 0

    while remaining:
        ordered = sorted(remaining, key=lambda p: (_manhattan(current, p), p[0], p[1]))
        best = ordered[0]
        best_dist = _manhattan(current, best)
        if len(ordered) > 1:
            second_dist = _manhattan(current, ordered[1])
            if second_dist == best_dist:
                tie_count += 1
            if second_dist <= best_dist + 2:
                close_choice_count += 1
        steps.append((current, best))
        remaining.remove(best)
        current = best

    return steps, tie_count, close_choice_count


def _analyze_layout(
    start: Coord,
    targets: list[Coord],
    shape: tuple[int, int],
    require_tie: bool | None = None,
) -> dict[str, object]:
    points = [start, *targets]
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]
    h, w = map(int, shape)

    if len(targets) < MIN_TARGETS or len(targets) > MAX_TARGETS:
        raise AssertionError("target count must be 4-6")
    if len(points) != len(set(points)):
        raise AssertionError("points must be distinct")
    if len(set(rows)) != len(points) or len(set(cols)) != len(points):
        raise AssertionError("no two endpoints may share a row or column")
    if not all(0 <= r < h and 0 <= c < w for r, c in points):
        raise AssertionError("point out of bounds")
    if max(rows) - min(rows) < 5 or max(cols) - min(cols) < 5:
        raise AssertionError("layout is too compact")

    steps, tie_count, close_choice_count = _trace_steps(start, targets)
    if require_tie is True and tie_count < 1:
        raise AssertionError("expected at least one tie step")
    if require_tie is False and tie_count != 0:
        raise AssertionError("unexpected tie step")
    if close_choice_count < 1:
        raise AssertionError("layout lacks meaningful nearest-neighbor competition")

    point_set = set(points)
    unique_path_cells: set[Coord] = set()
    total_path_cells = 0
    overlap_count = 0
    h_dirs: list[int] = []
    v_dirs: list[int] = []

    for src, dst in steps:
        cells = _l_path_cells(src, dst)
        if not cells:
            raise AssertionError("every segment must have a non-empty L path")
        if any(cell in point_set - {src, dst} for cell in cells):
            raise AssertionError("path may not pass through another endpoint")
        if not all(0 <= r < h and 0 <= c < w for r, c in cells):
            raise AssertionError("path out of bounds")

        total_path_cells += len(cells)
        overlap_count += sum(cell in unique_path_cells for cell in cells)
        unique_path_cells.update(cells)
        h_dirs.append(_sign(dst[1] - src[1]))
        v_dirs.append(_sign(dst[0] - src[0]))

    if total_path_cells < 12 or len(unique_path_cells) < 10:
        raise AssertionError("path network is too small")
    if overlap_count * 2 > total_path_cells:
        raise AssertionError("path overlap is too heavy for a clear sample")
    if len(set(h_dirs)) + len(set(v_dirs)) < 3:
        raise AssertionError("path directions are too uniform")

    return {
        "steps": steps,
        "tie_count": tie_count,
        "close_choice_count": close_choice_count,
        "path_cells": unique_path_cells,
    }


def _render_input(start: Coord, targets: list[Coord], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[start] = RED
    for point in targets:
        grid[point] = TEAL
    return grid


def _sample_general_layout(
    rng: np.random.Generator, h: int, w: int, n_targets: int
) -> tuple[Coord, list[Coord]]:
    rows = rng.choice(np.arange(1, h - 1), size=n_targets + 1, replace=False)
    cols = rng.choice(np.arange(1, w - 1), size=n_targets + 1, replace=False)
    rng.shuffle(rows)
    rng.shuffle(cols)
    points = [(int(r), int(c)) for r, c in zip(rows, cols)]
    return points[0], points[1:]


def _sample_tie_layout(
    rng: np.random.Generator, h: int, w: int, n_targets: int
) -> tuple[Coord, list[Coord]] | None:
    for _ in range(MAX_LAYOUT_ATTEMPTS):
        start = (int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2)))
        distance = int(rng.integers(4, 8))

        candidates: list[Coord] = []
        for dr in range(-distance + 1, distance):
            dc_abs = distance - abs(dr)
            if dr == 0 or dc_abs == 0:
                continue
            for dc in (-dc_abs, dc_abs):
                point = (start[0] + dr, start[1] + dc)
                if 1 <= point[0] < h - 1 and 1 <= point[1] < w - 1:
                    if point not in candidates:
                        candidates.append(point)

        pairs = [
            (a, b)
            for i, a in enumerate(candidates)
            for b in candidates[i + 1 :]
            if a[0] != b[0] and a[1] != b[1]
        ]
        if not pairs:
            continue

        first, second = pairs[int(rng.integers(0, len(pairs)))]
        used_rows = {start[0], first[0], second[0]}
        used_cols = {start[1], first[1], second[1]}
        available_rows = [r for r in range(1, h - 1) if r not in used_rows]
        available_cols = [c for c in range(1, w - 1) if c not in used_cols]
        need = n_targets - 2
        if len(available_rows) < need or len(available_cols) < need:
            continue

        rng.shuffle(available_rows)
        rng.shuffle(available_cols)
        rest = [(int(available_rows[i]), int(available_cols[i])) for i in range(need)]
        return start, [first, second, *rest]

    return None


def _validate_generated_instance(grid: np.ndarray, require_tie: bool) -> None:
    start, targets = _extract_points(grid)
    analysis = _analyze_layout(start, targets, grid.shape, require_tie=require_tie)
    out = main(grid)

    assert grid.ndim == 2
    assert out.shape == grid.shape
    assert set(map(int, np.unique(grid))).issubset({BLACK, RED, TEAL})
    assert set(map(int, np.unique(out))).issubset({BLACK, YELLOW, TEAL})
    assert np.count_nonzero(out == YELLOW) == len(targets) + 1
    assert np.count_nonzero(out == TEAL) == len(analysis["path_cells"])
    assert np.count_nonzero(out == RED) == 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the red start to all teal endpoints by a greedy chain of L paths, then swap node and path colors.

    Input structure:
    - The grid has a black background with exactly one red start pixel and several teal endpoint pixels.
    - Endpoints are single cells, and in valid puzzle instances no two endpoints share a row or a column.
    - The teal endpoints are sparse enough that each greedy L path can be drawn without crossing another endpoint.

    Rule:
    - Start at the red pixel and repeatedly choose the unvisited teal pixel with minimum Manhattan distance; break ties by smaller row, then smaller column.
    - For each chosen target, draw a right-angle path that goes horizontally first to the target column and then vertically to the target row, excluding the segment's endpoints.
    - After all teal pixels are visited, recolor every path cell from yellow to teal, and recolor the original red start and all original teal endpoints to yellow.

    Key insight:
    - The visible teal network in the output is not a static graph: it comes from a greedy nearest-neighbor traversal whose ordered L-path chain must be reconstructed before undoing the final color-role swap.
    """
    g = np.asarray(grid, dtype=int)
    start, targets = _extract_points(g)
    steps, _tie_count, _close_choice_count = _trace_steps(start, targets)

    out = np.zeros_like(g)
    for src, dst in steps:
        for r, c in _l_path_cells(src, dst):
            out[r, c] = TEAL

    out[start] = YELLOW
    for point in targets:
        out[point] = YELLOW
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(MAX_GENERATE_ATTEMPTS):
        h = int(rng.integers(MIN_SIZE, MAX_SIZE_EXCLUSIVE))
        w = int(rng.integers(MIN_SIZE, MAX_SIZE_EXCLUSIVE))
        n_targets = int(rng.integers(MIN_TARGETS, MAX_TARGETS + 1))
        require_tie = bool(rng.random() < TIE_MODE_PROB)

        for _ in range(MAX_LAYOUT_ATTEMPTS):
            if require_tie:
                layout = _sample_tie_layout(rng, h, w, n_targets)
                if layout is None:
                    continue
                start, targets = layout
            else:
                start, targets = _sample_general_layout(rng, h, w, n_targets)

            try:
                _analyze_layout(start, targets, (h, w), require_tie=require_tie)
            except AssertionError:
                continue

            grid = _render_input(start, targets, (h, w))
            _validate_generated_instance(grid, require_tie=require_tie)
            return grid

    raise RuntimeError("failed to generate a valid greedy L-path network input")
