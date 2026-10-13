from __future__ import annotations

import random

import numpy as np


BG = 7
WALL = 5
PATH = 6
TARGET = 1

WALKABLE = {BG, PATH, TARGET}
ORTHOGONALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
GEN_MAX_ATTEMPTS = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ORTHOGONALS:
        nr, nc = r + dr, c + dc
        if _in_bounds(shape, nr, nc):
            out.append((nr, nc))
    return out


def _find_unique(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    assert len(coords) == 1, f"expected exactly one pixel of color {color}"
    return tuple(map(int, coords[0]))


def _left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _trace_path(grid: np.ndarray) -> list[tuple[int, int]]:
    start = _find_unique(grid, PATH)
    goal = _find_unique(grid, TARGET)

    assert start[1] == 0, "start must lie on the left edge"
    assert goal[1] == grid.shape[1] - 1, "target must lie on the right edge"

    path = [start]
    cur = start
    direction = (0, 1)

    for _ in range(grid.shape[0] * grid.shape[1]):
        if cur == goal:
            return path

        nr, nc = cur[0] + direction[0], cur[1] + direction[1]
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) in WALKABLE:
            nxt = (nr, nc)
        else:
            candidates = []
            for new_dir in (_left(direction), _right(direction)):
                tr, tc = cur[0] + new_dir[0], cur[1] + new_dir[1]
                if _in_bounds(grid.shape, tr, tc) and int(grid[tr, tc]) in WALKABLE:
                    candidates.append((new_dir, (tr, tc)))
            assert len(candidates) == 1, "collision turn must have a unique open side"
            direction, nxt = candidates[0]
        path.append(nxt)
        cur = nxt

    raise AssertionError("particle failed to reach the target")


def _corner_indices(path: list[tuple[int, int]]) -> list[int]:
    corners: list[int] = []
    for i in range(1, len(path) - 1):
        r0, c0 = path[i - 1]
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        d1 = (r1 - r0, c1 - c0)
        d2 = (r2 - r1, c2 - c1)
        if d1 != d2:
            assert abs(d1[0]) + abs(d1[1]) == 1
            assert abs(d2[0]) + abs(d2[1]) == 1
            assert d1[0] != d2[0] and d1[1] != d2[1], "turns must be 90 degrees"
            corners.append(i)
    return corners


def main(grid: np.ndarray) -> np.ndarray:
    """Trace the unique corridor path from the left-edge particle to the right-edge goal and expand each 90-degree turn into a fixed 3x3 corner stamp.

    Input structure:
    - The background is orange, with gray wall pixels isolating a one-cell-wide corridor.
    - A single magenta particle sits at the corridor's left-edge entrance and a single blue target sits at its right-edge exit.
    - The corridor contains a unique non-branching walkable path made of orange cells between the endpoints.

    Rule:
    - Follow the unique 4-connected walkable path from the magenta start through orange corridor cells until the blue target.
    - Recolor every visited corridor cell except the start and target to magenta.
    - At each path cell where the step direction changes by 90 degrees, restore the center cell to orange, then apply a 3x3 stamp centered there: diagonals become magenta and orthogonal neighbors become gray.
    - Stamp pixels only onto orange background cells; any existing non-background wall, path, start, or target pixel is preserved.

    Key insight:
    - Because the walls isolate a single walkable corridor, the particle trajectory can be recovered as the unique start-to-goal path, and turn cells are exactly the path positions whose incoming and outgoing directions differ.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    path = _trace_path(g)
    corners = _corner_indices(path)

    out = g.copy()
    for r, c in path[1:-1]:
        out[r, c] = PATH

    for idx in corners:
        r, c = path[idx]
        out[r, c] = BG
        for dr, dc in ORTHOGONALS:
            nr, nc = r + dr, c + dc
            if _in_bounds(out.shape, nr, nc) and int(out[nr, nc]) == BG:
                out[nr, nc] = WALL
        for dr, dc in DIAGONALS:
            nr, nc = r + dr, c + dc
            if _in_bounds(out.shape, nr, nc) and int(out[nr, nc]) == BG:
                out[nr, nc] = PATH

    return out


def _line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("segments must be axis-aligned")


def _sample_spaced(values: list[int], count: int, min_gap: int) -> list[int] | None:
    if count == 0:
        return []
    for _ in range(100):
        shuffled = values[:]
        random.shuffle(shuffled)
        chosen: list[int] = []
        for value in shuffled:
            if all(abs(value - old) >= min_gap for old in chosen):
                chosen.append(value)
                if len(chosen) == count:
                    return chosen
    return None


def _build_path(height: int, width: int) -> list[tuple[int, int]] | None:
    excursion_count = random.choice([1, 2])
    row_candidates = list(range(2, height - 2))
    col_candidates = list(range(2, width - 2))

    rows = _sample_spaced(row_candidates, excursion_count + 1, 4)
    turn_cols = _sample_spaced(col_candidates, excursion_count, 4)
    if rows is None or turn_cols is None:
        return None

    turn_cols = sorted(turn_cols)
    waypoints: list[tuple[int, int]] = [(rows[0], 0)]
    for i, col in enumerate(turn_cols):
        waypoints.append((rows[i], col))
        waypoints.append((rows[i + 1], col))
    waypoints.append((rows[-1], width - 1))

    path: list[tuple[int, int]] = []
    for a, b in zip(waypoints, waypoints[1:]):
        segment = _line(a, b)
        if path:
            path.extend(segment[1:])
        else:
            path.extend(segment)

    if len(path) != len(set(path)):
        return None
    if len(_corner_indices(path)) != excursion_count * 2:
        return None
    return path


def _render_input(path: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    path_set = set(path)
    corner_cells = {path[i] for i in _corner_indices(path)}
    reserved_diagonals = {
        (r + dr, c + dc)
        for r, c in corner_cells
        for dr, dc in DIAGONALS
        if _in_bounds(shape, r + dr, c + dc)
    }

    for r, c in path:
        for dr, dc in ORTHOGONALS:
            nr, nc = r + dr, c + dc
            if not _in_bounds(shape, nr, nc):
                continue
            if (nr, nc) in path_set or (nr, nc) in reserved_diagonals:
                continue
            grid[nr, nc] = WALL

    sr, sc = path[0]
    tr, tc = path[-1]
    grid[sr, sc] = PATH
    grid[tr, tc] = TARGET
    return grid


def _valid_input(grid: np.ndarray) -> bool:
    try:
        path = _trace_path(grid)
    except AssertionError:
        return False

    if len(_corner_indices(path)) not in {2, 4}:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        height = random.randint(13, 19)
        width = random.randint(13, 20)
        path = _build_path(height, width)
        if path is None:
            continue
        grid = _render_input(path, (height, width))
        if _valid_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid corridor input")
