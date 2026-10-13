from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
CYAN = 8

TURN_LIMIT_FACTOR = 8
GENERATOR_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _green_start_states(grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREEN)]
    assert len(coords) == 2, "expected exactly one 2-cell green start marker"

    (r0, c0), (r1, c1) = coords
    assert abs(r0 - r1) + abs(c0 - c1) == 1, "green start marker must be a straight 2-cell segment"

    if r0 == r1:
        left, right = sorted(coords, key=lambda rc: rc[1])
        return [(left, (0, -1)), (right, (0, 1))]

    top, bottom = sorted(coords, key=lambda rc: rc[0])
    return [(top, (-1, 0)), (bottom, (1, 0))]


def _trace_from(
    grid: np.ndarray, start: tuple[int, int], direction: tuple[int, int]
) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    current = start
    path = {tuple(map(int, rc)) for rc in np.argwhere(grid == GREEN)}
    seen_states: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for _ in range(h * w * TURN_LIMIT_FACTOR):
        state = (current, direction)
        if state in seen_states:
            return None
        seen_states.add(state)

        nr = current[0] + direction[0]
        nc = current[1] + direction[1]
        if not _in_bounds((h, w), nr, nc):
            return None

        cell = int(grid[nr, nc])
        if cell == RED:
            return path
        if cell == BLACK:
            current = (nr, nc)
            path.add(current)
            continue
        if cell == CYAN:
            options = []
            for nd in (_turn_left(direction), _turn_right(direction)):
                rr = current[0] + nd[0]
                cc = current[1] + nd[1]
                if not _in_bounds((h, w), rr, cc):
                    continue
                if int(grid[rr, cc]) in (BLACK, RED):
                    options.append(nd)
            if len(options) != 1:
                return None
            direction = options[0]
            continue
        return None

    return None


def _trace_unique_beam(grid: np.ndarray) -> set[tuple[int, int]]:
    assert int(np.count_nonzero(grid == RED)) == 1, "expected exactly one red endpoint"

    traces = []
    for start, direction in _green_start_states(grid):
        path = _trace_from(grid, start, direction)
        if path is not None:
            traces.append(path)

    assert len(traces) == 1, "green segment orientation must determine a unique valid beam path"
    return traces[0]


def _fill_closed_black_regions(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    reachable = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if out[r, c] == BLACK and not reachable[r, c]:
                reachable[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if out[r, c] == BLACK and not reachable[r, c]:
                reachable[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if _in_bounds((h, w), rr, cc) and out[rr, cc] == BLACK and not reachable[rr, cc]:
                reachable[rr, cc] = True
                queue.append((rr, cc))

    out[(out == BLACK) & (~reachable)] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the bent green beam from its start marker to the red endpoint, then fill any black pockets trapped by the new wall.

    Input structure:
    - The grid is a black background with one straight 2-cell green segment, several cyan single pixels, and one red single pixel.
    - The green segment gives the beam's starting location and its initial axis.
    - Cyan pixels act as turn triggers, and the red pixel marks the stopping point.

    Rule:
    - Try the two outward orientations implied by the green segment; exactly one produces a valid beam.
    - The beam advances through black cells. When the next cell is cyan, it stays on its current cell and turns into the unique perpendicular direction whose immediate cell is in-bounds and not cyan.
    - When the next cell is red, the beam stops. Paint the whole traced beam green while preserving the cyan and red markers.
    - After the path is drawn, treat every non-black cell as a wall and recolor each 4-connected black region that cannot reach the grid edge to yellow.

    Key insight:
    - The cyan pixels do not get crossed; they steer the beam, and the finished green path turns the open background into a topology problem where only edge-reachable black cells survive.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    path = _trace_unique_beam(g)
    out = g.copy()
    for r, c in path:
        if int(out[r, c]) in (BLACK, GREEN):
            out[r, c] = GREEN
    return _fill_closed_black_regions(out)


def _build_canonical_template(kind: str, lengths: list[int]) -> np.ndarray | None:
    if kind == "loop4":
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    elif kind == "loop5":
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 1)]
    else:
        raise ValueError(f"unknown template kind: {kind}")

    current = (0, 1)
    inner = (0, 0)
    path_cells = {inner, current}
    special_cells: dict[tuple[int, int], int] = {inner: GREEN, current: GREEN}

    for i, (direction, length) in enumerate(zip(directions, lengths)):
        dr, dc = direction
        for _ in range(length):
            current = (current[0] + dr, current[1] + dc)
            if current in path_cells or current in special_cells:
                return None
            path_cells.add(current)

        if i == len(directions) - 1:
            continue

        next_direction = directions[i + 1]
        trigger = (current[0] + dr, current[1] + dc)
        blocker = (current[0] - next_direction[0], current[1] - next_direction[1])

        if trigger == blocker:
            return None
        for coord in (trigger, blocker):
            if coord in path_cells or coord in special_cells:
                return None
            special_cells[coord] = CYAN

    red = (current[0] + directions[-1][0], current[1] + directions[-1][1])
    if red in path_cells or red in special_cells:
        return None
    special_cells[red] = RED

    all_coords = list(path_cells) + list(special_cells.keys())
    min_r = min(r for r, _ in all_coords)
    max_r = max(r for r, _ in all_coords)
    min_c = min(c for _, c in all_coords)
    max_c = max(c for _, c in all_coords)

    h = max_r - min_r + 1
    w = max_c - min_c + 1
    patch = np.zeros((h, w), dtype=int)
    for (r, c), color in special_cells.items():
        patch[r - min_r, c - min_c] = color
    return patch


def _sample_template() -> np.ndarray | None:
    kind = random.choice(["loop4", "loop5"])
    if kind == "loop4":
        lengths = [
            random.randint(3, 7),
            random.randint(3, 7),
            random.randint(3, 8),
            random.randint(1, 5),
        ]
    else:
        lengths = [
            random.randint(3, 7),
            random.randint(3, 7),
            random.randint(3, 8),
            random.randint(1, 5),
            random.randint(1, 5),
        ]
    return _build_canonical_template(kind, lengths)


def _apply_random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    k = random.randrange(4)
    if k:
        out = np.rot90(out, k)
    return np.asarray(out, dtype=int).copy()


def _embed_in_canvas(patch: np.ndarray) -> np.ndarray:
    ph, pw = map(int, patch.shape)
    h = random.randint(ph + 2, min(30, ph + 10))
    w = random.randint(pw + 2, min(30, pw + 10))
    top = random.randint(1, h - ph - 1)
    left = random.randint(1, w - pw - 1)

    grid = np.zeros((h, w), dtype=int)
    grid[top : top + ph, left : left + pw] = patch
    return grid


def _cyan_are_isolated(grid: np.ndarray) -> bool:
    for r, c in np.argwhere(grid == CYAN):
        r = int(r)
        c = int(c)
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == CYAN:
                return False
    return True


def _valid_generated_sample(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if int(np.count_nonzero(grid == GREEN)) != 2:
        return False
    if int(np.count_nonzero(grid == RED)) != 1:
        return False
    if int(np.count_nonzero(grid == CYAN)) < 6:
        return False
    if not _cyan_are_isolated(grid):
        return False

    try:
        path = _trace_unique_beam(grid)
        out = main(grid)
    except Exception:
        return False

    yellow_count = int(np.count_nonzero(out == YELLOW))
    green_count = int(np.count_nonzero(out == GREEN))
    if yellow_count < 4:
        return False
    if green_count < len(path):
        return False
    if int(np.count_nonzero(out == BLACK)) == 0:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATOR_RETRIES):
        patch = _sample_template()
        if patch is None:
            continue
        patch = _apply_random_symmetry(patch)
        if max(patch.shape) > 24:
            continue
        grid = _embed_in_canvas(patch)
        if _valid_generated_sample(grid):
            return grid
    raise RuntimeError("failed to generate a valid beam-pocket ARC input")
