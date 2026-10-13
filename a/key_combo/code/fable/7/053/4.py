import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5

# Canonical ARC direction order: up, right, down, left.
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

GRID_MIN = 14
GRID_MAX = 22
GEN_RETRIES = 400
EXTRA_OBSTACLE_RANGE = (1, 4)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _advance(pos: tuple[int, int], dir_idx: int, steps: int = 1) -> tuple[int, int]:
    dr, dc = DIRS[dir_idx]
    return pos[0] + dr * steps, pos[1] + dc * steps


def _distance_to_edge(pos: tuple[int, int], dir_idx: int, shape: tuple[int, int]) -> int:
    r, c = pos
    h, w = shape
    if dir_idx == 0:
        return r
    if dir_idx == 1:
        return w - 1 - c
    if dir_idx == 2:
        return h - 1 - r
    return c


def _chebyshev_close(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _is_internal(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return 1 <= r < h - 1 and 1 <= c < w - 1


def _find_start(grid: np.ndarray) -> tuple[int, int]:
    red_cells = np.argwhere(grid == RED)
    assert red_cells.shape[0] == 1, "expected exactly one red start pixel"
    r, c = map(int, red_cells[0])
    h, w = grid.shape
    assert (r in (0, h - 1)) ^ (c in (0, w - 1)), "start must be on a non-corner edge"
    return r, c


def _start_direction(start: tuple[int, int], shape: tuple[int, int]) -> int:
    r, c = start
    h, w = shape
    if r == 0:
        return 2
    if r == h - 1:
        return 0
    if c == 0:
        return 1
    assert c == w - 1
    return 3


def _find_obstacle_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    comps: list[dict] = []
    for r, c in map(tuple, np.argwhere(grid == GREY)):
        rr, cc = int(r), int(c)
        if (rr, cc) in seen:
            continue
        q = deque([(rr, cc)])
        seen.add((rr, cc))
        cells: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            cells.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if not _in_bounds((h, w), nr, nc):
                    continue
                if grid[nr, nc] != GREY or (nr, nc) in seen:
                    continue
                seen.add((nr, nc))
                q.append((nr, nc))
        cells_set = set(cells)
        for cr, cc in cells:
            assert _is_internal((cr, cc), (h, w)), "obstacles must stay in the interior"
            for nr, nc in _neighbors4(cr, cc):
                if _in_bounds((h, w), nr, nc) and grid[nr, nc] == GREY:
                    assert (nr, nc) in cells_set, "grey obstacles must be 4-connected components"
        comps.append({"cells": cells, "size": len(cells)})

    for comp in comps:
        assert 1 <= comp["size"] <= 3, "obstacle size must encode 1/2/3 quarter-turns"
    for i, left in enumerate(comps):
        for right in comps[i + 1 :]:
            for a in left["cells"]:
                for b in right["cells"]:
                    assert not _chebyshev_close(a, b), "obstacles must be visually separated"
    return comps


def _simulate(
    grid: np.ndarray,
    turn_policy=None,
) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    colors = set(map(int, g.flat))
    assert colors <= {BLACK, RED, GREY}, "input should only contain black, red, and grey"

    start = _find_start(g)
    start_dir = _start_direction(start, g.shape)
    components = _find_obstacle_components(g)

    cell_to_size: dict[tuple[int, int], int] = {}
    for comp in components:
        for cell in comp["cells"]:
            cell_to_size[cell] = int(comp["size"])

    out = g.copy()
    pos = start
    dir_idx = start_dir
    seen_states = {(pos[0], pos[1], dir_idx)}
    move_cells: list[tuple[int, int]] = []
    hit_sizes: list[int] = []

    for _ in range(g.shape[0] * g.shape[1] * 8):
        nr, nc = _advance(pos, dir_idx, 1)
        if not _in_bounds(g.shape, nr, nc):
            out[pos] = RED
            return {
                "output": out,
                "start": start,
                "exit": pos,
                "move_cells": move_cells,
                "hit_sizes": hit_sizes,
            }

        if (nr, nc) in cell_to_size:
            real_turn = cell_to_size[(nr, nc)]
            hit_index = len(hit_sizes)
            hit_sizes.append(real_turn)
            turn = real_turn if turn_policy is None else int(turn_policy(hit_index, real_turn))
            assert turn in (1, 2, 3), "turn policy must return 1/2/3"
            dir_idx = (dir_idx + turn) % 4
            state = (pos[0], pos[1], dir_idx)
            assert state not in seen_states, "ray entered a loop"
            seen_states.add(state)
            continue

        assert g[nr, nc] == BLACK, "ray may only traverse black cells"
        pos = (nr, nc)
        assert pos != start, "generated samples should not revisit the start marker"
        move_cells.append(pos)
        out[nr, nc] = YELLOW
        state = (nr, nc, dir_idx)
        assert state not in seen_states, "ray entered a loop"
        seen_states.add(state)

    raise AssertionError("ray did not exit within the safety bound")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire a ray inward from the edge marker and rotate it clockwise by the hit obstacle's pixel count.

    Input structure:
    - Black background with exactly one red start pixel on a non-corner edge cell.
    - Several grey interior obstacles, each a visually separated 4-connected component of 1, 2, or 3 pixels.

    Rule:
    - Launch the ray perpendicular to the marked edge toward the grid interior.
    - Whenever the next cell is black, move into it and color that traversed cell yellow.
    - Whenever the next cell belongs to an obstacle, stay in the current cell and rotate clockwise by 1/2/3 quarter-turns according to that obstacle's pixel count.
    - When the next step would leave the grid, stop and color the last in-bounds ray cell red; keep all obstacles unchanged.

    Key insight:
    - The obstacle's area, not its shape or the hit face, is the whole turning program for the ray.
    """
    return _simulate(np.asarray(grid, dtype=int))["output"]


def _shape_patterns(size: int) -> list[list[tuple[int, int]]]:
    if size == 1:
        return [[(0, 0)]]
    if size == 2:
        return [
            [(0, 0), (1, 0)],
            [(0, 0), (-1, 0)],
            [(0, 0), (0, 1)],
            [(0, 0), (0, -1)],
        ]
    assert size == 3
    return [
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (-1, 0), (-2, 0)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (0, -1), (0, -2)],
        [(0, 0), (1, 0), (-1, 0)],
        [(0, 0), (0, 1), (0, -1)],
        [(0, 0), (1, 0), (0, 1)],
        [(0, 0), (1, 0), (0, -1)],
        [(0, 0), (-1, 0), (0, 1)],
        [(0, 0), (-1, 0), (0, -1)],
    ]


def _edge_starts(edge: str, h: int, w: int) -> list[tuple[int, int]]:
    if edge == "top":
        return [(0, c) for c in range(1, w - 1)]
    if edge == "bottom":
        return [(h - 1, c) for c in range(1, w - 1)]
    if edge == "left":
        return [(r, 0) for r in range(1, h - 1)]
    return [(r, w - 1) for r in range(1, h - 1)]


def _search_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    turn_sizes: list[int],
) -> dict | None:
    start_dir = _start_direction(start, shape)

    def rec(
        hit_idx: int,
        pos: tuple[int, int],
        dir_idx: int,
        seen_states: set[tuple[int, int, int]],
        path_cells: set[tuple[int, int]],
        anchors: list[tuple[tuple[int, int], int]],
        move_cells: list[tuple[int, int]],
    ) -> dict | None:
        if hit_idx == len(turn_sizes):
            dist = _distance_to_edge(pos, dir_idx, shape)
            if dist < 1:
                return None
            new_states = set()
            new_cells: list[tuple[int, int]] = []
            for step in range(1, dist + 1):
                cell = _advance(pos, dir_idx, step)
                state = (cell[0], cell[1], dir_idx)
                if state in seen_states:
                    return None
                if cell == start:
                    return None
                new_states.add(state)
                new_cells.append(cell)
            return {
                "turn_sizes": list(turn_sizes),
                "anchors": list(anchors),
                "move_cells": move_cells + new_cells,
                "path_cells": path_cells | set(new_cells),
                "exit": new_cells[-1],
            }

        dist = _distance_to_edge(pos, dir_idx, shape)
        max_run = dist - 2
        if max_run < 1:
            return None
        candidates = list(range(1, max_run + 1))
        random.shuffle(candidates)
        for run in candidates:
            new_states = set()
            new_cells: list[tuple[int, int]] = []
            valid = True
            for step in range(1, run + 1):
                cell = _advance(pos, dir_idx, step)
                state = (cell[0], cell[1], dir_idx)
                if state in seen_states or cell == start:
                    valid = False
                    break
                new_states.add(state)
                new_cells.append(cell)
            if not valid:
                continue

            end = new_cells[-1]
            anchor = _advance(pos, dir_idx, run + 1)
            if anchor == start or anchor in path_cells:
                continue
            if not _is_internal(anchor, shape):
                continue
            if any(_chebyshev_close(anchor, prev_anchor) for prev_anchor, _ in anchors):
                continue

            new_dir = (dir_idx + turn_sizes[hit_idx]) % 4
            pivot_state = (end[0], end[1], new_dir)
            if pivot_state in seen_states or pivot_state in new_states:
                continue

            result = rec(
                hit_idx + 1,
                end,
                new_dir,
                seen_states | new_states | {pivot_state},
                path_cells | set(new_cells),
                anchors + [(anchor, turn_sizes[hit_idx])],
                move_cells + new_cells,
            )
            if result is not None:
                return result
        return None

    return rec(
        0,
        start,
        start_dir,
        {(start[0], start[1], start_dir)},
        {start},
        [],
        [],
    )


def _cells_valid_for_obstacle(
    cells: list[tuple[int, int]],
    shape: tuple[int, int],
    path_cells: set[tuple[int, int]],
    start: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> bool:
    for cell in cells:
        if not _is_internal(cell, shape):
            return False
        if cell == start or cell in path_cells or cell in occupied:
            return False
    for cell in cells:
        for other in occupied:
            if _chebyshev_close(cell, other):
                return False
    return True


def _place_hit_obstacles(
    shape: tuple[int, int],
    start: tuple[int, int],
    path_cells: set[tuple[int, int]],
    anchors: list[tuple[tuple[int, int], int]],
) -> list[list[tuple[int, int]]] | None:
    placed: list[list[tuple[int, int]]] = []
    occupied: set[tuple[int, int]] = set()
    order = list(range(len(anchors)))
    order.sort(key=lambda i: anchors[i][1], reverse=True)
    obstacle_cells_by_index: list[list[tuple[int, int]] | None] = [None] * len(anchors)

    for idx in order:
        anchor, size = anchors[idx]
        patterns = _shape_patterns(size)
        random.shuffle(patterns)
        chosen = None
        for pattern in patterns:
            cells = [(anchor[0] + dr, anchor[1] + dc) for dr, dc in pattern]
            if _cells_valid_for_obstacle(cells, shape, path_cells, start, occupied):
                chosen = cells
                break
        if chosen is None:
            return None
        obstacle_cells_by_index[idx] = chosen
        occupied.update(chosen)

    for cells in obstacle_cells_by_index:
        assert cells is not None
        placed.append(cells)
    return placed


def _place_extra_obstacles(
    shape: tuple[int, int],
    start: tuple[int, int],
    path_cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    h, w = shape
    extras: list[list[tuple[int, int]]] = []
    targets = random.randint(*EXTRA_OBSTACLE_RANGE)
    anchors = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(anchors)

    while anchors and len(extras) < targets:
        anchor = anchors.pop()
        size = random.randint(1, 3)
        patterns = _shape_patterns(size)
        random.shuffle(patterns)
        for pattern in patterns:
            cells = [(anchor[0] + dr, anchor[1] + dc) for dr, dc in pattern]
            if any(_chebyshev_close(cell, start) for cell in cells):
                continue
            if not _cells_valid_for_obstacle(cells, shape, path_cells, start, occupied):
                continue
            extras.append(cells)
            occupied.update(cells)
            break
    return extras


def _render_input(
    shape: tuple[int, int],
    start: tuple[int, int],
    obstacles: list[list[tuple[int, int]]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    grid[start] = RED
    for cells in obstacles:
        for r, c in cells:
            grid[r, c] = GREY
    return grid


def _render_expected(grid: np.ndarray, move_cells: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r, c in move_cells:
        out[r, c] = YELLOW
    out[move_cells[-1]] = RED
    return out


def _validate_semantics(grid: np.ndarray, plan: dict, expected: np.ndarray) -> bool:
    try:
        actual = main(grid)
        trace = _simulate(grid)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if not np.array_equal(trace["output"], expected):
        return False
    if trace["hit_sizes"] != plan["turn_sizes"]:
        return False
    if set(trace["hit_sizes"]) != {1, 2, 3}:
        return False
    if trace["exit"] != plan["exit"]:
        return False
    if np.count_nonzero(actual == RED) != 2:
        return False
    if len(set(plan["move_cells"])) < 8:
        return False

    for i, real_turn in enumerate(plan["turn_sizes"]):
        wrong_turn = 1 if real_turn != 1 else 2
        try:
            alt = _simulate(
                grid,
                turn_policy=lambda hit_i, size, idx=i, wrong=wrong_turn: wrong if hit_i == idx else size,
            )["output"]
        except Exception:
            continue
        if np.array_equal(alt, actual):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    shape = (h, w)
    turn_sizes = random.sample([1, 2, 3], 3)

    edges = ["top", "bottom", "left", "right"]
    random.shuffle(edges)
    for edge in edges:
        starts = _edge_starts(edge, h, w)
        random.shuffle(starts)
        for start in starts[: min(len(starts), 10)]:
            plan = _search_path(shape, start, turn_sizes)
            if plan is None:
                continue
            hit_obstacles = _place_hit_obstacles(shape, start, plan["path_cells"], plan["anchors"])
            if hit_obstacles is None:
                continue
            occupied = {cell for obstacle in hit_obstacles for cell in obstacle}
            extra_obstacles = _place_extra_obstacles(shape, start, plan["path_cells"], occupied)
            if not extra_obstacles:
                continue
            grid = _render_input(shape, start, hit_obstacles + extra_obstacles)
            expected = _render_expected(grid, plan["move_cells"])
            if _validate_semantics(grid, plan, expected):
                return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-programming puzzle input")
