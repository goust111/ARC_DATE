from __future__ import annotations

import random

import numpy as np


RED = 2
BLUE = 1
GREEN = 3
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
TURN_PRIORITY = {1: 0, 0: 1, 3: 2, 2: 3}

GEN_COLOR_POOL = [0, 4, 5, 6, 7, 8, 9]
GRID_MIN = 10
GRID_MAX = 18
MAIN_LEN_MIN = 8
MAIN_LEN_MAX = 15
BRANCH_LEN_MAX = 4
GEN_RETRIES = 800


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]


def _dir_index(direction: tuple[int, int]) -> int:
    return DIRS.index((int(direction[0]), int(direction[1])))


def _right(direction: tuple[int, int]) -> tuple[int, int]:
    return DIRS[(_dir_index(direction) + 1) % 4]


def _left(direction: tuple[int, int]) -> tuple[int, int]:
    return DIRS[(_dir_index(direction) - 1) % 4]


def _turn_rank(forward: tuple[int, int], candidate: tuple[int, int]) -> int:
    delta = (_dir_index(candidate) - _dir_index(forward)) % 4
    return int(TURN_PRIORITY[delta])


def _step(cell: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    return int(cell[0] + direction[0]), int(cell[1] + direction[1])


def _edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _path_neighbors(mask: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for nr, nc in _neighbors4(r, c):
        if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]):
            out.append((nr, nc))
    return out


def _walked_cells(grid: np.ndarray) -> set[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    red_coords = np.argwhere(g == RED)
    blue_coords = np.argwhere(g == BLUE)
    assert red_coords.shape == (1, 2), "expected exactly one red endpoint"
    assert blue_coords.shape == (1, 2), "expected exactly one blue endpoint"
    red = tuple(map(int, red_coords[0]))
    blue = tuple(map(int, blue_coords[0]))

    colors = set(map(int, np.unique(g)))
    path_colors = colors - {bg, RED, BLUE}
    assert len(path_colors) == 1, "expected a single path color"
    path_color = int(next(iter(path_colors)))
    assert path_color != GREEN, "input path color must differ from the output green"

    path_mask = g == path_color
    assert bool(path_mask.any()), "expected at least one path pixel"
    assert _connected(path_mask), "path network must be 4-connected"

    start_neighbors = [p for p in _neighbors4(*red) if _in_bounds(g.shape, *p) and bool(path_mask[p])]
    end_neighbors = [p for p in _neighbors4(*blue) if _in_bounds(g.shape, *p) and bool(path_mask[p])]
    assert len(start_neighbors) == 1, "red endpoint must touch exactly one path pixel"
    assert len(end_neighbors) == 1, "blue endpoint must touch exactly one path pixel"
    start = start_neighbors[0]
    end = end_neighbors[0]

    forward = (start[0] - red[0], start[1] - red[1])
    cur = start
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    walked = {cur}
    limit = max(8, int(path_mask.sum()) * 6)

    for _ in range(limit):
        if cur == end:
            return walked
        candidates = []
        for nxt in _path_neighbors(path_mask, cur):
            edge = _edge(cur, nxt)
            if edge in visited_edges:
                continue
            direction = (nxt[0] - cur[0], nxt[1] - cur[1])
            candidates.append((_turn_rank(forward, direction), _dir_index(direction), direction, nxt, edge))
        assert candidates, "walker got stuck before reaching the blue endpoint"
        candidates.sort()
        _, _, direction, nxt, edge = candidates[0]
        visited_edges.add(edge)
        forward = direction
        cur = nxt
        walked.add(cur)

    raise AssertionError("walker did not reach the blue endpoint in time")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the 4-connected path network from the red marker, choosing the rightmost
    untraversed branch at each junction, and recolor only the visited path pixels green.
    """
    g = np.asarray(grid, dtype=int)
    walked = _walked_cells(g)
    out = g.copy()
    for r, c in walked:
        out[r, c] = GREEN
    return out


def _choose_weighted(rng: random.Random, weighted: list[tuple[tuple[int, int], int]]) -> tuple[int, int]:
    total = sum(weight for _, weight in weighted)
    pick = rng.randint(1, total)
    running = 0
    for item, weight in weighted:
        running += weight
        if pick <= running:
            return item
    return weighted[-1][0]


def _can_add(
    cell: tuple[int, int],
    parent: tuple[int, int] | None,
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> bool:
    r, c = cell
    if not (1 <= r < shape[0] - 1 and 1 <= c < shape[1] - 1):
        return False
    if cell in occupied or cell in blocked:
        return False
    neighbors = {nbr for nbr in _neighbors4(r, c) if nbr in occupied}
    if parent is None:
        return not neighbors
    return neighbors == {parent}


def _grow_main_path(
    rng: random.Random,
    start: tuple[int, int],
    initial_dir: tuple[int, int],
    target_len: int,
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    occupied = {start}
    path = [start]
    cur = start
    forward = initial_dir

    while len(path) < target_len:
        weighted: list[tuple[tuple[int, int], int]] = []
        for direction, weight in [(_right(forward), 4), (forward, 4), (_left(forward), 2)]:
            nxt = _step(cur, direction)
            if _can_add(nxt, cur, occupied, shape, blocked):
                weighted.append((direction, weight))
        if not weighted:
            break
        direction = _choose_weighted(rng, weighted)
        cur = _step(cur, direction)
        occupied.add(cur)
        path.append(cur)
        forward = direction

    if len(path) < MAIN_LEN_MIN:
        return None
    return path


def _branch_infos(
    path: list[tuple[int, int]],
    initial_dir: tuple[int, int],
) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    for i in range(len(path) - 1):
        incoming = initial_dir if i == 0 else (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        outgoing = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        main_rank = _turn_rank(incoming, outgoing)
        allowed = []
        for direction in (_right(incoming), incoming, _left(incoming)):
            if direction == outgoing:
                continue
            if _turn_rank(incoming, direction) > main_rank:
                allowed.append(direction)
        infos.append(
            {
                "root": path[i],
                "rank": main_rank,
                "allowed": allowed,
            }
        )
    return infos


def _available_branch_dirs(
    root: tuple[int, int],
    directions: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    out = []
    for direction in directions:
        first = _step(root, direction)
        if _can_add(first, root, occupied, shape, blocked):
            out.append(direction)
    return out


def _grow_branch(
    rng: random.Random,
    root: tuple[int, int],
    first_dir: tuple[int, int],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    max_len: int,
) -> list[tuple[int, int]] | None:
    branch: list[tuple[int, int]] = []
    branch_occupied = set(occupied)
    cur = root
    forward = first_dir
    target_len = rng.randint(1, max_len)

    for step_index in range(target_len):
        weighted: list[tuple[tuple[int, int], int]] = []
        options = [forward] if step_index == 0 else [forward, _right(forward), _left(forward)]
        weights = {forward: 5, _right(forward): 2, _left(forward): 2}
        for direction in options:
            nxt = _step(cur, direction)
            if _can_add(nxt, cur, branch_occupied, shape, blocked):
                weighted.append((direction, weights[direction]))
        if not weighted:
            break
        direction = _choose_weighted(rng, weighted)
        cur = _step(cur, direction)
        branch.append(cur)
        branch_occupied.add(cur)
        forward = direction

    if not branch:
        return None
    return branch


def _adjacent_path_cells(
    path_cells: set[tuple[int, int]],
    point: tuple[int, int],
) -> list[tuple[int, int]]:
    return [nbr for nbr in _neighbors4(*point) if nbr in path_cells]


def _build_candidate(rng: random.Random, mode: str) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    shape = (h, w)

    bg = rng.choice(GEN_COLOR_POOL)
    path_color = rng.choice([color for color in GEN_COLOR_POOL if color != bg])

    start = (rng.randint(2, h - 3), rng.randint(2, w - 3))
    initial_dir = rng.choice(DIRS)
    red = (start[0] - initial_dir[0], start[1] - initial_dir[1])
    if not _in_bounds(shape, *red):
        return None

    blocked = {red}
    for nbr in _neighbors4(*red):
        if _in_bounds(shape, *nbr) and nbr != start:
            blocked.add(nbr)

    max_target = min(MAIN_LEN_MAX, max(MAIN_LEN_MIN, (h - 2) * (w - 2) // 3))
    min_target = MAIN_LEN_MIN if mode != "mixed" else max(MAIN_LEN_MIN + 1, 10)
    if min_target > max_target:
        min_target = MAIN_LEN_MIN
    target_len = rng.randint(min_target, max_target)
    path = _grow_main_path(rng, start, initial_dir, target_len, shape, blocked)
    if path is None:
        return None

    path_cells = set(path)
    infos = _branch_infos(path, initial_dir)
    needed_ranks = [0] if mode == "right" else [1] if mode == "straight" else [0, 1]

    used_dirs: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for rank in needed_ranks:
        candidates = []
        for info in infos:
            if int(info["rank"]) != rank:
                continue
            root = info["root"]
            allowed = [d for d in info["allowed"] if d not in used_dirs.get(root, set())]
            available = _available_branch_dirs(root, allowed, path_cells, shape, blocked)
            if available:
                candidates.append((root, available))
        if not candidates:
            return None
        root, available = rng.choice(candidates)
        direction = rng.choice(available)
        branch = _grow_branch(rng, root, direction, path_cells, shape, blocked, BRANCH_LEN_MAX)
        if branch is None:
            return None
        path_cells.update(branch)
        used_dirs.setdefault(root, set()).add(direction)

    extra_budget = rng.randint(0, 3)
    shuffled_infos = infos[:]
    rng.shuffle(shuffled_infos)
    for info in shuffled_infos:
        if extra_budget <= 0:
            break
        if int(info["rank"]) not in (0, 1):
            continue
        root = info["root"]
        allowed = [d for d in info["allowed"] if d not in used_dirs.get(root, set())]
        available = _available_branch_dirs(root, allowed, path_cells, shape, blocked)
        if not available:
            continue
        if rng.random() > 0.45:
            continue
        direction = rng.choice(available)
        branch = _grow_branch(rng, root, direction, path_cells, shape, blocked, BRANCH_LEN_MAX)
        if branch is None:
            continue
        path_cells.update(branch)
        used_dirs.setdefault(root, set()).add(direction)
        extra_budget -= 1

    end = path[-1]
    final_dir = initial_dir if len(path) == 1 else (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])
    blue = (end[0] + final_dir[0], end[1] + final_dir[1])
    if not _in_bounds(shape, *blue):
        return None
    if blue in path_cells or blue == red:
        return None

    if _adjacent_path_cells(path_cells, red) != [start]:
        return None
    if _adjacent_path_cells(path_cells, blue) != [end]:
        return None
    if len(path_cells) <= len(path):
        return None

    degree_values = []
    for cell in path_cells:
        degree_values.append(sum(1 for nbr in _neighbors4(*cell) if nbr in path_cells))
    if max(degree_values, default=0) < 3:
        return None

    grid = np.full(shape, bg, dtype=int)
    for r, c in path_cells:
        grid[r, c] = path_color
    grid[red] = RED
    grid[blue] = BLUE

    if _get_bg(grid) != bg:
        return None

    try:
        walked = _walked_cells(grid)
        out = main(grid)
    except Exception:
        return None

    if walked != set(path):
        return None
    if int(np.count_nonzero(out == GREEN)) != len(path):
        return None
    if not np.array_equal(out == RED, grid == RED):
        return None
    if not np.array_equal(out == BLUE, grid == BLUE):
        return None
    if int(np.count_nonzero(out == path_color)) != len(path_cells) - len(path):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    mode = rng.choice(["right", "straight", "mixed"])
    for _ in range(GEN_RETRIES):
        candidate = _build_candidate(rng, mode)
        if candidate is not None:
            return candidate
    raise RuntimeError(f"failed to generate a valid input for mode={mode}")
