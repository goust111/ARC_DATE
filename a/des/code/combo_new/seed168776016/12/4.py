import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
ENDPOINT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
GEN_ATTEMPTS = 400
PATH_BUILD_ATTEMPTS = 140
PATH_MODE_CHOICES = ("line", "walk")


def _neighbors4(cell: tuple[int, int], shape: tuple[int, int]):
    r, c = cell
    h, w = shape
    for dr, dc in NEIGHBORS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cell = q.popleft()
                comp.append(cell)
                for nr, nc in _neighbors4(cell, mask.shape):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _degree(cell: tuple[int, int], points: set[tuple[int, int]], shape: tuple[int, int]) -> int:
    return sum((nbr in points) for nbr in _neighbors4(cell, shape))


def _trace_path(
    points: set[tuple[int, int]],
    start: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    order = [start]
    prev = None
    cur = start
    while True:
        nxts = [nbr for nbr in _neighbors4(cur, shape) if nbr in points and nbr != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path must not branch"
        prev, cur = cur, nxts[0]
        order.append(cur)
    assert len(order) == len(points), "path traversal must cover the component"
    return order


def _ordered_component_path(
    grid: np.ndarray, comp: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], int, int]:
    points = set(comp)
    degrees = {cell: _degree(cell, points, grid.shape) for cell in comp}
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "each path must have exactly two endpoints"
    assert all(deg in (1, 2) for deg in degrees.values()), "simple paths only"

    colored_endpoints = [cell for cell in endpoints if int(grid[cell]) not in (BG, GREY)]
    assert len(colored_endpoints) == 2, "both endpoints must be colored"
    assert set(colored_endpoints) == set(endpoints), "only endpoints may be non-grey"

    for cell in comp:
        if cell not in endpoints:
            assert int(grid[cell]) == GREY, "internal path cells must be grey"

    start, end = sorted(colored_endpoints)
    ordered = _trace_path(points, start, grid.shape)
    assert ordered[-1] == end, "trace must end at the other endpoint"
    return ordered, int(grid[start]), int(grid[end])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every thin path so each half inherits the color of the nearer endpoint.

    Input structure:
    - The background is black.
    - Each foreground component is a 1-pixel-wide 4-connected simple path.
    - Internal path cells are grey, and the two degree-1 endpoint cells are non-black,
      non-grey colors with different colors on the same path.
    - Distinct paths are disjoint and do not branch.

    Rule:
    - Process each foreground component independently as a single path.
    - Order the path from the row-major earlier colored endpoint to the later one.
    - Let L be the number of path cells, including both colored endpoints.
    - Paint the first ceil(L/2) cells with the first endpoint color and the remaining
      cells with the other endpoint color, replacing every grey cell.

    Key insight:
    - The task is to recover each path's topology first, then split that ordered path at
      its midpoint so color assignment depends on path distance rather than local geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for comp in _components(g != BG):
        ordered, first_color, second_color = _ordered_component_path(g, comp)
        split = (len(ordered) + 1) // 2
        for cell in ordered[:split]:
            out[cell] = first_color
        for cell in ordered[split:]:
            out[cell] = second_color
    return out


def _path_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(2, len(path)):
        dr1 = path[i - 1][0] - path[i - 2][0]
        dc1 = path[i - 1][1] - path[i - 2][1]
        dr2 = path[i][0] - path[i - 1][0]
        dc2 = path[i][1] - path[i - 1][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def _halo8(path: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    halo: set[tuple[int, int]] = set()
    for r, c in path:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                halo.add((rr, cc))
    return halo


def _sample_lengths(count: int, rng: random.Random, h: int, w: int) -> list[int]:
    max_len = min(14, h + w // 3)
    if count >= 3:
        max_len = min(max_len, 11)
    if count == 4:
        max_len = min(max_len, 9)

    parities = [rng.randrange(2) for _ in range(count)]
    if count >= 2 and len(set(parities)) == 1:
        parities[0] = 0
        parities[1] = 1

    lengths: list[int] = []
    for parity in parities:
        if parity == 0:
            choices = [n for n in range(4, max_len + 1) if n % 2 == 0]
        else:
            choices = [n for n in range(5, max_len + 1) if n % 2 == 1]
        lengths.append(rng.choice(choices))
    rng.shuffle(lengths)
    return lengths


def _max_path_count(h: int, w: int) -> int:
    area = h * w
    if area >= 360:
        return 4
    if area >= 250:
        return 3
    if area >= 170:
        return 2
    return 1


def _line_path(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    length: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    h, w = shape
    orientations = []
    if h >= length:
        orientations.append((1, 0))
    if w >= length:
        orientations.append((0, 1))
    if not orientations:
        return None

    for _ in range(PATH_BUILD_ATTEMPTS):
        dr, dc = rng.choice(orientations)
        if dr == 1:
            r0 = rng.randrange(0, h - length + 1)
            c0 = rng.randrange(w)
        else:
            r0 = rng.randrange(h)
            c0 = rng.randrange(0, w - length + 1)
        path = [(r0 + i * dr, c0 + i * dc) for i in range(length)]
        if any(cell in blocked for cell in path):
            continue
        return path
    return None


def _candidate_ok(
    nxt: tuple[int, int],
    cur: tuple[int, int],
    path_set: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    if nxt in blocked or nxt in path_set:
        return False
    adjacent = {nbr for nbr in _neighbors4(nxt, shape) if nbr in path_set}
    return adjacent == {cur}


def _has_future_step(
    cur: tuple[int, int],
    path_set: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    for nxt in _neighbors4(cur, shape):
        if _candidate_ok(nxt, cur, path_set, blocked, shape):
            return True
    return False


def _dfs_walk(
    path: list[tuple[int, int]],
    path_set: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    target_len: int,
    shape: tuple[int, int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    if len(path) == target_len:
        return list(path)

    cur = path[-1]
    prev = path[-2] if len(path) >= 2 else None
    candidates = []
    for nxt in _neighbors4(cur, shape):
        if not _candidate_ok(nxt, cur, path_set, blocked, shape):
            continue
        if len(path) + 1 < target_len:
            path_set.add(nxt)
            future_ok = _has_future_step(nxt, path_set, blocked, shape)
            path_set.remove(nxt)
            if not future_ok:
                continue
        candidates.append(nxt)

    if prev is not None and len(candidates) > 1 and rng.random() < 0.5:
        straight = (cur[0] - prev[0], cur[1] - prev[1])
        straight_next = [nxt for nxt in candidates if (nxt[0] - cur[0], nxt[1] - cur[1]) == straight]
        other_next = [nxt for nxt in candidates if nxt not in straight_next]
        rng.shuffle(straight_next)
        rng.shuffle(other_next)
        candidates = straight_next + other_next
    else:
        rng.shuffle(candidates)

    for nxt in candidates:
        path.append(nxt)
        path_set.add(nxt)
        result = _dfs_walk(path, path_set, blocked, target_len, shape, rng)
        if result is not None:
            return result
        path.pop()
        path_set.remove(nxt)
    return None


def _walk_path(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    length: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    h, w = shape
    free_cells = [(r, c) for r in range(h) for c in range(w) if (r, c) not in blocked]
    rng.shuffle(free_cells)
    for start in free_cells[: min(len(free_cells), PATH_BUILD_ATTEMPTS)]:
        result = _dfs_walk([start], {start}, blocked, length, shape, rng)
        if result is not None and _path_turns(result) >= 1:
            return result
    return None


def _build_path(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    length: int,
    mode: str,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    if mode == "line":
        return _line_path(shape, blocked, length, rng)
    return _walk_path(shape, blocked, length, rng)


def _paint_input(
    shape: tuple[int, int],
    paths: list[list[tuple[int, int]]],
    rng: random.Random,
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    colors = list(ENDPOINT_COLORS)
    rng.shuffle(colors)

    for i, path in enumerate(paths):
        color_a = colors[2 * i]
        color_b = colors[2 * i + 1]
        if rng.random() < 0.5:
            color_a, color_b = color_b, color_a

        for cell in path[1:-1]:
            grid[cell] = GREY
        grid[path[0]] = color_a
        grid[path[-1]] = color_b
    return grid


def _valid_generated_instance(grid: np.ndarray, expected_count: int) -> bool:
    try:
        comps = _components(grid != BG)
        if len(comps) != expected_count:
            return False
        solved = main(grid)
        if np.any(solved == GREY):
            return False
        for comp in comps:
            ordered, first_color, second_color = _ordered_component_path(grid, comp)
            split = (len(ordered) + 1) // 2
            for cell in ordered[:split]:
                if int(solved[cell]) != first_color:
                    return False
            for cell in ordered[split:]:
                if int(solved[cell]) != second_color:
                    return False
    except AssertionError:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    max_paths = _max_path_count(h, w)
    path_count = rng.randint(1, max_paths)

    lengths = _sample_lengths(path_count, rng, h, w)
    blocked: set[tuple[int, int]] = set()
    paths: list[list[tuple[int, int]]] = []

    for idx, length in enumerate(lengths):
        modes = list(PATH_MODE_CHOICES)
        rng.shuffle(modes)
        if path_count == 1 and idx == 0:
            preferred = rng.choice(PATH_MODE_CHOICES)
            modes = [preferred] + [m for m in modes if m != preferred]
        path = None
        for mode in modes:
            path = _build_path((h, w), blocked, length, mode, rng)
            if path is not None:
                break
        if path is None:
            return None
        paths.append(path)
        blocked |= _halo8(path, (h, w))

    grid = _paint_input((h, w), paths, rng)
    if not _valid_generated_instance(grid, path_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
