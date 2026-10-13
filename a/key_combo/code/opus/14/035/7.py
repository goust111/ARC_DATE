from __future__ import annotations

from collections import deque
import random

import numpy as np


class Color:
    BLACK = 0
    YELLOW = 4
    GREY = 5
    TEAL = 8
    MAROON = 9


Cell = tuple[int, int]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _simulate_rays(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    seed_rows = sorted(int(r) for r, c in np.argwhere(g == Color.YELLOW) if c == 0)
    for r in seed_rows:
        hit = False
        for c in range(1, w):
            if int(g[r, c]) == Color.GREY:
                hit = True
                for rr in (r - 1, r + 1):
                    if in_bounds(g.shape, rr, c) and int(g[rr, c]) == Color.BLACK:
                        out[rr, c] = Color.YELLOW
                break
            if int(g[r, c]) == Color.BLACK:
                out[r, c] = Color.YELLOW
        assert hit, "each seed row must hit an obstacle"
    return out


def _forbidden_from_yellow(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(grid == Color.YELLOW):
        r = int(r)
        c = int(c)
        forbidden[r, c] = True
        for nr, nc in neighbors8(r, c):
            if in_bounds(grid.shape, nr, nc):
                forbidden[nr, nc] = True
    return forbidden


def _track_cells(grid: np.ndarray) -> set[Cell]:
    return {
        (int(r), int(c))
        for r, c in np.argwhere((grid == Color.TEAL) | (grid == Color.MAROON))
    }


def _simple_path_count(passable: set[Cell], start: Cell, end: Cell, limit: int = 2) -> int:
    count = 0
    stack: list[tuple[Cell, set[Cell]]] = [(start, {start})]
    while stack:
        cell, seen = stack.pop()
        if cell == end:
            count += 1
            if count >= limit:
                return count
            continue
        r, c = cell
        nxts = []
        for nr, nc in neighbors4(r, c):
            nxt = (nr, nc)
            if nxt in passable and nxt not in seen:
                nxts.append(nxt)
        for nxt in sorted(nxts, reverse=True):
            stack.append((nxt, seen | {nxt}))
    return count


def _unique_simple_path(passable: set[Cell], start: Cell, end: Cell) -> list[Cell]:
    found: list[Cell] | None = None
    stack: list[tuple[Cell, list[Cell], set[Cell]]] = [(start, [start], {start})]
    while stack:
        cell, path, seen = stack.pop()
        if cell == end:
            if found is not None:
                raise AssertionError("expected a unique simple path")
            found = path
            continue
        r, c = cell
        nxts = []
        for nr, nc in neighbors4(r, c):
            nxt = (nr, nc)
            if nxt in passable and nxt not in seen:
                nxts.append(nxt)
        for nxt in sorted(nxts, reverse=True):
            stack.append((nxt, path + [nxt], seen | {nxt}))
    assert found is not None, "endpoints must remain connected"
    return found


def _shortest_path_multiplicity(passable: set[Cell], start: Cell, end: Cell, cap: int = 3) -> int:
    dist = {start: 0}
    ways = {start: 1}
    q = deque([start])
    while q:
        cell = q.popleft()
        r, c = cell
        for nr, nc in neighbors4(r, c):
            nxt = (nr, nc)
            if nxt not in passable:
                continue
            nd = dist[cell] + 1
            if nxt not in dist:
                dist[nxt] = nd
                ways[nxt] = min(cap, ways[cell])
                q.append(nxt)
            elif dist[nxt] == nd:
                ways[nxt] = min(cap, ways[nxt] + ways[cell])
    return ways.get(end, 0)


def _passable_after_forbidden(grid: np.ndarray, forbidden: np.ndarray) -> set[Cell]:
    passable = set()
    for r, c in np.argwhere((grid == Color.TEAL) | (grid == Color.MAROON)):
        r = int(r)
        c = int(c)
        if int(grid[r, c]) == Color.MAROON or not bool(forbidden[r, c]):
            passable.add((r, c))
    return passable


def main(grid: np.ndarray) -> np.ndarray:
    """Construct the ray-based exclusion zone and recolor the sole safe maroon-to-maroon track route.

    Input structure:
    - A black background contains 1-2 yellow seed pixels on the left border, several gray obstacle pixels, a teal 4-connected track, and exactly two maroon endpoints embedded in that track.
    - The teal track contains branch choices, so without any exclusion there are multiple endpoint-to-endpoint routes.

    Rule:
    - From each yellow seed, extend a horizontal yellow ray rightward through black cells until the ray reaches a gray obstacle; the obstacle stays gray and creates one-cell yellow side wakes directly above and below when those cells are black.
    - Every yellow cell, including seeds, ray cells, and side wakes, and every cell in each yellow cell's 8-neighborhood becomes forbidden.
    - In the teal/maroon track graph, remove forbidden teal cells while keeping the maroon endpoints, then find the unique remaining 4-connected simple path between the endpoints and recolor its teal cells to maroon.
    - Keep all non-path teal cells, yellow ray cells, and gray obstacles unchanged.

    Key insight:
    - The ray simulation is only an intermediate computation: its side wakes expand the forbidden field just enough to cut away decoy branches, so the intended endpoint-to-endpoint path becomes uniquely recoverable only after the exclusion zone is built.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    endpoints = [tuple(map(int, p)) for p in np.argwhere(g == Color.MAROON)]
    assert len(endpoints) == 2

    simulated = _simulate_rays(g)
    forbidden = _forbidden_from_yellow(simulated)
    passable = _passable_after_forbidden(g, forbidden)
    assert endpoints[0] in passable and endpoints[1] in passable
    assert _simple_path_count(passable, endpoints[0], endpoints[1], limit=2) == 1
    path = _unique_simple_path(passable, endpoints[0], endpoints[1])

    out = simulated.copy()
    for r, c in path:
        if int(g[r, c]) == Color.TEAL:
            out[r, c] = Color.MAROON
    return out


def _vertical_module_cells(top: int, center: int, half_width: int) -> set[Cell]:
    bottom = top + 4
    left = center - half_width
    right = center + half_width
    cells = set()
    for c in range(left, right + 1):
        cells.add((top, c))
        cells.add((bottom, c))
    for r in range(top, bottom + 1):
        cells.add((r, left))
        cells.add((r, right))
    return cells


def _paint_track(grid: np.ndarray, cells: set[Cell], endpoints: tuple[Cell, Cell]) -> None:
    for r, c in cells:
        grid[r, c] = Color.TEAL
    for r, c in endpoints:
        grid[r, c] = Color.MAROON


def _make_candidate(rng: random.Random) -> np.ndarray:
    module_count = rng.choice([1, 2])
    half_width = rng.choice([2, 3])
    center = rng.randint(half_width + 5, half_width + 8)
    top_margin = rng.randint(2, 4)
    spacings = [rng.randint(2, 4) for _ in range(module_count - 1)]
    bottom_margin = rng.randint(4, 6)
    module_tops = [top_margin]
    for gap in spacings:
        module_tops.append(module_tops[-1] + 4 + gap)
    height = module_tops[-1] + 4 + bottom_margin
    width = center + half_width + rng.randint(3, 6)

    assert height <= 30 and width <= 30
    grid = np.zeros((height, width), dtype=int)

    track: set[Cell] = set()
    top_endpoint = (0, center)
    bottom_endpoint = (height - 1, center)
    track.add(top_endpoint)
    track.add(bottom_endpoint)

    for r in range(1, module_tops[0]):
        track.add((r, center))
    for i, top in enumerate(module_tops):
        track |= _vertical_module_cells(top, center, half_width)
        bottom = top + 4
        next_top = module_tops[i + 1] if i + 1 < module_count else None
        if next_top is None:
            for r in range(bottom + 1, height - 1):
                track.add((r, center))
        else:
            for r in range(bottom + 1, next_top):
                track.add((r, center))

    _paint_track(grid, track, (top_endpoint, bottom_endpoint))

    seeds: list[Cell] = []
    real_obstacles: list[Cell] = []
    wake_blocked: set[Cell] = set()
    for top in module_tops:
        seed_r = top + 2
        left_col = center - half_width
        obstacle = (seed_r, left_col - 1)
        seed = (seed_r, 0)
        seeds.append(seed)
        real_obstacles.append(obstacle)
        grid[seed] = Color.YELLOW
        grid[obstacle] = Color.GREY
        wake_blocked.add((seed_r - 1, obstacle[1]))
        wake_blocked.add((seed_r + 1, obstacle[1]))

    obstacle_target = rng.randint(max(3, module_count + 1), 5)
    extras_needed = obstacle_target - len(real_obstacles)
    occupied = track | set(seeds) | set(real_obstacles) | wake_blocked
    extra_obstacles: set[Cell] = set()
    attempts = 0
    while len(extra_obstacles) < extras_needed and attempts < 500:
        attempts += 1
        r = rng.randrange(height)
        c = rng.randrange(1, width)
        cell = (r, c)
        if cell in occupied or cell in extra_obstacles:
            continue
        if any(r == sr and c < oc for (sr, _), (_, oc) in zip(seeds, real_obstacles)):
            continue
        if grid[cell] != Color.BLACK:
            continue
        if any(abs(r - tr) <= 1 and abs(c - tc) <= 1 for tr, tc in track):
            continue
        extra_obstacles.add(cell)
    if len(extra_obstacles) != extras_needed:
        raise AssertionError("failed to place distractor obstacles")
    for cell in extra_obstacles:
        grid[cell] = Color.GREY

    return grid


def _validate_candidate(grid: np.ndarray) -> bool:
    endpoints = [tuple(map(int, p)) for p in np.argwhere(grid == Color.MAROON)]
    if len(endpoints) != 2:
        return False

    track = _track_cells(grid)
    simple_before = _simple_path_count(track, endpoints[0], endpoints[1], limit=2)
    if simple_before < 2:
        return False
    if _shortest_path_multiplicity(track, endpoints[0], endpoints[1], cap=3) < 2:
        return False

    simulated = _simulate_rays(grid)
    original_track = (grid == Color.TEAL) | (grid == Color.MAROON)
    if np.any((simulated == Color.YELLOW) & original_track):
        return False
    if np.any((grid == Color.GREY) & original_track):
        return False

    forbidden = _forbidden_from_yellow(simulated)
    if any(forbidden[r, c] for r, c in endpoints):
        return False
    passable = _passable_after_forbidden(grid, forbidden)
    if _simple_path_count(passable, endpoints[0], endpoints[1], limit=2) != 1:
        return False
    path = _unique_simple_path(passable, endpoints[0], endpoints[1])
    if sum(int(grid[r, c] == Color.TEAL) for r, c in path) < 6:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    recolored = (out == Color.MAROON) & (grid == Color.TEAL)
    if int(recolored.sum()) != sum(int(grid[r, c] == Color.TEAL) for r, c in path):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random(random.randrange(1 << 30))
        grid = _make_candidate(rng)
        if _validate_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
