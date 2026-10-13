import heapq
from collections import Counter, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
WALL = 5
INF = 10**9
GEN_ATTEMPTS = 300


def get_bg(grid: np.ndarray) -> int:
    """Reference-compatible background heuristic: most common color."""
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    """Reference-compatible 4-neighborhood in (r, c) order."""
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _endpoint_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    assert len(pts) == 2, "expected exactly two red endpoints"
    pts.sort()
    assert pts[0][0] == 0 and pts[1][0] == grid.shape[0] - 1, "endpoints must lie on top/bottom borders"
    return pts


def _candidate_axes(width: int) -> list[int]:
    center = width - 1
    order = sorted(range(2 * width - 1), key=lambda k: (abs(k - center), k))
    return order


def _paint_output(grid: np.ndarray, orbit_cells: list[list[tuple[int, int]]], chosen_orbits: set[int]) -> np.ndarray:
    out = grid.copy()
    for idx in chosen_orbits:
        for r, c in orbit_cells[idx]:
            out[r, c] = BLUE
    return out


def _blue_connects_endpoints(grid: np.ndarray) -> bool:
    top, bottom = _endpoint_cells(grid)
    blue = grid == BLUE
    start = [rc for rc in neighbors4(*top) if in_bounds(grid.shape, *rc) and blue[rc]]
    goals = {rc for rc in neighbors4(*bottom) if in_bounds(grid.shape, *rc) and blue[rc]}
    if not start or not goals:
        return False
    q = deque(start)
    seen = set(start)
    while q:
        cell = q.popleft()
        if cell in goals:
            return True
        for nxt in neighbors4(*cell):
            if in_bounds(grid.shape, *nxt) and blue[nxt] and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return False


def _is_symmetric_blue(grid: np.ndarray, axis_k: int) -> bool:
    h, w = grid.shape
    for r, c in np.argwhere(grid == BLUE):
        mc = axis_k - int(c)
        if not (0 <= mc < w) or grid[int(r), mc] != BLUE:
            return False
    return True


def _build_axis_problem(grid: np.ndarray, axis_k: int):
    h, w = grid.shape
    endpoints = _endpoint_cells(grid)
    endpoint_set = set(endpoints)
    blue_cells = [tuple(map(int, rc)) for rc in np.argwhere(grid == BLUE)]

    orbit_cells: list[list[tuple[int, int]]] = []
    orbit_costs: list[int] = []
    mandatory_orbits: set[int] = set()
    cell_to_orbit: dict[tuple[int, int], int] = {}

    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if cell in cell_to_orbit or cell in endpoint_set or grid[cell] == WALL:
                continue
            mc = axis_k - c
            if not (0 <= mc < w):
                if grid[cell] == BLUE:
                    return None
                continue
            mirror = (r, int(mc))
            if mirror in endpoint_set or grid[mirror] == WALL:
                if grid[cell] == BLUE or grid[mirror] == BLUE:
                    return None
                continue
            cells = [cell] if mirror == cell else sorted([cell, mirror])
            idx = len(orbit_cells)
            orbit_cells.append(cells)
            orbit_costs.append(sum(1 for rc in cells if grid[rc] == BG))
            for rc in cells:
                cell_to_orbit[rc] = idx
            if any(grid[rc] == BLUE for rc in cells):
                mandatory_orbits.add(idx)

    if any(cell not in cell_to_orbit for cell in blue_cells):
        return None

    endpoint_nodes = [len(orbit_cells), len(orbit_cells) + 1]
    node_costs = orbit_costs + [0, 0]
    adjacency = [set() for _ in range(len(node_costs))]
    endpoint_node_of = {endpoints[0]: endpoint_nodes[0], endpoints[1]: endpoint_nodes[1]}

    for idx, cells in enumerate(orbit_cells):
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if not in_bounds(grid.shape, nr, nc):
                    continue
                other = (nr, nc)
                if other in cell_to_orbit:
                    j = cell_to_orbit[other]
                    if j != idx:
                        adjacency[idx].add(j)
                        adjacency[j].add(idx)
                elif other in endpoint_node_of:
                    ep = endpoint_node_of[other]
                    adjacency[idx].add(ep)
                    adjacency[ep].add(idx)

    terminals = endpoint_nodes + sorted(mandatory_orbits)
    return {
        "axis_k": axis_k,
        "orbit_cells": orbit_cells,
        "node_costs": node_costs,
        "adjacency": [sorted(ns) for ns in adjacency],
        "terminals": terminals,
    }


def _steiner_tree(node_costs: list[int], adjacency: list[list[int]], terminals: list[int]):
    t = len(terminals)
    n = len(node_costs)
    full = (1 << t) - 1
    dist = np.full((1 << t, n), INF, dtype=int)
    parent: list[list[tuple | None]] = [[None] * n for _ in range(1 << t)]

    for i, node in enumerate(terminals):
        mask = 1 << i
        dist[mask, node] = node_costs[node]
        parent[mask][node] = ("start",)

    for mask in range(1, full + 1):
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if sub < other:
                merged = dist[sub] + dist[other] - np.array(node_costs, dtype=int)
                better = merged < dist[mask]
                if np.any(better):
                    idxs = np.where(better)[0]
                    dist[mask, idxs] = merged[idxs]
                    for v in idxs:
                        parent[mask][int(v)] = ("merge", int(sub), int(other))
            sub = (sub - 1) & mask

        pq = [(int(dist[mask, v]), int(v)) for v in range(n) if dist[mask, v] < INF]
        heapq.heapify(pq)
        while pq:
            cur, v = heapq.heappop(pq)
            if cur != int(dist[mask, v]):
                continue
            for nxt in adjacency[v]:
                nd = cur + node_costs[nxt]
                if nd < int(dist[mask, nxt]):
                    dist[mask, nxt] = nd
                    parent[mask][nxt] = ("move", int(v))
                    heapq.heappush(pq, (nd, nxt))

    best_cost = int(dist[full].min())
    if best_cost >= INF:
        return None
    best_node = int(np.argmin(dist[full]))

    chosen: set[int] = set()

    def collect(mask: int, node: int):
        chosen.add(node)
        info = parent[mask][node]
        assert info is not None
        kind = info[0]
        if kind == "start":
            return
        if kind == "move":
            collect(mask, info[1])
            return
        if kind == "merge":
            collect(info[1], node)
            collect(info[2], node)
            return
        raise AssertionError(f"unknown parent kind: {kind}")

    collect(full, best_node)
    return {"cost": best_cost, "chosen_nodes": chosen}


def _solve_for_axis(grid: np.ndarray, axis_k: int):
    problem = _build_axis_problem(grid, axis_k)
    if problem is None:
        return None
    solved = _steiner_tree(problem["node_costs"], problem["adjacency"], problem["terminals"])
    if solved is None:
        return None

    orbit_count = len(problem["orbit_cells"])
    chosen_orbits = {node for node in solved["chosen_nodes"] if node < orbit_count}
    out = _paint_output(grid, problem["orbit_cells"], chosen_orbits)
    if not _is_symmetric_blue(out, axis_k):
        return None
    if not _blue_connects_endpoints(out):
        return None
    return {"axis_k": axis_k, "cost": solved["cost"], "output": out}


def _solve_candidates(grid: np.ndarray):
    h, w = grid.shape
    assert h <= 30 and w <= 30
    assert get_bg(grid) in {BG, WALL}
    assert set(np.unique(grid)).issubset({BG, BLUE, RED, WALL})

    candidates = []
    for axis_k in _candidate_axes(w):
        solved = _solve_for_axis(grid, axis_k)
        if solved is not None:
            candidates.append(solved)
    if not candidates:
        raise AssertionError("no valid symmetric-connected completion found")
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add the fewest blue pixels needed to make the partial pattern vertically mirror-symmetric and to connect the two red endpoints.

    Input structure:
    - A black background contains gray wall pixels that form thin orthogonal barriers and leave a passable corridor network.
    - Two red endpoint pixels lie on the top and bottom borders, and some blue pixels mark an incomplete pattern inside the passable area.
    - The blue pixels suggest a vertical mirror axis, but the current pattern is not yet symmetric and does not yet connect the endpoints.

    Rule:
    - Keep the gray walls and red endpoints unchanged.
    - Choose a vertical mirror axis and add blue pixels only on passable black cells so that every blue pixel has its mirror mate across that axis.
    - Among all mirror-symmetric blue completions whose blue cells form a 4-connected route between the two red endpoints, return the one that adds the fewest pixels.

    Key insight:
    - The problem is a simultaneous constraint satisfaction task: treat mirrored cell pairs as indivisible orbit nodes, then find the minimum-cost connected blue structure that includes the existing blue clues and touches both endpoints.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    candidates = _solve_candidates(g)
    best_cost = min(item["cost"] for item in candidates)
    best = [item for item in candidates if item["cost"] == best_cost]
    outputs = {item["output"].tobytes(): item["output"] for item in best}
    assert len(outputs) == 1, "generator is expected to provide a unique minimal output"
    return next(iter(outputs.values()))


def _choose_gaps(height: int, gap_count: int) -> list[tuple[int, int]]:
    if gap_count == 1:
        start = np.random.randint(3, height - 5)
        end = start + np.random.randint(1, min(4, height - start - 3))
        return [(start, end)]

    while True:
        first_start = int(np.random.randint(3, height - 7))
        first_len = int(np.random.randint(1, 3))
        first_end = first_start + first_len - 1
        second_low = first_end + 4
        second_high = height - 4
        if second_low > second_high:
            continue
        second_start = int(np.random.randint(second_low, second_high + 1))
        second_len_max = min(2, height - 3 - second_start)
        if second_len_max < 1:
            continue
        second_len = int(np.random.randint(1, second_len_max + 1))
        second_end = second_start + second_len - 1
        gaps = [(first_start, first_end), (second_start, second_end)]
        if second_end <= height - 3:
            return gaps


def _build_corridor_grid() -> tuple[np.ndarray, int, int, list[tuple[int, int]]]:
    outer_d = int(np.random.choice([2, 3]))
    margin = int(np.random.choice([1, 2]))
    width = 2 * (outer_d + margin) + 1
    height = int(np.random.randint(11, 18))
    axis_c = width // 2
    gap_count = int(np.random.choice([1, 2])) if height >= 14 else 1
    gaps = _choose_gaps(height, gap_count)

    grid = np.full((height, width), WALL, dtype=int)
    grid[0, axis_c] = RED
    grid[-1, axis_c] = RED

    gap_rows = {r for start, end in gaps for r in range(start, end + 1)}
    for r in range(1, height - 1):
        if r not in gap_rows:
            grid[r, axis_c] = BG
        grid[r, axis_c - outer_d] = BG
        grid[r, axis_c + outer_d] = BG

    for start, end in gaps:
        for row in (start - 1, end + 1):
            grid[row, axis_c - outer_d : axis_c + outer_d + 1] = BG

    return grid, axis_c, outer_d, gaps


def _route_sample_rows(height: int, gaps: list[tuple[int, int]]) -> list[int]:
    rows = []
    first_start, first_end = gaps[0]
    rows.append(np.random.randint(1, first_start))
    for start, end in gaps:
        rows.append(np.random.randint(start, end + 1))
    last_start, last_end = gaps[-1]
    rows.append(np.random.randint(last_end + 1, height - 1))
    if len(gaps) == 2:
        rows.append(np.random.randint(gaps[0][1] + 1, gaps[1][0]))
    return sorted(set(rows))


def _add_partial_blue(grid: np.ndarray, axis_c: int, outer_d: int, gaps: list[tuple[int, int]]):
    height = grid.shape[0]
    side = int(np.random.choice([-1, 1]))
    outer_col = axis_c + side * outer_d

    for row in _route_sample_rows(height, gaps):
        if any(start <= row <= end for start, end in gaps):
            grid[row, outer_col] = BLUE
        else:
            grid[row, axis_c] = BLUE

    for start, end in gaps:
        row = (start + end) // 2
        grid[row, outer_col] = BLUE


def _unique_best_solution(grid: np.ndarray) -> bool:
    candidates = _solve_candidates(grid)
    best_cost = min(item["cost"] for item in candidates)
    best = [item for item in candidates if item["cost"] == best_cost]
    outputs = {item["output"].tobytes() for item in best}
    return len(outputs) == 1


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid, axis_c, outer_d, gaps = _build_corridor_grid()
        _add_partial_blue(grid, axis_c, outer_d, gaps)

        if np.count_nonzero(grid == BLUE) < 4:
            continue
        if all(grid[r, axis_c] == BLUE for r in range(1, grid.shape[0] - 1)):
            continue
        if not _unique_best_solution(grid):
            continue

        out = main(grid)
        added = int(np.count_nonzero(out == BLUE) - np.count_nonzero(grid == BLUE))
        if added < 6:
            continue
        if not _blue_connects_endpoints(out):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
