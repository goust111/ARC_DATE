from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GRAY = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

ENDPOINT_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
PATH_RETRIES = 80


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _degree_grid(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    deg = np.zeros_like(mask, dtype=int)
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]):
                continue
            deg[r, c] = sum(int(mask[nr, nc]) for nr, nc in _neighbors4(r, c, h, w))
    return deg


def _trace_component_path(
    comp: list[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    comp_set = set(comp)
    path = [start]
    prev: tuple[int, int] | None = None
    curr = start
    while curr != end:
        nxt = [pt for pt in _neighbors4(curr[0], curr[1], h, w) if pt in comp_set and pt != prev]
        assert len(nxt) == 1
        prev, curr = curr, nxt[0]
        path.append(curr)
        assert len(path) <= len(comp_set)
    assert len(path) == len(comp_set)
    return path


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mask = g != BG
    deg = _degree_grid(mask)
    out = g.copy()

    for comp in _components(mask):
        endpoints = [pt for pt in comp if int(deg[pt[0], pt[1]]) == 1]
        assert len(endpoints) == 2
        a, b = sorted(endpoints)
        assert int(g[a[0], a[1]]) == int(g[b[0], b[1]])

        path = _trace_component_path(comp, a, b, h, w)
        if len(path) <= 2:
            continue

        internal_colors = [int(g[r, c]) for r, c in path[1:-1]]
        if not internal_colors:
            continue
        gray = int(Counter(internal_colors).most_common(1)[0][0])

        stations = [idx for idx in range(1, len(path) - 1) if int(g[path[idx][0], path[idx][1]]) != gray]
        if not stations:
            continue

        prev_idx = 0
        last_station_color = int(g[path[stations[-1]][0], path[stations[-1]][1]])
        for station_idx in stations:
            station_color = int(g[path[station_idx][0], path[station_idx][1]])
            for idx in range(prev_idx + 1, station_idx):
                r, c = path[idx]
                if int(g[r, c]) == gray:
                    out[r, c] = station_color
            prev_idx = station_idx
        for idx in range(prev_idx + 1, len(path) - 1):
            r, c = path[idx]
            if int(g[r, c]) == gray:
                out[r, c] = last_station_color

    return out


def _valid_extension(
    pt: tuple[int, int],
    prev: tuple[int, int] | None,
    occupied: np.ndarray,
    path_set: set[tuple[int, int]],
) -> bool:
    r, c = pt
    h, w = occupied.shape
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if bool(occupied[r, c]) or pt in path_set:
        return False
    for nr, nc in _neighbors4(r, c, h, w):
        if prev is not None and (nr, nc) == prev:
            continue
        if bool(occupied[nr, nc]) or (nr, nc) in path_set:
            return False
    return True


def _future_degree(
    pt: tuple[int, int],
    occupied: np.ndarray,
    path_set: set[tuple[int, int]],
    prev: tuple[int, int],
) -> int:
    h, w = occupied.shape
    score = 0
    for nr, nc in _neighbors4(pt[0], pt[1], h, w):
        if _valid_extension((nr, nc), pt, occupied, path_set | {pt}):
            score += 1
    return score


def _random_simple_path(
    h: int,
    w: int,
    occupied: np.ndarray,
    total_nodes: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    starts = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if _valid_extension((r, c), None, occupied, set())
    ]
    if not starts:
        return None
    rng.shuffle(starts)

    def dfs(path: list[tuple[int, int]], path_set: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(path) == total_nodes:
            return path.copy()
        curr = path[-1]
        candidates = []
        for nr, nc in _neighbors4(curr[0], curr[1], h, w):
            nxt = (nr, nc)
            if _valid_extension(nxt, curr, occupied, path_set):
                candidates.append(nxt)
        rng.shuffle(candidates)
        candidates.sort(key=lambda pt: _future_degree(pt, occupied, path_set, curr), reverse=True)
        for nxt in candidates:
            path.append(nxt)
            path_set.add(nxt)
            result = dfs(path, path_set)
            if result is not None:
                return result
            path_set.remove(nxt)
            path.pop()
        return None

    for start in starts[:PATH_RETRIES]:
        result = dfs([start], {start})
        if result is not None:
            return result
    return None


def _station_indices(segment_lengths: list[int]) -> list[int]:
    indices: list[int] = []
    pos = 0
    for seg_len in segment_lengths[:-1]:
        pos += seg_len + 1
        indices.append(pos)
    return indices


def _paint_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    endpoint_color: int,
    station_colors: list[int],
    segment_lengths: list[int],
) -> None:
    for idx, (r, c) in enumerate(path):
        if idx == 0 or idx == len(path) - 1:
            grid[r, c] = endpoint_color
        else:
            grid[r, c] = GRAY
    for idx, color in zip(_station_indices(segment_lengths), station_colors):
        r, c = path[idx]
        grid[r, c] = color


def _count_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    num_paths = rng.choice([2, 3])
    h = rng.randint(14, 22)
    w = rng.randint(14, 22)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    endpoint_colors = rng.sample(ENDPOINT_PALETTE, num_paths)
    relay_pool = [c for c in ENDPOINT_PALETTE if c not in endpoint_colors]

    station_counts = [rng.choice([1, 2, 2, 3]) for _ in range(num_paths)]
    if all(count == 1 for count in station_counts):
        station_counts[rng.randrange(num_paths)] = 2

    complex_paths = 0
    turnful_paths = 0

    for endpoint_color, station_count in zip(endpoint_colors, station_counts):
        segment_lengths = [rng.randint(2, 4) for _ in range(station_count + 1)]
        total_nodes = 2 + station_count + sum(segment_lengths)
        path = _random_simple_path(h, w, occupied, total_nodes, rng)
        if path is None:
            return None
        if path[0] > path[-1]:
            path = list(reversed(path))

        station_colors = rng.sample(relay_pool, station_count)
        _paint_path(grid, path, endpoint_color, station_colors, segment_lengths)
        for r, c in path:
            occupied[r, c] = True

        if station_count >= 2:
            complex_paths += 1
        if _count_turns(path) >= 1:
            turnful_paths += 1

    if complex_paths == 0 or turnful_paths == 0:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
