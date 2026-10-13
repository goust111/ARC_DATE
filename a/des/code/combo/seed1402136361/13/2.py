from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
ALL_COLORS = list(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[Coord]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _constellations(grid: np.ndarray, bg: int) -> dict[int, list[Coord]]:
    groups: dict[int, list[Coord]] = {}
    for r_raw, c_raw in np.argwhere(grid != bg):
        r = int(r_raw)
        c = int(c_raw)
        color = int(grid[r, c])
        groups.setdefault(color, []).append((r, c))
    for color in groups:
        groups[color].sort()
    return groups


def _constellation_sort_key(item: tuple[int, list[Coord]]) -> tuple[object, ...]:
    _, points = item
    return (len(points), points[0], tuple(points))


def _mst_edges(points: list[Coord]) -> list[tuple[Coord, Coord]]:
    if len(points) < 2:
        return []

    edges: list[tuple[int, Coord, Coord, int, int]] = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            a = points[i]
            b = points[j]
            p, q = sorted((a, b))
            dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
            edges.append((dist, p, q, i, j))
    edges.sort(key=lambda item: (item[0], item[1], item[2]))

    dsu = _DSU(len(points))
    chosen: list[tuple[Coord, Coord]] = []
    for _, p, q, i, j in edges:
        if dsu.union(i, j):
            chosen.append((p, q))
            if len(chosen) == len(points) - 1:
                break
    return chosen


def _horizontal_then_vertical_path(a: Coord, b: Coord) -> list[Coord]:
    start, end = sorted((a, b))
    r, c = start
    tr, tc = end
    path = [(r, c)]

    step_c = 0 if tc == c else (1 if tc > c else -1)
    while c != tc:
        c += step_c
        path.append((r, c))

    step_r = 0 if tr == r else (1 if tr > r else -1)
    while r != tr:
        r += step_r
        path.append((r, c))
    return path


def _solve(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    groups = _constellations(g, bg)
    ordered = sorted(groups.items(), key=_constellation_sort_key)

    out = g.copy()
    meta = {
        "num_colors": len(ordered),
        "edge_count": 0,
        "bent_edge_count": 0,
        "straight_edge_count": 0,
        "painted_cells": 0,
        "blocked_foreign": 0,
        "reused_self": 0,
        "size_tie": len({len(points) for _, points in ordered}) < len(ordered),
    }

    for color, points in ordered:
        for a, b in _mst_edges(points):
            meta["edge_count"] += 1
            if a[0] != b[0] and a[1] != b[1]:
                meta["bent_edge_count"] += 1
            else:
                meta["straight_edge_count"] += 1

            path = _horizontal_then_vertical_path(a, b)
            for idx, (r, c) in enumerate(path):
                is_endpoint = idx == 0 or idx == len(path) - 1
                if out[r, c] == bg:
                    out[r, c] = color
                    if not is_endpoint:
                        meta["painted_cells"] += 1
                elif not is_endpoint:
                    if int(out[r, c]) == color:
                        meta["reused_self"] += 1
                    else:
                        meta["blocked_foreign"] += 1

    if return_meta:
        return out, meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group isolated same-colored pixels into constellations and connect each constellation
    with horizontal-then-vertical MST edges, painting only background cells.
    """
    return _solve(grid)


def _sample_counts(scenario: str, num_colors: int) -> list[int]:
    while True:
        counts = [random.randint(2, 5) for _ in range(num_colors)]
        if max(counts) < 3:
            continue
        if scenario == "tie" and len(set(counts)) == len(counts):
            continue
        if scenario != "tie" and random.random() < 0.7 and len(set(counts)) != len(counts):
            continue
        return counts


def _sample_shape(scenario: str) -> tuple[int, int, int]:
    if scenario == "blocked":
        h = random.randint(9, 12)
        w = random.randint(9, 12)
        num_colors = random.randint(3, 4)
    elif scenario == "tie":
        h = random.randint(10, 15)
        w = random.randint(10, 15)
        num_colors = random.randint(3, 4)
    else:
        h = random.randint(11, 17)
        w = random.randint(11, 17)
        num_colors = random.randint(2, 4)
    return h, w, num_colors


def _biased_coord(h: int, w: int, scenario: str) -> Coord:
    if scenario == "blocked" and h > 4 and w > 4 and random.random() < 0.8:
        return (random.randint(1, h - 2), random.randint(1, w - 2))
    return (random.randrange(h), random.randrange(w))


def _can_place(grid: np.ndarray, bg: int, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    if int(grid[r, c]) != bg:
        return False

    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != bg:
            return False

    for nr, nc in _neighbors8(r, c):
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == color:
            return False
    return True


def _place_points(
    grid: np.ndarray,
    bg: int,
    color: int,
    count: int,
    scenario: str,
) -> bool:
    attempts = 0
    placed = 0
    while placed < count and attempts < 500:
        attempts += 1
        r, c = _biased_coord(*grid.shape, scenario)
        if not _can_place(grid, bg, r, c, color):
            continue
        grid[r, c] = color
        placed += 1
    return placed == count


def _basic_input_checks(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(int(v) for v in grid.flat)
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return False

    h, w = grid.shape
    for r_raw, c_raw in np.argwhere(grid != bg):
        r = int(r_raw)
        c = int(c_raw)
        color = int(grid[r, c])
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) != (r, c):
                if int(grid[nr, nc]) == color:
                    return False
    return True


def _scenario_accepts(scenario: str, meta: dict[str, int | bool]) -> bool:
    if meta["num_colors"] < 2:
        return False
    if meta["edge_count"] < 2:
        return False
    if meta["painted_cells"] < 3:
        return False
    if meta["bent_edge_count"] < 1:
        return False

    if scenario == "clean":
        return meta["blocked_foreign"] == 0
    if scenario == "blocked":
        return meta["blocked_foreign"] >= 1
    if scenario == "tie":
        return bool(meta["size_tie"])
    return False


def _gen_attempt(scenario: str) -> np.ndarray | None:
    h, w, num_colors = _sample_shape(scenario)
    bg = random.choice(ALL_COLORS)
    palette = [color for color in ALL_COLORS if color != bg]
    random.shuffle(palette)
    colors = palette[:num_colors]
    counts = _sample_counts(scenario, num_colors)

    grid = np.full((h, w), bg, dtype=int)
    color_specs = list(zip(colors, counts))
    color_specs.sort(key=lambda item: item[1], reverse=True)
    for color, count in color_specs:
        if not _place_points(grid, bg, color, count, scenario):
            return None

    if not _basic_input_checks(grid, bg):
        return None

    out, meta = _solve(grid, return_meta=True)
    if np.array_equal(grid, out):
        return None
    if not _scenario_accepts(scenario, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    scenarios = ["clean", "blocked", "tie"]
    for _ in range(MAX_GENERATION_ATTEMPTS):
        scenario = random.choice(scenarios)
        grid = _gen_attempt(scenario)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid constellation input")
