import random
from collections import Counter, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
INPUT_COLORS = [5, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (13, 24)
GRID_W_RANGE = (13, 24)
MAP_H_RANGE = (8, 18)
MAP_W_RANGE = (8, 18)
REGION_COUNT_RANGE = (6, 11)
SEED_GAP = 3
MIN_REGION_AREA = 6
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _dominant_core_color(grid: np.ndarray, marker: tuple[int, int]) -> int:
    r, c = marker
    around = []
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            color = int(grid[nr, nc])
            if color not in (BG, YELLOW):
                around.append(color)
    if not around:
        for nr in range(max(0, r - 1), min(grid.shape[0], r + 2)):
            for nc in range(max(0, c - 1), min(grid.shape[1], c + 2)):
                color = int(grid[nr, nc])
                if color not in (BG, YELLOW):
                    around.append(color)
    assert around, "yellow marker must sit inside a colored region"
    return Counter(around).most_common(1)[0][0]


def _extract_regions(grid: np.ndarray):
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    regions = []
    next_id = 0
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or labels[r, c] != -1:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_id
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == -1 and int(grid[nr, nc]) == color:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            regions.append({"color": color, "cells": cells})
            next_id += 1
    return labels, regions


def _region_graph(labels: np.ndarray, region_count: int):
    h, w = labels.shape
    graph = [set() for _ in range(region_count)]
    for r in range(h):
        for c in range(w):
            a = int(labels[r, c])
            if a < 0:
                continue
            for nr, nc in ((r + 1, c), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w:
                    b = int(labels[nr, nc])
                    if b >= 0 and b != a:
                        graph[a].add(b)
                        graph[b].add(a)
    return graph


def _graph_distances(graph: list[set[int]], start: int) -> dict[int, int]:
    dist = {start: 0}
    q = deque([start])
    while q:
        node = q.popleft()
        for nxt in graph[node]:
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                q.append(nxt)
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor regions by graph distance from the yellow-marked core region.

    Distance 1 -> red, distance 2 -> blue, distance 3 -> green.
    The core keeps its original color and the yellow marker is removed.
    Regions at distance >= 4 remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    marker_cells = np.argwhere(g == YELLOW)
    assert marker_cells.shape[0] == 1, "expected a unique yellow marker"
    marker = tuple(int(v) for v in marker_cells[0])

    patched = g.copy()
    core_color = _dominant_core_color(g, marker)
    patched[marker] = core_color

    labels, regions = _extract_regions(patched)
    core_id = int(labels[marker])
    assert core_id >= 0, "yellow marker must belong to the core region"

    graph = _region_graph(labels, len(regions))
    dist = _graph_distances(graph, core_id)

    out = patched.copy()
    for region_id, region in enumerate(regions):
        d = dist.get(region_id)
        if d == 1:
            new_color = RED
        elif d == 2:
            new_color = BLUE
        elif d == 3:
            new_color = GREEN
        else:
            continue
        rr, cc = zip(*region["cells"])
        out[np.array(rr), np.array(cc)] = new_color
    return out


def _sample_seeds(map_h: int, map_w: int, count: int, rng: random.Random):
    cells = [(r, c) for r in range(map_h) for c in range(map_w)]
    rng.shuffle(cells)
    seeds = []
    for r, c in cells:
        if all(abs(r - sr) + abs(c - sc) >= SEED_GAP for sr, sc in seeds):
            seeds.append((r, c))
            if len(seeds) == count:
                return seeds
    return None


def _grow_partition(map_h: int, map_w: int, seeds: list[tuple[int, int]], rng: random.Random):
    owners = np.full((map_h, map_w), -1, dtype=int)
    q = deque()
    order = list(range(len(seeds)))
    rng.shuffle(order)
    for idx in order:
        r, c = seeds[idx]
        owners[r, c] = idx
        q.append((r, c, idx))
    while q:
        r, c, idx = q.popleft()
        nbrs = list(_neighbors4(r, c))
        rng.shuffle(nbrs)
        for nr, nc in nbrs:
            if 0 <= nr < map_h and 0 <= nc < map_w and owners[nr, nc] == -1:
                owners[nr, nc] = idx
                q.append((nr, nc, idx))
    return owners


def _region_cells_from_owners(owners: np.ndarray, count: int):
    regions = [[] for _ in range(count)]
    for r in range(owners.shape[0]):
        for c in range(owners.shape[1]):
            regions[int(owners[r, c])].append((r, c))
    return regions


def _interior_cells(cells: list[tuple[int, int]], owner_set: set[tuple[int, int]]):
    interior = []
    for r, c in cells:
        if all((r + dr, c + dc) in owner_set for dr, dc in DIRS4):
            interior.append((r, c))
    return interior


def _color_graph(graph: list[set[int]], rng: random.Random):
    order = sorted(range(len(graph)), key=lambda i: (-len(graph[i]), rng.random()))
    assignment = {}

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        node = order[pos]
        choices = INPUT_COLORS[:]
        rng.shuffle(choices)
        for color in choices:
            if all(assignment.get(nei) != color for nei in graph[node]):
                assignment[node] = color
                if backtrack(pos + 1):
                    return True
                del assignment[node]
        return False

    if not backtrack(0):
        return None
    return assignment


def _make_attempt(rng: random.Random):
    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(*GRID_W_RANGE)

    map_h = rng.randint(MAP_H_RANGE[0], min(MAP_H_RANGE[1], grid_h - 2))
    map_w = rng.randint(MAP_W_RANGE[0], min(MAP_W_RANGE[1], grid_w - 2))
    top = rng.randint(1, grid_h - map_h - 1)
    left = rng.randint(1, grid_w - map_w - 1)

    area = map_h * map_w
    max_regions = min(REGION_COUNT_RANGE[1], area // MIN_REGION_AREA)
    if max_regions < REGION_COUNT_RANGE[0]:
        return None
    region_count = rng.randint(REGION_COUNT_RANGE[0], max_regions)

    seeds = _sample_seeds(map_h, map_w, region_count, rng)
    if seeds is None:
        return None
    owners = _grow_partition(map_h, map_w, seeds, rng)
    regions = _region_cells_from_owners(owners, region_count)
    if min(len(cells) for cells in regions) < MIN_REGION_AREA:
        return None

    graph = _region_graph(owners, region_count)
    coloring = _color_graph(graph, rng)
    if coloring is None:
        return None

    target_far = rng.random() < 0.5
    candidates = []
    for region_id, cells in enumerate(regions):
        cell_set = set(cells)
        interior = _interior_cells(cells, cell_set)
        if not interior:
            continue
        dist = _graph_distances(graph, region_id)
        if len(dist) != region_count:
            continue
        max_dist = max(dist.values())
        counts = Counter(dist.values())
        if not all(counts.get(layer, 0) > 0 for layer in (1, 2, 3)):
            continue
        if target_far and max_dist < 4:
            continue
        if not target_far and max_dist != 3:
            continue
        candidates.append((region_id, interior, dist))
    if not candidates:
        return None

    core_id, interior_cells, dist = rng.choice(candidates)
    marker_r, marker_c = rng.choice(interior_cells)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for r in range(map_h):
        for c in range(map_w):
            region_id = int(owners[r, c])
            grid[top + r, left + c] = coloring[region_id]
    grid[top + marker_r, left + marker_c] = YELLOW

    out = main(grid)
    core_color = coloring[core_id]
    if int(out[top + marker_r, left + marker_c]) != core_color:
        return None
    seen_layers = {int(out[top + r, left + c]) for region_id, cells in enumerate(regions) for r, c in cells if dist[region_id] in (1, 2, 3)}
    if not {RED, BLUE, GREEN}.issubset(seen_layers):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _make_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
