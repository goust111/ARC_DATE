from __future__ import annotations

from collections import deque
import random

import numpy as np


BG_COLOR = 0
WALL_COLOR = 2
SEED_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]

GEN_RETRY_LIMIT = 400
CELL_SIZE_OPTIONS = [3, 4]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_open_components(grid: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    h, w = grid.shape
    comp_id = np.full((h, w), -1, dtype=int)
    components: list[list[tuple[int, int]]] = []
    seeds_by_comp: list[list[tuple[int, int]]] = []

    next_id = 0
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == WALL_COLOR or int(comp_id[r, c]) != -1:
                continue

            queue = deque([(r, c)])
            comp_id[r, c] = next_id
            cells: list[tuple[int, int]] = []
            seeds: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                if int(grid[cr, cc]) not in (BG_COLOR, WALL_COLOR):
                    seeds.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != WALL_COLOR and int(comp_id[nr, nc]) == -1:
                        comp_id[nr, nc] = next_id
                        queue.append((nr, nc))

            components.append(cells)
            seeds_by_comp.append(seeds)
            next_id += 1

    return comp_id, components, seeds_by_comp


def _region_adjacency(
    grid: np.ndarray,
    comp_id: np.ndarray,
    valid_regions: set[int],
) -> dict[int, set[int]]:
    h, w = grid.shape
    adjacency = {region_id: set() for region_id in valid_regions}

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != WALL_COLOR:
                continue

            left = int(comp_id[r, c - 1]) if c - 1 >= 0 and int(grid[r, c - 1]) != WALL_COLOR else -1
            right = int(comp_id[r, c + 1]) if c + 1 < w and int(grid[r, c + 1]) != WALL_COLOR else -1
            up = int(comp_id[r - 1, c]) if r - 1 >= 0 and int(grid[r - 1, c]) != WALL_COLOR else -1
            down = int(comp_id[r + 1, c]) if r + 1 < h and int(grid[r + 1, c]) != WALL_COLOR else -1

            for a, b in ((left, right), (up, down)):
                if a in valid_regions and b in valid_regions and a != b:
                    adjacency[a].add(b)
                    adjacency[b].add(a)

    return adjacency


def _layer_distances(grid: np.ndarray, cells: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    h, w = grid.shape
    region = set(cells)
    queue = deque()
    distances: dict[tuple[int, int], int] = {}

    for r, c in cells:
        if any(0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == WALL_COLOR for nr, nc in _neighbors4(r, c)):
            distances[(r, c)] = 0
            queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in region and (nr, nc) not in distances:
                distances[(nr, nc)] = distances[(r, c)] + 1
                queue.append((nr, nc))

    return distances


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each seeded enclosed region with as many inward color layers as its number of neighboring regions.

    Input structure:
    - Black background cells are partitioned by a one-pixel-thick red wall network.
    - Every enclosed non-red region contains exactly one non-black, non-red seed pixel.
    - Different regions are separated only by red wall pixels and keep the same overall grid size.

    Rule:
    - Identify every enclosed region as a 4-connected non-red component and locate its unique seed color.
    - Build a region adjacency graph: two regions are neighbors when they appear on opposite sides of the same red wall segment.
    - For each region with neighbor count N, fill all cells whose inward distance from the red wall is less than N with the seed color; if the region is thinner than N layers, fill the whole region.
    - Keep red walls unchanged and leave any deeper unfilled center cells black.

    Key insight:
    - The visible fill depth is not determined by the seed position itself but by the region topology: count neighbors first, then convert that count into concentric wall-to-center layers.
    """
    g = np.asarray(grid, dtype=int)
    comp_id, components, seeds_by_comp = _find_open_components(g)

    valid_regions: set[int] = set()
    seed_color_by_region: dict[int, int] = {}
    for region_id, seeds in enumerate(seeds_by_comp):
        if len(seeds) == 1:
            valid_regions.add(region_id)
            sr, sc = seeds[0]
            seed_color_by_region[region_id] = int(g[sr, sc])

    adjacency = _region_adjacency(g, comp_id, valid_regions)

    out = g.copy()
    for region_id in valid_regions:
        cells = components[region_id]
        for r, c in cells:
            out[r, c] = BG_COLOR

        layer_count = len(adjacency[region_id])
        if layer_count <= 0:
            continue

        distances = _layer_distances(g, cells)
        fill_color = seed_color_by_region[region_id]
        for (r, c), depth in distances.items():
            if depth < layer_count:
                out[r, c] = fill_color

    return out


def _coarse_neighbors(rows: int, cols: int, r: int, c: int) -> list[tuple[int, int]]:
    nbrs = []
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < rows and 0 <= nc < cols:
            nbrs.append((nr, nc))
    return nbrs


def _sample_region_count(cell_count: int) -> int:
    max_regions = min(len(SEED_COLORS), cell_count)
    options = list(range(2, max_regions + 1))
    weights = [1 if count == 2 else 2 for count in options]
    if max_regions >= 5:
        weights = [1 if count in (2, max_regions) else 3 for count in options]
    return random.choices(options, weights=weights, k=1)[0]


def _grow_connected_partition(rows: int, cols: int, region_count: int) -> np.ndarray | None:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    seed_cells = random.sample(cells, region_count)

    assignment = {cell: idx for idx, cell in enumerate(seed_cells)}
    frontiers: list[set[tuple[int, int]]] = [set() for _ in range(region_count)]
    for idx, cell in enumerate(seed_cells):
        for nbr in _coarse_neighbors(rows, cols, cell[0], cell[1]):
            if nbr not in assignment:
                frontiers[idx].add(nbr)

    unassigned = set(cells) - set(seed_cells)
    while unassigned:
        candidates = [idx for idx in range(region_count) if any(cell in unassigned for cell in frontiers[idx])]
        if not candidates:
            return None

        region_id = random.choice(candidates)
        choices = [cell for cell in frontiers[region_id] if cell in unassigned]
        cell = random.choice(choices)
        assignment[cell] = region_id
        unassigned.remove(cell)

        for nbr in _coarse_neighbors(rows, cols, cell[0], cell[1]):
            if nbr in unassigned:
                frontiers[region_id].add(nbr)

    coarse = np.zeros((rows, cols), dtype=int)
    for (r, c), region_id in assignment.items():
        coarse[r, c] = region_id
    return coarse


def _coarse_adjacency(coarse: np.ndarray, region_count: int) -> dict[int, set[int]]:
    rows, cols = coarse.shape
    adjacency = {region_id: set() for region_id in range(region_count)}
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows and int(coarse[r, c]) != int(coarse[r + 1, c]):
                a, b = int(coarse[r, c]), int(coarse[r + 1, c])
                adjacency[a].add(b)
                adjacency[b].add(a)
            if c + 1 < cols and int(coarse[r, c]) != int(coarse[r, c + 1]):
                a, b = int(coarse[r, c]), int(coarse[r, c + 1])
                adjacency[a].add(b)
                adjacency[b].add(a)
    return adjacency


def _render_partition(coarse: np.ndarray, cell_size: int) -> np.ndarray:
    rows, cols = coarse.shape
    height = rows * cell_size + (rows + 1)
    width = cols * cell_size + (cols + 1)
    grid = np.full((height, width), WALL_COLOR, dtype=int)

    for r in range(rows):
        for c in range(cols):
            r0 = 1 + r * (cell_size + 1)
            c0 = 1 + c * (cell_size + 1)
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = BG_COLOR

            if c + 1 < cols and int(coarse[r, c]) == int(coarse[r, c + 1]):
                grid[r0 : r0 + cell_size, c0 + cell_size] = BG_COLOR
            if r + 1 < rows and int(coarse[r, c]) == int(coarse[r + 1, c]):
                grid[r0 + cell_size, c0 : c0 + cell_size] = BG_COLOR

    return grid


def _interesting_partition(coarse: np.ndarray, adjacency: dict[int, set[int]]) -> bool:
    neighbor_counts = [len(adjacency[region_id]) for region_id in sorted(adjacency)]
    return max(neighbor_counts) >= 2 and len(set(neighbor_counts)) >= 2


def _seed_region_pixels(grid: np.ndarray, region_count: int) -> bool:
    comp_id, components, seeds_by_comp = _find_open_components(grid)
    if len(components) != region_count or any(seeds for seeds in seeds_by_comp):
        return False

    colors = random.sample(SEED_COLORS, region_count)
    fully_fillable = False
    partially_fillable = False

    valid_regions = set(range(region_count))
    adjacency = _region_adjacency(grid, comp_id, valid_regions)

    for region_id, cells in enumerate(components):
        distances = _layer_distances(grid, cells)
        if not distances:
            return False

        max_depth = max(distances.values())
        layer_count = len(adjacency[region_id])
        fully_fillable |= layer_count > max_depth
        partially_fillable |= layer_count <= max_depth

        best_depth = max_depth
        candidates = [cell for cell, depth in distances.items() if depth == best_depth]
        sr, sc = random.choice(candidates)
        grid[sr, sc] = colors[region_id]

    return fully_fillable and partially_fillable


def _make_candidate() -> np.ndarray | None:
    cell_size = random.choice(CELL_SIZE_OPTIONS)
    max_coarse = 5 if cell_size == 4 else 6
    coarse_rows = random.randint(3, max_coarse)
    coarse_cols = random.randint(3, max_coarse)
    region_count = _sample_region_count(coarse_rows * coarse_cols)

    coarse = _grow_connected_partition(coarse_rows, coarse_cols, region_count)
    if coarse is None:
        return None

    adjacency = _coarse_adjacency(coarse, region_count)
    if not _interesting_partition(coarse, adjacency):
        return None

    grid = _render_partition(coarse, cell_size)
    if not _seed_region_pixels(grid, region_count):
        return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _make_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid region-layer ARC input")
