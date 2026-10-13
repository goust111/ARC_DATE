from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _same_color_component(
    grid: np.ndarray, start_r: int, start_c: int, visited: np.ndarray
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], set[int]]:
    height, width = grid.shape
    color = int(grid[start_r, start_c])
    queue = deque([(start_r, start_c)])
    visited[start_r, start_c] = True

    coords: list[tuple[int, int]] = []
    boundary: list[tuple[int, int]] = []
    neighbor_colors: set[int] = set()

    while queue:
        r, c = queue.popleft()
        coords.append((r, c))

        is_boundary = False
        for rr, cc in _neighbors4(r, c, height, width):
            value = int(grid[rr, cc])
            if value == color:
                if not visited[rr, cc]:
                    visited[rr, cc] = True
                    queue.append((rr, cc))
            else:
                neighbor_colors.add(value)
                is_boundary = True

        if is_boundary:
            boundary.append((r, c))

    return color, coords, boundary, neighbor_colors


def _layer_distances(mask: np.ndarray, boundary: list[tuple[int, int]]) -> np.ndarray:
    dist = np.full(mask.shape, -1, dtype=int)
    queue = deque(boundary)
    for r, c in boundary:
        dist[r, c] = 0

    height, width = mask.shape
    while queue:
        r, c = queue.popleft()
        for rr, cc in _neighbors4(r, c, height, width):
            if not mask[rr, cc] or dist[rr, cc] != -1:
                continue
            dist[rr, cc] = dist[r, c] + 1
            queue.append((rr, cc))

    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """Repaint each region as alternating inward layers whose count matches its distinct neighboring colors.

    Input structure:
    - The grid has no black background; it is fully partitioned into large 4-connected single-color regions.
    - Regions touch each other directly along edges, and each region borders at least two differently colored neighbors.
    - A color may appear in more than one disconnected place, so regions must be detected by connected components rather than by color alone.

    Rule:
    - For each 4-connected monochrome region, count how many distinct colors touch it across 4-neighbor region boundaries; call that value N.
    - Starting from the region cells that touch a different color, peel inward by graph distance inside the region: distance 0 is layer 1, distance 1 is layer 2, and so on.
    - Paint the first N layers in alternating colors: odd-numbered layers keep the region's original color, even-numbered layers become grey. Any deeper remaining core cells become black.

    Key insight:
    - The number of concentric repaint layers is not encoded by size or shape alone; it is determined by how many different neighboring region colors the boundary touches.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    height, width = g.shape
    visited = np.zeros((height, width), dtype=bool)
    out = np.full_like(g, BLACK)

    for r in range(height):
        for c in range(width):
            if visited[r, c]:
                continue

            color, coords, boundary, neighbor_colors = _same_color_component(g, r, c, visited)
            if not boundary or not neighbor_colors:
                for rr, cc in coords:
                    out[rr, cc] = color
                continue

            mask = np.zeros_like(visited)
            rows, cols = zip(*coords)
            mask[rows, cols] = True
            dist = _layer_distances(mask, boundary)
            layers = len(neighbor_colors)

            for rr, cc in coords:
                depth = int(dist[rr, cc])
                if 0 <= depth < layers:
                    out[rr, cc] = color if depth % 2 == 0 else GREY

    return out


GEN_INPUT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_HEIGHT_RANGE = (12, 24)
GEN_WIDTH_RANGE = (12, 24)
GEN_REGION_COUNT_RANGE = (4, 8)
GEN_RETRY_LIMIT = 600
GEN_BRANCH_MODES = ("core", "exhausted", "mixed")
GEN_GROWTH_STYLES = ("compact", "balanced", "stringy")


def _choose_seed_positions(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(cells)
    min_dist = 3 if min(height, width) >= 16 else 2

    seeds: list[tuple[int, int]] = []
    for r, c in cells:
        if all(abs(r - rr) + abs(c - cc) >= min_dist for rr, cc in seeds):
            seeds.append((r, c))
            if len(seeds) == count:
                return seeds
    return None


def _target_areas(total_cells: int, count: int) -> tuple[list[int], int]:
    min_area = max(8, total_cells // (count * 3))
    min_area = min(min_area, total_cells // count)
    slack = total_cells - min_area * count

    weights = [random.random() + 0.35 for _ in range(count)]
    total_weight = sum(weights)
    targets = [min_area + int(slack * weight / total_weight) for weight in weights]

    missing = total_cells - sum(targets)
    order = sorted(range(count), key=lambda idx: weights[idx], reverse=True)
    for i in range(missing):
        targets[order[i % count]] += 1

    return targets, min_area


def _growth_style(branch_mode: str) -> str:
    if branch_mode == "core":
        return random.choice(("compact", "balanced"))
    if branch_mode == "exhausted":
        return random.choice(("balanced", "stringy"))
    return random.choice(GEN_GROWTH_STYLES)


def _pick_frontier_cell(
    regions: np.ndarray,
    frontier: set[tuple[int, int]],
    region_id: int,
    seed: tuple[int, int],
    style: str,
) -> tuple[int, int] | None:
    if not frontier:
        return None

    height, width = regions.shape
    options = tuple(frontier)
    weights: list[float] = []
    seed_r, seed_c = seed

    for r, c in options:
        same_adj = 0
        other_adj = 0
        for rr, cc in _neighbors4(r, c, height, width):
            value = int(regions[rr, cc])
            if value == region_id:
                same_adj += 1
            elif value != -1:
                other_adj += 1

        dist = abs(r - seed_r) + abs(c - seed_c)
        if style == "compact":
            weight = (same_adj + 1) ** 3 / (dist + 1)
        elif style == "stringy":
            weight = (5 - same_adj) * (dist + 1) * (1.4 if other_adj == 0 else 1.0)
        else:
            weight = (same_adj + 1) * (dist + 2) * (1.25 if other_adj == 0 else 1.0)

        weights.append(max(weight, 0.01))

    return random.choices(options, weights=weights, k=1)[0]


def _grow_partition(height: int, width: int, count: int, style: str) -> tuple[np.ndarray, int] | None:
    seeds = _choose_seed_positions(height, width, count)
    if seeds is None:
        return None

    targets, min_area = _target_areas(height * width, count)
    regions = np.full((height, width), -1, dtype=int)
    frontiers = [set() for _ in range(count)]
    areas = [1] * count

    for region_id, (r, c) in enumerate(seeds):
        regions[r, c] = region_id

    for region_id, (r, c) in enumerate(seeds):
        for rr, cc in _neighbors4(r, c, height, width):
            if regions[rr, cc] == -1:
                frontiers[region_id].add((rr, cc))

    unassigned = height * width - count
    while unassigned > 0:
        available = [idx for idx in range(count) if frontiers[idx]]
        if not available:
            return None

        weights = []
        for idx in available:
            deficit = max(targets[idx] - areas[idx], 1)
            weights.append(deficit * max(len(frontiers[idx]), 1))

        region_id = random.choices(available, weights=weights, k=1)[0]
        cell = _pick_frontier_cell(regions, frontiers[region_id], region_id, seeds[region_id], style)
        if cell is None:
            frontiers[region_id].clear()
            continue

        r, c = cell
        if regions[r, c] != -1:
            frontiers[region_id].discard(cell)
            continue

        regions[r, c] = region_id
        areas[region_id] += 1
        unassigned -= 1

        for frontier in frontiers:
            frontier.discard(cell)
        for rr, cc in _neighbors4(r, c, height, width):
            if regions[rr, cc] == -1:
                frontiers[region_id].add((rr, cc))

    if min(areas) < min_area:
        return None
    return regions, min_area


def _adjacency(region_grid: np.ndarray, count: int) -> list[set[int]]:
    height, width = region_grid.shape
    neighbors = [set() for _ in range(count)]

    for r in range(height):
        for c in range(width):
            cur = int(region_grid[r, c])
            if r + 1 < height:
                down = int(region_grid[r + 1, c])
                if down != cur:
                    neighbors[cur].add(down)
                    neighbors[down].add(cur)
            if c + 1 < width:
                right = int(region_grid[r, c + 1])
                if right != cur:
                    neighbors[cur].add(right)
                    neighbors[right].add(cur)

    return neighbors


def _profile_regions(region_grid: np.ndarray, neighbors: list[set[int]]):
    profiles = []
    height, width = region_grid.shape

    for region_id, adjacent in enumerate(neighbors):
        mask = region_grid == region_id
        coords = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
        boundary = []
        for r, c in coords:
            if any(int(region_grid[rr, cc]) != region_id for rr, cc in _neighbors4(r, c, height, width)):
                boundary.append((r, c))

        if not boundary:
            return None

        dist = _layer_distances(mask, boundary)
        profiles.append(
            {
                "region": region_id,
                "area": int(mask.sum()),
                "neighbor_count": len(adjacent),
                "max_dist": int(dist[mask].max()),
            }
        )

    return profiles


def _colorize_regions(region_grid: np.ndarray, count: int) -> np.ndarray:
    palette = random.sample(GEN_INPUT_COLORS, count)
    return np.asarray(palette, dtype=int)[region_grid]


def _meets_branch_requirement(profiles: list[dict[str, int]], branch_mode: str) -> bool:
    has_core = any(p["max_dist"] >= p["neighbor_count"] for p in profiles)
    has_exhausted = any(p["max_dist"] < p["neighbor_count"] for p in profiles)

    if branch_mode == "core":
        return has_core
    if branch_mode == "exhausted":
        return has_exhausted
    return has_core and has_exhausted


def _gen_attempt() -> np.ndarray | None:
    branch_mode = random.choice(GEN_BRANCH_MODES)
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    max_regions = min(GEN_REGION_COUNT_RANGE[1], len(GEN_INPUT_COLORS), (height * width) // 8)
    if max_regions < GEN_REGION_COUNT_RANGE[0]:
        return None

    count = random.randint(GEN_REGION_COUNT_RANGE[0], max_regions)
    style = _growth_style(branch_mode)
    grown = _grow_partition(height, width, count, style)
    if grown is None:
        return None

    region_grid, _min_area = grown
    neighbors = _adjacency(region_grid, count)
    profiles = _profile_regions(region_grid, neighbors)
    if profiles is None:
        return None

    degrees = [p["neighbor_count"] for p in profiles]
    if min(degrees) < 2:
        return None
    if max(degrees) < 3:
        return None
    if len(set(degrees)) < 2:
        return None
    if max(p["max_dist"] for p in profiles) < 1:
        return None
    if max(p["max_dist"] for p in profiles) < 2:
        return None
    if not _meets_branch_requirement(profiles, branch_mode):
        return None

    inp = _colorize_regions(region_grid, count)
    out = main(inp)
    if np.any(inp == BLACK) or np.any(inp == GREY):
        return None
    if not np.any(out == GREY):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layered-neighbor-count map")
