from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
SEPARATOR = 9
SEED_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)

GEN_RETRY_LIMIT = 300
COARSE_H_RANGE = (3, 6)
COARSE_W_RANGE = (3, 6)
CELL_SIZE_RANGE = (2, 4)
REGION_COUNT_RANGE = (3, 6)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each seed by a shape-dependent number of flood-fill steps inside its maroon-delimited region.

    Input structure:
    - The background is black and maroon one-pixel horizontal/vertical line segments partition the grid.
    - Every non-maroon 4-connected region contains exactly one single-pixel seed whose color is neither black nor maroon.
    - Different regions use different seed colors, and the separator lines remain fixed barriers.

    Rule:
    - For each non-maroon region, trace its rectilinear boundary and count how many straight horizontal or vertical side segments it has.
    - Let the allowed expansion count be floor(side_count / 2); starting from the seed, perform that many 4-neighbor flood-fill layers but only inside the same region.
    - Recolor every reached black cell with the seed color, keep the seed itself unchanged, and leave maroon separators untouched.

    Key insight:
    - The fill radius is not tied to area; it is determined by the region's boundary complexity, so more turns in the outline allow more flood-fill layers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    seen_seed_colors: set[int] = set()
    for region_mask, seed_pos, seed_color in _extract_regions(g):
        assert seed_color not in seen_seed_colors, "seed colors must be unique across regions"
        seen_seed_colors.add(seed_color)

        side_count = _count_boundary_sides(region_mask)
        max_steps = side_count // 2
        reached = _cells_within_steps(region_mask, seed_pos, max_steps)
        fill_mask = reached & (g == BLACK)
        out[fill_mask] = seed_color

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    coarse_h = rng.randint(*COARSE_H_RANGE)
    coarse_w = rng.randint(*COARSE_W_RANGE)
    max_cell_h = min(CELL_SIZE_RANGE[1], (30 - (coarse_h - 1)) // coarse_h)
    max_cell_w = min(CELL_SIZE_RANGE[1], (30 - (coarse_w - 1)) // coarse_w)
    if max_cell_h < CELL_SIZE_RANGE[0] or max_cell_w < CELL_SIZE_RANGE[0]:
        return None

    cell_h = rng.randint(CELL_SIZE_RANGE[0], max_cell_h)
    cell_w = rng.randint(CELL_SIZE_RANGE[0], max_cell_w)

    total_cells = coarse_h * coarse_w
    max_regions = min(REGION_COUNT_RANGE[1], len(SEED_COLORS), total_cells - 1)
    if max_regions < REGION_COUNT_RANGE[0]:
        return None
    region_count = rng.randint(REGION_COUNT_RANGE[0], max_regions)

    labels = _random_connected_partition(coarse_h, coarse_w, region_count, rng)
    grid = _render_partition(labels, cell_h, cell_w)
    blank_regions = _components(grid != SEPARATOR)
    if len(blank_regions) != region_count:
        return None

    side_counts = []
    saw_rectangle = False
    saw_irregular = False
    for coords in blank_regions:
        region_mask = np.zeros_like(grid, dtype=bool)
        rows, cols = zip(*coords)
        region_mask[rows, cols] = True
        side_counts.append(_count_boundary_sides(region_mask))
        if _is_full_rectangle(region_mask):
            saw_rectangle = True
        else:
            saw_irregular = True
    if len(set(side_counts)) < 2 or not saw_rectangle or not saw_irregular:
        return None

    seed_colors = rng.sample(SEED_COLORS, region_count)
    for color, coords in zip(seed_colors, blank_regions):
        region_mask = np.zeros_like(grid, dtype=bool)
        rows, cols = zip(*coords)
        region_mask[rows, cols] = True
        seed_r, seed_c = _choose_seed_cell(region_mask, rng)
        grid[seed_r, seed_c] = color

    try:
        parsed = _extract_regions(grid)
    except AssertionError:
        return None
    if len(parsed) != region_count:
        return None

    parsed_steps = [_count_boundary_sides(region_mask) // 2 for region_mask, _seed, _color in parsed]
    if len(set(parsed_steps)) < 2:
        return None

    return grid


def _extract_regions(grid: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int], int]]:
    mask = grid != SEPARATOR
    regions = []
    for coords in _components(mask):
        seed_cells = [(r, c) for r, c in coords if int(grid[r, c]) not in (BLACK, SEPARATOR)]
        assert len(seed_cells) == 1, "each region must contain exactly one seed pixel"
        seed_pos = seed_cells[0]
        seed_color = int(grid[seed_pos])

        region_mask = np.zeros_like(mask, dtype=bool)
        rows, cols = zip(*coords)
        region_mask[rows, cols] = True
        regions.append((region_mask, seed_pos, seed_color))

    assert regions, "expected at least one region"
    return regions


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            result.append(coords)

    return result


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _count_boundary_sides(region_mask: np.ndarray) -> int:
    top = np.zeros_like(region_mask, dtype=bool)
    bottom = np.zeros_like(region_mask, dtype=bool)
    left = np.zeros_like(region_mask, dtype=bool)
    right = np.zeros_like(region_mask, dtype=bool)

    top[0, :] = region_mask[0, :]
    top[1:, :] = region_mask[1:, :] & ~region_mask[:-1, :]
    bottom[-1, :] = region_mask[-1, :]
    bottom[:-1, :] = region_mask[:-1, :] & ~region_mask[1:, :]
    left[:, 0] = region_mask[:, 0]
    left[:, 1:] = region_mask[:, 1:] & ~region_mask[:, :-1]
    right[:, -1] = region_mask[:, -1]
    right[:, :-1] = region_mask[:, :-1] & ~region_mask[:, 1:]

    return (
        _count_row_runs(top)
        + _count_row_runs(bottom)
        + _count_col_runs(left)
        + _count_col_runs(right)
    )


def _count_row_runs(mask: np.ndarray) -> int:
    starts = mask.copy()
    starts[:, 1:] &= ~mask[:, :-1]
    return int(starts.sum())


def _count_col_runs(mask: np.ndarray) -> int:
    starts = mask.copy()
    starts[1:, :] &= ~mask[:-1, :]
    return int(starts.sum())


def _cells_within_steps(
    region_mask: np.ndarray, seed_pos: tuple[int, int], max_steps: int
) -> np.ndarray:
    h, w = region_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    sr, sc = seed_pos
    dist[sr, sc] = 0
    queue = deque([seed_pos])

    while queue:
        r, c = queue.popleft()
        if dist[r, c] >= max_steps:
            continue
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and region_mask[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))

    return dist >= 0


def _random_connected_partition(
    h: int, w: int, region_count: int, rng: random.Random
) -> np.ndarray:
    labels = np.full((h, w), -1, dtype=int)
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    seeds = rng.sample(all_cells, region_count)
    frontiers = [set() for _ in range(region_count)]

    for label, (r, c) in enumerate(seeds):
        labels[r, c] = label
        for nr, nc in _coarse_neighbors(r, c, h, w):
            if labels[nr, nc] == -1:
                frontiers[label].add((nr, nc))

    remaining = h * w - region_count
    while remaining > 0:
        active = [label for label in range(region_count) if frontiers[label]]
        if not active:
            raise RuntimeError("partition frontiers unexpectedly exhausted")

        sizes = [int(np.sum(labels == label)) for label in active]
        min_size = min(sizes)
        preferred = [label for label in active if int(np.sum(labels == label)) == min_size]
        label = rng.choice(preferred if rng.random() < 0.7 else active)

        cell = rng.choice(tuple(frontiers[label]))
        frontiers[label].discard(cell)
        r, c = cell
        if labels[r, c] != -1:
            continue
        labels[r, c] = label
        remaining -= 1
        for nr, nc in _coarse_neighbors(r, c, h, w):
            if labels[nr, nc] == -1:
                frontiers[label].add((nr, nc))

    return labels


def _coarse_neighbors(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _render_partition(labels: np.ndarray, cell_h: int, cell_w: int) -> np.ndarray:
    coarse_h, coarse_w = labels.shape
    height = coarse_h * cell_h + (coarse_h - 1)
    width = coarse_w * cell_w + (coarse_w - 1)
    grid = np.full((height, width), BLACK, dtype=int)

    for split_r in range(1, coarse_h):
        wall_r = split_r * (cell_h + 1) - 1
        for c in range(coarse_w):
            if int(labels[split_r - 1, c]) != int(labels[split_r, c]):
                c0 = c * (cell_w + 1)
                grid[wall_r, c0 : c0 + cell_w] = SEPARATOR

    for split_c in range(1, coarse_w):
        wall_c = split_c * (cell_w + 1) - 1
        for r in range(coarse_h):
            if int(labels[r, split_c - 1]) != int(labels[r, split_c]):
                r0 = r * (cell_h + 1)
                grid[r0 : r0 + cell_h, wall_c] = SEPARATOR

    for split_r in range(1, coarse_h):
        wall_r = split_r * (cell_h + 1) - 1
        for split_c in range(1, coarse_w):
            wall_c = split_c * (cell_w + 1) - 1
            block = {
                int(labels[split_r - 1, split_c - 1]),
                int(labels[split_r - 1, split_c]),
                int(labels[split_r, split_c - 1]),
                int(labels[split_r, split_c]),
            }
            if len(block) > 1:
                grid[wall_r, wall_c] = SEPARATOR

    return grid


def _choose_seed_cell(region_mask: np.ndarray, rng: random.Random) -> tuple[int, int]:
    boundary_dist = _distance_from_boundary(region_mask)
    region_coords = np.argwhere(region_mask)
    values = boundary_dist[region_mask]
    max_dist = int(values.max())
    threshold = max(0, max_dist - 1)
    candidates = [
        (int(r), int(c))
        for r, c in region_coords
        if int(boundary_dist[int(r), int(c)]) >= threshold
    ]
    return rng.choice(candidates)


def _distance_from_boundary(region_mask: np.ndarray) -> np.ndarray:
    h, w = region_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    queue: deque[tuple[int, int]] = deque()

    for r, c in np.argwhere(region_mask):
        rr = int(r)
        cc = int(c)
        if _is_boundary_cell(region_mask, rr, cc):
            dist[rr, cc] = 0
            queue.append((rr, cc))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and region_mask[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))

    return dist


def _is_boundary_cell(region_mask: np.ndarray, r: int, c: int) -> bool:
    h, w = region_mask.shape
    for nr, nc in _neighbors4(r, c):
        if not (0 <= nr < h and 0 <= nc < w) or not region_mask[nr, nc]:
            return True
    return False


def _is_full_rectangle(region_mask: np.ndarray) -> bool:
    coords = np.argwhere(region_mask)
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return bool(np.all(region_mask[r0 : r1 + 1, c0 : c1 + 1]))
