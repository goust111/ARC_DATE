from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
SEED_COLORS = (BLUE, RED, GREEN)

GRID_MIN = 10
GRID_MAX = 18
MIN_SEED_DISTANCE_SQ = 36
MAX_RADIUS = 3
MAX_ATTEMPTS = 400


def _boundary_distance(r: int, c: int, height: int, width: int) -> int:
    return min(r, c, height - 1 - r, width - 1 - c) + 1


def _effective_radius(r: int, c: int, height: int, width: int) -> int:
    return min(_boundary_distance(r, c, height, width) - 1, MAX_RADIUS)


def _shape_cells(color: int, r: int, c: int, height: int, width: int) -> set[tuple[int, int]]:
    radius = _effective_radius(r, c, height, width)
    cells: set[tuple[int, int]] = set()

    if color == RED:
        for rr in range(r - radius, r + radius + 1):
            for cc in range(c - radius, c + radius + 1):
                if 0 <= rr < height and 0 <= cc < width:
                    cells.add((rr, cc))
        return cells

    if color == GREEN:
        for dr in range(-radius, radius + 1):
            rr = r + dr
            span = radius - abs(dr)
            for cc in range(c - span, c + span + 1):
                if 0 <= rr < height and 0 <= cc < width:
                    cells.add((rr, cc))
        return cells

    assert color == BLUE, f"unexpected seed color: {color}"
    for dc in range(-radius, radius + 1):
        cc = c + dc
        if 0 <= cc < width:
            cells.add((r, cc))
    for dr in range(-radius, radius + 1):
        rr = r + dr
        if 0 <= rr < height:
            cells.add((rr, c))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a color-coded pattern around each seed.

    Red grows into a filled square, green into a diamond, and blue into a cross.
    The size comes from the seed's distance to the nearest boundary, capped at radius 3.
    Later seeds in row-major order overwrite earlier growth on overlaps.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    out = np.full_like(g, BLACK)

    seeds = sorted((int(r), int(c), int(g[r, c])) for r, c in np.argwhere(g != BLACK))
    for r, c, color in seeds:
        assert color in SEED_COLORS, f"unexpected seed color: {color}"
        for rr, cc in _shape_cells(color, r, c, height, width):
            out[rr, cc] = color

    return out


def _seeds_far_enough(candidate: tuple[int, int], placed: list[tuple[int, int, int]]) -> bool:
    r, c = candidate
    for pr, pc, _color in placed:
        if (r - pr) * (r - pr) + (c - pc) * (c - pc) < MIN_SEED_DISTANCE_SQ:
            return False
    return True


def _positions_by_radius(height: int, width: int) -> dict[int, list[tuple[int, int]]]:
    buckets = {radius: [] for radius in range(MAX_RADIUS + 1)}
    for r in range(height):
        for c in range(width):
            buckets[_effective_radius(r, c, height, width)].append((r, c))
    return buckets


def _pick_position(
    color: int,
    radius: int,
    height: int,
    width: int,
    radius_positions: dict[int, list[tuple[int, int]]],
    placed: list[tuple[int, int, int]],
    occupied: set[tuple[int, int]],
) -> tuple[int, int, set[tuple[int, int]]] | None:
    candidates: list[tuple[int, int, set[tuple[int, int]]]] = []
    for r, c in radius_positions[radius]:
        if not _seeds_far_enough((r, c), placed):
            continue
        shape = _shape_cells(color, r, c, height, width)
        if shape & occupied:
            continue
        candidates.append((r, c, shape))
    return random.choice(candidates) if candidates else None


def _overlap_pairs(height: int, width: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if _effective_radius(r, c, height, width) != MAX_RADIUS:
                continue
            if c + 6 < width and _effective_radius(r, c + 6, height, width) == MAX_RADIUS:
                pairs.append(((r, c), (r, c + 6)))
            if r + 6 < height and _effective_radius(r + 6, c, height, width) == MAX_RADIUS:
                pairs.append(((r, c), (r + 6, c)))
    return pairs


def _sample_radii(feasible_radii: list[int], count: int) -> list[int]:
    pool = feasible_radii[:]
    random.shuffle(pool)
    radii: list[int] = []
    while len(radii) < count:
        if pool:
            radii.append(pool.pop())
        else:
            radii.append(random.choice(feasible_radii))
    return radii


def _seed_specs(feasible_radii: list[int], n_seeds: int, want_overlap: bool) -> list[tuple[int, int]]:
    colors = list(SEED_COLORS)
    random.shuffle(colors)

    specs: list[tuple[int, int]] = []
    if want_overlap:
        specs.append((colors[0], MAX_RADIUS))
        specs.append((colors[1], MAX_RADIUS))
        specs.append((colors[2], random.choice(feasible_radii)))
    else:
        for color, radius in zip(colors, _sample_radii(feasible_radii, len(colors))):
            specs.append((color, radius))

    for _ in range(n_seeds - len(colors)):
        specs.append((random.choice(SEED_COLORS), random.choice(feasible_radii)))
    return specs


def _pair_overlap_count(seeds: list[tuple[int, int, int]], height: int, width: int) -> int:
    masks = [_shape_cells(color, r, c, height, width) for r, c, color in seeds]
    count = 0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if masks[i] & masks[j]:
                count += 1
    return count


def _build_grid(height: int, width: int, specs: list[tuple[int, int]], want_overlap: bool) -> np.ndarray | None:
    radius_positions = _positions_by_radius(height, width)
    placed: list[tuple[int, int, int]] = []
    occupied: set[tuple[int, int]] = set()

    if want_overlap:
        pairs = _overlap_pairs(height, width)
        random.shuffle(pairs)
        if not pairs:
            return None

        pair_color_a = specs[0][0]
        pair_color_b = specs[1][0]
        pair_placed = False
        for (r0, c0), (r1, c1) in pairs:
            shape0 = _shape_cells(pair_color_a, r0, c0, height, width)
            shape1 = _shape_cells(pair_color_b, r1, c1, height, width)
            if not shape0 & shape1:
                continue
            placed.extend([(r0, c0, pair_color_a), (r1, c1, pair_color_b)])
            occupied |= shape0 | shape1
            pair_placed = True
            break
        if not pair_placed:
            return None
        remaining = specs[2:]
    else:
        remaining = specs

    for color, radius in remaining:
        picked = _pick_position(color, radius, height, width, radius_positions, placed, occupied)
        if picked is None:
            return None
        r, c, shape = picked
        placed.append((r, c, color))
        occupied |= shape

    grid = np.full((height, width), BLACK, dtype=int)
    for r, c, color in placed:
        grid[r, c] = color
    return grid


def _valid_sample(grid: np.ndarray, want_overlap: bool) -> bool:
    seeds = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(grid != BLACK)]
    if {color for _r, _c, color in seeds} != set(SEED_COLORS):
        return False

    for i, (r0, c0, _color0) in enumerate(seeds):
        for r1, c1, _color1 in seeds[i + 1 :]:
            if (r0 - r1) * (r0 - r1) + (c0 - c1) * (c0 - c1) < MIN_SEED_DISTANCE_SQ:
                return False

    overlap_count = _pair_overlap_count(seeds, grid.shape[0], grid.shape[1])
    if want_overlap and overlap_count < 1:
        return False
    if not want_overlap and overlap_count != 0:
        return False
    if overlap_count > 1:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.count_nonzero(out) > int(grid.size * 0.7):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        want_overlap = random.random() < 0.45

        size_min = 13 if want_overlap else GRID_MIN
        height = random.randint(size_min, GRID_MAX)
        width = random.randint(size_min, GRID_MAX)
        feasible_radii = [
            radius
            for radius, positions in _positions_by_radius(height, width).items()
            if positions
        ]
        if not feasible_radii:
            continue
        if want_overlap and MAX_RADIUS not in feasible_radii:
            continue

        max_seeds = 3
        if height * width >= 170:
            max_seeds += 1
        if height * width >= 250:
            max_seeds += 1
        n_seeds = random.randint(3, max_seeds)

        specs = _seed_specs(feasible_radii, n_seeds, want_overlap)
        grid = _build_grid(height, width, specs, want_overlap)
        if grid is None:
            continue
        if _valid_sample(grid, want_overlap):
            return grid

    raise RuntimeError("failed to generate a valid input grid")
