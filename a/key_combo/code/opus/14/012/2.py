from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

SEED_COLORS = (BLUE, RED, GREEN, YELLOW)
ODD_DISTANCES = (3, 5)
EVEN_DISTANCES = (2, 4, 6)
GRID_MIN = 14
GRID_MAX = 18
WALL_COUNT_RANGE = (10, 18)
GEN_ATTEMPTS = 500

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class SeedSpec:
    r: int
    c: int
    color: int
    d: int
    witness: tuple[int, int]

    @property
    def pos(self) -> tuple[int, int]:
        return (self.r, self.c)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _seed_cells(grid: np.ndarray) -> list[tuple[int, int, int]]:
    out = []
    for r, c in np.argwhere(np.isin(grid, SEED_COLORS)):
        out.append((int(r), int(c), int(grid[int(r), int(c)])))
    return sorted(out)


def _wall_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)]


def _directions_for_distance(d: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if d % 2 == 1:
        return ((0, -1), (0, 1))
    return ((-1, 0), (1, 0))


def _nearest_wall_distance(grid: np.ndarray, r: int, c: int) -> int:
    walls = _wall_cells(grid)
    assert walls, "expected at least one grey obstacle"
    return min(manhattan((r, c), wall) for wall in walls)


def _ray_cells_and_stops(
    grid: np.ndarray, r: int, c: int, d: int
) -> tuple[list[tuple[int, int]], list[dict[str, object]]]:
    cells: list[tuple[int, int]] = []
    legs: list[dict[str, object]] = []
    for dr, dc in _directions_for_distance(d):
        leg_cells: list[tuple[int, int]] = []
        stop = "length"
        for step in range(1, d + 1):
            rr = r + dr * step
            cc = c + dc * step
            if not in_bounds(grid.shape, rr, cc):
                stop = "boundary"
                break
            if int(grid[rr, cc]) == GREY:
                stop = "obstacle"
                break
            leg_cells.append((rr, cc))
        cells.extend(leg_cells)
        legs.append({"dir": (dr, dc), "cells": tuple(leg_cells), "stop": stop})
    return cells, legs


def _solve_with_metadata(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    seeds = _seed_cells(g)
    walls = _wall_cells(g)
    assert walls, "expected grey obstacles"
    assert seeds, "expected colored seed pixels"

    info: list[dict[str, object]] = []
    for r, c, color in seeds:
        d = min(manhattan((r, c), wall) for wall in walls)
        cells, legs = _ray_cells_and_stops(g, r, c, d)
        info.append(
            {
                "pos": (r, c),
                "color": color,
                "d": d,
                "cells": tuple(cells),
                "legs": legs,
            }
        )

    out = g.copy()
    for item in sorted(info, key=lambda x: (-int(x["d"]), -int(x["pos"][0]), -int(x["pos"][1]))):
        color = int(item["color"])
        for rr, cc in item["cells"]:
            out[int(rr), int(cc)] = color

    out[g == GREY] = GREY
    for r, c, color in seeds:
        out[r, c] = color
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast color rays whose direction and length are jointly encoded by each seed's distance to the nearest grey obstacle.

    Input structure:
    - The grid is a black background containing scattered grey obstacle pixels and 3-4 colored seed pixels.
    - Seed colors are distinct and come from blue, red, green, and yellow; grey pixels are fixed walls.
    - The same nearest-obstacle Manhattan distance both selects the seed's ray orientation and sets its ray length.

    Rule:
    - For each seed, compute d = Manhattan distance to the nearest grey obstacle pixel.
    - If d is odd, extend horizontal rays left and right; if d is even, extend vertical rays up and down.
    - Each ray paints up to d cells, stopping early at the grid boundary or just before a grey obstacle; grey pixels are never overwritten.
    - Draw rays in layering order: larger d first, then for ties larger row first, then larger column first, so smaller d / smaller-row / smaller-column seeds overwrite later.
    - Keep all original seed pixels and grey obstacle pixels unchanged.

    Key insight:
    - The obstacle distance is a double-coded parameter: its parity flips the ray axis, while its magnitude sets the reach, and overlap visibility is resolved by distance-based occlusion order rather than by seed color.
    """
    out, _ = _solve_with_metadata(np.asarray(grid, dtype=int))
    return out


def _witness_mode(spec: SeedSpec) -> str:
    wr, wc = spec.witness
    if spec.d % 2 == 1:
        return "ray" if wr == spec.r else "orth"
    return "ray" if wc == spec.c else "orth"


def _pairwise_seed_gap_ok(seeds: list[SeedSpec], min_gap: int = 4) -> bool:
    for i, a in enumerate(seeds):
        for b in seeds[i + 1 :]:
            if manhattan(a.pos, b.pos) < min_gap:
                return False
    return True


def _seed_clear_of_walls(seed: tuple[int, int], d: int, walls: set[tuple[int, int]]) -> bool:
    return all(manhattan(seed, wall) > d for wall in walls)


def _witness_clear_for_other_seeds(
    witness: tuple[int, int], seeds: list[SeedSpec], planned_d: int
) -> bool:
    for other in seeds:
        if manhattan((other.r, other.c), witness) <= other.d:
            return False
    return True


def _find_witness(
    H: int,
    W: int,
    r: int,
    c: int,
    d: int,
    mode: str,
    seeds: list[SeedSpec],
    occupied: set[tuple[int, int]],
) -> tuple[int, int] | None:
    if d % 2 == 1:
        candidates = [(r, c - d), (r, c + d)] if mode == "ray" else [(r - d, c), (r + d, c)]
    else:
        candidates = [(r - d, c), (r + d, c)] if mode == "ray" else [(r, c - d), (r, c + d)]
    random.shuffle(candidates)
    for wr, wc in candidates:
        if not in_bounds((H, W), wr, wc):
            continue
        if (wr, wc) in occupied:
            continue
        if not _witness_clear_for_other_seeds((wr, wc), seeds, d):
            continue
        return (wr, wc)
    return None


def _sample_extra_seed(
    H: int,
    W: int,
    color: int,
    existing_seeds: list[SeedSpec],
    occupied: set[tuple[int, int]],
    parity: str | None = None,
    witness_mode: str | None = None,
    require_boundary: bool = False,
) -> SeedSpec | None:
    d_choices = list(ODD_DISTANCES + EVEN_DISTANCES)
    if parity == "odd":
        d_choices = list(ODD_DISTANCES)
    elif parity == "even":
        d_choices = list(EVEN_DISTANCES)
    for _ in range(250):
        d = random.choice(d_choices)
        r = random.randrange(H)
        c = random.randrange(W)
        if (r, c) in occupied:
            continue
        if any(manhattan((r, c), other.pos) < 4 for other in existing_seeds):
            continue
        if not _seed_clear_of_walls((r, c), d, occupied - {seed.pos for seed in existing_seeds}):
            continue
        if require_boundary:
            if d % 2 == 1 and min(c, W - 1 - c) >= d:
                continue
            if d % 2 == 0 and min(r, H - 1 - r) >= d:
                continue
        mode = witness_mode
        if mode is None:
            mode = random.choice(["orth", "ray"])
        witness = _find_witness(H, W, r, c, d, mode, existing_seeds, occupied)
        if witness is None:
            continue
        return SeedSpec(r=r, c=c, color=color, d=d, witness=witness)
    return None


def _build_cross_scene(H: int, W: int, colors: list[int], n_seeds: int) -> tuple[list[SeedSpec], str] | None:
    d1 = random.choice(ODD_DISTANCES)
    d2 = random.choice([d for d in EVEN_DISTANCES if d <= max(H, W) - 3])

    c1_hi = min(d1 - 1, 3, W - 4)
    if c1_hi < 1:
        return None
    r1_lo = d1 + 1
    r1_hi = H - d1 - 2
    if r1_lo > r1_hi:
        return None

    for _ in range(200):
        r1 = random.randint(r1_lo, r1_hi)
        c1 = random.randint(1, c1_hi)
        witness1 = random.choice([(r1 - d1, c1), (r1 + d1, c1)])
        if not in_bounds((H, W), *witness1):
            continue

        a = random.randint(1, d1)
        b_lo = max(1, 4 - a)
        if b_lo > d2:
            continue
        b = random.randint(b_lo, d2)
        c2 = c1 + a
        if not (1 <= c2 < W - 1):
            continue

        row_options = [r1 - b, r1 + b]
        random.shuffle(row_options)
        for r2 in row_options:
            if not in_bounds((H, W), r2, c2):
                continue
            seeds = [SeedSpec(r1, c1, colors[0], d1, witness1)]
            occupied = {seeds[0].pos, witness1}
            witness2 = _find_witness(H, W, r2, c2, d2, "orth", seeds, occupied | {(r2, c2)})
            if witness2 is None:
                continue
            seeds.append(SeedSpec(r2, c2, colors[1], d2, witness2))
            if not _pairwise_seed_gap_ok(seeds):
                continue
            occupied = {seed.pos for seed in seeds} | {seed.witness for seed in seeds}

            third = _sample_extra_seed(
                H,
                W,
                colors[2],
                seeds,
                occupied,
                witness_mode="ray",
                require_boundary=False,
            )
            if third is None:
                continue
            seeds.append(third)
            if n_seeds == 4:
                occupied = {seed.pos for seed in seeds} | {seed.witness for seed in seeds}
                fourth = _sample_extra_seed(
                    H,
                    W,
                    colors[3],
                    seeds,
                    occupied,
                    require_boundary=True,
                )
                if fourth is None:
                    continue
                seeds.append(fourth)
            return seeds, "cross"
    return None


def _build_horizontal_tie_scene(H: int, W: int, colors: list[int], n_seeds: int) -> tuple[list[SeedSpec], str] | None:
    d = random.choice(ODD_DISTANCES)
    row_lo = d + 1
    row_hi = H - d - 2
    if row_lo > row_hi:
        return None

    for _ in range(200):
        r = random.randint(row_lo, row_hi)
        max_gap = min(2 * d, W - 5)
        if max_gap < 4:
            continue
        gap = random.randint(4, max_gap)
        c1_lo = 1
        c1_hi = W - gap - 2
        if c1_lo > c1_hi:
            continue
        c1 = random.randint(c1_lo, c1_hi)
        c2 = c1 + gap

        witness1 = random.choice([(r - d, c1), (r + d, c1)])
        witness2 = random.choice([(r - d, c2), (r + d, c2)])
        if not (in_bounds((H, W), *witness1) and in_bounds((H, W), *witness2)):
            continue
        seeds = [
            SeedSpec(r, c1, colors[0], d, witness1),
            SeedSpec(r, c2, colors[1], d, witness2),
        ]
        if not _pairwise_seed_gap_ok(seeds):
            continue
        if not _witness_clear_for_other_seeds(witness1, [seeds[1]], d):
            continue
        if not _witness_clear_for_other_seeds(witness2, [seeds[0]], d):
            continue

        overlap_lo = max(c1 + 1, c2 - d)
        overlap_hi = min(c1 + d, c2 - 1)
        if overlap_lo > overlap_hi:
            continue
        c3 = random.randint(overlap_lo, overlap_hi)
        d3 = random.choice(EVEN_DISTANCES)
        b = random.randint(1, d3)
        row_candidates = [r - b, r + b]
        random.shuffle(row_candidates)
        added = False
        for r3 in row_candidates:
            if not in_bounds((H, W), r3, c3):
                continue
            occupied = {seed.pos for seed in seeds} | {seed.witness for seed in seeds} | {(r3, c3)}
            witness3 = _find_witness(H, W, r3, c3, d3, "ray", seeds, occupied)
            if witness3 is None:
                continue
            seeds.append(SeedSpec(r3, c3, colors[2], d3, witness3))
            added = True
            break
        if not added:
            continue

        if n_seeds == 4:
            occupied = {seed.pos for seed in seeds} | {seed.witness for seed in seeds}
            fourth = _sample_extra_seed(
                H,
                W,
                colors[3],
                seeds,
                occupied,
                parity="even",
                witness_mode="orth",
                require_boundary=True,
            )
            if fourth is None:
                continue
            seeds.append(fourth)
        return seeds, "h_tie"
    return None


def _build_vertical_tie_scene(H: int, W: int, colors: list[int], n_seeds: int) -> tuple[list[SeedSpec], str] | None:
    d = random.choice([dd for dd in EVEN_DISTANCES if dd <= H - 3])
    col_lo = d + 1
    col_hi = W - d - 2
    if col_lo > col_hi:
        return None

    for _ in range(200):
        c = random.randint(col_lo, col_hi)
        max_gap = min(2 * d, H - 5)
        if max_gap < 4:
            continue
        gap = random.randint(4, max_gap)
        r1_lo = 1
        r1_hi = H - gap - 2
        if r1_lo > r1_hi:
            continue
        r1 = random.randint(r1_lo, r1_hi)
        r2 = r1 + gap

        witness1 = random.choice([(r1, c - d), (r1, c + d)])
        witness2 = random.choice([(r2, c - d), (r2, c + d)])
        if not (in_bounds((H, W), *witness1) and in_bounds((H, W), *witness2)):
            continue
        seeds = [
            SeedSpec(r1, c, colors[0], d, witness1),
            SeedSpec(r2, c, colors[1], d, witness2),
        ]
        if not _pairwise_seed_gap_ok(seeds):
            continue
        if not _witness_clear_for_other_seeds(witness1, [seeds[1]], d):
            continue
        if not _witness_clear_for_other_seeds(witness2, [seeds[0]], d):
            continue

        overlap_lo = max(r1 + 1, r2 - d)
        overlap_hi = min(r1 + d, r2 - 1)
        if overlap_lo > overlap_hi:
            continue
        r3 = random.randint(overlap_lo, overlap_hi)
        d3 = random.choice(ODD_DISTANCES)
        a = random.randint(1, d3)
        col_candidates = [c - a, c + a]
        random.shuffle(col_candidates)
        added = False
        for c3 in col_candidates:
            if not in_bounds((H, W), r3, c3):
                continue
            occupied = {seed.pos for seed in seeds} | {seed.witness for seed in seeds} | {(r3, c3)}
            witness3 = _find_witness(H, W, r3, c3, d3, "ray", seeds, occupied)
            if witness3 is None:
                continue
            seeds.append(SeedSpec(r3, c3, colors[2], d3, witness3))
            added = True
            break
        if not added:
            continue

        if n_seeds == 4:
            occupied = {seed.pos for seed in seeds} | {seed.witness for seed in seeds}
            fourth = _sample_extra_seed(
                H,
                W,
                colors[3],
                seeds,
                occupied,
                parity="odd",
                witness_mode="orth",
                require_boundary=True,
            )
            if fourth is None:
                continue
            seeds.append(fourth)
        return seeds, "v_tie"
    return None


def _add_filler_walls(H: int, W: int, seeds: list[SeedSpec], walls: set[tuple[int, int]], target: int) -> bool:
    occupied = {seed.pos for seed in seeds} | set(walls)
    for _ in range(1200):
        if len(walls) >= target:
            return True
        if walls and random.random() < 0.75:
            ar, ac = random.choice(tuple(walls))
            choices = [(ar + dr, ac + dc) for dr, dc in DIR4]
            random.shuffle(choices)
        else:
            choices = [(random.randrange(H), random.randrange(W))]
        for wr, wc in choices:
            if not in_bounds((H, W), wr, wc):
                continue
            if (wr, wc) in occupied:
                continue
            if any(manhattan((seed.r, seed.c), (wr, wc)) <= seed.d for seed in seeds):
                continue
            walls.add((wr, wc))
            occupied.add((wr, wc))
            break
    return len(walls) >= target


def _coverage_map(info: list[dict[str, object]]) -> dict[tuple[int, int], list[dict[str, object]]]:
    coverage: dict[tuple[int, int], list[dict[str, object]]] = {}
    for item in info:
        for cell in item["cells"]:
            coverage.setdefault((int(cell[0]), int(cell[1])), []).append(item)
    return coverage


def _validate_scene(grid: np.ndarray, seeds: list[SeedSpec], kind: str) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    seen_colors = sorted(int(grid[seed.r, seed.c]) for seed in seeds)
    if len(seeds) not in (3, 4) or len(set(seen_colors)) != len(seeds):
        return False

    walls = _wall_cells(grid)
    if not (WALL_COUNT_RANGE[0] <= len(walls) <= 20):
        return False

    if not _pairwise_seed_gap_ok(seeds):
        return False

    actual_ds = []
    for seed in seeds:
        d = _nearest_wall_distance(grid, seed.r, seed.c)
        if d != seed.d:
            return False
        if sum(manhattan(seed.pos, wall) == seed.d for wall in walls) != 1:
            return False
        if manhattan(seed.pos, seed.witness) != seed.d:
            return False
        actual_ds.append(d)

    if min(actual_ds) < 2 or len(set(actual_ds)) == 1:
        return False
    if not any(d % 2 == 1 for d in actual_ds):
        return False
    if not any(d % 2 == 0 for d in actual_ds):
        return False

    if sum(_witness_mode(seed) == "orth" for seed in seeds) < 2:
        return False
    if sum(_witness_mode(seed) == "ray" for seed in seeds) < 1:
        return False

    out, info = _solve_with_metadata(grid)
    if np.array_equal(out, grid):
        return False

    coverage = _coverage_map(info)
    overlap_cells = [cell for cell, items in coverage.items() if len(items) >= 2 and int(grid[cell]) == BLACK]
    if not overlap_cells:
        return False

    if kind == "cross":
        if not any(len({int(item["d"]) for item in coverage[cell]}) >= 2 for cell in overlap_cells):
            return False
    else:
        if not any(len({int(item["d"]) for item in coverage[cell]}) == 1 for cell in overlap_cells):
            return False

    stop_reasons = {str(leg["stop"]) for item in info for leg in item["legs"]}
    if "boundary" not in stop_reasons or "obstacle" not in stop_reasons:
        return False

    new_pixels = int(np.count_nonzero(out != grid))
    if new_pixels < 6:
        return False
    return True


def _build_grid_from_scene(H: int, W: int, seeds: list[SeedSpec]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    walls = {seed.witness for seed in seeds}
    target_walls = random.randint(*WALL_COUNT_RANGE)
    if not _add_filler_walls(H, W, seeds, walls, target_walls):
        raise RuntimeError("failed to place filler walls")

    for wr, wc in walls:
        grid[wr, wc] = GREY
    for seed in seeds:
        grid[seed.r, seed.c] = seed.color
    return grid


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(GRID_MIN, GRID_MAX)
    W = random.randint(GRID_MIN, GRID_MAX)
    colors = random.sample(list(SEED_COLORS), k=random.choice([3, 4]))
    n_seeds = len(colors)

    builders = [_build_cross_scene, _build_horizontal_tie_scene, _build_vertical_tie_scene]
    random.shuffle(builders)
    for builder in builders:
        built = builder(H, W, colors, n_seeds)
        if built is None:
            continue
        seeds, kind = built
        try:
            grid = _build_grid_from_scene(H, W, seeds)
        except RuntimeError:
            continue
        if _validate_scene(grid, seeds, kind):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid obstacle-distance ray scene")
