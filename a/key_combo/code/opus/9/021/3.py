from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
MARKER = 5
BOUNDARY = 7
SEED_PALETTE = (1, 2, 3, 4, 6)
NEUTRAL_OWNER = -2
BOUNDARY_OWNER = -1

GRID_ROWS = (13, 17)
GRID_COLS = (13, 17)
SEED_COUNT_RANGE = (3, 5)
MARKER_COUNT_RANGE = (4, 8)
MAX_GENERATION_ATTEMPTS = 400

SEED_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _component_cells(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    pts = {tuple(map(int, rc)) for rc in np.argwhere(grid == int(color))}
    comps: list[list[tuple[int, int]]] = []
    while pts:
        start = pts.pop()
        stack = [start]
        comp = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in pts:
                    pts.remove((nr, nc))
                    stack.append((nr, nc))
                    comp.append((nr, nc))
        comps.append(sorted(comp))
    return comps


def _extract_seed_regions(grid: np.ndarray) -> list[dict]:
    colors = sorted(int(c) for c in np.unique(grid) if int(c) not in (BG, MARKER, BOUNDARY))
    assert 3 <= len(colors) <= 5, "expected 3 to 5 distinct seed colors"

    regions = []
    for color in colors:
        comps = _component_cells(grid, color)
        assert len(comps) == 1, "each seed color must form exactly one connected seed object"
        cells = comps[0]
        assert 2 <= len(cells) <= 3, "each seed object must have size 2 or 3"
        regions.append({"color": color, "cells": cells})
    return regions


def _solve_with_meta(grid: np.ndarray, *, require_markers: bool = True) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert BOUNDARY not in set(map(int, np.unique(g))), "input should not already contain boundary pixels"

    regions = _extract_seed_regions(g)
    owner = np.full(g.shape, NEUTRAL_OWNER, dtype=int)
    for rid, region in enumerate(regions):
        for r, c in region["cells"]:
            owner[r, c] = rid

    while True:
        claims: dict[tuple[int, int], set[int]] = {}
        for rid in range(len(regions)):
            for r, c in np.argwhere(owner == rid):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if _in_bounds(g.shape, nr, nc) and owner[nr, nc] == NEUTRAL_OWNER and int(g[nr, nc]) in (BG, MARKER):
                        claims.setdefault((nr, nc), set()).add(rid)
        if not claims:
            break

        for (r, c), claimers in claims.items():
            owner[r, c] = next(iter(claimers)) if len(claimers) == 1 else BOUNDARY_OWNER

    assert not np.any(owner == NEUTRAL_OWNER), "growth should cover the whole grid"

    grown = np.full(g.shape, BG, dtype=int)
    for rid, region in enumerate(regions):
        grown[owner == rid] = int(region["color"])
    grown[owner == BOUNDARY_OWNER] = BOUNDARY

    areas = {rid: int(np.sum(owner == rid)) for rid in range(len(regions))}
    boundary_cells = [tuple(map(int, rc)) for rc in np.argwhere(owner == BOUNDARY_OWNER)]
    adjacency = {rid: 0 for rid in range(len(regions))}
    for r, c in boundary_cells:
        seen: set[int] = set()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(owner.shape, nr, nc) and int(owner[nr, nc]) >= 0:
                seen.add(int(owner[nr, nc]))
        for rid in seen:
            adjacency[rid] += 1

    marker_positions = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER)]
    votes = Counter({rid: 0 for rid in range(len(regions))})
    for r, c in marker_positions:
        rid = int(owner[r, c])
        assert rid >= 0, "markers may not end up on a collision boundary"
        votes[rid] += 1

    meta = {
        "regions": regions,
        "owner": owner,
        "grown": grown,
        "areas": areas,
        "boundary_cells": boundary_cells,
        "boundary_adjacency": adjacency,
    }
    if not marker_positions:
        assert not require_markers, "expected at least one marker"
        return meta

    top_vote = max(votes.values())
    winners = [rid for rid, count in votes.items() if count == top_vote]
    assert len(winners) == 1, "vote winner must be unique"
    winner = winners[0]

    absorb = np.zeros_like(owner, dtype=bool)
    for r, c in boundary_cells:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(owner.shape, nr, nc) and int(owner[nr, nc]) == winner:
                absorb[r, c] = True
                break

    final = grown.copy()
    final[absorb] = int(regions[winner]["color"])
    meta.update(
        final=final,
        votes=votes,
        winner=winner,
        absorbed_count=int(absorb.sum()),
        remaining_boundary_count=int(np.sum(final == BOUNDARY)),
    )
    return meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow all seed regions synchronously, let captured gray markers vote for a winner, then let that winner absorb adjacent collision boundary cells.

    Input structure:
    - A black background contains 3 to 5 small seed objects, each 4-connected, size 2 or 3, and a different non-gray color.
    - Several gray single-pixel markers sit on background cells and are not initially 4-adjacent to any seed.
    - No orange boundary pixels are present in the input; orange appears only after simultaneous growth collisions.

    Rule:
    - All seed regions expand step by step through 4-neighbor background and gray cells at the same time.
    - If two or more regions first reach a cell in the same step, that cell becomes an orange boundary and belongs to no region.
    - After growth finishes, each gray marker votes for the region that covered it, the unique highest-vote region wins, and every orange boundary cell 4-adjacent to that winner is recolored to the winner's color.

    Key insight:
    - The geometry alone creates the collision boundary, but the gray markers decide which region gets to redraw only its own adjacent slice of that boundary.
    """
    return _solve_with_meta(np.asarray(grid, dtype=int))["final"]


def _seed_separation_ok(seed_cells: list[list[tuple[int, int]]], new_cells: list[tuple[int, int]]) -> bool:
    for cell in new_cells:
        for other_seed in seed_cells:
            for other in other_seed:
                if _manhattan(cell, other) < 4:
                    return False
    return True


def _make_seed_grid(rng: random.Random) -> tuple[np.ndarray, list[list[tuple[int, int]]]] | None:
    rows = rng.randint(*GRID_ROWS)
    cols = rng.randint(*GRID_COLS)
    seed_count = rng.randint(*SEED_COUNT_RANGE)
    colors = rng.sample(list(SEED_PALETTE), seed_count)

    grid = np.full((rows, cols), BG, dtype=int)
    seed_cells: list[list[tuple[int, int]]] = []
    shape_choices = list(SEED_SHAPES)
    shape_weights = [1, 1, 2, 2, 3, 3, 3, 3]

    for color in colors:
        placed = False
        for _ in range(500):
            shape = rng.choices(shape_choices, weights=shape_weights, k=1)[0]
            h = max(r for r, _c in shape) + 1
            w = max(c for _r, c in shape) + 1
            r0 = rng.randrange(rows - h + 1)
            c0 = rng.randrange(cols - w + 1)
            cells = [(r0 + r, c0 + c) for r, c in shape]
            if any(int(grid[r, c]) != BG for r, c in cells):
                continue
            if not _seed_separation_ok(seed_cells, cells):
                continue
            for r, c in cells:
                grid[r, c] = int(color)
            seed_cells.append(cells)
            placed = True
            break
        if not placed:
            return None
    return grid, seed_cells


def _seed_adjacency_mask(shape: tuple[int, int], seed_cells: list[list[tuple[int, int]]]) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for cells in seed_cells:
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(shape, nr, nc):
                    blocked.add((nr, nc))
    return blocked


def _boundary_distances(owner: np.ndarray) -> dict[tuple[int, int], int]:
    boundary = [tuple(map(int, rc)) for rc in np.argwhere(owner == BOUNDARY_OWNER)]
    dists: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(owner >= 0):
        cell = (int(r), int(c))
        dists[cell] = min(_manhattan(cell, b) for b in boundary)
    return dists


def _enumerate_vote_vectors(
    n_regions: int,
    total: int,
    capacities: list[int],
    winner: int,
    require_all_distinct: bool,
    support_min: int,
) -> list[tuple[int, ...]]:
    vectors: list[tuple[int, ...]] = []

    def rec(idx: int, remaining: int, current: list[int]) -> None:
        if idx == n_regions:
            if remaining != 0:
                return
            if sum(v > 0 for v in current) < support_min:
                return
            if current[winner] < 2:
                return
            if max(current) != current[winner] or current.count(current[winner]) != 1:
                return
            if require_all_distinct and len(set(current)) != n_regions:
                return
            vectors.append(tuple(current))
            return

        upper = min(capacities[idx], remaining)
        for value in range(upper + 1):
            current.append(value)
            rec(idx + 1, remaining - value, current)
            current.pop()

    rec(0, total, [])
    return vectors


def _pick_marker_positions(
    rng: random.Random,
    region_cells: dict[int, list[tuple[int, int]]],
    vote_vector: tuple[int, ...],
    winner: int,
    boundary_dist: dict[tuple[int, int], int],
) -> list[tuple[int, int]] | None:
    positive_regions = [rid for rid, count in enumerate(vote_vector) if count > 0]
    order = sorted(positive_regions, key=lambda rid: (rid != winner, -vote_vector[rid]))

    for _ in range(120):
        chosen: list[tuple[int, int]] = []
        used: set[tuple[int, int]] = set()
        ok = True

        for rid in order:
            count = vote_vector[rid]
            cells = region_cells[rid][:]
            rng.shuffle(cells)
            if rid == winner:
                cells.sort(key=lambda cell: (boundary_dist[cell], rng.random()))
            else:
                cells.sort(key=lambda cell: (-boundary_dist[cell], rng.random()))

            picked = 0
            for cell in cells:
                if any(_manhattan(cell, other) == 1 for other in used):
                    continue
                chosen.append(cell)
                used.add(cell)
                picked += 1
                if picked == count:
                    break
            if picked != count:
                ok = False
                break

        if not ok:
            continue

        dists = [boundary_dist[cell] for cell in chosen]
        if min(dists) > 2:
            continue
        if max(dists) < 3 and any(dist >= 3 for dist in boundary_dist.values()):
            continue
        return chosen
    return None


def _validate_generated(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    try:
        meta = _solve_with_meta(g)
        out = main(g)
    except AssertionError:
        return False

    if not np.array_equal(out, meta["final"]):
        return False

    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER)]
    if not (MARKER_COUNT_RANGE[0] <= len(markers) <= MARKER_COUNT_RANGE[1]):
        return False

    seed_regions = meta["regions"]
    if not (SEED_COUNT_RANGE[0] <= len(seed_regions) <= SEED_COUNT_RANGE[1]):
        return False

    all_seed_cells = [region["cells"] for region in seed_regions]
    for i, cells_a in enumerate(all_seed_cells):
        for cells_b in all_seed_cells[i + 1 :]:
            if min(_manhattan(a, b) for a in cells_a for b in cells_b) < 4:
                return False

    for r, c in markers:
        if any((nr, nc) in {cell for region in all_seed_cells for cell in region} for nr, nc in _neighbors4(r, c)):
            return False

    vote_values = [int(meta["votes"][rid]) for rid in range(len(seed_regions))]
    if sum(v > 0 for v in vote_values) < 2:
        return False
    if len(seed_regions) % 2 == 0 and len(set(vote_values)) != len(vote_values):
        return False

    if meta["absorbed_count"] < 1:
        return False
    if meta["remaining_boundary_count"] < 1:
        return False
    if np.array_equal(meta["grown"], meta["final"]):
        return False

    owner = meta["owner"]
    boundary_dist = _boundary_distances(owner)
    marker_dists = [boundary_dist[(r, c)] for r, c in markers]
    if min(marker_dists) > 2:
        return False

    max_area = max(meta["areas"].values())
    non_max_with_boundary = [
        rid
        for rid, area in meta["areas"].items()
        if area < max_area and meta["boundary_adjacency"][rid] > 0
    ]
    if non_max_with_boundary and meta["winner"] not in non_max_with_boundary:
        return False

    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    base = _make_seed_grid(rng)
    if base is None:
        return None

    seed_grid, seed_cells = base
    try:
        seed_meta = _solve_with_meta(seed_grid.copy(), require_markers=False)
    except AssertionError:
        return None
    if len(seed_meta["boundary_cells"]) < 4:
        return None

    max_area = max(seed_meta["areas"].values())
    preferred_winners = [
        rid
        for rid, area in seed_meta["areas"].items()
        if area < max_area and seed_meta["boundary_adjacency"][rid] > 0
    ]
    if not preferred_winners:
        return None

    blocked = _seed_adjacency_mask(seed_grid.shape, seed_cells)
    boundary_dist = _boundary_distances(seed_meta["owner"])
    region_cells = {rid: [] for rid in range(len(seed_meta["regions"]))}
    for r in range(seed_grid.shape[0]):
        for c in range(seed_grid.shape[1]):
            rid = int(seed_meta["owner"][r, c])
            if rid < 0:
                continue
            if int(seed_grid[r, c]) != BG:
                continue
            if (r, c) in blocked:
                continue
            region_cells[rid].append((r, c))

    if any(not cells for cells in region_cells.values()):
        return None

    totals = list(range(MARKER_COUNT_RANGE[0], MARKER_COUNT_RANGE[1] + 1))
    rng.shuffle(totals)
    winner_options = preferred_winners[:]
    rng.shuffle(winner_options)

    for total_markers in totals:
        support_min = rng.randint(2, min(4, len(region_cells)))
        require_all_distinct = len(region_cells) % 2 == 0
        for winner in winner_options:
            capacities = [len(region_cells[rid]) for rid in range(len(region_cells))]
            vectors = _enumerate_vote_vectors(
                n_regions=len(region_cells),
                total=total_markers,
                capacities=capacities,
                winner=winner,
                require_all_distinct=require_all_distinct,
                support_min=support_min,
            )
            rng.shuffle(vectors)
            for vote_vector in vectors[:90]:
                markers = _pick_marker_positions(rng, region_cells, vote_vector, winner, boundary_dist)
                if markers is None:
                    continue

                grid = seed_grid.copy()
                for r, c in markers:
                    grid[r, c] = MARKER
                if _validate_generated(grid):
                    return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-3 input")
