import random

import numpy as np


BG = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_H_BOUNDS = (8, 18)
GRID_W_BOUNDS = (8, 18)
SEED_COUNT_BOUNDS = (2, 6)
MIN_GROWTH_LEN = 2
MAX_OCCUPANCY_RATIO = 0.42
GEN_ATTEMPTS = 400


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _distance_to_edge(h: int, w: int, r: int, c: int, dr: int, dc: int) -> int:
    if dr == -1:
        return r
    if dr == 1:
        return h - 1 - r
    if dc == -1:
        return c
    return w - 1 - c


def _growth_cells(h: int, w: int, r: int, c: int, dr: int, dc: int) -> list[tuple[int, int]]:
    dist = _distance_to_edge(h, w, r, c, dr, dc)
    return [(r + dr * step, c + dc * step) for step in range(1, dist + 1)]


def _neighbors8(r: int, c: int) -> set[tuple[int, int]]:
    return {
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
    }


def _choose_directions(count: int) -> list[tuple[int, int]]:
    order = list(DIRS)
    random.shuffle(order)
    dirs = [order[i % len(order)] for i in range(count)]
    random.shuffle(dirs)
    return dirs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed away from its adjacent grey pointer until the grid edge.

    Input structure:
    - Black background.
    - Colored singleton seed pixels.
    - Each seed has one grey 4-neighbor that points from the opposite side.

    Rule:
    - The grey neighbor marks the side behind the seed.
    - The seed grows in the direction from grey to seed.
    - Growth length is the number of cells from the seed to the edge in that direction.
    - Grey markers are erased.
    - Seeds remain in place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    seed_positions = [
        (int(r), int(c), int(g[r, c]))
        for r, c in np.argwhere((g != BG) & (g != GREY))
    ]
    assert seed_positions, "expected at least one seed"

    grey_positions = {
        (int(r), int(c))
        for r, c in np.argwhere(g == GREY)
    }

    seed_map = {(r, c): color for r, c, color in seed_positions}
    grey_usage: dict[tuple[int, int], int] = {}
    planned: dict[tuple[int, int], int] = {}

    for r, c, color in seed_positions:
        markers = []
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if _in_bounds(h, w, nr, nc) and int(g[nr, nc]) == GREY:
                markers.append((nr, nc))
        assert len(markers) == 1, "each seed must have exactly one adjacent grey marker"

        mr, mc = markers[0]
        grey_usage[(mr, mc)] = grey_usage.get((mr, mc), 0) + 1

        grow_dr = r - mr
        grow_dc = c - mc
        length = _distance_to_edge(h, w, r, c, grow_dr, grow_dc)
        assert length >= 1, "each seed must have room to grow toward the edge"

        for rr, cc in _growth_cells(h, w, r, c, grow_dr, grow_dc):
            if (rr, cc) in seed_map:
                assert seed_map[(rr, cc)] == color, "growth lines cannot cross another seed"
            if (rr, cc) in grey_positions:
                raise AssertionError("growth lines cannot cross another grey marker")
            prev = planned.get((rr, cc))
            assert prev in (None, color), "growth lines cannot overlap with different colors"
            planned[(rr, cc)] = color

    assert set(grey_usage) == grey_positions, "all grey pixels must serve as direction markers"
    assert all(count == 1 for count in grey_usage.values()), "each grey marker must belong to one seed"

    out = g.copy()
    out[out == GREY] = BG
    for (r, c), color in planned.items():
        out[r, c] = color
    return out


def _placement_for_seed(
    h: int,
    w: int,
    r: int,
    c: int,
    direction: tuple[int, int],
) -> tuple[tuple[int, int], list[tuple[int, int]], set[tuple[int, int]]] | None:
    dr, dc = direction
    marker = (r - dr, c - dc)
    if not _in_bounds(h, w, marker[0], marker[1]):
        return None

    dist = _distance_to_edge(h, w, r, c, dr, dc)
    if dist < MIN_GROWTH_LEN:
        return None

    grown = _growth_cells(h, w, r, c, dr, dc)
    footprint = {(r, c), marker, *grown}
    return marker, grown, footprint


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_BOUNDS)
    w = random.randint(*GRID_W_BOUNDS)
    max_seed_count = min(SEED_COUNT_BOUNDS[1], max(SEED_COUNT_BOUNDS[0], (h * w) // 30))
    seed_count = random.randint(SEED_COUNT_BOUNDS[0], max_seed_count)

    directions = _choose_directions(seed_count)
    colors = random.sample(SEED_COLORS, seed_count)

    grid = np.full((h, w), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()
    anchor_halo: set[tuple[int, int]] = set()
    placements: list[tuple[int, int, int, int, int, int]] = []

    for color, direction in zip(colors, directions):
        dr, dc = direction
        candidates: list[tuple[int, int, tuple[int, int], list[tuple[int, int]], set[tuple[int, int]]]] = []

        for r in range(h):
            for c in range(w):
                placement = _placement_for_seed(h, w, r, c, direction)
                if placement is None:
                    continue
                marker, grown, footprint = placement
                if footprint & occupied:
                    continue
                if footprint & anchor_halo:
                    continue
                candidates.append((r, c, marker, grown, footprint))

        if not candidates:
            return None

        random.shuffle(candidates)
        candidates.sort(key=lambda item: abs(len(item[3]) - ((h + w) // 6)))
        r, c, marker, grown, footprint = random.choice(candidates[: min(10, len(candidates))])

        grid[r, c] = color
        grid[marker] = GREY
        occupied |= footprint

        for anchor in ((r, c), marker):
            for ar, ac in _neighbors8(*anchor):
                if _in_bounds(h, w, ar, ac):
                    anchor_halo.add((ar, ac))

        placements.append((r, c, marker[0], marker[1], dr, dc))

    if len(occupied) > int(MAX_OCCUPANCY_RATIO * h * w):
        return None

    # Validate the structural assumptions that main() relies on.
    out = main(grid)
    assert out.shape == grid.shape
    assert not np.any(out == GREY)
    assert np.count_nonzero(grid == GREY) == seed_count

    for r, c, mr, mc, dr, dc in placements:
        assert grid[r, c] in SEED_COLORS
        assert grid[mr, mc] == GREY
        expected_dr = r - mr
        expected_dc = c - mc
        assert (expected_dr, expected_dc) == (dr, dc)
        dist = _distance_to_edge(h, w, r, c, dr, dc)
        for rr, cc in _growth_cells(h, w, r, c, dr, dc):
            assert out[rr, cc] == grid[r, c]
        assert dist >= MIN_GROWTH_LEN

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid grid")
