from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
YELLOW = 4
HOLE = 6
MAX_RING = 4
GEN_ATTEMPTS = 400
GEN_SIZES = (9, 11, 13)


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _ring_color(dist: int) -> int:
    assert 1 <= dist <= MAX_RING
    return RED if dist % 2 == 1 else BLUE


def _find_seed(grid: np.ndarray) -> tuple[int, int]:
    seeds = np.argwhere(grid == YELLOW)
    assert len(seeds) == 1, "expected exactly one yellow seed"
    return int(seeds[0, 0]), int(seeds[0, 1])


def _rotation_orbit(center: tuple[int, int], cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    cr, cc = center
    dr = int(cell[0]) - cr
    dc = int(cell[1]) - cc
    orbit = (
        (cr + dr, cc + dc),
        (cr - dc, cc + dr),
        (cr - dr, cc - dc),
        (cr + dc, cc - dr),
    )
    assert len(set(orbit)) == 4, "non-center ring cells must have 4-point quarter-turn orbits"
    return orbit


def _full_pattern(shape: tuple[int, int], center: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    cr, cc = center
    out[cr, cc] = YELLOW
    for r in range(shape[0]):
        for c in range(shape[1]):
            dist = _chebyshev((r, c), center)
            if 1 <= dist <= MAX_RING:
                out[r, c] = _ring_color(dist)
    return out


def _orbit_map(shape: tuple[int, int], center: tuple[int, int]) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    orbits_by_ring: dict[int, list[tuple[tuple[int, int], ...]]] = {d: [] for d in range(1, MAX_RING + 1)}
    seen: set[tuple[tuple[int, int], ...]] = set()
    for r in range(shape[0]):
        for c in range(shape[1]):
            dist = _chebyshev((r, c), center)
            if not (1 <= dist <= MAX_RING):
                continue
            orbit = tuple(sorted(_rotation_orbit(center, (r, c))))
            if orbit in seen:
                continue
            seen.add(orbit)
            orbits_by_ring[dist].append(orbit)
    return orbits_by_ring


def _validate_input(grid: np.ndarray) -> tuple[int, int]:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2, "expected a 2D grid"
    h, w = map(int, grid.shape)
    assert h == w and h % 2 == 1, "expected an odd square grid"

    center = _find_seed(grid)
    assert center == (h // 2, w // 2), "seed must sit at the geometric center"
    assert min(center[0], center[1], h - 1 - center[0], w - 1 - center[1]) >= MAX_RING

    seen_holes = 0
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            dist = _chebyshev((r, c), center)
            if color == YELLOW:
                assert dist == 0, "yellow is reserved for the center seed"
                continue
            if color == BACKGROUND:
                continue
            if color == HOLE:
                assert 1 <= dist <= MAX_RING, "holes may only damage ring cells"
                seen_holes += 1
                continue
            assert color in (RED, BLUE), "unexpected foreground color"
            assert 1 <= dist <= MAX_RING, "ring colors must stay inside the four rings"
            assert color == _ring_color(dist), "visible ring cells must obey the alternating rule"

    return center


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the occluded concentric Chebyshev rings and complete any missing ring cells around the center seed.

    Input structure:
    - An odd square grid has black background and one yellow seed at the geometric center.
    - Cells at Chebyshev distances 1..4 from that seed belong to alternating red and blue square rings.
    - Some ring cells are missing as black gaps, and some visible ring cells are replaced by magenta occluders.

    Rule:
    - Treat magenta as damage rather than a final color.
    - Use each cell's Chebyshev distance from the yellow seed to determine its intended ring color: odd distances are red and even distances are blue.
    - For every magenta cell, verify that at least one cell in its 90-degree rotation orbit around the seed is visibly painted with that same intended color, then replace the magenta cell with the intended color and fill any black cells whose distance is at most 4 with their ring color.

    Key insight:
    - The final image is a four-ring square target determined jointly by distance-from-center and quarter-turn symmetry, so occlusions can be repaired and missing ring segments can be completed from those two constraints.
    """
    g = np.asarray(grid, dtype=int)
    center = _validate_input(g)

    for row, col in np.argwhere(g == HOLE):
        row = int(row)
        col = int(col)
        dist = _chebyshev((row, col), center)
        expected = _ring_color(dist)
        orbit = _rotation_orbit(center, (row, col))
        visible_support = 0
        for rr, cc in orbit:
            color = int(g[rr, cc])
            if color in (RED, BLUE):
                assert color == expected, "visible orbit mates must agree with the damaged ring color"
                visible_support += 1
        assert visible_support >= 1, "every hole must have a visible quarter-turn witness"

    return _full_pattern(g.shape, center)


def _render_visible_orbits(
    grid: np.ndarray,
    orbits_by_ring: dict[int, list[tuple[tuple[int, int], ...]]],
) -> tuple[np.ndarray, list[tuple[tuple[int, int], ...]]]:
    visible_orbits: list[tuple[tuple[int, int], ...]] = []
    mandatory_black_ring = random.randint(1, MAX_RING)
    for dist, orbits in orbits_by_ring.items():
        random.shuffle(orbits)
        visible_count = random.randint(1, len(orbits))
        if dist != mandatory_black_ring and visible_count == len(orbits) and random.random() < 0.7:
            visible_count -= 1
        if dist == mandatory_black_ring and visible_count == len(orbits):
            visible_count -= 1
        chosen = set(orbits[:visible_count])
        color = _ring_color(dist)
        for orbit in orbits:
            if orbit in chosen:
                visible_orbits.append(orbit)
                for r, c in orbit:
                    grid[r, c] = color
    assert np.count_nonzero(grid == BACKGROUND) > (grid.size - (2 * MAX_RING + 1) ** 2)
    return grid, visible_orbits


def _choose_hole_positions(
    visible_orbits: list[tuple[tuple[int, int], ...]],
    center: tuple[int, int],
) -> list[tuple[int, int]]:
    odd_orbits = [orbit for orbit in visible_orbits if _chebyshev(orbit[0], center) % 2 == 1]
    even_orbits = [orbit for orbit in visible_orbits if _chebyshev(orbit[0], center) % 2 == 0]
    assert odd_orbits and even_orbits

    total_holes = random.randint(3, 6)
    selected_orbits: list[tuple[tuple[int, int], ...]] = [random.choice(odd_orbits), random.choice(even_orbits)]
    remaining = [orbit for orbit in visible_orbits if orbit not in selected_orbits]
    random.shuffle(remaining)
    max_orbits = min(total_holes, 4)
    extra_count = random.randint(0, min(len(remaining), max_orbits - len(selected_orbits)))
    selected_orbits.extend(remaining[:extra_count])

    hole_counts = {orbit: 1 for orbit in selected_orbits}
    remaining_holes = total_holes - len(selected_orbits)
    while remaining_holes > 0:
        candidates = [orbit for orbit in selected_orbits if hole_counts[orbit] < len(orbit) - 1]
        assert candidates
        orbit = random.choice(candidates)
        hole_counts[orbit] += 1
        remaining_holes -= 1

    holes: list[tuple[int, int]] = []
    for orbit in selected_orbits:
        count = hole_counts[orbit]
        cells = list(orbit)
        random.shuffle(cells)
        holes.extend(cells[:count])
    return holes


def _nondegenerate(grid: np.ndarray) -> bool:
    center = _find_seed(grid)
    hole_positions = [tuple(map(int, rc)) for rc in np.argwhere(grid == HOLE)]
    if not (3 <= len(hole_positions) <= 6):
        return False

    has_black_gap = False
    visible_rings = {d: 0 for d in range(1, MAX_RING + 1)}
    hole_parities = set()
    support_sizes: list[int] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            dist = _chebyshev((r, c), center)
            if 1 <= dist <= MAX_RING:
                if grid[r, c] == BACKGROUND:
                    has_black_gap = True
                elif grid[r, c] in (RED, BLUE):
                    visible_rings[dist] += 1

    for hole in hole_positions:
        dist = _chebyshev(hole, center)
        hole_parities.add(dist % 2)
        expected = _ring_color(dist)
        support = sum(int(grid[r, c] == expected) for r, c in _rotation_orbit(center, hole))
        if support < 1:
            return False
        support_sizes.append(support)

    return (
        has_black_gap
        and all(count > 0 for count in visible_rings.values())
        and hole_parities == {0, 1}
        and max(support_sizes) <= 3
        and min(support_sizes) >= 1
    )


def _generate_once() -> np.ndarray | None:
    size = random.choice(GEN_SIZES)
    center = (size // 2, size // 2)
    grid = np.zeros((size, size), dtype=int)
    grid[center] = YELLOW

    orbits_by_ring = _orbit_map(grid.shape, center)
    for dist, orbits in orbits_by_ring.items():
        if not orbits:
            return None
        if len(orbits) >= 2:
            random.shuffle(orbits)
            orbits_by_ring[dist] = orbits

    grid, visible_orbits = _render_visible_orbits(grid, orbits_by_ring)
    if len(visible_orbits) < 2:
        return None

    holes = _choose_hole_positions(visible_orbits, center)
    for r, c in holes:
        grid[r, c] = HOLE

    try:
        out = main(grid)
    except Exception:
        return None
    if not np.array_equal(out, _full_pattern(grid.shape, center)):
        return None
    if not _nondegenerate(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chebyshev-ring occlusion instance")
