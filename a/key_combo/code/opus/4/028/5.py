import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6

GRID_CHOICES = tuple(range(15, 22))
GEN_RETRIES = 400
MASK_COUNT_CHOICES = (
    (1, 1, 1, 1),
    (2, 1, 1, 1),
    (2, 2, 1, 1),
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _singleton_coord(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    assert coords.shape == (1, 2), f"expected one pixel of color {color}"
    return int(coords[0, 0]), int(coords[0, 1])


def _diagonal_mask(shape: tuple[int, int], anchor: tuple[int, int]) -> np.ndarray:
    h, w = shape
    r0, c0 = anchor
    diff = r0 - c0
    summ = r0 + c0
    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        c = r - diff
        if 0 <= c < w:
            mask[r, c] = True
        c = summ - r
        if 0 <= c < w:
            mask[r, c] = True
    return mask


def _rotate90(center: tuple[int, int], coord: tuple[int, int], turns: int = 1) -> tuple[int, int]:
    cr, cc = center
    dr = coord[0] - cr
    dc = coord[1] - cc
    for _ in range(turns % 4):
        dr, dc = -dc, dr
    return cr + dr, cc + dc


def _orbit(center: tuple[int, int], coord: tuple[int, int]) -> list[tuple[int, int]]:
    return [_rotate90(center, coord, k) for k in range(4)]


def _canonical_orbit(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in cells))


def _infer_center_and_rays(
    shape: tuple[int, int], red_pt: tuple[int, int], blue_pt: tuple[int, int]
) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    red_mask = _diagonal_mask(shape, red_pt)
    blue_mask = _diagonal_mask(shape, blue_pt)
    overlap = np.argwhere(red_mask & blue_mask)
    assert overlap.shape == (1, 2), "expected exactly one red/blue diagonal intersection"
    center = int(overlap[0, 0]), int(overlap[0, 1])
    return center, red_mask, blue_mask


def _compose_output(
    shape: tuple[int, int],
    red_pt: tuple[int, int],
    blue_pt: tuple[int, int],
    green_cells: set[tuple[int, int]],
    mask_cells: set[tuple[int, int]],
) -> np.ndarray:
    center, red_mask, blue_mask = _infer_center_and_rays(shape, red_pt, blue_pt)
    completed_green = set(green_cells) | set(mask_cells)

    out = np.full(shape, BLACK, dtype=int)
    out[red_mask] = RED
    out[blue_mask] = BLUE
    if center not in completed_green:
        out[center] = YELLOW
    for r, c in completed_green:
        out[r, c] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the red/blue diagonal rays, use their unique crossing as a rotation center, and fill pink masks from that C4 orbit.

    Input structure:
    - Black background on a square grid.
    - Exactly one red singleton and one blue singleton appear away from each other.
    - Green pixels and pink masked pixels form a sparse partial pattern around the hidden center.

    Rule:
    - Extend both diagonals through the red pixel in red, and both diagonals through the blue pixel in blue.
    - The unique cell where a red diagonal and a blue diagonal overlap is the center; if no green occupies it, mark that cell yellow.
    - For every pink pixel, rotate it around that center by 90°, 180°, and 270°; if its orbit contains green, replace the pink pixel with green.
    - Green pixels, including newly filled ones, override any diagonal ray colors underneath them.

    Key insight:
    - The colored singletons are not the target objects; their diagonal crossing is a geometric locator that supplies the only center needed to complete the C4-symmetric pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1]

    red_pt = _singleton_coord(g, RED)
    blue_pt = _singleton_coord(g, BLUE)
    center, red_mask, blue_mask = _infer_center_and_rays(g.shape, red_pt, blue_pt)

    completed_green = set((int(r), int(c)) for r, c in np.argwhere(g == GREEN))
    for r, c in np.argwhere(g == PINK):
        rc = int(r), int(c)
        orbit = _orbit(center, rc)
        assert all(_in_bounds(g.shape, rr, cc) for rr, cc in orbit)
        assert any(int(g[rr, cc]) == GREEN for rr, cc in orbit), "every mask orbit needs visible green"
        completed_green.add(rc)

    out = np.full_like(g, BLACK)
    out[red_mask] = RED
    out[blue_mask] = BLUE
    if center not in completed_green:
        out[center] = YELLOW
    for r, c in completed_green:
        out[r, c] = GREEN
    return out


def _sample_ray_spec(n: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    for _ in range(500):
        center = (random.randint(4, n - 5), random.randint(4, n - 5))
        branch = random.randint(0, 1)
        s1 = random.choice((-1, 1))
        s2 = random.choice((-1, 1))
        max_a = min(center[0] if s1 < 0 else n - 1 - center[0], center[1] if branch == 1 and s1 > 0 else n - 1)
        max_b = min(center[0] if s2 < 0 else n - 1 - center[0], center[1] if branch == 0 and s2 > 0 else n - 1)
        # Use simple bounded search instead of dense closed-form constraints.
        for a in random.sample(list(range(2, 6)), k=4):
            for b in random.sample(list(range(2, 6)), k=4):
                if a == b:
                    continue
                if branch == 0:
                    red_pt = (center[0] + s1 * a, center[1] + s1 * a)
                    blue_pt = (center[0] + s2 * b, center[1] - s2 * b)
                    other = (center[0] + s1 * a + s2 * b, center[1] + s1 * a - s2 * b)
                else:
                    red_pt = (center[0] + s1 * a, center[1] - s1 * a)
                    blue_pt = (center[0] + s2 * b, center[1] + s2 * b)
                    other = (center[0] + s1 * a + s2 * b, center[1] - s1 * a + s2 * b)
                if not _in_bounds((n, n), *red_pt) or not _in_bounds((n, n), *blue_pt):
                    continue
                if red_pt[0] == blue_pt[0] or red_pt[1] == blue_pt[1]:
                    continue
                if abs(red_pt[0] - blue_pt[0]) == abs(red_pt[1] - blue_pt[1]):
                    continue
                if _in_bounds((n, n), *other):
                    continue
                overlap = np.argwhere(_diagonal_mask((n, n), red_pt) & _diagonal_mask((n, n), blue_pt))
                if overlap.shape != (1, 2):
                    continue
                if tuple(map(int, overlap[0])) != center:
                    continue
                return center, red_pt, blue_pt
    raise RuntimeError("failed to sample a valid ray configuration")


def _candidate_orbits(
    n: int,
    center: tuple[int, int],
    red_pt: tuple[int, int],
    blue_pt: tuple[int, int],
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    red_mask = _diagonal_mask((n, n), red_pt)
    blue_mask = _diagonal_mask((n, n), blue_pt)
    seen: set[tuple[tuple[int, int], ...]] = set()
    line_orbits: list[list[tuple[int, int]]] = []
    off_orbits: list[list[tuple[int, int]]] = []
    for dr in range(-5, 6):
        for dc in range(-5, 6):
            if (dr, dc) == (0, 0):
                continue
            base = (center[0] + dr, center[1] + dc)
            if not _in_bounds((n, n), *base):
                continue
            orbit = _orbit(center, base)
            if len(set(orbit)) != 4:
                continue
            if not all(_in_bounds((n, n), r, c) for r, c in orbit):
                continue
            if red_pt in orbit or blue_pt in orbit:
                continue
            if any(max(abs(r - red_pt[0]), abs(c - red_pt[1])) <= 1 for r, c in orbit):
                continue
            if any(max(abs(r - blue_pt[0]), abs(c - blue_pt[1])) <= 1 for r, c in orbit):
                continue
            key = _canonical_orbit(orbit)
            if key in seen:
                continue
            seen.add(key)
            on_line = any(red_mask[r, c] or blue_mask[r, c] for r, c in orbit)
            if on_line:
                line_orbits.append(list(key))
            else:
                off_orbits.append(list(key))
    random.shuffle(line_orbits)
    random.shuffle(off_orbits)
    return line_orbits, off_orbits


def _orbit_fits(selected: list[list[tuple[int, int]]], orbit: list[tuple[int, int]]) -> bool:
    for chosen in selected:
        for r0, c0 in chosen:
            for r1, c1 in orbit:
                if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                    return False
    return True


def _pick_orbits(
    n: int,
    center: tuple[int, int],
    red_pt: tuple[int, int],
    blue_pt: tuple[int, int],
) -> list[list[tuple[int, int]]]:
    line_orbits, off_orbits = _candidate_orbits(n, center, red_pt, blue_pt)
    want_line = bool(line_orbits) and random.random() < 0.45
    for _ in range(120):
        selected: list[list[tuple[int, int]]] = []
        pools = []
        if want_line:
            pools.append(line_orbits)
        pools.extend([off_orbits, off_orbits, off_orbits, line_orbits + off_orbits])
        for pool in pools:
            choices = pool[:]
            random.shuffle(choices)
            for orbit in choices:
                if orbit in selected:
                    continue
                if _orbit_fits(selected, orbit):
                    selected.append(orbit)
                    break
        if len(selected) == 4:
            if want_line and not any(any(_diagonal_mask((n, n), red_pt)[r, c] or _diagonal_mask((n, n), blue_pt)[r, c] for r, c in orbit) for orbit in selected):
                continue
            return selected
    raise RuntimeError("failed to pick four compatible orbits")


def _build_pattern(
    orbits: list[list[tuple[int, int]]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    mask_counts = list(random.choice(MASK_COUNT_CHOICES))
    random.shuffle(mask_counts)
    green_cells: set[tuple[int, int]] = set()
    mask_cells: set[tuple[int, int]] = set()
    for orbit, mask_count in zip(orbits, mask_counts):
        cells = orbit[:]
        random.shuffle(cells)
        mask_subset = set(cells[:mask_count])
        green_subset = set(cells[mask_count:])
        assert green_subset
        mask_cells |= mask_subset
        green_cells |= green_subset
    return green_cells, mask_cells


def _validate_semantics(
    grid: np.ndarray,
    center: tuple[int, int],
    red_pt: tuple[int, int],
    blue_pt: tuple[int, int],
    green_cells: set[tuple[int, int]],
    mask_cells: set[tuple[int, int]],
) -> None:
    assert grid.shape[0] == grid.shape[1]
    assert max(grid.shape) <= 30
    assert len(green_cells) in {10, 11, 12}
    assert len(mask_cells) in {4, 5, 6}
    assert _singleton_coord(grid, RED) == red_pt
    assert _singleton_coord(grid, BLUE) == blue_pt

    inferred_center, red_mask, blue_mask = _infer_center_and_rays(grid.shape, red_pt, blue_pt)
    assert inferred_center == center
    assert center not in green_cells
    assert center not in mask_cells

    occupied = green_cells | mask_cells
    orbit_keys = set()
    for cell in occupied:
        orbit = _orbit(center, cell)
        assert all(_in_bounds(grid.shape, r, c) for r, c in orbit)
        assert set(orbit) <= occupied
        orbit_keys.add(_canonical_orbit(orbit))
    assert len(orbit_keys) == 4

    for cell in mask_cells:
        orbit = _orbit(center, cell)
        assert any(pos in green_cells for pos in orbit)

    # Keep the auxiliary line-overlap rules active in some samples without making the grid crowded.
    line_overlap_count = sum(1 for r, c in occupied if red_mask[r, c] or blue_mask[r, c])
    assert line_overlap_count <= 4

    expected = _compose_output(grid.shape, red_pt, blue_pt, green_cells, mask_cells)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert not np.any(actual == PINK)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        n = random.choice(GRID_CHOICES)
        center, red_pt, blue_pt = _sample_ray_spec(n)
        try:
            orbits = _pick_orbits(n, center, red_pt, blue_pt)
        except RuntimeError:
            continue
        green_cells, mask_cells = _build_pattern(orbits)

        grid = np.full((n, n), BLACK, dtype=int)
        grid[red_pt] = RED
        grid[blue_pt] = BLUE
        for r, c in green_cells:
            grid[r, c] = GREEN
        for r, c in mask_cells:
            grid[r, c] = PINK

        try:
            _validate_semantics(grid, center, red_pt, blue_pt, green_cells, mask_cells)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid diagonal-center symmetry puzzle")
