from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

CLASS_TO_COLOR = {1: ORANGE, 2: BLUE, 3: RED}

# Canonical half-shapes. Each one is a connected green fragment whose
# point-symmetric completion has 2/4/6 raw L-windows, corresponding to
# bend classes 1/2/3 after collapsing symmetric pairs.
TEMPLATES: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    1: (
        ((-3, 1), (-3, 2), (-2, 1)),
        ((-4, 2), (-4, 3), (-3, 2)),
        ((-4, 3), (-4, 4), (-3, 4)),
    ),
    2: (
        ((-4, 1), (-4, 2), (-4, 3), (-3, 2)),
        ((-4, 1), (-3, 1), (-3, 2), (-2, 1)),
        ((-4, 2), (-3, 2), (-3, 3), (-2, 2)),
    ),
    3: (
        ((-4, 1), (-4, 2), (-3, 2), (-3, 3), (-2, 2)),
        ((-4, 1), (-3, 1), (-3, 2), (-2, 2), (-2, 3)),
        ((-4, 2), (-3, 1), (-3, 2), (-3, 3), (-2, 1)),
    ),
}

SCENE_LAYOUTS: tuple[tuple[tuple[int, tuple[tuple[int, int], ...]], ...], ...] = (
    (
        (1, ((-3, 1), (-3, 2), (-2, 1))),
        (2, ((1, 4), (2, 3), (2, 4), (3, 4))),
        (3, ((1, -4), (2, -4), (2, -3), (2, -2), (3, -3))),
    ),
    (
        (1, ((-3, 1), (-3, 2), (-2, 1))),
        (2, ((3, 2), (4, 1), (4, 2), (4, 3))),
        (3, ((1, -4), (2, -4), (2, -3), (2, -2), (3, -3))),
    ),
    (
        (1, ((-3, 1), (-3, 2), (-2, 1))),
        (2, ((3, 2), (4, 1), (4, 2), (4, 3))),
        (3, ((-3, -3), (-2, -4), (-2, -3), (-1, -3), (-1, -2))),
    ),
    (
        (1, ((-3, 1), (-3, 2), (-2, 1))),
        (2, ((-3, -4), (-2, -4), (-2, -3), (-1, -4))),
        (3, ((2, 2), (3, 2), (3, 3), (4, 1), (4, 2))),
    ),
    (
        (1, ((-3, 1), (-3, 2), (-2, 1))),
        (1, ((1, 2), (1, 3), (2, 3))),
        (2, ((-4, -2), (-3, -3), (-3, -2), (-2, -2))),
        (3, ((1, -4), (2, -4), (2, -3), (2, -2), (3, -3))),
    ),
    (
        (1, ((-3, 1), (-3, 2), (-2, 1))),
        (1, ((2, 1), (3, 1), (3, 2))),
        (2, ((-3, -3), (-2, -4), (-2, -3), (-2, -2))),
        (3, ((1, -4), (2, -4), (2, -3), (2, -2), (3, -3))),
    ),
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, color: int) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            comp: set[tuple[int, int]] = set()
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                comp.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _reflect(coord: tuple[int, int], pivot: tuple[int, int]) -> tuple[int, int]:
    r, c = coord
    pr, pc = pivot
    return 2 * pr - r, 2 * pc - c


def _completed_shape(
    component: set[tuple[int, int]],
    pivot: tuple[int, int],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    full = set(component)
    for cell in component:
        rr, cc = _reflect(cell, pivot)
        if _in_bounds(shape, rr, cc):
            full.add((rr, cc))
    return full


def _raw_bend_count(cells: set[tuple[int, int]]) -> int:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    total = 0
    for r in range(min(rows), max(rows)):
        for c in range(min(cols), max(cols)):
            count = 0
            for dr in (0, 1):
                for dc in (0, 1):
                    if (r + dr, c + dc) in cells:
                        count += 1
            if count == 3:
                total += 1
    return total


def _bend_class(cells: set[tuple[int, int]]) -> int:
    raw = _raw_bend_count(cells)
    if raw % 2 == 0 and raw // 2 in CLASS_TO_COLOR:
        return raw // 2
    if raw in CLASS_TO_COLOR:
        return raw
    raise AssertionError(f"unexpected bend count: {raw}")


def _transform_coords(
    coords: tuple[tuple[int, int], ...],
    rotation: int,
    mirror: bool,
) -> tuple[tuple[int, int], ...]:
    transformed: list[tuple[int, int]] = []
    for r, c in coords:
        if mirror:
            c = -c
        for _ in range(rotation % 4):
            r, c = c, -r
        transformed.append((r, c))
    return tuple(sorted(transformed))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each green half-shape by 180-degree reflection around the grey pivot, then recolor by bend count.

    Input structure:
    - A black background contains exactly one grey pivot pixel.
    - Several separated green 4-connected fragments appear around that pivot.
    - Each green fragment is one preserved half of a point-symmetric bent shape.

    Rule:
    - For every green pixel, reflect it through the grey pivot by a 180-degree turn and
      add the reflected green pixel when it stays in bounds.
    - Treat each original green fragment together with its reflected pixels as one completed shape.
    - Count the completed shape's L-shaped 2x2 bends and recolor bend classes 1, 2, 3
      to orange, blue, and red respectively, then remove the grey pivot.

    Key insight:
    - The bend topology is not readable from the half-fragment alone; the fragment must first
      be completed through point symmetry before its bend class can be determined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pivots = np.argwhere(g == GREY)
    assert len(pivots) == 1, "expected exactly one grey pivot"
    pivot = tuple(map(int, pivots[0]))

    output = np.full_like(g, BACKGROUND)
    for component in _find_components(g, GREEN):
        full_shape = _completed_shape(component, pivot, g.shape)
        color = CLASS_TO_COLOR[_bend_class(full_shape)]
        for r, c in full_shape:
            output[r, c] = color
    return output


def _min_chebyshev_distance(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    return min(max(abs(r1 - r2), abs(c1 - c2)) for r1, c1 in a for r2, c2 in b)


def _placeable(
    candidate_half: set[tuple[int, int]],
    pivot: tuple[int, int],
    shape: tuple[int, int],
    placed_halves: list[set[tuple[int, int]]],
    occupied_full: set[tuple[int, int]],
) -> bool:
    if any(not _in_bounds(shape, r, c) for r, c in candidate_half):
        return False
    if pivot in candidate_half:
        return False

    candidate_full = _completed_shape(candidate_half, pivot, shape)
    if any(not _in_bounds(shape, r, c) for r, c in candidate_full):
        return False
    if occupied_full & candidate_full:
        return False

    for other_half in placed_halves:
        if _min_chebyshev_distance(candidate_half, other_half) < 3:
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(15, 23)
    w = random.randint(15, 23)
    pr = random.randint(5, h - 6)
    pc = random.randint(5, w - 6)
    pivot = (pr, pc)
    rotation = random.randint(0, 3)
    mirror = bool(random.getrandbits(1))
    layout = random.choice(SCENE_LAYOUTS)

    placed_halves: list[set[tuple[int, int]]] = []
    occupied_full: set[tuple[int, int]] = set()
    for bend_class, coords in layout:
        rel = _transform_coords(coords, rotation=rotation, mirror=mirror)
        candidate_half = {(pr + dr, pc + dc) for dr, dc in rel}
        if not _placeable(candidate_half, pivot, (h, w), placed_halves, occupied_full):
            return None
        placed_halves.append(candidate_half)
        occupied_full |= _completed_shape(candidate_half, pivot, (h, w))

    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[pivot] = GREY
    for half in placed_halves:
        for r, c in half:
            grid[r, c] = GREEN

    # Make sure the generated fragments stayed separated and the solver recovers all three colors.
    if len(_find_components(grid, GREEN)) != len(placed_halves):
        return None
    out = main(grid)
    colors = set(map(int, np.unique(out[out != BACKGROUND])))
    if colors != {BLUE, RED, ORANGE}:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
