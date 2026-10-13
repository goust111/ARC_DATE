from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

AREA_TO_COLOR = {
    1: GREEN,
    2: BLUE,
    3: YELLOW,
}

SIZE1_SHAPES = [((0, 0),)]
SIZE2_SHAPES = [
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
]
SIZE3_SHAPES = [
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
]
SIZE_TO_SHAPES = {
    1: SIZE1_SHAPES,
    2: SIZE2_SHAPES,
    3: SIZE3_SHAPES,
}

COUNT_OPTIONS = [
    (1, 2, 3),
    (1, 3, 2),
    (2, 1, 3),
    (2, 3, 1),
    (3, 1, 2),
    (3, 2, 1),
    (1, 2, 4),
    (1, 4, 2),
    (2, 1, 4),
    (2, 4, 1),
    (4, 1, 2),
    (4, 2, 1),
]
COUNT_BAG: list[tuple[int, int, int]] = []
MAX_GENERATION_ATTEMPTS = 400


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def find_components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    seen = np.zeros(grid.shape, dtype=bool)
    components: list[np.ndarray] = []
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def draw_square_border(
    grid: np.ndarray,
    center: tuple[int, int],
    layer: int,
    color: int,
    blocked: np.ndarray,
) -> None:
    r, c = center
    if layer == 0:
        if not blocked[r, c]:
            grid[r, c] = color
        return

    r0, r1 = r - layer, r + layer
    c0, c1 = c - layer, c + layer

    for cc in range(c0, c1 + 1):
        if in_bounds(grid.shape, r0, cc) and not blocked[r0, cc]:
            grid[r0, cc] = color
        if in_bounds(grid.shape, r1, cc) and not blocked[r1, cc]:
            grid[r1, cc] = color
    for rr in range(r0, r1 + 1):
        if in_bounds(grid.shape, rr, c0) and not blocked[rr, c0]:
            grid[rr, c0] = color
        if in_bounds(grid.shape, rr, c1) and not blocked[rr, c1]:
            grid[rr, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor black components by area and grow a seed-centered concentric square histogram from the same counts.

    Input structure:
    - A grey background contains several black 4-connected components and exactly one red seed pixel.
    - Every black component has area 1, 2, or 3, and the seed is inside the grid rather than on the border.
    - The seed is not 4-adjacent to any black component, and unaffected cells remain grey.

    Rule:
    - Count black 4-connected components by area: let A1, A2, and A3 be the numbers of size-1, size-2, and size-3 components.
    - Recolor size-1 components green, size-2 components blue, and size-3 components yellow.
    - Around the red seed, keep layer 0 as the red center, then draw square-border layers 1..A1 in green, the next A2 layers in blue, and the next A3 layers in yellow.
    - If a square-border cell coincides with a recolored component cell, keep the component color.

    Key insight:
    - The same area classification is used twice: once to recolor each component in place, and once to set the widths of the concentric square color bands.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    seed_coords = np.argwhere(g == RED)
    assert len(seed_coords) == 1, "expected exactly one red seed"
    seed = tuple(int(v) for v in seed_coords[0])

    components = find_components_of_color(g, BLACK)
    counts = {1: 0, 2: 0, 3: 0}
    component_mask = np.zeros(g.shape, dtype=bool)
    for coords in components:
        area = int(coords.shape[0])
        if area not in AREA_TO_COLOR:
            continue
        counts[area] += 1
        color = AREA_TO_COLOR[area]
        out[coords[:, 0], coords[:, 1]] = color
        component_mask[coords[:, 0], coords[:, 1]] = True

    out[seed] = RED
    layer_colors = (
        [GREEN] * counts[1]
        + [BLUE] * counts[2]
        + [YELLOW] * counts[3]
    )
    for layer, color in enumerate(layer_colors, start=1):
        draw_square_border(out, seed, layer, color, component_mask)

    return out


def next_count_tuple() -> tuple[int, int, int]:
    global COUNT_BAG
    if not COUNT_BAG:
        COUNT_BAG = COUNT_OPTIONS[:]
        random.shuffle(COUNT_BAG)
    return COUNT_BAG.pop()


def can_place_shape(grid: np.ndarray, abs_cells: list[tuple[int, int]]) -> bool:
    abs_set = set(abs_cells)
    for r, c in abs_cells:
        if not in_bounds(grid.shape, r, c) or int(grid[r, c]) != GREY:
            return False
    for r, c in abs_cells:
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == BLACK and (nr, nc) not in abs_set:
                return False
    return True


def place_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...]) -> bool:
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    candidates = [
        (r0, c0)
        for r0 in range(grid.shape[0] - max_r)
        for c0 in range(grid.shape[1] - max_c)
    ]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        abs_cells = [(r0 + r, c0 + c) for r, c in shape]
        if can_place_shape(grid, abs_cells):
            for r, c in abs_cells:
                grid[r, c] = BLACK
            return True
    return False


def place_components(grid: np.ndarray, counts: tuple[int, int, int]) -> bool:
    n1, n2, n3 = counts
    components = (
        [(3, random.choice(SIZE3_SHAPES)) for _ in range(n3)]
        + [(2, random.choice(SIZE2_SHAPES)) for _ in range(n2)]
        + [(1, SIZE1_SHAPES[0]) for _ in range(n1)]
    )
    random.shuffle(components)
    components.sort(key=lambda item: item[0], reverse=True)
    for _, shape in components:
        if not place_shape(grid, shape):
            return False
    return True


def black_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in rc) for rc in np.argwhere(grid == BLACK)]


def ring_overlap_count(seed: tuple[int, int], cells: list[tuple[int, int]], layers: int) -> int:
    sr, sc = seed
    overlap = 0
    for r, c in cells:
        dist = max(abs(r - sr), abs(c - sc))
        if 1 <= dist <= layers:
            overlap += 1
    return overlap


def choose_seed(grid: np.ndarray, total_layers: int) -> tuple[int, int] | None:
    black = black_cells(grid)
    candidates: list[tuple[int, int, int, bool]] = []
    for r in range(1, grid.shape[0] - 1):
        for c in range(1, grid.shape[1] - 1):
            if int(grid[r, c]) != GREY:
                continue
            if any(in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == BLACK for nr, nc in neighbors4(r, c)):
                continue
            overlap = ring_overlap_count((r, c), black, total_layers)
            if overlap > 4:
                continue
            clipped = total_layers > min(r, c, grid.shape[0] - 1 - r, grid.shape[1] - 1 - c)
            candidates.append((r, c, overlap, clipped))

    if not candidates:
        return None

    want_overlap = random.random() < 0.55
    want_clipped = random.random() < 0.5

    def sample(pool: list[tuple[int, int, int, bool]]) -> tuple[int, int] | None:
        if not pool:
            return None
        r, c, _, _ = random.choice(pool)
        return r, c

    if want_overlap:
        preferred = [item for item in candidates if 1 <= item[2] <= 4 and item[3] == want_clipped]
        fallback = [item for item in candidates if 1 <= item[2] <= 4]
    else:
        preferred = [item for item in candidates if item[2] == 0 and item[3] == want_clipped]
        fallback = [item for item in candidates if item[2] == 0]

    return sample(preferred) or sample(fallback) or sample(candidates)


def valid_input(grid: np.ndarray, counts: tuple[int, int, int]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    seed_coords = np.argwhere(grid == RED)
    if len(seed_coords) != 1:
        return False
    sr, sc = (int(seed_coords[0, 0]), int(seed_coords[0, 1]))
    if sr in (0, grid.shape[0] - 1) or sc in (0, grid.shape[1] - 1):
        return False

    components = find_components_of_color(grid, BLACK)
    areas = sorted(int(comp.shape[0]) for comp in components)
    expected = sorted([1] * counts[0] + [2] * counts[1] + [3] * counts[2])
    if areas != expected:
        return False

    for nr, nc in neighbors4(sr, sc):
        if int(grid[nr, nc]) == BLACK:
            return False

    out = main(grid)
    if out.shape != grid.shape:
        return False

    overlap = ring_overlap_count((sr, sc), black_cells(grid), sum(counts))
    if overlap > 4:
        return False

    return bool(np.any(out != grid))


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        counts = next_count_tuple()
        rows = random.randint(13, 18)
        cols = random.randint(13, 18)
        grid = np.full((rows, cols), GREY, dtype=int)

        if not place_components(grid, counts):
            continue

        seed = choose_seed(grid, sum(counts))
        if seed is None:
            continue
        grid[seed] = RED

        if valid_input(grid, counts):
            return grid

    raise RuntimeError("failed to generate a valid input")
