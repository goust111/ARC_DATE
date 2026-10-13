from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

CHECKER_BLOCK = np.array([[BLUE, RED], [RED, BLUE]], dtype=int)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_MIN = 8
GRID_MAX = 15
SINGLE_COUNTS = (2, 3)
MULTI_COUNTS = (1, 2)
MULTI_AREA_MIN = 3
MULTI_AREA_MAX = 8
SEPARATION_RADIUS = 2
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 120
SHAPE_RETRIES = 200


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in NEIGHBORS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _connected_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if grid[r, c] != GREEN or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if grid[nr, nc] == GREEN and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(cells)
    return components


def _transform(grid: np.ndarray) -> np.ndarray:
    height, width = grid.shape
    out = np.full((2 * height, 2 * width), BLACK, dtype=int)

    for component in _connected_components(grid):
        block = np.full((2, 2), GREEN, dtype=int) if len(component) == 1 else CHECKER_BLOCK
        for r, c in component:
            out[2 * r : 2 * r + 2, 2 * c : 2 * c + 2] = block
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Upscale the black-and-green canvas by 2x, turning isolated green pixels into solid blocks and grouped green pixels into checkerboard blocks.

    Input structure:
    - The grid is a black background containing only green foreground pixels.
    - Green pixels form several 4-connected components, including isolated single pixels and irregular multi-pixel objects.

    Rule:
    - Create a 2H x 2W output grid.
    - Each black input pixel becomes a 2x2 black block.
    - Each green component is classified by its 4-connected area.
    - Area-1 green pixels become solid 2x2 green blocks, while pixels in larger green components become 2x2 blue/red checkerboard blocks.

    Key insight:
    - The replacement pattern is chosen per pixel from the size of its connected component, so solving requires connected-component analysis before local upscaling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, GREEN}, "expected only black and green pixels"
    return _transform(g)


def _normalize_shape(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _shape_is_irregular(shape: list[tuple[int, int]]) -> bool:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    height = max(rows) + 1
    width = max(cols) + 1
    return height > 1 and width > 1 and len(shape) < height * width


def _generate_blob(area: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    for _ in range(SHAPE_RETRIES):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            weights: list[int] = []
            frontier_set = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    candidate = (r + dr, c + dc)
                    if candidate in cells or candidate in frontier_set:
                        continue
                    frontier_set.add(candidate)
                    shared_edges = sum((candidate[0] + rr, candidate[1] + cc) in cells for rr, cc in NEIGHBORS4)
                    frontier.append(candidate)
                    weights.append(2 if shared_edges >= 2 else 1)
            index = int(rng.choice(len(frontier), p=np.array(weights, dtype=float) / sum(weights)))
            cells.add(frontier[index])

        shape = _normalize_shape(cells)
        if _shape_is_irregular(shape):
            return shape
    raise RuntimeError(f"failed to generate irregular blob of area {area}")


def _paint_forbidden(forbidden: np.ndarray, cells: list[tuple[int, int]]) -> None:
    height, width = forbidden.shape
    for r, c in cells:
        r0 = max(0, r - SEPARATION_RADIUS)
        r1 = min(height, r + SEPARATION_RADIUS + 1)
        c0 = max(0, c - SEPARATION_RADIUS)
        c1 = min(width, c + SEPARATION_RADIUS + 1)
        forbidden[r0:r1, c0:c1] = True


def _place_shape(
    grid: np.ndarray,
    forbidden: np.ndarray,
    shape: list[tuple[int, int]],
    rng: np.random.Generator,
) -> bool:
    height, width = grid.shape
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1
    positions = [
        (top, left)
        for top in range(height - shape_h + 1)
        for left in range(width - shape_w + 1)
    ]
    order = rng.permutation(len(positions))

    for idx in order[:PLACEMENT_RETRIES]:
        top, left = positions[int(idx)]
        placed = [(top + r, left + c) for r, c in shape]
        if any(forbidden[r, c] for r, c in placed):
            continue
        for r, c in placed:
            grid[r, c] = GREEN
        _paint_forbidden(forbidden, placed)
        return True
    return False


def _choose_grid_shape(
    num_multi: int,
    num_single: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    base = GRID_MIN + (num_multi - 1) * 2 + (num_single - 2)
    low = min(base, GRID_MAX)
    height = int(rng.integers(low, GRID_MAX + 1))
    width = int(rng.integers(low, GRID_MAX + 1))
    return height, width


def _component_spacing_ok(components: list[list[tuple[int, int]]]) -> bool:
    for i, comp_a in enumerate(components):
        for comp_b in components[i + 1 :]:
            for ra, ca in comp_a:
                for rb, cb in comp_b:
                    if max(abs(ra - rb), abs(ca - cb)) <= SEPARATION_RADIUS:
                        return False
    return True


def _valid_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 15:
        return False
    if set(map(int, np.unique(grid))) - {BLACK, GREEN}:
        return False

    components = _connected_components(grid)
    sizes = sorted(len(comp) for comp in components)
    singles = sum(size == 1 for size in sizes)
    multis = [size for size in sizes if size > 1]

    if singles not in SINGLE_COUNTS:
        return False
    if len(multis) not in MULTI_COUNTS:
        return False
    if any(size < MULTI_AREA_MIN or size > MULTI_AREA_MAX for size in multis):
        return False
    if not _component_spacing_ok(components):
        return False

    for comp in components:
        if len(comp) == 1:
            continue
        shape = _normalize_shape(set(comp))
        if not _shape_is_irregular(shape):
            return False
    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    num_multi = int(rng.choice(MULTI_COUNTS))
    num_single = int(rng.choice(SINGLE_COUNTS))
    height, width = _choose_grid_shape(num_multi, num_single, rng)

    grid = np.full((height, width), BLACK, dtype=int)
    forbidden = np.zeros((height, width), dtype=bool)

    areas = [int(rng.integers(MULTI_AREA_MIN, MULTI_AREA_MAX + 1)) for _ in range(num_multi)]
    areas.sort(reverse=True)

    for area in areas:
        shape = _generate_blob(area, rng)
        if not _place_shape(grid, forbidden, shape, rng):
            return None

    for _ in range(num_single):
        if not _place_shape(grid, forbidden, [(0, 0)], rng):
            return None

    if not _valid_input(grid):
        return None

    expected = _transform(grid)
    if max(expected.shape) > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-4 input")
