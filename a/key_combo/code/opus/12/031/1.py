from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 5
COMPONENT_COLOR = 1
MARKER_COLOR = 2
ODD_COLOR = 3
EVEN_COLOR = 4

GRID_MIN = 12
GRID_MAX = 16
MARKER_COUNT_RANGE = (2, 4)
COMPONENT_COUNT_RANGE = (3, 6)
COMPONENT_SIZE_RANGE = (3, 8)
GENERATION_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    seen = np.zeros(grid.shape, dtype=bool)
    components: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        queue = deque([(r, c)])
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while queue:
            rr, cc = queue.popleft()
            coords.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if _in_bounds(grid.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        components.append(np.array(coords, dtype=int))
    return components


def _marker_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(grid == MARKER_COLOR)]


def _build_rays(markers: list[tuple[int, int]]) -> list[tuple[str, int]]:
    rays: list[tuple[str, int]] = []
    for r, c in markers:
        rays.append(("diag", r - c))
        rays.append(("anti", r + c))
    return rays


def _component_ray_count(coords: np.ndarray, rays: list[tuple[str, int]]) -> int:
    diag_ids = {int(r - c) for r, c in coords}
    anti_ids = {int(r + c) for r, c in coords}
    return sum(
        1
        for kind, value in rays
        if (kind == "diag" and value in diag_ids) or (kind == "anti" and value in anti_ids)
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count which diagonal rays pass through each blue connected component and recolor the component by hit-count parity.

    Input structure:
    - The background is grey.
    - Red singleton marker pixels are scattered away from the blue foreground.
    - Blue foreground pixels form several separate 4-connected components with irregular shapes.

    Rule:
    - From every red marker, take the full NW-SE diagonal and the full NE-SW diagonal through that marker as logical rays.
    - For each blue 4-connected component, count how many of those rays pass through at least one pixel of the component, counting each ray at most once per component.
    - Recolor the whole component green if its hit count is odd, or yellow if its hit count is even and nonzero, while leaving markers and background unchanged.

    Key insight:
    - The diagonals are never drawn; they act only as geometric tests whose per-component vote parity determines the recolor.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    markers = _marker_positions(g)
    rays = _build_rays(markers)
    for coords in _components_of_color(g, COMPONENT_COLOR):
        count = _component_ray_count(coords, rays)
        if count % 2 == 1:
            out[coords[:, 0], coords[:, 1]] = ODD_COLOR
        elif count > 0:
            out[coords[:, 0], coords[:, 1]] = EVEN_COLOR

    return out


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _transform_shape(
    shape: tuple[tuple[int, int], ...], rotation: int, flip: bool
) -> tuple[tuple[int, int], ...]:
    cells = [(r, c) for r, c in shape]
    for _ in range(rotation % 4):
        cells = [(-c, r) for r, c in cells]
    if flip:
        cells = [(r, -c) for r, c in cells]
    return _normalize_shape(set(cells))


def _random_irregular_shape(rng: random.Random, size: int) -> tuple[tuple[int, int], ...]:
    for _ in range(80):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in cells:
                        continue
                    trial = cells | {(nr, nc)}
                    r0, c0, r1, c1 = _bbox(trial)
                    if r1 - r0 + 1 <= 4 and c1 - c0 + 1 <= 4:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != size:
            continue
        r0, c0, r1, c1 = _bbox(cells)
        if (r1 - r0 + 1) * (c1 - c0 + 1) == size:
            continue
        base = _normalize_shape(cells)
        return _transform_shape(base, rng.randrange(4), bool(rng.randrange(2)))

    fallback = {
        3: ((0, 0), (1, 0), (1, 1)),
        4: ((0, 0), (1, 0), (2, 0), (2, 1)),
        5: ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
        6: ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)),
        7: ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (3, 2)),
        8: ((0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 1), (3, 1), (3, 2)),
    }
    return _transform_shape(fallback[size], rng.randrange(4), bool(rng.randrange(2)))


def _place_markers(
    rng: random.Random, height: int, width: int, marker_count: int
) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    rng.shuffle(cells)
    markers: list[tuple[int, int]] = []
    diag_ids: set[int] = set()
    anti_ids: set[int] = set()
    for r, c in cells:
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in markers):
            continue
        if r - c in diag_ids or r + c in anti_ids:
            continue
        markers.append((r, c))
        diag_ids.add(r - c)
        anti_ids.add(r + c)
        if len(markers) == marker_count:
            return markers
    return None


def _shape_hits(
    placed_cells: list[tuple[int, int]], markers: list[tuple[int, int]]
) -> int:
    rays = _build_rays(markers)
    coords = np.array(placed_cells, dtype=int)
    return _component_ray_count(coords, rays)


def _can_place_component(
    grid: np.ndarray, cells: list[tuple[int, int]]
) -> bool:
    cell_set = set(cells)
    for r, c in cells:
        if not _in_bounds(grid.shape, r, c):
            return False
        if grid[r, c] != BACKGROUND:
            return False
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(grid.shape, nr, nc):
                continue
            if (nr, nc) in cell_set:
                continue
            if grid[nr, nc] in (COMPONENT_COLOR, MARKER_COLOR):
                return False
    return True


def _component_cells(
    shape: tuple[tuple[int, int], ...], top: int, left: int
) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in shape]


def _place_components(
    rng: random.Random,
    grid: np.ndarray,
    markers: list[tuple[int, int]],
    target_components: int,
) -> list[int] | None:
    hit_counts: list[int] = []
    height, width = grid.shape
    for _ in range(target_components):
        placed = False
        for _ in range(220):
            size = rng.randint(*COMPONENT_SIZE_RANGE)
            shape = _random_irregular_shape(rng, size)
            max_r = max(r for r, _ in shape)
            max_c = max(c for _, c in shape)
            top = rng.randint(0, height - max_r - 1)
            left = rng.randint(0, width - max_c - 1)
            cells = _component_cells(shape, top, left)
            if not _can_place_component(grid, cells):
                continue
            hit_count = _shape_hits(cells, markers)
            if hit_count == 0:
                continue
            if target_components >= 4 and len(hit_counts) == 0 and hit_count == 1:
                continue
            for r, c in cells:
                grid[r, c] = COMPONENT_COLOR
            hit_counts.append(hit_count)
            placed = True
            break
        if not placed:
            return None
    return hit_counts


def _validate_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) - {BACKGROUND, COMPONENT_COLOR, MARKER_COLOR}:
        return False

    markers = _marker_positions(grid)
    if not (MARKER_COUNT_RANGE[0] <= len(markers) <= MARKER_COUNT_RANGE[1]):
        return False
    if len({r - c for r, c in markers}) != len(markers):
        return False
    if len({r + c for r, c in markers}) != len(markers):
        return False
    for i, (r, c) in enumerate(markers):
        for rr, cc in markers[i + 1 :]:
            if max(abs(r - rr), abs(c - cc)) <= 1:
                return False
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == COMPONENT_COLOR:
                return False

    components = _components_of_color(grid, COMPONENT_COLOR)
    if not (COMPONENT_COUNT_RANGE[0] <= len(components) <= COMPONENT_COUNT_RANGE[1]):
        return False

    rays = _build_rays(markers)
    hit_counts: list[int] = []
    for coords in components:
        area = int(coords.shape[0])
        if not (COMPONENT_SIZE_RANGE[0] <= area <= COMPONENT_SIZE_RANGE[1]):
            return False
        rows = coords[:, 0]
        cols = coords[:, 1]
        if (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1) == area:
            return False
        count = _component_ray_count(coords, rays)
        if count == 0:
            return False
        hit_counts.append(count)

    if not any(count % 2 == 1 for count in hit_counts):
        return False
    if not any(count % 2 == 0 for count in hit_counts):
        return False
    if max(hit_counts) < 2:
        return False
    if len(set(hit_counts)) < 2:
        return False

    expected = grid.copy()
    for coords, count in zip(components, hit_counts):
        expected[coords[:, 0], coords[:, 1]] = ODD_COLOR if count % 2 == 1 else EVEN_COLOR
    return np.array_equal(main(grid), expected)


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    marker_count = rng.randint(*MARKER_COUNT_RANGE)
    component_count = rng.randint(*COMPONENT_COUNT_RANGE)

    markers = _place_markers(rng, height, width, marker_count)
    if markers is None:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for r, c in markers:
        grid[r, c] = MARKER_COLOR

    hit_counts = _place_components(rng, grid, markers, component_count)
    if hit_counts is None:
        return None

    if not _validate_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-ray parity input")
