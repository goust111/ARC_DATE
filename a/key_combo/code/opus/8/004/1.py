from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
BLUE = 1
RED = 2
GREEN = 3

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CORNERS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
MAX_GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _detect_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (
                grid[r, c] == YELLOW
                and grid[r - 1, c] == YELLOW
                and grid[r + 1, c] == YELLOW
                and grid[r, c - 1] == YELLOW
                and grid[r, c + 1] == YELLOW
            ):
                centers.append((r, c))
    return centers


def _ray_color(index_along_ray: int) -> int:
    cycle = (GREEN, BLUE, RED)
    return cycle[(index_along_ray - 1) % 3]


def _emit_ray_cells(grid: np.ndarray, center: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int, int]]:
    r, c = center
    dr, dc = direction
    cells: list[tuple[int, int, int]] = []
    step = 2
    ray_index = 1
    while True:
        nr = r + dr * step
        nc = c + dc * step
        if not _in_bounds(grid.shape, nr, nc):
            break
        if grid[nr, nc] != BG:
            break
        cells.append((nr, nc, _ray_color(ray_index)))
        step += 1
        ray_index += 1
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """Emit four periodic rays from every yellow plus center while preserving the original foreground.

    Input structure:
    - The grid is a black background with several yellow 4-connected foreground regions.
    - Some regions contain a true "+" center: one yellow pixel whose four orthogonal neighbors are also yellow.
    - No other input colors are present, and non-plus yellow regions are inert obstacles.

    Rule:
    - Detect every yellow plus center.
    - From each center, cast rays up, down, left, and right, starting just beyond the arm tip; each ray continues only through cells that were black in the original input and stops before the first non-black cell or the grid edge.
    - Color ray cells by step index along that ray with the repeating sequence green, blue, red; if multiple centers propose the same cell, keep the color from the lexicographically smallest source center `(row, col)`.
    - Leave all original yellow cells unchanged.

    Key insight:
    - The yellow plus is not output decoration but the emitter definition, and ray conflicts are resolved by source-center priority rather than by draw order or direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "grid exceeds ARC size limit"

    centers = sorted(_detect_centers(g))
    out = g.copy()
    claims: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}

    for center in centers:
        for direction in DIRECTIONS:
            for r, c, color in _emit_ray_cells(g, center, direction):
                key = (r, c)
                prev = claims.get(key)
                if prev is None or center < prev[0]:
                    claims[key] = (center, color)

    for (r, c), (_center, color) in claims.items():
        out[r, c] = color
    return out


def _plus_cells(center: tuple[int, int], extra_corners: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r, c = center
    cells = {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
    for dr, dc in extra_corners:
        cells.add((r + dr, c + dc))
    return cells


def _can_place(grid: np.ndarray, cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    cell_set = set(cells)
    for r, c in cell_set:
        if not _in_bounds(grid.shape, r, c) or grid[r, c] != BG:
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in cell_set:
                    continue
                if _in_bounds(grid.shape, rr, cc) and grid[rr, cc] == YELLOW:
                    return False
    return True


def _paint(grid: np.ndarray, cells: set[tuple[int, int]]) -> None:
    for r, c in cells:
        grid[r, c] = YELLOW


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _has_cross(cells: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if (
            (r - 1, c) in cells
            and (r + 1, c) in cells
            and (r, c - 1) in cells
            and (r, c + 1) in cells
        ):
            return True
    return False


def _random_connected_blob(size: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIRECTIONS:
                nbr = (r + dr, c + dc)
                if nbr not in cells:
                    frontier.append(nbr)
        cells.add(random.choice(frontier))
    return _normalize(cells)


def _place_blob_randomly(grid: np.ndarray, size: int) -> bool:
    for _ in range(80):
        blob = _random_connected_blob(size)
        if _has_cross(blob):
            continue
        h = max(r for r, _ in blob) + 1
        w = max(c for _, c in blob) + 1
        top = random.randint(0, grid.shape[0] - h)
        left = random.randint(0, grid.shape[1] - w)
        placed = {(top + r, left + c) for r, c in blob}
        if _can_place(grid, placed):
            _paint(grid, placed)
            return True
    return False


def _place_blocker_on_ray(grid: np.ndarray, center: tuple[int, int], direction: tuple[int, int]) -> bool:
    dr, dc = direction
    if dr == 0:
        perp_options = [(-1, 0), (1, 0)]
    else:
        perp_options = [(0, -1), (0, 1)]

    distances = list(range(4, 8))
    random.shuffle(distances)
    for distance in distances:
        anchor = (center[0] + dr * distance, center[1] + dc * distance)
        if not _in_bounds(grid.shape, *anchor):
            continue
        pr, pc = random.choice(perp_options)
        shape_options = [
            {(0, 0), (pr, pc)},
            {(0, 0), (pr, pc), (pr + dr, pc + dc)},
            {(0, 0), (pr, pc), (pr - dr, pc - dc)},
        ]
        random.shuffle(shape_options)
        for offsets in shape_options:
            placed = {(anchor[0] + r, anchor[1] + c) for r, c in offsets}
            if _has_cross(_normalize(placed)):
                continue
            if _can_place(grid, placed):
                _paint(grid, placed)
                return True
    return False


def _is_far_enough(candidate: tuple[int, int], centers: list[tuple[int, int]]) -> bool:
    r, c = candidate
    for rr, cc in centers:
        dr = abs(r - rr)
        dc = abs(c - cc)
        if dr <= 2 and dc <= 2:
            return False
        if dr + dc < 5:
            return False
        if r == rr and dc < 5:
            return False
        if c == cc and dr < 5:
            return False
    return True


def _random_center(shape: tuple[int, int], centers: list[tuple[int, int]]) -> tuple[int, int] | None:
    rows = list(range(3, shape[0] - 3))
    cols = list(range(3, shape[1] - 3))
    candidates = [(r, c) for r in rows for c in cols]
    random.shuffle(candidates)
    for candidate in candidates:
        if _is_far_enough(candidate, centers):
            return candidate
    return None


def _sample_centers(shape: tuple[int, int]) -> tuple[list[tuple[int, int]], str]:
    h, w = shape
    scenario = random.choices(
        ["single", "isolated", "aligned", "perpendicular"],
        weights=[2, 3, 3, 3],
        k=1,
    )[0]

    if scenario == "single":
        center = _random_center(shape, [])
        assert center is not None
        return [center], scenario

    if scenario == "aligned":
        horizontal = random.random() < 0.5
        if horizontal:
            r = random.randint(3, h - 4)
            c1 = random.randint(3, w - 9)
            c2 = random.randint(c1 + 5, w - 4)
            centers = [(r, c1), (r, c2)]
        else:
            c = random.randint(3, w - 4)
            r1 = random.randint(3, h - 9)
            r2 = random.randint(r1 + 5, h - 4)
            centers = [(r1, c), (r2, c)]
        if random.random() < 0.45:
            extra = _random_center(shape, centers)
            if extra is not None:
                centers.append(extra)
        return sorted(centers), scenario

    if scenario == "perpendicular":
        r1 = random.randint(3, h - 9)
        c1 = random.randint(3, w - 9)
        r2 = random.randint(r1 + 5, h - 4)
        c2 = random.randint(c1 + 5, w - 4)
        centers = [(r1, c1), (r2, c2)]
        if random.random() < 0.35:
            extra = _random_center(shape, centers)
            if extra is not None:
                centers.append(extra)
        return sorted(centers), scenario

    count = random.choice([2, 3])
    centers: list[tuple[int, int]] = []
    while len(centers) < count:
        center = _random_center(shape, centers)
        if center is None:
            break
        centers.append(center)
    if not centers:
        center = _random_center(shape, [])
        assert center is not None
        centers = [center]
    return sorted(centers), scenario


def _clearance_ok(grid: np.ndarray, center: tuple[int, int]) -> bool:
    r, c = center
    for dr, dc in DIRECTIONS:
        for step in (2, 3):
            nr = r + dr * step
            nc = c + dc * step
            if not _in_bounds(grid.shape, nr, nc):
                return False
            if grid[nr, nc] != BG:
                return False
    return True


def _make_cross_component(grid: np.ndarray, center: tuple[int, int]) -> bool:
    corner_count = random.randint(0, 4)
    choices = CORNERS[:]
    random.shuffle(choices)
    cells = _plus_cells(center, set(choices[:corner_count]))
    if not _can_place(grid, cells):
        return False
    _paint(grid, cells)
    return True


def _build_input() -> np.ndarray | None:
    h = random.randint(13, 18)
    w = random.randint(13, 18)
    grid = np.zeros((h, w), dtype=int)

    centers, scenario = _sample_centers(grid.shape)
    if not centers:
        return None

    for center in centers:
        if not _make_cross_component(grid, center):
            return None

    if scenario in {"single", "isolated"} and random.random() < 0.7:
        blocker_attempts = random.randint(1, min(2, len(centers)))
        center_order = centers[:]
        random.shuffle(center_order)
        for center in center_order[:blocker_attempts]:
            directions = DIRECTIONS[:]
            random.shuffle(directions)
            for direction in directions:
                if _place_blocker_on_ray(grid, center, direction):
                    break

    free_blob_count = random.randint(0, 2)
    for _ in range(free_blob_count):
        size = random.randint(3, 7)
        _place_blob_randomly(grid, size)

    detected = sorted(_detect_centers(grid))
    if detected != sorted(centers):
        return None
    if not all(_clearance_ok(grid, center) for center in centers):
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-ray-cross input")
