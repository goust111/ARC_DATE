from __future__ import annotations

import random

import numpy as np


BG = 8
DIR_ORDER = ("up", "down", "left", "right")
DIR_OFFSETS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


# Generator configuration.
GRID_HEIGHT_BOUNDS = (14, 21)
GRID_WIDTH_BOUNDS = (14, 21)
SPRITE_COUNT_BOUNDS = (2, 4)
AREA_BOUNDS = (5, 15)
LOCAL_DIM_BOUNDS = (2, 6)
MAX_LOCAL_BOX_AREA = 24
PALETTE = (0, 1, 2, 3, 4, 5, 6, 7, 9)
SHAPE_ATTEMPTS = 250
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yield int(r + dr), int(c + dc)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _is_bg_or_oob(grid: np.ndarray, r: int, c: int) -> bool:
    return not _in_bounds(grid.shape, r, c) or int(grid[r, c]) == BG


def _find_components(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))

            components.append((color, coords))

    components.sort(
        key=lambda item: (
            min(r for r, _ in item[1]),
            min(c for _, c in item[1]),
            int(item[0]),
        )
    )
    return components


def _directional_side_pixels(
    grid: np.ndarray, coords: list[tuple[int, int]]
) -> dict[str, list[tuple[int, int]]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    top_row = min(rows)
    bottom_row = max(rows)
    left_col = min(cols)
    right_col = max(cols)

    buckets = {name: [] for name in DIR_ORDER}
    for r, c in coords:
        if r == top_row and _is_bg_or_oob(grid, r - 1, c):
            buckets["up"].append((int(r), int(c)))
        if r == bottom_row and _is_bg_or_oob(grid, r + 1, c):
            buckets["down"].append((int(r), int(c)))
        if c == left_col and _is_bg_or_oob(grid, r, c - 1):
            buckets["left"].append((int(r), int(c)))
        if c == right_col and _is_bg_or_oob(grid, r, c + 1):
            buckets["right"].append((int(r), int(c)))
    return buckets


def _dominant_direction(
    buckets: dict[str, list[tuple[int, int]]]
) -> str | None:
    counts = {name: len(buckets[name]) for name in DIR_ORDER}
    best = max(counts.values(), default=0)
    winners = [name for name in DIR_ORDER if counts[name] == best and best > 0]
    if len(winners) != 1:
        return None
    return winners[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each sprite outward on its most exposed side.

    For isolated sprites, counting every exposed top/down/left/right edge cell
    leads to unavoidable opposite-side ties. This implementation therefore uses
    the outer side pixels of the sprite bbox: top row, bottom row, left column,
    and right column pixels that face background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for color, coords in _find_components(g):
        side_pixels = _directional_side_pixels(g, coords)
        direction = _dominant_direction(side_pixels)
        if direction is None:
            continue

        dr, dc = DIR_OFFSETS[direction]
        for r, c in side_pixels[direction]:
            nr = int(r + dr)
            nc = int(c + dc)
            if _in_bounds(g.shape, nr, nc) and int(out[nr, nc]) == BG:
                out[nr, nc] = int(color)
    return out


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((int(r - min_r), int(c - min_c)) for r, c in cells)


def _random_polyomino(area: int, box_h: int, box_w: int) -> list[tuple[int, int]] | None:
    start = (random.randrange(box_h), random.randrange(box_w))
    cells = {start}
    frontier: list[tuple[int, int]] = []

    def add_frontier(cell: tuple[int, int]) -> None:
        r, c = cell
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < box_h and 0 <= nc < box_w):
                continue
            if (nr, nc) in cells or (nr, nc) in frontier:
                continue
            frontier.append((int(nr), int(nc)))

    add_frontier(start)
    while len(cells) < area and frontier:
        idx = random.randrange(len(frontier))
        cell = frontier.pop(idx)
        if cell in cells:
            continue
        cells.add(cell)
        add_frontier(cell)

    if len(cells) != area:
        return None
    return _normalize(cells)


def _shape_dims(coords: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _local_side_pixels(
    coords: list[tuple[int, int]]
) -> dict[str, list[tuple[int, int]]]:
    box_h, box_w = _shape_dims(coords)
    local_grid = np.full((box_h + 2, box_w + 2), BG, dtype=int)
    for r, c in coords:
        local_grid[r + 1, c + 1] = 1
    shifted = [(r + 1, c + 1) for r, c in coords]
    buckets = _directional_side_pixels(local_grid, shifted)
    return {
        name: [(r - 1, c - 1) for r, c in pts]
        for name, pts in buckets.items()
    }


def _generate_shape_for_direction(
    direction: str,
) -> tuple[list[tuple[int, int]], int, int]:
    for _ in range(SHAPE_ATTEMPTS):
        area = random.randint(*AREA_BOUNDS)
        dims = [
            (box_h, box_w)
            for box_h in range(LOCAL_DIM_BOUNDS[0], LOCAL_DIM_BOUNDS[1] + 1)
            for box_w in range(LOCAL_DIM_BOUNDS[0], LOCAL_DIM_BOUNDS[1] + 1)
            if area <= box_h * box_w <= MAX_LOCAL_BOX_AREA
        ]
        if not dims:
            return None
        box_h, box_w = random.choice(dims)
        coords = _random_polyomino(area, box_h, box_w)
        if coords is None:
            continue

        shape_h, shape_w = _shape_dims(coords)
        if shape_h < 2 or shape_w < 2:
            continue
        if len(coords) == shape_h * shape_w:
            continue

        buckets = _local_side_pixels(coords)
        if _dominant_direction(buckets) != direction:
            continue
        if len(buckets[direction]) < 2:
            continue
        return coords, shape_h, shape_w
    return None


def _can_place(reserved: np.ndarray, top: int, left: int, shape_h: int, shape_w: int) -> bool:
    return not bool(reserved[top - 1 : top + shape_h + 1, left - 1 : left + shape_w + 1].any())


def _place_shape(
    grid: np.ndarray,
    reserved: np.ndarray,
    coords: list[tuple[int, int]],
    color: int,
    top: int,
    left: int,
) -> None:
    for r, c in coords:
        grid[top + r, left + c] = int(color)
    shape_h, shape_w = _shape_dims(coords)
    reserved[top - 1 : top + shape_h + 1, left - 1 : left + shape_w + 1] = True


def _validate_generated_grid(grid: np.ndarray, expected_count: int) -> bool:
    components = _find_components(grid)
    if len(components) != expected_count:
        return False

    colors = [int(color) for color, _ in components]
    if len(set(colors)) != len(colors):
        return False

    for _color, coords in components:
        if not (AREA_BOUNDS[0] <= len(coords) <= AREA_BOUNDS[1]):
            return False
        buckets = _directional_side_pixels(grid, coords)
        if _dominant_direction(buckets) is None:
            return False

    out = main(grid)
    return bool(np.any(out != grid))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_BOUNDS)
    width = random.randint(*GRID_WIDTH_BOUNDS)
    sprite_count = random.randint(*SPRITE_COUNT_BOUNDS)
    colors = random.sample(PALETTE, sprite_count)
    directions = random.sample(list(DIR_ORDER), sprite_count)

    grid = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    for color, direction in zip(colors, directions):
        shape = _generate_shape_for_direction(direction)
        if shape is None:
            return None
        coords, shape_h, shape_w = shape

        positions = [
            (top, left)
            for top in range(1, height - shape_h)
            for left in range(1, width - shape_w)
            if _can_place(reserved, top, left, shape_h, shape_w)
        ]
        if not positions:
            return None

        top, left = random.choice(positions)
        _place_shape(grid, reserved, coords, color, top, left)

    if not _validate_generated_grid(grid, sprite_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
