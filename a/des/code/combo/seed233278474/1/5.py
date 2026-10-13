from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
BAND_COLORS = (1, 3, 4, 6, 7, 8, 9)

GREY_OFFSETS = (
    (-2, 0),
    (-1, -1),
    (-1, 1),
    (0, -2),
    (0, 2),
    (1, -1),
    (1, 1),
    (2, 0),
)
POINTER_OFFSETS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
CLOCK_MASK_OFFSETS = set(GREY_OFFSETS) | set(POINTER_OFFSETS.values()) | {(0, 0)}

GRID_HEIGHT_RANGE = (12, 20)
GRID_WIDTH_RANGE = (12, 20)
CLOCK_COUNT_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 150


def _clock_direction(grid: np.ndarray, r: int, c: int) -> str | None:
    h, w = map(int, grid.shape)
    if not (3 <= r <= h - 4 and 3 <= c <= w - 4):
        return None
    if int(grid[r, c]) != BLACK:
        return None

    for dr, dc in GREY_OFFSETS:
        if int(grid[r + dr, c + dc]) != GREY:
            return None

    pointer_dirs: list[str] = []
    for direction, (dr, dc) in POINTER_OFFSETS.items():
        value = int(grid[r + dr, c + dc])
        if value == RED:
            pointer_dirs.append(direction)
        elif value != BLACK:
            return None
    if len(pointer_dirs) != 1:
        return None

    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if (dr, dc) in CLOCK_MASK_OFFSETS:
                continue
            if int(grid[r + dr, c + dc]) != BLACK:
                return None

    return pointer_dirs[0]


def _find_clocks(grid: np.ndarray) -> list[tuple[int, int, str]]:
    h, w = map(int, grid.shape)
    clocks: list[tuple[int, int, str]] = []
    for r in range(3, h - 3):
        for c in range(3, w - 3):
            direction = _clock_direction(grid, r, c)
            if direction is not None:
                clocks.append((r, c, direction))
    return clocks


def _border_hit(shape: tuple[int, int], r: int, c: int, direction: str) -> tuple[int, int]:
    h, w = map(int, shape)
    if direction == "up":
        return 0, c
    if direction == "down":
        return h - 1, c
    if direction == "left":
        return r, 0
    return r, w - 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each clock center and pointer using the border-band segment selected by
    the pointer direction and the clock's row/column position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for r, c, direction in _find_clocks(g):
        hit_r, hit_c = _border_hit(g.shape, r, c, direction)
        color = int(g[hit_r, hit_c])
        dr, dc = POINTER_OFFSETS[direction]
        out[r, c] = color
        out[r + dr, c + dc] = color
    return out


def _random_partition(total: int) -> list[int]:
    min_parts = max(2, (total + 4) // 5)
    max_parts = min(5, total // 2)
    choices = [parts for parts in range(min_parts, max_parts + 1) if 2 * parts <= total <= 5 * parts]
    assert choices, f"no valid partition for length {total}"
    parts = random.choice(choices)

    lengths = [2] * parts
    remaining = total - 2 * parts
    capacities = [3] * parts
    while remaining > 0:
        idx = random.choice([i for i, cap in enumerate(capacities) if cap > 0])
        add = random.randint(1, min(capacities[idx], remaining))
        lengths[idx] += add
        capacities[idx] -= add
        remaining -= add
    random.shuffle(lengths)
    return lengths


def _segment_colors(parts: int, palette: tuple[int, ...]) -> list[int]:
    colors = [random.choice(palette)]
    while len(colors) < parts:
        prev = colors[-1]
        choices = [color for color in palette if color != prev]
        colors.append(random.choice(choices))
    return colors


def _paint_band(grid: np.ndarray, cells: list[tuple[int, int]], palette: tuple[int, ...]) -> None:
    lengths = _random_partition(len(cells))
    colors = _segment_colors(len(lengths), palette)

    idx = 0
    for length, color in zip(lengths, colors):
        for _ in range(length):
            r, c = cells[idx]
            grid[r, c] = color
            idx += 1


def _clock_boxes_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) < 5 and abs(a[1] - b[1]) < 5


def _sample_centers(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(3, h - 3) for c in range(3, w - 3)]
    for _ in range(PLACEMENT_ATTEMPTS):
        random.shuffle(candidates)
        centers: list[tuple[int, int]] = []
        for center in candidates:
            if all(not _clock_boxes_overlap(center, other) for other in centers):
                centers.append(center)
                if len(centers) == count:
                    return centers
    return None


def _draw_clock(grid: np.ndarray, center: tuple[int, int], direction: str) -> None:
    r, c = center
    for dr, dc in GREY_OFFSETS:
        grid[r + dr, c + dc] = GREY
    dr, dc = POINTER_OFFSETS[direction]
    grid[r + dr, c + dc] = RED
    grid[r, c] = BLACK


def _sample_directions(count: int) -> list[str]:
    directions = [random.choice(tuple(POINTER_OFFSETS)) for _ in range(count)]
    if count >= 3 and len(set(directions)) == 1:
        directions[-1] = random.choice([d for d in POINTER_OFFSETS if d != directions[-1]])
    return directions


def _border_color(grid: np.ndarray, r: int, c: int, direction: str) -> int:
    hit_r, hit_c = _border_hit(grid.shape, r, c, direction)
    return int(grid[hit_r, hit_c])


def _validate_generated(grid: np.ndarray, expected_count: int) -> bool:
    clocks = _find_clocks(grid)
    if len(clocks) != expected_count:
        return False

    hit_colors = [_border_color(grid, r, c, direction) for r, c, direction in clocks]
    if len(set(hit_colors)) < 2:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False

    changed = np.argwhere(out != grid)
    if changed.shape[0] != 2 * expected_count:
        return False

    if not np.array_equal(out[0, :], grid[0, :]):
        return False
    if not np.array_equal(out[-1, :], grid[-1, :]):
        return False
    if not np.array_equal(out[1:-1, 0], grid[1:-1, 0]):
        return False
    if not np.array_equal(out[1:-1, -1], grid[1:-1, -1]):
        return False

    for r, c, direction in clocks:
        color = _border_color(grid, r, c, direction)
        dr, dc = POINTER_OFFSETS[direction]
        if int(out[r, c]) != color or int(out[r + dr, c + dc]) != color:
            return False
        for off_r, off_c in GREY_OFFSETS:
            if int(out[r + off_r, c + off_c]) != GREY:
                return False

    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    count = random.randint(*CLOCK_COUNT_RANGE)

    grid = np.full((h, w), BLACK, dtype=int)
    palette_size = random.randint(4, min(6, len(BAND_COLORS)))
    palette = tuple(random.sample(BAND_COLORS, palette_size))

    _paint_band(grid, [(0, c) for c in range(w)], palette)
    _paint_band(grid, [(h - 1, c) for c in range(w)], palette)
    _paint_band(grid, [(r, 0) for r in range(1, h - 1)], palette)
    _paint_band(grid, [(r, w - 1) for r in range(1, h - 1)], palette)

    centers = _sample_centers(h, w, count)
    if centers is None:
        return None

    directions = _sample_directions(count)
    random.shuffle(centers)
    for center, direction in zip(centers, directions):
        _draw_clock(grid, center, direction)

    if not _validate_generated(grid, count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
