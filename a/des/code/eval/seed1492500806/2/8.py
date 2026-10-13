import random
from collections import Counter

import numpy as np


BLACK = 0
GRAY = 5
MAROON = 9
PAIR_COLORS = (1, 2, 3, 4, 6, 7, 8)

BRANCH_TYPES = (
    "horizontal_clear",
    "horizontal_blocked",
    "vertical_clear",
    "vertical_blocked",
    "diagonal",
)

GRID_HEIGHT_RANGE = (12, 18)
GRID_WIDTH_RANGE = (12, 18)
GENERATE_RETRIES = 400
PAIR_PLACEMENT_RETRIES = 250
EXTRA_OBSTACLE_RANGE = (1, 6)


def _pair_cells(color: int, grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    coords = np.argwhere(grid == color)
    assert coords.shape == (2, 2), f"color {color} must appear exactly twice"
    a = (int(coords[0, 0]), int(coords[0, 1]))
    b = (int(coords[1, 0]), int(coords[1, 1]))
    return a, b


def _paint_row_segment(
    original: np.ndarray,
    out: np.ndarray,
    row: int,
    left: tuple[int, int],
    right: tuple[int, int],
    color: int,
) -> None:
    obstacle_col = None
    for c in range(left[1] + 1, right[1]):
        if int(original[row, c]) == GRAY:
            obstacle_col = c
            break

    stop = right[1] if obstacle_col is None else obstacle_col
    for c in range(left[1] + 1, stop):
        if int(original[row, c]) == BLACK:
            out[row, c] = color

    if obstacle_col is not None:
        out[right] = MAROON


def _paint_col_segment(
    original: np.ndarray,
    out: np.ndarray,
    col: int,
    top: tuple[int, int],
    bottom: tuple[int, int],
    color: int,
) -> None:
    obstacle_row = None
    for r in range(top[0] + 1, bottom[0]):
        if int(original[r, col]) == GRAY:
            obstacle_row = r
            break

    stop = bottom[0] if obstacle_row is None else obstacle_row
    for r in range(top[0] + 1, stop):
        if int(original[r, col]) == BLACK:
            out[r, col] = color

    if obstacle_row is not None:
        out[bottom] = MAROON


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect aligned same-color pixel pairs, but mark invalid or blocked pairs with maroon.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    colors = sorted(int(v) for v in np.unique(g) if int(v) not in (BLACK, GRAY))
    for color in colors:
        a, b = _pair_cells(color, g)
        if a[0] == b[0]:
            left, right = sorted((a, b), key=lambda p: p[1])
            _paint_row_segment(g, out, a[0], left, right, color)
        elif a[1] == b[1]:
            top, bottom = sorted((a, b), key=lambda p: p[0])
            _paint_col_segment(g, out, a[1], top, bottom, color)
        else:
            out[a] = MAROON
            out[b] = MAROON
    return out


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        row = a[0]
        c0, c1 = sorted((a[1], b[1]))
        return [(row, c) for c in range(c0, c1 + 1)]
    if a[1] == b[1]:
        col = a[1]
        r0, r1 = sorted((a[0], b[0]))
        return [(r, col) for r in range(r0, r1 + 1)]
    return [a, b]


def _cells_available(cells: list[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    return all(cell not in reserved for cell in cells)


def _place_horizontal_clear(
    grid: np.ndarray, reserved: set[tuple[int, int]], color: int
) -> bool:
    height, width = grid.shape
    for _ in range(PAIR_PLACEMENT_RETRIES):
        row = random.randrange(height)
        left_col = random.randrange(width - 2)
        right_col = random.randrange(left_col + 2, width)
        left = (row, left_col)
        right = (row, right_col)
        cells = _segment_cells(left, right)
        if not _cells_available(cells, reserved):
            continue
        grid[left] = color
        grid[right] = color
        reserved.update(cells)
        return True
    return False


def _place_horizontal_blocked(
    grid: np.ndarray, reserved: set[tuple[int, int]], color: int
) -> bool:
    height, width = grid.shape
    if width < 5:
        return False
    for _ in range(PAIR_PLACEMENT_RETRIES):
        row = random.randrange(height)
        left_col = random.randrange(width - 4)
        right_col = random.randrange(left_col + 4, width)
        obstacle_col = random.randrange(left_col + 2, right_col - 1)
        left = (row, left_col)
        right = (row, right_col)
        obstacle = (row, obstacle_col)
        cells = _segment_cells(left, right)
        if not _cells_available(cells, reserved):
            continue
        grid[left] = color
        grid[right] = color
        grid[obstacle] = GRAY
        reserved.update(cells)
        return True
    return False


def _place_vertical_clear(
    grid: np.ndarray, reserved: set[tuple[int, int]], color: int
) -> bool:
    height, width = grid.shape
    for _ in range(PAIR_PLACEMENT_RETRIES):
        col = random.randrange(width)
        top_row = random.randrange(height - 2)
        bottom_row = random.randrange(top_row + 2, height)
        top = (top_row, col)
        bottom = (bottom_row, col)
        cells = _segment_cells(top, bottom)
        if not _cells_available(cells, reserved):
            continue
        grid[top] = color
        grid[bottom] = color
        reserved.update(cells)
        return True
    return False


def _place_vertical_blocked(
    grid: np.ndarray, reserved: set[tuple[int, int]], color: int
) -> bool:
    height, width = grid.shape
    if height < 5:
        return False
    for _ in range(PAIR_PLACEMENT_RETRIES):
        col = random.randrange(width)
        top_row = random.randrange(height - 4)
        bottom_row = random.randrange(top_row + 4, height)
        obstacle_row = random.randrange(top_row + 2, bottom_row - 1)
        top = (top_row, col)
        bottom = (bottom_row, col)
        obstacle = (obstacle_row, col)
        cells = _segment_cells(top, bottom)
        if not _cells_available(cells, reserved):
            continue
        grid[top] = color
        grid[bottom] = color
        grid[obstacle] = GRAY
        reserved.update(cells)
        return True
    return False


def _place_diagonal(grid: np.ndarray, reserved: set[tuple[int, int]], color: int) -> bool:
    height, width = grid.shape
    for _ in range(PAIR_PLACEMENT_RETRIES):
        a = (random.randrange(height), random.randrange(width))
        b = (random.randrange(height), random.randrange(width))
        if a == b or a[0] == b[0] or a[1] == b[1]:
            continue
        if not _cells_available([a, b], reserved):
            continue
        grid[a] = color
        grid[b] = color
        reserved.add(a)
        reserved.add(b)
        return True
    return False


PLACERS = {
    "horizontal_clear": _place_horizontal_clear,
    "horizontal_blocked": _place_horizontal_blocked,
    "vertical_clear": _place_vertical_clear,
    "vertical_blocked": _place_vertical_blocked,
    "diagonal": _place_diagonal,
}


def _branch_plan() -> list[str]:
    branches = list(BRANCH_TYPES)
    if random.random() < 0.5:
        branches.append(random.choice(BRANCH_TYPES))
    random.shuffle(branches)
    return branches


def _endpoint_halo(
    endpoints: set[tuple[int, int]], height: int, width: int
) -> set[tuple[int, int]]:
    halo: set[tuple[int, int]] = set()
    for r, c in endpoints:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    halo.add((nr, nc))
    return halo


def _validate_structure(grid: np.ndarray, branch_counts: Counter[str]) -> None:
    assert grid.ndim == 2
    colors = [int(v) for v in np.unique(grid) if int(v) not in (BLACK, GRAY, MAROON)]
    assert len(colors) == sum(branch_counts.values())
    for color in colors:
        assert int(np.count_nonzero(grid == color)) == 2
    assert int(np.count_nonzero(grid == GRAY)) >= 1
    for branch in BRANCH_TYPES:
        assert branch_counts[branch] >= 1


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((height, width), BLACK, dtype=int)
    reserved: set[tuple[int, int]] = set()
    branches = _branch_plan()
    colors = random.sample(PAIR_COLORS, len(branches))
    branch_counts = Counter(branches)

    for color, branch in zip(colors, branches):
        if not PLACERS[branch](grid, reserved, color):
            return None

    endpoints = {
        (int(r), int(c))
        for color in colors
        for r, c in np.argwhere(grid == color)
    }
    halo = _endpoint_halo(endpoints, height, width)
    free_cells = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) not in reserved and (r, c) not in halo
    ]
    if not free_cells:
        free_cells = [(r, c) for r in range(height) for c in range(width) if (r, c) not in reserved]
    if not free_cells:
        return None

    extra_obstacles = random.randint(*EXTRA_OBSTACLE_RANGE)
    extra_obstacles = min(extra_obstacles, len(free_cells))
    for cell in random.sample(free_cells, extra_obstacles):
        grid[cell] = GRAY

    _validate_structure(grid, branch_counts)
    solved = main(grid)
    assert solved.shape == grid.shape
    assert np.any(solved != grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATE_RETRIES} attempts")
