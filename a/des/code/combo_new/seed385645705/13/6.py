from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
YELLOW = 4
ORANGE = 7
SEED_PALETTE = (1, 2, 3, 6, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_MIN = 12
GRID_MAX = 24
GEN_ATTEMPTS = 500
SEGMENT_MIN_LEN = 2
SEGMENT_MAX_LEN = 8

BRANCH_BLOCKED = "blocked"
BRANCH_YELLOW = "yellow"
BRANCH_ORANGE = "orange"


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _infer_wall_color(grid: np.ndarray, bg: int) -> int | None:
    colors = [int(color) for color in np.unique(grid) if int(color) != int(bg)]
    if not colors:
        return None
    counts = {color: int(np.count_nonzero(grid == int(color))) for color in colors}
    wall_color, wall_count = max(counts.items(), key=lambda item: item[1])
    if wall_count <= 1:
        return None
    return int(wall_color)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow orthogonal seed rays until obstacles and recolor crossings by how many seed colors meet there.

    Input structure:
    - The background is the dominant color.
    - One non-background color forms one-pixel-thick horizontal or vertical wall segments.
    - Two to four isolated seed pixels of different colors sit on background cells and do not touch walls.

    Rule:
    - Each seed emits one-pixel rays up, down, left, and right in the seed's own color.
    - A ray stops before a wall pixel, before another seed pixel, or when it would leave the grid.
    - A background pixel reached by exactly one seed keeps that seed color; a background pixel reached by exactly two different seed colors becomes yellow; a background pixel reached by three or more different seed colors becomes orange.
    - Wall cells and original seed cells stay unchanged.

    Key insight:
    - The only moving information is straight row/column visibility from each seed, so the output is determined by line-of-sight blocked by walls and by counting how many distinct seed rays can pass through each background cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = _background_color(g)
    wall_color = _infer_wall_color(g, bg)
    wall_mask = (g == int(wall_color)) if wall_color is not None else np.zeros((h, w), dtype=bool)
    seed_mask = (g != int(bg)) & ~wall_mask

    seed_coords = np.argwhere(seed_mask)
    assert 2 <= int(seed_coords.shape[0]) <= 4, "expected 2 to 4 seed pixels"

    seed_colors = [int(g[r, c]) for r, c in seed_coords]
    assert len(seed_colors) == len(set(seed_colors)), "seed colors must be unique"
    for r, c in seed_coords:
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w:
                assert not wall_mask[nr, nc], "seed must not touch a wall"

    ray_bits = np.zeros((h, w), dtype=np.uint8)

    for index, (r0_raw, c0_raw) in enumerate(seed_coords):
        r0 = int(r0_raw)
        c0 = int(c0_raw)
        for dr, dc in DIR4:
            r = r0
            c = c0
            while True:
                r += int(dr)
                c += int(dc)
                if not (0 <= r < h and 0 <= c < w):
                    break
                if wall_mask[r, c] or seed_mask[r, c]:
                    break
                assert int(g[r, c]) == int(bg), "rays may only pass through background cells"
                ray_bits[r, c] |= np.uint8(1 << index)

    out = g.copy()
    for r, c in np.argwhere(g == int(bg)):
        bitmask = int(ray_bits[r, c])
        if bitmask == 0:
            continue
        hit_count = int(bitmask.bit_count())
        if hit_count == 1:
            out[r, c] = int(seed_colors[int(bitmask.bit_length() - 1)])
        elif hit_count == 2:
            out[r, c] = YELLOW
        else:
            out[r, c] = ORANGE

    return out


def _vertical_segment_cells(height: int, col: int, through_row: int) -> list[tuple[int, int]]:
    max_len = min(SEGMENT_MAX_LEN, height)
    length = random.randint(SEGMENT_MIN_LEN, max_len)
    top_lo = max(0, through_row - length + 1)
    top_hi = min(through_row, height - length)
    top = random.randint(top_lo, top_hi)
    return [(r, col) for r in range(top, top + length)]


def _horizontal_segment_cells(width: int, row: int, through_col: int) -> list[tuple[int, int]]:
    max_len = min(SEGMENT_MAX_LEN, width)
    length = random.randint(SEGMENT_MIN_LEN, max_len)
    left_lo = max(0, through_col - length + 1)
    left_hi = min(through_col, width - length)
    left = random.randint(left_lo, left_hi)
    return [(row, c) for c in range(left, left + length)]


def _can_place_segment(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    seed_cells: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                if (nr, nc) in seed_cells:
                    return False
                if int(grid[nr, nc]) == WALL and (nr, nc) not in cell_set:
                    return False
    return True


def _paint_segment(grid: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        grid[r, c] = WALL


def _try_add_vertical_wall(
    grid: np.ndarray,
    through_row: int,
    col_lo: int,
    col_hi: int,
    seed_cells: set[tuple[int, int]],
    attempts: int = 40,
) -> bool:
    if col_lo > col_hi:
        return False
    for _ in range(attempts):
        col = random.randint(col_lo, col_hi)
        cells = _vertical_segment_cells(grid.shape[0], col, through_row)
        if _can_place_segment(grid, cells, seed_cells):
            _paint_segment(grid, cells)
            return True
    return False


def _try_add_horizontal_wall(
    grid: np.ndarray,
    through_col: int,
    row_lo: int,
    row_hi: int,
    seed_cells: set[tuple[int, int]],
    attempts: int = 40,
) -> bool:
    if row_lo > row_hi:
        return False
    for _ in range(attempts):
        row = random.randint(row_lo, row_hi)
        cells = _horizontal_segment_cells(grid.shape[1], row, through_col)
        if _can_place_segment(grid, cells, seed_cells):
            _paint_segment(grid, cells)
            return True
    return False


def _seed_ok(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != BG:
        return False
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) != BG:
            return False
    return True


def _add_seed(grid: np.ndarray, r: int, c: int, color: int) -> None:
    assert _seed_ok(grid, r, c), "seed location must be isolated and on background"
    grid[r, c] = int(color)


def _random_shape() -> tuple[int, int]:
    return random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX)


def _apply_random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.array(out, dtype=int, copy=True)


def _validate_structure(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False

    bg = _background_color(grid)
    wall_color = _infer_wall_color(grid, bg)
    if wall_color is None:
        return False

    wall_mask = grid == int(wall_color)
    if int(np.count_nonzero(wall_mask)) < 2:
        return False

    seed_coords = np.argwhere((grid != int(bg)) & ~wall_mask)
    if not (2 <= int(seed_coords.shape[0]) <= 4):
        return False

    colors = [int(grid[r, c]) for r, c in seed_coords]
    if len(colors) != len(set(colors)):
        return False

    for color in colors:
        if int(np.count_nonzero(grid == int(color))) != 1:
            return False

    for r, c in seed_coords:
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if wall_mask[nr, nc]:
                    return False
    return True


def _output_matches_branch(inp: np.ndarray, branch: str) -> bool:
    out = main(inp)
    yellow_count = int(np.count_nonzero(out == YELLOW))
    orange_count = int(np.count_nonzero(out == ORANGE))
    if np.array_equal(inp, out):
        return False
    if branch == BRANCH_BLOCKED:
        return yellow_count == 0 and orange_count == 0
    if branch == BRANCH_YELLOW:
        return yellow_count > 0 and orange_count == 0
    assert branch == BRANCH_ORANGE
    return orange_count > 0


def _build_blocked_pair() -> np.ndarray | None:
    h, w = _random_shape()
    grid = np.full((h, w), BG, dtype=int)

    row = random.randint(3, h - 4)
    left = random.randint(2, w - 8)
    right = random.randint(left + 4, w - 3)
    colors = random.sample(SEED_PALETTE, 2)

    _add_seed(grid, row, left, colors[0])
    _add_seed(grid, row, right, colors[1])
    seed_cells = {(row, left), (row, right)}

    if not _try_add_vertical_wall(grid, row, left + 2, right - 2, seed_cells):
        return None

    if random.random() < 0.5:
        _try_add_vertical_wall(grid, row, 0, left - 2, seed_cells)
    if random.random() < 0.5:
        _try_add_vertical_wall(grid, row, right + 2, w - 1, seed_cells)

    return grid


def _build_yellow_pair() -> np.ndarray | None:
    h, w = _random_shape()
    grid = np.full((h, w), BG, dtype=int)

    row = random.randint(3, h - 4)
    left = random.randint(2, w - 8)
    right = random.randint(left + 4, w - 3)
    colors = random.sample(SEED_PALETTE, 2)

    _add_seed(grid, row, left, colors[0])
    _add_seed(grid, row, right, colors[1])
    seed_cells = {(row, left), (row, right)}

    wall_placed = False
    choices = []
    if left >= 2:
        choices.append("left")
    if right <= w - 3:
        choices.append("right")
    random.shuffle(choices)
    target_count = 1 if len(choices) == 1 else random.randint(1, len(choices))
    for side in choices[:target_count]:
        if side == "left":
            wall_placed |= _try_add_vertical_wall(grid, row, 0, left - 2, seed_cells)
        else:
            wall_placed |= _try_add_vertical_wall(grid, row, right + 2, w - 1, seed_cells)
    if not wall_placed:
        return None

    return grid


def _build_orange_scene() -> np.ndarray | None:
    h, w = _random_shape()
    grid = np.full((h, w), BG, dtype=int)

    row = random.randint(4, h - 5)
    left = random.randint(2, w - 10)
    right = random.randint(left + 6, w - 3)
    center = random.randint(left + 2, right - 2)

    use_four = random.random() < 0.5
    top = random.randint(1, row - 2)
    bottom = random.randint(row + 2, h - 2) if use_four else None
    colors = random.sample(SEED_PALETTE, 4 if use_four else 3)

    _add_seed(grid, row, left, colors[0])
    _add_seed(grid, row, right, colors[1])
    _add_seed(grid, top, center, colors[2])
    seed_cells = {(row, left), (row, right), (top, center)}
    if use_four and bottom is not None:
        _add_seed(grid, bottom, center, colors[3])
        seed_cells.add((bottom, center))

    wall_placed = False
    wall_placed |= _try_add_vertical_wall(grid, row, 0, left - 2, seed_cells)
    wall_placed |= _try_add_vertical_wall(grid, row, right + 2, w - 1, seed_cells)
    wall_placed |= _try_add_horizontal_wall(grid, center, 0, top - 2, seed_cells)
    if use_four and bottom is not None:
        wall_placed |= _try_add_horizontal_wall(grid, center, bottom + 2, h - 1, seed_cells)
    if not wall_placed:
        return None

    return grid


def _gen_attempt(branch: str) -> np.ndarray | None:
    if branch == BRANCH_BLOCKED:
        grid = _build_blocked_pair()
    elif branch == BRANCH_YELLOW:
        grid = _build_yellow_pair()
    else:
        grid = _build_orange_scene()
    if grid is None:
        return None

    grid = _apply_random_symmetry(grid)
    if not _validate_structure(grid):
        return None
    if not _output_matches_branch(grid, branch):
        return None
    return grid


def generate_input() -> np.ndarray:
    branches = [BRANCH_BLOCKED, BRANCH_YELLOW, BRANCH_ORANGE]
    for _ in range(GEN_ATTEMPTS):
        branch = random.choice(branches)
        grid = _gen_attempt(branch)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed-ray obstacle task")
