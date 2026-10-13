import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
OUTER_MASK = np.ones((5, 5), dtype=bool)
OUTER_MASK[1:4, 1:4] = False
GEN_RETRIES = 200


def _mode(values) -> int:
    flat = [int(v) for v in np.asarray(values).ravel()]
    return Counter(flat).most_common(1)[0][0]


def _edge_colors(grid: np.ndarray) -> dict[tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    return {
        (-1, 0): _mode(g[0, 1:-1]),
        (0, 1): _mode(g[1:-1, -1]),
        (1, 0): _mode(g[-1, 1:-1]),
        (0, -1): _mode(g[1:-1, 0]),
    }


def _find_clocks(grid: np.ndarray) -> list[tuple[int, int, tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    clocks: list[tuple[int, int, tuple[int, int]]] = []

    for r in range(1, h - 5):
        for c in range(1, w - 5):
            block = g[r : r + 5, c : c + 5]
            frame_color = int(block[0, 0])
            if frame_color == BLACK:
                continue
            if not np.all(block[OUTER_MASK] == frame_color):
                continue

            inner = block[1:4, 1:4]
            center_color = int(inner[1, 1])
            if center_color == BLACK or center_color == frame_color:
                continue

            pointer_dirs = []
            for dr, dc in DIRS4:
                if int(inner[1 + dr, 1 + dc]) == center_color:
                    pointer_dirs.append((dr, dc))
            if len(pointer_dirs) != 1:
                continue

            colored_cells = {tuple(map(int, rc)) for rc in np.argwhere(inner == center_color)}
            dr, dc = pointer_dirs[0]
            if colored_cells != {(1, 1), (1 + dr, 1 + dc)}:
                continue

            fill_mask = np.ones((3, 3), dtype=bool)
            fill_mask[1, 1] = False
            fill_mask[1 + dr, 1 + dc] = False
            if not np.all(inner[fill_mask] == BLACK):
                continue

            clocks.append((r, c, (dr, dc)))

    return clocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each clock face using the side-band color indicated by its pointer.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    side_colors = _edge_colors(g)

    for r, c, direction in _find_clocks(g):
        face = out[r + 1 : r + 4, c + 1 : c + 4]
        face[face == BLACK] = int(side_colors[direction])

    return out


def _paint_bands(grid: np.ndarray, rng: random.Random) -> None:
    grid[0, :] = RED
    grid[-1, :] = GREEN
    grid[:, 0] = YELLOW
    grid[:, -1] = BLUE

    grid[0, 0] = rng.choice((RED, YELLOW))
    grid[0, -1] = rng.choice((RED, BLUE))
    grid[-1, 0] = rng.choice((GREEN, YELLOW))
    grid[-1, -1] = rng.choice((GREEN, BLUE))


def _sample_directions(rng: random.Random, count: int) -> list[tuple[int, int]]:
    pool = DIRS4[:]
    rng.shuffle(pool)
    if count <= len(pool):
        return pool[:count]

    result = pool[:]
    while len(result) < count:
        extra = DIRS4[:]
        rng.shuffle(extra)
        result.extend(extra)
    return result[:count]


def _too_close(pos_a: tuple[int, int], pos_b: tuple[int, int]) -> bool:
    r1, c1 = pos_a
    r2, c2 = pos_b
    rows_separated = r1 + 4 < r2 - 1 or r2 + 4 < r1 - 1
    cols_separated = c1 + 4 < c2 - 1 or c2 + 4 < c1 - 1
    return not (rows_separated or cols_separated)


def _place_positions(rng: random.Random, h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(2, h - 5) for c in range(2, w - 5)]
    rng.shuffle(candidates)

    positions: list[tuple[int, int]] = []
    for pos in candidates:
        if all(not _too_close(pos, existing) for existing in positions):
            positions.append(pos)
            if len(positions) == count:
                return positions
    return None


def _paint_clock(grid: np.ndarray, top: int, left: int, direction: tuple[int, int]) -> None:
    grid[top, left : left + 5] = GREY
    grid[top + 4, left : left + 5] = GREY
    grid[top : top + 5, left] = GREY
    grid[top : top + 5, left + 4] = GREY

    center_r = top + 2
    center_c = left + 2
    grid[center_r, center_c] = ORANGE
    dr, dc = direction
    grid[center_r + dr, center_c + dc] = ORANGE


def _valid_input(grid: np.ndarray, expected_count: int) -> bool:
    if grid.ndim != 2 or min(grid.shape) < 7:
        return False
    if not np.all(grid[0, 1:-1] == RED):
        return False
    if not np.all(grid[1:-1, -1] == BLUE):
        return False
    if not np.all(grid[-1, 1:-1] == GREEN):
        return False
    if not np.all(grid[1:-1, 0] == YELLOW):
        return False
    clocks = _find_clocks(grid)
    if len(clocks) != expected_count:
        return False
    return not np.array_equal(main(grid), grid)


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(GEN_RETRIES):
        h = rng.randint(15, 25)
        w = rng.randint(15, 25)
        row_slots = len(range(2, h - 5, 6))
        col_slots = len(range(2, w - 5, 6))
        max_clocks = min(6, row_slots * col_slots)
        if max_clocks < 2:
            continue

        clock_count = rng.randint(2, max_clocks)
        positions = _place_positions(rng, h, w, clock_count)
        if positions is None:
            continue

        grid = np.full((h, w), BLACK, dtype=int)
        _paint_bands(grid, rng)

        directions = _sample_directions(rng, clock_count)
        rng.shuffle(directions)
        for (top, left), direction in zip(positions, directions):
            _paint_clock(grid, top, left, direction)

        if _valid_input(grid, clock_count):
            return grid

    raise RuntimeError("generate_input: exceeded retry limit")
