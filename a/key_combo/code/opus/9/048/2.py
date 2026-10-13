import random
from collections import deque

import numpy as np


BG = 0
RED = 2
TEAL = 8

GRID_HEIGHT_RANGE = (11, 16)
GRID_WIDTH_RANGE = (10, 16)
TEAL_COUNT_RANGE = (2, 4)
RED_COUNT_RANGE = (2, 5)
GENERATION_RETRIES = 500

_BRANCH_CURSOR = 0
_RECENT_OUTPUT_COUNTS: list[int] = []


def _count_components(mask: np.ndarray) -> int:
    """Count 4-connected True components in a boolean mask."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            seen[r, c] = True
            queue = deque([(r, c)])
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
    return int(count)


def _draw_paths(grid: np.ndarray, *, mode: str = "exact") -> np.ndarray:
    """
    Draw the teal flow paths.

    mode:
    - "exact": match the puzzle rule.
    - "straight": ignore all red deflections.
    - "connected": also fill the skipped cell in a deflection column, creating a connected staircase.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    source_rows = [int(r) for r in np.where(g[:, 0] == TEAL)[0]]
    red_cols = {int(c) for c in np.where(g[0] == RED)[0]}

    for start_r in sorted(source_rows):
        r = int(start_r)
        out[r, 0] = TEAL
        for c in range(1, w):
            if mode == "straight":
                out[r, c] = TEAL
                continue

            if c in red_cols:
                if mode == "connected":
                    if r + 1 >= h:
                        raise ValueError("connected variant would leave the grid")
                    out[r, c] = TEAL
                    r += 1
                    out[r, c] = TEAL
                else:
                    r += 1
                    if r >= h:
                        raise ValueError("exact path would leave the grid")
                    out[r, c] = TEAL
            else:
                out[r, c] = TEAL

    return out


def _background_region_count(grid: np.ndarray) -> int:
    return _count_components(np.asarray(grid, dtype=int) == BG)


def _choose_rows(height: int, red_count: int, teal_count: int, branch: int) -> list[int] | None:
    max_start_row = int(height) - 1 - int(red_count)
    if max_start_row < 1:
        return None
    pool = list(range(1, max_start_row + 1))
    if len(pool) < int(teal_count):
        return None

    if branch % 2 == 1 and max_start_row - teal_count + 1 >= 1 and random.random() < 0.55:
        start = random.randint(1, max_start_row - teal_count + 1)
        return [start + i for i in range(int(teal_count))]

    return sorted(random.sample(pool, int(teal_count)))


def _choose_red_cols(width: int, red_count: int, branch: int) -> list[int]:
    if branch == 0:
        cols = set()
        attempts = 0
        while len(cols) < int(red_count) and attempts < 80:
            attempts += 1
            c = random.randint(1, int(width) - 1)
            if all(abs(c - other) >= 2 for other in cols):
                cols.add(c)
        while len(cols) < int(red_count):
            cols.add(random.randint(1, int(width) - 1))
        return sorted(cols)

    if branch == 1:
        run = min(random.randint(2, min(3, int(red_count))), int(width) - 1)
        start = random.randint(1, int(width) - run)
        cols = set(range(start, start + run))
        while len(cols) < int(red_count):
            cols.add(random.randint(1, int(width) - 1))
        return sorted(cols)

    if branch == 2:
        cols = {1}
        if red_count >= 4 and width >= 6 and random.random() < 0.65:
            cols.add(2)
        while len(cols) < int(red_count):
            cols.add(random.randint(1, int(width) - 1))
        return sorted(cols)

    cols = {int(width) - 1}
    if red_count >= 4 and width >= 6 and random.random() < 0.65:
        cols.add(int(width) - 2)
    while len(cols) < int(red_count):
        cols.add(random.randint(1, int(width) - 1))
    return sorted(cols)


def _make_candidate(branch: int) -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    red_count = random.randint(RED_COUNT_RANGE[0], min(RED_COUNT_RANGE[1], width - 1))
    teal_count = random.randint(*TEAL_COUNT_RANGE)
    rows = _choose_rows(height, red_count, teal_count, branch)
    if rows is None:
        return None

    red_cols = _choose_red_cols(width, red_count, branch)
    grid = np.zeros((height, width), dtype=int)
    grid[rows, 0] = TEAL
    grid[0, red_cols] = RED
    return grid


def _structure_ok(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if not set(map(int, np.unique(g))).issubset({BG, RED, TEAL}):
        return False
    if int(g[0, 0]) != BG:
        return False
    if np.any(g[1:, 1:] != BG):
        return False
    if np.any((g[:, 0] != BG) & (g[:, 0] != TEAL)):
        return False
    if np.any((g[0, :] != BG) & (g[0, :] != RED)):
        return False

    teal_rows = np.where(g[:, 0] == TEAL)[0]
    red_cols = np.where(g[0, :] == RED)[0]
    if not (TEAL_COUNT_RANGE[0] <= len(teal_rows) <= TEAL_COUNT_RANGE[1]):
        return False
    if not (RED_COUNT_RANGE[0] <= len(red_cols) <= RED_COUNT_RANGE[1]):
        return False
    if len(set(map(int, teal_rows))) != len(teal_rows):
        return False
    if len(set(map(int, red_cols))) != len(red_cols):
        return False
    if np.any(teal_rows == 0) or np.any(red_cols == 0):
        return False
    if int(teal_rows.max()) + len(red_cols) >= g.shape[0]:
        return False
    return True


def _semantic_ok(grid: np.ndarray) -> bool:
    if not _structure_ok(grid):
        return False

    try:
        exact = _draw_paths(grid, mode="exact")
        straight = _draw_paths(grid, mode="straight")
        connected = _draw_paths(grid, mode="connected")
    except ValueError:
        return False

    teal_count = int(np.count_nonzero(grid[:, 0] == TEAL))
    exact_count = _background_region_count(exact)
    straight_count = _background_region_count(straight)
    connected_count = _background_region_count(connected)

    if exact_count <= 1:
        return False
    if exact_count == teal_count + 1:
        return False
    if exact_count == straight_count:
        return False
    if exact_count == connected_count:
        return False
    if exact_count < 4:
        return False

    out = main(grid)
    if out.shape != (exact_count, 1):
        return False
    if np.any(out != BG):
        return False

    recent = _RECENT_OUTPUT_COUNTS[-4:]
    if exact_count in recent and random.random() < 0.8:
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw teal flow paths deflected by top-edge red pixels, count the black regions they create, and return that count as a black column.

    Input structure:
    - The grid uses black as background, with teal source pixels only on the left edge and red deflector pixels only on the top edge.
    - The interior starts empty; each teal pixel marks one path source and each red top-edge pixel marks one deflection column.

    Rule:
    - From each teal source, move right one column at a time until the right edge is reached.
    - When entering a column whose top-edge cell is red, place that path cell one row lower than the previous column; otherwise keep the same row.
    - Treat all drawn teal path pixels together with the original red pixels as foreground barriers.
    - Count the 4-connected black background regions and output an `N x 1` black grid, where `N` is that region count.

    Key insight:
    - The visible teal traces matter only as dynamically bent barriers: the red columns control where diagonal gaps appear, and those gaps determine how many black regions remain connected.
    """
    drawn = _draw_paths(np.asarray(grid, dtype=int), mode="exact")
    region_count = _background_region_count(drawn)
    return np.zeros((int(region_count), 1), dtype=int)


def generate_input() -> np.ndarray:
    global _BRANCH_CURSOR

    for _ in range(GENERATION_RETRIES):
        branch = _BRANCH_CURSOR % 4
        grid = _make_candidate(branch)
        if grid is None:
            continue
        if not _semantic_ok(grid):
            continue

        exact_count = int(main(grid).shape[0])
        _RECENT_OUTPUT_COUNTS.append(exact_count)
        if len(_RECENT_OUTPUT_COUNTS) > 12:
            del _RECENT_OUTPUT_COUNTS[:-12]
        _BRANCH_CURSOR += 1
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 2")
