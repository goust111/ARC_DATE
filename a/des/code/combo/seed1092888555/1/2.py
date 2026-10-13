import random

import numpy as np


BG = 8
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 9]
GRID_HEIGHT_RANGE = (8, 18)
GRID_WIDTH_RANGE = (8, 18)
TRACK_COUNT_RANGE = (2, 6)
MAX_PIXELS_PER_TRACK = 5
MAX_GENERATION_ATTEMPTS = 400


def _diag_cells(h: int, w: int, d: int) -> list[tuple[int, int]]:
    if d >= 0:
        r, c = d, 0
    else:
        r, c = 0, -d

    cells: list[tuple[int, int]] = []
    while r < h and c < w:
        cells.append((r, c))
        r += 1
        c += 1
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """Compress each occupied main-diagonal track to one edge based on parity."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    out = np.full((h, w), BG, dtype=int)
    for d in range(-(w - 1), h):
        cells = _diag_cells(h, w, d)
        colors = [int(g[r, c]) for r, c in cells if int(g[r, c]) != BG]
        if not colors:
            continue

        n = len(colors)
        target_cells = cells[:n] if n % 2 == 1 else cells[-n:]
        for (r, c), color in zip(target_cells, colors):
            out[r, c] = color

    return out


def _valid_track_ids(h: int, w: int) -> list[int]:
    return [d for d in range(-(w - 1), h) if len(_diag_cells(h, w, d)) >= 2]


def _count_choices(length: int, parity: int) -> list[int]:
    upper = min(MAX_PIXELS_PER_TRACK, length)
    return [n for n in range(1, upper + 1) if n % 2 == parity]


def _sample_non_target_indices(length: int, count: int, parity: int) -> list[int] | None:
    target = list(range(count)) if parity == 1 else list(range(length - count, length))

    for _ in range(80):
        indices = sorted(random.sample(range(length), count))
        if indices != target:
            return indices
    return None


def _balanced_parities(track_count: int) -> list[int]:
    odd = track_count // 2
    even = track_count // 2
    if track_count % 2 == 1:
        if random.random() < 0.5:
            odd += 1
        else:
            even += 1
    parities = [1] * odd + [0] * even
    random.shuffle(parities)
    return parities


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    valid_ids = _valid_track_ids(h, w)
    if len(valid_ids) < TRACK_COUNT_RANGE[0]:
        return None

    track_count = random.randint(TRACK_COUNT_RANGE[0], min(TRACK_COUNT_RANGE[1], len(valid_ids)))
    diag_ids = random.sample(valid_ids, track_count)
    parities = _balanced_parities(track_count)

    grid = np.full((h, w), BG, dtype=int)
    branch_counts = {0: 0, 1: 0}

    for d, parity in zip(diag_ids, parities):
        cells = _diag_cells(h, w, d)
        choices = _count_choices(len(cells), parity)
        if not choices:
            return None

        count = random.choice(choices)
        indices = _sample_non_target_indices(len(cells), count, parity)
        if indices is None:
            return None

        branch_counts[parity] += 1
        for idx in indices:
            r, c = cells[idx]
            grid[r, c] = random.choice(COLOR_POOL)

    out = main(grid)
    if np.array_equal(grid, out):
        return None
    if track_count >= 2 and (branch_counts[0] == 0 or branch_counts[1] == 0):
        return None
    if np.count_nonzero(grid != BG) < 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid diagonal-compression input")
