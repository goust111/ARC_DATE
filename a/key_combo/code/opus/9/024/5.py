from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4

HEIGHT_CHOICES = [11, 13, 15, 17, 19]
WIDTH_CHOICES = [11, 13, 15, 17, 19]
SIGNAL_COUNT_RANGE = (9, 15)
NOISE_COUNT_CAP = 10
TOTAL_FOREGROUND_RANGE = (15, 25)
MAX_GENERATION_ATTEMPTS = 400
SIGNAL_STEPS = [
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
    (-2, 0),
    (2, 0),
    (0, -2),
    (0, 2),
]


def main(grid: np.ndarray) -> np.ndarray:
    """Filter foreground by row-column parity, then mirror the surviving signal across the vertical axis.

    Input structure:
    - The grid has a black background and odd height/width, so a vertical center column exists.
    - Foreground cells appear in two visible roles: signal cells lie on even `(row + col)` parity in the left half, while noise cells lie on odd parity.
    - Signal cells are placed so that their left-right mirror positions are empty in the input.

    Rule:
    - Treat every non-black cell with even `(row + col)` parity as signal and every non-black cell with odd parity as noise.
    - Remove all noise cells.
    - Keep each signal cell in place.
    - Reflect every signal cell across the grid's vertical center column; paint each newly created mirror cell yellow, but do not overwrite an already kept signal cell.

    Key insight:
    - The true object is not defined by color alone; the useful pixels are exactly the foreground cells whose coordinates satisfy the even parity test, and only that filtered set should be mirrored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1, "grid must have odd height and width"

    rows, cols = np.indices(g.shape)
    signal_mask = (g != BLACK) & ((rows + cols) % 2 == 0)

    out = np.zeros_like(g)
    out[signal_mask] = g[signal_mask]

    c_mid = w // 2
    signal_rows, signal_cols = np.where(signal_mask)
    mirror_cols = 2 * c_mid - signal_cols
    valid = (0 <= mirror_cols) & (mirror_cols < w)
    for r, c in zip(signal_rows[valid], mirror_cols[valid]):
        if out[r, c] == BLACK:
            out[r, c] = YELLOW

    return out


def _signal_cells(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    c_mid = w // 2
    return [(r, c) for r in range(h) for c in range(c_mid) if (r + c) % 2 == 0]


def _noise_cells(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    return [(r, c) for r in range(h) for c in range(w) if (r + c) % 2 == 1]


def _grow_signal(shape: tuple[int, int], count: int) -> set[tuple[int, int]] | None:
    allowed = set(_signal_cells(shape))
    h, w = shape
    c_mid = w // 2
    interior_starts = [
        (r, c)
        for r, c in allowed
        if 1 <= r < h - 1 and c < c_mid - 1
    ]
    starts = interior_starts or list(allowed)
    start = random.choice(starts)
    used = {start}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(cell: tuple[int, int]) -> None:
        r, c = cell
        for dr, dc in SIGNAL_STEPS:
            nxt = (r + dr, c + dc)
            if nxt in allowed and nxt not in used:
                frontier.add(nxt)

    add_frontier(start)
    while len(used) < count:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        used.add(cell)
        add_frontier(cell)

    rows = sorted(r for r, _ in used)
    cols = sorted(c for _, c in used)
    if len(set(rows)) < 3 or len(set(cols)) < 2:
        return None
    if rows[-1] - rows[0] < 3 or cols[-1] - cols[0] < 2:
        return None
    return used


def _choose_counts() -> tuple[int, int]:
    signal_count = random.randint(*SIGNAL_COUNT_RANGE)
    noise_low = max(4, TOTAL_FOREGROUND_RANGE[0] - signal_count)
    noise_high = min(NOISE_COUNT_CAP, signal_count - 1, TOTAL_FOREGROUND_RANGE[1] - signal_count)
    if noise_low > noise_high:
        raise RuntimeError("invalid count bounds")
    noise_count = random.randint(noise_low, noise_high)
    return signal_count, noise_count


def _sample_noise(shape: tuple[int, int], count: int, preferred_rows: set[int]) -> set[tuple[int, int]] | None:
    h, w = shape
    c_mid = w // 2
    all_cells = set(_noise_cells(shape))
    left_cells = [cell for cell in all_cells if cell[1] < c_mid]
    right_cells = [cell for cell in all_cells if cell[1] > c_mid]
    mid_cells = [cell for cell in all_cells if cell[1] == c_mid]
    near_rows = [cell for cell in all_cells if cell[0] in preferred_rows]

    mode = random.choice(["balanced", "left_bias", "right_bias"])
    chosen: set[tuple[int, int]] = set()

    def pick(pool: list[tuple[int, int]], need: int) -> bool:
        available = [cell for cell in pool if cell not in chosen]
        if len(available) < need:
            return False
        chosen.update(random.sample(available, need))
        return True

    if count >= 2:
        if mode == "balanced":
            if not pick(left_cells, 1) or not pick(right_cells, 1):
                return None
        elif mode == "left_bias":
            if not pick(left_cells, min(len(left_cells), max(2, count // 2))):
                return None
        else:
            if not pick(right_cells, min(len(right_cells), max(2, count // 2))):
                return None

    if near_rows and random.random() < 0.75:
        need = min(len(near_rows), max(1, count // 3))
        if not pick(near_rows, need):
            return None

    if mid_cells and random.random() < 0.5:
        pick(mid_cells, 1)

    remaining = count - len(chosen)
    if remaining < 0:
        if len(chosen) == count:
            return chosen
        return None
    if not pick(list(all_cells), remaining):
        return None
    return chosen


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    c_mid = w // 2
    out = np.zeros_like(grid)
    signal_coords = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLACK and (r + c) % 2 == 0:
                out[r, c] = int(grid[r, c])
                signal_coords.append((r, c))
    for r, c in signal_coords:
        mc = 2 * c_mid - c
        if out[r, mc] == BLACK:
            out[r, mc] = YELLOW
    return out


def _validate_instance(inp: np.ndarray) -> bool:
    if not isinstance(inp, np.ndarray) or inp.ndim != 2:
        return False
    h, w = inp.shape
    if max(inp.shape) > 30 or h % 2 == 0 or w % 2 == 0:
        return False
    if not set(np.unique(inp)).issubset({BLACK, RED, GREEN}):
        return False

    green = np.argwhere(inp == GREEN)
    red = np.argwhere(inp == RED)
    signal_count = int(len(green))
    noise_count = int(len(red))
    if not (signal_count > noise_count and TOTAL_FOREGROUND_RANGE[0] <= signal_count + noise_count <= TOTAL_FOREGROUND_RANGE[1]):
        return False
    if signal_count < SIGNAL_COUNT_RANGE[0]:
        return False
    if noise_count < 4:
        return False

    c_mid = w // 2
    if any((int(r + c) % 2) != 0 or int(c) >= c_mid for r, c in green):
        return False
    if any((int(r + c) % 2) != 1 for r, c in red):
        return False

    rows = [int(r) for r, _ in green]
    cols = [int(c) for _, c in green]
    if len(set(rows)) < 3 or len(set(cols)) < 2:
        return False
    if max(rows) - min(rows) < 3 or max(cols) - min(cols) < 2:
        return False

    mirrors = {(int(r), 2 * c_mid - int(c)) for r, c in green}
    if len(mirrors) != signal_count:
        return False
    for r, c in mirrors:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(inp[r, c]) != BLACK:
            return False

    left_noise = sum(1 for _, c in red if int(c) < c_mid)
    right_noise = sum(1 for _, c in red if int(c) > c_mid)
    if left_noise == 0 or right_noise == 0:
        return False

    expected = _oracle_output(inp)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, actual):
        return False
    if set(np.unique(actual)) - {BLACK, GREEN, YELLOW}:
        return False
    if int(np.count_nonzero(actual == GREEN)) != signal_count:
        return False
    if int(np.count_nonzero(actual == YELLOW)) != signal_count:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    shape = (random.choice(HEIGHT_CHOICES), random.choice(WIDTH_CHOICES))
    signal_count, noise_count = _choose_counts()
    signal = _grow_signal(shape, signal_count)
    if signal is None:
        return None

    noise = _sample_noise(shape, noise_count, preferred_rows={r for r, _ in signal})
    if noise is None:
        return None

    grid = np.zeros(shape, dtype=int)
    for r, c in signal:
        grid[r, c] = GREEN
    for r, c in noise:
        grid[r, c] = RED

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid parity-mirror ARC input")
