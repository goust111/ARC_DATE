from __future__ import annotations

import hashlib
import random

import numpy as np


BG = 0
TOKEN = 1
RED = 2
YELLOW = 4
DIVIDER = 5

MAX_ATTEMPTS = 600
_SEEN_INPUTS: set[str] = set()
_SEEN_OUTPUTS: set[str] = set()


def _full_width_divider_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == DIVIDER)]
    assert len(rows) == 1, "expected exactly one full-width grey divider row"
    return int(rows[0])


def _extract_master(grid: np.ndarray, divider_r: int) -> np.ndarray:
    lower = grid[divider_r + 1 :]
    mask = np.isin(lower, (RED, YELLOW))
    rows = np.flatnonzero(mask.any(axis=1))
    assert rows.size > 0, "expected a colored motif below the divider"
    r0 = int(rows[0])
    r1 = int(rows[-1])
    master = lower[r0 : r1 + 1, :].copy()
    assert 2 <= master.shape[0] <= 3, "master motif must be 2-3 rows tall"
    assert np.any(master == RED) and np.any(master == YELLOW), "both motif colors must appear"
    return master


def _column_heights(grid: np.ndarray, divider_r: int) -> list[int]:
    upper = grid[:divider_r]
    heights = [int(np.count_nonzero(upper[:, c] == TOKEN)) for c in range(grid.shape[1])]
    assert all(h >= 1 for h in heights), "every column must contain at least one token"
    return heights


def _render_output(master: np.ndarray, heights: list[int]) -> np.ndarray:
    h, w = master.shape
    max_height = max(heights)
    out = np.zeros((h * max_height, w), dtype=int)
    for c, stack_h in enumerate(heights):
        start = h * (max_height - stack_h)
        for k in range(stack_h):
            r0 = start + k * h
            out[r0 : r0 + h, c] = master[:, c]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert token stack heights above a divider into per-column vertical repetitions of the lower motif.

    Input structure:
    - A black background grid contains exactly one full-width grey divider row.
    - Above the divider, blue single-cell tokens are scattered, with at least one token in every column.
    - Below the divider, a 2-3-row full-width motif made from red and yellow pixels sits in the lower panel.

    Rule:
    - Let every blue token fall straight down within its own column until the tokens stack immediately above the grey divider.
    - For each column c, count the resulting stack height H_c; this count equals the number of repetitions for that column.
    - Take the lower motif as the master pattern and repeat its column-c vertical strip H_c times, bottom-aligned inside an output of height master_height * max(H_c); unused cells stay black.

    Key insight:
    - The upper panel does not contribute visible output pixels directly; after gravity, each column's token count becomes a repetition factor that vertically stretches the corresponding column of the lower motif.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider_r = _full_width_divider_row(g)
    master = _extract_master(g, divider_r)
    heights = _column_heights(g, divider_r)
    assert master.shape[1] == g.shape[1], "upper width and motif width must match"
    return _render_output(master, heights)


def _pattern_pool(height: int) -> list[tuple[int, ...]]:
    pool: list[tuple[int, ...]] = []
    for code in range(3 ** height):
        vals = []
        x = code
        for _ in range(height):
            vals.append((BG, RED, YELLOW)[x % 3])
            x //= 3
        pat = tuple(vals)
        if all(v == BG for v in pat):
            continue
        pool.append(pat)
    return pool


def _sample_master(height: int, width: int) -> np.ndarray:
    patterns = _pattern_pool(height)
    while True:
        unique_count = random.randint(3, min(5, len(patterns), width))
        chosen = random.sample(patterns, unique_count)
        cols = [random.choice(chosen) for _ in range(width)]
        if len(set(cols)) < 2:
            continue
        if not any(cols.count(col) >= 2 for col in set(cols)):
            continue
        master = np.array(cols, dtype=int).T
        if not np.any(master == RED) or not np.any(master == YELLOW):
            continue
        if not np.any(master == BG):
            continue
        if np.any(np.all(master == BG, axis=0)):
            continue
        if not np.all(np.any(master != BG, axis=1)):
            continue
        return master


def _sample_heights(width: int, max_height: int) -> list[int]:
    target_distinct = min(max_height, 3)
    while True:
        heights = [random.randint(1, max_height) for _ in range(width)]
        if len(set(heights)) < 2:
            continue
        if len(set(heights)) < target_distinct:
            continue
        return heights


def _scatter_rows(top_h: int, height: int) -> list[int]:
    rows = sorted(random.sample(range(top_h), height))
    return rows


def _semantic_checks(inp: np.ndarray, expected_out: np.ndarray) -> bool:
    divider_r = _full_width_divider_row(inp)
    master = _extract_master(inp, divider_r)
    heights = _column_heights(inp, divider_r)

    if not np.array_equal(main(inp), expected_out):
        return False
    if len(set(heights)) < 2:
        return False

    col_tuples = [tuple(map(int, master[:, c])) for c in range(master.shape[1])]
    same_pattern_diff_height = any(
        col_tuples[i] == col_tuples[j] and heights[i] != heights[j]
        for i in range(len(heights))
        for j in range(i + 1, len(heights))
    )
    same_height_diff_pattern = any(
        heights[i] == heights[j] and col_tuples[i] != col_tuples[j]
        for i in range(len(heights))
        for j in range(i + 1, len(heights))
    )
    if not same_pattern_diff_height:
        return False
    if not same_height_diff_pattern:
        return False

    already_stacked_cols = 0
    upper = inp[:divider_r]
    for c, h in enumerate(heights):
        final_rows = set(range(divider_r - h, divider_r))
        token_rows = set(map(int, np.flatnonzero(upper[:, c] == TOKEN)))
        if token_rows == final_rows:
            already_stacked_cols += 1
    if already_stacked_cols == len(heights):
        return False

    if expected_out.shape[0] == master.shape[0]:
        return False
    if len({tuple(expected_out[:, c]) for c in range(expected_out.shape[1])}) < 2:
        return False
    return True


def _attempt() -> np.ndarray | None:
    width = random.randint(7, 11)
    master_h = random.choice((2, 3))
    max_height = random.randint(2, 4)
    top_h = random.randint(max_height + 2, min(9, max_height + 5))
    lower_gap = random.randint(0, 2)
    lower_tail = random.randint(0, 2)
    total_h = top_h + 1 + lower_gap + master_h + lower_tail
    if total_h > 30:
        return None

    master = _sample_master(master_h, width)
    heights = _sample_heights(width, max_height)

    grid = np.zeros((total_h, width), dtype=int)
    divider_r = top_h
    grid[divider_r, :] = DIVIDER

    for c, h in enumerate(heights):
        rows = _scatter_rows(top_h, h)
        for r in rows:
            grid[r, c] = TOKEN

    master_r0 = divider_r + 1 + lower_gap
    grid[master_r0 : master_r0 + master_h, :] = master

    out = _render_output(master, heights)
    if max(out.shape) > 30:
        return None
    if not _semantic_checks(grid, out):
        return None

    in_sig = hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()
    out_sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
    if in_sig in _SEEN_INPUTS or out_sig in _SEEN_OUTPUTS:
        return None
    _SEEN_INPUTS.add(in_sig)
    _SEEN_OUTPUTS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate input for puzzle 2")
