from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREY = 5

MAX_ATTEMPTS = 400


def _control_length(grid: np.ndarray) -> int:
    row0 = np.asarray(grid[0], dtype=int)
    n = 0
    while n < row0.size and int(row0[n]) == RED:
        n += 1
    assert 3 <= n <= 5, "control bar length must be in [3, 5]"
    assert np.all(row0[n:] == BG), "row 0 must contain only the leading control bar"
    return n


def _run_lengths(n: int) -> list[int]:
    return list(range(1, n + 1)) + list(range(n - 1, 0, -1))


def _sequence_from_length(n: int) -> np.ndarray:
    vals: list[int] = []
    runs = _run_lengths(n)
    for i, run in enumerate(runs):
        vals.extend([RED] * run)
        if i != len(runs) - 1:
            vals.append(BG)
    return np.asarray(vals, dtype=int)


def _wrap_sequence(sequence: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    assert h >= 2 and w >= 1
    capacity = (h - 1) * w
    assert len(sequence) <= capacity, "grid is too short to hold the wrapped sequence"

    out = np.zeros((h, w), dtype=int)
    flat = out[1:].reshape(-1)
    flat[: len(sequence)] = sequence
    return out


def _settle_column(column: np.ndarray) -> np.ndarray:
    out = np.asarray(column, dtype=int).copy()
    out[out == RED] = BG

    obstacle_rows = np.flatnonzero(column == GREY)
    boundaries = [-1, *map(int, obstacle_rows), len(column)]
    for start_block, end_block in zip(boundaries, boundaries[1:]):
        r0 = start_block + 1
        r1 = end_block - 1
        if r0 > r1:
            continue
        count = int(np.sum(column[r0 : r1 + 1] == RED))
        if count == 0:
            continue
        out[r1 - count + 1 : r1 + 1] = RED
    return out


def _apply_gravity(pattern: np.ndarray, obstacle_mask: np.ndarray) -> np.ndarray:
    working = np.where(obstacle_mask, GREY, BG)
    red_cells = pattern == RED
    assert not np.any(obstacle_mask & red_cells), "obstacles must not overlap wrapped red cells"
    working[red_cells] = RED

    out = working.copy()
    for c in range(working.shape[1]):
        out[:, c] = _settle_column(working[:, c])
    return out


def _global_pack_by_column(pattern: np.ndarray, obstacle_mask: np.ndarray) -> np.ndarray:
    h, w = pattern.shape
    out = np.where(obstacle_mask, GREY, BG)
    for c in range(w):
        count = int(np.sum(pattern[:, c] == RED))
        free_rows = [r for r in range(h - 1, -1, -1) if not obstacle_mask[r, c]]
        for r in free_rows[:count]:
            out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert the control bar into a wrapped red run sequence, then let every column settle around gray blockers.

    Input structure:
    - Black background with one red control bar on row 0 starting at column 0.
    - Gray obstacle pixels may appear below row 0 and remain fixed.
    - No other foreground objects are present; the control bar length is the only generator for new red pixels.

    Rule:
    - Read the row-0 red bar length n and generate run lengths 1,2,...,n,n-1,...,1 with one black gap between consecutive red runs.
    - Clear row 0, wrap that one-dimensional sequence row-major from row 1 across the existing grid width, and leave unused tail cells black.
    - For each column independently, let its red pixels fall downward without crossing gray obstacles; within each obstacle-bounded vertical segment they stack at the bottom of that segment.

    Key insight:
    - The bar controls how many red pixels each wrapped column receives, while the obstacle layout decides which vertical segment each column can finally occupy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert set(map(int, np.unique(g))) <= {BG, RED, GREY}, "expected only black, red, and gray"
    assert not np.any(g[-1] == GREY), "obstacles must not occupy the bottom row"
    assert not np.any(g[0] == GREY), "obstacles must not occupy row 0"

    n = _control_length(g)
    sequence = _sequence_from_length(n)
    wrapped = _wrap_sequence(sequence, g.shape)
    obstacle_mask = g == GREY
    return _apply_gravity(wrapped, obstacle_mask)


def _column_red_rows(grid: np.ndarray, col: int) -> list[int]:
    return [int(r) for r in np.flatnonzero(grid[:, col] == RED)]


def _pick_split_obstacle(wrapped: np.ndarray) -> tuple[int, int] | None:
    h, w = wrapped.shape
    candidates: list[tuple[int, int]] = []
    for c in range(w):
        red_rows = _column_red_rows(wrapped, c)
        if len(red_rows) < 2:
            continue
        for r in range(1, h - 1):
            if wrapped[r, c] != BG:
                continue
            above = any(rr < r for rr in red_rows)
            below = any(rr > r for rr in red_rows)
            lower_capacity = (h - 1) - r
            reds_below = sum(rr > r for rr in red_rows)
            if above and below and lower_capacity > reds_below:
                candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _pick_landing_obstacle(wrapped: np.ndarray, used: set[tuple[int, int]]) -> tuple[int, int] | None:
    h, w = wrapped.shape
    candidates: list[tuple[int, int]] = []
    for c in range(w):
        red_rows = _column_red_rows(wrapped, c)
        if not red_rows:
            continue
        lowest = max(red_rows)
        for r in range(lowest + 1, h - 1):
            if wrapped[r, c] == BG and (r, c) not in used:
                candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _obstacle_height_signature(mask: np.ndarray) -> tuple[int | None, ...]:
    sig: list[int | None] = []
    for c in range(mask.shape[1]):
        rows = np.flatnonzero(mask[:, c])
        sig.append(None if len(rows) == 0 else int(rows[0]))
    return tuple(sig)


def _has_height_variation(mask: np.ndarray) -> bool:
    return len(set(_obstacle_height_signature(mask))) >= 2


def _semantic_checks(inp: np.ndarray) -> bool:
    h, w = inp.shape
    if h > 30 or w > 30:
        return False

    n = _control_length(inp)
    wrapped = _wrap_sequence(_sequence_from_length(n), inp.shape)
    obstacle_mask = inp == GREY

    if np.any(obstacle_mask & (wrapped == RED)):
        return False
    if np.array_equal(inp[1:], wrapped[1:]):
        return False
    if not _has_height_variation(obstacle_mask):
        return False

    out = main(inp)
    if np.array_equal(out, wrapped):
        return False

    obstacle_columns = {int(c) for c in np.flatnonzero(obstacle_mask.any(axis=0))}
    if len(obstacle_columns) < 2:
        return False

    no_obstacle_input = np.zeros_like(inp)
    no_obstacle_input[0, :n] = RED
    no_obstacle_out = main(no_obstacle_input)
    if np.array_equal(out, no_obstacle_out):
        return False

    simplified = _global_pack_by_column(wrapped, obstacle_mask)
    if np.array_equal(out, simplified):
        return False

    red_columns = [c for c in range(w) if np.any(out[:, c] == RED)]
    if len(red_columns) < 3:
        return False
    top_rows = [int(np.flatnonzero(out[:, c] == RED)[0]) for c in red_columns]
    if len(set(top_rows)) < 2:
        return False

    if not any(
        c in obstacle_columns and not np.array_equal(out[:, c], no_obstacle_out[:, c])
        for c in red_columns
    ):
        return False

    if not any(np.any(out[:, c] == RED) and obstacle_mask[:, c].any() for c in range(w)):
        return False

    return True


def _choose_shape(n: int) -> tuple[int, int] | None:
    seq_len = len(_sequence_from_length(n))
    for _ in range(80):
        w = random.randint(max(6, n + 2), 10)
        min_h = max(9, 2 + (seq_len + w - 1) // w)
        max_h = min(15, 30)
        if min_h > max_h:
            continue
        h = random.randint(min_h, max_h)
        return h, w
    return None


def _available_obstacle_cells(wrapped: np.ndarray, used: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = wrapped.shape
    cells = []
    for r in range(1, h - 1):
        for c in range(w):
            if wrapped[r, c] == BG and (r, c) not in used:
                cells.append((r, c))
    return cells


def _attempt_input() -> np.ndarray | None:
    n = random.randint(3, 5)
    shape = _choose_shape(n)
    if shape is None:
        return None
    h, w = shape

    wrapped = _wrap_sequence(_sequence_from_length(n), (h, w))
    split = _pick_split_obstacle(wrapped)
    if split is None:
        return None

    used = {split}
    landing = _pick_landing_obstacle(wrapped, used)
    if landing is None:
        return None
    used.add(landing)

    total_obstacles = random.randint(3, 6)
    extras = _available_obstacle_cells(wrapped, used)
    if len(extras) < total_obstacles - len(used):
        return None
    extra_obstacles = random.sample(extras, total_obstacles - len(used))

    grid = np.zeros((h, w), dtype=int)
    grid[0, :n] = RED
    for r, c in [*used, *extra_obstacles]:
        grid[r, c] = GREY

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
