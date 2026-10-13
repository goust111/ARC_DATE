from __future__ import annotations

import random

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_HEIGHT_RANGE = (8, 18)
GRID_WIDTH_RANGE = (8, 18)
ACTIVE_COLUMN_MIN = 4
ACTIVE_COLUMN_MAX = 7
PALETTE_SIZE_RANGE = (4, 6)
MAX_SEQUENCE_LEN = 7
GENERATION_ATTEMPTS = 400


def _column_sequence_bottom_up(column: np.ndarray) -> list[int]:
    rows = np.flatnonzero(column != BLACK)
    return [int(column[r]) for r in rows[::-1]]


def _stack_reduce(colors_bottom_up: list[int]) -> list[int]:
    stack: list[int] = []
    for color in colors_bottom_up:
        if stack and stack[-1] == color:
            stack.pop()
        else:
            stack.append(int(color))
    return stack


def _pair_elimination_reduce(colors_bottom_up: list[int]) -> tuple[list[int], int]:
    work = list(map(int, colors_bottom_up))
    rounds = 0
    while True:
        next_work: list[int] = []
        changed = False
        i = 0
        while i < len(work):
            if i + 1 < len(work) and work[i] == work[i + 1]:
                changed = True
                i += 2
            else:
                next_work.append(work[i])
                i += 1
        if not changed:
            return next_work, rounds
        rounds += 1
        work = next_work


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let each column fall straight down while equal-color collisions annihilate until the column stabilizes.

    Input structure:
    - The grid uses black background and scattered non-black pixels.
    - Each column may contain several colored pixels separated by arbitrary vertical gaps.
    - Pixels never move sideways, so columns are independent.

    Rule:
    - Process every column independently from bottom to top.
    - A pixel falls until it lands on the floor or on top of the current settled stack in that column.
    - If the falling pixel would land directly above a settled pixel of the same color, both pixels disappear.
    - After all pixels in a column have been processed, place the surviving pixels packed against the bottom in their final order.

    Key insight:
    - Exact row coordinates are irrelevant; each column depends only on its bottom-to-top color sequence, reduced by a stack that pops when two equal colors become adjacent.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    height, width = g.shape
    out = np.zeros_like(g)
    for c in range(width):
        reduced = _stack_reduce(_column_sequence_bottom_up(g[:, c]))
        for i, color in enumerate(reduced):
            out[height - 1 - i, c] = int(color)
    return out


def _build_stable_sequence_of_length(palette: list[int], length: int) -> list[int]:
    seq = [random.choice(palette)]
    while len(seq) < length:
        color = random.choice(palette)
        if color != seq[-1]:
            seq.append(color)
    return seq


def _build_stable_sequence(palette: list[int], max_len: int) -> list[int]:
    return _build_stable_sequence_of_length(palette, random.randint(2, max_len))


def _build_one_round_sequence(palette: list[int], max_len: int) -> list[int]:
    for _ in range(100):
        length = random.randint(3, min(max_len, 6))
        seq = _build_stable_sequence_of_length(palette, length)
        pair_start = random.randint(0, length - 2)
        seq[pair_start + 1] = seq[pair_start]
        reduced, rounds = _pair_elimination_reduce(seq)
        if rounds == 1 and 0 < len(reduced) < len(seq):
            return seq
    a, b = random.sample(palette, 2)
    return [a, a, b]


def _build_chain_empty_sequence(palette: list[int], max_len: int) -> list[int]:
    options: list[list[int]] = []
    a, b = random.sample(palette, 2)
    options.append([a, b, b, a])
    if max_len >= 6:
        a, b, c = random.sample(palette, 3)
        options.append([a, b, c, c, b, a])
        a, b, c = random.sample(palette, 3)
        options.append([a, a, b, c, c, b])
    random.shuffle(options)
    for seq in options:
        reduced, rounds = _pair_elimination_reduce(seq)
        if rounds >= 2 and not reduced:
            return seq
    return [a, b, b, a]


def _build_chain_residue_sequence(palette: list[int], max_len: int) -> list[int]:
    options: list[list[int]] = []
    a, b, c = random.sample(palette, 3)
    options.append([c, a, b, b, a])
    options.append([a, b, b, a, c])
    if max_len >= 7:
        a, b, c, d = random.sample(palette, 4)
        options.append([d, a, b, c, c, b, a])
    random.shuffle(options)
    for seq in options:
        reduced, rounds = _pair_elimination_reduce(seq)
        if rounds >= 2 and reduced:
            return seq
    return [c, a, b, b, a]


def _build_sequence(kind: str, palette: list[int], max_len: int) -> list[int]:
    if kind == "stable":
        return _build_stable_sequence(palette, max_len)
    if kind == "one_round":
        return _build_one_round_sequence(palette, max_len)
    if kind == "chain_empty":
        return _build_chain_empty_sequence(palette, max_len)
    if kind == "chain_residue":
        return _build_chain_residue_sequence(palette, max_len)
    raise ValueError(f"unknown sequence kind: {kind}")


def _choose_active_columns(width: int, count: int) -> list[int]:
    candidates = [list(range(offset, width, 2)) for offset in (0, 1)]
    feasible = [cols for cols in candidates if len(cols) >= count]
    assert feasible, "grid width too small for spaced active columns"
    return sorted(random.sample(random.choice(feasible), count))


def _sample_rows(height: int, count: int) -> list[int]:
    assert 1 <= count <= height - 2
    for _ in range(60):
        rows = sorted(random.sample(range(height - 1), count))
        if any(b - a > 1 for a, b in zip(rows, rows[1:])):
            return rows
    return sorted(random.sample(range(height - 1), count))


def _paint_sequence(grid: np.ndarray, column: int, colors_bottom_up: list[int]) -> None:
    rows = _sample_rows(grid.shape[0], len(colors_bottom_up))
    for color, row in zip(colors_bottom_up, rows[::-1]):
        grid[int(row), int(column)] = int(color)


def _validate_example(grid: np.ndarray) -> bool:
    out = main(grid)
    if grid.shape != out.shape or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    active_cols = [c for c in range(grid.shape[1]) if np.any(grid[:, c] != BLACK)]
    if len(active_cols) < ACTIVE_COLUMN_MIN:
        return False

    stable_cols = 0
    one_round_cols = 0
    chain_cols = 0
    empty_after_cols = 0
    nonempty_after_cols = 0
    visibly_changed_cols = 0
    gapped_cols = 0

    for c in active_cols:
        in_col = grid[:, c]
        out_col = out[:, c]
        input_seq = _column_sequence_bottom_up(in_col)
        stack_result = _stack_reduce(input_seq)
        pair_result, rounds = _pair_elimination_reduce(input_seq)
        if stack_result != pair_result:
            return False
        if _column_sequence_bottom_up(out_col) != pair_result:
            return False
        if not np.array_equal(in_col, out_col):
            visibly_changed_cols += 1
        if rounds == 0:
            stable_cols += 1
        if rounds == 1:
            one_round_cols += 1
        if rounds >= 2:
            chain_cols += 1
        if pair_result:
            nonempty_after_cols += 1
        else:
            empty_after_cols += 1
        rows = np.flatnonzero(in_col != BLACK)
        if rows[-1] < grid.shape[0] - 1 or np.any(np.diff(rows) > 1):
            gapped_cols += 1

    return (
        stable_cols >= 1
        and one_round_cols >= 1
        and chain_cols >= 2
        and empty_after_cols >= 1
        and nonempty_after_cols >= 2
        and visibly_changed_cols >= 3
        and gapped_cols >= len(active_cols)
        and np.count_nonzero(out) < np.count_nonzero(grid)
    )


def _generate_candidate() -> np.ndarray:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    active_max = min(ACTIVE_COLUMN_MAX, (width + 1) // 2)
    active_count = random.randint(ACTIVE_COLUMN_MIN, active_max)
    palette_size = random.randint(*PALETTE_SIZE_RANGE)
    palette = random.sample(COLORS, palette_size)
    max_len = min(MAX_SEQUENCE_LEN, height - 2)

    kinds = ["stable", "one_round", "chain_empty", "chain_residue"]
    extra_pool = ["stable", "one_round", "chain_empty", "chain_residue"]
    while len(kinds) < active_count:
        kinds.append(random.choice(extra_pool))
    random.shuffle(kinds)

    grid = np.zeros((height, width), dtype=int)
    for column, kind in zip(_choose_active_columns(width, active_count), kinds):
        seq = _build_sequence(kind, palette, max_len)
        _paint_sequence(grid, column, seq)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if _validate_example(grid):
            return grid
    raise RuntimeError("failed to generate a valid same-color-annihilation column puzzle")
