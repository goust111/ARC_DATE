from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
TEAL = 8

HEIGHT = 6
WIDTH = 24
MIN_SOURCE_COUNT = 3
MAX_SOURCE_COUNT = 5
MIN_RED_COUNT = 4
MAX_RED_COUNT = 8
PERIOD_OPTIONS = (3, 4, 6, 8, 12)
_VALID_CASE_PARAMS: list[
    tuple[int, tuple[int, ...], tuple[int, ...], tuple[bool, ...]]
] | None = None


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _mirror_period(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    _, w = g.shape
    for period in _divisors(w):
        base = g[:, :period]
        ok = True
        for start in range(0, w, period):
            block = g[:, start : start + period]
            if not (
                np.array_equal(block, base) or np.array_equal(block, np.fliplr(base))
            ):
                ok = False
                break
        if ok:
            return period
    raise ValueError("no reflected horizontal period found")


def _paint_teal(block: np.ndarray, r: int, c: int) -> None:
    if block[r, c] == BLACK:
        block[r, c] = TEAL


def _simulate_block(red_mask: np.ndarray, source_rows: tuple[int, ...], mirrored: bool) -> np.ndarray:
    period = int(red_mask.shape[1])
    block = np.zeros((HEIGHT, period), dtype=int)
    block[0, red_mask[0]] = RED

    direction = -1 if mirrored else 1
    start_col = period - 1 if mirrored else 0

    for start_row in source_rows:
        row = int(start_row)
        col = int(start_col)
        _paint_teal(block, row, col)
        if red_mask[0, col] and row + 1 < HEIGHT:
            row += 1
            _paint_teal(block, row, col)

        while True:
            nxt = col + direction
            if not (0 <= nxt < period):
                break
            col = nxt
            _paint_teal(block, row, col)
            if red_mask[0, col] and row + 1 < HEIGHT:
                row += 1
                _paint_teal(block, row, col)

    return block


def _top_red_mask(grid: np.ndarray) -> np.ndarray:
    mask = np.zeros((1, grid.shape[1]), dtype=bool)
    mask[0] = np.asarray(grid[0] == RED, dtype=bool)
    return mask


def _source_rows_from_grid(grid: np.ndarray) -> tuple[int, ...]:
    return tuple(int(r) for r in np.flatnonzero(np.asarray(grid[:, 0] == TEAL)))


def _seed_period_and_orientations(grid: np.ndarray) -> tuple[int, list[bool]]:
    red_mask = _top_red_mask(grid)
    period = _mirror_period(red_mask.astype(int))
    base = red_mask[:, :period]
    orientations: list[bool] = []

    for block_idx, start in enumerate(range(0, red_mask.shape[1], period)):
        block = red_mask[:, start : start + period]
        same = np.array_equal(block, base)
        flip = np.array_equal(block, np.fliplr(base))
        if not same and not flip:
            raise ValueError("top-edge pattern is not period-consistent")
        if block_idx == 0 and not same:
            raise ValueError("the visible first block must be the base orientation")
        orientations.append(False if same else True)

    return period, orientations


def _reconstruct_completed_pattern(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    period, orientations = _seed_period_and_orientations(g)
    red_mask = _top_red_mask(g)
    source_rows = _source_rows_from_grid(g)

    completed = np.zeros_like(g)
    for block_idx, mirrored in enumerate(orientations):
        start = block_idx * period
        stop = start + period
        block_red = red_mask[:, start:stop]
        completed[:, start:stop] = _simulate_block(block_red, source_rows, mirrored)

    return completed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replay the edge-sourced flow inside the smallest reflected horizontal repeat and return its leftmost unit.

    Input structure:
    - The grid is a 6x24 black canvas with teal source pixels on the left edge.
    - Red pixels lie on the top edge and are arranged as repeated horizontal blocks,
      where later blocks may be horizontal reflections of the first block.
    - The visible input only shows the source pixels at the global left edge and the
      red top-edge deflectors.

    Rule:
    - Infer the smallest horizontal block width whose top-edge red pattern repeats up
      to left-right reflection.
    - Reconstruct the completed trajectory pattern block by block: base blocks replay
      the teal flow from left to right, while reflected blocks replay the same local
      flow in horizontal mirror from right to left; every red column pushes the path
      one cell downward within that block while red pixels remain red.
    - After the full repeated trajectory pattern is reconstructed, detect its smallest
      reflected horizontal period and return the leftmost block of that width.

    Key insight:
    - The top edge only specifies the repeating deflector schedule; the crop target is
      the smallest completed flow tile once mirrored repeats are expanded into their
      full trajectory blocks.
    """
    g = np.asarray(grid, dtype=int)
    if g.shape != (HEIGHT, WIDTH):
        raise ValueError(f"expected a {HEIGHT}x{WIDTH} grid")

    source_rows = _source_rows_from_grid(g)
    if len(source_rows) < 3:
        raise ValueError("expected at least three teal edge sources")
    if any(b - a <= 1 for a, b in zip(source_rows, source_rows[1:])):
        raise ValueError("teal edge sources must be pairwise non-adjacent")

    completed = _reconstruct_completed_pattern(g)
    period = _mirror_period(completed)
    return completed[:, :period].copy()

def _build_input(
    period: int,
    source_rows: tuple[int, ...],
    base_positions: tuple[int, ...],
    flags: tuple[bool, ...] | list[bool],
) -> np.ndarray:
    grid = np.zeros((HEIGHT, WIDTH), dtype=int)
    grid[list(source_rows), 0] = TEAL

    repeats = WIDTH // period
    for block_idx in range(repeats):
        positions = tuple(int(pos) for pos in base_positions)
        if flags[block_idx]:
            positions = tuple(period - 1 - pos for pos in positions)
        if block_idx == 0 and 0 in positions and 0 in source_rows:
            raise ValueError("the global top-left corner cannot host both source and red")
        for pos in positions:
            grid[0, block_idx * period + pos] = RED

    return grid


def _case_is_valid(grid: np.ndarray, target_period: int) -> bool:
    try:
        red_count = int(np.count_nonzero(grid == RED))
        if red_count < MIN_RED_COUNT or red_count > MAX_RED_COUNT:
            return False

        if _mirror_period(_top_red_mask(grid).astype(int)) != target_period:
            return False

        source_rows = _source_rows_from_grid(grid)
        if not (MIN_SOURCE_COUNT <= len(source_rows) <= MAX_SOURCE_COUNT):
            return False
        if any(b - a <= 1 for a, b in zip(source_rows, source_rows[1:])):
            return False

        completed = _reconstruct_completed_pattern(grid)
        if _mirror_period(completed) != target_period:
            return False

        output = main(grid)
        if output.shape != (HEIGHT, target_period):
            return False

        for smaller in _divisors(target_period):
            if smaller == target_period:
                continue
            base = output[:, :smaller]
            ok = True
            for start in range(0, target_period, smaller):
                block = output[:, start : start + smaller]
                if not (
                    np.array_equal(block, base) or np.array_equal(block, np.fliplr(base))
                ):
                    ok = False
                    break
            if ok:
                return False
    except Exception:
        return False

    return True


def _source_row_options() -> list[tuple[int, ...]]:
    options: list[tuple[int, ...]] = []

    def backtrack(start: int, chosen: list[int]) -> None:
        if MIN_SOURCE_COUNT <= len(chosen) <= MAX_SOURCE_COUNT:
            options.append(tuple(chosen))
        if len(chosen) == MAX_SOURCE_COUNT:
            return
        for row in range(start, HEIGHT):
            if chosen and row - chosen[-1] <= 1:
                continue
            chosen.append(row)
            backtrack(row + 2, chosen)
            chosen.pop()

    backtrack(0, [])
    return options


def _base_red_position_options(period: int) -> list[tuple[int, ...]]:
    repeats = WIDTH // period
    per_block = [
        count
        for count in range(1, period + 1)
        if MIN_RED_COUNT <= count * repeats <= MAX_RED_COUNT
    ]
    options: list[tuple[int, ...]] = []

    def choose(start: int, chosen: list[int], remaining: int) -> None:
        if remaining == 0:
            options.append(tuple(chosen))
            return
        last = period - remaining
        for pos in range(start, last + 1):
            chosen.append(pos)
            choose(pos + 1, chosen, remaining - 1)
            chosen.pop()

    for count in per_block:
        choose(0, [], count)

    return options


def _orientation_sequences(repeats: int) -> list[tuple[bool, ...]]:
    if repeats <= 1:
        return [(False,)]
    return [
        tuple([False, *tail])
        for tail_bits in range(1, 1 << (repeats - 1))
        for tail in [[bool((tail_bits >> bit) & 1) for bit in range(repeats - 1)]]
    ]


def _valid_case_params() -> list[tuple[int, tuple[int, ...], tuple[int, ...], tuple[bool, ...]]]:
    global _VALID_CASE_PARAMS
    if _VALID_CASE_PARAMS is not None:
        return _VALID_CASE_PARAMS

    cases: list[tuple[int, tuple[int, ...], tuple[int, ...], tuple[bool, ...]]] = []
    source_options = _source_row_options()

    for period in PERIOD_OPTIONS:
        for source_rows in source_options:
            for base_positions in _base_red_position_options(period):
                for flags in _orientation_sequences(WIDTH // period):
                    try:
                        grid = _build_input(period, source_rows, base_positions, flags)
                    except ValueError:
                        continue
                    if _case_is_valid(grid, period):
                        cases.append((period, source_rows, base_positions, flags))

    if not cases:
        raise RuntimeError("failed to precompute any valid reflected flow-period samples")

    _VALID_CASE_PARAMS = cases
    return cases


def generate_input() -> np.ndarray:
    period, source_rows, base_positions, flags = random.choice(_valid_case_params())
    return _build_input(period, source_rows, base_positions, flags)
