from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
GREEN = 3
PALETTE = (1, 2, 4, 5, 6, 7, 8, 9)
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))

SIZE_MIN = 8
SIZE_MAX = 12
FOREGROUND_MIN = 5
FOREGROUND_MAX = 12
COUNT_CHOICES = (1, 2, 3)
GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _settle_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    height, width = g.shape
    for col in range(width):
        values = g[:, col][g[:, col] != BLACK]
        if values.size:
            out[height - values.size :, col] = values
    return out


def _draw_spiral(base: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    out = np.asarray(base, dtype=int).copy()
    height, width = out.shape
    stats = {
        "steps": 1,
        "wall_block": 0,
        "settled_block": 0,
        "green_block": 0,
        "lookahead_settled": 0,
        "lookahead_green": 0,
    }

    if out[0, 0] != BLACK:
        return out, stats

    r, c = 0, 0
    dir_idx = 0
    out[r, c] = GREEN

    while True:
        moved = False
        for _ in range(4):
            dr, dc = DIRS[dir_idx]
            nr, nc = r + dr, c + dc
            if not _in_bounds(out.shape, nr, nc):
                stats["wall_block"] += 1
                dir_idx = (dir_idx + 1) % 4
                continue
            if out[nr, nc] != BLACK:
                if out[nr, nc] == GREEN:
                    stats["green_block"] += 1
                else:
                    stats["settled_block"] += 1
                dir_idx = (dir_idx + 1) % 4
                continue

            fr, fc = r + 2 * dr, c + 2 * dc
            if _in_bounds(out.shape, fr, fc) and out[fr, fc] != BLACK:
                if out[fr, fc] == GREEN:
                    stats["lookahead_green"] += 1
                else:
                    stats["lookahead_settled"] += 1
                dir_idx = (dir_idx + 1) % 4
                continue

            r, c = nr, nc
            out[r, c] = GREEN
            stats["steps"] += 1
            moved = True
            break

        if not moved:
            return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Settle each column downward, then draw a clockwise green spiral that avoids the settled obstacles.

    Input structure:
    - The grid is a small square canvas with black background.
    - Sparse non-black single-cell foreground pixels are scattered across multiple columns.
    - The top-left cell is background, and foreground colors are preserved during settling.

    Rule:
    - First compact every non-black column value to the bottom of its own column, preserving top-to-bottom order.
    - Then draw a green spiral from the top-left corner, initially moving right, on top of that settled layout.
    - A move is allowed only if the next cell is in bounds and black; otherwise rotate clockwise and try again.
    - Even when the next cell is black, rotate clockwise instead if the cell two steps ahead exists and is already non-black.
    - Stop when no direction satisfies those conditions; keep both the settled colored pixels and the green spiral.

    Key insight:
    - The colored pixels matter only after gravity compacts them into a bottom obstacle field, and that obstacle field changes the spiral's turns and stopping point.
    """
    settled = _settle_grid(np.asarray(grid, dtype=int))
    out, _stats = _draw_spiral(settled)
    return out


def _random_counts(size: int, total: int) -> list[int] | None:
    valid_options = []
    for column_count in range(3, min(size - 1, 7) + 1):
        if not (column_count <= total <= 3 * column_count):
            continue
        for packed in itertools.product(COUNT_CHOICES, repeat=column_count):
            if sum(packed) == total and any(count >= 2 for count in packed):
                valid_options.append(packed)
    if not valid_options:
        return None

    packed = list(random.choice(valid_options))
    random.shuffle(packed)
    columns = random.sample(range(size), len(packed))
    counts = [0] * size
    for col, count in zip(columns, packed):
        counts[col] = count
    return counts


def _choose_positions(
    size: int,
    count: int,
    *,
    forbid_top_left: bool = False,
    require_unsettled: bool = False,
    require_gap: bool = False,
) -> tuple[int, ...] | None:
    settled = tuple(range(size - count, size))
    candidates = []
    for positions in itertools.combinations(range(size), count):
        if forbid_top_left and 0 in positions:
            continue
        if require_unsettled and positions == settled:
            continue
        if require_gap and not any(positions[i + 1] - positions[i] > 1 for i in range(count - 1)):
            continue
        candidates.append(positions)
    if not candidates:
        return None

    weights = []
    for positions in candidates:
        top_bias = sum(size - r for r in positions)
        weights.append(top_bias + (8 if positions != settled else 1))
    return random.choices(candidates, weights=weights, k=1)[0]


def _scatter_from_settled(settled: np.ndarray, counts: list[int]) -> np.ndarray | None:
    size = settled.shape[0]
    populated = [col for col, count in enumerate(counts) if count]
    multi_cols = [col for col in populated if counts[col] >= 2]
    if len(populated) < 3 or not multi_cols:
        return None

    gap_col = random.choice(multi_cols)
    unsettled_cols = {gap_col}
    other_cols = [col for col in populated if col != gap_col]
    if other_cols:
        unsettled_cols.add(random.choice(other_cols))
    for col in other_cols:
        if random.random() < 0.45:
            unsettled_cols.add(col)

    inp = np.zeros_like(settled)
    for col in populated:
        count = counts[col]
        values = settled[size - count :, col].tolist()
        positions = _choose_positions(
            size,
            count,
            forbid_top_left=(col == 0),
            require_unsettled=(col in unsettled_cols),
            require_gap=(col == gap_col),
        )
        if positions is None:
            return None
        for row, value in zip(positions, values):
            inp[row, col] = value

    if inp[0, 0] != BLACK:
        return None
    return inp


def _feature_mode_ok(stats: dict[str, int], mode: str) -> bool:
    if mode == "lookahead_obstacle":
        return stats["lookahead_settled"] > 0 and stats["settled_block"] > 0
    if mode == "self_spiral":
        return stats["lookahead_green"] > 0 and stats["wall_block"] > 0
    return (
        stats["lookahead_settled"] > 0
        and stats["lookahead_green"] > 0
        and stats["settled_block"] > 0
    )


def _semantic_checks(inp: np.ndarray, mode: str) -> bool:
    settled = _settle_grid(inp)
    out, stats = _draw_spiral(settled)
    original_spiral, _ = _draw_spiral(inp)
    empty_spiral, _ = _draw_spiral(np.zeros_like(inp))

    if not _feature_mode_ok(stats, mode):
        return False
    if np.array_equal(inp, settled):
        return False
    if np.array_equal(out, settled):
        return False
    if np.array_equal(out, original_spiral):
        return False
    if np.array_equal(out, empty_spiral):
        return False
    if np.count_nonzero(inp) < FOREGROUND_MIN or np.count_nonzero(inp) > FOREGROUND_MAX:
        return False
    if np.count_nonzero(out == GREEN) < max(6, inp.shape[0] + 1):
        return False
    if not np.any((out != BLACK) & (out != GREEN)):
        return False

    unsettled_cols = sum(
        not np.array_equal(inp[:, col], settled[:, col])
        for col in range(inp.shape[1])
    )
    if unsettled_cols < 2:
        return False

    gap_cols = 0
    for col in range(inp.shape[1]):
        rows = np.flatnonzero(inp[:, col] != BLACK)
        if rows.size >= 2 and np.any(np.diff(rows) > 1):
            gap_cols += 1
    return gap_cols >= 1


def _gen_attempt(mode: str) -> np.ndarray | None:
    size = random.randint(SIZE_MIN, SIZE_MAX)
    total = random.randint(FOREGROUND_MIN, FOREGROUND_MAX)
    counts = _random_counts(size, total)
    if counts is None:
        return None

    settled = np.zeros((size, size), dtype=int)
    for col, count in enumerate(counts):
        if count == 0:
            continue
        values = random.choices(PALETTE, k=count)
        settled[size - count :, col] = values

    inp = _scatter_from_settled(settled, counts)
    if inp is None:
        return None
    return inp if _semantic_checks(inp, mode) else None


def generate_input() -> np.ndarray:
    modes = ("lookahead_obstacle", "self_spiral", "mixed")
    for _ in range(GEN_ATTEMPTS):
        mode = random.choice(modes)
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
