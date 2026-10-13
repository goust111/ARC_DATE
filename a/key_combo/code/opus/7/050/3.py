import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
BAR = 5
SEED_COLORS = (BLUE, RED)
MAX_GEN_ATTEMPTS = 400


def _find_bar_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == BAR)]
    assert rows, "expected one full-span grey bar"
    return int(rows[0])


def _settle_above_bar(grid: np.ndarray, bar_row: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full((h, w), BG, dtype=int)
    out[bar_row, :] = BAR

    for c in range(w):
        stack = [int(g[r, c]) for r in range(bar_row) if int(g[r, c]) not in (BG, BAR)]
        n = len(stack)
        if n == 0:
            continue
        out[bar_row - n : bar_row, c] = np.array(stack, dtype=int)

    return out


def _seed_columns(settled: np.ndarray, bar_row: int) -> list[tuple[int, int]]:
    seeds: list[tuple[int, int]] = []
    for c in range(settled.shape[1]):
        color = int(settled[bar_row - 1, c]) if bar_row > 0 else BG
        if color != BG:
            seeds.append((c, color))
    return seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let falling colored pixels settle on a full-width grey bar, then use only the bottom contact pixels as periodic stripe seeds.

    Input structure:
    - The grid has black background and one grey horizontal bar spanning the full width.
    - Above the bar, red and blue pixels are scattered, with some columns containing vertical stacks.
    - Below the bar, the grid is empty background.

    Rule:
    - Drop every red/blue pixel straight downward within its own column until it lands on the bar or on another fallen pixel.
    - In each non-empty column, only the pixel directly touching the bar is the seed; higher stacked pixels stay visible but do not affect the stripe rule.
    - Compute the stripe period as the minimum gap between adjacent seed columns.
    - Copy each seed color to every column below the bar whose column index matches that seed modulo the period; unmatched columns remain black.

    Key insight:
    - The apparent stack contents are a distraction: only the bar-touching bottom layer matters, and those seed columns define a repeating stripe pattern below.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bar_row = _find_bar_row(g)
    settled = _settle_above_bar(g, bar_row)
    seeds = _seed_columns(settled, bar_row)
    assert len(seeds) >= 2, "expected at least two seed columns"

    seed_cols = [c for c, _ in seeds]
    period = min(b - a for a, b in zip(seed_cols, seed_cols[1:]))
    assert period > 0

    residue_to_color: dict[int, int] = {}
    for c, color in seeds:
        residue = c % period
        residue_to_color.setdefault(residue, color)

    out = settled.copy()
    if bar_row + 1 < g.shape[0]:
        for c in range(g.shape[1]):
            color = residue_to_color.get(c % period, BG)
            if color != BG:
                out[bar_row + 1 :, c] = color

    return out


def _choose_period_and_columns(width: int) -> tuple[int, list[int]]:
    options: list[tuple[int, int, list[int], int]] = []
    for period in range(2, min(7, width - 1) + 1):
        for residue in range(period):
            columns = list(range(residue, width, period))
            if len(columns) < 4:
                continue
            remainder = width % period
            score = 0 if remainder == 0 else 1 if remainder in (1, period - 1) else 2
            options.append((score, period, columns, residue))

    if not options:
        raise ValueError("no valid period choices")

    best_score = min(score for score, *_ in options)
    filtered = [item for item in options if item[0] <= best_score + 1]
    _score, period, columns, _residue = random.choice(filtered)

    max_k = min(6, len(columns) - 1)
    k = random.randint(3, max_k)
    start = random.randint(0, len(columns) - k)
    chosen = columns[start : start + k]
    assert len(chosen) >= 3
    assert len(chosen) < len(columns)
    return period, chosen


def _place_column_stack(grid: np.ndarray, bar_row: int, col: int, height: int, bottom_color: int) -> None:
    rows = sorted(random.sample(range(bar_row), height))
    colors = [random.choice(SEED_COLORS) for _ in range(height)]
    colors[-1] = bottom_color
    for r, color in zip(rows, colors):
        grid[r, col] = color


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        height = random.randint(12, 18)
        width = random.randint(15, 24)
        bar_row = random.randint(4, height - 5)

        try:
            period, seed_cols = _choose_period_and_columns(width)
        except ValueError:
            continue

        grid = np.zeros((height, width), dtype=int)
        grid[bar_row, :] = BAR

        bottom_color = random.choice(SEED_COLORS)
        distractor_color = RED if bottom_color == BLUE else BLUE

        heights = [random.choices([1, 2, 3], weights=[4, 4, 2])[0] for _ in seed_cols]
        if max(heights) == 1:
            heights[random.randrange(len(heights))] = random.choice([2, 3])
        stacked_idx = random.choice([i for i, h in enumerate(heights) if h > 1])

        for i, (col, stack_h) in enumerate(zip(seed_cols, heights)):
            _place_column_stack(grid, bar_row, col, stack_h, bottom_color)
            if i == stacked_idx:
                rows = [r for r in range(bar_row) if grid[r, col] != BG]
                rows.sort()
                if len(rows) > 1:
                    grid[rows[0], col] = distractor_color
                    grid[rows[-1], col] = bottom_color

        settled = _settle_above_bar(grid, bar_row)
        seeds = _seed_columns(settled, bar_row)
        if [c for c, _ in seeds] != seed_cols:
            continue
        if any(color != bottom_color for _, color in seeds):
            continue

        computed_period = min(b - a for a, b in zip(seed_cols, seed_cols[1:]))
        if computed_period != period:
            continue

        output = main(grid)
        stripe_cols = [c for c in range(width) if int(output[bar_row + 1, c]) == bottom_color] if bar_row + 1 < height else []
        if not stripe_cols:
            continue
        if not any(c not in seed_cols for c in stripe_cols):
            continue
        if distractor_color not in grid:
            continue

        return grid

    raise RuntimeError("generate_input failed to produce a valid grid")
