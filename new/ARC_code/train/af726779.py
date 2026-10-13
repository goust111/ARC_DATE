from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg
import random
from functools import lru_cache


def _children_between_gapped_pairs(mask: np.ndarray) -> np.ndarray:
    children = np.zeros(mask.shape, dtype=bool)
    children[1:-1] = mask[:-2] & ~mask[1:-1] & mask[2:]
    return children


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow an alternating pink/orange triangular trace downward from the horizontal orange seed row.

    Input structure:
    - The grid has a green background and one horizontal row containing all orange foreground pixels.
    - The seed row may contain isolated orange pixels and adjacent orange runs.
    - Pink is absent from the input and is used for the first generated row.

    Rule:
    - Copy the input grid.
    - Two rows below the current active row, place a child pixel between each active pair
      whose middle cell in the current row is background.
    - Alternate generated row colors between pink and orange, stopping when no children
      can be generated or the next row would leave the grid.

    Key insight:
    - Only an active-background-active triplet in the current row reproduces; contiguous
      active runs do not produce children through occupied middle cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg == Color.GREEN, "expected green background"
    assert Color.PINK not in colors_present(g), "pink must be absent from the input"

    foreground_colors = colors_present(g, exclude={bg})
    assert foreground_colors == [Color.ORANGE], "expected orange as the only foreground color"

    seed_coords = np.argwhere(g == Color.ORANGE)
    assert seed_coords.size > 0, "expected at least one orange seed pixel"
    seed_rows = set(map(int, seed_coords[:, 0]))
    assert len(seed_rows) == 1, "expected all orange seed pixels on one row"
    seed_row = seed_rows.pop()

    out = g.copy()
    active = g[seed_row, :] == Color.ORANGE
    paint_color = Color.PINK
    row = seed_row + 2
    while row < g.shape[0]:
        children = _children_between_gapped_pairs(active)
        if not np.any(children):
            break
        assert np.all(out[row, children] == bg), "generated children must land on background"
        out[row, children] = paint_color
        active = children
        paint_color = Color.ORANGE if paint_color == Color.PINK else Color.PINK
        row += 2

    return out

# --------------- generator (att2) ---------------

ROW_BOUNDS = (9, 21)
COL_BOUNDS = (15, 24)
SEED_ROW_BOUNDS = (1, 4)
EXTRA_ROWS_BELOW_LAST_GEN_BOUNDS = (1, 8)
LEFT_MARGIN_MAX = 3
RIGHT_MARGIN_MAX = 3
RUN_COUNT_BOUNDS = (3, 9)
ORANGE_TOTAL_BOUNDS = (9, 12)
RUN_LENGTH_BOUNDS = (1, 7)
GAP_WIDTH = 1
LEVEL_CHOICES = tuple(range(1, 9))
LEVEL_WEIGHTS = (5, 4, 3, 2, 1, 0.5, 0.3, 0.2)
LONG_RUN_WEIGHT_BASE = 1
LONG_RUN_WEIGHT_EXP = 8
LENGTH_ENERGY_EXP = 8
RUN_COUNT_WEIGHT_EXP = 10
MAX_GENERATION_ATTEMPTS = 300


def _gen_children_between_gapped_pairs(mask: np.ndarray) -> np.ndarray:
    children = np.zeros(mask.shape, dtype=bool)
    children[1:-1] = mask[:-2] & ~mask[1:-1] & mask[2:]
    return children


@lru_cache(maxsize=None)
def _gen_run_compositions(parts: int, total: int) -> tuple[tuple[int, ...], ...]:
    if parts == 0:
        return ((),) if total == 0 else ()

    choices = []
    min_run, max_run = RUN_LENGTH_BOUNDS
    min_value = max(min_run, total - max_run * (parts - 1))
    max_value = min(max_run, total - min_run * (parts - 1))
    for value in range(min_value, max_value + 1):
        for rest in _gen_run_compositions(parts - 1, total - value):
            choices.append((value,) + rest)
    return tuple(choices)


@lru_cache(maxsize=None)
def _gen_level_count_from_runs(run_lengths: tuple[int, ...]) -> int:
    span = sum(run_lengths) + GAP_WIDTH * (len(run_lengths) - 1)
    active = np.zeros(span, dtype=bool)
    col = 0
    for length in run_lengths:
        active[col : col + length] = True
        col += length + GAP_WIDTH

    levels = 0
    while True:
        children = _gen_children_between_gapped_pairs(active)
        if not np.any(children):
            return levels
        levels += 1
        active = children


@lru_cache(maxsize=None)
def _gen_seed_options(cols: int, target_levels: int) -> tuple[tuple[tuple[int, ...], int], ...]:
    options = []
    for run_count in range(RUN_COUNT_BOUNDS[0], RUN_COUNT_BOUNDS[1] + 1):
        for orange_total in range(ORANGE_TOTAL_BOUNDS[0], ORANGE_TOTAL_BOUNDS[1] + 1):
            if orange_total < run_count:
                continue
            span = orange_total + GAP_WIDTH * (run_count - 1)
            margin_total = cols - span
            if margin_total < 0 or margin_total > LEFT_MARGIN_MAX + RIGHT_MARGIN_MAX:
                continue
            left_min = max(0, margin_total - RIGHT_MARGIN_MAX)
            left_max = min(LEFT_MARGIN_MAX, margin_total)
            left_margins = tuple(range(left_min, left_max + 1))
            if not left_margins:
                continue

            for run_lengths in _gen_run_compositions(run_count, orange_total):
                if _gen_level_count_from_runs(run_lengths) == target_levels:
                    for left_margin in left_margins:
                        options.append((run_lengths, left_margin))
    return tuple(options)


def _gen_row_seed_pairs(target_levels: int) -> list[tuple[int, int]]:
    pairs = []
    for rows in range(ROW_BOUNDS[0], ROW_BOUNDS[1] + 1):
        for seed_row in range(SEED_ROW_BOUNDS[0], SEED_ROW_BOUNDS[1] + 1):
            last_gen_row = seed_row + 2 * target_levels
            extra_rows = rows - 1 - last_gen_row
            if EXTRA_ROWS_BELOW_LAST_GEN_BOUNDS[0] <= extra_rows <= EXTRA_ROWS_BELOW_LAST_GEN_BOUNDS[1]:
                pairs.append((rows, seed_row))
    return pairs


def _gen_place_seed(rows: int, cols: int, seed_row: int, run_lengths: tuple[int, ...], left_margin: int) -> np.ndarray:
    grid = np.full((rows, cols), Color.GREEN, dtype=int)
    col = left_margin
    for length in run_lengths:
        grid[seed_row, col : col + length] = Color.ORANGE
        col += length + GAP_WIDTH
    return grid


def _gen_seed_option_weight(run_lengths: tuple[int, ...]) -> float:
    long_runs = sum(length > 1 for length in run_lengths)
    length_energy = sum(length * length for length in run_lengths)
    return (
        (LONG_RUN_WEIGHT_BASE + long_runs) ** LONG_RUN_WEIGHT_EXP
        * length_energy**LENGTH_ENERGY_EXP
        / len(run_lengths) ** RUN_COUNT_WEIGHT_EXP
    )


def _gen_attempt() -> np.ndarray | None:
    target_levels = random.choices(LEVEL_CHOICES, weights=LEVEL_WEIGHTS, k=1)[0]

    row_seed_pairs = _gen_row_seed_pairs(target_levels)
    if not row_seed_pairs:
        return None
    rows, seed_row = random.choice(row_seed_pairs)

    col_choices = [cols for cols in range(COL_BOUNDS[0], COL_BOUNDS[1] + 1) if _gen_seed_options(cols, target_levels)]
    if not col_choices:
        return None
    cols = random.choice(col_choices)

    grouped_options = {}
    for run_lengths, left_margin in _gen_seed_options(cols, target_levels):
        grouped_options.setdefault(sum(run_lengths), []).append((run_lengths, left_margin))
    run_group = random.choice(list(grouped_options.values()))
    run_weights = [_gen_seed_option_weight(run_lengths) for run_lengths, _ in run_group]
    run_lengths, left_margin = random.choices(run_group, weights=run_weights, k=1)[0]

    grid = _gen_place_seed(rows, cols, seed_row, run_lengths, left_margin)
    out = main(grid)
    generated_rows = np.any(out != grid, axis=1)
    if int(np.count_nonzero(generated_rows)) != target_levels:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate af726779 input")
