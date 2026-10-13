from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_HELPER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import Color


HEIGHT_BOUNDS = (10, 24)
WIDTH_BOUNDS = (12, 24)
SEED_ROW_BOUNDS = (1, 5)
MAX_LEVELS = 7
MAX_GENERATION_ATTEMPTS = 400


def _children_between_gapped_pairs(mask: np.ndarray) -> np.ndarray:
    children = np.zeros(mask.shape, dtype=bool)
    children[1:-1] = mask[:-2] & ~mask[1:-1] & mask[2:]
    return children


def _has_isolated_and_run(mask: np.ndarray) -> bool:
    runs = []
    c = 0
    while c < mask.size:
        if not mask[c]:
            c += 1
            continue
        c0 = c
        while c < mask.size and mask[c]:
            c += 1
        runs.append(c - c0)
    return any(length == 1 for length in runs) and any(length >= 2 for length in runs)


def _generation_masks(seed_mask: np.ndarray) -> list[np.ndarray]:
    generations: list[np.ndarray] = []
    active = seed_mask.copy()
    while True:
        children = _children_between_gapped_pairs(active)
        if not np.any(children):
            break
        generations.append(children)
        active = children
    return generations


def _build_state(shape: tuple[int, int], seed_row: int, seed_mask: np.ndarray, levels: int | None = None) -> np.ndarray:
    out = np.full(shape, int(Color.GREEN), dtype=int)
    out[int(seed_row), seed_mask] = int(Color.ORANGE)

    generations = _generation_masks(seed_mask)
    if levels is None:
        levels = len(generations)

    color = int(Color.PINK)
    for index, mask in enumerate(generations[: int(levels)]):
        row = int(seed_row + 2 * (index + 1))
        if row >= shape[0]:
            break
        out[row, mask] = color
        color = int(Color.ORANGE) if color == int(Color.PINK) else int(Color.PINK)
    return out


def _find_seed_row_and_mask(grid: np.ndarray) -> tuple[int, np.ndarray]:
    non_bg_rows = np.flatnonzero(np.any(grid != int(Color.GREEN), axis=1))
    assert non_bg_rows.size > 0, "expected at least one foreground row"
    seed_row = int(non_bg_rows[0])
    seed_mask = grid[seed_row] == int(Color.ORANGE)
    assert np.any(seed_mask), "expected an orange seed row"
    return seed_row, seed_mask


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Complete the alternating growth below the orange seed row, unless the input is
    already the halted full-growth state, in which case clear back to the seed row.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    seed_row, seed_mask = _find_seed_row_and_mask(grid)
    terminal = _build_state(grid.shape, seed_row, seed_mask)
    if np.array_equal(grid, terminal):
        return _build_state(grid.shape, seed_row, seed_mask, levels=0)
    return terminal


def _sample_seed_mask(width: int) -> np.ndarray | None:
    for _ in range(120):
        run_count = random.randint(4, min(8, max(4, width // 2)))
        run_lengths = [1]
        run_lengths.append(random.randint(2, 4))
        for _idx in range(run_count - 2):
            run_lengths.append(random.randint(1, 4))
        random.shuffle(run_lengths)

        gaps = [random.choice((1, 1, 1, 2)) for _ in range(run_count - 1)]
        span = sum(run_lengths) + sum(gaps)
        if span > width - 2:
            continue

        left_margin = random.randint(0, width - span)
        mask = np.zeros(width, dtype=bool)
        col = left_margin
        for index, length in enumerate(run_lengths):
            mask[col : col + length] = True
            col += length
            if index < len(gaps):
                col += gaps[index]

        if not _has_isolated_and_run(mask):
            continue
        generations = _generation_masks(mask)
        if 1 <= len(generations) <= MAX_LEVELS:
            return mask
    return None


def _sample_problem() -> tuple[np.ndarray, int, np.ndarray, int]:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        width = random.randint(*WIDTH_BOUNDS)
        seed_mask = _sample_seed_mask(width)
        if seed_mask is None:
            continue

        full_levels = len(_generation_masks(seed_mask))
        seed_row = random.randint(*SEED_ROW_BOUNDS)
        min_height = seed_row + 2 * full_levels + 2
        if min_height > HEIGHT_BOUNDS[1]:
            continue
        height = random.randint(max(HEIGHT_BOUNDS[0], min_height), HEIGHT_BOUNDS[1])

        mode = random.choices(
            ("seed_only", "partial", "complete"),
            weights=(2, 4, 2),
            k=1,
        )[0]
        if mode == "seed_only" or full_levels == 0:
            shown_levels = 0
        elif mode == "complete":
            shown_levels = full_levels
        else:
            shown_levels = random.randint(1, full_levels - 1) if full_levels >= 2 else 0

        grid = _build_state((height, width), seed_row, seed_mask, levels=shown_levels)
        return grid, seed_row, seed_mask, full_levels
    raise RuntimeError("failed to generate a valid input")


def generate_input() -> np.ndarray:
    return _sample_problem()[0]
