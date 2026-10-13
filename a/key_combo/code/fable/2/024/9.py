from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg


GRID_HEIGHT_BOUNDS = (12, 20)
GRID_WIDTH_MIN = 14
GRID_WIDTH_MAX = 26
SEED_GAP_MIN = 5
SEED_GAP_MAX = 11
MANDATORY_RUN_LEN = 3
OPTIONAL_OVERLAP_RUN_PROB = 0.75
OPTIONAL_OVERLAP_RUN_LENGTHS = (2, 3)
EXTRA_RUN_COUNT_BOUNDS = (1, 4)
EXTRA_RUN_LENGTH_BOUNDS = (1, 4)
GEN_RETRY_LIMIT = 500
SEED_COLORS = (1, 2, 4, 6, 7, 8, 9)


def _extract_seeds(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    seed_cols = [
        int(c)
        for c, value in enumerate(g[0])
        if int(value) not in (Color.BLACK, Color.GREEN, Color.GREY)
    ]
    assert len(seed_cols) == 2, "expected exactly two top-row seeds"

    seeds = [(int(c), int(g[0, c])) for c in seed_cols]
    seed_colors = [color for _c, color in seeds]
    assert len(set(seed_colors)) == 2, "seed colors must be distinct"
    assert not np.any(g[1:] == Color.GREEN), "green is reserved for overlap output"
    assert np.all((g[1:] == Color.BLACK) | (g[1:] == Color.GREY)), "only grey blockers may appear below the top row"
    return seeds


def _full_cone_mask(shape: tuple[int, int], seed_col: int) -> np.ndarray:
    height, width = map(int, shape)
    rr, cc = np.indices((height, width))
    return np.abs(cc - int(seed_col)) <= rr


def _propagate_one(blockers: np.ndarray, seed_col: int) -> np.ndarray:
    height, width = map(int, blockers.shape)
    lit = np.zeros((height, width), dtype=bool)
    lit[0, int(seed_col)] = True

    for r in range(1, height):
        c0 = max(0, int(seed_col) - r)
        c1 = min(width - 1, int(seed_col) + r)
        for c in range(c0, c1 + 1):
            if blockers[r, c]:
                continue
            if lit[r - 1, max(0, c - 1) : min(width, c + 2)].any():
                lit[r, c] = True
    return lit


def _merge_masks(
    blockers: np.ndarray,
    seed_colors: tuple[int, int],
    masks: tuple[np.ndarray, np.ndarray],
    *,
    overlap_green: bool = True,
) -> np.ndarray:
    out = np.full(blockers.shape, Color.BLACK, dtype=int)
    out[blockers] = Color.GREY

    left_mask, right_mask = masks
    left_color, right_color = map(int, seed_colors)

    if overlap_green:
        out[left_mask & ~right_mask & ~blockers] = left_color
        out[right_mask & ~left_mask & ~blockers] = right_color
        out[left_mask & right_mask & ~blockers] = Color.GREEN
    else:
        out[left_mask & ~blockers] = left_color
        out[right_mask & ~blockers] = right_color
    return out


def _triangle_shadow_mask(shape: tuple[int, int], seed_col: int, blockers: np.ndarray) -> np.ndarray:
    full = _full_cone_mask(shape, seed_col)
    height, width = map(int, shape)
    shadow = np.zeros((height, width), dtype=bool)

    for r_raw, c_raw in np.argwhere(blockers & full):
        r = int(r_raw)
        c = int(c_raw)
        for rr in range(r + 1, height):
            span = rr - r
            c0 = max(0, c - span)
            c1 = min(width - 1, c + span)
            shadow[rr, c0 : c1 + 1] = True

    return full & ~blockers & ~shadow


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow two top-edge color cones downward around grey blockers, turning overlap cells green.

    Input structure:
    - The background is black.
    - The top row contains exactly two one-cell seeds of distinct colors; neither seed color is green or grey.
    - Any non-black cell below the top row is a grey blocker cell.

    Rule:
    - For each seed independently, propagate downward row by row: a non-blocker cell is lit by that seed iff it lies within the seed's cone range (column distance at most row distance) and at least one of the three cells directly above or diagonally above was already lit by the same seed.
    - Grey blocker cells stay grey and stop that seed on the blocked cell itself, but light may continue later through diagonal parents that remain lit.
    - Paint cells lit by only one cone with that seed's color, and paint cells lit by both cones green.

    Key insight:
    - The illuminated region must be computed by recursive per-row reachability, not by subtracting static geometric shadows from a triangle: blockers create dark pockets, yet diagonal paths can reconnect below them, and green comes from intersecting the two independently propagated masks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    assert bg == Color.BLACK, "expected black background"

    seeds = _extract_seeds(g)
    blockers = g == Color.GREY
    seed_colors = (int(seeds[0][1]), int(seeds[1][1]))
    masks = tuple(_propagate_one(blockers, seed_col) for seed_col, _color in seeds)
    return _merge_masks(blockers, seed_colors, masks)


def _build_input(shape: tuple[int, int], seed_cols: tuple[int, int], seed_colors: tuple[int, int], blockers: np.ndarray) -> np.ndarray:
    grid = np.full(shape, Color.BLACK, dtype=int)
    for seed_col, seed_color in zip(seed_cols, seed_colors):
        grid[0, int(seed_col)] = int(seed_color)
    grid[blockers] = Color.GREY
    return grid


def _prune_redundant_blockers(grid: np.ndarray) -> np.ndarray:
    seeds = _extract_seeds(grid)
    seed_cols = (int(seeds[0][0]), int(seeds[1][0]))
    seed_colors = (int(seeds[0][1]), int(seeds[1][1]))
    blockers = (np.asarray(grid, dtype=int) == Color.GREY)
    baseline = main(grid)

    blocker_cells = list(map(tuple, np.argwhere(blockers)))
    random.shuffle(blocker_cells)
    for r_raw, c_raw in blocker_cells:
        r = int(r_raw)
        c = int(c_raw)
        blockers[r, c] = False
        candidate = _build_input(grid.shape, seed_cols, seed_colors, blockers)
        if not np.array_equal(main(candidate), baseline):
            blockers[r, c] = True

    return _build_input(grid.shape, seed_cols, seed_colors, blockers)


def _semantic_checks(grid: np.ndarray) -> bool:
    seeds = _extract_seeds(grid)
    seed_cols = (int(seeds[0][0]), int(seeds[1][0]))
    seed_colors = (int(seeds[0][1]), int(seeds[1][1]))
    blockers = (np.asarray(grid, dtype=int) == Color.GREY)
    height, width = map(int, grid.shape)
    area = int(height * width)

    masks = tuple(_propagate_one(blockers, seed_col) for seed_col in seed_cols)
    correct = _merge_masks(blockers, seed_colors, masks)

    full_masks = tuple(_full_cone_mask(grid.shape, seed_col) & ~blockers for seed_col in seed_cols)
    shadow_masks = tuple(_triangle_shadow_mask(grid.shape, seed_col, blockers) for seed_col in seed_cols)
    wrong_no_shadow = _merge_masks(blockers, seed_colors, full_masks)
    wrong_triangle = _merge_masks(blockers, seed_colors, shadow_masks)
    wrong_no_green = _merge_masks(blockers, seed_colors, masks, overlap_green=False)

    left_mask, right_mask = masks
    lit = int(np.count_nonzero(left_mask | right_mask))
    green = int(np.count_nonzero(left_mask & right_mask))
    exclusive_left = int(np.count_nonzero(left_mask & ~right_mask))
    exclusive_right = int(np.count_nonzero(right_mask & ~left_mask))
    shadow_cells = int(np.count_nonzero((full_masks[0] | full_masks[1]) & ~(left_mask | right_mask) & ~blockers))
    bypass_left = int(np.count_nonzero(left_mask & ~shadow_masks[0]))
    bypass_right = int(np.count_nonzero(right_mask & ~shadow_masks[1]))
    blocker_count = int(np.count_nonzero(blockers))
    lit_ratio = lit / area
    green_ratio = green / max(1, lit)

    return all(
        [
            np.array_equal(main(grid), correct),
            green >= 3,
            exclusive_left >= 6,
            exclusive_right >= 6,
            shadow_cells >= 2,
            bypass_left >= 1,
            bypass_right >= 1,
            4 <= blocker_count <= max(12, area // 12),
            0.18 <= lit_ratio <= 0.72,
            0.08 <= green_ratio <= 0.42,
            np.any((left_mask | right_mask)[-1]),
            not np.array_equal(correct, grid),
            not np.array_equal(correct, wrong_no_shadow),
            not np.array_equal(correct, wrong_triangle),
            not np.array_equal(correct, wrong_no_green),
        ]
    )


def _sample_shape() -> tuple[int, int]:
    height = random.randint(*GRID_HEIGHT_BOUNDS)
    width_lo = max(GRID_WIDTH_MIN, height + 1)
    width_hi = min(GRID_WIDTH_MAX, height + 8)
    return int(height), int(random.randint(width_lo, width_hi))


def _sample_seed_cols(height: int, width: int) -> tuple[int, int] | None:
    max_gap = min(SEED_GAP_MAX, width - 4, height - 2)
    if max_gap < SEED_GAP_MIN:
        return None
    min_gap = min(max(SEED_GAP_MIN, height // 3), max_gap)
    gap = random.randint(min_gap, max_gap)
    left = random.randint(1, width - gap - 2)
    return int(left), int(left + gap)


def _place_mandatory_run(blockers: np.ndarray, seed_col: int, forbidden_rows: set[int]) -> bool:
    height, _width = map(int, blockers.shape)
    row_choices = [r for r in range(2, height - 3) if all(abs(r - taken) >= 2 for taken in forbidden_rows)]
    if not row_choices:
        return False

    row = int(random.choice(row_choices))
    start_lo = int(seed_col - row + 1)
    start_hi = int(seed_col + row - MANDATORY_RUN_LEN)
    if start_lo > start_hi:
        return False

    start = random.randint(start_lo, start_hi)
    blockers[row, start : start + MANDATORY_RUN_LEN] = True
    forbidden_rows.add(row)
    return True


def _place_overlap_run(blockers: np.ndarray, seed_cols: tuple[int, int], forbidden_rows: set[int]) -> None:
    if random.random() >= OPTIONAL_OVERLAP_RUN_PROB:
        return

    height, width = map(int, blockers.shape)
    left_seed, right_seed = seed_cols
    row_choices = [
        r
        for r in range(max(3, (right_seed - left_seed) // 2 + 1), height - 2)
        if all(abs(r - taken) >= 1 for taken in forbidden_rows)
    ]
    if not row_choices:
        return

    row = int(random.choice(row_choices))
    length = int(random.choice(OPTIONAL_OVERLAP_RUN_LENGTHS))
    start_lo = max(1, max(left_seed - row, right_seed - row))
    start_hi = min(width - length, min(left_seed + row, right_seed + row) - (length - 1))
    if start_lo > start_hi:
        return

    start = random.randint(start_lo, start_hi)
    blockers[row, start : start + length] = True
    forbidden_rows.add(row)


def _place_extra_runs(blockers: np.ndarray, seed_cols: tuple[int, int]) -> None:
    union_mask = _full_cone_mask(blockers.shape, seed_cols[0]) | _full_cone_mask(blockers.shape, seed_cols[1])
    height, width = map(int, blockers.shape)

    for _ in range(random.randint(*EXTRA_RUN_COUNT_BOUNDS)):
        row = random.randint(1, height - 2)
        length = random.randint(*EXTRA_RUN_LENGTH_BOUNDS)
        starts = [c0 for c0 in range(0, width - length + 1) if np.any(union_mask[row, c0 : c0 + length])]
        if not starts:
            continue
        start = int(random.choice(starts))
        blockers[row, start : start + length] = True


def _generate_candidate() -> np.ndarray | None:
    height, width = _sample_shape()
    seed_cols = _sample_seed_cols(height, width)
    if seed_cols is None:
        return None

    seed_colors = tuple(random.sample(SEED_COLORS, 2))
    blockers = np.zeros((height, width), dtype=bool)
    blocked_rows: set[int] = set()

    if not _place_mandatory_run(blockers, seed_cols[0], blocked_rows):
        return None
    if not _place_mandatory_run(blockers, seed_cols[1], blocked_rows):
        return None

    _place_overlap_run(blockers, seed_cols, blocked_rows)
    _place_extra_runs(blockers, seed_cols)

    grid = _build_input((height, width), seed_cols, seed_colors, blockers)
    grid = _prune_redundant_blockers(grid)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid two-cone shadow instance")
