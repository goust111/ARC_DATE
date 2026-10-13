from __future__ import annotations

import random

import numpy as np


BG = 0
PLATFORM = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 500

SHAPE_OFFSETS = {
    0: [(0, 0)],
    1: [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)],
    2: [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)],
    3: [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)],
}


def _platform_rows(grid: np.ndarray) -> list[int]:
    return [int(r) for r in np.flatnonzero(np.any(grid == PLATFORM, axis=1))]


def _seed_columns(grid: np.ndarray) -> list[int]:
    return [int(c) for c in np.flatnonzero((grid[0] != BG) & (grid[0] != PLATFORM))]


def _simulate_seed(grid: np.ndarray, col: int, platform_rows: list[int]) -> tuple[int, int]:
    passed = 0
    for row in platform_rows:
        if int(grid[row, col]) == PLATFORM:
            return passed, row - 1
        passed += 1
    return passed, grid.shape[0] - 1


def _paint_shape(
    out: np.ndarray,
    base_platform_mask: np.ndarray,
    center_r: int,
    center_c: int,
    color: int,
    shape_id: int,
) -> None:
    for dr, dc in SHAPE_OFFSETS[shape_id]:
        rr = center_r + dr
        cc = center_c + dc
        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and not base_platform_mask[rr, cc]:
            out[rr, cc] = color


def _drawn_shape_cells(
    shape_id: int,
    center_r: int,
    center_c: int,
    platform_mask: np.ndarray,
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    h, w = platform_mask.shape
    for dr, dc in SHAPE_OFFSETS[shape_id]:
        rr = center_r + dr
        cc = center_c + dc
        if 0 <= rr < h and 0 <= cc < w and not platform_mask[rr, cc]:
            cells.add((rr, cc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop top-row seed pixels through gapped platforms and expand each landing point into the shape selected by its pass count.

    Input structure:
    - The background is black.
    - Row 0 contains 2-4 single-pixel colored seeds in distinct columns.
    - Lower rows contain 1-3 horizontal grey platform lines; each line spans most of the width but has single-column gaps.

    Rule:
    - Each seed falls straight down in its own column.
    - At each platform row, a grey cell blocks the seed and makes it land one row above; a gap lets it pass and increments the pass count.
    - A seed that passes all platforms lands on the bottom row.
    - After landing, erase the original seed and paint the shape chosen by the pass count:
      pass count 0 -> single pixel, 1 -> solid 3x3 block, 2 -> X shape, 3 -> diamond (center + four cardinals).
    - Platform cells remain grey; if expanded shapes overlap, process seeds from left to right and let later seeds overwrite earlier non-platform cells.

    Key insight:
    - The same platform row plays two roles at once: its gaps count transformations for some columns while its grey cells stop other columns, so motion, counting, and final shape selection are chained together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    platform_rows = _platform_rows(g)
    seed_cols = _seed_columns(g)
    assert 1 <= len(platform_rows) <= 3
    assert 2 <= len(seed_cols) <= 4

    out = np.where(g == PLATFORM, PLATFORM, BG).astype(int)
    platform_mask = g == PLATFORM

    for col in sorted(seed_cols):
        color = int(g[0, col])
        passed, landing_row = _simulate_seed(g, col, platform_rows)
        assert 0 <= passed <= 3
        _paint_shape(out, platform_mask, landing_row, col, color, passed)

    return out


def _choose_platform_rows(n_platforms: int) -> list[int] | None:
    first = random.randint(3, 5)
    rows = [first]
    for _ in range(1, n_platforms):
        rows.append(rows[-1] + random.randint(4, 6))
    height = rows[-1] + random.randint(4, 6) + 1
    if height > 30:
        return None
    return rows


def _choose_pass_counts(n_platforms: int, n_seeds: int) -> list[int]:
    patterns = {
        1: [
            [0, 1],
            [0, 1, 1],
            [0, 0, 1],
            [0, 1, 1, 1],
        ],
        2: [
            [0, 1],
            [0, 2],
            [1, 2],
            [0, 1, 2],
            [0, 2, 2],
            [0, 1, 1, 2],
            [0, 1, 2, 2],
        ],
        3: [
            [0, 3],
            [1, 3],
            [2, 3],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
            [0, 1, 2, 3],
            [0, 3, 3],
            [1, 3, 3],
            [0, 2, 3, 3],
        ],
    }
    options = [p for p in patterns[n_platforms] if len(p) == n_seeds]
    values = list(random.choice(options))
    random.shuffle(values)
    return values


def _choose_columns(width: int, n_seeds: int) -> list[int] | None:
    candidates = list(range(1, width - 1))
    random.shuffle(candidates)
    chosen: list[int] = []
    for col in candidates:
        if all(abs(col - prev) >= 3 for prev in chosen):
            chosen.append(col)
            if len(chosen) == n_seeds:
                return sorted(chosen)
    return None


def _make_platform_row(width: int, forced_gaps: set[int], forbidden_gaps: set[int]) -> np.ndarray:
    row = np.zeros(width, dtype=int)
    row[1 : width - 1] = PLATFORM
    gap_pool = [c for c in range(1, width - 1) if c not in forbidden_gaps and c not in forced_gaps]
    extra_gap_count = random.randint(0, min(2, len(gap_pool)))
    extra_gaps = set(random.sample(gap_pool, extra_gap_count)) if extra_gap_count else set()
    all_gaps = forced_gaps | extra_gaps
    if not all_gaps:
        spare = [c for c in range(2, width - 2) if c not in forbidden_gaps]
        all_gaps = {random.choice(spare)}
    for col in all_gaps:
        row[col] = BG
    return row


def _landing_row_for_plan(pass_count: int, platform_rows: list[int], height: int) -> int:
    if pass_count < len(platform_rows):
        return platform_rows[pass_count] - 1
    return height - 1


def _instance_is_valid(grid: np.ndarray, plans: dict[int, int]) -> bool:
    if not (grid.ndim == 2 and max(grid.shape) <= 30):
        return False
    if not np.all(grid[0] != PLATFORM):
        return False

    platform_rows = _platform_rows(grid)
    if platform_rows != sorted(platform_rows):
        return False
    if len(platform_rows) != 0:
        gaps = np.diff(platform_rows)
        if np.any(gaps < 4):
            return False
    for row in platform_rows:
        grey_count = int(np.sum(grid[row] == PLATFORM))
        if grey_count < grid.shape[1] - 5:
            return False
        if not np.any(grid[row] == BG):
            return False

    seen: set[tuple[int, int]] = set()
    platform_mask = grid == PLATFORM
    for col in sorted(plans):
        passed, landing_row = _simulate_seed(grid, col, platform_rows)
        if passed != plans[col]:
            return False
        cells = _drawn_shape_cells(passed, landing_row, col, platform_mask)
        if seen & cells:
            return False
        seen |= cells

    try:
        out = main(grid)
    except Exception:
        return False
    return bool(out.shape == grid.shape)


def _attempt() -> np.ndarray | None:
    n_platforms = random.choices([1, 2, 3], weights=[2, 3, 3], k=1)[0]
    platform_rows = _choose_platform_rows(n_platforms)
    if platform_rows is None:
        return None

    height = platform_rows[-1] + random.randint(4, 6) + 1
    if height > 30:
        return None

    n_seeds = random.randint(2, 4)
    width_lo = max(11, 3 * n_seeds + 3)
    width_hi = min(18, 30)
    if width_lo > width_hi:
        return None
    width = random.randint(width_lo, width_hi)

    cols = _choose_columns(width, n_seeds)
    if cols is None:
        return None
    pass_counts = _choose_pass_counts(n_platforms, n_seeds)
    plans = dict(zip(cols, pass_counts))

    grid = np.zeros((height, width), dtype=int)
    colors = random.sample(SEED_COLORS, n_seeds)
    for col, color in zip(cols, colors):
        grid[0, col] = color

    for i, row_idx in enumerate(platform_rows):
        forced_gaps = {col for col, passes in plans.items() if passes > i}
        forbidden = {col for col, passes in plans.items() if passes <= i}
        grid[row_idx] = _make_platform_row(width, forced_gaps, forbidden)

    if not any(plan > 0 for plan in plans.values()):
        return None
    if not any(
        any(plans[col] > i for col in cols) and any(plans[col] <= i for col in cols)
        for i in range(n_platforms)
    ):
        return None

    if not _instance_is_valid(grid, plans):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid gravity-platform puzzle input")
