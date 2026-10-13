from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8

SEED_PALETTE = (BLUE, YELLOW, ORANGE)
FORBIDDEN_SEED_COLORS = {BLACK, GREEN, GREY, RED, TEAL}
POST_OFFSETS = ((-1, -2), (-1, 2), (0, -2), (0, 2), (1, -2), (1, 2))

HEIGHT_RANGE = (20, 24)
WIDTH_RANGE = (18, 24)
SEED_COUNT_RANGE = (2, 3)
CONTROL_COUNT_RANGE = (2, 3)
CONTROL_MARGIN = 3
CONTROL_ROW_GAP = 5
TIE_PROBABILITY = 0.65
MAX_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _seed_rows(grid: np.ndarray) -> list[tuple[int, int]]:
    seeds: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        vals = np.unique(grid[r])
        if vals.size != 1:
            continue
        color = int(vals[0])
        if color not in FORBIDDEN_SEED_COLORS:
            seeds.append((r, color))
    assert 2 <= len(seeds) <= 3, "expected 2-3 full-width seed rows"
    assert len({color for _r, color in seeds}) == len(seeds), "seed colors must be distinct"
    return seeds


def _control_points(grid: np.ndarray) -> list[tuple[int, int]]:
    controls = sorted((int(r), int(c)) for r, c in np.argwhere(grid == GREEN))
    assert 2 <= len(controls) <= 3, "expected 2-3 green control pixels"
    return controls


def _nearest_seed_color(row: int, seeds: list[tuple[int, int]]) -> int:
    return min(seeds, key=lambda item: (abs(row - item[0]), item[0]))[1]


def _paint_pixel(out: np.ndarray, r: int, c: int, color: int) -> None:
    if _in_bounds(out.shape, r, c):
        out[r, c] = color


def _scaffold_box(control: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = control
    return r - 2, c - 2, r + 2, c + 2


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _has_tie(row: int, seed_rows: list[int]) -> bool:
    distances = sorted(abs(row - seed_row) for seed_row in seed_rows)
    return len(distances) >= 2 and distances[0] == distances[1]


def _oracle_render(grid: np.ndarray) -> np.ndarray:
    seeds = _seed_rows(grid)
    controls = _control_points(grid)

    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape

    for r, c in sorted(controls):
        for cc in range(c - 2, c + 3):
            _paint_pixel(out, r - 2, cc, GREY)

        _paint_pixel(out, r - 1, c, GREY)
        for dr, dc in POST_OFFSETS:
            _paint_pixel(out, r + dr, c + dc, RED)

        baseline_row = r + 2
        baseline_color = _nearest_seed_color(r, seeds)
        if 0 <= baseline_row < h:
            out[baseline_row, :] = baseline_color
            left = max(0, c - 2)
            right = min(w, c + 3)
            out[baseline_row, left:right] = TEAL

        _paint_pixel(out, r, c, GREEN)

    for seed_row, _color in seeds:
        out[seed_row, :] = grid[seed_row, :]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each green control pixel into a fixed scaffold whose baseline color comes from the nearest horizontal seed row.

    Input structure
    - The grid has a black background, 2-3 full-width monochrome seed rows, and 2-3 green control pixels.
    - Seed rows are uniform non-black rows whose color is not green, grey, red, or teal.
    - Each control pixel acts as the center of a 5x5 local scaffold template, with clipping if the template would cross the grid edge.

    Rule
    - For each control pixel at (r, c), draw a grey 5-cell cap on row r-2, a grey stem at (r-1, c), and red side posts at offsets (+/-2) on rows r-1, r, and r+1, while keeping the center pixel green.
    - Find the seed row with minimum vertical distance to the control row; if two seed rows are equally close, choose the upper one.
    - Paint row r+2 across the full width with that chosen seed color, then recolor columns c-2..c+2 on that row to teal.
    - Preserve the original seed rows in place.

    Key insight
    - The scaffold geometry is fixed by the control pixel, but the baseline color is determined independently by the control pixel's nearest seed row, so both structures must be combined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _oracle_render(g)


def _sample_control_rows(height: int, count: int) -> list[int] | None:
    candidates = list(range(CONTROL_MARGIN, height - CONTROL_MARGIN))
    for _ in range(200):
        rows = sorted(random.sample(candidates, count))
        if all(b - a >= CONTROL_ROW_GAP for a, b in zip(rows, rows[1:])):
            return rows
    return None


def _sample_seed_rows(height: int, control_rows: list[int], count: int, want_tie: bool) -> list[int] | None:
    forbidden_rows = set()
    for row in control_rows:
        forbidden_rows.update(range(row - 2, row + 3))

    allowed = [r for r in range(1, height - 1) if r not in forbidden_rows]
    if len(allowed) < count:
        return None

    for _ in range(250):
        rows = sorted(random.sample(allowed, count))
        tie_rows = [row for row in control_rows if _has_tie(row, rows)]
        if want_tie and not tie_rows:
            continue
        if not want_tie and tie_rows:
            continue
        return rows
    return None


def _build_input(
    height: int,
    width: int,
    seed_rows: list[int],
    seed_colors: list[int],
    controls: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for row, color in zip(seed_rows, seed_colors):
        grid[row, :] = color
    for r, c in controls:
        grid[r, c] = GREEN
    return grid


def _validate_input(grid: np.ndarray, require_tie: bool) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    h, w = grid.shape
    if max(h, w) > 30:
        return False

    if not set(int(v) for v in np.unique(grid)) <= {BLACK, BLUE, GREEN, YELLOW, ORANGE}:
        return False

    try:
        seeds = _seed_rows(grid)
        controls = _control_points(grid)
    except AssertionError:
        return False

    seed_rows = [row for row, _color in seeds]
    baseline_colors = []

    for r, c in controls:
        if not (CONTROL_MARGIN <= r < h - CONTROL_MARGIN and CONTROL_MARGIN <= c < w - CONTROL_MARGIN):
            return False
        if r in seed_rows:
            return False
        if grid[r, c] != GREEN:
            return False
        for rr in range(r - 2, r + 3):
            if rr in seed_rows:
                return False
        baseline_colors.append(_nearest_seed_color(r, seeds))

    for i, control_a in enumerate(controls):
        box_a = _scaffold_box(control_a)
        for control_b in controls[i + 1 :]:
            if _boxes_overlap(box_a, _scaffold_box(control_b)):
                return False
            if abs(control_a[0] - control_b[0]) < CONTROL_ROW_GAP:
                return False

    if len({r + 2 for r, _c in controls}) != len(controls):
        return False

    ties = [_has_tie(r, seed_rows) for r, _c in controls]
    if require_tie and not any(ties):
        return False
    if not require_tie and any(ties):
        return False

    if len(set(baseline_colors)) < 2:
        return False

    expected_nonzero = set(seed_rows) | {r for r, _c in controls}
    for r in range(h):
        row = grid[r]
        if r in seed_rows:
            if len(np.unique(row)) != 1:
                return False
        else:
            nonzero_cols = np.flatnonzero(row)
            if len(nonzero_cols) == 0:
                continue
            if r not in {cr for cr, _cc in controls}:
                return False
            if len(nonzero_cols) != sum(1 for cr, _cc in controls if cr == r):
                return False
            for c in nonzero_cols:
                if grid[r, c] != GREEN:
                    return False

    control_rows = {r for r, _c in controls}
    if not expected_nonzero <= (set(seed_rows) | control_rows):
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    control_count = random.randint(*CONTROL_COUNT_RANGE)
    seed_count = random.randint(*SEED_COUNT_RANGE)

    control_rows = _sample_control_rows(height, control_count)
    if control_rows is None:
        return None

    control_cols = random.sample(list(range(CONTROL_MARGIN, width - CONTROL_MARGIN)), control_count)
    controls = list(zip(control_rows, control_cols))

    want_tie = random.random() < TIE_PROBABILITY
    seed_rows = _sample_seed_rows(height, control_rows, seed_count, want_tie)
    if seed_rows is None:
        return None

    seed_colors = random.sample(list(SEED_PALETTE), seed_count)
    grid = _build_input(height, width, seed_rows, seed_colors, controls)

    if not _validate_input(grid, require_tie=want_tie):
        return None

    expected = _oracle_render(grid)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
