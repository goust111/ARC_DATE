from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

MAX_GENERATION_ATTEMPTS = 400
HEIGHT_RANGE = (10, 16)
WIDTH_RANGE = (12, 18)
RED_COUNT_OPTIONS = (3, 4)
EXTRA_BLUE_LINE_MAX = 2
MULTI_BARRIER_PROBABILITY = 0.65


def _analyze_blue_pairs(grid: np.ndarray) -> list[tuple[int, int, int]]:
    pairs: list[tuple[int, int, int]] = []
    for r in range(grid.shape[0]):
        cols = np.flatnonzero(grid[r] == BLUE)
        if len(cols) == 0:
            continue
        assert len(cols) == 2, "expected exactly one blue pair on each used row"
        c0, c1 = map(int, cols)
        assert c1 - c0 >= 2, "blue pair must leave at least one black gap"
        assert np.all(grid[r, c0 + 1 : c1] == BLACK), "blue endpoints must not be pre-connected"
        pairs.append((int(r), c0, c1))
    assert pairs, "expected at least one blue pair"
    return sorted(pairs)


def _analyze_red_sources(grid: np.ndarray) -> list[tuple[int, int]]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(grid == RED)]
    assert coords, "expected at least one red source"
    cols = [c for _, c in coords]
    assert len(cols) == len(set(cols)), "expected at most one red source per column"
    return sorted(coords)


def _barrier_rows_by_column(
    shape: tuple[int, int], blue_pairs: list[tuple[int, int, int]]
) -> list[list[int]]:
    _, width = shape
    barrier_rows: list[list[int]] = [[] for _ in range(width)]
    for r, c0, c1 in blue_pairs:
        for c in range(c0, c1 + 1):
            barrier_rows[c].append(r)
    for rows in barrier_rows:
        rows.sort()
    return barrier_rows


def _solve(
    grid: np.ndarray,
    blue_pairs: list[tuple[int, int, int]] | None = None,
    red_sources: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED}), "unexpected colors"

    pairs = _analyze_blue_pairs(g) if blue_pairs is None else blue_pairs
    sources = _analyze_red_sources(g) if red_sources is None else red_sources

    blue_rows = {r for r, _, _ in pairs}
    assert all(r not in blue_rows for r, _ in sources), "red sources must not share rows with blue pairs"

    out = g.copy()
    for r, c0, c1 in pairs:
        out[r, c0 : c1 + 1] = BLUE

    barriers = _barrier_rows_by_column(g.shape, pairs)
    height = g.shape[0]
    for r, c in sources:
        stop_r = height - 1
        for barrier_r in barriers[c]:
            if barrier_r > r:
                stop_r = barrier_r - 1
                break
        out[r : stop_r + 1, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw horizontal blue barriers between paired blue pixels, then let each red source fall straight down until the nearest barrier or the bottom edge.

    Input structure:
    - The grid is black background with single red source pixels and blue pixels that occur as same-row endpoint pairs.
    - Each used blue row contains exactly one separated blue pair, so the intended horizontal barrier on that row is uniquely determined.
    - Red sources occupy distinct columns and do not share rows with the blue pairs.

    Rule:
    - For every blue pair on row r, fill every cell from the left endpoint to the right endpoint with blue to form a horizontal barrier.
    - For every red source at (r, c), extend red downward in column c through black cells.
    - Stop that downward propagation immediately before the first blue barrier reached in the same column; if no barrier lies below, continue to the bottom row.
    - Preserve all untouched black cells, and never overwrite blue with red.

    Key insight:
    - The visible blue endpoints are only clues: they must first be connected into full horizontal barriers, and those newly created barrier cells then determine where each red column stops.
    """
    return _solve(grid)


def _pick_special_columns(width: int) -> tuple[int, int, int]:
    interior = list(range(1, width - 1))
    for _ in range(200):
        cols = sorted(random.sample(interior, 3))
        if cols[1] - cols[0] >= 3 and cols[2] - cols[1] >= 3:
            if random.random() < 0.5:
                return cols[1], cols[2], cols[0]
            return cols[1], cols[0], cols[2]
    raise RuntimeError("failed to choose spaced special columns")


def _span_covering_target(
    width: int, target: int, forbidden: set[int]
) -> tuple[int, int] | None:
    left_limit = max([-1] + [c for c in forbidden if c < target]) + 1
    right_limit = min([width] + [c for c in forbidden if c > target]) - 1
    if left_limit > target - 1 or right_limit < target + 1:
        return None
    left = random.randint(left_limit, target - 1)
    right = random.randint(target + 1, right_limit)
    return int(left), int(right)


def _span_avoiding_columns(width: int, forbidden: set[int]) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for c0 in range(width):
        for c1 in range(c0 + 2, width):
            if any(c0 <= col <= c1 for col in forbidden):
                continue
            candidates.append((c0, c1))
    if not candidates:
        return None
    return random.choice(candidates)


def _semantic_profile(
    grid: np.ndarray,
) -> tuple[
    list[tuple[int, int, int]],
    list[tuple[int, int]],
    dict[int, list[int]],
    dict[int, list[tuple[int, int, int]]],
]:
    blue_pairs = _analyze_blue_pairs(grid)
    red_sources = _analyze_red_sources(grid)
    rows_by_col = _barrier_rows_by_column(grid.shape, blue_pairs)
    spans_by_col: dict[int, list[tuple[int, int, int]]] = {c: [] for c in range(grid.shape[1])}
    for pair in blue_pairs:
        r, c0, c1 = pair
        for c in range(c0, c1 + 1):
            spans_by_col[c].append(pair)
    return blue_pairs, red_sources, {c: rows_by_col[c] for c in range(grid.shape[1])}, spans_by_col


def _validate_generated_input(grid: np.ndarray, expect_multi_barrier: bool) -> None:
    blue_pairs, red_sources, rows_by_col, spans_by_col = _semantic_profile(grid)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert set(map(int, np.unique(grid))).issubset({BLACK, BLUE, RED})
    assert len(blue_pairs) >= 2
    assert len(red_sources) >= 3

    blue_rows = {r for r, _, _ in blue_pairs}
    source_rows = [r for r, _ in red_sources]
    assert all(r not in blue_rows for r in source_rows)

    blocked = 0
    unblocked = 0
    single_barrier = 0
    multi_barrier = 0
    strict_interior_barrier = 0
    for r, c in red_sources:
        below = [row for row in rows_by_col[c] if row > r]
        if below:
            blocked += 1
            if len(below) == 1:
                single_barrier += 1
            if len(below) >= 2:
                multi_barrier += 1
            if any(c0 < c < c1 for row, c0, c1 in spans_by_col[c] if row in below):
                strict_interior_barrier += 1
        else:
            unblocked += 1

    assert blocked >= 1, "need at least one truncated red source"
    assert unblocked >= 1, "need at least one source that reaches the bottom"
    assert single_barrier >= 1, "need at least one single-barrier example"
    assert strict_interior_barrier >= 1, "need a source blocked by a filled interior line cell"
    if expect_multi_barrier:
        assert multi_barrier >= 1, "need a multi-barrier column when that branch is requested"

    out = _solve(grid, blue_pairs=blue_pairs, red_sources=red_sources)
    assert np.array_equal(out, main(grid))
    changed = int(np.count_nonzero(out != grid))
    assert changed >= len(blue_pairs) + len(red_sources), "output should reflect both line drawing and propagation"


def _build_attempt(expect_multi_barrier: bool) -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)

    source_band_end = max(2, height // 3 - 1)
    source_rows = list(range(source_band_end + 1))
    blue_row_choices = list(range(source_band_end + 2, height - 1))
    if len(source_rows) < 3 or len(blue_row_choices) < 2 + int(expect_multi_barrier):
        return None

    red_count = random.choice(RED_COUNT_OPTIONS)
    block_col, single_col, clear_col = _pick_special_columns(width)
    special_cols = {block_col, single_col, clear_col}

    designated_blue = 2 + int(expect_multi_barrier)
    blue_count = designated_blue + random.randint(0, EXTRA_BLUE_LINE_MAX)
    if blue_count > len(blue_row_choices):
        blue_count = len(blue_row_choices)
    if blue_count < designated_blue:
        return None

    chosen_blue_rows = random.sample(blue_row_choices, blue_count)
    block_rows = sorted(random.sample(chosen_blue_rows, 1 + int(expect_multi_barrier)))
    remaining_rows = [r for r in chosen_blue_rows if r not in block_rows]
    if not remaining_rows:
        return None
    single_row = random.choice(remaining_rows)
    extra_rows = sorted(r for r in remaining_rows if r != single_row)

    spans_by_row: dict[int, tuple[int, int]] = {}
    for row in block_rows:
        span = _span_covering_target(width, block_col, {single_col, clear_col})
        if span is None:
            return None
        spans_by_row[row] = span

    single_span = _span_covering_target(width, single_col, {block_col, clear_col})
    if single_span is None:
        return None
    spans_by_row[single_row] = single_span

    for row in extra_rows:
        span = _span_avoiding_columns(width, special_cols)
        if span is None:
            return None
        spans_by_row[row] = span

    grid = np.full((height, width), BLACK, dtype=int)
    for r, (c0, c1) in spans_by_row.items():
        grid[r, c0] = BLUE
        grid[r, c1] = BLUE

    mandatory_cols = [block_col, single_col, clear_col]
    extra_cols = [c for c in range(width) if c not in special_cols]
    random.shuffle(extra_cols)
    chosen_red_cols = mandatory_cols + extra_cols[: red_count - len(mandatory_cols)]
    if len(chosen_red_cols) != red_count:
        return None

    if len(source_rows) >= len(mandatory_cols):
        chosen_source_rows = random.sample(source_rows, len(mandatory_cols))
    else:
        return None
    while len(chosen_source_rows) < red_count:
        chosen_source_rows.append(random.choice(source_rows))

    assignments = list(zip(chosen_source_rows, chosen_red_cols))
    random.shuffle(assignments)
    for r, c in assignments:
        if grid[r, c] != BLACK:
            return None
        grid[r, c] = RED

    try:
        _validate_generated_input(grid, expect_multi_barrier=expect_multi_barrier)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        expect_multi_barrier = random.random() < MULTI_BARRIER_PROBABILITY
        grid = _build_attempt(expect_multi_barrier=expect_multi_barrier)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
