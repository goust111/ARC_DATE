from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
TEAL = 8

ENCODE_ORDER = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]

GENERATION_RETRIES = 400
ROW_SAMPLE_RETRIES = 200
PAIR_CYCLE: list[int] = []
GENERATION_CALLS = 0


def _encode_count(n_segments: int) -> np.ndarray:
    assert 0 <= n_segments <= len(ENCODE_ORDER)
    out = np.zeros((3, 3), dtype=int)
    for r, c in ENCODE_ORDER[:n_segments]:
        out[r, c] = BLUE
    return out


def _sorted_red_pixels(grid: np.ndarray) -> list[tuple[int, int]]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(grid == RED)]
    coords.sort()
    return coords


def _slide_red_pixels(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    out[out == RED] = BLACK

    final_positions: list[tuple[int, int]] = []
    occupied: set[tuple[int, int]] = set()

    for r, _c in _sorted_red_pixels(g):
        barrier_cols = [int(c) for c in np.flatnonzero(g[r] == GREY)]
        assert len(barrier_cols) <= 1, "each row may contain at most one grey barrier"
        target_c = barrier_cols[0] + 1 if barrier_cols else 0

        # The puzzle specifies row-order processing plus a right-shift fallback
        # if the nominal destination is already occupied by a previously placed red.
        while (r, target_c) in occupied:
            target_c += 1
            assert target_c < w, "collision resolution must stay inside the row"
            assert g[r, target_c] != GREY, "collision resolution cannot cover a barrier"

        occupied.add((r, target_c))
        final_positions.append((r, target_c))
        out[r, target_c] = RED

    return out, final_positions


def _connect_aligned_pairs(slid_grid: np.ndarray, final_positions: list[tuple[int, int]]) -> tuple[np.ndarray, int]:
    out = slid_grid.copy()
    by_col: dict[int, list[int]] = {}
    for r, c in final_positions:
        by_col.setdefault(int(c), []).append(int(r))

    n_segments = 0
    for c, rows in by_col.items():
        assert len(rows) <= 2, "generator and puzzle both forbid three or more reds in one final column"
        if len(rows) != 2:
            continue
        r0, r1 = sorted(rows)
        assert r1 > r0
        if r1 > r0 + 1:
            segment = out[r0 + 1 : r1, c]
            assert np.all(segment == BLACK), "cyan segments may only pass through background cells"
            out[r0 + 1 : r1, c] = TEAL
        n_segments += 1

    return out, n_segments


def _count_input_column_pairs(grid: np.ndarray) -> tuple[int, int]:
    cols = [int(c) for _r, c in _sorted_red_pixels(grid)]
    counts = Counter(cols)
    return sum(1 for count in counts.values() if count == 2), max(counts.values(), default=0)


def _pair_count_from_positions(final_positions: list[tuple[int, int]]) -> int:
    counts = Counter(int(c) for _r, c in final_positions)
    return sum(1 for count in counts.values() if count == 2)


def _barrier_rows(grid: np.ndarray) -> dict[int, int]:
    rows: dict[int, int] = {}
    for r in range(grid.shape[0]):
        cols = [int(c) for c in np.flatnonzero(grid[r] == GREY)]
        assert len(cols) <= 1
        if cols:
            rows[int(r)] = cols[0]
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each red pixel left to its row stopper, connect the red pairs that land in the same column, and encode the number of such connections as a 3x3 blue pattern.

    Input structure:
    - A black-background grid contains isolated red pixels and isolated grey barrier pixels.
    - Red pixels are sparse and act independently within their own rows.
    - Each row has at most one grey barrier; rows without a barrier use the left edge as the stop.

    Rule:
    - Process red pixels in ascending original row order.
    - In each red pixel's row, erase its original location and place it immediately to the right of that row's grey barrier, or at column 0 if the row has no barrier; if that destination is already occupied by a previously placed red, shift right until the first free cell.
    - After sliding, whenever a column contains exactly two red pixels, fill the black cells strictly between them with teal and count one vertical connection for that column.
    - Return a 3x3 black grid whose first N cells in the fixed order (0,0), (0,2), (1,1), (2,0), (2,2) are blue, where N is the number of teal connections.

    Key insight:
    - The barriers do not matter directly in the output: they only control the slide endpoints, which create or destroy column alignments, which in turn determine how many vertical links exist and therefore which sparse 3x3 code is emitted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    allowed = {BLACK, RED, GREY}
    assert set(map(int, np.unique(g))) <= allowed, "input may only contain black, red, and grey"

    slid, final_positions = _slide_red_pixels(g)
    _connected, n_segments = _connect_aligned_pairs(slid, final_positions)
    return _encode_count(n_segments)


def _next_target_pair_count() -> tuple[int, int]:
    global PAIR_CYCLE, GENERATION_CALLS
    if not PAIR_CYCLE:
        PAIR_CYCLE = [1, 2, 3, 4, 5]
        random.shuffle(PAIR_CYCLE)
    GENERATION_CALLS += 1
    return PAIR_CYCLE.pop(), GENERATION_CALLS


def _sample_row_layout(height: int, red_count: int, n_pairs: int) -> tuple[list[tuple[int, int]], list[int]] | None:
    for _ in range(ROW_SAMPLE_RETRIES):
        rows = sorted(random.sample(range(height), red_count))
        pair_rows = [(rows[2 * i], rows[2 * i + 1]) for i in range(n_pairs)]
        if any(r1 <= r0 + 1 for r0, r1 in pair_rows):
            continue
        singletons = rows[2 * n_pairs :]
        return pair_rows, singletons
    return None


def _critical_barrier_exists(grid: np.ndarray, expected_pairs: int) -> bool:
    red_rows = {int(r) for r, _c in _sorted_red_pixels(grid)}
    barrier_rows = _barrier_rows(grid)
    for r, c in barrier_rows.items():
        if r not in red_rows:
            continue
        mutated = grid.copy()
        mutated[r, c] = BLACK
        try:
            _slid, final_positions = _slide_red_pixels(mutated)
        except AssertionError:
            return True
        if _pair_count_from_positions(final_positions) != expected_pairs:
            return True
    return False


def _semantic_checks(grid: np.ndarray, expected_pairs: int) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) - {BLACK, RED, GREY}:
        return False

    red_pixels = _sorted_red_pixels(grid)
    if len(red_pixels) < 3:
        return False
    if any(c < grid.shape[1] // 2 for _r, c in red_pixels):
        return False

    barrier_rows = _barrier_rows(grid)
    red_rows = {int(r) for r, _c in red_pixels}
    red_rows_with_barrier = red_rows & barrier_rows.keys()
    red_rows_without_barrier = red_rows - barrier_rows.keys()
    if not red_rows_with_barrier or not red_rows_without_barrier:
        return False

    try:
        slid, final_positions = _slide_red_pixels(grid)
        connected, pair_count = _connect_aligned_pairs(slid, final_positions)
    except AssertionError:
        return False

    if pair_count != expected_pairs:
        return False
    if _pair_count_from_positions(final_positions) != expected_pairs:
        return False
    if not np.array_equal(main(grid), _encode_count(expected_pairs)):
        return False

    input_pairs, input_max_col_mult = _count_input_column_pairs(grid)
    if input_max_col_mult > 2:
        return False
    if input_pairs == expected_pairs:
        return False
    if not any(int(c0) != int(c1) for (_r0, c0), (_r1, c1) in zip(red_pixels, final_positions)):
        return False
    if not _critical_barrier_exists(grid, expected_pairs):
        return False

    final_by_col = Counter(int(c) for _r, c in final_positions)
    if max(final_by_col.values(), default=0) > 2:
        return False
    for c, count in final_by_col.items():
        if count != 2:
            continue
        rows = sorted(int(r) for r, cc in final_positions if int(cc) == int(c))
        if rows[1] <= rows[0] + 1:
            return False
        if not np.all(connected[rows[0] + 1 : rows[1], c] == TEAL):
            return False

    return True


def _compose_candidate(n_pairs: int, call_index: int) -> np.ndarray | None:
    prefer_zero_singleton = call_index % 2 == 1
    extra_min = max(1, 3 - 2 * n_pairs)
    extra_max = min(2, 10 - 2 * n_pairs)
    if extra_max < extra_min:
        extra_min = extra_max = max(0, extra_max)
    extra_singletons = random.randint(extra_min, extra_max)
    if n_pairs == 5:
        extra_singletons = 0
    if prefer_zero_singleton and extra_singletons == 0 and n_pairs < 5:
        extra_singletons = 1

    red_count = 2 * n_pairs + extra_singletons
    min_height = max(10, red_count + n_pairs + 1)
    max_height = min(20, min_height + 4)
    if min_height > max_height:
        return None
    height = random.randint(min_height, max_height)

    layout = _sample_row_layout(height, red_count, n_pairs)
    if layout is None:
        return None
    pair_rows, singleton_rows = layout

    distinct_final_cols = n_pairs + extra_singletons
    min_width = max(11, distinct_final_cols + 5)
    max_width = min(18, min_width + 4)
    if min_width > max_width:
        return None
    width = random.randint(min_width, max_width)

    candidate_cols = list(range(0, max(2, width // 2 + 3)))
    if len(candidate_cols) < distinct_final_cols:
        return None

    zero_mode = "singleton" if prefer_zero_singleton and extra_singletons > 0 else "pair"
    if n_pairs == 1:
        zero_mode = "singleton"
    remaining_cols = [c for c in candidate_cols if c != 0]
    if len(remaining_cols) < distinct_final_cols - 1:
        return None
    random.shuffle(remaining_cols)

    pair_cols: list[int] = []
    singleton_cols: list[int] = []
    if zero_mode == "pair":
        pair_cols.append(0)
        pair_cols.extend(remaining_cols[: n_pairs - 1])
        singleton_cols = remaining_cols[n_pairs - 1 : n_pairs - 1 + extra_singletons]
    else:
        singleton_cols.append(0)
        pair_cols = remaining_cols[:n_pairs]
        singleton_cols.extend(remaining_cols[n_pairs : n_pairs + extra_singletons - 1])

    if len(pair_cols) != n_pairs or len(singleton_cols) != extra_singletons:
        return None
    if not any(col > 0 for col in pair_cols):
        return None

    target_by_row: dict[int, int] = {}
    for col, (r0, r1) in zip(pair_cols, pair_rows):
        target_by_row[int(r0)] = int(col)
        target_by_row[int(r1)] = int(col)
    for row, col in zip(singleton_rows, singleton_cols):
        target_by_row[int(row)] = int(col)

    forbidden_noise: set[tuple[int, int]] = set()
    for col, (r0, r1) in zip(pair_cols, pair_rows):
        for r in range(r0 + 1, r1):
            forbidden_noise.add((int(r), int(col)))

    grid = np.zeros((height, width), dtype=int)
    input_red_rows = sorted(target_by_row)
    for r in input_red_rows:
        target_c = target_by_row[r]
        start_lo = max(width // 2, target_c + 1)
        start_hi = width - 1
        if start_lo > start_hi:
            return None
        start_c = random.randint(start_lo, start_hi)
        grid[r, start_c] = RED
        if target_c > 0:
            grid[r, target_c - 1] = GREY

    empty_rows = [r for r in range(height) if r not in target_by_row]
    random.shuffle(empty_rows)
    noise_rows = empty_rows[: random.randint(1, max(1, min(3, len(empty_rows))))]
    for r in noise_rows:
        valid_cols = [c for c in range(width) if (r, c) not in forbidden_noise]
        if not valid_cols:
            return None
        grid[r, random.choice(valid_cols)] = GREY

    return grid


def generate_input() -> np.ndarray:
    target_pairs, call_index = _next_target_pair_count()
    for _ in range(GENERATION_RETRIES):
        grid = _compose_candidate(target_pairs, call_index)
        if grid is None:
            continue
        if _semantic_checks(grid, target_pairs):
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle instance with {target_pairs} aligned pairs")
