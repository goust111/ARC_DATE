from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


Pair = tuple[int, int, int, int]  # barrier_row, upper_row, lower_row, col

PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 500
MAX_MARKERS_PER_ROW = 3


def _uniform_rows(grid: np.ndarray) -> list[int]:
    return [r for r in range(grid.shape[0]) if np.all(grid[r] == grid[r, 0])]


def _segments(height: int, barriers: list[int]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    prev = -1
    for barrier in barriers:
        segments.append((prev + 1, barrier - 1))
        prev = barrier
    segments.append((prev + 1, height - 1))
    return segments


def _row_mode_nonblack(row: np.ndarray) -> int:
    values = [int(v) for v in row if int(v) != 0]
    assert values, "non-uniform rows must contain at least one non-black cell"
    ranked = Counter(values).most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique non-black row mode"
    return int(ranked[0][0])


def _detect_pairs(grid: np.ndarray) -> tuple[list[int], list[Pair]]:
    height, width = grid.shape
    barriers = _uniform_rows(grid)
    pairs: list[Pair] = []

    prev = -1
    for idx, barrier in enumerate(barriers):
        next_barrier = barriers[idx + 1] if idx + 1 < len(barriers) else height
        upper_rows = range(prev + 1, barrier)
        lower_rows = range(barrier + 1, next_barrier)
        for col in range(width):
            upper_hits = [r for r in upper_rows if int(grid[r, col]) == 0]
            lower_hits = [r for r in lower_rows if int(grid[r, col]) == 0]
            if len(upper_hits) == 1 and len(lower_hits) == 1:
                pairs.append((barrier, upper_hits[0], lower_hits[0], col))
        prev = barrier
    return barriers, pairs


def main(grid: np.ndarray) -> np.ndarray:
    """Detect solid-color rows as implicit barriers, then swap each matched black-marker pair across its barrier.

    Input structure:
    - Some rows are fully uniform solid-color rows; these rows are the implicit barriers.
    - All other rows are non-uniform colored rows with a clear most-common non-black color.
    - Black marker pixels appear in matching same-column pairs on opposite sides of a barrier row.

    Rule:
    - Detect every row whose cells are all the same color.
    - For each barrier row and each column, if there is exactly one black marker in the segment above the barrier
      and exactly one black marker in the segment below it, treat them as a pair.
    - Restore the two source cells to their row's most-common non-black color, then paint black cells at the
      barrier-adjacent destination cells: just above and just below that barrier in the same column.

    Key insight:
    - The barrier is not marked by a special color; it must be inferred from row uniformity before the cross-barrier
      pairing and adjacent-position swap can be applied.
    """
    g = np.asarray(grid, dtype=int)
    barriers, pairs = _detect_pairs(g)
    barrier_set = set(barriers)
    row_fill = {r: _row_mode_nonblack(g[r]) for r in range(g.shape[0]) if r not in barrier_set}

    out = g.copy()
    for barrier, upper_row, lower_row, col in pairs:
        out[upper_row, col] = row_fill[upper_row]
        out[lower_row, col] = row_fill[lower_row]

    for barrier, _upper_row, _lower_row, col in pairs:
        out[barrier - 1, col] = 0
        out[barrier + 1, col] = 0

    return out


def _sample_layout() -> tuple[int, int, list[int], list[tuple[int, int]]]:
    barrier_count = random.choices([2, 3], weights=[3, 2])[0]
    for _ in range(100):
        segment_lengths = [random.randint(2, 4) for _ in range(barrier_count + 1)]
        height = sum(segment_lengths) + barrier_count
        if 10 <= height <= 16:
            barriers: list[int] = []
            cursor = segment_lengths[0]
            for idx in range(barrier_count):
                barriers.append(cursor)
                cursor += 1 + segment_lengths[idx + 1]
            return height, barrier_count, barriers, _segments(height, barriers)
    raise AssertionError("failed to sample a valid layout")


def _sample_pair_counts(barrier_count: int, width: int) -> list[int]:
    counts = [1] * barrier_count
    extra_budget = max(0, min(width - 2 - barrier_count, barrier_count * 2))
    if extra_budget > 0:
        extra = random.randint(1, extra_budget)
        for _ in range(extra):
            choices = [idx for idx in range(barrier_count) if counts[idx] < 3]
            counts[random.choice(choices)] += 1
    if all(count == 1 for count in counts) and width >= barrier_count + 3:
        counts[random.randrange(barrier_count)] += 1
    assert sum(counts) <= width - 2
    return counts


def _pick_row(seg: tuple[int, int], row_load: dict[int, int], prefer: int | None = None) -> int:
    rows = [r for r in range(seg[0], seg[1] + 1) if row_load.get(r, 0) < MAX_MARKERS_PER_ROW]
    assert rows, "segment ran out of row capacity"
    if prefer in rows:
        return int(prefer)
    min_load = min(row_load.get(r, 0) for r in rows)
    light_rows = [r for r in rows if row_load.get(r, 0) == min_load]
    return int(random.choice(light_rows))


def _make_pairs(
    width: int,
    barriers: list[int],
    segments: list[tuple[int, int]],
    pair_counts: list[int],
) -> list[Pair]:
    total_pairs = sum(pair_counts)
    columns = random.sample(range(width), total_pairs)
    row_load: dict[int, int] = defaultdict(int)
    pairs: list[Pair] = []

    ambiguous_barrier = random.randrange(len(barriers) - 1)
    cursor = 0
    for idx, barrier in enumerate(barriers):
        upper_seg = segments[idx]
        lower_seg = segments[idx + 1]
        for local_idx in range(pair_counts[idx]):
            col = int(columns[cursor])
            cursor += 1

            prefer_upper = None
            prefer_lower = None
            if idx == ambiguous_barrier and local_idx == 0:
                prefer_lower = lower_seg[1]
            elif idx > 0 and random.random() < 0.35:
                prefer_upper = upper_seg[0]
            elif idx + 1 < len(barriers) and random.random() < 0.35:
                prefer_lower = lower_seg[1]

            upper_row = _pick_row(upper_seg, row_load, prefer=prefer_upper)
            row_load[upper_row] += 1
            lower_row = _pick_row(lower_seg, row_load, prefer=prefer_lower)
            row_load[lower_row] += 1
            pairs.append((barrier, upper_row, lower_row, col))

    assert len({col for _barrier, _u, _l, col in pairs}) == len(pairs)
    return pairs


def _assign_row_colors(
    height: int,
    width: int,
    barriers: list[int],
    pairs: list[Pair],
) -> tuple[list[int], dict[int, int], dict[int, int], dict[int, list[int]]]:
    barrier_colors = random.sample(PALETTE, len(barriers))
    barrier_set = set(barriers)
    nonbar_rows = [r for r in range(height) if r not in barrier_set]
    random.shuffle(nonbar_rows)

    row_base: dict[int, int] = {}
    for idx, row in enumerate(nonbar_rows):
        if idx < len(barrier_colors):
            row_base[row] = int(barrier_colors[idx])
        else:
            row_base[row] = int(random.choice(PALETTE))

    marker_cols_by_row: dict[int, set[int]] = defaultdict(set)
    for _barrier, upper_row, lower_row, col in pairs:
        marker_cols_by_row[upper_row].add(col)
        marker_cols_by_row[lower_row].add(col)

    row_accent: dict[int, int] = {}
    row_accent_cols: dict[int, list[int]] = {}
    for row in nonbar_rows:
        base = row_base[row]
        accent = int(random.choice([color for color in PALETTE if color != base]))
        allowed = [c for c in range(width) if c not in marker_cols_by_row[row]]
        assert allowed, "each non-uniform row needs at least one non-marker accent cell"
        accent_count = min(len(allowed), random.randint(1, 2))
        accent_cols = random.sample(allowed, accent_count)
        row_accent[row] = accent
        row_accent_cols[row] = accent_cols

    return barrier_colors, row_base, row_accent, row_accent_cols


def _build_input(
    height: int,
    width: int,
    barriers: list[int],
    barrier_colors: list[int],
    row_base: dict[int, int],
    row_accent: dict[int, int],
    row_accent_cols: dict[int, list[int]],
    pairs: list[Pair],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    barrier_color_by_row = dict(zip(barriers, barrier_colors))
    barrier_set = set(barriers)

    for row in range(height):
        if row in barrier_set:
            grid[row, :] = barrier_color_by_row[row]
        else:
            grid[row, :] = row_base[row]
            for col in row_accent_cols[row]:
                grid[row, col] = row_accent[row]

    for _barrier, upper_row, lower_row, col in pairs:
        grid[upper_row, col] = 0
        grid[lower_row, col] = 0

    return grid


def _expected_output(grid: np.ndarray, row_base: dict[int, int], pairs: list[Pair]) -> np.ndarray:
    out = grid.copy()
    for barrier, upper_row, lower_row, col in pairs:
        out[upper_row, col] = row_base[upper_row]
        out[lower_row, col] = row_base[lower_row]
    for barrier, _upper_row, _lower_row, col in pairs:
        out[barrier - 1, col] = 0
        out[barrier + 1, col] = 0
    return out


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    barriers: list[int],
    barrier_colors: list[int],
    row_base: dict[int, int],
    pairs: list[Pair],
) -> None:
    height, width = grid.shape
    assert height <= 30 and width <= 30
    assert _uniform_rows(grid) == barriers
    assert not np.array_equal(grid, expected)

    barrier_set = set(barriers)
    source_cells = {(upper_row, col) for _barrier, upper_row, _lower_row, col in pairs}
    source_cells |= {(lower_row, col) for _barrier, _upper_row, lower_row, col in pairs}
    black_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == 0)}
    assert black_cells == source_cells
    assert len(source_cells) == 2 * len(pairs)
    assert len({(barrier - 1, col) for barrier, _u, _l, col in pairs} | {(barrier + 1, col) for barrier, _u, _l, col in pairs}) == 2 * len(pairs)

    barrier_index = {barrier: idx for idx, barrier in enumerate(barriers)}
    ambiguous_seen = False
    for barrier, upper_row, lower_row, col in pairs:
        idx = barrier_index[barrier]
        lower_seg = (barrier + 1, barriers[idx + 1] - 1 if idx + 1 < len(barriers) else height - 1)
        between = [r for r in barriers if upper_row < r < lower_row]
        assert between == [barrier]
        assert grid[upper_row, col] == 0 and grid[lower_row, col] == 0
        assert lower_seg[0] <= lower_row <= lower_seg[1]
        if idx + 1 < len(barriers) and lower_row == lower_seg[1]:
            ambiguous_seen = True

    assert ambiguous_seen

    for row in range(height):
        if row in barrier_set:
            assert np.all(grid[row] == grid[row, 0])
        else:
            assert not np.all(grid[row] == grid[row, 0])
            assert _row_mode_nonblack(grid[row]) == row_base[row]

    for color in barrier_colors:
        assert any(base == color for base in row_base.values())

    assert np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    try:
        height, _barrier_count, barriers, segments = _sample_layout()
        width = random.randint(10, 15)
        pair_counts = _sample_pair_counts(len(barriers), width)
        pairs = _make_pairs(width, barriers, segments, pair_counts)
        barrier_colors, row_base, row_accent, row_accent_cols = _assign_row_colors(height, width, barriers, pairs)
        grid = _build_input(height, width, barriers, barrier_colors, row_base, row_accent, row_accent_cols, pairs)
        expected = _expected_output(grid, row_base, pairs)
        _validate_instance(grid, expected, barriers, barrier_colors, row_base, pairs)
        return grid
    except (AssertionError, ValueError):
        return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid implicit-barrier swap puzzle")
