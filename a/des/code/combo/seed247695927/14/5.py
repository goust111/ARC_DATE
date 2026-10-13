import random

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the innermost mirrored marker pair(s) by filling their horizontal span.

    Input structure
    - The grid uses exactly one background color and one foreground color.
    - Every non-empty row contains exactly two foreground pixels mirrored
      across the vertical center.

    Rule
    - Find the non-empty row(s) with the smallest horizontal gap between the
      two mirrored markers.
    - Fill all pixels between those markers with the foreground color.
    - Leave every other cell unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    colors = sorted(set(map(int, g.ravel())) - {bg})
    assert len(colors) == 1, "expected exactly one foreground color"
    fg = colors[0]

    height, width = g.shape
    pairs: list[tuple[int, int, int]] = []
    for r in range(height):
        cols = np.flatnonzero(g[r] == fg)
        assert cols.size in (0, 2), "each foreground row must have exactly two markers"
        if cols.size == 2:
            c0, c1 = map(int, cols)
            assert c0 < c1, "markers must occupy distinct columns"
            assert c0 + c1 == width - 1, "markers must be mirrored across the vertical center"
            pairs.append((c1 - c0, c0, r))

    assert pairs, "expected at least one mirrored marker pair"
    min_gap = min(gap for gap, _c0, _r in pairs)

    out = g.copy()
    for gap, c0, r in pairs:
        if gap == min_gap:
            c1 = width - 1 - c0
            out[r, c0 : c1 + 1] = fg
    return out


HEIGHT_RANGE = (4, 10)
WIDTH_RANGE = (5, 15)
MAX_ATTEMPTS = 400


def _choose_gap_levels(width: int, row_count: int) -> list[int]:
    possible_gaps = [width - 1 - 2 * c0 for c0 in range(width // 2)]
    assert possible_gaps

    if len(possible_gaps) == 1 or row_count == 1:
        return [possible_gaps[0]]

    level_cap = min(len(possible_gaps), row_count, 4)
    level_count = random.randint(2, level_cap)
    return sorted(random.sample(possible_gaps, level_count))


def _allocate_rows(total_rows: int, level_count: int) -> list[int]:
    counts = [1] * level_count
    remaining = total_rows - level_count
    while remaining > 0:
        counts[random.randrange(level_count)] += 1
        remaining -= 1
    random.shuffle(counts)
    return counts


def _gap_to_columns(width: int, gap: int) -> tuple[int, int]:
    c0 = (width - 1 - gap) // 2
    c1 = width - 1 - c0
    assert c0 + c1 == width - 1
    assert c1 - c0 == gap
    return c0, c1


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)

    bg, fg = random.sample(range(10), 2)
    row_count = random.randint(1, height)

    gap_levels = _choose_gap_levels(width, row_count)
    min_gap = gap_levels[0]

    if len(gap_levels) == 1:
        selected_rows = row_count
        level_counts = [row_count]
    else:
        selected_rows = random.randint(1, max(1, row_count // 2 + 1))
        selected_rows = min(selected_rows, row_count - (len(gap_levels) - 1))
        selected_rows = max(selected_rows, 1)

        other_counts = _allocate_rows(row_count - selected_rows, len(gap_levels) - 1)
        level_counts = [selected_rows] + other_counts

    assert sum(level_counts) == row_count
    assert level_counts[0] >= 1
    if len(gap_levels) > 1:
        assert all(count >= 1 for count in level_counts[1:])

    rows = sorted(random.sample(range(height), row_count))
    random.shuffle(rows)

    grid = np.full((height, width), bg, dtype=int)
    idx = 0
    for gap, count in zip(gap_levels, level_counts):
        c0, c1 = _gap_to_columns(width, gap)
        for _ in range(count):
            r = rows[idx]
            idx += 1
            grid[r, c0] = fg
            grid[r, c1] = fg

    nonempty_rows = [r for r in range(height) if np.count_nonzero(grid[r] == fg) == 2]
    assert len(nonempty_rows) == row_count

    gaps = []
    for r in nonempty_rows:
        cols = np.flatnonzero(grid[r] == fg)
        gaps.append(int(cols[1] - cols[0]))

    assert min(gaps) == min_gap
    assert gaps.count(min_gap) == level_counts[0]
    assert len(set(gaps)) == len(gap_levels)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
