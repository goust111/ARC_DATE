import random

import numpy as np


BG = 0
PALETTE_COLORS = (1, 2, 3, 4, 5, 6, 7, 8, 9)

GRID_H_RANGE = (6, 16)
GRID_W_RANGE = (11, 28)
LEGEND_LEN_RANGE = (2, 5)
GEN_RETRY_LIMIT = 300
ROW_SCRAMBLE_RETRIES = 40


def _legend_order(grid: np.ndarray) -> list[int]:
    cols = np.flatnonzero(grid[0] != BG)
    legend = [int(grid[0, c]) for c in cols]
    assert legend, "top row must contain a non-empty color legend"
    assert len(set(legend)) == len(legend), "legend colors must be unique"
    return legend


def _repack_from_counts(width: int, priority_counts: list[tuple[int, int]]) -> np.ndarray:
    row = np.full(width, BG, dtype=int)
    total = sum(count for _color, count in priority_counts)
    if total == 0:
        return row

    assert total <= width, "foreground count must fit on the row"
    start = (width - total) // 2
    available = list(range(start, start + total))
    center = (width - 1) / 2.0

    for color, count in priority_counts:
        if count == 0:
            continue
        # When two cells are equally central, use the left cell first.
        ranked = sorted(available, key=lambda col: (abs(col - center), col))
        chosen = ranked[:count]
        row[chosen] = int(color)
        chosen_set = set(chosen)
        available = [col for col in available if col not in chosen_set]

    return row


def _repack_row(row: np.ndarray, priorities: list[int]) -> np.ndarray:
    counts = [(color, int(np.count_nonzero(row == color))) for color in priorities]
    return _repack_from_counts(int(row.shape[0]), counts)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each row's mixed foreground colors away from the center according to a top-row priority legend.

    Input structure:
    - Background is black.
    - The top row contains a left-to-right legend of distinct non-black colors; farther left means higher priority.
    - Each lower row contains zero or more foreground pixels whose colors are drawn from that legend.

    Rule:
    - Leave the legend row unchanged.
    - For each lower row, count how many pixels belong to each legend color and ignore their original columns.
    - Center the row's total foreground span, then assign the currently most central free cells to the highest-priority color, the next-most-central free cells to the next color, and so on until all counts are placed.
    - This keeps high-priority colors in the middle, pushes lower-priority colors toward the edges, and preserves every color's pixel count.

    Key insight:
    - The legend controls spatial priority rather than recoloring: each row is rebuilt only from per-color counts, with priority deciding distance from the row center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    priorities = _legend_order(g)
    allowed = set(priorities)

    out = np.full_like(g, BG)
    out[0] = g[0]

    for r in range(1, g.shape[0]):
        row = g[r]
        row_colors = {int(v) for v in row if int(v) != BG}
        assert row_colors <= allowed, "all foreground colors must appear in the legend"
        out[r] = _repack_row(row, priorities)

    return out


def _positive_partition(total: int, parts: int, rng: random.Random) -> list[int]:
    assert parts >= 1 and total >= parts
    if parts == 1:
        return [total]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _sample_row_counts(colors: list[int], width: int, rng: random.Random) -> dict[int, int]:
    k = len(colors)
    max_total = min(width - 2, k * 5)
    min_total = k
    assert max_total >= min_total
    total = rng.randint(min_total, max_total)
    parts = _positive_partition(total, k, rng)
    rng.shuffle(parts)
    return {color: count for color, count in zip(colors, parts)}


def _scramble_row(counts: dict[int, int], width: int, priorities: list[int], rng: random.Random) -> np.ndarray | None:
    total = sum(counts.values())
    colors = [color for color, count in counts.items() for _ in range(count)]

    for _ in range(ROW_SCRAMBLE_RETRIES):
        row = np.full(width, BG, dtype=int)
        cols = sorted(rng.sample(range(width), total))
        shuffled = colors[:]
        rng.shuffle(shuffled)
        for col, color in zip(cols, shuffled):
            row[col] = int(color)
        if not np.array_equal(row, _repack_row(row, priorities)):
            return row
    return None


def _sample_subset(priorities: list[int], row_index: int, n_active: int, rng: random.Random) -> list[int]:
    n_colors = len(priorities)
    if row_index == 0:
        return priorities[:]
    if row_index == 1 and n_colors >= 3:
        subset_size = rng.randint(2, n_colors - 1)
        return rng.sample(priorities, subset_size)
    subset_size = rng.randint(2, n_colors)
    subset = rng.sample(priorities, subset_size)
    if row_index == n_active - 1 and n_colors >= 3 and len(subset) == 2:
        extra = rng.choice([c for c in priorities if c not in subset])
        subset.append(extra)
    return subset


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)

    max_colors = min(LEGEND_LEN_RANGE[1], width - 2)
    if max_colors < LEGEND_LEN_RANGE[0]:
        return None
    n_colors = rng.randint(LEGEND_LEN_RANGE[0], max_colors)
    priorities = rng.sample(PALETTE_COLORS, n_colors)

    grid = np.zeros((height, width), dtype=int)
    legend_cols = sorted(rng.sample(range(width), n_colors))
    for col, color in zip(legend_cols, priorities):
        grid[0, col] = int(color)

    lower_rows = list(range(1, height))
    n_active = rng.randint(max(2, len(lower_rows) // 2), len(lower_rows))
    active_rows = sorted(rng.sample(lower_rows, n_active))

    changed_rows = 0
    for idx, row_idx in enumerate(active_rows):
        subset = _sample_subset(priorities, idx, n_active, rng)
        counts = _sample_row_counts(subset, width, rng)
        row = _scramble_row(counts, width, priorities, rng)
        if row is None:
            return None
        grid[row_idx] = row
        changed_rows += int(not np.array_equal(row, _repack_row(row, priorities)))

    if changed_rows == 0:
        return None

    used_below = {int(v) for v in grid[1:].ravel() if int(v) != BG}
    if used_below != set(priorities):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
