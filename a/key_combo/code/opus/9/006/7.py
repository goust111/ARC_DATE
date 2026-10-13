import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

SEPARATOR_COLOR = GREEN
SOURCE_COLORS = (BLUE, RED, YELLOW, ORANGE)
HEADER_MAP = {
    (1, 2): GREY,
    (3, 4): PINK,
    (5, 6): TEAL,
    (7, 8): MAROON,
}
BUCKETS = ((1, 2), (3, 4), (5, 6), (7, 8))
_GEN_COUNTER = 0


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _separator_positions(grid: np.ndarray) -> tuple[list[int], list[int]]:
    row_seps = [int(r) for r in np.where(np.all(grid == SEPARATOR_COLOR, axis=1))[0]]
    col_seps = [int(c) for c in np.where(np.all(grid == SEPARATOR_COLOR, axis=0))[0]]
    return row_seps, col_seps


def _bands(seps: list[int], limit: int) -> list[tuple[int, int]]:
    if not seps:
        return []
    spans = []
    ordered = sorted(set(int(x) for x in seps if 0 <= int(x) < limit))
    for a, b in zip(ordered, ordered[1:]):
        if b - a > 1:
            spans.append((a + 1, b - 1))
    return spans


def _parse_table(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_seps, col_seps = _separator_positions(grid)
    row_bands = _bands(row_seps, grid.shape[0])
    col_bands = _bands(col_seps, grid.shape[1])
    assert 3 <= len(row_bands) <= 4, "expected 3-4 table rows"
    assert 3 <= len(col_bands) <= 4, "expected 3-4 table columns"
    return row_bands, col_bands


def _compact_cell(cell: np.ndarray) -> np.ndarray:
    out = np.zeros_like(cell)
    h, w = cell.shape
    for c in range(w):
        vals = [int(cell[r, c]) for r in range(h - 1, -1, -1) if int(cell[r, c]) != BG]
        for offset, val in enumerate(vals):
            out[h - 1 - offset, c] = val
    return out


def _header_color(total: int) -> int:
    if total <= 2:
        return GREY
    if total <= 4:
        return PINK
    if total <= 6:
        return TEAL
    return MAROON


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compact pixels downward inside each table cell and encode each data-column total into the header cell color.

    Input structure:
    - The background is black, and full-span green separator lines form a rectangular table with 3-4 rows and 3-4 columns of cells.
    - The top row of cells is a header row whose interiors are initially black.
    - Every non-header column uses one foreground color from blue/red/yellow/orange, with pixels scattered inside each cell but never on the separator lines.

    Rule:
    - Detect the green separator rows and columns and recover every cell interior between adjacent separator lines.
    - For each non-header cell, keep each pixel in its current local column but compact all foreground pixels downward within that cell so they stack tightly at the bottom.
    - For each table column, count all foreground pixels across the non-header cells and fill the corresponding header cell with grey for totals 1-2, pink for 3-4, teal for 5-6, and maroon for 7-8.
    - Leave the green separator lines unchanged.

    Key insight:
    - The puzzle combines a local per-cell gravity operation with a global per-table-column count, so the header color depends on aggregated pixel totals rather than on the source foreground color or the original pixel positions.
    """
    g = np.asarray(grid, dtype=int)
    row_bands, col_bands = _parse_table(g)

    out = g.copy()
    for col_idx, (c0, c1) in enumerate(col_bands):
        total = 0
        for r0, r1 in row_bands[1:]:
            compacted = _compact_cell(out[r0 : r1 + 1, c0 : c1 + 1])
            out[r0 : r1 + 1, c0 : c1 + 1] = compacted
            total += int(np.count_nonzero(compacted))
        hr0, hr1 = row_bands[0]
        out[hr0 : hr1 + 1, c0 : c1 + 1] = _header_color(total)
    return out


def _all_partitions(total: int, parts: int, cap: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)] if 0 <= total <= cap else []
    out = []
    for first in range(min(cap, total) + 1):
        for rest in _all_partitions(total - first, parts - 1, cap):
            out.append((first,) + rest)
    return out


def _pick_totals(n_cols: int, rng: random.Random, mode: int) -> list[int]:
    if mode == 0:
        assert n_cols == 4
        totals = [rng.randint(lo, hi) for lo, hi in BUCKETS]
        rng.shuffle(totals)
        return totals

    bucket_ids = list(range(4))
    rng.shuffle(bucket_ids)
    chosen = bucket_ids[: min(4, n_cols)]
    while len(chosen) < n_cols:
        chosen.append(rng.choice(bucket_ids))
    totals = [rng.randint(BUCKETS[idx][0], BUCKETS[idx][1]) for idx in chosen]
    rng.shuffle(totals)
    if len(set(totals)) == 1:
        totals[0] = 1 if totals[0] != 1 else 8
    return totals


def _pick_partitions(totals: list[int], data_rows: int, rng: random.Random) -> list[tuple[int, ...]] | None:
    for _ in range(300):
        parts = []
        for total in totals:
            choices = _all_partitions(total, data_rows, 4)
            if not choices:
                return None
            rng.shuffle(choices)
            scored = sorted(
                choices,
                key=lambda p: (
                    0 if any(v == 0 for v in p) else 1,
                    0 if any(v >= 2 for v in p) else 1,
                    -len(set(p)),
                    rng.random(),
                ),
            )
            parts.append(scored[0])

        nonempty_counts = [sum(v > 0 for v in p) for p in parts]
        has_pair = any(
            nonempty_counts[i] == nonempty_counts[j] and totals[i] != totals[j]
            for i in range(len(totals))
            for j in range(i + 1, len(totals))
        )
        has_zero = any(v == 0 for part in parts for v in part)
        has_multi = any(v >= 2 for part in parts for v in part)
        if has_pair and has_zero and has_multi:
            return parts
    return None


def _random_positions(h: int, w: int, count: int, rng: random.Random) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(cells)
    if count == 0:
        return []

    if count >= 2 and rng.random() < 0.6:
        col = rng.randrange(w)
        rows = list(range(h))
        rng.shuffle(rows)
        pos = [(rows[i], col) for i in range(min(count, h))]
        used = set(pos)
        while len(pos) < count:
            candidate = rng.choice(cells)
            if candidate not in used:
                pos.append(candidate)
                used.add(candidate)
        return pos

    return cells[:count]


def _build_grid(
    row_heights: list[int],
    col_widths: list[int],
    totals: list[int],
    partitions: list[tuple[int, ...]],
    col_colors: list[int],
    rng: random.Random,
) -> np.ndarray:
    h = sum(row_heights) + len(row_heights) + 1
    w = sum(col_widths) + len(col_widths) + 1
    grid = np.zeros((h, w), dtype=int)

    row_lines = [0]
    acc = 0
    for height in row_heights:
        acc += height + 1
        row_lines.append(acc)
    col_lines = [0]
    acc = 0
    for width in col_widths:
        acc += width + 1
        col_lines.append(acc)

    grid[row_lines, :] = SEPARATOR_COLOR
    grid[:, col_lines] = SEPARATOR_COLOR

    row_bands = _bands(row_lines, h)
    col_bands = _bands(col_lines, w)

    for col_idx, (c0, c1) in enumerate(col_bands):
        color = col_colors[col_idx]
        for data_row_idx, (r0, r1) in enumerate(row_bands[1:]):
            count = partitions[col_idx][data_row_idx]
            if count == 0:
                continue
            positions = _random_positions(r1 - r0 + 1, c1 - c0 + 1, count, rng)
            for rr, cc in positions:
                grid[r0 + rr, c0 + cc] = color

    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False
    if _get_bg(grid) != BG:
        return False

    row_bands, col_bands = _parse_table(grid)
    if not all(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BG) for c0, c1 in col_bands for r0, r1 in row_bands[:1]):
        return False

    data_rows = row_bands[1:]
    totals = []
    source_colors = []
    changed_cells = 0
    nonempty_counts = []
    seen_zero = False
    seen_multi = False
    for c0, c1 in col_bands:
        col_colors = sorted(set(int(v) for v in np.unique(grid[:, c0 : c1 + 1])) - {BG})
        col_colors = [v for v in col_colors if v != SEPARATOR_COLOR]
        if len(col_colors) != 1 or col_colors[0] not in SOURCE_COLORS:
            return False
        source_colors.append(col_colors[0])
        total = 0
        nonempty = 0
        for r0, r1 in data_rows:
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            compacted = _compact_cell(cell)
            count = int(np.count_nonzero(cell))
            total += count
            if count > 0:
                nonempty += 1
            if count == 0:
                seen_zero = True
            if count >= 2:
                seen_multi = True
            if not np.array_equal(cell, compacted):
                changed_cells += 1
        totals.append(total)
        nonempty_counts.append(nonempty)

    if len(set(source_colors)) != len(source_colors):
        return False
    if not all(1 <= total <= 8 for total in totals):
        return False
    if len(set(totals)) == 1:
        return False
    if len({_header_color(total) for total in totals}) < 2:
        return False
    if not seen_zero or not seen_multi or changed_cells == 0:
        return False
    if not any(
        nonempty_counts[i] == nonempty_counts[j] and totals[i] != totals[j]
        for i in range(len(totals))
        for j in range(i + 1, len(totals))
    ):
        return False

    out = main(grid)
    if np.any(out[np.all(grid == SEPARATOR_COLOR, axis=1), :] != SEPARATOR_COLOR):
        return False
    if np.any(out[:, np.all(grid == SEPARATOR_COLOR, axis=0)] != SEPARATOR_COLOR):
        return False

    for col_idx, (c0, c1) in enumerate(col_bands):
        hr0, hr1 = row_bands[0]
        header = out[hr0 : hr1 + 1, c0 : c1 + 1]
        if not np.all(header == _header_color(totals[col_idx])):
            return False
        for r0, r1 in data_rows:
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], _compact_cell(cell)):
                return False
    return True


def _gen_attempt(rng: random.Random, mode: int) -> np.ndarray | None:
    n_rows = rng.choice([3, 4])
    data_rows = n_rows - 1
    n_cols = 4 if mode == 0 else rng.choices([3, 4], weights=[2, 3])[0]

    row_heights = [rng.randint(3, 4)] + [rng.randint(3, 5) for _ in range(data_rows)]
    col_widths = [rng.randint(3, 6) for _ in range(n_cols)]
    if sum(row_heights) + len(row_heights) + 1 > 30:
        return None
    if sum(col_widths) + len(col_widths) + 1 > 30:
        return None

    totals = _pick_totals(n_cols, rng, mode)
    partitions = _pick_partitions(totals, data_rows, rng)
    if partitions is None:
        return None

    col_colors = rng.sample(list(SOURCE_COLORS), n_cols)
    grid = _build_grid(row_heights, col_widths, totals, partitions, col_colors, rng)
    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    _GEN_COUNTER += 1
    mode = _GEN_COUNTER % 3
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng, mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
