from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _spans_from_bar_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return non-bar runs as half-open spans [start, end)."""
    mask = np.asarray(mask, dtype=bool)
    spans: list[tuple[int, int]] = []
    start = None
    for idx, is_bar in enumerate(mask.tolist()):
        if is_bar:
            if start is not None:
                spans.append((start, idx))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color in map(int, np.unique(grid)):
        row_mask = np.all(grid == color, axis=1)
        col_mask = np.all(grid == color, axis=0)
        if not row_mask.any() or not col_mask.any():
            continue
        if not (row_mask[0] and row_mask[-1] and col_mask[0] and col_mask[-1]):
            continue
        row_spans = _spans_from_bar_mask(row_mask)
        col_spans = _spans_from_bar_mask(col_mask)
        if len(row_spans) >= 4 and len(col_spans) >= 4:
            candidates.append((color, row_spans, col_spans))
    assert len(candidates) == 1, "expected one separator color forming a framed lattice"
    return candidates[0]


def _infer_background(grid: np.ndarray, separator: int) -> int:
    values, counts = np.unique(grid[grid != separator], return_counts=True)
    assert len(values) > 0
    return int(values[np.argmax(counts)])


def _cell_color(cell: np.ndarray, bg: int, separator: int) -> int | None:
    assert not np.any(cell == separator), "separator color must stay on lattice bars"
    non_bg = sorted(set(map(int, np.unique(cell))) - {bg})
    if not non_bg:
        return None
    assert len(non_bg) == 1, "each cell may contain at most one sprite color"
    color = int(non_bg[0])
    mask = cell == color
    box = _bbox(mask)
    assert box is not None, "non-empty sprite must have a bbox"
    r0, c0, r1, c1 = box
    assert int(mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1), "sprite must be a solid rectangle"
    return color


def _analyze_grid(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], list[list[int | None]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    separator, row_spans, col_spans = _find_separator(g)
    bg = _infer_background(g, separator)

    cell_colors: list[list[int | None]] = []
    for r0, r1 in row_spans:
        row: list[int | None] = []
        for c0, c1 in col_spans:
            row.append(_cell_color(g[r0:r1, c0:c1], bg, separator))
        cell_colors.append(row)
    return separator, bg, row_spans, col_spans, cell_colors


def _counts_from_cells(cell_colors: list[list[int | None]]) -> tuple[dict[int, int], dict[int, int]]:
    n_rows = len(cell_colors)
    n_cols = len(cell_colors[0])
    edge_counts: Counter[int] = Counter()
    inner_counts: Counter[int] = Counter()
    for r, row in enumerate(cell_colors):
        for c, color in enumerate(row):
            if color is None:
                continue
            if r in (0, n_rows - 1) or c in (0, n_cols - 1):
                edge_counts[color] += 1
            else:
                inner_counts[color] += 1
    return dict(edge_counts), dict(inner_counts)


def main(grid: np.ndarray) -> np.ndarray:
    """Summarize each sprite color into one bar whose length comes from interior count and whose row order comes from perimeter count.

    Input structure:
    - A single separator color forms a complete framed lattice that partitions the grid into a rectangular array of cells.
    - Some cells are empty, and each occupied cell contains one solid monochrome rectangular sprite on the background.
    - Every sprite color appears at least once on the outer ring of cells and at least once in the strictly interior cells.

    Rule:
    - Count, for each sprite color, how many occupied cells of that color lie on the array perimeter and how many lie in the array interior.
    - Sort colors by increasing perimeter count.
    - Output one left-aligned horizontal bar per color, with bar length equal to that color's interior count and padding filled with background.

    Key insight:
    - The same visible color carries two independent statistics: perimeter frequency chooses the row, while interior frequency chooses the bar length.
    """
    _separator, bg, _row_spans, _col_spans, cell_colors = _analyze_grid(grid)
    edge_counts, inner_counts = _counts_from_cells(cell_colors)

    assert edge_counts, "expected at least one sprite color"
    colors = sorted(edge_counts)
    assert set(colors) == set(inner_counts), "each sprite color must appear on both perimeter and interior"
    assert len(set(edge_counts.values())) == len(colors), "perimeter counts must be unique"
    assert len(set(inner_counts.values())) == len(colors), "interior counts must be unique"

    ordered_colors = sorted(colors, key=lambda color: edge_counts[color])
    width = max(inner_counts.values())
    out = np.full((len(ordered_colors), width), bg, dtype=int)
    for r, color in enumerate(ordered_colors):
        out[r, : inner_counts[color]] = color
    return out


MIN_CELL_ROWS = 4
MAX_CELL_ROWS = 5
MIN_CELL_COLS = 4
MAX_CELL_COLS = 6
CELL_SIZE_RANGE = (3, 4)
GAP_RANGE = (1, 2)
COLOR_POOL = tuple(range(1, 10))
MAX_ATTEMPTS = 600


def _sample_geometry() -> tuple[list[int], list[int], int, int] | None:
    for _ in range(60):
        n_rows = random.randint(MIN_CELL_ROWS, MAX_CELL_ROWS)
        n_cols = random.randint(MIN_CELL_COLS, MAX_CELL_COLS)
        row_gap = random.randint(*GAP_RANGE)
        col_gap = random.randint(*GAP_RANGE)
        row_heights = [random.randint(*CELL_SIZE_RANGE) for _ in range(n_rows)]
        col_widths = [random.randint(*CELL_SIZE_RANGE) for _ in range(n_cols)]
        height = sum(row_heights) + (n_rows + 1) * row_gap
        width = sum(col_widths) + (n_cols + 1) * col_gap
        if height <= 30 and width <= 30:
            return row_heights, col_widths, row_gap, col_gap
    return None


def _sample_distinct_counts(n: int, capacity: int) -> list[int] | None:
    upper = min(capacity, 8)
    choices = list(range(1, upper + 1))
    if len(choices) < n:
        return None
    for _ in range(120):
        counts = random.sample(choices, n)
        if sum(counts) <= capacity:
            return counts
    return None


def _anti_shortcut_ok(edge_counts: dict[int, int], inner_counts: dict[int, int]) -> bool:
    colors = sorted(edge_counts)
    order_by_edge = sorted(colors, key=lambda color: edge_counts[color])
    order_by_inner = sorted(colors, key=lambda color: inner_counts[color])
    order_by_total = sorted(colors, key=lambda color: edge_counts[color] + inner_counts[color])
    if order_by_edge == order_by_inner or order_by_edge == order_by_total:
        return False
    if any(edge_counts[color] == inner_counts[color] for color in colors):
        return False
    return True


def _expected_output(bg: int, colors: list[int], edge_counts: dict[int, int], inner_counts: dict[int, int]) -> np.ndarray:
    ordered = sorted(colors, key=lambda color: edge_counts[color])
    out = np.full((len(ordered), max(inner_counts.values())), bg, dtype=int)
    for r, color in enumerate(ordered):
        out[r, : inner_counts[color]] = color
    return out


def _sample_case_spec() -> tuple[
    int,
    int,
    list[int],
    list[int],
    int,
    int,
    list[int],
    dict[int, int],
    dict[int, int],
] | None:
    geometry = _sample_geometry()
    if geometry is None:
        return None
    row_heights, col_widths, _row_gap, _col_gap = geometry
    n_rows = len(row_heights)
    n_cols = len(col_widths)
    perimeter_slots = 2 * n_rows + 2 * n_cols - 4
    interior_slots = (n_rows - 2) * (n_cols - 2)
    max_colors = min(4, perimeter_slots, interior_slots)
    if max_colors < 2:
        return None

    for _ in range(120):
        n_colors = random.randint(2, max_colors)
        edge_values = _sample_distinct_counts(n_colors, perimeter_slots - random.randint(0, 2))
        inner_values = _sample_distinct_counts(n_colors, interior_slots - random.randint(0, 2))
        if edge_values is None or inner_values is None:
            continue
        if sum(edge_values) > perimeter_slots or sum(inner_values) > interior_slots:
            continue
        palette = random.sample(COLOR_POOL, n_colors + 2)
        separator = palette[0]
        bg = palette[1]
        colors = palette[2:]
        edge_counts = {color: count for color, count in zip(colors, edge_values)}
        inner_counts = {color: count for color, count in zip(colors, inner_values)}
        if _anti_shortcut_ok(edge_counts, inner_counts):
            return separator, bg, row_heights, col_widths, geometry[2], geometry[3], colors, edge_counts, inner_counts
    return None


def _place_counts(
    positions: list[tuple[int, int]],
    colors: list[int],
    counts: dict[int, int],
) -> dict[tuple[int, int], int]:
    shuffled = positions[:]
    random.shuffle(shuffled)
    out: dict[tuple[int, int], int] = {}
    idx = 0
    for color in random.sample(colors, len(colors)):
        for _ in range(counts[color]):
            out[shuffled[idx]] = color
            idx += 1
    return out


def _build_cell_spans(lengths: list[int], gap: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = gap
    for size in lengths:
        spans.append((pos, pos + size))
        pos += size + gap
    return spans


def _render_grid(
    separator: int,
    bg: int,
    row_heights: list[int],
    col_widths: list[int],
    row_gap: int,
    col_gap: int,
    placements: dict[tuple[int, int], int],
) -> np.ndarray:
    row_spans = _build_cell_spans(row_heights, row_gap)
    col_spans = _build_cell_spans(col_widths, col_gap)
    height = row_spans[-1][1] + row_gap
    width = col_spans[-1][1] + col_gap

    grid = np.full((height, width), separator, dtype=int)
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            grid[r0:r1, c0:c1] = bg
            color = placements.get((r, c))
            if color is None:
                continue
            cell_h = r1 - r0
            cell_w = c1 - c0
            sprite_h = random.randint(1, max(1, cell_h - 2))
            sprite_w = random.randint(1, max(1, cell_w - 2))
            top = random.randint(r0, r1 - sprite_h)
            left = random.randint(c0, c1 - sprite_w)
            grid[top : top + sprite_h, left : left + sprite_w] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    spec = _sample_case_spec()
    if spec is None:
        return None
    separator, bg, row_heights, col_widths, row_gap, col_gap, colors, edge_counts, inner_counts = spec
    n_rows = len(row_heights)
    n_cols = len(col_widths)

    perimeter_positions = [
        (r, c)
        for r in range(n_rows)
        for c in range(n_cols)
        if r in (0, n_rows - 1) or c in (0, n_cols - 1)
    ]
    interior_positions = [
        (r, c)
        for r in range(1, n_rows - 1)
        for c in range(1, n_cols - 1)
    ]

    placements = {}
    placements.update(_place_counts(perimeter_positions, colors, edge_counts))
    placements.update(_place_counts(interior_positions, colors, inner_counts))
    grid = _render_grid(separator, bg, row_heights, col_widths, row_gap, col_gap, placements)

    try:
        found_separator, found_bg, found_row_spans, found_col_spans, cell_colors = _analyze_grid(grid)
    except Exception:
        return None
    if found_separator != separator or found_bg != bg:
        return None
    if len(found_row_spans) != n_rows or len(found_col_spans) != n_cols:
        return None

    found_edge, found_inner = _counts_from_cells(cell_colors)
    if found_edge != edge_counts or found_inner != inner_counts:
        return None
    if not _anti_shortcut_ok(found_edge, found_inner):
        return None

    expected = _expected_output(bg, colors, edge_counts, inner_counts)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
