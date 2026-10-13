from __future__ import annotations

import numpy as np


"""
Reorder each query row's color blocks to match the reference row order.

Input structure
- Background is black (0).
- Row 0 is the reference row: contiguous non-black color blocks, one block per color.
- Each later non-empty row is a query row using the same color set in a different order.
- Query rows are visually separated by all-black rows.

Rule
- Read the left-to-right color order from the reference row.
- For each query row, reorder its blocks by that reference order.
- Keep every block's width unchanged; only their horizontal order changes.
- Preserve the reference row and background rows.
"""


BG = 0
PALETTE = tuple(range(1, 10))
COLOR_COUNT_RANGE = (3, 7)  # high bound exclusive
QUERY_COUNT_RANGE = (2, 6)  # high bound exclusive
BLOCK_WIDTH_RANGE = (1, 4)  # high bound exclusive
SIDE_PADDING_RANGE = (0, 4)  # high bound exclusive
MAX_GENERATION_ATTEMPTS = 300


def _segments(row: np.ndarray, bg: int = BG) -> list[tuple[int, int, int]]:
    """Return contiguous non-background runs as (start, end, color)."""
    row = np.asarray(row, dtype=int)
    segments: list[tuple[int, int, int]] = []
    c = 0
    width = int(row.shape[0])
    while c < width:
        if int(row[c]) == bg:
            c += 1
            continue
        start = c
        color = int(row[c])
        while c < width and int(row[c]) == color:
            c += 1
        segments.append((start, c, color))
    return segments


def _reference_order(grid: np.ndarray) -> list[int]:
    ref_segments = _segments(grid[0], BG)
    assert len(ref_segments) >= 2, "reference row must contain at least two color blocks"
    order = [color for _start, _end, color in ref_segments]
    assert len(order) == len(set(order)), "reference row colors must be unique"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 2, "expected a reference row and at least one query row"

    ref_order = _reference_order(g)
    priority = {color: idx for idx, color in enumerate(ref_order)}

    out = g.copy()
    for r in range(1, g.shape[0]):
        segments = _segments(g[r], BG)
        if not segments:
            continue

        colors = [color for _start, _end, color in segments]
        assert len(colors) == len(set(colors)), "query row colors must be unique"
        assert set(colors) == set(ref_order), "query row must use the same colors as the reference row"

        left = segments[0][0]
        right = segments[-1][1]
        total_width = sum(end - start for start, end, _color in segments)
        assert right - left == total_width, "query row blocks must be contiguous"

        ordered = sorted(segments, key=lambda seg: priority[seg[2]])
        out[r, left:right] = BG
        cursor = left
        for start, end, color in ordered:
            block_w = end - start
            out[r, cursor : cursor + block_w] = color
            cursor += block_w

    return out


def _sample_widths(colors: list[int]) -> dict[int, int]:
    return {int(color): int(np.random.randint(*BLOCK_WIDTH_RANGE)) for color in colors}


def _scrambled_order(colors: list[int]) -> list[int]:
    base = list(colors)
    for _ in range(20):
        mode = int(np.random.randint(3))
        if mode == 0:
            candidate = list(reversed(base))
        elif mode == 1:
            shift = int(np.random.randint(1, len(base)))
            candidate = base[shift:] + base[:shift]
        else:
            candidate = list(np.random.permutation(base))
        if candidate != base:
            return [int(color) for color in candidate]
    candidate = base[:]
    i, j = np.random.choice(len(base), size=2, replace=False)
    candidate[int(i)], candidate[int(j)] = candidate[int(j)], candidate[int(i)]
    return [int(color) for color in candidate]


def _build_row(order: list[int], widths: dict[int, int], left_pad: int, row_width: int) -> np.ndarray:
    row = np.full(row_width, BG, dtype=int)
    cursor = int(left_pad)
    for color in order:
        block_w = int(widths[color])
        row[cursor : cursor + block_w] = int(color)
        cursor += block_w
    return row


def _row_spec(order: list[int], widths: dict[int, int], left_pad: int) -> tuple[list[int], dict[int, int], int, int]:
    active_width = sum(int(widths[color]) for color in order)
    return order, widths, int(left_pad), int(active_width)


def _gen_attempt() -> np.ndarray | None:
    color_count = int(np.random.randint(*COLOR_COUNT_RANGE))
    query_count = int(np.random.randint(*QUERY_COUNT_RANGE))
    colors = [int(color) for color in np.random.choice(PALETTE, size=color_count, replace=False)]

    row_specs: list[tuple[list[int], dict[int, int], int, int]] = []

    ref_left = int(np.random.randint(*SIDE_PADDING_RANGE))
    ref_widths = _sample_widths(colors)
    row_specs.append(_row_spec(colors, ref_widths, ref_left))

    seen_orders: set[tuple[int, ...]] = set()
    for _ in range(query_count):
        order = _scrambled_order(colors)
        seen_orders.add(tuple(order))
        left_pad = int(np.random.randint(*SIDE_PADDING_RANGE))
        widths = _sample_widths(colors)
        row_specs.append(_row_spec(order, widths, left_pad))

    if not seen_orders:
        return None

    row_width = max(left_pad + active_width for _order, _widths, left_pad, active_width in row_specs)
    row_width += int(np.random.randint(*SIDE_PADDING_RANGE))

    height = 1 + query_count + (query_count - 1)
    grid = np.full((height, row_width), BG, dtype=int)

    order, widths, left_pad, _active_width = row_specs[0]
    grid[0] = _build_row(order, widths, left_pad, row_width)

    for idx, spec in enumerate(row_specs[1:]):
        row_index = 1 + 2 * idx
        order, widths, left_pad, _active_width = spec
        grid[row_index] = _build_row(order, widths, left_pad, row_width)

    if not _valid_generated_grid(grid):
        return None
    return grid


def _valid_generated_grid(grid: np.ndarray) -> bool:
    ref_order = _reference_order(grid)
    query_rows = 0
    distinct_orders: set[tuple[int, ...]] = set()

    for r in range(1, grid.shape[0]):
        segments = _segments(grid[r], BG)
        if not segments:
            continue
        query_rows += 1
        order = tuple(color for _start, _end, color in segments)
        if set(order) != set(ref_order):
            return False
        if order == tuple(ref_order):
            return False
        distinct_orders.add(order)
        if any((end - start) not in {1, 2, 3} for start, end, _color in segments):
            return False

    return query_rows >= 2 and len(distinct_orders) >= 1


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
