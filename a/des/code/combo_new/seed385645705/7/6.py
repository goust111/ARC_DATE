from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _background(g: np.ndarray) -> int:
    counts = Counter(int(v) for v in g.ravel())
    return counts.most_common(1)[0][0]


def _ruler_colors(line: np.ndarray, bg: int) -> list[int]:
    colors: list[int] = []
    prev = None
    for raw in line:
        color = int(raw)
        if color == bg:
            prev = color
            continue
        if color != prev:
            colors.append(color)
        prev = color
    assert colors, "each ruler must contain at least one colored segment"
    assert len(colors) == len(set(colors)), "ruler segment colors must be unique"
    return colors


def _read_legend(g: np.ndarray, bg: int) -> tuple[int, int, dict[tuple[int, int], int]]:
    row_colors = _ruler_colors(g[1:, 0], bg)
    col_colors = _ruler_colors(g[0, 1:], bg)
    rh = len(row_colors)
    cw = len(col_colors)

    r0 = g.shape[0] - rh - 1
    c0 = g.shape[1] - cw - 1
    assert r0 > 0 and c0 > 0, "legend must sit inside the bottom-right corner"
    assert int(g[r0, c0]) == bg, "legend corner cell must be background"

    row_headers = [int(g[r0 + i, c0]) for i in range(1, rh + 1)]
    col_headers = [int(g[r0, c0 + j]) for j in range(1, cw + 1)]
    assert len(set(row_headers)) == rh and set(row_headers) == set(row_colors)
    assert len(set(col_headers)) == cw and set(col_headers) == set(col_colors)

    table: dict[tuple[int, int], int] = {}
    for i, row_color in enumerate(row_headers, start=1):
        for j, col_color in enumerate(col_headers, start=1):
            table[(row_color, col_color)] = int(g[r0 + i, c0 + j])
    return r0, c0, table


def _monochrome_components(g: np.ndarray, mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    seen = np.zeros(g.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr_raw, sc_raw in np.argwhere(mask):
        sr = int(sr_raw)
        sc = int(sc_raw)
        if seen[sr, sc]:
            continue
        color = int(g[sr, sc])
        stack = [(sr, sc)]
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                    if mask[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
        comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each inner solid rectangle by looking up the row-ruler and column-ruler color pair in the corner mixing table.

    Input structure:
    - The dominant color is the background.
    - The first row is a top ruler made of contiguous color segments, and the first column is a left ruler made of contiguous color segments.
    - The interior contains separated monochrome solid rectangles on the background.
    - The bottom-right corner contains a mixing legend whose top row and left column are color headers, with result colors inside the table.

    Rule:
    - Read the bottom-right legend as a mapping from (left-ruler row color, top-ruler column color) to a result color.
    - For each solid rectangle outside the rulers and outside the legend, take the center cell of its bounding box.
    - Use the first-column ruler color at that center row and the first-row ruler color at that center column to look up the rectangle's new color.
    - Recolor the whole rectangle with that result color while leaving the rulers and legend unchanged.

    Key insight:
    - The original rectangle colors are distractors; the visible row and column rulers project each rectangle to a two-dimensional lookup table in the corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background(g)
    legend_r0, legend_c0, table = _read_legend(g, bg)

    object_mask = g != bg
    object_mask[0, :] = False
    object_mask[:, 0] = False
    object_mask[legend_r0:, legend_c0:] = False

    out = g.copy()
    for comp in _monochrome_components(g, object_mask):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        color = int(g[r0, c0])
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert len(comp) == area, "interior foreground components must be solid rectangles"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color)

        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        row_color = int(g[center_r, 0])
        col_color = int(g[0, center_c])
        assert (row_color, col_color) in table
        out[r0 : r1 + 1, c0 : c1 + 1] = table[(row_color, col_color)]

    return out


# --------------- generator ---------------

COLOR_IDS = tuple(range(10))
GENERATE_RETRIES = 400
GRID_SIZE_RANGE = (18, 30)
RULER_COLOR_COUNTS = (2, 3)
SEGMENT_MIN = 3
RECT_SIZE_RANGE = (1, 4)
PLACE_RETRIES_PER_RECT = 500


def _partition(total: int, parts: int, minimum: int) -> list[int] | None:
    extra = total - parts * minimum
    if extra < 0:
        return None
    vals = [minimum] * parts
    for _ in range(extra):
        vals[random.randrange(parts)] += 1
    random.shuffle(vals)
    return vals


def _intervals(lengths: list[int]) -> list[tuple[int, int]]:
    intervals = []
    pos = 1
    for length in lengths:
        intervals.append((pos, pos + length - 1))
        pos += length
    return intervals


def _paint_rulers(
    grid: np.ndarray,
    top_order: list[int],
    top_lengths: list[int],
    left_order: list[int],
    left_lengths: list[int],
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    col_spans: dict[int, tuple[int, int]] = {}
    row_spans: dict[int, tuple[int, int]] = {}

    for color, (c0, c1) in zip(top_order, _intervals(top_lengths)):
        grid[0, c0 : c1 + 1] = color
        col_spans[color] = (c0, c1)

    for color, (r0, r1) in zip(left_order, _intervals(left_lengths)):
        grid[r0 : r1 + 1, 0] = color
        row_spans[color] = (r0, r1)

    return row_spans, col_spans


def _sample_table_values(size: int, bg: int) -> list[int]:
    colors = [c for c in COLOR_IDS if c != bg]
    needed_distinct = min(3, size * size)
    for _ in range(100):
        vals = [random.choice(colors) for _ in range(size * size)]
        if len(set(vals)) >= needed_distinct:
            return vals
    return (colors * (size * size))[: size * size]


def _draw_legend(
    grid: np.ndarray,
    colors: list[int],
    bg: int,
) -> tuple[int, int, dict[tuple[int, int], int]]:
    k = len(colors)
    r0 = grid.shape[0] - k - 1
    c0 = grid.shape[1] - k - 1
    row_headers = random.sample(colors, k)
    col_headers = random.sample(colors, k)
    values = _sample_table_values(k, bg)

    grid[r0, c0] = bg
    for j, color in enumerate(col_headers, start=1):
        grid[r0, c0 + j] = color
    for i, color in enumerate(row_headers, start=1):
        grid[r0 + i, c0] = color

    table: dict[tuple[int, int], int] = {}
    n = 0
    for i, row_color in enumerate(row_headers, start=1):
        for j, col_color in enumerate(col_headers, start=1):
            result = values[n]
            n += 1
            grid[r0 + i, c0 + j] = result
            table[(row_color, col_color)] = result
    return r0, c0, table


def _expanded_clear(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = occupied.shape
    er0 = max(0, r0 - 1)
    ec0 = max(0, c0 - 1)
    er1 = min(h - 1, r1 + 1)
    ec1 = min(w - 1, c1 + 1)
    return not bool(occupied[er0 : er1 + 1, ec0 : ec1 + 1].any())


def _place_rectangle(
    grid: np.ndarray,
    occupied: np.ndarray,
    row_span: tuple[int, int],
    col_span: tuple[int, int],
    result_color: int,
    bg: int,
) -> bool:
    h, w = grid.shape
    source_choices = [c for c in COLOR_IDS if c != bg and c != result_color]

    for _ in range(PLACE_RETRIES_PER_RECT):
        rh = random.randint(*RECT_SIZE_RANGE)
        rw = random.randint(*RECT_SIZE_RANGE)
        if rh * rw == 1:
            continue
        center_r = random.randint(row_span[0], row_span[1])
        center_c = random.randint(col_span[0], col_span[1])
        r0 = center_r - (rh - 1) // 2
        c0 = center_c - (rw - 1) // 2
        r1 = r0 + rh - 1
        c1 = c0 + rw - 1
        if not (1 <= r0 <= r1 < h and 1 <= c0 <= c1 < w):
            continue
        if not _expanded_clear(occupied, r0, c0, r1, c1):
            continue

        color = random.choice(source_choices)
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(COLOR_IDS)
    k = random.choice(RULER_COLOR_COUNTS)
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    if h - 1 < k * SEGMENT_MIN or w - 1 < k * SEGMENT_MIN:
        return None

    colors = random.sample([c for c in COLOR_IDS if c != bg], k)
    top_order = random.sample(colors, k)
    left_order = random.sample(colors, k)
    top_lengths = _partition(w - 1, k, SEGMENT_MIN)
    left_lengths = _partition(h - 1, k, SEGMENT_MIN)
    if top_lengths is None or left_lengths is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    row_spans, col_spans = _paint_rulers(grid, top_order, top_lengths, left_order, left_lengths)
    legend_r0, legend_c0, table = _draw_legend(grid, colors, bg)

    occupied = np.zeros((h, w), dtype=bool)
    occupied[0, :] = True
    occupied[:, 0] = True
    occupied[max(1, legend_r0 - 1) :, max(1, legend_c0 - 1) :] = True

    all_pairs = [(rcol, ccol) for rcol in colors for ccol in colors]
    random.shuffle(all_pairs)
    object_count = random.randint(k + 2, min(k * k, 8))
    pairs = all_pairs[:object_count]

    if k == 3:
        rows_used = {r for r, _ in pairs}
        cols_used = {c for _, c in pairs}
        for color in colors:
            if color not in rows_used:
                pairs[-1] = (color, pairs[-1][1])
                rows_used.add(color)
            if color not in cols_used:
                pairs[-1] = (pairs[-1][0], color)
                cols_used.add(color)

    for row_color, col_color in pairs:
        result_color = table[(row_color, col_color)]
        if not _place_rectangle(grid, occupied, row_spans[row_color], col_spans[col_color], result_color, bg):
            return None

    if _background(grid) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mixing-table ruler input")
