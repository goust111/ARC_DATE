from __future__ import annotations

import random

import numpy as np


BG = 0
START = 5
BLOCK_COLORS = (1, 2, 3, 4, 6)

GRID_H_RANGE = (9, 13)
GRID_W_RANGE = (11, 15)
RECT_COUNT_CHOICES = (4, 5)
RECT_H_RANGE = (2, 4)
RECT_W_RANGE = (2, 5)
MAX_ATTEMPTS = 400


Rect = tuple[int, int, int, int, int]  # (color, r0, c0, h, w)


def _snake_positions(shape: tuple[int, int]):
    h, w = shape
    for r in range(h):
        cols = range(w) if r % 2 == 0 else range(w - 1, -1, -1)
        for c in cols:
            yield r, c


def _snake_index(r: int, c: int, width: int) -> int:
    return r * width + (c if r % 2 == 0 else width - 1 - c)


def _histogram_from_scan(grid: np.ndarray) -> tuple[list[int], dict[int, int]]:
    seen: set[int] = set()
    order: list[int] = []
    counts: dict[int, int] = {}
    for r, c in _snake_positions(grid.shape):
        color = int(grid[r, c])
        if color in (BG, START):
            continue
        counts[color] = counts.get(color, 0) + 1
        if color not in seen:
            seen.add(color)
            order.append(color)
    return order, counts


def _bars_from_order(order: list[int], counts: dict[int, int]) -> np.ndarray:
    assert order, "expected at least one colored block on the path"
    out = np.zeros((max(counts.values()), len(order)), dtype=int)
    for col, color in enumerate(order):
        out[-counts[color] :, col] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Turn the snake-scan order of colored blocks into a bottom-aligned histogram.

    Input structure:
    - A black grid contains one grey start marker at the top-left corner.
    - Several other colors form separated foreground blocks on the background.
    - The intended scan is a full-row snake path: row 0 left-to-right, row 1
      right-to-left, then alternating until the last row.

    Rule:
    - Start from the grey marker and follow the snake path through the whole grid.
    - Ignore black cells and the grey start marker; record every other visited color.
    - For each color, count how many path cells of that color are visited in total.
    - Order the output columns by each color's first appearance along the snake path.
    - Build a histogram whose width is the number of encountered colors and whose
      height is the largest count; each column is bottom-aligned and filled with
      its color for exactly that count.

    Key insight:
    - The snake traversal is needed twice: it decides column order by first encounter,
      and it also supplies the per-color path lengths that become bar heights.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] > 0 and g.shape[1] > 0, "expected a non-empty grid"
    assert int(g[0, 0]) == START, "expected the grey start marker at (0, 0)"
    assert int(np.count_nonzero(g == START)) == 1, "expected exactly one grey start marker"

    order, counts = _histogram_from_scan(g)
    assert counts, "expected at least one colored block"
    return _bars_from_order(order, counts)


def _gap_ok(rects: list[Rect], r0: int, c0: int, h: int, w: int) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for _color, pr0, pc0, ph, pw in rects:
        pr1 = pr0 + ph - 1
        pc1 = pc0 + pw - 1
        if not (r1 + 1 < pr0 or pr1 + 1 < r0 or c1 + 1 < pc0 or pc1 + 1 < c0):
            return False
    return True


def _place_random_rect(
    color: int,
    rects: list[Rect],
    height: int,
    width: int,
    top_rows: range,
) -> Rect | None:
    top_options = list(top_rows)
    random.shuffle(top_options)
    for r0 in top_options:
        max_h = min(RECT_H_RANGE[1], height - r0)
        if max_h < RECT_H_RANGE[0]:
            continue
        h = random.randint(RECT_H_RANGE[0], max_h)
        max_w = min(RECT_W_RANGE[1], width)
        if max_w < RECT_W_RANGE[0]:
            continue
        w = random.randint(RECT_W_RANGE[0], max_w)
        c_min = 1 if r0 == 0 else 0
        c_max = width - w
        if c_min > c_max:
            continue
        cols = list(range(c_min, c_max + 1))
        random.shuffle(cols)
        for c0 in cols:
            if _gap_ok(rects, r0, c0, h, w):
                return (color, r0, c0, h, w)
    return None


def _place_reversal_pair(
    colors: tuple[int, int],
    rects: list[Rect],
    height: int,
    width: int,
    reversal_row: int,
) -> tuple[Rect, Rect] | None:
    for _ in range(120):
        h1 = random.randint(2, min(RECT_H_RANGE[1], height - reversal_row))
        h2 = random.randint(2, min(RECT_H_RANGE[1], height - reversal_row))
        w1 = random.randint(2, min(4, width - 4))
        w2 = random.randint(2, min(4, width - 4))
        min_gap = 2
        if w1 + w2 + min_gap > width:
            continue
        left_max = width - (w1 + w2 + min_gap)
        c1 = random.randint(0, left_max)
        c2 = random.randint(c1 + w1 + min_gap, width - w2)
        left_rect = (colors[0], reversal_row, c1, h1, w1)
        right_rect = (colors[1], reversal_row, c2, h2, w2)
        if _gap_ok(rects, reversal_row, c1, h1, w1) and _gap_ok(rects, reversal_row, c2, h2, w2):
            return left_rect, right_rect
    return None


def _rect_order_from_geometry(rects: list[Rect], width: int) -> list[int]:
    ranked = []
    for color, r0, c0, h, w in rects:
        first_c = c0 if r0 % 2 == 0 else c0 + w - 1
        ranked.append((_snake_index(r0, first_c, width), color))
    ranked.sort()
    return [color for _, color in ranked]


def _expected_from_rects(rects: list[Rect], width: int) -> tuple[np.ndarray, list[int], dict[int, int]]:
    order = _rect_order_from_geometry(rects, width)
    counts = {color: h * w for color, _r0, _c0, h, w in rects}
    return _bars_from_order(order, counts), order, counts


def _render_grid(height: int, width: int, rects: list[Rect]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[0, 0] = START
    for color, r0, c0, h, w in rects:
        grid[r0 : r0 + h, c0 : c0 + w] = color
    return grid


def _solid_rectangle(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))


def _validate_candidate(
    grid: np.ndarray,
    rects: list[Rect],
    reversal_colors: tuple[int, int],
) -> bool:
    if int(np.count_nonzero(grid == START)) != 1 or int(grid[0, 0]) != START:
        return False

    expected, geom_order, counts = _expected_from_rects(rects, grid.shape[1])
    scan_order, scan_counts = _histogram_from_scan(grid)
    if geom_order != scan_order or counts != scan_counts:
        return False

    colors = {color for color, *_rest in rects}
    if set(map(int, np.unique(grid))) != colors | {BG, START}:
        return False

    for color in colors:
        if not _solid_rectangle(grid == color):
            return False

    row_major = [color for color, *_rest in sorted(rects, key=lambda item: (item[1], item[2]))]
    if geom_order == row_major:
        return False
    if geom_order == sorted(geom_order) or geom_order == sorted(geom_order, reverse=True):
        return False

    heights_in_order = [counts[color] for color in geom_order]
    if heights_in_order == sorted(heights_in_order) or heights_in_order == sorted(heights_in_order, reverse=True):
        return False
    if len(set(heights_in_order)) < 3:
        return False

    left_to_right_reversal = [
        color
        for color, _r0, c0, _h, _w in sorted(
            [rect for rect in rects if rect[0] in reversal_colors], key=lambda item: item[2]
        )
    ]
    path_reversal = [color for color in geom_order if color in reversal_colors]
    if path_reversal != left_to_right_reversal[::-1]:
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    n_rects = random.choice(RECT_COUNT_CHOICES)
    colors = random.sample(BLOCK_COLORS, n_rects)

    odd_rows = [r for r in range(3, height - 3) if r % 2 == 1]
    if not odd_rows:
        return None
    reversal_row = random.choice(odd_rows)

    rects: list[Rect] = [(START, 0, 0, 1, 1)]
    placed: list[Rect] = []

    reversal_pair = _place_reversal_pair((colors[0], colors[1]), rects, height, width, reversal_row)
    if reversal_pair is None:
        return None
    for rect in reversal_pair:
        placed.append(rect)
        rects.append(rect)

    above = _place_random_rect(colors[2], rects, height, width, range(0, reversal_row - 1))
    below = _place_random_rect(colors[3], rects, height, width, range(reversal_row + 1, height - 1))
    if above is None or below is None:
        return None
    for rect in (above, below):
        placed.append(rect)
        rects.append(rect)

    if n_rects == 5:
        extra_bands = [range(0, reversal_row - 1), range(reversal_row + 1, height - 1)]
        random.shuffle(extra_bands)
        extra = None
        for band in extra_bands:
            extra = _place_random_rect(colors[4], rects, height, width, band)
            if extra is not None:
                break
        if extra is None:
            return None
        placed.append(extra)
        rects.append(extra)

    grid = _render_grid(height, width, placed)
    if _validate_candidate(grid, placed, (colors[0], colors[1])):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
