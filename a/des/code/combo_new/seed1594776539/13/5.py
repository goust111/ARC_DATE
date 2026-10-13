from __future__ import annotations

import math
import random

import numpy as np


BG = 0
SEPARATOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_HEIGHT_RANGE = (7, 30)
GRID_WIDTH_RANGE = (8, 30)
LEGEND_COUNT_RANGE = (2, 6)

TOTAL_FILL_DIVISOR = 7
MAX_COLOR_PLACEMENT_TRIES = 220
MAX_GENERATION_ATTEMPTS = 400

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = NEIGHBORS4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _legend_rows_and_colors(grid: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    legend_rows = [int(r) for r in np.flatnonzero(grid[:, 0] != bg)]
    assert legend_rows, "expected at least one legend pixel"
    assert all(
        legend_rows[i + 1] - legend_rows[i] == 2 for i in range(len(legend_rows) - 1)
    ), "legend pixels must be separated by one background cell"

    legend_colors = [int(grid[r, 0]) for r in legend_rows]
    assert len(legend_colors) == len(set(legend_colors)), "legend colors must be distinct"

    sep_col = grid[:, 1]
    assert np.all(sep_col == sep_col[0]), "expected a constant separator column"
    assert int(sep_col[0]) != bg, "separator must be non-background"
    return legend_rows, legend_colors


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[1] >= 3, "expected legend, separator, and interior columns"

    bg = _background_color(g)
    legend_rows, legend_colors = _legend_rows_and_colors(g, bg)
    interior = g[:, 2:]

    counts = {color: int(np.count_nonzero(interior == color)) for color in legend_colors}

    out = g.copy()
    out[:, 2:] = bg
    max_bar = out.shape[1] - 2
    for row, color in zip(legend_rows, legend_colors, strict=True):
        count = counts[color]
        assert count <= max_bar, "counted bar exceeds available width"
        if count:
            out[row, 2 : 2 + count] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the left-edge legend, total the interior pixels of each legend color, and redraw them as horizontal bars.

    Input structure:
    - The background is black and the leftmost column contains a vertical legend
      of distinct colored pixels, separated by one black cell between neighbors.
    - Column 1 is a solid gray separator that splits the legend from the interior.
    - The interior region on the right contains scattered 4-connected monochrome
      clusters whose colors all come from the legend.

    Rule:
    - For each legend color, count the total number of interior pixels of that
      color across all of its disconnected clusters.
    - Erase the entire interior back to the black background.
    - On each legend row, draw a horizontal bar of the same color starting
      immediately to the right of the separator, with length equal to the count.
      Keep the legend column and the separator column unchanged.

    Key insight:
    - The edge pixels are a legend that chooses both which colors matter and
      which rows the summarized bars belong to; the bar length is the total pixel
      count per color, not the number of disconnected clusters.
    """
    return _transform(grid)


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _rotate(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(-c, r) for r, c in cells}


def _reflect(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(r, -c) for r, c in cells}


def _random_polyomino(size: int) -> list[tuple[int, int]]:
    assert 1 <= size <= 8
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = {
            (r + dr, c + dc)
            for r, c in cells
            for dr, dc in NEIGHBORS4
            if (r + dr, c + dc) not in cells
        }
        cells.add(random.choice(list(frontier)))

    for _ in range(random.randrange(4)):
        cells = _rotate(cells)
    if random.random() < 0.5:
        cells = _reflect(cells)
    return _normalize(cells)


def _bounded_composition(total: int, parts: int, lower: int, upper: int) -> list[int] | None:
    if parts * lower > total or parts * upper < total:
        return None

    out: list[int] = []
    remaining = total
    for idx in range(parts):
        slots_left = parts - idx - 1
        lo = max(lower, remaining - upper * slots_left)
        hi = min(upper, remaining - lower * slots_left)
        if lo > hi:
            return None
        piece = random.randint(lo, hi)
        out.append(piece)
        remaining -= piece

    random.shuffle(out)
    return out


def _cluster_sizes(total: int) -> list[int]:
    min_parts = max(1, math.ceil(total / 8))
    max_parts = min(total, 5)
    n_parts = random.randint(min_parts, max_parts)
    sizes = _bounded_composition(total, n_parts, 1, 8)
    assert sizes is not None
    return sizes


def _place_cluster(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    size: int,
) -> bool:
    h, w = grid.shape
    for _ in range(MAX_COLOR_PLACEMENT_TRIES):
        shape = _random_polyomino(size)
        shape_h = max(r for r, _c in shape) + 1
        shape_w = max(c for _r, c in shape) + 1
        if shape_h > h or shape_w > w - 2:
            continue

        r0 = random.randint(0, h - shape_h)
        c0 = random.randint(2, w - shape_w)
        coords = [(r0 + r, c0 + c) for r, c in shape]
        if any(blocked[r, c] for r, c in coords):
            continue

        for r, c in coords:
            grid[r, c] = color
        for r, c in coords:
            blocked[r, c] = True
            for dr, dc in NEIGHBORS8:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True
        return True
    return False


def _expected_output(
    grid: np.ndarray,
    legend_rows: list[int],
    legend_colors: list[int],
    totals: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    out[:, 2:] = BG
    for row, color in zip(legend_rows, legend_colors, strict=True):
        count = totals[color]
        if count:
            out[row, 2 : 2 + count] = color
    return out


def _generate_attempt() -> np.ndarray | None:
    n_legend = random.randint(*LEGEND_COUNT_RANGE)
    min_h = max(GRID_HEIGHT_RANGE[0], 2 * n_legend - 1)
    if min_h > GRID_HEIGHT_RANGE[1]:
        return None

    height = random.randint(min_h, GRID_HEIGHT_RANGE[1])
    width = random.randint(*GRID_WIDTH_RANGE)
    bar_limit = width - 2
    if n_legend > len(PALETTE) or bar_limit < 1:
        return None

    max_total = min((height * bar_limit) // TOTAL_FILL_DIVISOR, n_legend * bar_limit)
    active_count = random.randint(1, n_legend)
    if max_total < active_count:
        return None

    grid = np.full((height, width), BG, dtype=int)
    legend_start = random.randint(0, height - (2 * n_legend - 1))
    legend_rows = [legend_start + 2 * i for i in range(n_legend)]
    legend_colors = random.sample(PALETTE, n_legend)

    for row, color in zip(legend_rows, legend_colors, strict=True):
        grid[row, 0] = color
    grid[:, 1] = SEPARATOR

    active_colors = set(random.sample(legend_colors, active_count))
    total_pixels = random.randint(active_count, max_total)
    active_totals = _bounded_composition(total_pixels, active_count, 1, bar_limit)
    if active_totals is None:
        return None

    totals = {color: 0 for color in legend_colors}
    for color, total in zip(active_colors, active_totals, strict=True):
        totals[int(color)] = int(total)

    blocked = np.zeros_like(grid, dtype=bool)
    cluster_specs: list[tuple[int, int]] = []
    for color in legend_colors:
        if totals[color] == 0:
            continue
        for size in _cluster_sizes(totals[color]):
            cluster_specs.append((int(size), int(color)))

    random.shuffle(cluster_specs)
    cluster_specs.sort(reverse=True)
    for size, color in cluster_specs:
        if not _place_cluster(grid, blocked, color, size):
            return None

    expected = _expected_output(grid, legend_rows, legend_colors, totals)
    solved = _transform(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
