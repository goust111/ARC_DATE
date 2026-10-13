from __future__ import annotations

import itertools
import random

import numpy as np


OUTPUT_CYCLE = (2, 8, 5)
INPUT_COLOR_POOL = (0, 1, 3, 4, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 400


def _colors_present(grid: np.ndarray) -> list[int]:
    return sorted(set(int(v) for v in np.asarray(grid).ravel()))


def _candidate_products(
    grid: np.ndarray,
) -> list[tuple[tuple[int, int], np.ndarray, np.ndarray, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    candidates = []
    for pair in itertools.combinations(_colors_present(g), 2):
        mask = np.isin(g, pair)
        rows = mask.any(axis=1)
        cols = mask.any(axis=0)
        if not rows.any() or not cols.any():
            continue
        if int(mask.sum()) != int(rows.sum()) * int(cols.sum()):
            continue
        subgrid = g[np.ix_(rows, cols)]
        if set(map(int, subgrid.ravel())) != set(pair):
            continue
        candidates.append((pair, rows, cols, subgrid))
    return candidates


def _extract_hidden_subgrid(grid: np.ndarray) -> np.ndarray:
    candidates = _candidate_products(grid)
    assert len(candidates) == 1, "expected exactly one two-color row/column product"
    return candidates[0][3].copy()


def _recolor_subgrid(subgrid: np.ndarray) -> np.ndarray:
    g = np.asarray(subgrid, dtype=int)
    colors = _colors_present(g)
    assert len(colors) == 2, "extracted subgrid must be exactly two colors"

    counts = {color: int(np.count_nonzero(g == color)) for color in colors}
    items = sorted(counts.items(), key=lambda item: item[1])
    assert items[0][1] != items[1][1], "foreground and background counts must differ"
    foreground = items[0][0]

    coords = [(int(r), int(c)) for r, c in np.argwhere(g == foreground)]
    assert len(coords) >= 4, "foreground must contain at least four pixels"

    if g.shape[0] >= g.shape[1]:
        ordered = sorted(coords)
    else:
        ordered = sorted(coords, key=lambda rc: (rc[1], rc[0]))

    out = g.copy()
    for i, (r, c) in enumerate(ordered):
        out[r, c] = OUTPUT_CYCLE[i % len(OUTPUT_CYCLE)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the hidden two-color row/column product image, then recolor its minority pixels in scan order chosen by the extracted shape.

    Input structure:
    - A small noisy grid contains 4-5 colors overall.
    - Exactly one color pair occupies every cell of the cross product of its active rows and active columns.
    - All remaining colors are filler outside that cross product.

    Rule:
    - Find the unique color pair whose union forms a complete row/column product.
    - Keep only those active rows and columns to obtain the compact two-color subgrid.
    - In that subgrid, treat the less frequent color as foreground and the more frequent color as background.
    - If the extracted subgrid is at least as tall as wide, order foreground pixels row-major; otherwise order them column-major.
    - Recolor the ordered foreground pixels cyclically with red(2), teal(8), grey(5), leaving the background color unchanged.

    Key insight:
    - The extraction result is not only the hidden content; its aspect ratio determines the scan direction that fixes the recoloring order.
    """
    subgrid = _extract_hidden_subgrid(np.asarray(grid, dtype=int))
    return _recolor_subgrid(subgrid)


def _choose_subgrid_shape() -> tuple[int, int]:
    tall_shapes = [(4, 3), (5, 3), (5, 4), (6, 3), (6, 4), (6, 5)]
    wide_shapes = [(3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)]
    return random.choice(tall_shapes if random.random() < 0.5 else wide_shapes)


def _choose_active_indices(count: int) -> tuple[list[int], int]:
    extra = random.randint(4, 6)
    total = count + extra
    gaps = [0] * (count + 1)
    for _ in range(extra):
        gaps[random.randrange(count + 1)] += 1

    indices = []
    cur = gaps[0]
    for i in range(count):
        indices.append(cur)
        cur += 1 + gaps[i + 1]
    assert len(indices) == count and indices[-1] < total
    return indices, total


def _non_adjacent_foreground_positions(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(120):
        coords = random.sample(cells, count)
        coord_set = set(coords)
        if any((r + 1, c) in coord_set or (r, c + 1) in coord_set for r, c in coords):
            continue
        if len({r for r, _ in coords}) < 2 or len({c for _, c in coords}) < 2:
            continue
        return coords
    return None


def _build_subgrid(height: int, width: int) -> np.ndarray | None:
    area = height * width
    max_fg = min((area - 1) // 2, max(4, int(area * 0.35)))
    if max_fg < 4:
        return None

    fg_count = random.randint(4, max_fg)
    fg_positions = _non_adjacent_foreground_positions(height, width, fg_count)
    if fg_positions is None:
        return None

    pair = random.sample(INPUT_COLOR_POOL, 2)
    foreground = random.choice(pair)
    background = pair[0] if pair[1] == foreground else pair[1]

    subgrid = np.full((height, width), background, dtype=int)
    for r, c in fg_positions:
        subgrid[r, c] = foreground
    return subgrid


def _fill_noise(
    grid: np.ndarray,
    active_rows: list[int],
    active_cols: list[int],
    fillers: list[int],
) -> None:
    h, w = grid.shape
    row_bias = random.randrange(7)
    col_bias = random.randrange(7)
    diag_bias = random.randrange(7)
    active_row_set = set(active_rows)
    active_col_set = set(active_cols)

    for r in range(h):
        for c in range(w):
            if r in active_row_set and c in active_col_set:
                continue
            if r not in active_row_set and c not in active_col_set:
                idx = (r + c + diag_bias) % len(fillers)
            elif r not in active_row_set:
                idx = (r + row_bias) % len(fillers)
            else:
                idx = (c + col_bias) % len(fillers)
            grid[r, c] = fillers[idx]

    outside = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if r not in active_row_set or c not in active_col_set
    ]
    random.shuffle(outside)
    for color, cell in zip(fillers, outside):
        grid[cell] = color


def _generate_attempt() -> np.ndarray | None:
    sub_h, sub_w = _choose_subgrid_shape()
    subgrid = _build_subgrid(sub_h, sub_w)
    if subgrid is None:
        return None

    active_rows, grid_h = _choose_active_indices(sub_h)
    active_cols, grid_w = _choose_active_indices(sub_w)

    filler_count = random.choice((2, 3))
    used = set(_colors_present(subgrid))
    filler_pool = [color for color in INPUT_COLOR_POOL if color not in used]
    fillers = random.sample(filler_pool, filler_count)

    grid = np.empty((grid_h, grid_w), dtype=int)
    _fill_noise(grid, active_rows, active_cols, fillers)
    grid[np.ix_(active_rows, active_cols)] = subgrid

    if len(_colors_present(grid)) != 2 + filler_count:
        return None
    if max(grid.shape) > 30:
        return None

    candidates = _candidate_products(grid)
    if len(candidates) != 1:
        return None

    expected = _recolor_subgrid(subgrid)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
