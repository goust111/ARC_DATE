from __future__ import annotations

import random

import numpy as np


RED = 2


def _infer_row_colors(grid: np.ndarray) -> list[int]:
    """Recover the underlying band color for each row, ignoring red seeds."""
    row_colors: list[int | None] = []
    for row in grid:
        colors = np.unique(row[row != RED])
        if len(colors) == 1:
            row_colors.append(int(colors[0]))
        elif len(colors) == 0:
            row_colors.append(None)
        else:
            raise AssertionError("each row must belong to exactly one non-red band color")

    while any(color is None for color in row_colors):
        changed = False
        for r, color in enumerate(row_colors):
            if color is not None:
                continue
            neighbors: list[int] = []
            if r > 0 and row_colors[r - 1] is not None:
                neighbors.append(int(row_colors[r - 1]))
            if r + 1 < len(row_colors) and row_colors[r + 1] is not None:
                neighbors.append(int(row_colors[r + 1]))
            if neighbors and len(set(neighbors)) == 1:
                row_colors[r] = neighbors[0]
                changed = True
        if not changed:
            break

    assert all(color is not None for color in row_colors), "failed to infer all band colors"
    return [int(color) for color in row_colors]


def _band_spans(grid: np.ndarray) -> list[tuple[int, int]]:
    row_colors = _infer_row_colors(grid)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(row_colors):
        stop = start + 1
        while stop < len(row_colors) and row_colors[stop] == row_colors[start]:
            stop += 1
        spans.append((start, stop))
        start = stop
    return spans


def _paint_growth(out: np.ndarray, row: int, col: int, length: int, step: int) -> None:
    width = out.shape[1]
    for offset in range(length):
        cc = col + step * offset
        if not (0 <= cc < width):
            break
        out[row, cc] = RED


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each red seed horizontally by its band's height.

    Odd-indexed horizontal bands grow right; even-indexed bands grow left.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    height, width = g.shape
    assert height > 0 and width > 0, "grid must be non-empty"

    bands = _band_spans(g)
    out = g.copy()

    for band_idx, (r0, r1) in enumerate(bands, start=1):
        band_height = r1 - r0
        step = 1 if band_idx % 2 == 1 else -1
        for r in range(r0, r1):
            for c in np.flatnonzero(g[r] == RED):
                _paint_growth(out, r, int(c), band_height, step)

    return out


# --------------- generator ---------------

PALETTE = (0, 1, 3, 4, 5, 6, 7, 8, 9)
BAND_COUNT_RANGE = (3, 7)
BAND_HEIGHT_RANGE = (1, 4)
TOTAL_HEIGHT_RANGE = (8, 18)
WIDTH_RANGE = (8, 18)
EXTRA_SEEDED_BANDS_MAX = 3
MULTI_ROW_SEED_PROB = 0.55
MULTI_SEED_ROW_PROB = 0.35
MAX_GENERATION_ATTEMPTS = 300


def _sample_band_heights() -> list[int]:
    for _ in range(100):
        band_count = random.randint(*BAND_COUNT_RANGE)
        heights = [random.randint(*BAND_HEIGHT_RANGE) for _ in range(band_count)]
        total_height = sum(heights)
        if TOTAL_HEIGHT_RANGE[0] <= total_height <= TOTAL_HEIGHT_RANGE[1] and len(set(heights)) >= 2:
            return heights
    raise RuntimeError("failed to sample band heights")


def _interior_columns(width: int, band_height: int, step: int) -> list[int]:
    if step == 1:
        return list(range(0, width - band_height + 1))
    return list(range(band_height - 1, width))


def _edge_clip_columns(width: int, band_height: int, step: int) -> list[int]:
    if step == 1:
        cols = list(range(max(0, width - band_height + 1), width))
        return cols or [width - 1]
    cols = list(range(0, min(width, band_height)))
    return cols or [0]


def _sample_seed_columns(width: int, band_height: int, step: int, force_edge_clip: bool) -> list[int]:
    cols: set[int] = set()
    if force_edge_clip:
        cols.add(width - 1 if step == 1 else 0)

    target = 1 + int(random.random() < MULTI_SEED_ROW_PROB)
    while len(cols) < target:
        roll = random.random()
        if roll < 0.45:
            choices = _interior_columns(width, band_height, step)
        elif roll < 0.8:
            choices = _edge_clip_columns(width, band_height, step)
        else:
            choices = list(range(width))
        cols.add(random.choice(choices))

    if len(cols) >= width:
        cols.remove(random.choice(sorted(cols)))
    return sorted(cols)


def _choose_seed_bands(heights: list[int]) -> tuple[set[int], int, int]:
    odd_candidates = [i for i in range(len(heights)) if (i + 1) % 2 == 1]
    even_candidates = [i for i in range(len(heights)) if (i + 1) % 2 == 0]
    assert odd_candidates and even_candidates, "generator requires both odd and even bands"

    odd_preferred = [i for i in odd_candidates if heights[i] > 1] or odd_candidates
    even_preferred = [i for i in even_candidates if heights[i] > 1] or even_candidates
    odd_anchor = random.choice(odd_preferred)
    even_anchor = random.choice(even_preferred)

    seeded = {odd_anchor, even_anchor}
    extra_candidates = [i for i in range(len(heights)) if i not in seeded]
    extra_count = random.randint(0, min(EXTRA_SEEDED_BANDS_MAX, len(extra_candidates)))
    if extra_count:
        seeded.update(random.sample(extra_candidates, extra_count))
    return seeded, odd_anchor, even_anchor


def _seed_rows_for_band(row0: int, row1: int) -> list[int]:
    rows = list(range(row0, row1))
    if len(rows) == 1:
        return rows
    count = 1 + int(random.random() < MULTI_ROW_SEED_PROB)
    count = min(count, len(rows))
    return sorted(random.sample(rows, count))


def _build_seed_map(heights: list[int], width: int) -> dict[int, list[int]]:
    seed_map: dict[int, list[int]] = {}
    seeded_bands, odd_anchor, even_anchor = _choose_seed_bands(heights)

    row0 = 0
    for band_idx, band_height in enumerate(heights):
        row1 = row0 + band_height
        if band_idx in seeded_bands:
            rows = _seed_rows_for_band(row0, row1)
            step = 1 if (band_idx + 1) % 2 == 1 else -1
            forced_row = rows[0]
            for r in rows:
                force_edge = (
                    r == forced_row and (band_idx == odd_anchor or band_idx == even_anchor)
                )
                seed_map[r] = _sample_seed_columns(width, band_height, step, force_edge)
        row0 = row1

    return seed_map


def _render_input(heights: list[int], colors: list[int], width: int, seed_map: dict[int, list[int]]) -> np.ndarray:
    total_height = sum(heights)
    grid = np.empty((total_height, width), dtype=int)
    row0 = 0
    for band_height, color in zip(heights, colors):
        row1 = row0 + band_height
        grid[row0:row1, :] = color
        row0 = row1

    for r, cols in seed_map.items():
        grid[r, cols] = RED

    assert not np.any(np.all(grid == RED, axis=1)), "each row needs visible band color"
    return grid


def _render_output(
    heights: list[int], colors: list[int], width: int, seed_map: dict[int, list[int]]
) -> np.ndarray:
    total_height = sum(heights)
    out = np.empty((total_height, width), dtype=int)
    row0 = 0
    for band_idx, (band_height, color) in enumerate(zip(heights, colors), start=1):
        row1 = row0 + band_height
        out[row0:row1, :] = color
        step = 1 if band_idx % 2 == 1 else -1
        for r in range(row0, row1):
            for c in seed_map.get(r, []):
                _paint_growth(out, r, c, band_height, step)
        row0 = row1
    return out


def _gen_attempt() -> np.ndarray | None:
    heights = _sample_band_heights()
    width = random.randint(*WIDTH_RANGE)
    colors = random.sample(PALETTE, len(heights))
    seed_map = _build_seed_map(heights, width)

    inp = _render_input(heights, colors, width, seed_map)
    out = _render_output(heights, colors, width, seed_map)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
