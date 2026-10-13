from __future__ import annotations

from collections import Counter

import numpy as np


BLACK = 0
MAROON = 9
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8)
GEN_RETRIES = 400
SEGMENT_COUNT_RANGE = (2, 5)
SEGMENT_WIDTH_RANGE = (4, 8)
HEIGHT_RANGE = (6, 18)


def _segment_bounds(width: int, separator_cols: list[int]) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start = 0
    for col in separator_cols:
        if start < col:
            bounds.append((start, col))
        start = col + 1
    if start < width:
        bounds.append((start, width))
    return bounds


def _sample_counts(
    rng: np.random.Generator, color_count: int, occupied: int
) -> np.ndarray | None:
    for _ in range(100):
        counts = np.ones(color_count, dtype=int)
        for _ in range(occupied - color_count):
            counts[int(rng.integers(0, color_count))] += 1
        max_count = int(counts.max())
        if int(np.sum(counts == max_count)) == 1:
            return counts
    return None


def _sample_row_votes(
    rng: np.random.Generator, width: int, palette: list[int]
) -> tuple[np.ndarray, int, bool] | None:
    max_colors = min(3, len(palette), width - 1)
    if max_colors <= 0:
        return None

    for _ in range(200):
        color_count = int(rng.integers(1, max_colors + 1))
        chosen = [int(v) for v in rng.choice(palette, size=color_count, replace=False)]
        occupied_upper = min(width - 1, max(color_count, (width // 2) + 1))
        occupied = int(rng.integers(color_count, occupied_upper + 1))
        counts = _sample_counts(rng, color_count, occupied)
        if counts is None:
            continue

        row = np.full(width, BLACK, dtype=int)
        positions = [int(v) for v in rng.choice(width, size=occupied, replace=False)]
        values: list[int] = []
        for color, count in zip(chosen, counts):
            values.extend([int(color)] * int(count))
        rng.shuffle(values)
        row[positions] = values
        winner = int(chosen[int(np.argmax(counts))])
        return row, winner, color_count > 1

    return None


def _bounded_composition(
    rng: np.random.Generator, total: int, parts: int, lo: int, hi: int
) -> list[int] | None:
    if parts * lo > total or parts * hi < total:
        return None

    values = [lo] * parts
    remaining = total - parts * lo
    for _ in range(remaining):
        candidates = [idx for idx, value in enumerate(values) if value < hi]
        if not candidates:
            return None
        pick = int(rng.choice(candidates))
        values[pick] += 1
    rng.shuffle(values)
    return [int(v) for v in values]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each separator-bounded row segment with that segment row's majority vote color.

    Input structure:
    - The background is black.
    - One or more full-height maroon columns act as vertical separators.
    - Between separators, each row contains scattered single-cell vote pixels in non-black, non-maroon colors.

    Rule:
    - Split the grid into vertical segments at the maroon separator columns.
    - For every row inside each segment, count the non-black, non-maroon colors already present there.
    - Find the unique most frequent color for that row segment and recolor every black cell in that row segment to that winner.
    - Keep the maroon separator columns and all original colored vote pixels unchanged.

    Key insight:
    - The maroon columns are independent voting boundaries, so the same row can have different winning colors in different vertical segments.
    """
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)

    separator_cols = [int(c) for c in range(width) if bool(np.all(g[:, c] == MAROON))]
    segments = _segment_bounds(width, separator_cols)
    if not segments:
        segments = [(0, width)]

    out = g.copy()
    for r in range(height):
        for c0, c1 in segments:
            row = g[r, c0:c1]
            votes = row[(row != BLACK) & (row != MAROON)]
            assert votes.size > 0, "each row segment must contain at least one vote pixel"

            counts = Counter(map(int, votes))
            max_count = max(counts.values())
            winners = [color for color, count in counts.items() if count == max_count]
            assert len(winners) == 1, "each row segment must have a unique winning color"

            filled = row.copy()
            filled[row == BLACK] = int(winners[0])
            out[r, c0:c1] = filled

    return out


def _validate_grid(grid: np.ndarray) -> bool:
    height, width = map(int, grid.shape)
    if height > 30 or width > 30:
        return False

    separator_cols = [int(c) for c in range(width) if bool(np.all(grid[:, c] == MAROON))]
    if len(separator_cols) < 1:
        return False

    segments = _segment_bounds(width, separator_cols)
    if len(segments) < 2:
        return False

    saw_multicolor_row = False
    saw_row_disagreement = False
    separator_set = set(separator_cols)
    for r in range(height):
        winners: list[int] = []
        for c0, c1 in segments:
            row = grid[r, c0:c1]
            if not np.any(row == BLACK):
                return False
            votes = row[(row != BLACK) & (row != MAROON)]
            if votes.size == 0:
                return False
            counts = Counter(map(int, votes))
            max_count = max(counts.values())
            if sum(count == max_count for count in counts.values()) != 1:
                return False
            if len(counts) > 1:
                saw_multicolor_row = True
            winners.append(next(color for color, count in counts.items() if count == max_count))

        if len(set(winners)) > 1:
            saw_row_disagreement = True

    if not saw_multicolor_row or not saw_row_disagreement:
        return False

    for c in range(width):
        if c in separator_set:
            continue
        col = grid[:, c]
        if int(col[0]) != BLACK and bool(np.all(col == col[0])):
            return False

    out = main(grid)
    return not np.array_equal(out, grid)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    segment_count = int(rng.integers(SEGMENT_COUNT_RANGE[0], SEGMENT_COUNT_RANGE[1] + 1))
    separator_count = segment_count - 1
    min_total_width = segment_count * SEGMENT_WIDTH_RANGE[0] + separator_count
    max_total_width = min(30, segment_count * SEGMENT_WIDTH_RANGE[1] + separator_count)
    total_width = int(rng.integers(min_total_width, max_total_width + 1))

    widths = _bounded_composition(
        rng,
        total_width - separator_count,
        segment_count,
        SEGMENT_WIDTH_RANGE[0],
        SEGMENT_WIDTH_RANGE[1],
    )
    if widths is None:
        return None

    height = int(rng.integers(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1))
    grid = np.full((height, total_width), BLACK, dtype=int)

    segments: list[tuple[int, int]] = []
    palettes: list[list[int]] = []
    col = 0
    for idx, seg_width in enumerate(widths):
        segments.append((col, col + seg_width))
        palette_size = int(rng.integers(2, min(4, len(COLOR_POOL)) + 1))
        palettes.append(
            [int(v) for v in rng.choice(COLOR_POOL, size=palette_size, replace=False)]
        )
        col += seg_width
        if idx != segment_count - 1:
            grid[:, col] = MAROON
            col += 1

    for r in range(height):
        for (c0, c1), palette in zip(segments, palettes):
            sampled = _sample_row_votes(rng, c1 - c0, palette)
            if sampled is None:
                return None
            row, _winner, _is_multicolor = sampled
            grid[r, c0:c1] = row

    if not _validate_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
