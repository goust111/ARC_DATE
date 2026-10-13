from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
DIVIDER = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_LIMIT = 30
MIN_REGION_SIZE = 3
MAX_REGION_SIZE = 8
MAX_ATTEMPTS = 300
LAYOUT_MODES = ("rows", "cols", "both")


def _divider_color(grid: np.ndarray) -> int:
    candidates: list[tuple[int, int]] = []
    for color in map(int, np.unique(grid)):
        row_lines = np.all(grid == color, axis=1)
        col_lines = np.all(grid == color, axis=0)
        if not row_lines.any() and not col_lines.any():
            continue
        mask = grid == color
        coverage = row_lines[:, None] | col_lines[None, :]
        if np.array_equal(mask, coverage):
            candidates.append((int(mask.sum()), color))
    assert candidates, "expected a structural divider color"
    return max(candidates)[1]


def _background_color(grid: np.ndarray, divider: int) -> int:
    values = grid[grid != divider]
    assert values.size > 0, "expected non-divider cells"
    return Counter(int(value) for value in values).most_common(1)[0][0]


def _spans(line_mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for index, is_line in enumerate(line_mask.tolist()):
        if is_line:
            if start < index:
                spans.append((start, index))
            start = index + 1
    if start < len(line_mask):
        spans.append((start, len(line_mask)))
    assert spans, "expected at least one region span"
    return spans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each rectangular pane from its local color vote, using the minority color as an inner border.

    Input structure:
    - Black background fills the empty cells inside rectangular panes.
    - One gray color forms single-cell full rows and/or full columns that divide the grid into panes.
    - Each pane contains sparse vote pixels of exactly two non-background, non-divider colors, and those two counts are never tied.

    Rule:
    - Detect the gray divider rows and columns, then treat every gap between them as one pane.
    - In each pane, count the two vote colors and choose the more frequent one as the majority fill color.
    - Repaint the pane's entire interior with the majority color, but overwrite the pane's inside border ring with the minority color.
    - Keep the gray divider lines unchanged.

    Key insight:
    The scattered colored pixels are only local votes: the less frequent color is still important because it becomes the salient border while the majority color fills the rest.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider = _divider_color(g)
    bg = _background_color(g, divider)
    row_spans = _spans(np.all(g == divider, axis=1))
    col_spans = _spans(np.all(g == divider, axis=0))

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            patch = g[r0:r1, c0:c1]
            counts = Counter(
                int(value)
                for value in patch.ravel()
                if int(value) not in (bg, divider)
            )
            assert len(counts) == 2, "each pane must contain exactly two vote colors"
            ranked = counts.most_common(2)
            assert ranked[0][1] != ranked[1][1], "vote colors must have a unique majority"
            majority = ranked[0][0]
            minority = ranked[1][0]

            out[r0:r1, c0:c1] = majority
            out[r0, c0:c1] = minority
            out[r1 - 1, c0:c1] = minority
            out[r0:r1, c0] = minority
            out[r0:r1, c1 - 1] = minority

    return out


def _sample_region_sizes(
    rng: random.Random,
    count: int,
    min_size: int,
    max_size: int,
    limit: int,
) -> list[int] | None:
    min_total = count * min_size + (count - 1)
    max_total = min(limit, count * max_size + (count - 1))
    if min_total > max_total:
        return None

    total = rng.randint(min_total, max_total)
    remaining = total - (count - 1) - count * min_size
    sizes = [min_size] * count

    while remaining > 0:
        idx = rng.randrange(count)
        if sizes[idx] < max_size:
            sizes[idx] += 1
            remaining -= 1

    rng.shuffle(sizes)
    return sizes


def _build_regions(sizes: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for index, size in enumerate(sizes):
        spans.append((cursor, cursor + size))
        cursor += size
        if index < len(sizes) - 1:
            cursor += 1
    return spans


def _place_votes(
    rng: random.Random,
    grid: np.ndarray,
    row_span: tuple[int, int],
    col_span: tuple[int, int],
) -> None:
    r0, r1 = row_span
    c0, c1 = col_span
    area = (r1 - r0) * (c1 - c0)
    max_votes = min(8, max(3, area // 3 + 1))
    total_votes = rng.randint(3, max_votes)
    minority_count = rng.randint(1, (total_votes - 1) // 2)
    majority_count = total_votes - minority_count

    colors = rng.sample(PALETTE, 2)
    if rng.random() < 0.5:
        majority_color, minority_color = colors
    else:
        minority_color, majority_color = colors

    cells = [(r, c) for r in range(r0, r1) for c in range(c0, c1)]
    chosen = rng.sample(cells, total_votes)
    rng.shuffle(chosen)

    for r, c in chosen[:majority_count]:
        grid[r, c] = majority_color
    for r, c in chosen[majority_count:]:
        grid[r, c] = minority_color


def _layout_counts(rng: random.Random) -> tuple[int, int]:
    mode = rng.choice(LAYOUT_MODES)
    if mode == "rows":
        return rng.randint(2, 4), 1
    if mode == "cols":
        return 1, rng.randint(2, 4)
    return rng.randint(2, 4), rng.randint(2, 4)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_count, col_count = _layout_counts(rng)
    row_sizes = _sample_region_sizes(
        rng,
        count=row_count,
        min_size=MIN_REGION_SIZE,
        max_size=MAX_REGION_SIZE,
        limit=GRID_LIMIT,
    )
    col_sizes = _sample_region_sizes(
        rng,
        count=col_count,
        min_size=MIN_REGION_SIZE,
        max_size=MAX_REGION_SIZE,
        limit=GRID_LIMIT,
    )
    if row_sizes is None or col_sizes is None:
        return None

    height = sum(row_sizes) + (row_count - 1)
    width = sum(col_sizes) + (col_count - 1)
    if height > GRID_LIMIT or width > GRID_LIMIT:
        return None

    grid = np.full((height, width), BG, dtype=int)
    row_spans = _build_regions(row_sizes)
    col_spans = _build_regions(col_sizes)

    divider_rows = np.cumsum(row_sizes[:-1]) + np.arange(row_count - 1)
    divider_cols = np.cumsum(col_sizes[:-1]) + np.arange(col_count - 1)
    if divider_rows.size:
        grid[divider_rows, :] = DIVIDER
    if divider_cols.size:
        grid[:, divider_cols] = DIVIDER

    for row_span in row_spans:
        for col_span in col_spans:
            _place_votes(rng, grid, row_span, col_span)

    if _is_valid_input(grid):
        return grid
    return None


def _is_valid_input(grid: np.ndarray) -> bool:
    try:
        divider = _divider_color(grid)
        bg = _background_color(grid, divider)
        row_spans = _spans(np.all(grid == divider, axis=1))
        col_spans = _spans(np.all(grid == divider, axis=0))
    except AssertionError:
        return False

    if len(row_spans) * len(col_spans) < 2:
        return False

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            if min(r1 - r0, c1 - c0) < MIN_REGION_SIZE:
                return False
            patch = grid[r0:r1, c0:c1]
            counts = Counter(
                int(value)
                for value in patch.ravel()
                if int(value) not in (bg, divider)
            )
            if len(counts) != 2:
                return False
            ranked = counts.most_common(2)
            if ranked[0][1] == ranked[1][1]:
                return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
