import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 9)
MAX_GRID_SIDE = 30
MAX_GENERATION_ATTEMPTS = 300


def _dominant_color(grid: np.ndarray) -> int:
    values = [int(v) for v in np.asarray(grid).reshape(-1)]
    return int(Counter(values).most_common(1)[0][0])


def _separator_color(grid: np.ndarray, bg: int) -> int:
    best_color = None
    best_score = -1
    for value in np.unique(grid):
        color = int(value)
        if color == bg:
            continue
        full_rows = np.all(grid == color, axis=1)
        full_cols = np.all(grid == color, axis=0)
        score = int(full_rows.sum() + full_cols.sum())
        if full_rows.any() and full_cols.any() and score > best_score:
            best_color = color
            best_score = score
    assert best_color is not None, "expected a separator color spanning full rows and columns"
    return best_color


def _non_separator_spans(full_lines: np.ndarray) -> list[slice]:
    spans: list[slice] = []
    start = None
    for idx, is_sep in enumerate(full_lines.tolist()):
        if not is_sep and start is None:
            start = idx
        elif is_sep and start is not None:
            spans.append(slice(start, idx))
            start = None
    if start is not None:
        spans.append(slice(start, len(full_lines)))
    return spans


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = mask.shape
    dst_h, dst_w = shape
    row_idx = (np.arange(dst_h) * src_h) // dst_h
    col_idx = (np.arange(dst_w) * src_w) // dst_w
    return mask[row_idx][:, col_idx]


def _reference_color(
    grid: np.ndarray, row_spans: list[slice], ref_col_span: slice, bg: int, sep: int
) -> int:
    values: list[int] = []
    for row_span in row_spans:
        cell = grid[row_span, ref_col_span]
        values.extend(
            int(v)
            for v in cell.reshape(-1)
            if int(v) != bg and int(v) != sep
        )
    assert values, "reference panel must contain a foreground template color"
    return int(Counter(values).most_common(1)[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    For each target cell, count its same-color markers, select the matching
    reference-row template, and repaint that template in the marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    sep = _separator_color(g, bg)

    row_spans = _non_separator_spans(np.all(g == sep, axis=1))
    col_spans = _non_separator_spans(np.all(g == sep, axis=0))
    assert len(row_spans) >= 2, "need at least two cell rows"
    assert len(col_spans) >= 2, "need a reference column plus targets"

    row_heights = {row_span.stop - row_span.start for row_span in row_spans}
    col_widths = {col_span.stop - col_span.start for col_span in col_spans}
    assert len(row_heights) == 1 and len(col_widths) == 1, "all cells must be equal-sized"
    cell_h = next(iter(row_heights))
    cell_w = next(iter(col_widths))
    assert cell_h == cell_w, "cells must be square"

    ref_color = _reference_color(g, row_spans, col_spans[0], bg, sep)
    reference_masks: list[np.ndarray] = []
    for row_span in row_spans:
        mask = g[row_span, col_spans[0]] == ref_color
        assert mask.any(), "each reference cell must contain a pattern"
        reference_masks.append(mask)

    out = g.copy()
    for row_span in row_spans:
        for col_span in col_spans[1:]:
            cell = g[row_span, col_span]
            values = [int(v) for v in cell.reshape(-1) if int(v) != bg]
            assert values, "each target cell must contain markers"
            marker_counts = Counter(values)
            assert len(marker_counts) == 1, "markers inside a target cell must share one color"
            marker_color, marker_total = next(iter(marker_counts.items()))
            assert 1 <= marker_total <= len(reference_masks), "marker count must index a reference row"

            template = _resize_mask(reference_masks[marker_total - 1], cell.shape)
            painted = np.full(cell.shape, bg, dtype=int)
            painted[template] = int(marker_color)
            out[row_span, col_span] = painted

    return out


BASE_PATTERNS = [
    np.array(
        [[0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0]],
        dtype=bool,
    ),
    np.array(
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        dtype=bool,
    ),
    np.array(
        [[1, 0, 0, 0, 1],
         [0, 1, 0, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [1, 0, 0, 0, 1]],
        dtype=bool,
    ),
    np.array(
        [[1, 1, 1, 1, 1],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [1, 1, 1, 1, 1]],
        dtype=bool,
    ),
    np.array(
        [[0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [1, 1, 1, 1, 1],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0]],
        dtype=bool,
    ),
    np.array(
        [[1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 1, 1, 1, 1]],
        dtype=bool,
    ),
    np.array(
        [[1, 1, 1, 1, 1],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0]],
        dtype=bool,
    ),
    np.array(
        [[0, 0, 1, 0, 0],
         [0, 1, 1, 1, 0],
         [1, 1, 1, 1, 1],
         [0, 1, 1, 1, 0],
         [0, 0, 1, 0, 0]],
        dtype=bool,
    ),
]


def _dihedral(mask: np.ndarray, variant: int) -> np.ndarray:
    out = np.rot90(mask, variant % 4)
    if variant >= 4:
        out = np.fliplr(out)
    return out


def _cell_origin(row: int, col: int, cell_size: int) -> tuple[int, int]:
    return 1 + row * (cell_size + 1), 1 + col * (cell_size + 1)


def _choose_reference_masks(row_count: int, cell_size: int, rng: random.Random) -> list[np.ndarray]:
    for _ in range(100):
        choices = rng.sample(range(len(BASE_PATTERNS)), row_count)
        masks = [
            _resize_mask(_dihedral(BASE_PATTERNS[idx], rng.randrange(8)), (cell_size, cell_size))
            for idx in choices
        ]
        if any(not mask.any() or mask.all() for mask in masks):
            continue
        if len({mask.tobytes() for mask in masks}) != row_count:
            continue
        return masks
    raise RuntimeError("failed to sample distinct reference masks")


def _sample_marker_points(cell_size: int, count: int, rng: random.Random) -> list[tuple[int, int]]:
    points = [(r, c) for r in range(cell_size) for c in range(cell_size)]
    for gap in (2, 1, 0):
        for _ in range(80):
            rng.shuffle(points)
            chosen: list[tuple[int, int]] = []
            for point in points:
                if all(
                    max(abs(point[0] - other[0]), abs(point[1] - other[1])) > gap
                    for other in chosen
                ):
                    chosen.append(point)
                    if len(chosen) == count:
                        return chosen
    return rng.sample(points, count)


def _marker_color_grid(row_count: int, target_cols: int, rng: random.Random) -> list[list[int]]:
    colors = [[BLACK] * target_cols for _ in range(row_count)]
    for r in range(row_count):
        for c in range(target_cols):
            blocked = set()
            if r > 0:
                blocked.add(colors[r - 1][c])
            if c > 0:
                blocked.add(colors[r][c - 1])
            choices = [color for color in MARKER_COLORS if color not in blocked]
            colors[r][c] = rng.choice(choices or list(MARKER_COLORS))
    return colors


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_count = rng.randint(2, 5)
    target_cols = rng.randint(2, 4)
    total_cols = target_cols + 1

    max_cell_h = (MAX_GRID_SIDE - (row_count + 1)) // row_count
    max_cell_w = (MAX_GRID_SIDE - (total_cols + 1)) // total_cols
    max_cell = min(max_cell_h, max_cell_w, 7)
    if max_cell < 4:
        return None
    cell_size = rng.randint(4, max_cell)

    reference_masks = _choose_reference_masks(row_count, cell_size, rng)
    counts = list(range(1, row_count + 1)) * target_cols
    rng.shuffle(counts)
    count_grid = np.array(counts, dtype=int).reshape(row_count, target_cols)
    color_grid = _marker_color_grid(row_count, target_cols, rng)

    height = row_count * cell_size + row_count + 1
    width = total_cols * cell_size + total_cols + 1
    grid = np.full((height, width), BLACK, dtype=int)

    for r in range(0, height, cell_size + 1):
        grid[r, :] = GREY
    for c in range(0, width, cell_size + 1):
        grid[:, c] = GREY

    for row in range(row_count):
        top, left = _cell_origin(row, 0, cell_size)
        ref_mask = reference_masks[row]
        grid[top : top + cell_size, left : left + cell_size][ref_mask] = AZURE

    for row in range(row_count):
        for col in range(target_cols):
            top, left = _cell_origin(row, col + 1, cell_size)
            count = int(count_grid[row, col])
            color = int(color_grid[row][col])
            for dr, dc in _sample_marker_points(cell_size, count, rng):
                grid[top + dr, left + dc] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
