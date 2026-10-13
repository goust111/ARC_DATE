import random

import numpy as np
from math import ceil


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_cross_axes(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    candidates: list[tuple[int, np.ndarray, np.ndarray]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        full_rows = np.where(np.all(grid == color, axis=1))[0]
        full_cols = np.where(np.all(grid == color, axis=0))[0]
        if len(full_rows) and len(full_cols):
            candidates.append((color, full_rows, full_cols))

    assert len(candidates) == 1, "expected one cross color"
    axis_color, rows, cols = candidates[0]
    assert len(rows) == 1 and len(cols) == 1, "expected one full cross row and one full cross column"
    return axis_color, int(rows[0]), int(cols[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete sparse marker pixels by reflecting them across a full-row/full-column cross.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    axis_color, axis_r, axis_c = _find_cross_axes(g, bg)
    markers = np.argwhere((g != bg) & (g != axis_color))
    assert len(markers) > 0, "expected markers away from the cross"

    out = g.copy()
    for r0, c0 in markers:
        r = int(r0)
        c = int(c0)
        color = int(g[r, c])
        assert r != axis_r and c != axis_c, "markers must not lie on the cross"

        for rr in (r, 2 * axis_r - r):
            for cc in (c, 2 * axis_c - c):
                if not in_bounds(g.shape, rr, cc):
                    continue
                existing = int(out[rr, cc])
                assert existing in (bg, color), "conflicting mirrored colors"
                out[rr, cc] = color
    return out


BG = 7
PALETTE = tuple(range(10))
GRID_SIZE_RANGE = (8, 18)
AXIS_MARGIN = 2
MARKER_COUNT_RANGE = (5, 12)
MARKER_DENSITY_RANGE = (0.18, 0.45)
COLOR_COUNT_RANGE = (2, 6)
MAX_MARKERS_PER_COLOR = 3
MAX_GENERATION_ATTEMPTS = 300


def _quadrant_cells(h: int, w: int, axis_r: int, axis_c: int, quadrant: int) -> list[tuple[int, int]]:
    row_range = range(0, axis_r) if quadrant < 2 else range(axis_r + 1, h)
    col_range = range(0, axis_c) if quadrant % 2 == 0 else range(axis_c + 1, w)
    return [(r, c) for r in row_range for c in col_range]


def _valid_marker_counts(area: int) -> list[int]:
    lo, hi = MARKER_COUNT_RANGE
    min_density, max_density = MARKER_DENSITY_RANGE
    counts = []
    for n in range(lo, hi + 1):
        if n <= area and min_density <= n / area <= max_density:
            counts.append(n)
    return counts


def _sample_layout():
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)

    row_lo, row_hi = AXIS_MARGIN, h - AXIS_MARGIN - 1
    col_lo, col_hi = AXIS_MARGIN, w - AXIS_MARGIN - 1
    if row_lo > row_hi or col_lo > col_hi:
        return None

    axis_r = random.randint(row_lo, row_hi)
    axis_c = random.randint(col_lo, col_hi)

    options = []
    for quadrant in range(4):
        cells = _quadrant_cells(h, w, axis_r, axis_c, quadrant)
        counts = _valid_marker_counts(len(cells))
        if counts:
            options.append((quadrant, cells, counts))
    if not options:
        return None

    quadrant, cells, counts = random.choice(options)
    marker_count = random.choice(counts)
    min_colors = max(COLOR_COUNT_RANGE[0], ceil(marker_count / MAX_MARKERS_PER_COLOR))
    max_colors = min(COLOR_COUNT_RANGE[1], marker_count)
    if min_colors > max_colors:
        return None
    color_count = random.randint(min_colors, max_colors)
    return h, w, axis_r, axis_c, quadrant, cells, marker_count, color_count


def _marker_color_counts(marker_count: int, color_count: int) -> list[int]:
    counts = [1] * color_count
    for _ in range(marker_count - color_count):
        growable = [i for i, count in enumerate(counts) if count < MAX_MARKERS_PER_COLOR]
        counts[random.choice(growable)] += 1
    random.shuffle(counts)
    return counts


def _sample_markers(cells: list[tuple[int, int]], marker_count: int, color_count: int):
    cross_color = random.choice([c for c in PALETTE if c != BG])
    marker_palette = [c for c in PALETTE if c not in (BG, cross_color)]
    marker_colors = random.sample(marker_palette, color_count)
    marker_cells = random.sample(cells, marker_count)

    assignments: list[int] = []
    for color, count in zip(marker_colors, _marker_color_counts(marker_count, color_count)):
        assignments.extend([color] * count)
    random.shuffle(assignments)
    return cross_color, marker_cells, assignments


def _render_input(
    h: int,
    w: int,
    axis_r: int,
    axis_c: int,
    cross_color: int,
    marker_cells: list[tuple[int, int]],
    assignments: list[int],
) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    grid[axis_r, :] = cross_color
    grid[:, axis_c] = cross_color

    for (r, c), color in zip(marker_cells, assignments):
        if grid[r, c] != BG:
            return None
        grid[r, c] = color

    try:
        solved = main(grid)
    except AssertionError:
        return None

    for r in range(h):
        for c in range(w):
            if grid[r, c] == cross_color:
                continue
            if solved[r, c] != BG and solved[r, c] != grid[r, c]:
                mirrored_positions = {
                    (r, c),
                    (2 * axis_r - r, c),
                    (r, 2 * axis_c - c),
                    (2 * axis_r - r, 2 * axis_c - c),
                }
                colors = {
                    int(grid[rr, cc])
                    for rr, cc in mirrored_positions
                    if in_bounds((h, w), rr, cc) and grid[rr, cc] not in (BG, cross_color)
                }
                if len(colors) > 1:
                    return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        layout = _sample_layout()
        if layout is None:
            continue

        h, w, axis_r, axis_c, _quadrant, cells, marker_count, color_count = layout
        sampled = _sample_markers(cells, marker_count, color_count)
        cross_color, marker_cells, assignments = sampled
        grid = _render_input(h, w, axis_r, axis_c, cross_color, marker_cells, assignments)
        if grid is not None:
            return grid

    raise RuntimeError("failed to generate a valid input")
