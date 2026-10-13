import itertools
import random
from collections import Counter

import numpy as np


BACKGROUND = 5
MARKER_COUNTS = (3, 4)
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_ROWS = (13, 18)
GRID_COLS = (13, 18)
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_markers(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    coords = np.argwhere(g != bg)
    assert len(coords) in MARKER_COUNTS, "expected 3 or 4 markers"

    markers = [(int(r), int(c), int(g[r, c])) for r, c in coords]
    assert len({color for _r, _c, color in markers}) == len(markers), "marker colors must be distinct"
    assert len({r for r, _c, _color in markers}) == len(markers), "marker rows must be distinct"
    assert len({c for _r, c, _color in markers}) == len(markers), "marker cols must be distinct"
    assert all(0 < r < g.shape[0] - 1 for r, _c, _color in markers), "markers may not touch top/bottom border"
    assert all(0 < c < g.shape[1] - 1 for _r, c, _color in markers), "markers may not touch left/right border"
    return bg, sorted(markers, key=lambda item: item[1])


def _spans(sorted_indices: list[int], limit: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    prev = -1
    for idx in sorted_indices:
        start = prev + 1
        end = idx - 1
        if start <= end:
            spans.append((start, end))
        prev = idx
    if prev + 1 <= limit - 1:
        spans.append((prev + 1, limit - 1))
    return spans


def _draw_cross_grid(shape: tuple[int, int], bg: int, markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for r, c, color in markers:
        out[r, :] = color
        out[:, c] = color
    return out


def _solve_from_markers(shape: tuple[int, int], bg: int, markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = _draw_cross_grid(shape, bg, markers)
    palette = [color for _r, _c, color in markers]
    row_spans = _spans(sorted(r for r, _c, _color in markers), shape[0])
    col_spans = _spans(sorted(c for _r, c, _color in markers), shape[1])

    for row_index, (r0, r1) in enumerate(row_spans):
        for col_index, (c0, c1) in enumerate(col_spans):
            out[r0 : r1 + 1, c0 : c1 + 1] = palette[(row_index + col_index) % len(palette)]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw marker-colored full-canvas crosses, then cyclically color the rectangular regions that the crosses create.

    Input structure:
    - The grid has a dominant grey background and exactly 3 or 4 isolated colored marker pixels.
    - The markers have distinct colors, distinct rows, and distinct columns, and none lies on the canvas border.
    - No other foreground objects are present.

    Rule:
    - Sort the markers by column from left to right.
    - For each marker in that order, draw a full-width horizontal line on its row and a full-height vertical line on its column in the marker's color; later markers overwrite earlier ones at intersections.
    - The line rows and line columns split the canvas into rectangular background regions. Index the regions by row band from top to bottom and by column band from left to right, then fill each region with palette[(row_index + col_index) mod palette_length], where the palette is the left-to-right marker colors.

    Key insight:
    - The markers do not directly label the regions; they first create a line grid, and the region color depends on the region's row/column index within that induced partition.
    """
    bg, markers = _extract_markers(grid)
    return _solve_from_markers(np.asarray(grid, dtype=int).shape, bg, markers)


def _random_composition(parts: int, total: int, max_part: int = 5) -> list[int] | None:
    if total < parts or total > parts * max_part:
        return None
    for _ in range(200):
        cuts = sorted(random.sample(range(1, total), parts - 1))
        values = [cuts[0], *[b - a for a, b in zip(cuts, cuts[1:])], total - cuts[-1]]
        if max(values) <= max_part and len(set(values)) >= 2 and any(v >= 3 for v in values):
            return values
    return None


def _line_positions(segments: list[int]) -> list[int]:
    positions: list[int] = []
    cursor = 0
    for seg in segments[:-1]:
        cursor += seg
        positions.append(cursor)
        cursor += 1
    return positions


def _independent_region_color(palette: list[int], row_index: int, col_index: int) -> int:
    return palette[(row_index + col_index) % len(palette)]


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        bg, markers = _extract_markers(grid)
        out = main(grid)
    except AssertionError:
        return False

    if bg != BACKGROUND:
        return False

    n = len(markers)
    if n not in MARKER_COUNTS:
        return False

    palette = [color for _r, _c, color in markers]
    if palette == sorted(palette) or palette == sorted(palette, reverse=True):
        return False

    row_order = tuple(sorted(range(n), key=lambda i: markers[i][0]))
    if row_order == tuple(range(n)):
        return False

    rows = sorted(r for r, _c, _color in markers)
    cols = sorted(c for _r, c, _color in markers)
    row_spans = _spans(rows, grid.shape[0])
    col_spans = _spans(cols, grid.shape[1])
    if len(row_spans) != n + 1 or len(col_spans) != n + 1:
        return False

    line_grid = _draw_cross_grid(grid.shape, bg, markers)
    line_mask = np.zeros(grid.shape, dtype=bool)
    line_mask[rows, :] = True
    line_mask[:, cols] = True
    if not np.array_equal(out[line_mask], line_grid[line_mask]):
        return False
    if np.array_equal(out, line_grid):
        return False

    filled_colors = set()
    for row_index, (r0, r1) in enumerate(row_spans):
        for col_index, (c0, c1) in enumerate(col_spans):
            block = out[r0 : r1 + 1, c0 : c1 + 1]
            expected = _independent_region_color(palette, row_index, col_index)
            if not np.all(block == expected):
                return False
            filled_colors.add(expected)

    if filled_colors != set(palette):
        return False

    if set(np.unique(out)) != set(palette):
        return False

    if sum(1 for size in [b - a + 1 for a, b in row_spans] if size >= 3) == 0:
        return False
    if sum(1 for size in [b - a + 1 for a, b in col_spans] if size >= 3) == 0:
        return False

    counts = Counter(out.ravel().tolist())
    if min(counts[color] for color in palette) < 3:
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    n = random.choice(MARKER_COUNTS)
    region_count = n + 1

    row_total = random.randint(max(region_count, GRID_ROWS[0] - n), GRID_ROWS[1] - n)
    col_total = random.randint(max(region_count, GRID_COLS[0] - n), GRID_COLS[1] - n)
    row_segments = _random_composition(region_count, row_total)
    col_segments = _random_composition(region_count, col_total)
    if row_segments is None or col_segments is None:
        return None

    rows = _line_positions(row_segments)
    cols = _line_positions(col_segments)
    h = sum(row_segments) + n
    w = sum(col_segments) + n
    if not (GRID_ROWS[0] <= h <= GRID_ROWS[1] and GRID_COLS[0] <= w <= GRID_COLS[1]):
        return None

    palette = random.sample(MARKER_COLORS, n)
    if palette == sorted(palette) or palette == sorted(palette, reverse=True):
        return None

    permutations = [perm for perm in itertools.permutations(range(n)) if perm != tuple(range(n))]
    row_permutation = random.choice(permutations)
    grid = np.full((h, w), BACKGROUND, dtype=int)
    for col_rank, row_rank in enumerate(row_permutation):
        grid[rows[row_rank], cols[col_rank]] = palette[col_rank]

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid instance for puzzle 7")
