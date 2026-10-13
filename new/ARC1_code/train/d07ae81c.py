from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import in_bounds, neighbors8


def _base_colors_by_frequency(g: np.ndarray) -> set[int]:
    counts = Counter(map(int, g.ravel()))
    assert len(counts) == 4, "expected two base colors and two marker colors"

    ordered_counts = sorted(counts.values(), reverse=True)
    assert ordered_counts[1] > ordered_counts[2], "base/marker frequency split is ambiguous"
    base_cutoff = ordered_counts[1]

    base_colors = {color for color, count in counts.items() if count >= base_cutoff}
    assert len(base_colors) == 2, "expected exactly two large base colors"
    return base_colors


def _marker_pixels(g: np.ndarray, base_colors: set[int]) -> list[tuple[int, int, int]]:
    base_mask = np.isin(g, list(base_colors))
    markers = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere(~base_mask)]
    assert markers, "expected at least one marker pixel"
    return markers


def _base_to_marker_map(
    g: np.ndarray,
    base_colors: set[int],
    markers: list[tuple[int, int, int]],
) -> dict[int, int]:
    base_to_marker: dict[int, int] = {}
    marker_to_base: dict[int, int] = {}

    for r, c, marker in markers:
        neighbor_counts: Counter[int] = Counter()
        for rr, cc in neighbors8(r, c):
            if not in_bounds(g.shape, rr, cc):
                continue
            color = int(g[rr, cc])
            if color in base_colors:
                neighbor_counts[color] += 1

        assert neighbor_counts, "marker must touch at least one base color"
        best_count = max(neighbor_counts.values())
        candidate_bases = [color for color, count in neighbor_counts.items() if count == best_count]
        assert len(candidate_bases) == 1, "marker's underlying base color must be unique"
        base = candidate_bases[0]

        if base in base_to_marker:
            assert base_to_marker[base] == marker, "one base color cannot have two marker colors"
        else:
            base_to_marker[base] = marker

        if marker in marker_to_base:
            assert marker_to_base[marker] == base, "one marker color cannot represent two bases"
        else:
            marker_to_base[marker] = base

    assert set(base_to_marker) == base_colors, "each base color must have a marker color"
    assert len(base_to_marker) == len(marker_to_base) == 2, "expected a one-to-one base/marker color map"
    return base_to_marker


def main(grid: np.ndarray) -> np.ndarray:
    """
    Activate both diagonals through each marker pixel and recolor activated base-color cells to that base color's paired marker color.

    Input structure
    - The canvas is filled by two large base colors.
    - The other colors occur as marker pixels placed in those base-color regions.
    - A marker's surrounding base-color neighborhood identifies which base color the marker color represents.

    Rule
    - For every marker pixel, activate the full NW-SE and NE-SW diagonals through that pixel, all the way to the grid edges.
    - Starting from the original grid, every activated cell whose color is one of the two base colors changes to that base color's paired marker color.
    - Original marker pixels remain unchanged, including where diagonal paths cross them.

    Key insight
    - The markers do not paint only their own color; they define diagonal paths, and the color written on each path cell is chosen by the cell's original base color.
    """
    g = np.asarray(grid, dtype=int)
    base_colors = _base_colors_by_frequency(g)
    markers = _marker_pixels(g, base_colors)
    base_to_marker = _base_to_marker_map(g, base_colors, markers)

    rows, cols = np.indices(g.shape)
    path_mask = np.zeros(g.shape, dtype=bool)
    for r, c, _marker in markers:
        path_mask |= (rows - cols == r - c) | (rows + cols == r + c)

    out = g.copy()
    for base, marker in base_to_marker.items():
        out[path_mask & (g == base)] = marker
    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, and retry bounds.
CANVAS_MIN_SIZE, CANVAS_MAX_SIZE = 12, 19
COLOR_POOL = np.arange(1, 10)
GENERATION_ATTEMPTS = 300
# Alternating band layout bounds.
COLUMN_COUNT_OPTIONS = (3, 4)
COLUMN_WIDTH_MIN, COLUMN_WIDTH_MAX = 2, 10
ROW_COUNT_OPTIONS = (1, 3, 5)
ROW_HEIGHT_MIN, ROW_HEIGHT_MAX = 2, 15
# Layout acceptance and marker-placement constraints.
MAJORITY_AREA_RATIO_MIN, MAJORITY_AREA_RATIO_MAX = 0.53, 0.72
MARKER_EDGE_MARGIN = 1
BASE_MARKERS_PER_COLOR = 1
EXTRA_MARKER_PROBABILITY = 0.5
EXTRA_MARKER_MIN_CANDIDATES = BASE_MARKERS_PER_COLOR + 1
MARKER_SPACING = 2


def _gen_partitions(total, n_parts, lo, hi):
    out = []

    def rec(remaining, k, prefix):
        if k == 0:
            if remaining == 0:
                out.append(tuple(prefix))
            return
        low = max(lo, remaining - hi * (k - 1))
        high = min(hi, remaining - lo * (k - 1))
        for value in range(low, high + 1):
            prefix.append(value)
            rec(remaining - value, k - 1, prefix)
            prefix.pop()

    rec(total, n_parts, [])
    return out


def _gen_choose(seq):
    return seq[int(np.random.randint(len(seq)))]


def _gen_neighborhood_cells(grid, color):
    h, w = grid.shape
    cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) == color and np.all(grid[r - 1 : r + 2, c - 1 : c + 2] == color):
                cells.append((r, c))
    return cells


def _gen_take_spaced(cells, n, used):
    available = [cell for cell in cells if all(max(abs(cell[0] - r), abs(cell[1] - c)) >= MARKER_SPACING for r, c in used)]
    chosen = []
    for _ in range(n):
        if not available:
            return None
        cell = _gen_choose(available)
        chosen.append(cell)
        used.append(cell)
        available = [other for other in available if max(abs(other[0] - cell[0]), abs(other[1] - cell[1])) >= MARKER_SPACING]
    return chosen


def _gen_choose_layout(h, w):
    """Choose alternating row/column bands; return (row_heights, col_widths) or None."""
    col_count = int(_gen_choose(COLUMN_COUNT_OPTIONS))
    col_parts = []
    for part in _gen_partitions(w, col_count, COLUMN_WIDTH_MIN, COLUMN_WIDTH_MAX):
        min_width = sum(part[0::2])
        maj_width = sum(part[1::2])
        min_interior = sum(max(0, x - 2 * MARKER_EDGE_MARGIN) for x in part[0::2])
        if min_width < maj_width and min_interior > 0:
            col_parts.append(part)
    if not col_parts: return None
    col_widths = _gen_choose(col_parts)

    valid_row_counts = [n for n in ROW_COUNT_OPTIONS if ROW_HEIGHT_MIN * n <= h <= ROW_HEIGHT_MAX * n]
    row_count = int(_gen_choose(valid_row_counts))
    row_parts = []
    for part in _gen_partitions(h, row_count, ROW_HEIGHT_MIN, ROW_HEIGHT_MAX):
        mixed_interior = sum(max(0, x - 2 * MARKER_EDGE_MARGIN) for i, x in enumerate(part) if i % 2 == 0)
        if mixed_interior > 0:
            row_parts.append(part)
    if not row_parts: return None
    return _gen_choose(row_parts), col_widths


def _gen_base_grid(h, w, row_heights, col_widths, maj_color, min_color):
    grid = np.full((h, w), maj_color, dtype=int)
    r0 = 0
    for row_i, row_h in enumerate(row_heights):
        if row_i % 2 == 0:
            c0 = 0
            for col_i, col_w in enumerate(col_widths):
                if col_i % 2 == 0:
                    grid[r0 : r0 + row_h, c0 : c0 + col_w] = min_color
                c0 += col_w
        r0 += row_h
    return grid


def _gen_add_markers(grid, maj_color, min_color, maj_marker, min_marker):
    """Place marker pixels in base-color interiors; mutates grid and reports success."""
    candidates = {maj_color: _gen_neighborhood_cells(grid, maj_color), min_color: _gen_neighborhood_cells(grid, min_color)}
    if not candidates[maj_color] or not candidates[min_color]:
        return False

    marker_counts = {maj_color: BASE_MARKERS_PER_COLOR, min_color: BASE_MARKERS_PER_COLOR}
    if np.random.random() < EXTRA_MARKER_PROBABILITY:
        extra_options = [color for color in (maj_color, min_color) if len(candidates[color]) >= EXTRA_MARKER_MIN_CANDIDATES]
        if not extra_options:
            return False
        marker_counts[int(_gen_choose(extra_options))] += 1

    used = []
    marker_map = {maj_color: maj_marker, min_color: min_marker}
    for base_color in np.random.permutation([maj_color, min_color]):
        chosen = _gen_take_spaced(candidates[int(base_color)], marker_counts[int(base_color)], used)
        if chosen is None:
            return False
        for r, c in chosen:
            grid[r, c] = marker_map[int(base_color)]
    return True


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(CANVAS_MIN_SIZE, CANVAS_MAX_SIZE + 1))
    w = int(np.random.randint(CANVAS_MIN_SIZE, CANVAS_MAX_SIZE + 1))

    layout = _gen_choose_layout(h, w)
    if layout is None: return None
    row_heights, col_widths = layout

    colors = np.random.choice(COLOR_POOL, size=4, replace=False)
    maj_color, min_color, maj_marker, min_marker = map(int, colors)

    grid = _gen_base_grid(h, w, row_heights, col_widths, maj_color, min_color)
    maj_area = int(np.count_nonzero(grid == maj_color))
    min_area = int(np.count_nonzero(grid == min_color))
    area_ratio = maj_area / (h * w)
    if maj_area <= min_area or not (MAJORITY_AREA_RATIO_MIN <= area_ratio <= MAJORITY_AREA_RATIO_MAX):
        return None

    if not _gen_add_markers(grid, maj_color, min_color, maj_marker, min_marker): return None

    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid d07ae81c input")
