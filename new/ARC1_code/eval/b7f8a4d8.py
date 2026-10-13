from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import bbox_coords, components_mask, paste
import random

BBox = tuple[int, int, int, int]


def _unique_most_common(counts: Counter[int], what: str) -> int:
    ranked = counts.most_common()
    assert ranked, f"expected at least one {what}"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], f"expected unique {what}"
    return int(ranked[0][0])


def _interior_regions(g: np.ndarray, bg: int, frame: int) -> list[tuple[int, BBox]]:
    regions: list[tuple[int, BBox]] = []
    for coords in components_mask(g != bg, connectivity=4):
        values = set(map(int, g[coords[:, 0], coords[:, 1]]))
        if frame not in values:
            continue

        interior_colors = [color for color in values if color != frame]
        if not interior_colors:
            continue
        assert len(interior_colors) == 1, "each tile region must have one interior color"

        color = int(interior_colors[0])
        interior_coords = coords[g[coords[:, 0], coords[:, 1]] == color]
        r0, c0, r1, c1 = bbox_coords(interior_coords)
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "interior block must be solid"
        regions.append((color, (r0, c0, r1, c1)))

    assert regions, "expected framed tile interiors"
    return regions


def _paint_rectangular_connectors(g: np.ndarray, bg: int, markers: dict[int, list[BBox]]) -> dict[tuple[int, int], int]:
    planned: dict[tuple[int, int], int] = {}

    for color, boxes in markers.items():
        assert len(boxes) == 4, "each marker color must appear at four rectangle corners"

        row_intervals = sorted({(box[0], box[2]) for box in boxes})
        col_intervals = sorted({(box[1], box[3]) for box in boxes})
        assert len(row_intervals) == 2, "marker corners must define two row bands"
        assert len(col_intervals) == 2, "marker corners must define two column bands"

        expected = {(rows, cols) for rows in row_intervals for cols in col_intervals}
        actual = {((box[0], box[2]), (box[1], box[3])) for box in boxes}
        assert actual == expected, "marker regions must occupy all four rectangle corners"

        (_top0, top1), (bottom0, _bottom1) = row_intervals
        (_left0, left1), (right0, _right1) = col_intervals

        for r0, r1 in row_intervals:
            for r in range(r0, r1 + 1):
                for c in range(left1 + 1, right0):
                    if int(g[r, c]) == bg:
                        previous = planned.get((r, c))
                        assert previous is None or previous == color, "conflicting connector colors"
                        planned[(r, c)] = color

        for c0, c1 in col_intervals:
            for c in range(c0, c1 + 1):
                for r in range(top1 + 1, bottom0):
                    if int(g[r, c]) == bg:
                        previous = planned.get((r, c))
                        assert previous is None or previous == color, "conflicting connector colors"
                        planned[(r, c)] = color

    return planned


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw same-colored rectangular connector bands through background channels between four matching marker blocks.

    Input structure
    - A background lattice of separated framed tiles; the frame color is the unique most frequent non-background color.
    - Each tile region contains one solid non-frame interior block.
    - The unique most common interior color is the default; every other interior color marks four rectangle corners.

    Rule
    - For each marker color, find its two row bands and two column bands.
    - Paint that color through background cells along the two horizontal and two vertical sides of the implied rectangle.
    - Preserve all original frame, default interior, marker, and off-path background pixels.

    Key insight
    - The repeated tile frames define which cells are obstacles and which cells are background channels; matching interior colors supply rectangle corners independent of tile and marker size.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(g[0, 0])
    assert np.all(g[0, :] == bg), "top row must be background"

    non_bg = [int(v) for v in g.ravel() if int(v) != bg]
    frame = _unique_most_common(Counter(non_bg), "frame color")

    regions = _interior_regions(g, bg, frame)
    default = _unique_most_common(Counter(color for color, _box in regions), "default interior color")

    markers: dict[int, list[BBox]] = defaultdict(list)
    for color, box in regions:
        if color != default:
            markers[int(color)].append(box)
    assert markers, "expected at least one marker color"

    out = g.copy()
    for (r, c), color in _paint_rectangular_connectors(g, bg, markers).items():
        out[r, c] = color
    return out

# --------------- generator (o5x) ---------------

# Canvas and lattice bounds.
GEN_BG = 0
GEN_COLORS = range(1, 10)
GEN_PERIODS, GEN_INTERIOR_SIZES = range(5, 8), range(1, 4)
GEN_TILE_ROWS, GEN_TILE_COLS = range(4, 6), range(4, 7)
GEN_HEIGHT_MIN, GEN_HEIGHT_MAX = 23, 30
GEN_WIDTH_MIN, GEN_WIDTH_MAX = 28, 30
GEN_MARGIN = GEN_FRAME = 1
GEN_GAP_MIN, GEN_GAP_MAX = 1, 3

# Marker placement and default-tile constraints.
GEN_MARKER_COUNTS = (1, 2)
GEN_MARKER_TILES = 4
GEN_DEFAULT_TILES_MIN, GEN_DEFAULT_TILES_MAX = 8, 22
GEN_ROW_SPAN_MIN, GEN_ROW_SPAN_MAX = 2, 3
GEN_COL_SPAN_MIN, GEN_COL_SPAN_MAX = 1, 3

# Generation retry budget.
GEN_MAX_ATTEMPTS = 300

def _gen_lattice_options() -> list[tuple[int, int, int, int, int, int, int]]:
    """Enumerate valid lattice/canvas tuples in sampling order."""
    options = []
    for period in GEN_PERIODS:
        for interior_size in GEN_INTERIOR_SIZES:
            gap = period - interior_size - 2
            if not GEN_GAP_MIN <= gap <= GEN_GAP_MAX:
                continue
            for tile_rows in GEN_TILE_ROWS:
                min_h = GEN_MARGIN + (tile_rows - 1) * period + GEN_FRAME + interior_size
                max_h = GEN_MARGIN + tile_rows * period + GEN_FRAME
                for tile_cols in GEN_TILE_COLS:
                    min_w = GEN_MARGIN + (tile_cols - 1) * period + GEN_FRAME + interior_size
                    max_w = GEN_MARGIN + tile_cols * period + GEN_FRAME
                    for height in range(max(GEN_HEIGHT_MIN, min_h), min(GEN_HEIGHT_MAX, max_h) + 1):
                        for width in range(max(GEN_WIDTH_MIN, min_w), min(GEN_WIDTH_MAX, max_w) + 1):
                            options.append((period, interior_size, gap, tile_rows, tile_cols, height, width))
    return options

_GEN_LATTICE_OPTIONS = _gen_lattice_options()

def _gen_pairs(n: int, lo: int, hi: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(n) for b in range(a + 1, n) if lo <= b - a <= hi]

def _gen_rectangles(tile_rows: int, tile_cols: int, marker_count: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Sample marker rectangles without sharing row or column bands."""
    row_pairs = _gen_pairs(tile_rows, GEN_ROW_SPAN_MIN, GEN_ROW_SPAN_MAX)
    col_pairs = _gen_pairs(tile_cols, GEN_COL_SPAN_MIN, GEN_COL_SPAN_MAX)

    if marker_count == 1:
        return [(random.choice(row_pairs), random.choice(col_pairs))]

    row_pair_choices = [
        (a, b)
        for i, a in enumerate(row_pairs)
        for b in row_pairs[i + 1 :]
        if set(a).isdisjoint(b)
    ]
    col_pair_choices = [
        (a, b)
        for i, a in enumerate(col_pairs)
        for b in col_pairs[i + 1 :]
        if set(a).isdisjoint(b)
    ]
    rows = list(random.choice(row_pair_choices))
    cols = list(random.choice(col_pair_choices))
    random.shuffle(rows)
    random.shuffle(cols)
    return list(zip(rows, cols))

def _gen_tile_patch(interior_size: int, frame_color: int, interior_color: int) -> np.ndarray:
    tile_size = interior_size + 2 * GEN_FRAME
    tile = np.full((tile_size, tile_size), frame_color, dtype=int)
    tile[1:-1, 1:-1] = interior_color
    return tile

def _gen_interior_box(period: int, interior_size: int, row_i: int, col_i: int) -> tuple[int, int, int, int]:
    r0 = GEN_MARGIN + row_i * period + GEN_FRAME
    c0 = GEN_MARGIN + col_i * period + GEN_FRAME
    return (r0, c0, r0 + interior_size - 1, c0 + interior_size - 1)

def _gen_marker_maps(period: int, interior_size: int, marker_colors: list[int], rectangles):
    """Map marker tile indices and marker colors to their four interior boxes."""
    marker_by_tile: dict[tuple[int, int], int] = {}
    marker_boxes: dict[int, list[tuple[int, int, int, int]]] = {}
    for color, (row_pair, col_pair) in zip(marker_colors, rectangles):
        boxes = []
        for row_i in row_pair:
            for col_i in col_pair:
                marker_by_tile[(row_i, col_i)] = color
                boxes.append(_gen_interior_box(period, interior_size, row_i, col_i))
        marker_boxes[color] = boxes
    return marker_by_tile, marker_boxes

def _gen_render_lattice(layout, frame_color: int, default_color: int, marker_by_tile: dict[tuple[int, int], int]) -> np.ndarray:
    """Render the clipped tile lattice before connector cells are painted."""
    period, interior_size, _gap, tile_rows, tile_cols, height, width = layout
    base = np.full((height, width), GEN_BG, dtype=int)
    for row_i, r0 in enumerate(range(GEN_MARGIN, height, period)):
        for col_i, c0 in enumerate(range(GEN_MARGIN, width, period)):
            interior_color = default_color
            if row_i < tile_rows and col_i < tile_cols:
                interior_color = marker_by_tile.get((row_i, col_i), default_color)
            paste(
                base,
                _gen_tile_patch(interior_size, frame_color, interior_color),
                r0,
                c0,
                transparent=None,
                clip=True,
            )
    return base

def _gen_connector_cells(grid: np.ndarray, bg: int, marker_boxes: dict[int, list[tuple[int, int, int, int]]]):
    """Plan all rectangle connector cells, returning None on color conflict."""
    planned: dict[tuple[int, int], int] = {}
    for color, boxes in marker_boxes.items():
        row_intervals = sorted({(box[0], box[2]) for box in boxes})
        col_intervals = sorted({(box[1], box[3]) for box in boxes})
        (_top0, top1), (bottom0, _bottom1) = row_intervals
        (_left0, left1), (right0, _right1) = col_intervals

        for r0, r1 in row_intervals:
            for r in range(r0, r1 + 1):
                for c in range(left1 + 1, right0):
                    if int(grid[r, c]) == bg:
                        old = planned.get((r, c))
                        if old is not None and old != color:
                            return None
                        planned[(r, c)] = color
        for c0, c1 in col_intervals:
            for c in range(c0, c1 + 1):
                for r in range(top1 + 1, bottom0):
                    if int(grid[r, c]) == bg:
                        old = planned.get((r, c))
                        if old is not None and old != color:
                            return None
                        planned[(r, c)] = color
    return planned

def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input and keep it only if the solver oracle matches."""
    layout = random.choice(_GEN_LATTICE_OPTIONS)
    period, interior_size, _gap, tile_rows, tile_cols, _height, _width = layout

    marker_choices = [
        m for m in GEN_MARKER_COUNTS
        if GEN_DEFAULT_TILES_MIN <= tile_rows * tile_cols - GEN_MARKER_TILES * m <= GEN_DEFAULT_TILES_MAX
    ]
    marker_count = random.choice(marker_choices)
    frame_color, default_color, *marker_colors = random.sample(GEN_COLORS, 2 + marker_count)

    rectangles = _gen_rectangles(tile_rows, tile_cols, marker_count)
    marker_by_tile, marker_boxes = _gen_marker_maps(period, interior_size, marker_colors, rectangles)

    base = _gen_render_lattice(layout, frame_color, default_color, marker_by_tile)
    planned = _gen_connector_cells(base, GEN_BG, marker_boxes)
    if planned is None:
        return None

    out = base.copy()
    for (r, c), color in planned.items():
        out[r, c] = color

    inp = out.copy()
    for r, c in planned:
        inp[r, c] = GEN_BG

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp

def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b7f8a4d8 input")
