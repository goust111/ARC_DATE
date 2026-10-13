from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


BACKGROUND = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GENERATION_ATTEMPTS = 400
LAYER_COUNT_RANGE = (3, 5)
OUTER_THICKNESS_CHOICES = (1, 1, 2, 2, 3)
MIN_GRID_SIZE = 14
MAX_GRID_SIZE = 30
LEGEND_MARKER_GAP = 2


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boundary_color(rect: np.ndarray, top: int, left: int, bottom: int, right: int) -> int:
    parts = [rect[top, left : right + 1]]
    if bottom > top:
        parts.append(rect[bottom, left : right + 1])
    if bottom - top >= 2:
        parts.append(rect[top + 1 : bottom, left])
    if right > left and bottom - top >= 2:
        parts.append(rect[top + 1 : bottom, right])
    boundary = np.concatenate([part.reshape(-1) for part in parts if part.size])
    colors = np.unique(boundary)
    assert colors.size == 1, "each visible shell must have a uniform boundary color"
    return int(colors[0])


def _read_layer_spec(rect: np.ndarray) -> tuple[list[int], list[int]]:
    top, left = 0, 0
    bottom, right = rect.shape[0] - 1, rect.shape[1] - 1
    colors: list[int] = []
    thicknesses: list[int] = []

    while top <= bottom and left <= right:
        color = _boundary_color(rect, top, left, bottom, right)
        thickness = 0
        while top <= bottom and left <= right and _boundary_color(rect, top, left, bottom, right) == color:
            thickness += 1
            top += 1
            left += 1
            bottom -= 1
            right -= 1
        colors.append(color)
        thicknesses.append(thickness)

    assert len(colors) >= 2, "expected at least two nested layers"
    assert len(colors) == len(set(colors)), "layer colors must be distinct"
    return colors, thicknesses


def _analyze_input(grid: np.ndarray) -> tuple[BBox, Coord, list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    assert bg == BACKGROUND, "expected black background"

    comps = _components_mask(g != bg)
    assert len(comps) == 2, "expected one legend and one marker"

    marker_comp = [comp for comp in comps if comp.shape[0] == 1]
    legend_comp = [comp for comp in comps if comp.shape[0] > 1]
    assert len(marker_comp) == 1 and len(legend_comp) == 1, "expected exactly one isolated marker"

    marker = tuple(map(int, marker_comp[0][0]))
    legend_bbox = _bbox_coords(legend_comp[0])
    r0, c0, r1, c1 = legend_bbox
    legend = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(legend != bg), "legend must fill its bounding box"

    colors_outer_to_inner, widths_outer_to_inner = _read_layer_spec(legend)
    assert int(g[marker]) == colors_outer_to_inner[-1], "marker color must equal the innermost layer color"
    return legend_bbox, marker, colors_outer_to_inner, widths_outer_to_inner


def _render_rings(
    shape: tuple[int, int],
    marker: Coord,
    colors_inner_to_outer: list[int],
    widths_inner_to_outer: list[int],
) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    mr, mc = marker
    out[mr, mc] = int(colors_inner_to_outer[0])

    rows, cols = np.indices(shape)
    dist = np.maximum(np.abs(rows - mr), np.abs(cols - mc))

    radius = 0
    for color, width in zip(colors_inner_to_outer, widths_inner_to_outer):
        band = (dist >= radius + 1) & (dist <= radius + width)
        out[band] = int(color)
        radius += int(width)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Regrow the nested-rectangle legend as Chebyshev-distance square rings around the lone marker.

    Input structure:
    - The grid has a black background, one solid foreground rectangle, and one isolated marker pixel.
    - The rectangle is fully filled and decomposes into distinct nested rectangular color layers.
    - The marker color matches the innermost legend layer and is separated from the rectangle.

    Rule:
    - Read the legend from inside to outside as a sequence of layer colors and layer thicknesses.
    - Erase the legend rectangle, keep the marker pixel, and around that marker draw
      concentric square Chebyshev bands starting at distance 1.
    - The first band uses the innermost color and its thickness; each next outer band
      uses the next legend color and thickness; parts outside the grid are clipped.

    Key insight:
    - The rectangle is only a color-thickness legend: its rectangular geometry must be
      abstracted into shell widths before it can be regrown as square distance isolines.
    """
    _legend_bbox, marker, colors_outer_to_inner, widths_outer_to_inner = _analyze_input(grid)
    return _render_rings(
        grid.shape,
        marker,
        list(reversed(colors_outer_to_inner)),
        list(reversed(widths_outer_to_inner)),
    )


def _sample_inner_rect() -> tuple[int, int]:
    short_side = random.randint(1, 6)
    long_side = min(short_side + random.randint(2, 6), 8)
    if random.random() < 0.5:
        return short_side, long_side
    return long_side, short_side


def _build_legend(
    colors_outer_to_inner: list[int],
    widths_outer_to_inner: list[int],
    inner_shape: tuple[int, int],
) -> np.ndarray:
    inner_h, inner_w = inner_shape
    outer_shells = widths_outer_to_inner[:-1]
    rect_h = inner_h + 2 * sum(outer_shells)
    rect_w = inner_w + 2 * sum(outer_shells)
    rect = np.full((rect_h, rect_w), BACKGROUND, dtype=int)

    top = left = 0
    bottom = rect_h
    right = rect_w
    for idx, color in enumerate(colors_outer_to_inner):
        rect[top:bottom, left:right] = int(color)
        if idx < len(outer_shells):
            t = int(outer_shells[idx])
            top += t
            left += t
            bottom -= t
            right -= t
    return rect


def _marker_candidates(
    shape: tuple[int, int],
    total_radius: int,
    outermost_width: int,
    clipped: bool,
) -> list[Coord]:
    h, w = shape
    min_visible_outer_start = total_radius - outermost_width + 1
    cells: list[Coord] = []
    for r in range(h):
        for c in range(w):
            min_edge = min(r, h - 1 - r, c, w - 1 - c)
            max_edge = max(r, h - 1 - r, c, w - 1 - c)
            if clipped:
                if min_edge < total_radius and max_edge >= min_visible_outer_start:
                    cells.append((r, c))
            else:
                if min_edge >= total_radius:
                    cells.append((r, c))
    return cells


def _legend_positions(shape: tuple[int, int], legend_shape: tuple[int, int], marker: Coord) -> list[Coord]:
    h, w = shape
    lh, lw = legend_shape
    mr, mc = marker
    out: list[Coord] = []
    for r0 in range(h - lh + 1):
        r1 = r0 + lh - 1
        for c0 in range(w - lw + 1):
            c1 = c0 + lw - 1
            if r0 - LEGEND_MARKER_GAP <= mr <= r1 + LEGEND_MARKER_GAP and c0 - LEGEND_MARKER_GAP <= mc <= c1 + LEGEND_MARKER_GAP:
                continue
            out.append((r0, c0))
    return out


def _sample_candidate(clipped: bool) -> np.ndarray | None:
    layer_count = random.randint(*LAYER_COUNT_RANGE)
    inner_h, inner_w = _sample_inner_rect()
    inner_width = (min(inner_h, inner_w) + 1) // 2
    outer_widths = [int(random.choice(OUTER_THICKNESS_CHOICES)) for _ in range(layer_count - 1)]
    widths_outer_to_inner = outer_widths + [inner_width]
    if len(set(widths_outer_to_inner)) == 1:
        return None

    colors_outer_to_inner = random.sample(PALETTE, layer_count)
    legend = _build_legend(colors_outer_to_inner, widths_outer_to_inner, (inner_h, inner_w))
    legend_h, legend_w = legend.shape
    if abs(legend_h - legend_w) < 2:
        return None

    colors_check, widths_check = _read_layer_spec(legend)
    if colors_check != colors_outer_to_inner or widths_check != widths_outer_to_inner:
        return None

    total_radius = sum(widths_outer_to_inner)
    if total_radius > 10:
        return None

    h_min = max(MIN_GRID_SIZE, legend_h + 2, total_radius + 3)
    w_min = max(MIN_GRID_SIZE, legend_w + 2, total_radius + 3)
    if not clipped:
        h_min = max(h_min, 2 * total_radius + 1)
        w_min = max(w_min, 2 * total_radius + 1)
    if h_min > MAX_GRID_SIZE or w_min > MAX_GRID_SIZE:
        return None

    h = random.randint(h_min, min(MAX_GRID_SIZE, h_min + 8))
    w = random.randint(w_min, min(MAX_GRID_SIZE, w_min + 8))
    markers = _marker_candidates((h, w), total_radius, widths_outer_to_inner[0], clipped)
    if not markers:
        return None
    marker = random.choice(markers)

    positions = _legend_positions((h, w), legend.shape, marker)
    if not positions:
        return None
    legend_r0, legend_c0 = random.choice(positions)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[legend_r0 : legend_r0 + legend_h, legend_c0 : legend_c0 + legend_w] = legend
    grid[marker] = colors_outer_to_inner[-1]
    return grid


def _validate_candidate(grid: np.ndarray, clipped: bool) -> bool:
    try:
        legend_bbox, marker, colors_outer_to_inner, widths_outer_to_inner = _analyze_input(grid)
    except AssertionError:
        return False

    r0, c0, r1, c1 = legend_bbox
    if abs((r1 - r0 + 1) - (c1 - c0 + 1)) < 2:
        return False
    if len(set(widths_outer_to_inner)) == 1:
        return False

    widths_inner_to_outer = list(reversed(widths_outer_to_inner))
    colors_inner_to_outer = list(reversed(colors_outer_to_inner))
    total_radius = sum(widths_inner_to_outer)
    outermost_width = widths_inner_to_outer[-1]

    mr, mc = marker
    h, w = grid.shape
    min_edge = min(mr, h - 1 - mr, mc, w - 1 - mc)
    max_edge = max(mr, h - 1 - mr, mc, w - 1 - mc)
    if clipped:
        if min_edge >= total_radius:
            return False
    else:
        if min_edge < total_radius:
            return False
    if max_edge < total_radius - outermost_width + 1:
        return False

    expected = _render_rings(grid.shape, marker, colors_inner_to_outer, widths_inner_to_outer)
    try:
        out = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(out, expected):
        return False
    if np.array_equal(out, grid):
        return False

    rows, cols = np.indices(grid.shape)
    dist = np.maximum(np.abs(rows - mr), np.abs(cols - mc))
    radius = 0
    for color, width in zip(colors_inner_to_outer, widths_inner_to_outer):
        band = (dist >= radius + 1) & (dist <= radius + width)
        if not np.any(band):
            return False
        if not np.any(out[band] == color):
            return False
        radius += width
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        clipped = random.random() < 0.2
        candidate = _sample_candidate(clipped)
        if candidate is None:
            continue
        if _validate_candidate(candidate, clipped):
            return candidate
    raise RuntimeError("failed to generate a valid input for puzzle 049/4")
