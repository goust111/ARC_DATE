from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
Segment = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _shape_signature(points: list[Coord]) -> tuple[int, int, tuple[Coord, ...]]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    norm = tuple(sorted((r - r0, c - c0) for r, c in points))
    return (r1 - r0 + 1, c1 - c0 + 1, norm)


def _mask_signature(mask: np.ndarray) -> tuple[int, int, tuple[Coord, ...]]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return _shape_signature(coords)


def _segments(length: int, separators: list[int]) -> list[Segment]:
    sep_set = set(int(x) for x in separators)
    parts: list[Segment] = []
    start: int | None = None
    for i in range(length):
        if i in sep_set:
            if start is not None:
                parts.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        parts.append((start, length))
    return parts


def _region_outlier(region: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    comps = _components(region != bg)
    assert 2 <= len(comps) <= 4

    by_shape: Counter[tuple[int, int, tuple[Coord, ...]]] = Counter()
    entries: list[tuple[list[Coord], tuple[int, int, tuple[Coord, ...]], int]] = []
    for comp in comps:
        colors = {int(region[r, c]) for r, c in comp}
        assert len(colors) == 1
        color = colors.pop()
        sig = _shape_signature(comp)
        by_shape[sig] += 1
        entries.append((comp, sig, color))

    assert len(by_shape) == 2
    rare = [sig for sig, count in by_shape.items() if count == 1]
    common = [sig for sig, count in by_shape.items() if count > 1]
    assert len(rare) == 1 and len(common) == 1

    odd_sig = rare[0]
    for comp, sig, color in entries:
        if sig == odd_sig:
            return [(r, c, color) for r, c in comp]
    raise AssertionError("missing outlier object")


def _divider_candidate(
    grid: np.ndarray, bg: int, color: int
) -> tuple[list[int], list[int], list[Segment], list[Segment]] | None:
    h, w = grid.shape
    full_rows = [r for r in range(h) if bool(np.all(grid[r, :] == color))]
    full_cols = [c for c in range(w) if bool(np.all(grid[:, c] == color))]
    if not full_rows or not full_cols:
        return None

    line_mask = np.zeros((h, w), dtype=bool)
    line_mask[full_rows, :] = True
    line_mask[:, full_cols] = True
    if bool(np.any((grid == color) & ~line_mask)):
        return None

    row_parts = _segments(h, full_rows)
    col_parts = _segments(w, full_cols)
    if len(row_parts) < 2 or len(col_parts) < 2:
        return None

    try:
        for r0, r1 in row_parts:
            for c0, c1 in col_parts:
                _region_outlier(grid[r0:r1, c0:c1], bg)
    except AssertionError:
        return None
    return full_rows, full_cols, row_parts, col_parts


def _find_dividers(
    grid: np.ndarray, bg: int
) -> tuple[int, list[Segment], list[Segment]]:
    candidates: list[tuple[int, list[Segment], list[Segment]]] = []
    for value in np.unique(grid):
        color = int(value)
        if color == bg:
            continue
        info = _divider_candidate(grid, bg, color)
        if info is not None:
            _rows, _cols, row_parts, col_parts = info
            candidates.append((color, row_parts, col_parts))
    assert len(candidates) == 1
    divider_color, row_parts, col_parts = candidates[0]
    return divider_color, row_parts, col_parts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the unique-shape object from each divider-defined region.

    The input is partitioned by one full-row/full-column divider color.
    Each region contains several small 4-connected monochrome objects, and
    exactly one object's translation-normalized shape appears only once inside
    that region. The output removes divider lines and all non-outlier objects,
    while preserving each selected object's color, shape, and local position
    inside its region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    _divider, row_parts, col_parts = _find_dividers(g, bg)

    out_h = sum(r1 - r0 for r0, r1 in row_parts)
    out_w = sum(c1 - c0 for c0, c1 in col_parts)
    out = np.full((out_h, out_w), bg, dtype=int)

    out_r = 0
    for r0, r1 in row_parts:
        out_c = 0
        for c0, c1 in col_parts:
            region = g[r0:r1, c0:c1]
            for rr, cc, color in _region_outlier(region, bg):
                out[out_r + rr, out_c + cc] = color
            out_c += c1 - c0
        out_r += r1 - r0

    return out


# --------------- generator ---------------

_BASE_SHAPES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
]

REGION_ROW_RANGE = (2, 3)
REGION_COL_RANGE = (2, 3)
REGION_HEIGHT_RANGE = (6, 9)
REGION_WIDTH_RANGE = (6, 9)
OBJECT_COUNT_CHOICES = (3, 4)
GENERATION_RETRIES = 500


def _random_shape() -> np.ndarray:
    shape = random.choice(_BASE_SHAPES)
    transformed = np.rot90(shape, random.randint(0, 3))
    if random.random() < 0.5:
        transformed = np.fliplr(transformed)
    return np.array(transformed, dtype=bool, copy=True)


def _sample_distinct_shapes() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(100):
        majority = _random_shape()
        odd = _random_shape()
        if _mask_signature(majority) != _mask_signature(odd):
            return majority, odd
    raise RuntimeError("failed to sample distinct shapes")


def _candidate_positions(forbidden: np.ndarray, shape: np.ndarray) -> list[Coord]:
    rh, rw = forbidden.shape
    sh, sw = shape.shape
    positions: list[Coord] = []
    for r in range(1, rh - sh):
        for c in range(1, rw - sw):
            if bool(forbidden[r : r + sh, c : c + sw].any()):
                continue
            positions.append((r, c))
    return positions


def _paint_shape(canvas: np.ndarray, shape: np.ndarray, top: int, left: int, color: int) -> None:
    sh, sw = shape.shape
    patch = canvas[top : top + sh, left : left + sw]
    patch[shape] = color


def _place_region(region_h: int, region_w: int, bg: int, colors: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    object_count = random.choice(OBJECT_COUNT_CHOICES)
    majority_shape, odd_shape = _sample_distinct_shapes()

    shapes = [majority_shape.copy() for _ in range(object_count - 1)] + [odd_shape.copy()]
    odd_index = object_count - 1
    object_colors = random.sample(colors, object_count)

    region = np.full((region_h, region_w), bg, dtype=int)
    expected = np.full((region_h, region_w), bg, dtype=int)
    forbidden = np.zeros((region_h, region_w), dtype=bool)

    place_order = sorted(
        range(object_count),
        key=lambda idx: (shapes[idx].shape[0] * shapes[idx].shape[1], int(shapes[idx].sum())),
        reverse=True,
    )

    placements: dict[int, Coord] = {}
    for idx in place_order:
        positions = _candidate_positions(forbidden, shapes[idx])
        if not positions:
            return None
        top, left = random.choice(positions)
        placements[idx] = (top, left)
        sh, sw = shapes[idx].shape
        r0 = max(0, top - 1)
        r1 = min(region_h, top + sh + 1)
        c0 = max(0, left - 1)
        c1 = min(region_w, left + sw + 1)
        forbidden[r0:r1, c0:c1] = True

    for idx, (top, left) in placements.items():
        color = int(object_colors[idx])
        _paint_shape(region, shapes[idx], top, left, color)
        if idx == odd_index:
            _paint_shape(expected, shapes[idx], top, left, color)

    return region, expected


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, 9)
    divider = random.choice([c for c in range(10) if c != bg])
    object_colors = [c for c in range(10) if c not in {bg, divider}]

    n_region_rows = random.randint(*REGION_ROW_RANGE)
    n_region_cols = random.randint(*REGION_COL_RANGE)
    row_heights = [random.randint(*REGION_HEIGHT_RANGE) for _ in range(n_region_rows)]
    col_widths = [random.randint(*REGION_WIDTH_RANGE) for _ in range(n_region_cols)]

    grid_h = sum(row_heights) + (n_region_rows - 1)
    grid_w = sum(col_widths) + (n_region_cols - 1)
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    expected = np.full((sum(row_heights), sum(col_widths)), bg, dtype=int)

    div_r = 0
    for height in row_heights[:-1]:
        div_r += height
        grid[div_r, :] = divider
        div_r += 1

    div_c = 0
    for width in col_widths[:-1]:
        div_c += width
        grid[:, div_c] = divider
        div_c += 1

    in_r = 0
    out_r = 0
    for row_idx, region_h in enumerate(row_heights):
        in_c = 0
        out_c = 0
        for col_idx, region_w in enumerate(col_widths):
            placed = _place_region(region_h, region_w, bg, object_colors)
            if placed is None:
                return None
            region, odd_only = placed
            grid[in_r : in_r + region_h, in_c : in_c + region_w] = region
            expected[out_r : out_r + region_h, out_c : out_c + region_w] = odd_only
            in_c += region_w + (0 if col_idx == n_region_cols - 1 else 1)
            out_c += region_w
        in_r += region_h + (0 if row_idx == n_region_rows - 1 else 1)
        out_r += region_h

    if _get_bg(grid) != bg:
        return None

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
