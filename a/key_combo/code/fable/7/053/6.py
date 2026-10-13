from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 1000
Rect = tuple[int, int, int, int, int, int]  # color, r0, r1, c0, c1, area


def _contiguous_span(values: np.ndarray) -> tuple[int, int] | None:
    if values.size == 0:
        return None
    ordered = sorted(set(map(int, values)))
    if ordered != list(range(ordered[0], ordered[-1] + 1)):
        return None
    return int(ordered[0]), int(ordered[-1])


def _intersection_area(a: Rect, b: Rect) -> int:
    _ca, ar0, ar1, ac0, ac1, _aa = a
    _cb, br0, br1, bc0, bc1, _ba = b
    h = min(ar1, br1) - max(ar0, br0) + 1
    w = min(ac1, bc1) - max(ac0, bc0) + 1
    return max(0, h) * max(0, w)


def _parse_projection_grid(grid: np.ndarray) -> tuple[int, list[Rect]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)

    for band in range(2, min(height, width)):
        if np.any(g[:band, :band] != BG):
            continue
        if np.any(g[band:, band:] != BG):
            continue

        top = g[:band, band:]
        left = g[band:, :band]
        top_colors = {int(x) for x in np.unique(top) if int(x) != BG}
        left_colors = {int(x) for x in np.unique(left) if int(x) != BG}
        if not top_colors or top_colors != left_colors or len(top_colors) != band:
            continue

        rects: list[Rect] = []
        used_top_rows: set[int] = set()
        used_left_cols: set[int] = set()
        ok = True

        for color in top_colors:
            top_coords = np.argwhere(top == color)
            left_coords = np.argwhere(left == color)
            if top_coords.size == 0 or left_coords.size == 0:
                ok = False
                break

            top_rows = set(map(int, top_coords[:, 0]))
            left_cols = set(map(int, left_coords[:, 1]))
            if len(top_rows) != 1 or len(left_cols) != 1:
                ok = False
                break

            top_row = int(next(iter(top_rows)))
            left_col = int(next(iter(left_cols)))
            if top_row in used_top_rows or left_col in used_left_cols:
                ok = False
                break
            used_top_rows.add(top_row)
            used_left_cols.add(left_col)

            col_span = _contiguous_span(top_coords[:, 1])
            row_span = _contiguous_span(left_coords[:, 0])
            if col_span is None or row_span is None:
                ok = False
                break

            c0, c1 = col_span
            r0, r1 = row_span
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            rects.append((int(color), int(r0), int(r1), int(c0), int(c1), int(area)))

        if not ok:
            continue

        areas = [rect[5] for rect in rects]
        if len(set(areas)) != len(areas):
            continue
        if not any(_intersection_area(a, b) > 0 for i, a in enumerate(rects) for b in rects[i + 1 :]):
            continue
        return band, rects

    raise AssertionError("could not parse a valid paired projection-band layout")


def _paint_rectangles(base: np.ndarray, band: int, rects: list[Rect]) -> np.ndarray:
    out = np.array(base, dtype=int, copy=True)
    for color, r0, r1, c0, c1, _area in sorted(rects, key=lambda rect: rect[5], reverse=True):
        out[band + r0 : band + r1 + 1, band + c0 : band + c1 + 1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct solid rectangles from paired top/left color projections and resolve overlaps by area.

    Input structure:
    - The grid contains a top projection band and a left projection band, with a black square at their corner.
    - Each non-black color appears once as a contiguous horizontal segment in the top band and once as a contiguous vertical segment in the left band.
    - The interior region is entirely black in the input; rectangle areas are all distinct and at least one pair overlaps.

    Rule:
    - For each color, take the row interval from its left-band segment and the column interval from its top-band segment.
    - Paint the corresponding solid rectangle into the interior region.
    - Preserve the projection bands, and in overlaps let the smaller-area rectangle overwrite the larger-area rectangle.

    Key insight:
    - The visible strips are orthogonal projections of hidden rectangles, and the layer order is determined only by rectangle area rather than color id or spatial order.
    """
    g = np.asarray(grid, dtype=int)
    band, rects = _parse_projection_grid(g)
    return _paint_rectangles(g, band, rects)


def _owner_map(rects: list[Rect], inner_h: int, inner_w: int) -> tuple[np.ndarray, np.ndarray]:
    cover_count = np.zeros((inner_h, inner_w), dtype=int)
    best_area = np.full((inner_h, inner_w), 10_000, dtype=int)
    owner = np.zeros((inner_h, inner_w), dtype=int)

    for color, r0, r1, c0, c1, area in rects:
        cover_count[r0 : r1 + 1, c0 : c1 + 1] += 1
        area_window = best_area[r0 : r1 + 1, c0 : c1 + 1]
        mask = area < area_window
        area_window[mask] = area
        owner_window = owner[r0 : r1 + 1, c0 : c1 + 1]
        owner_window[mask] = int(color)
    return cover_count, owner


def _valid_rectangles(rects: list[Rect], inner_h: int, inner_w: int) -> bool:
    areas = [rect[5] for rect in rects]
    if len(set(areas)) != len(areas):
        return False

    overlaps = [_intersection_area(a, b) for i, a in enumerate(rects) for b in rects[i + 1 :]]
    if not overlaps or max(overlaps) < 4:
        return False
    if not any(area == 0 for area in overlaps):
        return False

    cover_count, owner = _owner_map(rects, inner_h, inner_w)
    union = cover_count > 0
    overlap = cover_count >= 2
    union_pixels = int(np.count_nonzero(union))
    overlap_pixels = int(np.count_nonzero(overlap))
    if union_pixels == 0 or not (4 <= overlap_pixels <= max(8, union_pixels // 3)):
        return False

    visible_colors = {int(x) for x in np.unique(owner[union]) if int(x) != BG}
    if len(visible_colors) != len(rects):
        return False

    for color, r0, r1, c0, c1, _area in rects:
        visible = owner[r0 : r1 + 1, c0 : c1 + 1] == color
        if int(np.count_nonzero(visible)) < 2:
            return False

    mixed_dimension = False
    for i, a in enumerate(rects):
        ah = a[2] - a[1] + 1
        aw = a[4] - a[3] + 1
        for b in rects[i + 1 :]:
            if _intersection_area(a, b) == 0:
                continue
            bh = b[2] - b[1] + 1
            bw = b[4] - b[3] + 1
            if a[5] < b[5] and (ah > bh or aw > bw):
                mixed_dimension = True
            if b[5] < a[5] and (bh > ah or bw > aw):
                mixed_dimension = True
    return mixed_dimension


def _sample_rectangles(count: int, inner_h: int, inner_w: int) -> list[Rect] | None:
    if inner_h < 10 or inner_w < 10:
        return None

    core_h = random.randint(2, 4)
    core_w = random.randint(2, 4)
    if inner_h - core_h < 6 or inner_w - core_w < 6:
        return None
    core_r0 = random.randint(3, inner_h - core_h - 3)
    core_c0 = random.randint(3, inner_w - core_w - 3)
    core_r1 = core_r0 + core_h - 1
    core_c1 = core_c0 + core_w - 1

    rect_specs: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        for _attempt in range(120):
            up = random.randint(0, min(5, core_r0))
            down = random.randint(0, min(5, inner_h - 1 - core_r1))
            left = random.randint(0, min(5, core_c0))
            right = random.randint(0, min(5, inner_w - 1 - core_c1))
            spec = (core_r0 - up, core_r1 + down, core_c0 - left, core_c1 + right)
            area = (spec[1] - spec[0] + 1) * (spec[3] - spec[2] + 1)
            if spec in rect_specs:
                continue
            if any((r1 - r0 + 1) * (c1 - c0 + 1) == area for r0, r1, c0, c1 in rect_specs):
                continue
            rect_specs.append(spec)
            break
        else:
            return None

    random.shuffle(rect_specs)

    extra_disjoint = random.randint(1, max(1, count - 2))
    for _ in range(extra_disjoint):
        placed = False
        for _attempt in range(150):
            h = random.randint(3, min(8, inner_h - 1))
            w = random.randint(3, min(8, inner_w - 1))
            r0 = random.randint(0, inner_h - h)
            c0 = random.randint(0, inner_w - w)
            cand = (r0, r0 + h - 1, c0, c0 + w - 1)
            area = h * w
            if any((rr1 - rr0 + 1) * (cc1 - cc0 + 1) == area for rr0, rr1, cc0, cc1 in rect_specs):
                continue
            intersections = [
                max(0, min(cand[1], rr1) - max(cand[0], rr0) + 1)
                * max(0, min(cand[3], cc1) - max(cand[2], cc0) + 1)
                for rr0, rr1, cc0, cc1 in rect_specs
            ]
            if sum(area > 0 for area in intersections) not in (0, 1):
                continue
            rect_specs[random.randrange(len(rect_specs))] = cand
            placed = True
            break
        if not placed:
            return None

    colors = random.sample(PALETTE, count)
    rects = []
    for color, (r0, r1, c0, c1) in zip(colors, rect_specs):
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        rects.append((int(color), int(r0), int(r1), int(c0), int(c1), int(area)))
    return rects if _valid_rectangles(rects, inner_h, inner_w) else None


def _render_input(rects: list[Rect], band: int, inner_h: int, inner_w: int) -> tuple[np.ndarray, list[int], list[int]]:
    grid = np.zeros((band + inner_h, band + inner_w), dtype=int)
    top_lanes = list(range(band))
    left_lanes = list(range(band))
    random.shuffle(top_lanes)
    random.shuffle(left_lanes)

    for (color, r0, r1, c0, c1, _area), top_lane, left_lane in zip(rects, top_lanes, left_lanes):
        grid[top_lane, band + c0 : band + c1 + 1] = int(color)
        grid[band + r0 : band + r1 + 1, left_lane] = int(color)
    return grid, top_lanes, left_lanes


def _orders_are_nondegenerate(rects: list[Rect], top_lanes: list[int], left_lanes: list[int]) -> bool:
    by_area_small = [color for color, *_rest in sorted(rects, key=lambda rect: rect[5])]
    by_area_large = list(reversed(by_area_small))
    by_color = [color for color, *_rest in sorted(rects)]
    by_top = [rects[i][0] for i in np.argsort(np.array(top_lanes))]
    by_left = [rects[i][0] for i in np.argsort(np.array(left_lanes))]
    bad_orders = {tuple(by_area_small), tuple(by_area_large)}
    return tuple(by_color) not in bad_orders and tuple(by_top) not in bad_orders and tuple(by_left) not in bad_orders


def _make_attempt() -> np.ndarray | None:
    count = random.choice((3, 4, 4, 5))
    band = count
    inner_h = random.randint(12, 25 - band)
    inner_w = random.randint(12, 25 - band)
    if band + inner_h > 30 or band + inner_w > 30:
        return None

    rects = _sample_rectangles(count, inner_h, inner_w)
    if rects is None:
        return None

    grid, top_lanes, left_lanes = _render_input(rects, band, inner_h, inner_w)
    if not _orders_are_nondegenerate(rects, top_lanes, left_lanes):
        return None

    try:
        parsed_band, parsed_rects = _parse_projection_grid(grid)
        actual = main(grid)
        expected = _paint_rectangles(grid, parsed_band, parsed_rects)
    except AssertionError:
        return None

    if parsed_band != band or not np.array_equal(actual, expected):
        return None
    if np.any(grid[:band, :band] != BG) or np.any(grid[band:, band:] != BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid projection-rectangle instance")
