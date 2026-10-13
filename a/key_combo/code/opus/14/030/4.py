from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox, colors_present, get_bg


Rect = tuple[int, int, int, int]

BACKGROUND = 0
MARKER = 7
PALETTE = (1, 2, 3, 4, 5, 6, 8, 9)
GEN_ATTEMPTS = 600


def _area(rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _center(rect: Rect) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    return (r0 + r1 + 1) // 2, (c0 + c1 + 1) // 2


def _contains(outer: Rect, inner: Rect) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _overlap_area(a: Rect, b: Rect) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _point_in_rect(rect: Rect, point: tuple[int, int]) -> bool:
    r, c = point
    return rect[0] <= r <= rect[2] and rect[1] <= c <= rect[3]


def _point_in_strict_interior(rect: Rect, point: tuple[int, int]) -> bool:
    r, c = point
    return rect[0] < r < rect[2] and rect[1] < c < rect[3]


def _point_on_boundary(rect: Rect, point: tuple[int, int]) -> bool:
    r, c = point
    return _point_in_rect(rect, point) and (r in (rect[0], rect[2]) or c in (rect[1], rect[3]))


def _dist(marker: tuple[int, int], rect: Rect) -> int:
    cr, cc = _center(rect)
    return abs(marker[0] - cr) + abs(marker[1] - cc)


def _draw_rect(out: np.ndarray, rect: Rect, color: int) -> None:
    r0, c0, r1, c1 = rect
    out[r0 : r1 + 1, c0 : c1 + 1] = color


def _ordered_layers(rects: list[tuple[int, Rect]], marker: tuple[int, int]) -> list[tuple[int, Rect]]:
    return sorted(rects, key=lambda item: (-_dist(marker, item[1]), -_area(item[1])))


def _stack_layers(rects: list[tuple[int, Rect]], shape: tuple[int, int], marker: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for color, rect in _ordered_layers(rects, marker):
        _draw_rect(out, rect, color)
    return out


def _render(rects: list[tuple[int, Rect]], shape: tuple[int, int], marker: tuple[int, int]) -> np.ndarray:
    out = _stack_layers(rects, shape, marker)
    out[marker] = MARKER
    return out


def _render_area_sorted(
    rects: list[tuple[int, Rect]], shape: tuple[int, int], marker: tuple[int, int], reverse: bool
) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for color, rect in sorted(rects, key=lambda item: _area(item[1]), reverse=reverse):
        _draw_rect(out, rect, color)
    out[marker] = MARKER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete sparse rectangle samples and stack the filled rectangles by marker-centered Manhattan distance.

    Input structure:
    - The background is black.
    - Each non-black, non-orange foreground color appears as sparse pixels lying on the boundary of one axis-aligned rectangle.
    - Exactly one orange marker pixel appears anywhere on the grid and is not part of the rectangle colors.

    Rule:
    - For each rectangle color, take the bounding box of all pixels of that color as the completed filled rectangle.
    - Round each rectangle center to the nearest grid cell, then compute its Manhattan distance to the orange marker.
    - Draw the rectangles from larger distance to smaller distance; if two distances tie, draw the larger-area rectangle first so the smaller one is painted later and covers it.
    - Keep the orange marker pixel on the top layer and leave every cell outside all completed rectangles black.

    Key insight:
    - The sparse pixels only identify rectangle extents; the visible overlap pattern is determined by distance-to-marker ordering rather than by size or nesting alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    assert bg == BACKGROUND, "expected black background"

    marker_coords = np.argwhere(g == MARKER)
    assert marker_coords.shape == (1, 2), "expected exactly one orange marker"
    marker = tuple(map(int, marker_coords[0]))

    rects: list[tuple[int, Rect]] = []
    for color in colors_present(g, exclude={bg, MARKER}):
        rect = bbox(g == color)
        assert rect is not None, "every rectangle color must be present"
        rects.append((int(color), tuple(map(int, rect))))

    assert 3 <= len(rects) <= 5, "expected 3-5 rectangle colors"
    areas = [_area(rect) for _, rect in rects]
    assert len(set(areas)) == len(areas), "rectangle areas must be unique"
    return _render(rects, g.shape, marker)


def _random_rect_inside(outer: Rect, min_h: int = 3, min_w: int = 3) -> Rect | None:
    max_top = min(3, outer[2] - outer[0] - min_h)
    max_left = min(3, outer[3] - outer[1] - min_w)
    if max_top < 1 or max_left < 1:
        return None

    for _ in range(120):
        top = random.randint(1, max_top)
        bottom = random.randint(1, min(3, outer[2] - outer[0] - min_h - top + 1))
        left = random.randint(1, max_left)
        right = random.randint(1, min(3, outer[3] - outer[1] - min_w - left + 1))
        inner = (outer[0] + top, outer[1] + left, outer[2] - bottom, outer[3] - right)
        if inner[2] - inner[0] + 1 < min_h or inner[3] - inner[1] + 1 < min_w:
            continue
        if _center(inner) != _center(outer):
            return inner
    return None


def _random_partial_rect(base: Rect, shape: tuple[int, int], existing: list[Rect]) -> Rect | None:
    h, w = shape
    base_h = base[2] - base[0] + 1
    base_w = base[3] - base[1] + 1
    min_h = 4
    min_w = 4
    max_h = min(9, max(min_h, min(h - 1, base_h + 2)))
    max_w = min(9, max(min_w, min(w - 1, base_w + 2)))
    if max_h < min_h or max_w < min_w:
        return None

    for _ in range(240):
        hh = random.randint(min_h, max_h)
        ww = random.randint(min_w, max_w)
        r0_lo = max(0, base[0] - hh + 2)
        r0_hi = min(h - hh, base[2] - 1)
        c0_lo = max(0, base[1] - ww + 2)
        c0_hi = min(w - ww, base[3] - 1)
        if r0_lo > r0_hi or c0_lo > c0_hi:
            continue
        r0 = random.randint(r0_lo, r0_hi)
        c0 = random.randint(c0_lo, c0_hi)
        rect = (r0, c0, r0 + hh - 1, c0 + ww - 1)
        if _contains(base, rect) or _contains(rect, base):
            continue
        ov = _overlap_area(rect, base)
        if ov < 4 or ov >= int(0.75 * min(_area(rect), _area(base))):
            continue
        if any(_overlap_area(rect, other) > 0 and _contains(other, rect) for other in existing):
            continue
        return rect
    return None


def _boundary_points(rect: Rect) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return sorted(pts)


def _pick_samples(rect: Rect, blocked: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = rect
    top = [(r0, c) for c in range(c0 + 1, c1)] or [(r0, c0), (r0, c1)]
    bottom = [(r1, c) for c in range(c0 + 1, c1)] or [(r1, c0), (r1, c1)]
    left = [(r, c0) for r in range(r0 + 1, r1)] or [(r0, c0), (r1, c0)]
    right = [(r, c1) for r in range(r0 + 1, r1)] or [(r0, c1), (r1, c1)]

    for _ in range(120):
        anchors = [
            random.choice(top),
            random.choice(bottom),
            random.choice(left),
            random.choice(right),
        ]
        if any(pt in blocked for pt in anchors):
            continue
        unique = list(dict.fromkeys(anchors))
        if len(unique) < 4:
            continue
        boundary = [pt for pt in _boundary_points(rect) if pt not in blocked and pt not in unique]
        random.shuffle(boundary)
        extra_count = random.randint(0, min(4, len(boundary)))
        samples = unique + boundary[:extra_count]
        if 4 <= len(samples) <= 8:
            return samples
    return None


def _relation_summary(rects: list[Rect]) -> tuple[int, int]:
    nested = 0
    partial = 0
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _contains(rects[i], rects[j]) or _contains(rects[j], rects[i]):
                nested += 1
            elif _overlap_area(rects[i], rects[j]) > 0:
                partial += 1
    return nested, partial


def _is_full_nesting_chain(rects: list[Rect]) -> bool:
    ordered = sorted(rects, key=_area, reverse=True)
    return all(_contains(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1))


def _variable_stacks(rects: list[tuple[int, Rect]], shape: tuple[int, int]) -> bool:
    seen: set[bytes] = set()
    for r in range(shape[0]):
        for c in range(shape[1]):
            seen.add(_stack_layers(rects, shape, (r, c)).tobytes())
            if len(seen) >= 2:
                return True
    return False


def _pick_marker(rects: list[tuple[int, Rect]], shape: tuple[int, int]) -> tuple[int, int] | None:
    all_cells = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
    random.shuffle(all_cells)
    desire_tie = random.random() < 0.35
    modes = random.choices(["inside", "outside", "any"], weights=[4, 4, 2], k=4)
    modes.append("any")

    for mode in modes:
        for marker in all_cells:
            if any(_point_on_boundary(rect, marker) for _, rect in rects):
                continue
            if mode == "inside" and not any(_point_in_strict_interior(rect, marker) for _, rect in rects):
                continue
            if mode == "outside" and any(_point_in_rect(rect, marker) for _, rect in rects):
                continue

            dists = [_dist(marker, rect) for _, rect in rects]
            has_tie = len(set(dists)) < len(dists)
            if desire_tie and not has_tie:
                continue
            if not desire_tie and has_tie and random.random() < 0.65:
                continue

            actual = _render(rects, shape, marker)
            if np.array_equal(actual, _render_area_sorted(rects, shape, marker, reverse=True)):
                continue
            if np.array_equal(actual, _render_area_sorted(rects, shape, marker, reverse=False)):
                continue
            return marker
    return None


def _build_rectangles(shape: tuple[int, int], count: int) -> list[Rect] | None:
    h, w = shape
    max_outer_h = min(10, h - 3)
    max_outer_w = min(10, w - 3)
    if max_outer_h < 7 or max_outer_w < 7:
        return None

    outer_h = random.randint(7, max_outer_h)
    outer_w = random.randint(7, max_outer_w)
    outer = (
        random.randint(1, h - outer_h - 1),
        random.randint(1, w - outer_w - 1),
        0,
        0,
    )
    outer = (outer[0], outer[1], outer[0] + outer_h - 1, outer[1] + outer_w - 1)

    inner = _random_rect_inside(outer)
    if inner is None:
        return None

    rects = [outer, inner]
    partial = _random_partial_rect(outer, shape, rects)
    if partial is None:
        return None
    rects.append(partial)
    smallest = inner

    while len(rects) < count:
        if random.random() < 0.55:
            nxt = _random_rect_inside(smallest)
            if nxt is not None:
                rects.append(nxt)
                smallest = nxt
                continue
        base = random.choice(rects)
        nxt = _random_partial_rect(base, shape, rects)
        if nxt is not None:
            rects.append(nxt)
            continue
        nxt = _random_rect_inside(smallest)
        if nxt is not None:
            rects.append(nxt)
            smallest = nxt
            continue
        return None

    areas = [_area(rect) for rect in rects]
    if len(set(areas)) != len(areas):
        return None

    nested_pairs, partial_pairs = _relation_summary(rects)
    if nested_pairs == 0 and partial_pairs == 0:
        return None
    if partial_pairs == 0:
        return None
    if _is_full_nesting_chain(rects):
        return None

    union = np.zeros(shape, dtype=bool)
    for rect in rects:
        union[rect[0] : rect[2] + 1, rect[1] : rect[3] + 1] = True
    ratio = float(union.mean())
    if not (0.18 <= ratio <= 0.48):
        return None

    centers = [_center(rect) for rect in rects]
    if len(set(centers)) != len(centers):
        return None
    return rects


def _validate_input(grid: np.ndarray) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert int(get_bg(grid)) == BACKGROUND
    marker_coords = np.argwhere(grid == MARKER)
    assert marker_coords.shape == (1, 2)

    rect_colors = colors_present(grid, exclude={BACKGROUND, MARKER})
    assert 3 <= len(rect_colors) <= 5
    rects: list[Rect] = []
    for color in rect_colors:
        mask = grid == color
        rect = bbox(mask)
        assert rect is not None
        rect = tuple(map(int, rect))
        rects.append(rect)
        pixels = [tuple(map(int, pt)) for pt in np.argwhere(mask)]
        assert 4 <= len(pixels) <= 8
        sides = {
            "top" for r, _ in pixels if r == rect[0]
        } | {
            "bottom" for r, _ in pixels if r == rect[2]
        } | {
            "left" for _, c in pixels if c == rect[1]
        } | {
            "right" for _, c in pixels if c == rect[3]
        }
        assert {"top", "bottom", "left", "right"} <= sides
        assert all(_point_on_boundary(rect, pt) for pt in pixels)

    areas = [_area(rect) for rect in rects]
    assert len(set(areas)) == len(areas)
    _, partial_pairs = _relation_summary(rects)
    assert partial_pairs >= 1
    assert not _is_full_nesting_chain(rects)


def _validate_semantics(grid: np.ndarray) -> None:
    marker = tuple(map(int, np.argwhere(grid == MARKER)[0]))
    rects = [
        (int(color), tuple(map(int, bbox(grid == color))))
        for color in colors_present(grid, exclude={BACKGROUND, MARKER})
    ]
    actual = _render(rects, grid.shape, marker)
    assert np.array_equal(main(grid), actual)
    assert not np.array_equal(actual, _render_area_sorted(rects, grid.shape, marker, reverse=True))
    assert not np.array_equal(actual, _render_area_sorted(rects, grid.shape, marker, reverse=False))

    visible_patterns: set[bytes] = set()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] != BACKGROUND:
                continue
            visible_patterns.add(_stack_layers(rects, grid.shape, (r, c)).tobytes())
            if len(visible_patterns) >= 2:
                return
    raise AssertionError("marker movement must change the visible stacked result")


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(13, 18), random.randint(13, 18))
    count = random.randint(3, 5)
    rects_only = _build_rectangles(shape, count)
    if rects_only is None:
        return None

    colors = random.sample(PALETTE, k=count)
    rects = list(zip(colors, rects_only))
    marker = _pick_marker(rects, shape)
    if marker is None:
        return None

    grid = np.full(shape, BACKGROUND, dtype=int)
    blocked = {marker}
    for color, rect in rects:
        samples = _pick_samples(rect, blocked)
        if samples is None:
            return None
        for r, c in samples:
            grid[r, c] = color
        blocked.update(samples)

    grid[marker] = MARKER

    try:
        _validate_input(grid)
        _validate_semantics(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid distance-stacking rectangle input")
