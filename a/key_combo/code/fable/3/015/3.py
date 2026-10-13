from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

try:
    from arc_utils import bbox_coords, colors_present, get_bg
except ModuleNotFoundError:
    _file = Path(__file__).resolve()
    _arc_utils_path = None
    for parent in _file.parents:
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.is_file():
            _arc_utils_path = candidate.parent
            break
    if _arc_utils_path is None:
        raise
    sys.path.insert(0, str(_arc_utils_path))
    from arc_utils import bbox_coords, colors_present, get_bg


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
PALETTE_SIZE = len(PALETTE)


class Rect(NamedTuple):
    r0: int
    c0: int
    r1: int
    c1: int


def _rect_area(rect: Rect) -> int:
    return (rect.r1 - rect.r0 + 1) * (rect.c1 - rect.c0 + 1)


def _rects_intersect(a: Rect, b: Rect) -> bool:
    return not (
        a.r1 < b.r0
        or b.r1 < a.r0
        or a.c1 < b.c0
        or b.c1 < a.c0
    )


def _marker_pairs(rect: Rect) -> tuple[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]]:
    return (
        ((rect.r0, rect.c0), (rect.r1, rect.c1)),
        ((rect.r0, rect.c1), (rect.r1, rect.c0)),
    )


def _sample_rectangle(height: int, width: int, used_areas: set[int], *, min_span: int = 1, max_tries: int = 5000) -> Rect | None:
    for _ in range(max_tries):
        r0 = random.randint(0, height - min_span - 1)
        r1 = random.randint(r0 + min_span, height - 1)
        c0 = random.randint(0, width - min_span - 1)
        c1 = random.randint(c0 + min_span, width - 1)
        rect = Rect(r0, c0, r1, c1)
        area = _rect_area(rect)
        if area in used_areas:
            continue
        return rect
    return None


def _sample_rectangle_disjoint_from(
    height: int,
    width: int,
    existing: list[Rect],
    used_areas: set[int],
    *,
    min_span: int = 1,
    max_tries: int = 5000,
) -> Rect | None:
    for _ in range(max_tries):
        rect = _sample_rectangle(height, width, used_areas, min_span=min_span, max_tries=1)
        if rect is None:
            return None
        if any(_rects_intersect(rect, other) for other in existing):
            continue
        return rect
    return None


def _sample_rectangle_overlapping(
    height: int,
    width: int,
    anchors: list[Rect],
    used_areas: set[int],
    *,
    must_overlap_two: bool = False,
    pair_anchor: Rect | None = None,
    max_tries: int = 8000,
) -> Rect | None:
    if not anchors:
        return _sample_rectangle(height, width, used_areas, max_tries=max_tries)

    for _ in range(max_tries):
        rect = _sample_rectangle(height, width, used_areas, max_tries=1)
        if rect is None:
            return None
        if pair_anchor is not None and not _rects_intersect(rect, pair_anchor):
            continue
        if must_overlap_two:
            if not all(_rects_intersect(rect, a) for a in anchors):
                continue
            return rect
        if any(_rects_intersect(rect, a) for a in anchors):
            return rect
    return None


def _sample_rectangle_inside(outer: Rect, used_areas: set[int]) -> Rect | None:
    if outer.r1 - outer.r0 < 2 or outer.c1 - outer.c0 < 2:
        return None

    for _ in range(5000):
        r0 = random.randint(outer.r0 + 1, outer.r1 - 1)
        r1 = random.randint(r0 + 1, outer.r1)
        c0 = random.randint(outer.c0 + 1, outer.c1 - 1)
        c1 = random.randint(c0 + 1, outer.c1)
        if r1 > outer.r1 - 1 or c1 > outer.c1 - 1:
            continue
        rect = Rect(r0, c0, r1, c1)
        area = _rect_area(rect)
        if area in used_areas:
            continue
        return rect
    return None


def _coverage(rects: list[Rect], h: int, w: int) -> np.ndarray:
    cnt = np.zeros((h, w), dtype=np.uint8)
    for rect in rects:
        cnt[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] += 1
    return cnt


def _build_mode_rectangles(height: int, width: int, count: int, mode: str) -> list[Rect] | None:
    if count < 2:
        return None

    used_areas: set[int] = set()

    if mode == "sparse":
        rects: list[Rect] = []
        for _ in range(300):
            if len(rects) >= count:
                break
            rect = _sample_rectangle_disjoint_from(height, width, rects, used_areas, min_span=random.randint(1, 4), max_tries=200)
            if rect is None:
                break
            rects.append(rect)
            used_areas.add(_rect_area(rect))
        if len(rects) != count:
            return None
        return rects

    if mode == "pair":
        rects = []
        first = _sample_rectangle(height, width, used_areas, min_span=random.randint(1, 4))
        if first is None:
            return None
        rects.append(first)
        used_areas.add(_rect_area(first))

        second = _sample_rectangle_overlapping(
            height,
            width,
            anchors=[first],
            used_areas=used_areas,
            must_overlap_two=False,
            max_tries=2000,
        )
        if second is None:
            return None
        rects.append(second)
        used_areas.add(_rect_area(second))

        while len(rects) < count:
            # keep the overlap structure readable, place remaining rectangles disjoint.
            rect = _sample_rectangle_disjoint_from(height, width, rects[:2], used_areas, min_span=1, max_tries=2000)
            if rect is None:
                return None
            rects.append(rect)
            used_areas.add(_rect_area(rect))
        return rects

    if mode == "triple":
        # Build forced three-coverage structure: outer -> inner -> crossing third.
        for _ in range(500):
            outer = _sample_rectangle(height, width, used_areas, min_span=random.randint(3, 6))
            if outer is None:
                return None
            area = _rect_area(outer)
            if area in used_areas:
                continue
            rects = [outer]
            used = used_areas | {area}

            inner = _sample_rectangle_inside(outer, used)
            if inner is None:
                continue
            rects.append(inner)
            used.add(_rect_area(inner))

            third = _sample_rectangle_overlapping(
                height,
                width,
                anchors=[inner, outer],
                used_areas=used,
                must_overlap_two=True,
                max_tries=2000,
            )
            if third is None:
                continue
            rects.append(third)
            used.add(_rect_area(third))

            while len(rects) < count:
                # additional rectangles mostly disjoint to keep triple structure visible.
                rect = _sample_rectangle_disjoint_from(height, width, rects[:3], used, min_span=1, max_tries=3000)
                if rect is None:
                    break
                rects.append(rect)
                used.add(_rect_area(rect))
            if len(rects) != count:
                return None
            return rects
        return None

    # dense: force many intersections with optional multi-layer overlap.
    rects = []
    for _ in range(500):
        if not rects:
            first = _sample_rectangle(height, width, used_areas, min_span=random.randint(1, 6))
            if first is None:
                continue
            rects.append(first)
            used_areas.add(_rect_area(first))
            continue

        if random.random() < 0.7:
            rect = _sample_rectangle_overlapping(height, width, anchors=rects, used_areas=used_areas, max_tries=3000)
        else:
            rect = _sample_rectangle(height, width, used_areas, min_span=random.randint(1, 4), max_tries=1500)

        if rect is None:
            continue
        rects.append(rect)
        used_areas.add(_rect_area(rect))

        if len(rects) == count:
            return rects
    return None


def _mode_passes(rects: list[Rect], mode: str) -> bool:
    if not rects:
        return False
    h = max(r.r1 for r in rects) + 1
    w = max(r.c1 for r in rects) + 1
    cnt = _coverage(rects, h, w)
    odd_pixels = int(np.count_nonzero(cnt % 2 == 1))
    if odd_pixels == 0:
        return False

    even_overlap = np.any((cnt > 0) & (cnt % 2 == 0))
    has_three = np.any(cnt >= 3)

    if mode == "sparse":
        return cnt.max() <= 1 and odd_pixels > 0 and not even_overlap
    if mode == "pair":
        return even_overlap and cnt.max() <= 2
    if mode == "triple":
        return has_three
    # dense
    return even_overlap and odd_pixels > 0 and cnt.max() >= 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the implicit rectangle from each same-color pair, and reveal cells with odd coverage by the smallest covering rectangle.

    Input structure:
    - A single dominant background color.
    - Each non-background color appears exactly twice, forming opposite corners of an implicit solid rectangle.
    - The two same-color pixels are on different rows and different columns; rectangle areas are pairwise unique.

    Rule:
    - For each color pair, define a solid rectangle using its two pixels as opposite corners.
    - Count coverage for each cell by all defined rectangles.
    - Cells covered an even number of times (including zero) are black.
    - Cells covered an odd number of times take the color of the smallest-area rectangle among all rectangles covering that cell.

    Key insight:
    - Coverage parity creates cancellative overlap, so only odd-count cells survive, and parity ties are resolved by area-order, not stacking order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert colors, "expected at least one non-background color"

    rects: list[tuple[int, int, int, int, int, int]] = []
    seen_colors = set[int]()
    seen_areas = set[int]()

    for color in colors:
        assert color not in seen_colors, f"duplicate color marker: {color}"
        seen_colors.add(color)

        coords = np.argwhere(g == color)
        assert coords.shape[0] == 2, f"each color must appear exactly twice: color {color}"
        r0, c0, r1, c1 = bbox_coords(coords)
        assert r0 != r1, f"marker pair must not share row: color {color}"
        assert c0 != c1, f"marker pair must not share column: color {color}"
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert area not in seen_areas, "rectangle areas must be unique"
        seen_areas.add(area)
        rects.append((r0, c0, r1, c1, color, area))

    h, w = g.shape
    coverage = np.zeros((h, w), dtype=np.uint8)
    for r0, c0, r1, c1, _color, _area in rects:
        coverage[r0 : r1 + 1, c0 : c1 + 1] += 1

    odd = (coverage & 1).astype(bool)
    out = np.full_like(g, bg)
    inf_area = np.iinfo(np.int32).max
    best_area = np.full((h, w), inf_area, dtype=np.int32)
    best_color = np.zeros((h, w), dtype=int)

    for r0, c0, r1, c1, color, area in sorted(rects, key=lambda t: t[5]):
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if not odd[r, c]:
                    continue
                if area < best_area[r, c]:
                    best_area[r, c] = area
                    best_color[r, c] = color

    out[odd] = best_color[odd]
    return out


def _place_markers(height: int, width: int, rects: list[Rect], colors: list[int]) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()
    for rect, color in zip(rects, colors):
        placed = False
        for m0, m1 in _marker_pairs(rect):
            if m0 in occupied or m1 in occupied:
                continue
            grid[m0] = color
            grid[m1] = color
            occupied.add(m0)
            occupied.add(m1)
            placed = True
            break
        if not placed:
            return None
    return grid


def _build_candidate() -> np.ndarray | None:
    mode = random.choices(
        population=["sparse", "pair", "triple", "dense"],
        weights=[0.20, 0.30, 0.25, 0.25],
        k=1,
    )[0]

    height = random.randint(12, 30)
    width = random.randint(12, 30)
    max_colors = min(8, (height * width) // 40, PALETTE_SIZE)
    color_count = random.randint(3, max(3, max_colors))
    if max_colors < 2:
        return None
    color_count = min(color_count, max_colors)
    if color_count < 2:
        color_count = 2

    rect_count = color_count
    for _ in range(200):
        rects = _build_mode_rectangles(height, width, rect_count, mode)
        if rects is None:
            continue
        if not _mode_passes(rects, mode):
            continue

        areas = [_rect_area(r) for r in rects]
        if len(set(areas)) != len(areas):
            continue

        # Ensure a visible signal and enough background.
        cnt = _coverage(rects, height, width)
        if np.count_nonzero(cnt == 0) < 4:
            continue

        colors = random.sample(PALETTE, rect_count)
        grid = _place_markers(height, width, rects, colors)
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue

        # Keep only non-degenerate transforms with meaningful overlap branches.
        if mode == "dense" and (out == BG).all():
            continue
        if out.shape != grid.shape:
            continue
        if out.size == 0:
            continue
        if tuple(np.unique(out).tolist()) == (BG,):
            continue

        if (mode == "triple" and np.max(_coverage(rects, height, width)) < 3):
            continue
        if mode == "pair" and not np.any(_coverage(rects, height, width) == 2):
            continue
        if mode == "sparse":
            cnt = _coverage(rects, height, width)
            if np.any(cnt > 1):
                continue

        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 3.py input")

