from __future__ import annotations

from collections import Counter
from pathlib import Path
import importlib.util
import sys
from typing import Iterable

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")

try:
    if _ARC_UTILS_PATH.exists():
        spec = importlib.util.spec_from_file_location("_arc_utils_shared", _ARC_UTILS_PATH)
        if spec is not None and spec.loader is not None:
            _arc_utils = importlib.util.module_from_spec(spec)
            sys.modules.setdefault("_arc_utils_shared", _arc_utils)
            spec.loader.exec_module(_arc_utils)
            Color = _arc_utils.Color
            get_bg = _arc_utils.get_bg
        else:
            raise ImportError("failed to load arc_utils spec")
    else:
        raise ImportError("arc_utils.py is unavailable")
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        GRAY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9

        BACKGROUND = 0
        NOT_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


Rect = tuple[int, int, int, int, int, int]  # color, r0, r1, c0, c1, area
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GEN_RETRIES = 600


def _randint_inclusive(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _rects_overlap(a: Rect, b: Rect) -> bool:
    return _intervals_overlap(a[1], a[2], b[1], b[2]) and _intervals_overlap(a[3], a[4], b[3], b[4])


def _parse_rectangles(grid: np.ndarray) -> list[Rect]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    rectangles: list[Rect] = []

    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue

        horizontal = coords[(coords[:, 0] == 0) | (coords[:, 0] == h - 1)]
        vertical = coords[(coords[:, 1] == 0) | (coords[:, 1] == w - 1)]
        assert horizontal.size > 0 and vertical.size > 0, "each color must have one horizontal and one vertical edge segment"
        assert len(horizontal) + len(vertical) == len(coords), "foreground must stay on the border"

        h_rows = set(map(int, horizontal[:, 0]))
        v_cols = set(map(int, vertical[:, 1]))
        assert len(h_rows) == 1 and len(v_cols) == 1, "segments must lie on one border side per orientation"

        h_cols = sorted(int(c) for c in horizontal[:, 1])
        v_rows = sorted(int(r) for r in vertical[:, 0])
        assert h_cols == list(range(h_cols[0], h_cols[-1] + 1)), "horizontal segment must be contiguous"
        assert v_rows == list(range(v_rows[0], v_rows[-1] + 1)), "vertical segment must be contiguous"
        assert 0 < h_cols[0] and h_cols[-1] < w - 1, "horizontal segment must avoid corners"
        assert 0 < v_rows[0] and v_rows[-1] < h - 1, "vertical segment must avoid corners"

        r0, r1 = v_rows[0], v_rows[-1]
        c0, c1 = h_cols[0], h_cols[-1]
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        rectangles.append((color, r0, r1, c0, c1, area))

    areas = [rect[5] for rect in rectangles]
    assert len(areas) == len(set(areas)), "rectangle areas must be unique"
    return rectangles


def _expected_from_rectangles(rectangles: Iterable[Rect], shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    cover = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    ordered = sorted(rectangles, key=lambda rect: rect[5], reverse=True)

    for _color, r0, r1, c0, c1, _area in ordered:
        cover[r0 : r1 + 1, c0 : c1 + 1] += 1

    overlap_mask = cover >= 2
    for color, r0, r1, c0, c1, _area in ordered:
        region = overlap_mask[r0 : r1 + 1, c0 : c1 + 1]
        out_region = out[r0 : r1 + 1, c0 : c1 + 1]
        out_region[region] = int(color)
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Rebuild hidden rectangles from matching edge projections and keep only their overlap fragments.

    Input structure:
    - The grid has a black background and an empty interior.
    - Colored segments appear only on the four outer borders.
    - Each color appears exactly twice: once on a horizontal border and once on a vertical border.

    Rule:
    - Pair the two same-color border segments.
    - The horizontal segment gives that hidden rectangle's column span, and the vertical segment gives its row span.
    - In the output, erase every border segment and leave only cells covered by at least two hidden rectangles.
    - When several hidden rectangles cover the same visible overlap cell, paint it with the color of the smallest-area covering rectangle.

    Key insight:
    - The colored border marks are just projections of rectangles that never appear in the input interior; only the stacked overlap scraps become visible, with smaller rectangles sitting above larger ones.
    """
    rectangles = _parse_rectangles(np.asarray(input_grid, dtype=int))
    return _expected_from_rectangles(rectangles, np.asarray(input_grid).shape)


def _random_rect(rng: np.random.Generator, h: int, w: int) -> Rect:
    max_height = min(h - 2, max(2, h // 2))
    max_width = min(w - 2, max(2, w // 2))
    height = _randint_inclusive(rng, 1, max_height)
    width = _randint_inclusive(rng, 1, max_width)
    r0 = _randint_inclusive(rng, 1, h - 1 - height)
    c0 = _randint_inclusive(rng, 1, w - 1 - width)
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    return (-1, r0, r1, c0, c1, height * width)


def _make_overlapping_rect(base: Rect, rng: np.random.Generator, h: int, w: int) -> Rect:
    _, br0, br1, bc0, bc1, _ = base
    for _ in range(80):
        rect = _random_rect(rng, h, w)
        if _rects_overlap(base, rect):
            return rect
    raise RuntimeError("failed to make overlapping rectangle")


def _make_contained_rect(base: Rect, rng: np.random.Generator) -> Rect:
    _, br0, br1, bc0, bc1, _ = base
    if br0 == br1 and bc0 == bc1:
        return base
    for _ in range(80):
        r0 = _randint_inclusive(rng, br0, br1)
        r1 = _randint_inclusive(rng, r0, br1)
        c0 = _randint_inclusive(rng, bc0, bc1)
        c1 = _randint_inclusive(rng, c0, bc1)
        rect = (-1, r0, r1, c0, c1, (r1 - r0 + 1) * (c1 - c0 + 1))
        if rect[5] < base[5]:
            return rect
    raise RuntimeError("failed to make contained rectangle")


def _assign_two_sides(rng: np.random.Generator, spans: list[tuple[int, int]]) -> list[int] | None:
    order = sorted(range(len(spans)), key=lambda idx: (spans[idx][0], spans[idx][1]))
    ends = [-10**9, -10**9]
    assignment = [-1] * len(spans)

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        start, end = spans[idx]
        side_order = [0, 1]
        rng.shuffle(side_order)
        for side in side_order:
            if start > ends[side]:
                old_end = ends[side]
                ends[side] = end
                assignment[idx] = side
                if backtrack(pos + 1):
                    return True
                assignment[idx] = -1
                ends[side] = old_end
        return False

    if backtrack(0):
        return assignment
    return None


def _make_mode_rectangles(rng: np.random.Generator, h: int, w: int, count: int) -> list[Rect]:
    mode = rng.choice(["pair", "chain", "contain"], p=[0.4, 0.35, 0.25])
    rects: list[Rect] = []

    if mode == "contain":
        outer = _random_rect(rng, h, w)
        inner = _make_contained_rect(outer, rng)
        rects = [outer, inner]
        while len(rects) < count:
            candidate = _random_rect(rng, h, w)
            if rng.random() < 0.6:
                candidate = _make_overlapping_rect(rects[rng.integers(0, len(rects))], rng, h, w)
            rects.append(candidate)
    elif mode == "chain":
        first = _random_rect(rng, h, w)
        second = _make_overlapping_rect(first, rng, h, w)
        rects = [first, second]
        while len(rects) < count:
            rects.append(_make_overlapping_rect(rects[-1], rng, h, w))
    else:
        first = _random_rect(rng, h, w)
        second = _make_overlapping_rect(first, rng, h, w)
        rects = [first, second]
        while len(rects) < count:
            candidate = _random_rect(rng, h, w)
            if rng.random() < 0.45:
                candidate = _make_overlapping_rect(rects[rng.integers(0, len(rects))], rng, h, w)
            rects.append(candidate)

    return rects


def _valid_rectangles(rects: list[Rect], h: int, w: int) -> bool:
    areas = [rect[5] for rect in rects]
    if len(areas) != len(set(areas)):
        return False

    row_spans = [(rect[1], rect[2]) for rect in rects]
    col_spans = [(rect[3], rect[4]) for rect in rects]
    if _assign_two_sides(np.random.default_rng(0), row_spans) is None:
        return False
    if _assign_two_sides(np.random.default_rng(1), col_spans) is None:
        return False

    cover = np.zeros((h, w), dtype=int)
    for _color, r0, r1, c0, c1, _area in rects:
        cover[r0 : r1 + 1, c0 : c1 + 1] += 1

    overlap_cells = int(np.count_nonzero(cover >= 2))
    if overlap_cells == 0:
        return False
    interior_area = max(1, (h - 2) * (w - 2))
    if overlap_cells > interior_area // 2:
        return False

    if not any(rect[1] == rect[2] or rect[3] == rect[4] for rect in rects) and len(rects) >= 3:
        # Bias the generator toward thin edge cases often enough for training.
        return False

    return True


def _generate_case(rng: np.random.Generator, *, return_rectangles: bool = False):
    for _ in range(GEN_RETRIES):
        h = _randint_inclusive(rng, 10, 24)
        w = _randint_inclusive(rng, 10, 24)
        count = int(rng.choice([2, 3, 4, 5], p=[0.2, 0.35, 0.3, 0.15]))
        try:
            base_rects = _make_mode_rectangles(rng, h, w, count)
        except RuntimeError:
            continue
        if not _valid_rectangles(base_rects, h, w):
            continue

        row_assign = _assign_two_sides(rng, [(rect[1], rect[2]) for rect in base_rects])
        col_assign = _assign_two_sides(rng, [(rect[3], rect[4]) for rect in base_rects])
        if row_assign is None or col_assign is None:
            continue

        colors = list(rng.choice(PALETTE, size=count, replace=False))
        rects = [
            (int(colors[i]), base_rects[i][1], base_rects[i][2], base_rects[i][3], base_rects[i][4], base_rects[i][5])
            for i in range(count)
        ]

        grid = np.zeros((h, w), dtype=int)
        for idx, (color, r0, r1, c0, c1, _area) in enumerate(rects):
            border_row = 0 if col_assign[idx] == 0 else h - 1
            border_col = 0 if row_assign[idx] == 0 else w - 1
            grid[border_row, c0 : c1 + 1] = color
            grid[r0 : r1 + 1, border_col] = color

        if not np.array_equal(main(grid), _expected_from_rectangles(rects, (h, w))):
            continue
        if return_rectangles:
            return grid, rects
        return grid
    raise RuntimeError("failed to generate valid puzzle input")


def generate_input() -> np.ndarray:
    return _generate_case(np.random.default_rng(), return_rectangles=False)
