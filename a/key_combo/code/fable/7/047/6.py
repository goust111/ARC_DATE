from __future__ import annotations

import random
from typing import Iterable

import numpy as np


Rect = tuple[int, int, int, int]  # (r0, c0, r1, c1), inclusive

BACKGROUND = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _rect_height(rect: Rect) -> int:
    return int(rect[2] - rect[0] + 1)


def _rect_width(rect: Rect) -> int:
    return int(rect[3] - rect[1] + 1)


def _rect_area(rect: Rect) -> int:
    return _rect_height(rect) * _rect_width(rect)


def _rect_corners(rect: Rect) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]


def _border_cells(rect: Rect) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _paint_border(grid: np.ndarray, rect: Rect, color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _three_corner_rect(coords: Iterable[tuple[int, int]]) -> Rect | None:
    pts = {(int(r), int(c)) for r, c in coords}
    if len(pts) != 3:
        return None
    rows = sorted({r for r, _ in pts})
    cols = sorted({c for _, c in pts})
    if len(rows) != 2 or len(cols) != 2:
        return None
    rect = (rows[0], cols[0], rows[1], cols[1])
    if not pts.issubset(set(_rect_corners(rect))):
        return None
    return rect


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each color that leaves exactly three rectangle corners into a full rectangular frame.

    Input structure:
    - A black background contains scattered isolated colored pixels.
    - Some colors appear exactly three times, with the three pixels forming three corners
      of an axis-aligned rectangle whose fourth corner is absent.
    - Other colors are distractors: they appear a different number of times, or their
      three pixels do not satisfy the row/column pattern of a rectangle corner set.

    Rule:
    - Remove every distractor color completely.
    - For each valid three-corner color, infer the missing fourth corner and draw the
      full one-cell-thick border of that rectangle.
    - Draw valid rectangles from larger area to smaller area so smaller rectangles
      overwrite larger ones on overlaps.

    Key insight:
    - A usable color is identified only by the hidden combinatorics of its three residual
      corner pixels: exactly two rows, exactly two columns, and one missing corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.zeros_like(g)
    rectangles: list[tuple[int, Rect, int]] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != BACKGROUND):
        coords = np.argwhere(g == int(color))
        if coords.shape[0] != 3:
            continue
        rect = _three_corner_rect((tuple(rc) for rc in coords))
        if rect is None:
            continue
        rectangles.append((_rect_area(rect), rect, int(color)))

    rectangles.sort(key=lambda item: (-int(item[0]), int(item[2])))
    for _area, rect, color in rectangles:
        _paint_border(out, rect, color)
    return out


def _random_rect(height: int, width: int, *, min_side: int = 3, max_side: int = 12) -> Rect:
    h = random.randint(min_side, min(max_side, height))
    w = random.randint(min_side, min(max_side, width))
    r0 = random.randint(0, height - h)
    c0 = random.randint(0, width - w)
    return (r0, c0, r0 + h - 1, c0 + w - 1)


def _place_independent_rect(
    height: int,
    width: int,
    used_areas: set[int],
    used_corners: set[tuple[int, int]],
    occupied_borders: set[tuple[int, int]],
) -> Rect | None:
    for _ in range(240):
        rect = _random_rect(height, width)
        if _rect_area(rect) in used_areas:
            continue
        corners = set(_rect_corners(rect))
        if corners & used_corners:
            continue
        border = _border_cells(rect)
        if border & occupied_borders:
            continue
        return rect
    return None


def _place_overlap_rect(
    base: Rect,
    height: int,
    width: int,
    used_areas: set[int],
    used_corners: set[tuple[int, int]],
    occupied_borders: set[tuple[int, int]],
) -> Rect | None:
    r0, c0, r1, c1 = base
    base_h = _rect_height(base)
    base_w = _rect_width(base)
    options = ["share_top", "share_bottom", "share_left", "share_right"]
    random.shuffle(options)

    for mode in options:
        for _ in range(80):
            if mode in {"share_top", "share_bottom"}:
                if base_h < 4 or base_w < 5:
                    break
                h = random.randint(3, base_h - 1)
                w = random.randint(3, base_w - 2)
                left_min = c0 + 1
                left_max = c1 - w
                if left_min > left_max:
                    continue
                left = random.randint(left_min, left_max)
                top = r0 if mode == "share_top" else r1 - h + 1
                rect = (top, left, top + h - 1, left + w - 1)
            else:
                if base_w < 4 or base_h < 5:
                    break
                h = random.randint(3, base_h - 2)
                w = random.randint(3, base_w - 1)
                top_min = r0 + 1
                top_max = r1 - h
                if top_min > top_max:
                    continue
                top = random.randint(top_min, top_max)
                left = c0 if mode == "share_left" else c1 - w + 1
                rect = (top, left, top + h - 1, left + w - 1)

            if _rect_area(rect) in used_areas:
                continue
            corners = set(_rect_corners(rect))
            if corners & used_corners:
                continue
            border = _border_cells(rect)
            overlap = border & occupied_borders
            if not overlap:
                continue
            if len(overlap) > max(1, len(border) // 3):
                continue
            return rect
    return None


def _choose_missing_corner(rect: Rect) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    corners = _rect_corners(rect)
    missing = random.choice(corners)
    remain = [pt for pt in corners if pt != missing]
    return missing, remain


def _add_invalid_three_noise(
    grid: np.ndarray,
    color: int,
    used_cells: set[tuple[int, int]],
) -> bool:
    height, width = map(int, grid.shape)
    modes = ["row", "col", "diag"]
    random.shuffle(modes)

    for mode in modes:
        for _ in range(60):
            if mode == "row" and width >= 3:
                r = random.randrange(height)
                cols = random.sample(range(width), 3)
                cells = [(r, c) for c in cols]
            elif mode == "col" and height >= 3:
                c = random.randrange(width)
                rows = random.sample(range(height), 3)
                cells = [(r, c) for r in rows]
            else:
                if height < 3 or width < 3:
                    continue
                rows = random.sample(range(height), 3)
                cols = random.sample(range(width), 3)
                cells = list(zip(rows, cols))
            if any(cell in used_cells for cell in cells):
                continue
            if _three_corner_rect(cells) is not None:
                continue
            for r, c in cells:
                grid[r, c] = int(color)
                used_cells.add((r, c))
            return True
    return False


def _add_count_noise(
    grid: np.ndarray,
    color: int,
    used_cells: set[tuple[int, int]],
) -> bool:
    height, width = map(int, grid.shape)
    count = random.choice([1, 2, 4, 5])
    available = [(r, c) for r in range(height) for c in range(width) if (r, c) not in used_cells]
    if len(available) < count:
        return False

    for _ in range(80):
        cells = random.sample(available, count)
        for r, c in cells:
            grid[r, c] = int(color)
            used_cells.add((r, c))
        return True
    return False


def _build_expected(shape: tuple[int, int], specs: list[tuple[Rect, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rect, color in sorted(specs, key=lambda item: (-_rect_area(item[0]), int(item[1]))):
        _paint_border(out, rect, color)
    return out


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(14, 28)
    width = random.randint(14, 28)
    overlap_case = random.random() < 0.55
    valid_count = random.choices([1, 2, 3, 4], weights=[1, 4, 4, 2])[0]
    if overlap_case:
        valid_count = max(2, valid_count)

    used_areas: set[int] = set()
    used_corners: set[tuple[int, int]] = set()
    occupied_borders: set[tuple[int, int]] = set()
    rects: list[Rect] = []

    if overlap_case:
        base = None
        for _ in range(120):
            candidate = _random_rect(height, width, min_side=6, max_side=13)
            if _rect_area(candidate) in used_areas:
                continue
            rects = [candidate]
            used_areas.add(_rect_area(candidate))
            used_corners |= set(_rect_corners(candidate))
            occupied_borders |= _border_cells(candidate)
            base = candidate
            break
        if base is None:
            return None

        overlap_rect = _place_overlap_rect(base, height, width, used_areas, used_corners, occupied_borders)
        if overlap_rect is None:
            return None
        rects.append(overlap_rect)
        used_areas.add(_rect_area(overlap_rect))
        used_corners |= set(_rect_corners(overlap_rect))
        occupied_borders |= _border_cells(overlap_rect)

    while len(rects) < valid_count:
        rect = _place_independent_rect(height, width, used_areas, used_corners, occupied_borders)
        if rect is None:
            return None
        rects.append(rect)
        used_areas.add(_rect_area(rect))
        used_corners |= set(_rect_corners(rect))
        occupied_borders |= _border_cells(rect)

    max_noise = min(4, 9 - len(rects))
    if max_noise <= 0:
        return None
    noise_count = random.randint(1, max_noise)
    colors = random.sample(PALETTE, len(rects) + noise_count)
    rect_colors = colors[: len(rects)]
    noise_colors = colors[len(rects) :]

    grid = np.zeros((height, width), dtype=int)
    used_input_cells: set[tuple[int, int]] = set()
    specs = list(zip(rects, rect_colors))
    for rect, color in specs:
        _missing, visible = _choose_missing_corner(rect)
        for r, c in visible:
            if (r, c) in used_input_cells:
                return None
            grid[r, c] = int(color)
            used_input_cells.add((r, c))

    has_invalid_three = False
    has_other_count = False
    for idx, color in enumerate(noise_colors):
        want_invalid_three = idx == 0 or (not has_invalid_three and random.random() < 0.65)
        ok = (
            _add_invalid_three_noise(grid, color, used_input_cells)
            if want_invalid_three
            else _add_count_noise(grid, color, used_input_cells)
        )
        if not ok:
            ok = (
                _add_count_noise(grid, color, used_input_cells)
                if want_invalid_three
                else _add_invalid_three_noise(grid, color, used_input_cells)
            )
            want_invalid_three = not want_invalid_three
        if not ok:
            return None
        has_invalid_three |= want_invalid_three
        has_other_count |= not want_invalid_three

    if not has_other_count and noise_colors:
        extra_color_choices = [c for c in PALETTE if c not in colors]
        if extra_color_choices:
            color = random.choice(extra_color_choices)
            if not _add_count_noise(grid, color, used_input_cells):
                return None

    expected = _build_expected((height, width), specs)
    if not np.array_equal(main(grid), expected):
        return None
    if overlap_case:
        overlap_cells = _border_cells(rects[0]) & _border_cells(rects[1])
        if not overlap_cells:
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 6")
