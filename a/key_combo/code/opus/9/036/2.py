from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 5
FRAME = 1
MARKER_COLORS = (2, 3, 4)

MIN_INNER = 3
MAX_INNER = 6
MIN_OUTER = MIN_INNER + 2
MAX_OUTER = MAX_INNER + 2

GRID_MIN = 12
GRID_MAX = 16
GEN_ATTEMPTS = 1200

Rect = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _border_cells(rect: Rect) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _strict_interior_cells(rect: Rect) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}


def _inside_bbox(rect: Rect, cell: tuple[int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = rect
    return r0 <= r <= r1 and c0 <= c <= c1


def _inside_interior(rect: Rect, cell: tuple[int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = rect
    return r0 < r < r1 and c0 < c < c1


def _geometric_inner_area(rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 - 1) * (c1 - c0 - 1)


def _strict_overlap(a: Rect, b: Rect) -> bool:
    return not (a[2] <= b[0] + 1 or b[2] <= a[0] + 1 or a[3] <= b[1] + 1 or b[3] <= a[1] + 1)


def _actual_area(grid: np.ndarray, rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    return int(np.count_nonzero(grid[r0 + 1 : r1, c0 + 1 : c1] == BACKGROUND))


def _enumerate_rectangles(grid: np.ndarray) -> list[Rect]:
    h, w = grid.shape
    rects: list[Rect] = []
    for r0 in range(0, h - MIN_OUTER + 1):
        for r1 in range(r0 + MIN_OUTER - 1, min(h - 1, r0 + MAX_OUTER - 1) + 1):
            if not (MIN_INNER <= r1 - r0 - 1 <= MAX_INNER):
                continue
            for c0 in range(0, w - MIN_OUTER + 1):
                for c1 in range(c0 + MIN_OUTER - 1, min(w - 1, c0 + MAX_OUTER - 1) + 1):
                    if not (MIN_INNER <= c1 - c0 - 1 <= MAX_INNER):
                        continue
                    if grid[r0, c0] != FRAME or grid[r0, c1] != FRAME or grid[r1, c0] != FRAME or grid[r1, c1] != FRAME:
                        continue
                    if not np.all(grid[r0, c0 : c1 + 1] == FRAME):
                        continue
                    if not np.all(grid[r1, c0 : c1 + 1] == FRAME):
                        continue
                    if not np.all(grid[r0 : r1 + 1, c0] == FRAME):
                        continue
                    if not np.all(grid[r0 : r1 + 1, c1] == FRAME):
                        continue
                    if np.count_nonzero(grid[r0 + 1 : r1, c0 + 1 : c1] == BACKGROUND) == 0:
                        continue
                    rects.append((r0, c0, r1, c1))
    return rects


def _marker_matches_rect(grid: np.ndarray, marker: tuple[int, int], rect: Rect) -> bool:
    if _inside_bbox(rect, marker):
        return False
    border = _border_cells(rect)
    for nr, nc in _neighbors4(*marker):
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == FRAME and (nr, nc) in border:
            return True
    return False


def _solve_from_rects(grid: np.ndarray, rects: list[Rect], fill_colors: dict[Rect, int]) -> np.ndarray:
    areas = {rect: _actual_area(grid, rect) for rect in rects}
    assert len(set(areas.values())) == len(rects), "frame areas must be distinct"

    out = np.array(grid, copy=True)
    for rect in sorted(rects, key=lambda item: areas[item]):
        r0, c0, r1, c1 = rect
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        src = grid[r0 + 1 : r1, c0 + 1 : c1]
        fill_mask = (src == BACKGROUND) & (patch == BACKGROUND)
        patch[fill_mask] = int(fill_colors[rect])

    marker_mask = (grid != BACKGROUND) & (grid != FRAME)
    out[marker_mask] = BACKGROUND
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow blue rectangle from its adjacent external marker, resolving interior overlaps by smaller area.

    Input structure:
    - The grid uses grey as background, blue as the shared one-pixel-thick frame color,
      and 2-3 single-pixel external markers in distinct non-blue colors.
    - Each blue object is an axis-aligned hollow rectangular frame whose interior size is
      between 3x3 and 6x6, and some frame interiors overlap in space.
    - Every marker lies outside the frames and is 4-neighbor adjacent to exactly one frame border.

    Rule:
    - Detect all hollow blue rectangles and assign each one the color of its unique adjacent marker.
    - Fill only the grey pixels inside each frame; leave every blue border pixel unchanged.
    - If a grey pixel belongs to multiple frame interiors, color it with the fill color of the
      frame whose original grey interior area is smaller, then erase all marker pixels back to grey.

    Key insight:
    - The marker does not indicate where to paint directly; it selects a frame, and overlap pixels
      are decided by comparing each frame's original interior grey area rather than by draw order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert _get_bg(g) == BACKGROUND, "expected grey background"

    rects = _enumerate_rectangles(g)
    assert 2 <= len(rects) <= 3, "expected 2-3 hollow rectangles"

    marker_coords = np.argwhere((g != BACKGROUND) & (g != FRAME))
    assert marker_coords.shape[0] == len(rects), "expected one marker per frame"
    marker_colors = [int(g[r, c]) for r, c in marker_coords]
    assert len(set(marker_colors)) == len(marker_colors), "marker colors must be distinct"

    fill_colors: dict[Rect, int] = {}
    for r, c in marker_coords:
        marker = (int(r), int(c))
        matches = [rect for rect in rects if _marker_matches_rect(g, marker, rect)]
        assert len(matches) == 1, "each marker must touch exactly one frame"
        fill_colors[matches[0]] = int(g[r, c])

    assert len(fill_colors) == len(rects), "every frame must receive exactly one fill color"
    return _solve_from_rects(g, rects, fill_colors)


def _render_borders(shape: tuple[int, int], rects: list[Rect]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for rect in rects:
        for r, c in _border_cells(rect):
            grid[r, c] = FRAME
    return grid


def _sample_rect(h: int, w: int, anchor: Rect | None) -> Rect | None:
    outer_h = random.randint(MIN_OUTER, MAX_OUTER)
    outer_w = random.randint(MIN_OUTER, MAX_OUTER)
    r_low, r_high = 1, h - outer_h - 1
    c_low, c_high = 1, w - outer_w - 1
    if r_low > r_high or c_low > c_high:
        return None

    if anchor is not None:
        ar0, ac0, ar1, ac1 = anchor
        r_low = max(r_low, ar0 - outer_h + 3)
        r_high = min(r_high, ar1 - 2)
        c_low = max(c_low, ac0 - outer_w + 3)
        c_high = min(c_high, ac1 - 2)
        if r_low > r_high or c_low > c_high:
            return None

    r0 = random.randint(r_low, r_high)
    c0 = random.randint(c_low, c_high)
    return (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)


def _visible_overlap_cells(grid: np.ndarray, rects: list[Rect]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] != BACKGROUND:
                continue
            cover = sum(1 for rect in rects if _inside_interior(rect, (r, c)))
            if cover >= 2:
                cells.append((r, c))
    return cells


def _assign_markers(border_grid: np.ndarray, rects: list[Rect]) -> dict[Rect, tuple[tuple[int, int], int]] | None:
    colors = random.sample(list(MARKER_COLORS), len(rects))
    marker_info: dict[Rect, tuple[tuple[int, int], int]] = {}

    for rect, color in zip(random.sample(rects, len(rects)), colors):
        r0, c0, r1, c1 = rect
        candidates = []
        if r0 > 0:
            candidates.extend((r0 - 1, c) for c in range(c0 + 1, c1))
        if r1 + 1 < border_grid.shape[0]:
            candidates.extend((r1 + 1, c) for c in range(c0 + 1, c1))
        if c0 > 0:
            candidates.extend((r, c0 - 1) for r in range(r0 + 1, r1))
        if c1 + 1 < border_grid.shape[1]:
            candidates.extend((r, c1 + 1) for r in range(r0 + 1, r1))
        random.shuffle(candidates)

        for cell in candidates:
            if border_grid[cell] != BACKGROUND:
                continue
            if any(_inside_bbox(other, cell) for other in rects if other != rect):
                continue
            matches = [other for other in rects if _marker_matches_rect(border_grid, cell, other)]
            if len(matches) != 1 or matches[0] != rect:
                continue
            marker_info[rect] = (cell, color)
            border_grid[cell] = color
            break
        if rect not in marker_info:
            return None

    return marker_info


def _has_area_inversion(grid: np.ndarray, rects: list[Rect]) -> bool:
    actual = {rect: _actual_area(grid, rect) for rect in rects}
    geometric = {rect: _geometric_inner_area(rect) for rect in rects}
    for i, rect_a in enumerate(rects):
        for rect_b in rects[i + 1 :]:
            if actual[rect_a] == actual[rect_b] or geometric[rect_a] == geometric[rect_b]:
                continue
            overlap = False
            for r, c in _strict_interior_cells(rect_a) & _strict_interior_cells(rect_b):
                if grid[r, c] == BACKGROUND:
                    overlap = True
                    break
            if not overlap:
                continue
            actual_winner = rect_a if actual[rect_a] < actual[rect_b] else rect_b
            geometric_winner = rect_a if geometric[rect_a] < geometric[rect_b] else rect_b
            if actual_winner != geometric_winner:
                return True
    return False


def _semantic_checks(grid: np.ndarray, rects: list[Rect], marker_info: dict[Rect, tuple[tuple[int, int], int]]) -> bool:
    detected = sorted(_enumerate_rectangles(grid))
    if detected != sorted(rects):
        return False

    areas = [_actual_area(grid, rect) for rect in rects]
    if any(area <= 0 for area in areas) or len(set(areas)) != len(areas):
        return False

    overlap_cells = _visible_overlap_cells(grid, rects)
    if len(overlap_cells) < 2:
        return False

    fill_colors = {rect: info[1] for rect, info in marker_info.items()}
    expected = _solve_from_rects(grid, rects, fill_colors)

    for rect, color in fill_colors.items():
        if int(np.count_nonzero(expected == color)) < 2:
            return False
        visible = 0
        for cell in _strict_interior_cells(rect):
            if expected[cell] == color:
                visible += 1
        if visible == 0:
            return False

    for marker, _color in marker_info.values():
        if expected[marker] != BACKGROUND:
            return False

    try:
        solved = main(grid)
    except AssertionError:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    frame_count = 2 if random.random() < 0.55 else 3
    require_inversion = frame_count == 3 and random.random() < 0.45

    rects: list[Rect] = []
    for idx in range(frame_count):
        placed = False
        for _ in range(120):
            anchor = None
            if idx > 0:
                anchor = random.choice(rects) if random.random() < 0.85 else None
            rect = _sample_rect(h, w, anchor)
            if rect is None or rect in rects:
                continue
            trial = rects + [rect]
            if idx > 0 and not any(_strict_overlap(a, b) for i, a in enumerate(trial) for b in trial[i + 1 :]):
                continue
            rects = trial
            placed = True
            break
        if not placed:
            return None

    border_grid = _render_borders((h, w), rects)
    if sorted(_enumerate_rectangles(border_grid)) != sorted(rects):
        return None

    if len(_visible_overlap_cells(border_grid, rects)) < 2:
        return None

    if require_inversion and not _has_area_inversion(border_grid, rects):
        return None

    marker_grid = border_grid.copy()
    marker_info = _assign_markers(marker_grid, rects)
    if marker_info is None:
        return None

    if not _semantic_checks(marker_grid, rects, marker_info):
        return None
    return marker_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
