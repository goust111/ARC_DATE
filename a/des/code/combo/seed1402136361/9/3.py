from __future__ import annotations

import random

import numpy as np


MAROON = 9
GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (12, 22)
RECT_COUNT_RANGE = (2, 5)
MIN_SIDE_SPAN = 2
MAX_BORDER_FILL_RATIO = 0.36
MIN_UNIQUE_BORDER_CELLS = 6
MAX_GENERATION_ATTEMPTS = 400

Rect = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.argmax(np.bincount(values, minlength=10)))


def _border_points(rect: Rect) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _all_corners(rect: Rect) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}


def _flip_vertical(rect: Rect, height: int) -> Rect:
    r0, c0, r1, c1 = rect
    return (height - 1 - r1, c0, height - 1 - r0, c1)


def _flip_horizontal(rect: Rect, width: int) -> Rect:
    r0, c0, r1, c1 = rect
    return (r0, width - 1 - c1, r1, width - 1 - c0)


def _transpose_rect(rect: Rect) -> Rect:
    r0, c0, r1, c1 = rect
    return (c0, r0, c1, r1)


def _normalize_rect(rect: Rect) -> Rect:
    r0, c0, r1, c1 = rect
    return (min(r0, r1), min(c0, c1), max(r0, r1), max(c0, c1))


def _apply_random_flips(rects: list[Rect], height: int, width: int, rng: random.Random) -> list[Rect]:
    result = list(rects)
    if rng.random() < 0.5:
        result = [_flip_vertical(rect, height) for rect in result]
    if rng.random() < 0.5:
        result = [_flip_horizontal(rect, width) for rect in result]
    return [_normalize_rect(rect) for rect in result]


def _rectangles_from_grid(grid: np.ndarray) -> list[tuple[int, Rect]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = [color for color in np.unique(g) if int(color) != bg]
    rectangles: list[tuple[int, Rect]] = []
    for color in colors:
        coords = np.argwhere(g == int(color))
        assert coords.shape[0] == 2, "each foreground color must appear exactly twice"
        (r_a, c_a), (r_b, c_b) = coords
        assert int(r_a) != int(r_b) and int(c_a) != int(c_b), "corner pair must define a non-degenerate rectangle"
        rectangles.append(
            (
                int(color),
                (min(int(r_a), int(r_b)), min(int(c_a), int(c_b)), max(int(r_a), int(r_b)), max(int(c_a), int(c_b))),
            )
        )
    return rectangles


def _coverage_from_rectangles(rectangles: list[Rect], shape: tuple[int, int]) -> tuple[np.ndarray, list[set[tuple[int, int]]]]:
    cover = np.zeros(shape, dtype=int)
    border_sets: list[set[tuple[int, int]]] = []
    for rect in rectangles:
        pts = _border_points(rect)
        border_sets.append(pts)
        for r, c in pts:
            cover[r, c] += 1
    return cover, border_sets


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    rectangles = _rectangles_from_grid(g)
    cover = np.zeros(g.shape, dtype=int)
    first_color = np.full(g.shape, bg, dtype=int)

    for color, rect in rectangles:
        for r, c in _border_points(rect):
            cover[r, c] += 1
            if cover[r, c] == 1:
                first_color[r, c] = color

    out = np.full(g.shape, bg, dtype=int)
    out[cover == 1] = first_color[cover == 1]
    out[cover >= 2] = MAROON
    return out


def _random_rectangle(height: int, width: int, rng: random.Random) -> Rect | None:
    if height <= MIN_SIDE_SPAN or width <= MIN_SIDE_SPAN:
        return None
    for _ in range(80):
        r0 = rng.randint(0, height - MIN_SIDE_SPAN - 1)
        r1 = rng.randint(r0 + MIN_SIDE_SPAN, height - 1)
        c0 = rng.randint(0, width - MIN_SIDE_SPAN - 1)
        c1 = rng.randint(c0 + MIN_SIDE_SPAN, width - 1)
        rect = (r0, c0, r1, c1)
        if (r1 - r0) + (c1 - c0) >= 6:
            return rect
    return None


def _disjoint_candidate_ok(candidate: Rect, placed: list[Rect]) -> bool:
    candidate_corners = _all_corners(candidate)
    candidate_border = _border_points(candidate)
    for rect in placed:
        if candidate_corners & _all_corners(rect):
            return False
        if candidate_border & _border_points(rect):
            return False
    return True


def _extend_with_disjoint_rectangles(
    base: list[Rect], height: int, width: int, target_count: int, rng: random.Random
) -> list[Rect] | None:
    for _ in range(200):
        rects = list(base)
        for _ in range(600):
            if len(rects) == target_count:
                return rects
            candidate = _random_rectangle(height, width, rng)
            if candidate is None:
                continue
            if _disjoint_candidate_ok(candidate, rects):
                rects.append(candidate)
        if len(rects) == target_count:
            return rects
    return None


def _build_cross_pair_canonical(height: int, width: int, rng: random.Random) -> list[Rect] | None:
    if height < 6 or width < 7:
        return None
    for _ in range(200):
        r0 = rng.randint(1, height - 4)
        r1 = rng.randint(r0 + MIN_SIDE_SPAN, height - 2)
        c0 = rng.randint(0, width - 6)
        c1 = rng.randint(c0 + 3, width - 3)
        a = (r0, c0, r1, c1)

        b_left = rng.randint(c0 + 1, c1 - 1)
        b_right = rng.randint(c1 + 1, width - 1)
        b_top = rng.randint(0, r0 - 1)
        b_bottom = rng.randint(r1 + 1, height - 1)
        b = (b_top, b_left, b_bottom, b_right)

        if not (_all_corners(a) & _all_corners(b)):
            return [a, b]
    return None


def _build_cross_pair(height: int, width: int, rng: random.Random) -> list[Rect] | None:
    if rng.random() < 0.5:
        rects = _build_cross_pair_canonical(height, width, rng)
    else:
        rects_t = _build_cross_pair_canonical(width, height, rng)
        rects = None if rects_t is None else [_transpose_rect(rect) for rect in rects_t]
    if rects is None:
        return None
    return _apply_random_flips(rects, height, width, rng)


def _build_shared_edge_pair_canonical(height: int, width: int, rng: random.Random) -> list[Rect] | None:
    if height < 7 or width < 7:
        return None
    for _ in range(200):
        shared_col = rng.randint(2, width - 3)
        left_col = rng.randint(0, shared_col - 2)
        right_col = rng.randint(shared_col + 2, width - 1)

        a_top = rng.randint(0, height - 5)
        a_bottom = rng.randint(a_top + 4, height - 1)
        b_top = rng.randint(a_top + 1, a_bottom - 3)
        b_bottom = rng.randint(b_top + 2, a_bottom - 1)

        a = (a_top, left_col, a_bottom, shared_col)
        b = (b_top, shared_col, b_bottom, right_col)

        if not (_all_corners(a) & _all_corners(b)):
            return [a, b]
    return None


def _build_shared_edge_pair(height: int, width: int, rng: random.Random) -> list[Rect] | None:
    if rng.random() < 0.5:
        rects = _build_shared_edge_pair_canonical(height, width, rng)
    else:
        rects_t = _build_shared_edge_pair_canonical(width, height, rng)
        rects = None if rects_t is None else [_transpose_rect(rect) for rect in rects_t]
    if rects is None:
        return None
    return _apply_random_flips(rects, height, width, rng)


def _layout_is_good(rects: list[Rect], height: int, width: int, want_overlap: bool) -> bool:
    cover, border_sets = _coverage_from_rectangles(rects, (height, width))
    overlap_cells = int(np.count_nonzero(cover >= 2))
    if want_overlap != (overlap_cells > 0):
        return False
    if want_overlap and overlap_cells > max(height, width):
        return False

    filled = int(np.count_nonzero(cover > 0))
    if filled > int(height * width * MAX_BORDER_FILL_RATIO):
        return False

    for pts in border_sets:
        unique_cells = sum(cover[r, c] == 1 for r, c in pts)
        if unique_cells < MIN_UNIQUE_BORDER_CELLS:
            return False
    return True


def _grid_from_rectangles(rects: list[Rect], height: int, width: int, bg: int, rng: random.Random) -> np.ndarray:
    color_pool = [color for color in range(10) if color not in {bg, MAROON}]
    colors = rng.sample(color_pool, len(rects))
    grid = np.full((height, width), bg, dtype=int)
    for color, rect in zip(colors, rects):
        r0, c0, r1, c1 = rect
        if rng.random() < 0.5:
            corners = ((r0, c0), (r1, c1))
        else:
            corners = ((r0, c1), (r1, c0))
        for r, c in corners:
            assert grid[r, c] == bg
            grid[r, c] = color
    return grid


def _generated_grid_is_good(grid: np.ndarray, want_overlap: bool) -> bool:
    bg = _get_bg(grid)
    colors = [color for color in np.unique(grid) if int(color) != bg]
    if len(colors) < RECT_COUNT_RANGE[0]:
        return False
    if any(int(np.count_nonzero(grid == int(color))) != 2 for color in colors):
        return False

    out = main(grid)
    maroon_cells = int(np.count_nonzero(out == MAROON))
    if want_overlap != (maroon_cells > 0):
        return False
    if maroon_cells > int(grid.size * 0.18):
        return False
    for color in colors:
        if int(np.count_nonzero(out == int(color))) < MIN_UNIQUE_BORDER_CELLS:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    bg = rng.choice([color for color in range(10) if color != MAROON])
    rect_count = rng.randint(*RECT_COUNT_RANGE)

    want_overlap = rng.random() < 0.5
    if want_overlap:
        if rng.random() < 0.5:
            base = _build_cross_pair(height, width, rng)
        else:
            base = _build_shared_edge_pair(height, width, rng)
    else:
        base = []

    if base is None:
        return None

    rects = _extend_with_disjoint_rectangles(base, height, width, rect_count, rng)
    if rects is None or not _layout_is_good(rects, height, width, want_overlap):
        return None

    grid = _grid_from_rectangles(rects, height, width, bg, rng)
    if not _generated_grid_is_good(grid, want_overlap):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
