import random

import numpy as np


MAROON = 9
MAX_ATTEMPTS = 400

GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (10, 18)
RECT_COUNT_RANGE = (2, 4)
RECT_SIDE_RANGE = (2, 6)
MAX_COVER_FRACTION = 0.55
MIN_UNIQUE_AREA = 3
MIN_OVERLAP_AREA = 2


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _ordered_marker_pairs(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    first_seen: dict[int, int] = {}
    positions: dict[int, list[tuple[int, int]]] = {}
    h, w = g.shape

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg:
                continue
            first_seen.setdefault(color, r * w + c)
            positions.setdefault(color, []).append((r, c))

    assert positions, "expected at least one marker pair"
    for color, coords in positions.items():
        assert len(coords) == 2, f"color {color} must appear exactly twice"
        (r0, c0), (r1, c1) = coords
        assert r0 != r1 and c0 != c1, "markers must be opposite rectangle corners, not a line"

    return sorted(positions.items(), key=lambda item: first_seen[item[0]])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangle implied by a same-color diagonal marker pair and color overlaps maroon.

    Input structure:
    - The background is the dominant color.
    - Every non-background color appears exactly twice as isolated marker pixels.
    - Each pair marks opposite corners of an axis-aligned rectangle.

    Rule:
    - Process colors in the row-major order of their first appearance.
    - Fill each marker-defined rectangle with that marker color.
    - Any cell covered by two or more rectangles ends up maroon.
    - Marker cells themselves are part of the filled rectangles.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    ordered_pairs = _ordered_marker_pairs(g, bg)

    cover_count = np.zeros(g.shape, dtype=int)
    paint = np.full(g.shape, bg, dtype=int)

    for color, coords in ordered_pairs:
        (r0, c0), (r1, c1) = coords
        rr0, rr1 = sorted((r0, r1))
        cc0, cc1 = sorted((c0, c1))
        cover_count[rr0 : rr1 + 1, cc0 : cc1 + 1] += 1
        paint[rr0 : rr1 + 1, cc0 : cc1 + 1] = color

    out = np.full(g.shape, bg, dtype=int)
    filled = cover_count > 0
    out[filled] = paint[filled]
    out[cover_count >= 2] = MAROON
    return out


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _marker_positions(rect: tuple[int, int, int, int], diag: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    if diag == 0:
        return [(r0, c0), (r1, c1)]
    return [(r0, c1), (r1, c0)]


def _markers_are_clear(rects: list[tuple[int, int, int, int]], diags: list[int]) -> bool:
    seen: list[tuple[int, int]] = []
    for rect, diag in zip(rects, diags):
        for r, c in _marker_positions(rect, diag):
            for rr, cc in seen:
                if abs(r - rr) + abs(c - cc) <= 1:
                    return False
            seen.append((r, c))
    return True


def _layout_metrics(
    rects: list[tuple[int, int, int, int]], shape: tuple[int, int]
) -> tuple[np.ndarray, int, list[int]]:
    cover = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rects:
        cover[r0 : r1 + 1, c0 : c1 + 1] += 1

    overlap_area = int(np.count_nonzero(cover >= 2))
    unique_areas = []
    for r0, c0, r1, c1 in rects:
        unique_areas.append(int(np.count_nonzero(cover[r0 : r1 + 1, c0 : c1 + 1] == 1)))
    return cover, overlap_area, unique_areas


def _sample_rect(rng: random.Random, h: int, w: int) -> tuple[int, int, int, int]:
    side_hi_h = min(RECT_SIDE_RANGE[1], h)
    side_hi_w = min(RECT_SIDE_RANGE[1], w)
    rect_h = rng.randint(RECT_SIDE_RANGE[0], side_hi_h)
    rect_w = rng.randint(RECT_SIDE_RANGE[0], side_hi_w)
    r0 = rng.randint(0, h - rect_h)
    c0 = rng.randint(0, w - rect_w)
    return (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)


def _sample_overlapping_rect(
    rng: random.Random, h: int, w: int, base: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    for _ in range(120):
        rect = _sample_rect(rng, h, w)
        inter = _intersection_area(rect, base)
        if inter < MIN_OVERLAP_AREA:
            continue
        if inter >= min(_rect_area(rect), _rect_area(base)):
            continue
        return rect
    return None


def _sample_rectangles(
    rng: random.Random, h: int, w: int, rect_count: int, require_overlap: bool
) -> list[tuple[int, int, int, int]] | None:
    rects = [_sample_rect(rng, h, w)]

    if require_overlap:
        second = _sample_overlapping_rect(rng, h, w, rects[0])
        if second is None:
            return None
        rects.append(second)

    while len(rects) < rect_count:
        candidate = _sample_rect(rng, h, w)
        if candidate in rects:
            continue
        rects.append(candidate)

    cover, overlap_area, unique_areas = _layout_metrics(rects, (h, w))
    if require_overlap and overlap_area < MIN_OVERLAP_AREA:
        return None
    if not require_overlap and overlap_area != 0:
        return None
    if int(cover.max()) > 2:
        return None
    if np.count_nonzero(cover) > int(h * w * MAX_COVER_FRACTION):
        return None
    if any(area < MIN_UNIQUE_AREA for area in unique_areas):
        return None
    return rects


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    rect_count = rng.randint(*RECT_COUNT_RANGE)
    require_overlap = bool(rng.randint(0, 1))

    rects = _sample_rectangles(rng, h, w, rect_count, require_overlap)
    if rects is None:
        return None

    diags = [rng.randint(0, 1) for _ in rects]
    if not _markers_are_clear(rects, diags):
        return None

    bg_choices = [color for color in range(10) if color != MAROON]
    bg = rng.choice(bg_choices)
    marker_colors = [color for color in range(10) if color not in (bg, MAROON)]
    rng.shuffle(marker_colors)
    marker_colors = marker_colors[:rect_count]

    grid = np.full((h, w), bg, dtype=int)
    for color, rect, diag in zip(marker_colors, rects, diags):
        for r, c in _marker_positions(rect, diag):
            if grid[r, c] != bg:
                return None
            grid[r, c] = color

    out = main(grid)
    has_maroon = bool(np.any(out == MAROON))
    if has_maroon != require_overlap:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
