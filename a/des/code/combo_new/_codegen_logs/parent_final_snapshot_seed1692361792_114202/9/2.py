from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))

PAIR_COUNT_OPTIONS = (2, 3, 3, 4)
GRID_SIZE_BY_COUNT = {
    2: (10, 18, 10, 18),
    3: (12, 22, 12, 22),
    4: (15, 26, 15, 26),
}
OVERLAP_PROB = 0.5
NESTED_OVERLAP_PROB = 0.5
MAX_OUTPUT_FILL_FRACTION = 0.55
GEN_RETRIES = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw each rectangle implied by a same-color diagonal pair, recoloring rectangles by area rank from a left-edge legend.

    Input structure:
    - The background is black.
    - The leftmost column contains distinct non-black singleton pixels whose top-to-bottom order is a color-ranking legend.
    - Every other non-black color appears exactly twice outside the leftmost column, and the two pixels lie in different rows and columns so they define opposite corners of an axis-aligned rectangle.

    Rule:
    - Read the left-column singleton colors from top to bottom as the rank-to-color legend.
    - For each duplicated color, build the inclusive axis-aligned rectangle whose two matching pixels are diagonal corners.
    - Compute every rectangle area and sort the rectangles from smallest to largest; the areas are unique.
    - Assign the k-th legend color to the k-th smallest rectangle, then redraw all rectangles on a blank black canvas of the same size.
    - When rectangles overlap, larger rectangles are painted first and smaller rectangles overwrite them, so the smallest-area rectangle is visible on top.

    Key insight:
    - The colored input pixels are only corner markers plus an independent legend; solving requires pairing equal colors into hidden rectangles, ranking those rectangles by area, and using the left-edge legend as the new fill lookup.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_colors, ranked_rects = _parse_task(g)
    mapped_rects = [
        (rect, fill_color)
        for rect, fill_color in zip(ranked_rects, legend_colors)
    ]

    out = np.zeros_like(g)
    for rect, fill_color in sorted(mapped_rects, key=lambda item: _rect_area(item[0]), reverse=True):
        r0, c0, r1, c1 = rect
        out[r0 : r1 + 1, c0 : c1 + 1] = fill_color
    return out


def _parse_task(grid: np.ndarray) -> tuple[list[int], list[tuple[int, int, int, int]]]:
    colors = [int(color) for color in np.unique(grid) if int(color) != BG]
    assert colors, "expected at least one non-background color"

    legend: list[tuple[int, int]] = []
    rects: list[tuple[int, int, int, int]] = []
    seen_areas: set[int] = set()

    for color in colors:
        pts = [tuple(map(int, pt)) for pt in np.argwhere(grid == color)]
        if len(pts) == 1:
            r, c = pts[0]
            assert c == 0, "legend singletons must be in the leftmost column"
            legend.append((r, color))
            continue

        assert len(pts) == 2, "each non-legend color must appear exactly twice"
        (r0, c0), (r1, c1) = pts
        assert c0 != 0 and c1 != 0, "pair markers may not occupy the legend column"
        assert r0 != r1 and c0 != c1, "pair markers must form a diagonal"
        rect = (min(r0, r1), min(c0, c1), max(r0, r1), max(c0, c1))
        area = _rect_area(rect)
        assert area not in seen_areas, "rectangle areas must be unique"
        seen_areas.add(area)
        rects.append(rect)

    assert legend and rects and len(legend) == len(rects), "legend size must match rectangle count"
    legend_colors = [color for _, color in sorted(legend)]
    ranked_rects = sorted(rects, key=_rect_area)
    return legend_colors, ranked_rects


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _rect_corners(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}


def _rect_intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    rows = max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1)
    cols = max(0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
    return rows * cols


def _rect_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] <= inner[2] <= outer[2] and outer[1] <= inner[1] <= inner[3] <= outer[3]


def _expanded_rect(rect: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    return (
        max(0, rect[0] - pad),
        max(1, rect[1] - pad),
        min(h - 1, rect[2] + pad),
        min(w - 1, rect[3] + pad),
    )


def _sample_rect(
    rng: random.Random,
    h: int,
    w: int,
    min_h: int = 2,
    max_h: int | None = None,
    min_w: int = 2,
    max_w: int | None = None,
) -> tuple[int, int, int, int] | None:
    rect_max_h = min(h, h if max_h is None else max_h)
    rect_max_w = min(w - 1, (w - 1) if max_w is None else max_w)
    if rect_max_h < min_h or rect_max_w < min_w:
        return None

    rect_h = rng.randint(min_h, rect_max_h)
    rect_w = rng.randint(min_w, rect_max_w)
    r0 = rng.randint(0, h - rect_h)
    c0 = rng.randint(1, w - rect_w)
    return (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)


def _corners_disjoint(rect: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]) -> bool:
    corners = _rect_corners(rect)
    return all(corners.isdisjoint(_rect_corners(other)) for other in others)


def _place_isolated_rect(
    rng: random.Random,
    h: int,
    w: int,
    existing: list[tuple[int, int, int, int]],
    used_areas: set[int],
) -> tuple[int, int, int, int] | None:
    for _ in range(200):
        rect = _sample_rect(rng, h, w, max_h=min(10, h), max_w=min(10, w - 1))
        if rect is None:
            return None
        if _rect_area(rect) in used_areas:
            continue
        if not _corners_disjoint(rect, existing):
            continue
        if any(_rect_intersection_area(rect, other) > 0 for other in existing):
            continue
        expanded = _expanded_rect(rect, 1, h, w)
        if any(_rect_intersection_area(expanded, other) > 0 for other in existing):
            continue
        return rect
    return None


def _place_nested_rect(
    rng: random.Random,
    base: tuple[int, int, int, int],
    used_areas: set[int],
    existing: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    base_h = base[2] - base[0] + 1
    base_w = base[3] - base[1] + 1
    if base_h < 3 or base_w < 3:
        return None

    for _ in range(150):
        rect_h = rng.randint(2, base_h - 1)
        rect_w = rng.randint(2, base_w - 1)
        r0 = rng.randint(base[0], base[2] - rect_h + 1)
        c0 = rng.randint(base[1], base[3] - rect_w + 1)
        rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if _rect_area(rect) in used_areas:
            continue
        if rect == base or not _rect_contains(base, rect):
            continue
        if not _corners_disjoint(rect, existing):
            continue
        return rect
    return None


def _place_partial_overlap_rect(
    rng: random.Random,
    h: int,
    w: int,
    base: tuple[int, int, int, int],
    used_areas: set[int],
    existing: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    base_area = _rect_area(base)
    for _ in range(200):
        rect = _sample_rect(rng, h, w, min_h=3, max_h=min(10, h), min_w=3, max_w=min(10, w - 1))
        if rect is None:
            return None
        area = _rect_area(rect)
        if area in used_areas:
            continue
        if not _corners_disjoint(rect, existing):
            continue

        inter = _rect_intersection_area(base, rect)
        if inter == 0:
            continue
        if _rect_contains(base, rect) or _rect_contains(rect, base):
            continue
        if inter >= min(base_area, area):
            continue
        return rect
    return None


def _place_overlap_pair(
    rng: random.Random,
    h: int,
    w: int,
    used_areas: set[int],
) -> list[tuple[int, int, int, int]] | None:
    for _ in range(150):
        base = _sample_rect(rng, h, w, min_h=4, max_h=min(12, h), min_w=4, max_w=min(12, w - 1))
        if base is None:
            return None
        base_area = _rect_area(base)
        if base_area in used_areas:
            continue

        current = [base]
        current_areas = used_areas | {base_area}
        if rng.random() < NESTED_OVERLAP_PROB:
            other = _place_nested_rect(rng, base, current_areas, current)
            if other is None:
                other = _place_partial_overlap_rect(rng, h, w, base, current_areas, current)
        else:
            other = _place_partial_overlap_rect(rng, h, w, base, current_areas, current)
            if other is None:
                other = _place_nested_rect(rng, base, current_areas, current)
        if other is None:
            continue
        return [base, other]
    return None


def _render_output(
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
    legend_colors: list[int],
) -> np.ndarray:
    ranked = list(zip(sorted(rects, key=_rect_area), legend_colors))
    out = np.zeros(shape, dtype=int)
    for rect, color in sorted(ranked, key=lambda item: _rect_area(item[0]), reverse=True):
        r0, c0, r1, c1 = rect
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _pair_points(
    rect: tuple[int, int, int, int],
    diag: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = rect
    if diag == "main":
        return (r0, c0), (r1, c1)
    return (r0, c1), (r1, c0)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice(PAIR_COUNT_OPTIONS)
    h_min, h_max, w_min, w_max = GRID_SIZE_BY_COUNT[pair_count]
    h = rng.randint(h_min, h_max)
    w = rng.randint(w_min, w_max)
    if pair_count * 2 > len(COLOR_POOL) or h < pair_count or w < 4:
        return None

    require_overlap = rng.random() < OVERLAP_PROB
    rects: list[tuple[int, int, int, int]] = []
    used_areas: set[int] = set()

    if require_overlap:
        overlap_pair = _place_overlap_pair(rng, h, w, used_areas)
        if overlap_pair is None:
            return None
        rects.extend(overlap_pair)
        used_areas.update(_rect_area(rect) for rect in overlap_pair)

    while len(rects) < pair_count:
        rect = _place_isolated_rect(rng, h, w, rects, used_areas)
        if rect is None:
            return None
        rects.append(rect)
        used_areas.add(_rect_area(rect))

    if not require_overlap:
        assert all(
            _rect_intersection_area(rects[i], rects[j]) == 0
            for i in range(len(rects))
            for j in range(i + 1, len(rects))
        )
    elif not any(
        _rect_intersection_area(rects[i], rects[j]) > 0
        for i in range(len(rects))
        for j in range(i + 1, len(rects))
    ):
        return None

    output = _render_output((h, w), rects, list(range(1, pair_count + 1)))
    if int(np.count_nonzero(output)) > int(MAX_OUTPUT_FILL_FRACTION * h * w):
        return None

    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    pair_colors = colors[:pair_count]
    legend_rank_colors = colors[pair_count : 2 * pair_count]
    legend_rows = sorted(rng.sample(range(h), pair_count))

    grid = np.zeros((h, w), dtype=int)
    for row, color in zip(legend_rows, legend_rank_colors):
        grid[row, 0] = color

    used_points = {(row, 0) for row in legend_rows}
    for rect, pair_color in zip(rects, pair_colors):
        diag = rng.choice(("main", "anti"))
        pts = _pair_points(rect, diag)
        if any(pt in used_points for pt in pts):
            return None
        used_points.update(pts)
        for r, c in pts:
            grid[r, c] = pair_color

    expected = _render_output((h, w), rects, legend_rank_colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
