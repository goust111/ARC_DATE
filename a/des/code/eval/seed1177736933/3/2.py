from __future__ import annotations

import random
from collections import Counter

import numpy as np


PALETTE = tuple(range(1, 10))
BACKGROUND = 0
RECT_COUNT_RANGE = (3, 5)
CENTER_SIDE_MAX = 4
OUTER_SIDE_MAX = 6
SCENE_HEIGHT_RANGE = (9, 18)
GRID_WIDTH_RANGE = (14, 26)
TRAILING_MARKER_BLANKS = (1, 4)
MAX_OUTPUT_SIDE = 30
GENERATION_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values = Counter(map(int, np.asarray(grid, dtype=int).reshape(-1)))
    return int(values.most_common(1)[0][0])


def _parse_marker_row(marker_row: np.ndarray, bg: int) -> dict[int, int]:
    row = np.asarray(marker_row, dtype=int)
    positions = [int(c) for c in np.flatnonzero(row != bg)]
    assert len(positions) >= 3, "expected 3-5 marker pixels"

    thickness_by_color: dict[int, int] = {}
    previous = -1
    for col in positions:
        color = int(row[col])
        assert color not in thickness_by_color, "marker colors must be unique"
        gap = int(col if previous < 0 else col - previous - 1)
        assert gap >= 0, "marker spacing must be non-negative"
        thickness_by_color[color] = gap
        previous = col

    return thickness_by_color


def _extract_rectangles(body: np.ndarray, bg: int) -> list[dict[str, int]]:
    g = np.asarray(body, dtype=int)
    rectangles: list[dict[str, int]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        points = np.argwhere(g == color)
        assert len(points) > 0
        r0, c0 = map(int, points.min(axis=0))
        r1, c1 = map(int, points.max(axis=0))
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        area = int(len(points))
        assert area == h * w, "each fragment must be a solid rectangle"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle must be filled"
        rectangles.append(
            {
                "color": color,
                "r0": r0,
                "c0": c0,
                "h": h,
                "w": w,
                "area": area,
            }
        )

    assert RECT_COUNT_RANGE[0] <= len(rectangles) <= RECT_COUNT_RANGE[1]
    areas = [rect["area"] for rect in rectangles]
    assert len(set(areas)) == len(areas), "rectangle areas must be unique"
    return sorted(rectangles, key=lambda rect: rect["area"])


def _embed_centered(grid: np.ndarray, side: int, bg: int) -> np.ndarray:
    h, w = map(int, grid.shape)
    assert side >= max(h, w)
    out = np.full((side, side), bg, dtype=int)
    top = (side - h) // 2
    left = (side - w) // 2
    out[top : top + h, left : left + w] = grid
    return out


def _wrap_square_frame(grid: np.ndarray, color: int, thickness: int, bg: int) -> np.ndarray:
    base_side = int(max(grid.shape))
    inner = _embed_centered(np.asarray(grid, dtype=int), base_side, bg)
    if thickness <= 0:
        return inner

    side = int(base_side + 2 * thickness)
    out = np.full((side, side), color, dtype=int)
    out[thickness : side - thickness, thickness : side - thickness] = bg
    out[thickness : thickness + base_side, thickness : thickness + base_side] = inner
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered solid rectangles by area and rebuild them as nested square frames.

    The bottom row encodes per-color frame thickness via the blank count before each
    marker pixel. The smallest-area rectangle stays a solid center block with its
    original aspect ratio; each larger rectangle contributes one outer square frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 3 and g.shape[1] >= 3, "grid too small"

    bg = _most_common_color(g)
    marker_row = g[-1]
    thickness_by_color = _parse_marker_row(marker_row, bg)

    body = g[:-1]
    rectangles = _extract_rectangles(body, bg)
    body_colors = {rect["color"] for rect in rectangles}
    assert body_colors == set(thickness_by_color), "marker colors must match fragment colors"

    center = rectangles[0]
    out = np.full((center["h"], center["w"]), center["color"], dtype=int)
    for rect in rectangles[1:]:
        out = _wrap_square_frame(out, rect["color"], thickness_by_color[rect["color"]], bg)

    return out


def _sample_center_dims() -> tuple[int, int]:
    branch = random.choice(("wide", "tall", "square"))
    if branch == "square":
        side = random.randint(2, CENTER_SIDE_MAX)
        return side, side
    if branch == "wide":
        h = random.randint(1, 3)
        w = random.randint(h + 1, CENTER_SIDE_MAX)
        return h, w
    w = random.randint(1, 3)
    h = random.randint(w + 1, CENTER_SIDE_MAX)
    return h, w


def _sample_rectangle_dims(count: int) -> list[tuple[int, int]]:
    center = _sample_center_dims()
    dims = [center]
    used_areas = {center[0] * center[1]}
    center_area = center[0] * center[1]

    while len(dims) < count:
        h = random.randint(1, OUTER_SIDE_MAX)
        w = random.randint(1, OUTER_SIDE_MAX)
        area = h * w
        if area <= center_area or area in used_areas:
            continue
        dims.append((h, w))
        used_areas.add(area)

    return sorted(dims, key=lambda shape: shape[0] * shape[1])


def _sample_marker_layout(colors: list[int]) -> tuple[list[int], list[int], dict[int, int], int]:
    n = len(colors)
    marker_order = colors[:]
    random.shuffle(marker_order)

    first_gap = random.randint(1, 4)
    between_gaps = random.sample((1, 2, 3, 4), n - 1)
    positions = [first_gap]
    thickness_by_color = {marker_order[0]: first_gap}
    for index, gap in enumerate(between_gaps, start=1):
        positions.append(positions[-1] + gap + 1)
        thickness_by_color[marker_order[index]] = gap

    trailing = random.randint(*TRAILING_MARKER_BLANKS)
    width = int(positions[-1] + 1 + trailing)
    return marker_order, positions, thickness_by_color, width


def _place_rectangles(
    scene_h: int, scene_w: int, rectangles: list[tuple[int, int, int]]
) -> list[tuple[int, int, int, int, int]] | None:
    occ = np.zeros((scene_h, scene_w), dtype=bool)
    placed: list[tuple[int, int, int, int, int]] = []

    for color, h, w in sorted(rectangles, key=lambda item: item[1] * item[2], reverse=True):
        candidates: list[tuple[int, int]] = []
        for r in range(scene_h - h + 1):
            for c in range(scene_w - w + 1):
                r0 = max(0, r - 1)
                r1 = min(scene_h, r + h + 1)
                c0 = max(0, c - 1)
                c1 = min(scene_w, c + w + 1)
                if not occ[r0:r1, c0:c1].any():
                    candidates.append((r, c))
        if not candidates:
            return None

        r, c = random.choice(candidates)
        placed.append((color, r, c, h, w))
        occ[max(0, r - 1) : min(scene_h, r + h + 1), max(0, c - 1) : min(scene_w, c + w + 1)] = True

    return placed


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*RECT_COUNT_RANGE)
    colors = random.sample(PALETTE, count)
    dims = _sample_rectangle_dims(count)
    rectangles = [(colors[i], dims[i][0], dims[i][1]) for i in range(count)]
    area_sorted_colors = [item[0] for item in rectangles]

    marker_order, marker_positions, thickness_by_color, marker_width = _sample_marker_layout(colors)
    if marker_order == area_sorted_colors:
        return None

    center_max = max(dims[0])
    outer_sum = sum(
        thickness_by_color[color]
        for color in area_sorted_colors[1:]
    )
    if center_max + 2 * outer_sum > MAX_OUTPUT_SIDE:
        return None

    scene_h = random.randint(*SCENE_HEIGHT_RANGE)
    scene_w = random.randint(max(marker_width, GRID_WIDTH_RANGE[0]), GRID_WIDTH_RANGE[1])
    placements = _place_rectangles(scene_h, scene_w, rectangles)
    if placements is None:
        return None

    grid = np.full((scene_h + 2, scene_w), BACKGROUND, dtype=int)
    for color, r, c, h, w in placements:
        grid[r : r + h, c : c + w] = color

    marker_colors = {color for color, _r, _c, _h, _w in placements}
    if marker_colors != set(colors):
        return None

    for pos, color in zip(marker_positions, marker_order):
        grid[-1, pos] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape[0] != out.shape[1]:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame ARC input")
