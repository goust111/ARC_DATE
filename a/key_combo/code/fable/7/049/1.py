from __future__ import annotations

import random
from collections import Counter

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _bbox_of_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == int(color))
    if coords.size == 0:
        raise ValueError(f"color {color} is absent")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _topo_bottom_to_top(
    rects: dict[int, tuple[int, int, int, int]],
    above: dict[int, set[int]],
) -> list[int]:
    indeg = {color: 0 for color in rects}
    below_of = {color: set() for color in rects}
    for lower, uppers in above.items():
        for upper in uppers:
            indeg[upper] += 1
            below_of[upper].add(lower)

    order: list[int] = []
    ready = [color for color, deg in indeg.items() if deg == 0]
    ready.sort(key=lambda color: rects[color])

    while ready:
        color = ready.pop(0)
        order.append(color)
        released: list[int] = []
        for upper in sorted(above[color], key=lambda c: rects[c]):
            indeg[upper] -= 1
            below_of[upper].discard(color)
            if indeg[upper] == 0:
                released.append(upper)
        ready.extend(released)
        ready.sort(key=lambda c: rects[c])

    if len(order) != len(rects):
        raise ValueError("occlusion graph must be acyclic")
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct each partially hidden solid rectangle, infer the bottom-to-top layer order, and drop the rectangles to stack them.

    Input structure:
    - The background is black and each non-black color belongs to one solid rectangle.
    - Rectangles partially overlap, so only the topmost color is visible at each cell.
    - Every rectangle still shows pixels on all four extremes of its true bounding box, letting the visible bbox recover the full rectangle.

    Rule:
    - For each color, treat the visible pixels' bounding box as that rectangle's full solid shape.
    - Whenever another non-background color appears inside that bbox, it must lie above the current rectangle; collect all such above/below relations and topologically sort them from bottom to top.
    - Process rectangles in that bottom-to-top order: keep each rectangle's columns fixed and drop the full solid bbox vertically until it hits the grid bottom or the top of an already settled rectangle.

    Key insight:
    - The decisive ordering signal is occlusion, not initial height: hidden interior cells reveal which colors sit above which, and those reconstructed rectangles then fall as rigid solid blocks.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    h, w = map(int, g.shape)
    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    if not colors:
        return g.copy()

    rects = {color: _bbox_of_color(g, color) for color in colors}
    above = {color: set() for color in colors}
    for color, (r0, c0, r1, c1) in rects.items():
        patch_colors = colors_present(g[r0 : r1 + 1, c0 : c1 + 1], exclude={bg, color})
        above[color].update(patch_colors)

    order = _topo_bottom_to_top(rects, above)

    out = np.full((h, w), bg, dtype=int)
    column_top = np.full(w, h, dtype=int)
    for color in order:
        r0, c0, r1, c1 = rects[color]
        rect_h = r1 - r0 + 1
        support = int(column_top[c0 : c1 + 1].min())
        final_r0 = support - rect_h
        if final_r0 < 0:
            raise ValueError("stacked rectangles exceed grid height")
        out[final_r0:support, c0 : c1 + 1] = int(color)
        column_top[c0 : c1 + 1] = final_r0

    return out


BG = 0
COLORS = tuple(range(1, 10))
GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
RECT_COUNT_RANGE = (2, 4)
OUTER_H_MIN = 5
OUTER_W_MIN = 5
OUTER_H_MAX = 12
OUTER_W_MAX = 14
MARGIN_MIN = 1
MARGIN_MAX = 3
HEIGHT_MIN = 2
WIDTH_MIN = 2
GEN_ATTEMPTS = 400


def _sample_local_bboxes(count: int) -> list[tuple[int, int, int, int]] | None:
    outer_h = random.randint(max(OUTER_H_MIN, count * 2 + 1), OUTER_H_MAX)
    outer_w = random.randint(max(OUTER_W_MIN, count * 2 + 1), OUTER_W_MAX)
    boxes = [(0, 0, outer_h - 1, outer_w - 1)]

    for _ in range(1, count):
        pr0, pc0, pr1, pc1 = boxes[-1]
        prev_h = pr1 - pr0 + 1
        prev_w = pc1 - pc0 + 1

        top_max = min(MARGIN_MAX, prev_h - HEIGHT_MIN - 1)
        bottom_max = min(MARGIN_MAX, prev_h - HEIGHT_MIN - 1)
        left_max = min(MARGIN_MAX, prev_w - WIDTH_MIN - 1)
        right_max = min(MARGIN_MAX, prev_w - WIDTH_MIN - 1)
        if min(top_max, bottom_max, left_max, right_max) < MARGIN_MIN:
            return None

        for _attempt in range(40):
            top = random.randint(MARGIN_MIN, top_max)
            bottom = random.randint(MARGIN_MIN, min(bottom_max, prev_h - top - HEIGHT_MIN))
            left = random.randint(MARGIN_MIN, left_max)
            right = random.randint(MARGIN_MIN, min(right_max, prev_w - left - WIDTH_MIN))
            inner_h = prev_h - top - bottom
            inner_w = prev_w - left - right
            if inner_h >= HEIGHT_MIN and inner_w >= WIDTH_MIN:
                boxes.append((pr0 + top, pc0 + left, pr1 - bottom, pc1 - right))
                break
        else:
            return None

    return boxes


def _shift_boxes(
    boxes: list[tuple[int, int, int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int, int, int]]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in boxes]


def _render_occluded_scene(
    shape: tuple[int, int],
    colors: list[int],
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for color, (r0, c0, r1, c1) in zip(colors, boxes):
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return grid


def _render_stacked_output(
    shape: tuple[int, int],
    colors: list[int],
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    column_top = np.full(w, h, dtype=int)
    for color, (_r0, c0, r1, c1) in zip(colors, boxes):
        rect_h = r1 - _r0 + 1
        support = int(column_top[c0 : c1 + 1].min())
        final_r0 = support - rect_h
        if final_r0 < 0:
            raise ValueError("generator created an over-tall stack")
        out[final_r0:support, c0 : c1 + 1] = int(color)
        column_top[c0 : c1 + 1] = final_r0
    return out


def _edge_visibility_ok(grid: np.ndarray, color: int, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    mask = grid == int(color)
    return bool(
        mask[r0, c0 : c1 + 1].any()
        and mask[r1, c0 : c1 + 1].any()
        and mask[r0 : r1 + 1, c0].any()
        and mask[r0 : r1 + 1, c1].any()
    )


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*RECT_COUNT_RANGE)
    local_boxes = _sample_local_bboxes(count)
    if local_boxes is None:
        return None

    outer_r0, outer_c0, outer_r1, outer_c1 = local_boxes[0]
    outer_h = outer_r1 - outer_r0 + 1
    outer_w = outer_c1 - outer_c0 + 1
    heights = [r1 - r0 + 1 for r0, _c0, r1, _c1 in local_boxes]
    min_h = max(sum(heights) + 1, outer_h + 2, GRID_H_RANGE[0])
    max_h = GRID_H_RANGE[1]
    min_w = max(outer_w + 1, GRID_W_RANGE[0])
    max_w = GRID_W_RANGE[1]
    if min_h > max_h or min_w > max_w:
        return None

    grid_h = random.randint(min_h, max_h)
    grid_w = random.randint(min_w, max_w)

    max_scene_r0 = grid_h - outer_h - 1
    max_scene_c0 = grid_w - outer_w
    if max_scene_r0 < 0 or max_scene_c0 < 0:
        return None
    scene_r0 = random.randint(0, max_scene_r0)
    scene_c0 = random.randint(0, max_scene_c0)
    boxes = _shift_boxes(local_boxes, scene_r0, scene_c0)

    colors = random.sample(list(COLORS), count)
    inp = _render_occluded_scene((grid_h, grid_w), colors, boxes)

    for color, box in zip(colors, boxes):
        if _bbox_of_color(inp, color) != box:
            return None
        if not _edge_visibility_ok(inp, color, box):
            return None

    expected = _render_stacked_output((grid_h, grid_w), colors, boxes)
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
