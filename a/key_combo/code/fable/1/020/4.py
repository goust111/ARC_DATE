from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox, colors_present, crop_bbox, get_bg


BBox = tuple[int, int, int, int]


def _box_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def _box_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def _intersect_boxes(a: BBox, b: BBox) -> BBox | None:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _boxes_from_grid(grid: np.ndarray, bg: int) -> dict[int, BBox]:
    boxes: dict[int, BBox] = {}
    for color in colors_present(grid, exclude={bg}):
        box = bbox(grid == int(color))
        assert box is not None, f"missing pixels for color {color}"
        boxes[int(color)] = box
    assert boxes, "expected at least one colored rectangle"
    return boxes


def _overlap_edges(grid: np.ndarray, boxes: dict[int, BBox]) -> tuple[dict[int, set[int]], dict[int, int], int]:
    colors = sorted(boxes)
    edges = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    overlap_pairs = 0

    for a, b in combinations(colors, 2):
        inter = _intersect_boxes(boxes[a], boxes[b])
        if inter is None:
            continue
        overlap_pairs += 1
        region = crop_bbox(grid, inter)
        visible = set(int(v) for v in np.unique(region))
        seen = visible & {int(a), int(b)}
        assert len(seen) <= 1, "an overlap cannot reveal both rectangle colors"
        if not seen:
            continue
        upper = next(iter(seen))
        lower = b if upper == a else a
        if upper not in edges[lower]:
            edges[lower].add(upper)
            indegree[upper] += 1

    return edges, indegree, overlap_pairs


def _unique_stack_order(grid: np.ndarray, boxes: dict[int, BBox]) -> list[int]:
    colors = sorted(boxes)
    edges, indegree, _overlap_pairs = _overlap_edges(grid, boxes)
    remaining = set(colors)
    order: list[int] = []

    while remaining:
        frontier = sorted(color for color in remaining if indegree[color] == 0)
        assert len(frontier) == 1, "overlap constraints must imply one unique bottom-to-top order"
        color = frontier[0]
        remaining.remove(color)
        order.append(color)
        for nxt in sorted(edges[color]):
            indegree[nxt] -= 1
            assert indegree[nxt] >= 0, "indegree underflow indicates a cycle"

    return order


def _render_rectangles(shape: tuple[int, int], boxes: dict[int, BBox], order: list[int], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for color in order:
        r0, c0, r1, c1 = boxes[int(color)]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full stacked rectangles, reverse their layer order, and mirror the back view left-to-right.

    Input structure
    - A dominant background surrounds several distinct-colored solid rectangles that may overlap.
    - Each color's visible pixels belong to one rectangle, and that visible color mask's bbox is the rectangle's true bbox.
    - Overlap evidence reveals one unique global stacking order from bottom to top.

    Rule
    - For every non-background color, recover the full rectangle by filling its visible bbox.
    - Compare every pair of overlapping recovered bboxes: whenever their overlap visibly shows one of the two colors, that shown color is above the other rectangle.
    - Use the uniquely determined bottom-to-top order, redraw the rectangles in the opposite order, then mirror the whole result horizontally.

    Key insight
    - The front view already encodes both hidden geometry and hidden depth: bbox recovers each opaque plate, overlap winners recover the layer order, and a true back view needs both layer reversal and left-right mirroring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(get_bg(g))
    boxes = _boxes_from_grid(g, bg)
    order = _unique_stack_order(g, boxes)
    reverse_render = _render_rectangles(g.shape, boxes, list(reversed(order)), bg)
    return np.fliplr(reverse_render)


# --------------- generator (o5x) ---------------

GEN_BG = 0
GEN_COLORS = tuple(range(1, 10))
GEN_RECT_COUNT_RANGE = (3, 5)
GEN_SIDE_RANGE = (5, 7)
GEN_EXTENSION_MAX = 3
GEN_MIN_EXPOSED = 2
GEN_CANVAS_EXTRA_RANGE = (4, 10)
GEN_MARGIN = 1
GEN_MAX_ATTEMPTS = 400
GEN_DIRECTIONS = ("se", "sw", "ne", "nw")


def _place_relative(prev: BBox, height: int, width: int, direction: str, ov_h: int, ov_w: int) -> BBox:
    if direction == "se":
        r0 = int(prev[2] - ov_h + 1)
        c0 = int(prev[3] - ov_w + 1)
        return r0, c0, r0 + height - 1, c0 + width - 1
    if direction == "sw":
        r0 = int(prev[2] - ov_h + 1)
        c1 = int(prev[1] + ov_w - 1)
        return r0, c1 - width + 1, r0 + height - 1, c1
    if direction == "ne":
        r1 = int(prev[0] + ov_h - 1)
        c0 = int(prev[3] - ov_w + 1)
        return r1 - height + 1, c0, r1, c0 + width - 1
    assert direction == "nw"
    r1 = int(prev[0] + ov_h - 1)
    c1 = int(prev[1] + ov_w - 1)
    return r1 - height + 1, c1 - width + 1, r1, c1


def _shift_boxes(boxes: list[BBox], dr: int, dc: int) -> list[BBox]:
    return [
        (int(r0 + dr), int(c0 + dc), int(r1 + dr), int(c1 + dc))
        for r0, c0, r1, c1 in boxes
    ]


def _edge_is_visible(grid: np.ndarray, color: int, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    return bool(
        np.any(grid[r0, c0 : c1 + 1] == color)
        and np.any(grid[r1, c0 : c1 + 1] == color)
        and np.any(grid[r0 : r1 + 1, c0] == color)
        and np.any(grid[r0 : r1 + 1, c1] == color)
    )


def _sample_chain_boxes(count: int) -> list[BBox] | None:
    first_h = random.randint(*GEN_SIDE_RANGE)
    first_w = random.randint(*GEN_SIDE_RANGE)
    boxes = [(0, 0, first_h - 1, first_w - 1)]

    for _ in range(1, count):
        prev = boxes[-1]
        height = random.randint(*GEN_SIDE_RANGE)
        width = random.randint(*GEN_SIDE_RANGE)

        ov_h_lo = max(GEN_MIN_EXPOSED, height - GEN_EXTENSION_MAX)
        ov_h_hi = min(_box_h(prev) - GEN_MIN_EXPOSED, height - GEN_MIN_EXPOSED)
        ov_w_lo = max(GEN_MIN_EXPOSED, width - GEN_EXTENSION_MAX)
        ov_w_hi = min(_box_w(prev) - GEN_MIN_EXPOSED, width - GEN_MIN_EXPOSED)
        if ov_h_lo > ov_h_hi or ov_w_lo > ov_w_hi:
            return None

        candidates: list[BBox] = []
        directions = list(GEN_DIRECTIONS)
        random.shuffle(directions)
        ov_h_vals = list(range(ov_h_lo, ov_h_hi + 1))
        ov_w_vals = list(range(ov_w_lo, ov_w_hi + 1))
        random.shuffle(ov_h_vals)
        random.shuffle(ov_w_vals)

        for direction in directions:
            for ov_h in ov_h_vals:
                for ov_w in ov_w_vals:
                    box = _place_relative(prev, height, width, direction, ov_h, ov_w)
                    if any(_intersect_boxes(box, older) is not None for older in boxes[:-1]):
                        continue
                    candidates.append(box)

        if not candidates:
            return None
        boxes.append(random.choice(candidates))

    return boxes


def _boxes_with_canvas(boxes: list[BBox]) -> tuple[tuple[int, int], list[BBox]] | None:
    min_r = min(box[0] for box in boxes)
    min_c = min(box[1] for box in boxes)
    max_r = max(box[2] for box in boxes)
    max_c = max(box[3] for box in boxes)
    scene_h = max_r - min_r + 1
    scene_w = max_c - min_c + 1

    extra_h_max = min(GEN_CANVAS_EXTRA_RANGE[1], 30 - scene_h)
    extra_w_max = min(GEN_CANVAS_EXTRA_RANGE[1], 30 - scene_w)
    if extra_h_max < GEN_CANVAS_EXTRA_RANGE[0] or extra_w_max < GEN_CANVAS_EXTRA_RANGE[0]:
        return None

    extra_h = random.randint(GEN_CANVAS_EXTRA_RANGE[0], extra_h_max)
    extra_w = random.randint(GEN_CANVAS_EXTRA_RANGE[0], extra_w_max)
    height = scene_h + extra_h
    width = scene_w + extra_w

    top_margin = random.randint(GEN_MARGIN, extra_h - GEN_MARGIN)
    left_margin = random.randint(GEN_MARGIN, extra_w - GEN_MARGIN)
    shifted = _shift_boxes(boxes, top_margin - min_r, left_margin - min_c)
    return (height, width), shifted


def _validate_instance(grid: np.ndarray, boxes: dict[int, BBox], order: list[int], bg: int) -> bool:
    if int(get_bg(grid)) != int(bg):
        return False

    recovered_boxes = _boxes_from_grid(grid, bg)
    if recovered_boxes != boxes:
        return False

    if any(not _edge_is_visible(grid, color, box) for color, box in boxes.items()):
        return False

    try:
        recovered_order = _unique_stack_order(grid, recovered_boxes)
    except AssertionError:
        return False
    if recovered_order != order:
        return False

    edges, _indegree, overlap_pairs = _overlap_edges(grid, recovered_boxes)
    if overlap_pairs < len(order) - 1:
        return False
    edge_count = sum(len(dsts) for dsts in edges.values())
    if edge_count < len(order) - 1:
        return False

    front = _render_rectangles(grid.shape, boxes, order, bg)
    if not np.array_equal(front, grid):
        return False
    reverse_only = _render_rectangles(grid.shape, boxes, list(reversed(order)), bg)
    back_view = np.fliplr(reverse_only)

    if np.array_equal(reverse_only, front):
        return False
    if np.array_equal(back_view, reverse_only):
        return False
    if np.array_equal(back_view, np.fliplr(front)):
        return False
    if np.array_equal(back_view, front):
        return False
    if set(colors_present(back_view, exclude={bg})) != set(order):
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, back_view)


def _generate_attempt() -> np.ndarray | None:
    count = random.randint(*GEN_RECT_COUNT_RANGE)
    chain_boxes = _sample_chain_boxes(count)
    if chain_boxes is None:
        return None

    canvas = _boxes_with_canvas(chain_boxes)
    if canvas is None:
        return None

    shape, shifted_boxes = canvas
    colors = random.sample(GEN_COLORS, count)
    order = [int(color) for color in colors]
    boxes = {color: box for color, box in zip(order, shifted_boxes)}
    grid = _render_rectangles(shape, boxes, order, GEN_BG)

    if np.count_nonzero(grid != GEN_BG) >= grid.size // 2:
        return None
    if not _validate_instance(grid, boxes, order, GEN_BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid back-view occlusion puzzle input")
