import itertools
import random
from typing import Iterable

import numpy as np


Box = tuple[int, int, int, int]

BG = 0
PALETTE = [1, 2, 3, 4, 7]
MAX_GENERATION_ATTEMPTS = 400

# Valid 3-box width/height families. The 4/6/8 span guarantees strict cross-over
# while the varying heights keep the stacked output diverse.
THREE_BOX_HEIGHTS = [
    (6, 5, 4),
    (7, 5, 4),
    (8, 5, 4),
    (7, 6, 4),
    (8, 6, 4),
    (8, 7, 4),
    (7, 6, 5),
    (8, 6, 5),
    (8, 7, 5),
    (8, 7, 6),
]
THREE_BOX_WIDTHS = (4, 6, 8)

# Hand-picked valid 4-box templates normalized to their bbox union.
FOUR_BOX_TEMPLATES = [
    {
        "shape": (9, 10),
        "boxes": [
            (0, 3, 7, 9),
            (2, 2, 6, 6),
            (1, 4, 8, 8),
            (2, 0, 5, 7),
        ],
    },
    {
        "shape": (8, 9),
        "boxes": [
            (1, 3, 6, 6),
            (2, 2, 6, 7),
            (0, 0, 7, 4),
            (1, 1, 4, 8),
        ],
    },
    {
        "shape": (8, 8),
        "boxes": [
            (2, 2, 5, 5),
            (1, 1, 6, 5),
            (2, 0, 6, 7),
            (0, 0, 7, 5),
        ],
    },
]


def _box_h(box: Box) -> int:
    return box[2] - box[0] + 1


def _box_w(box: Box) -> int:
    return box[3] - box[1] + 1


def _box_area(box: Box) -> int:
    return _box_h(box) * _box_w(box)


def _translate_box(box: Box, dr: int, dc: int) -> Box:
    r0, c0, r1, c1 = box
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _draw_frame(grid: np.ndarray, box: Box, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _perimeter_points(box: Box) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _bbox_union(boxes: Iterable[Box]) -> Box:
    boxes = list(boxes)
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _normalize_scene(shape: tuple[int, int], boxes: list[Box]) -> tuple[tuple[int, int], list[Box]]:
    del shape
    union = _bbox_union(boxes)
    r0, c0, r1, c1 = union
    shifted = [_translate_box(box, -r0, -c0) for box in boxes]
    return (r1 - r0 + 1, c1 - c0 + 1), shifted


def _transform_point(
    r: int,
    c: int,
    shape: tuple[int, int],
    transpose: bool,
    flip_v: bool,
    flip_h: bool,
) -> tuple[int, int]:
    h, w = shape
    if transpose:
        r, c = c, r
        h, w = w, h
    if flip_v:
        r = h - 1 - r
    if flip_h:
        c = w - 1 - c
    return r, c


def _transform_box(
    box: Box,
    shape: tuple[int, int],
    transpose: bool,
    flip_v: bool,
    flip_h: bool,
) -> Box:
    r0, c0, r1, c1 = box
    corners = [
        _transform_point(r0, c0, shape, transpose, flip_v, flip_h),
        _transform_point(r0, c1, shape, transpose, flip_v, flip_h),
        _transform_point(r1, c0, shape, transpose, flip_v, flip_h),
        _transform_point(r1, c1, shape, transpose, flip_v, flip_h),
    ]
    rows = [p[0] for p in corners]
    cols = [p[1] for p in corners]
    return min(rows), min(cols), max(rows), max(cols)


def _transform_scene(
    shape: tuple[int, int],
    boxes: list[Box],
    transpose: bool,
    flip_v: bool,
    flip_h: bool,
) -> tuple[tuple[int, int], list[Box]]:
    new_shape = (shape[1], shape[0]) if transpose else shape
    new_boxes = [_transform_box(box, shape, transpose, flip_v, flip_h) for box in boxes]
    return _normalize_scene(new_shape, new_boxes)


def _render_scene(shape: tuple[int, int], boxes_by_color: dict[int, Box], order_front: list[int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color in reversed(order_front):
        _draw_frame(grid, boxes_by_color[color], color)
    return grid


def _visible_opposite_sides(grid: np.ndarray, box: Box, color: int) -> bool:
    r0, c0, r1, c1 = box
    left = bool(np.all(grid[r0 : r1 + 1, c0] == color))
    right = bool(np.all(grid[r0 : r1 + 1, c1] == color))
    top = bool(np.all(grid[r0, c0 : c1 + 1] == color))
    bottom = bool(np.all(grid[r1, c0 : c1 + 1] == color))
    return (left and right) or (top and bottom)


def _pair_overlap_is_large_enough(box_a: Box, box_b: Box) -> bool:
    r0 = max(box_a[0], box_b[0])
    c0 = max(box_a[1], box_b[1])
    r1 = min(box_a[2], box_b[2])
    c1 = min(box_a[3], box_b[3])
    return (r1 - r0 + 1) >= 2 and (c1 - c0 + 1) >= 2


def _pair_front_color(grid: np.ndarray, color_a: int, box_a: Box, color_b: int, box_b: Box) -> int:
    pts = _perimeter_points(box_a) & _perimeter_points(box_b)
    assert pts, "every frame pair must have at least one border-border crossing"
    visible = {int(grid[r, c]) for r, c in pts if int(grid[r, c]) in (color_a, color_b)}
    assert visible, "a frame pair must expose at least one direct occlusion witness"
    assert len(visible) == 1, "pairwise visibility witnesses must agree on one front frame"
    return next(iter(visible))


def _recover_boxes(grid: np.ndarray) -> dict[int, Box]:
    colors = sorted(int(c) for c in np.unique(grid) if int(c) != BG)
    assert 3 <= len(colors) <= 4, "expected 3-4 colored frames"
    boxes: dict[int, Box] = {}
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.size > 0
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        box = int(r0), int(c0), int(r1), int(c1)
        assert 4 <= _box_h(box) <= 8 and 4 <= _box_w(box) <= 8
        for rr, cc in coords:
            assert rr in (r0, r1) or cc in (c0, c1), "visible pixels must lie on a frame border"
        boxes[color] = box
    return boxes


def _recover_depth_order(grid: np.ndarray, boxes: dict[int, Box]) -> list[int]:
    colors = list(boxes)
    edges = {color: set() for color in colors}
    indeg = {color: 0 for color in colors}
    for color_a, color_b in itertools.combinations(colors, 2):
        front = _pair_front_color(grid, color_a, boxes[color_a], color_b, boxes[color_b])
        back = color_b if front == color_a else color_a
        if back not in edges[front]:
            edges[front].add(back)
            indeg[back] += 1

    order: list[int] = []
    remaining = set(colors)
    while remaining:
        zeros = sorted(color for color in remaining if indeg[color] == 0)
        assert len(zeros) == 1, "pairwise occlusions must define a unique total depth order"
        front = zeros[0]
        order.append(front)
        remaining.remove(front)
        for back in sorted(edges[front]):
            indeg[back] -= 1
    return order


def _stack_frames(order_front: list[int], boxes: dict[int, Box]) -> np.ndarray:
    widths = [_box_w(boxes[color]) for color in order_front]
    heights = [_box_h(boxes[color]) for color in order_front]
    out_h = sum(heights) + len(order_front) - 1
    out_w = max(widths)
    assert out_h <= 30 and out_w <= 30
    out = np.zeros((out_h, out_w), dtype=int)
    row = 0
    for color in order_front:
        h = _box_h(boxes[color])
        w = _box_w(boxes[color])
        col = (out_w - w) // 2
        _draw_frame(out, (row, col, row + h - 1, col + w - 1), color)
        row += h + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full hollow frames and restack them by front-to-back depth.

    Input structure:
    - The grid is mostly black background.
    - It contains 3-4 differently colored 1-pixel-thick hollow rectangles whose
      visible borders may be partially hidden by other frames in front.
    - Each color still exposes enough border pixels to recover that frame's full
      bounding box, and every pair of frames has at least one visible border
      crossing that reveals which one is in front.

    Rule:
    - For each color, recover the full rectangle by taking the min/max row and
      min/max column of its visible pixels.
    - For every pair of recovered rectangles, inspect the pixels where their full
      borders intersect; the visible color at those crossings is the front frame.
    - Topologically sort the resulting front-before-back relations and output the
      complete hollow frames stacked vertically from front to back, separated by
      one black row and centered horizontally within the widest frame.

    Key insight:
    - Occlusion removes border pixels but also encodes depth: the remaining color
      extremes recover each full frame, while the surviving crossing colors reveal
      the depth order needed for the final stack.
    """
    g = np.asarray(grid, dtype=int)
    boxes = _recover_boxes(g)
    order_front = _recover_depth_order(g, boxes)
    return _stack_frames(order_front, boxes)


def _scene_is_nontrivial(order_front: list[int], boxes_by_color: dict[int, Box]) -> bool:
    top_left = sorted(order_front, key=lambda color: (boxes_by_color[color][0], boxes_by_color[color][1], color))
    area_asc = sorted(order_front, key=lambda color: (_box_area(boxes_by_color[color]), boxes_by_color[color][0], color))
    area_desc = list(reversed(area_asc))
    color_asc = sorted(order_front)
    color_desc = list(reversed(color_asc))
    return order_front not in (top_left, area_asc, area_desc, color_asc, color_desc)


def _validate_scene(grid: np.ndarray, order_front: list[int], boxes_by_color: dict[int, Box]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not _scene_is_nontrivial(order_front, boxes_by_color):
        return False
    if sum(_box_h(boxes_by_color[color]) for color in order_front) + len(order_front) - 1 > 30:
        return False

    for color in order_front:
        if not _visible_opposite_sides(grid, boxes_by_color[color], color):
            return False

    for color_a, color_b in itertools.combinations(order_front, 2):
        if not _pair_overlap_is_large_enough(boxes_by_color[color_a], boxes_by_color[color_b]):
            return False
        pts = _perimeter_points(boxes_by_color[color_a]) & _perimeter_points(boxes_by_color[color_b])
        if not pts:
            return False
        front = color_a if order_front.index(color_a) < order_front.index(color_b) else color_b
        back = color_b if front == color_a else color_a
        values = {int(grid[r, c]) for r, c in pts}
        if front not in values or back in values:
            return False

    recovered_boxes = _recover_boxes(grid)
    if recovered_boxes != boxes_by_color:
        return False
    recovered_order = _recover_depth_order(grid, recovered_boxes)
    if recovered_order != order_front:
        return False
    expected = _stack_frames(order_front, boxes_by_color)
    actual = main(grid)
    return bool(np.array_equal(actual, expected))


def _instantiate_scene(
    shape: tuple[int, int],
    depth_boxes: list[Box],
    rng: random.Random,
) -> tuple[np.ndarray, list[int], dict[int, Box]] | None:
    shape, depth_boxes = _normalize_scene(shape, depth_boxes)
    transpose = bool(rng.randrange(2))
    flip_v = bool(rng.randrange(2))
    flip_h = bool(rng.randrange(2))
    shape, depth_boxes = _transform_scene(shape, depth_boxes, transpose, flip_v, flip_h)

    scene_h, scene_w = shape
    canvas_h = rng.randint(max(14, scene_h), 16)
    canvas_w = rng.randint(max(14, scene_w), 16)
    off_r = rng.randint(0, canvas_h - scene_h)
    off_c = rng.randint(0, canvas_w - scene_w)

    order_front = rng.sample(PALETTE, len(depth_boxes))
    boxes_by_color = {
        color: _translate_box(box, off_r, off_c)
        for color, box in zip(order_front, depth_boxes)
    }
    grid = _render_scene((canvas_h, canvas_w), boxes_by_color, order_front)
    if not _validate_scene(grid, order_front, boxes_by_color):
        return None
    return grid, order_front, boxes_by_color


def _generate_three_box_scene(rng: random.Random) -> np.ndarray | None:
    local_shape = (16, 16)
    for _ in range(120):
        heights = rng.choice(THREE_BOX_HEIGHTS)
        base_r = rng.randint(6, 9)
        base_c = rng.randint(6, 9)
        depth_boxes: list[Box] = []
        for width, height in zip(THREE_BOX_WIDTHS, heights):
            dr = rng.randint(-2, 2)
            dc = rng.randint(-2, 2)
            center_r = base_r + dr
            center_c = base_c + dc
            r0 = center_r - height // 2
            c0 = center_c - width // 2
            box = (r0, c0, r0 + height - 1, c0 + width - 1)
            if box[0] < 0 or box[1] < 0 or box[2] >= local_shape[0] or box[3] >= local_shape[1]:
                depth_boxes = []
                break
            depth_boxes.append(box)
        if len(depth_boxes) != 3:
            continue
        result = _instantiate_scene(local_shape, depth_boxes, rng)
        if result is not None:
            return result[0]
    return None


def _generate_four_box_scene(rng: random.Random) -> np.ndarray | None:
    template = rng.choice(FOUR_BOX_TEMPLATES)
    result = _instantiate_scene(template["shape"], list(template["boxes"]), rng)
    return None if result is None else result[0]


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        if rng.random() < 0.45:
            grid = _generate_four_box_scene(rng)
        else:
            grid = _generate_three_box_scene(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapping-frame scene")
