from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

from arc_utils import (
    Color,
    bbox,
    bbox_coords,
    bbox_h,
    bbox_w,
    colors_present,
    components_mask,
    get_bg,
)


BBox = tuple[int, int, int, int]
_INPUT_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
_SIDES = ("top", "bottom", "left", "right")
_MAX_ATTEMPTS = 400


def _rect_area(box: BBox) -> int:
    return int(bbox_h(box) * bbox_w(box))


def _shift_box(box: BBox, dr: int, dc: int) -> BBox:
    r0, c0, r1, c1 = box
    return int(r0 + dr), int(c0 + dc), int(r1 + dr), int(c1 + dc)


def _strict_contains(outer: BBox, inner: BBox) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and inner[2] < outer[2]
        and inner[3] < outer[3]
    )


def _draw_frame(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _sorted_layers(grid: np.ndarray) -> tuple[int, list[tuple[int, int, BBox]]]:
    bg = int(get_bg(grid))
    layers: list[tuple[int, int, BBox]] = []
    for color in colors_present(grid, exclude={bg}):
        box = bbox(grid == int(color))
        assert box is not None, "every foreground color must define a bbox"
        layers.append((_rect_area(box), int(color), box))

    assert len(layers) >= 2, "expected at least two nested layers"
    areas = [area for area, _color, _box in layers]
    assert len(set(areas)) == len(areas), "layer bbox areas must be unique"

    layers.sort(key=lambda item: item[0], reverse=True)
    for i in range(len(layers) - 1):
        assert _strict_contains(
            layers[i][2], layers[i + 1][2]
        ), "layer bboxes must be strictly nested"
    return bg, layers


def _frame_canvas(shape: tuple[int, int], layers: list[tuple[int, int, BBox]], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for _area, color, box in layers:
        _draw_frame(out, box, color)
    return out


def _component_touches_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    height, width = map(int, shape)
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == height - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == width - 1)
    )


def _component_depth(r: int, c: int, boxes: list[BBox]) -> int:
    return sum(
        int(r0 < r < r1 and c0 < c < c1)
        for r0, c0, r1, c1 in boxes
    )


def _candidate_gap_components(frame: np.ndarray, boxes: list[BBox], bg: int) -> list[np.ndarray]:
    comps: list[np.ndarray] = []
    black_mask = frame == int(bg)
    for coords in components_mask(black_mask, connectivity=4):
        if _component_touches_boundary(coords, frame.shape):
            continue
        r, c = map(int, coords[0])
        depth = _component_depth(r, c, boxes)
        if 1 <= depth < len(boxes):
            comps.append(coords)
    return comps


def _fill_square_gaps(frame: np.ndarray, boxes: list[BBox], bg: int) -> np.ndarray:
    out = frame.copy()
    for coords in _candidate_gap_components(frame, boxes, bg):
        box = bbox_coords(coords)
        height = bbox_h(box)
        width = bbox_w(box)
        if height == width and int(coords.shape[0]) == height * width:
            out[coords[:, 0], coords[:, 1]] = Color.GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete sparse nested rectangle samples into frames, then fill only square inter-frame holes with gray.

    Input structure:
    - The background is black and 3-4 non-black colors appear as sparse samples of axis-aligned rectangles.
    - Each color's pixels define one bbox, and the bboxes are strictly nested with distinct areas.
    - The sparse samples do not already show the full rectangles; the full frame geometry must be inferred from the bboxes.

    Rule:
    - For each foreground color, complete its bbox into a one-cell-thick rectangular frame and redraw the frames from largest bbox to smallest.
    - Find every 4-connected black component that lies strictly between two adjacent nesting depths, excluding the outside background and the innermost cavity.
    - If such a component has a square bbox and completely fills that bbox, recolor the whole component gray; otherwise leave it black.

    Key insight:
    - The decisive holes are not explicit in the input; they emerge only after reconstructing the nested rectangles, and only the square fully-rectangular inter-frame gaps are filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, layers = _sorted_layers(g)
    boxes = [box for _area, _color, box in layers]
    frame = _frame_canvas(g.shape, layers, bg)
    return _fill_square_gaps(frame, boxes, bg)


def _wrap_boxes(boxes: list[BBox], side: str, gap: int) -> list[BBox]:
    child = boxes[0]
    child_h = bbox_h(child)
    child_w = bbox_w(child)

    if side == "right":
        parent_h, parent_w = child_h + 2, child_w + int(gap) + 2
        dr, dc = 1, 1
    elif side == "left":
        parent_h, parent_w = child_h + 2, child_w + int(gap) + 2
        dr, dc = 1, 1 + int(gap)
    elif side == "top":
        parent_h, parent_w = child_h + int(gap) + 2, child_w + 2
        dr, dc = 1 + int(gap), 1
    else:
        parent_h, parent_w = child_h + int(gap) + 2, child_w + 2
        dr, dc = 1, 1

    shifted = [_shift_box(box, dr, dc) for box in boxes]
    parent = (0, 0, int(parent_h - 1), int(parent_w - 1))
    return [parent] + shifted


def _boundary_lines_unique(boxes: list[BBox]) -> bool:
    seen: set[tuple[str, int]] = set()
    for r0, c0, r1, c1 in boxes:
        keys = (("r", int(r0)), ("r", int(r1)), ("c", int(c0)), ("c", int(c1)))
        for key in keys:
            if key in seen:
                return False
            seen.add(key)
    return True


def _non_square_gap(span: int) -> int:
    choices = [gap for gap in range(1, 5) if gap != int(span)]
    return random.choice(choices)


def _build_boxes() -> list[BBox] | None:
    layer_count = random.choice((3, 4))
    gap_kinds = ["square", "non_square"]
    while len(gap_kinds) < layer_count - 1:
        gap_kinds.append(random.choice(("square", "non_square")))
    random.shuffle(gap_kinds)

    boxes: list[BBox] = [(0, 0, random.randint(3, 5), random.randint(3, 5))]
    for kind in gap_kinds:
        child = boxes[0]
        side = random.choice(_SIDES)
        span = bbox_h(child) if side in ("left", "right") else bbox_w(child)
        gap = int(span) if kind == "square" else _non_square_gap(span)
        boxes = _wrap_boxes(boxes, side, gap)
        if max(bbox_h(boxes[0]), bbox_w(boxes[0])) > 20:
            return None

    areas = [_rect_area(box) for box in boxes]
    if len(set(areas)) != len(areas):
        return None
    if not _boundary_lines_unique(boxes):
        return None
    return boxes


def _place_boxes(boxes: list[BBox]) -> tuple[tuple[int, int], list[BBox]] | None:
    outer = boxes[0]
    outer_h = bbox_h(outer)
    outer_w = bbox_w(outer)
    max_h = min(24, outer_h + 6)
    max_w = min(24, outer_w + 6)
    min_h = outer_h + 2
    min_w = outer_w + 2
    if min_h > max_h or min_w > max_w:
        return None

    height = random.randint(min_h, max_h)
    width = random.randint(min_w, max_w)
    top = random.randint(1, height - outer_h - 1)
    left = random.randint(1, width - outer_w - 1)
    placed = [_shift_box(box, top, left) for box in boxes]
    return (height, width), placed


def _sparse_points(box: BBox) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    rm = (r0 + r1) // 2
    cm = (c0 + c1) // 2
    ordered = [
        (r0, c0),
        (r0, c1),
        (r1, c0),
        (r1, c1),
        (r0, cm),
        (r1, cm),
        (rm, c0),
        (rm, c1),
    ]
    points: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for point in ordered:
        if point not in seen:
            seen.add(point)
            points.append(point)
    return points


def _render_input(shape: tuple[int, int], boxes: list[BBox], colors: list[int]) -> np.ndarray:
    grid = np.full(shape, Color.BLACK, dtype=int)
    for box, color in zip(boxes, colors):
        points = _sparse_points(box)
        assert 4 <= len(points) <= 8
        for r, c in points:
            grid[r, c] = int(color)
    return grid


def _gap_stats(frame: np.ndarray, boxes: list[BBox]) -> list[tuple[bool, bool]]:
    stats: list[tuple[bool, bool]] = []
    for coords in _candidate_gap_components(frame, boxes, Color.BLACK):
        box = bbox_coords(coords)
        is_square = bbox_h(box) == bbox_w(box)
        fills_bbox = int(coords.shape[0]) == bbox_h(box) * bbox_w(box)
        stats.append((is_square, fills_bbox))
    return stats


def _valid_sample(grid: np.ndarray) -> bool:
    try:
        bg, layers = _sorted_layers(grid)
        assert bg == Color.BLACK
        assert 3 <= len(layers) <= 4
        colors = [color for _area, color, _box in layers]
        assert len(set(colors)) == len(colors)
        for color in colors:
            count = int(np.count_nonzero(grid == color))
            assert 4 <= count <= 8

        boxes = [box for _area, _color, box in layers]
        frame = _frame_canvas(grid.shape, layers, bg)
        stats = _gap_stats(frame, boxes)
        assert stats
        assert any(is_square and fills_bbox for is_square, fills_bbox in stats)
        assert any(not (is_square and fills_bbox) for is_square, fills_bbox in stats)

        expected = _fill_square_gaps(frame, boxes, bg)
        actual = main(grid)
        assert np.array_equal(actual, expected)
        assert int(np.count_nonzero(actual == Color.GREY)) > 0
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    boxes = _build_boxes()
    if boxes is None:
        return None
    placed = _place_boxes(boxes)
    if placed is None:
        return None

    shape, placed_boxes = placed
    colors = random.sample(_INPUT_COLORS, len(placed_boxes))
    grid = _render_input(shape, placed_boxes, colors)
    if _valid_sample(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame square-gap sample")
