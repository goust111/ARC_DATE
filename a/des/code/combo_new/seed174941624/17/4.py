import random
from typing import List, Sequence, Tuple

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GRID_SIZE = 30
GENERATION_RETRIES = 400
GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 28)
RECT_COUNT_RANGE = (3, 6)
MAX_STACKS = 3
PLACEMENT_RETRIES = 150
OUTER_GAP = 1


BBox = Tuple[int, int, int, int]


def _bbox_area(bbox: BBox) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _paint_rect(canvas: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    canvas[r0 : r1 + 1, c0 : c1 + 1] = color


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boxes_are_separate(a: BBox, b: BBox, gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _merge_preserving_stack_order(stacks: Sequence[Sequence[BBox]]) -> List[BBox]:
    positions = [0] * len(stacks)
    active = [idx for idx, stack in enumerate(stacks) if stack]
    merged: List[BBox] = []
    while active:
        stack_idx = random.choice(active)
        merged.append(stacks[stack_idx][positions[stack_idx]])
        positions[stack_idx] += 1
        if positions[stack_idx] == len(stacks[stack_idx]):
            active.remove(stack_idx)
    return merged


def _sample_stack_depths(rect_count: int) -> List[int]:
    stack_count = random.randint(1, min(MAX_STACKS, rect_count - 1))
    depths = [1] * stack_count
    remaining = rect_count - stack_count
    while remaining > 0:
        depths[random.randrange(stack_count)] += 1
        remaining -= 1
    random.shuffle(depths)
    return depths


def _place_outer_boxes(
    height: int, width: int, dims: Sequence[Tuple[int, int]]
) -> List[BBox] | None:
    ordered = sorted(
        enumerate(dims), key=lambda item: item[1][0] * item[1][1], reverse=True
    )
    placed: dict[int, BBox] = {}
    accepted: List[BBox] = []
    for idx, (box_h, box_w) in ordered:
        found = None
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(0, height - box_h)
            c0 = random.randint(1, width - box_w)
            bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if all(_boxes_are_separate(bbox, other, gap=OUTER_GAP) for other in accepted):
                found = bbox
                break
        if found is None:
            return None
        placed[idx] = found
        accepted.append(found)
    return [placed[idx] for idx in range(len(dims))]


def _build_nested_stack(outer_bbox: BBox, depth: int) -> List[BBox] | None:
    chain = [outer_bbox]
    parent = outer_bbox
    for level in range(1, depth):
        remaining = depth - level
        pr0, pc0, pr1, pc1 = parent
        parent_h = pr1 - pr0 + 1
        parent_w = pc1 - pc0 + 1
        min_child_h = 2 * remaining
        min_child_w = 2 * remaining
        max_child_h = parent_h - 2
        max_child_w = parent_w - 2
        if max_child_h < min_child_h or max_child_w < min_child_w:
            return None

        child_h = random.randint(min_child_h, max_child_h)
        child_w = random.randint(min_child_w, max_child_w)
        child_r0_min = pr0 + 1
        child_r0_max = pr1 - child_h
        child_c0_min = pc0 + 1
        child_c0_max = pc1 - child_w
        if child_r0_min > child_r0_max or child_c0_min > child_c0_max:
            return None

        child_r0 = random.randint(child_r0_min, child_r0_max)
        child_c0 = random.randint(child_c0_min, child_c0_max)
        child = (
            child_r0,
            child_c0,
            child_r0 + child_h - 1,
            child_c0 + child_w - 1,
        )
        chain.append(child)
        parent = child
    return chain


def _pick_marker_rows(height: int, rect_count: int) -> List[int]:
    if height >= 2 * rect_count and random.random() < 0.75:
        candidates = list(range(height))
        rows: List[int] = []
        while candidates and len(rows) < rect_count:
            idx = random.randrange(len(candidates))
            row = candidates.pop(idx)
            rows.append(row)
            candidates = [r for r in candidates if abs(r - row) >= 2]
        if len(rows) == rect_count:
            return sorted(rows)
    return sorted(random.sample(range(height), rect_count))


def _redraw_by_area(rects: Sequence[dict], shape: Tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    ordering = sorted(rects, key=lambda rect: (rect["area"], rect["order"]))
    for rect in ordering:
        _paint_rect(out, rect["bbox"], rect["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the original solid rectangles from the left-edge draw-order markers, then redraw them from smallest area to largest.

    Input structure:
    - The background is black.
    - Column 0 contains one-pixel color markers whose top-to-bottom order gives the original rectangle draw order from bottom layer to top layer.
    - Every non-black color also appears in the main scene as the visible part of one unique solid axis-aligned rectangle, possibly partially covered by later rectangles.

    Rule:
    - Ignore the left marker column when locating the rectangles.
    - For each marker color, collect all visible scene pixels of that color and recover the rectangle's full bounding box from the min/max visible row and column.
    - Compute each recovered rectangle's area, clear the canvas, then redraw the rectangles in ascending area order so smaller rectangles become lower layers and larger rectangles become upper layers.
    - Do not redraw the left-edge markers.

    Key insight:
    - The marker column identifies which colors correspond to layered rectangles, while each rectangle's surviving edge pixels still reveal its full bounding box even after occlusion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)
    assert width >= 2

    marker_rows = [r for r in range(height) if int(g[r, 0]) != BG]
    marker_colors = [int(g[r, 0]) for r in marker_rows]
    assert marker_colors
    assert len(marker_colors) == len(set(marker_colors))

    scene = g.copy()
    scene[:, 0] = BG
    visible_colors = set(map(int, np.unique(scene))) - {BG}
    assert visible_colors == set(marker_colors)

    rects = []
    for order, color in enumerate(marker_colors):
        coords = np.argwhere(scene == color)
        assert coords.size > 0
        bbox = _bbox_from_coords(coords)
        rects.append(
            {
                "color": color,
                "bbox": bbox,
                "area": _bbox_area(bbox),
                "order": order,
            }
        )

    return _redraw_by_area(rects, (height, width))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    rect_count = random.randint(*RECT_COUNT_RANGE)

    depths = _sample_stack_depths(rect_count)
    max_depth = max(depths)
    if height < 2 * max_depth or width < 2 * max_depth + 1:
        return None

    outer_dims: List[Tuple[int, int]] = []
    for depth in depths:
        min_h = 2 * depth
        min_w = 2 * depth
        max_h = min(height - 1, min_h + 7)
        max_w = min(width - 2, min_w + 9)
        if max_h < min_h or max_w < min_w:
            return None
        outer_dims.append(
            (
                random.randint(min_h, max_h),
                random.randint(min_w, max_w),
            )
        )

    outer_boxes = _place_outer_boxes(height, width, outer_dims)
    if outer_boxes is None:
        return None

    stacks: List[List[BBox]] = []
    for outer_bbox, depth in zip(outer_boxes, depths):
        stack = _build_nested_stack(outer_bbox, depth)
        if stack is None:
            return None
        stacks.append(stack)

    ordered_boxes = _merge_preserving_stack_order(stacks)
    areas = [_bbox_area(bbox) for bbox in ordered_boxes]
    if len(set(areas)) != len(areas):
        return None

    colors = random.sample(PALETTE, rect_count)
    rects = [
        {
            "color": color,
            "bbox": bbox,
            "area": _bbox_area(bbox),
            "order": order,
        }
        for order, (color, bbox) in enumerate(zip(colors, ordered_boxes))
    ]

    scene = np.full((height, width), BG, dtype=int)
    for rect in rects:
        _paint_rect(scene, rect["bbox"], rect["color"])

    has_occlusion = False
    for rect in rects:
        color = rect["color"]
        bbox = rect["bbox"]
        coords = np.argwhere(scene == color)
        if coords.size == 0:
            return None
        if _bbox_from_coords(coords) != bbox:
            return None
        if int(coords.shape[0]) < rect["area"]:
            has_occlusion = True
    if not has_occlusion:
        return None

    marker_rows = _pick_marker_rows(height, rect_count)
    grid = scene.copy()
    for row, rect in zip(marker_rows, rects):
        grid[row, 0] = rect["color"]

    expected = _redraw_by_area(rects, (height, width))
    if not np.array_equal(main(grid), expected):
        return None

    assert max(grid.shape) <= MAX_GRID_SIZE
    assert max(expected.shape) <= MAX_GRID_SIZE
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
