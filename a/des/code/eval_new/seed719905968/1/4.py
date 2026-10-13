import random
from typing import Dict, List, Tuple

import numpy as np


BG = 0
GREY = 5
AZURE = 8
FRAGMENT_COLORS = (1, 2, 3, 4, 6, 7, 9)
MAX_RETRIES = 400


def _bbox(coords: np.ndarray) -> Tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> List[int]:
    excluded = set() if exclude is None else set(exclude)
    return [int(color) for color in np.unique(grid) if int(color) not in excluded]


def _separator_col(grid: np.ndarray) -> int:
    cols = np.flatnonzero(np.all(grid == GREY, axis=0))
    assert len(cols) == 1, "expected one full-height grey separator"
    return int(cols[0])


def _frame_bbox(grid: np.ndarray) -> Tuple[int, int, int, int]:
    coords = np.argwhere(grid == AZURE)
    assert coords.size, "expected one azure target frame"
    r0, c0, r1, c1 = _bbox(coords)
    frame_mask = np.zeros_like(grid, dtype=bool)
    frame_mask[r0, c0 : c1 + 1] = True
    frame_mask[r1, c0 : c1 + 1] = True
    frame_mask[r0 : r1 + 1, c0] = True
    frame_mask[r0 : r1 + 1, c1] = True
    assert np.array_equal(grid == AZURE, frame_mask), "azure pixels must form one rectangular border"
    return r0, c0, r1, c1


def _infer_order(fragments: Dict[int, Dict[str, np.ndarray | Tuple[int, int, int, int]]]) -> List[int]:
    colors = list(fragments)
    above_of: Dict[int, set[int]] = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}

    for lower in colors:
        lr0, lc0, lr1, lc1 = fragments[lower]["bbox"]  # type: ignore[index]
        for upper in colors:
            if upper == lower:
                continue
            coords = fragments[upper]["coords"]  # type: ignore[index]
            inside = (
                (coords[:, 0] >= lr0)
                & (coords[:, 0] <= lr1)
                & (coords[:, 1] >= lc0)
                & (coords[:, 1] <= lc1)
            )
            if np.any(inside):
                above_of[lower].add(upper)

    for lower, uppers in above_of.items():
        for upper in uppers:
            indegree[upper] += 1

    order: List[int] = []
    remaining = set(colors)
    while remaining:
        available = [color for color in remaining if indegree[color] == 0]
        assert len(available) == 1, "occlusion evidence must define a unique total order"
        current = available[0]
        remaining.remove(current)
        order.append(current)
        for upper in above_of[current]:
            indegree[upper] -= 1
    return order


def _centered_rect_start(
    inner_bbox: Tuple[int, int, int, int], rect_h: int, rect_w: int
) -> Tuple[int, int]:
    ir0, ic0, ir1, ic1 = inner_bbox
    inner_h = ir1 - ir0 + 1
    inner_w = ic1 - ic0 + 1
    assert rect_h <= inner_h and rect_w <= inner_w
    top = ir0 + (inner_h - rect_h) // 2
    left = ic0 + (inner_w - rect_w) // 2
    return top, left


def _render_target(
    grid: np.ndarray,
    frame_bbox: Tuple[int, int, int, int],
    ordered_rects: List[Tuple[int, int, int]],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    fr0, fc0, fr1, fc1 = frame_bbox
    inner_bbox = (fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1)
    out[fr0 + 1 : fr1, fc0 + 1 : fc1] = BG
    for color, rect_h, rect_w in ordered_rects:
        top, left = _centered_rect_start(inner_bbox, rect_h, rect_w)
        out[top : top + rect_h, left : left + rect_w] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the occluded rectangles and restack them inside the azure target frame.

    Input structure:
    - Black background with a full-height grey divider splitting the grid into left and right regions.
    - The left region contains several distinct-colored visible fragments produced by overlapping solid axis-aligned rectangles.
    - The right region contains one azure rectangular border whose interior is black.

    Rule:
    - For each fragment color, recover its full rectangle as the bounding box of all visible pixels of that color.
    - Infer the bottom-to-top stacking order from occlusion: if color A is visible inside color B's bounding box, then A lies above B.
    - Keep the input grid layout, but fill the azure frame's interior by drawing the recovered rectangles from bottom to top, each centered in the target area with ties biased upward and leftward.

    Key insight:
    - The irregular-looking left fragments are cropped views of full rectangles, and visible pixels appearing inside another fragment's bounding box reveal the hidden stacking order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _separator_col(g)
    frame_bbox = _frame_bbox(g)
    left = g[:, :sep]

    fragment_colors = _colors_present(left, exclude={BG, GREY, AZURE})
    assert len(fragment_colors) >= 2, "expected multiple colored fragments"

    fragments: Dict[int, Dict[str, np.ndarray | Tuple[int, int, int, int]]] = {}
    for color in fragment_colors:
        coords = np.argwhere(left == color)
        assert coords.size, "fragment must be non-empty"
        fragments[color] = {"coords": coords, "bbox": _bbox(coords)}

    order = _infer_order(fragments)
    rects = []
    for color in order:
        r0, c0, r1, c1 = fragments[color]["bbox"]  # type: ignore[index]
        rects.append((color, r1 - r0 + 1, c1 - c0 + 1))

    return _render_target(g, frame_bbox, rects)


def _normalize_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    min_r = min(r0 for r0, _c0, _r1, _c1 in boxes)
    min_c = min(c0 for _r0, c0, _r1, _c1 in boxes)
    return [(r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c) for r0, c0, r1, c1 in boxes]


def _nested_boxes(n_rects: int) -> List[Tuple[int, int, int, int]] | None:
    top_h = random.randint(2, 4)
    top_w = random.randint(2, 4)
    boxes = [(0, 0, top_h - 1, top_w - 1)]  # topmost first
    for _ in range(n_rects - 1):
        r0, c0, r1, c1 = boxes[-1]
        grow_top = random.randint(1, 2)
        grow_bottom = random.randint(1, 2)
        grow_left = random.randint(1, 2)
        grow_right = random.randint(1, 2)
        boxes.append(
            (r0 - grow_top, c0 - grow_left, r1 + grow_bottom, c1 + grow_right)
        )
    boxes = _normalize_boxes(boxes)
    br0, bc0, br1, bc1 = boxes[-1]
    bottom_h = br1 - br0 + 1
    bottom_w = bc1 - bc0 + 1
    if max(bottom_h, bottom_w) > 12:
        return None
    return list(reversed(boxes))  # bottom-to-top


def _draw_rect(grid: np.ndarray, bbox: Tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _shift_box(
    bbox: Tuple[int, int, int, int], dr: int, dc: int
) -> Tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _gen_attempt() -> np.ndarray | None:
    n_rects = random.randint(3, 5)
    local_boxes = _nested_boxes(n_rects)
    if local_boxes is None:
        return None

    _, _, br1, bc1 = local_boxes[0]
    bottom_h = br1 + 1
    bottom_w = bc1 + 1

    target_inner_h = bottom_h + random.randint(0, 4)
    target_inner_w = bottom_w + random.randint(0, 4)
    target_outer_h = target_inner_h + 2
    target_outer_w = target_inner_w + 2

    left_region_w = bottom_w + random.randint(2, 5)
    right_region_w = target_outer_w + random.randint(2, 5)
    grid_w = left_region_w + 1 + right_region_w

    grid_h = max(bottom_h, target_outer_h) + random.randint(2, 5)
    if grid_h > 30 or grid_w > 30:
        return None
    if left_region_w - bottom_w < 2 or right_region_w - target_outer_w < 2 or grid_h - bottom_h < 2 or grid_h - target_outer_h < 2:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    sep = left_region_w
    grid[:, sep] = GREY

    left_top = random.randint(1, grid_h - bottom_h - 1)
    left_left = random.randint(1, left_region_w - bottom_w - 1)
    frame_top = random.randint(1, grid_h - target_outer_h - 1)
    frame_left = sep + 1 + random.randint(1, right_region_w - target_outer_w - 1)

    colors = random.sample(FRAGMENT_COLORS, n_rects)

    global_boxes = [_shift_box(box, left_top, left_left) for box in local_boxes]
    for color, box in zip(colors, global_boxes):
        _draw_rect(grid, box, color)

    fr0, fc0 = frame_top, frame_left
    fr1 = fr0 + target_outer_h - 1
    fc1 = fc0 + target_outer_w - 1
    grid[fr0, fc0 : fc1 + 1] = AZURE
    grid[fr1, fc0 : fc1 + 1] = AZURE
    grid[fr0 : fr1 + 1, fc0] = AZURE
    grid[fr0 : fr1 + 1, fc1] = AZURE

    ordered_rects = []
    for color, box in zip(colors, local_boxes):
        r0, c0, r1, c1 = box
        ordered_rects.append((color, r1 - r0 + 1, c1 - c0 + 1))
    expected = _render_target(grid, (fr0, fc0, fr1, fc1), ordered_rects)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} attempts")
