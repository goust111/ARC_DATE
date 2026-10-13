from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
SEPARATOR_COLOR = 5
RECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
VISIBLE_CORNERS = ("tl", "tr", "bl", "br")

MIN_RECT_COUNT = 2
MAX_RECT_COUNT = 4
TOP_SIZE_RANGE = (2, 4)
GAP_RANGE = (1, 2)
TOP_MARGIN_RANGE = (1, 3)
BOTTOM_MARGIN_RANGE = (1, 3)
LEFT_MARGIN_RANGE = (0, 2)
MAX_HEIGHT = 18
MAX_WIDTH = 30
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return _bbox_h(bbox) * _bbox_w(bbox)


def _overlap_bbox(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return (int(r0), int(c0), int(r1), int(c1))


def _contains_bbox(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(w - 1):
        col = grid[:, c]
        if int(col[0]) == bg or not np.all(col == col[0]):
            continue
        if not np.any(grid[:, :c] != bg):
            continue
        if np.any(grid[:, c + 1 :] != bg):
            continue
        candidates.append((int(c), int(col[0])))
    assert len(candidates) == 1, "expected a unique separator column"
    return candidates[0]


def _visible_in_bbox(scene: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(np.any(scene[r0 : r1 + 1, c0 : c1 + 1] == color))


def _depth_order(
    scene: np.ndarray, rects: dict[int, tuple[int, int, int, int]]
) -> list[int]:
    edges = {color: set() for color in rects}
    indeg = {color: 0 for color in rects}

    for a, b in combinations(rects, 2):
        bbox_a = rects[a]
        bbox_b = rects[b]
        if _overlap_bbox(bbox_a, bbox_b) is None:
            continue

        a_visible_in_b = _visible_in_bbox(scene, a, bbox_b)
        b_visible_in_a = _visible_in_bbox(scene, b, bbox_a)

        if b_visible_in_a and not a_visible_in_b:
            lower, upper = a, b
        elif a_visible_in_b and not b_visible_in_a:
            lower, upper = b, a
        elif _contains_bbox(bbox_a, bbox_b) and bbox_a != bbox_b:
            lower, upper = a, b
        elif _contains_bbox(bbox_b, bbox_a) and bbox_a != bbox_b:
            lower, upper = b, a
        else:
            raise AssertionError("ambiguous overlap relation")

        if upper not in edges[lower]:
            edges[lower].add(upper)
            indeg[upper] += 1

    def sort_key(color: int) -> tuple[int, int, int, int]:
        bbox = rects[color]
        return (-_bbox_area(bbox), bbox[0], bbox[1], bbox[2])

    ready = sorted([color for color, deg in indeg.items() if deg == 0], key=sort_key)
    order: list[int] = []
    while ready:
        color = ready.pop(0)
        order.append(color)
        for nxt in sorted(edges[color], key=sort_key):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=sort_key)

    assert len(order) == len(rects), "depth graph must be acyclic and complete"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct hidden solid rectangles from an occluded stack and pack them in depth order.

    Input structure:
    - The background is black.
    - The left side contains several unique-color filled rectangles that overlap.
    - A full-height monochrome separator column splits the scene from a blank output area.

    Rule:
    - Recover each rectangle from the bounding box implied by its visible corner pixels.
    - Infer bottom-to-top order from which color is visible inside another rectangle's span.
    - Draw the recovered rectangles into the right-side output area, top-aligned and separated
      by one black column, while keeping the scene and separator unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    sep_col, _sep_color = _find_separator(g, bg)

    scene = g[:, :sep_col]
    out = g.copy()
    out[:, sep_col + 1 :] = bg

    colors = [int(color) for color in np.unique(scene) if int(color) != bg]
    assert colors, "expected at least one visible rectangle color"

    rects: dict[int, tuple[int, int, int, int]] = {}
    for color in colors:
        coords = np.argwhere(scene == color)
        assert coords.size > 0
        rects[color] = _bbox_from_coords(coords)

    order = _depth_order(scene, rects)

    out_c = sep_col + 1
    for color in order:
        bbox = rects[color]
        height = _bbox_h(bbox)
        width = _bbox_w(bbox)
        assert out_c + width <= g.shape[1], "output area is too narrow"
        out[:height, out_c : out_c + width] = color
        out_c += width + 1

    return out


def _shift_bboxes(
    boxes: list[tuple[int, int, int, int]], dr: int, dc: int
) -> list[tuple[int, int, int, int]]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in boxes]


def _add_outer_box(
    boxes: list[tuple[int, int, int, int]], visible_corner: str, dv: int, dh: int
) -> list[tuple[int, int, int, int]]:
    inner = boxes[0]
    inner_h = _bbox_h(inner)
    inner_w = _bbox_w(inner)

    if visible_corner == "tl":
        shifted = _shift_bboxes(boxes, dv, dh)
    elif visible_corner == "tr":
        shifted = _shift_bboxes(boxes, dv, 0)
    elif visible_corner == "bl":
        shifted = _shift_bboxes(boxes, 0, dh)
    else:
        assert visible_corner == "br"
        shifted = boxes[:]

    outer = (0, 0, inner_h + dv - 1, inner_w + dh - 1)
    return [outer] + shifted


def _build_boxes() -> list[tuple[int, int, int, int]] | None:
    count = random.randint(MIN_RECT_COUNT, MAX_RECT_COUNT)
    top_h = random.randint(*TOP_SIZE_RANGE)
    top_w = random.randint(*TOP_SIZE_RANGE)
    boxes = [(0, 0, top_h - 1, top_w - 1)]

    for _ in range(count - 1):
        dv = random.randint(*GAP_RANGE)
        dh = random.randint(*GAP_RANGE)
        corner = random.choice(VISIBLE_CORNERS)
        boxes = _add_outer_box(boxes, corner, dv, dh)

    packed_w = sum(_bbox_w(bbox) for bbox in boxes) + len(boxes) - 1
    outer_h = _bbox_h(boxes[0])
    outer_w = _bbox_w(boxes[0])
    if packed_w + outer_w + 1 > MAX_WIDTH:
        return None
    if outer_h + TOP_MARGIN_RANGE[1] + BOTTOM_MARGIN_RANGE[1] > MAX_HEIGHT:
        return None
    return boxes


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _render_expected(
    inp: np.ndarray,
    sep_col: int,
    colors: list[int],
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = inp.copy()
    out[:, sep_col + 1 :] = BG
    c = sep_col + 1
    for color, bbox in zip(colors, boxes):
        h = _bbox_h(bbox)
        w = _bbox_w(bbox)
        out[:h, c : c + w] = int(color)
        c += w + 1
    return out


def _gen_attempt() -> np.ndarray | None:
    boxes = _build_boxes()
    if boxes is None:
        return None

    outer_h = _bbox_h(boxes[0])
    outer_w = _bbox_w(boxes[0])
    packed_w = sum(_bbox_w(bbox) for bbox in boxes) + len(boxes) - 1
    top_margin = random.randint(*TOP_MARGIN_RANGE)
    bottom_margin = random.randint(*BOTTOM_MARGIN_RANGE)
    left_margin = random.randint(*LEFT_MARGIN_RANGE)

    grid_h = top_margin + outer_h + bottom_margin
    grid_w = left_margin + outer_w + 1 + packed_w
    if grid_h > MAX_HEIGHT or grid_w > MAX_WIDTH:
        return None

    colors = random.sample(RECT_COLORS, len(boxes))
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    placed_boxes = [
        (
            bbox[0] + top_margin,
            bbox[1] + left_margin,
            bbox[2] + top_margin,
            bbox[3] + left_margin,
        )
        for bbox in boxes
    ]

    for color, bbox in zip(colors, placed_boxes):
        _paint_rect(grid, bbox, color)

    sep_col = left_margin + outer_w
    grid[:, sep_col] = SEPARATOR_COLOR

    scene = grid[:, :sep_col]
    for color, bbox in zip(colors, placed_boxes):
        coords = np.argwhere(scene == color)
        if coords.size == 0:
            return None
        if _bbox_from_coords(coords) != bbox:
            return None

    expected = _render_expected(grid, sep_col, colors, placed_boxes)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
