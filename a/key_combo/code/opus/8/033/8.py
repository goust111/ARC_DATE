from __future__ import annotations

import random

import numpy as np


ColorSeq = list[int]
BBox = tuple[int, int, int, int]


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _intersect(a: BBox, b: BBox) -> BBox | None:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _paint_rect(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _bboxes_from_visible(grid: np.ndarray, colors: ColorSeq) -> dict[int, BBox]:
    boxes: dict[int, BBox] = {}
    for color in colors:
        box = _bbox(grid == int(color))
        assert box is not None, "every rectangle color must remain visible"
        boxes[int(color)] = box
    return boxes


def _direct_edges(grid: np.ndarray, boxes: dict[int, BBox], colors: ColorSeq) -> dict[int, set[int]]:
    edges = {int(color): set() for color in colors}
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            inter = _intersect(boxes[int(a)], boxes[int(b)])
            if inter is None:
                continue
            r0, c0, r1, c1 = inter
            vals = set(map(int, np.unique(grid[r0 : r1 + 1, c0 : c1 + 1])))
            assert 0 not in vals, "overlap area cannot expose background"
            assert len(vals) == 1, "pairwise overlap must reveal exactly one visible color"
            top = next(iter(vals))
            assert top in (int(a), int(b)), "a third rectangle cannot enter a pairwise overlap"
            if top == int(a):
                edges[int(b)].add(int(a))
            else:
                edges[int(a)].add(int(b))
    return edges


def _total_order_from_edges(colors: ColorSeq, edges: dict[int, set[int]]) -> list[int]:
    k = len(colors)
    index = {int(color): i for i, color in enumerate(colors)}
    reach = np.zeros((k, k), dtype=bool)
    for low, highs in edges.items():
        for high in highs:
            reach[index[int(low)], index[int(high)]] = True

    for mid in range(k):
        reach |= reach[:, [mid]] & reach[[mid], :]

    assert not np.any(np.diag(reach)), "occlusion order cannot contain a cycle"
    for i in range(k):
        for j in range(i + 1, k):
            assert reach[i, j] != reach[j, i], "all rectangles must be comparable after transitive closure"

    above_counts = reach.sum(axis=1)
    assert len(set(map(int, above_counts))) == k, "total order must assign a unique depth rank to every rectangle"
    order_idx = list(np.argsort(-above_counts))
    return [int(colors[i]) for i in order_idx]


def _solve_order(grid: np.ndarray) -> list[int]:
    colors = [int(color) for color in np.unique(grid) if int(color) != 0]
    assert 3 <= len(colors) <= 5, "expected 3 to 5 visible rectangles"
    boxes = _bboxes_from_visible(grid, colors)
    edges = _direct_edges(grid, boxes, colors)
    return _total_order_from_edges(colors, edges)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the bottom-to-top order of overlapping filled rectangles and compress it into one row.

    Input structure:
    - The grid has black background plus 3 to 5 differently colored filled rectangles.
    - Rectangles are axis-aligned, partially overlap, and only the topmost color is visible in each covered cell.
    - Every rectangle still exposes at least one visible pixel, and the visible pixels of each color have the same bbox as that rectangle's full extent.

    Rule:
    - For each visible color, treat the bbox of its visible pixels as that rectangle's full rectangle.
    - For every pair of rectangles whose bboxes overlap, inspect the overlap area: if it is entirely one rectangle's color, that rectangle is above the other.
    - Take the transitive closure of all such pairwise above/below relations to recover the unique full layer order.
    - Output a 1xK row listing the rectangle colors from bottom layer at the left to top layer at the right.

    Key insight:
    - No single overlap reveals the whole answer; the rectangles must be compared pairwise and then globally ordered by transitive occlusion constraints.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    order = _solve_order(g)
    return np.array([order], dtype=int)


_BG = 0
_PALETTE = (1, 2, 3, 4, 6)
_COUNT_RANGE = (3, 5)
_SIDE_RANGE = (3, 6)
_CANVAS_MAX = 30
_GEN_ATTEMPTS = 500


def _shift_bbox(box: BBox, dr: int, dc: int) -> BBox:
    return int(box[0] + dr), int(box[1] + dc), int(box[2] + dr), int(box[3] + dc)


def _place_next(prev: BBox, cur_h: int, cur_w: int, sign_r: int, sign_c: int, overlap_h: int, overlap_w: int) -> BBox:
    if sign_r > 0:
        r0 = int(prev[2] - overlap_h + 1)
    else:
        r0 = int(prev[0] - cur_h + overlap_h)

    if sign_c > 0:
        c0 = int(prev[3] - overlap_w + 1)
    else:
        c0 = int(prev[1] - cur_w + overlap_w)

    return int(r0), int(c0), int(r0 + cur_h - 1), int(c0 + cur_w - 1)


def _render(rects: list[BBox], colors: ColorSeq) -> np.ndarray:
    h = max(int(box[2]) for box in rects) + 1
    w = max(int(box[3]) for box in rects) + 1
    grid = np.full((h, w), _BG, dtype=int)
    for box, color in zip(rects, colors):
        _paint_rect(grid, box, int(color))
    return grid


def _chain_rectangles() -> tuple[list[BBox], list[int]] | None:
    count = random.randint(*_COUNT_RANGE)
    colors = random.sample(list(_PALETTE), count)
    sizes = [
        (
            random.randint(_SIDE_RANGE[0], _SIDE_RANGE[1]),
            random.randint(_SIDE_RANGE[0], _SIDE_RANGE[1]),
        )
        for _ in range(count)
    ]
    sign_r, sign_c = random.choice(((-1, -1), (-1, 1), (1, -1), (1, 1)))
    tight_axis = random.choice(("row", "col"))

    rects: list[BBox] = []
    h0, w0 = sizes[0]
    rects.append((0, 0, h0 - 1, w0 - 1))

    for i in range(1, count):
        prev = rects[-1]
        prev_h = int(prev[2] - prev[0] + 1)
        prev_w = int(prev[3] - prev[1] + 1)
        cur_h, cur_w = sizes[i]

        if tight_axis == "row":
            overlap_h = 1
            overlap_w = random.randint(1, min(3, prev_w - 1, cur_w - 1))
        else:
            overlap_h = random.randint(1, min(3, prev_h - 1, cur_h - 1))
            overlap_w = 1

        cur = _place_next(prev, cur_h, cur_w, sign_r, sign_c, overlap_h, overlap_w)

        for older in rects[:-1]:
            if _intersect(older, cur) is not None:
                return None
        rects.append(cur)

    min_r = min(int(box[0]) for box in rects)
    min_c = min(int(box[1]) for box in rects)
    rects = [_shift_bbox(box, -min_r, -min_c) for box in rects]

    total_h = max(int(box[2]) for box in rects) + 1
    total_w = max(int(box[3]) for box in rects) + 1
    if total_h > _CANVAS_MAX or total_w > _CANVAS_MAX:
        return None

    slack_h = _CANVAS_MAX - total_h
    slack_w = _CANVAS_MAX - total_w
    pad_r = random.randint(0, min(4, slack_h))
    pad_c = random.randint(0, min(4, slack_w))
    rects = [_shift_bbox(box, pad_r, pad_c) for box in rects]
    return rects, colors


def _validate_scene(rects: list[BBox], colors: ColorSeq, grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > _CANVAS_MAX:
        return False

    boxes = _bboxes_from_visible(grid, colors)
    expected_boxes = {int(color): rect for color, rect in zip(colors, rects)}
    if boxes != expected_boxes:
        return False

    for i, box_a in enumerate(rects):
        for j, box_b in enumerate(rects):
            if i >= j:
                continue
            inter = _intersect(box_a, box_b)
            if abs(i - j) == 1:
                if inter is None:
                    return False
            elif inter is not None:
                return False

    direct = _direct_edges(grid, boxes, colors)
    expected_direct = {int(color): set() for color in colors}
    for i in range(len(colors) - 1):
        expected_direct[int(colors[i])].add(int(colors[i + 1]))
    if direct != expected_direct:
        return False

    try:
        order = _total_order_from_edges(colors, direct)
    except AssertionError:
        return False
    if order != [int(color) for color in colors]:
        return False

    visible_areas = {int(color): int(np.count_nonzero(grid == int(color))) for color in colors}
    if any(area <= 0 for area in visible_areas.values()):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    sampled = _chain_rectangles()
    if sampled is None:
        return None
    rects, colors = sampled
    grid = _render(rects, colors)
    if not _validate_scene(rects, colors, grid):
        return None
    expected = np.array([[int(color) for color in colors]], dtype=int)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapping-rectangle layering example")
