from __future__ import annotations

import itertools
import random

import numpy as np


GREY = 5
BG = 0
INPUT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MIN_GRID = 12
MAX_GRID = 24
MAX_ATTEMPTS = 500


BBox = tuple[int, int, int, int]


def _dominant_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _intersection(a: BBox, b: BBox) -> BBox | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return (r0, c0, r1, c1)


def _contains(box: BBox, r: int, c: int) -> bool:
    return box[0] <= r <= box[2] and box[1] <= c <= box[3]


def _extract_rectangles(grid: np.ndarray, bg: int) -> dict[int, BBox]:
    rects: dict[int, BBox] = {}
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        rects[color] = _bbox_from_coords(coords)
    return rects


def _render_visible(shape: tuple[int, int], rects: dict[int, BBox], order: tuple[int, ...], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for color in order:
        r0, c0, r1, c1 = rects[int(color)]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _valid_orders(grid: np.ndarray, rects: dict[int, BBox], bg: int) -> list[tuple[int, ...]]:
    colors = tuple(int(color) for color in rects)
    valid = []
    for order in itertools.permutations(colors):
        if np.array_equal(_render_visible(grid.shape, rects, order, bg), grid):
            valid.append(order)
    return valid


def _grey_output(grid: np.ndarray, rects: dict[int, BBox], order: tuple[int, ...]) -> np.ndarray:
    out = np.array(grid, copy=True, dtype=int)
    for lower, upper in zip(order, order[1:]):
        overlap = _intersection(rects[int(lower)], rects[int(upper)])
        if overlap is None:
            continue
        r0, c0, r1, c1 = overlap
        out[r0 : r1 + 1, c0 : c1 + 1] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden rectangle bboxes and layer order, then grey the overlap
    of each directly adjacent layer pair.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    rects = _extract_rectangles(g, bg)
    valid = _valid_orders(g, rects, bg)
    assert valid, "expected at least one layer order consistent with the visible rectangles"

    outputs: dict[bytes, np.ndarray] = {}
    for order in valid:
        out = _grey_output(g, rects, order)
        outputs.setdefault(out.tobytes(), out)

    if len(outputs) == 1:
        return next(iter(outputs.values()))

    # Fallback for unexpected ambiguous inputs.
    return _grey_output(g, rects, min(valid))


def _boxes_extent(boxes: list[BBox]) -> tuple[int, int]:
    max_r = max(box[2] for box in boxes)
    max_c = max(box[3] for box in boxes)
    return max_r + 1, max_c + 1


def _normalize_boxes(boxes: list[BBox]) -> list[BBox]:
    min_r = min(box[0] for box in boxes)
    min_c = min(box[1] for box in boxes)
    return [(r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c) for r0, c0, r1, c1 in boxes]


def _flip_rows(boxes: list[BBox]) -> list[BBox]:
    height, _width = _boxes_extent(boxes)
    return [(height - 1 - r1, c0, height - 1 - r0, c1) for r0, c0, r1, c1 in boxes]


def _flip_cols(boxes: list[BBox]) -> list[BBox]:
    _height, width = _boxes_extent(boxes)
    return [(r0, width - 1 - c1, r1, width - 1 - c0) for r0, c0, r1, c1 in boxes]


def _shift_boxes(boxes: list[BBox], dr: int, dc: int) -> list[BBox]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in boxes]


def _union_area(boxes: list[BBox]) -> int:
    norm = _normalize_boxes(boxes)
    height, width = _boxes_extent(norm)
    mask = np.zeros((height, width), dtype=bool)
    for r0, c0, r1, c1 in norm:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return int(mask.sum())


def _has_non_adjacent_overlap(boxes: list[BBox]) -> bool:
    for i in range(len(boxes)):
        for j in range(i + 2, len(boxes)):
            if _intersection(boxes[i], boxes[j]) is not None:
                return True
    return False


def _all_consecutive_overlap(boxes: list[BBox]) -> bool:
    return all(_intersection(a, b) is not None for a, b in zip(boxes, boxes[1:]))


def _sample_canvas(ext_h: int, ext_w: int, union_area: int) -> tuple[int, int] | None:
    area_floor = union_area * 2 + 12
    options = [
        (h, w)
        for h in range(max(MIN_GRID, ext_h + 2), MAX_GRID + 1)
        for w in range(max(MIN_GRID, ext_w + 2), MAX_GRID + 1)
        if h * w >= area_floor
    ]
    if not options:
        return None
    return random.choice(options)


def _nested_boxes(count: int) -> list[BBox] | None:
    min_side = max(8, 2 * count + 3)
    max_side = min(14, MAX_GRID - 4)
    if min_side > max_side:
        return None

    outer_h = random.randint(min_side, max_side)
    outer_w = random.randint(min_side, max_side)
    boxes = [(0, 0, outer_h - 1, outer_w - 1)]

    for _ in range(count - 1):
        r0, c0, r1, c1 = boxes[-1]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        choices = []
        for top in range(1, min(3, height - 2) + 1):
            for bottom in range(1, min(3, height - top - 1) + 1):
                if height - top - bottom < 3:
                    continue
                for left in range(1, min(3, width - 2) + 1):
                    for right in range(1, min(3, width - left - 1) + 1):
                        if width - left - right < 3:
                            continue
                        choices.append((top, bottom, left, right))
        if not choices:
            return None
        top, bottom, left, right = random.choice(choices)
        boxes.append((r0 + top, c0 + left, r1 - bottom, c1 - right))

    return boxes


def _chain_boxes(count: int) -> list[BBox] | None:
    side_max = {3: 8, 4: 7, 5: 6}[count]
    orientation = random.choice(("horizontal", "vertical"))
    first_h = random.randint(4, side_max)
    first_w = random.randint(4, side_max)
    boxes = [(0, 0, first_h - 1, first_w - 1)]

    for _ in range(count - 1):
        pr0, pc0, pr1, pc1 = boxes[-1]
        prev_h = pr1 - pr0 + 1
        prev_w = pc1 - pc0 + 1
        height = max(4, min(side_max, prev_h + random.randint(-1, 1)))
        width = max(4, min(side_max, prev_w + random.randint(-1, 1)))

        if orientation == "horizontal":
            max_overlap = min(3, prev_w - 1, width - 1)
            if max_overlap < 2:
                return None
            overlap_w = random.randint(2, max_overlap)
            shift_c = prev_w - overlap_w
            deltas = [
                delta
                for delta in range(-(height - 2), prev_h - 1)
                if not (delta <= 0 and delta + height - 1 >= prev_h - 1)
            ]
            if not deltas:
                return None
            delta_r = random.choice(deltas)
            boxes.append((pr0 + delta_r, pc0 + shift_c, pr0 + delta_r + height - 1, pc0 + shift_c + width - 1))
        else:
            max_overlap = min(3, prev_h - 1, height - 1)
            if max_overlap < 2:
                return None
            overlap_h = random.randint(2, max_overlap)
            shift_r = prev_h - overlap_h
            deltas = [
                delta
                for delta in range(-(width - 2), prev_w - 1)
                if not (delta <= 0 and delta + width - 1 >= prev_w - 1)
            ]
            if not deltas:
                return None
            delta_c = random.choice(deltas)
            boxes.append((pr0 + shift_r, pc0 + delta_c, pr0 + shift_r + height - 1, pc0 + delta_c + width - 1))

    boxes = _normalize_boxes(boxes)
    if random.random() < 0.5:
        boxes = _flip_rows(boxes)
    if random.random() < 0.5:
        boxes = _flip_cols(boxes)
    boxes = _normalize_boxes(boxes)

    if _has_non_adjacent_overlap(boxes):
        return None
    return boxes


def _candidate_ok(grid: np.ndarray, rects: dict[int, BBox], order: tuple[int, ...], family: str) -> bool:
    if np.count_nonzero(grid == BG) <= grid.size // 2:
        return False

    boxes = [rects[color] for color in order]
    if not _all_consecutive_overlap(boxes):
        return False

    has_non_adjacent = _has_non_adjacent_overlap(boxes)
    if family == "chain" and has_non_adjacent:
        return False
    if family == "nested" and not has_non_adjacent:
        return False

    for color in order:
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            return False
        if _bbox_from_coords(coords) != rects[color]:
            return False

    extracted = _extract_rectangles(grid, BG)
    valid = _valid_orders(grid, extracted, BG)
    if len(valid) != 1 or valid[0] != order:
        return False

    expected = _grey_output(grid, rects, order)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.count_nonzero(actual == GREY) == 0:
        return False
    return True


def _build_candidate(family: str) -> np.ndarray | None:
    count = random.randint(3, 5)
    colors = tuple(random.sample(INPUT_COLORS, count))
    if family == "nested":
        base_boxes = _nested_boxes(count)
    else:
        base_boxes = _chain_boxes(count)
    if base_boxes is None:
        return None

    base_boxes = _normalize_boxes(base_boxes)
    ext_h, ext_w = _boxes_extent(base_boxes)
    canvas = _sample_canvas(ext_h, ext_w, _union_area(base_boxes))
    if canvas is None:
        return None
    height, width = canvas

    top = random.randint(1, height - ext_h - 1)
    left = random.randint(1, width - ext_w - 1)
    boxes = _shift_boxes(base_boxes, top, left)
    rects = {color: box for color, box in zip(colors, boxes)}
    order = colors
    grid = _render_visible((height, width), rects, order, BG)

    if not _candidate_ok(grid, rects, order, family):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        family = "chain" if random.random() < 0.5 else "nested"
        grid = _build_candidate(family)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rectangle-layering input")
