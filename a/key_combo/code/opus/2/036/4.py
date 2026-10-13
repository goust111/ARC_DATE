from __future__ import annotations

import math
import random

import numpy as np


Color = type(
    "Color",
    (),
    {
        "BLACK": 0,
        "BLUE": 1,
        "RED": 2,
        "GREEN": 3,
        "YELLOW": 4,
        "ORANGE": 7,
    },
)

BBox = tuple[int, int, int, int]  # (r0, c0, r1, c1), inclusive

INPUT_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE)
RANK_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
AREA_TO_DIMS: dict[int, tuple[tuple[int, int], ...]] = {
    30: ((5, 6), (6, 5)),
    25: ((5, 5),),
    24: ((4, 6), (6, 4)),
    20: ((4, 5), (5, 4)),
    18: ((3, 6), (6, 3)),
    16: ((4, 4),),
    15: ((3, 5), (5, 3)),
    12: ((3, 4), (4, 3)),
    10: ((2, 5), (5, 2)),
    9: ((3, 3),),
    8: ((2, 4), (4, 2)),
    6: ((2, 3), (3, 2)),
}

GEN_RETRY_LIMIT = 400


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(set(int(v) for v in np.unique(grid)))
    if exclude is None:
        return colors
    return [color for color in colors if color not in exclude]


def bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def bbox_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def bbox_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def bbox_area(box: BBox) -> int:
    return bbox_h(box) * bbox_w(box)


def shift_bbox(box: BBox, dr: int, dc: int) -> BBox:
    r0, c0, r1, c1 = box
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def bbox_union(boxes: list[BBox]) -> BBox:
    r0 = min(box[0] for box in boxes)
    c0 = min(box[1] for box in boxes)
    r1 = max(box[2] for box in boxes)
    c1 = max(box[3] for box in boxes)
    return int(r0), int(c0), int(r1), int(c1)


def intersect_bbox(a: BBox, b: BBox) -> BBox | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def paint_rect(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each occluded solid rectangle from its visible footprint, then recolor rectangles by restored area rank.

    Input structure:
    - The background is black.
    - There are 3 or 4 solid axis-aligned rectangles in distinct non-black colors.
    - Overlaps may hide parts of a rectangle, but the visible pixels of each color still determine that rectangle's full bounding box.

    Rule:
    - For each non-black color, find the bounding box of all visible pixels of that color.
    - If that box contains only foreground colors, treat the full box as the completed rectangle of that color.
    - Compute every completed rectangle's area and sort the rectangles from largest to smallest.
    - Recolor the largest, second, third, and optional fourth rectangles to blue, red, green, and yellow.
    - Redraw them in that largest-to-smallest order so smaller rectangles remain visible on top inside overlaps.

    Key insight:
    - Final colors depend only on restored rectangle area rank, so the hidden pixels must be recovered before ranking.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rects: list[dict[str, int | BBox]] = []
    for color in colors_present(g, exclude={Color.BLACK}):
        box = bbox(g == color)
        assert box is not None, "every foreground color must have visible pixels"
        patch = crop_bbox(g, box)
        assert np.all(patch != Color.BLACK), "completed rectangle footprints must be fully foreground"
        rects.append({"color": color, "bbox": box, "area": bbox_area(box)})

    assert 3 <= len(rects) <= 4
    areas = [int(rect["area"]) for rect in rects]
    assert len(set(areas)) == len(areas), "restored areas must be unique"

    ranked = sorted(rects, key=lambda rect: int(rect["area"]), reverse=True)
    out = np.zeros_like(g)
    for rank, rect in enumerate(ranked):
        paint_rect(out, rect["bbox"], RANK_COLORS[rank])
    return out


def _choose_area_sequence(n_rects: int) -> list[int]:
    largest = random.choice((20, 24, 25, 30))
    second_choices = [area for area in (18, 16, 15, 12) if area < largest]
    second = random.choice(second_choices)

    third_pool = [area for area in (16, 15, 12, 10, 9) if area < second and area != second]
    third = random.choice(third_pool)

    if n_rects == 3:
        return [largest, second, third]

    fourth_pool = [area for area in (12, 10, 9, 8, 6) if area < third and area not in {second, third}]
    fourth = random.choice(fourth_pool)
    return [largest, second, third, fourth]


def _choose_sizes(n_rects: int) -> list[tuple[int, int]]:
    return [random.choice(AREA_TO_DIMS[area]) for area in _choose_area_sequence(n_rects)]


def _candidate_boxes(target: BBox, size: tuple[int, int]) -> list[BBox]:
    target_area = bbox_area(target)
    h, w = size
    min_overlap = max(2, int(math.ceil(target_area * 0.18)))
    max_overlap = max(min_overlap, int(math.floor(target_area * 0.42)))

    boxes: list[BBox] = []
    for r0 in range(target[0] - h + 1, target[2] + 1):
        for c0 in range(target[1] - w + 1, target[3] + 1):
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            inter = intersect_bbox(box, target)
            if inter is None:
                continue
            if not (min_overlap <= bbox_area(inter) <= max_overlap):
                continue
            if inter == target:
                continue
            if inter[0] == target[0] and bbox_w(inter) == bbox_w(target):
                continue
            if inter[2] == target[2] and bbox_w(inter) == bbox_w(target):
                continue
            if inter[1] == target[1] and bbox_h(inter) == bbox_h(target):
                continue
            if inter[3] == target[3] and bbox_h(inter) == bbox_h(target):
                continue
            boxes.append(box)
    random.shuffle(boxes)
    return boxes


def _target_order(rects: list[dict[str, object]], idx: int) -> list[int]:
    valid = [i for i, rect in enumerate(rects) if bbox_h(rect["bbox"]) >= 3 and bbox_w(rect["bbox"]) >= 3]
    weighted: list[int] = []
    if 0 in valid:
        weighted.extend([0] * (4 if idx == 2 else 3))
    if 1 in valid and idx >= 2:
        weighted.extend([1] * 2)
    if 2 in valid and idx >= 3:
        weighted.append(2)
    if not weighted:
        return []

    random.shuffle(weighted)
    order: list[int] = []
    for target in weighted + valid:
        if target not in order:
            order.append(target)
    return order


def _place_layout(sizes: list[tuple[int, int]]) -> list[dict[str, object]] | None:
    first_h, first_w = sizes[0]
    rects: list[dict[str, object]] = [{"bbox": (0, 0, first_h - 1, first_w - 1), "area": first_h * first_w}]

    for idx, size in enumerate(sizes[1:], start=1):
        placed = False
        for target_index in _target_order(rects, idx):
            candidates = _candidate_boxes(rects[target_index]["bbox"], size)
            for box in candidates:
                trial = rects + [{"bbox": box, "area": size[0] * size[1]}]
                union = bbox_union([rect["bbox"] for rect in trial])
                if bbox_h(union) > 14 or bbox_w(union) > 14:
                    continue
                rects.append({"bbox": box, "area": size[0] * size[1]})
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return rects


def _build_input(shape: tuple[int, int], rects: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rect in rects:
        paint_rect(grid, rect["bbox"], rect["color"])
    return grid


def _expected_output(shape: tuple[int, int], rects: list[dict[str, object]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    ranked = sorted(rects, key=lambda rect: int(rect["area"]), reverse=True)
    for rank, rect in enumerate(ranked):
        paint_rect(out, rect["bbox"], RANK_COLORS[rank])
    return out


def _apply_symmetry(grid: np.ndarray, turns: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(grid, turns)
    if flip_lr:
        out = np.fliplr(out)
    return out.copy()


def _has_required_inversion(grid: np.ndarray, rects: list[dict[str, object]]) -> bool:
    visible = {int(rect["color"]): int(np.count_nonzero(grid == int(rect["color"]))) for rect in rects}
    for i, rect_i in enumerate(rects):
        for rect_j in rects[i + 1 :]:
            area_i = int(rect_i["area"])
            area_j = int(rect_j["area"])
            vis_i = visible[int(rect_i["color"])]
            vis_j = visible[int(rect_j["color"])]
            if area_i > area_j and vis_i < vis_j:
                return True
            if area_j > area_i and vis_j < vis_i:
                return True
    return False


def _valid_input(grid: np.ndarray, rects: list[dict[str, object]]) -> bool:
    coverage = np.zeros(grid.shape, dtype=int)
    overlap_pairs = 0
    for i, rect in enumerate(rects):
        box = rect["bbox"]
        coverage[box[0] : box[2] + 1, box[1] : box[3] + 1] += 1
        for other in rects[i + 1 :]:
            if intersect_bbox(box, other["bbox"]) is not None:
                overlap_pairs += 1

    overlap_pixels = int(np.count_nonzero(coverage > 1))
    if overlap_pairs == 0 or overlap_pixels < 3:
        return False
    if overlap_pixels > int(np.count_nonzero(coverage) * 0.33):
        return False

    occluded = 0
    for rect in rects:
        color = int(rect["color"])
        box = rect["bbox"]
        area = int(rect["area"])
        visible_mask = grid == color
        if int(visible_mask.sum()) < int(math.ceil(area * 0.60)):
            return False
        if bbox(visible_mask) != box:
            return False
        patch = crop_bbox(grid, box)
        if np.any(patch == Color.BLACK):
            return False
        if np.any(patch != color):
            occluded += 1

    return occluded >= 1 and _has_required_inversion(grid, rects)


def _sample_rects() -> list[dict[str, object]] | None:
    n_rects = random.choice((3, 4))
    sizes = _choose_sizes(n_rects)
    rects = _place_layout(sizes)
    if rects is None:
        return None

    colors = random.sample(INPUT_COLORS, n_rects)
    for rect, color in zip(rects, colors):
        rect["color"] = color
    return rects


def _embed_rects(rects: list[dict[str, object]]) -> tuple[tuple[int, int], list[dict[str, object]]] | None:
    union = bbox_union([rect["bbox"] for rect in rects])
    uh = bbox_h(union)
    uw = bbox_w(union)

    top = random.randint(1, 3)
    bottom = random.randint(1, 4)
    left = random.randint(1, 3)
    right = random.randint(1, 4)
    height = max(11, uh + top + bottom)
    width = max(11, uw + left + right)
    if height > 18 or width > 18:
        return None

    shift_r = top - union[0]
    shift_c = left - union[1]
    shifted: list[dict[str, object]] = []
    for rect in rects:
        shifted.append(
            {
                "bbox": shift_bbox(rect["bbox"], shift_r, shift_c),
                "area": rect["area"],
                "color": rect["color"],
            }
        )
    return (height, width), shifted


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    rects = _sample_rects()
    if rects is None:
        return None

    embedded = _embed_rects(rects)
    if embedded is None:
        return None
    shape, shifted = embedded

    inp = _build_input(shape, shifted)
    if not _valid_input(inp, shifted):
        return None

    out = _expected_output(shape, shifted)
    turns = random.randrange(4)
    flip_lr = random.random() < 0.5
    inp = _apply_symmetry(inp, turns, flip_lr)
    out = _apply_symmetry(out, turns, flip_lr)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        case = _generate_case()
        if case is None:
            continue
        inp, expected = case
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid occluded-rectangle ranking input")
