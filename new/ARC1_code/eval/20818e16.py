from __future__ import annotations

import numpy as np

from arc_utils import bbox, bbox_h, bbox_w, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack the colored rectangles by size after aligning their upper-left corners.

    Input structure
    - The dominant color is background.
    - Each foreground color is the visible part of one filled rectangle.
    - A color's full rectangle is recovered from the bounding box of its visible pixels.

    Rule
    - Build one rectangle for each foreground color using that color's bounding-box height and width.
    - Use the largest rectangle's height and width as the output shape.
    - Paint all rectangles at the output upper-left corner from largest area to smallest area.

    Key insight
    - The input positions only reveal each rectangle's size; the output re-stacks the same rectangles at a common origin.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    rectangles = []
    for color in colors_present(g, exclude={bg}):
        box = bbox(g == color)
        assert box is not None, "expected every foreground color to be present"
        height = bbox_h(box)
        width = bbox_w(box)
        area = height * width
        rectangles.append((int(color), height, width, area))

    assert rectangles, "expected at least one foreground rectangle"
    areas = [area for _color, _height, _width, area in rectangles]
    assert len(set(areas)) == len(areas), "rectangle areas must give a unique layer order"

    rectangles.sort(key=lambda rect: rect[3], reverse=True)
    _base_color, out_height, out_width, _base_area = rectangles[0]
    assert out_height == max(height for _color, height, _width, _area in rectangles)
    assert out_width == max(width for _color, _height, width, _area in rectangles)

    out = np.full((out_height, out_width), bg, dtype=int)
    for color, height, width, _area in rectangles:
        out[:height, :width] = color
    return out

# --------------- generator (o5x) ---------------

import itertools
from arc_utils import paste


# Canvas/palette bounds; rng.integers stop values are exclusive.
GRID_H_MIN, GRID_H_STOP = 14, 17
GRID_W_MIN, GRID_W_STOP = 15, 17
ARC_COLOR_COUNT = 10

# Rectangle count/size bounds.
RECT_COUNT_MIN, RECT_COUNT_STOP = 3, 5
LARGEST_H_MIN, LARGEST_H_STOP = 5, 9
LARGEST_W_MIN, LARGEST_W_STOP = 5, 11
SMALL_RECT_H_MIN = 1
SMALL_RECT_W_MIN = 2
AREA_BUDGET_FRACTION = 0.44

# Layout constraints.
THREE_RECT_ISOLATED_COUNT = 1
FOUR_RECT_ISOLATED_MIN, FOUR_RECT_ISOLATED_STOP = 1, 3

# Retry budget.
MAX_GENERATION_ATTEMPTS = 300


def _gen_bbox_at(r, c, h, w):
    return (int(r), int(c), int(r + h - 1), int(c + w - 1))


def _gen_bboxes_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _gen_extends_outside(inner, outer):
    return inner[0] < outer[0] or inner[1] < outer[1] or inner[2] > outer[2] or inner[3] > outer[3]


def _gen_choice(rng, items):
    return items[int(rng.integers(0, len(items)))]


def _gen_base_visible_full(base_rect, occluder_bboxes):
    r0, c0, r1, c1 = base_rect["bbox"]
    h, w = base_rect["h"], base_rect["w"]
    visible = np.ones((h, w), dtype=bool)
    for b in occluder_bboxes:
        rr0 = max(r0, b[0])
        cc0 = max(c0, b[1])
        rr1 = min(r1, b[2])
        cc1 = min(c1, b[3])
        if rr0 <= rr1 and cc0 <= cc1:
            visible[rr0 - r0 : rr1 - r0 + 1, cc0 - c0 : cc1 - c0 + 1] = False
    visible_box = bbox(visible)
    if visible_box is None:
        return False
    return visible_box == (0, 0, h - 1, w - 1)


def _gen_pick_rectangles(rng, grid_h, grid_w, n_rects, fg_colors):
    largest_h = int(rng.integers(LARGEST_H_MIN, LARGEST_H_STOP))
    largest_w = int(rng.integers(LARGEST_W_MIN, LARGEST_W_STOP))
    largest_area = largest_h * largest_w
    area_budget = int(AREA_BUDGET_FRACTION * grid_h * grid_w)

    candidates = [
        (h, w, h * w)
        for h in range(SMALL_RECT_H_MIN, largest_h)
        for w in range(SMALL_RECT_W_MIN, largest_w)
    ]
    valid_groups = []
    for group in itertools.combinations(candidates, n_rects - 1):
        areas = [area for _h, _w, area in group]
        if len(set(areas)) != len(areas):
            continue
        if largest_area + sum(areas) <= area_budget:
            valid_groups.append(group)
    if not valid_groups:
        return None

    smaller_dims = list(_gen_choice(rng, valid_groups))
    rng.shuffle(smaller_dims)
    dims = [(largest_h, largest_w, largest_area), *smaller_dims]
    rects = [
        {"color": int(color), "h": int(h), "w": int(w), "area": int(area)}
        for color, (h, w, area) in zip(fg_colors, dims)
    ]
    rects.sort(key=lambda rect: rect["area"], reverse=True)
    return rects


def _gen_partition_rectangles(rng, n_rects):
    """Choose which smaller rectangles cluster with the largest one."""
    isolated_count = THREE_RECT_ISOLATED_COUNT
    if n_rects != 3:
        isolated_count = int(rng.integers(FOUR_RECT_ISOLATED_MIN, FOUR_RECT_ISOLATED_STOP))
    cluster_count = n_rects - isolated_count

    smaller_indices = list(range(1, n_rects))
    rng.shuffle(smaller_indices)
    split = cluster_count - 1
    return [0] + smaller_indices[:split], smaller_indices[split:]


def _gen_overlap_positions(grid_h, grid_w, rect, base_bbox):
    out = []
    h, w = rect["h"], rect["w"]
    for r in range(0, grid_h - h + 1):
        for c in range(0, grid_w - w + 1):
            b = _gen_bbox_at(r, c, h, w)
            if _gen_bboxes_overlap(b, base_bbox) and _gen_extends_outside(b, base_bbox):
                out.append((r, c, b))
    return out


def _gen_nonoverlap_positions(grid_h, grid_w, rect, blocked_bboxes):
    out = []
    h, w = rect["h"], rect["w"]
    for r in range(0, grid_h - h + 1):
        for c in range(0, grid_w - w + 1):
            b = _gen_bbox_at(r, c, h, w)
            if all(not _gen_bboxes_overlap(b, other) for other in blocked_bboxes):
                out.append((r, c, b))
    return out


def _gen_place_cluster(rng, grid_h, grid_w, rects, cluster_indices):
    """Place clustered rectangles after the base has been positioned."""
    base = rects[0]
    if len(cluster_indices) == 2:
        idx = cluster_indices[1]
        candidates = _gen_overlap_positions(grid_h, grid_w, rects[idx], base["bbox"])
        candidates = [
            item for item in candidates
            if _gen_base_visible_full(base, [item[2]])
        ]
        if not candidates:
            return False
        r, c, b = _gen_choice(rng, candidates)
        rects[idx]["pos"] = (r, c)
        rects[idx]["bbox"] = b
        return True

    if len(cluster_indices) == 3:
        idx_a, idx_b = cluster_indices[1], cluster_indices[2]
        cand_a = _gen_overlap_positions(grid_h, grid_w, rects[idx_a], base["bbox"])
        cand_b = _gen_overlap_positions(grid_h, grid_w, rects[idx_b], base["bbox"])
        pairs = []
        for a in cand_a:
            for b in cand_b:
                if _gen_bboxes_overlap(a[2], b[2]):
                    continue
                if _gen_base_visible_full(base, [a[2], b[2]]):
                    pairs.append((a, b))
        if not pairs:
            return False
        (ra, ca, ba), (rb, cb, bb) = _gen_choice(rng, pairs)
        rects[idx_a]["pos"] = (ra, ca)
        rects[idx_a]["bbox"] = ba
        rects[idx_b]["pos"] = (rb, cb)
        rects[idx_b]["bbox"] = bb
        return True

    return True


def _gen_place_isolated(rng, grid_h, grid_w, rects, isolated_indices):
    blocked = [rect["bbox"] for rect in rects if "bbox" in rect]
    for idx in isolated_indices:
        candidates = _gen_nonoverlap_positions(grid_h, grid_w, rects[idx], blocked)
        if not candidates:
            return False
        r, c, b = _gen_choice(rng, candidates)
        rects[idx]["pos"] = (r, c)
        rects[idx]["bbox"] = b
        blocked.append(b)
    return True


def _gen_place_rectangles(rng, grid_h, grid_w, rects):
    cluster_indices, isolated_indices = _gen_partition_rectangles(rng, len(rects))
    base = rects[0]
    base_r = int(rng.integers(0, grid_h - base["h"] + 1))
    base_c = int(rng.integers(0, grid_w - base["w"] + 1))
    base["bbox"] = _gen_bbox_at(base_r, base_c, base["h"], base["w"])
    base["pos"] = (base_r, base_c)
    if not _gen_place_cluster(rng, grid_h, grid_w, rects, cluster_indices):
        return None
    if not _gen_place_isolated(rng, grid_h, grid_w, rects, isolated_indices):
        return None
    if any("pos" not in rect for rect in rects):
        return None
    return rects


def _gen_paint_rectangles(grid, rects, *, at_origin):
    for rect in sorted(rects, key=lambda item: item["area"], reverse=True):
        patch = np.full((rect["h"], rect["w"]), rect["color"], dtype=int)
        pos = (0, 0) if at_origin else rect["pos"]
        paste(grid, patch, *pos)
    return grid


def _gen_expected_output(bg, rects):
    largest = max(rects, key=lambda rect: rect["area"])
    out = np.full((largest["h"], largest["w"]), int(bg), dtype=int)
    _gen_paint_rectangles(out, rects, at_origin=True)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    grid_h = int(rng.integers(GRID_H_MIN, GRID_H_STOP))
    grid_w = int(rng.integers(GRID_W_MIN, GRID_W_STOP))
    n_rects = int(rng.integers(RECT_COUNT_MIN, RECT_COUNT_STOP))

    colors = list(range(ARC_COLOR_COUNT))
    rng.shuffle(colors)
    bg = int(colors[0])
    fg_colors = [int(c) for c in colors[1 : n_rects + 1]]

    rects = _gen_pick_rectangles(rng, grid_h, grid_w, n_rects, fg_colors)
    if rects is None:
        return None
    rects = _gen_place_rectangles(rng, grid_h, grid_w, rects)
    if rects is None:
        return None

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    _gen_paint_rectangles(inp, rects, at_origin=False)

    expected = _gen_expected_output(bg, rects)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 20818e16 input")
