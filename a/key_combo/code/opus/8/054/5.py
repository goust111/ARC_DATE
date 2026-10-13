from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GREY = 5
RECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 500


BBox = tuple[int, int, int, int]


def _bbox_from_mask(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _overlap_box(a: BBox, b: BBox) -> BBox | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _overlap_area(a: BBox, b: BBox) -> int:
    box = _overlap_box(a, b)
    if box is None:
        return 0
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _recover_rectangles(grid: np.ndarray) -> list[dict]:
    rects = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        box = _bbox_from_mask(grid == color)
        assert box is not None
        rects.append({"color": color, "bbox": box})
    assert 4 <= len(rects) <= 6
    return rects


def _build_constraints(grid: np.ndarray, rects: list[dict]) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    by_color = {rect["color"]: rect for rect in rects}
    above: dict[int, set[int]] = {color: set() for color in by_color}
    below: dict[int, set[int]] = {color: set() for color in by_color}

    colors = [rect["color"] for rect in rects]
    for i, color_a in enumerate(colors):
        box_a = by_color[color_a]["bbox"]
        for color_b in colors[i + 1 :]:
            box_b = by_color[color_b]["bbox"]
            overlap = _overlap_box(box_a, box_b)
            if overlap is None:
                continue
            r0, c0, r1, c1 = overlap
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            a_visible = bool(np.any(patch == color_a))
            b_visible = bool(np.any(patch == color_b))
            assert not (a_visible and b_visible), "pairwise overlap cannot expose both colors"
            if a_visible:
                below[color_a].add(color_b)
                above[color_b].add(color_a)
            elif b_visible:
                below[color_b].add(color_a)
                above[color_a].add(color_b)

    return above, below


def _compute_depths(rects: list[dict], above: dict[int, set[int]], below: dict[int, set[int]]) -> tuple[dict[int, int], dict[int, set[int]]]:
    colors = [rect["color"] for rect in rects]
    indeg = {color: len(below[color]) for color in colors}
    q = deque(sorted(color for color in colors if indeg[color] == 0))
    topo = []
    while q:
        color = q.popleft()
        topo.append(color)
        for nxt in sorted(above[color]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    assert len(topo) == len(colors), "layer constraints must form a DAG"

    depth = {color: 1 for color in colors}
    for color in topo:
        for nxt in above[color]:
            depth[nxt] = max(depth[nxt], depth[color] + 1)

    reach = {color: set() for color in colors}
    for color in reversed(topo):
        all_up = set(above[color])
        for nxt in above[color]:
            all_up |= reach[nxt]
        reach[color] = all_up

    return depth, reach


def _pair_gap(left_box: BBox, right_box: BBox) -> BBox | None:
    if left_box[3] >= right_box[1]:
        return None
    row0 = max(left_box[0], right_box[0])
    row1 = min(left_box[2], right_box[2])
    if row0 > row1:
        return None
    gap = (row0, left_box[3] + 1, row1, right_box[1] - 1)
    if gap[1] > gap[3]:
        return None
    return gap


def _analyze(grid: np.ndarray) -> dict:
    rects = _recover_rectangles(grid)
    rect_by_color = {rect["color"]: rect for rect in rects}
    above, below = _build_constraints(grid, rects)
    depth, reach = _compute_depths(rects, above, below)

    fill_actions = []
    blocked_actions = []
    colors = [rect["color"] for rect in rects]

    for i, color_a in enumerate(colors):
        box_a = rect_by_color[color_a]["bbox"]
        for color_b in colors[i + 1 :]:
            box_b = rect_by_color[color_b]["bbox"]
            if depth[color_a] != depth[color_b]:
                continue
            if color_b in reach[color_a] or color_a in reach[color_b]:
                continue

            if box_a[3] < box_b[1]:
                left_color, right_color = color_a, color_b
                left_box, right_box = box_a, box_b
            elif box_b[3] < box_a[1]:
                left_color, right_color = color_b, color_a
                left_box, right_box = box_b, box_a
            else:
                continue

            gap = _pair_gap(left_box, right_box)
            if gap is None:
                continue

            blockers = []
            for other in rects:
                other_color = other["color"]
                if other_color in (left_color, right_color):
                    continue
                if depth[other_color] > depth[left_color] and _overlap_area(other["bbox"], gap) > 0:
                    blockers.append(other_color)

            if blockers:
                blocked_actions.append(
                    {
                        "pair": (left_color, right_color),
                        "depth": depth[left_color],
                        "gap": gap,
                        "blockers": tuple(sorted(blockers)),
                    }
                )
                continue

            next_depth = depth[left_color] + 1
            candidates = [rect for rect in rects if depth[rect["color"]] == next_depth]
            if not candidates:
                fill_color = GREY
            else:
                scored = sorted(
                    ((-_overlap_area(rect["bbox"], gap), rect["bbox"], rect["color"]) for rect in candidates)
                )
                assert len(scored) == 1 or scored[0][0] < scored[1][0], "next-layer overlap tie is ambiguous"
                fill_color = scored[0][2]

            fill_actions.append(
                {
                    "pair": (left_color, right_color),
                    "depth": depth[left_color],
                    "gap": gap,
                    "fill_color": fill_color,
                }
            )

    fill_actions.sort(key=lambda item: (item["gap"][0], item["gap"][1], item["gap"][2], item["gap"][3]))
    blocked_actions.sort(key=lambda item: (item["gap"][0], item["gap"][1], item["gap"][2], item["gap"][3]))
    return {
        "rects": rects,
        "rect_by_color": rect_by_color,
        "above": above,
        "below": below,
        "depth": depth,
        "reach": reach,
        "fills": fill_actions,
        "blocked": blocked_actions,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer rectangular occlusion layers and fill eligible same-depth horizontal gaps with the adjacent upper-layer color.

    Input structure
    - The grid has black background and 4 to 6 uniquely colored solid rectangles, some partially hidden by overlaps.
    - Each color corresponds to one rectangle; the visible pixels of that color may be fragmented by higher rectangles, but their full extent is its visible-color bounding box.
    - Some horizontally separated rectangles share a depth level, share rows, and have a black gap between their bounding boxes.

    Rule
    - Recover each rectangle as the bounding box of its visible color.
    - For every overlapping rectangle pair, use the visible color inside their overlap to infer which rectangle is above the other, then compute layer depths from those constraints.
    - For every same-depth rectangle pair whose column spans are disjoint and row spans overlap, inspect the horizontal gap over the shared rows.
    - If no higher-depth rectangle's recovered bounding box intersects that gap, fill only the black cells in the gap with the color of the unique best depth+1 rectangle, chosen by maximum bbox-overlap area with the gap; if there is no depth+1 rectangle, fill with grey.
    - Process eligible gaps in top-left reading order and keep all existing non-black pixels unchanged.

    Key insight
    - The visible colors in rectangle-overlap areas determine a global depth partial order, and that inferred layer structure is then reused to decide both which gaps belong to same-depth pairs and which color should be used to fill them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    analysis = _analyze(g)
    out = g.copy()
    for action in analysis["fills"]:
        r0, c0, r1, c1 = action["gap"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == BG] = action["fill_color"]
        out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


def _place_rect(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _boxes_within(shape: tuple[int, int], boxes: dict[str, BBox]) -> bool:
    h, w = shape
    for r0, c0, r1, c1 in boxes.values():
        if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
            return False
    return True


def _render_scene(shape: tuple[int, int], layers: list[list[tuple[str, int, BBox]]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for layer in layers:
        for _, color, box in layer:
            _place_rect(grid, box, color)
    return grid


def _new_pixels(input_grid: np.ndarray, output_grid: np.ndarray, region: BBox) -> int:
    r0, c0, r1, c1 = region
    before = input_grid[r0 : r1 + 1, c0 : c1 + 1]
    after = output_grid[r0 : r1 + 1, c0 : c1 + 1]
    return int(np.sum((before == BG) & (after != BG)))


def _ordered_pair_from_boxes(color_map: dict[str, int], boxes: dict[str, BBox], left_name: str, right_name: str) -> tuple[int, int]:
    if boxes[left_name][1] <= boxes[right_name][1]:
        return color_map[left_name], color_map[right_name]
    return color_map[right_name], color_map[left_name]


def _build_template(include_top_pair: bool) -> tuple[np.ndarray, dict]:
    h = random.randint(16, 22)
    w = random.randint(20, 27)

    bottom_h = random.randint(4, 5)
    mid_h = random.randint(5, 6)
    top_h = random.randint(4, 5)

    overlap_bottom_mid = random.randint(2, 3)
    overlap_mid_top = random.randint(2, 3)

    top_r = random.randint(1, 3)
    mid_r = top_r + top_h - overlap_mid_top
    bottom_r = mid_r + mid_h - overlap_bottom_mid
    if bottom_r + bottom_h + 1 >= h:
        shift = bottom_r + bottom_h + 2 - h
        top_r -= shift
        mid_r -= shift
        bottom_r -= shift
    if top_r < 1 or bottom_r + bottom_h >= h:
        raise AssertionError("row layout failed")

    w_a = random.randint(4, 5)
    w_b = random.randint(4, 5)
    w_c = random.randint(4, 5)
    gap_ab = random.randint(2, 3)
    gap_bc = random.randint(1, 3)

    overlap_db = random.randint(2, 3)
    overlap_dc = random.randint(1, 2)
    d_w = overlap_db + gap_bc + overlap_dc
    assert 4 <= d_w <= 7

    overlap_de = random.randint(1, 2)
    w_e = random.randint(4, 5)

    if include_top_pair:
        overlap_df = random.randint(1, 2)
        w_f = random.randint(4, 5)
        left_extra_f = w_f - overlap_df
        right_extra_e = w_e - overlap_de
        if overlap_df + overlap_de > d_w - 1:
            raise AssertionError("top pair must leave a visible middle gap over D")
    else:
        overlap_df = 0
        w_f = 0
        left_extra_f = 0
        right_extra_e = w_e - overlap_de

    min_total_w = w_a + gap_ab + w_b + gap_bc + w_c
    if min_total_w + 2 > w:
        raise AssertionError("canvas too narrow")

    max_ca = w - min_total_w - 1
    c_a = random.randint(1, max(1, max_ca))
    c_b = c_a + w_a + gap_ab
    c_c = c_b + w_b + gap_bc

    c_d = c_b + w_b - overlap_db
    d_end = c_c + overlap_dc - 1
    assert d_end - c_d + 1 == d_w
    assert c_d >= c_b

    if include_top_pair:
        c_f = c_d - left_extra_f
        f_end = c_d + overlap_df - 1
        c_e = d_end - overlap_de + 1
        e_end = c_e + w_e - 1
        if c_f <= c_a + w_a + gap_ab - 1:
            raise AssertionError("top-left rectangle intrudes into fillable lower gap")
    else:
        c_e = d_end - overlap_de + 1
        e_end = c_e + w_e - 1
        c_f = f_end = -1

    boxes = {
        "A": (bottom_r, c_a, bottom_r + bottom_h - 1, c_a + w_a - 1),
        "B": (bottom_r, c_b, bottom_r + bottom_h - 1, c_b + w_b - 1),
        "C": (bottom_r, c_c, bottom_r + bottom_h - 1, c_c + w_c - 1),
        "D": (mid_r, c_d, mid_r + mid_h - 1, d_end),
        "E": (top_r, c_e, top_r + top_h - 1, e_end),
    }
    if include_top_pair:
        boxes["F"] = (top_r, c_f, top_r + top_h - 1, f_end)

    if not _boxes_within((h, w), boxes):
        raise AssertionError("box out of bounds")

    if _overlap_area(boxes["A"], boxes["D"]) != 0:
        raise AssertionError("middle rectangle must not overlap A")
    if _overlap_area(boxes["A"], boxes["B"]) != 0:
        raise AssertionError("bottom rectangles must stay disjoint")
    if _overlap_area(boxes["B"], boxes["C"]) != 0:
        raise AssertionError("bottom rectangles must stay disjoint")
    if include_top_pair and _overlap_area(boxes["E"], boxes["F"]) != 0:
        raise AssertionError("top pair must stay disjoint")

    colors = random.sample(RECT_COLORS, len(boxes))
    color_map = {name: color for name, color in zip(sorted(boxes), colors)}

    layer0 = [("A", color_map["A"], boxes["A"]), ("B", color_map["B"], boxes["B"]), ("C", color_map["C"], boxes["C"])]
    layer1 = [("D", color_map["D"], boxes["D"])]
    layer2 = [("E", color_map["E"], boxes["E"])]
    if include_top_pair:
        layer2.append(("F", color_map["F"], boxes["F"]))

    grid = _render_scene((h, w), [layer0, layer1, layer2])
    meta = {"boxes": boxes, "color_map": color_map, "include_top_pair": include_top_pair}
    return grid, meta


def _validate_generated_case(grid: np.ndarray, meta: dict) -> bool:
    try:
        analysis = _analyze(grid)
        out = main(grid)
    except Exception:
        return False
    boxes = meta["boxes"]
    color_map = meta["color_map"]
    include_top_pair = meta["include_top_pair"]

    expected_boxes = {color_map[name]: box for name, box in boxes.items()}
    actual_boxes = {rect["color"]: rect["bbox"] for rect in analysis["rects"]}
    if expected_boxes != actual_boxes:
        return False

    depth = analysis["depth"]
    if depth[color_map["A"]] != 1 or depth[color_map["B"]] != 1 or depth[color_map["C"]] != 1:
        return False
    if depth[color_map["D"]] != 2:
        return False
    if depth[color_map["E"]] != 3:
        return False
    if include_top_pair and depth[color_map["F"]] != 3:
        return False

    fills = {tuple(action["pair"]): action for action in analysis["fills"]}
    blocked = {tuple(action["pair"]): action for action in analysis["blocked"]}

    ab_pair = _ordered_pair_from_boxes(color_map, boxes, "A", "B")
    bc_pair = _ordered_pair_from_boxes(color_map, boxes, "B", "C")

    if ab_pair not in fills:
        return False
    if fills[ab_pair]["fill_color"] != color_map["D"]:
        return False
    if _new_pixels(grid, out, fills[ab_pair]["gap"]) <= 0:
        return False

    if bc_pair not in blocked:
        return False
    gap_bc = blocked[bc_pair]["gap"]
    r0, c0, r1, c1 = gap_bc
    if np.any(out[r0 : r1 + 1, c0 : c1 + 1] != grid[r0 : r1 + 1, c0 : c1 + 1]):
        return False

    if include_top_pair:
        ef_pair = _ordered_pair_from_boxes(color_map, boxes, "E", "F")
        if ef_pair not in fills:
            return False
        if fills[ef_pair]["fill_color"] != GREY:
            return False
        if _new_pixels(grid, out, fills[ef_pair]["gap"]) <= 0:
            return False
        if not np.any(out == GREY):
            return False
    else:
        if np.any(out == GREY):
            return False

    if len(analysis["blocked"]) < 1 or len(analysis["fills"]) < 1:
        return False

    # Reject overly simple outputs where all newly painted cells share one region/color.
    new_mask = (grid == BG) & (out != BG)
    new_colors = {int(out[r, c]) for r, c in np.argwhere(new_mask)}
    if include_top_pair:
        if len(new_colors) < 2:
            return False
    else:
        if len(new_colors) < 1:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        include_top_pair = random.random() < 0.7
        try:
            grid, meta = _build_template(include_top_pair)
        except AssertionError:
            continue
        if _validate_generated_case(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid rectangle-layer puzzle input")
