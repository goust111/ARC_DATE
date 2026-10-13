from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BAR = 5
CONTROL = 9
PALETTE = (1, 2, 3, 4, 7, 8)
GEN_ATTEMPTS = 1200
MIN_GAP = 3


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bar_cells(axis: str, center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    if axis == "horizontal":
        return [(r, c - 1), (r, c), (r, c + 1)]
    return [(r - 1, c), (r, c), (r + 1, c)]


def _find_marker(g: np.ndarray) -> tuple[str, tuple[int, int], tuple[int, int], list[tuple[int, int]], tuple[int, int]]:
    bar_coords = np.argwhere(g == BAR)
    control_coords = np.argwhere(g == CONTROL)
    assert bar_coords.shape == (3, 2), "expected a unique three-pixel bar"
    assert control_coords.shape == (1, 2), "expected a unique control pixel"

    rows = sorted(set(int(v) for v in bar_coords[:, 0]))
    cols = sorted(set(int(v) for v in bar_coords[:, 1]))
    if len(rows) == 1 and len(cols) == 3 and cols == list(range(cols[0], cols[0] + 3)):
        axis = "horizontal"
        center = (rows[0], cols[1])
    elif len(cols) == 1 and len(rows) == 3 and rows == list(range(rows[0], rows[0] + 3)):
        axis = "vertical"
        center = (rows[1], cols[0])
    else:
        raise AssertionError("bar must be a straight contiguous length-3 segment")

    control = tuple(int(v) for v in control_coords[0])
    assert abs(control[0] - center[0]) + abs(control[1] - center[1]) == 1, "control must touch bar center"
    if axis == "horizontal":
        assert control[1] == center[1], "horizontal bar must have vertical control"
    else:
        assert control[0] == center[0], "vertical bar must have horizontal control"

    step = (center[0] - control[0], center[1] - control[1])
    return axis, center, control, _bar_cells(axis, center), step


def _ray_cells(center: tuple[int, int], step: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r = center[0] + step[0]
    c = center[1] + step[1]
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((int(r), int(c)))
        r += step[0]
        c += step[1]
    return cells


def _extract_objects(g: np.ndarray) -> list[dict]:
    mask = (g != BG) & (g != BAR) & (g != CONTROL)
    comps = _components(mask)
    objects: list[dict] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        side = int(r1 - r0 + 1)
        assert side == int(c1 - c0 + 1), "object bbox must be square"
        assert side in (5, 7), "outer square side must be 5 or 7"
        crop = g[r0 : r1 + 1, c0 : c1 + 1]
        border_mask = np.zeros_like(crop, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        border_vals = {int(v) for v in np.unique(crop[border_mask])}
        inner_vals = {int(v) for v in np.unique(crop[~border_mask])}
        assert len(border_vals) == 1 and len(inner_vals) == 1, "expected single-color border and inner square"
        border_color = next(iter(border_vals))
        inner_color = next(iter(inner_vals))
        assert border_color not in (BG, BAR, CONTROL)
        assert inner_color not in (BG, BAR, CONTROL)
        assert border_color != inner_color
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "coords": coords,
                "coords_set": {tuple(map(int, rc)) for rc in coords},
                "side": side,
                "inner_side": side - 2,
                "inner_color": inner_color,
                "border_color": border_color,
                "center": ((r0 + r1) // 2, (c0 + c1) // 2),
            }
        )
    assert 3 <= len(objects) <= 4, "expected three or four concentric-square objects"
    return objects


def _paint_rect(out: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    h, w = out.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(h - 1, int(r1))
    cc1 = min(w - 1, int(c1))
    if rr0 <= rr1 and cc0 <= cc1:
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(color)


def _growth_rects(bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox
    return [
        (r0 - 2, c0, r0 - 1, c1),
        (r1 + 1, c0, r1 + 2, c1),
        (r0, c0 - 2, r1, c0 - 1),
        (r0, c1 + 1, r1, c1 + 2),
    ]


def _draw_swapped_square(out: np.ndarray, obj: dict) -> None:
    r0, c0, r1, c1 = obj["bbox"]
    out[r0 : r1 + 1, c0 : c1 + 1] = int(obj["inner_color"])
    out[r0 + 1 : r1, c0 + 1 : c1] = int(obj["border_color"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap every concentric square's colors, then grow only the ray-hit ones using their original border colors.

    Input structure:
    - A black background contains one grey three-pixel marker bar and one maroon control pixel touching the bar center.
    - Three or four separated concentric square objects occupy the remaining foreground.
    - Each object is a solid 3x3 or 5x5 inner square wrapped by a one-pixel border of a different color.

    Rule:
    - Cast a straight ray from the marker-bar center in the direction away from the control pixel and mark every concentric square touched by that ray.
    - For every concentric square, remember its original border color, then swap the inner-square color with the border color.
    - After all swaps, extend each hit square by four outward rectangles of length two and width equal to the square's outer side, colored with that remembered original border color.
    - Remove the marker bar and control pixel; if growth rectangles overlap, later ones overwrite earlier ones in top-left-to-bottom-right center order.

    Key insight:
    - Growth depends on which object the ray selects, but the growth color depends on the border color from before the swap, so the solver must track original color identity rather than current location.
    """
    g = np.asarray(grid, dtype=int)
    _, center, _control, _bar_cells, step = _find_marker(g)
    objects = _extract_objects(g)
    ray = set(_ray_cells(center, step, g.shape))

    out = g.copy()
    out[(out == BAR) | (out == CONTROL)] = BG
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
    for obj in objects:
        _draw_swapped_square(out, obj)

    hit_objects = [obj for obj in objects if obj["coords_set"] & ray]
    for obj in sorted(hit_objects, key=lambda d: (d["center"][0], d["center"][1], d["bbox"][0], d["bbox"][1])):
        for rect in _growth_rects(obj["bbox"]):
            _paint_rect(out, rect, obj["border_color"])
    return out


def _gap_between_boxes(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    col_gap = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return max(row_gap, col_gap)


def _rects_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _box_covers_marker_cells(bbox: tuple[int, int, int, int], cells: set[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = bbox
    return any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in cells)


def _choose_marker(h: int, w: int) -> tuple[str, tuple[int, int], tuple[int, int], list[tuple[int, int]], tuple[int, int]]:
    candidates = []
    for step in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        axis = "horizontal" if step[0] != 0 else "vertical"
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                center = (r, c)
                bar = _bar_cells(axis, center)
                control = (r - step[0], c - step[1])
                if not (0 <= control[0] < h and 0 <= control[1] < w):
                    continue
                if not all(0 <= rr < h and 0 <= cc < w for rr, cc in bar):
                    continue
                ray = _ray_cells(center, step, (h, w))
                if len(ray) >= 6:
                    candidates.append((axis, center, control, bar, step))
    return random.choice(candidates)


def _covers_line(bbox: tuple[int, int, int, int], center: tuple[int, int], step: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    if step[0] != 0:
        return c0 <= center[1] <= c1
    return r0 <= center[0] <= r1


def _on_ray_side(bbox: tuple[int, int, int, int], center: tuple[int, int], step: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    if step == (-1, 0):
        return r1 < center[0]
    if step == (1, 0):
        return r0 > center[0]
    if step == (0, -1):
        return c1 < center[1]
    return c0 > center[1]


def _on_control_side(bbox: tuple[int, int, int, int], center: tuple[int, int], step: tuple[int, int]) -> bool:
    return _on_ray_side(bbox, center, (-step[0], -step[1]))


def _valid_bbox(
    bbox: tuple[int, int, int, int],
    *,
    hit: bool,
    placed: list[dict],
    marker_cells: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
        return False
    if hit and (r0 < 2 or c0 < 2 or r1 >= h - 2 or c1 >= w - 2):
        return False
    if _box_covers_marker_cells(bbox, marker_cells):
        return False

    new_growths = _growth_rects(bbox) if hit else []
    for other in placed:
        if _gap_between_boxes(bbox, other["bbox"]) < MIN_GAP:
            return False
        if any(_rects_intersect(bbox, rect) for rect in other.get("growth_rects", [])):
            return False
        if hit and any(_rects_intersect(other["bbox"], rect) for rect in new_growths):
            return False
    return True


def _pair_layout_ok(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    step: tuple[int, int],
) -> bool:
    if _gap_between_boxes(a, b) < MIN_GAP:
        return False
    if step[0] != 0:
        col_overlap = min(a[3], b[3]) >= max(a[1], b[1])
        row_gap = max(0, b[0] - a[2] - 1, a[0] - b[2] - 1)
        return col_overlap and row_gap in (3, 4)
    row_overlap = min(a[2], b[2]) >= max(a[0], b[0])
    col_gap = max(0, b[1] - a[3] - 1, a[1] - b[3] - 1)
    return row_overlap and col_gap in (3, 4)


def _candidate_bboxes(
    side: int,
    mode: str,
    center: tuple[int, int],
    step: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    h, w = shape
    boxes: list[tuple[int, int, int, int]] = []
    for r0 in range(1, h - side):
        for c0 in range(1, w - side):
            bbox = (r0, c0, r0 + side - 1, c0 + side - 1)
            covers = _covers_line(bbox, center, step)
            ray_side = _on_ray_side(bbox, center, step)
            control_side = _on_control_side(bbox, center, step)
            if mode == "hit" and covers and ray_side:
                boxes.append(bbox)
            elif mode == "opposite_aligned" and covers and control_side:
                boxes.append(bbox)
            elif mode == "same_side_offline" and (not covers) and ray_side:
                boxes.append(bbox)
            elif mode == "opposite_offline" and (not covers) and control_side:
                boxes.append(bbox)
    random.shuffle(boxes)
    return boxes


def _choose_pairs(count: int) -> list[tuple[int, int]]:
    if count == 3:
        colors = list(PALETTE)
        random.shuffle(colors)
        pairs = [(colors[0], colors[1]), (colors[2], colors[3]), (colors[4], colors[5])]
        random.shuffle(pairs)
        return pairs

    pair_pool = [(a, b) for i, a in enumerate(PALETTE) for b in PALETTE[i + 1 :]]
    while True:
        pairs = random.sample(pair_pool, 4)
        used = {}
        for a, b in pairs:
            used[a] = used.get(a, 0) + 1
            used[b] = used.get(b, 0) + 1
        if len(used) >= 5 and max(used.values()) <= 2:
            random.shuffle(pairs)
            return pairs


def _choose_object_specs(count: int) -> list[dict]:
    inner_sides = [3, 5]
    while len(inner_sides) < count:
        inner_sides.append(random.choice([3, 5]))
    random.shuffle(inner_sides)
    pairs = _choose_pairs(count)
    specs = []
    for inner_side, pair in zip(inner_sides, pairs):
        if random.random() < 0.5:
            inner_color, border_color = pair
        else:
            border_color, inner_color = pair
        specs.append(
            {
                "inner_side": inner_side,
                "side": inner_side + 2,
                "inner_color": int(inner_color),
                "border_color": int(border_color),
            }
        )
    return specs


def _render_input(case: dict) -> np.ndarray:
    grid = np.zeros((case["h"], case["w"]), dtype=int)
    for obj in case["objects"]:
        r0, c0, r1, c1 = obj["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(obj["border_color"])
        grid[r0 + 1 : r1, c0 + 1 : c1] = int(obj["inner_color"])
    for r, c in case["bar_cells"]:
        grid[r, c] = BAR
    grid[case["control"]] = CONTROL
    return grid


def _render_expected(case: dict) -> np.ndarray:
    out = np.zeros((case["h"], case["w"]), dtype=int)
    for obj in case["objects"]:
        _draw_swapped_square(out, obj)
    hits = [obj for obj in case["objects"] if obj["hit"]]
    for obj in sorted(hits, key=lambda d: (d["center"][0], d["center"][1], d["bbox"][0], d["bbox"][1])):
        for rect in _growth_rects(obj["bbox"]):
            _paint_rect(out, rect, obj["border_color"])
    return out


def _place_objects(case: dict) -> bool:
    h, w = case["h"], case["w"]
    center, step = case["center"], case["step"]
    marker_cells = set(case["bar_cells"]) | {case["control"]}
    count = len(case["objects"])

    miss_count = sum(1 for obj in case["objects"] if not obj["hit"])
    modes = []
    if miss_count:
        modes.append("opposite_aligned")
        for _ in range(miss_count - 1):
            modes.append(random.choice(["same_side_offline", "opposite_offline"]))
        random.shuffle(modes)

    miss_i = 0
    placement_order = sorted(
        range(count),
        key=lambda i: (
            0 if case["objects"][i]["hit"] else 1,
            -case["objects"][i]["side"],
            random.random(),
        ),
    )

    placed: list[dict] = []
    preplaced: set[int] = set()
    hit_indices = [i for i in placement_order if case["objects"][i]["hit"]]
    if len(hit_indices) >= 2:
        i0, i1 = hit_indices[:2]
        obj0 = case["objects"][i0]
        obj1 = case["objects"][i1]
        cand0 = _candidate_bboxes(obj0["side"], "hit", center, step, (h, w))
        cand1 = _candidate_bboxes(obj1["side"], "hit", center, step, (h, w))
        found_pair = False
        for bbox0 in cand0[:150]:
            if not _valid_bbox(bbox0, hit=True, placed=placed, marker_cells=marker_cells, shape=(h, w)):
                continue
            temp_obj = {"bbox": bbox0, "growth_rects": _growth_rects(bbox0)}
            for bbox1 in cand1[:150]:
                if not _pair_layout_ok(bbox0, bbox1, step):
                    continue
                if not _valid_bbox(bbox1, hit=True, placed=[temp_obj], marker_cells=marker_cells, shape=(h, w)):
                    continue
                for idx, bbox in ((i0, bbox0), (i1, bbox1)):
                    obj = case["objects"][idx]
                    obj["bbox"] = bbox
                    obj["center"] = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    obj["growth_rects"] = _growth_rects(bbox)
                    placed.append(obj)
                    preplaced.add(idx)
                found_pair = True
                break
            if found_pair:
                break
        if not found_pair:
            return False

    for idx in placement_order:
        if idx in preplaced:
            continue
        obj = case["objects"][idx]
        mode = "hit" if obj["hit"] else modes[miss_i]
        if not obj["hit"]:
            miss_i += 1

        candidates = _candidate_bboxes(obj["side"], mode, center, step, (h, w))
        if mode == "opposite_aligned" and not candidates:
            candidates = _candidate_bboxes(obj["side"], "opposite_offline", center, step, (h, w))
        for bbox in candidates:
            if _valid_bbox(bbox, hit=obj["hit"], placed=placed, marker_cells=marker_cells, shape=(h, w)):
                obj["bbox"] = bbox
                obj["center"] = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                obj["growth_rects"] = _growth_rects(bbox) if obj["hit"] else []
                placed.append(obj)
                break
        else:
            return False
    return True


def _build_case() -> dict | None:
    h = random.randint(21, 27)
    w = random.randint(21, 27)
    axis, center, control, bar_cells, step = _choose_marker(h, w)
    count = random.choice([4, 4, 3])
    specs = _choose_object_specs(count)
    hit_count = 2
    hit_indices = set(random.sample(range(count), hit_count))
    case = {
        "h": h,
        "w": w,
        "axis": axis,
        "center": center,
        "control": control,
        "bar_cells": bar_cells,
        "step": step,
        "objects": [],
    }
    for i, spec in enumerate(specs):
        case["objects"].append(
            {
                **spec,
                "hit": i in hit_indices,
            }
        )
    if not _place_objects(case):
        return None
    return case


def _object_from_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _validate_case(case: dict) -> bool:
    inp = _render_input(case)
    out = _render_expected(case)
    solved = main(inp)
    if not np.array_equal(out, solved):
        return False

    axis, center, control, bar_cells, step = _find_marker(inp)
    if axis != case["axis"] or center != case["center"] or control != case["control"] or step != case["step"]:
        return False

    parsed = _extract_objects(inp)
    if len(parsed) != len(case["objects"]):
        return False

    ray = set(_ray_cells(center, step, inp.shape))
    parsed_hits = sum(1 for obj in parsed if obj["coords_set"] & ray)
    expected_hits = sum(1 for obj in case["objects"] if obj["hit"])
    if parsed_hits != expected_hits or not (1 <= parsed_hits < len(parsed)):
        return False

    for obj in case["objects"]:
        if obj["hit"]:
            r0, c0, r1, c1 = obj["bbox"]
            if not (r0 >= 2 and c0 >= 2 and r1 < case["h"] - 2 and c1 < case["w"] - 2):
                return False
            for other in case["objects"]:
                if other is obj:
                    continue
                for rect in _growth_rects(obj["bbox"]):
                    if _rects_intersect(rect, other["bbox"]):
                        return False

    if np.any((solved == BAR) | (solved == CONTROL)):
        return False

    for obj in case["objects"]:
        crop = _object_from_bbox(solved, obj["bbox"])
        border_mask = np.zeros_like(crop, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        if set(np.unique(crop[border_mask])) != {obj["inner_color"]}:
            return False
        if set(np.unique(crop[~border_mask])) != {obj["border_color"]}:
            return False

    opposite_side_misses = 0
    for obj in case["objects"]:
        if not obj["hit"] and _covers_line(obj["bbox"], case["center"], case["step"]) and _on_control_side(obj["bbox"], case["center"], case["step"]):
            opposite_side_misses += 1
    if opposite_side_misses == 0:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _build_case()
        if case is None:
            continue
        if _validate_case(case):
            return _render_input(case)
    raise RuntimeError("failed to generate a valid ARC input")
