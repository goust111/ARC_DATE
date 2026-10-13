from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_h, bbox_w, draw_line, find_components, get_bg, paste


BG = 0
AREA_TO_DELTA = {
    4: (-2, 0),
    5: (0, 3),
    6: (3, 0),
    7: (0, -2),
}
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
COLOR_POOL = tuple(range(1, 10))

TEMPLATES = {
    4: (
        np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
        np.array([[0, 0, 1], [0, 1, 1], [1, 0, 0]], dtype=int),
        np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=int),
    ),
    5: (
        np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=int),
        np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0], [0, 0, 1]], dtype=int),
    ),
    6: (
        np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=int),
        np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=int),
        np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=int),
    ),
    7: (
        np.array([[1, 1, 1], [1, 1, 1], [0, 0, 1]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1]], dtype=int),
        np.array([[1, 1, 0], [1, 1, 1], [1, 0, 1]], dtype=int),
    ),
}

MAX_ATTEMPTS = 400


def _bar_midpoint(bar: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bar["bbox"])
    return (r0 + r1) // 2, (c0 + c1) // 2


def _is_bar(comp: dict) -> bool:
    h = bbox_h(comp["bbox"])
    w = bbox_w(comp["bbox"])
    return int(comp["area"]) == 3 and ((h == 3 and w == 1) or (h == 1 and w == 3))


def _ray_hit_distance(start: tuple[int, int], direction: tuple[int, int], bbox: tuple[int, int, int, int]) -> int | None:
    sr, sc = map(int, start)
    dr, dc = map(int, direction)
    r0, c0, r1, c1 = map(int, bbox)

    if dr == 0:
        if not (r0 <= sr <= r1):
            return None
        if dc == 1 and c0 > sc:
            return c0 - sc
        if dc == -1 and c1 < sc:
            return sc - c1
        return None

    if not (c0 <= sc <= c1):
        return None
    if dr == 1 and r0 > sr:
        return r0 - sr
    if dr == -1 and r1 < sr:
        return sr - r1
    return None


def _extract_scene(grid: np.ndarray) -> tuple[int, tuple[int, int], tuple[int, int], list[dict]]:
    bg = get_bg(grid)
    comps = find_components(
        grid,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )

    bars = [comp for comp in comps if _is_bar(comp)]
    singles = [comp for comp in comps if int(comp["area"]) == 1]
    glyphs = [
        comp
        for comp in comps
        if int(comp["area"]) in AREA_TO_DELTA and bbox_h(comp["bbox"]) == 3 and bbox_w(comp["bbox"]) == 3
    ]

    matches: list[tuple[dict, dict, tuple[int, int], tuple[int, int]]] = []
    for bar in bars:
        mid = _bar_midpoint(bar)
        bar_h = bbox_h(bar["bbox"])
        bar_w = bbox_w(bar["bbox"])
        for single in singles:
            sr, sc = map(int, single["coords"][0])
            dr = mid[0] - sr
            dc = mid[1] - sc
            if abs(dr) + abs(dc) != 1:
                continue
            if bar_h == 3 and bar_w == 1 and dr == 0 and dc != 0:
                matches.append((bar, single, mid, (dr, dc)))
            if bar_h == 1 and bar_w == 3 and dc == 0 and dr != 0:
                matches.append((bar, single, mid, (dr, dc)))

    assert len(matches) == 1, "expected a unique marker bar + control pixel pair"
    _bar, _control, start, direction = matches[0]
    assert glyphs, "expected at least one 3x3 glyph"
    return int(bg), start, direction, glyphs


def _render_polyline(
    shape: tuple[int, int],
    bg: int,
    start: tuple[int, int],
    ordered_glyphs: list[dict],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    cur_r, cur_c = map(int, start)
    for glyph in ordered_glyphs:
        delta_r, delta_c = AREA_TO_DELTA[int(glyph["area"])]
        end_r = cur_r + delta_r
        end_c = cur_c + delta_c
        assert 0 <= end_r < shape[0] and 0 <= end_c < shape[1], "decoded polyline exits the canvas"
        draw_line(out, cur_r, cur_c, end_r=end_r, end_c=end_c, color=int(glyph["color"]))
        cur_r, cur_c = end_r, end_c
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the ray-ordered 3x3 glyphs into a colored polyline drawn on a blank canvas.

    Input structure:
    - The background contains several scattered monochrome 3x3 glyphs.
    - One extra object is a straight three-pixel marker bar with a single control pixel touching its midpoint.

    Rule:
    - Use the marker-bar midpoint as the start point, and point a ray away from the control pixel.
    - Only glyphs whose bounding boxes are crossed by that ray are read, ordered by first hit along the ray.
    - A glyph's pixel count encodes one segment: area 4 -> up 2, area 5 -> right 3, area 6 -> down 3, area 7 -> left 2; the segment color is the glyph color.
    - On an otherwise empty canvas of the same size, draw those segments sequentially from the start point; erase every input object.

    Key insight:
    - The glyphs are not read by rows or columns: the ray's hit order selects the instructions, and off-ray glyphs are decoys.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg, start, direction, glyphs = _extract_scene(g)

    hits: list[tuple[int, dict]] = []
    for glyph in glyphs:
        dist = _ray_hit_distance(start, direction, glyph["bbox"])
        if dist is not None:
            hits.append((dist, glyph))

    assert hits, "expected at least one ray-hit glyph"
    hits.sort(key=lambda item: int(item[0]))
    assert len({int(dist) for dist, _glyph in hits}) == len(hits), "ray-hit order must be unambiguous"
    ordered = [glyph for _dist, glyph in hits]
    return _render_polyline(g.shape, bg, start, ordered)


def _glyph_patch(area: int, color: int) -> np.ndarray:
    template = random.choice(TEMPLATES[int(area)])
    return template * int(color)


def _segment_points(start: tuple[int, int], delta: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = map(int, delta)
    steps = abs(dr) + abs(dc)
    step_r = 0 if dr == 0 else dr // abs(dr)
    step_c = 0 if dc == 0 else dc // abs(dc)
    sr, sc = map(int, start)
    return [(sr + step_r * i, sc + step_c * i) for i in range(steps + 1)]


def _trace_sequence(areas: list[int]) -> dict | None:
    cur = (0, 0)
    visited = {(0, 0)}
    all_points = [(0, 0)]
    directions: list[tuple[int, int]] = []

    for area in areas:
        delta = AREA_TO_DELTA[int(area)]
        pts = _segment_points(cur, delta)
        for pt in pts[1:]:
            if pt in visited:
                return None
        visited.update(pts[1:])
        all_points.extend(pts[1:])
        cur = pts[-1]
        directions.append((np.sign(delta[0]), np.sign(delta[1])))

    rows = [r for r, _c in all_points]
    cols = [c for _r, c in all_points]
    turns = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i - 1])
    if turns < 2:
        return None
    return {
        "end": cur,
        "points": all_points,
        "bbox": (min(rows), min(cols), max(rows), max(cols)),
        "turns": turns,
    }


def _sample_areas() -> tuple[list[int], dict] | None:
    for _ in range(120):
        n_hit = random.randint(4, 5)
        areas = [4, 5, 6, 7]
        if n_hit == 5:
            areas.append(random.choice(areas))
        random.shuffle(areas)
        traced = _trace_sequence(areas)
        if traced is None:
            continue
        return areas, traced
    return None


def _sample_hit_boxes(n_hit: int, direction: tuple[int, int]) -> list[tuple[int, int, int, int]] | None:
    for _ in range(120):
        near = [random.randint(3, 4)]
        for _idx in range(1, n_hit):
            near.append(near[-1] + random.randint(4, 5))
        aligns = [random.choice((-2, -1, 0)) for _ in range(n_hit)]
        if len(set(aligns)) < 2:
            continue

        boxes: list[tuple[int, int, int, int]] = []
        dr, dc = direction
        for dist, align in zip(near, aligns):
            if (dr, dc) == (0, 1):
                boxes.append((align, dist, align + 2, dist + 2))
            elif (dr, dc) == (0, -1):
                boxes.append((align, -dist - 2, align + 2, -dist))
            elif (dr, dc) == (1, 0):
                boxes.append((dist, align, dist + 2, align + 2))
            else:
                boxes.append((-dist - 2, align, -dist, align + 2))

        return boxes
    return None


def _bar_control_rel(direction: tuple[int, int]) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    if direction in {(0, 1), (0, -1)}:
        control = (0, -1) if direction == (0, 1) else (0, 1)
        return (-1, 0, 1, 0), control
    control = (1, 0) if direction == (-1, 0) else (-1, 0)
    return (0, -1, 0, 1), control


def _union_bbox(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _shift_box(box: tuple[int, int, int, int], origin: tuple[int, int]) -> tuple[int, int, int, int]:
    sr, sc = map(int, origin)
    r0, c0, r1, c1 = map(int, box)
    return r0 + sr, c0 + sc, r1 + sr, c1 + sc


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, bg: int) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    for dr, dc in np.argwhere(patch != bg):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        if int(grid[rr, cc]) != bg:
            return False
        rr0 = max(0, rr - 1)
        rr1 = min(grid.shape[0], rr + 2)
        cc0 = max(0, cc - 1)
        cc1 = min(grid.shape[1], cc + 2)
        if np.any(grid[rr0:rr1, cc0:cc1] != bg):
            return False
    return True


def _perp_gap(start: tuple[int, int], direction: tuple[int, int], bbox: tuple[int, int, int, int]) -> int:
    sr, sc = map(int, start)
    r0, c0, r1, c1 = map(int, bbox)
    if direction[0] == 0:
        if r0 <= sr <= r1:
            return 0
        if sr < r0:
            return r0 - sr
        return sr - r1
    if c0 <= sc <= c1:
        return 0
    if sc < c0:
        return c0 - sc
    return sc - c1


def _is_forward_box(start: tuple[int, int], direction: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    sr, sc = map(int, start)
    r0, c0, r1, c1 = map(int, bbox)
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return c0 > sc
    if (dr, dc) == (0, -1):
        return c1 < sc
    if (dr, dc) == (1, 0):
        return r0 > sr
    return r1 < sr


def _all_candidate_boxes(
    grid: np.ndarray,
    patch: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    *,
    predicate,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r0 in range(grid.shape[0] - patch.shape[0] + 1):
        for c0 in range(grid.shape[1] - patch.shape[1] + 1):
            bbox = (r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1)
            if _ray_hit_distance(start, direction, bbox) is not None:
                continue
            if not predicate(bbox):
                continue
            if _can_place_patch(grid, patch, r0, c0, bg=BG):
                candidates.append((r0, c0))
    random.shuffle(candidates)
    return candidates


def _place_decoy(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    area: int,
    color: int,
    *,
    predicate,
    decoys: list[dict],
    decoy_id: int,
) -> bool:
    patch = _glyph_patch(area, color)
    candidates = _all_candidate_boxes(grid, patch, start, direction, predicate=predicate)
    if not candidates:
        return False
    r0, c0 = candidates[0]
    paste(grid, patch, r0, c0, transparent=BG)
    decoys.append({"id": f"d{decoy_id}", "bbox": (r0, c0, r0 + 2, c0 + 2), "area": int(area), "color": int(color)})
    return True


def _render_expected(shape: tuple[int, int], start: tuple[int, int], areas: list[int], colors: list[int]) -> np.ndarray:
    ordered = [{"area": int(area), "color": int(color)} for area, color in zip(areas, colors)]
    return _render_polyline(shape, BG, start, ordered)


def _attempt() -> np.ndarray | None:
    sampled = _sample_areas()
    if sampled is None:
        return None
    areas, traced = sampled
    direction = random.choice(DIRS)
    hit_rel_boxes = _sample_hit_boxes(len(areas), direction)
    if hit_rel_boxes is None:
        return None

    bar_rel_box, control_rel = _bar_control_rel(direction)
    path_box = traced["bbox"]
    rel_union = _union_bbox([path_box, bar_rel_box, *hit_rel_boxes])
    min_r, min_c, max_r, max_c = rel_union

    min_h = max(16, max_r - min_r + 3)
    min_w = max(16, max_c - min_c + 3)
    if min_h > 30 or min_w > 30:
        return None

    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)
    r_lo = 1 - min_r
    r_hi = height - 2 - max_r
    c_lo = 1 - min_c
    c_hi = width - 2 - max_c
    if r_lo > r_hi or c_lo > c_hi:
        return None

    start = (random.randint(r_lo, r_hi), random.randint(c_lo, c_hi))
    grid = np.full((height, width), BG, dtype=int)

    palette = list(COLOR_POOL)
    random.shuffle(palette)
    bar_color = palette.pop()
    control_color = palette.pop()
    hit_colors = [palette.pop() for _ in areas]

    if direction in {(0, 1), (0, -1)}:
        bar_patch = np.full((3, 1), bar_color, dtype=int)
        paste(grid, bar_patch, start[0] - 1, start[1])
    else:
        bar_patch = np.full((1, 3), bar_color, dtype=int)
        paste(grid, bar_patch, start[0], start[1] - 1)
    grid[start[0] + control_rel[0], start[1] + control_rel[1]] = control_color

    hits: list[dict] = []
    for idx, (area, color, rel_box) in enumerate(zip(areas, hit_colors, hit_rel_boxes)):
        patch = _glyph_patch(area, color)
        r0, c0, _r1, _c1 = _shift_box(rel_box, start)
        if not _can_place_patch(grid, patch, r0, c0, bg=BG):
            return None
        paste(grid, patch, r0, c0, transparent=BG)
        hits.append({"id": f"h{idx}", "bbox": (r0, c0, r0 + 2, c0 + 2), "area": int(area), "color": int(color)})

    if not all(_ray_hit_distance(start, direction, hit["bbox"]) is not None for hit in hits):
        return None

    decoys: list[dict] = []
    near_area = random.choice(tuple(AREA_TO_DELTA))
    near_color = random.choice(COLOR_POOL)
    if not _place_decoy(
        grid,
        start,
        direction,
        near_area,
        near_color,
        predicate=lambda bbox: _is_forward_box(start, direction, bbox) and _perp_gap(start, direction, bbox) == 1,
        decoys=decoys,
        decoy_id=0,
    ):
        return None

    first_hit = hits[0]["bbox"]
    if direction == (0, 1):
        order_pred = lambda bbox: bbox[1] < first_hit[1]
    elif direction == (0, -1):
        order_pred = lambda bbox: bbox[1] > first_hit[1]
    elif direction == (1, 0):
        order_pred = lambda bbox: bbox[0] < first_hit[0]
    else:
        order_pred = lambda bbox: bbox[0] > first_hit[0]

    order_area = random.choice(tuple(AREA_TO_DELTA))
    order_color = random.choice(COLOR_POOL)
    if not _place_decoy(
        grid,
        start,
        direction,
        order_area,
        order_color,
        predicate=order_pred,
        decoys=decoys,
        decoy_id=1,
    ):
        return None

    target_decoys = random.randint(2, 4)
    for decoy_id in range(2, target_decoys):
        area = random.choice(tuple(AREA_TO_DELTA))
        color = random.choice(COLOR_POOL)
        placed = _place_decoy(
            grid,
            start,
            direction,
            area,
            color,
            predicate=lambda bbox: True,
            decoys=decoys,
            decoy_id=decoy_id,
        )
        if not placed:
            return None

    if not all(_ray_hit_distance(start, direction, decoy["bbox"]) is None for decoy in decoys):
        return None

    all_glyphs = hits + decoys
    row_order = [item["id"] for item in sorted(all_glyphs, key=lambda item: (item["bbox"][0], item["bbox"][1]))]
    col_order = [item["id"] for item in sorted(all_glyphs, key=lambda item: (item["bbox"][1], item["bbox"][0]))]
    hit_ids = [item["id"] for item in hits]
    if row_order[: len(hit_ids)] == hit_ids:
        return None
    if col_order[: len(hit_ids)] == hit_ids:
        return None

    expected = _render_expected(grid.shape, start, areas, hit_colors)
    if len({int(v) for v in expected.ravel() if int(v) != BG}) < 3:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-ordered glyph decoding input")
