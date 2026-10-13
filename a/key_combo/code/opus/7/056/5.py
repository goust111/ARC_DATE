from __future__ import annotations

import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MIN_GRID = 15
MAX_GRID = 19
MIN_OBJECTS = 3
MAX_OBJECTS = 5
MIN_BBOX_GAP = 3
GEN_RETRIES = 400


SHAPE_TEMPLATES = {
    2: [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    ],
    3: [
        ((0, -1), (0, 0), (0, 1)),
        ((-1, 0), (0, 0), (1, 0)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ],
    4: [
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, -1), (0, 0), (0, 1), (1, 0)),
        ((1, -1), (1, 0), (1, 1), (0, 0)),
        ((-1, 0), (0, 0), (1, 0), (1, 1)),
        ((-1, 1), (0, 1), (1, 1), (1, 0)),
        ((0, -1), (0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, -1), (1, 0)),
        ((0, -1), (0, 0), (0, 1), (0, 2)),
        ((-1, 0), (0, 0), (1, 0), (2, 0)),
    ],
    5: [
        ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)),
        ((0, -2), (0, -1), (0, 0), (0, 1), (0, 2)),
        ((-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)),
        ((-1, -1), (-1, 0), (0, 0), (1, 0), (1, 1)),
        ((-1, 1), (-1, 0), (0, 0), (1, 0), (1, -1)),
        ((-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)),
        ((1, -1), (1, 0), (1, 1), (0, 0), (-1, 0)),
        ((-1, 0), (0, 0), (1, 0), (1, 1), (0, 1)),
    ],
}


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _template_info(cells: tuple[tuple[int, int], ...]) -> dict:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return {
        "cells": tuple((int(r), int(c)) for r, c in cells),
        "min_r": int(min(rows)),
        "max_r": int(max(rows)),
        "min_c": int(min(cols)),
        "max_c": int(max(cols)),
        "h": int(max(rows) - min(rows) + 1),
        "w": int(max(cols) - min(cols) + 1),
    }


TEMPLATES = {count: [_template_info(shape) for shape in shapes] for count, shapes in SHAPE_TEMPLATES.items()}


def _validate_templates() -> None:
    for count, templates in TEMPLATES.items():
        for template in templates:
            cells = set(template["cells"])
            assert len(cells) == count
            seen = set()
            stack = [next(iter(cells))]
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                for nbr in _neighbors4(*cur):
                    if nbr in cells and nbr not in seen:
                        stack.append(nbr)
            assert seen == cells
            assert (template["min_r"] + template["max_r"]) // 2 == 0
            assert (template["min_c"] + template["max_c"]) // 2 == 0


_validate_templates()


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BLACK or seen[r, c]:
                continue

            coords = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            objects.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "count": len(coords),
                    "bbox": (r0, c0, r1, c1),
                    "center": ((r0 + r1) // 2, (c0 + c1) // 2),
                }
            )
    return objects


def _ring_value(obj: dict, r: int, c: int) -> int | None:
    cr, cc = obj["center"]
    dist = max(abs(int(r) - int(cr)), abs(int(c) - int(cc)))
    if dist >= int(obj["count"]):
        return None
    return int(obj["color"]) if dist % 2 == 0 else BLACK


def _draw_object_rings(out: np.ndarray, obj: dict) -> None:
    h, w = out.shape
    cr, cc = obj["center"]
    for radius in range(int(obj["count"])):
        value = int(obj["color"]) if radius % 2 == 0 else BLACK
        r0 = cr - radius
        r1 = cr + radius
        c0 = cc - radius
        c1 = cc + radius
        if 0 <= r0 < h:
            for c in range(max(0, c0), min(w - 1, c1) + 1):
                out[r0, c] = value
        if radius > 0 and 0 <= r1 < h:
            for c in range(max(0, c0), min(w - 1, c1) + 1):
                out[r1, c] = value
        for r in range(max(0, r0 + 1), min(h - 1, r1 - 1) + 1):
            if 0 <= c0 < w:
                out[r, c0] = value
            if radius > 0 and 0 <= c1 < w:
                out[r, c1] = value


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each small object with alternating concentric square rings whose depth equals its pixel count.

    Input structure:
    - The grid is a black background containing 3-5 separated 4-connected monochromatic foreground objects.
    - Each object is small, irregular, and uses its own non-black color.
    - The object's bounding box center determines where the replacement structure will be anchored.

    Rule:
    - Count the pixels in each foreground object and compute the floor-centered midpoint of its bounding box.
    - Erase the original shape and draw exactly that many concentric square layers around the center.
    - Odd-numbered layers use the object's color, even-numbered layers use black, and overlaps are resolved by
      giving priority to larger objects, then to smaller top-row bounding boxes, then to smaller left columns.

    Key insight:
    - Shape is discarded entirely: only pixel count, object color, and bounding-box center survive into the output,
      while the count also doubles as the overlap-priority rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _extract_objects(g)
    ordered = sorted(objects, key=lambda obj: (int(obj["count"]), -int(obj["bbox"][0]), -int(obj["bbox"][1])))
    out = np.zeros_like(g)
    for obj in ordered:
        _draw_object_rings(out, obj)
    return out


def _bbox_from_center(obj: dict) -> tuple[int, int, int, int]:
    cr, cc = obj["center"]
    return (
        int(cr + obj["template"]["min_r"]),
        int(cc + obj["template"]["min_c"]),
        int(cr + obj["template"]["max_r"]),
        int(cc + obj["template"]["max_c"]),
    )


def _ring_box(obj: dict) -> tuple[int, int, int, int]:
    cr, cc = obj["center"]
    radius = int(obj["count"]) - 1
    return int(cr - radius), int(cc - radius), int(cr + radius), int(cc + radius)


def _box_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    row_gap = max(0, int(b[0]) - int(a[2]) - 1, int(a[0]) - int(b[2]) - 1)
    col_gap = max(0, int(b[1]) - int(a[3]) - 1, int(a[1]) - int(b[3]) - 1)
    return int(max(row_gap, col_gap))


def _overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return 0
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _has_visible_ring_conflict(a: dict, b: dict) -> bool:
    box_a = _ring_box(a)
    box_b = _ring_box(b)
    r0 = max(box_a[0], box_b[0])
    c0 = max(box_a[1], box_b[1])
    r1 = min(box_a[2], box_b[2])
    c1 = min(box_a[3], box_b[3])
    if r0 > r1 or c0 > c1:
        return False
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            va = _ring_value(a, r, c)
            vb = _ring_value(b, r, c)
            if va is not None and vb is not None and va != vb:
                return True
    return False


def _pick_template(count: int, *, min_row: int | None = None) -> dict:
    choices = TEMPLATES[count]
    if min_row is not None:
        choices = [template for template in choices if int(template["min_r"]) == int(min_row)]
    return dict(random.choice(choices))


def _make_object(color: int, count: int, template: dict) -> dict:
    obj = {
        "color": int(color),
        "count": int(count),
        "template": template,
    }
    obj["center"] = (0, 0)
    return obj


def _place_center_random(obj: dict, h: int, w: int, placed: list[dict], *, allow_ring_overlap: bool) -> bool:
    radius = int(obj["count"]) - 1
    rows = list(range(radius, h - radius))
    cols = list(range(radius, w - radius))
    random.shuffle(rows)
    random.shuffle(cols)
    for cr in rows:
        for cc in cols:
            obj["center"] = (int(cr), int(cc))
            bbox = _bbox_from_center(obj)
            ok = True
            for other in placed:
                if _box_gap(bbox, _bbox_from_center(other)) < MIN_BBOX_GAP:
                    ok = False
                    break
                if not allow_ring_overlap and _overlap_area(_ring_box(obj), _ring_box(other)) > 0:
                    ok = False
                    break
            if ok:
                return True
    return False


def _safe_anchor_range(delta: int, radius_a: int, radius_b: int, limit: int) -> tuple[int, int] | None:
    low = max(radius_a, radius_b - delta)
    high = min(limit - 1 - radius_a, limit - 1 - radius_b - delta)
    if low > high:
        return None
    return int(low), int(high)


def _place_overlap_pair(a: dict, b: dict, h: int, w: int, mode: str) -> bool:
    radius_sum = (int(a["count"]) - 1) + (int(b["count"]) - 1)
    for _ in range(200):
        if mode == "tie_left":
            delta_r = int(a["template"]["min_r"]) - int(b["template"]["min_r"])
            if abs(delta_r) > radius_sum:
                continue
            delta_c = random.choice((-radius_sum, radius_sum))
        else:
            delta_r = random.choice((-radius_sum, radius_sum))
            delta_c = random.choice((-radius_sum, radius_sum))

        row_range = _safe_anchor_range(delta_r, int(a["count"]) - 1, int(b["count"]) - 1, h)
        col_range = _safe_anchor_range(delta_c, int(a["count"]) - 1, int(b["count"]) - 1, w)
        if row_range is None or col_range is None:
            continue

        a["center"] = (
            random.randint(row_range[0], row_range[1]),
            random.randint(col_range[0], col_range[1]),
        )
        b["center"] = (int(a["center"][0] + delta_r), int(a["center"][1] + delta_c))

        bbox_a = _bbox_from_center(a)
        bbox_b = _bbox_from_center(b)
        if _box_gap(bbox_a, bbox_b) < MIN_BBOX_GAP:
            continue

        overlap = _overlap_area(_ring_box(a), _ring_box(b))
        if overlap <= 0 or overlap > 8:
            continue
        if not _has_visible_ring_conflict(a, b):
            continue

        top_diff = int(a["center"][0] + a["template"]["min_r"]) - int(b["center"][0] + b["template"]["min_r"])
        if mode == "tie_top" and top_diff == 0:
            continue
        if mode == "tie_left" and top_diff != 0:
            continue
        return True
    return False


def _scenario_name() -> str:
    roll = random.random()
    if roll < 0.45:
        return "independent"
    if roll < 0.80:
        return "overlap_diff"
    if roll < 0.95:
        return "tie_top"
    return "tie_left"


def _build_objects(scenario: str, k: int) -> tuple[list[dict], tuple[int, int] | None]:
    colors = random.sample(PALETTE, k)
    objects: list[dict] = []
    pair: tuple[int, int] | None = None

    if scenario == "independent":
        counts = [random.randint(2, 5) for _ in range(k)]
        for color, count in zip(colors, counts):
            objects.append(_make_object(color, count, _pick_template(count)))
        return objects, None

    if scenario == "overlap_diff":
        count_a = random.choice((2, 3, 4, 5))
        count_b = random.choice([n for n in (2, 3, 4, 5) if n != count_a])
        pair_counts = [count_a, count_b]
    elif scenario == "tie_top":
        shared = random.choice((3, 5))
        pair_counts = [shared, shared]
    else:
        shared = random.choice((3, 4))
        pair_counts = [shared, shared]

    objects.append(_make_object(colors[0], pair_counts[0], _pick_template(pair_counts[0])))
    if scenario == "tie_left":
        first_min = int(objects[0]["template"]["min_r"])
        possible = [template for template in TEMPLATES[pair_counts[1]] if int(template["min_r"]) != first_min]
        second_template = dict(random.choice(possible))
    else:
        second_template = _pick_template(pair_counts[1])
    objects.append(_make_object(colors[1], pair_counts[1], second_template))
    pair = (0, 1)

    for color in colors[2:]:
        count = random.randint(2, 5)
        objects.append(_make_object(color, count, _pick_template(count)))

    random.shuffle(objects[2:])
    return objects, pair


def _place_objects(objects: list[dict], pair: tuple[int, int] | None, scenario: str, h: int, w: int) -> bool:
    placed: list[dict] = []
    if pair is not None:
        a = objects[pair[0]]
        b = objects[pair[1]]
        if scenario == "overlap_diff":
            if not _place_overlap_pair(a, b, h, w, "diff"):
                return False
        elif scenario == "tie_top":
            if not _place_overlap_pair(a, b, h, w, "tie_top"):
                return False
        else:
            if not _place_overlap_pair(a, b, h, w, "tie_left"):
                return False
        placed.extend([a, b])

    for idx, obj in enumerate(objects):
        if pair is not None and idx in pair:
            continue
        if not _place_center_random(obj, h, w, placed, allow_ring_overlap=False):
            return False
        placed.append(obj)
    return True


def _render_input(objects: list[dict], h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for obj in objects:
        cr, cc = obj["center"]
        for dr, dc in obj["template"]["cells"]:
            grid[int(cr + dr), int(cc + dc)] = int(obj["color"])
    return grid


def _validate_instance(grid: np.ndarray, scenario: str, pair: tuple[int, int] | None) -> bool:
    objects = _extract_objects(grid)
    if not (MIN_OBJECTS <= len(objects) <= MAX_OBJECTS):
        return False

    colors = [int(obj["color"]) for obj in objects]
    if len(colors) != len(set(colors)):
        return False

    for obj in objects:
        if not (2 <= int(obj["count"]) <= 5):
            return False
        radius = int(obj["count"]) - 1
        cr, cc = obj["center"]
        h, w = grid.shape
        if not (radius <= cr < h - radius and radius <= cc < w - radius):
            return False

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if _box_gap(objects[i]["bbox"], objects[j]["bbox"]) < MIN_BBOX_GAP:
                return False

    output = main(grid)
    if np.array_equal(grid, output):
        return False

    if scenario == "independent":
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                if _overlap_area(_ring_box(objects[i]), _ring_box(objects[j])) > 0:
                    return False
    elif pair is not None:
        pair_objs = sorted(objects, key=lambda obj: (-int(obj["count"]), int(obj["bbox"][0]), int(obj["bbox"][1])))[:2]
        if _overlap_area(_ring_box(pair_objs[0]), _ring_box(pair_objs[1])) <= 0:
            return False
        if not _has_visible_ring_conflict(pair_objs[0], pair_objs[1]):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        scenario = _scenario_name()
        h = random.randint(MIN_GRID, MAX_GRID)
        w = random.randint(MIN_GRID, MAX_GRID)
        k = random.randint(MIN_OBJECTS, MAX_OBJECTS)
        objects, pair = _build_objects(scenario, k)
        if not _place_objects(objects, pair, scenario, h, w):
            continue
        grid = _render_input(objects, h, w)
        if _validate_instance(grid, scenario, pair):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
