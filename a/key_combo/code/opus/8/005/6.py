from __future__ import annotations

import random
from typing import Iterable

import numpy as np


OUTER_COLORS = (1, 2)
INNER_COLORS = (3, 4, 6, 7)
MARKER_COLORS = (5, 8)
BASE_COLORS = OUTER_COLORS + INNER_COLORS
SIDES = ("top", "bottom", "left", "right")


def _prefix(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.int16).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    total = int(prefix[r1, c1])
    if r0 > 0:
        total -= int(prefix[r0 - 1, c1])
    if c0 > 0:
        total -= int(prefix[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(prefix[r0 - 1, c0 - 1])
    return total


def _perimeter_sum(prefix: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    total = _rect_sum(prefix, r0, c0, r1, c1)
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        total -= _rect_sum(prefix, r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    return total


def _border_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    coords.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    coords.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    coords.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return coords


def _side_coords(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if side == "left":
        return [(r, c0) for r in range(r0, r1 + 1)]
    return [(r, c1) for r in range(r0, r1 + 1)]


def _side_values(grid: np.ndarray, bbox: tuple[int, int, int, int], side: str) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return grid[r0, c0 : c1 + 1]
    if side == "bottom":
        return grid[r1, c0 : c1 + 1]
    if side == "left":
        return grid[r0 : r1 + 1, c0]
    return grid[r0 : r1 + 1, c1]


def _valid_border_pattern(grid: np.ndarray, bbox: tuple[int, int, int, int], base_color: int) -> bool:
    allowed = {base_color, *MARKER_COLORS}
    side_has_base = 0
    for side in SIDES:
        vals = _side_values(grid, bbox, side)
        if any(int(v) not in allowed for v in vals):
            return False
        if np.any(vals == base_color):
            side_has_base += 1
    return side_has_base >= 2


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    found: dict[tuple[tuple[int, int, int, int], int], dict] = {}
    for base_color in BASE_COLORS:
        allowed_mask = (grid == base_color) | (grid == MARKER_COLORS[0]) | (grid == MARKER_COLORS[1])
        for comp in _components(allowed_mask):
            base_hits = sum(int(grid[r, c] == base_color) for r, c in comp)
            if base_hits == 0:
                continue
            r0 = int(comp[:, 0].min())
            c0 = int(comp[:, 1].min())
            r1 = int(comp[:, 0].max())
            c1 = int(comp[:, 1].max())
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            bbox = (r0, c0, r1, c1)
            perim = 2 * ((r1 - r0) + (c1 - c0))
            if len(comp) != perim:
                continue
            if not all(r in {r0, r1} or c in {c0, c1} for r, c in comp):
                continue
            comp_set = {(int(r), int(c)) for r, c in comp}
            if any(coord not in comp_set for coord in _border_coords(bbox)):
                continue
            if not _valid_border_pattern(grid, bbox, base_color):
                continue
            found[(bbox, base_color)] = {
                "bbox": bbox,
                "base_color": int(base_color),
                "parent": None,
                "children": [],
                "contact": None,
                "depth": 0,
            }
    rects = list(found.values())
    rects.sort(key=lambda item: ((item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1]), item["bbox"]))
    return rects


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 <= ir0 and oc0 <= ic0 and ir1 <= or1 and ic1 <= oc1 and outer != inner


def _area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _attach_hierarchy(rects: list[dict]) -> None:
    for i, rect in enumerate(rects):
        parent = None
        parent_area = None
        for j, other in enumerate(rects):
            if i == j or not _contains(other["bbox"], rect["bbox"]):
                continue
            candidate_area = _area(other["bbox"])
            if parent is None or candidate_area < parent_area:
                parent = j
                parent_area = candidate_area
        rect["parent"] = parent
    for rect in rects:
        rect["children"] = []
    for i, rect in enumerate(rects):
        if rect["parent"] is not None:
            rects[rect["parent"]]["children"].append(i)
    for i in range(len(rects)):
        depth = 0
        p = rects[i]["parent"]
        while p is not None:
            depth += 1
            p = rects[p]["parent"]
        rects[i]["depth"] = depth


def _root_index(rects: list[dict], idx: int) -> int:
    while rects[idx]["parent"] is not None:
        idx = rects[idx]["parent"]
    return idx


def _contact_with_parent(grid: np.ndarray, rect: dict, parent: dict) -> tuple[str, int] | None:
    bbox = rect["bbox"]
    pb = parent["bbox"]
    matches: list[tuple[str, int]] = []
    for side in SIDES:
        if side == "top" and bbox[0] != pb[0]:
            continue
        if side == "bottom" and bbox[2] != pb[2]:
            continue
        if side == "left" and bbox[1] != pb[1]:
            continue
        if side == "right" and bbox[3] != pb[3]:
            continue
        vals = _side_values(grid, bbox, side)
        uniques = {int(v) for v in vals}
        if len(uniques) == 1 and next(iter(uniques)) in MARKER_COLORS:
            matches.append((side, next(iter(uniques))))
    if len(matches) == 1:
        return matches[0]
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each non-outer rectangular frame by either its contact-side marker or its outermost container color, then fill its black interior.

    Input structure:
    - The background is black, with one or two outer rectangular borders in blue or red.
    - Each outer rectangle contains smaller rectangular borders in green, yellow, magenta, or orange, and some of those smaller rectangles share exactly one side with their direct container.
    - Shared-side contact is marked by gray or teal pixels placed only on the overlapping border segment.

    Rule:
    - Detect the rectangular containment hierarchy, allowing an inner rectangle to share one full side with its direct container.
    - If a non-outer rectangle has a fully marked shared side with its parent, recolor that rectangle to the marker color and keep the shared container segment in that marker color.
    - Otherwise recolor the non-outer rectangle to the color of the outermost rectangle that contains it, then fill any black pixels inside the recolored rectangle with the same new color.

    Key insight:
    - The same nesting geometry splits into two mutually exclusive cases: touching a parent side activates marker inheritance, while floating rectangles inherit the outermost container color.
    """
    g = np.asarray(grid, dtype=int)
    rects = _find_rectangles(g)
    if not rects:
        return g.copy()

    _attach_hierarchy(rects)
    protected: list[set[tuple[int, int]]] = [set() for _ in rects]
    for idx, rect in enumerate(rects):
        parent_idx = rect["parent"]
        if parent_idx is None:
            continue
        contact = _contact_with_parent(g, rect, rects[parent_idx])
        rect["contact"] = contact
        if contact is not None:
            side, _marker = contact
            protected[parent_idx].update(_side_coords(rect["bbox"], side))

    out = g.copy()
    order = sorted(
        [i for i, rect in enumerate(rects) if rect["parent"] is not None],
        key=lambda i: (rects[i]["depth"], _area(rects[i]["bbox"])),
        reverse=True,
    )
    for idx in order:
        rect = rects[idx]
        if rect["contact"] is not None:
            new_color = int(rect["contact"][1])
        else:
            root = rects[_root_index(rects, idx)]
            new_color = int(root["base_color"])

        for r, c in _border_coords(rect["bbox"]):
            if (r, c) not in protected[idx]:
                out[r, c] = new_color

        r0, c0, r1, c1 = rect["bbox"]
        if r1 - r0 >= 2 and c1 - c0 >= 2:
            interior = out[r0 + 1 : r1, c0 + 1 : c1]
            interior[interior == 0] = new_color

    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _sample_contact_bbox(
    outer: tuple[int, int, int, int], used: Iterable[tuple[int, int, int, int]], side: str
) -> tuple[int, int, int, int] | None:
    or0, oc0, or1, oc1 = outer
    oh = or1 - or0 + 1
    ow = oc1 - oc0 + 1
    attempts = 80
    while attempts > 0:
        attempts -= 1
        if side in {"top", "bottom"}:
            width = random.randint(3, max(3, ow - 2))
            height = random.randint(3, max(3, min(7, oh - 2)))
            if width > ow - 2:
                continue
            c0 = random.randint(oc0 + 1, oc1 - width)
            c1 = c0 + width - 1
            if side == "top":
                r0, r1 = or0, or0 + height - 1
            else:
                r1, r0 = or1, or1 - height + 1
        else:
            height = random.randint(3, max(3, oh - 2))
            width = random.randint(3, max(3, min(7, ow - 2)))
            if height > oh - 2:
                continue
            r0 = random.randint(or0 + 1, or1 - height)
            r1 = r0 + height - 1
            if side == "left":
                c0, c1 = oc0, oc0 + width - 1
            else:
                c1, c0 = oc1, oc1 - width + 1
        box = (r0, c0, r1, c1)
        if all(_boxes_separate(box, other) for other in used):
            return box
    return None


def _sample_floating_bbox(
    outer: tuple[int, int, int, int], used: Iterable[tuple[int, int, int, int]]
) -> tuple[int, int, int, int] | None:
    or0, oc0, or1, oc1 = outer
    oh = or1 - or0 + 1
    ow = oc1 - oc0 + 1
    attempts = 120
    while attempts > 0:
        attempts -= 1
        height = random.randint(3, max(3, min(7, oh - 4)))
        width = random.randint(3, max(3, min(7, ow - 4)))
        if height > oh - 4 or width > ow - 4:
            continue
        r0 = random.randint(or0 + 2, or1 - height - 1)
        c0 = random.randint(oc0 + 2, oc1 - width - 1)
        box = (r0, c0, r0 + height - 1, c0 + width - 1)
        if all(_boxes_separate(box, other) for other in used):
            return box
    return None


def _paint_inner(
    grid: np.ndarray, bbox: tuple[int, int, int, int], base_color: int, touch_side: str | None, marker: int | None
) -> None:
    _draw_frame(grid, bbox, base_color)
    if touch_side is not None and marker is not None:
        coords = _side_coords(bbox, touch_side)
        for r, c in coords:
            grid[r, c] = marker


def _build_outer_children(grid: np.ndarray, outer: tuple[int, int, int, int]) -> bool:
    count = random.choice((2, 3))
    colors = random.sample(INNER_COLORS, count)
    used: list[tuple[int, int, int, int]] = []
    contact_sides = random.sample(list(SIDES), random.choice((1, 2)))
    kinds = ["contact", "floating"]
    if count == 3:
        kinds.append(random.choice(("contact", "floating")))
    random.shuffle(kinds)
    children: list[tuple[tuple[int, int, int, int], int, str | None, int | None]] = []

    for i, kind in enumerate(kinds):
        if kind == "contact":
            side_pool = [side for side in SIDES if side not in contact_sides[: i]]
            side = random.choice(side_pool) if side_pool else random.choice(SIDES)
            bbox = _sample_contact_bbox(outer, used, side)
            if bbox is None:
                return False
            marker = random.choice(MARKER_COLORS)
            children.append((bbox, colors[i], side, marker))
            used.append(bbox)
        else:
            bbox = _sample_floating_bbox(outer, used)
            if bbox is None:
                return False
            children.append((bbox, colors[i], None, None))
            used.append(bbox)

    if not any(side is not None for _, _, side, _ in children):
        return False
    if not any(side is None for _, _, side, _ in children):
        return False

    for bbox, color, side, marker in children:
        _paint_inner(grid, bbox, color, side, marker)
    return True


def _layout_outers() -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[int]] | None:
    outer_count = random.choice((1, 1, 2))
    if outer_count == 1:
        oh = random.randint(10, 15)
        ow = random.randint(10, 15)
        H = oh + random.randint(4, 9)
        W = ow + random.randint(4, 9)
        if H > 30 or W > 30:
            return None
        grid = np.zeros((H, W), dtype=int)
        r0 = random.randint(1, H - oh - 1)
        c0 = random.randint(1, W - ow - 1)
        outer = (r0, c0, r0 + oh - 1, c0 + ow - 1)
        return grid, [outer], [random.choice(OUTER_COLORS)]

    layout = random.choice(("horizontal", "vertical"))
    a_h, a_w = random.randint(10, 13), random.randint(10, 13)
    b_h, b_w = random.randint(10, 13), random.randint(10, 13)
    gap = random.randint(2, 4)
    margin = random.randint(1, 2)
    if layout == "horizontal":
        H = max(a_h, b_h) + 2 * margin + random.randint(0, 2)
        W = a_w + b_w + gap + 2 * margin + random.randint(0, 2)
        if H > 30 or W > 30:
            return None
        grid = np.zeros((H, W), dtype=int)
        a_r0 = random.randint(margin, H - a_h - margin)
        b_r0 = random.randint(margin, H - b_h - margin)
        a_c0 = margin
        b_c0 = W - margin - b_w
    else:
        H = a_h + b_h + gap + 2 * margin + random.randint(0, 2)
        W = max(a_w, b_w) + 2 * margin + random.randint(0, 2)
        if H > 30 or W > 30:
            return None
        grid = np.zeros((H, W), dtype=int)
        a_r0 = margin
        b_r0 = H - margin - b_h
        a_c0 = random.randint(margin, W - a_w - margin)
        b_c0 = random.randint(margin, W - b_w - margin)
    outer_a = (a_r0, a_c0, a_r0 + a_h - 1, a_c0 + a_w - 1)
    outer_b = (b_r0, b_c0, b_r0 + b_h - 1, b_c0 + b_w - 1)
    colors = random.sample(list(OUTER_COLORS), 2)
    return grid, [outer_a, outer_b], colors


def generate_input() -> np.ndarray:
    for _ in range(300):
        layout = _layout_outers()
        if layout is None:
            continue
        grid, outers, colors = layout
        ok = True
        for outer, color in zip(outers, colors):
            _draw_frame(grid, outer, color)
            if not _build_outer_children(grid, outer):
                ok = False
                break
        if not ok:
            continue
        out = main(grid)
        if np.array_equal(out, grid):
            continue
        if grid.ndim == 2 and out.ndim == 2 and max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
