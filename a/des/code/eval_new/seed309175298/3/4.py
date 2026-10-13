from __future__ import annotations

import random

import numpy as np


BG = 0
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
DIR_VECTORS = {
    "R": (0, 1),
    "L": (0, -1),
    "D": (1, 0),
    "U": (-1, 0),
}
PALETTE = tuple(range(1, 10))
MAX_ARROW_BODY = 6
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    if np.any(grid == BG):
        return BG
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _same_color_components8(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            components.append({"color": color, "cells": cells})
    return components


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _is_solid_rectangle(cells: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _match_arrow(cells: list[tuple[int, int]]) -> dict | None:
    coords = set(cells)
    r0, c0, r1, c1 = _bbox(cells)

    if r1 - r0 == 2 and c1 - c0 >= 3:
        mid = r0 + 1
        expected = {(mid, c1), (r0, c1 - 1), (r1, c1 - 1)}
        expected.update((mid, c) for c in range(c0, c1 - 1))
        if coords == expected and (c1 - c0 - 1) >= 2:
            return {"direction": "R", "tip": (mid, c1)}

        expected = {(mid, c0), (r0, c0 + 1), (r1, c0 + 1)}
        expected.update((mid, c) for c in range(c0 + 2, c1 + 1))
        if coords == expected and (c1 - c0 - 1) >= 2:
            return {"direction": "L", "tip": (mid, c0)}

    if c1 - c0 == 2 and r1 - r0 >= 3:
        mid = c0 + 1
        expected = {(r1, mid), (r1 - 1, c0), (r1 - 1, c1)}
        expected.update((r, mid) for r in range(r0, r1 - 1))
        if coords == expected and (r1 - r0 - 1) >= 2:
            return {"direction": "D", "tip": (r1, mid)}

        expected = {(r0, mid), (r0 + 1, c0), (r0 + 1, c1)}
        expected.update((r, mid) for r in range(r0 + 2, r1 + 1))
        if coords == expected and (r1 - r0 - 1) >= 2:
            return {"direction": "U", "tip": (r0, mid)}

    return None


def _detect_parts(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict], dict[tuple[int, int], int]]:
    objects: list[dict] = []
    arrows: list[dict] = []
    cell_to_object: dict[tuple[int, int], int] = {}

    for comp in _same_color_components8(grid, bg):
        color = int(comp["color"])
        cells = comp["cells"]
        if _is_solid_rectangle(cells):
            bbox = _bbox(cells)
            obj_idx = len(objects)
            objects.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "area": _rect_area(bbox),
                }
            )
            for cell in cells:
                cell_to_object[cell] = obj_idx
            continue

        arrow = _match_arrow(cells)
        assert arrow is not None, "every non-rectangle component must be a valid arrow"
        arrows.append({"color": color, "direction": arrow["direction"], "tip": arrow["tip"]})

    assert objects, "expected at least one rectangular object"
    return objects, arrows, cell_to_object


def _find_target_object(
    tip: tuple[int, int],
    direction: str,
    cell_to_object: dict[tuple[int, int], int],
    shape: tuple[int, int],
) -> int:
    h, w = shape
    dr, dc = DIR_VECTORS[direction]
    r, c = tip[0] + dr, tip[1] + dc
    while 0 <= r < h and 0 <= c < w:
        obj_idx = cell_to_object.get((r, c))
        if obj_idx is not None:
            return obj_idx
        r += dr
        c += dc
    raise AssertionError("arrow must point to a nearest rectangle object")


def _paint_objects(objects: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    order = sorted(
        range(len(objects)),
        key=lambda idx: (int(objects[idx].get("count", 0)), -int(objects[idx]["area"]), idx),
    )
    for idx in order:
        obj = objects[idx]
        grow = int(obj.get("count", 0))
        r0, c0, r1, c1 = obj["bbox"]
        rr0 = max(0, r0 - grow)
        cc0 = max(0, c0 - grow)
        rr1 = min(h - 1, r1 + grow)
        cc1 = min(w - 1, c1 + grow)
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the arrows and enlarge each rectangle by the number of arrows pointing at it.

    Input structure:
    - The background is black, with several monochrome rectangular objects scattered on it.
    - Separate monochrome arrow components sit in the gaps between objects.
    - Each arrow is an orthogonal 1-pixel shaft of length at least 2 plus a 3-pixel V-shaped head whose tip faces one rectangle.

    Rule:
    - Identify every rectangle and every arrow, including the arrow direction from its V-shaped head.
    - The target of an arrow is the first rectangular object encountered when walking forward from the arrow tip along its pointing direction.
    - Count how many arrows point to each rectangle; if a rectangle receives N arrows, expand its bbox by N cells on every side, and if N=0 keep it unchanged.
    - Erase all arrows in the output.
    - If expanded rectangles overlap, the rectangle with more incoming arrows is on top; if the counts tie, the rectangle with the smaller original area is on top.

    Key insight:
    - Arrow colors are irrelevant labels: the real signal is the arrow head orientation, which turns each arrow into one unit of geometric growth for the nearest rectangle it points toward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    objects, arrows, cell_to_object = _detect_parts(g, bg)
    counts = [0] * len(objects)
    for arrow in arrows:
        target = _find_target_object(arrow["tip"], arrow["direction"], cell_to_object, g.shape)
        counts[target] += 1

    grown_objects = []
    for idx, obj in enumerate(objects):
        grown = dict(obj)
        grown["count"] = counts[idx]
        grown_objects.append(grown)

    return _paint_objects(grown_objects, g.shape, bg)


def _expanded_bbox(obj: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = obj["bbox"]
    grow = int(obj["count"])
    return r0 - grow, c0 - grow, r1 + grow, c1 + grow


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _rect_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _cells_clear(cells: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or occupied[r, c]:
            return False
    for r, c in cells:
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if np.any(occupied[r0:r1, c0:c1]):
            return False
    return True


def _mark_cells(grid: np.ndarray, occupied: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color
        occupied[r, c] = True


def _arrow_cells(direction: str, tip: tuple[int, int], body_len: int) -> list[tuple[int, int]]:
    tr, tc = tip
    if direction == "R":
        cells = [(tr, tc), (tr - 1, tc - 1), (tr + 1, tc - 1)]
        cells.extend((tr, c) for c in range(tc - body_len - 1, tc - 1))
        return cells
    if direction == "L":
        cells = [(tr, tc), (tr - 1, tc + 1), (tr + 1, tc + 1)]
        cells.extend((tr, c) for c in range(tc + 2, tc + body_len + 2))
        return cells
    if direction == "D":
        cells = [(tr, tc), (tr - 1, tc - 1), (tr - 1, tc + 1)]
        cells.extend((r, tc) for r in range(tr - body_len - 1, tr - 1))
        return cells
    cells = [(tr, tc), (tr + 1, tc - 1), (tr + 1, tc + 1)]
    cells.extend((r, tc) for r in range(tr + 2, tr + body_len + 2))
    return cells


def _arrow_gap_cell(direction: str, tip: tuple[int, int]) -> tuple[int, int]:
    dr, dc = DIR_VECTORS[direction]
    return tip[0] + dr, tip[1] + dc


def _arrow_candidates(obj: dict, occupied: np.ndarray) -> list[dict]:
    h, w = occupied.shape
    r0, c0, r1, c1 = obj["bbox"]
    candidates: list[dict] = []

    for r in range(r0, r1 + 1):
        tip = (r, c0 - 2)
        max_len = min(MAX_ARROW_BODY, c0 - 3)
        gap = _arrow_gap_cell("R", tip)
        if 0 <= gap[0] < h and 0 <= gap[1] < w and not occupied[gap]:
            for body_len in range(2, max_len + 1):
                cells = _arrow_cells("R", tip, body_len)
                if _cells_clear(cells, occupied):
                    candidates.append({"direction": "R", "tip": tip, "cells": cells})

        tip = (r, c1 + 2)
        max_len = min(MAX_ARROW_BODY, w - c1 - 4)
        gap = _arrow_gap_cell("L", tip)
        if 0 <= gap[0] < h and 0 <= gap[1] < w and not occupied[gap]:
            for body_len in range(2, max_len + 1):
                cells = _arrow_cells("L", tip, body_len)
                if _cells_clear(cells, occupied):
                    candidates.append({"direction": "L", "tip": tip, "cells": cells})

    for c in range(c0, c1 + 1):
        tip = (r0 - 2, c)
        max_len = min(MAX_ARROW_BODY, r0 - 3)
        gap = _arrow_gap_cell("D", tip)
        if 0 <= gap[0] < h and 0 <= gap[1] < w and not occupied[gap]:
            for body_len in range(2, max_len + 1):
                cells = _arrow_cells("D", tip, body_len)
                if _cells_clear(cells, occupied):
                    candidates.append({"direction": "D", "tip": tip, "cells": cells})

        tip = (r1 + 2, c)
        max_len = min(MAX_ARROW_BODY, h - r1 - 4)
        gap = _arrow_gap_cell("U", tip)
        if 0 <= gap[0] < h and 0 <= gap[1] < w and not occupied[gap]:
            for body_len in range(2, max_len + 1):
                cells = _arrow_cells("U", tip, body_len)
                if _cells_clear(cells, occupied):
                    candidates.append({"direction": "U", "tip": tip, "cells": cells})

    return candidates


def _pick_arrow_candidate(obj: dict, occupied: np.ndarray, rng: random.Random) -> dict | None:
    candidates = _arrow_candidates(obj, occupied)
    if not candidates:
        return None
    grouped: dict[tuple[str, tuple[int, int]], list[dict]] = {}
    for cand in candidates:
        key = (cand["direction"], cand["tip"])
        grouped.setdefault(key, []).append(cand)
    group = rng.choice(list(grouped.values()))
    return rng.choice(group)


def _sample_size(rng: random.Random, lo: int = 2, hi: int = 4) -> tuple[int, int]:
    return rng.randint(lo, hi), rng.randint(lo, hi)


def _try_place_bbox(
    objects: list[dict],
    occupied: np.ndarray,
    bbox: tuple[int, int, int, int],
    count: int,
) -> bool:
    cells = _rect_cells(bbox)
    if not _cells_clear(cells, occupied):
        return False
    for r, c in cells:
        occupied[r, c] = True
    objects.append({"bbox": bbox, "area": _rect_area(bbox), "count": count})
    return True


def _build_independent_objects(h: int, w: int, rng: random.Random) -> list[dict] | None:
    count_patterns = [
        [0, 1, 2],
        [0, 0, 1, 2],
        [0, 1, 1, 2],
    ]
    counts = list(rng.choice(count_patterns))
    anchors = [(5, 5), (5, w - 10), (h - 10, 5), (h - 10, w - 10)]
    rng.shuffle(anchors)
    rng.shuffle(counts)

    occupied = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for idx, count in enumerate(counts):
        placed = False
        for _ in range(40):
            hh, ww = _sample_size(rng)
            base_r, base_c = anchors[idx]
            r0 = base_r + rng.randint(0, 2)
            c0 = base_c + rng.randint(0, 2)
            bbox = (r0, c0, r0 + hh - 1, c0 + ww - 1)
            if _try_place_bbox(objects, occupied, bbox, count):
                placed = True
                break
        if not placed:
            return None

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if _bboxes_overlap(_expanded_bbox(objects[i]), _expanded_bbox(objects[j])):
                return None
    return objects


def _place_extra_objects(
    objects: list[dict],
    occupied: np.ndarray,
    counts: list[int],
    h: int,
    w: int,
    rng: random.Random,
) -> bool:
    anchors = [(5, 5), (5, w - 10), (h - 10, 5), (h - 10, w - 10)]
    rng.shuffle(anchors)
    for count in counts:
        placed = False
        for base_r, base_c in anchors:
            for _ in range(20):
                hh, ww = _sample_size(rng)
                r0 = base_r + rng.randint(0, 2)
                c0 = base_c + rng.randint(0, 2)
                bbox = (r0, c0, r0 + hh - 1, c0 + ww - 1)
                before = len(objects)
                if not _try_place_bbox(objects, occupied, bbox, count):
                    continue
                if any(
                    _bboxes_overlap(_expanded_bbox(objects[-1]), _expanded_bbox(objects[k]))
                    for k in range(before)
                ):
                    cells = _rect_cells(bbox)
                    for r, c in cells:
                        occupied[r, c] = False
                    objects.pop()
                    continue
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _build_overlap_objects(h: int, w: int, rng: random.Random, tie: bool) -> list[dict] | None:
    pair_counts = (2, 2) if tie else rng.choice([(2, 1), (3, 1)])
    extra_counts = list(rng.choice([[0], [0, 1]]))
    orientation = rng.choice(["H", "V"])

    occupied = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    if orientation == "H":
        h1, w1 = _sample_size(rng)
        h2, w2 = _sample_size(rng)
        if tie and h1 * w1 == h2 * w2:
            return None
        gap_max = pair_counts[0] + pair_counts[1] - 1
        if gap_max < 2:
            return None
        gap = rng.randint(2, min(3, gap_max))
        total_w = w1 + gap + w2
        if w - total_w - 12 < 0:
            return None
        r0 = rng.randint(7, h - max(h1, h2) - 7)
        c0 = rng.randint(6, w - total_w - 6)
        bbox1 = (r0, c0, r0 + h1 - 1, c0 + w1 - 1)
        c2 = c0 + w1 + gap
        bbox2 = (r0, c2, r0 + h2 - 1, c2 + w2 - 1)
    else:
        h1, w1 = _sample_size(rng)
        h2, w2 = _sample_size(rng)
        if tie and h1 * w1 == h2 * w2:
            return None
        gap_max = pair_counts[0] + pair_counts[1] - 1
        if gap_max < 2:
            return None
        gap = rng.randint(2, min(3, gap_max))
        total_h = h1 + gap + h2
        if h - total_h - 12 < 0:
            return None
        r0 = rng.randint(6, h - total_h - 6)
        c0 = rng.randint(7, w - max(w1, w2) - 7)
        bbox1 = (r0, c0, r0 + h1 - 1, c0 + w1 - 1)
        r2 = r0 + h1 + gap
        bbox2 = (r2, c0, r2 + h2 - 1, c0 + w2 - 1)

    if not _try_place_bbox(objects, occupied, bbox1, pair_counts[0]):
        return None
    if not _try_place_bbox(objects, occupied, bbox2, pair_counts[1]):
        return None
    if not _bboxes_overlap(_expanded_bbox(objects[0]), _expanded_bbox(objects[1])):
        return None
    if tie and objects[0]["area"] == objects[1]["area"]:
        return None

    if not _place_extra_objects(objects, occupied, extra_counts, h, w, rng):
        return None
    return objects


def _assign_colors(objects: list[dict], rng: random.Random) -> list[int] | None:
    total_colors = len(objects) + sum(int(obj["count"]) for obj in objects)
    if total_colors > len(PALETTE):
        return None
    return rng.sample(PALETTE, total_colors)


def _build_grid_from_objects(objects: list[dict], shape: tuple[int, int], rng: random.Random) -> np.ndarray | None:
    h, w = shape
    palette = _assign_colors(objects, rng)
    if palette is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for idx, obj in enumerate(objects):
        obj["color"] = palette[idx]
        _mark_cells(grid, occupied, _rect_cells(obj["bbox"]), obj["color"])

    arrow_color_idx = len(objects)
    order = sorted(range(len(objects)), key=lambda idx: (-int(objects[idx]["count"]), idx))
    for obj_idx in order:
        for _ in range(int(objects[obj_idx]["count"])):
            cand = _pick_arrow_candidate(objects[obj_idx], occupied, rng)
            if cand is None:
                return None
            color = palette[arrow_color_idx]
            arrow_color_idx += 1
            _mark_cells(grid, occupied, cand["cells"], color)

    expected = _paint_objects(objects, shape, BG)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(24, 30)
    w = rng.randint(24, 30)
    scenario = rng.choice(["independent", "overlap_diff", "overlap_tie"])

    if scenario == "independent":
        objects = _build_independent_objects(h, w, rng)
    elif scenario == "overlap_diff":
        objects = _build_overlap_objects(h, w, rng, tie=False)
    else:
        objects = _build_overlap_objects(h, w, rng, tie=True)
    if objects is None:
        return None

    return _build_grid_from_objects(objects, (h, w), rng)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
