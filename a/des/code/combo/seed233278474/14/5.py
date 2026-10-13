from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
MAROON = 9
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8)

GEN_RETRIES = 400
PAIR_RETRIES = 120
SINGLE_RETRIES = 80
GRID_MIN = 12
GRID_MAX = 28
RECT_H_MAX = 5
RECT_W_MAX = 6
PADDING = 1


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.array(list(coords), dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_objects(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    mask = (grid != BG) & (grid != MAROON)
    comps = _components_mask(mask)
    comp_index = np.full(grid.shape, -1, dtype=int)
    objects: list[dict] = []
    for idx, pts in enumerate(comps):
        for r, c in pts:
            comp_index[r, c] = idx
        r0, c0, r1, c1 = _bbox(pts)
        color = int(grid[pts[0][0], pts[0][1]])
        objects.append(
            {
                "id": idx,
                "coords": pts,
                "bbox": (r0, c0, r1, c1),
                "area": len(pts),
                "color": color,
                "top_left": (r0, c0),
            }
        )
    return objects, comp_index


def _scan_object(
    grid: np.ndarray,
    comp_index: np.ndarray,
    start: tuple[int, int],
    dr: int,
    dc: int,
) -> tuple[int, int] | None:
    h, w = grid.shape
    r, c = start
    gap = 0
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        value = int(grid[r, c])
        if value == BG:
            gap += 1
            r += dr
            c += dc
            continue
        if value == MAROON:
            return None
        return int(comp_index[r, c]), gap
    return None


def _resolve_marker(
    grid: np.ndarray,
    objects: list[dict],
    comp_index: np.ndarray,
    marker: tuple[int, int],
    require_unique_axis: bool,
) -> tuple[str, int, int] | None:
    r, c = marker
    left = _scan_object(grid, comp_index, marker, 0, -1)
    right = _scan_object(grid, comp_index, marker, 0, 1)
    up = _scan_object(grid, comp_index, marker, -1, 0)
    down = _scan_object(grid, comp_index, marker, 1, 0)

    candidates: list[tuple[int, str, int, int]] = []
    if left is not None and right is not None and left[0] != right[0]:
        candidates.append((left[1] + right[1], "h", left[0], right[0]))
    if up is not None and down is not None and up[0] != down[0]:
        candidates.append((up[1] + down[1], "v", up[0], down[0]))

    if not candidates:
        return None
    if require_unique_axis and len(candidates) != 1:
        return None

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    _, axis, a, b = candidates[0]

    # Prefer canonical ordering for downstream tie breaking.
    obj_a = objects[a]
    obj_b = objects[b]
    if obj_b["top_left"] < obj_a["top_left"]:
        a, b = b, a
    return axis, a, b


def _merge_color(obj_a: dict, obj_b: dict) -> int:
    key_a = (int(obj_a["area"]), int(obj_a["top_left"][0]), int(obj_a["top_left"][1]))
    key_b = (int(obj_b["area"]), int(obj_b["top_left"][0]), int(obj_b["top_left"][1]))
    return int(obj_a["color"] if key_a <= key_b else obj_b["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge rectangle pairs indicated by maroon bridge pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects, comp_index = _extract_objects(g)
    out = g.copy()

    markers = [tuple(map(int, pt)) for pt in np.argwhere(g == MAROON)]
    for marker in sorted(markers):
        resolved = _resolve_marker(g, objects, comp_index, marker, require_unique_axis=False)
        if resolved is None:
            continue
        _axis, a, b = resolved
        obj_a = objects[a]
        obj_b = objects[b]
        fill = _merge_color(obj_a, obj_b)
        r0 = min(int(obj_a["bbox"][0]), int(obj_b["bbox"][0]))
        c0 = min(int(obj_a["bbox"][1]), int(obj_b["bbox"][1]))
        r1 = max(int(obj_a["bbox"][2]), int(obj_b["bbox"][2]))
        c1 = max(int(obj_a["bbox"][3]), int(obj_b["bbox"][3]))
        out[r0 : r1 + 1, c0 : c1 + 1] = fill
    return out


def _sample_dims(rng: random.Random, area: int | None = None) -> tuple[int, int]:
    if area is None:
        return rng.randint(1, RECT_H_MAX), rng.randint(1, RECT_W_MAX)

    choices = [
        (h, w)
        for h in range(1, RECT_H_MAX + 1)
        for w in range(1, RECT_W_MAX + 1)
        if h * w == area
    ]
    return rng.choice(choices)


def _sample_pair_patch(rng: random.Random, orientation: str) -> tuple[list[tuple[int, int, int]], int, int]:
    tie = rng.random() < 0.5
    same_top_tie = tie and orientation == "h" and rng.random() < 0.5
    color_a, color_b = rng.sample(OBJECT_COLORS, 2)

    if tie:
        areas = sorted({h * w for h in range(1, RECT_H_MAX + 1) for w in range(1, RECT_W_MAX + 1)})
        area = rng.choice(areas)
        h1, w1 = _sample_dims(rng, area)
        h2, w2 = _sample_dims(rng, area)
    else:
        h1, w1 = _sample_dims(rng)
        h2, w2 = _sample_dims(rng)
        while h1 * w1 == h2 * w2:
            h2, w2 = _sample_dims(rng)

    cells: list[tuple[int, int, int]] = []
    if orientation == "h":
        if same_top_tie:
            top1 = 0
            top2 = 0
            bridge_row = rng.randrange(min(h1, h2))
        else:
            anchor1 = rng.randrange(h1)
            anchor2 = rng.randrange(h2)
            bridge_row = max(anchor1, anchor2)
            top1 = bridge_row - anchor1
            top2 = bridge_row - anchor2

        gap_left = rng.randint(0, 3)
        gap_right = rng.randint(0, 3)
        left1 = 0
        marker_c = left1 + w1 + gap_left
        left2 = marker_c + gap_right + 1
        height = max(top1 + h1, top2 + h2)
        width = left2 + w2

        for r in range(top1, top1 + h1):
            for c in range(left1, left1 + w1):
                cells.append((r, c, color_a))
        for r in range(top2, top2 + h2):
            for c in range(left2, left2 + w2):
                cells.append((r, c, color_b))
        cells.append((bridge_row, marker_c, MAROON))
        return cells, height, width

    anchor1 = rng.randrange(w1)
    anchor2 = rng.randrange(w2)
    bridge_c = max(anchor1, anchor2)
    left1 = bridge_c - anchor1
    left2 = bridge_c - anchor2
    gap_up = rng.randint(0, 3)
    gap_down = rng.randint(0, 3)
    top1 = 0
    marker_r = top1 + h1 + gap_up
    top2 = marker_r + gap_down + 1
    height = top2 + h2
    width = max(left1 + w1, left2 + w2)

    for r in range(top1, top1 + h1):
        for c in range(left1, left1 + w1):
            cells.append((r, c, color_a))
    for r in range(top2, top2 + h2):
        for c in range(left2, left2 + w2):
            cells.append((r, c, color_b))
    cells.append((marker_r, bridge_c, MAROON))
    return cells, height, width


def _sample_single_patch(rng: random.Random) -> tuple[list[tuple[int, int, int]], int, int]:
    h, w = _sample_dims(rng)
    color = rng.choice(OBJECT_COLORS)
    cells = [(r, c, color) for r in range(h) for c in range(w)]
    return cells, h, w


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_patch(
    grid: np.ndarray,
    blocked: np.ndarray,
    cells: list[tuple[int, int, int]],
    patch_h: int,
    patch_w: int,
    rng: random.Random,
    retries: int,
) -> bool:
    h, w = grid.shape
    for _ in range(retries):
        top = rng.randint(0, h - patch_h)
        left = rng.randint(0, w - patch_w)
        r0 = max(0, top - PADDING)
        c0 = max(0, left - PADDING)
        r1 = min(h, top + patch_h + PADDING)
        c1 = min(w, left + patch_w + PADDING)
        if blocked[r0:r1, c0:c1].any():
            continue
        for dr, dc, color in cells:
            grid[top + dr, left + dc] = color
        blocked[r0:r1, c0:c1] = True
        return True
    return False


def _validate_grid(grid: np.ndarray) -> bool:
    if grid.ndim != 2:
        return False
    objects, comp_index = _extract_objects(grid)
    if not objects:
        return False

    # All non-marker objects must be solid rectangles.
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        color = int(obj["color"])
        if int(obj["area"]) != (r1 - r0 + 1) * (c1 - c0 + 1):
            return False
        if not bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)):
            return False

    markers = [tuple(map(int, pt)) for pt in np.argwhere(grid == MAROON)]
    if not markers:
        return False
    if len(_components_mask(grid == MAROON)) != len(markers):
        return False

    used: set[int] = set()
    merged_boxes: list[tuple[int, int, int, int]] = []
    for marker in sorted(markers):
        resolved = _resolve_marker(grid, objects, comp_index, marker, require_unique_axis=True)
        if resolved is None:
            return False
        _axis, a, b = resolved
        if a in used or b in used:
            return False
        used.add(a)
        used.add(b)
        obj_a = objects[a]
        obj_b = objects[b]
        box = (
            min(int(obj_a["bbox"][0]), int(obj_b["bbox"][0])),
            min(int(obj_a["bbox"][1]), int(obj_b["bbox"][1])),
            max(int(obj_a["bbox"][2]), int(obj_b["bbox"][2])),
            max(int(obj_a["bbox"][3]), int(obj_b["bbox"][3])),
        )
        for other in objects:
            oid = int(other["id"])
            if oid in (a, b):
                continue
            or0, oc0, or1, oc1 = other["bbox"]
            if _rects_overlap(box, (or0, oc0, or1, oc1)):
                return False
        for other_marker in markers:
            if other_marker == marker:
                continue
            if box[0] <= other_marker[0] <= box[2] and box[1] <= other_marker[1] <= box[3]:
                return False
        for prev in merged_boxes:
            if _rects_overlap(box, prev):
                return False
        merged_boxes.append(box)
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice((1, 2, 3))
    single_count = rng.choice((0, 1, 2, 3))
    cluster_count = pair_count + single_count

    min_side = max(GRID_MIN, 8 + 3 * cluster_count)
    max_side = min(GRID_MAX, min_side + 8)
    if min_side > max_side:
        return None
    h = rng.randint(min_side, max_side)
    w = rng.randint(min_side, max_side)

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    orientations = [rng.choice(("h", "v")) for _ in range(pair_count)]
    if pair_count >= 2 and len(set(orientations)) == 1:
        orientations[rng.randrange(pair_count)] = "v" if orientations[0] == "h" else "h"

    for orientation in orientations:
        cells, patch_h, patch_w = _sample_pair_patch(rng, orientation)
        if patch_h > h or patch_w > w:
            return None
        if not _place_patch(grid, blocked, cells, patch_h, patch_w, rng, PAIR_RETRIES):
            return None

    for _ in range(single_count):
        cells, patch_h, patch_w = _sample_single_patch(rng)
        if patch_h > h or patch_w > w:
            return None
        if not _place_patch(grid, blocked, cells, patch_h, patch_w, rng, SINGLE_RETRIES):
            return None

    return grid if _validate_grid(grid) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
