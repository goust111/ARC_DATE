from __future__ import annotations

import hashlib
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
PINK = 6
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _binary_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = int(cr + dr), int(cc + dc)
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_monochrome_components(grid: np.ndarray, ignored_colors: set[int]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(set(map(int, np.unique(g))) - {int(c) for c in ignored_colors})
    comps: list[dict] = []
    for color in colors:
        for coords in _binary_components(g == int(color)):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _component_patch(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    rows = comp["coords"][:, 0] - int(r0)
    cols = comp["coords"][:, 1] - int(c0)
    patch[rows, cols] = int(comp["color"])
    return patch


def _stamp_on_black(canvas: np.ndarray, patch: np.ndarray, anchor: tuple[int, int], color: int) -> None:
    h, w = patch.shape
    center_r, center_c = int(h // 2), int(w // 2)
    top = int(anchor[0] - center_r)
    left = int(anchor[1] - center_c)
    for pr, pc in np.argwhere(patch != BLACK):
        rr = int(top + pr)
        cc = int(left + pc)
        if _in_bounds(canvas.shape, rr, cc) and int(canvas[rr, cc]) == BLACK:
            canvas[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter tiny noise, extract the largest component as a cropped stamp template, and place it at magenta anchors.

    Input structure:
    - A black-background grid contains one large blue foreground object, several tiny red/green noise components,
      and 1-3 magenta singleton anchors.
    - Foreground components are monochrome and use 4-connectivity; anchors are not part of the template object.

    Rule:
    - Find all monochrome 4-connected foreground components except the magenta anchors.
    - Treat every area-1 or area-2 component as noise, erase those components, and select the unique largest remaining
      component as the signal object.
    - Crop the signal object's tight bbox to form a template, preserving any internal black holes; erase the original
      signal object; then stamp the template in blue at every magenta anchor so the template bbox center (rounded
      down/right for even sizes) aligns with the anchor, filling only black cells and keeping anchors unchanged.

    Key insight:
    - The template is not given separately: it must be recovered by isolating the unique largest monochrome component
      before cropping, otherwise nearby or internal noise corrupts the stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _find_monochrome_components(g, ignored_colors={BLACK, PINK})
    assert comps, "expected at least one non-anchor foreground component"

    max_area = max(int(comp["area"]) for comp in comps)
    signal_candidates = [comp for comp in comps if int(comp["area"]) == int(max_area)]
    assert len(signal_candidates) == 1 and int(max_area) >= 3, "expected a unique large signal component"
    signal = signal_candidates[0]

    out = g.copy()
    for comp in comps:
        if int(comp["area"]) <= 2 or comp is signal:
            out[comp["coords"][:, 0], comp["coords"][:, 1]] = BLACK

    template = _component_patch(signal)
    anchors = [tuple(map(int, pos)) for pos in np.argwhere(g == PINK)]
    for anchor in anchors:
        _stamp_on_black(out, template, anchor, int(signal["color"]))
    return out


_TEMPLATE_ROWS = [
    ["###", "###", "##."],
    ["####", ".###", ".##."],
    ["####", "#..#", "####"],
    ["###.", "#.#.", "####", ".##."],
    ["#####", "#...#", "#####"],
    ["####", "#..#", "#.##", "####"],
    ["###.", "####", ".###", ".#.."],
    ["#####", "####.", "..###"],
    ["###", "#.#", "###", ".#."],
    ["####.", "##.##", ".####"],
]

_SEEN_OUTPUTS: set[str] = set()
_GEN_CALLS = 0
_MAX_GEN_ATTEMPTS = 500


def _rows_to_mask(rows: list[str]) -> np.ndarray:
    mask = np.array([[1 if ch == "#" else 0 for ch in row] for row in rows], dtype=int)
    comps = _binary_components(mask != 0)
    assert len(comps) == 1, "template must be 4-connected"
    area = int(mask.sum())
    assert 8 <= area <= 15, f"template area out of range: {area}"
    return mask


_TEMPLATE_LIBRARY = [_rows_to_mask(rows) for rows in _TEMPLATE_ROWS]


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _hole_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    bg = mask == 0
    holes: list[tuple[int, int]] = []
    for comp in _binary_components(bg):
        if any(int(r) in (0, mask.shape[0] - 1) or int(c) in (0, mask.shape[1] - 1) for r, c in comp):
            continue
        holes.extend((int(r), int(c)) for r, c in comp)
    return holes


def _template_features(mask: np.ndarray) -> dict:
    holes = _hole_cells(mask)
    h, w = mask.shape
    return {
        "mask": mask,
        "area": int(mask.sum()),
        "h": int(h),
        "w": int(w),
        "hole_cells": holes,
        "has_hole": bool(holes),
        "even_dim": bool(h % 2 == 0 or w % 2 == 0),
        "center_filled": bool(mask[h // 2, w // 2] != 0),
    }


def _feature_request(call_idx: int) -> dict:
    cycle = [
        {"force_hole": True, "force_hole_noise": True},
        {"force_anchor_count": 3, "force_even_dim": True},
        {"force_anchor_count": 1, "force_center_filled": False},
    ]
    req = cycle[int(call_idx % len(cycle))].copy()
    req.setdefault("force_hole", random.random() < 0.45)
    req.setdefault("force_even_dim", random.random() < 0.5)
    req.setdefault("force_center_filled", random.choice([True, False]))
    if "force_anchor_count" not in req:
        req["force_anchor_count"] = random.randint(1, 3)
    return req


def _choose_template(req: dict) -> dict | None:
    candidates: list[dict] = []
    for base in _TEMPLATE_LIBRARY:
        for _ in range(8):
            feat = _template_features(_transform_mask(base))
            if req.get("force_hole") and not feat["has_hole"]:
                continue
            if req.get("force_even_dim") and not feat["even_dim"]:
                continue
            if bool(feat["center_filled"]) != bool(req.get("force_center_filled", feat["center_filled"])):
                continue
            candidates.append(feat)
    if not candidates:
        return None
    return random.choice(candidates)


def _anchor_bbox(anchor: tuple[int, int], patch_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = map(int, patch_shape)
    center_r, center_c = int(h // 2), int(w // 2)
    top = int(anchor[0] - center_r)
    left = int(anchor[1] - center_c)
    return int(top), int(left), int(top + h - 1), int(left + w - 1)


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    pad = int(pad)
    return not (
        int(a[2]) + pad < int(b[0])
        or int(b[2]) + pad < int(a[0])
        or int(a[3]) + pad < int(b[1])
        or int(b[3]) + pad < int(a[1])
    )


def _stamp_cells(anchor: tuple[int, int], mask: np.ndarray) -> set[tuple[int, int]]:
    h, w = mask.shape
    top = int(anchor[0] - h // 2)
    left = int(anchor[1] - w // 2)
    return {
        (int(top + r), int(left + c))
        for r, c in np.argwhere(mask != 0)
    }


def _place_anchors(h: int, w: int, mask: np.ndarray, n_anchors: int) -> tuple[list[tuple[int, int]], list[tuple[int, int, int, int]]] | None:
    center_r, center_c = int(mask.shape[0] // 2), int(mask.shape[1] // 2)
    candidates = [
        (int(r), int(c))
        for r in range(center_r, int(h - mask.shape[0] + center_r + 1))
        for c in range(center_c, int(w - mask.shape[1] + center_c + 1))
    ]
    random.shuffle(candidates)
    for pad in (1, 0):
        anchors: list[tuple[int, int]] = []
        bboxes: list[tuple[int, int, int, int]] = []
        occupied: set[tuple[int, int]] = set()
        for cand in candidates:
            bbox = _anchor_bbox(cand, mask.shape)
            cells = _stamp_cells(cand, mask) | {cand}
            if any(_boxes_intersect(bbox, prev, pad=pad) for prev in bboxes):
                continue
            if occupied & cells:
                continue
            anchors.append(cand)
            bboxes.append(bbox)
            occupied |= cells
            if len(anchors) == int(n_anchors):
                return anchors, bboxes
    return None


def _build_expected(shape: tuple[int, int], mask: np.ndarray, anchors: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c in anchors:
        out[int(r), int(c)] = PINK
    patch = np.where(mask != 0, BLUE, BLACK)
    for anchor in anchors:
        _stamp_on_black(out, patch, anchor, BLUE)
    return out


def _signal_patch(mask: np.ndarray) -> np.ndarray:
    return np.where(mask != 0, BLUE, BLACK)


def _can_place_signal(
    top: int,
    left: int,
    mask: np.ndarray,
    anchors: list[tuple[int, int]],
    stamp_bboxes: list[tuple[int, int, int, int]],
    grid_shape: tuple[int, int],
) -> bool:
    h, w = mask.shape
    bbox = (int(top), int(left), int(top + h - 1), int(left + w - 1))
    if any(_boxes_intersect(bbox, stamp_bbox, pad=1) for stamp_bbox in stamp_bboxes):
        return False
    cells = {(int(top + r), int(left + c)) for r, c in np.argwhere(mask != 0)}
    for ar, ac in anchors:
        if (int(ar), int(ac)) in cells:
            return False
        for cr, cc in cells:
            if abs(int(ar) - int(cr)) + abs(int(ac) - int(cc)) == 1:
                return False
    return all(_in_bounds(grid_shape, r, c) for r, c in cells)


def _place_signal(grid: np.ndarray, mask: np.ndarray, anchors: list[tuple[int, int]], stamp_bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = mask.shape
    candidates = [
        (int(top), int(left))
        for top in range(int(h - ph + 1))
        for left in range(int(w - pw + 1))
        if _can_place_signal(top, left, mask, anchors, stamp_bboxes, grid.shape)
    ]
    if not candidates:
        return None
    top, left = random.choice(candidates)
    patch = _signal_patch(mask)
    grid[top : top + ph, left : left + pw][patch != 0] = BLUE
    return int(top), int(left)


def _point_in_boxes(point: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> bool:
    r, c = map(int, point)
    return any(int(b[0]) <= r <= int(b[2]) and int(b[1]) <= c <= int(b[3]) for b in boxes)


def _noise_candidates(size: int, shape: tuple[int, int]) -> list[list[tuple[int, int]]]:
    h, w = map(int, shape)
    out: list[list[tuple[int, int]]] = []
    if int(size) == 1:
        for r in range(h):
            for c in range(w):
                out.append([(int(r), int(c))])
        return out
    for r in range(h):
        for c in range(w):
            if c + 1 < w:
                out.append([(int(r), int(c)), (int(r), int(c + 1))])
            if r + 1 < h:
                out.append([(int(r), int(c)), (int(r + 1), int(c))])
    return out


def _valid_noise_cells(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    color: int,
    *,
    allow_touch_diff: bool,
    require_touch_diff: bool,
    avoid_boxes: list[tuple[int, int, int, int]],
) -> bool:
    cell_set = {(int(r), int(c)) for r, c in cells}
    touched_diff = False
    for r, c in cell_set:
        if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != BLACK:
            return False
        if _point_in_boxes((int(r), int(c)), avoid_boxes):
            return False
        for dr, dc in DIRS4:
            nr, nc = int(r + dr), int(c + dc)
            if (nr, nc) in cell_set:
                continue
            if not _in_bounds(grid.shape, nr, nc):
                continue
            neigh = int(grid[nr, nc])
            if neigh == PINK:
                return False
            if neigh == color:
                return False
            if neigh != BLACK:
                if not allow_touch_diff:
                    return False
                touched_diff = True
    return (not require_touch_diff) or touched_diff


def _place_noise_component(
    grid: np.ndarray,
    size: int,
    color: int,
    *,
    allow_touch_diff: bool,
    require_touch_diff: bool,
    avoid_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]] | None:
    candidates = _noise_candidates(size, grid.shape)
    random.shuffle(candidates)
    for cells in candidates:
        if _valid_noise_cells(
            grid,
            cells,
            color,
            allow_touch_diff=allow_touch_diff,
            require_touch_diff=require_touch_diff,
            avoid_boxes=avoid_boxes,
        ):
            for r, c in cells:
                grid[int(r), int(c)] = int(color)
            return [(int(r), int(c)) for r, c in cells]
    return None


def _noise_size_partition(total: int) -> list[int] | None:
    candidates: list[list[int]] = []
    for n_comps in range(2, 5):
        for bits in range(1 << n_comps):
            sizes = [1 + ((bits >> i) & 1) for i in range(n_comps)]
            if sum(sizes) == int(total):
                candidates.append(sizes)
    if not candidates:
        return None
    return random.choice(candidates)


def _stamp_from_boolean_mask(shape: tuple[int, int], bool_mask: np.ndarray, anchors: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c in anchors:
        out[int(r), int(c)] = PINK
    patch = np.where(bool_mask, BLUE, BLACK)
    for anchor in anchors:
        _stamp_on_black(out, patch, anchor, BLUE)
    return out


def _wrong_raw_bbox_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    comps = _find_monochrome_components(g, ignored_colors={BLACK, PINK})
    signal = max(comps, key=lambda comp: int(comp["area"]))
    r0, c0, r1, c1 = map(int, signal["bbox"])
    raw_mask = g[r0 : r1 + 1, c0 : c1 + 1] != BLACK
    anchors = [tuple(map(int, pos)) for pos in np.argwhere(g == PINK)]
    return _stamp_from_boolean_mask(g.shape, raw_mask, anchors)


def _wrong_binary_union_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    comps = _binary_components((g != BLACK) & (g != PINK))
    signal_coords = max(comps, key=lambda coords: int(coords.shape[0]))
    r0, c0, r1, c1 = _bbox(signal_coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in signal_coords:
        mask[int(r) - r0, int(c) - c0] = True
    anchors = [tuple(map(int, pos)) for pos in np.argwhere(g == PINK)]
    return _stamp_from_boolean_mask(g.shape, mask, anchors)


def _output_signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _validate_example(
    grid: np.ndarray,
    expected: np.ndarray,
    mask: np.ndarray,
    signal_top_left: tuple[int, int],
    anchors: list[tuple[int, int]],
    *,
    hole_noise_used: bool,
    adjacent_noise_used: bool,
) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    comps = _find_monochrome_components(grid, ignored_colors={BLACK, PINK})
    if not comps:
        return False
    max_area = max(int(comp["area"]) for comp in comps)
    largest = [comp for comp in comps if int(comp["area"]) == int(max_area)]
    if len(largest) != 1 or int(max_area) != int(mask.sum()) or int(largest[0]["color"]) != BLUE:
        return False

    signal_patch = _component_patch(largest[0])
    if not np.array_equal(signal_patch, _signal_patch(mask)):
        return False

    noise_comps = [comp for comp in comps if comp is not largest[0]]
    noise_pixels = sum(int(comp["area"]) for comp in noise_comps)
    if not (3 <= noise_pixels <= 6):
        return False
    if any(int(comp["area"]) > 2 for comp in noise_comps):
        return False
    if any(int(comp["color"]) not in (RED, GREEN) for comp in noise_comps):
        return False

    anchor_set = {(int(r), int(c)) for r, c in anchors}
    fg_non_anchor = {(int(r), int(c)) for r, c in np.argwhere((grid != BLACK) & (grid != PINK))}
    for ar, ac in anchor_set:
        for dr, dc in DIRS4:
            if (int(ar + dr), int(ac + dc)) in fg_non_anchor:
                return False

    if hole_noise_used:
        top, left = map(int, signal_top_left)
        hole_positions = {(int(top + r), int(left + c)) for r, c in _hole_cells(mask)}
        if not any(pos in hole_positions for comp in noise_comps for pos in map(tuple, comp["coords"].tolist())):
            return False
        if np.array_equal(_wrong_raw_bbox_output(grid), expected):
            return False

    if adjacent_noise_used and np.array_equal(_wrong_binary_union_output(grid), expected):
        return False

    if _output_signature(expected) in _SEEN_OUTPUTS:
        return False
    return True


def _gen_attempt(call_idx: int) -> np.ndarray | None:
    req = _feature_request(call_idx)
    chosen = _choose_template(req)
    if chosen is None:
        return None

    mask = np.asarray(chosen["mask"], dtype=int)
    n_anchors = int(req["force_anchor_count"])
    min_side = max(18, int(mask.shape[0] + 8), int(mask.shape[1] + 8))
    max_side = 24
    if min_side > max_side:
        return None
    h = random.randint(min_side, max_side)
    w = random.randint(min_side, max_side)

    anchor_data = _place_anchors(h, w, mask, n_anchors)
    if anchor_data is None:
        return None
    anchors, stamp_bboxes = anchor_data
    expected = _build_expected((h, w), mask, anchors)

    grid = np.zeros((h, w), dtype=int)
    for r, c in anchors:
        grid[int(r), int(c)] = PINK

    signal_pos = _place_signal(grid, mask, anchors, stamp_bboxes)
    if signal_pos is None:
        return None

    hole_noise_used = bool(req.get("force_hole_noise", False) and chosen["has_hole"])
    if not hole_noise_used:
        hole_noise_used = bool(chosen["has_hole"] and random.random() < 0.45)

    total_noise = random.randint(3, 6)
    used_noise_pixels = 0
    avoid_boxes = stamp_bboxes

    if hole_noise_used:
        hole_cells = chosen["hole_cells"]
        if not hole_cells:
            return None
        hr, hc = random.choice(hole_cells)
        rr = int(signal_pos[0] + hr)
        cc = int(signal_pos[1] + hc)
        if int(grid[rr, cc]) != BLACK:
            return None
        grid[rr, cc] = random.choice([RED, GREEN])
        used_noise_pixels += 1

    remaining = int(total_noise - used_noise_pixels)
    sizes = _noise_size_partition(remaining)
    if sizes is None:
        return None

    adjacent_noise_used = not hole_noise_used or random.random() < 0.55
    touch_index = 0 if adjacent_noise_used and sizes else -1
    random.shuffle(sizes)
    if touch_index >= 0 and sizes:
        sizes.insert(0, sizes.pop())

    for i, size in enumerate(sizes):
        placed = _place_noise_component(
            grid,
            int(size),
            random.choice([RED, GREEN]),
            allow_touch_diff=bool(i == touch_index),
            require_touch_diff=bool(i == touch_index),
            avoid_boxes=avoid_boxes,
        )
        if placed is None:
            return None

    if not _validate_example(
        grid,
        expected,
        mask,
        signal_pos,
        anchors,
        hole_noise_used=hole_noise_used,
        adjacent_noise_used=adjacent_noise_used,
    ):
        return None

    _SEEN_OUTPUTS.add(_output_signature(expected))
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALLS
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(_GEN_CALLS)
        if grid is not None:
            _GEN_CALLS += 1
            return grid
    raise RuntimeError("failed to generate a valid input")
