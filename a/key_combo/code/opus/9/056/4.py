from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RECT_COLORS = (2, 3, 4, 6)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_TOTAL_RECTS = (3, 4)
GEN_GRID_SIDE = (16, 21)
GEN_OBJECT_H = (7, 11)
GEN_OBJECT_W = (9, 12)
GEN_SMALL_OCCLUDERS = [(2, 2), (1, 4), (2, 3), (1, 5)]
GEN_LARGE_OCCLUDERS = [(4, 4), (4, 5), (5, 5)]
GEN_SURVIVOR_PATTERNS = {
    2: [[(3, 3), (2, 5)], [(2, 5), (3, 4)], [(3, 4), (3, 5)]],
    3: [[(3, 3), (3, 3), (2, 5)], [(2, 5), (3, 4), (3, 4)], [(3, 4), (3, 4), (3, 5)]],
}
GEN_ATTEMPTS = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _find_monochrome_components(grid: np.ndarray, colors: tuple[int, ...]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    wanted = set(map(int, colors))
    seen = np.zeros((H, W), dtype=bool)
    components = []

    for r0 in range(H):
        for c0 in range(W):
            color = int(g[r0, c0])
            if color not in wanted or seen[r0, c0]:
                continue
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            coords = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_from_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _is_filled_rectangle(grid: np.ndarray, component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool(np.all(patch == int(component["color"])))


def _outlier_index(areas: list[int]) -> int:
    vals = np.array(areas, dtype=float)
    median = float(np.median(vals))
    distances = np.abs(vals - median)
    idx = int(np.argmax(distances))
    if np.count_nonzero(np.isclose(distances, distances[idx])) != 1:
        raise AssertionError("expected a unique area outlier")
    return idx


def _complete_blue_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    blue_mask = out == BLUE
    box = _bbox_from_mask(blue_mask)
    if box is None:
        raise AssertionError("expected visible blue pixels")
    _r0, c0, _r1, c1 = box
    source = np.argwhere(blue_mask)
    for r, c in source:
        mc = c0 + c1 - int(c)
        if out[int(r), mc] == BLACK:
            out[int(r), mc] = BLUE
    return out


def _transform_with_rectangle_choice(grid: np.ndarray, rect_index: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    rects = _find_monochrome_components(g, RECT_COLORS)
    if not (3 <= len(rects) <= 4):
        raise AssertionError("expected 3 or 4 rectangle components")
    for rect in rects:
        if not _is_filled_rectangle(g, rect):
            raise AssertionError("expected every non-blue foreground component to be a filled rectangle")
    chosen = rects[int(rect_index)]
    out = g.copy()
    rr, cc = chosen["coords"][:, 0], chosen["coords"][:, 1]
    out[rr, cc] = BLACK
    return _complete_blue_symmetry(out)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the rectangle whose area is the unique outlier, then mirror-complete the blue object.

    Input structure:
    - Black background with one blue object that would be left-right symmetric if unobstructed.
    - Also present are 3 or 4 non-black, non-blue filled rectangles with distinct colors.
    - Exactly one rectangle overlaps the blue object and hides part of it.

    Rule:
    - Identify every rectangle as a monochromatic 4-connected component.
    - Compute the area of each rectangle and select the unique one farthest from the median area.
    - Erase that outlier rectangle to black.
    - Find the visible blue object's bounding-box midline and mirror every remaining blue cell across that vertical axis, filling only black cells.

    Key insight:
    - The occluder is not marked by color or position; only its area is anomalous relative to the other rectangles, and symmetry repair only works after that rectangle is removed.
    """
    g = np.asarray(grid, dtype=int)
    rects = _find_monochrome_components(g, RECT_COLORS)
    if not (3 <= len(rects) <= 4):
        raise AssertionError("expected 3 or 4 rectangle components")
    for rect in rects:
        if not _is_filled_rectangle(g, rect):
            raise AssertionError("expected every non-blue foreground component to be a filled rectangle")
    idx = _outlier_index([rect["area"] for rect in rects])
    return _transform_with_rectangle_choice(g, idx)


def _make_symmetric_object(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    max_inset = max(1, w // 2 - 1)
    insets = np.zeros(h, dtype=int)
    center = int(rng.integers(max(1, h // 2 - 1), min(h - 1, h // 2 + 2)))
    extra_center = 1 if rng.random() < 0.35 and center + 1 < h else 0
    for r in range(center - 1, -1, -1):
        insets[r] = min(max_inset, insets[r + 1] + int(rng.choice([0, 0, 1])))
    for r in range(center + extra_center + 1, h):
        insets[r] = min(max_inset, insets[r - 1] + int(rng.choice([0, 0, 1])))
    if len(set(map(int, insets))) == 1:
        insets[0] = min(max_inset, 1)
        insets[-1] = min(max_inset, 1)

    mask = np.zeros((h, w), dtype=bool)
    for r, inset in enumerate(insets):
        mask[r, int(inset) : w - int(inset)] = True
    return mask


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_survivors(
    out: np.ndarray,
    occupied: np.ndarray,
    dims_list: list[tuple[int, int]],
    colors: list[int],
    rng: np.random.Generator,
) -> list[dict] | None:
    H, W = out.shape
    order = list(range(len(dims_list)))
    rng.shuffle(order)
    order.sort(key=lambda i: dims_list[i][0] * dims_list[i][1], reverse=True)
    placed = [None] * len(dims_list)
    for i in order:
        h, w = dims_list[i]
        positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
        rng.shuffle(positions)
        for r0, c0 in positions:
            if not _can_place(occupied, r0, c0, h, w, pad=1):
                continue
            out[r0 : r0 + h, c0 : c0 + w] = colors[i]
            occupied[r0 : r0 + h, c0 : c0 + w] = True
            placed[i] = {"color": colors[i], "dims": (h, w), "pos": (r0, c0), "area": h * w}
            break
        if placed[i] is None:
            return None
    return placed


def _occluder_candidates(
    object_mask: np.ndarray,
    object_bbox: tuple[int, int, int, int],
    forbidden: np.ndarray,
    rect_h: int,
    rect_w: int,
    side: str,
) -> list[tuple[int, int]]:
    H, W = object_mask.shape
    r0_obj, c0_obj, r1_obj, c1_obj = object_bbox
    want_overlap = max(3, min(6, (rect_h * rect_w) // 3))
    candidates = []
    for r0 in range(max(0, r0_obj - rect_h + 1), min(H - rect_h, r1_obj) + 1):
        for c0 in range(max(0, c0_obj - rect_w + 1), min(W - rect_w, c1_obj) + 1):
            if forbidden[max(0, r0 - 1) : min(H, r0 + rect_h + 1), max(0, c0 - 1) : min(W, c0 + rect_w + 1)].any():
                continue
            overlap = object_mask[r0 : r0 + rect_h, c0 : c0 + rect_w]
            pts = np.argwhere(overlap)
            if int(pts.shape[0]) < want_overlap:
                continue
            ok = True
            visible = object_mask.copy()
            visible[r0 : r0 + rect_h, c0 : c0 + rect_w] &= False
            if _bbox_from_mask(visible) != object_bbox:
                continue
            for dr, dc in pts:
                gr = r0 + int(dr)
                gc = c0 + int(dc)
                mc = c0_obj + c1_obj - gc
                if side == "left" and not (gc < mc):
                    ok = False
                    break
                if side == "right" and not (gc > mc):
                    ok = False
                    break
                if not object_mask[gr, mc]:
                    ok = False
                    break
                if r0 <= gr < r0 + rect_h and c0 <= mc < c0 + rect_w:
                    ok = False
                    break
            if ok:
                candidates.append((r0, c0))
    return candidates


def _scene_from_input(grid: np.ndarray) -> tuple[list[dict], int, np.ndarray]:
    rects = _find_monochrome_components(grid, RECT_COLORS)
    idx = _outlier_index([rect["area"] for rect in rects])
    erased = np.array(grid, dtype=int, copy=True)
    rr, cc = rects[idx]["coords"][:, 0], rects[idx]["coords"][:, 1]
    erased[rr, cc] = BLACK
    return rects, idx, erased


def _validate_scene(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        rects, idx, erased = _scene_from_input(inp)
        if len(rects) not in (3, 4):
            return False
        if any(not _is_filled_rectangle(inp, rect) for rect in rects):
            return False
        areas = [rect["area"] for rect in rects]
        survivors = [areas[i] for i in range(len(areas)) if i != idx]
        if max(survivors) > 1.3 * min(survivors):
            return False
        if not (areas[idx] >= 1.5 * max(survivors) or areas[idx] <= 0.5 * min(survivors)):
            return False
        repaired = _complete_blue_symmetry(erased)
        if np.array_equal(repaired, erased):
            return False
        if not np.array_equal(repaired, expected):
            return False
        if not np.array_equal(main(inp), expected):
            return False
        wrong = int(np.argmin(areas)) if idx == int(np.argmax(areas)) else int(np.argmax(areas))
        if wrong == idx or np.array_equal(_transform_with_rectangle_choice(inp, wrong), expected):
            return False
        if np.array_equal(_complete_blue_symmetry(inp), expected):
            return False
        blue_box_erased = _bbox_from_mask(erased == BLUE)
        blue_box_out = _bbox_from_mask(expected == BLUE)
        if blue_box_erased != blue_box_out:
            return False
        return not np.array_equal(inp, expected)
    except AssertionError:
        return False


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    total_rects = int(rng.integers(GEN_TOTAL_RECTS[0], GEN_TOTAL_RECTS[1] + 1))
    survivor_count = total_rects - 1
    survivor_dims = [tuple(x) for x in GEN_SURVIVOR_PATTERNS[survivor_count][int(rng.integers(len(GEN_SURVIVOR_PATTERNS[survivor_count])))]]
    survivor_areas = [h * w for h, w in survivor_dims]
    large_outlier = bool(rng.integers(0, 2))
    occluder_pool = GEN_LARGE_OCCLUDERS if large_outlier else GEN_SMALL_OCCLUDERS
    valid_occluders = []
    for h, w in occluder_pool:
        area = h * w
        if large_outlier and area >= 1.5 * max(survivor_areas):
            valid_occluders.append((h, w))
        if (not large_outlier) and area <= 0.5 * min(survivor_areas):
            valid_occluders.append((h, w))
    if not valid_occluders:
        return None
    occ_h, occ_w = valid_occluders[int(rng.integers(len(valid_occluders)))]

    H = int(rng.integers(GEN_GRID_SIDE[0], GEN_GRID_SIDE[1]))
    W = int(rng.integers(GEN_GRID_SIDE[0], GEN_GRID_SIDE[1]))
    obj_h_min = max(GEN_OBJECT_H[0], occ_h + 3)
    obj_w_min = max(GEN_OBJECT_W[0], min(W - 4, 9))
    if obj_h_min >= GEN_OBJECT_H[1] or obj_w_min >= GEN_OBJECT_W[1]:
        return None
    obj_h = int(rng.integers(obj_h_min, min(GEN_OBJECT_H[1], H - 3)))
    obj_w = int(rng.integers(obj_w_min, min(GEN_OBJECT_W[1], W - 3)))
    mask_local = _make_symmetric_object(obj_h, obj_w, rng)

    row_choices = list(range(2, H - obj_h - 1))
    col_choices = list(range(2, W - obj_w - 1))
    if not row_choices or not col_choices:
        return None
    obj_r0 = int(rng.choice(row_choices))
    obj_c0 = int(rng.choice(col_choices))
    obj_r1, obj_c1 = obj_r0 + obj_h - 1, obj_c0 + obj_w - 1

    expected = np.zeros((H, W), dtype=int)
    object_mask = np.zeros((H, W), dtype=bool)
    object_mask[obj_r0 : obj_r0 + obj_h, obj_c0 : obj_c0 + obj_w] = mask_local
    expected[object_mask] = BLUE

    occupied = object_mask.copy()
    rect_colors = list(rng.permutation(np.array(RECT_COLORS, dtype=int))[:total_rects])
    occluder_color = rect_colors[0]
    survivor_colors = rect_colors[1:]
    survivors = _place_survivors(expected, occupied, survivor_dims, survivor_colors, rng)
    if survivors is None:
        return None

    forbidden = occupied & ~object_mask
    sides = ["left", "right"]
    rng.shuffle(sides)
    candidates = []
    for side in sides:
        candidates.extend(_occluder_candidates(object_mask, (obj_r0, obj_c0, obj_r1, obj_c1), forbidden, occ_h, occ_w, side))
    if not candidates:
        return None
    occ_r0, occ_c0 = candidates[int(rng.integers(len(candidates)))]

    inp = expected.copy()
    inp[occ_r0 : occ_r0 + occ_h, occ_c0 : occ_c0 + occ_w] = occluder_color
    return inp if _validate_scene(inp, expected) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
