# Combo variant of 3a25b0d8 (index 16)
# combo_spec: D5/0+D6/1
# Name: 完整画布+源留痕迹
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior and trace
#    - Original task: detect `src` (outline + colored cavity fills) and `dst`
#      (same outline family + empty holes). Match source subsprites to destination
#      holes by centroid row/column rank, flood-fill each dst hole with the matched
#      source color, then tight-crop so `src` disappears from the final output.
#    - Fused task:
#      1. the color transfer step stays the same;
#      2. output keeps the full original canvas instead of cropping;
#      3. the colored cells inside `src` are not left untouched and not erased to bg:
#         every source subsprite cell is rewritten to one uniform marker color
#         (fixed grey here) to show that its color was "taken away".
#    - Small trace:
#      if a red L-shaped subsprite in src matches the upper-right dst hole, old output
#      contained only the cropped dst with that hole turned red. New output keeps the
#      whole canvas, the dst hole still turns red, and the old red L-shape inside src
#      becomes a grey L-shape at the same absolute coordinates.
#
# b) Complexity calibration
#    - Only two post-match changes are introduced: framing and source afterimage.
#      The correspondence logic is still one stable mechanism: centroid-rank matching.
#    - The marker color is a single reserved color for the whole sample, not a
#      per-cavity recoloring rule. That keeps the visual difference obvious without
#      exploding the color reasoning burden.
#    - We keep 4-9 cavities and 3-7 fill colors, so repeated colors still support the
#      original positional inference while the grey traces make the transfer observable.
#
# c) Hidden assumptions in the original main() that break here
#    - `return tight_crop(dest)` depended on "src can disappear safely". This is no
#      longer valid because the output must preserve the input frame and also show the
#      modified src.
#    - The old solver only needed source centroids/colors. The fused rule also needs
#      the absolute source fill pixels so it can rewrite them to the marker color.
#    - The old generator could allow bbox overlap as long as current non-bg pixels did
#      not collide. That becomes invalid here: filling dst holes can activate cells that
#      were previously bg, so overlapping bboxes could create output collisions.
#
# d) Placement legality
#    - Generator now places src and dst as disjoint rectangles on the canvas, using
#      solid occupancy blocks rather than transparent sprite masks during placement.
#      This guarantees that src traces and filled dst holes never overwrite each other.
#
# e) Visual observability
#    - Across a few examples, humans see two simultaneous regularities:
#      1. output size always equals input size;
#      2. colored source pieces turn into one uniform grey trace while those colors
#         reappear inside the other object.
#    - No invisible property such as numeric color ordering is used. Matching still
#      depends only on visible relative hole/subsprite arrangement.
#
# f) Visual cleanliness
#    - The trace uses one reserved color and stays exactly on the old source shapes,
#      so the src object remains readable: outline stays intact, only the interior
#      content changes.
#    - Rectangle-level separation between objects avoids clutter from inter-object
#      overlap in the full-canvas output.
#
# g) Inferability of color behavior
#    - The only color mapping remains "copy source color to dst by positional rank".
#      Source-side recoloring is deliberately simpler: all transferred source cells
#      become the same marker color. This is easy to spot from 2-3 samples.
#
# h) New solver-side assumptions introduced here
#    - There are still exactly two top-level 8-connected objects. Generator enforces it.
#    - The grey marker color is absent from the input palette. Generator reserves color 5.
#    - Src and dst output patches must not overlap after filling/marking. Generator
#      guarantees rectangle-level separation when placing the two cropped objects.
#
# i) Non-degeneracy metric used after testing
#    - `trace_pixels`: number of cells that change from a non-marker input fill color to
#      the reserved grey marker in the output. This must be >0 in every generated sample.
#    - `same_shape`: output.shape == input.shape. This verifies the D5/0 full-canvas part.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


MARKER_COLOR = 5

CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)

CAVITY_COUNT_RANGE = (4, 9)
FILL_COLOR_COUNT_RANGE = (3, 7)

ROW_HEIGHT_CHOICES = [1, 2, 2, 3, 3, 3]
LAYOUT_K_CHOICES_FULL = [1, 2, 2, 3, 3]
LAYOUT_K_CHOICES_SMALL = [1, 2, 2]
ROW_GAP_CHOICES = [1, 1, 2]

CENTERED_SLOT_WIDTHS = [3, 3, 5, 5, 7]
PAIR_SLOT_WIDTHS = [2, 2, 3, 3, 4]
SIDE_SLOT_WIDTHS = [2, 2, 3]
MID_SLOT_WIDTHS = [1, 3]
PAIR_AXIS_GAP_CHOICES = [1, 1, 3]
TRIO_MID_SIDE_GAP = 1

NONRECT_PROBABILITY = 0.8
MAX_MUTATE_ATTEMPTS = 40
COLOR_MISMATCH_RATE = 0.10

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2

MAX_GEN_ATTEMPTS = 400
MAX_LAYOUT_R_BUDGET = 24


def _components(mask: np.ndarray, *, connectivity: int = 8) -> list[dict]:
    lab, n = label_mask(mask, connectivity=connectivity)
    comps: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        comps.append(
            {
                "cid": int(cid),
                "pts": pts,
                "size": int(len(pts)),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    comps.sort(key=lambda d: int(d["size"]), reverse=True)
    return comps


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    comps = _components(grid != int(bg), connectivity=8)
    objs: list[dict] = []
    for comp in comps[:2]:
        r0, c0, r1, c1 = comp["bbox"]
        sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        for rr, cc in comp["pts"]:
            sub[int(rr) - int(r0), int(cc) - int(c0)] = int(grid[int(rr), int(cc)])
        objs.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "sub": sub,
            }
        )
    return objs


def _enclosed_bg_components(sub: np.ndarray, bg: int) -> list[dict]:
    lab, n = label_mask(sub == int(bg), connectivity=4)
    h, w = sub.shape
    holes: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if int(r0) == 0 or int(c0) == 0 or int(r1) == h - 1 or int(c1) == w - 1:
            continue
        holes.append(
            {
                "pts": pts,
                "cent": (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean())))),
                "size": int(len(pts)),
            }
        )
    return holes


def _outline_color(sub: np.ndarray, bg: int) -> int:
    cols = [int(v) for v in np.unique(sub) if int(v) != int(bg)]
    if not cols:
        return int(bg)
    if len(cols) == 1:
        return int(cols[0])
    cnt = Counter(map(int, sub.flatten()))
    return int(max(cols, key=lambda col: (cnt[int(col)], -int(col))))


def _colored_components(sub: np.ndarray, bg: int, outline_col: int) -> list[dict]:
    mask = (sub != int(bg)) & (sub != int(outline_col))
    lab, n = label_mask(mask, connectivity=4)
    subs: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        col = Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0]
        subs.append(
            {
                "pts": pts,
                "cent": (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean())))),
                "color": int(col),
            }
        )
    return subs


def _rank_maps(points: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], dict[int, int], dict[int, int]]:
    rows = sorted(set(int(r) for r, _ in points))
    cols = sorted(set(int(c) for _, c in points))
    rrank = {int(r): i for i, r in enumerate(rows)}
    crank = {int(c): i for i, c in enumerate(cols)}
    keys = [(int(rrank[int(r)]), int(crank[int(c)])) for r, c in points]
    return keys, rrank, crank


def _transform_point(p: tuple[int, int], h0: int, w0: int, k: int, flip: bool) -> tuple[int, int]:
    r, c = int(p[0]), int(p[1])
    if int(k) == 0:
        rr, cc = int(r), int(c)
        h2, w2 = int(h0), int(w0)
    elif int(k) == 1:
        rr, cc = int(w0 - 1 - c), int(r)
        h2, w2 = int(w0), int(h0)
    elif int(k) == 2:
        rr, cc = int(h0 - 1 - r), int(w0 - 1 - c)
        h2, w2 = int(h0), int(w0)
    else:
        rr, cc = int(c), int(h0 - 1 - r)
        h2, w2 = int(w0), int(h0)
    if bool(flip):
        cc = int(w2 - 1 - cc)
    return int(rr), int(cc)


def _classify_src_dst(objs: list[dict], bg: int) -> tuple[dict, dict]:
    feats = []
    for obj in objs:
        sub = obj["sub"]
        non_bg_colors = [int(v) for v in np.unique(sub) if int(v) != int(bg)]
        holes = _enclosed_bg_components(sub, bg)
        feats.append(
            {
                "obj": obj,
                "n_colors": int(len(non_bg_colors)),
                "hole_area": int(sum(int(h["size"]) for h in holes)),
            }
        )
    if len(feats) != 2:
        raise ValueError("expected exactly two objects")
    feats.sort(key=lambda d: (int(d["n_colors"]), int(d["hole_area"])))
    if int(feats[0]["n_colors"]) != int(feats[1]["n_colors"]):
        dest_feat = feats[0]
        src_feat = feats[1]
    else:
        dest_feat = max(feats, key=lambda d: int(d["hole_area"]))
        src_feat = min(feats, key=lambda d: int(d["hole_area"]))
    return src_feat["obj"], dest_feat["obj"]


def _choose_marker_color(grid: np.ndarray, bg: int, *, preferred: int = MARKER_COLOR) -> int:
    present = {int(v) for v in np.unique(grid)}
    if int(preferred) != int(bg) and int(preferred) not in present:
        return int(preferred)
    for col in range(10):
        if int(col) != int(bg) and int(col) not in present:
            return int(col)
    for col in range(10):
        if int(col) != int(bg):
            return int(col)
    return int(preferred)


def _solve_local_objects(src_sub: np.ndarray, dest_sub: np.ndarray, bg: int, marker_color: int) -> tuple[np.ndarray, np.ndarray]:
    outline_col = _outline_color(dest_sub, bg)
    dest_holes = _enclosed_bg_components(dest_sub, bg)
    src_subs = _colored_components(src_sub, bg, outline_col)
    if not dest_holes or not src_subs:
        return src_sub.copy(), dest_sub.copy()

    dest_pts = [h["cent"] for h in dest_holes]
    dest_keys, d_rr, d_cc = _rank_maps(dest_pts)

    best = None
    for k in range(4):
        for flip in (False, True):
            tpts = [_transform_point(s["cent"], int(src_sub.shape[0]), int(src_sub.shape[1]), int(k), bool(flip)) for s in src_subs]
            tkeys, _s_rr, _s_cc = _rank_maps(tpts)
            if sorted(tkeys) == sorted(dest_keys):
                best = (int(k), bool(flip))
                break
        if best is not None:
            break
    if best is None:
        best = (0, False)
    k, flip = best

    dest_list = []
    for hole in dest_holes:
        key = (int(d_rr[int(hole["cent"][0])]), int(d_cc[int(hole["cent"][1])]))
        dest_list.append((key, hole["cent"], hole))
    dest_list.sort(key=lambda item: (item[0][0], item[0][1], item[1][0], item[1][1]))

    tpts_all = [_transform_point(s["cent"], int(src_sub.shape[0]), int(src_sub.shape[1]), int(k), bool(flip)) for s in src_subs]
    _keys, s_rr, s_cc = _rank_maps(tpts_all)

    src_list = []
    for sub in src_subs:
        tp = _transform_point(sub["cent"], int(src_sub.shape[0]), int(src_sub.shape[1]), int(k), bool(flip))
        key = (int(s_rr[int(tp[0])]), int(s_cc[int(tp[1])]))
        src_list.append((key, tp, int(sub["color"]), sub["pts"]))
    src_list.sort(key=lambda item: (item[0][0], item[0][1], item[1][0], item[1][1]))

    src_marked = src_sub.copy()
    dest_filled = dest_sub.copy()

    for (_src_key, _tp, col, _src_pts), (_dst_key, _dp, hole) in zip(src_list, dest_list):
        for rr, cc in hole["pts"]:
            dest_filled[int(rr), int(cc)] = int(col)

    for _src_key, _tp, _col, src_pts in src_list:
        for rr, cc in src_pts:
            src_marked[int(rr), int(cc)] = int(marker_color)

    return src_marked, dest_filled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Full-canvas + source-trace variant.

    Fill dst holes with the matched src colors, but keep the original canvas size and
    rewrite all src subsprite pixels to one marker color to show that the colors were
    transferred out of src.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    marker_color = int(_choose_marker_color(g, bg))

    objs = _extract_objects(g, bg)
    if len(objs) < 2:
        raise ValueError("main: expected two objects")
    src_obj, dest_obj = _classify_src_dst(objs, bg)

    src_marked, dest_filled = _solve_local_objects(src_obj["sub"], dest_obj["sub"], bg, marker_color)

    out = g.copy()
    src_r0, src_c0, _src_r1, _src_c1 = src_obj["bbox"]
    dest_r0, dest_c0, _dest_r1, _dest_c1 = dest_obj["bbox"]
    paste(out, src_marked, int(src_r0), int(src_c0))
    paste(out, dest_filled, int(dest_r0), int(dest_c0))
    return out


def _random_rect_shape(h: int, w: int) -> np.ndarray:
    return np.ones((int(h), int(w)), dtype=bool)


def _mutate_shape(mask: np.ndarray, *, n_remove_max: int, symmetric: bool = False) -> np.ndarray:
    m = mask.copy()
    h, w = m.shape
    n_remove = random.randint(1, max(1, int(n_remove_max)))
    removed = 0
    tried = 0
    while removed < n_remove and tried < MAX_MUTATE_ATTEMPTS:
        tried += 1
        cand = []
        for r in range(h):
            for c in range(w):
                if not bool(m[int(r), int(c)]):
                    continue
                nb = 0
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(m[int(nr), int(nc)]):
                        nb += 1
                if nb <= 2:
                    cand.append((int(r), int(c)))
        if not cand:
            break
        r, c = random.choice(cand)
        to_remove = [(int(r), int(c))]
        if bool(symmetric):
            mc = int(w - 1 - c)
            if (int(r), int(mc)) != (int(r), int(c)):
                to_remove.append((int(r), int(mc)))
        m2 = m.copy()
        for rr, cc in to_remove:
            m2[int(rr), int(cc)] = False
        if not m2.any():
            continue
        _, n_comp = label_mask(m2, connectivity=4)
        if int(n_comp) == 1:
            m = m2
            removed += len(to_remove)
    return m if m.any() else mask.copy()


def _gen_shape(h: int, w: int, *, nonrect: bool, symmetric: bool = False) -> np.ndarray:
    m = _random_rect_shape(h, w)
    if bool(nonrect) and int(h) * int(w) >= 3:
        m = _mutate_shape(m, n_remove_max=max(1, (int(h) * int(w)) // 2), symmetric=symmetric)
    return m


def _mirror_shape(mask: np.ndarray) -> np.ndarray:
    return mask[:, ::-1].copy()


def _gen_dst_src_pair(h: int, w: int, *, symmetric: bool) -> tuple[np.ndarray, np.ndarray]:
    nonrect = random.random() < NONRECT_PROBABILITY
    return (
        _gen_shape(h, w, nonrect=nonrect, symmetric=symmetric),
        _gen_shape(h, w, nonrect=nonrect, symmetric=symmetric),
    )


def _add_centered_slot(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    w = random.choice(CENTERED_SLOT_WIDTHS)
    dst_shape, src_shape = _gen_dst_src_pair(row_h, w, symmetric=True)
    slots.append(
        {
            "r0": int(cur_r),
            "col_offset": -int(w // 2),
            "shape_dst": dst_shape,
            "shape_src": src_shape,
            "pair_id": int(pair_id),
        }
    )


def _add_mirror_pair_slots(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    w = random.choice(PAIR_SLOT_WIDTHS)
    dst_l, src_l = _gen_dst_src_pair(row_h, w, symmetric=False)
    dst_r = _mirror_shape(dst_l)
    src_r = _mirror_shape(src_l)
    axis_gap = random.choice(PAIR_AXIS_GAP_CHOICES)
    if int(axis_gap) % 2 == 0:
        axis_gap += 1
    half = int(axis_gap // 2)
    slots.append(
        {
            "r0": int(cur_r),
            "col_offset": -int(half + w),
            "shape_dst": dst_l,
            "shape_src": src_l,
            "pair_id": int(pair_id),
        }
    )
    slots.append(
        {
            "r0": int(cur_r),
            "col_offset": int(half + 1),
            "shape_dst": dst_r,
            "shape_src": src_r,
            "pair_id": int(pair_id),
        }
    )


def _add_trio_slots(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    w_side = random.choice(SIDE_SLOT_WIDTHS)
    w_mid = random.choice(MID_SLOT_WIDTHS)
    dst_l, src_l = _gen_dst_src_pair(row_h, w_side, symmetric=False)
    dst_r = _mirror_shape(dst_l)
    src_r = _mirror_shape(src_l)
    dst_m, src_m = _gen_dst_src_pair(row_h, w_mid, symmetric=True)
    mid_half = int(w_mid // 2)
    slots.append(
        {
            "r0": int(cur_r),
            "col_offset": -int(mid_half + w_side + TRIO_MID_SIDE_GAP),
            "shape_dst": dst_l,
            "shape_src": src_l,
            "pair_id": int(pair_id),
        }
    )
    slots.append(
        {
            "r0": int(cur_r),
            "col_offset": -int(mid_half),
            "shape_dst": dst_m,
            "shape_src": src_m,
            "pair_id": int(pair_id + 1),
        }
    )
    slots.append(
        {
            "r0": int(cur_r),
            "col_offset": int(mid_half + TRIO_MID_SIDE_GAP + 1),
            "shape_dst": dst_r,
            "shape_src": src_r,
            "pair_id": int(pair_id),
        }
    )


def _pick_layout_k(budget: int) -> int:
    if int(budget) == 1:
        return 1
    if int(budget) == 2:
        return int(random.choice(LAYOUT_K_CHOICES_SMALL))
    return int(random.choice(LAYOUT_K_CHOICES_FULL))


def _design_layout(n_target: int) -> list[dict]:
    slots: list[dict] = []
    cur_r = 0
    pair_id = 0
    budget = int(n_target)
    while len(slots) < int(n_target) and cur_r < MAX_LAYOUT_R_BUDGET and budget > 0:
        row_h = random.choice(ROW_HEIGHT_CHOICES)
        k = _pick_layout_k(budget)
        if int(k) == 1:
            _add_centered_slot(slots, cur_r, pair_id, row_h)
            pair_id += 1
            budget -= 1
        elif int(k) == 2:
            _add_mirror_pair_slots(slots, cur_r, pair_id, row_h)
            pair_id += 1
            budget -= 2
        else:
            _add_trio_slots(slots, cur_r, pair_id, row_h)
            pair_id += 2
            budget -= 3
        cur_r += int(row_h) + int(random.choice(ROW_GAP_CHOICES))
    return slots


def _render_object(
    layout: list[dict],
    shapes: list[np.ndarray],
    colors: list[int],
    bg: int,
    outline_col: int,
    *,
    filled: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    placed = []
    for slot_idx, (slot, shape) in enumerate(zip(layout, shapes)):
        rr, cc = np.where(shape)
        if len(rr) == 0:
            continue
        placed.append((int(slot_idx), slot["r0"] + rr, slot["col_offset"] + cc))
    if not placed:
        return None, None

    all_rows = np.concatenate([p[1] for p in placed])
    all_cols = np.concatenate([p[2] for p in placed])
    r_min, r_max = int(all_rows.min()), int(all_rows.max())
    c_min, c_max = int(all_cols.min()), int(all_cols.max())

    pad = 2
    h = int(r_max - r_min + 1 + 2 * pad)
    w = int(c_max - c_min + 1 + 2 * pad)
    canvas = np.full((h, w), int(bg), dtype=int)
    fill_mask = np.zeros((h, w), dtype=bool)

    for slot_idx, rows, cols in placed:
        rr = rows - int(r_min) + pad
        cc = cols - int(c_min) + pad
        canvas[rr, cc] = int(colors[int(slot_idx)])
        fill_mask[rr, cc] = True

    outline_mask = np.zeros_like(fill_mask)
    outline_mask[1:, :] |= fill_mask[:-1, :]
    outline_mask[:-1, :] |= fill_mask[1:, :]
    outline_mask[:, 1:] |= fill_mask[:, :-1]
    outline_mask[:, :-1] |= fill_mask[:, 1:]
    outline_mask &= ~fill_mask
    canvas[outline_mask] = int(outline_col)

    filled_grid = canvas.copy()
    if not bool(filled):
        canvas[fill_mask] = int(bg)

    b = bbox(canvas != int(bg))
    if b is None:
        return None, None
    cropped = crop_bbox(canvas, b)
    r0, c0, r1, c1 = b
    filled_cropped = filled_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    return cropped, filled_cropped


def _count_4conn_cavities(filled_grid: np.ndarray, bg: int, outline_col: int) -> int:
    mask = (filled_grid != int(bg)) & (filled_grid != int(outline_col))
    _, n = label_mask(mask, connectivity=4)
    return int(n)


def _count_enclosed_bg_holes(empty_grid: np.ndarray, bg: int) -> int:
    lab, n = label_mask(empty_grid == int(bg), connectivity=4)
    h, w = empty_grid.shape
    count = 0
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if int(r0) > 0 and int(c0) > 0 and int(r1) < h - 1 and int(c1) < w - 1:
            count += 1
    return int(count)


def _is_8connected(grid: np.ndarray, bg: int) -> bool:
    mask = grid != int(bg)
    if not mask.any():
        return False
    _, n = label_mask(mask, connectivity=8)
    return int(n) == 1


def _size_in_spec(grid: np.ndarray) -> bool:
    h, w = grid.shape
    return OBJECT_H_RANGE[0] <= int(h) <= OBJECT_H_RANGE[1] and OBJECT_W_RANGE[0] <= int(w) <= OBJECT_W_RANGE[1]


def _assign_pair_colors(layout: list[dict], fill_pool: list[int]) -> list[int]:
    pair_ids: list[int] = []
    for slot in layout:
        if int(slot["pair_id"]) not in pair_ids:
            pair_ids.append(int(slot["pair_id"]))
    n_pairs = len(pair_ids)
    n_colors = len(fill_pool)
    if n_pairs >= n_colors:
        pair_colors = list(fill_pool) + [int(random.choice(fill_pool)) for _ in range(n_pairs - n_colors)]
    else:
        pair_colors = list(fill_pool[:n_pairs])
    random.shuffle(pair_colors)
    pair_color_map = dict(zip(pair_ids, pair_colors))
    colors = [int(pair_color_map[int(slot["pair_id"])]) for slot in layout]
    for i in range(len(colors)):
        if random.random() < COLOR_MISMATCH_RATE:
            colors[i] = int(random.choice(fill_pool))
    return colors


def _pick_bg_and_palette() -> tuple[int, list[int]]:
    bg_choices = [c for c in range(10) if c != MARKER_COLOR]
    bg = int(random.choice(bg_choices))
    palette = [c for c in range(10) if c not in (int(bg), int(MARKER_COLOR))]
    random.shuffle(palette)
    return int(bg), [int(c) for c in palette]


def _build_object_pair(
    layout: list[dict],
    colors: list[int],
    bg: int,
    outline_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    n_cavities = len(layout)
    dst_shapes = [slot["shape_dst"] for slot in layout]
    src_shapes = [slot["shape_src"] for slot in layout]

    dst_empty, dst_filled = _render_object(layout, dst_shapes, colors, bg, outline_col, filled=False)
    if dst_empty is None or dst_filled is None:
        return None
    if not _is_8connected(dst_empty, bg):
        return None
    if _count_enclosed_bg_holes(dst_empty, bg) != int(n_cavities):
        return None
    if not _size_in_spec(dst_empty):
        return None

    src_input, src_filled = _render_object(layout, src_shapes, colors, bg, outline_col, filled=True)
    if src_input is None or src_filled is None:
        return None
    if not _is_8connected(src_input, bg):
        return None
    if _count_4conn_cavities(src_input, bg, outline_col) != int(n_cavities):
        return None
    if not _size_in_spec(src_input):
        return None

    marker_colors = [int(MARKER_COLOR)] * int(n_cavities)
    src_marked, _src_marked_filled = _render_object(layout, src_shapes, marker_colors, bg, outline_col, filled=True)
    if src_marked is None:
        return None
    if src_marked.shape != src_input.shape:
        return None
    if np.array_equal(src_marked, src_input):
        return None

    return src_input, dst_empty, src_marked, dst_filled


def _place_on_canvas(src: np.ndarray, dst: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]] | None:
    canvas_h = random.randint(*CANVAS_SIZE_RANGE)
    canvas_w = random.randint(*CANVAS_SIZE_RANGE)

    occ = np.zeros((int(canvas_h), int(canvas_w)), dtype=int)
    src_block = np.ones(src.shape, dtype=int)
    dst_block = np.ones(dst.shape, dtype=int)
    try:
        src_pos = random_free_location(
            occ,
            src_block,
            transparent=0,
            border=PLACEMENT_BORDER,
            padding=PLACEMENT_PADDING,
        )
        paste(occ, src_block, int(src_pos[0]), int(src_pos[1]), transparent=0)
        dst_pos = random_free_location(
            occ,
            dst_block,
            transparent=0,
            border=PLACEMENT_BORDER,
            padding=PLACEMENT_PADDING,
        )
    except ValueError:
        return None

    canvas = np.full((int(canvas_h), int(canvas_w)), int(bg), dtype=int)
    paste(canvas, src, int(src_pos[0]), int(src_pos[1]), transparent=bg)
    paste(canvas, dst, int(dst_pos[0]), int(dst_pos[1]), transparent=bg)
    return canvas, (int(src_pos[0]), int(src_pos[1])), (int(dst_pos[0]), int(dst_pos[1]))


def _expected_canvas(
    src_marked: np.ndarray,
    dst_filled: np.ndarray,
    src_pos: tuple[int, int],
    dst_pos: tuple[int, int],
    canvas_shape: tuple[int, int],
    bg: int,
) -> np.ndarray:
    out = np.full(tuple(map(int, canvas_shape)), int(bg), dtype=int)
    paste(out, src_marked, int(src_pos[0]), int(src_pos[1]), transparent=bg)
    paste(out, dst_filled, int(dst_pos[0]), int(dst_pos[1]), transparent=bg)
    return out


def trace_pixels(inp: np.ndarray, out: np.ndarray, *, marker_color: int = MARKER_COLOR) -> int:
    inp_arr = np.array(inp, dtype=int)
    out_arr = np.array(out, dtype=int)
    changed_to_marker = (inp_arr != out_arr) & (out_arr == int(marker_color))
    return int(np.count_nonzero(changed_to_marker))


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline_col = int(palette[0])

    layout = _design_layout(random.randint(*CAVITY_COUNT_RANGE))
    if len(layout) < CAVITY_COUNT_RANGE[0]:
        return None
    n_cavities = len(layout)

    fc_lo, fc_hi = FILL_COLOR_COUNT_RANGE
    max_cols = min(int(fc_hi), int(n_cavities), len(palette) - 1)
    if max_cols < int(fc_lo):
        return None
    n_fill_cols = random.randint(int(fc_lo), int(max_cols))
    fill_pool = [int(c) for c in palette[1 : 1 + n_fill_cols]]
    colors = _assign_pair_colors(layout, fill_pool)

    built = _build_object_pair(layout, colors, bg, outline_col)
    if built is None:
        return None
    src_input, dst_empty, src_marked, dst_filled = built

    placed = _place_on_canvas(src_input, dst_empty, bg)
    if placed is None:
        return None
    inp, src_pos, dst_pos = placed
    expected = _expected_canvas(src_marked, dst_filled, src_pos, dst_pos, inp.shape, bg)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not isinstance(actual, np.ndarray):
        return None
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    if trace_pixels(inp, actual, marker_color=MARKER_COLOR) <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
