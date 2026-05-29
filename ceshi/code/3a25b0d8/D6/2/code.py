# Variant of 3a25b0d8
# Dimension: D6 — 轨迹/源对象去留 (trajectory)
# Change: 默认消失 — src 不出现在输出中（因为 tight crop 只包含 dst）... → src 和 dst 交换内容
#
# Design analysis for the D6 variant
# ==================================
# a) Concrete behavior and trace
#    - Original: input has two outline-colored 8-connected objects. `src` contains colored
#      subsprites inside cavities; `dst` contains empty bg holes. Output keeps only `dst`
#      and fills each hole with the matched `src` color.
#    - Variant: output keeps BOTH objects at their original relative positions.
#      `dst` still receives `src` colors, but `src` also changes: erase each colored
#      subsprite and write back the matched `dst` hole SHAPE using the shared outline color.
#      This is a genuine bidirectional swap: colors move from src to dst; hole silhouettes
#      move from dst to src.
#    - Small trace on one matched pair:
#      original pair:
#        src cavity bbox = 3x3 filled by color 7 in an L-shape
#        dst hole bbox = same 3x3 but empty T-shape
#      old output:
#        dst hole becomes solid color 7 T-shape; src vanishes because cropping keeps only dst
#      new output:
#        dst hole still becomes color 7 T-shape
#        src first loses the old L-shaped color 7 fill, then gains an outline-colored T-shape
#        in the same slot; final crop contains both modified objects.
#
# b) Complexity calibration
#    - The matching rule stays the original centroid-rank pairing. This keeps inference focused
#      on the D6 change instead of introducing a second new matching mechanism.
#    - To make the bidirectional swap visible, generator keeps 4-9 cavities and independently
#      mutates src/dst shapes per slot; corresponding src/dst shapes are forced to differ when
#      possible, so the src-side rewrite is visually noticeable.
#
# c) Hidden assumptions from the old solver that break here
#    - `return tight_crop(dest)` depended on src disappearing. No longer valid; we must rebuild
#      a canvas with both transformed objects and crop their union.
#    - Old solver only needed src colors, so it stored src component centroids/colors only.
#      New solver also needs destination hole SHAPES and source slot bboxes to redraw src.
#    - Old generator allowed bbox overlap as long as non-bg pixels did not collide. That was
#      harmless because output only kept dst. Now src can light up previously-bg interior cells,
#      so bbox overlap can cause new output collisions. Generator must therefore enforce
#      rectangle-level separation between src and dst placements.
#
# d) Placement legality
#    - Generator places the two cropped objects as non-overlapping rectangles with a fixed gap.
#      This guarantees the swapped-in outline pixels inside src never collide with dst.
#    - Solver pastes the transformed subgrids back at their original bbox positions, so
#      generate_input() must preserve exactly two 8-connected top-level objects.
#
# e) Visual observability
#    - Across 2-4 examples, humans can now see that output contains TWO objects instead of one.
#    - On the dst side, colors arrive as before.
#    - On the src side, colored blobs disappear and are replaced by outline-colored silhouettes.
#      Because slot bboxes stay aligned but shapes differ, the swap is visible at the pixel level.
#
# f) Visual cleanliness
#    - Using the same shared outline color for the imported dst shapes keeps the src object legible:
#      frame and inserted silhouettes read as one object family.
#    - Bbox separation avoids inter-object clutter in the union crop.
#
# g) Color inferability
#    - No new arbitrary color mapping is introduced.
#    - The only color motion is the old, already-structured src->dst transfer by centroid rank.
#      The reverse transfer carries SHAPE only, using the already-visible outline color.
#
# h) New solver-side assumptions introduced by this variant
#    - Matching still assumes each paired src subsprite and dst hole share the same template bbox
#      dimensions. The generator guarantees this by deriving both from the same per-slot `(h, w)`.
#    - Because src rewriting uses those slot-local coordinates, generator must not rotate or flip
#      the whole object relative to the other one. This remains true here, matching the original
#      generator behavior.
#
# i) Variant metric used after testing
#    - We will verify non-degeneracy by checking that generated outputs contain two 8-connected
#      objects and that the src-side swapped object differs from the original src in every sample.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Bidirectional swap variant of 3a25b0d8.

    - `dst` holes are filled by matched `src` colors.
    - `src` colored subsprites are erased and replaced by the matched `dst` hole shapes,
      rendered in the shared outline color.
    - Output keeps both transformed objects at their original relative positions, then
      tight-crops the union.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def _components(mask: np.ndarray) -> list[dict]:
        lab, n = label_mask(mask, connectivity=8)
        comps = []
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
                    "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
        comps.sort(key=lambda d: int(d["size"]), reverse=True)
        return comps

    def _bbox_of(group: list[dict]) -> tuple[int, int, int, int]:
        r0 = min(int(d["bbox"][0]) for d in group)
        c0 = min(int(d["bbox"][1]) for d in group)
        r1 = max(int(d["bbox"][2]) for d in group)
        c1 = max(int(d["bbox"][3]) for d in group)
        return int(r0), int(c0), int(r1), int(c1)

    def _extract_group_subgrid(group: list[dict]) -> np.ndarray:
        r0, c0, r1, c1 = _bbox_of(group)
        sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        for d in group:
            for rr, cc in d["pts"]:
                sub[int(rr) - int(r0), int(cc) - int(c0)] = int(g[int(rr), int(cc)])
        return sub

    def _enclosed_bg_components(sub: np.ndarray) -> list[dict]:
        m = sub == int(bg)
        lab, n = label_mask(m, connectivity=4)
        hs, ws = sub.shape
        holes = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if bool(r0 == 0 or c0 == 0 or r1 == int(hs - 1) or c1 == int(ws - 1)):
                continue
            holes.append(
                {
                    "pts": pts,
                    "cent": (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean())))),
                    "size": int(len(pts)),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
        return holes

    def _outline_col(sub: np.ndarray) -> int:
        cnt = Counter(map(int, sub.flatten()))
        opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
        return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)

    def _colored_components(sub: np.ndarray) -> list[dict]:
        outc = _outline_col(sub)
        m = (sub != int(bg)) & (sub != int(outc))
        lab, n = label_mask(m, connectivity=4)
        subs = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            col = Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0]
            subs.append(
                {
                    "pts": pts,
                    "cent": (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean())))),
                    "color": int(col),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
        return subs

    def _rank_keys(pts: list[tuple[int, int]]):
        rows = sorted(set(int(r) for r, _c in pts))
        cols = sorted(set(int(c) for _r, c in pts))
        rrank = {r: i for i, r in enumerate(rows)}
        crank = {c: i for i, c in enumerate(cols)}
        keys = [(int(rrank[r]), int(crank[c])) for r, c in pts]
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

    comps = _components(g != int(bg))
    assert len(comps) >= 2
    groups = [[comps[0]], [comps[1]]]
    objs = []
    for grp in groups:
        objs.append({"bbox": _bbox_of(grp), "sub": _extract_group_subgrid(grp)})

    holes_list = [_enclosed_bg_components(o["sub"]) for o in objs]
    hole_area = [sum(int(h["size"]) for h in hh) for hh in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)

    dest = objs[dest_i]["sub"].copy()
    src = objs[src_i]["sub"].copy()
    dest_bbox = objs[dest_i]["bbox"]
    src_bbox = objs[src_i]["bbox"]
    outline_col = _outline_col(dest)

    dest_holes = _enclosed_bg_components(dest)
    src_subs = _colored_components(src)
    if not dest_holes or not src_subs:
        out = np.full_like(g, bg)
        paste(out, src, int(src_bbox[0]), int(src_bbox[1]), transparent=bg)
        paste(out, dest, int(dest_bbox[0]), int(dest_bbox[1]), transparent=bg)
        b = bbox(out != int(bg))
        assert b is not None
        return crop_bbox(out, b)

    dest_pts = [h["cent"] for h in dest_holes]
    dest_keys, d_rr, d_cc = _rank_keys(dest_pts)

    best = None
    for k in range(4):
        for flip in [False, True]:
            tpts = [_transform_point(s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip)) for s in src_subs]
            tkeys, _s_rr, _s_cc = _rank_keys(tpts)
            if sorted(tkeys) == sorted(dest_keys):
                best = (int(k), bool(flip))
                break
        if best is not None:
            break
    if best is None:
        best = (0, False)
    k, flip = best

    dest_list = []
    for h in dest_holes:
        key = (int(d_rr[int(h["cent"][0])]), int(d_cc[int(h["cent"][1])]))
        dest_list.append((key, h["cent"], h))
    dest_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    tpts_all = [_transform_point(s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip)) for s in src_subs]
    _keys, s_rr, s_cc = _rank_keys(tpts_all)

    src_list = []
    for s in src_subs:
        tp = _transform_point(s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
        key = (int(s_rr[int(tp[0])]), int(s_cc[int(tp[1])]))
        src_list.append((key, tp, s))
    src_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    src_swapped = src.copy()
    for s in src_subs:
        for rr, cc in s["pts"]:
            src_swapped[int(rr), int(cc)] = int(bg)

    for (_k1, _p1, s), (_k2, _p2, h) in zip(src_list, dest_list):
        for rr, cc in h["pts"]:
            dest[int(rr), int(cc)] = int(s["color"])

        sr0, sc0, sr1, sc1 = s["bbox"]
        hr0, hc0, _hr1, _hc1 = h["bbox"]
        for rr, cc in h["pts"]:
            src_swapped[int(sr0 + (rr - hr0)), int(sc0 + (cc - hc0))] = int(outline_col)

    out = np.full_like(g, bg)
    paste(out, src_swapped, int(src_bbox[0]), int(src_bbox[1]), transparent=bg)
    paste(out, dest, int(dest_bbox[0]), int(dest_bbox[1]), transparent=bg)
    b = bbox(out != int(bg))
    assert b is not None
    return crop_bbox(out, b)


# --------------- generator (D6 bidirectional swap) ---------------

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
SHAPE_DIFFERENCE_ATTEMPTS = 20

COLOR_MISMATCH_RATE = 0.10

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
MAX_RECT_PLACE_ATTEMPTS = 120

MAX_GEN_ATTEMPTS = 300
MAX_LAYOUT_R_BUDGET = 24


def _random_rect_shape(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _mutate_shape(mask: np.ndarray, *, n_remove_max: int, symmetric: bool = False) -> np.ndarray:
    m = mask.copy()
    h, w = m.shape
    n_remove = random.randint(1, max(1, n_remove_max))
    tried = 0
    removed = 0
    while removed < n_remove and tried < MAX_MUTATE_ATTEMPTS:
        tried += 1
        cand = []
        for r in range(h):
            for c in range(w):
                if not m[r, c]:
                    continue
                nb = 0
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and m[nr, nc]:
                        nb += 1
                if nb <= 2:
                    cand.append((r, c))
        if not cand:
            break
        r, c = random.choice(cand)
        to_remove = [(r, c)]
        if symmetric:
            mc = w - 1 - c
            if (r, mc) != (r, c):
                to_remove.append((r, mc))
        m2 = m.copy()
        for rr, cc in to_remove:
            m2[rr, cc] = False
        if not m2.any():
            continue
        _, n_components = label_mask(m2, connectivity=4)
        if int(n_components) == 1:
            m = m2
            removed += len(to_remove)
    if not m.any():
        return mask.copy()
    return m


def _gen_shape(h: int, w: int, *, nonrect: bool, symmetric: bool = False) -> np.ndarray:
    m = _random_rect_shape(h, w)
    if nonrect and h * w >= 3:
        m = _mutate_shape(m, n_remove_max=max(1, (h * w) // 2), symmetric=symmetric)
    return m


def _mirror_shape(m: np.ndarray) -> np.ndarray:
    return m[:, ::-1].copy()


def _gen_dst_src_pair(h: int, w: int, *, symmetric: bool) -> tuple[np.ndarray, np.ndarray]:
    nonrect = random.random() < NONRECT_PROBABILITY
    dst = _gen_shape(h, w, nonrect=nonrect, symmetric=symmetric)
    for _ in range(SHAPE_DIFFERENCE_ATTEMPTS):
        src = _gen_shape(h, w, nonrect=nonrect, symmetric=symmetric)
        if not np.array_equal(dst, src):
            return dst, src
    return dst, src


def _add_centered_slot(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    w = random.choice(CENTERED_SLOT_WIDTHS)
    dst, src = _gen_dst_src_pair(row_h, w, symmetric=True)
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -(w // 2),
            "shape_dst": dst,
            "shape_src": src,
            "pair_id": pair_id,
        }
    )


def _add_mirror_pair_slots(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    w = random.choice(PAIR_SLOT_WIDTHS)
    dst_l, src_l = _gen_dst_src_pair(row_h, w, symmetric=False)
    axis_gap = random.choice(PAIR_AXIS_GAP_CHOICES)
    if axis_gap % 2 == 0:
        axis_gap += 1
    half = axis_gap // 2
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -(half + w),
            "shape_dst": dst_l,
            "shape_src": src_l,
            "pair_id": pair_id,
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": half + 1,
            "shape_dst": _mirror_shape(dst_l),
            "shape_src": _mirror_shape(src_l),
            "pair_id": pair_id,
        }
    )


def _add_trio_slots(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    w_side = random.choice(SIDE_SLOT_WIDTHS)
    w_mid = random.choice(MID_SLOT_WIDTHS)
    dst_l, src_l = _gen_dst_src_pair(row_h, w_side, symmetric=False)
    dst_m, src_m = _gen_dst_src_pair(row_h, w_mid, symmetric=True)
    mid_half = w_mid // 2
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -(mid_half + w_side + TRIO_MID_SIDE_GAP),
            "shape_dst": dst_l,
            "shape_src": src_l,
            "pair_id": pair_id,
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -mid_half,
            "shape_dst": dst_m,
            "shape_src": src_m,
            "pair_id": pair_id + 1,
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": mid_half + TRIO_MID_SIDE_GAP + 1,
            "shape_dst": _mirror_shape(dst_l),
            "shape_src": _mirror_shape(src_l),
            "pair_id": pair_id,
        }
    )


def _pick_layout_k(budget: int) -> int:
    if budget == 1:
        return 1
    if budget == 2:
        return random.choice(LAYOUT_K_CHOICES_SMALL)
    return random.choice(LAYOUT_K_CHOICES_FULL)


def _design_layout(n_target: int) -> list[dict]:
    slots: list[dict] = []
    cur_r = 0
    pair_id = 0
    budget = n_target
    while len(slots) < n_target and cur_r < MAX_LAYOUT_R_BUDGET and budget > 0:
        row_h = random.choice(ROW_HEIGHT_CHOICES)
        k = _pick_layout_k(budget)
        if k == 1:
            _add_centered_slot(slots, cur_r, pair_id, row_h)
            pair_id += 1
            budget -= 1
        elif k == 2:
            _add_mirror_pair_slots(slots, cur_r, pair_id, row_h)
            pair_id += 1
            budget -= 2
        else:
            _add_trio_slots(slots, cur_r, pair_id, row_h)
            pair_id += 2
            budget -= 3
        cur_r += row_h + random.choice(ROW_GAP_CHOICES)
    return slots


def _render_object(
    layout: list[dict],
    shapes: list[np.ndarray],
    colors: list[int],
    bg: int,
    outline_col: int,
    *,
    filled: bool,
    return_slots: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None] | tuple[np.ndarray | None, np.ndarray | None, list[dict]]:
    placed = []
    for slot_idx, (slot, sh) in enumerate(zip(layout, shapes)):
        rr, cc = np.where(sh)
        if len(rr) == 0:
            continue
        placed.append(
            {
                "slot_idx": slot_idx,
                "rows": slot["r0"] + rr,
                "cols": slot["col_offset"] + cc,
                "shape": sh,
                "origin_r": slot["r0"],
                "origin_c": slot["col_offset"],
            }
        )
    if not placed:
        return (None, None, []) if return_slots else (None, None)

    all_rows = np.concatenate([p["rows"] for p in placed])
    all_cols = np.concatenate([p["cols"] for p in placed])
    r_min, r_max = int(all_rows.min()), int(all_rows.max())
    c_min, c_max = int(all_cols.min()), int(all_cols.max())

    pad = 2
    h = (r_max - r_min + 1) + 2 * pad
    w = (c_max - c_min + 1) + 2 * pad

    canvas = np.full((h, w), bg, dtype=int)
    fill_mask = np.zeros((h, w), dtype=bool)
    slot_meta = [None] * len(shapes)
    for p in placed:
        rr = p["rows"] - r_min + pad
        cc = p["cols"] - c_min + pad
        canvas[rr, cc] = colors[p["slot_idx"]]
        fill_mask[rr, cc] = True
        slot_meta[p["slot_idx"]] = {
            "origin_r": int(p["origin_r"] - r_min + pad),
            "origin_c": int(p["origin_c"] - c_min + pad),
            "shape": p["shape"],
        }

    outline_mask = np.zeros_like(fill_mask)
    outline_mask[1:, :] |= fill_mask[:-1, :]
    outline_mask[:-1, :] |= fill_mask[1:, :]
    outline_mask[:, 1:] |= fill_mask[:, :-1]
    outline_mask[:, :-1] |= fill_mask[:, 1:]
    outline_mask &= ~fill_mask
    canvas[outline_mask] = outline_col

    filled_grid = canvas.copy()
    if not filled:
        canvas[fill_mask] = bg

    b = bbox(canvas != bg)
    if b is None:
        return (None, None, []) if return_slots else (None, None)

    cropped = crop_bbox(canvas, b)
    r0b, c0b, r1b, c1b = b
    filled_cropped = filled_grid[r0b : r1b + 1, c0b : c1b + 1]

    if not return_slots:
        return cropped, filled_cropped

    adjusted = []
    for meta in slot_meta:
        assert meta is not None
        adjusted.append(
            {
                "origin_r": int(meta["origin_r"] - r0b),
                "origin_c": int(meta["origin_c"] - c0b),
                "shape": meta["shape"],
            }
        )
    return cropped, filled_cropped, adjusted


def _count_4conn_cavities(filled_grid: np.ndarray, bg: int, outline_col: int) -> int:
    mask = (filled_grid != bg) & (filled_grid != outline_col)
    _, n = label_mask(mask, connectivity=4)
    return int(n)


def _count_enclosed_bg_holes(empty_grid: np.ndarray, bg: int) -> int:
    mask = empty_grid == bg
    lab, n = label_mask(mask, connectivity=4)
    h, w = empty_grid.shape
    count = 0
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == cid)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r0 > 0 and c0 > 0 and r1 < h - 1 and c1 < w - 1:
            count += 1
    return count


def _is_8connected(grid: np.ndarray, bg: int) -> bool:
    mask = grid != bg
    if not mask.any():
        return False
    _, n = label_mask(mask, connectivity=8)
    return int(n) == 1


def _size_in_spec(grid: np.ndarray) -> bool:
    h, w = grid.shape
    return OBJECT_H_RANGE[0] <= h <= OBJECT_H_RANGE[1] and OBJECT_W_RANGE[0] <= w <= OBJECT_W_RANGE[1]


def _assign_pair_colors(layout: list[dict], fill_pool: list[int]) -> list[int]:
    unique_pair_ids: list[int] = []
    for slot in layout:
        if slot["pair_id"] not in unique_pair_ids:
            unique_pair_ids.append(slot["pair_id"])
    n_pairs = len(unique_pair_ids)
    n_colors = len(fill_pool)

    if n_pairs >= n_colors:
        pair_colors = list(fill_pool) + [random.choice(fill_pool) for _ in range(n_pairs - n_colors)]
    else:
        pair_colors = list(fill_pool)[:n_pairs]
    random.shuffle(pair_colors)

    pair_color_map = dict(zip(unique_pair_ids, pair_colors))
    colors = [pair_color_map[slot["pair_id"]] for slot in layout]
    for i in range(len(colors)):
        if random.random() < COLOR_MISMATCH_RATE:
            colors[i] = random.choice(fill_pool)
    return colors


def _pick_bg_and_palette() -> tuple[int, list[int]]:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    return bg, palette


def _make_swapped_src(
    layout: list[dict],
    src_shapes: list[np.ndarray],
    dst_shapes: list[np.ndarray],
    colors: list[int],
    bg: int,
    outline_col: int,
) -> np.ndarray | None:
    src_cropped, _src_filled, slot_meta = _render_object(
        layout, src_shapes, colors, bg, outline_col, filled=True, return_slots=True
    )
    if src_cropped is None:
        return None
    swapped = src_cropped.copy()
    fill_mask = (swapped != bg) & (swapped != outline_col)
    swapped[fill_mask] = bg
    for meta, dst_shape in zip(slot_meta, dst_shapes):
        rr, cc = np.where(dst_shape)
        swapped[meta["origin_r"] + rr, meta["origin_c"] + cc] = outline_col
    return swapped


def _build_object_pair(
    layout: list[dict],
    colors: list[int],
    bg: int,
    outline_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n_cavities = len(layout)
    dst_shapes = [slot["shape_dst"] for slot in layout]
    src_shapes = [slot["shape_src"] for slot in layout]

    dst_cropped, dst_filled = _render_object(layout, dst_shapes, colors, bg, outline_col, filled=False)
    if dst_cropped is None or dst_filled is None:
        return None
    if not _is_8connected(dst_cropped, bg):
        return None
    if _count_enclosed_bg_holes(dst_cropped, bg) != n_cavities:
        return None
    if not _size_in_spec(dst_cropped):
        return None

    src_cropped, _ = _render_object(layout, src_shapes, colors, bg, outline_col, filled=True)
    if src_cropped is None:
        return None
    if not _is_8connected(src_cropped, bg):
        return None
    if _count_4conn_cavities(src_cropped, bg, outline_col) != n_cavities:
        return None
    if not _size_in_spec(src_cropped):
        return None

    src_swapped = _make_swapped_src(layout, src_shapes, dst_shapes, colors, bg, outline_col)
    if src_swapped is None:
        return None
    if np.array_equal(src_swapped, src_cropped):
        return None

    return src_cropped, dst_cropped, src_swapped, dst_filled


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _place_on_canvas(
    src: np.ndarray,
    dst: np.ndarray,
    src_swapped: np.ndarray,
    dst_filled: np.ndarray,
    bg: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    canvas_h = random.randint(*CANVAS_SIZE_RANGE)
    canvas_w = random.randint(*CANVAS_SIZE_RANGE)
    src_h, src_w = src.shape
    dst_h, dst_w = dst.shape
    if src_h + 2 * PLACEMENT_BORDER > canvas_h or src_w + 2 * PLACEMENT_BORDER > canvas_w:
        return None
    if dst_h + 2 * PLACEMENT_BORDER > canvas_h or dst_w + 2 * PLACEMENT_BORDER > canvas_w:
        return None

    for _ in range(MAX_RECT_PLACE_ATTEMPTS):
        sr = random.randint(PLACEMENT_BORDER, canvas_h - PLACEMENT_BORDER - src_h)
        sc = random.randint(PLACEMENT_BORDER, canvas_w - PLACEMENT_BORDER - src_w)
        dr = random.randint(PLACEMENT_BORDER, canvas_h - PLACEMENT_BORDER - dst_h)
        dc = random.randint(PLACEMENT_BORDER, canvas_w - PLACEMENT_BORDER - dst_w)

        src_rect = (sr, sc, sr + src_h - 1, sc + src_w - 1)
        dst_rect = (dr, dc, dr + dst_h - 1, dc + dst_w - 1)
        if _rects_overlap(src_rect, dst_rect, pad=PLACEMENT_PADDING):
            continue

        canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
        paste(canvas, src, sr, sc, transparent=bg)
        paste(canvas, dst, dr, dc, transparent=bg)

        out = np.full((canvas_h, canvas_w), bg, dtype=int)
        paste(out, src_swapped, sr, sc, transparent=bg)
        paste(out, dst_filled, dr, dc, transparent=bg)
        b = bbox(out != bg)
        if b is None:
            continue
        expected = crop_bbox(out, b)
        return canvas, expected
    return None


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline_col = palette[0]

    layout = _design_layout(random.randint(*CAVITY_COUNT_RANGE))
    if len(layout) < CAVITY_COUNT_RANGE[0]:
        return None
    n_cavities = len(layout)

    fc_lo, fc_hi = FILL_COLOR_COUNT_RANGE
    max_cols = min(fc_hi, n_cavities, len(palette) - 1)
    if max_cols < fc_lo:
        return None
    n_fill_cols = random.randint(fc_lo, max_cols)
    fill_pool = palette[1 : 1 + n_fill_cols]
    colors = _assign_pair_colors(layout, fill_pool)

    built = _build_object_pair(layout, colors, bg, outline_col)
    if built is None:
        return None
    src, dst, src_swapped, dst_filled = built

    placed = _place_on_canvas(src, dst, src_swapped, dst_filled, bg)
    if placed is None:
        return None
    canvas, expected = placed

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not isinstance(actual, np.ndarray):
        return None
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
