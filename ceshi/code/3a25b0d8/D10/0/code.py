# Variant of 3a25b0d8
# Dimension: D10 — src-dst 基数关系 (multi_step)
# Change: 1:1 — 画布上恰好一个 src 和一个 dst... → 多个 src 对一个 dst (many:1)
#
# Design analysis for the D10 variant
# - Concrete behavior:
#   Input now contains one empty dst outline object plus 2-3 partial src objects.
#   Every partial src keeps the full outline skeleton and the full cavity layout, but
#   only a subset of cavities is colored; the remaining cavities stay bg. The output is
#   still the tight crop of the dst with every hole filled, but the color evidence must
#   be aggregated across multiple src objects.
# - Behavior trace:
#   Original: pick the only src, rank-match all colored cavity centroids to all dst holes,
#   fill dst.
#   Variant: identify the unique fill-free object as dst, then for each remaining partial
#   src first align its full cavity layout (colored cavities + empty cavities together) to
#   the dst layout, then copy only that src's colored subset into dst. The solver now has
#   an explicit aggregation step over multiple objects.
# - Complexity calibration:
#   Partial srcs are not reduced to isolated colored blobs; they preserve the full cavity
#   geometry, so humans can visually see that several similar source frames jointly describe
#   one target frame. Each partial src gets at least two colored cavities to avoid trivial
#   singleton evidence and to keep the pattern visible.
# - Implicit-assumption audit of the original main():
#   `_extract_two_objects()` relied on exactly two 8-connected objects; invalid here.
#   `src_i = 1 - dest_i` relied on exactly one non-dst object; invalid here.
#   Transform search used only src colored centroids; invalid for partial srcs because a
#   subset loses absolute rank information. The fix is to align on the full source cavity
#   layout (colored components plus empty holes), then project only colored cavities.
#   The original zip-based fill loop assumed one complete src list; invalid here, replaced
#   by repeated per-source fills keyed by destination rank positions.
# - Placement legality:
#   Generator now places 3-4 objects on one canvas, so it slightly enlarges the canvas and
#   slightly tightens object size bounds. Objects are still placed independently with no
#   overlap and a visible gap.
# - Visual observability:
#   Across examples the key visible change is repeated similar source frames around one dst.
#   Each source shows complementary colored subsets on the same skeleton, making the
#   "collect colors from several sources" rule inferable from 2-3 examples.
# - Visual cleanliness:
#   All objects remain separated by padding; no new markers or overlays are introduced.
#   The only new visual element is repetition of the same source frame with partial fills.
# - Color inference:
#   Colors are unchanged from the original rule: each cavity's color is copied directly
#   from whichever partial src shows that cavity filled. No hidden color remapping is used.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill holes in one dst object using color evidence gathered from several partial src objects.

    Input structure:
    - bg = background.
    - One object is the dst: same outline framework, every cavity empty.
    - 2-3 other objects are partial srcs: same framework, but each colors only a subset
      of the cavities and leaves the rest empty.

    Rule:
    - Find all 8-connected objects.
    - Pick the unique fill-free object as dst.
    - For each remaining object, align its full cavity layout to dst and transfer only its
      colored cavities.
    - Tight-crop the filled dst.
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
            cent = (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))
            holes.append({"pts": pts, "cent": cent, "size": int(len(pts))})
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
            col = Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0]
            cent = (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))
            subs.append({"cent": cent, "color": int(col), "size": int(len(pts))})
        return subs

    def _all_slot_centers(sub: np.ndarray) -> list[tuple[int, int]]:
        holes = _enclosed_bg_components(sub)
        fills = _colored_components(sub)
        return [tuple(h["cent"]) for h in holes] + [tuple(f["cent"]) for f in fills]

    def _rank_keys(pts: list[tuple[int, int]]):
        rows = sorted(set(int(r) for r, _c in pts))
        cols = sorted(set(int(c) for _r, c in pts))
        rrank = {r: i for i, r in enumerate(rows)}
        crank = {c: i for i, c in enumerate(cols)}
        keys = [(int(rrank[r]), int(crank[c])) for r, c in pts]
        return keys, rrank, crank

    def _transform_point(
        p: tuple[int, int], h0: int, w0: int, k: int, flip: bool
    ) -> tuple[int, int]:
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

    def _tight_crop(sub: np.ndarray) -> np.ndarray:
        m = sub != int(bg)
        pts = np.argwhere(m)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        return sub[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

    comps = _components(g != int(bg))
    assert len(comps) >= 2

    objs = [_extract_group_subgrid([comp]) for comp in comps]
    infos = []
    for idx, obj in enumerate(objs):
        holes = _enclosed_bg_components(obj)
        fills = _colored_components(obj)
        infos.append(
            {
                "idx": int(idx),
                "obj": obj,
                "holes": holes,
                "fills": fills,
                "slot_centers": _all_slot_centers(obj),
                "hole_area": int(sum(int(h["size"]) for h in holes)),
            }
        )

    dest_candidates = [info for info in infos if info["holes"] and not info["fills"]]
    if dest_candidates:
        dest_info = max(dest_candidates, key=lambda d: (len(d["holes"]), d["hole_area"]))
    else:
        dest_info = max(infos, key=lambda d: (len(d["holes"]), d["hole_area"]))

    dest = dest_info["obj"].copy()
    dest_holes = dest_info["holes"]
    if not dest_holes:
        return _tight_crop(dest)

    dest_pts = [tuple(h["cent"]) for h in dest_holes]
    dest_keys, d_rr, d_cc = _rank_keys(dest_pts)

    dest_by_key: dict[tuple[int, int], dict] = {}
    for h in dest_holes:
        key = (int(d_rr[int(h["cent"][0])]), int(d_cc[int(h["cent"][1])]))
        dest_by_key[key] = h

    for info in infos:
        if int(info["idx"]) == int(dest_info["idx"]) or not info["fills"]:
            continue

        slot_pts = [tuple(p) for p in info["slot_centers"]]
        if len(slot_pts) != len(dest_pts):
            continue

        best = None
        best_tpts = None
        for k in range(4):
            for flip in [False, True]:
                tpts = [
                    _transform_point(p, int(info["obj"].shape[0]), int(info["obj"].shape[1]), int(k), bool(flip))
                    for p in slot_pts
                ]
                tkeys, _s_rr, _s_cc = _rank_keys(tpts)
                if sorted(tkeys) == sorted(dest_keys):
                    best = (int(k), bool(flip))
                    best_tpts = tpts
                    break
            if best is not None:
                break
        if best is None:
            best = (0, False)
            best_tpts = slot_pts
        k, flip = best
        _keys, s_rr, s_cc = _rank_keys(best_tpts)

        for s in info["fills"]:
            tp = _transform_point(
                tuple(s["cent"]),
                int(info["obj"].shape[0]),
                int(info["obj"].shape[1]),
                int(k),
                bool(flip),
            )
            key = (int(s_rr[int(tp[0])]), int(s_cc[int(tp[1])]))
            hole = dest_by_key.get(key)
            if hole is None:
                continue
            for rr, cc in hole["pts"]:
                dest[int(rr), int(cc)] = int(s["color"])

    return _tight_crop(dest)


# --------------- generator ---------------

CANVAS_SIZE_RANGE = (24, 34)
OBJECT_H_RANGE = (7, 15)
OBJECT_W_RANGE = (7, 11)

# Cavity structure
CAVITY_COUNT_RANGE = (4, 8)
FILL_COLOR_COUNT_RANGE = (3, 7)

# Layout row patterns (weighted choices)
ROW_HEIGHT_CHOICES = [1, 2, 2, 3, 3, 3]
LAYOUT_K_CHOICES_FULL = [1, 2, 2, 3, 3]
LAYOUT_K_CHOICES_SMALL = [1, 2, 2]
ROW_GAP_CHOICES = [1, 1, 2]

# Cavity slot widths per layout type
CENTERED_SLOT_WIDTHS = [3, 3, 5, 5, 7]
PAIR_SLOT_WIDTHS = [2, 2, 3, 3, 4]
SIDE_SLOT_WIDTHS = [2, 2, 3]
MID_SLOT_WIDTHS = [1, 3]
PAIR_AXIS_GAP_CHOICES = [1, 1, 3]
TRIO_MID_SIDE_GAP = 1

# Shape mutation
NONRECT_PROBABILITY = 0.8
MAX_MUTATE_ATTEMPTS = 40

# Color assignment
COLOR_MISMATCH_RATE = 0.10

# Placement
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2

# Multi-source split
SRC_COUNT_CHOICES = [2, 2, 3]
MIN_COLORED_PER_SRC = 2

# Generation retry
MAX_GEN_ATTEMPTS = 400
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
        limit = max(1, (h * w) // 2)
        m = _mutate_shape(m, n_remove_max=limit, symmetric=symmetric)
    return m


def _mirror_shape(m: np.ndarray) -> np.ndarray:
    return m[:, ::-1].copy()


def _gen_dst_src_pair(h: int, w: int, *, symmetric: bool) -> tuple[np.ndarray, np.ndarray]:
    nonrect = random.random() < NONRECT_PROBABILITY
    return (
        _gen_shape(h, w, nonrect=nonrect, symmetric=symmetric),
        _gen_shape(h, w, nonrect=nonrect, symmetric=symmetric),
    )


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
    dst_r = _mirror_shape(dst_l)
    src_r = _mirror_shape(src_l)
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
            "shape_dst": dst_r,
            "shape_src": src_r,
            "pair_id": pair_id,
        }
    )


def _add_trio_slots(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    w_side = random.choice(SIDE_SLOT_WIDTHS)
    w_mid = random.choice(MID_SLOT_WIDTHS)
    dst_l, src_l = _gen_dst_src_pair(row_h, w_side, symmetric=False)
    dst_r = _mirror_shape(dst_l)
    src_r = _mirror_shape(src_l)
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
            "shape_dst": dst_r,
            "shape_src": src_r,
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


def _placed_slot_geometry(
    layout: list[dict], shapes: list[np.ndarray], bg: int
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], tuple[int, int, int, int]] | tuple[None, None]:
    placed_slots = []
    for slot_idx, (slot, sh) in enumerate(zip(layout, shapes)):
        rr, cc = np.where(sh)
        if len(rr) == 0:
            continue
        placed_slots.append((slot_idx, slot["r0"] + rr, slot["col_offset"] + cc))
    if not placed_slots:
        return None, None
    all_rows = np.concatenate([p[1] for p in placed_slots])
    all_cols = np.concatenate([p[2] for p in placed_slots])
    r_min, r_max = int(all_rows.min()), int(all_rows.max())
    c_min, c_max = int(all_cols.min()), int(all_cols.max())
    return placed_slots, (r_min, c_min, r_max, c_max)


def _render_object(
    layout: list[dict], shapes: list[np.ndarray], colors: list[int], bg: int, outline_col: int, *, filled: bool
) -> tuple[np.ndarray | None, np.ndarray | None]:
    placed_slots, bounds = _placed_slot_geometry(layout, shapes, bg)
    if placed_slots is None or bounds is None:
        return None, None
    r_min, c_min, r_max, c_max = bounds
    pad = 2
    h = (r_max - r_min + 1) + 2 * pad
    w = (c_max - c_min + 1) + 2 * pad

    canvas = np.full((h, w), bg, dtype=int)
    fill_mask = np.zeros((h, w), dtype=bool)
    for slot_idx, rows, cols in placed_slots:
        rr = rows - r_min + pad
        cc = cols - c_min + pad
        canvas[rr, cc] = colors[slot_idx]
        fill_mask[rr, cc] = True

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
        return None, None
    cropped = crop_bbox(canvas, b)
    r0b, c0b, r1b, c1b = b
    filled_cropped = filled_grid[r0b : r1b + 1, c0b : c1b + 1]
    return cropped, filled_cropped


def _render_partial_source(
    layout: list[dict],
    shapes: list[np.ndarray],
    colors: list[int],
    bg: int,
    outline_col: int,
    active_slots: set[int],
) -> np.ndarray | None:
    placed_slots, bounds = _placed_slot_geometry(layout, shapes, bg)
    if placed_slots is None or bounds is None:
        return None
    r_min, c_min, r_max, c_max = bounds
    pad = 2
    h = (r_max - r_min + 1) + 2 * pad
    w = (c_max - c_min + 1) + 2 * pad

    canvas = np.full((h, w), bg, dtype=int)
    shape_mask = np.zeros((h, w), dtype=bool)
    active_mask = np.zeros((h, w), dtype=bool)
    for slot_idx, rows, cols in placed_slots:
        rr = rows - r_min + pad
        cc = cols - c_min + pad
        shape_mask[rr, cc] = True
        if slot_idx in active_slots:
            canvas[rr, cc] = colors[slot_idx]
            active_mask[rr, cc] = True

    outline_mask = np.zeros_like(shape_mask)
    outline_mask[1:, :] |= shape_mask[:-1, :]
    outline_mask[:-1, :] |= shape_mask[1:, :]
    outline_mask[:, 1:] |= shape_mask[:, :-1]
    outline_mask[:, :-1] |= shape_mask[:, 1:]
    outline_mask &= ~shape_mask
    canvas[outline_mask] = outline_col

    b = bbox(canvas != bg)
    if b is None:
        return None
    return crop_bbox(canvas, b)


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


def _split_slots_across_sources(n_slots: int) -> list[list[int]]:
    max_sources = min(n_slots // MIN_COLORED_PER_SRC, max(SRC_COUNT_CHOICES))
    if max_sources < 2:
        return []
    if max_sources >= 3 and random.random() < 0.6:
        n_sources = 3
    else:
        n_sources = 2
    sizes = [MIN_COLORED_PER_SRC] * n_sources
    remaining = n_slots - n_sources * MIN_COLORED_PER_SRC
    while remaining > 0:
        sizes[random.randrange(n_sources)] += 1
        remaining -= 1
    indices = list(range(n_slots))
    random.shuffle(indices)
    groups = []
    start = 0
    for size in sizes:
        subset = sorted(indices[start : start + size])
        groups.append(subset)
        start += size
    return groups


def _build_object_family(
    layout: list[dict], colors: list[int], bg: int, outline_col: int
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray] | None:
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

    source_groups = _split_slots_across_sources(n_cavities)
    if not source_groups:
        return None

    partial_sources: list[np.ndarray] = []
    for subset in source_groups:
        src_patch = _render_partial_source(layout, src_shapes, colors, bg, outline_col, set(subset))
        if src_patch is None:
            return None
        if not _is_8connected(src_patch, bg):
            return None
        if _count_4conn_cavities(src_patch, bg, outline_col) != len(subset):
            return None
        if _count_enclosed_bg_holes(src_patch, bg) + _count_4conn_cavities(src_patch, bg, outline_col) != n_cavities:
            return None
        if not _size_in_spec(src_patch):
            return None
        partial_sources.append(src_patch)

    return partial_sources, dst_cropped, dst_filled


def _place_on_canvas(patches: list[np.ndarray], bg: int) -> np.ndarray | None:
    min_h = max(CANVAS_SIZE_RANGE[0], max(int(p.shape[0]) for p in patches) + 2 * PLACEMENT_BORDER)
    min_w = max(CANVAS_SIZE_RANGE[0], max(int(p.shape[1]) for p in patches) + 2 * PLACEMENT_BORDER)
    if min_h > CANVAS_SIZE_RANGE[1] or min_w > CANVAS_SIZE_RANGE[1]:
        return None

    for _ in range(25):
        canvas_h = random.randint(min_h, CANVAS_SIZE_RANGE[1])
        canvas_w = random.randint(min_w, CANVAS_SIZE_RANGE[1])
        canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
        order = list(range(len(patches)))
        random.shuffle(order)
        order.sort(key=lambda i: int(patches[i].shape[0] * patches[i].shape[1]), reverse=True)
        ok = True
        for idx in order:
            patch = patches[idx]
            try:
                r, c = random_free_location(
                    canvas,
                    patch,
                    transparent=bg,
                    border=PLACEMENT_BORDER,
                    padding=PLACEMENT_PADDING,
                )
            except ValueError:
                ok = False
                break
            paste(canvas, patch, r, c, transparent=bg)
        if ok:
            return canvas
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

    built = _build_object_family(layout, colors, bg, outline_col)
    if built is None:
        return None
    srcs, dst, expected = built

    canvas = _place_on_canvas(srcs + [dst], bg)
    if canvas is None:
        return None

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
