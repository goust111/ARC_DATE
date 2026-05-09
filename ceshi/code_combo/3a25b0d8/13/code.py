from __future__ import annotations

# Combo variant of 3a25b0d8 (index 13)
# combo_spec: D4/0+D16/0
# Name: 图例驱动颜色映射
#
# Design analysis
# =============================================================================
# a) Concrete behavior and execution trace
#    Base task:
#    - The canvas contains exactly two large 8-connected outline objects.
#    - src has colored cavity-fills; dst has matching empty holes.
#    - main() matches src subsprites to dst holes by centroid rank after trying
#      rotation / flip, then copies each src color directly into the paired hole.
#
#    Fused variant:
#    - Keep the original src/dst geometry and matching rule intact.
#    - Add a third object: a 2xN horizontal legend strip attached to the canvas
#      top edge or bottom edge.
#    - The legend's top row lists source colors; the bottom row lists the colors
#      they should become at fill time. Column pairing defines the mapping.
#    - Example trace:
#        src colors by rank        = [red, green, red, yellow]
#        legend top row            = [yellow, red, green]
#        legend bottom row         = [maroon, blue, yellow]
#        decoded map               = {yellow->maroon, red->blue, green->yellow}
#        original output colors    = [red, green, red, yellow]
#        fused-variant output      = [blue, yellow, blue, maroon]
#      The only changed step is the final color choice, but now it depends on a
#      visible symbolic object on the canvas edge, so D4 and D16 act together.
#
# b) Complexity calibration
#    - Legend width is tied to the fill-color pool size and kept in the range 3-5.
#      This is large enough to require reading the legend, but still compact
#      enough for humans to decode from 2-4 examples.
#    - Mapping is always a derangement on the used src color pool, so every used
#      src color changes. This prevents degeneration to the original direct-fill
#      behavior.
#    - We preserve the original cavity counts, LR-symmetric packed layouts, and
#      centroid-rank matching so the new concept remains specifically "read the
#      edge legend, then recolor before filling", rather than changing multiple
#      unrelated mechanisms at once.
#
# c) Implicit-assumption audit
#    - Original "_extract_two_objects()" assumed the grid had only src and dst.
#      That premise breaks immediately once the legend is added. The new solver
#      scans all 8-connected components and explicitly identifies the legend.
#    - Original fill loop assumed "fill color = src subsprite color". This is no
#      longer valid; the solver must decode a mapping first.
#    - Original object-identification logic could rely on the two largest objects.
#      That remains safe only because generator guarantees the legend is much
#      smaller than src/dst and has a unique 2xN full-strip signature on an edge.
#    - New solver assumption: exactly one border-touching 2xN fully-filled strip
#      exists and every used src color appears in its top row. Generator enforces
#      both properties.
#
# d) Placement legality
#    - Generator places the legend first, flush with row 0-1 or row H-2:H-1.
#    - src and dst are then placed with the standard padding helper against the
#      already-occupied canvas, so they cannot overlap the legend or merge into it
#      under 8-connectivity.
#    - src and dst still use the original non-overlap / in-bounds guarantees.
#
# e) Visual observability
#    - The legend is visually separate from src/dst because it is tiny, border-
#      attached, and rectangular, while src/dst are large interior outline
#      objects. Humans can spot that it is an instruction strip, not a third task
#      object to transform.
#    - The mapping is directly readable columnwise: top color becomes bottom color.
#      No hidden ordering depends on numeric color ids or non-visual metadata.
#
# f) Visual cleanliness
#    - The legend contains no outline and no background holes, so it looks like a
#      clean 2-row swatch strip rather than a noisy extra object.
#    - Because objects are separated by placement padding, the edge legend does
#      not visually fuse with src/dst.
#
# g) Inferability of color change
#    - Color count is capped at 5, which is small enough for human inspection.
#    - Every legend column is used by at least one src cavity color, so each
#      training sample provides direct evidence that the legend matters.
#    - Since the mapping is a bijection / derangement, multiple holes often reuse
#      the same mapped color, reinforcing the rule instead of introducing sparse,
#      one-off recolorings.
#
# New solver assumptions review
# -----------------------------------------------------------------------------
# - The legend is the only non-bg component whose bbox has height 2, touches the
#   top or bottom edge, and is fully filled (no bg inside its bbox).
# - Top-row legend colors are unique and bottom-row legend colors are unique.
# - Every used src fill color has a legend entry, and direct-fill output differs
#   from mapped output on every generated sample.
# - Src and dst remain the two largest non-legend components.

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match src cavity-fills to dst holes by centroid rank, but recolor each src
    color through a border-attached 2xN legend strip before filling.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

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
        Hs, Ws = sub.shape
        holes = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if bool(r0 == 0 or c0 == 0 or r1 == int(Hs - 1) or c1 == int(Ws - 1)):
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
            subs.append({"cent": cent, "color": int(col)})
        return subs

    def _rank_keys(pts: list[tuple[int, int]]):
        rows = sorted(set(int(r) for r, _c in pts))
        cols = sorted(set(int(c) for _r, c in pts))
        rrank = {r: i for i, r in enumerate(rows)}
        crank = {c: i for i, c in enumerate(cols)}
        keys = [(int(rrank[r]), int(crank[c])) for r, c in pts]
        return keys, rrank, crank

    def _transform_point(
        p: tuple[int, int], H0: int, W0: int, k: int, flip: bool
    ) -> tuple[int, int]:
        r, c = int(p[0]), int(p[1])
        if int(k) == 0:
            rr, cc = int(r), int(c)
            H2, W2 = int(H0), int(W0)
        elif int(k) == 1:
            rr, cc = int(W0 - 1 - c), int(r)
            H2, W2 = int(W0), int(H0)
        elif int(k) == 2:
            rr, cc = int(H0 - 1 - r), int(W0 - 1 - c)
            H2, W2 = int(H0), int(W0)
        else:
            rr, cc = int(c), int(H0 - 1 - r)
            H2, W2 = int(W0), int(H0)
        if bool(flip):
            cc = int(W2 - 1 - cc)
        return int(rr), int(cc)

    def _parse_edge_legend(sub: np.ndarray, box: tuple[int, int, int, int]) -> dict[int, int] | None:
        r0, _c0, r1, _c1 = box
        if sub.shape[0] != 2 or sub.shape[1] < 3:
            return None
        if np.any(sub == int(bg)):
            return None
        touches_top = int(r0) == 0 and int(r1) == 1
        touches_bottom = int(r0) == int(H - 2) and int(r1) == int(H - 1)
        if not (touches_top or touches_bottom):
            return None
        top = [int(v) for v in sub[0, :]]
        bot = [int(v) for v in sub[1, :]]
        if len(set(top)) != len(top) or len(set(bot)) != len(bot):
            return None
        mapping = {int(src_col): int(dst_col) for src_col, dst_col in zip(top, bot)}
        return mapping if mapping else None

    comps = _components(g != int(bg))
    infos = []
    for comp in comps:
        sub = _extract_group_subgrid([comp])
        info = {
            "sub": sub,
            "size": int(comp["size"]),
            "bbox": comp["bbox"],
            "holes": _enclosed_bg_components(sub),
            "legend_map": _parse_edge_legend(sub, comp["bbox"]),
        }
        infos.append(info)

    legend_candidates = [i for i, info in enumerate(infos) if info["legend_map"]]
    legend_idx = None
    if legend_candidates:
        legend_idx = max(
            legend_candidates,
            key=lambda i: (
                len(infos[i]["legend_map"]),
                -int(infos[i]["size"]),
            ),
        )
    color_map = infos[legend_idx]["legend_map"] if legend_idx is not None else {}

    object_infos = [info for i, info in enumerate(infos) if i != legend_idx]
    object_infos.sort(key=lambda d: int(d["size"]), reverse=True)
    assert len(object_infos) >= 2

    objs = [object_infos[0]["sub"], object_infos[1]["sub"]]
    holes_list = [object_infos[0]["holes"], object_infos[1]["holes"]]
    hole_area = [sum(int(h["size"]) for h in hh) for hh in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    dest = objs[dest_i].copy()
    src = objs[src_i].copy()

    dest_holes = _enclosed_bg_components(dest)
    src_subs = _colored_components(src)
    if not dest_holes or not src_subs:
        m = dest != int(bg)
        pts = np.argwhere(m)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        return dest[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

    dest_pts = [h["cent"] for h in dest_holes]
    dest_keys, d_rr, d_cc = _rank_keys(dest_pts)

    best = None
    for k in range(4):
        for flip in [False, True]:
            tpts = [
                _transform_point(s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
                for s in src_subs
            ]
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

    tpts_all = [
        _transform_point(s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
        for s in src_subs
    ]
    _keys, s_rr, s_cc = _rank_keys(tpts_all)

    src_list = []
    for s in src_subs:
        tp = _transform_point(s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
        key = (int(s_rr[int(tp[0])]), int(s_cc[int(tp[1])]))
        src_list.append((key, tp, int(s["color"])))
    src_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    for (_k1, _p1, src_col), (_k2, _p2, h) in zip(src_list, dest_list):
        fill_col = int(color_map.get(int(src_col), int(src_col)))
        for rr, cc in h["pts"]:
            dest[int(rr), int(cc)] = int(fill_col)

    m = dest != int(bg)
    pts = np.argwhere(m)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return dest[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


# --------------- generator ---------------

CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)

CAVITY_COUNT_RANGE = (4, 9)
FILL_COLOR_COUNT_RANGE = (3, 5)

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

MAX_GEN_ATTEMPTS = 300
MAX_LAYOUT_R_BUDGET = 24


def _random_rect_shape(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _mutate_shape(mask: np.ndarray, *, n_remove_max: int, symmetric: bool = False) -> np.ndarray:
    m = mask.copy()
    H, W = m.shape
    n_remove = random.randint(1, max(1, n_remove_max))
    tried = 0
    removed = 0
    while removed < n_remove and tried < MAX_MUTATE_ATTEMPTS:
        tried += 1
        cand = []
        for r in range(H):
            for c in range(W):
                if not m[r, c]:
                    continue
                nb = 0
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and m[nr, nc]:
                        nb += 1
                if nb <= 2:
                    cand.append((r, c))
        if not cand:
            break
        r, c = random.choice(cand)
        to_remove = [(r, c)]
        if symmetric:
            mc = W - 1 - c
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
    dst_L, src_L = _gen_dst_src_pair(row_h, w, symmetric=False)
    dst_R = _mirror_shape(dst_L)
    src_R = _mirror_shape(src_L)
    axis_gap = random.choice(PAIR_AXIS_GAP_CHOICES)
    if axis_gap % 2 == 0:
        axis_gap += 1
    half = axis_gap // 2
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -(half + w),
            "shape_dst": dst_L,
            "shape_src": src_L,
            "pair_id": pair_id,
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": half + 1,
            "shape_dst": dst_R,
            "shape_src": src_R,
            "pair_id": pair_id,
        }
    )


def _add_trio_slots(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    w_side = random.choice(SIDE_SLOT_WIDTHS)
    w_mid = random.choice(MID_SLOT_WIDTHS)
    dst_L, src_L = _gen_dst_src_pair(row_h, w_side, symmetric=False)
    dst_R = _mirror_shape(dst_L)
    src_R = _mirror_shape(src_L)
    dst_M, src_M = _gen_dst_src_pair(row_h, w_mid, symmetric=True)
    mid_half = w_mid // 2
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -(mid_half + w_side + TRIO_MID_SIDE_GAP),
            "shape_dst": dst_L,
            "shape_src": src_L,
            "pair_id": pair_id,
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -mid_half,
            "shape_dst": dst_M,
            "shape_src": src_M,
            "pair_id": pair_id + 1,
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": mid_half + TRIO_MID_SIDE_GAP + 1,
            "shape_dst": dst_R,
            "shape_src": src_R,
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
) -> tuple[np.ndarray | None, np.ndarray | None]:
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

    pad = 2
    H = (r_max - r_min + 1) + 2 * pad
    W = (c_max - c_min + 1) + 2 * pad

    canvas = np.full((H, W), bg, dtype=int)
    fill_mask = np.zeros((H, W), dtype=bool)
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


def _count_4conn_cavities(filled_grid: np.ndarray, bg: int, outline_col: int) -> int:
    mask = (filled_grid != bg) & (filled_grid != outline_col)
    _, n = label_mask(mask, connectivity=4)
    return int(n)


def _count_enclosed_bg_holes(empty_grid: np.ndarray, bg: int) -> int:
    mask = empty_grid == bg
    lab, n = label_mask(mask, connectivity=4)
    H, W = empty_grid.shape
    count = 0
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == cid)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r0 > 0 and c0 > 0 and r1 < H - 1 and c1 < W - 1:
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


def _pair_id_count(layout: list[dict]) -> int:
    return len({int(slot["pair_id"]) for slot in layout})


def _ensure_color_coverage(colors: list[int], fill_pool: list[int]) -> list[int]:
    out = list(colors)
    idxs = list(range(len(out)))
    random.shuffle(idxs)
    cols = list(fill_pool)
    random.shuffle(cols)
    for idx, col in zip(idxs, cols):
        out[int(idx)] = int(col)
    return out


def _pick_bg_and_palette() -> tuple[int, list[int]]:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    return bg, palette


def _make_derangement(colors: list[int]) -> dict[int, int]:
    src = list(colors)
    dst = list(colors)
    while True:
        random.shuffle(dst)
        if all(int(a) != int(b) for a, b in zip(src, dst)):
            return {int(a): int(b) for a, b in zip(src, dst)}


def _build_edge_legend(color_map: dict[int, int]) -> np.ndarray:
    pairs = list(color_map.items())
    random.shuffle(pairs)
    legend = np.zeros((2, len(pairs)), dtype=int)
    for i, (src_col, dst_col) in enumerate(pairs):
        legend[0, int(i)] = int(src_col)
        legend[1, int(i)] = int(dst_col)
    return legend


def _build_object_pair(
    layout: list[dict],
    src_colors: list[int],
    dst_colors: list[int],
    bg: int,
    outline_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n_cavities = len(layout)
    dst_shapes = [slot["shape_dst"] for slot in layout]
    src_shapes = [slot["shape_src"] for slot in layout]

    dst_cropped, dst_filled = _render_object(
        layout, dst_shapes, dst_colors, bg, outline_col, filled=False
    )
    if dst_cropped is None:
        return None
    if not _is_8connected(dst_cropped, bg):
        return None
    if _count_enclosed_bg_holes(dst_cropped, bg) != n_cavities:
        return None
    if not _size_in_spec(dst_cropped):
        return None

    src_cropped, _ = _render_object(layout, src_shapes, src_colors, bg, outline_col, filled=True)
    if src_cropped is None:
        return None
    if not _is_8connected(src_cropped, bg):
        return None
    if _count_4conn_cavities(src_cropped, bg, outline_col) != n_cavities:
        return None
    if not _size_in_spec(src_cropped):
        return None

    return src_cropped, dst_cropped, dst_filled


def _place_on_canvas(src: np.ndarray, dst: np.ndarray, legend: np.ndarray, bg: int) -> np.ndarray | None:
    canvas_h = random.randint(*CANVAS_SIZE_RANGE)
    canvas_w = random.randint(*CANVAS_SIZE_RANGE)
    if legend.shape[1] > canvas_w:
        return None

    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    legend_top = random.choice([True, False])
    legend_r = 0 if legend_top else canvas_h - legend.shape[0]

    c_lo = 1 if canvas_w - legend.shape[1] >= 3 else 0
    c_hi = canvas_w - legend.shape[1] - 1 if canvas_w - legend.shape[1] >= 3 else canvas_w - legend.shape[1]
    if c_hi < c_lo:
        return None
    legend_c = random.randint(int(c_lo), int(c_hi))
    paste(canvas, legend, int(legend_r), int(legend_c), transparent=bg)

    for patch in [src, dst]:
        try:
            r, c = random_free_location(
                canvas,
                patch,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
            )
        except ValueError:
            return None
        paste(canvas, patch, int(r), int(c), transparent=bg)
    return canvas


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline_col = palette[0]

    layout = _design_layout(random.randint(*CAVITY_COUNT_RANGE))
    if len(layout) < CAVITY_COUNT_RANGE[0]:
        return None
    if _pair_id_count(layout) < FILL_COLOR_COUNT_RANGE[0]:
        return None

    max_cols = min(FILL_COLOR_COUNT_RANGE[1], _pair_id_count(layout), len(palette) - 1)
    if max_cols < FILL_COLOR_COUNT_RANGE[0]:
        return None
    n_fill_cols = random.randint(FILL_COLOR_COUNT_RANGE[0], max_cols)
    fill_pool = palette[1 : 1 + n_fill_cols]

    src_colors = _assign_pair_colors(layout, fill_pool)
    src_colors = _ensure_color_coverage(src_colors, fill_pool)
    color_map = _make_derangement(fill_pool)
    dst_colors = [int(color_map[int(col)]) for col in src_colors]

    built = _build_object_pair(layout, src_colors, dst_colors, bg, outline_col)
    if built is None:
        return None
    src, dst, expected = built
    legend = _build_edge_legend(color_map)

    canvas = _place_on_canvas(src, dst, legend, bg)
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
