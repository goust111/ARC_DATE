# Variant of 3a25b0d8
# Dimension: D9 — 颜色分配规则 (meta)
# Change: 镜像对空腔默认共享同一颜色，10%的空腔允许不匹配；使用 3-7 种填充色... → 颜色按行分配

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


# Design analysis for the D9 variant
# a) Concrete behavior
#    Original task: each src cavity contributes its own color after centroid-rank matching.
#    Variant: cavities are still matched by geometry, but color is inferred per cavity-row.
#    In the src object, each cavity row exposes one colored clue cavity; that clue color applies
#    to all cavities in the same row-rank, while different row-ranks use different colors. In the
#    dst object, every hole in row-rank 0 gets the color of src row-rank 0, every hole in
#    row-rank 1 gets the color of src row-rank 1, etc. Example trace:
#      - Original row layout: top row has left/right cavities in colors 2 and 7, bottom row has
#        center cavity in color 4. Solver copies colors cavity-by-cavity.
#      - Variant row layout: top row has one clue cavity colored 2 and one sibling cavity left
#        empty; bottom row has one clue cavity colored 4. Also, src and dst may disagree on
#        left/right spacing inside the same row. Solver first groups src clue cavities by
#        transformed row-rank, learns {row 0 -> 2, row 1 -> 4}, then fills all dst holes in each
#        row with that row color. Within-row column alignment is no longer a reliable signal and is
#        intentionally ignored.
#    The geometric rank matching still matters at the row level, but the coloring stage now uses
#    row groups instead of cavity identities.
# b) Complexity calibration
#    A weak version of this variant would allow many singleton rows, making the new rule hard to
#    notice and visually close to the original. To keep the D9 change observable, the generator
#    biases toward 3-5 cavity rows and requires at least two rows with multiple cavities via
#    curated row plans. This keeps repeated same-color evidence inside each sample while still
#    allowing mixed row widths and 1/2/3 cavities per row.
# c) Original assumptions that break
#    - `_colored_components()` is still useful, but now it extracts the visible row clues rather
#      than a full set of cavity colors. The solver must combine those clues with src empty holes
#      to recover the full row structure.
#    - The original final zip of `src_list` and `dest_list` assumes each cavity keeps an
#      independent color. That is no longer the intended semantics. If left unchanged, the solver
#      would still work on many generated samples only because same-row cavities happen to share a
#      color, but it would not be explicitly solving the new row-level rule.
#    - Therefore the last stage must change from "copy each matched cavity's color" to
#      "infer a color per transformed src row-rank, then paint all dst holes in that row-rank."
#    - To prevent degeneration, the generator also stops preserving exact per-cavity column ranks
#      between src and dst. The old exact-key matcher is no longer sufficient.
# d) Placement legality
#    Placement stays identical to the original: src and dst are separate 8-connected objects,
#    placed on a 22-30 canvas with non-overlap enforced by `random_free_location(..., padding=2)`.
#    Since row grouping is internal to each object and the solver still extracts the two largest
#    8-connected components, no extra spatial relation between src and dst is needed.
# e) Visual observability
#    A human should see one clue color per src row and a row-wise recoloring in the output. The
#    generator uses distinct colors per row, not per cavity, and prefers plans with multiple
#    multi-hole rows so 2-4 examples are enough to notice the "one clue colors the whole row"
#    pattern.
#    No invisible feature (like numeric color id ordering) is used in the reasoning.
# f) Visual cleanliness
#    The variant does not add extra markers. Objects remain outline-only vs partly-filled outline,
#    which is still visually clean. The only extra structure is a sparse row clue in src, which is
#    easier to parse than per-cavity arbitrary colors.
# g) Color inferability
#    Distinct fill colors equal the number of cavity rows (typically 3-5), so the palette is small.
#    Each row contributes exactly one visible clue color in src, and the output repeats that color
#    across the whole row. This avoids the original mirror-pair mismatch noise because such noise
#    would obscure the new row rule.
# 9) New solver assumptions introduced by this variant
#    - After the chosen transform, src and dst share the same row-rank structure, but they do not
#      need to share the same within-row column ranks.
#    - Every transformed src row-rank contains exactly one visible clue cavity.
#    - Every dst row-rank exists in the src row-color map.
#    The generator guarantees these by building src and dst from the same row-annotated layout and
#    assigning one distinct clue color per row_id while allowing per-object row spacing to differ.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill dst holes using row-level colors inferred from src, then tight-crop.

    Input structure remains the same as 3a25b0d8:
    - Two similar 8-connected objects share one outline color.
    - Source contains one colored clue cavity per row plus other empty holes.
    - Destination contains empty holes.

    D9 variant rule:
    - Match src cavity-row arrangement to dst hole rows by centroid rank.
    - Infer one color per src clue row (same row-rank => same color).
    - Fill all dst holes in each row-rank with that row's color.
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

    def _extract_two_objects() -> list[list[dict]]:
        comps = _components(g != int(bg))
        if len(comps) < 2:
            return [comps]
        return [[comps[0]], [comps[1]]]

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
            subs.append({"cent": cent, "color": int(col)})
        return subs

    def _rank_keys(pts: list[tuple[int, int]]):
        rows = sorted(set(int(r) for r, _c in pts))
        cols = sorted(set(int(c) for _r, c in pts))
        rrank = {r: i for i, r in enumerate(rows)}
        crank = {c: i for i, c in enumerate(cols)}
        keys = [(int(rrank[r]), int(crank[c])) for r, c in pts]
        return keys, rrank, crank

    def _row_signature(pts: list[tuple[int, int]]):
        keys, rrank, crank = _rank_keys(pts)
        counts = [0] * len(rrank)
        for row_key, _col_key in keys:
            counts[int(row_key)] += 1
        return tuple(counts), keys, rrank, crank

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

    def _tight_crop_non_bg(sub: np.ndarray) -> np.ndarray:
        m = sub != int(bg)
        pts = np.argwhere(m)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        return sub[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

    groups = _extract_two_objects()
    assert not (len(groups) < 2)
    objs = [_extract_group_subgrid(grp) for grp in groups]

    holes_list = [_enclosed_bg_components(o) for o in objs]
    hole_area = [sum(int(h["size"]) for h in hh) for hh in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    dest = objs[dest_i].copy()
    src = objs[src_i].copy()

    dest_holes = _enclosed_bg_components(dest)
    src_empty_holes = _enclosed_bg_components(src)
    src_subs = _colored_components(src)
    if not dest_holes or not src_subs:
        return _tight_crop_non_bg(dest)

    dest_pts = [h["cent"] for h in dest_holes]
    dest_row_sig, dest_keys, d_rr, d_cc = _row_signature(dest_pts)
    src_all_pts = [s["cent"] for s in src_subs] + [h["cent"] for h in src_empty_holes]
    if not src_all_pts:
        return _tight_crop_non_bg(dest)

    best = None
    for k in range(4):
        for flip in [False, True]:
            tpts = [
                _transform_point(p, int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
                for p in src_all_pts
            ]
            t_row_sig, _tkeys, _s_rr, _s_cc = _row_signature(tpts)
            if t_row_sig == dest_row_sig:
                best = (int(k), bool(flip))
                break
        if best is not None:
            break
    if best is None:
        best = (0, False)
    k, flip = best

    dest_rows: dict[int, list[dict]] = {}
    dest_list = []
    for h in dest_holes:
        row_key = int(d_rr[int(h["cent"][0])])
        col_key = int(d_cc[int(h["cent"][1])])
        dest_rows.setdefault(row_key, []).append(h)
        dest_list.append(((row_key, col_key), h["cent"], h))
    dest_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    tpts_all = [
        _transform_point(s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
        for s in src_subs
    ]
    _row_sig, _keys, s_rr, s_cc = _row_signature(tpts_all)

    row_to_colors: dict[int, list[int]] = {}
    src_list = []
    for s, tp in zip(src_subs, tpts_all):
        row_key = int(s_rr[int(tp[0])])
        col_key = int(s_cc[int(tp[1])])
        color = int(s["color"])
        row_to_colors.setdefault(row_key, []).append(color)
        src_list.append(((row_key, col_key), tp, color))
    src_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    row_color_map = {
        row_key: int(Counter(cols).most_common(1)[0][0]) for row_key, cols in row_to_colors.items()
    }
    rowwise_ok = (
        set(row_color_map) == set(dest_rows)
        and all(len(set(cols)) == 1 for cols in row_to_colors.values())
    )

    if rowwise_ok:
        for row_key, holes in dest_rows.items():
            fill_color = int(row_color_map[row_key])
            for h in holes:
                for rr, cc in h["pts"]:
                    dest[int(rr), int(cc)] = fill_color
    else:
        # Robust fallback: preserve the original cavity-wise copying logic if a non-variant
        # input violates the new row-color assumption.
        for (_k1, _p1, col), (_k2, _p2, h) in zip(src_list, dest_list):
            for rr, cc in h["pts"]:
                dest[int(rr), int(cc)] = int(col)

    return _tight_crop_non_bg(dest)


# --------------- generator for the D9 row-color variant ---------------

CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)

# Cavity structure: minimum raised to 5 so the row-color cue is visually supported.
CAVITY_COUNT_RANGE = (5, 9)
FILL_COLOR_COUNT_RANGE = (3, 5)

# Layout row plans: each integer is the number of cavities in that row.
# Plans are curated so multiple rows usually contain repeated same-row evidence.
ROW_PLAN_CHOICES = [
    [2, 1, 2],
    [2, 2, 1],
    [1, 2, 2],
    [2, 2, 2],
    [3, 1, 2],
    [2, 1, 3],
    [3, 2, 1],
    [1, 2, 3],
    [2, 3, 1],
    [2, 3, 2],
    [3, 2, 2],
    [2, 2, 3],
    [2, 2, 1, 2],
    [2, 1, 2, 2],
    [1, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 1, 2, 1, 2],
    [1, 2, 1, 2, 1],
    [2, 2, 1, 2, 2],
    [1, 2, 2, 1, 2],
]
ROW_HEIGHT_CHOICES = [1, 2, 2, 3, 3]
ROW_HEIGHT_CHOICES_DENSE = [1, 1, 2, 2]
ROW_GAP_CHOICES = [1, 1, 2]
ROW_GAP_CHOICES_DENSE = [1, 1]

CENTERED_SLOT_WIDTHS = [3, 3, 5, 5, 7]
PAIR_SLOT_WIDTHS = [2, 2, 3, 3, 4]
SIDE_SLOT_WIDTHS = [2, 2, 3]
MID_SLOT_WIDTHS = [1, 3]
PAIR_AXIS_GAP_CHOICES = [1, 1, 3]
TRIO_MID_SIDE_GAP = 1

NONRECT_PROBABILITY = 0.8
MAX_MUTATE_ATTEMPTS = 40

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2

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


def _add_centered_slot(
    slots: list[dict], cur_r: int, row_id: int, pair_id: int, row_h: int
) -> int:
    w_dst = random.choice(CENTERED_SLOT_WIDTHS)
    w_src = random.choice(CENTERED_SLOT_WIDTHS)
    dst = _gen_shape(row_h, w_dst, nonrect=(random.random() < NONRECT_PROBABILITY), symmetric=True)
    src = _gen_shape(row_h, w_src, nonrect=(random.random() < NONRECT_PROBABILITY), symmetric=True)
    slots.append(
        {
            "r0": cur_r,
            "col_offset_dst": -(w_dst // 2),
            "col_offset_src": -(w_src // 2),
            "shape_dst": dst,
            "shape_src": src,
            "pair_id": pair_id,
            "row_id": row_id,
        }
    )
    return pair_id + 1


def _add_mirror_pair_slots(
    slots: list[dict], cur_r: int, row_id: int, pair_id: int, row_h: int
) -> int:
    w_dst = random.choice(PAIR_SLOT_WIDTHS)
    w_src = random.choice(PAIR_SLOT_WIDTHS)
    dst_l = _gen_shape(row_h, w_dst, nonrect=(random.random() < NONRECT_PROBABILITY), symmetric=False)
    src_l = _gen_shape(row_h, w_src, nonrect=(random.random() < NONRECT_PROBABILITY), symmetric=False)
    dst_r = _mirror_shape(dst_l)
    src_r = _mirror_shape(src_l)
    axis_gap_dst = random.choice(PAIR_AXIS_GAP_CHOICES)
    if axis_gap_dst % 2 == 0:
        axis_gap_dst += 1
    half_dst = axis_gap_dst // 2
    axis_gap_src = random.choice(PAIR_AXIS_GAP_CHOICES)
    if axis_gap_src % 2 == 0:
        axis_gap_src += 1
    half_src = axis_gap_src // 2
    slots.append(
        {
            "r0": cur_r,
            "col_offset_dst": -(half_dst + w_dst),
            "col_offset_src": -(half_src + w_src),
            "shape_dst": dst_l,
            "shape_src": src_l,
            "pair_id": pair_id,
            "row_id": row_id,
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset_dst": half_dst + 1,
            "col_offset_src": half_src + 1,
            "shape_dst": dst_r,
            "shape_src": src_r,
            "pair_id": pair_id,
            "row_id": row_id,
        }
    )
    return pair_id + 1


def _add_trio_slots(
    slots: list[dict], cur_r: int, row_id: int, pair_id: int, row_h: int
) -> int:
    w_side_dst = random.choice(SIDE_SLOT_WIDTHS)
    w_mid_dst = random.choice(MID_SLOT_WIDTHS)
    w_side_src = random.choice(SIDE_SLOT_WIDTHS)
    w_mid_src = random.choice(MID_SLOT_WIDTHS)
    dst_l = _gen_shape(
        row_h, w_side_dst, nonrect=(random.random() < NONRECT_PROBABILITY), symmetric=False
    )
    src_l = _gen_shape(
        row_h, w_side_src, nonrect=(random.random() < NONRECT_PROBABILITY), symmetric=False
    )
    dst_r = _mirror_shape(dst_l)
    src_r = _mirror_shape(src_l)
    dst_m = _gen_shape(
        row_h, w_mid_dst, nonrect=(random.random() < NONRECT_PROBABILITY), symmetric=True
    )
    src_m = _gen_shape(
        row_h, w_mid_src, nonrect=(random.random() < NONRECT_PROBABILITY), symmetric=True
    )
    mid_half_dst = w_mid_dst // 2
    mid_half_src = w_mid_src // 2
    slots.append(
        {
            "r0": cur_r,
            "col_offset_dst": -(mid_half_dst + w_side_dst + TRIO_MID_SIDE_GAP),
            "col_offset_src": -(mid_half_src + w_side_src + TRIO_MID_SIDE_GAP),
            "shape_dst": dst_l,
            "shape_src": src_l,
            "pair_id": pair_id,
            "row_id": row_id,
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset_dst": -mid_half_dst,
            "col_offset_src": -mid_half_src,
            "shape_dst": dst_m,
            "shape_src": src_m,
            "pair_id": pair_id + 1,
            "row_id": row_id,
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset_dst": mid_half_dst + TRIO_MID_SIDE_GAP + 1,
            "col_offset_src": mid_half_src + TRIO_MID_SIDE_GAP + 1,
            "shape_dst": dst_r,
            "shape_src": src_r,
            "pair_id": pair_id,
            "row_id": row_id,
        }
    )
    return pair_id + 2


def _design_layout() -> list[dict] | None:
    for _ in range(200):
        plan = list(random.choice(ROW_PLAN_CHOICES))
        n_rows = len(plan)
        if not (FILL_COLOR_COUNT_RANGE[0] <= n_rows <= FILL_COLOR_COUNT_RANGE[1]):
            continue
        if not (CAVITY_COUNT_RANGE[0] <= sum(plan) <= CAVITY_COUNT_RANGE[1]):
            continue
        if sum(1 for k in plan if k >= 2) < 2:
            continue

        slots: list[dict] = []
        cur_r = 0
        pair_id = 0
        row_height_choices = ROW_HEIGHT_CHOICES_DENSE if n_rows >= 5 else ROW_HEIGHT_CHOICES
        row_gap_choices = ROW_GAP_CHOICES_DENSE if n_rows >= 5 else ROW_GAP_CHOICES
        ok = True
        for row_id, k in enumerate(plan):
            row_h = random.choice(row_height_choices)
            if k == 1:
                pair_id = _add_centered_slot(slots, cur_r, row_id, pair_id, row_h)
            elif k == 2:
                pair_id = _add_mirror_pair_slots(slots, cur_r, row_id, pair_id, row_h)
            else:
                pair_id = _add_trio_slots(slots, cur_r, row_id, pair_id, row_h)
            cur_r += row_h
            if row_id != n_rows - 1:
                cur_r += random.choice(row_gap_choices)
            if cur_r > MAX_LAYOUT_R_BUDGET:
                ok = False
                break
        if ok:
            return slots
    return None


def _render_object(
    layout: list[dict],
    shapes: list[np.ndarray],
    colors: list[int],
    bg: int,
    outline_col: int,
    *,
    offset_key: str,
    visible_mask: list[bool],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    placed_slots = []
    for slot_idx, (slot, sh) in enumerate(zip(layout, shapes)):
        rr, cc = np.where(sh)
        if len(rr) == 0:
            continue
        placed_slots.append((slot_idx, slot["r0"] + rr, int(slot[offset_key]) + cc))
    if not placed_slots:
        return None, None

    all_rows = np.concatenate([p[1] for p in placed_slots])
    all_cols = np.concatenate([p[2] for p in placed_slots])
    r_min, r_max = int(all_rows.min()), int(all_rows.max())
    c_min, c_max = int(all_cols.min()), int(all_cols.max())

    pad = 2
    h = (r_max - r_min + 1) + 2 * pad
    w = (c_max - c_min + 1) + 2 * pad

    canvas = np.full((h, w), bg, dtype=int)
    full_canvas = np.full((h, w), bg, dtype=int)
    structure_mask = np.zeros((h, w), dtype=bool)
    for slot_idx, rows, cols in placed_slots:
        rr = rows - r_min + pad
        cc = cols - c_min + pad
        structure_mask[rr, cc] = True
        full_canvas[rr, cc] = colors[slot_idx]
        if visible_mask[slot_idx]:
            canvas[rr, cc] = colors[slot_idx]

    outline_mask = np.zeros_like(structure_mask)
    outline_mask[1:, :] |= structure_mask[:-1, :]
    outline_mask[:-1, :] |= structure_mask[1:, :]
    outline_mask[:, 1:] |= structure_mask[:, :-1]
    outline_mask[:, :-1] |= structure_mask[:, 1:]
    outline_mask &= ~structure_mask
    canvas[outline_mask] = outline_col
    full_canvas[outline_mask] = outline_col

    b = bbox(canvas != bg)
    if b is None:
        return None, None
    cropped = crop_bbox(canvas, b)
    r0b, c0b, r1b, c1b = b
    filled_cropped = full_canvas[r0b : r1b + 1, c0b : c1b + 1]
    return cropped, filled_cropped


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


def _outline_col_local(sub: np.ndarray, bg: int) -> int:
    cnt = Counter(map(int, sub.flatten()))
    opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
    return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)


def _hole_centroids(sub: np.ndarray, bg: int) -> list[tuple[int, int]]:
    mask = sub == int(bg)
    lab, n = label_mask(mask, connectivity=4)
    h, w = sub.shape
    pts_out: list[tuple[int, int]] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        pts_out.append((int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean())))))
    return pts_out


def _colored_centroids(sub: np.ndarray, bg: int) -> list[tuple[int, int]]:
    outc = _outline_col_local(sub, bg)
    mask = (sub != int(bg)) & (sub != int(outc))
    lab, n = label_mask(mask, connectivity=4)
    pts_out: list[tuple[int, int]] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        pts_out.append((int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean())))))
    return pts_out


def _rank_keys_global(pts: list[tuple[int, int]]):
    rows = sorted(set(int(r) for r, _c in pts))
    cols = sorted(set(int(c) for _r, c in pts))
    rrank = {r: i for i, r in enumerate(rows)}
    crank = {c: i for i, c in enumerate(cols)}
    keys = [(int(rrank[r]), int(crank[c])) for r, c in pts]
    return keys, rrank, crank


def _row_signature_from_pts(pts: list[tuple[int, int]]) -> tuple[int, ...]:
    keys, rrank, _crank = _rank_keys_global(pts)
    counts = [0] * len(rrank)
    for row_key, _col_key in keys:
        counts[int(row_key)] += 1
    return tuple(counts)


def _variant_signal_active(src: np.ndarray, dst: np.ndarray, bg: int) -> bool:
    src_pts = _colored_centroids(src, bg)
    dst_pts = _hole_centroids(dst, bg)
    if len(src_pts) >= len(dst_pts):
        return False
    src_row_sig = _row_signature_from_pts(src_pts)
    dst_row_sig = _row_signature_from_pts(dst_pts)
    if len(src_row_sig) != len(dst_row_sig):
        return False
    if any(v != 1 for v in src_row_sig):
        return False
    return max(dst_row_sig, default=0) >= 2


def _make_src_cue_mask(layout: list[dict]) -> list[bool]:
    cue_mask = [False] * len(layout)
    by_row: dict[int, list[int]] = {}
    for idx, slot in enumerate(layout):
        by_row.setdefault(int(slot["row_id"]), []).append(idx)
    for indices in by_row.values():
        cue_mask[random.choice(indices)] = True
    return cue_mask


def _assign_row_colors(layout: list[dict], fill_pool: list[int]) -> list[int]:
    row_ids: list[int] = []
    for slot in layout:
        row_id = int(slot["row_id"])
        if row_id not in row_ids:
            row_ids.append(row_id)
    chosen = random.sample(fill_pool, len(row_ids))
    random.shuffle(chosen)
    row_color_map = dict(zip(row_ids, chosen))
    return [int(row_color_map[int(slot["row_id"])]) for slot in layout]


def _pick_bg_and_palette() -> tuple[int, list[int]]:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    return bg, palette


def _build_object_pair(
    layout: list[dict], colors: list[int], bg: int, outline_col: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n_cavities = len(layout)
    n_rows = len({int(slot["row_id"]) for slot in layout})
    dst_shapes = [slot["shape_dst"] for slot in layout]
    src_shapes = [slot["shape_src"] for slot in layout]
    dst_visible_mask = [False] * n_cavities
    src_visible_mask = _make_src_cue_mask(layout)

    dst_cropped, dst_filled = _render_object(
        layout,
        dst_shapes,
        colors,
        bg,
        outline_col,
        offset_key="col_offset_dst",
        visible_mask=dst_visible_mask,
    )
    if dst_cropped is None:
        return None
    if not _is_8connected(dst_cropped, bg):
        return None
    if _count_enclosed_bg_holes(dst_cropped, bg) != n_cavities:
        return None
    if not _size_in_spec(dst_cropped):
        return None

    src_cropped, _ = _render_object(
        layout,
        src_shapes,
        colors,
        bg,
        outline_col,
        offset_key="col_offset_src",
        visible_mask=src_visible_mask,
    )
    if src_cropped is None:
        return None
    if not _is_8connected(src_cropped, bg):
        return None
    if _count_4conn_cavities(src_cropped, bg, outline_col) != n_rows:
        return None
    if not _size_in_spec(src_cropped):
        return None
    if not _variant_signal_active(src_cropped, dst_cropped, bg):
        return None

    return src_cropped, dst_cropped, dst_filled


def _place_on_canvas(src: np.ndarray, dst: np.ndarray, bg: int) -> np.ndarray | None:
    canvas_h = random.randint(*CANVAS_SIZE_RANGE)
    canvas_w = random.randint(*CANVAS_SIZE_RANGE)
    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    try:
        r, c = random_free_location(
            canvas, src, transparent=bg, border=PLACEMENT_BORDER, padding=PLACEMENT_PADDING
        )
    except ValueError:
        return None
    paste(canvas, src, r, c, transparent=bg)
    try:
        r, c = random_free_location(
            canvas, dst, transparent=bg, border=PLACEMENT_BORDER, padding=PLACEMENT_PADDING
        )
    except ValueError:
        return None
    paste(canvas, dst, r, c, transparent=bg)
    return canvas


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline_col = palette[0]

    layout = _design_layout()
    if layout is None:
        return None
    row_ids = sorted({int(slot["row_id"]) for slot in layout})
    n_fill_cols = len(row_ids)
    if not (FILL_COLOR_COUNT_RANGE[0] <= n_fill_cols <= FILL_COLOR_COUNT_RANGE[1]):
        return None
    if len(palette) - 1 < n_fill_cols:
        return None
    fill_pool = palette[1 : 1 + n_fill_cols]
    colors = _assign_row_colors(layout, fill_pool)

    built = _build_object_pair(layout, colors, bg, outline_col)
    if built is None:
        return None
    src, dst, expected = built

    canvas = _place_on_canvas(src, dst, bg)
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
