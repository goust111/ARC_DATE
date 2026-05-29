# Combo variant of 3a25b0d8 (index 29)
# combo_spec: D3/0+D16/2+D18/2
# Name: 预填色dst+控制像素+选择性填充

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, get_bg, label_mask, paste, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for the fused variant
# -----------------------------------------------------------------------------
#
# fused_rule:
#   keep:
#     - there are still exactly two main 8-connected objects: src and dst
#     - src still contains one colored subsprite per cavity
#     - dst and src are still matched by centroid rank after optional rigid transform
#     - output is still a tight crop of the completed dst object
#   add_1_D3:
#     - dst is no longer purely outline+bg holes: a strict subset of the holes is already
#       filled with the correct final color in the input
#     - solver must detect those prefilled interior components and treat them as already
#       solved slots rather than as src-like evidence
#   add_2_D16:
#     - the outer canvas border contains 1-pixel control markers
#     - each control pixel color is one of the src fill colors
#     - the set of border-marker colors is the "allowed colors" legend
#   add_3_D18:
#     - even among still-empty dst slots, only rectangular holes are eligible
#     - irregular holes remain bg in the output
#   final_behavior:
#     - build the full dst slot template from:
#         enclosed empty bg holes + enclosed prefilled colored components
#     - match src subsprites to dst slots by centroid-rank pattern
#     - for each matched dst slot:
#         if slot is already filled: leave it alone
#         else if slot hole is rectangular AND matched color is present on the border:
#             fill it
#         else:
#             keep it empty
#
# concrete_trace:
#   - suppose dst has five slots ordered by centroid rank:
#       s1 red rectangle empty
#       s2 green rectangle already prefilled
#       s3 blue L-shape empty
#       s4 yellow rectangle empty
#       s5 red staircase empty
#   - suppose border controls are {red, yellow}
#   - original task would fill s1..s5 from src
#   - D3-only would skip s2 but still fill every other empty slot
#   - D18-only would fill s1 and s4, skip s3 and s5, but would also fill any other
#     rectangular color whether or not the border authorizes it
#   - fused task does:
#       s1 fill (empty + rectangular + red is controlled)
#       s2 keep as is (already filled)
#       s3 skip (controlled color absent and non-rectangular anyway)
#       s4 fill (empty + rectangular + yellow is controlled)
#       s5 skip (red is controlled but the hole is non-rectangular)
#   - this makes all three added mechanisms visible in one sample:
#       prefilled slots, color-gated empty slots, and shape-gated empty slots
#
# complexity_calibration:
#   - control pixels encode only a small color subset (typically 2 colors), so the legend
#     is readable from 2-4 examples and does not explode into arbitrary many cases
#   - generator enforces both:
#       * at least one rectangular hole blocked only by control-color absence
#       * at least one active-color hole blocked only by non-rectangular shape
#     so D16 and D18 both matter materially
#   - prefilled holes are drawn only from the final-eligible subset; this prevents noisy
#     contradictions like an "inactive" color already being filled in dst
#
# hidden_assumption_audit_for_main:
#   - original `top 2 components are the objects` still works because controls are single
#     border pixels and far smaller than src/dst; generator guarantees this
#   - original `dest = object with more bg-hole evidence` still works only if dst keeps at
#     least one empty hole; generator guarantees at least one eligible-but-unfilled slot
#   - original `_colored_components(dest)` was invalid because dst now contains its own
#     colored interiors; fused solver uses them as destination slots, not as source data
#   - original unconditional fill loop is invalid; fused solver gates on:
#       `slot kind == empty`, `slot is rectangular`, `matched color in control set`
#   - original most-common-nonbg outline heuristic can fail when dst is partially filled;
#     border-based outline detection is used instead
#   - new assumption: border non-bg pixels are controls, not object pixels
#     generator guarantees objects stay at least 1 cell away from the canvas edge
#
# placement_legality:
#   - src and dst are placed first with `random_free_location`, so they do not overlap or
#     leave the canvas
#   - control pixels are added only on the outer border after object placement, using bg
#     cells that are separated from the objects by the mandatory border margin
#
# visual_observability_and_cleanliness:
#   - the border markers are sparse isolated pixels, so they read as a legend rather than
#     as a third large object
#   - rectangular-vs-irregular dst holes are visible from the outline alone
#   - prefilled cavities show the "skip already solved slots" rule directly
#   - the grid still contains only the original two objects plus a small border legend
#
# generator_guarantees_for_solver_assumptions:
#   - at least 3 dst rectangular holes and at least 1 irregular dst hole
#   - at least 2 final-eligible slots, with at least 1 prefilled and 1 still empty
#   - at least 1 rectangular inactive-color slot (control filter matters)
#   - at least 1 irregular active-color slot (shape filter matters)
#   - expected output differs from:
#       * control-only filling
#       * rectangle-only filling
#       * fill-everything baseline


# =============================================================================
# Shared solver helpers
# =============================================================================


def _components(mask: np.ndarray, *, connectivity: int = 8) -> list[dict]:
    lab, n = label_mask(mask, connectivity=connectivity)
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


def _extract_group_subgrid(grid: np.ndarray, bg: int, group: list[dict]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_of(group)
    sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for d in group:
        for rr, cc in d["pts"]:
            sub[int(rr) - int(r0), int(cc) - int(c0)] = int(grid[int(rr), int(cc)])
    return sub


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    mask = grid != int(bg)
    pts = np.argwhere(mask)
    if pts.size == 0:
        return grid.copy()
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _outline_col(sub: np.ndarray, bg: int) -> int:
    h, w = sub.shape
    border_vals = []
    for c in range(w):
        border_vals.append(int(sub[0, c]))
        border_vals.append(int(sub[h - 1, c]))
    for r in range(1, h - 1):
        border_vals.append(int(sub[r, 0]))
        border_vals.append(int(sub[r, w - 1]))
    border_vals = [v for v in border_vals if int(v) != int(bg)]
    if border_vals:
        return int(Counter(border_vals).most_common(1)[0][0])
    cnt = Counter(map(int, sub.flatten()))
    opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
    return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)


def _is_rect_pts(pts: np.ndarray) -> bool:
    if pts.size == 0:
        return False
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(len(pts)) == int((r1 - r0 + 1) * (c1 - c0 + 1))


def _enclosed_bg_components(sub: np.ndarray, bg: int) -> list[dict]:
    mask = sub == int(bg)
    lab, n = label_mask(mask, connectivity=4)
    h, w = sub.shape
    holes = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if bool(r0 == 0 or c0 == 0 or r1 == int(h - 1) or c1 == int(w - 1)):
            continue
        holes.append(
            {
                "pts": pts,
                "cent": (
                    int(round(float(pts[:, 0].mean()))),
                    int(round(float(pts[:, 1].mean()))),
                ),
                "size": int(len(pts)),
                "is_rect": bool(_is_rect_pts(pts)),
            }
        )
    return holes


def _colored_components(
    sub: np.ndarray,
    bg: int,
    *,
    outline_col: int | None = None,
    interior_only: bool = True,
) -> list[dict]:
    outc = int(_outline_col(sub, bg) if outline_col is None else outline_col)
    mask = (sub != int(bg)) & (sub != int(outc))
    lab, n = label_mask(mask, connectivity=4)
    h, w = sub.shape
    comps = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if interior_only and bool(r0 == 0 or c0 == 0 or r1 == int(h - 1) or c1 == int(w - 1)):
            continue
        vals = sub[lab == int(cid)]
        col = Counter(map(int, vals.flatten())).most_common(1)[0][0]
        comps.append(
            {
                "pts": pts,
                "cent": (
                    int(round(float(pts[:, 0].mean()))),
                    int(round(float(pts[:, 1].mean()))),
                ),
                "size": int(len(pts)),
                "color": int(col),
                "is_rect": bool(_is_rect_pts(pts)),
            }
        )
    return comps


def _dest_slots(sub: np.ndarray, bg: int) -> list[dict]:
    outc = _outline_col(sub, bg)
    slots = []
    for hole in _enclosed_bg_components(sub, bg):
        slots.append(
            {
                "kind": "empty",
                "pts": hole["pts"],
                "cent": hole["cent"],
                "size": hole["size"],
                "is_rect": bool(hole["is_rect"]),
            }
        )
    for comp in _colored_components(sub, bg, outline_col=outc, interior_only=True):
        slots.append(
            {
                "kind": "filled",
                "pts": comp["pts"],
                "cent": comp["cent"],
                "size": comp["size"],
                "color": int(comp["color"]),
                "is_rect": bool(comp["is_rect"]),
            }
        )
    return slots


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


def _extract_scene(grid: np.ndarray, bg: int) -> tuple[list[np.ndarray], set[int]]:
    comps = _components(grid != int(bg), connectivity=8)
    assert len(comps) >= 2
    object_groups = [[comps[0]], [comps[1]]]
    objects = [_extract_group_subgrid(grid, bg, grp) for grp in object_groups]

    object_cells = set()
    for grp in object_groups:
        for comp in grp:
            for rr, cc in comp["pts"]:
                object_cells.add((int(rr), int(cc)))

    h, w = grid.shape
    ctrl_colors: set[int] = set()
    border_pts = (
        [(0, c) for c in range(w)]
        + [(h - 1, c) for c in range(w)]
        + [(r, 0) for r in range(1, h - 1)]
        + [(r, w - 1) for r in range(1, h - 1)]
    )
    for r, c in border_pts:
        if (int(r), int(c)) in object_cells:
            continue
        val = int(grid[int(r), int(c)])
        if int(val) != int(bg):
            ctrl_colors.add(int(val))
    return objects, ctrl_colors


def _ordered_match(src_subs: list[dict], dest_slots: list[dict], src_shape: tuple[int, int]):
    dest_pts = [slot["cent"] for slot in dest_slots]
    dest_keys, d_rr, d_cc = _rank_keys(dest_pts)

    best = None
    for k in range(4):
        for flip in [False, True]:
            tpts = [
                _transform_point(s["cent"], int(src_shape[0]), int(src_shape[1]), int(k), bool(flip))
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
    for slot in dest_slots:
        key = (int(d_rr[int(slot["cent"][0])]), int(d_cc[int(slot["cent"][1])]))
        dest_list.append((key, slot["cent"], slot))
    dest_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    tpts_all = [
        _transform_point(s["cent"], int(src_shape[0]), int(src_shape[1]), int(k), bool(flip))
        for s in src_subs
    ]
    _keys, s_rr, s_cc = _rank_keys(tpts_all)

    src_list = []
    for s in src_subs:
        tp = _transform_point(s["cent"], int(src_shape[0]), int(src_shape[1]), int(k), bool(flip))
        key = (int(s_rr[int(tp[0])]), int(s_cc[int(tp[1])]))
        src_list.append((key, tp, int(s["color"])))
    src_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))
    return src_list, dest_list


def _solve_pair(
    dest: np.ndarray,
    src: np.ndarray,
    bg: int,
    active_colors: set[int],
    *,
    use_control: bool,
    use_rect_filter: bool,
) -> np.ndarray:
    dest = np.array(dest, dtype=int).copy()
    src = np.array(src, dtype=int)

    dest_slots = _dest_slots(dest, bg)
    src_subs = _colored_components(src, bg, interior_only=True)
    if not dest_slots or not src_subs or len(dest_slots) != len(src_subs):
        return _tight_crop_non_bg(dest, bg)

    src_list, dest_list = _ordered_match(src_subs, dest_slots, src.shape)

    for (_src_key, _src_pt, col), (_dst_key, _dst_pt, slot) in zip(src_list, dest_list):
        if slot["kind"] != "empty":
            continue
        if bool(use_rect_filter) and not bool(slot["is_rect"]):
            continue
        if bool(use_control) and int(col) not in active_colors:
            continue
        for rr, cc in slot["pts"]:
            dest[int(rr), int(cc)] = int(col)
    return _tight_crop_non_bg(dest, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    1. Ignore border control pixels when extracting the two main objects.
    2. Treat dst slots as `empty holes + already-filled interior components`.
    3. Match src subsprites to those slots by centroid-rank pattern.
    4. Fill only slots that are:
         - still empty,
         - rectangular,
         - and whose matched src color appears among the border control pixels.
    5. Tight-crop the resulting dst object.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    objs, ctrl_colors = _extract_scene(g, bg)
    holes_list = [_enclosed_bg_components(o, bg) for o in objs]
    hole_counts = [len(hh) for hh in holes_list]
    if max(hole_counts) > 0:
        dest_i = int(np.argmax(hole_counts))
    else:
        hole_area = [sum(int(h["size"]) for h in hh) for hh in holes_list]
        dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)

    dest = objs[dest_i].copy()
    src = objs[src_i].copy()
    return _solve_pair(dest, src, bg, ctrl_colors, use_control=True, use_rect_filter=True)


# =============================================================================
# Generator constants
# =============================================================================


CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)

CAVITY_COUNT_RANGE = (5, 8)
FILL_COLOR_COUNT_RANGE = (3, 5)

ROW_HEIGHT_CHOICES = [2, 2, 3, 3]
LAYOUT_K_CHOICES_FULL = [1, 2, 2, 3, 3]
LAYOUT_K_CHOICES_SMALL = [1, 2, 2]
ROW_GAP_CHOICES = [1, 1, 2]

CENTERED_SLOT_WIDTHS = [3, 3, 5]
PAIR_SLOT_WIDTHS = [2, 2, 3, 3, 4]
SIDE_SLOT_WIDTHS = [2, 2, 3]
MID_SLOT_WIDTHS = [3]
PAIR_AXIS_GAP_CHOICES = [1, 1, 3]
TRIO_MID_SIDE_GAP = 1

MAX_MUTATE_ATTEMPTS = 40
NONRECT_RETRIES = 24
RECT_HOLE_PROBABILITY = 0.5
MIN_RECT_HOLES = 3
MIN_NONRECT_HOLES = 1

COLOR_MISMATCH_RATE = 0.10
ACTIVE_CONTROL_COUNT = 2
PREFILL_MID_BIAS_PROB = 0.75

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2

MAX_GEN_ATTEMPTS = 600
MAX_LAYOUT_R_BUDGET = 24


# =============================================================================
# Shape helpers
# =============================================================================


def _random_rect_shape(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _is_rect_mask(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    return bool(_is_rect_pts(pts))


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


def _fallback_nonrect_shape(h: int, w: int, *, symmetric: bool) -> np.ndarray:
    m = np.ones((h, w), dtype=bool)
    if symmetric:
        m[0, 0] = False
        m[0, w - 1] = False
    else:
        m[0, 0] = False
    return m


def _gen_nonrect_shape(h: int, w: int, *, symmetric: bool = False) -> np.ndarray:
    if h < 2 or w < 2:
        raise ValueError("non-rect generation requires h,w >= 2")
    base = _random_rect_shape(h, w)
    limit = max(1, (h * w) // 2)
    for _ in range(NONRECT_RETRIES):
        m = _mutate_shape(base, n_remove_max=limit, symmetric=symmetric)
        if m.any() and not _is_rect_mask(m):
            return m
    return _fallback_nonrect_shape(h, w, symmetric=symmetric)


def _mirror_shape(m: np.ndarray) -> np.ndarray:
    return m[:, ::-1].copy()


def _gen_dst_src_pair(h: int, w: int, *, symmetric: bool, dst_rect: bool) -> tuple[np.ndarray, np.ndarray]:
    if bool(dst_rect):
        dst = _random_rect_shape(h, w)
        src = _gen_nonrect_shape(h, w, symmetric=symmetric)
    else:
        dst = _gen_nonrect_shape(h, w, symmetric=symmetric)
        src = _random_rect_shape(h, w)
    return dst, src


# =============================================================================
# Layout helpers
# =============================================================================


def _random_fillable_flag() -> bool:
    return bool(random.random() < RECT_HOLE_PROBABILITY)


def _add_centered_slot(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    w = random.choice(CENTERED_SLOT_WIDTHS)
    fillable = _random_fillable_flag()
    dst, src = _gen_dst_src_pair(row_h, w, symmetric=True, dst_rect=fillable)
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -(w // 2),
            "shape_dst": dst,
            "shape_src": src,
            "pair_id": pair_id,
            "fillable": bool(fillable),
        }
    )


def _add_mirror_pair_slots(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    w = random.choice(PAIR_SLOT_WIDTHS)
    fillable = _random_fillable_flag()
    dst_l, src_l = _gen_dst_src_pair(row_h, w, symmetric=False, dst_rect=fillable)
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
            "fillable": bool(fillable),
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": half + 1,
            "shape_dst": dst_r,
            "shape_src": src_r,
            "pair_id": pair_id,
            "fillable": bool(fillable),
        }
    )


def _add_trio_slots(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    w_side = random.choice(SIDE_SLOT_WIDTHS)
    w_mid = random.choice(MID_SLOT_WIDTHS)

    outer_fillable = _random_fillable_flag()
    center_fillable = _random_fillable_flag()

    dst_l, src_l = _gen_dst_src_pair(row_h, w_side, symmetric=False, dst_rect=outer_fillable)
    dst_r = _mirror_shape(dst_l)
    src_r = _mirror_shape(src_l)
    dst_m, src_m = _gen_dst_src_pair(row_h, w_mid, symmetric=True, dst_rect=center_fillable)

    mid_half = w_mid // 2
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -(mid_half + w_side + TRIO_MID_SIDE_GAP),
            "shape_dst": dst_l,
            "shape_src": src_l,
            "pair_id": pair_id,
            "fillable": bool(outer_fillable),
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -mid_half,
            "shape_dst": dst_m,
            "shape_src": src_m,
            "pair_id": pair_id + 1,
            "fillable": bool(center_fillable),
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": mid_half + TRIO_MID_SIDE_GAP + 1,
            "shape_dst": dst_r,
            "shape_src": src_r,
            "pair_id": pair_id,
            "fillable": bool(outer_fillable),
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


# =============================================================================
# Rendering and validation
# =============================================================================


def _render_object_data(
    layout: list[dict],
    shapes: list[np.ndarray],
    colors: list[int],
    bg: int,
    outline_col: int,
) -> dict | None:
    placed_slots = []
    for slot_idx, (slot, sh) in enumerate(zip(layout, shapes)):
        rr, cc = np.where(sh)
        if len(rr) == 0:
            continue
        placed_slots.append((slot_idx, slot["r0"] + rr, slot["col_offset"] + cc))
    if not placed_slots:
        return None

    all_rows = np.concatenate([p[1] for p in placed_slots])
    all_cols = np.concatenate([p[2] for p in placed_slots])
    r_min, r_max = int(all_rows.min()), int(all_rows.max())
    c_min, c_max = int(all_cols.min()), int(all_cols.max())

    pad = 2
    h = (r_max - r_min + 1) + 2 * pad
    w = (c_max - c_min + 1) + 2 * pad

    filled_grid = np.full((h, w), bg, dtype=int)
    fill_mask = np.zeros((h, w), dtype=bool)
    slot_pts_full = [None] * len(layout)
    for slot_idx, rows, cols in placed_slots:
        rr = rows - r_min + pad
        cc = cols - c_min + pad
        filled_grid[rr, cc] = int(colors[slot_idx])
        fill_mask[rr, cc] = True
        slot_pts_full[slot_idx] = np.column_stack([rr, cc]).astype(int)

    outline_mask = np.zeros_like(fill_mask)
    outline_mask[1:, :] |= fill_mask[:-1, :]
    outline_mask[:-1, :] |= fill_mask[1:, :]
    outline_mask[:, 1:] |= fill_mask[:, :-1]
    outline_mask[:, :-1] |= fill_mask[:, 1:]
    outline_mask &= ~fill_mask
    filled_grid[outline_mask] = int(outline_col)

    empty_grid = filled_grid.copy()
    empty_grid[fill_mask] = int(bg)

    b = bbox(filled_grid != bg)
    if b is None:
        return None
    r0b, c0b, r1b, c1b = b
    cropped_empty = empty_grid[r0b : r1b + 1, c0b : c1b + 1]
    cropped_filled = filled_grid[r0b : r1b + 1, c0b : c1b + 1]
    shift = np.array([r0b, c0b], dtype=int)
    slot_pts = []
    for pts in slot_pts_full:
        if pts is None:
            return None
        slot_pts.append((pts - shift).astype(int))
    return {"empty": cropped_empty, "filled": cropped_filled, "slot_pts": slot_pts}


def _count_4conn_cavities(filled_grid: np.ndarray, bg: int, outline_col: int) -> int:
    mask = (filled_grid != bg) & (filled_grid != outline_col)
    _, n = label_mask(mask, connectivity=4)
    return int(n)


def _count_enclosed_bg_holes(empty_grid: np.ndarray, bg: int) -> int:
    holes = _enclosed_bg_components(empty_grid, bg)
    return int(len(holes))


def _count_filled_interior_components(grid: np.ndarray, bg: int, outline_col: int) -> int:
    return int(len(_colored_components(grid, bg, outline_col=outline_col, interior_only=True)))


def _count_total_dest_slots(grid: np.ndarray, bg: int, outline_col: int) -> int:
    return int(_count_enclosed_bg_holes(grid, bg) + _count_filled_interior_components(grid, bg, outline_col))


def _is_8connected(grid: np.ndarray, bg: int) -> bool:
    mask = grid != bg
    if not mask.any():
        return False
    _, n = label_mask(mask, connectivity=8)
    return int(n) == 1


def _size_in_spec(grid: np.ndarray) -> bool:
    h, w = grid.shape
    return OBJECT_H_RANGE[0] <= h <= OBJECT_H_RANGE[1] and OBJECT_W_RANGE[0] <= w <= OBJECT_W_RANGE[1]


# =============================================================================
# Color and control helpers
# =============================================================================


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


def _force_fused_color_pattern(layout: list[dict], fill_pool: list[int]) -> tuple[list[int], set[int]] | None:
    rect_idxs = [i for i, slot in enumerate(layout) if bool(slot["fillable"])]
    nonrect_idxs = [i for i, slot in enumerate(layout) if not bool(slot["fillable"])]
    if len(rect_idxs) < MIN_RECT_HOLES or len(nonrect_idxs) < MIN_NONRECT_HOLES:
        return None
    if len(fill_pool) < 3:
        return None

    colors = _assign_pair_colors(layout, fill_pool)
    active_a, active_b, inactive_c = random.sample(fill_pool, 3)

    rect_active_a = random.choice(rect_idxs)
    rem = [i for i in rect_idxs if i != rect_active_a]
    rect_active_b = random.choice(rem)
    rem = [i for i in rem if i != rect_active_b]
    rect_inactive = random.choice(rem)
    nonrect_active = random.choice(nonrect_idxs)

    colors[rect_active_a] = int(active_a)
    colors[rect_active_b] = int(active_b)
    colors[rect_inactive] = int(inactive_c)
    colors[nonrect_active] = int(active_a)

    active_colors = {int(active_a), int(active_b)}
    return colors, active_colors


def _choose_prefilled_indices(slot_pts: list[np.ndarray], eligible_indices: list[int]) -> list[int]:
    if len(eligible_indices) < 2:
        return []
    counts = list(range(1, len(eligible_indices)))
    mid_counts = [k for k in counts if 1 < k < len(eligible_indices) - 1]
    if mid_counts and random.random() < PREFILL_MID_BIAS_PROB:
        n_prefilled = random.choice(mid_counts)
    else:
        n_prefilled = random.choice(counts)

    cents = []
    for pts in slot_pts:
        cents.append(
            (
                int(round(float(pts[:, 0].mean()))),
                int(round(float(pts[:, 1].mean()))),
            )
        )
    keys, _, _ = _rank_keys(cents)
    ordered_eligible = sorted(eligible_indices, key=lambda i: (keys[i][0], keys[i][1], cents[i][0], cents[i][1]))
    bad_prefix = set(ordered_eligible[:n_prefilled])
    bad_suffix = set(ordered_eligible[-n_prefilled:])
    for _ in range(40):
        cand = sorted(random.sample(eligible_indices, n_prefilled))
        sc = set(cand)
        if sc != bad_prefix and sc != bad_suffix:
            return cand
    return sorted(random.sample(eligible_indices, n_prefilled))


def _apply_slot_fills(base_grid: np.ndarray, slot_pts: list[np.ndarray], colors: list[int], indices: list[int]) -> np.ndarray:
    out = np.array(base_grid, dtype=int).copy()
    for idx in indices:
        pts = slot_pts[idx]
        out[pts[:, 0], pts[:, 1]] = int(colors[idx])
    return out


def _choose_control_columns(w: int, n: int) -> list[int] | None:
    cols = list(range(1, w - 1))
    random.shuffle(cols)
    picked: list[int] = []
    for c in cols:
        if all(abs(int(c) - int(pc)) >= 2 for pc in picked):
            picked.append(int(c))
            if len(picked) == n:
                return sorted(picked)
    return None


# =============================================================================
# Generation pipeline
# =============================================================================


def _build_object_pair(
    layout: list[dict],
    colors: list[int],
    active_colors: set[int],
    bg: int,
    outline_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n_cavities = len(layout)
    dst_shapes = [slot["shape_dst"] for slot in layout]
    src_shapes = [slot["shape_src"] for slot in layout]

    dst_data = _render_object_data(layout, dst_shapes, colors, bg, outline_col)
    if dst_data is None:
        return None
    dst_empty = dst_data["empty"]
    if not _is_8connected(dst_empty, bg):
        return None
    if _count_enclosed_bg_holes(dst_empty, bg) != n_cavities:
        return None
    if _count_filled_interior_components(dst_empty, bg, outline_col) != 0:
        return None
    if not _size_in_spec(dst_empty):
        return None

    src_data = _render_object_data(layout, src_shapes, colors, bg, outline_col)
    if src_data is None:
        return None
    src_cropped = src_data["filled"]
    if not _is_8connected(src_cropped, bg):
        return None
    if _count_4conn_cavities(src_cropped, bg, outline_col) != n_cavities:
        return None
    if len(_enclosed_bg_components(src_cropped, bg)) != 0:
        return None
    if not _size_in_spec(src_cropped):
        return None

    eligible_idxs = [i for i, slot in enumerate(layout) if bool(slot["fillable"]) and int(colors[i]) in active_colors]
    rect_inactive = [i for i, slot in enumerate(layout) if bool(slot["fillable"]) and int(colors[i]) not in active_colors]
    active_nonrect = [i for i, slot in enumerate(layout) if (not bool(slot["fillable"])) and int(colors[i]) in active_colors]
    if len(eligible_idxs) < 2 or not rect_inactive or not active_nonrect:
        return None

    prefilled_indices = _choose_prefilled_indices(dst_data["slot_pts"], eligible_idxs)
    if not prefilled_indices or len(prefilled_indices) >= len(eligible_idxs):
        return None

    dst_partial = _apply_slot_fills(dst_empty, dst_data["slot_pts"], colors, prefilled_indices)
    expected = _apply_slot_fills(dst_empty, dst_data["slot_pts"], colors, eligible_idxs)

    if not _is_8connected(dst_partial, bg):
        return None
    if not _size_in_spec(dst_partial):
        return None

    n_empty = _count_enclosed_bg_holes(dst_partial, bg)
    n_prefilled = _count_filled_interior_components(dst_partial, bg, outline_col)
    if n_empty <= 0 or n_prefilled <= 0:
        return None
    if n_empty + n_prefilled != n_cavities:
        return None
    if _count_total_dest_slots(dst_partial, bg, outline_col) != n_cavities:
        return None

    fused = _solve_pair(dst_partial.copy(), src_cropped.copy(), bg, active_colors, use_control=True, use_rect_filter=True)
    ctrl_only = _solve_pair(dst_partial.copy(), src_cropped.copy(), bg, active_colors, use_control=True, use_rect_filter=False)
    rect_only = _solve_pair(dst_partial.copy(), src_cropped.copy(), bg, active_colors, use_control=False, use_rect_filter=True)
    full_fill = _solve_pair(dst_partial.copy(), src_cropped.copy(), bg, active_colors, use_control=False, use_rect_filter=False)
    if not np.array_equal(fused, expected):
        return None
    if np.array_equal(fused, ctrl_only):
        return None
    if np.array_equal(fused, rect_only):
        return None
    if np.array_equal(fused, full_fill):
        return None
    return src_cropped, dst_partial, expected


def _place_on_canvas(src: np.ndarray, dst: np.ndarray, active_colors: set[int], bg: int) -> np.ndarray | None:
    canvas_h = random.randint(*CANVAS_SIZE_RANGE)
    canvas_w = random.randint(*CANVAS_SIZE_RANGE)
    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    try:
        r, c = random_free_location(canvas, src, transparent=bg, border=PLACEMENT_BORDER, padding=PLACEMENT_PADDING)
    except ValueError:
        return None
    paste(canvas, src, r, c, transparent=bg)
    try:
        r, c = random_free_location(canvas, dst, transparent=bg, border=PLACEMENT_BORDER, padding=PLACEMENT_PADDING)
    except ValueError:
        return None
    paste(canvas, dst, r, c, transparent=bg)

    ctrl_cols = _choose_control_columns(canvas_w, len(active_colors))
    if ctrl_cols is None:
        return None
    for c, col in zip(ctrl_cols, sorted(active_colors)):
        canvas[0, int(c)] = int(col)
    return canvas


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    objs, ctrl_colors = _extract_scene(g, bg)
    holes_list = [_enclosed_bg_components(o, bg) for o in objs]
    hole_counts = [len(hh) for hh in holes_list]
    dest_i = int(np.argmax(hole_counts))
    src_i = int(1 - dest_i)
    dest = objs[dest_i]
    src = objs[src_i]

    dest_slots = _dest_slots(dest, bg)
    src_subs = _colored_components(src, bg, interior_only=True)
    if not dest_slots or len(dest_slots) != len(src_subs):
        return {
            "prefilled": 0,
            "fillable_empty": 0,
            "rect_inactive": 0,
            "active_nonrect": 0,
            "active_colors": len(ctrl_colors),
            "diff_vs_ctrl_only": 0,
            "diff_vs_rect_only": 0,
        }

    src_list, dest_list = _ordered_match(src_subs, dest_slots, src.shape)
    prefilled = 0
    fillable_empty = 0
    rect_inactive = 0
    active_nonrect = 0
    for (_src_key, _src_pt, col), (_dst_key, _dst_pt, slot) in zip(src_list, dest_list):
        is_active = int(col) in ctrl_colors
        if slot["kind"] == "filled":
            prefilled += 1
        elif bool(slot["is_rect"]) and is_active:
            fillable_empty += 1
        elif bool(slot["is_rect"]) and not is_active:
            rect_inactive += 1
        elif (not bool(slot["is_rect"])) and is_active:
            active_nonrect += 1

    fused = _solve_pair(dest.copy(), src.copy(), bg, ctrl_colors, use_control=True, use_rect_filter=True)
    ctrl_only = _solve_pair(dest.copy(), src.copy(), bg, ctrl_colors, use_control=True, use_rect_filter=False)
    rect_only = _solve_pair(dest.copy(), src.copy(), bg, ctrl_colors, use_control=False, use_rect_filter=True)
    return {
        "prefilled": int(prefilled),
        "fillable_empty": int(fillable_empty),
        "rect_inactive": int(rect_inactive),
        "active_nonrect": int(active_nonrect),
        "active_colors": int(len(ctrl_colors)),
        "diff_vs_ctrl_only": int(not np.array_equal(fused, ctrl_only)),
        "diff_vs_rect_only": int(not np.array_equal(fused, rect_only)),
    }


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline_col = palette[0]

    layout = _design_layout(random.randint(*CAVITY_COUNT_RANGE))
    if len(layout) < CAVITY_COUNT_RANGE[0]:
        return None

    rect_count = sum(1 for slot in layout if bool(slot["fillable"]))
    nonrect_count = len(layout) - rect_count
    if rect_count < MIN_RECT_HOLES or nonrect_count < MIN_NONRECT_HOLES:
        return None

    fc_lo, fc_hi = FILL_COLOR_COUNT_RANGE
    max_cols = min(fc_hi, len(layout), len(palette) - 1)
    if max_cols < fc_lo:
        return None
    n_fill_cols = random.randint(fc_lo, max_cols)
    fill_pool = palette[1 : 1 + n_fill_cols]
    forced = _force_fused_color_pattern(layout, fill_pool)
    if forced is None:
        return None
    colors, active_colors = forced
    if len(active_colors) != ACTIVE_CONTROL_COUNT:
        return None

    built = _build_object_pair(layout, colors, active_colors, bg, outline_col)
    if built is None:
        return None
    src, dst_partial, expected = built

    canvas = _place_on_canvas(src, dst_partial, active_colors, bg)
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

    metrics = _variant_metrics(canvas)
    if metrics["prefilled"] <= 0:
        return None
    if metrics["fillable_empty"] <= 0:
        return None
    if metrics["rect_inactive"] <= 0:
        return None
    if metrics["active_nonrect"] <= 0:
        return None
    if metrics["diff_vs_ctrl_only"] != 1:
        return None
    if metrics["diff_vs_rect_only"] != 1:
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
