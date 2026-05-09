# Variant of 3a25b0d8
# Dimension: D3 — 目标对象结构 (dst)
# Change: 单个 8-连通轮廓框架，洞为背景色，洞的排列（质心秩序）与 src 子精灵对应
#         → dst 包含部分预填色的洞；solver 需要区分已填洞与空洞，只补剩余空洞

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location

# -----------------------------------------------------------------------------
# Design analysis for the D3 variant
# -----------------------------------------------------------------------------
#
# variant_spec:
#   core_change:
#     before: dst is an outline-only 8-connected frame; every interior cavity is bg.
#     after: dst keeps the same outline structure, but a strict non-empty proper subset
#       of its cavities is already filled with the correct src-derived color.
#       The remaining cavities stay bg and must be completed in the output.
#
#   concrete_pixel_behavior:
#     src:
#       unchanged from the original task: one 8-connected outlined object whose interior
#       cavities are fully filled, one 4-connected colored component per cavity.
#     dst:
#       still the same outline color and same cavity-position template, but now each
#       interior slot is in exactly one of two visible states:
#       - empty hole: an enclosed bg component
#       - prefilled hole: an enclosed non-bg, non-outline 4-connected component
#       Both kinds participate in the same centroid-rank layout. Output is dst fully
#       filled, so only the empty subset changes.
#
#   behavior_trace:
#     example:
#       src has 5 colored subsprites ordered by centroid rank:
#         [(top,left)=red, (top,right)=green, (mid)=yellow, (bot,left)=blue, (bot,right)=red]
#       original dst has 5 bg holes. Original solver extracts those 5 holes and fills all.
#       variant dst might already contain green at (top,right) and blue at (bot,left), with
#       the other 3 slots still bg. The old solver only sees the 3 bg holes, so its rank
#       list length no longer matches src's 5 subsprites and the pairing collapses.
#       The new solver instead builds dst's slot template from:
#         3 enclosed bg holes + 2 enclosed colored interior components = 5 total slots,
#       aligns that full template to src, then writes colors only into the 3 bg slots.
#
#   complexity_calibration:
#     - We keep the original 4-9 slot layouts and the same src/dst asymmetry in cavity
#       shapes, so the task still requires positional matching rather than trivial copying.
#     - Generator enforces at least one prefilled slot and at least one empty slot, so the
#       variant is visually observable and never degenerates back to the original task.
#     - Prefill count is biased away from extremes when possible, so typical samples show a
#       mixed state instead of "almost original" or "already solved".
#
#   implicit_assumption_audit:
#     - Original `dest_i = argmax(hole_area)` assumed dst contains all holes and src none.
#       This still holds only if dst retains at least one empty hole; generator guarantees
#       that. Solver now chooses dst by enclosed-hole count first and falls back only if
#       needed.
#     - Original `_colored_components(dest)` would have misinterpreted dst's prefilled
#       interiors as if dst were the source object. The new solver intentionally uses
#       colored interior components on dst as part of the destination slot template.
#     - Original fill loop overwrote every matched hole unconditionally. In the variant,
#       unconditional filling is no longer semantically correct because some slots are
#       already solved. The new loop writes only into slots tagged as empty holes.
#     - Original outline-color heuristic used the most common non-bg color. With partially
#       filled dst, a large fill color could occasionally rival the outline. We switch to
#       a border-based outline-color detector, because after tight object extraction only
#       outline pixels can touch the subgrid border.
#
#   placement_legality:
#     - Objects are still placed independently with the original border/padding rules via
#       `random_free_location`, so they never overlap or leave the canvas.
#     - The partial dst keeps the same bbox as the empty/full versions; no extra placement
#       constraints are introduced.
#
#   visual_observability:
#     - Across 2-4 ARC examples, the human-visible difference is direct: dst already shows
#       some correctly colored interior patches while other cavities remain blank.
#     - The rule remains inferable from structure, not numeric color ids: prefilled colors
#       match src by position, and unfilled cavities complete the same positional pattern.
#
#   visual_cleanliness:
#     - We do not add extra markers or third object types; the scene still contains exactly
#       the original two objects with one added state inside dst's cavities.
#     - Using the same outline and fill vocabulary keeps the grid readable.
#
#   color_inference:
#     - Colors are unchanged from the original task: 3-7 fill colors with repeated support.
#       Prefilled dst regions reuse the same slot-color mapping, so they provide extra
#       evidence rather than introducing a new mapping rule.
#
#   generator_guarantees_for_new_main:
#     - dst has at least 1 empty hole and at least 1 prefilled interior component.
#     - The total slot count detected as
#         enclosed_bg_holes + enclosed_colored_components
#       exactly matches the number of src subsprites.
#     - Prefilled dst colors are already correct, so the solver's "fill only empty holes"
#       policy is sufficient to reach the expected output.


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


def _extract_two_objects(grid: np.ndarray, bg: int) -> list[list[dict]]:
    comps = _components(grid != int(bg))
    if len(comps) < 2:
        return [comps]
    return [[comps[0]], [comps[1]]]


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


def _enclosed_bg_components(sub: np.ndarray, bg: int) -> list[dict]:
    mask = sub == int(bg)
    lab, n = label_mask(mask, connectivity=4)
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


def _outline_col(sub: np.ndarray, bg: int) -> int:
    H, W = sub.shape
    border_vals = []
    for c in range(W):
        border_vals.append(int(sub[0, c]))
        border_vals.append(int(sub[H - 1, c]))
    for r in range(1, H - 1):
        border_vals.append(int(sub[r, 0]))
        border_vals.append(int(sub[r, W - 1]))
    border_vals = [v for v in border_vals if int(v) != int(bg)]
    if border_vals:
        return int(Counter(border_vals).most_common(1)[0][0])
    cnt = Counter(map(int, sub.flatten()))
    opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
    return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)


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
    H, W = sub.shape
    comps = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if interior_only and bool(r0 == 0 or c0 == 0 or r1 == int(H - 1) or c1 == int(W - 1)):
            continue
        vals = sub[lab == int(cid)]
        col = Counter(map(int, vals.flatten())).most_common(1)[0][0]
        cent = (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))
        comps.append(
            {
                "pts": pts,
                "cent": cent,
                "size": int(len(pts)),
                "color": int(col),
            }
        )
    return comps


def _dest_slots(sub: np.ndarray, bg: int) -> list[dict]:
    outc = _outline_col(sub, bg)
    slots = []
    for h in _enclosed_bg_components(sub, bg):
        slots.append(
            {
                "kind": "empty",
                "pts": h["pts"],
                "cent": h["cent"],
                "size": h["size"],
            }
        )
    for comp in _colored_components(sub, bg, outline_col=outc, interior_only=True):
        slots.append(
            {
                "kind": "filled",
                "pts": comp["pts"],
                "cent": comp["cent"],
                "size": comp["size"],
                "color": comp["color"],
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


def _transform_point(p: tuple[int, int], H0: int, W0: int, k: int, flip: bool) -> tuple[int, int]:
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


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    mask = grid != int(bg)
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _count_filled_interior_components(grid: np.ndarray, bg: int, outline_col: int) -> int:
    return len(_colored_components(grid, bg, outline_col=outline_col, interior_only=True))


def _count_total_dest_slots(grid: np.ndarray, bg: int, outline_col: int) -> int:
    return len(_enclosed_bg_components(grid, bg)) + _count_filled_interior_components(grid, bg, outline_col)


def _is_8connected(grid: np.ndarray, bg: int) -> bool:
    mask = grid != int(bg)
    if not mask.any():
        return False
    _, n = label_mask(mask, connectivity=8)
    return int(n) == 1


def _size_in_spec(grid: np.ndarray) -> bool:
    h, w = grid.shape
    return (OBJECT_H_RANGE[0] <= h <= OBJECT_H_RANGE[1]
            and OBJECT_W_RANGE[0] <= w <= OBJECT_W_RANGE[1])


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    """Diagnostic-only helper used during degeneration checks."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    groups = _extract_two_objects(g, bg)
    if len(groups) < 2:
        return {"empty_holes": 0, "prefilled_holes": 0, "total_slots": 0}
    objs = [_extract_group_subgrid(g, bg, grp) for grp in groups]
    hole_counts = [len(_enclosed_bg_components(o, bg)) for o in objs]
    dest_i = int(np.argmax(hole_counts))
    dest = objs[dest_i]
    outc = _outline_col(dest, bg)
    empty = len(_enclosed_bg_components(dest, bg))
    prefilled = _count_filled_interior_components(dest, bg, outc)
    return {
        "empty_holes": int(empty),
        "prefilled_holes": int(prefilled),
        "total_slots": int(empty + prefilled),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only the still-empty holes in the destination object, preserving already-solved
    cavities, then tight-crop the completed destination.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    groups = _extract_two_objects(g, bg)
    assert len(groups) >= 2
    objs = [_extract_group_subgrid(g, bg, grp) for grp in groups]

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

    dest_slots = _dest_slots(dest, bg)
    src_subs = _colored_components(src, bg, interior_only=True)
    if not dest_slots or not src_subs or len(dest_slots) != len(src_subs):
        return _tight_crop_non_bg(dest, bg)

    dest_pts = [slot["cent"] for slot in dest_slots]
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
    for slot in dest_slots:
        key = (int(d_rr[int(slot["cent"][0])]), int(d_cc[int(slot["cent"][1])]))
        dest_list.append((key, slot["cent"], slot))
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

    for (_src_key, _src_pt, col), (_dst_key, _dst_pt, slot) in zip(src_list, dest_list):
        if slot["kind"] != "empty":
            continue
        for rr, cc in slot["pts"]:
            dest[int(rr), int(cc)] = int(col)

    return _tight_crop_non_bg(dest, bg)


# --------------- generator (D3 variant: partially prefilled dst) ---------------

CANVAS_SIZE_RANGE = (22, 30)           # canvas H/W range
OBJECT_H_RANGE = (7, 17)               # src/dst object height bounds
OBJECT_W_RANGE = (7, 13)               # src/dst object width bounds

# Cavity structure
CAVITY_COUNT_RANGE = (4, 9)            # number of cavities per object
FILL_COLOR_COUNT_RANGE = (3, 7)        # number of distinct fill colors

# Layout row patterns (weighted choices)
ROW_HEIGHT_CHOICES = [1, 2, 2, 3, 3, 3]           # per-row cavity height
LAYOUT_K_CHOICES_FULL = [1, 2, 2, 3, 3]           # k=1 centered, k=2 pair, k=3 L+C+R
LAYOUT_K_CHOICES_SMALL = [1, 2, 2]                # when remaining budget <= 2
ROW_GAP_CHOICES = [1, 1, 2]                       # empty rows between cavity rows

# Cavity slot widths per layout type
CENTERED_SLOT_WIDTHS = [3, 3, 5, 5, 7]            # k=1 (odd widths only)
PAIR_SLOT_WIDTHS = [2, 2, 3, 3, 4]                # k=2 per-slot width
SIDE_SLOT_WIDTHS = [2, 2, 3]                      # k=3 side slot width
MID_SLOT_WIDTHS = [1, 3]                          # k=3 center slot width (odd)
PAIR_AXIS_GAP_CHOICES = [1, 1, 3]                 # k=2 empty cols at axis (forced odd)
TRIO_MID_SIDE_GAP = 1                             # k=3 empty cols between center and side

# Shape mutation
NONRECT_PROBABILITY = 0.8              # P(cavity is non-rectangular, else solid rect)
MAX_MUTATE_ATTEMPTS = 40               # iterations per shape mutation

# Color assignment
COLOR_MISMATCH_RATE = 0.10             # P(cavity breaks mirror-pair color sharing)

# Partial dst filling
PREFILL_MID_BIAS_PROB = 0.75           # prefer mixed prefill counts when possible

# Placement
PLACEMENT_BORDER = 1                   # min distance from canvas border
PLACEMENT_PADDING = 2                  # min gap between src and dst objects

# Generation retry
MAX_GEN_ATTEMPTS = 300                 # outer retry loop for generate_input()
MAX_LAYOUT_R_BUDGET = 24               # layout design vertical budget (before crop)


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
    H = (r_max - r_min + 1) + 2 * pad
    W = (c_max - c_min + 1) + 2 * pad

    filled_grid = np.full((H, W), bg, dtype=int)
    fill_mask = np.zeros((H, W), dtype=bool)
    slot_pts_full = [None] * len(layout)
    for slot_idx, rows, cols in placed_slots:
        rr = rows - r_min + pad
        cc = cols - c_min + pad
        filled_grid[rr, cc] = colors[slot_idx]
        fill_mask[rr, cc] = True
        slot_pts_full[slot_idx] = np.column_stack([rr, cc]).astype(int)

    outline_mask = np.zeros_like(fill_mask)
    outline_mask[1:, :] |= fill_mask[:-1, :]
    outline_mask[:-1, :] |= fill_mask[1:, :]
    outline_mask[:, 1:] |= fill_mask[:, :-1]
    outline_mask[:, :-1] |= fill_mask[:, 1:]
    outline_mask &= ~fill_mask
    filled_grid[outline_mask] = outline_col

    empty_grid = filled_grid.copy()
    empty_grid[fill_mask] = bg

    b = bbox(filled_grid != bg)
    if b is None:
        return None
    r0b, c0b, r1b, c1b = b
    cropped_empty = empty_grid[r0b:r1b + 1, c0b:c1b + 1]
    cropped_filled = filled_grid[r0b:r1b + 1, c0b:c1b + 1]
    slot_pts = []
    shift = np.array([r0b, c0b], dtype=int)
    for pts in slot_pts_full:
        if pts is None:
            return None
        slot_pts.append((pts - shift).astype(int))
    return {"empty": cropped_empty, "filled": cropped_filled, "slot_pts": slot_pts}


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


def _choose_prefilled_indices(slot_pts: list[np.ndarray]) -> list[int]:
    n_cavities = len(slot_pts)
    counts = list(range(1, n_cavities))
    mid_counts = [k for k in counts if 2 <= k <= n_cavities - 2]
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
    ordered = sorted(range(n_cavities), key=lambda i: (keys[i][0], keys[i][1], cents[i][0], cents[i][1]))
    bad_prefix = set(ordered[:n_prefilled])
    bad_suffix = set(ordered[-n_prefilled:])
    for _ in range(40):
        cand = sorted(random.sample(range(n_cavities), n_prefilled))
        sc = set(cand)
        if sc != bad_prefix and sc != bad_suffix:
            return cand
    return sorted(random.sample(range(n_cavities), n_prefilled))


def _build_partial_dst(
    dst_empty: np.ndarray,
    slot_pts: list[np.ndarray],
    colors: list[int],
    prefilled_indices: list[int],
) -> np.ndarray:
    dst_partial = dst_empty.copy()
    for idx in prefilled_indices:
        pts = slot_pts[idx]
        dst_partial[pts[:, 0], pts[:, 1]] = int(colors[idx])
    return dst_partial


def _build_object_pair(
    layout: list[dict],
    colors: list[int],
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
    dst_filled = dst_data["filled"]
    if not _is_8connected(dst_empty, bg):
        return None
    if len(_enclosed_bg_components(dst_empty, bg)) != n_cavities:
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
    if _count_filled_interior_components(src_cropped, bg, outline_col) != n_cavities:
        return None
    if len(_enclosed_bg_components(src_cropped, bg)) != 0:
        return None
    if not _size_in_spec(src_cropped):
        return None

    prefilled_indices = _choose_prefilled_indices(dst_data["slot_pts"])
    dst_partial = _build_partial_dst(dst_empty, dst_data["slot_pts"], colors, prefilled_indices)
    if not _is_8connected(dst_partial, bg):
        return None
    if not _size_in_spec(dst_partial):
        return None

    n_empty = len(_enclosed_bg_components(dst_partial, bg))
    n_prefilled = _count_filled_interior_components(dst_partial, bg, outline_col)
    if n_empty <= 0 or n_prefilled <= 0:
        return None
    if n_empty + n_prefilled != n_cavities:
        return None
    if _count_total_dest_slots(dst_partial, bg, outline_col) != n_cavities:
        return None

    return src_cropped, dst_partial, dst_filled


def _place_on_canvas(src: np.ndarray, dst: np.ndarray, bg: int) -> np.ndarray | None:
    canvas_h = random.randint(*CANVAS_SIZE_RANGE)
    canvas_w = random.randint(*CANVAS_SIZE_RANGE)
    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    try:
        r, c = random_free_location(
            canvas,
            src,
            transparent=bg,
            border=PLACEMENT_BORDER,
            padding=PLACEMENT_PADDING,
        )
    except ValueError:
        return None
    paste(canvas, src, r, c, transparent=bg)
    try:
        r, c = random_free_location(
            canvas,
            dst,
            transparent=bg,
            border=PLACEMENT_BORDER,
            padding=PLACEMENT_PADDING,
        )
    except ValueError:
        return None
    paste(canvas, dst, r, c, transparent=bg)
    return canvas


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
    fill_pool = palette[1:1 + n_fill_cols]
    colors = _assign_pair_colors(layout, fill_pool)

    built = _build_object_pair(layout, colors, bg, outline_col)
    if built is None:
        return None
    src, dst_partial, expected = built

    canvas = _place_on_canvas(src, dst_partial, bg)
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
    if metrics["prefilled_holes"] <= 0 or metrics["empty_holes"] <= 0:
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
