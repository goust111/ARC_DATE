# Combo variant of 3a25b0d8 (index 22)
# combo_spec: D16/2+D18/2
# Name: 控制像素+选择性填充

"""
Design analysis for the combo variant
=====================================

a) Concretization and behavior trace
------------------------------------
Base task:
  - detect the two non-bg objects
  - choose dst as the object with the larger enclosed bg-hole area
  - centroid-rank match src colored subsprites to dst holes
  - fill every dst hole with the matched src color

Combo rule:
  - keep the same src/dst extraction and centroid-rank matching
  - add top-edge control pixels whose colors are copied from a subset of src fill colors
  - interpret those border colors as an allow-list of transferable colors
  - after matching, fill a dst hole iff:
      1. the dst hole is rectangular in its own bbox, and
      2. the matched src color appears among the control pixels

Small trace:
  - suppose dst has four holes in rank order:
      H1 = 2x3 rectangle, H2 = L-shape, H3 = 2x2 rectangle, H4 = staircase
  - matched src colors are:
      H1<-red, H2<-blue, H3<-green, H4<-red
  - top-edge control pixels are [red, blue]
  - original solver fills all four
  - D18-only would fill H1 and H3
  - control-only would fill H1, H2, H4
  - combo fills only H1:
      H2 blocked by shape, H3 blocked by control color, H4 blocked by shape

b) Complexity calibration
-------------------------
The control vocabulary is intentionally small: 1-3 distinct control colors, always reused
from src fills. This keeps the legend human-visible and avoids hidden numeric conventions.
The generator enforces three simultaneously visible categories:
  - at least one filled rectangular allowed hole
  - at least one rectangular disallowed hole
  - at least one non-rectangular allowed hole
That prevents degeneration into either single variant.

c) Hidden-assumption review against the original main()
-------------------------------------------------------
Original assumptions and status after fusion:
  - "all non-bg components are object pixels":
      no longer true because border control pixels are also non-bg; solver must exclude
      border non-bg cells before object extraction.
  - "every matched hole is fillable":
      no longer true; the final paint loop must branch on rectangularity and control color.
  - "dst is the object with larger hole area":
      still true because control pixels are excluded and src remains filled while dst keeps
      bg holes in the input.
  - "outline color is the only large non-bg color":
      still true inside extracted object subgrids because controls are not included there.
  - "transform search is unaffected":
      still true because control pixels do not participate in centroid matching.

d) Placement legality
---------------------
Objects are placed first using random_free_location with border=1, guaranteeing that the
outer canvas border stays empty. Control pixels are then written only onto that free border,
so they never overlap objects or connect to them. Src and dst still use the original
non-overlap placement pipeline.

e) Visual observability
-----------------------
Humans can infer the rule from a few examples because:
  - control colors visibly repeat colors already present in src subsprites
  - some rectangular holes with allowed colors fill
  - some rectangular holes with disallowed colors stay empty
  - some non-rectangular holes with allowed colors also stay empty
So neither color alone nor shape alone explains the output; the conjunction is observable.

f) Visual cleanliness
---------------------
Controls are isolated single pixels on the top border, away from both objects. The objects
keep the original clean two-object layout and the D18 rectangular-vs-irregular contrast.
No marks are drawn inside the objects beyond the original fills.

g) Change inferability
----------------------
The color rule is structural, not numeric: "border colors name the transferable src colors".
The shape rule is also structural: "only rectangular dst holes can receive transfer".
The generator keeps the number of distinct fill colors moderate and reuses colors across
multiple cavities, so a small number of examples is enough to support the legend.

h) Quantitative anti-degeneration metric
----------------------------------------
For each generated instance, compute:
  - filled_count            = rectangular AND allowed
  - blocked_by_control      = rectangular AND not allowed
  - blocked_by_shape        = non-rectangular AND allowed
  - blocked_by_both         = non-rectangular AND not allowed
The generator rejects any instance with filled_count == 0, blocked_by_control == 0, or
blocked_by_shape == 0. It also rejects cases where the combo output matches either:
  - the original "fill everything" output
  - the D18-only "rectangles only" output
  - the D16-only "allowed colors only" output

i) New solver assumptions that the generator must guarantee
-----------------------------------------------------------
  - all control pixels lie on the outer border and are the only border non-bg cells
  - control colors come from src fill colors, never from bg or outline
  - there are exactly two 8-connected non-control objects
  - rectangularity is visually meaningful, so hole heights/widths are >= 2 where needed
  - at least one allowed rectangular hole exists, otherwise output would equal the empty dst
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


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
    m = grid != int(bg)
    pts = np.argwhere(m)
    if pts.size == 0:
        return grid.copy()
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _is_rect_pts(pts: np.ndarray) -> bool:
    if pts.size == 0:
        return False
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(len(pts)) == int((r1 - r0 + 1) * (c1 - c0 + 1))


def _enclosed_bg_components(sub: np.ndarray, bg: int) -> list[dict]:
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
                "cent": (
                    int(round(float(pts[:, 0].mean()))),
                    int(round(float(pts[:, 1].mean()))),
                ),
                "size": int(len(pts)),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "is_rect": bool(_is_rect_pts(pts)),
            }
        )
    return holes


def _outline_col(sub: np.ndarray, bg: int) -> int:
    cnt = Counter(map(int, sub.flatten()))
    opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
    return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)


def _colored_components(sub: np.ndarray, bg: int) -> list[dict]:
    outc = _outline_col(sub, bg)
    m = (sub != int(bg)) & (sub != int(outc))
    lab, n = label_mask(m, connectivity=4)
    subs = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        col = Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0]
        subs.append(
            {
                "cent": (
                    int(round(float(pts[:, 0].mean()))),
                    int(round(float(pts[:, 1].mean()))),
                ),
                "color": int(col),
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


def _control_points_and_palette(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], set[int]]:
    h, w = grid.shape
    ctrl_pts: list[tuple[int, int]] = []
    for c in range(w):
        if int(grid[0, c]) != int(bg):
            ctrl_pts.append((0, int(c)))
        if int(h) > 1 and int(grid[h - 1, c]) != int(bg):
            ctrl_pts.append((int(h - 1), int(c)))
    for r in range(1, h - 1):
        if int(grid[r, 0]) != int(bg):
            ctrl_pts.append((int(r), 0))
        if int(w) > 1 and int(grid[r, w - 1]) != int(bg):
            ctrl_pts.append((int(r), int(w - 1)))
    palette = {int(grid[r, c]) for r, c in ctrl_pts}
    return ctrl_pts, palette


def _extract_two_objects_without_controls(grid: np.ndarray, bg: int) -> tuple[list[np.ndarray], set[int]]:
    ctrl_pts, allowed_colors = _control_points_and_palette(grid, bg)
    obj_mask = grid != int(bg)
    for r, c in ctrl_pts:
        obj_mask[int(r), int(c)] = False
    comps = _components(obj_mask, connectivity=8)
    assert len(comps) >= 2
    groups = [[comps[0]], [comps[1]]]
    objs = [_extract_group_subgrid(grid, bg, grp) for grp in groups]
    return objs, allowed_colors


def _solve_pair(
    dest: np.ndarray,
    src: np.ndarray,
    bg: int,
    *,
    selective: bool,
    allowed_colors: set[int] | None,
) -> np.ndarray:
    dest = np.array(dest, dtype=int).copy()
    src = np.array(src, dtype=int)

    dest_holes = _enclosed_bg_components(dest, bg)
    src_subs = _colored_components(src, bg)
    if not dest_holes or not src_subs:
        return _tight_crop_non_bg(dest, bg)

    dest_pts = [h["cent"] for h in dest_holes]
    dest_keys, d_rr, d_cc = _rank_keys(dest_pts)

    best = None
    for k in range(4):
        for flip in [False, True]:
            tpts = [
                _transform_point(
                    s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip)
                )
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
    for hole in dest_holes:
        key = (int(d_rr[int(hole["cent"][0])]), int(d_cc[int(hole["cent"][1])]))
        dest_list.append((key, hole["cent"], hole))
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

    for (_k1, _p1, col), (_k2, _p2, hole) in zip(src_list, dest_list):
        if bool(selective) and not bool(hole["is_rect"]):
            continue
        if allowed_colors is not None and int(col) not in allowed_colors:
            continue
        for rr, cc in hole["pts"]:
            dest[int(rr), int(cc)] = int(col)

    return _tight_crop_non_bg(dest, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    1. Border non-bg pixels form a color legend; their colors define an allow-list.
    2. Ignore those border control pixels when extracting the two main objects.
    3. Match src colored subsprites to dst holes by centroid-rank order.
    4. Fill only dst holes that are both rectangular and matched to an allowed color.
    5. Tight-crop the resulting dst object.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    objs, allowed_colors = _extract_two_objects_without_controls(g, bg)
    holes_list = [_enclosed_bg_components(o, bg) for o in objs]
    hole_area = [sum(int(h["size"]) for h in hh) for hh in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    return _solve_pair(
        objs[dest_i],
        objs[src_i],
        bg,
        selective=True,
        allowed_colors=allowed_colors,
    )


# =============================================================================
# Generator
# =============================================================================


CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)

CAVITY_COUNT_RANGE = (4, 8)
FILL_COLOR_COUNT_RANGE = (3, 7)
ALLOWED_COLOR_COUNT_RANGE = (1, 3)

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

COLOR_MISMATCH_RATE = 0.10

MIN_FILLED_HOLES = 1
MIN_BLOCKED_BY_CONTROL = 1
MIN_BLOCKED_BY_SHAPE = 1

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2

MAX_GEN_ATTEMPTS = 700
MAX_LAYOUT_R_BUDGET = 24


# =============================================================================
# Shape primitives
# =============================================================================


def _random_rect_shape(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _is_rect_mask(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(len(pts)) == int((r1 - r0 + 1) * (c1 - c0 + 1))


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


def _gen_dst_src_pair(
    h: int, w: int, *, symmetric: bool, dst_rect: bool
) -> tuple[np.ndarray, np.ndarray]:
    if bool(dst_rect):
        dst = _random_rect_shape(h, w)
        src = _gen_nonrect_shape(h, w, symmetric=symmetric)
    else:
        dst = _gen_nonrect_shape(h, w, symmetric=symmetric)
        src = _random_rect_shape(h, w)
    return dst, src


# =============================================================================
# Layout design
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
        pts = np.argwhere(lab == int(cid))
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


# =============================================================================
# Color assignment and control selection
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


def _selection_stats(layout: list[dict], colors: list[int], allowed_colors: set[int]) -> dict[str, int]:
    filled = 0
    blocked_by_control = 0
    blocked_by_shape = 0
    blocked_by_both = 0
    for slot, col in zip(layout, colors):
        fillable = bool(slot["fillable"])
        allowed = int(col) in allowed_colors
        if fillable and allowed:
            filled += 1
        elif fillable and not allowed:
            blocked_by_control += 1
        elif (not fillable) and allowed:
            blocked_by_shape += 1
        else:
            blocked_by_both += 1
    return {
        "filled": int(filled),
        "blocked_by_control": int(blocked_by_control),
        "blocked_by_shape": int(blocked_by_shape),
        "blocked_by_both": int(blocked_by_both),
    }


def _choose_allowed_colors(layout: list[dict], colors: list[int]) -> tuple[set[int], dict[str, int]] | None:
    used = sorted(set(map(int, colors)))
    if len(used) < 2:
        return None

    lo, hi = ALLOWED_COLOR_COUNT_RANGE
    hi = min(hi, len(used) - 1)
    if hi < lo:
        return None

    candidates: list[tuple[set[int], dict[str, int]]] = []
    for k in range(lo, hi + 1):
        for subset in combinations(used, k):
            allowed = set(map(int, subset))
            stats = _selection_stats(layout, colors, allowed)
            if (
                stats["filled"] >= MIN_FILLED_HOLES
                and stats["blocked_by_control"] >= MIN_BLOCKED_BY_CONTROL
                and stats["blocked_by_shape"] >= MIN_BLOCKED_BY_SHAPE
            ):
                candidates.append((allowed, stats))
    if not candidates:
        return None
    return random.choice(candidates)


# =============================================================================
# Generation pipeline
# =============================================================================


def _pick_bg_and_palette() -> tuple[int, list[int]]:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    return bg, palette


def _build_object_pair(
    layout: list[dict], colors: list[int], bg: int, outline_col: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, set[int], dict[str, int]] | None:
    n_cavities = len(layout)
    fillable_count = sum(1 for slot in layout if bool(slot["fillable"]))
    skipped_count = n_cavities - fillable_count
    if fillable_count < 2 or skipped_count < 2:
        return None

    chosen = _choose_allowed_colors(layout, colors)
    if chosen is None:
        return None
    allowed_colors, selection_stats = chosen

    dst_shapes = [slot["shape_dst"] for slot in layout]
    src_shapes = [slot["shape_src"] for slot in layout]

    dst_cropped, _dst_filled_all = _render_object(layout, dst_shapes, colors, bg, outline_col, filled=False)
    if dst_cropped is None:
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

    expected = _solve_pair(
        dst_cropped.copy(),
        src_cropped.copy(),
        bg,
        selective=True,
        allowed_colors=allowed_colors,
    )
    full_fill = _solve_pair(
        dst_cropped.copy(),
        src_cropped.copy(),
        bg,
        selective=False,
        allowed_colors=None,
    )
    rect_only = _solve_pair(
        dst_cropped.copy(),
        src_cropped.copy(),
        bg,
        selective=True,
        allowed_colors=None,
    )
    control_only = _solve_pair(
        dst_cropped.copy(),
        src_cropped.copy(),
        bg,
        selective=False,
        allowed_colors=allowed_colors,
    )

    if np.array_equal(expected, dst_cropped):
        return None
    if np.array_equal(expected, full_fill):
        return None
    if np.array_equal(expected, rect_only):
        return None
    if np.array_equal(expected, control_only):
        return None
    return src_cropped, dst_cropped, expected, allowed_colors, selection_stats


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


def _add_control_pixels(canvas: np.ndarray, bg: int, allowed_colors: set[int]) -> np.ndarray | None:
    g = np.array(canvas, dtype=int).copy()
    cols = list(range(1, int(g.shape[1]) - 1))
    if len(cols) < len(allowed_colors):
        return None
    chosen_cols = sorted(random.sample(cols, len(allowed_colors)))
    for c, col in zip(chosen_cols, sorted(allowed_colors)):
        if int(g[0, int(c)]) != int(bg):
            return None
        g[0, int(c)] = int(col)
    return g


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
    src, dst, expected, allowed_colors, selection_stats = built

    canvas = _place_on_canvas(src, dst, bg)
    if canvas is None:
        return None
    canvas = _add_control_pixels(canvas, bg, allowed_colors)
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

    # Extra guard: the control signal must actually matter on this concrete canvas.
    full_fill = _solve_pair(dst.copy(), src.copy(), bg, selective=False, allowed_colors=None)
    rect_only = _solve_pair(dst.copy(), src.copy(), bg, selective=True, allowed_colors=None)
    control_only = _solve_pair(dst.copy(), src.copy(), bg, selective=False, allowed_colors=allowed_colors)
    if np.array_equal(actual, full_fill) or np.array_equal(actual, rect_only) or np.array_equal(actual, control_only):
        return None
    if (
        selection_stats["filled"] < MIN_FILLED_HOLES
        or selection_stats["blocked_by_control"] < MIN_BLOCKED_BY_CONTROL
        or selection_stats["blocked_by_shape"] < MIN_BLOCKED_BY_SHAPE
    ):
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
