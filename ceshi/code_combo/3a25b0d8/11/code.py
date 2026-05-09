# Combo variant of 3a25b0d8 (index 11)
# combo_spec: D3/0+D18/2
# Name: 部分预填色dst+选择性填充

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, get_bg, label_mask, paste, random_free_location

"""
Design analysis for the combo variant
=====================================

combo_rule:
  summary:
    Keep the original centroid-rank matching between src colored subsprites and dst slot
    positions, but change dst so that:
      1. some fillable slots are already prefilled with the correct color
      2. among the remaining empty holes, only rectangular holes should be filled
      3. non-rectangular empty holes must stay blank
    The solver therefore has to distinguish three visible slot states inside dst:
      - prefilled: already colored interior component, leave unchanged
      - pending: empty rectangular hole, fill from matched src color
      - skipped: empty non-rectangular hole, preserve as bg

  behavior_trace:
    Small concrete trace:
      - src has 5 subsprites ordered by centroid rank:
          top-left=red, top-right=green, middle=yellow, bottom-left=blue, bottom-right=red
      - original dst had 5 empty holes; original solver would fill all 5.
      - combo dst may show:
          top-left already green? impossible, because prefills must be correct
          top-right already green (prefilled)
          middle empty rectangle (pending -> becomes yellow)
          bottom-left empty staircase (skipped -> stays bg)
          bottom-right empty rectangle (pending -> becomes red)
      - old base solver breaks because it only reads empty holes and loses the prefilled
        slot from the layout.
      - D3-only solver recovers the full layout, but would still incorrectly fill the
        staircase.
      - combo solver uses the full slot template
          prefilled colored interiors + empty holes
        to align against src, then paints only the empty slots whose shape is rectangular.

  complexity_calibration:
    - The positional rule from the original task remains necessary; colors still come from
      src by centroid-rank, not by local copying inside dst.
    - The selective condition is deliberately binary and visual: rectangle vs non-rectangle.
      This avoids hidden numeric thresholds while remaining easy to observe from 2-4 ARC
      examples.
    - Generator forces all three categories to appear in each sample:
        prefilled_count >= 1
        pending_fill_count >= 1
        skipped_count >= 1
      so the combo cannot silently collapse to either single variant.
    - Prefills are chosen only from fillable rectangular slots. This is a deliberate
      simplification to keep the human-facing rule crisp: colored dst interiors are
      examples of "already solved" slots, while empty rectangles vs empty irregular holes
      show the selective branch directly.

  implicit_assumption_audit:
    - Original `dest_i = argmax(hole_area)` assumed dst is the only object with holes.
      Still mostly true, but partial prefilling reduces hole area. The new solver chooses
      dst by enclosed-hole count first and falls back to hole area only if needed.
    - Original `_outline_col` used global non-bg majority. With partial prefilling, a large
      fill color can rival the outline, so we switch to a border-based outline detector:
      after tight extraction, only outline pixels touch the subgrid border.
    - Original destination template was `dest_holes`. That assumption fails because some
      destination slots are already colored. We now build the template from:
        enclosed bg holes + enclosed interior colored components
    - Original fill loop wrote to every matched dst hole. That assumption fails because the
      combo adds a second decision step. The new write condition is:
        slot.kind == "empty" and slot.is_rect
    - New assumption introduced by the solver:
        prefilled interior components are already correct and should be preserved.
      The generator guarantees this by deriving prefills from the same slot-color mapping as
      the expected output.

  placement_legality:
    - src and dst are still generated independently, validated as single 8-connected
      objects, and placed with `random_free_location`, so there is no overlap or overflow.
    - Partial prefilling does not change the object bbox because it only recolors interior
      hole cells.

  visual_observability:
    - A human can see two distinct cues in dst:
        some cavities already contain the final color
        among the still-empty cavities, rectangles get filled while irregular holes do not
    - The rule does not depend on hidden color ids or coordinate arithmetic beyond visible
      slot order.

  visual_cleanliness:
    - No extra markers or third object types are introduced.
    - We restrict the cavity vocabulary to heights/widths that make rectangularity visible,
      avoiding ambiguous one-cell-wide "rectangles".
    - Prefilling is limited to existing cavity shapes, so there is no additional clutter.

  color_inference:
    - Colors are still explained structurally by src position.
    - Repeated colors remain common because mirror-pair assignment is preserved.
    - The selective branch is shape-driven only, preventing color-based ambiguity.

  generator_guarantees:
    - `len(dest_slots) == len(src_subsprites)` under the solver's own detectors.
    - dst always contains at least one prefilled slot, one pending rectangular hole, and
      one skipped irregular hole.
    - src/dst shapes for corresponding slots are still independently mutated so the task
      cannot be solved by shape copying instead of position matching.
    - Generated samples are rejected if combo output matches the "fill every empty hole"
      baseline on the same partial-dst input, because that would mean the selective branch
      is not actually active.

  review_of_new_main_assumptions:
    - `_dest_slots` assumes every prefilled slot appears as one interior non-outline
      4-connected component; generator fills whole cavities with one color, so true.
    - `slot.is_rect` is only evaluated on empty holes; generator keeps skipped holes empty,
      so the rectangularity test is directly visible.
    - Matching assumes rank-key uniqueness is stable across src and dst. Generator uses the
      original layout construction, so slot centroid rows/cols stay rankable.
"""


# =============================================================================
# Solver helpers
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


def _extract_two_objects(grid: np.ndarray, bg: int) -> list[list[dict]]:
    comps = _components(grid != int(bg), connectivity=8)
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
    for comp in group:
        for rr, cc in comp["pts"]:
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


def _is_rect_pts(pts: np.ndarray) -> bool:
    if pts.size == 0:
        return False
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(len(pts)) == int((r1 - r0 + 1) * (c1 - c0 + 1))


def _enclosed_bg_components(sub: np.ndarray, bg: int) -> list[dict]:
    mask = sub == int(bg)
    lab, n = label_mask(mask, connectivity=4)
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
                "is_rect": bool(_is_rect_pts(pts)),
            }
        )
    return holes


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
    opts = [(color, n) for color, n in cnt.items() if int(color) != int(bg)]
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
        color = Counter(map(int, vals.flatten())).most_common(1)[0][0]
        comps.append(
            {
                "pts": pts,
                "cent": (
                    int(round(float(pts[:, 0].mean()))),
                    int(round(float(pts[:, 1].mean()))),
                ),
                "size": int(len(pts)),
                "color": int(color),
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
                "color": comp["color"],
            }
        )
    return slots


def _rank_keys(pts: list[tuple[int, int]]):
    rows = sorted(set(int(r) for r, _ in pts))
    cols = sorted(set(int(c) for _, c in pts))
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


def _solve_pair(dest: np.ndarray, src: np.ndarray, bg: int, *, selective: bool) -> np.ndarray:
    dest = np.array(dest, dtype=int).copy()
    src = np.array(src, dtype=int)

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

    for (_src_key, _src_pt, color), (_dst_key, _dst_pt, slot) in zip(src_list, dest_list):
        if slot["kind"] != "empty":
            continue
        if bool(selective) and not bool(slot["is_rect"]):
            continue
        for rr, cc in slot["pts"]:
            dest[int(rr), int(cc)] = int(color)

    return _tight_crop_non_bg(dest, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    1. Find src and partially filled dst.
    2. Build dst's slot layout from both empty holes and already-colored interior slots.
    3. Match slot order to src by centroid-rank pattern.
    4. Fill only dst slots that are both empty and rectangular.
    5. Preserve prefilled slots and irregular empty holes, then tight-crop dst.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    groups = _extract_two_objects(g, bg)
    assert len(groups) >= 2
    objs = [_extract_group_subgrid(g, bg, grp) for grp in groups]

    hole_lists = [_enclosed_bg_components(obj, bg) for obj in objs]
    hole_counts = [len(hs) for hs in hole_lists]
    if max(hole_counts) > 0:
        dest_i = int(np.argmax(hole_counts))
    else:
        hole_area = [sum(int(h["size"]) for h in hs) for hs in hole_lists]
        dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    return _solve_pair(objs[dest_i], objs[src_i], bg, selective=True)


# =============================================================================
# Generator constants
# =============================================================================


CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)

CAVITY_COUNT_RANGE = (4, 8)
FILL_COLOR_COUNT_RANGE = (3, 7)

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
RECT_HOLE_PROBABILITY = 0.58

COLOR_MISMATCH_RATE = 0.10
PREFILL_MID_BIAS_PROB = 0.75

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2

MIN_FILLABLE_HOLES = 2
MIN_SKIPPED_HOLES = 1

MAX_GEN_ATTEMPTS = 500
MAX_LAYOUT_R_BUDGET = 24


# =============================================================================
# Generator helpers
# =============================================================================


def _random_rect_shape(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _is_rect_mask(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
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


def _mirror_shape(mask: np.ndarray) -> np.ndarray:
    return mask[:, ::-1].copy()


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


def _render_object_data(
    layout: list[dict],
    shapes: list[np.ndarray],
    colors: list[int],
    bg: int,
    outline_col: int,
) -> dict | None:
    placed_slots = []
    for slot_idx, (slot, shape) in enumerate(zip(layout, shapes)):
        rr, cc = np.where(shape)
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


def _count_enclosed_bg_holes(grid: np.ndarray, bg: int) -> int:
    return len(_enclosed_bg_components(grid, bg))


def _count_filled_interior_components(grid: np.ndarray, bg: int, outline_col: int) -> int:
    return len(_colored_components(grid, bg, outline_col=outline_col, interior_only=True))


def _count_total_dest_slots(grid: np.ndarray, bg: int, outline_col: int) -> int:
    return _count_enclosed_bg_holes(grid, bg) + _count_filled_interior_components(grid, bg, outline_col)


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
    for idx in range(len(colors)):
        if random.random() < COLOR_MISMATCH_RATE:
            colors[idx] = random.choice(fill_pool)
    return colors


def _pick_bg_and_palette() -> tuple[int, list[int]]:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    return bg, palette


def _choose_prefilled_indices(fillable_indices: list[int]) -> list[int]:
    assert len(fillable_indices) >= 2
    counts = list(range(1, len(fillable_indices)))
    mid_counts = [k for k in counts if 1 < k < len(fillable_indices) - 1]
    if mid_counts and random.random() < PREFILL_MID_BIAS_PROB:
        n_prefilled = random.choice(mid_counts)
    else:
        n_prefilled = random.choice(counts)
    return sorted(random.sample(fillable_indices, n_prefilled))


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
    layout: list[dict], colors: list[int], bg: int, outline_col: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n_cavities = len(layout)
    fillable_indices = [i for i, slot in enumerate(layout) if bool(slot["fillable"])]
    skipped_indices = [i for i, slot in enumerate(layout) if not bool(slot["fillable"])]
    if len(fillable_indices) < MIN_FILLABLE_HOLES or len(skipped_indices) < MIN_SKIPPED_HOLES:
        return None

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
    if not _size_in_spec(dst_empty):
        return None

    src_data = _render_object_data(layout, src_shapes, colors, bg, outline_col)
    if src_data is None:
        return None
    src = src_data["filled"]
    if not _is_8connected(src, bg):
        return None
    if _count_4conn_cavities(src, bg, outline_col) != n_cavities:
        return None
    if not _size_in_spec(src):
        return None

    prefilled_indices = _choose_prefilled_indices(fillable_indices)
    pending_fill_indices = [idx for idx in fillable_indices if idx not in set(prefilled_indices)]
    if not pending_fill_indices:
        return None

    dst_partial = _build_partial_dst(dst_empty, dst_data["slot_pts"], colors, prefilled_indices)
    if not _is_8connected(dst_partial, bg):
        return None
    if not _size_in_spec(dst_partial):
        return None

    if _count_filled_interior_components(dst_partial, bg, outline_col) != len(prefilled_indices):
        return None
    if _count_total_dest_slots(dst_partial, bg, outline_col) != n_cavities:
        return None

    expected = _solve_pair(dst_partial.copy(), src.copy(), bg, selective=True)
    full_fill = _solve_pair(dst_partial.copy(), src.copy(), bg, selective=False)
    if expected.shape != dst_partial.shape:
        return None
    if np.array_equal(expected, _tight_crop_non_bg(dst_partial, bg)):
        return None
    if np.array_equal(expected, full_fill):
        return None
    return src, dst_partial, expected


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


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    groups = _extract_two_objects(g, bg)
    if len(groups) < 2:
        return {
            "prefilled": 0,
            "pending_fill": 0,
            "skipped": 0,
            "total_slots": 0,
            "selective_diff": 0,
        }
    objs = [_extract_group_subgrid(g, bg, grp) for grp in groups]
    hole_lists = [_enclosed_bg_components(obj, bg) for obj in objs]
    hole_counts = [len(hs) for hs in hole_lists]
    if max(hole_counts) > 0:
        dest_i = int(np.argmax(hole_counts))
    else:
        hole_area = [sum(int(h["size"]) for h in hs) for hs in hole_lists]
        dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)

    dest = objs[dest_i]
    src = objs[src_i]
    outc = _outline_col(dest, bg)
    dest_slots = _dest_slots(dest, bg)

    prefilled = sum(1 for slot in dest_slots if slot["kind"] == "filled")
    pending_fill = sum(1 for slot in dest_slots if slot["kind"] == "empty" and bool(slot["is_rect"]))
    skipped = sum(1 for slot in dest_slots if slot["kind"] == "empty" and not bool(slot["is_rect"]))

    selective = _solve_pair(dest.copy(), src.copy(), bg, selective=True)
    full_fill = _solve_pair(dest.copy(), src.copy(), bg, selective=False)
    return {
        "prefilled": int(prefilled),
        "pending_fill": int(pending_fill),
        "skipped": int(skipped),
        "total_slots": int(len(dest_slots)),
        "filled_components": int(_count_filled_interior_components(dest, bg, outc)),
        "selective_diff": int(not np.array_equal(selective, full_fill)),
    }


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
    if metrics["prefilled"] <= 0 or metrics["pending_fill"] <= 0 or metrics["skipped"] <= 0:
        return None
    if metrics["selective_diff"] != 1:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
