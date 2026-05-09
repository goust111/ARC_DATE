# Combo variant of 3a25b0d8 (index 30)
# combo_spec: D10/2+D11/0+D15/0
# Name: 多配对+干扰框架+轮廓色分组

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location

"""
Design analysis for combo D10/2 + D11/0 + D15/0
================================================

a) Concrete behavior and trace
------------------------------
- Original task:
  the canvas has exactly one filled `src` and one empty-hole `dst`. They share one outline
  color, so the solver can ignore color grouping and just decide which of the two objects
  is the receiver, then transfer colors by centroid row/column rank.
- This combo:
  the canvas has 2-3 true pairs, plus 1-2 extra empty frames. Each true pair uses a fixed
  visible outline-color code, for example blue-outline `src` -> red-outline `dst`, or
  green-outline `src` -> yellow-outline `dst`. The distractors reuse one of the receiver
  colors, so color only narrows the search to a group; it does not identify the true `dst`
  by itself.
- Concrete mini-trace:
  pair A = blue `src`, red true `dst`, red distractor
  pair B = green `src`, yellow true `dst`
  The solver first groups objects by the fixed outline-color codebook, so blue only
  interacts with red frames and green only interacts with yellow frames. Inside the red
  group, there are two empty frames, so the solver must run the original centroid-rank
  check and keep only the red frame whose hole arrangement matches the blue source.
  Then it solves the yellow group the same way. Output is the union of the filled true
  `dst` objects only, tight-cropped.
- Difference from original:
  the old logic was "pick one donor and one receiver globally".
  The new logic is "split the scene into 2-3 color-coded families, discard same-color
  distractor frames inside each family, then solve each surviving pair locally".

b) Complexity calibration
-------------------------
- I keep the original cavity-layout and centroid-rank transfer rule inside each family.
- I use a small fixed outline codebook:
    blue -> red, green -> yellow, grey -> pink
  This keeps the pairing rule human-visible across a few ARC examples.
- Fill colors are restricted to orange/teal/maroon, so outline colors never appear as
  fills. That avoids accidental color collisions.
- Distractors reuse receiver colors instead of inventing new colors. This makes the D11
  change visible without turning the scene into a color-legend puzzle.
- I cap pair count at 3 and distractor count at 2, and slightly shrink object sizes while
  enlarging the canvas. The multi-object scene stays legible and placement stays feasible.

c) Implicit-assumption audit of the original main()
---------------------------------------------------
- `_extract_two_objects()`:
  broken. The combo input has 5-8 objects, not 2.
- `dest_i = argmax(hole_area)` and `src_i = 1 - dest_i`:
  broken. There is no single global source/target.
- Shared-outline assumption:
  broken by D15/0. Source and destination use different outline colors, so role and pair
  grouping must consult outline color.
- "One dst per pair color" assumption from the D10/2 + D15/0 combo:
  broken by D11/0. A receiver-color group can now contain the true `dst` plus one extra
  empty frame, so color grouping alone is insufficient.
- One global `zip(src_list, dest_list)` fill loop:
  broken. Filling must happen independently inside each pair after the correct frame is
  selected.

d) Placement legality
---------------------
- Objects are still placed with at least one-cell padding under 8-connectivity, so they
  never merge into one component.
- Because 2-3 pairs plus 1-2 distractors can yield up to 8 objects, the canvas is larger
  than in the original task and placement retries sample fresh canvas sizes.
- Expected output is constructed by pasting only the filled true `dst`s at their placed
  locations, so no post-placement motion is introduced.

e) Visual observability
-----------------------
- Humans can see the repeated outline codebook across examples: blue donors feed red
  receivers, green donors feed yellow receivers, grey donors feed pink receivers.
- D15/0 is visible because donor and receiver outlines differ inside every true pair.
- D11/0 is visible because some receiver-color groups contain two same-color empty frames,
  but only one survives into the output after being filled.

f) Visual cleanliness
---------------------
- No labels, text, or overlays are added.
- Outline colors and fill colors are disjoint, so object roles remain visually clean.
- Distractors are the same kind of outline-only frame as the real `dst`, which makes the
  selection challenge clear without adding clutter in the same local area.

g) Color-change inferability
----------------------------
- Cavity fill colors are still copied directly from the matched source subsprites.
- The only added color logic is the small fixed source-outline -> destination-outline
  codebook. Because it recurs across examples and uses few colors, it remains inferable.
- Distractors do not introduce any new recoloring rule; they only add same-color receiver
  candidates whose geometry does not match.

h) Post-implementation combo metric
-----------------------------------
- Core metric:
  generated scenes must contain 2 or 3 true outline-coded pairs and at least 1 distractor.
- Supporting metrics:
  1. every used true pair has `src_outline != dst_outline`,
  2. at least one receiver-color group has multiple empty frames,
  3. only one empty frame in that group matches the source under centroid-rank signature.
- These metrics detect degeneration back to the D10/2 + D15/0 case with no distractor,
  or back to the original one-pair task.

i) New solver assumptions and generator guarantees
--------------------------------------------------
- Assumption: each object's outline color is the dominant non-background color.
  Guarantee: generator rejects objects violating that.
- Assumption: outline colors come from the fixed codebook, and fill colors are disjoint.
  Guarantee: generator draws from separate outline and fill palettes only.
- Assumption: each used pair id contributes exactly one filled `src`, at least one empty
  receiver-color frame, and exactly one frame in that group matches the source signature.
  Guarantee: generator constructs true pairs directly and rejects distractors that match.
- Assumption: output is the union of the filled true `dst`s, not all empty frames.
  Guarantee: expected-output construction pastes only the true filled receivers.
"""


PAIR_COLOR_CODES = [
    (1, 2),  # blue -> red
    (3, 4),  # green -> yellow
    (5, 6),  # grey -> pink
]
FILL_COLOR_CODES = [7, 8, 9]  # orange, teal, maroon
SRC_TO_PAIR = {int(src): idx for idx, (src, _dst) in enumerate(PAIR_COLOR_CODES)}
DST_TO_PAIR = {int(dst): idx for idx, (_src, dst) in enumerate(PAIR_COLOR_CODES)}

CANVAS_SIZE_RANGE_TWO = (30, 38)
CANVAS_SIZE_RANGE_THREE = (34, 42)
OBJECT_H_RANGE = (6, 13)
OBJECT_W_RANGE = (6, 11)

CAVITY_COUNT_RANGE_TWO = (3, 5)
CAVITY_COUNT_RANGE_THREE = (3, 4)

ROW_HEIGHT_CHOICES = [1, 2, 2]
LAYOUT_K_CHOICES_FULL = [1, 2, 2]
LAYOUT_K_CHOICES_SMALL = [1, 2]
ROW_GAP_CHOICES = [1, 2, 2]

CENTERED_SLOT_WIDTHS = [3, 5]
PAIR_SLOT_WIDTHS = [2, 3]
SIDE_SLOT_WIDTHS = [2, 2, 3]
MID_SLOT_WIDTHS = [1, 3]
PAIR_AXIS_GAP_CHOICES = [1, 1, 3]
TRIO_MID_SIDE_GAP = 1

NONRECT_PROBABILITY = 0.65
MAX_MUTATE_ATTEMPTS = 40
COLOR_MISMATCH_RATE = 0.10

DISTRACTOR_COUNT_CHOICES = [1, 2]
DISTRACTOR_AREA_RATIO_RANGE = (0.75, 1.35)
MAX_DISTRACTOR_ATTEMPTS = 200

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 1
CANVAS_PLACEMENT_ATTEMPTS = 30

MAX_GEN_ATTEMPTS = 400
MAX_LAYOUT_R_BUDGET = 16


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


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    out = []
    for comp in _components(grid != int(bg)):
        r0, c0, r1, c1 = comp["bbox"]
        sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        for rr, cc in comp["pts"]:
            sub[int(rr) - int(r0), int(cc) - int(c0)] = int(grid[int(rr), int(cc)])
        out.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "grid": sub,
                "size": int(comp["size"]),
            }
        )
    return out


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
        cent = (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))
        holes.append({"pts": pts, "cent": cent, "size": int(len(pts))})
    return holes


def _outline_col(sub: np.ndarray, bg: int) -> int:
    vals = [int(v) for v in sub.flatten() if int(v) != int(bg)]
    if not vals:
        return int(bg)
    return int(Counter(vals).most_common(1)[0][0])


def _colored_components(sub: np.ndarray, bg: int, outline_col: int | None = None) -> list[dict]:
    outc = int(_outline_col(sub, bg) if outline_col is None else outline_col)
    mask = (sub != int(bg)) & (sub != int(outc))
    lab, n = label_mask(mask, connectivity=4)
    subs = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        col = Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0]
        cent = (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))
        subs.append({"pts": pts, "cent": cent, "color": int(col), "size": int(len(pts))})
    return subs


def _rank_keys(pts: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], dict[int, int], dict[int, int]]:
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
        _h2, w2 = int(h0), int(w0)
    elif int(k) == 1:
        rr, cc = int(w0 - 1 - c), int(r)
        _h2, w2 = int(w0), int(h0)
    elif int(k) == 2:
        rr, cc = int(h0 - 1 - r), int(w0 - 1 - c)
        _h2, w2 = int(h0), int(w0)
    else:
        rr, cc = int(c), int(h0 - 1 - r)
        _h2, w2 = int(w0), int(h0)
    if bool(flip):
        cc = int(w2 - 1 - cc)
    return int(rr), int(cc)


def _signature_from_points(pts: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    if not pts:
        return tuple()
    keys, _rr, _cc = _rank_keys(pts)
    return tuple(sorted((int(rk), int(ck)) for rk, ck in keys))


def _all_transformed_signatures(src_subs: list[dict], src_shape: tuple[int, int]) -> set[tuple[tuple[int, int], ...]]:
    h0, w0 = map(int, src_shape)
    sigs: set[tuple[tuple[int, int], ...]] = set()
    for k in range(4):
        for flip in (False, True):
            pts = [_transform_point(tuple(s["cent"]), h0, w0, int(k), bool(flip)) for s in src_subs]
            sigs.add(_signature_from_points(pts))
    return sigs


def _match_transform(
    src_subs: list[dict], src_shape: tuple[int, int], dest_holes: list[dict]
) -> tuple[int, bool] | None:
    if not src_subs or not dest_holes or len(src_subs) != len(dest_holes):
        return None
    target_sig = _signature_from_points([tuple(h["cent"]) for h in dest_holes])
    h0, w0 = map(int, src_shape)
    for k in range(4):
        for flip in (False, True):
            pts = [_transform_point(tuple(s["cent"]), h0, w0, int(k), bool(flip)) for s in src_subs]
            if _signature_from_points(pts) == target_sig:
                return int(k), bool(flip)
    return None


def _tight_crop_non_bg(sub: np.ndarray, bg: int) -> np.ndarray:
    m = sub != int(bg)
    if not bool(m.any()):
        return sub.copy()
    b = bbox(m)
    assert b is not None
    return crop_bbox(sub, b)


def _hole_area(sub: np.ndarray, bg: int) -> int:
    return int(sum(int(h["size"]) for h in _enclosed_bg_components(sub, bg)))


def _solve_pair(
    src: np.ndarray, dest: np.ndarray, bg: int, match: tuple[int, bool] | None = None
) -> np.ndarray:
    dest = np.array(dest, dtype=int).copy()
    src = np.array(src, dtype=int)

    dest_holes = _enclosed_bg_components(dest, bg)
    src_outline = _outline_col(src, bg)
    src_subs = _colored_components(src, bg, src_outline)
    if not dest_holes or not src_subs:
        return _tight_crop_non_bg(dest, bg)

    if match is None:
        match = _match_transform(src_subs, src.shape, dest_holes)
    if match is None:
        match = (0, False)
    k, flip = match

    dest_pts = [tuple(h["cent"]) for h in dest_holes]
    _dest_keys, d_rr, d_cc = _rank_keys(dest_pts)

    dest_list = []
    for h in dest_holes:
        key = (int(d_rr[int(h["cent"][0])]), int(d_cc[int(h["cent"][1])]))
        dest_list.append((key, tuple(h["cent"]), h))
    dest_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    tpts_all = [
        _transform_point(tuple(s["cent"]), int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
        for s in src_subs
    ]
    _keys, s_rr, s_cc = _rank_keys(tpts_all)

    src_list = []
    for s in src_subs:
        tp = _transform_point(tuple(s["cent"]), int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
        key = (int(s_rr[int(tp[0])]), int(s_cc[int(tp[1])]))
        src_list.append((key, tp, int(s["color"])))
    src_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    for (_k1, _p1, col), (_k2, _p2, hole) in zip(src_list, dest_list):
        for rr, cc in hole["pts"]:
            dest[int(rr), int(cc)] = int(col)
    return dest


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo solver:
    1. extract all objects,
    2. use the fixed outline-color codebook to form 2-3 source/receiver families,
    3. inside each family, reject same-color distractor frames via centroid-rank matching,
    4. fill only the true receivers,
    5. return the tight crop of the union of all filled true receivers.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    objects = _extract_objects(g, bg)
    assert objects

    infos = []
    for idx, obj in enumerate(objects):
        sub = obj["grid"]
        outline = _outline_col(sub, bg)
        holes = _enclosed_bg_components(sub, bg)
        fills = _colored_components(sub, bg, outline)
        role = None
        pair_id = None
        if int(outline) in SRC_TO_PAIR:
            role = "src"
            pair_id = int(SRC_TO_PAIR[int(outline)])
        elif int(outline) in DST_TO_PAIR:
            role = "dst"
            pair_id = int(DST_TO_PAIR[int(outline)])
        infos.append(
            {
                "index": int(idx),
                "bbox": obj["bbox"],
                "grid": sub,
                "size": int(obj["size"]),
                "outline": int(outline),
                "holes": holes,
                "hole_area": int(sum(int(h["size"]) for h in holes)),
                "fills": fills,
                "role": role,
                "pair_id": pair_id,
            }
        )

    out = np.full_like(g, int(bg))
    filled_any = False

    for pid in range(len(PAIR_COLOR_CODES)):
        srcs = [info for info in infos if info["pair_id"] == int(pid) and info["role"] == "src" and info["fills"]]
        dsts = [info for info in infos if info["pair_id"] == int(pid) and info["role"] == "dst" and info["holes"]]
        if len(srcs) != 1 or not dsts:
            continue

        src_info = srcs[0]
        matches: list[tuple[dict, tuple[int, bool]]] = []
        for dst_info in dsts:
            match = _match_transform(src_info["fills"], src_info["grid"].shape, dst_info["holes"])
            if match is not None:
                matches.append((dst_info, match))

        if matches:
            dst_info, match = max(
                matches,
                key=lambda item: (
                    int(item[0]["hole_area"]),
                    int(item[0]["size"]),
                    -int(item[0]["index"]),
                ),
            )
        else:
            dst_info = max(
                dsts,
                key=lambda item: (int(item["hole_area"]), int(item["size"]), -int(item["index"])),
            )
            match = _match_transform(src_info["fills"], src_info["grid"].shape, dst_info["holes"])

        filled = _solve_pair(src_info["grid"], dst_info["grid"], bg, match)
        r0, c0, _r1, _c1 = dst_info["bbox"]
        paste(out, filled, int(r0), int(c0), transparent=bg)
        filled_any = True

    if not filled_any:
        src_candidates = [info for info in infos if info["fills"]]
        dst_candidates = [info for info in infos if info["holes"] and not info["fills"]]
        if src_candidates and dst_candidates:
            src_info = src_candidates[0]
            dst_info = max(dst_candidates, key=lambda d: (int(d["hole_area"]), int(d["size"])))
            match = _match_transform(src_info["fills"], src_info["grid"].shape, dst_info["holes"])
            filled = _solve_pair(src_info["grid"], dst_info["grid"], bg, match)
            r0, c0, _r1, _c1 = dst_info["bbox"]
            paste(out, filled, int(r0), int(c0), transparent=bg)
            filled_any = True

    if not filled_any:
        return _tight_crop_non_bg(g, bg)

    return _tight_crop_non_bg(out, bg)


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
    slots.append({"r0": cur_r, "col_offset": -(w // 2), "shape_dst": dst, "shape_src": src, "pair_id": pair_id})


def _add_mirror_pair_slots(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    w = random.choice(PAIR_SLOT_WIDTHS)
    dst_l, src_l = _gen_dst_src_pair(row_h, w, symmetric=False)
    dst_r = _mirror_shape(dst_l)
    src_r = _mirror_shape(src_l)
    axis_gap = random.choice(PAIR_AXIS_GAP_CHOICES)
    if axis_gap % 2 == 0:
        axis_gap += 1
    half = axis_gap // 2
    slots.append({"r0": cur_r, "col_offset": -(half + w), "shape_dst": dst_l, "shape_src": src_l, "pair_id": pair_id})
    slots.append({"r0": cur_r, "col_offset": half + 1, "shape_dst": dst_r, "shape_src": src_r, "pair_id": pair_id})


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
    slots.append({"r0": cur_r, "col_offset": -mid_half, "shape_dst": dst_m, "shape_src": src_m, "pair_id": pair_id + 1})
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


def _dominant_non_bg_color(grid: np.ndarray, bg: int) -> int:
    vals = [int(v) for v in grid.flatten() if int(v) != int(bg)]
    if not vals:
        return int(bg)
    return int(Counter(vals).most_common(1)[0][0])


def _outline_is_unique_mode(grid: np.ndarray, bg: int, outline_col: int) -> bool:
    vals = [int(v) for v in grid.flatten() if int(v) != int(bg)]
    if not vals:
        return False
    cnt = Counter(vals)
    if int(outline_col) not in cnt:
        return False
    others = [n for c, n in cnt.items() if int(c) != int(outline_col)]
    return cnt[int(outline_col)] > (max(others) if others else -1)


def _assign_pair_colors(layout: list[dict], fill_pool: list[int]) -> list[int]:
    unique_pair_ids = []
    for slot in layout:
        if slot["pair_id"] not in unique_pair_ids:
            unique_pair_ids.append(slot["pair_id"])
    pair_color_map = {pid: fill_pool[i % len(fill_pool)] for i, pid in enumerate(unique_pair_ids)}
    colors = [int(pair_color_map[slot["pair_id"]]) for slot in layout]
    for i in range(len(colors)):
        if random.random() < COLOR_MISMATCH_RATE:
            colors[i] = int(random.choice(fill_pool))
    return colors


def _cavity_count_range(n_pairs: int) -> tuple[int, int]:
    return CAVITY_COUNT_RANGE_THREE if int(n_pairs) >= 3 else CAVITY_COUNT_RANGE_TWO


def _build_object_pair(
    layout: list[dict],
    colors: list[int],
    bg: int,
    src_outline_col: int,
    dst_outline_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n_cavities = len(layout)
    dst_shapes = [slot["shape_dst"] for slot in layout]
    src_shapes = [slot["shape_src"] for slot in layout]

    dst_cropped, dst_filled = _render_object(layout, dst_shapes, colors, bg, dst_outline_col, filled=False)
    if dst_cropped is None or dst_filled is None:
        return None
    if not _is_8connected(dst_cropped, bg):
        return None
    if _count_enclosed_bg_holes(dst_cropped, bg) != n_cavities:
        return None
    if not _size_in_spec(dst_cropped):
        return None
    if _dominant_non_bg_color(dst_cropped, bg) != int(dst_outline_col):
        return None

    src_cropped, _ = _render_object(layout, src_shapes, colors, bg, src_outline_col, filled=True)
    if src_cropped is None:
        return None
    if not _is_8connected(src_cropped, bg):
        return None
    if _count_4conn_cavities(src_cropped, bg, src_outline_col) != n_cavities:
        return None
    if not _size_in_spec(src_cropped):
        return None
    if not _outline_is_unique_mode(src_cropped, bg, src_outline_col):
        return None
    if _dominant_non_bg_color(src_cropped, bg) != int(src_outline_col):
        return None
    if _match_transform(
        _colored_components(src_cropped, bg, src_outline_col),
        src_cropped.shape,
        _enclosed_bg_components(dst_cropped, bg),
    ) is None:
        return None

    return src_cropped, dst_cropped, dst_filled


def _build_empty_frame(layout: list[dict], colors: list[int], bg: int, outline_col: int) -> np.ndarray | None:
    shapes = [slot["shape_dst"] for slot in layout]
    empty_cropped, _filled = _render_object(layout, shapes, colors, bg, outline_col, filled=False)
    if empty_cropped is None:
        return None
    if not _is_8connected(empty_cropped, bg):
        return None
    if _count_enclosed_bg_holes(empty_cropped, bg) != len(layout):
        return None
    if not _size_in_spec(empty_cropped):
        return None
    if _dominant_non_bg_color(empty_cropped, bg) != int(outline_col):
        return None
    return empty_cropped


def _build_distractor_for_pair(
    src: np.ndarray,
    dst: np.ndarray,
    colors: list[int],
    bg: int,
    dst_outline_col: int,
) -> np.ndarray | None:
    src_outline = _outline_col(src, bg)
    src_subs = _colored_components(src, bg, src_outline)
    if not src_subs:
        return None
    src_sigs = _all_transformed_signatures(src_subs, src.shape)
    target_holes = len(src_subs)
    target_area = max(1, _hole_area(dst, bg))

    for _attempt in range(MAX_DISTRACTOR_ATTEMPTS):
        layout = _design_layout(target_holes)
        if len(layout) != target_holes:
            continue
        candidate = _build_empty_frame(layout, colors, bg, dst_outline_col)
        if candidate is None:
            continue
        holes = _enclosed_bg_components(candidate, bg)
        sig = _signature_from_points([tuple(h["cent"]) for h in holes])
        if sig in src_sigs:
            continue
        if _match_transform(src_subs, src.shape, holes) is not None:
            continue
        area = _hole_area(candidate, bg)
        ratio = float(area) / float(target_area)
        if not (DISTRACTOR_AREA_RATIO_RANGE[0] <= ratio <= DISTRACTOR_AREA_RATIO_RANGE[1]):
            continue
        return candidate
    return None


def _place_scene(
    records: list[dict],
    bg: int,
    n_pairs: int,
) -> tuple[np.ndarray, dict[tuple[str, int, int], tuple[int, int]]] | None:
    for _attempt in range(CANVAS_PLACEMENT_ATTEMPTS):
        if int(n_pairs) >= 3 or len(records) >= 7:
            canvas_h = random.randint(*CANVAS_SIZE_RANGE_THREE)
            canvas_w = random.randint(*CANVAS_SIZE_RANGE_THREE)
        else:
            canvas_h = random.randint(*CANVAS_SIZE_RANGE_TWO)
            canvas_w = random.randint(*CANVAS_SIZE_RANGE_TWO)
        canvas = np.full((canvas_h, canvas_w), bg, dtype=int)

        placements: dict[tuple[str, int, int], tuple[int, int]] = {}
        ok = True
        for rec in sorted(records, key=lambda r: int(r["grid"].shape[0] * r["grid"].shape[1]), reverse=True):
            try:
                r0, c0 = random_free_location(
                    canvas,
                    rec["grid"],
                    transparent=bg,
                    border=PLACEMENT_BORDER,
                    padding=PLACEMENT_PADDING,
                )
            except ValueError:
                ok = False
                break
            paste(canvas, rec["grid"], int(r0), int(c0), transparent=bg)
            placements[rec["key"]] = (int(r0), int(c0))
        if ok:
            return canvas, placements
    return None


def _choose_pair_codes(bg: int) -> list[tuple[int, int, int]]:
    valid = []
    for pid, (src_col, dst_col) in enumerate(PAIR_COLOR_CODES):
        if int(bg) in (int(src_col), int(dst_col)):
            continue
        valid.append((int(pid), int(src_col), int(dst_col)))
    if len(valid) <= 2:
        return valid
    if random.random() < 0.5:
        return valid
    return random.sample(valid, 2)


def _choose_distractor_pair_ids(pair_ids: list[int]) -> list[int]:
    n = min(len(pair_ids), random.choice(DISTRACTOR_COUNT_CHOICES))
    return random.sample(pair_ids, n)


def _scene_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    objects = _extract_objects(g, bg)
    groups: dict[int, dict[str, int]] = {}
    n_src = 0
    n_dst_like = 0
    for obj in objects:
        sub = obj["grid"]
        outline = _outline_col(sub, bg)
        holes = _enclosed_bg_components(sub, bg)
        fills = _colored_components(sub, bg, outline)
        if int(outline) in SRC_TO_PAIR and fills:
            pid = int(SRC_TO_PAIR[int(outline)])
            groups.setdefault(pid, {"src": 0, "dst": 0, "match": 0})
            groups[pid]["src"] += 1
            n_src += 1
        elif int(outline) in DST_TO_PAIR and holes:
            pid = int(DST_TO_PAIR[int(outline)])
            groups.setdefault(pid, {"src": 0, "dst": 0, "match": 0})
            groups[pid]["dst"] += 1
            n_dst_like += 1

    for pid, info in groups.items():
        src_outline = PAIR_COLOR_CODES[int(pid)][0]
        src_candidates = [
            obj["grid"]
            for obj in objects
            if _outline_col(obj["grid"], bg) == int(src_outline)
            and _colored_components(obj["grid"], bg, int(src_outline))
        ]
        dst_candidates = [
            obj["grid"]
            for obj in objects
            if _outline_col(obj["grid"], bg) == int(PAIR_COLOR_CODES[int(pid)][1])
            and _enclosed_bg_components(obj["grid"], bg)
        ]
        if len(src_candidates) == 1:
            src_subs = _colored_components(src_candidates[0], bg, int(src_outline))
            for dst_grid in dst_candidates:
                if _match_transform(src_subs, src_candidates[0].shape, _enclosed_bg_components(dst_grid, bg)) is not None:
                    info["match"] += 1

    n_true_pairs = sum(1 for info in groups.values() if info["src"] == 1 and info["match"] == 1)
    n_groups_with_competition = sum(1 for info in groups.values() if info["dst"] >= 2)
    n_distractors = sum(max(0, info["dst"] - info["match"]) for info in groups.values())
    return {
        "n_objects": int(len(objects)),
        "n_src": int(n_src),
        "n_dst_like": int(n_dst_like),
        "n_true_pairs": int(n_true_pairs),
        "n_groups_with_competition": int(n_groups_with_competition),
        "n_distractors": int(n_distractors),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, 9)
    pair_codes = _choose_pair_codes(bg)
    n_pairs = len(pair_codes)
    if n_pairs < 2:
        return None

    fill_pool = [int(c) for c in FILL_COLOR_CODES if int(c) != int(bg)]
    if len(fill_pool) < 2:
        return None

    records: list[dict] = []
    expected_records: list[dict] = []
    pair_records: dict[int, dict] = {}
    cavity_lo, cavity_hi = _cavity_count_range(n_pairs)

    for pid, src_outline_col, dst_outline_col in pair_codes:
        layout = _design_layout(random.randint(cavity_lo, cavity_hi))
        if len(layout) < cavity_lo:
            return None
        colors = _assign_pair_colors(layout, fill_pool)
        built = _build_object_pair(layout, colors, bg, src_outline_col, dst_outline_col)
        if built is None:
            return None
        src, dst, dst_filled = built
        records.append({"kind": "src", "pair_id": pid, "key": ("src", int(pid), 0), "grid": src})
        records.append({"kind": "dst", "pair_id": pid, "key": ("dst", int(pid), 0), "grid": dst})
        expected_records.append({"key": ("dst", int(pid), 0), "grid": dst_filled})
        pair_records[int(pid)] = {
            "src": src,
            "dst": dst,
            "dst_outline_col": int(dst_outline_col),
            "colors": list(colors),
        }

    for slot_idx, pid in enumerate(_choose_distractor_pair_ids(list(pair_records.keys())), start=1):
        pair_rec = pair_records[int(pid)]
        distractor = _build_distractor_for_pair(
            pair_rec["src"],
            pair_rec["dst"],
            pair_rec["colors"],
            bg,
            int(pair_rec["dst_outline_col"]),
        )
        if distractor is None:
            return None
        records.append(
            {
                "kind": "distractor",
                "pair_id": int(pid),
                "key": ("distractor", int(pid), int(slot_idx)),
                "grid": distractor,
            }
        )

    placed = _place_scene(records, bg, n_pairs)
    if placed is None:
        return None
    canvas, placements = placed

    expected_canvas = np.full_like(canvas, bg)
    for rec in expected_records:
        pos = placements[rec["key"]]
        paste(expected_canvas, rec["grid"], int(pos[0]), int(pos[1]), transparent=bg)
    expected = _tight_crop_non_bg(expected_canvas, bg)

    metrics = _scene_metrics(canvas)
    if metrics["n_true_pairs"] < 2 or metrics["n_distractors"] < 1 or metrics["n_groups_with_competition"] < 1:
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
