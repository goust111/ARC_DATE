# Combo variant of 3a25b0d8 (index 20)
# combo_spec: D10/2+D15/0
# Name: 多配对分组+轮廓色区分

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location

"""
Design analysis for combo D10/2 + D15/0
=======================================

a) Concrete behavior and trace
------------------------------
- Original task:
  one filled `src` object and one empty-hole `dst` object share one outline color.
  The solver identifies the larger-hole object as `dst`, matches cavity centroids by
  row/col rank, fills the holes, then tight-crops the filled `dst`.
- Combo task:
  the canvas now contains 2-3 independent `src`/`dst` pairs, and each pair uses its
  own fixed outline-color code. Within a pair, `src` and `dst` outlines are different
  colors, so role recognition and pair grouping are both color-driven before geometry
  matching starts.
- Concrete mini-trace:
  pair A = blue-outline `src` + red-outline `dst`
  pair B = green-outline `src` + yellow-outline `dst`
  Input contains four objects. The solver first groups objects by the visible code
  `blue->red` and `green->yellow`, then for each group executes the original centroid-
  rank transfer rule, and finally keeps only the filled `dst` objects, preserving their
  input positions and tight-cropping the union of all filled `dst`s.
- Difference from original:
  the old logic was “pick one donor and one receiver globally”.
  The new logic is “partition the scene into 2-3 color-coded donor/receiver pairs, then
  solve each pair locally”.

b) Complexity calibration
-------------------------
- I keep the original cavity-layout and centroid-rank matching mechanics inside each pair.
- I do not introduce arbitrary per-sample color arithmetic. Instead I use a small fixed
  codebook of visible outline pairs:
    blue->red, green->yellow, grey->pink
- Fill colors are restricted to orange/teal/maroon, so outline colors never double as
  fills. That keeps the pair code visually legible across 2-4 examples.
- The generator uses 2 pairs whenever the background blocks one code color, and 2 or 3
  pairs otherwise. This makes the D10 change visible without exploding scene density.

c) Implicit-assumption audit of the original main()
---------------------------------------------------
- `_extract_two_objects()`:
  broken. The combo input has 4-6 objects, not 2.
- `dest_i = argmax(hole_area)` and `src_i = 1 - dest_i`:
  broken. There is no single global `dst` or `src`.
- Shared-outline assumption:
  broken by D15/0. A pair is no longer “same outline color”, so outline color can no
  longer be ignored after object extraction.
- One zip fill loop over one `src` list and one `dst` list:
  broken. Filling must happen per matched pair.
- Tight crop of one destination object:
  changed. The output is now the union of all filled `dst` objects.

d) Placement legality
---------------------
- Multiple objects make the original 22-30 canvas unreliable, so the generator slightly
  enlarges the canvas and slightly tightens object size bounds.
- Objects are still placed with at least one-cell padding under 8-connectivity, so they
  never merge into one connected component.
- Output placement is inherited from the placed `dst` objects only; no extra motion is
  introduced after filling.

e) Visual observability
-----------------------
- The pairing rule is human-visible because the same outline-color code recurs across
  examples: blue donors always fill red receivers, green donors fill yellow receivers,
  and grey donors fill pink receivers.
- D15/0 is also visible because donor and receiver outlines differ within every pair.
- The surviving outlines in the output are always the `dst` colors, reinforcing the role
  cue from training examples.

f) Visual cleanliness
---------------------
- No legends, labels, or overlays are added.
- Outline colors are reserved for outlines only; fill colors are reserved for cavity
  contents only. This prevents code-color / fill-color collisions.
- Objects remain separated by padding, so multi-pair scenes stay readable.

g) Color-change inferability
----------------------------
- There is no hidden recoloring rule on cavity fills: each hole still copies the matching
  source subsprite color directly.
- The only color logic added by the combo is the small fixed outline-pair codebook.
- Because the codebook has only three visible mappings and they recur across examples,
  humans can infer it from few-shot ARC supervision.

h) Post-implementation combo metric
-----------------------------------
- Core metric: generated inputs should contain 2 or 3 valid outline-coded pairs.
- Supporting metrics:
  1. every used outline color belongs to the fixed codebook,
  2. every used pair has one colored `src` and one hole-only `dst`,
  3. the input should rarely collapse to 1 pair, and never collapse to shared outlines.
- This detects degeneration back toward the original single-pair or same-outline task.

i) New solver assumptions and generator guarantees
--------------------------------------------------
- Assumption: each object's outline color is its dominant non-bg color.
  Guarantee: generator rejects samples violating that.
- Assumption: outline colors are drawn from the fixed `src_outline -> dst_outline`
  codebook, and outline colors never appear as fills.
  Guarantee: generator uses disjoint outline/fill palettes.
- Assumption: each used pair id contributes exactly one filled `src` and one empty-hole
  `dst`.
  Guarantee: generator constructs pairs directly and verifies them with the oracle check.
- Assumption: output is the union of the filled `dst` objects, not the union of all
  objects.
  Guarantee: expected-output construction pastes only filled `dst`s before cropping.
"""


PAIR_COLOR_CODES = [
    (1, 2),  # blue -> red
    (3, 4),  # green -> yellow
    (5, 6),  # grey -> pink
]
FILL_COLOR_CODES = [7, 8, 9]  # orange, teal, maroon
SRC_TO_PAIR = {int(src): idx for idx, (src, _dst) in enumerate(PAIR_COLOR_CODES)}
DST_TO_PAIR = {int(dst): idx for idx, (_src, dst) in enumerate(PAIR_COLOR_CODES)}

CANVAS_SIZE_RANGE_TWO = (26, 34)
CANVAS_SIZE_RANGE_THREE = (30, 38)
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

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 1

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


def _tight_crop_non_bg(sub: np.ndarray, bg: int) -> np.ndarray:
    m = sub != int(bg)
    if not bool(m.any()):
        return sub.copy()
    b = bbox(m)
    assert b is not None
    return crop_bbox(sub, b)


def _solve_pair(src: np.ndarray, dest: np.ndarray, bg: int) -> np.ndarray:
    dest = np.array(dest, dtype=int).copy()
    src = np.array(src, dtype=int)

    dest_holes = _enclosed_bg_components(dest, bg)
    src_outline = _outline_col(src, bg)
    src_subs = _colored_components(src, bg, src_outline)
    if not dest_holes or not src_subs:
        return _tight_crop_non_bg(dest, bg)

    dest_pts = [tuple(h["cent"]) for h in dest_holes]
    dest_keys, d_rr, d_cc = _rank_keys(dest_pts)

    best = None
    for k in range(4):
        for flip in [False, True]:
            tpts = [_transform_point(tuple(s["cent"]), int(src.shape[0]), int(src.shape[1]), int(k), bool(flip)) for s in src_subs]
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
        dest_list.append((key, tuple(h["cent"]), h))
    dest_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    tpts_all = [_transform_point(tuple(s["cent"]), int(src.shape[0]), int(src.shape[1]), int(k), bool(flip)) for s in src_subs]
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
    - find all objects,
    - infer each object's outline color and role,
    - pair objects via the fixed outline-color codebook,
    - solve each pair independently with the original centroid-rank fill rule,
    - return the tight crop of the union of filled dst objects.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    objects = _extract_objects(g, bg)
    assert objects

    infos = []
    for obj in objects:
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
                "bbox": obj["bbox"],
                "grid": sub,
                "outline": int(outline),
                "holes": holes,
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
        if len(srcs) != 1 or len(dsts) != 1:
            continue
        src_info = srcs[0]
        dst_info = dsts[0]
        filled = _solve_pair(src_info["grid"], dst_info["grid"], bg)
        r0, c0, _r1, _c1 = dst_info["bbox"]
        paste(out, filled, int(r0), int(c0), transparent=bg)
        filled_any = True

    if not filled_any:
        # Structural fallback for malformed inputs: preserve the old single-pair behavior.
        src_candidates = [info for info in infos if info["fills"]]
        dst_candidates = [info for info in infos if info["holes"] and not info["fills"]]
        if src_candidates and dst_candidates:
            src_info = src_candidates[0]
            dst_info = max(dst_candidates, key=lambda d: sum(int(h["size"]) for h in d["holes"]))
            filled = _solve_pair(src_info["grid"], dst_info["grid"], bg)
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

    return src_cropped, dst_cropped, dst_filled


def _place_scene(
    records: list[dict],
    bg: int,
    n_pairs: int,
) -> tuple[np.ndarray, dict[tuple[str, int], tuple[int, int]]] | None:
    if int(n_pairs) >= 3:
        canvas_h = random.randint(*CANVAS_SIZE_RANGE_THREE)
        canvas_w = random.randint(*CANVAS_SIZE_RANGE_THREE)
    else:
        canvas_h = random.randint(*CANVAS_SIZE_RANGE_TWO)
        canvas_w = random.randint(*CANVAS_SIZE_RANGE_TWO)
    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)

    placements: dict[tuple[str, int], tuple[int, int]] = {}
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
            return None
        paste(canvas, rec["grid"], int(r0), int(c0), transparent=bg)
        placements[(str(rec["kind"]), int(rec["pair_id"]))] = (int(r0), int(c0))
    return canvas, placements


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
        records.append({"kind": "src", "pair_id": pid, "grid": src})
        records.append({"kind": "dst", "pair_id": pid, "grid": dst})
        expected_records.append({"pair_id": pid, "grid": dst_filled})

    placed = _place_scene(records, bg, n_pairs)
    if placed is None:
        return None
    canvas, placements = placed

    expected_canvas = np.full_like(canvas, bg)
    for rec in expected_records:
        pos = placements[("dst", int(rec["pair_id"]))]
        paste(expected_canvas, rec["grid"], int(pos[0]), int(pos[1]), transparent=bg)
    expected = _tight_crop_non_bg(expected_canvas, bg)

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
