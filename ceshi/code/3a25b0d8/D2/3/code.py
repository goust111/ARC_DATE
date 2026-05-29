# Variant of 3a25b0d8
# Dimension: D2 — 源对象结构 (src)
# Change: 单个 8-连通精灵：轮廓骨架 + 彩色子精灵填充空腔；LR 对称；4-9 个空腔... → src 子精灵包含多色渐变
#
# Design analysis
# spec:
#   variant_focus: D2 only. Keep the two-object scene, the LR-symmetric outline skeleton,
#     and centroid-rank matching. Change only the source cavity internals from single-color
#     fills to two-color structured fills.
#   concrete_visual:
#     src:
#       object: one 8-connected outline object, same as original.
#       cavities: 4-9 4-connected cavity masks.
#       fill_rule: every cavity uses exactly two non-bg colors in a local checkerboard
#         pattern anchored to that cavity slot's bbox top-left.
#       visibility_constraints:
#         - every cavity has area >= 2
#         - every cavity occupies its full seed bbox, so the local coordinate frame is
#           stable and visible in both src and dst
#         - source and destination shapes still differ, but they share the same slot bbox
#     dst:
#       object: same outline-only object family as before, with empty holes.
#       output: each dst hole is filled with the matched source cavity's full checkerboard
#         pattern, not a single color.
#   behavior_trace:
#     original:
#       - `_colored_components()` labels non-outline, non-bg cells and collapses each cavity
#         to one majority color.
#       - filling dst writes one scalar color into every hole cell.
#     variant:
#       - source cavity extraction must keep the full two-color parity rule.
#       - filling dst reconstructs the checkerboard over the destination hole bbox and writes
#         parity-dependent colors per cell.
#       difference:
#         a 2x3 cavity like
#           2 3 2
#           3 2 3
#         used to be reduced to either color 2 or 3 by majority logic; now it must stay
#         structured and reproduce as a checkerboard inside the matched dst hole.
#   complexity_calibration:
#     - Use one pattern family (checkerboard) globally so humans can infer the rule from a
#       few examples.
#     - Still vary the two colors per cavity and keep independent src/dst cavity shapes, so
#       the task is visibly different from the original and not trivialized to pure color
#       copying.
#     - Avoid many pattern families because sparse irregular masks would make pattern
#       identification ambiguous from only 2-4 examples.
#   implicit_assumption_audit_main:
#     - Original `_outline_col()` assumed the outline was the most frequent non-bg color.
#       With multi-color cavities that can fail, so the variant reads the outline from the
#       object's outer border instead.
#     - Original `_colored_components()` assumed one color per cavity and stored only that
#       scalar. This is invalid now; the new extractor stores bbox + parity colors.
#     - Original unconditional hole fill `dest[rr, cc] = col` was harmless only because every
#       cavity had one color. Now each cell color depends on local `(r + c) % 2`.
#   placement_legality:
#     - Generator still places two independently cropped objects on a 22-30 canvas using the
#       shared helper that prevents overlap and keeps border/padding constraints.
#     - Output-first validation still checks `main(input) == expected`.
#   observability:
#     - Humans can see that each src cavity contains alternating colors, and the output fills
#       each dst hole with the same alternation by arrangement.
#     - No ordering depends on numeric color ids; only spatial centroid ranks are used.
#   visual_cleanliness:
#     - Keep a single outline color and limited fill palette.
#     - Remove 1-row cavity seeds so cavities can differ in shape while still showing a clear
#       two-color interior.
#   color_inference:
#     - Each cavity uses only two colors, repeated across multiple cells by the checkerboard,
#       so the color relation is local and structurally supported rather than arbitrary.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill holes in the target object using the matched source cavity's two-color checkerboard.

    Input structure:
    - bg = background.
    - Two 8-connected objects share an outline color.
    - Source cavities are filled with two-color checkerboards.
    - Target cavities are empty bg holes.

    Rule:
    - Find the two objects.
    - Target = object with larger enclosed bg area.
    - Match source cavities to target holes by centroid row/col rank order.
    - Rebuild the matched source cavity's checkerboard inside the target hole bbox.
    - Tight-crop the filled target.
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
            holes.append(
                {
                    "pts": pts,
                    "cent": cent,
                    "size": int(len(pts)),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
        return holes

    def _outline_col(sub: np.ndarray) -> int:
        border_vals = []
        if sub.size == 0:
            return int(bg)
        border_vals.extend(map(int, sub[0, :].tolist()))
        border_vals.extend(map(int, sub[-1, :].tolist()))
        if sub.shape[0] > 2:
            border_vals.extend(map(int, sub[1:-1, 0].tolist()))
            border_vals.extend(map(int, sub[1:-1, -1].tolist()))
        border_non_bg = [v for v in border_vals if int(v) != int(bg)]
        if border_non_bg:
            return int(Counter(border_non_bg).most_common(1)[0][0])
        cnt = Counter(map(int, sub.flatten()))
        opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
        return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)

    def _infer_checker_spec(
        sub: np.ndarray, pts: np.ndarray
    ) -> tuple[tuple[int, int, int, int], int, int]:
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        vals = [int(sub[int(rr), int(cc)]) for rr, cc in pts]
        uniq = sorted(set(vals))
        if not uniq:
            return (int(r0), int(c0), int(r1), int(c1)), int(bg), int(bg)
        candidates = []
        if len(uniq) == 1:
            candidates = [(int(uniq[0]), int(uniq[0]))]
        else:
            for even_col in uniq:
                for odd_col in uniq:
                    if int(even_col) != int(odd_col):
                        candidates.append((int(even_col), int(odd_col)))
        best_pair = candidates[0]
        best_mismatch = None
        for even_col, odd_col in candidates:
            mismatch = 0
            for rr, cc in pts:
                expect = int(even_col) if (int(rr - r0 + cc - c0) % 2 == 0) else int(odd_col)
                mismatch += int(int(sub[int(rr), int(cc)]) != expect)
            if best_mismatch is None or mismatch < best_mismatch:
                best_mismatch = mismatch
                best_pair = (int(even_col), int(odd_col))
        return (int(r0), int(c0), int(r1), int(c1)), int(best_pair[0]), int(best_pair[1])

    def _source_cavity_specs(sub: np.ndarray) -> list[dict]:
        outc = _outline_col(sub)
        m = (sub != int(bg)) & (sub != int(outc))
        lab, n = label_mask(m, connectivity=4)
        specs = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            bbox0, even_col, odd_col = _infer_checker_spec(sub, pts)
            cent = (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))
            specs.append(
                {
                    "cent": cent,
                    "bbox": bbox0,
                    "even_color": int(even_col),
                    "odd_color": int(odd_col),
                }
            )
        return specs

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
    src_specs = _source_cavity_specs(src)
    if not dest_holes or not src_specs:
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
                for s in src_specs
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
        for s in src_specs
    ]
    _keys, s_rr, s_cc = _rank_keys(tpts_all)

    src_list = []
    for s in src_specs:
        tp = _transform_point(s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
        key = (int(s_rr[int(tp[0])]), int(s_cc[int(tp[1])]))
        src_list.append((key, tp, int(s["even_color"]), int(s["odd_color"])))
    src_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    for (_k1, _p1, even_col, odd_col), (_k2, _p2, h) in zip(src_list, dest_list):
        r0, c0, _r1, _c1 = h["bbox"]
        for rr, cc in h["pts"]:
            local_parity = int((int(rr) - int(r0) + int(cc) - int(c0)) % 2)
            dest[int(rr), int(cc)] = int(even_col) if local_parity == 0 else int(odd_col)

    m = dest != int(bg)
    pts = np.argwhere(m)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return dest[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


# --------------- generator ---------------

CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)

# Cavity structure
CAVITY_COUNT_RANGE = (4, 9)
FILL_COLOR_COUNT_RANGE = (4, 7)

# Layout row patterns (weighted choices)
ROW_HEIGHT_CHOICES = [2, 2, 3, 3, 3]
LAYOUT_K_CHOICES_FULL = [1, 2, 2, 3, 3]
LAYOUT_K_CHOICES_SMALL = [1, 2, 2]
ROW_GAP_CHOICES = [1, 1, 2]

# Cavity slot widths per layout type
CENTERED_SLOT_WIDTHS = [3, 3, 5, 5, 7]
PAIR_SLOT_WIDTHS = [2, 2, 3, 3, 4]
SIDE_SLOT_WIDTHS = [2, 2, 3]
MID_SLOT_WIDTHS = [3]
PAIR_AXIS_GAP_CHOICES = [1, 1, 3]
TRIO_MID_SIDE_GAP = 1

# Shape mutation
NONRECT_PROBABILITY = 0.8
MAX_MUTATE_ATTEMPTS = 40
SHAPE_RETRY_LIMIT = 30
PAIR_BUILD_RETRY_LIMIT = 40

# Pattern/color assignment
COLOR_MISMATCH_RATE = 0.10

# Placement
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2

# Generation retry
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


def _shape_uses_full_bbox(mask: np.ndarray) -> bool:
    b = bbox(mask)
    if b is None:
        return False
    h, w = mask.shape
    return tuple(map(int, b)) == (0, 0, int(h - 1), int(w - 1))


def _shape_supports_checker(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) < 2:
        return False
    if not _shape_uses_full_bbox(mask):
        return False
    parities = {int((int(r) + int(c)) % 2) for r, c in pts}
    return len(parities) == 2


def _gen_shape(h: int, w: int, *, nonrect: bool, symmetric: bool = False) -> np.ndarray:
    rect = _random_rect_shape(h, w)
    if not nonrect or h * w < 4:
        return rect
    for _ in range(SHAPE_RETRY_LIMIT):
        m = rect.copy()
        limit = max(1, (h * w) // 2)
        m = _mutate_shape(m, n_remove_max=limit, symmetric=symmetric)
        if _shape_supports_checker(m):
            return m
    return rect


def _mirror_shape(m: np.ndarray) -> np.ndarray:
    return m[:, ::-1].copy()


def _gen_dst_src_pair(h: int, w: int, *, symmetric: bool) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(PAIR_BUILD_RETRY_LIMIT):
        nonrect_dst = random.random() < NONRECT_PROBABILITY
        nonrect_src = random.random() < NONRECT_PROBABILITY
        dst = _gen_shape(h, w, nonrect=nonrect_dst, symmetric=symmetric)
        src = _gen_shape(h, w, nonrect=nonrect_src, symmetric=symmetric)
        if _shape_supports_checker(dst) and _shape_supports_checker(src) and not np.array_equal(dst, src):
            return dst, src
    rect = _random_rect_shape(h, w)
    if symmetric:
        # deterministic fallback: full rectangle vs rectangle missing one interior-adjacent corner pair
        src = rect.copy()
        src[0, 0] = False
        src[0, -1] = False
        if _shape_supports_checker(src):
            return rect, src
    src = rect.copy()
    src[0, 0] = False
    if _shape_supports_checker(src):
        return rect, src
    return rect, rect


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


def _slot_checker_color(pattern_spec: dict, local_r: int, local_c: int) -> int:
    parity = int((int(local_r) + int(local_c)) % 2)
    return int(pattern_spec["even_color"]) if parity == 0 else int(pattern_spec["odd_color"])


def _render_object(
    layout: list[dict], shapes: list[np.ndarray], pattern_specs: list[dict], bg: int, outline_col: int, *, filled: bool
) -> tuple[np.ndarray | None, np.ndarray | None]:
    placed_slots = []
    for slot_idx, (slot, sh) in enumerate(zip(layout, shapes)):
        rr, cc = np.where(sh)
        if len(rr) == 0:
            continue
        placed_slots.append((slot_idx, slot["r0"] + rr, slot["col_offset"] + cc, rr, cc))
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
    for slot_idx, rows, cols, local_rr, local_cc in placed_slots:
        rr = rows - r_min + pad
        cc = cols - c_min + pad
        spec = pattern_specs[slot_idx]
        for gr, gc, lr, lc in zip(rr, cc, local_rr, local_cc):
            canvas[int(gr), int(gc)] = _slot_checker_color(spec, int(lr), int(lc))
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


def _pick_bg_and_palette() -> tuple[int, list[int]]:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    return bg, palette


def _assign_pair_patterns(layout: list[dict], fill_pool: list[int]) -> list[dict]:
    unique_pair_ids: list[int] = []
    for slot in layout:
        if slot["pair_id"] not in unique_pair_ids:
            unique_pair_ids.append(slot["pair_id"])
    random.shuffle(unique_pair_ids)
    base_specs: dict[int, dict] = {}
    n_fill = len(fill_pool)
    offset = random.randint(1, max(1, n_fill - 1)) if n_fill > 1 else 0
    for i, pid in enumerate(unique_pair_ids):
        a = int(fill_pool[i % n_fill])
        b = int(fill_pool[(i + offset) % n_fill])
        if a == b:
            b = int(fill_pool[(i + offset + 1) % n_fill])
        if random.random() < 0.5:
            a, b = b, a
        base_specs[int(pid)] = {"even_color": int(a), "odd_color": int(b)}
    pattern_specs = [dict(base_specs[int(slot["pair_id"])]) for slot in layout]
    for spec in pattern_specs:
        if random.random() < COLOR_MISMATCH_RATE:
            a, b = random.sample(fill_pool, 2)
            if random.random() < 0.5:
                a, b = b, a
            spec["even_color"] = int(a)
            spec["odd_color"] = int(b)
    return pattern_specs


def _build_object_pair(
    layout: list[dict], pattern_specs: list[dict], bg: int, outline_col: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n_cavities = len(layout)
    dst_shapes = [slot["shape_dst"] for slot in layout]
    src_shapes = [slot["shape_src"] for slot in layout]

    dst_cropped, dst_filled = _render_object(layout, dst_shapes, pattern_specs, bg, outline_col, filled=False)
    if dst_cropped is None or dst_filled is None:
        return None
    if not _is_8connected(dst_cropped, bg):
        return None
    if _count_enclosed_bg_holes(dst_cropped, bg) != n_cavities:
        return None
    if not _size_in_spec(dst_cropped):
        return None

    src_cropped, _ = _render_object(layout, src_shapes, pattern_specs, bg, outline_col, filled=True)
    if src_cropped is None:
        return None
    if not _is_8connected(src_cropped, bg):
        return None
    if _count_4conn_cavities(src_cropped, bg, outline_col) != n_cavities:
        return None
    if not _size_in_spec(src_cropped):
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

    layout = _design_layout(random.randint(*CAVITY_COUNT_RANGE))
    if len(layout) < CAVITY_COUNT_RANGE[0]:
        return None
    n_cavities = len(layout)

    fc_lo, fc_hi = FILL_COLOR_COUNT_RANGE
    max_cols = min(fc_hi, len(palette) - 1, max(4, n_cavities))
    if max_cols < fc_lo:
        return None
    n_fill_cols = random.randint(fc_lo, max_cols)
    fill_pool = palette[1 : 1 + n_fill_cols]
    pattern_specs = _assign_pair_patterns(layout, fill_pool)

    built = _build_object_pair(layout, pattern_specs, bg, outline_col)
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
