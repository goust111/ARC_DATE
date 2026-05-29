# Combo variant of 3a25b0d8 (index 18)
# combo_spec: D8/0+D9/2
# Name: 上下对称+颜色按行分配

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


# Design analysis for the combo variant
# a) Concrete behavior
#    Original 3a25b0d8 uses two similar outline objects. Src contains one color per cavity, dst
#    contains empty holes, and the solver copies colors cavity-by-cavity after matching centroid
#    rank keys. The original silhouettes are LR-symmetric.
#    This combo changes two linked layers at once:
#    - D8/0: the outline/hole scaffold becomes UD-symmetric instead of LR-symmetric.
#    - D9/2: colors are no longer per cavity; they are assigned per row-group. All cavities in one
#      row share a color, and the mirrored top/bottom rows share that same row-group color.
#    To force real interaction between the dimensions, src shows only one visible colored clue per
#    mirrored row-group. Example with actual row counts [2,1,2,1,2]:
#    - Original: five cavities would each be colored in src and copied one-by-one to dst.
#    - Combo: src still has the five-row UD-symmetric hole layout, but only three clue cavities are
#      visible: one for rows 0/4, one for rows 1/3, and one for center row 2. Output fills rows
#      0 and 4 with one color, rows 1 and 3 with another, and row 2 with a third.
#    Difference: the solver must recover full row structure from src holes plus sparse clues, then
#    lift colors from row-level clues to mirrored row-groups.
# b) Complexity calibration
#    A weak version would accidentally stay close to the original if every row were singleton or if
#    src exposed a clue on every row. To avoid that, the generator uses only 5-row palindromic row
#    plans whose total cavity count stays in the original 7-9 range and that contain repeated
#    multi-cavity evidence. There are always exactly 3 row-groups / 3 colors, so the rule is
#    visible but not noisy.
# c) Broken original assumptions
#    - Original `_colored_components()` assumed colored pieces enumerate all cavities. Now they
#      enumerate only row-group clues.
#    - Original transform matching compared cavity-level rank keys. Here within-row column layouts
#      may differ between src and dst, so only the row-count signature is reliable.
#    - Original final zip assumed one color per cavity. That is false: color now belongs to
#      `min(row_rank, mirrored_row_rank)`.
#    The solver therefore reconstructs all src cavity rows from `colored + empty` cavities, ranks
#    rows, folds them into mirrored row-groups, and only then paints dst.
# d) Placement legality
#    Src and dst are still placed independently by `random_free_location(..., padding=2)`, so they
#    never overlap or leave the canvas. The object generator validates 8-connectivity and size
#    before placement, then rechecks with the solver oracle.
# e) Visual observability
#    Humans can see three coupled cues in 2-4 examples:
#    - dst outlines are vertically mirrored top/bottom rather than left/right,
#    - same-row cavities end up same-colored in the output,
#    - top/bottom mirrored rows also share that color even when src only shows one clue for the
#      pair. No invisible property such as numeric color ordering is used.
# f) Visual cleanliness
#    No extra markers are added. The scene still contains two outline objects. Sparse clue fills in
#    src keep the picture legible, and the generator rejects accidental LR-symmetric silhouettes so
#    the D8 change stays visible instead of collapsing into dual-axis symmetry.
# g) Color inferability
#    The palette is capped at exactly 3 fill colors, one per mirrored row-group. Each sample
#    repeats those colors across several holes and across a mirrored row pair, which provides
#    enough structural support to infer the mapping from a few demonstrations.
# 9) New solver assumptions and how generation guarantees them
#    - The true row structure is a 5-row palindromic signature. `_design_layout()` always emits
#      that pattern.
#    - Every mirrored row-group has at least one visible src clue. `_make_src_cue_mask()` picks one
#      clue per group.
#    - Mirrored rows share one color. `_assign_group_colors()` colors by row_group id.
#    - Exact column ranks are not needed. Src/dst use independent asymmetric row offsets, but both
#      preserve the same row counts, which is the signal `main()` matches on.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill dst holes using colors inferred per mirrored row-group, then tight-crop.

    Combo rule:
    - Outline/hole scaffolds are UD-symmetric.
    - Cavities are matched at row level, not cavity level.
    - Same-row cavities share a color, and mirrored rows share that same color.
    - Src exposes only one visible clue cavity per mirrored row-group.
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
            if bool(r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
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
        mask = (sub != int(bg)) & (sub != int(outc))
        lab, n = label_mask(mask, connectivity=4)
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
        mask = sub != int(bg)
        pts = np.argwhere(mask)
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
    dest_row_sig, _dest_keys, d_rr, d_cc = _row_signature(dest_pts)
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

    n_rows = len(dest_row_sig)
    dest_rows: dict[int, list[dict]] = {}
    dest_groups: dict[int, list[dict]] = {}
    dest_list = []
    for h in dest_holes:
        row_key = int(d_rr[int(h["cent"][0])])
        col_key = int(d_cc[int(h["cent"][1])])
        group_key = min(row_key, int(n_rows - 1 - row_key))
        dest_rows.setdefault(row_key, []).append(h)
        dest_groups.setdefault(group_key, []).append(h)
        dest_list.append(((row_key, col_key), h["cent"], h))
    dest_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    t_all_pts = [
        _transform_point(p, int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
        for p in src_all_pts
    ]
    _row_sig, _keys, s_rr_all, _s_cc_all = _row_signature(t_all_pts)

    row_to_colors: dict[int, list[int]] = {}
    group_to_colors: dict[int, list[int]] = {}
    src_list = []
    for s in src_subs:
        tp = _transform_point(s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
        row_key = int(s_rr_all[int(tp[0])])
        group_key = min(row_key, int(n_rows - 1 - row_key))
        color = int(s["color"])
        row_to_colors.setdefault(row_key, []).append(color)
        group_to_colors.setdefault(group_key, []).append(color)
        src_list.append(((row_key, 0), tp, color))
    src_list.sort(key=lambda t: (t[0][0], t[1][0], t[1][1]))

    group_color_map = {
        group_key: int(Counter(cols).most_common(1)[0][0])
        for group_key, cols in group_to_colors.items()
    }
    row_color_map = {
        row_key: int(Counter(cols).most_common(1)[0][0]) for row_key, cols in row_to_colors.items()
    }

    groupwise_ok = (
        set(group_color_map) == set(dest_groups)
        and all(len(set(cols)) == 1 for cols in group_to_colors.values())
    )
    rowwise_ok = (
        set(row_color_map) == set(dest_rows)
        and all(len(set(cols)) == 1 for cols in row_to_colors.values())
    )

    if groupwise_ok:
        for group_key, holes in dest_groups.items():
            fill_color = int(group_color_map[group_key])
            for h in holes:
                for rr, cc in h["pts"]:
                    dest[int(rr), int(cc)] = fill_color
    elif rowwise_ok:
        for row_key, holes in dest_rows.items():
            fill_color = int(row_color_map[row_key])
            for h in holes:
                for rr, cc in h["pts"]:
                    dest[int(rr), int(cc)] = fill_color
    else:
        for (_k1, _p1, col), (_k2, _p2, h) in zip(src_list, dest_list):
            for rr, cc in h["pts"]:
                dest[int(rr), int(cc)] = int(col)

    return _tight_crop_non_bg(dest)


CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)

CAVITY_COUNT_RANGE = (7, 9)
FILL_COLOR_COUNT_RANGE = (3, 3)

# Each group plan [a, b, c] expands to the 5-row palindromic actual plan [a, b, c, b, a].
GROUP_PLAN_CHOICES = [
    [2, 1, 2],
    [2, 2, 1],
    [1, 2, 2],
    [2, 1, 3],
    [1, 2, 3],
]

ROW_HEIGHT_CHOICES_PAIR = [1, 1, 2, 2]
ROW_HEIGHT_CHOICES_CENTER = [1, 2, 2, 3]
ROW_GAP_CHOICES = [1, 1, 2]

SINGLE_WIDTH_CHOICES = [1, 2, 3]
PAIR_WIDTH_CHOICES = [2, 2, 3, 3, 4]
TRIO_WIDTH_CHOICES = [1, 1, 2, 2]
ROW_LEFT_PAD_CHOICES = [0, 0, 1, 2]
ROW_GAP_X_CHOICES = [1, 1, 2]

NONRECT_PROBABILITY = 0.8
MAX_MUTATE_ATTEMPTS = 40

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2

MAX_GEN_ATTEMPTS = 500
MAX_LAYOUT_R_BUDGET = 24


def _random_rect_shape(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _mutate_shape(
    mask: np.ndarray, *, n_remove_max: int, axis: str | None = None
) -> np.ndarray:
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
        if axis == "lr":
            mc = w - 1 - c
            if (r, mc) != (r, c):
                to_remove.append((r, mc))
        elif axis == "ud":
            mr = h - 1 - r
            if (mr, c) != (r, c):
                to_remove.append((mr, c))
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


def _gen_shape(h: int, w: int, *, nonrect: bool, axis: str | None = None) -> np.ndarray:
    m = _random_rect_shape(h, w)
    if nonrect and h * w >= 3:
        limit = max(1, (h * w) // 2)
        m = _mutate_shape(m, n_remove_max=limit, axis=axis)
    return m


def _mirror_ud(m: np.ndarray) -> np.ndarray:
    return m[::-1, :].copy()


def _pick_widths(k: int) -> list[int]:
    if k == 1:
        return [random.choice(SINGLE_WIDTH_CHOICES)]
    if k == 2:
        return [random.choice(PAIR_WIDTH_CHOICES), random.choice(PAIR_WIDTH_CHOICES)]
    return [random.choice(TRIO_WIDTH_CHOICES) for _ in range(3)]


def _sample_row_offsets(widths: list[int]) -> list[int]:
    offsets = []
    x = random.choice(ROW_LEFT_PAD_CHOICES)
    for idx, w in enumerate(widths):
        offsets.append(int(x))
        x += int(w)
        if idx != len(widths) - 1:
            x += random.choice(ROW_GAP_X_CHOICES)
    return offsets


def _expand_group_plan(group_plan: list[int]) -> list[int]:
    return list(group_plan) + list(group_plan[-2::-1])


def _add_slots_for_pair(
    slots: list[dict],
    *,
    row_id_top: int,
    row_id_bottom: int,
    row_group: int,
    r0_top: int,
    r0_bottom: int,
    row_h: int,
    k: int,
) -> None:
    widths_dst = _pick_widths(k)
    widths_src = _pick_widths(k)
    offs_dst = _sample_row_offsets(widths_dst)
    offs_src = _sample_row_offsets(widths_src)
    for idx in range(k):
        nonrect_dst = random.random() < NONRECT_PROBABILITY
        nonrect_src = random.random() < NONRECT_PROBABILITY
        dst_top = _gen_shape(row_h, widths_dst[idx], nonrect=nonrect_dst, axis=None)
        src_top = _gen_shape(row_h, widths_src[idx], nonrect=nonrect_src, axis=None)
        slots.append(
            {
                "r0": r0_top,
                "row_id": row_id_top,
                "row_group": row_group,
                "col_offset_dst": offs_dst[idx],
                "col_offset_src": offs_src[idx],
                "shape_dst": dst_top,
                "shape_src": src_top,
            }
        )
        slots.append(
            {
                "r0": r0_bottom,
                "row_id": row_id_bottom,
                "row_group": row_group,
                "col_offset_dst": offs_dst[idx],
                "col_offset_src": offs_src[idx],
                "shape_dst": _mirror_ud(dst_top),
                "shape_src": _mirror_ud(src_top),
            }
        )


def _add_slots_for_center(
    slots: list[dict], *, row_id: int, row_group: int, r0: int, row_h: int, k: int
) -> None:
    widths_dst = _pick_widths(k)
    widths_src = _pick_widths(k)
    offs_dst = _sample_row_offsets(widths_dst)
    offs_src = _sample_row_offsets(widths_src)
    for idx in range(k):
        nonrect_dst = random.random() < NONRECT_PROBABILITY
        nonrect_src = random.random() < NONRECT_PROBABILITY
        slots.append(
            {
                "r0": r0,
                "row_id": row_id,
                "row_group": row_group,
                "col_offset_dst": offs_dst[idx],
                "col_offset_src": offs_src[idx],
                "shape_dst": _gen_shape(row_h, widths_dst[idx], nonrect=nonrect_dst, axis="ud"),
                "shape_src": _gen_shape(row_h, widths_src[idx], nonrect=nonrect_src, axis="ud"),
            }
        )


def _design_layout() -> list[dict] | None:
    for _ in range(300):
        group_plan = list(random.choice(GROUP_PLAN_CHOICES))
        plan = _expand_group_plan(group_plan)
        if not (CAVITY_COUNT_RANGE[0] <= sum(plan) <= CAVITY_COUNT_RANGE[1]):
            continue

        h0 = random.choice(ROW_HEIGHT_CHOICES_PAIR)
        h1 = random.choice(ROW_HEIGHT_CHOICES_PAIR)
        h2 = random.choice(ROW_HEIGHT_CHOICES_CENTER)
        g0 = random.choice(ROW_GAP_CHOICES)
        g1 = random.choice(ROW_GAP_CHOICES)
        heights = [h0, h1, h2, h1, h0]
        starts = [0] * 5
        starts[1] = heights[0] + g0
        starts[2] = starts[1] + heights[1] + g1
        starts[3] = starts[2] + heights[2] + g1
        starts[4] = starts[3] + heights[3] + g0
        total_h = starts[4] + heights[4]
        if total_h > MAX_LAYOUT_R_BUDGET:
            continue

        slots: list[dict] = []
        _add_slots_for_pair(
            slots,
            row_id_top=0,
            row_id_bottom=4,
            row_group=0,
            r0_top=starts[0],
            r0_bottom=starts[4],
            row_h=heights[0],
            k=plan[0],
        )
        _add_slots_for_pair(
            slots,
            row_id_top=1,
            row_id_bottom=3,
            row_group=1,
            r0_top=starts[1],
            r0_bottom=starts[3],
            row_h=heights[1],
            k=plan[1],
        )
        _add_slots_for_center(
            slots, row_id=2, row_group=2, r0=starts[2], row_h=heights[2], k=plan[2]
        )
        return slots
    return None


def _render_object(
    layout: list[dict],
    colors: list[int],
    bg: int,
    outline_col: int,
    *,
    visible_mask: list[bool],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    placed_slots = []
    for slot_idx, slot in enumerate(layout):
        rr, cc = np.where(slot["_shape"])
        if len(rr) == 0:
            continue
        placed_slots.append((slot_idx, slot["r0"] + rr, int(slot["_offset"]) + cc))
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


def _render_dst(
    layout: list[dict], colors: list[int], bg: int, outline_col: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    local = []
    for slot in layout:
        local.append({**slot, "_shape": slot["shape_dst"], "_offset": slot["col_offset_dst"]})
    return _render_object(local, colors, bg, outline_col, visible_mask=[False] * len(local))


def _render_src(
    layout: list[dict], colors: list[int], bg: int, outline_col: int, visible_mask: list[bool]
) -> tuple[np.ndarray | None, np.ndarray | None]:
    local = []
    for slot in layout:
        local.append({**slot, "_shape": slot["shape_src"], "_offset": slot["col_offset_src"]})
    return _render_object(local, colors, bg, outline_col, visible_mask=visible_mask)


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
        if pts.size == 0:
            continue
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


def _colored_centroids_and_colors(sub: np.ndarray, bg: int) -> list[tuple[tuple[int, int], int]]:
    cnt = Counter(map(int, sub.flatten()))
    opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
    outc = int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)
    mask = (sub != int(bg)) & (sub != int(outc))
    lab, n = label_mask(mask, connectivity=4)
    out = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        col = Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0]
        cent = (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))
        out.append((cent, int(col)))
    return out


def _row_signature_from_pts(pts: list[tuple[int, int]]) -> tuple[int, ...]:
    rows = sorted(set(int(r) for r, _c in pts))
    rrank = {r: i for i, r in enumerate(rows)}
    counts = [0] * len(rrank)
    for r, _c in pts:
        counts[int(rrank[int(r)])] += 1
    return tuple(counts)


def _is_ud_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, mask[::-1, :]))


def _is_lr_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, mask[:, ::-1]))


def _variant_signal_active(src: np.ndarray, dst: np.ndarray, expected: np.ndarray, bg: int) -> bool:
    dst_mask = dst != int(bg)
    if not _is_ud_symmetric(dst_mask):
        return False
    if _is_lr_symmetric(dst_mask):
        return False

    dst_pts = _hole_centroids(dst, bg)
    src_clues = _colored_centroids_and_colors(src, bg)
    if len(dst_pts) <= len(src_clues):
        return False
    row_sig = _row_signature_from_pts(dst_pts)
    if len(row_sig) != 5 or tuple(row_sig) != tuple(row_sig[::-1]):
        return False
    if len(src_clues) != 3:
        return False

    exp_colored = _colored_centroids_and_colors(expected, bg)
    if len(exp_colored) != len(dst_pts):
        return False
    return True


def _make_src_cue_mask(layout: list[dict]) -> list[bool]:
    cue_mask = [False] * len(layout)
    by_group_rows: dict[int, dict[int, list[int]]] = {}
    for idx, slot in enumerate(layout):
        by_group_rows.setdefault(int(slot["row_group"]), {}).setdefault(int(slot["row_id"]), []).append(idx)
    for rows in by_group_rows.values():
        chosen_row = random.choice(sorted(rows))
        cue_mask[random.choice(rows[chosen_row])] = True
    return cue_mask


def _assign_group_colors(layout: list[dict], fill_pool: list[int]) -> list[int]:
    row_groups = sorted({int(slot["row_group"]) for slot in layout})
    chosen = random.sample(fill_pool, len(row_groups))
    random.shuffle(chosen)
    group_to_color = dict(zip(row_groups, chosen))
    return [int(group_to_color[int(slot["row_group"])]) for slot in layout]


def _pick_bg_and_palette() -> tuple[int, list[int]]:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    return bg, palette


def _build_object_pair(
    layout: list[dict], colors: list[int], bg: int, outline_col: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n_cavities = len(layout)
    n_groups = len({int(slot["row_group"]) for slot in layout})
    cue_mask = _make_src_cue_mask(layout)

    dst_cropped, dst_filled = _render_dst(layout, colors, bg, outline_col)
    if dst_cropped is None or dst_filled is None:
        return None
    if not _is_8connected(dst_cropped, bg):
        return None
    if _count_enclosed_bg_holes(dst_cropped, bg) != n_cavities:
        return None
    if not _size_in_spec(dst_cropped):
        return None

    src_cropped, _src_filled = _render_src(layout, colors, bg, outline_col, cue_mask)
    if src_cropped is None:
        return None
    if not _is_8connected(src_cropped, bg):
        return None
    if _count_4conn_cavities(src_cropped, bg, outline_col) != n_groups:
        return None
    if not _size_in_spec(src_cropped):
        return None
    if not _variant_signal_active(src_cropped, dst_cropped, dst_filled, bg):
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
    n_fill_cols = FILL_COLOR_COUNT_RANGE[0]
    if len(palette) - 1 < n_fill_cols:
        return None
    fill_pool = palette[1 : 1 + n_fill_cols]
    colors = _assign_group_colors(layout, fill_pool)

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
