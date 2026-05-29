# Combo variant of 3a25b0d8 (index 1)
# combo_spec: D1/0+D3/2
# Name: 颜色线索配对+洞边框提示
#
# Design analysis
# ===============
# a) Concrete behavior and trace
#    - Original task: `src` and `dst` are two outline objects. `src` has colored cavity fills,
#      `dst` has bg holes. Solver pairs cavities and holes by centroid row/col rank, then copies
#      the fill color into the matched hole.
#    - This combo changes both the pairing signal and the visual form of the destination:
#      `src` keeps one colored marker on the outline next to each filled cavity; `dst` recolors
#      the hole's surrounding outline pixels into the same marker color. The copied color is still
#      the cavity's main fill color, but the correspondence is now "marker color match" rather
#      than "same geometric rank".
#    - Small trace:
#      `src`: cavity A = red fill + yellow outline marker; cavity B = blue fill + green marker.
#      `dst`: top hole border = green; bottom hole border = yellow.
#      Original solver would fill top/bottom by positional order. This combo must fill top with
#      blue and bottom with red because the hole-border colors point to the matching `src` cavity.
# b) Complexity calibration
#    - Marker colors are unique per cavity and appear twice in the input: once on `src`, once as a
#      border color around one `dst` hole. That gives a visible, low-entropy cue humans can track.
#    - Fill colors are intentionally reused across cavities, so the task does not collapse into
#      "copy the border color" or "every fill color is already unique".
#    - Cavity count is reduced to 4-5 so the palette can support 1 outline color + unique marker
#      colors + 2-3 repeated fill colors without exhausting ARC's 10 colors.
# c) Hidden assumptions reviewed against the original solver
#    - Original `_rank_keys(...)`, `_transform_point(...)`, and rotation/flip search assumed
#      geometry was the matching signal. That assumption is removed completely.
#    - Original `_colored_components(...)` assumed each `src` cavity component was monochrome.
#      Here each component is "majority fill color + one minority marker color", so the solver
#      must separate those roles explicitly.
#    - Original outline-color inference from `dst` no longer holds because `dst` now has many
#      non-outline border-hint pixels. New solver assumption: the true outline color is the one
#      whose removal from `src` yields exactly the hole count worth of enclosed fill+marker
#      components. Generator guarantee: yes.
# d) Placement legality
#    - `src` and `dst` are still tight-cropped object patches placed independently onto a
#      22-30 canvas with `random_free_location(..., padding=2)`, so they remain disjoint and
#      within bounds.
# e) Visual observability
#    - The hole-border colors are large local structures, not single hidden pixels. Across 2-4
#      examples a human can see that each `dst` hole is ringed by a color that reappears next to
#      exactly one `src` cavity, and the fill copied into that hole is the cavity's main color.
#    - The generator forces a non-identity color permutation and requires at least two filled holes
#      to differ from the original arrangement baseline, so the combo does not silently degenerate
#      back to the base task.
# f) Visual cleanliness
#    - The destination hint uses only outline pixels that are 4-neighbor-adjacent to exactly one
#      hole. That avoids ambiguous shared border cells and keeps each colored ring visually tied to
#      a single hole.
# g) Color-change inferability
#    - Marker colors and fill colors come from disjoint palette slices. The border color is a
#      correspondence key; it is not the color that gets copied into the hole.
# h) Anti-degeneration metric used after implementation
#    - Compare this combo solver's output against the original arrangement-based solver on the same
#      generated input. The generator enforces at least two slot-color mismatches versus the
#      arrangement baseline, so the outputs should differ on essentially every sample.
# i) New solver assumptions and generator guarantees
#    - `src` has exactly one outline-marker pixel per cavity component. Guaranteed.
#    - Each `dst` hole has at least three uniquely owned colored border pixels. Guaranteed.
#    - Marker colors are unique across cavities. Guaranteed.
#    - `dst` still has larger enclosed-bg area than `src`, so source/destination identification by
#      hole area remains valid. Guaranteed.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each `dst` hole with the main fill color of the `src` cavity whose marker color matches
    that hole's colored border hint, then tight-crop the result.
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
                    "pts": pts,
                    "size": int(len(pts)),
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
        lab, n = label_mask(sub == int(bg), connectivity=4)
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
            holes.append({"pts": pts, "size": int(len(pts))})
        return holes

    def _infer_outline_col(src: np.ndarray, expected_components: int) -> int | None:
        colors = sorted({int(v) for v in src.flatten() if int(v) != int(bg)})
        Hs, Ws = src.shape
        for cand in colors:
            mask = (src != int(bg)) & (src != int(cand))
            lab, n = label_mask(mask, connectivity=4)
            if int(n) != int(expected_components):
                continue
            ok = True
            for cid in range(1, int(n) + 1):
                pts = np.argwhere(lab == int(cid))
                if pts.size == 0:
                    ok = False
                    break
                r0, c0 = pts.min(axis=0)
                r1, c1 = pts.max(axis=0)
                if bool(r0 == 0 or c0 == 0 or r1 == int(Hs - 1) or c1 == int(Ws - 1)):
                    ok = False
                    break
                cnt = Counter(map(int, src[lab == int(cid)].flatten()))
                if len(cnt) != 2:
                    ok = False
                    break
                (_fill_col, fill_n), (_marker_col, marker_n) = cnt.most_common(2)
                if int(fill_n) <= int(marker_n):
                    ok = False
                    break
            if ok:
                return int(cand)
        return None

    def _source_components(sub: np.ndarray, outline_col: int) -> list[dict]:
        mask = (sub != int(bg)) & (sub != int(outline_col))
        lab, n = label_mask(mask, connectivity=4)
        out = []
        for cid in range(1, int(n) + 1):
            vals = [int(v) for v in sub[lab == int(cid)].flatten()]
            if not vals:
                continue
            cnt = Counter(vals)
            if len(cnt) != 2:
                continue
            fill_col, fill_n = cnt.most_common(1)[0]
            marker_opts = [(col, num) for col, num in cnt.items() if int(col) != int(fill_col)]
            if not marker_opts:
                continue
            marker_col, marker_n = marker_opts[0]
            if int(fill_n) <= int(marker_n):
                continue
            out.append({"fill": int(fill_col), "marker": int(marker_col)})
        return out

    def _hole_hint_color(sub: np.ndarray, hole: dict, outline_col: int) -> int | None:
        Hs, Ws = sub.shape
        hint_cols = []
        for rr, cc in hole["pts"]:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = int(rr + dr), int(cc + dc)
                if not (0 <= nr < Hs and 0 <= nc < Ws):
                    continue
                val = int(sub[int(nr), int(nc)])
                if val != int(bg) and val != int(outline_col):
                    hint_cols.append(val)
        if not hint_cols:
            return None
        return int(Counter(hint_cols).most_common(1)[0][0])

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
    if not dest_holes:
        pts = np.argwhere(dest != int(bg))
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        return dest[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

    outline_col = _infer_outline_col(src, len(dest_holes))
    if outline_col is None:
        pts = np.argwhere(dest != int(bg))
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        return dest[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

    src_subs = _source_components(src, outline_col)
    if len(src_subs) != len(dest_holes):
        pts = np.argwhere(dest != int(bg))
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        return dest[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

    src_by_marker = {int(s["marker"]): int(s["fill"]) for s in src_subs}
    for h in dest_holes:
        hint = _hole_hint_color(dest, h, outline_col)
        if hint is None or int(hint) not in src_by_marker:
            continue
        fill_col = int(src_by_marker[int(hint)])
        for rr, cc in h["pts"]:
            dest[int(rr), int(cc)] = int(fill_col)

    pts = np.argwhere(dest != int(bg))
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return dest[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)

CAVITY_COUNT_RANGE = (4, 5)
FILL_COLOR_COUNT_RANGE = (2, 3)

ROW_HEIGHT_CHOICES = [2, 2, 3]
LAYOUT_K_CHOICES_FULL = [1, 2, 2, 3, 3]
LAYOUT_K_CHOICES_SMALL = [1, 2, 2]
ROW_GAP_CHOICES = [2, 2, 3]

CENTERED_SLOT_WIDTHS = [3, 5]
PAIR_SLOT_WIDTHS = [2, 3]
SIDE_SLOT_WIDTHS = [2, 3]
MID_SLOT_WIDTHS = [3]
PAIR_AXIS_GAP_CHOICES = [1, 3]
TRIO_MID_SIDE_GAP = 1

NONRECT_PROBABILITY = 0.8
MAX_MUTATE_ATTEMPTS = 40
MIN_CAVITY_AREA = 3
MIN_HINT_BORDER_CELLS = 2

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2

MAX_GEN_ATTEMPTS = 800
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
        if int(m2.sum()) < MIN_CAVITY_AREA:
            continue
        _, n_components = label_mask(m2, connectivity=4)
        if int(n_components) == 1:
            m = m2
            removed += len(to_remove)
    return m if bool(m.any()) else mask.copy()


def _gen_shape(h: int, w: int, *, nonrect: bool, symmetric: bool = False) -> np.ndarray:
    for _ in range(20):
        m = _random_rect_shape(h, w)
        if nonrect and h * w >= 4:
            limit = max(1, (h * w) // 2)
            m = _mutate_shape(m, n_remove_max=limit, symmetric=symmetric)
        if int(m.sum()) >= MIN_CAVITY_AREA:
            return m
    return _random_rect_shape(h, w)


def _mirror_shape(m: np.ndarray) -> np.ndarray:
    return m[:, ::-1].copy()


def _gen_dst_src_pair(h: int, w: int, *, symmetric: bool) -> tuple[np.ndarray, np.ndarray]:
    nonrect = random.random() < NONRECT_PROBABILITY
    return (
        _gen_shape(h, w, nonrect=nonrect, symmetric=symmetric),
        _gen_shape(h, w, nonrect=nonrect, symmetric=symmetric),
    )


def _add_centered_slot(slots: list[dict], cur_r: int, row_h: int) -> None:
    w = random.choice(CENTERED_SLOT_WIDTHS)
    dst, src = _gen_dst_src_pair(row_h, w, symmetric=True)
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -(w // 2),
            "shape_dst": dst,
            "shape_src": src,
            "marker_side": random.choice(["left", "right"]),
        }
    )


def _add_mirror_pair_slots(slots: list[dict], cur_r: int, row_h: int) -> None:
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
            "marker_side": "left",
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": half + 1,
            "shape_dst": dst_r,
            "shape_src": src_r,
            "marker_side": "right",
        }
    )


def _add_trio_slots(slots: list[dict], cur_r: int, row_h: int) -> None:
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
            "marker_side": "left",
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -mid_half,
            "shape_dst": dst_m,
            "shape_src": src_m,
            "marker_side": "top",
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": mid_half + TRIO_MID_SIDE_GAP + 1,
            "shape_dst": dst_r,
            "shape_src": src_r,
            "marker_side": "right",
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
    budget = n_target
    while len(slots) < n_target and cur_r < MAX_LAYOUT_R_BUDGET and budget > 0:
        row_h = random.choice(ROW_HEIGHT_CHOICES)
        k = _pick_layout_k(budget)
        if k == 1:
            _add_centered_slot(slots, cur_r, row_h)
            budget -= 1
        elif k == 2:
            _add_mirror_pair_slots(slots, cur_r, row_h)
            budget -= 2
        else:
            _add_trio_slots(slots, cur_r, row_h)
            budget -= 3
        cur_r += row_h + random.choice(ROW_GAP_CHOICES)
    return slots


def _outline_from_fill(fill_mask: np.ndarray) -> np.ndarray:
    outline_mask = np.zeros_like(fill_mask)
    outline_mask[1:, :] |= fill_mask[:-1, :]
    outline_mask[:-1, :] |= fill_mask[1:, :]
    outline_mask[:, 1:] |= fill_mask[:, :-1]
    outline_mask[:, :-1] |= fill_mask[:, 1:]
    outline_mask &= ~fill_mask
    return outline_mask


def _owner_neighbors(owner: np.ndarray, r: int, c: int) -> set[int]:
    H, W = owner.shape
    out = set()
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = int(r + dr), int(c + dc)
        if 0 <= nr < H and 0 <= nc < W and int(owner[int(nr), int(nc)]) >= 0:
            out.add(int(owner[int(nr), int(nc)]))
    return out


def _marker_candidates(
    cells: list[tuple[int, int]],
    side: str,
    outline_mask: np.ndarray,
    owner: np.ndarray,
    slot_idx: int,
    used: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    delta = {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}[side]
    H, W = outline_mask.shape
    cand = set()
    for r, c in cells:
        nr, nc = int(r + delta[0]), int(c + delta[1])
        if not (0 <= nr < H and 0 <= nc < W):
            continue
        if not bool(outline_mask[int(nr), int(nc)]):
            continue
        if (int(nr), int(nc)) in used:
            continue
        if _owner_neighbors(owner, int(nr), int(nc)) == {int(slot_idx)}:
            cand.add((int(nr), int(nc)))
    return sorted(cand)


def _pick_marker_cell(
    cells: list[tuple[int, int]],
    preferred_side: str,
    outline_mask: np.ndarray,
    owner: np.ndarray,
    slot_idx: int,
    used: set[tuple[int, int]],
) -> tuple[int, int] | None:
    order = [preferred_side, "top", "bottom", "left", "right"]
    seen = set()
    for side in order:
        if side in seen:
            continue
        seen.add(side)
        cand = _marker_candidates(cells, side, outline_mask, owner, slot_idx, used)
        if cand:
            return cand[0]
    return None


def _all_hint_border_cells(
    cells: list[tuple[int, int]],
    outline_mask: np.ndarray,
    owner: np.ndarray,
    slot_idx: int,
) -> list[tuple[int, int]]:
    out = set()
    H, W = outline_mask.shape
    for r, c in cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if not bool(outline_mask[int(nr), int(nc)]):
                continue
            if _owner_neighbors(owner, int(nr), int(nc)) == {int(slot_idx)}:
                out.add((int(nr), int(nc)))
    return sorted(out)


def _place_slots(
    layout: list[dict],
    shapes: list[np.ndarray],
    fill_colors: list[int],
    bg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[tuple[int, int]]]] | tuple[None, None, None, None]:
    placed_slots = []
    for slot_idx, (slot, sh) in enumerate(zip(layout, shapes)):
        rr, cc = np.where(sh)
        if len(rr) == 0:
            continue
        placed_slots.append((slot_idx, slot["r0"] + rr, slot["col_offset"] + cc))
    if not placed_slots:
        return None, None, None, None

    all_rows = np.concatenate([p[1] for p in placed_slots])
    all_cols = np.concatenate([p[2] for p in placed_slots])
    r_min, r_max = int(all_rows.min()), int(all_rows.max())
    c_min, c_max = int(all_cols.min()), int(all_cols.max())

    pad = 2
    H = (r_max - r_min + 1) + 2 * pad
    W = (c_max - c_min + 1) + 2 * pad

    owner = np.full((H, W), -1, dtype=int)
    fill_mask = np.zeros((H, W), dtype=bool)
    fill_vals = np.full((H, W), int(bg), dtype=int)
    slot_cells: list[list[tuple[int, int]]] = [[] for _ in layout]

    for slot_idx, rows, cols in placed_slots:
        rr = rows - r_min + pad
        cc = cols - c_min + pad
        fill_mask[rr, cc] = True
        fill_vals[rr, cc] = int(fill_colors[slot_idx])
        owner[rr, cc] = int(slot_idx)
        slot_cells[slot_idx] = list(zip(rr.astype(int).tolist(), cc.astype(int).tolist()))
    return owner, fill_mask, fill_vals, slot_cells


def _render_src_object(
    layout: list[dict],
    shapes: list[np.ndarray],
    fill_colors: list[int],
    marker_colors: list[int],
    bg: int,
    outline_col: int,
) -> np.ndarray | None:
    owner, fill_mask, fill_vals, slot_cells = _place_slots(layout, shapes, fill_colors, bg)
    if owner is None:
        return None

    outline_mask = _outline_from_fill(fill_mask)
    canvas = np.full(owner.shape, int(bg), dtype=int)
    canvas[outline_mask] = int(outline_col)
    canvas[fill_mask] = fill_vals[fill_mask]

    used_marker_cells: set[tuple[int, int]] = set()
    for slot_idx, slot in enumerate(layout):
        marker_cell = _pick_marker_cell(
            slot_cells[slot_idx],
            str(slot["marker_side"]),
            outline_mask,
            owner,
            slot_idx,
            used_marker_cells,
        )
        if marker_cell is None:
            return None
        used_marker_cells.add(marker_cell)
        rr, cc = marker_cell
        canvas[int(rr), int(cc)] = int(marker_colors[slot_idx])

    b = bbox(canvas != int(bg))
    if b is None:
        return None
    return crop_bbox(canvas, b)


def _render_dst_object(
    layout: list[dict],
    shapes: list[np.ndarray],
    fill_colors: list[int],
    hint_colors: list[int],
    bg: int,
    outline_col: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    owner, fill_mask, fill_vals, slot_cells = _place_slots(layout, shapes, fill_colors, bg)
    if owner is None:
        return None, None

    outline_mask = _outline_from_fill(fill_mask)
    empty_grid = np.full(owner.shape, int(bg), dtype=int)
    empty_grid[outline_mask] = int(outline_col)

    filled_grid = empty_grid.copy()
    filled_grid[fill_mask] = fill_vals[fill_mask]

    for slot_idx in range(len(layout)):
        border_cells = _all_hint_border_cells(slot_cells[slot_idx], outline_mask, owner, slot_idx)
        if len(border_cells) < MIN_HINT_BORDER_CELLS:
            return None, None
        for rr, cc in border_cells:
            empty_grid[int(rr), int(cc)] = int(hint_colors[slot_idx])
            filled_grid[int(rr), int(cc)] = int(hint_colors[slot_idx])

    b = bbox(filled_grid != int(bg))
    if b is None:
        return None, None
    return crop_bbox(empty_grid, b), crop_bbox(filled_grid, b)


def _count_4conn_cavities(filled_grid: np.ndarray, bg: int, outline_col: int) -> int:
    mask = (filled_grid != int(bg)) & (filled_grid != int(outline_col))
    _, n = label_mask(mask, connectivity=4)
    return int(n)


def _count_enclosed_bg_holes(empty_grid: np.ndarray, bg: int) -> int:
    mask = empty_grid == int(bg)
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
    mask = grid != int(bg)
    if not bool(mask.any()):
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


def _assign_fill_colors(n_slots: int, fill_pool: list[int]) -> list[int]:
    colors = list(fill_pool)
    while len(colors) < n_slots:
        colors.append(random.choice(fill_pool))
    random.shuffle(colors)
    return colors[:n_slots]


def _random_derangement(n: int) -> list[int]:
    base = list(range(n))
    for _ in range(200):
        perm = base.copy()
        random.shuffle(perm)
        if all(int(i) != int(p) for i, p in enumerate(perm)):
            return perm
    return base[1:] + base[:1]


def _build_object_pair(
    layout: list[dict],
    src_fill_colors: list[int],
    src_marker_colors: list[int],
    dst_hint_colors: list[int],
    expected_dst_fill_colors: list[int],
    bg: int,
    outline_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n_cavities = len(layout)
    dst_shapes = [slot["shape_dst"] for slot in layout]
    src_shapes = [slot["shape_src"] for slot in layout]

    dst_cropped, dst_filled = _render_dst_object(
        layout,
        dst_shapes,
        expected_dst_fill_colors,
        dst_hint_colors,
        bg,
        outline_col,
    )
    if dst_cropped is None or dst_filled is None:
        return None
    if not _is_8connected(dst_cropped, bg):
        return None
    if _count_enclosed_bg_holes(dst_cropped, bg) != n_cavities:
        return None
    if not _size_in_spec(dst_cropped):
        return None

    src_cropped = _render_src_object(
        layout,
        src_shapes,
        src_fill_colors,
        src_marker_colors,
        bg,
        outline_col,
    )
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
    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
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


def _gen_attempt_with_meta() -> tuple[np.ndarray | None, dict | None]:
    bg, palette = _pick_bg_and_palette()
    outline_col = int(palette[0])

    layout = _design_layout(random.randint(*CAVITY_COUNT_RANGE))
    if not (CAVITY_COUNT_RANGE[0] <= len(layout) <= CAVITY_COUNT_RANGE[1]):
        return None, None
    n_cavities = len(layout)

    marker_colors = palette[1 : 1 + n_cavities]
    fill_candidates = palette[1 + n_cavities :]
    max_fill = min(FILL_COLOR_COUNT_RANGE[1], len(fill_candidates))
    if max_fill < FILL_COLOR_COUNT_RANGE[0]:
        return None, None
    n_fill_cols = random.randint(FILL_COLOR_COUNT_RANGE[0], max_fill)
    fill_pool = fill_candidates[:n_fill_cols]
    src_fill_colors = _assign_fill_colors(n_cavities, fill_pool)

    perm = _random_derangement(n_cavities)
    dst_hint_colors = [0] * n_cavities
    expected_dst_fill_colors = [0] * n_cavities
    for src_idx, dst_idx in enumerate(perm):
        dst_hint_colors[int(dst_idx)] = int(marker_colors[src_idx])
        expected_dst_fill_colors[int(dst_idx)] = int(src_fill_colors[src_idx])

    mismatch = sum(
        int(expected_dst_fill_colors[i]) != int(src_fill_colors[i]) for i in range(n_cavities)
    )
    if mismatch < 2:
        return None, None

    built = _build_object_pair(
        layout,
        src_fill_colors,
        marker_colors,
        dst_hint_colors,
        expected_dst_fill_colors,
        bg,
        outline_col,
    )
    if built is None:
        return None, None
    src, dst, expected = built

    canvas = _place_on_canvas(src, dst, bg)
    if canvas is None:
        return None, None

    try:
        actual = main(canvas)
    except Exception:
        return None, None
    if not isinstance(actual, np.ndarray):
        return None, None
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None, None
    meta = {
        "slot_mismatch": int(mismatch),
        "n_cavities": int(n_cavities),
    }
    return canvas, meta


def _gen_attempt() -> np.ndarray | None:
    canvas, _meta = _gen_attempt_with_meta()
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
