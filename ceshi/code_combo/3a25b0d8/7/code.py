from __future__ import annotations

# Combo variant of 3a25b0d8 (index 7)
# combo_spec: D1/0+D4/0
# Name: 颜色配对+颜色映射变换
#
# Design analysis
# ===============
# a) Concrete behavior and trace
#    - Original task: solver finds two large outline objects, matches src cavity-fills to
#      dst holes by centroid row/col rank, and copies the src fill color directly.
#    - This combo changes two separate stages in the causal chain:
#      1. Matching stage: src cavities and dst holes each carry a one-pixel marker on the
#         outline. Equal marker colors define the pairing, so geometry no longer decides
#         correspondence.
#      2. Painting stage: after a src cavity is matched, its majority fill color is not
#         copied directly. A third legend object shows a color map, and the solver must
#         transform the fill color through that map before painting the matched dst hole.
#    - Tiny trace:
#         src cavity A = fill red, marker yellow
#         src cavity B = fill blue, marker green
#         dst hole left has marker green
#         dst hole right has marker yellow
#         legend says red -> orange, blue -> grey
#      Original output would be [left=red, right=blue] by position.
#      D1-only output would be [left=blue, right=red] by marker.
#      This combo output is [left=grey, right=orange]: first pair by marker, then recolor.
# b) Complexity calibration
#    - ARC has only 10 total colors, so the generator fixes the puzzle at 4 cavities:
#      4 unique marker colors + 2 repeated src fill colors + 2 mapped legend colors
#      + 1 outline color + 1 bg color = exactly 10 colors.
#    - Using exactly 2 fill colors keeps the legend small and readable while still forcing
#      repeated evidence: each legend row is used by multiple cavities.
#    - Marker permutation is forced to be a non-identity derangement, and the resulting
#      destination color pattern must differ from arrangement-based matching in at least
#      two slots. That prevents the combo from collapsing into either the base task or a
#      purely decorative marker system.
# c) Hidden assumptions from the original solver that no longer hold
#    - Original geometric ranking / rotation search is invalid: matching is now by marker
#      attribute, so all row-rank logic must be removed.
#    - Original "colored component is monochrome" assumption is invalid on src because each
#      cavity component now contains majority fill color plus one minority marker pixel.
#    - Original "there are only two components" assumption is invalid because the legend is
#      a third object. The solver must explicitly detect and exclude the legend.
#    - Original "fill chosen color equals src color" assumption is invalid because legend
#      recoloring is mandatory after marker matching.
# d) Placement legality
#    - Generator places src, dst, and legend independently with `random_free_location`
#      using the same border/padding guarantees as the base task. This keeps all objects
#      in-bounds, non-overlapping, and 8-disconnected.
# e) Visual observability
#    - The marker colors each appear exactly twice across the input, once on src and once
#      on dst, which makes the pairing cue visually local and explicit.
#    - The legend is a separate 5-column card: outline | src_color | outline | dst_color |
#      outline. Humans can read it as a literal color key from a few examples.
#    - Because src fill colors repeat and marker pairing is non-identity, the viewer cannot
#      solve by position alone or by copying visible src colors directly.
# f) Visual cleanliness
#    - Markers are recolored outline pixels adjacent to exactly one cavity/hole, so they
#      read as tags rather than floating clutter and do not bridge neighboring regions.
#    - Legend is compact and structurally distinct from the framed cavity objects, so it is
#      unlikely to be confused with a third src/dst-like object.
# g) Inferability of the color change
#    - The legend covers every src fill color exactly once. Mapped colors are chosen from a
#      separate two-color pool, so every used fill color visibly changes in the output.
#    - Since the same mapped result appears on multiple holes, the transform is supported by
#      repetition rather than a one-off accidental recolor.
# h) Quantitative anti-degeneration metric used after implementation
#    - After coding, measure two Hamming-style differences on generated samples:
#      1. combo output vs arrangement+legend baseline
#      2. combo output vs marker-only baseline
#      The generator is accepted only if both differences are non-zero on essentially all
#      samples, and the first one is typically >= 2 filled holes.
# i) New solver assumptions and generator guarantees
#    - Legend follows a rigid 5-column schema with no bg pixels inside; generator guarantees.
#    - Each src cavity has exactly one unique marker color and one majority fill color;
#      generator guarantees via tagged rendering.
#    - Each dst hole has exactly one adjacent marker pixel owned by that hole; generator
#      guarantees by choosing outline cells with a unique neighboring cavity owner.
#    - Every src fill color appears in the legend; generator guarantees full coverage.

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair src cavities to dst holes by marker color, then recolor via a legend before filling.
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

    def _outline_col(sub: np.ndarray) -> int:
        cnt = Counter(map(int, sub.flatten()))
        opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
        return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)

    def _parse_legend(sub: np.ndarray) -> dict[int, int] | None:
        outline_col = _outline_col(sub)
        H, W = sub.shape
        if int(W) != 5 or int(H) < 4:
            return None
        if np.any(sub == int(bg)):
            return None
        if not np.all(sub[0, :] == int(outline_col)) or not np.all(sub[-1, :] == int(outline_col)):
            return None
        if not np.all(sub[:, 0] == int(outline_col)) or not np.all(sub[:, -1] == int(outline_col)):
            return None
        if not np.all(sub[:, 2] == int(outline_col)):
            return None
        mapping: dict[int, int] = {}
        seen_dst: set[int] = set()
        for r in range(1, int(H - 1)):
            src_col = int(sub[int(r), 1])
            dst_col = int(sub[int(r), 3])
            if src_col in (int(bg), int(outline_col)) or dst_col in (int(bg), int(outline_col)):
                return None
            if int(src_col) in mapping or int(dst_col) in seen_dst:
                return None
            mapping[int(src_col)] = int(dst_col)
            seen_dst.add(int(dst_col))
        return mapping if mapping else None

    def _source_components(sub: np.ndarray, outline_col: int) -> list[dict]:
        mask = (sub != int(bg)) & (sub != int(outline_col))
        lab, n = label_mask(mask, connectivity=4)
        out = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            vals = [int(v) for v in sub[lab == int(cid)].flatten()]
            cnt = Counter(vals)
            fill_col, fill_n = cnt.most_common(1)[0]
            marker_opts = [(col, num) for col, num in cnt.items() if int(col) != int(fill_col)]
            if not marker_opts:
                continue
            marker_col, marker_n = min(marker_opts, key=lambda t: (t[1], t[0]))
            if int(fill_n) <= int(marker_n):
                continue
            out.append({"fill": int(fill_col), "marker": int(marker_col)})
        return out

    def _hole_marker(sub: np.ndarray, hole: dict, outline_col: int) -> int | None:
        Hs, Ws = sub.shape
        marker_cols = []
        for rr, cc in hole["pts"]:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = int(rr + dr), int(cc + dc)
                if not (0 <= nr < Hs and 0 <= nc < Ws):
                    continue
                val = int(sub[int(nr), int(nc)])
                if val != int(bg) and val != int(outline_col):
                    marker_cols.append(int(val))
        if not marker_cols:
            return None
        return int(Counter(marker_cols).most_common(1)[0][0])

    comps = _components(g != int(bg))
    infos = []
    for comp in comps:
        sub = _extract_group_subgrid([comp])
        infos.append(
            {
                "sub": sub,
                "size": int(comp["size"]),
                "holes": _enclosed_bg_components(sub),
                "legend_map": _parse_legend(sub),
            }
        )

    legend_candidates = [i for i, info in enumerate(infos) if info["legend_map"]]
    legend_idx = None
    if legend_candidates:
        legend_idx = min(legend_candidates, key=lambda i: int(infos[i]["size"]))
    color_map = infos[legend_idx]["legend_map"] if legend_idx is not None else {}

    object_infos = [info for i, info in enumerate(infos) if i != legend_idx]
    object_infos.sort(key=lambda d: int(d["size"]), reverse=True)
    assert len(object_infos) >= 2

    objs = [object_infos[0]["sub"], object_infos[1]["sub"]]
    holes_list = [object_infos[0]["holes"], object_infos[1]["holes"]]
    hole_area = [sum(int(h["size"]) for h in hh) for hh in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    dest = objs[dest_i].copy()
    src = objs[src_i].copy()

    outline_col = _outline_col(dest)
    dest_holes = _enclosed_bg_components(dest)
    src_subs = _source_components(src, outline_col)
    if not dest_holes or not src_subs:
        pts = np.argwhere(dest != int(bg))
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        return dest[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

    src_by_marker = {int(s["marker"]): int(s["fill"]) for s in src_subs}
    for hole in dest_holes:
        marker = _hole_marker(dest, hole, outline_col)
        if marker is None or int(marker) not in src_by_marker:
            continue
        src_fill = int(src_by_marker[int(marker)])
        out_col = int(color_map.get(int(src_fill), int(src_fill)))
        for rr, cc in hole["pts"]:
            dest[int(rr), int(cc)] = int(out_col)

    pts = np.argwhere(dest != int(bg))
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return dest[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)

CAVITY_COUNT = 4
FILL_COLOR_COUNT = 2

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

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2

MAX_GEN_ATTEMPTS = 500
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


def _render_object(
    layout: list[dict],
    shapes: list[np.ndarray],
    fill_colors: list[int],
    marker_colors: list[int],
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

    outline_mask = _outline_from_fill(fill_mask)
    canvas = np.full((H, W), int(bg), dtype=int)
    canvas[outline_mask] = int(outline_col)

    filled_grid = canvas.copy()
    filled_grid[fill_mask] = fill_vals[fill_mask]
    if filled:
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
            return None, None
        used_marker_cells.add(marker_cell)
        rr, cc = marker_cell
        canvas[int(rr), int(cc)] = int(marker_colors[slot_idx])
        filled_grid[int(rr), int(cc)] = int(marker_colors[slot_idx])

    b = bbox(filled_grid != int(bg))
    if b is None:
        return None, None
    cropped = crop_bbox(canvas, b)
    filled_cropped = crop_bbox(filled_grid, b)
    return cropped, filled_cropped


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
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if int(r0) > 0 and int(c0) > 0 and int(r1) < int(H - 1) and int(c1) < int(W - 1):
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


def _assign_src_fill_colors(n_slots: int, fill_pool: list[int]) -> list[int]:
    colors = [int(fill_pool[i % len(fill_pool)]) for i in range(n_slots)]
    random.shuffle(colors)
    return colors


def _random_derangement(n: int) -> list[int]:
    base = list(range(n))
    for _ in range(200):
        perm = base.copy()
        random.shuffle(perm)
        if all(int(i) != int(p) for i, p in enumerate(perm)):
            return perm
    return base[1:] + base[:1]


def _build_legend(color_map: dict[int, int], outline_col: int) -> np.ndarray:
    rows = list(color_map.items())
    random.shuffle(rows)
    legend = np.full((len(rows) + 2, 5), int(outline_col), dtype=int)
    for i, (src_col, dst_col) in enumerate(rows, start=1):
        legend[int(i), 1] = int(src_col)
        legend[int(i), 3] = int(dst_col)
    return legend


def _build_object_pair(
    layout: list[dict],
    src_fill_colors: list[int],
    src_marker_colors: list[int],
    dest_marker_colors: list[int],
    expected_dest_fill_colors: list[int],
    bg: int,
    outline_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n_cavities = len(layout)
    dst_shapes = [slot["shape_dst"] for slot in layout]
    src_shapes = [slot["shape_src"] for slot in layout]

    dst_cropped, dst_filled = _render_object(
        layout,
        dst_shapes,
        expected_dest_fill_colors,
        dest_marker_colors,
        bg,
        outline_col,
        filled=False,
    )
    if dst_cropped is None or dst_filled is None:
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
        src_fill_colors,
        src_marker_colors,
        bg,
        outline_col,
        filled=True,
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


def _place_on_canvas(patches: list[np.ndarray], bg: int) -> np.ndarray | None:
    canvas_h = random.randint(*CANVAS_SIZE_RANGE)
    canvas_w = random.randint(*CANVAS_SIZE_RANGE)
    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    for patch in patches:
        try:
            r, c = random_free_location(
                canvas,
                patch,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
            )
        except ValueError:
            return None
        paste(canvas, patch, r, c, transparent=bg)
    return canvas


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline_col = int(palette[0])

    layout = _design_layout(CAVITY_COUNT)
    if len(layout) != CAVITY_COUNT:
        return None

    marker_colors = [int(c) for c in palette[1 : 1 + CAVITY_COUNT]]
    fill_pool = [int(c) for c in palette[1 + CAVITY_COUNT : 1 + CAVITY_COUNT + FILL_COLOR_COUNT]]
    mapped_pool = [int(c) for c in palette[1 + CAVITY_COUNT + FILL_COLOR_COUNT : 1 + CAVITY_COUNT + 2 * FILL_COLOR_COUNT]]
    if len(fill_pool) != FILL_COLOR_COUNT or len(mapped_pool) != FILL_COLOR_COUNT:
        return None

    src_fill_colors = _assign_src_fill_colors(CAVITY_COUNT, fill_pool)
    mapped_order = mapped_pool.copy()
    random.shuffle(mapped_order)
    color_map = {int(src): int(dst) for src, dst in zip(fill_pool, mapped_order)}

    perm = _random_derangement(CAVITY_COUNT)
    dest_marker_colors = [0] * CAVITY_COUNT
    dest_fill_before_map = [0] * CAVITY_COUNT
    expected_dest_fill_colors = [0] * CAVITY_COUNT
    for src_idx, dst_idx in enumerate(perm):
        dest_marker_colors[int(dst_idx)] = int(marker_colors[src_idx])
        src_fill = int(src_fill_colors[src_idx])
        dest_fill_before_map[int(dst_idx)] = int(src_fill)
        expected_dest_fill_colors[int(dst_idx)] = int(color_map[int(src_fill)])

    marker_mismatch = sum(
        int(dest_fill_before_map[i]) != int(src_fill_colors[i]) for i in range(CAVITY_COUNT)
    )
    if marker_mismatch < 2:
        return None

    if any(int(color_map[int(col)]) == int(col) for col in fill_pool):
        return None

    built = _build_object_pair(
        layout,
        src_fill_colors,
        marker_colors,
        dest_marker_colors,
        expected_dest_fill_colors,
        bg,
        outline_col,
    )
    if built is None:
        return None
    src, dst, expected = built
    legend = _build_legend(color_map, outline_col)

    canvas = _place_on_canvas([src, dst, legend], bg)
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
