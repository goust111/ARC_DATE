# Combo variant of 3a25b0d8 (index 14)
# combo_spec: D4/1+D5/0
# Name: 擦除源+完整画布输出
#
# Design analysis for the fused variant:
# a) Concrete behavior:
#    Small example: input has a left src outline with two colored cavities
#    (red on the upper cavity, green on the lower cavity) and a right dst outline
#    with two empty holes at the same centroid rank positions. The original task
#    outputs only the filled dst crop. This fused variant instead keeps the whole
#    canvas, fills the two dst holes with red/green, and erases the two src colored
#    regions back to bg. The visible effect is "color transfer": src loses colors
#    while dst gains them, and the outline-only src stays in place.
# b) Complexity calibration:
#    The matching cue remains the original row/column centroid-rank structure, so
#    the task is not simplified into raw color copying. At the same time we do not
#    add extra transform degrees of freedom in the generator; keeping the layout
#    identity avoids making 2-4 examples too ambiguous for humans.
# c) Hidden assumptions audited:
#    - The original solver assumed output can be cropped to dst. That is removed.
#    - The original fallback returned a cropped dst; here every fallback returns the
#      full canvas because grid_shape changed to full-canvas output.
#    - Because full-canvas output must preserve both objects, local bbox edits cannot
#      be pasted opaquely: overlapping bboxes would destroy unrelated pixels. The
#      solver therefore writes dst fills and src erasures in global coordinates.
#    - Role detection still assumes dst has larger enclosed-bg hole area than src,
#      because src cavities are color-filled while dst cavities are empty. The
#      generator continues to guarantee exactly two such objects.
# d) Placement legality:
#    The generator now places the two tight object patches by bbox, with a mandatory
#    empty gap and strict in-bounds checks. This is stronger than the original pixel-
#    only non-collision rule and keeps the full-canvas scenes visually separated.
# e) Visual observability:
#    The full canvas intentionally shows both halves of the transfer in one output:
#    dst gains colors, src loses them. Humans can infer the rule from repeated
#    structure without relying on hidden ordering by numeric color ids.
# f) Visual cleanliness:
#    Bbox separation prevents crowded overlap when both objects remain visible in the
#    output. The src outline remains readable after erasure because only interior
#    colored pixels are removed.
# g) Change inferability:
#    Colors are not remapped by arbitrary value; they move by positional matching.
#    Reused colors across mirrored slots and repeated cavity layouts provide enough
#    structure for a human to see that the same colors disappear from src and appear
#    in the corresponding dst holes.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste


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


def _extract_group_subgrid(grid: np.ndarray, bg: int, pts: np.ndarray, bb: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bb
    sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for rr, cc in pts:
        sub[int(rr) - int(r0), int(cc) - int(c0)] = int(grid[int(rr), int(cc)])
    return sub


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objs = []
    for comp in _components(grid != int(bg))[:2]:
        r0, c0, r1, c1 = comp["bbox"]
        objs.append(
            {
                "bbox": comp["bbox"],
                "offset": (int(r0), int(c0)),
                "pts": comp["pts"],
                "sub": _extract_group_subgrid(grid, bg, comp["pts"], comp["bbox"]),
            }
        )
    return objs


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
            }
        )
    return holes


def _outline_col(sub: np.ndarray, bg: int) -> int:
    cnt = Counter(map(int, sub.flatten()))
    opts = [(color, n) for color, n in cnt.items() if int(color) != int(bg)]
    return int(max(opts, key=lambda item: item[1])[0]) if opts else int(bg)


def _colored_components(sub: np.ndarray, bg: int) -> list[dict]:
    outc = _outline_col(sub, bg)
    mask = (sub != int(bg)) & (sub != int(outc))
    lab, n = label_mask(mask, connectivity=4)
    subs = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        color = Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0]
        subs.append(
            {
                "pts": pts,
                "cent": (
                    int(round(float(pts[:, 0].mean()))),
                    int(round(float(pts[:, 1].mean()))),
                ),
                "color": int(color),
            }
        )
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer colors from src cavities into dst holes on the full canvas.

    Output semantics for this combo:
    - dst holes are filled with the matched src colors.
    - src colored subcomponents are erased back to the background.
    - the full input canvas is preserved; nothing is cropped.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    objs = _extract_objects(g, bg)
    if len(objs) < 2:
        return out

    holes_list = [_enclosed_bg_components(obj["sub"], bg) for obj in objs]
    hole_area = [sum(int(h["size"]) for h in holes) for holes in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)

    dest_obj = objs[dest_i]
    src_obj = objs[src_i]
    dest_holes = holes_list[dest_i]
    src_subs = _colored_components(src_obj["sub"], bg)
    if not dest_holes or not src_subs or len(dest_holes) != len(src_subs):
        return out

    dest_pts = [h["cent"] for h in dest_holes]
    dest_keys, d_rr, d_cc = _rank_keys(dest_pts)

    best = None
    for k in range(4):
        for flip in [False, True]:
            tpts = [
                _transform_point(s["cent"], int(src_obj["sub"].shape[0]), int(src_obj["sub"].shape[1]), int(k), bool(flip))
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
    dest_list.sort(key=lambda item: (item[0][0], item[0][1], item[1][0], item[1][1]))

    tpts_all = [
        _transform_point(s["cent"], int(src_obj["sub"].shape[0]), int(src_obj["sub"].shape[1]), int(k), bool(flip))
        for s in src_subs
    ]
    _keys, s_rr, s_cc = _rank_keys(tpts_all)

    src_list = []
    for src_part in src_subs:
        tp = _transform_point(src_part["cent"], int(src_obj["sub"].shape[0]), int(src_obj["sub"].shape[1]), int(k), bool(flip))
        key = (int(s_rr[int(tp[0])]), int(s_cc[int(tp[1])]))
        src_list.append((key, tp, int(src_part["color"])))
    src_list.sort(key=lambda item: (item[0][0], item[0][1], item[1][0], item[1][1]))

    dr0, dc0 = dest_obj["offset"]
    for (_key1, _tp, color), (_key2, _cent, hole) in zip(src_list, dest_list):
        for rr, cc in hole["pts"]:
            out[int(dr0) + int(rr), int(dc0) + int(cc)] = int(color)

    sr0, sc0 = src_obj["offset"]
    for src_part in src_subs:
        for rr, cc in src_part["pts"]:
            out[int(sr0) + int(rr), int(sc0) + int(cc)] = int(bg)

    return out


# --------------- generator ---------------

CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)

CAVITY_COUNT_RANGE = (4, 9)
FILL_COLOR_COUNT_RANGE = (3, 7)

ROW_HEIGHT_CHOICES = [1, 2, 2, 3, 3, 3]
LAYOUT_K_CHOICES_FULL = [1, 2, 2, 3, 3]
LAYOUT_K_CHOICES_SMALL = [1, 2, 2]
ROW_GAP_CHOICES = [1, 1, 2]

CENTERED_SLOT_WIDTHS = [3, 3, 5, 5, 7]
PAIR_SLOT_WIDTHS = [2, 2, 3, 3, 4]
SIDE_SLOT_WIDTHS = [2, 2, 3]
MID_SLOT_WIDTHS = [1, 3]
PAIR_AXIS_GAP_CHOICES = [1, 1, 3]
TRIO_MID_SIDE_GAP = 1

NONRECT_PROBABILITY = 0.8
MAX_MUTATE_ATTEMPTS = 40

COLOR_MISMATCH_RATE = 0.10

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2

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


def _gen_shape(h: int, w: int, *, nonrect: bool, symmetric: bool = False) -> np.ndarray:
    m = _random_rect_shape(h, w)
    if nonrect and h * w >= 3:
        m = _mutate_shape(m, n_remove_max=max(1, (h * w) // 2), symmetric=symmetric)
    return m


def _mirror_shape(mask: np.ndarray) -> np.ndarray:
    return mask[:, ::-1].copy()


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
    dst_l, src_l = _gen_dst_src_pair(row_h, w, symmetric=False)
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
            "shape_dst": _mirror_shape(dst_l),
            "shape_src": _mirror_shape(src_l),
            "pair_id": pair_id,
        }
    )


def _add_trio_slots(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    w_side = random.choice(SIDE_SLOT_WIDTHS)
    w_mid = random.choice(MID_SLOT_WIDTHS)
    dst_l, src_l = _gen_dst_src_pair(row_h, w_side, symmetric=False)
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
            "shape_dst": _mirror_shape(dst_l),
            "shape_src": _mirror_shape(src_l),
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
    for slot_idx, (slot, shape) in enumerate(zip(layout, shapes)):
        rr, cc = np.where(shape)
        if len(rr) == 0:
            continue
        placed_slots.append((slot_idx, slot["r0"] + rr, slot["col_offset"] + cc))
    if not placed_slots:
        return None, None

    all_rows = np.concatenate([item[1] for item in placed_slots])
    all_cols = np.concatenate([item[2] for item in placed_slots])
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

    bb = bbox(canvas != bg)
    if bb is None:
        return None, None
    cropped = crop_bbox(canvas, bb)
    r0, c0, r1, c1 = bb
    filled_cropped = filled_grid[r0 : r1 + 1, c0 : c1 + 1]
    return cropped, filled_cropped


def _count_4conn_cavities(filled_grid: np.ndarray, bg: int, outline_col: int) -> int:
    mask = (filled_grid != bg) & (filled_grid != outline_col)
    _, n = label_mask(mask, connectivity=4)
    return int(n)


def _count_enclosed_bg_holes(empty_grid: np.ndarray, bg: int) -> int:
    lab, n = label_mask(empty_grid == int(bg), connectivity=4)
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
    mask = grid != int(bg)
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
    for i in range(len(colors)):
        if random.random() < COLOR_MISMATCH_RATE:
            colors[i] = random.choice(fill_pool)
    return colors


def _pick_bg_and_palette() -> tuple[int, list[int]]:
    bg = random.randint(0, 9)
    palette = [color for color in range(10) if color != bg]
    random.shuffle(palette)
    return bg, palette


def _build_object_pair(
    layout: list[dict],
    colors: list[int],
    bg: int,
    outline_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    n_cavities = len(layout)
    dst_shapes = [slot["shape_dst"] for slot in layout]
    src_shapes = [slot["shape_src"] for slot in layout]

    dst_empty, dst_filled = _render_object(layout, dst_shapes, colors, bg, outline_col, filled=False)
    if dst_empty is None or dst_filled is None:
        return None
    if not _is_8connected(dst_empty, bg):
        return None
    if _count_enclosed_bg_holes(dst_empty, bg) != n_cavities:
        return None
    if not _size_in_spec(dst_empty):
        return None

    src_filled, _src_filled_again = _render_object(layout, src_shapes, colors, bg, outline_col, filled=True)
    if src_filled is None:
        return None
    if not _is_8connected(src_filled, bg):
        return None
    if _count_4conn_cavities(src_filled, bg, outline_col) != n_cavities:
        return None
    if not _size_in_spec(src_filled):
        return None

    src_erased, _ = _render_object(layout, src_shapes, colors, bg, outline_col, filled=False)
    if src_erased is None:
        return None
    if not _is_8connected(src_erased, bg):
        return None
    if _count_enclosed_bg_holes(src_erased, bg) != n_cavities:
        return None
    if src_erased.shape != src_filled.shape:
        return None

    return src_filled, src_erased, dst_empty, dst_filled


def _separated_bbox(
    r0: int,
    c0: int,
    h0: int,
    w0: int,
    r1: int,
    c1: int,
    h1: int,
    w1: int,
    gap: int,
) -> bool:
    return (
        int(r0) + int(h0) + int(gap) <= int(r1)
        or int(r1) + int(h1) + int(gap) <= int(r0)
        or int(c0) + int(w0) + int(gap) <= int(c1)
        or int(c1) + int(w1) + int(gap) <= int(c0)
    )


def _all_positions(canvas_h: int, canvas_w: int, patch_h: int, patch_w: int) -> list[tuple[int, int]]:
    positions = []
    r_min = PLACEMENT_BORDER
    c_min = PLACEMENT_BORDER
    r_max = canvas_h - PLACEMENT_BORDER - patch_h
    c_max = canvas_w - PLACEMENT_BORDER - patch_w
    if r_max < r_min or c_max < c_min:
        return positions
    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            positions.append((int(r), int(c)))
    random.shuffle(positions)
    return positions


def _place_on_canvas(
    src_in: np.ndarray,
    src_out: np.ndarray,
    dst_in: np.ndarray,
    dst_out: np.ndarray,
    bg: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    canvas_h = random.randint(*CANVAS_SIZE_RANGE)
    canvas_w = random.randint(*CANVAS_SIZE_RANGE)
    src_h, src_w = src_in.shape
    dst_h, dst_w = dst_in.shape

    src_positions = _all_positions(canvas_h, canvas_w, src_h, src_w)
    if not src_positions:
        return None

    for sr, sc in src_positions:
        dst_positions = _all_positions(canvas_h, canvas_w, dst_h, dst_w)
        dst_positions = [
            (dr, dc)
            for dr, dc in dst_positions
            if _separated_bbox(sr, sc, src_h, src_w, dr, dc, dst_h, dst_w, PLACEMENT_PADDING)
        ]
        if not dst_positions:
            continue
        dr, dc = random.choice(dst_positions)
        inp = np.full((canvas_h, canvas_w), bg, dtype=int)
        out = np.full((canvas_h, canvas_w), bg, dtype=int)
        paste(inp, src_in, sr, sc, transparent=bg)
        paste(inp, dst_in, dr, dc, transparent=bg)
        paste(out, src_out, sr, sc, transparent=bg)
        paste(out, dst_out, dr, dc, transparent=bg)
        return inp, out
    return None


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
    src_filled, src_erased, dst_empty, dst_filled = built

    placed = _place_on_canvas(src_filled, src_erased, dst_empty, dst_filled, bg)
    if placed is None:
        return None
    canvas, expected = placed

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
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
