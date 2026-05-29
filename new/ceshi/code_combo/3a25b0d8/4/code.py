# Combo variant of 3a25b0d8 (index 4)
# combo_spec: D1/3+D8/2
# Name: 非平凡旋转匹配+无对称约束
#
# Design analysis
# a) Concrete behavior:
#    Original 3a25b0d8 already paired src subsprites to dst holes by centroid row/col
#    ranks, but its generator kept the visible src and dst in the same orientation and
#    also built LR-symmetric layouts. This combo removes both crutches. The generator
#    now builds an asymmetric cavity arrangement, renders dst as an outline-with-holes
#    object, renders src from the same arrangement with independently mutated cavity
#    shapes, then shows src only after a non-identity D4 transform. Example trace:
#    canonical dst hole centroids might rank as [(0,0), (0,2), (1,1), (2,0), (2,2)].
#    Visible src is a 90/180/270 rotation or flip of that asymmetric arrangement, so
#    under identity its rank keys differ; only after mentally rotating/flipping the
#    whole arrangement do the keys line up and the colors can be transferred.
# b) Complexity calibration:
#    The transform family is still small and structured: one unique non-identity D4
#    match per instance. Layouts use 4-9 cavities, 3-7 fill colors, repeated colors,
#    and 3+ distinct row/col ranks, so matching cannot collapse to trivial color lookup
#    or a single straight-line ordering.
# c) Implicit assumption audit for main():
#    - `extract_two_objects`: still assumes exactly two salient 8-connected objects;
#      generator places only src and dst.
#    - `dest = argmax(hole_area)`: still assumes dst has enclosed bg holes while src
#      does not; generator keeps src cavities filled and dst cavities empty.
#    - `_colored_components` assumes each subsprite is monochrome and 4-connected;
#      generator paints each cavity with one color and keeps cavity masks 4-connected.
#    - `_rank_keys` assumes centroid ranks are meaningful and equal in count; generator
#      derives src/dst from one layout and rejects samples with ambiguous transform hits.
#    - transform search used to have a dead non-identity branch because data were
#      identity and symmetric. Here generator explicitly rejects identity-compatible or
#      multi-transform-compatible layouts, so the nontrivial branch is required.
# d) Placement legality:
#    Src and dst are still placed independently with `random_free_location`, keeping
#    them in-bounds, non-overlapping, and separated by the required padding.
# e) Visual observability:
#    No hidden numeric property participates. The visible evidence is purely geometric:
#    both objects are asymmetric, and only one rotated/flipped view of src aligns with
#    dst's hole arrangement.
# f) Visual cleanliness:
#    The scene remains just two outlined objects on a blank canvas. There are no extra
#    markers or overlaid hints; asymmetry comes from the cavity arrangement itself.
# g) Color inferability:
#    Colors are reused across cavities, so the task remains "match positions after the
#    right transform" rather than "each color is unique, just copy by inspection".

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


TRANSFORM_PREFERENCE = [
    (1, False),
    (2, False),
    (3, False),
    (0, True),
    (2, True),
    (1, True),
    (3, True),
    (0, False),
]
GENERATOR_VISIBLE_TRANSFORMS = [t for t in TRANSFORM_PREFERENCE if t != (0, False)]


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


def _extract_two_objects(g: np.ndarray, bg: int) -> list[list[dict]]:
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


def _extract_group_subgrid(g: np.ndarray, group: list[dict], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_of(group)
    sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for d in group:
        for rr, cc in d["pts"]:
            sub[int(rr) - int(r0), int(cc) - int(c0)] = int(g[int(rr), int(cc)])
    return sub


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
        cent = (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))
        holes.append({"pts": pts, "cent": cent, "size": int(len(pts))})
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
        cent = (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))
        subs.append({"cent": cent, "color": int(col)})
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


def _transform_grid(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(np.array(arr, dtype=int), -int(k))
    if bool(flip):
        out = np.fliplr(out)
    return np.array(out, dtype=int)


def _tight_crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray:
    m = arr != int(bg)
    pts = np.argwhere(m)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _matching_transforms_from_parts(
    src_subs: list[dict],
    dest_holes: list[dict],
    src_shape: tuple[int, int],
) -> list[tuple[int, bool]]:
    if not src_subs or not dest_holes or len(src_subs) != len(dest_holes):
        return []
    dest_pts = [h["cent"] for h in dest_holes]
    dest_keys = sorted(_rank_keys(dest_pts)[0])
    matches = []
    for k, flip in TRANSFORM_PREFERENCE:
        tpts = [
            _transform_point(s["cent"], int(src_shape[0]), int(src_shape[1]), int(k), bool(flip))
            for s in src_subs
        ]
        tkeys = sorted(_rank_keys(tpts)[0])
        if tkeys == dest_keys:
            matches.append((int(k), bool(flip)))
    return matches


def _matching_transforms_for_grids(src: np.ndarray, dst: np.ndarray, bg: int) -> list[tuple[int, bool]]:
    return _matching_transforms_from_parts(
        _colored_components(src, bg),
        _enclosed_bg_components(dst, bg),
        src.shape,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill dst holes with src colors after the unique nontrivial rotation/flip alignment,
    then tight-crop the completed dst object.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    groups = _extract_two_objects(g, bg)
    assert not (len(groups) < 2)
    objs = [_extract_group_subgrid(g, grp, bg) for grp in groups]

    holes_list = [_enclosed_bg_components(o, bg) for o in objs]
    hole_area = [sum(int(h["size"]) for h in hh) for hh in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    dest = objs[dest_i].copy()
    src = objs[src_i].copy()

    dest_holes = _enclosed_bg_components(dest, bg)
    src_subs = _colored_components(src, bg)
    if not dest_holes or not src_subs:
        return _tight_crop_non_bg(dest, bg)

    dest_pts = [h["cent"] for h in dest_holes]
    _dest_keys, d_rr, d_cc = _rank_keys(dest_pts)
    matches = _matching_transforms_from_parts(src_subs, dest_holes, src.shape)
    k, flip = matches[0] if matches else (0, False)

    dest_list = []
    for h in dest_holes:
        key = (int(d_rr[int(h["cent"][0])]), int(d_cc[int(h["cent"][1])]))
        dest_list.append((key, h["cent"], h))
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

    for (_key1, _p1, col), (_key2, _p2, h) in zip(src_list, dest_list):
        for rr, cc in h["pts"]:
            dest[int(rr), int(cc)] = int(col)

    return _tight_crop_non_bg(dest, bg)


CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)

CAVITY_COUNT_RANGE = (4, 9)
FILL_COLOR_COUNT_RANGE = (3, 7)

ROW_HEIGHT_CHOICES = [1, 1, 2, 2, 3]
ROW_SLOT_CHOICES_FULL = [1, 2, 2, 3]
ROW_SLOT_CHOICES_SMALL = [1, 2, 2]
ROW_GAP_CHOICES = [1, 1, 2]
COL_GAP_CHOICES = [1, 1, 1, 2]
SLOT_W_CHOICES = [1, 2, 2, 3, 3, 4]
LAYOUT_WIDTH_CHOICES = [5, 6, 7, 8, 9, 10]

NONRECT_PROBABILITY = 0.8
MAX_MUTATE_ATTEMPTS = 40

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2

MAX_GEN_ATTEMPTS = 800
MAX_LAYOUT_HEIGHT = 11


def _random_rect_shape(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _mutate_shape(mask: np.ndarray, *, n_remove_max: int) -> np.ndarray:
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
        m2 = m.copy()
        m2[int(r), int(c)] = False
        if not m2.any():
            continue
        _, n_components = label_mask(m2, connectivity=4)
        if int(n_components) == 1:
            m = m2
            removed += 1
    if not m.any():
        return mask.copy()
    return m


def _gen_shape(h: int, w: int, *, nonrect: bool) -> np.ndarray:
    m = _random_rect_shape(h, w)
    if nonrect and h * w >= 3:
        m = _mutate_shape(m, n_remove_max=max(1, (h * w) // 2))
    return m


def _gen_dst_src_pair(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    nonrect = random.random() < NONRECT_PROBABILITY
    return (
        _gen_shape(h, w, nonrect=nonrect),
        _gen_shape(h, w, nonrect=nonrect),
    )


def _pick_row_slot_count(remaining: int) -> int:
    if remaining == 1:
        return 1
    if remaining == 2:
        return random.choice(ROW_SLOT_CHOICES_SMALL)
    return random.choice(ROW_SLOT_CHOICES_FULL)


def _design_layout(n_target: int) -> list[dict] | None:
    for _ in range(120):
        width_budget = random.choice(LAYOUT_WIDTH_CHOICES)
        slots: list[dict] = []
        cur_r = 0
        remaining = n_target
        prev_span: tuple[int, int] | None = None
        ok = True
        while remaining > 0 and cur_r < MAX_LAYOUT_HEIGHT:
            row_h = random.choice(ROW_HEIGHT_CHOICES)
            k = min(_pick_row_slot_count(remaining), remaining)
            row_ok = False
            for _row_try in range(40):
                widths = [random.choice(SLOT_W_CHOICES) for _ in range(k)]
                gaps = [random.choice(COL_GAP_CHOICES) for _ in range(max(0, k - 1))]
                total = int(sum(widths) + sum(gaps))
                if total > width_budget:
                    continue
                left = random.randint(0, int(width_budget - total))
                row_span = (int(left), int(left + total - 1))
                if prev_span is not None:
                    if row_span[1] < prev_span[0] - 1 or row_span[0] > prev_span[1] + 1:
                        continue
                c = int(left)
                row_slots = []
                for i, w in enumerate(widths):
                    dst_shape, src_shape = _gen_dst_src_pair(int(row_h), int(w))
                    row_slots.append(
                        {
                            "r0": int(cur_r),
                            "c0": int(c),
                            "shape_dst": dst_shape,
                            "shape_src": src_shape,
                        }
                    )
                    c += int(w)
                    if i < len(gaps):
                        c += int(gaps[i])
                slots.extend(row_slots)
                remaining -= int(k)
                prev_span = row_span
                row_ok = True
                break
            if not row_ok:
                ok = False
                break
            cur_r += int(row_h + random.choice(ROW_GAP_CHOICES))
        if ok and remaining == 0:
            return slots
    return None


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
        placed_slots.append((slot_idx, slot["r0"] + rr, slot["c0"] + cc))
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


def _mask_has_any_symmetry(grid: np.ndarray, bg: int) -> bool:
    m = (grid != int(bg)).astype(int)
    for k, flip in GENERATOR_VISIBLE_TRANSFORMS:
        t = _transform_grid(m, k, flip)
        if t.shape == m.shape and np.array_equal(t, m):
            return True
    return False


def _arrangement_is_rich(dst: np.ndarray, bg: int) -> bool:
    holes = _enclosed_bg_components(dst, bg)
    if len(holes) < CAVITY_COUNT_RANGE[0]:
        return False
    pts = [h["cent"] for h in holes]
    if len(set(int(r) for r, _c in pts)) < 3:
        return False
    if len(set(int(c) for _r, c in pts)) < 3:
        return False
    return True


def _assign_slot_colors(n_slots: int, fill_pool: list[int]) -> list[int]:
    colors = list(fill_pool[: min(len(fill_pool), n_slots)])
    while len(colors) < n_slots:
        colors.append(random.choice(fill_pool))
    random.shuffle(colors)
    return [int(c) for c in colors]


def _pick_bg_and_palette() -> tuple[int, list[int]]:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    return bg, palette


def _build_object_pair(
    layout: list[dict],
    colors: list[int],
    bg: int,
    outline_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n_cavities = len(layout)
    dst_shapes = [slot["shape_dst"] for slot in layout]
    src_shapes = [slot["shape_src"] for slot in layout]

    dst_cropped, dst_filled = _render_object(layout, dst_shapes, colors, bg, outline_col, filled=False)
    if dst_cropped is None:
        return None
    if not _is_8connected(dst_cropped, bg):
        return None
    if _count_enclosed_bg_holes(dst_cropped, bg) != n_cavities:
        return None
    if not _size_in_spec(dst_cropped):
        return None
    if not _arrangement_is_rich(dst_cropped, bg):
        return None
    if _mask_has_any_symmetry(dst_cropped, bg):
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
    if _mask_has_any_symmetry(src_cropped, bg):
        return None

    return src_cropped, dst_cropped, dst_filled


def _select_transformed_src(
    src: np.ndarray,
    dst: np.ndarray,
    bg: int,
) -> tuple[np.ndarray, tuple[int, bool], tuple[int, bool]] | None:
    choices = GENERATOR_VISIBLE_TRANSFORMS[:]
    random.shuffle(choices)
    for applied in choices:
        src_t = _transform_grid(src, applied[0], applied[1])
        if not _size_in_spec(src_t):
            continue
        if _mask_has_any_symmetry(src_t, bg):
            continue
        matches = _matching_transforms_for_grids(src_t, dst, bg)
        if len(matches) != 1:
            continue
        if matches[0] == (0, False):
            continue
        return src_t, applied, matches[0]
    return None


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
    outline_col = int(palette[0])

    layout = _design_layout(random.randint(*CAVITY_COUNT_RANGE))
    if layout is None or len(layout) < CAVITY_COUNT_RANGE[0]:
        return None
    n_cavities = len(layout)

    fc_lo, fc_hi = FILL_COLOR_COUNT_RANGE
    max_cols = min(fc_hi, len(palette) - 1, max(fc_lo, n_cavities - 1))
    if max_cols < fc_lo:
        return None
    n_fill_cols = random.randint(fc_lo, max_cols)
    fill_pool = palette[1 : 1 + n_fill_cols]
    colors = _assign_slot_colors(n_cavities, fill_pool)

    built = _build_object_pair(layout, colors, bg, outline_col)
    if built is None:
        return None
    src, dst, expected = built

    selected = _select_transformed_src(src, dst, bg)
    if selected is None:
        return None
    src_t, _applied, _solver_match = selected

    canvas = _place_on_canvas(src_t, dst, bg)
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
