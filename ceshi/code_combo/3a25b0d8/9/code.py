# Combo variant of 3a25b0d8 (index 9)
# combo_spec: D2/2+D3/3
# Name: src-dst均无对称约束
#
# Design analysis for this fused variant
# --------------------------------------
# a) Concrete behavior:
#    Original generator used LR-mirrored cavity rows, so many examples could be parsed
#    with a "left/right pair" shortcut even though the solver already matched by centroid
#    row/column rank. Here both src and dst are built from one connected slot occupancy
#    matrix, for example occupied slots {(0,0),(0,2),(1,1),(2,1),(2,2)} on a 3x3 grid.
#    Each occupied slot becomes a walled chamber containing one irregular hole/subsprite.
#    Src and dst then get independent chamber sizes and independent cavity shapes. Their
#    only preserved invariant is the set of centroid rank keys such as
#    {(0,0),(0,2),(1,1),(2,1),(2,2)}. Output is still dst with each hole filled by the
#    source color having the same rank key.
# b) Complexity calibration:
#    The layout degrees of freedom are limited to small connected occupancy matrices up to
#    4x4 with 4-9 occupied cells. This is visibly different from the mirrored seed task,
#    but still structured enough that humans can compare ranks from a few examples.
# c) Hidden assumptions checked against the old main():
#    - Old transform search assumed a possible rotate/flip relationship. The fused variant
#      does not generate such transforms, so main now matches in native coordinates only.
#    - Old source outline detection used the most common non-bg color inside src, which can
#      fail once asymmetric layouts change outline/fill area balance. The new main reads the
#      outline color from dst, where the outline is the only non-bg color.
#    - The pairing step assumes each hole/subsprite has a unique rank key. Generator now
#      validates src and dst key sets directly and retries otherwise.
# d) Placement legality:
#    Src and dst are cropped first, then placed with random_free_location() using explicit
#    border and padding, so they never overlap or leave the canvas.
# e) Visual observability:
#    Symmetry removal is visible at the silhouette level: both object masks are rejected if
#    they are LR-symmetric. Since src and dst also use different slot sizes/shapes, the
#    matching clue that remains visually stable is centroid rank order.
# f) Visual cleanliness:
#    Cavities live in tightly packed chambers that share walls when adjacent, so outlines
#    remain a single 8-connected object without cluttering the canvas or stacking markers.
# g) Inferability of color behavior:
#    Colors are copied by positional rank only. Fill palettes are kept to 3-5 colors with
#    repetition, so the user sees that color identity comes from matched locations rather
#    than from symmetry or from numeric color ids.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


def _components(mask: np.ndarray, *, connectivity: int = 8) -> list[dict]:
    lab, n = label_mask(mask, connectivity=connectivity)
    comps: list[dict] = []
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


def _extract_objects(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    comps = _components(grid != int(bg), connectivity=8)
    out: list[np.ndarray] = []
    for comp in comps[:2]:
        r0, c0, r1, c1 = comp["bbox"]
        sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        for rr, cc in comp["pts"]:
            sub[int(rr) - int(r0), int(cc) - int(c0)] = int(grid[int(rr), int(cc)])
        out.append(sub)
    return out


def _tight_crop(obj: np.ndarray, bg: int) -> np.ndarray:
    b = bbox(obj != int(bg))
    if b is None:
        return obj.copy()
    return crop_bbox(obj, b)


def _enclosed_bg_components(sub: np.ndarray, bg: int) -> list[dict]:
    lab, n = label_mask(sub == int(bg), connectivity=4)
    h, w = sub.shape
    holes: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if int(r0) == 0 or int(c0) == 0 or int(r1) == h - 1 or int(c1) == w - 1:
            continue
        cent = (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))
        holes.append({"pts": pts, "cent": cent, "size": int(len(pts))})
    return holes


def _rank_maps(points: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], dict[int, int], dict[int, int]]:
    rows = sorted(set(int(r) for r, _ in points))
    cols = sorted(set(int(c) for _, c in points))
    rrank = {int(r): i for i, r in enumerate(rows)}
    crank = {int(c): i for i, c in enumerate(cols)}
    keys = [(int(rrank[int(r)]), int(crank[int(c)])) for r, c in points]
    return keys, rrank, crank


def _outline_color(sub: np.ndarray, bg: int) -> int:
    cnt = Counter(map(int, sub.flatten()))
    opts = [(col, n) for col, n in cnt.items() if int(col) != int(bg)]
    return int(max(opts, key=lambda item: item[1])[0]) if opts else int(bg)


def _colored_components(sub: np.ndarray, bg: int, outline_col: int) -> list[dict]:
    mask = (sub != int(bg)) & (sub != int(outline_col))
    lab, n = label_mask(mask, connectivity=4)
    subs: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        col = Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0]
        cent = (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))
        subs.append({"pts": pts, "cent": cent, "color": int(col), "size": int(len(pts))})
    return subs


def _non_bg_colors(sub: np.ndarray, bg: int) -> set[int]:
    return {int(v) for v in np.unique(sub) if int(v) != int(bg)}


def _classify_src_dst(objs: list[np.ndarray], bg: int) -> tuple[np.ndarray, np.ndarray]:
    feats = []
    for obj in objs:
        holes = _enclosed_bg_components(obj, bg)
        feats.append(
            {
                "obj": obj,
                "hole_area": sum(int(h["size"]) for h in holes),
                "n_non_bg_colors": len(_non_bg_colors(obj, bg)),
            }
        )
    if len(feats) != 2:
        raise ValueError("expected exactly two objects")
    if sorted(f["n_non_bg_colors"] for f in feats) == [1, 2] or max(f["n_non_bg_colors"] for f in feats) > 1:
        src_feat = max(feats, key=lambda f: (int(f["n_non_bg_colors"]), -int(f["hole_area"])))
        dest_feat = min(feats, key=lambda f: (int(f["n_non_bg_colors"]), -int(f["hole_area"])))
        return src_feat["obj"].copy(), dest_feat["obj"].copy()
    dest_feat = max(feats, key=lambda f: int(f["hole_area"]))
    src_feat = min(feats, key=lambda f: int(f["hole_area"]))
    return src_feat["obj"].copy(), dest_feat["obj"].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused no-symmetry variant:
    - find src (outline + colored cavities) and dst (outline + empty holes)
    - compute centroid row/column rank keys in native coordinates
    - copy each src color into the dst hole with the same rank key
    - tight-crop the filled dst
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    objs = _extract_objects(g, bg)
    if len(objs) < 2:
        raise ValueError("main: expected two objects")
    src, dest = _classify_src_dst(objs, bg)

    outline_col = _outline_color(dest, bg)
    dest_holes = _enclosed_bg_components(dest, bg)
    src_subs = _colored_components(src, bg, outline_col)
    if not dest_holes or not src_subs:
        return _tight_crop(dest, bg)

    dest_pts = [h["cent"] for h in dest_holes]
    dest_keys, d_rr, d_cc = _rank_maps(dest_pts)
    src_pts = [s["cent"] for s in src_subs]
    src_keys, s_rr, s_cc = _rank_maps(src_pts)

    dest_map: dict[tuple[int, int], dict] = {}
    for hole in dest_holes:
        key = (int(d_rr[int(hole["cent"][0])]), int(d_cc[int(hole["cent"][1])]))
        dest_map[key] = hole

    src_map: dict[tuple[int, int], int] = {}
    for sub in src_subs:
        key = (int(s_rr[int(sub["cent"][0])]), int(s_cc[int(sub["cent"][1])]))
        src_map[key] = int(sub["color"])

    if set(src_map) != set(dest_map):
        # Fallback keeps behavior deterministic on off-spec inputs.
        src_list = []
        for sub in src_subs:
            key = (int(s_rr[int(sub["cent"][0])]), int(s_cc[int(sub["cent"][1])]))
            src_list.append((key, sub["cent"], int(sub["color"])))
        dest_list = []
        for hole in dest_holes:
            key = (int(d_rr[int(hole["cent"][0])]), int(d_cc[int(hole["cent"][1])]))
            dest_list.append((key, hole["cent"], hole))
        src_list.sort(key=lambda item: (item[0][0], item[0][1], item[1][0], item[1][1]))
        dest_list.sort(key=lambda item: (item[0][0], item[0][1], item[1][0], item[1][1]))
        for (_, _, col), (_, _, hole) in zip(src_list, dest_list):
            for rr, cc in hole["pts"]:
                dest[int(rr), int(cc)] = int(col)
        return _tight_crop(dest, bg)

    for key, hole in dest_map.items():
        col = int(src_map[key])
        for rr, cc in hole["pts"]:
            dest[int(rr), int(cc)] = int(col)
    return _tight_crop(dest, bg)


CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)

CAVITY_COUNT_RANGE = (4, 9)
FILL_COLOR_COUNT_RANGE = (3, 5)

ROW_HEIGHT_CHOICES = [2, 2, 2, 3]
COL_WIDTH_CHOICES = [2, 2, 2, 3]

NONRECT_PROBABILITY = 0.8
MAX_MUTATE_ATTEMPTS = 40
PAIR_SHAPE_ATTEMPTS = 10

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
MAX_GEN_ATTEMPTS = 500


def _random_rect_shape(h: int, w: int) -> np.ndarray:
    return np.ones((int(h), int(w)), dtype=bool)


def _mutate_shape(mask: np.ndarray, *, n_remove_max: int) -> np.ndarray:
    m = mask.copy()
    h, w = m.shape
    n_remove = random.randint(1, max(1, int(n_remove_max)))
    removed = 0
    tried = 0
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
        _, n_comp = label_mask(m2, connectivity=4)
        if int(n_comp) == 1:
            m = m2
            removed += 1
    return m if m.any() else mask.copy()


def _gen_shape(h: int, w: int) -> np.ndarray:
    m = _random_rect_shape(h, w)
    if h * w >= 4 and random.random() < NONRECT_PROBABILITY:
        m = _mutate_shape(m, n_remove_max=max(1, (h * w) // 2))
    return m


def _gen_distinct_shape_pair(h_src: int, w_src: int, h_dst: int, w_dst: int) -> tuple[np.ndarray, np.ndarray]:
    src = _gen_shape(h_src, w_src)
    dst = _gen_shape(h_dst, w_dst)
    if (int(h_src), int(w_src)) != (int(h_dst), int(w_dst)):
        return src, dst
    for _ in range(PAIR_SHAPE_ATTEMPTS):
        if not np.array_equal(src, dst):
            return src, dst
        src = _gen_shape(h_src, w_src)
        dst = _gen_shape(h_dst, w_dst)
    return src, _mutate_shape(dst, n_remove_max=1) if h_dst * w_dst >= 4 else dst


def _neighbors4_cells(r: int, c: int, rows: int, cols: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = int(r + dr), int(c + dc)
        if 0 <= nr < int(rows) and 0 <= nc < int(cols):
            out.append((int(nr), int(nc)))
    return out


def _is_lr_symmetric_mask(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, mask[:, ::-1]))


def _random_occupancy(rows: int, cols: int, n_target: int) -> np.ndarray | None:
    start = (random.randrange(int(rows)), random.randrange(int(cols)))
    cells = {start}
    while len(cells) < int(n_target):
        frontier = set()
        for r, c in cells:
            frontier.update(_neighbors4_cells(r, c, rows, cols))
        frontier.difference_update(cells)
        if not frontier:
            return None
        cells.add(random.choice(list(frontier)))
    occ = np.zeros((int(rows), int(cols)), dtype=bool)
    for r, c in cells:
        occ[int(r), int(c)] = True
    return occ


def _sample_grid_shape(n_cavities: int) -> tuple[int, int]:
    candidates = []
    for rows in (2, 3, 4):
        for cols in (2, 3, 4):
            if rows * cols >= int(n_cavities):
                candidates.append((rows, cols))
    roomy = [rc for rc in candidates if rc[0] * rc[1] > int(n_cavities)]
    pool = roomy if roomy else candidates
    pool.sort(key=lambda rc: (rc[0] * rc[1], abs(rc[0] - rc[1])))
    return random.choice(pool[: min(3, len(pool))])


def _sample_geometry(rows: int, cols: int) -> dict:
    row_heights = [random.choice(ROW_HEIGHT_CHOICES) for _ in range(int(rows))]
    col_widths = [random.choice(COL_WIDTH_CHOICES) for _ in range(int(cols))]

    row_starts = []
    cur = 0
    for i, rh in enumerate(row_heights):
        row_starts.append(int(cur))
        cur += int(rh) + 1

    col_starts = []
    cur = 0
    for i, cw in enumerate(col_widths):
        col_starts.append(int(cur))
        cur += int(cw) + 1

    return {
        "row_heights": row_heights,
        "col_widths": col_widths,
        "row_starts": row_starts,
        "col_starts": col_starts,
    }


def _slots_from_occ(occ: np.ndarray, geom: dict) -> list[dict]:
    rows, cols = occ.shape
    slots = []
    for r in range(int(rows)):
        for c in range(int(cols)):
            if not bool(occ[int(r), int(c)]):
                continue
            slots.append(
                {
                    "grid_rc": (int(r), int(c)),
                    "r0": int(geom["row_starts"][int(r)]),
                    "c0": int(geom["col_starts"][int(c)]),
                    "h": int(geom["row_heights"][int(r)]),
                    "w": int(geom["col_widths"][int(c)]),
                }
            )
    return slots


def _render_object(
    slots: list[dict],
    shapes: list[np.ndarray],
    colors: list[int],
    bg: int,
    outline_col: int,
    *,
    filled: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not slots:
        return None, None

    h_max = max(int(slot["r0"]) + int(slot["h"]) + 2 for slot in slots)
    w_max = max(int(slot["c0"]) + int(slot["w"]) + 2 for slot in slots)

    canvas = np.full((int(h_max), int(w_max)), int(bg), dtype=int)
    for slot in slots:
        r0, c0 = int(slot["r0"]), int(slot["c0"])
        h, w = int(slot["h"]), int(slot["w"])
        canvas[r0 : r0 + h + 2, c0 : c0 + w + 2] = int(outline_col)

    filled_canvas = canvas.copy()
    for slot, shape, col in zip(slots, shapes, colors):
        r0, c0 = int(slot["r0"]) + 1, int(slot["c0"]) + 1
        rr, cc = np.where(shape)
        if filled:
            canvas[r0 + rr, c0 + cc] = int(col)
        else:
            canvas[r0 + rr, c0 + cc] = int(bg)
        filled_canvas[r0 + rr, c0 + cc] = int(col)

    b = bbox(canvas != int(bg))
    if b is None:
        return None, None
    cropped = crop_bbox(canvas, b)
    r0, c0, r1, c1 = b
    filled_cropped = filled_canvas[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    return cropped, filled_cropped


def _count_4conn_cavities(filled_grid: np.ndarray, bg: int, outline_col: int) -> int:
    mask = (filled_grid != int(bg)) & (filled_grid != int(outline_col))
    _, n = label_mask(mask, connectivity=4)
    return int(n)


def _count_enclosed_bg_holes(empty_grid: np.ndarray, bg: int) -> int:
    lab, n = label_mask(empty_grid == int(bg), connectivity=4)
    h, w = empty_grid.shape
    count = 0
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if int(r0) > 0 and int(c0) > 0 and int(r1) < h - 1 and int(c1) < w - 1:
            count += 1
    return int(count)


def _is_8connected(grid: np.ndarray, bg: int) -> bool:
    mask = grid != int(bg)
    if not mask.any():
        return False
    _, n = label_mask(mask, connectivity=8)
    return int(n) == 1


def _size_in_spec(grid: np.ndarray) -> bool:
    h, w = grid.shape
    return OBJECT_H_RANGE[0] <= int(h) <= OBJECT_H_RANGE[1] and OBJECT_W_RANGE[0] <= int(w) <= OBJECT_W_RANGE[1]


def _pick_bg_and_palette() -> tuple[int, list[int]]:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != int(bg)]
    random.shuffle(palette)
    return int(bg), palette


def _assign_colors(n_cavities: int, fill_palette: list[int]) -> list[int]:
    n_fill = random.randint(FILL_COLOR_COUNT_RANGE[0], min(FILL_COLOR_COUNT_RANGE[1], int(n_cavities), len(fill_palette)))
    pool = list(fill_palette[:n_fill])
    colors = pool[:]
    while len(colors) < int(n_cavities):
        colors.append(random.choice(pool))
    random.shuffle(colors)
    return [int(c) for c in colors]


def _rank_key_set_from_centroids(centroids: list[tuple[int, int]]) -> set[tuple[int, int]]:
    keys, _rr, _cc = _rank_maps(centroids)
    return set((int(rk), int(ck)) for rk, ck in keys)


def _object_is_asymmetric(grid: np.ndarray, bg: int) -> bool:
    mask = grid != int(bg)
    return not bool(np.array_equal(mask, mask[:, ::-1]))


def _build_object_pair(
    occ: np.ndarray,
    colors: list[int],
    bg: int,
    outline_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    rows, cols = occ.shape
    n_cavities = int(np.sum(occ))
    for _ in range(24):
        src_geom = _sample_geometry(rows, cols)
        dst_geom = _sample_geometry(rows, cols)
        src_slots = _slots_from_occ(occ, src_geom)
        dst_slots = _slots_from_occ(occ, dst_geom)

        src_shapes: list[np.ndarray] = []
        dst_shapes: list[np.ndarray] = []
        for src_slot, dst_slot in zip(src_slots, dst_slots):
            src_shape, dst_shape = _gen_distinct_shape_pair(
                int(src_slot["h"]), int(src_slot["w"]), int(dst_slot["h"]), int(dst_slot["w"])
            )
            src_shapes.append(src_shape)
            dst_shapes.append(dst_shape)

        dst_cropped, expected = _render_object(dst_slots, dst_shapes, colors, bg, outline_col, filled=False)
        if dst_cropped is None or expected is None:
            continue
        src_cropped, _src_filled = _render_object(src_slots, src_shapes, colors, bg, outline_col, filled=True)
        if src_cropped is None:
            continue

        if not _size_in_spec(src_cropped) or not _size_in_spec(dst_cropped):
            continue
        if not _is_8connected(src_cropped, bg) or not _is_8connected(dst_cropped, bg):
            continue
        if _count_4conn_cavities(src_cropped, bg, outline_col) != n_cavities:
            continue
        if _count_enclosed_bg_holes(dst_cropped, bg) != n_cavities:
            continue
        if not _object_is_asymmetric(src_cropped, bg) or not _object_is_asymmetric(dst_cropped, bg):
            continue

        src_subs = _colored_components(src_cropped, bg, outline_col)
        dst_holes = _enclosed_bg_components(dst_cropped, bg)
        if len(src_subs) != n_cavities or len(dst_holes) != n_cavities:
            continue
        if _rank_key_set_from_centroids([s["cent"] for s in src_subs]) != _rank_key_set_from_centroids([h["cent"] for h in dst_holes]):
            continue
        return src_cropped, dst_cropped, expected
    return None


def _place_on_canvas(src: np.ndarray, dst: np.ndarray, bg: int) -> np.ndarray | None:
    canvas_h = random.randint(*CANVAS_SIZE_RANGE)
    canvas_w = random.randint(*CANVAS_SIZE_RANGE)
    canvas = np.full((int(canvas_h), int(canvas_w)), int(bg), dtype=int)
    try:
        r0, c0 = random_free_location(
            canvas,
            src,
            transparent=bg,
            border=PLACEMENT_BORDER,
            padding=PLACEMENT_PADDING,
        )
        paste(canvas, src, r0, c0, transparent=bg)
        r1, c1 = random_free_location(
            canvas,
            dst,
            transparent=bg,
            border=PLACEMENT_BORDER,
            padding=PLACEMENT_PADDING,
        )
        paste(canvas, dst, r1, c1, transparent=bg)
    except ValueError:
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    n_cavities = random.randint(*CAVITY_COUNT_RANGE)
    rows, cols = _sample_grid_shape(n_cavities)
    occ = _random_occupancy(rows, cols, n_cavities)
    if occ is None:
        return None

    bg, palette = _pick_bg_and_palette()
    outline_col = int(palette[0])
    colors = _assign_colors(n_cavities, palette[1:])

    built = _build_object_pair(occ, colors, bg, outline_col)
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
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
