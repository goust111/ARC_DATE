# Combo variant of 3a25b0d8 (index 19)
# combo_spec: D8/1+D13/2
# Name: 四重旋转对称+螺旋序匹配
#
# Design analysis for the fused variant
# -------------------------------------
# 1. Concrete behavior:
#    Original 3a25b0d8 sorts src subsprites / dst holes by (row_rank, col_rank).
#    That works for LR-symmetric row layouts but becomes ambiguous under C4 symmetry.
#    This variant therefore uses a visible square-spiral order:
#      outer UL -> outer UR -> outer LR -> outer LL -> inner UL -> inner UR -> inner LR -> inner LL
#    where UL/UR/LR/LL are the four quadrants around the object's centroid.
#    Example with two rings:
#      row/col order   = outer UL, outer UR, inner UL, inner UR, inner LL, inner LR, outer LL, outer LR
#      spiral order    = outer UL, outer UR, outer LR, outer LL, inner UL, inner UR, inner LR, inner LL
#    The solver must therefore stop relying on row/col ranks and instead sort by
#    radial shell (outside to inside) plus clockwise quadrant order.
#
# 2. Complexity calibration:
#    The generator always creates 8 cavities arranged as two rotationally repeated
#    quartets. Using only one ring would make "outside to inside" visually weak;
#    using more than two rings would make placement and human inference noisier.
#    Two repeated quartets are enough for the human to see the periodicity.
#
# 3. Hidden assumptions broken from the original solver:
#    - _rank_keys(...): no longer valid because many cavities share repeated row/col ranks.
#    - transform search by row/col signatures: degenerate under C4 symmetry, since 90-degree
#      rotations preserve the overall cavity set. The new generator therefore does not add an
#      extra global object rotation; instead the canonical orientation is the visible grid
#      orientation, and the spiral starts from the upper-left quadrant.
#    - deterministic tie-breaking by centroid coordinates alone: insufficient, because the
#      essential distinction is shell + clockwise phase, not lexicographic position.
#
# 4. Placement legality:
#    Objects are still rendered as sparse outline-only shapes around cavities, so random
#    placement on a 22-30 canvas remains feasible even when their tight bboxes are moderate.
#    random_free_location(...) guarantees no overlap, no border violation, and a padding gap.
#
# 5. Visual observability / cleanliness:
#    The quartets are placed in four quadrants and repeated on two shells, so the rotational
#    symmetry is obvious in 2-3 examples. Spiral matching is also visible because every sample
#    contains two shells, and the generator rejects cases where the old row/col rule would
#    accidentally produce the same coloring as the new spiral rule.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Find the two non-bg 8-connected objects.
    - dst = the object with enclosed bg holes; src = the one with colored cavity fills.
    - Both objects use a C4-symmetric cavity arrangement: two shells of four quadrants each.
    - Match src colors to dst holes by visible square-spiral order:
      outer shell first, clockwise UL->UR->LR->LL, then the next inner shell.
    - Fill dst holes with the matched colors and tight-crop.
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

    def _tight_crop_nonbg(sub: np.ndarray) -> np.ndarray:
        m = sub != int(bg)
        pts = np.argwhere(m)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        return sub[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

    def _enclosed_bg_components(sub: np.ndarray) -> list[dict]:
        m = sub == int(bg)
        lab, n = label_mask(m, connectivity=4)
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
            holes.append(
                {
                    "pts": pts,
                    "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                    "size": int(len(pts)),
                }
            )
        return holes

    def _outline_col(sub: np.ndarray) -> int:
        cnt = Counter(map(int, sub.flatten()))
        opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
        return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)

    def _colored_components(sub: np.ndarray) -> list[dict]:
        outc = _outline_col(sub)
        m = (sub != int(bg)) & (sub != int(outc))
        lab, n = label_mask(m, connectivity=4)
        subs = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            col = Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0]
            subs.append(
                {
                    "pts": pts,
                    "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                    "color": int(col),
                }
            )
        return subs

    def _quadrant_phase(cent: tuple[float, float], center: tuple[float, float]) -> int:
        dr = float(cent[0]) - float(center[0])
        dc = float(cent[1]) - float(center[1])
        eps = 1e-6
        if dr < -eps and dc < -eps:
            return 0  # UL
        if dr < -eps and dc >= -eps:
            return 1  # UR
        if dr >= -eps and dc >= -eps:
            return 2  # LR
        return 3      # LL

    def _spiral_sort(items: list[dict]) -> list[dict]:
        center = (
            float(np.mean([d["cent"][0] for d in items])),
            float(np.mean([d["cent"][1] for d in items])),
        )

        def _key(d: dict) -> tuple[float, int, float, float]:
            dr = float(d["cent"][0]) - center[0]
            dc = float(d["cent"][1]) - center[1]
            radial = max(abs(dr), abs(dc))
            phase = _quadrant_phase(d["cent"], center)
            return (-float(radial), int(phase), float(d["cent"][0]), float(d["cent"][1]))

        return sorted(items, key=_key)

    groups = _extract_two_objects()
    assert len(groups) >= 2
    objs = [_extract_group_subgrid(grp) for grp in groups[:2]]

    holes_list = [_enclosed_bg_components(o) for o in objs]
    hole_area = [sum(int(h["size"]) for h in hh) for hh in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    dest = objs[dest_i].copy()
    src = objs[src_i].copy()

    dest_holes = _enclosed_bg_components(dest)
    src_subs = _colored_components(src)
    if not dest_holes or not src_subs:
        return _tight_crop_nonbg(dest)

    dest_order = _spiral_sort(dest_holes)
    src_order = _spiral_sort(src_subs)

    for s, h in zip(src_order, dest_order):
        for rr, cc in h["pts"]:
            dest[int(rr), int(cc)] = int(s["color"])

    return _tight_crop_nonbg(dest)


CANVAS_SIZE_RANGE = (30, 30)
OBJECT_H_RANGE = (11, 15)
OBJECT_W_RANGE = (11, 15)

CAVITY_COUNT = 8
FILL_COLOR_COUNT_RANGE = (4, 6)
NONRECT_PROBABILITY = 0.85
MAX_MUTATE_ATTEMPTS = 40
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
MAX_GEN_ATTEMPTS = 400

BODY_SIZE = 13
OUTER_RADIUS = 4
INNER_RADIUS = 2
BOX_SIZE = 2
SPIRAL_SIGNS = [(-1, -1), (-1, 1), (1, 1), (1, -1)]  # UL, UR, LR, LL


def _random_rect_shape(size: int) -> np.ndarray:
    return np.ones((int(size), int(size)), dtype=bool)


def _mutate_shape(mask: np.ndarray, *, n_remove_max: int) -> np.ndarray:
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
        m2 = m.copy()
        m2[r, c] = False
        if not m2.any():
            continue
        _, n_components = label_mask(m2, connectivity=4)
        if int(n_components) == 1:
            m = m2
            removed += 1
    return m if m.any() else mask.copy()


def _gen_shape(size: int) -> np.ndarray:
    m = _random_rect_shape(size)
    if random.random() < NONRECT_PROBABILITY and size * size >= 3:
        limit = max(1, (size * size) // 2)
        m = _mutate_shape(m, n_remove_max=limit)
    return m


def _gen_distinct_pair(size: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(50):
        dst = _gen_shape(size)
        src = _gen_shape(size)
        if not np.array_equal(dst, src):
            return dst, src
    dst = np.ones((size, size), dtype=bool)
    src = np.eye(size, dtype=bool)
    if not src.any():
        src[0, 0] = True
    return dst, src


def _slot_top_left(radius: int, size: int, sign_r: int, sign_c: int) -> tuple[int, int]:
    center = BODY_SIZE // 2
    center_r = center + int(sign_r) * int(radius)
    center_c = center + int(sign_c) * int(radius)
    return int(center_r - size // 2), int(center_c - size // 2)


def _design_layout() -> list[dict]:
    layout: list[dict] = []
    ring_specs = [
        {"radius": int(OUTER_RADIUS), "size": int(BOX_SIZE)},
        {"radius": int(INNER_RADIUS), "size": int(BOX_SIZE)},
    ]

    for ring_idx, spec in enumerate(ring_specs):
        size = int(spec["size"])
        dst_base, src_base = _gen_distinct_pair(size)
        for step_idx, (sr, sc) in enumerate(SPIRAL_SIGNS):
            r0, c0 = _slot_top_left(int(spec["radius"]), size, int(sr), int(sc))
            layout.append(
                {
                    "ring_idx": int(ring_idx),
                    "step_idx": int(step_idx),
                    "r0": int(r0),
                    "c0": int(c0),
                    "shape_dst": np.rot90(dst_base, k=(-step_idx) % 4).copy(),
                    "shape_src": np.rot90(src_base, k=(-step_idx) % 4).copy(),
                }
            )
    return layout


def _render_object(
    layout: list[dict],
    shapes: list[np.ndarray],
    colors: list[int],
    bg: int,
    outline_col: int,
    *,
    filled: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    canvas = np.full((BODY_SIZE, BODY_SIZE), outline_col, dtype=int)
    filled_grid = canvas.copy()
    any_fill = False
    for slot_idx, (slot, sh) in enumerate(zip(layout, shapes)):
        rr, cc = np.where(sh)
        if len(rr) == 0:
            continue
        any_fill = True
        ar = int(slot["r0"]) + rr
        ac = int(slot["c0"]) + cc
        filled_grid[ar, ac] = int(colors[slot_idx])
        if filled:
            canvas[ar, ac] = int(colors[slot_idx])
        else:
            canvas[ar, ac] = int(bg)
    if not any_fill:
        return None, None
    return canvas.copy(), filled_grid.copy()


def _count_4conn_cavities(filled_grid: np.ndarray, bg: int, outline_col: int) -> int:
    mask = (filled_grid != bg) & (filled_grid != outline_col)
    _, n = label_mask(mask, connectivity=4)
    return int(n)


def _count_enclosed_bg_holes(empty_grid: np.ndarray, bg: int) -> int:
    mask = empty_grid == bg
    lab, n = label_mask(mask, connectivity=4)
    H, W = empty_grid.shape
    count = 0
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r0 > 0 and c0 > 0 and r1 < H - 1 and c1 < W - 1:
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


def _make_slot_colors(fill_pool: list[int], n_slots: int) -> list[int]:
    cols = []
    seed = fill_pool.copy()
    random.shuffle(seed)
    while len(cols) < n_slots:
        cols.extend(seed if random.random() < 0.6 else random.sample(fill_pool, len(fill_pool)))
    cols = cols[:n_slots]
    random.shuffle(cols)
    return [int(c) for c in cols]


def _rank_keys(pts: list[tuple[float, float]]) -> tuple[list[tuple[int, int]], dict[float, int], dict[float, int]]:
    rows = sorted(set(float(r) for r, _ in pts))
    cols = sorted(set(float(c) for _, c in pts))
    rrank = {r: i for i, r in enumerate(rows)}
    crank = {c: i for i, c in enumerate(cols)}
    keys = [(int(rrank[r]), int(crank[c])) for r, c in pts]
    return keys, rrank, crank


def _enclosed_bg_components(sub: np.ndarray, bg: int) -> list[dict]:
    m = sub == int(bg)
    lab, n = label_mask(m, connectivity=4)
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
        holes.append({"pts": pts, "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean()))})
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
        subs.append({"pts": pts, "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())), "color": int(col)})
    return subs


def _fill_by_rowcol(dest: np.ndarray, src: np.ndarray, bg: int) -> np.ndarray:
    out = dest.copy()
    dest_holes = _enclosed_bg_components(dest, bg)
    src_subs = _colored_components(src, bg)
    dest_pts = [h["cent"] for h in dest_holes]
    dest_keys, d_rr, d_cc = _rank_keys(dest_pts)

    dest_list = []
    for h in dest_holes:
        key = (int(d_rr[float(h["cent"][0])]), int(d_cc[float(h["cent"][1])]))
        dest_list.append((key, h["cent"], h))
    dest_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    src_pts = [s["cent"] for s in src_subs]
    _, s_rr, s_cc = _rank_keys(src_pts)
    src_list = []
    for s in src_subs:
        key = (int(s_rr[float(s["cent"][0])]), int(s_cc[float(s["cent"][1])]))
        src_list.append((key, s["cent"], int(s["color"])))
    src_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    for (_, _, col), (_, _, h) in zip(src_list, dest_list):
        for rr, cc in h["pts"]:
            out[int(rr), int(cc)] = int(col)
    return out


def _build_object_pair(
    layout: list[dict],
    colors: list[int],
    bg: int,
    outline_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    dst_shapes = [slot["shape_dst"] for slot in layout]
    src_shapes = [slot["shape_src"] for slot in layout]

    dst_cropped, dst_filled = _render_object(layout, dst_shapes, colors, bg, outline_col, filled=False)
    if dst_cropped is None or dst_filled is None:
        return None
    if not _is_8connected(dst_cropped, bg):
        return None
    if _count_enclosed_bg_holes(dst_cropped, bg) != CAVITY_COUNT:
        return None
    if not _size_in_spec(dst_cropped):
        return None

    src_cropped, _ = _render_object(layout, src_shapes, colors, bg, outline_col, filled=True)
    if src_cropped is None:
        return None
    if not _is_8connected(src_cropped, bg):
        return None
    if _count_4conn_cavities(src_cropped, bg, outline_col) != CAVITY_COUNT:
        return None
    if not _size_in_spec(src_cropped):
        return None

    wrong = _fill_by_rowcol(dst_cropped, src_cropped, bg)
    mismatch = int(np.sum(wrong != dst_filled))
    if mismatch <= 0:
        return None

    return src_cropped, dst_cropped, dst_filled, mismatch


def _place_on_canvas(src: np.ndarray, dst: np.ndarray, bg: int) -> np.ndarray | None:
    canvas_h = random.randint(*CANVAS_SIZE_RANGE)
    canvas_w = random.randint(*CANVAS_SIZE_RANGE)
    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    h, w = src.shape
    candidates = [
        (1, 1),
        (1, canvas_w - w - 1),
        (canvas_h - h - 1, 1),
        (canvas_h - h - 1, canvas_w - w - 1),
    ]
    random.shuffle(candidates)
    src_pos = candidates[0]
    dst_pos = candidates[1]
    paste(canvas, src, int(src_pos[0]), int(src_pos[1]), transparent=bg)
    paste(canvas, dst, int(dst_pos[0]), int(dst_pos[1]), transparent=bg)
    return canvas


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline_col = int(palette[0])
    n_fill_cols = random.randint(FILL_COLOR_COUNT_RANGE[0], FILL_COLOR_COUNT_RANGE[1])
    fill_pool = palette[1 : 1 + n_fill_cols]

    for _ in range(20):
        layout = _design_layout()
        colors = _make_slot_colors(fill_pool, len(layout))
        built = _build_object_pair(layout, colors, bg, outline_col)
        if built is None:
            continue
        src, dst, expected, _mismatch = built
        canvas = _place_on_canvas(src, dst, bg)
        if canvas is None:
            continue
        try:
            actual = main(canvas)
        except Exception:
            continue
        if actual.shape == expected.shape and np.array_equal(actual, expected):
            return canvas
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
