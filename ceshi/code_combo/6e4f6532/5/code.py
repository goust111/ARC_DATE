# Combo variant of 6e4f6532 (index 5)
# combo_spec: D1/3+D4/2
# Name: 边缘颜色匹配+负空间车站

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior:
#    Original task used isolated 9-clusters as stations and matched trains by
#    coupler-count. Here the "station" is a 1-cell background cavity inside a
#    local stripe cross: the four orthogonal neighbors of the hole reuse the
#    target region's boundary-stripe colors. Example: if a region is bounded by
#    {top=1,right=2,bottom=3,left=4}, then the station is a bg cell with
#    neighbors up/right/down/left = 1/2/3/4. A train still carries one 9
#    coupler, but all trains do, so count is useless. The solver must detect
#    the cavity as negative space, match the train whose marker-color set is
#    exactly {1,2,3,4}, then rotate/flip so each marker faces its same-color
#    regional boundary before docking the coupler into the hole.
# b) Complexity calibration:
#    The generator always creates 4 global stripes on the border plus one full
#    vertical and one full horizontal internal stripe, yielding 4 rectangular
#    regions with distinct 4-color boundary sets. This keeps the matching space
#    small and human-visible while making size-based matching impossible.
# c) Assumption audit versus the original solver:
#    - "station_groups()" on isolated 9s is invalid: stations are bg holes, so
#      the new solver scans for cavity cells whose 4-neighbors match regional
#      stripe colors.
#    - "groups_by_size[len(g)]" is invalid: all holes are size-1 on purpose, so
#      count no longer distinguishes targets.
#    - "skip drawing 9 because station already exists" is invalid: the output
#      must draw the coupler 9 into the hole, because the input station is bg.
#    - Region detection via global stripes still works, so that part is kept.
# d) Placement legality:
#    Docked patches are built around a single coupler cell and explicitly forbid
#    the four orthogonal cells around that coupler, because those belong to the
#    fixed station cross. Body cells therefore connect diagonally, which keeps
#    the train 8-connected in input/output while respecting the cavity.
# e/f/g) Visual observability and cleanliness:
#    The key variable is the 4-color marker set. Every sample has duplicate
#    coupler counts (all trains have one coupler), so the old rule is visibly
#    broken. Regions are large, object counts are only 2-3, and each station is
#    a small colored cross around a bg hole, keeping the grid legible.


OPS = [
    (False, 0),
    (False, 1),
    (False, 2),
    (False, 3),
    (True, 0),
    (True, 1),
    (True, 2),
    (True, 3),
]
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_COLOR_SIDE = {0: "top", 1: "right", 2: "bottom", 3: "left"}


GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 29)
GRID_MIN_AREA = 320
EDGE_THICK_RANGE = (2, 4)
INTERNAL_THICK = 2
MIN_REGION_DIM = 8
TRAIN_COUNTS = [2]
BODY_MIN = 6
BODY_MAX = 16
PATCH_H_RANGE = (4, 8)
PATCH_W_RANGE = (4, 8)
BODY_GROWTH_ITERS = 120
PATH_BOUNDS_PAD = 4
LAYOUT_RETRIES = 200
PATCH_RETRIES = 80
GENERATE_RETRIES = 300
COLOR_POOL = list(range(8))


def apply_transform(arr: np.ndarray, flip_ud: bool = False, rot_k: int = 0) -> np.ndarray:
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def transform_coords(coords, shape, flip_ud: bool = False, rot_k: int = 0):
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    out = apply_transform(mask, flip_ud=flip_ud, rot_k=rot_k)
    return set(map(tuple, np.argwhere(out)))


def inverse_patch(patch: np.ndarray, op) -> np.ndarray:
    for inv in OPS:
        cand = apply_transform(patch, *inv)
        if np.array_equal(apply_transform(cand, *op), patch):
            return cand
    raise ValueError("no inverse transform")


def bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def dims(coords):
    r0, c0, r1, c1 = bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def path_to(body, goal, blocked, bounds):
    if goal in body:
        return True
    r0, r1, c0, c1 = bounds
    q = deque(body)
    prev = {p: None for p in body}
    dirs = DIRS4[:]
    random.shuffle(dirs)
    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            p = (r + dr, c + dc)
            if p in prev or p in blocked:
                continue
            if not (r0 <= p[0] <= r1 and c0 <= p[1] <= c1):
                continue
            prev[p] = (r, c)
            if p == goal:
                cur = p
                while cur is not None:
                    body.add(cur)
                    cur = prev[cur]
                return True
            q.append(p)
    return False


def grow_body(body, blocked, target):
    for _ in range(BODY_GROWTH_ITERS):
        all_cells = body | blocked
        h, w = dims(all_cells)
        if len(body) >= target and h >= PATCH_H_RANGE[0] and w >= PATCH_W_RANGE[0]:
            break
        cands = []
        for r, c in body:
            for dr, dc in DIRS4:
                q = (r + dr, c + dc)
                if q in body or q in blocked:
                    continue
                nh, nw = dims(all_cells | {q})
                if nh <= PATCH_H_RANGE[1] and nw <= PATCH_W_RANGE[1]:
                    cands.append(q)
        if not cands or len(body) >= BODY_MAX:
            break
        body.add(random.choice(cands))


def patch_nonbg_abs(patch: np.ndarray, top_left, bg: int):
    r0, c0 = top_left
    for pr, pc in np.argwhere(patch != bg):
        yield int(r0 + pr), int(c0 + pc), int(patch[pr, pc])


def find_edge_stripes(grid: np.ndarray, bg: int):
    H, W = grid.shape
    stripe = np.zeros((H, W), dtype=bool)
    colors = set()

    top_h = 0
    for r in range(H):
        row = grid[r, :]
        if (row == row[0]).all() and row[0] != bg:
            top_h += 1
        else:
            break
    if top_h:
        stripe[:top_h, :] = True
        colors.add(int(grid[0, 0]))

    bot_h = 0
    for r in range(H - 1, -1, -1):
        row = grid[r, :]
        if (row == row[0]).all() and row[0] != bg:
            bot_h += 1
        else:
            break
    if bot_h:
        stripe[H - bot_h :, :] = True
        colors.add(int(grid[H - 1, 0]))

    inner_r0, inner_r1 = top_h, H - bot_h
    left_w = 0
    for c in range(W):
        col = grid[inner_r0:inner_r1, c]
        if col.size == 0:
            break
        if (col == col[0]).all() and col[0] != bg:
            left_w += 1
        else:
            break
    if left_w:
        stripe[inner_r0:inner_r1, :left_w] = True
        colors.add(int(grid[inner_r0, 0]))

    right_w = 0
    for c in range(W - 1, -1, -1):
        col = grid[inner_r0:inner_r1, c]
        if col.size == 0:
            break
        if (col == col[0]).all() and col[0] != bg:
            right_w += 1
        else:
            break
    if right_w:
        stripe[inner_r0:inner_r1, W - right_w :] = True
        colors.add(int(grid[inner_r0, W - 1]))
    return stripe, colors, (top_h, bot_h, left_w, right_w)


def find_internal_stripes(grid: np.ndarray, bg: int, edge_info):
    H, W = grid.shape
    top_h, bot_h, left_w, right_w = edge_info
    r0, r1, c0, c1 = top_h, H - bot_h, left_w, W - right_w
    stripe = np.zeros((H, W), dtype=bool)
    colors = set()

    c = c0
    while c < c1:
        col = grid[r0:r1, c]
        if col.size and bool((col != bg).all()):
            c2 = c
            while c2 + 1 < c1:
                col2 = grid[r0:r1, c2 + 1]
                if col2.size and bool((col2 != bg).all()):
                    c2 += 1
                else:
                    break
            stripe[r0:r1, c : c2 + 1] = True
            vals = grid[r0:r1, c : c2 + 1].flatten().tolist()
            mc = Counter(int(v) for v in vals if int(v) != bg).most_common(1)[0][0]
            colors.add(int(mc))
            c = c2 + 1
        else:
            c += 1

    r = r0
    while r < r1:
        row = grid[r, c0:c1]
        if row.size and bool((row != bg).all()):
            r2 = r
            while r2 + 1 < r1:
                row2 = grid[r2 + 1, c0:c1]
                if row2.size and bool((row2 != bg).all()):
                    r2 += 1
                else:
                    break
            stripe[r : r2 + 1, c0:c1] = True
            vals = grid[r : r2 + 1, c0:c1].flatten().tolist()
            mc = Counter(int(v) for v in vals if int(v) != bg).most_common(1)[0][0]
            colors.add(int(mc))
            r = r2 + 1
        else:
            r += 1
    return stripe, colors


def regions_from_stripes(stripe_mask: np.ndarray):
    non = ~stripe_mask
    lab, n = label(non)
    regs = []
    for k in range(1, n + 1):
        coords = np.argwhere(lab == k)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (r1 - r0 + 1) >= 6 and (c1 - c0 + 1) >= 6:
            regs.append({"r0": int(r0), "r1": int(r1), "c0": int(c0), "c1": int(c1)})
    regs.sort(key=lambda reg: (reg["r0"], reg["c0"]))
    return regs


def boundary_dirs_for_region(grid: np.ndarray, reg, stripe_colors):
    H, W = grid.shape
    dirs = {}
    r0, r1, c0, c1 = reg["r0"], reg["r1"], reg["c0"], reg["c1"]
    if r0 - 1 >= 0:
        mc = Counter(grid[r0 - 1, c0 : c1 + 1].tolist()).most_common(1)[0][0]
        if mc in stripe_colors:
            dirs[int(mc)] = 0
    if r1 + 1 < H:
        mc = Counter(grid[r1 + 1, c0 : c1 + 1].tolist()).most_common(1)[0][0]
        if mc in stripe_colors:
            dirs[int(mc)] = 2
    if c0 - 1 >= 0:
        mc = Counter(grid[r0 : r1 + 1, c0 - 1].tolist()).most_common(1)[0][0]
        if mc in stripe_colors:
            dirs[int(mc)] = 3
    if c1 + 1 < W:
        mc = Counter(grid[r0 : r1 + 1, c1 + 1].tolist()).most_common(1)[0][0]
        if mc in stripe_colors:
            dirs[int(mc)] = 1
    return dirs


def detect_cavities(grid: np.ndarray, bg: int, stripe_colors, regions):
    cavities = []
    for reg in regions:
        dirs = boundary_dirs_for_region(grid, reg, stripe_colors)
        assert len(dirs) >= 4
        dir_to_color = {v: k for k, v in dirs.items()}
        found = []
        for r in range(reg["r0"] + 1, reg["r1"]):
            for c in range(reg["c0"] + 1, reg["c1"]):
                if int(grid[r, c]) != bg:
                    continue
                if (
                    int(grid[r - 1, c]) == dir_to_color[0]
                    and int(grid[r, c + 1]) == dir_to_color[1]
                    and int(grid[r + 1, c]) == dir_to_color[2]
                    and int(grid[r, c - 1]) == dir_to_color[3]
                ):
                    found.append((int(r), int(c)))
        if len(found) == 1:
            cavities.append(
                {
                    "cell": found[0],
                    "region": reg,
                    "divider_dirs": dirs,
                    "color_set": frozenset(dirs.keys()),
                }
            )
    return cavities


def extract_trains(grid: np.ndarray, bg: int, stripe_mask: np.ndarray, stripe_colors):
    comp_mask = (grid != bg) & (~stripe_mask)
    lab, n = label(comp_mask, structure=np.ones((3, 3), dtype=int))
    trains = []
    for k in range(1, n + 1):
        coords = np.argwhere(lab == k)
        vals = [int(grid[r, c]) for r, c in coords]
        if 8 not in vals:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        bbox_arr = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        abs_coords = [(int(r), int(c)) for r, c in coords]
        body9, marker_pixels = [], []
        for r, c in abs_coords:
            v = int(grid[r, c])
            if v == 9:
                body9.append((int(r - r0), int(c - c0)))
            if v in stripe_colors:
                marker_pixels.append((v, int(r - r0), int(c - c0)))
        trains.append(
            {
                "abs_coords": abs_coords,
                "bbox": bbox_arr,
                "body9": body9,
                "marker_pixels": marker_pixels,
                "marker_set": frozenset(v for v, _, _ in marker_pixels),
            }
        )
    return trains


def rank_ops(train, divider_dirs):
    bbox0 = train["bbox"]
    h0, w0 = bbox0.shape
    markers_by_color = {}
    for v, r, c in train["marker_pixels"]:
        markers_by_color.setdefault(v, []).append((r, c))
    assert set(markers_by_color) == set(divider_dirs)
    assert train["body9"]
    ref_r = sum(r for r, _ in train["body9"]) / len(train["body9"])
    ref_c = sum(c for _, c in train["body9"]) / len(train["body9"])
    ranked = []
    for flip_ud in (False, True):
        for k in range(4):
            ref_t = list(
                transform_coords([(int(round(ref_r)), int(round(ref_c)))], (h0, w0), flip_ud, k)
            )
            rr_ref, cc_ref = ref_t[0]
            score = 0
            for color, pts in markers_by_color.items():
                mr = sum(r for r, _ in pts) / len(pts)
                mc = sum(c for _, c in pts) / len(pts)
                m_t = list(
                    transform_coords([(int(round(mr)), int(round(mc)))], (h0, w0), flip_ud, k)
                )
                rr_m, cc_m = m_t[0]
                dr, dc = rr_m - rr_ref, cc_m - cc_ref
                facing = (0 if dr < 0 else 2) if abs(dr) >= abs(dc) else (3 if dc < 0 else 1)
                if facing == divider_dirs[color]:
                    score += 1
            ranked.append((score, (flip_ud, k)))
    ranked.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
    assert ranked and ranked[0][0] == len(divider_dirs)
    return ranked


def xform_coords(coords, shape, flip_ud, rot_k):
    result = transform_coords(coords, shape, flip_ud, rot_k)
    new_shape = apply_transform(np.zeros(shape, dtype=int), flip_ud, rot_k).shape
    return list(result), new_shape


def analyze_input(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    edge_mask, edge_colors, edge_info = find_edge_stripes(g, bg)
    internal_mask, internal_colors = find_internal_stripes(g, bg, edge_info)
    stripe_mask = edge_mask | internal_mask
    stripe_colors = edge_colors | internal_colors
    regions = regions_from_stripes(stripe_mask)
    cavities = detect_cavities(g, bg, stripe_colors, regions)
    trains = extract_trains(g, bg, stripe_mask, stripe_colors)
    return {
        "bg": bg,
        "stripe_mask": stripe_mask,
        "stripe_colors": stripe_colors,
        "regions": regions,
        "cavities": cavities,
        "trains": trains,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock train sprites into stripe-coded background cavities by matching boundary-color sets.

    Input structure:
    - Thick outer stripes plus one full horizontal and one full vertical stripe partition the board into rectangular regions.
    - Color 8 forms train bodies; color 9 marks train couplers only.
    - Each target station is a background hole inside a small local stripe cross, and each train carries marker pixels colored with stripe colors.

    Rule:
    - Detect regions from the global stripes, then find the cavity in each used region by locating bg cells whose four orthogonal neighbors reproduce that region's stripe directions.
    - Match each train to the unique cavity whose region has the same marker-color set as the train.
    - Rotate/flip the train so every marker faces the same-color regional boundary, erase the original train, then translate so its coupler lands in the cavity cell.

    Key insight:
    - The 9 count no longer identifies the destination because every train uses the same coupler size; the target is encoded by the set of boundary-stripe colors, while the marker geometry still fixes the orientation.
    """
    info = analyze_input(grid)
    g = np.array(grid, dtype=int)
    bg = info["bg"]
    stripe_mask = info["stripe_mask"]
    cavities = info["cavities"]
    trains = info["trains"]

    cavity_by_color_set = {}
    for cav in cavities:
        key = cav["color_set"]
        assert key not in cavity_by_color_set
        cavity_by_color_set[key] = cav

    out = g.copy()
    for t in trains:
        for r, c in t["abs_coords"]:
            out[r, c] = bg

    for t in trains:
        assert len(t["body9"]) == 1, "Generator guarantees a single coupler"
        cav = cavity_by_color_set.get(t["marker_set"])
        assert cav is not None, "Marker color set must identify exactly one cavity"
        reg = cav["region"]
        chosen = None
        hole_r, hole_c = cav["cell"]
        for score, op in rank_ops(t, cav["divider_dirs"]):
            if score < len(cav["divider_dirs"]):
                break
            tbbox = apply_transform(t["bbox"], *op)
            t_body9, (h1, w1) = xform_coords(t["body9"], t["bbox"].shape, *op)
            assert len(t_body9) == 1
            off_r = hole_r - t_body9[0][0]
            off_c = hole_c - t_body9[0][1]
            ok = True
            for r in range(h1):
                for c in range(w1):
                    v = int(tbbox[r, c])
                    if v == bg:
                        continue
                    rr, cc = r + off_r, c + off_c
                    if not (reg["r0"] <= rr <= reg["r1"] and reg["c0"] <= cc <= reg["c1"]):
                        ok = False
                        break
                    if stripe_mask[rr, cc]:
                        ok = False
                        break
                    if out[rr, cc] != bg:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                chosen = (tbbox, h1, w1, off_r, off_c)
                break
        assert chosen is not None, "Docked train must fit its target region and cavity"
        tbbox, h1, w1, off_r, off_c = chosen
        for r in range(h1):
            for c in range(w1):
                v = int(tbbox[r, c])
                if v != bg:
                    out[r + off_r, c + off_c] = v
    return out


def layout():
    for _ in range(LAYOUT_RETRIES):
        top = random.randint(*EDGE_THICK_RANGE)
        bottom = random.randint(*EDGE_THICK_RANGE)
        left = random.randint(*EDGE_THICK_RANGE)
        right = random.randint(*EDGE_THICK_RANGE)
        candidates = []
        for H in range(*GRID_H_RANGE):
            for W in range(*GRID_W_RANGE):
                if H * W < GRID_MIN_AREA:
                    continue
                inner_h = H - top - bottom
                inner_w = W - left - right
                need_h = MIN_REGION_DIM * 2 + INTERNAL_THICK
                need_w = MIN_REGION_DIM * 2 + INTERNAL_THICK
                if inner_h >= need_h and inner_w >= need_w:
                    candidates.append((H, W))
        if not candidates:
            continue
        H, W = random.choice(candidates)

        pool = COLOR_POOL[:]
        bg = random.choice(pool)
        pool.remove(bg)
        stripe_cols = random.sample(pool, 6)
        col_top, col_bottom, col_left, col_right, col_v, col_h = stripe_cols

        grid = np.full((H, W), bg, dtype=int)
        grid[:top, :] = col_top
        grid[H - bottom :, :] = col_bottom
        grid[top : H - bottom, :left] = col_left
        grid[top : H - bottom, W - right :] = col_right

        c_lo = left + MIN_REGION_DIM
        c_hi = W - right - MIN_REGION_DIM - INTERNAL_THICK
        r_lo = top + MIN_REGION_DIM
        r_hi = H - bottom - MIN_REGION_DIM - INTERNAL_THICK
        if c_lo > c_hi or r_lo > r_hi:
            continue
        split_c = random.randint(c_lo, c_hi)
        split_r = random.randint(r_lo, r_hi)
        grid[top : H - bottom, split_c : split_c + INTERNAL_THICK] = col_v
        grid[split_r : split_r + INTERNAL_THICK, left : W - right] = col_h

        regions = [
            {
                "r0": top,
                "r1": split_r - 1,
                "c0": left,
                "c1": split_c - 1,
                "dirs": {col_top: 0, col_v: 1, col_h: 2, col_left: 3},
            },
            {
                "r0": top,
                "r1": split_r - 1,
                "c0": split_c + INTERNAL_THICK,
                "c1": W - right - 1,
                "dirs": {col_top: 0, col_right: 1, col_h: 2, col_v: 3},
            },
            {
                "r0": split_r + INTERNAL_THICK,
                "r1": H - bottom - 1,
                "c0": left,
                "c1": split_c - 1,
                "dirs": {col_h: 0, col_v: 1, col_bottom: 2, col_left: 3},
            },
            {
                "r0": split_r + INTERNAL_THICK,
                "r1": H - bottom - 1,
                "c0": split_c + INTERNAL_THICK,
                "c1": W - right - 1,
                "dirs": {col_h: 0, col_right: 1, col_bottom: 2, col_v: 3},
            },
        ]
        if all(
            reg["r1"] - reg["r0"] + 1 >= MIN_REGION_DIM and reg["c1"] - reg["c0"] + 1 >= MIN_REGION_DIM
            for reg in regions
        ):
            return grid, bg, regions
    return None


def marker_cell(direction: int):
    d = random.choice([2, 3])
    side = random.choice([-1, 1])
    if direction == 0:
        return (-d, side)
    if direction == 2:
        return (d, side)
    if direction == 3:
        return (side, -d)
    return (side, d)


def inward_cell(marker, direction: int):
    dr, dc = DIRS4[(direction + 2) % 4]
    return marker[0] + dr, marker[1] + dc


def build_docked_patch(bg: int, dirs):
    coupler = (0, 0)
    forbidden = {(-1, 0), (1, 0), (0, -1), (0, 1)}
    markers = []
    marker_cells = set()
    for color, direction in dirs.items():
        p = marker_cell(direction)
        if p == coupler or p in forbidden or p in marker_cells:
            return None
        markers.append((p, color, direction))
        marker_cells.add(p)

    blocked = {coupler} | forbidden | marker_cells
    corners = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    random.shuffle(corners)
    body = {corners[0]}

    seed = blocked | body
    sr0, sc0, sr1, sc1 = bbox(seed)
    bounds = (sr0 - PATH_BOUNDS_PAD, sr1 + PATH_BOUNDS_PAD, sc0 - PATH_BOUNDS_PAD, sc1 + PATH_BOUNDS_PAD)

    for p, _, direction in markers:
        goal = inward_cell(p, direction)
        if goal in blocked:
            return None
        if not path_to(body, goal, blocked, bounds):
            return None

    if not any(max(abs(r), abs(c)) == 1 for r, c in body):
        body.add(random.choice(corners))
    if len(body) > BODY_MAX:
        return None

    target = random.randint(max(BODY_MIN, len(body)), BODY_MAX)
    grow_body(body, blocked, target)
    all_cells = body | blocked
    h, w = dims(all_cells)
    if not (BODY_MIN <= len(body) <= BODY_MAX and PATCH_H_RANGE[0] <= h <= PATCH_H_RANGE[1] and PATCH_W_RANGE[0] <= w <= PATCH_W_RANGE[1]):
        return None

    r0, c0, _, _ = bbox(all_cells)
    patch = np.full((h, w), bg, dtype=int)
    for r, c in body:
        patch[r - r0, c - c0] = 8
    for (r, c), color, _ in markers:
        patch[r - r0, c - c0] = color
    patch[coupler[0] - r0, coupler[1] - c0] = 9
    coupler_rel = (coupler[0] - r0, coupler[1] - c0)
    return patch, coupler_rel


def try_place_station(base: np.ndarray, occupied: np.ndarray, region, patch: np.ndarray, coupler_rel, bg: int):
    H, W = base.shape
    nonbg = [(int(r), int(c), int(patch[r, c])) for r, c in np.argwhere(patch != bg)]
    blocked = dilate(occupied)
    centers = [
        (r, c)
        for r in range(region["r0"] + 2, region["r1"] - 1)
        for c in range(region["c0"] + 2, region["c1"] - 1)
    ]
    random.shuffle(centers)
    dir_to_color = {v: k for k, v in region["dirs"].items()}
    for cr, cc in centers:
        cross = [
            (cr - 1, cc, dir_to_color[0]),
            (cr, cc + 1, dir_to_color[1]),
            (cr + 1, cc, dir_to_color[2]),
            (cr, cc - 1, dir_to_color[3]),
        ]
        if any(not (0 <= r < H and 0 <= c < W) for r, c, _ in cross):
            continue
        if any(blocked[r, c] for r, c, _ in cross):
            continue
        top_left = (cr - coupler_rel[0], cc - coupler_rel[1])
        ok = True
        cells = []
        for rr, cc2, v in patch_nonbg_abs(patch, top_left, bg):
            if not (region["r0"] <= rr <= region["r1"] and region["c0"] <= cc2 <= region["c1"]):
                ok = False
                break
            if blocked[rr, cc2]:
                ok = False
                break
            cells.append((rr, cc2, v))
        if not ok:
            continue
        return (cr, cc), cross, cells
    return None


def place_docked_trains(base: np.ndarray, bg: int, regions):
    output = base.copy()
    occupied = output != bg
    n_trains = random.choice(TRAIN_COUNTS)
    chosen_regions = random.sample(regions, n_trains)
    trains = []

    for reg in chosen_regions:
        placed = None
        for _ in range(PATCH_RETRIES):
            built = build_docked_patch(bg, reg["dirs"])
            if built is None:
                continue
            patch, coupler_rel = built
            attempt = try_place_station(base, occupied, reg, patch, coupler_rel, bg)
            if attempt is None:
                continue
            center, cross, cells = attempt
            for r, c, v in cross:
                base[r, c] = v
            for r, c, _ in cross:
                occupied[r, c] = True
            for r, c, v in cells:
                output[r, c] = v
                occupied[r, c] = True
            trains.append(
                {
                    "patch": patch,
                    "target_center": center,
                    "region": reg,
                    "cross": cross,
                }
            )
            placed = True
            break
        if not placed:
            return None
    return output, trains, base


def patch_locations(region, patch: np.ndarray, bg: int, blocked: np.ndarray, margin: int = 1):
    H, W = blocked.shape
    ph, pw = patch.shape
    coords = [(int(r), int(c)) for r, c in np.argwhere(patch != bg)]
    locs = []
    r_start = max(region["r0"] + margin, 0)
    r_stop = min(region["r1"] - margin - ph + 2, H - ph + 1)
    c_start = max(region["c0"] + margin, 0)
    c_stop = min(region["c1"] - margin - pw + 2, W - pw + 1)
    for rr in range(r_start, r_stop):
        for cc in range(c_start, c_stop):
            ok = True
            for pr, pc in coords:
                if blocked[rr + pr, cc + pc]:
                    ok = False
                    break
            if ok:
                locs.append((rr, cc))
    return locs


def scatter_trains(base: np.ndarray, bg: int, regions, trains):
    input_grid = base.copy()
    valid_ops = [OPS[:] for _ in trains]
    ops = [random.choice(v) for v in valid_ops]
    if all(op == (False, 0) for op in ops):
        ops[0] = random.choice([op for op in OPS if op != (False, 0)])

    blocked = dilate(base != bg)
    order = list(range(len(trains)))
    random.shuffle(order)
    chosen = [None] * len(trains)

    for idx in order:
        t = trains[idx]
        op = ops[idx]
        in_patch = inverse_patch(t["patch"], op)
        reg_order = regions[:]
        random.shuffle(reg_order)
        all_locs = []
        for reg in reg_order:
            locs = patch_locations(reg, in_patch, bg, blocked, margin=1)
            all_locs.extend(locs)
        if not all_locs:
            return None
        r0, c0 = random.choice(all_locs)
        for rr, cc, v in patch_nonbg_abs(in_patch, (r0, c0), bg):
            input_grid[rr, cc] = v
        patch_mask = np.zeros_like(blocked, dtype=bool)
        for rr, cc, _ in patch_nonbg_abs(in_patch, (r0, c0), bg):
            patch_mask[rr, cc] = True
        blocked |= dilate(patch_mask)
        chosen[idx] = (r0, c0)

    return input_grid


def gen_attempt():
    laid = layout()
    if laid is None:
        return None
    base, bg, regions = laid
    placed = place_docked_trains(base.copy(), bg, regions)
    if placed is None:
        return None
    _output, trains, station_base = placed
    input_grid = scatter_trains(station_base, bg, regions, trains)
    if input_grid is None:
        return None
    counts = Counter(map(int, input_grid.flatten()))
    if counts[bg] != max(counts.values()):
        return None
    try:
        solved = main(input_grid)
        metrics = variant_metrics(input_grid)
        if (
            solved is not None
            and metrics["train_count"] == metrics["cavity_count"] >= 2
            and metrics["all_cavities_same_size"]
            and metrics["count_matching_ambiguous"]
            and all(v == 1 for v in metrics["unique_color_matches"])
        ):
            return input_grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        g = gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("failed to generate valid fused 6e4f6532 input")


def variant_metrics(grid: np.ndarray):
    info = analyze_input(grid)
    cavity_count = len(info["cavities"])
    train_count = len(info["trains"])
    cavity_sizes = [1 for _ in info["cavities"]]
    unique_color_matches = []
    for train in info["trains"]:
        matches = sum(train["marker_set"] == cav["color_set"] for cav in info["cavities"])
        unique_color_matches.append(matches)
    return {
        "train_count": train_count,
        "cavity_count": cavity_count,
        "all_cavities_same_size": len(set(cavity_sizes)) == 1 if cavity_sizes else False,
        "count_matching_ambiguous": len(set(cavity_sizes)) < cavity_count if cavity_sizes else False,
        "unique_color_matches": unique_color_matches,
    }
