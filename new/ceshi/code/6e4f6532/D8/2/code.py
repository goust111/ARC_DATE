# Variant of 6e4f6532
# Dimension: D8 — 源-目标基数关系 (meta)
# Change: 1:1 严格匹配——每个火车唯一对应一个车站（通过耦合器数量唯一确定），2-3组配对... → 有干扰项的匹配

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg, paste


# Design analysis for the D8 variant:
# a) Concrete behavior:
#    Real objects still follow the original task: a movable train is an 8-containing
#    component with attached 9 couplers, and a real station is an isolated 9-cluster
#    whose cardinality matches that train's coupler count. The variant adds two
#    kinds of clutter that remain unchanged from input to output:
#    - a fake train-like component made of 8 plus stripe-colored markers but no 9
#    - a fake station-like isolated 9-cluster of size 3 or 5
#    In the original logic "contains 8" was enough to treat a component as a train,
#    and every isolated 9-cluster was expected to be a real target. Here that would
#    wrongly erase the fake train or try to match the fake station, so the solver
#    must filter before docking.
# b) Complexity calibration:
#    The clutter is visually obvious but not overwhelming: exactly one fake train and
#    one fake station are injected. Their shapes resemble the real object families,
#    but only the real trains keep attached 9 couplers and only the real stations use
#    cardinalities from {1,2,4}. This preserves the original reasoning while making
#    the distractor step necessary and observable.
# c) Hidden assumptions review:
#    - "all 8-components are trains" no longer holds, so zero-coupler 8-components
#      must be ignored instead of asserted on.
#    - "all isolated 9-clusters participate in a unique size match" no longer holds,
#      so unmatched station sizes must be tolerated.
#    - "clear every 8-component before redocking" would destroy fake trains, so only
#      matched movable trains are erased.
# d) Placement legality:
#    The generator still uses output-first construction for the real trains, then adds
#    static decoys in locations that avoid stripes, real trains, real stations, and
#    future scatter placements via occupied-mask checks and dilation-based clearance.
# e) Visual observability:
#    In every sample, one extra 8-object stays fixed and one extra isolated 9-cluster
#    stays fixed. Across 2-4 examples, a human can see that only 8+9 coupled objects
#    move and that odd-sized isolated 9-clusters are ignored.
# f) Visual cleanliness:
#    Static decoys are kept separated from other non-background objects by one-cell
#    dilation, and fake stations also keep the same two-cell stripe clearance as real
#    stations, so the regions stay readable.
# g) Color inference:
#    No new color mapping is introduced. Marker colors still come directly from region
#    boundary stripe colors, and fake trains reuse the same marker mechanism so the
#    clutter looks task-consistent rather than introducing arbitrary colors.


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


def apply_transform(arr: np.ndarray, flip_ud: bool = False, rot_k: int = 0) -> np.ndarray:
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def transform_coords(coords, shape, flip_ud: bool = False, rot_k: int = 0):
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    out = apply_transform(mask, flip_ud=flip_ud, rot_k=rot_k)
    return set(map(tuple, np.argwhere(out)))


def _find_edge_stripes(grid: np.ndarray, bg: int):
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


def _find_internal_stripes(grid: np.ndarray, bg: int, edge_info):
    H, W = grid.shape
    top_h, bot_h, left_w, right_w = edge_info
    r0, r1, c0, c1 = top_h, H - bot_h, left_w, W - right_w
    stripe = np.zeros((H, W), dtype=bool)
    colors = set()

    c = c0
    while c < c1:
        col = grid[r0:r1, c]
        if col.size and (col == col[0]).all() and col[0] != bg:
            color = int(col[0])
            c2 = c
            while c2 + 1 < c1:
                col2 = grid[r0:r1, c2 + 1]
                if col2.size and (col2 == col2[0]).all() and int(col2[0]) == color:
                    c2 += 1
                else:
                    break
            stripe[r0:r1, c : c2 + 1] = True
            colors.add(color)
            c = c2 + 1
        else:
            c += 1

    r = r0
    while r < r1:
        row = grid[r, c0:c1]
        if row.size and (row == row[0]).all() and row[0] != bg:
            color = int(row[0])
            r2 = r
            while r2 + 1 < r1:
                row2 = grid[r2 + 1, c0:c1]
                if row2.size and (row2 == row2[0]).all() and int(row2[0]) == color:
                    r2 += 1
                else:
                    break
            stripe[r : r2 + 1, c0:c1] = True
            colors.add(color)
            r = r2 + 1
        else:
            r += 1

    return stripe, colors


def _station_groups(grid: np.ndarray):
    H, W = grid.shape
    sm = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 9:
                continue
            ok = all(
                grid[r + dr, c + dc] != 8
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr != 0 or dc != 0) and 0 <= r + dr < H and 0 <= c + dc < W
            )
            if ok:
                sm[r, c] = True
    lab, n = label(sm, structure=np.ones((3, 3), dtype=int))
    groups = []
    for k in range(1, n + 1):
        coords = np.argwhere(lab == k)
        groups.append([(int(r), int(c)) for r, c in coords])
    return groups


def _regions_from_stripes(stripe_mask: np.ndarray):
    lab, n = label(~stripe_mask)
    regs = []
    for k in range(1, n + 1):
        coords = np.argwhere(lab == k)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (r1 - r0 + 1) >= 5 and (c1 - c0 + 1) >= 5:
            regs.append({"r0": int(r0), "r1": int(r1), "c0": int(c0), "c1": int(c1)})
    return regs


def _boundary_dirs_for_region(grid: np.ndarray, reg, stripe_colors):
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


def _xform_coords(coords, shape, flip_ud, rot_k):
    result = transform_coords(coords, shape, flip_ud, rot_k)
    new_shape = apply_transform(np.zeros(shape, dtype=int), flip_ud, rot_k).shape
    return list(result), new_shape


def _marker_score(train, divider_dirs, op):
    bbox0 = train["bbox"]
    H0, W0 = bbox0.shape
    markers_by_color = defaultdict(list)
    for v, r, c in train["marker_pixels"]:
        if v in divider_dirs:
            markers_by_color[v].append((r, c))
    if not markers_by_color or not train["body9"]:
        return -1

    ref_r = sum(r for r, _ in train["body9"]) / len(train["body9"])
    ref_c = sum(c for _, c in train["body9"]) / len(train["body9"])
    ref_t = list(
        transform_coords([(int(round(ref_r)), int(round(ref_c)))], (H0, W0), op[0], op[1])
    )
    rr_ref, cc_ref = ref_t[0]
    score = 0
    for color, pts in markers_by_color.items():
        mr = sum(r for r, _ in pts) / len(pts)
        mc = sum(c for _, c in pts) / len(pts)
        m_t = list(
            transform_coords([(int(round(mr)), int(round(mc)))], (H0, W0), op[0], op[1])
        )
        rr_m, cc_m = m_t[0]
        dr, dc = rr_m - rr_ref, cc_m - cc_ref
        dirc = (0 if dr < 0 else 2) if abs(dr) >= abs(dc) else (3 if dc < 0 else 1)
        if dirc == divider_dirs[color]:
            score += 1
    return score


def _choose_best_op(train, divider_dirs):
    best, best_score = (False, 0), -1
    for flip_ud in (False, True):
        for k in range(4):
            score = _marker_score(train, divider_dirs, (flip_ud, k))
            if score > best_score:
                best_score, best = score, (flip_ud, k)
    return best, best_score


def _extract_train_components(grid: np.ndarray, bg: int, stripe_mask: np.ndarray, station_set):
    comp_mask = (grid != bg) & (~stripe_mask)
    lab, n = label(comp_mask)
    trains = []
    for k in range(1, n + 1):
        coords = np.argwhere(lab == k)
        vals = [int(grid[r, c]) for r, c in coords]
        if 8 not in vals:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        bbox = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        abs_coords = [(int(r), int(c)) for r, c in coords]
        body9, marker_pixels = [], []
        for r, c in abs_coords:
            v = int(grid[r, c])
            if v == 9 and (r, c) not in station_set:
                body9.append((int(r - r0), int(c - c0)))
            if v not in (bg, 8, 9):
                marker_pixels.append((v, int(r - r0), int(c - c0)))
        trains.append(
            {
                "abs_coords": abs_coords,
                "bbox": bbox,
                "body9": body9,
                "marker_pixels": marker_pixels,
            }
        )
    return trains


def analyze_variant(grid: np.ndarray) -> dict[str, int]:
    bg = get_bg(grid)
    edge_mask, edge_colors, edge_info = _find_edge_stripes(grid, bg)
    internal_mask, internal_colors = _find_internal_stripes(grid, bg, edge_info)
    stripe_mask = edge_mask | internal_mask
    stripe_colors = edge_colors | internal_colors
    regions = _regions_from_stripes(stripe_mask)
    groups = _station_groups(grid)
    station_set = {pt for g in groups for pt in g}
    trains = _extract_train_components(grid, bg, stripe_mask, station_set)

    groups_by_size = defaultdict(list)
    for g in groups:
        groups_by_size[len(g)].append(g)

    movable = 0
    for t in trains:
        n9 = len(t["body9"])
        if n9 == 0:
            continue
        candidates = groups_by_size.get(n9, [])
        if len(candidates) != 1:
            continue
        g = candidates[0]
        regs = [
            reg
            for reg in regions
            if all(reg["r0"] <= r <= reg["r1"] and reg["c0"] <= c <= reg["c1"] for r, c in g)
        ]
        if len(regs) != 1:
            continue
        _, score = _choose_best_op(t, _boundary_dirs_for_region(grid, regs[0], stripe_colors))
        if score > 0:
            movable += 1

    valid_sizes = {len(t["body9"]) for t in trains if len(t["body9"]) > 0}
    fake_stations = sum(1 for g in groups if len(g) not in valid_sizes)
    fake_trains = sum(1 for t in trains if len(t["body9"]) == 0)
    return {
        "station_groups": len(groups),
        "real_trains": movable,
        "fake_trains": fake_trains,
        "fake_stations": fake_stations,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move only the real train sprites onto matching station groups, while leaving decoys untouched.

    Input structure:
    - The canvas is partitioned by colored border/internal stripes into 1-2 rectangular regions.
    - Color 8 is the train body, color 9 serves both as true stations and as real train couplers.
    - Some extra clutter is present: an 8-component with no couplers and an isolated 9-cluster whose size does not belong to any real match.

    Rule:
    - Treat isolated 9-clusters as candidate stations and 8-components as candidate trains, but first discard distractors.
    - A movable train must carry attached non-station 9 couplers; a real target must be the unique isolated 9-cluster with the same coupler count.
    - Rotate/flip that train so its stripe-colored markers face the matching boundary stripes of the target region, erase it from its source position, and dock it so couplers coincide with the target station cells.

    Key insight:
    - The original size-matching rule still solves the real objects, but the new step is to filter noise first: 8-objects without couplers and isolated 9-clusters with unmatched sizes are static decoys and should be ignored.
    """
    inp = grid
    grid = inp.copy()
    H, W = grid.shape
    bg = get_bg(grid)

    edge_mask, edge_colors, edge_info = _find_edge_stripes(grid, bg)
    internal_mask, internal_colors = _find_internal_stripes(grid, bg, edge_info)
    stripe_mask = edge_mask | internal_mask
    stripe_colors = edge_colors | internal_colors
    regions = _regions_from_stripes(stripe_mask)

    groups = _station_groups(grid)
    groups_by_size = defaultdict(list)
    for g in groups:
        groups_by_size[len(g)].append(g)
    station_set = {pt for g in groups for pt in g}

    train_components = _extract_train_components(grid, bg, stripe_mask, station_set)
    movable = []
    for t in train_components:
        n9 = len(t["body9"])
        if n9 == 0:
            continue
        candidate_groups = groups_by_size.get(n9, [])
        if len(candidate_groups) != 1:
            continue
        g = candidate_groups[0]
        gset = set(g)
        matching_regions = [
            reg
            for reg in regions
            if all(reg["r0"] <= r <= reg["r1"] and reg["c0"] <= c <= reg["c1"] for r, c in g)
        ]
        if len(matching_regions) != 1:
            continue
        reg = matching_regions[0]
        divider_dirs = _boundary_dirs_for_region(grid, reg, stripe_colors)
        op, score = _choose_best_op(t, divider_dirs)
        if score <= 0:
            continue
        t_body9, _ = _xform_coords(t["body9"], t["bbox"].shape, *op)
        body9_set = set(t_body9)
        matching_offsets = {
            (sr - br, sc - bc)
            for br, bc in body9_set
            for sr, sc in gset
            if all((r + sr - br, c + sc - bc) in gset for r, c in body9_set)
        }
        if len(matching_offsets) != 1:
            continue
        off_r, off_c = next(iter(matching_offsets))
        movable.append({"train": t, "op": op, "reg": reg, "offset": (off_r, off_c)})

    out = grid.copy()
    for item in movable:
        for r, c in item["train"]["abs_coords"]:
            out[r, c] = bg

    for item in movable:
        train = item["train"]
        reg = item["reg"]
        off_r, off_c = item["offset"]
        tbbox = apply_transform(train["bbox"], *item["op"])
        H1, W1 = tbbox.shape
        ok = True
        for r in range(H1):
            for c in range(W1):
                v = int(tbbox[r, c])
                if v == bg or v == 9:
                    continue
                rr, cc = r + off_r, c + off_c
                if not (reg["r0"] <= rr <= reg["r1"] and reg["c0"] <= cc <= reg["c1"]):
                    ok = False
                    break
                if stripe_mask[rr, cc] or (out[rr, cc] != bg and out[rr, cc] != 9):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            raise AssertionError("Docked train must fit entirely within its target region")
        for r in range(H1):
            for c in range(W1):
                v = int(tbbox[r, c])
                if v != bg and v != 9:
                    out[r + off_r, c + off_c] = v
    return out


# --------------- generator ---------------

GRID_H_RANGE = (13, 31)
GRID_W_RANGE = (16, 29)
GRID_MIN_AREA = 300
MIN_REGION_DIM = 10
MIN_REGION_DIRS = 3
INTERNAL_STRIPE_THICK = 2

STRIPE_THICK_RANGE = (2, 4)
INTERNAL_PROB = 0.62

TRAIN_COUNTS = [2, 3]
STATION_SIZES = [1, 2, 4]

BODY_MIN = 5
BODY_MAX = 20
PATCH_H_RANGE = (4, 9)
PATCH_W_RANGE = (3, 8)

MARKER_DIST_RANGE = (2, 3)
MARKER_JITTER = [-1, 0, 1]
THREE_MARKER_PROB = 0.65

BODY_GROWTH_ITERS = 160
PATH_BOUNDS_PAD = 4

LAYOUT_RETRIES = 200
DOCK_RETRIES = 20
STATIC_DECOY_RETRIES = 40
GENERATE_RETRIES = 300

DECOY_STATION_COUNT = 1
DECOY_TRAIN_COUNT = 1

COLOR_POOL = list(range(8))

DECOY_STATION_SHAPES = [
    {(0, 0), (0, 1), (1, 0)},
    {(0, 0), (0, 1), (0, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)},
]


def _apply_transform(arr, flip_ud=False, rot_k=0):
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def _inverse_patch(patch, op):
    for inv in OPS:
        cand = _apply_transform(patch, *inv)
        if np.array_equal(_apply_transform(cand, *op), patch):
            return cand
    raise ValueError("no inverse transform")


def _valid_ops_for_patch(patch):
    hlo, hhi = PATCH_H_RANGE
    wlo, whi = PATCH_W_RANGE
    valid = []
    for op in OPS:
        h, w = _inverse_patch(patch, op).shape
        if hlo <= h <= hhi and wlo <= w <= whi:
            valid.append(op)
    return valid


def _bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _dims(coords):
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def _adjacent8(p, cells):
    r, c = p
    return any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in cells)


def _dilate(mask):
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


def _path_to(body, goal, blocked, bounds):
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


def _assign_region_dirs(regions, edge_col, top, bottom, left, right, H, W, internal):
    for reg in regions:
        dirs = {}
        if top and reg["r0"] == top:
            dirs[edge_col["top"]] = 0
        if bottom and reg["r1"] == H - bottom - 1:
            dirs[edge_col["bottom"]] = 2
        if left and reg["c0"] == left:
            dirs[edge_col["left"]] = 3
        if right and reg["c1"] == W - right - 1:
            dirs[edge_col["right"]] = 1
        if internal:
            kind, pos, col = internal
            if kind == "v":
                dirs[col] = 1 if reg["c1"] == pos - 1 else 3
            else:
                dirs[col] = 2 if reg["r1"] == pos - 1 else 0
        reg["dirs"] = dirs
        rh = reg["r1"] - reg["r0"] + 1
        rw = reg["c1"] - reg["c0"] + 1
        if rh < MIN_REGION_DIM or rw < MIN_REGION_DIM or len(dirs) < MIN_REGION_DIRS:
            return False
    return True


def _layout():
    for _ in range(LAYOUT_RETRIES):
        use_internal = random.random() < INTERNAL_PROB
        if use_internal:
            orient = random.choice(["v", "h"])
            if orient == "v":
                edges = ["left", "right", random.choice(["top", "bottom"])]
            else:
                edges = ["top", "bottom", random.choice(["left", "right"])]
        else:
            orient = None
            edges = random.sample(["top", "bottom", "left", "right"], random.choice([3, 4]))

        thick = {e: random.randint(*STRIPE_THICK_RANGE) for e in edges}
        top, bottom = thick.get("top", 0), thick.get("bottom", 0)
        left, right = thick.get("left", 0), thick.get("right", 0)

        min_inner_for_split = MIN_REGION_DIM * 2 + INTERNAL_STRIPE_THICK
        candidates = []
        for H in range(*GRID_H_RANGE):
            for W in range(*GRID_W_RANGE):
                if H * W < GRID_MIN_AREA:
                    continue
                ih, iw = H - top - bottom, W - left - right
                if ih < MIN_REGION_DIM or iw < MIN_REGION_DIM:
                    continue
                if orient == "v" and iw < min_inner_for_split:
                    continue
                if orient == "h" and ih < min_inner_for_split:
                    continue
                candidates.append((H, W))
        if not candidates:
            continue

        H, W = random.choice(candidates)
        pool = COLOR_POOL[:]
        bg = random.choice(pool)
        pool.remove(bg)
        stripe_cols = random.sample(pool, len(edges) + int(use_internal))
        edge_col = dict(zip(edges, stripe_cols[: len(edges)]))
        internal_col = stripe_cols[-1] if use_internal else None
        grid = np.full((H, W), bg, dtype=int)

        if top:
            grid[:top, :] = edge_col["top"]
        if bottom:
            grid[H - bottom :, :] = edge_col["bottom"]
        if left:
            grid[top : H - bottom, :left] = edge_col["left"]
        if right:
            grid[top : H - bottom, W - right :] = edge_col["right"]

        inner = {"r0": top, "r1": H - bottom - 1, "c0": left, "c1": W - right - 1}
        regions, internal = [], None
        if orient == "v":
            lo = left + MIN_REGION_DIM
            hi = W - right - MIN_REGION_DIM - INTERNAL_STRIPE_THICK
            if lo > hi:
                continue
            c = random.randint(lo, hi)
            grid[top : H - bottom, c : c + INTERNAL_STRIPE_THICK] = internal_col
            internal = ("v", c, internal_col)
            regions = [
                {"r0": top, "r1": H - bottom - 1, "c0": left, "c1": c - 1},
                {
                    "r0": top,
                    "r1": H - bottom - 1,
                    "c0": c + INTERNAL_STRIPE_THICK,
                    "c1": W - right - 1,
                },
            ]
        elif orient == "h":
            lo = top + MIN_REGION_DIM
            hi = H - bottom - MIN_REGION_DIM - INTERNAL_STRIPE_THICK
            if lo > hi:
                continue
            r = random.randint(lo, hi)
            grid[r : r + INTERNAL_STRIPE_THICK, left : W - right] = internal_col
            internal = ("h", r, internal_col)
            regions = [
                {"r0": top, "r1": r - 1, "c0": left, "c1": W - right - 1},
                {
                    "r0": r + INTERNAL_STRIPE_THICK,
                    "r1": H - bottom - 1,
                    "c0": left,
                    "c1": W - right - 1,
                },
            ]
        else:
            regions = [inner]

        if _assign_region_dirs(regions, edge_col, top, bottom, left, right, H, W, internal):
            return grid, bg, regions
    return None


def _station_shape(size):
    if size == 1:
        return {(0, 0)}
    if size == 2:
        return {(0, 0), (0, 1)} if random.random() < 0.5 else {(0, 0), (1, 0)}
    return {(0, 0), (0, 1), (1, 0), (1, 1)}


def _marker_specs(dirs):
    items = list(dirs.items())
    if len(items) >= 3 and random.random() < THREE_MARKER_PROB:
        return random.sample(items, 3)
    pairs = []
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            if (a[1] - b[1]) % 4 in (1, 3):
                pairs.append((a, b))
    return list(random.choice(pairs)) if pairs else random.sample(items, 2)


def _marker_cell(couplers, direction):
    r0, c0, r1, c1 = _bbox(couplers)
    d = random.randint(*MARKER_DIST_RANGE)
    jitter = random.choice(MARKER_JITTER)
    if direction == 0:
        return (r0 - d, random.randint(c0, c1) + jitter)
    if direction == 2:
        return (r1 + d, random.randint(c0, c1) + jitter)
    if direction == 3:
        return (random.randint(r0, r1) + jitter, c0 - d)
    return (random.randint(r0, r1) + jitter, c1 + d)


def _inward_cell(marker, direction):
    dr, dc = DIRS4[(direction + 2) % 4]
    return marker[0] + dr, marker[1] + dc


def _connect_couplers_to_body(couplers, body, blocked, bounds):
    for cp in couplers:
        if _adjacent8(cp, body):
            continue
        cr, cc = cp
        choices = [
            (cr + dr, cc + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc) and (cr + dr, cc + dc) not in blocked
        ]
        random.shuffle(choices)
        for q in choices:
            if _path_to(body, q, blocked, bounds):
                break
        else:
            return False
    return True


def _grow_body(body, blocked, target):
    for _ in range(BODY_GROWTH_ITERS):
        all_cells = body | blocked
        h, w = _dims(all_cells)
        if len(body) >= target and h >= PATCH_H_RANGE[0] and w >= PATCH_W_RANGE[0]:
            break
        cands = []
        for r, c in body:
            for dr, dc in DIRS4:
                q = (r + dr, c + dc)
                if q in body or q in blocked:
                    continue
                nh, nw = _dims(all_cells | {q})
                if nh <= PATCH_H_RANGE[1] and nw <= PATCH_W_RANGE[1]:
                    cands.append(q)
        if not cands or len(body) >= BODY_MAX:
            break
        h, w = _dims(all_cells)
        if h < PATCH_H_RANGE[0]:
            r0, _, r1, _ = _bbox(all_cells)
            exp = [q for q in cands if q[0] < r0 or q[0] > r1]
            cands = exp or cands
        elif w < PATCH_W_RANGE[0]:
            _, c0, _, c1 = _bbox(all_cells)
            exp = [q for q in cands if q[1] < c0 or q[1] > c1]
            cands = exp or cands
        body.add(random.choice(cands))


def _build_docked_patch(bg, dirs, size):
    couplers = _station_shape(size)
    specs = _marker_specs(dirs)
    markers, marker_cells = [], set()
    for color, direction in specs:
        p = _marker_cell(couplers, direction)
        if p in couplers or p in marker_cells:
            return None
        markers.append((p, color, direction))
        marker_cells.add(p)

    blocked = set(couplers) | marker_cells
    contacts = [q for r, c in couplers for dr, dc in DIRS4 if (q := (r + dr, c + dc)) not in blocked]
    if not contacts:
        return None

    body = {random.choice(contacts)}
    all_seed = set(couplers) | marker_cells | body
    sr0, sc0, sr1, sc1 = _bbox(all_seed)
    bounds = (sr0 - PATH_BOUNDS_PAD, sr1 + PATH_BOUNDS_PAD, sc0 - PATH_BOUNDS_PAD, sc1 + PATH_BOUNDS_PAD)

    for p, _, direction in markers:
        goal = _inward_cell(p, direction)
        if goal in blocked:
            return None
        if not _path_to(body, goal, blocked, bounds):
            return None

    if not _connect_couplers_to_body(couplers, body, blocked, bounds):
        return None

    if len(body) > BODY_MAX:
        return None
    target = random.randint(max(BODY_MIN, len(body)), BODY_MAX)
    _grow_body(body, blocked, target)

    all_cells = body | set(couplers) | marker_cells
    h, w = _dims(all_cells)
    if not (
        BODY_MIN <= len(body) <= BODY_MAX
        and PATCH_H_RANGE[0] <= h <= PATCH_H_RANGE[1]
        and PATCH_W_RANGE[0] <= w <= PATCH_W_RANGE[1]
    ):
        return None

    r0, c0, _, _ = _bbox(all_cells)
    patch = np.full((h, w), bg, dtype=int)
    for r, c in body:
        patch[r - r0, c - c0] = 8
    for (r, c), color, _ in markers:
        patch[r - r0, c - c0] = color
    rel_couplers = []
    for r, c in couplers:
        patch[r - r0, c - c0] = 9
        rel_couplers.append((r - r0, c - c0))
    return patch, rel_couplers


def _render_shape(cells, color, bg):
    r0, c0, r1, c1 = _bbox(cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    rel = []
    for r, c in cells:
        patch[r - r0, c - c0] = color
        rel.append((r - r0, c - c0))
    return patch, rel


def _build_decoy_train_patch(bg, dirs):
    for _ in range(DOCK_RETRIES):
        result = _build_docked_patch(bg, dirs, random.choice(STATION_SIZES))
        if result is None:
            continue
        patch, _ = result
        decoy = patch.copy()
        decoy[decoy == 9] = 8
        return decoy
    return None


def _patch_locations(region, patch, bg, occupied, coupler_rel=None, station_clearance=False, margin=0):
    H, W = occupied.shape
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
                if occupied[rr + pr, cc + pc]:
                    ok = False
                    break
            if not ok:
                continue
            if station_clearance and coupler_rel:
                for pr, pc in coupler_rel:
                    ar, ac = rr + pr, cc + pc
                    if not (
                        region["r0"] + 2 <= ar <= region["r1"] - 2
                        and region["c0"] + 2 <= ac <= region["c1"] - 2
                    ):
                        ok = False
                        break
            if ok:
                locs.append((rr, cc))
    return locs


def _place_docked_trains(base, bg, regions):
    output = base.copy()
    occupied = output != bg

    n_trains = random.choice(TRAIN_COUNTS)
    sizes = random.sample(STATION_SIZES, n_trains)
    trains = []
    for size in sizes:
        for _ in range(DOCK_RETRIES):
            reg = random.choice(regions)
            result = _build_docked_patch(bg, reg["dirs"], size)
            if result is None:
                continue
            patch, coupler_rel = result
            locs = _patch_locations(reg, patch, bg, _dilate(occupied), coupler_rel, True, 0)
            if not locs:
                continue
            r0, c0 = random.choice(locs)
            paste(output, patch, r0, c0, transparent=bg)
            occupied = output != bg
            station_abs = [(r0 + r, c0 + c) for r, c in coupler_rel]
            trains.append({"patch": patch, "station": station_abs})
            break
        else:
            return None
    return output, trains


def _place_static_decoys(base, output_real, bg, regions):
    decor = base.copy()
    output = output_real.copy()
    occupied = output != bg

    for _ in range(DECOY_STATION_COUNT):
        placed = False
        for _ in range(STATIC_DECOY_RETRIES):
            reg = random.choice(regions)
            cells = random.choice(DECOY_STATION_SHAPES)
            patch, rel = _render_shape(cells, 9, bg)
            locs = _patch_locations(reg, patch, bg, _dilate(occupied), rel, True, 0)
            if not locs:
                continue
            r0, c0 = random.choice(locs)
            paste(decor, patch, r0, c0, transparent=bg)
            paste(output, patch, r0, c0, transparent=bg)
            occupied = output != bg
            placed = True
            break
        if not placed:
            return None

    for _ in range(DECOY_TRAIN_COUNT):
        placed = False
        for _ in range(STATIC_DECOY_RETRIES):
            reg = random.choice(regions)
            patch = _build_decoy_train_patch(bg, reg["dirs"])
            if patch is None:
                continue
            locs = _patch_locations(reg, patch, bg, _dilate(occupied), None, False, 1)
            if not locs:
                continue
            r0, c0 = random.choice(locs)
            paste(decor, patch, r0, c0, transparent=bg)
            paste(output, patch, r0, c0, transparent=bg)
            occupied = output != bg
            placed = True
            break
        if not placed:
            return None

    return decor, output


def _scatter_trains(base, bg, regions, trains):
    H, W = base.shape
    input_grid = base.copy()
    station_mask = np.zeros((H, W), dtype=bool)
    for t in trains:
        for r, c in t["station"]:
            input_grid[r, c] = 9
            station_mask[r, c] = True

    valid_ops = []
    for t in trains:
        valid = _valid_ops_for_patch(t["patch"])
        if not valid:
            return None
        valid_ops.append(valid)
    ops = [random.choice(valid) for valid in valid_ops]
    if all(op == (False, 0) for op in ops):
        non_id = [op for op in valid_ops[0] if op != (False, 0)]
        if not non_id:
            return None
        ops[0] = random.choice(non_id)

    forbidden = _dilate(base != bg) | _dilate(station_mask)
    for t, op in zip(trains, ops):
        in_patch = _inverse_patch(t["patch"], op)
        regs = regions[:]
        random.shuffle(regs)
        all_locs = []
        for reg in regs:
            locs = _patch_locations(reg, in_patch, bg, forbidden, None, False, 1)
            all_locs.extend((reg, p) for p in locs)
        if not all_locs:
            return None
        _, (r0, c0) = random.choice(all_locs)
        paste(input_grid, in_patch, r0, c0, transparent=bg)
        ph, pw = in_patch.shape
        patch_mask = np.zeros((H, W), dtype=bool)
        patch_mask[r0 : r0 + ph, c0 : c0 + pw] = in_patch != bg
        forbidden |= _dilate(patch_mask)

    return input_grid


def _gen_attempt():
    layout = _layout()
    if layout is None:
        return None
    base, bg, regions = layout

    placed = _place_docked_trains(base, bg, regions)
    if placed is None:
        return None
    output_real, trains = placed

    decoyed = _place_static_decoys(base, output_real, bg, regions)
    if decoyed is None:
        return None
    decor, output = decoyed

    input_grid = _scatter_trains(decor, bg, regions, trains)
    if input_grid is None:
        return None

    counts = Counter(map(int, input_grid.flatten()))
    if counts[bg] != max(counts.values()):
        return None

    metrics = analyze_variant(input_grid)
    if metrics["fake_trains"] < 1 or metrics["fake_stations"] < 1:
        return None

    try:
        if np.array_equal(main(input_grid), output):
            return input_grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 6e4f6532 D8 variant input")
