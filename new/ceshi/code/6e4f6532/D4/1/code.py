from __future__ import annotations

import random
from collections import Counter

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg, paste

# Variant of 6e4f6532
# Dimension: D4 — target object: station structure (dst)
# Change: the train no longer carries stripe-colored orientation markers.
# Instead, each station keeps its 9-cell core and gains one adjacent indicator
# pixel. The indicator sits on the side opposite the required docking side, so
# the solver must read direction from the target object and orient the train by
# its unique body-to-coupler contact side.
#
# Design analysis
# a) Concrete behavior:
#    Matching still uses the original unique size mapping {1,2,4}, but
#    orientation is no longer source-coded. A size-2 train with the same coupler
#    shape can now only be docked by reading the destination's attached
#    indicator, which specifies the approach side.
# b) Complexity calibration:
#    The station indicator is a single pixel and the train body remains mostly
#    unchanged, so the visual change stays local to D4 while preserving the
#    original puzzle's region layout and docking mechanic.
# c) Old assumption audit:
#    The original solver assumed stripe-colored pixels inside each train and used
#    them to align with boundary stripes. Those assumptions are removed. The new
#    solver instead infers a train's current facing from the unique orthogonal
#    side where the 8-body touches its 9-couplers.
# d) Placement legality:
#    The generator still builds output first and inverse-transforms scattered
#    source trains. It now also reserves one indicator cell next to each station
#    and ensures the body touches the coupler core on exactly one side.
# e) Visual observability:
#    Every station carries an explicit indicator pixel, and source trains have no
#    stripe-colored orientation cue at all, so the direction transfer is visible.
# f) Visual cleanliness:
#    Indicators use one non-stripe, non-body, non-station color and remain
#    isolated from trains, keeping segmentation straightforward.
# g) Color inference:
#    The indicator color is not semantic; only its position relative to the
#    station core matters. This avoids introducing a new color-to-direction code.
# h) New solver assumptions and generator guarantees:
#    - Every station has exactly one adjacent indicator on a bbox side.
#    - The required docking side is the opposite of that indicator side.
#    - Each train's body touches its couplers on exactly one orthogonal side.
#    - At least one scattered train uses a non-identity transform.

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


def _bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _dims(coords):
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def _opposite_dir(direction: int) -> int:
    return (int(direction) + 2) % 4


def _contact_cell(couplers, direction: int):
    r0, c0, r1, c1 = _bbox(couplers)
    if direction == 0:
        cols = [c for r, c in couplers if r == r0]
        return (r0 - 1, random.choice(cols))
    if direction == 2:
        cols = [c for r, c in couplers if r == r1]
        return (r1 + 1, random.choice(cols))
    if direction == 3:
        rows = [r for r, c in couplers if c == c0]
        return (random.choice(rows), c0 - 1)
    rows = [r for r, c in couplers if c == c1]
    return (random.choice(rows), c1 + 1)


def _contact_side(couplers, body):
    body_set = set(body)
    dirs = set()
    for r, c in couplers:
        for direction, (dr, dc) in enumerate(DIRS4):
            if (r + dr, c + dc) in body_set:
                dirs.add(direction)
    return next(iter(dirs)) if len(dirs) == 1 else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock trains onto size-matched stations, orienting from target-side indicators.

    Input structure:
    - bg; colored stripes partition the grid into rectangular regions.
    - Color 8 forms train bodies; color 9 marks station cores and train couplers.
    - Each station also has one adjacent non-stripe indicator pixel.

    Rule:
    - Match trains to stations by coupler count as in the original task.
    - The indicator's side relative to the station core tells which side the
      train should dock from: the train approaches from the opposite side.
    - Rotate/flip the train until its unique body-to-coupler contact side matches
      that required docking side, then translate so couplers land on the station.
    """
    grid = np.asarray(grid, dtype=int)
    H, W = grid.shape
    bg = get_bg(grid)

    def find_edge_stripes():
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

    def find_internal_stripes(edge_info):
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

    def station_groups():
        sm = np.zeros((H, W), dtype=bool)
        for r in range(H):
            for c in range(W):
                if int(grid[r, c]) != 9:
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
            coords = [(int(r), int(c)) for r, c in np.argwhere(lab == k)]
            groups.append({"coords": coords, "coord_set": set(coords)})
        return groups

    def regions_from_stripes(stripe_mask):
        non = ~stripe_mask
        lab, n = label(non)
        regs = []
        for k in range(1, n + 1):
            coords = np.argwhere(lab == k)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            if (r1 - r0 + 1) >= 5 and (c1 - c0 + 1) >= 5:
                regs.append({"r0": int(r0), "r1": int(r1), "c0": int(c0), "c1": int(c1)})
        return regs

    def xform_coords(coords, shape, flip_ud, rot_k):
        result = transform_coords(coords, shape, flip_ud, rot_k)
        new_shape = apply_transform(np.zeros(shape), flip_ud, rot_k).shape
        return list(result), new_shape

    edge_mask, edge_colors, edge_info = find_edge_stripes()
    internal_mask, internal_colors = find_internal_stripes(edge_info)
    stripe_mask = edge_mask | internal_mask
    stripe_colors = edge_colors | internal_colors
    regions = regions_from_stripes(stripe_mask)

    groups = station_groups()
    groups_by_size = {}
    for g in groups:
        groups_by_size[len(g["coords"])] = g
    station_set = set(pt for g in groups for pt in g["coords"])

    def station_indicator_side(group):
        coords = group["coords"]
        r0, c0, r1, c1 = _bbox(coords)
        candidates = []
        probes = {
            0: [(r0 - 1, c) for c in range(c0, c1 + 1)],
            2: [(r1 + 1, c) for c in range(c0, c1 + 1)],
            3: [(r, c0 - 1) for r in range(r0, r1 + 1)],
            1: [(r, c1 + 1) for r in range(r0, r1 + 1)],
        }
        for side, cells in probes.items():
            for rr, cc in cells:
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                v = int(grid[rr, cc])
                if v in (bg, 8, 9) or stripe_mask[rr, cc]:
                    continue
                candidates.append((side, rr, cc))
        assert len(candidates) == 1, "Each station must expose exactly one indicator pixel"
        return int(candidates[0][0])

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
        body9 = []
        body8 = []
        for r, c in abs_coords:
            v = int(grid[r, c])
            rr, cc = int(r - r0), int(c - c0)
            if v == 9 and (r, c) not in station_set:
                body9.append((rr, cc))
            elif v == 8:
                body8.append((rr, cc))
        trains.append({"abs_coords": abs_coords, "bbox": bbox, "body9": body9, "body8": body8})

    out = grid.copy()
    for t in trains:
        for r, c in t["abs_coords"]:
            out[r, c] = bg

    for t in trains:
        n9 = len(t["body9"])
        assert n9 > 0, "Each train must carry at least one coupler"
        assert n9 in groups_by_size, f"Missing station for coupler count {n9}"
        g = groups_by_size[n9]
        gset = g["coord_set"]
        matching_regions = [
            reg
            for reg in regions
            if all(reg["r0"] <= r <= reg["r1"] and reg["c0"] <= c <= reg["c1"] for r, c in g["coords"])
        ]
        assert len(matching_regions) == 1, "Each station group must belong to exactly one region"
        reg = matching_regions[0]
        dock_dir = _opposite_dir(station_indicator_side(g))

        candidate_ops = []
        for op in OPS:
            t_body9 = list(transform_coords(t["body9"], t["bbox"].shape, *op))
            t_body8 = list(transform_coords(t["body8"], t["bbox"].shape, *op))
            if _contact_side(t_body9, t_body8) == dock_dir:
                candidate_ops.append(op)
        assert candidate_ops, "No orientation matches the target-side indicator"

        placed = False
        for op in candidate_ops:
            tbbox = apply_transform(t["bbox"], *op)
            t_body9, (H1, W1) = xform_coords(t["body9"], t["bbox"].shape, *op)
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
                    if stripe_mask[rr, cc] or out[rr, cc] != bg:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            for r in range(H1):
                for c in range(W1):
                    v = int(tbbox[r, c])
                    if v != bg and v != 9:
                        out[r + off_r, c + off_c] = v
            placed = True
            break
        assert placed, "Docked train must fit entirely within its target region"

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
BODY_GROWTH_ITERS = 160

LAYOUT_RETRIES = 200
DOCK_RETRIES = 20
GENERATE_RETRIES = 300


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
        pool = list(range(8))
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


def _grow_body(body, blocked, couplers, target):
    coupler_set = set(couplers)
    for _ in range(BODY_GROWTH_ITERS):
        occ = body | coupler_set
        h, w = _dims(occ)
        if len(body) >= target and h >= PATCH_H_RANGE[0] and w >= PATCH_W_RANGE[0]:
            break
        cands = []
        for r, c in body:
            for dr, dc in DIRS4:
                q = (r + dr, c + dc)
                if q in body or q in blocked:
                    continue
                nh, nw = _dims(occ | {q})
                if nh <= PATCH_H_RANGE[1] and nw <= PATCH_W_RANGE[1]:
                    cands.append(q)
        if not cands or len(body) >= BODY_MAX:
            break
        if h < PATCH_H_RANGE[0]:
            r0, _, r1, _ = _bbox(occ)
            exp = [q for q in cands if q[0] < r0 or q[0] > r1]
            cands = exp or cands
        elif w < PATCH_W_RANGE[0]:
            _, c0, _, c1 = _bbox(occ)
            exp = [q for q in cands if q[1] < c0 or q[1] > c1]
            cands = exp or cands
        body.add(random.choice(cands))


def _build_docked_patch(bg, size, dock_dir):
    couplers = _station_shape(size)
    anchor = _contact_cell(couplers, dock_dir)
    blocked = set(couplers)
    for r, c in couplers:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr != 0 or dc != 0:
                    blocked.add((r + dr, c + dc))
    blocked.discard(anchor)

    body = {anchor}
    target = random.randint(BODY_MIN, BODY_MAX)
    _grow_body(body, blocked, couplers, target)

    occ = body | set(couplers)
    h, w = _dims(occ)
    if not (
        BODY_MIN <= len(body) <= BODY_MAX
        and PATCH_H_RANGE[0] <= h <= PATCH_H_RANGE[1]
        and PATCH_W_RANGE[0] <= w <= PATCH_W_RANGE[1]
    ):
        return None

    r0, c0, _, _ = _bbox(occ)
    patch = np.full((h, w), bg, dtype=int)
    body_rel = []
    coupler_rel = []
    for r, c in body:
        rr, cc = r - r0, c - c0
        patch[rr, cc] = 8
        body_rel.append((rr, cc))
    for r, c in couplers:
        rr, cc = r - r0, c - c0
        patch[rr, cc] = 9
        coupler_rel.append((rr, cc))
    if _contact_side(coupler_rel, body_rel) != dock_dir:
        return None
    return patch, coupler_rel


def _indicator_rel(coupler_rel, dock_dir):
    return _contact_cell(set(coupler_rel), _opposite_dir(dock_dir))


def _patch_locations(region, patch, bg, occupied, coupler_rel=None, indicator_rel=None, margin=0):
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
            if coupler_rel:
                for pr, pc in coupler_rel:
                    ar, ac = rr + pr, cc + pc
                    if not (
                        region["r0"] + 2 <= ar <= region["r1"] - 2
                        and region["c0"] + 2 <= ac <= region["c1"] - 2
                    ):
                        ok = False
                        break
            if ok and indicator_rel is not None:
                ir, ic = indicator_rel
                ar, ac = rr + ir, cc + ic
                if not (
                    0 <= ar < H
                    and 0 <= ac < W
                    and region["r0"] <= ar <= region["r1"]
                    and region["c0"] <= ac <= region["c1"]
                    and not occupied[ar, ac]
                ):
                    ok = False
            if ok:
                locs.append((rr, cc))
    return locs


def _place_docked_trains(base, bg, regions, arrow_color):
    output = base.copy()
    occupied = output != bg
    n_trains = random.choice(TRAIN_COUNTS)
    sizes = random.sample(STATION_SIZES, n_trains)
    trains = []
    for size in sizes:
        for _ in range(DOCK_RETRIES):
            reg = random.choice(regions)
            dock_dir = random.choice(list(reg["dirs"].values()) or [0, 1, 2, 3])
            result = _build_docked_patch(bg, size, dock_dir)
            if result is None:
                continue
            patch, coupler_rel = result
            indicator_rel = _indicator_rel(coupler_rel, dock_dir)
            locs = _patch_locations(reg, patch, bg, _dilate(occupied), coupler_rel, indicator_rel, 0)
            if not locs:
                continue
            r0, c0 = random.choice(locs)
            paste(output, patch, r0, c0, transparent=bg)
            ir, ic = indicator_rel
            output[r0 + ir, c0 + ic] = arrow_color
            occupied = output != bg
            trains.append(
                {
                    "patch": patch,
                    "station": [(r0 + r, c0 + c) for r, c in coupler_rel],
                    "indicator": (r0 + ir, c0 + ic),
                }
            )
            break
        else:
            return None
    return output, trains


def _scatter_trains(base, bg, regions, trains, arrow_color):
    H, W = base.shape
    input_grid = base.copy()
    station_mask = np.zeros((H, W), dtype=bool)
    indicator_mask = np.zeros((H, W), dtype=bool)
    for t in trains:
        for r, c in t["station"]:
            input_grid[r, c] = 9
            station_mask[r, c] = True
        ir, ic = t["indicator"]
        input_grid[ir, ic] = arrow_color
        indicator_mask[ir, ic] = True

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

    forbidden = _dilate(base != bg) | _dilate(station_mask) | _dilate(indicator_mask)
    for t, op in zip(trains, ops):
        in_patch = _inverse_patch(t["patch"], op)
        regs = regions[:]
        random.shuffle(regs)
        all_locs = []
        for reg in regs:
            locs = _patch_locations(reg, in_patch, bg, forbidden, None, None, 1)
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
    stripe_colors = set(int(v) for v in np.unique(base) if int(v) != bg)
    arrow_candidates = [c for c in range(8) if c != bg and c not in stripe_colors]
    if not arrow_candidates:
        return None
    arrow_color = random.choice(arrow_candidates)

    placed = _place_docked_trains(base, bg, regions, arrow_color)
    if placed is None:
        return None
    output, trains = placed

    input_grid = _scatter_trains(base, bg, regions, trains, arrow_color)
    if input_grid is None:
        return None

    counts = Counter(map(int, input_grid.flatten()))
    if counts[bg] != max(counts.values()):
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
    raise RuntimeError("failed to generate valid 6e4f6532 D4-alt1 input")
