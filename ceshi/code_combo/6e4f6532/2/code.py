# Combo variant of 6e4f6532 (index 2)
# combo_spec: D1/1+D2/0
# Name: 双重形状推理（形状匹配+形状定向）
#
# Design analysis
# a) Concrete behavior:
#    A typical sample now shows 2-3 isolated station clusters that all contain five color-9 cells,
#    but their silhouettes differ, e.g. one is an L-pentomino while another is a P-pentomino.
#    In the original task a train would first pick a station by `len(couplers)` and then use stripe-
#    colored markers to choose a rotation. In this combo, both steps are replaced by shape reasoning:
#    the train first identifies its destination by canonical coupler geometry, and then the exact
#    drawn orientation of that station determines the only legal flip/rotation for docking.
# b) Complexity calibration:
#    The shape family is intentionally small and stable: a handful of asymmetric 5-cell polyominoes.
#    Keeping the count bucket repeated while varying only the silhouettes makes the new rule visible
#    without exploding the number of cases a human must memorize from 2-4 examples.
# c) Hidden-assumption audit:
#    The original `groups_by_size[len(g)]` lookup is invalid because multiple stations now share the
#    same size. The original marker-extraction and stripe-direction scoring are also invalid because
#    trains no longer carry any stripe-colored pixels. The new solver therefore depends on three new
#    premises only: station canonical signatures are unique, at least one size bucket is repeated,
#    and each chosen station shape has eight distinct D4 orientations so exact orientation is unique.
# d) Placement legality:
#    Generation still works output-first. Stations stay isolated from 8-cells, docked trains fit
#    fully inside a region, scattered trains avoid stripes/stations/each other, and the oracle
#    `main(input) == output` is checked before accepting any sample.
# e) Visual observability:
#    Every accepted sample contains same-size stations with visibly different 9-shapes, so count-only
#    matching is visibly insufficient. Because the station itself shows the final coupler layout, the
#    orientation signal is also directly visible without any hidden metadata.
# f) Visual cleanliness:
#    Removing marker colors actually simplifies the scene: regions and trains are still readable, and
#    the only important contrast is 8-body versus 9-coupler/station geometry.
# g) Color inference:
#    No new color mapping is introduced. Stripe colors still partition space only; they no longer
#    carry per-train meaning, so the combo does not ask the solver to infer an arbitrary palette code.
# h) New solver assumptions and generator guarantees:
#    - Each station canonical shape appears at most once in a sample.
#    - At least one repeated count bucket exists; in practice all stations use five 9-cells.
#    - Each chosen shape has eight distinct D4 images, so the target station orientation yields one
#      and only one docking transform after the destination has been identified by canonical shape.

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg, paste


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
    """Positional-args helper (kept local; only used by this solver/generator)."""
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def transform_coords(coords, shape, flip_ud: bool = False, rot_k: int = 0):
    """Transform a coord set by rendering it into a boolean mask first."""
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    out = apply_transform(mask, flip_ud=flip_ud, rot_k=rot_k)
    return set(map(tuple, np.argwhere(out)))


def normalize_coords(coords):
    """Translate coords so their bbox starts at (0, 0), then sort them."""
    pts = [(int(r), int(c)) for r, c in coords]
    if not pts:
        return tuple()
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def canonical_shape_signature(coords):
    """Canonicalize a pixel shape up to translation, rotation, and vertical flip."""
    norm = normalize_coords(coords)
    if not norm:
        return tuple()
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    variants = []
    for flip_ud, rot_k in OPS:
        variants.append(normalize_coords(transform_coords(norm, (h, w), flip_ud, rot_k)))
    return min(variants)


def xform_coords(coords, shape, flip_ud, rot_k):
    """Transform coords, returning both the transformed list and the new canvas shape."""
    result = transform_coords(coords, shape, flip_ud, rot_k)
    new_shape = apply_transform(np.zeros(shape, dtype=int), flip_ud, rot_k).shape
    return list(result), new_shape


def shape_from_key(shape_key):
    return set(SHAPE_LIBRARY[shape_key])


def transform_shape(coords, op):
    norm = normalize_coords(coords)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return set(normalize_coords(transform_coords(norm, (h, w), *op)))


def bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def dims(coords):
    r0, c0, r1, c1 = bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def adjacent8(p, cells):
    r, c = p
    return any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in cells)


def dilate(mask):
    """8-connected dilation via numpy shifts."""
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
    """BFS from *body* to *goal*, adding the path into *body* (mutates)."""
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


def inverse_patch(patch, op):
    """Find the pre-image of *patch* under *op* by brute force over D4."""
    for inv in OPS:
        cand = apply_transform(patch, *inv)
        if np.array_equal(apply_transform(cand, *op), patch):
            return cand
    raise ValueError("no inverse transform")


def valid_ops_for_patch(patch):
    """Return ops whose inverse keeps the patch within bbox size limits."""
    hlo, hhi = PATCH_H_RANGE
    wlo, whi = PATCH_W_RANGE
    valid = []
    for op in OPS:
        h, w = inverse_patch(patch, op).shape
        if hlo <= h <= hhi and wlo <= w <= whi:
            valid.append(op)
    return valid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock train sprites onto station groups by matching coupler geometry, then orient via the station's exact shape.

    Input structure:
    - bg; colored stripes (edge + internal) partition the canvas into rectangular regions.
    - Color 8 forms train bodies; color 9 marks both isolated stations and train-attached couplers.
    - Stations may share the same 9-count, but each station silhouette is a different asymmetric polyomino.
    - Trains contain no stripe-colored marker pixels in this combo; only body + couplers matter.

    Rule:
    - Stations = 8-connected 9-clusters not adjacent to any 8. Trains = connected components containing 8.
    - Match each train to the unique station whose attached coupler 9-shape is congruent under D4.
    - Among the 8 possible flips/rotations, choose the single one whose transformed coupler shape has the same
      exact orientation as that target station cluster.
    - Translate so couplers land exactly on the station cells, erase the train from its source location, and redraw
      only the non-bg, non-9 train pixels in the docked position.

    Key insight: Shape does double duty. The canonical 9-shape identifies which station belongs to a train, and the
    station's asymmetric drawn orientation uniquely determines how the train must be rotated or flipped to dock.
    """
    grid = grid.copy()
    H, W = grid.shape
    bg = get_bg(grid)

    # Reconstruct the hidden scene graph:
    # - divider stripes partition the canvas into rectangular regions
    # - isolated 9-clusters are stations
    # - components containing 8 are trains with attached 9-couplers
    # Then dock each train by:
    # - matching its canonical coupler geometry to one station class
    # - finding the unique D4 operation whose transformed coupler shape equals that station's exact orientation
    # - translating so the couplers coincide with the station cells

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
                if grid[r, c] != 9:
                    continue
                ok = all(
                    grid[r + dr, c + dc] != 8
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if (dr != 0 or dc != 0)
                    and 0 <= r + dr < H
                    and 0 <= c + dc < W
                )
                if ok:
                    sm[r, c] = True
        struct = np.ones((3, 3), dtype=int)
        lab, n = label(sm, structure=struct)
        groups = []
        for k in range(1, n + 1):
            coords = np.argwhere(lab == k)
            groups.append([(int(r), int(c)) for r, c in coords])
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

    def choose_unique_op(train, target_group):
        target_exact = normalize_coords(target_group)
        matches = []
        for flip_ud, rot_k in OPS:
            t_body9, _ = xform_coords(train["body9"], train["bbox"].shape, flip_ud, rot_k)
            if normalize_coords(t_body9) == target_exact:
                matches.append((flip_ud, rot_k))
        assert len(matches) == 1, "Station shape must determine a unique orientation"
        return matches[0]

    edge_mask, _, edge_info = find_edge_stripes()
    internal_mask, _ = find_internal_stripes(edge_info)
    stripe_mask = edge_mask | internal_mask
    regions = regions_from_stripes(stripe_mask)

    groups = station_groups()
    groups_by_sig = defaultdict(list)
    for g in groups:
        groups_by_sig[canonical_shape_signature(g)].append(g)
    station_set = {pt for g in groups for pt in g}

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
        bbox_arr = grid[r0:r1 + 1, c0:c1 + 1].copy()
        abs_coords = [(int(r), int(c)) for r, c in coords]
        body9 = []
        for r, c in abs_coords:
            if int(grid[r, c]) == 9 and (r, c) not in station_set:
                body9.append((int(r - r0), int(c - c0)))
        trains.append(
            {
                "abs_coords": abs_coords,
                "bbox": bbox_arr,
                "body9": body9,
                "coupler_sig": canonical_shape_signature(body9),
            }
        )

    out = grid.copy()
    for t in trains:
        for r, c in t["abs_coords"]:
            out[r, c] = bg

    for t in trains:
        assert t["body9"], "Each train must carry at least one coupler"
        candidate_groups = groups_by_sig.get(t["coupler_sig"])
        assert candidate_groups is not None and len(candidate_groups) == 1, (
            "Expected one station group for each canonical coupler shape"
        )
        g = candidate_groups[0]
        gset = set(g)
        matching_regions = [
            reg
            for reg in regions
            if all(reg["r0"] <= r <= reg["r1"] and reg["c0"] <= c <= reg["c1"] for r, c in g)
        ]
        assert len(matching_regions) == 1, "Each station group must belong to exactly one region"
        reg = matching_regions[0]
        op = choose_unique_op(t, g)
        tbbox = apply_transform(t["bbox"], *op)
        t_body9, (H1, W1) = xform_coords(t["body9"], t["bbox"].shape, *op)
        body9_set = set(t_body9)
        br0 = min(r for r, _ in body9_set)
        bc0 = min(c for _, c in body9_set)
        sr0 = min(r for r, _ in gset)
        sc0 = min(c for _, c in gset)
        off_r, off_c = sr0 - br0, sc0 - bc0
        assert {(r + off_r, c + off_c) for r, c in body9_set} == gset, (
            "Couplers must align exactly onto the station shape"
        )
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
        assert ok, "Docked train must fit entirely within its target region"
        for r in range(H1):
            for c in range(W1):
                v = int(tbbox[r, c])
                if v != bg and v != 9:
                    out[r + off_r, c + off_c] = v
    return out


# --------------- generator ---------------

# Canvas / grid sizing
GRID_H_RANGE = (13, 31)
GRID_W_RANGE = (16, 29)
GRID_MIN_AREA = 300
MIN_REGION_DIM = 10
MIN_REGION_DIRS = 3
INTERNAL_STRIPE_THICK = 2

# Edge stripes
STRIPE_THICK_RANGE = (2, 4)

# Internal-stripe probability
INTERNAL_PROB = 0.62

# Train / station counts
TRAIN_COUNTS = [2, 3]
SHAPE_LIBRARY = {
    "L5": frozenset({(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)}),
    "P5": frozenset({(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)}),
    "N5": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (3, 0)}),
    "F5": frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 0)}),
}

# Docked-patch body constraints
BODY_MIN = 5
BODY_MAX = 20
PATCH_H_RANGE = (4, 9)
PATCH_W_RANGE = (3, 8)

# Growth / pathfinding
BODY_GROWTH_ITERS = 160
PATH_BOUNDS_PAD = 4

# Retry budgets
LAYOUT_RETRIES = 200
DOCK_RETRIES = 30
GENERATE_RETRIES = 400

# Color pool (excludes 8=train body, 9=coupler/station)
COLOR_POOL = list(range(8))


def assign_region_dirs(regions, edge_col, top, bottom, left, right, H, W, internal):
    """Populate each region's direction map; stripes remain visual separators only."""
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


def layout():
    """Build the background grid with edge/internal stripes and rectangular regions."""
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
        edge_col = dict(zip(edges, stripe_cols[:len(edges)]))
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
                {"r0": top, "r1": H - bottom - 1, "c0": c + INTERNAL_STRIPE_THICK, "c1": W - right - 1},
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
                {"r0": r + INTERNAL_STRIPE_THICK, "r1": H - bottom - 1, "c0": left, "c1": W - right - 1},
            ]
        else:
            regions = [inner]

        if assign_region_dirs(regions, edge_col, top, bottom, left, right, H, W, internal):
            return grid, bg, regions
    return None


def choose_station_shape_keys(n_trains):
    """
    All chosen shapes have the same count, so count-only matching is guaranteed to fail.
    """
    return random.sample(list(SHAPE_LIBRARY.keys()), n_trains)


def connect_couplers_to_body(couplers, body, blocked, bounds):
    """Ensure every coupler is 8-adjacent to the body via BFS paths."""
    for cp in couplers:
        if adjacent8(cp, body):
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
            if path_to(body, q, blocked, bounds):
                break
        else:
            return False
    return True


def grow_body(body, blocked, target):
    """Grow *body* toward *target* cells while respecting bbox limits."""
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
        h, w = dims(all_cells)
        if h < PATCH_H_RANGE[0]:
            r0, _, r1, _ = bbox(all_cells)
            exp = [q for q in cands if q[0] < r0 or q[0] > r1]
            cands = exp or cands
        elif w < PATCH_W_RANGE[0]:
            _, c0, _, c1 = bbox(all_cells)
            exp = [q for q in cands if q[1] < c0 or q[1] > c1]
            cands = exp or cands
        body.add(random.choice(cands))


def build_docked_patch(bg, shape_key):
    """Build a docked train patch: 8-body + asymmetric 9-couplers."""
    couplers = shape_from_key(shape_key)
    blocked = set(couplers)
    contacts = [
        q
        for r, c in couplers
        for dr, dc in DIRS4
        if (q := (r + dr, c + dc)) not in blocked
    ]
    if not contacts:
        return None

    body = {random.choice(contacts)}
    all_seed = set(couplers) | body
    sr0, sc0, sr1, sc1 = bbox(all_seed)
    bounds = (sr0 - PATH_BOUNDS_PAD, sr1 + PATH_BOUNDS_PAD, sc0 - PATH_BOUNDS_PAD, sc1 + PATH_BOUNDS_PAD)

    if not connect_couplers_to_body(couplers, body, blocked, bounds):
        return None
    if len(body) > BODY_MAX:
        return None

    target = random.randint(max(BODY_MIN, len(body)), BODY_MAX)
    grow_body(body, blocked, target)

    all_cells = body | set(couplers)
    h, w = dims(all_cells)
    if not (
        BODY_MIN <= len(body) <= BODY_MAX
        and PATCH_H_RANGE[0] <= h <= PATCH_H_RANGE[1]
        and PATCH_W_RANGE[0] <= w <= PATCH_W_RANGE[1]
    ):
        return None

    r0, c0, _, _ = bbox(all_cells)
    patch = np.full((h, w), bg, dtype=int)
    for r, c in body:
        patch[r - r0, c - c0] = 8
    rel_couplers = []
    for r, c in couplers:
        patch[r - r0, c - c0] = 9
        rel_couplers.append((r - r0, c - c0))
    return patch, rel_couplers


def patch_locations(region, patch, bg, occupied, coupler_rel=None, station_clearance=False, margin=0):
    """Find all valid top-left placements for *patch* within *region*."""
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


def place_docked_trains(base, bg, regions):
    """Place N docked trains on the output grid; return (output, trains) or None."""
    H, W = base.shape
    output = base.copy()
    occupied = output != bg

    n_trains = random.choice(TRAIN_COUNTS)
    shape_keys = choose_station_shape_keys(n_trains)
    trains = []
    seen_station_sigs = set()
    for shape_key in shape_keys:
        for _ in range(DOCK_RETRIES):
            reg = random.choice(regions)
            result = build_docked_patch(bg, shape_key)
            if result is None:
                continue
            patch, coupler_rel = result
            coupler_sig = canonical_shape_signature(coupler_rel)
            if coupler_sig in seen_station_sigs:
                continue
            locs = patch_locations(reg, patch, bg, dilate(occupied), coupler_rel, True, 0)
            if not locs:
                continue
            r0, c0 = random.choice(locs)
            paste(output, patch, r0, c0, transparent=bg)
            occupied |= dilate((output != bg) & ~occupied) | (output != bg)
            station_abs = [(r0 + r, c0 + c) for r, c in coupler_rel]
            trains.append(
                {
                    "patch": patch,
                    "station": station_abs,
                    "shape_key": shape_key,
                    "coupler_sig": coupler_sig,
                }
            )
            seen_station_sigs.add(coupler_sig)
            break
        else:
            return None
    return output, trains


def scatter_trains(base, bg, regions, trains):
    """Apply random transforms and scatter trains onto the input grid."""
    H, W = base.shape
    input_grid = base.copy()
    station_mask = np.zeros((H, W), dtype=bool)
    for t in trains:
        for r, c in t["station"]:
            input_grid[r, c] = 9
            station_mask[r, c] = True

    valid_ops = []
    for t in trains:
        valid = valid_ops_for_patch(t["patch"])
        if not valid:
            return None
        valid_ops.append(valid)
    ops = [random.choice(valid) for valid in valid_ops]
    if all(op == (False, 0) for op in ops):
        non_id = [op for op in valid_ops[0] if op != (False, 0)]
        if not non_id:
            return None
        ops[0] = random.choice(non_id)

    forbidden = dilate(base != bg) | dilate(station_mask)
    for t, op in zip(trains, ops):
        in_patch = inverse_patch(t["patch"], op)
        regs = regions[:]
        random.shuffle(regs)
        all_locs = []
        for reg in regs:
            locs = patch_locations(reg, in_patch, bg, forbidden, None, False, 1)
            all_locs.extend((reg, p) for p in locs)
        if not all_locs:
            return None
        _, (r0, c0) = random.choice(all_locs)
        paste(input_grid, in_patch, r0, c0, transparent=bg)
        ph, pw = in_patch.shape
        patch_mask = np.zeros((H, W), dtype=bool)
        patch_mask[r0:r0 + ph, c0:c0 + pw] = in_patch != bg
        forbidden |= dilate(patch_mask)

    return input_grid


def repeated_count_shape_metric(trains):
    """How many count buckets contain multiple distinct canonical station shapes?"""
    by_count = defaultdict(set)
    for t in trains:
        sig = t["coupler_sig"]
        by_count[len(sig)].add(sig)
    return sum(1 for sigs in by_count.values() if len(sigs) >= 2)


def gen_attempt():
    built = layout()
    if built is None:
        return None
    base, bg, regions = built

    placed = place_docked_trains(base, bg, regions)
    if placed is None:
        return None
    output, trains = placed
    if repeated_count_shape_metric(trains) < 1:
        return None

    input_grid = scatter_trains(base, bg, regions, trains)
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
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 6e4f6532 combo input")


for _shape_name, _shape in SHAPE_LIBRARY.items():
    assert len({normalize_coords(transform_shape(_shape, op)) for op in OPS}) == 8, (
        f"{_shape_name} must have 8 distinct D4 orientations"
    )
