# Variant of 6e4f6532
# Dimension: D4 — target object: station structure (dst)
# Change: stations/couplers are no longer monochrome color-9 groups; each station
# is a connected multi-color signature, and each train carries the same signature.

from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import combinations
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg, paste


# Design analysis for the D4 variant
# a) Concretization and behavior trace:
#    The original puzzle used isolated color-9 clusters as stations and attached
#    color-9 train couplers, so train -> station matching was driven by coupler
#    count alone. Here, each station is an 8-connected cluster of non-stripe,
#    non-8 colors, and the attached train couplers use the same color multiset.
#    Example: two size-2 stations may both be dominoes, but one is {1,5} and the
#    other is {3,9}; both trains therefore have two couplers, and size-only
#    matching becomes ambiguous while color-signature matching resolves it.
# b) Complexity calibration:
#    To make the color logic visibly necessary, every sample includes duplicated
#    station sizes via patterns (2,2) or (2,2,4). All stations have at least two
#    cells, so the multi-color change is always visible. Signatures are unique but
#    drawn from a small palette, keeping the search space human-readable.
# c) Hidden assumptions from the old solver:
#    Any line that checked `grid == 9`, skipped `v == 9`, or allowed overlap with
#    existing `9`s depended on the old monochrome station invariant. Those spots
#    are replaced with structural checks:
#    - stations = isolated non-bg / non-8 / non-stripe components
#    - couplers = non-8 / non-stripe cells inside train components
#    - station overlap is validated by transformed coupler coordinates, not color 9
# d) Placement legality:
#    The generator stays output-first. It builds docked trains whose colored
#    couplers already sit on valid station cells inside a target region, then
#    inverse-transforms and scatters those full sprites. Station colors are chosen
#    from the non-body palette excluding bg and stripe colors, so solver parsing is
#    stable and no marker/station ambiguity is introduced.
# e) Visual observability:
#    Duplicate size-2 stations are present in every sample, so "count only" cannot
#    explain the examples. The repeated size class forces the solver to use the
#    visible station color signatures.
# f) Visual cleanliness:
#    Marker colors remain stripe colors and station colors never reuse stripe
#    colors, so trains still decompose cleanly into body / couplers / markers.
# g) Color inference:
#    Only a small number of station colors are used per sample, and signatures are
#    repeated across the coupled train and its destination station. The human cue is
#    the direct color-combination echo, not any hidden color-id arithmetic.
# 9) New solver assumptions and generator guarantees:
#    - Every station signature is unique across stations.
#    - No station color equals bg, 8, or any stripe color.
#    - Every train component contains at least one stripe-colored marker and at
#      least one non-stripe coupler cell.
#    - The transformed coupler shape is congruent to exactly one station component.


def apply_transform(arr: np.ndarray, flip_ud: bool = False, rot_k: int = 0) -> np.ndarray:
    """Positional-args helper (kept local; only used by this solver)."""
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def transform_coords(coords, shape, flip_ud: bool = False, rot_k: int = 0):
    """Positional-args helper (kept local; only used by this solver)."""
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    out = apply_transform(mask, flip_ud=flip_ud, rot_k=rot_k)
    return set(map(tuple, np.argwhere(out)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock train sprites onto multi-color station groups by matching coupler color signatures, then orient via divider-color markers.

    Input structure:
    - bg; colored stripes (edge + internal) partition grid into rectangular regions.
    - Color 8 forms train bodies; non-stripe, non-8 pixels encode either isolated stations or train-attached couplers.
    - Stripe-colored pixels on trains are orientation markers.

    Rule:
    - Stations = 8-connected non-stripe / non-8 clusters whose pixels have no 8-neighbor. Trains = components containing 8.
    - Match a train to the unique station whose coupler count and color multiset agree with the train's attached couplers.
    - Rotate/flip the train so marker pixels face the same-color boundary stripes of the target region, then translate so coupler cells coincide with the station cells.

    Key insight: duplicated station sizes remove count-only matching, so the visible multi-color coupler signature identifies the destination, while marker colors still determine orientation.
    """
    inp = grid
    grid = inp.copy()
    H, W = grid.shape
    bg = get_bg(grid)

    # This puzzle reconstructs several implicit structures:
    # - divider stripes (edge + internal) that partition the canvas into regions
    # - multi-color station groups = isolated non-stripe clusters with no 8-neighbor
    # - train sprites = components containing 8, with colored couplers and stripe markers
    # Then, for each train:
    # - match the target station by COUPLER COLOR SIGNATURE (not just count)
    # - choose orientation by marker-vs-stripe agreement
    # - translate so transformed couplers land exactly on the station cells

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

    def station_groups(stripe_colors):
        sm = np.zeros((H, W), dtype=bool)
        for r in range(H):
            for c in range(W):
                v = int(grid[r, c])
                if v == bg or v == 8 or v in stripe_colors:
                    continue
                ok = all(
                    grid[r + dr, c + dc] != 8
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if (dr != 0 or dc != 0) and 0 <= r + dr < H and 0 <= c + dc < W
                )
                if ok:
                    sm[r, c] = True
        struct = np.ones((3, 3), dtype=int)
        lab, n = label(sm, structure=struct)
        groups = []
        for k in range(1, n + 1):
            coords = [(int(r), int(c)) for r, c in np.argwhere(lab == k)]
            signature = tuple(sorted(int(grid[r, c]) for r, c in coords))
            groups.append({"coords": coords, "coord_set": set(coords), "signature": signature})
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

    def boundary_dirs_for_region(reg, stripe_colors):
        dirs = {}
        r0, r1, c0, c1 = reg["r0"], reg["r1"], reg["c0"], reg["c1"]
        if r0 - 1 >= 0:
            mc = Counter(grid[r0 - 1, c0:c1 + 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 0
        if r1 + 1 < H:
            mc = Counter(grid[r1 + 1, c0:c1 + 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 2
        if c0 - 1 >= 0:
            mc = Counter(grid[r0:r1 + 1, c0 - 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 3
        if c1 + 1 < W:
            mc = Counter(grid[r0:r1 + 1, c1 + 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 1
        return dirs

    def xform_coords(coords, shape, flip_ud, rot_k):
        result = transform_coords(coords, shape, flip_ud, rot_k)
        new_shape = apply_transform(np.zeros(shape), flip_ud, rot_k).shape
        return list(result), new_shape

    def choose_best_op(train, divider_dirs):
        bbox0 = train["bbox"]
        H0, W0 = bbox0.shape
        markers_by_color = defaultdict(list)
        for v, r, c in train["marker_pixels"]:
            if v in divider_dirs:
                markers_by_color[v].append((r, c))
        assert markers_by_color, "Each train must expose divider-colored markers"
        assert train["couplers"], "Each train must carry at least one colored coupler"
        ref_r = sum(r for r, _, _ in train["couplers"]) / len(train["couplers"])
        ref_c = sum(c for _, c, _ in train["couplers"]) / len(train["couplers"])
        best, best_score = (False, 0), -1
        for flip_ud in (False, True):
            for k in range(4):
                ref_t = list(
                    transform_coords([(int(round(ref_r)), int(round(ref_c)))], (H0, W0), flip_ud, k)
                )
                rr_ref, cc_ref = ref_t[0]
                score = 0
                for color, pts in markers_by_color.items():
                    mr = sum(r for r, _ in pts) / len(pts)
                    mc = sum(c for _, c in pts) / len(pts)
                    m_t = list(
                        transform_coords([(int(round(mr)), int(round(mc)))], (H0, W0), flip_ud, k)
                    )
                    rr_m, cc_m = m_t[0]
                    dr, dc = rr_m - rr_ref, cc_m - cc_ref
                    dirc = (0 if dr < 0 else 2) if abs(dr) >= abs(dc) else (3 if dc < 0 else 1)
                    if dirc == divider_dirs[color]:
                        score += 1
                if score > best_score:
                    best_score, best = score, (flip_ud, k)
        return best

    edge_mask, edge_colors, edge_info = find_edge_stripes()
    internal_mask, internal_colors = find_internal_stripes(edge_info)
    stripe_mask = edge_mask | internal_mask
    stripe_colors = edge_colors | internal_colors
    regions = regions_from_stripes(stripe_mask)

    groups = station_groups(stripe_colors)
    groups_by_signature = defaultdict(list)
    for g in groups:
        groups_by_signature[g["signature"]].append(g)

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
        bbox = grid[r0:r1 + 1, c0:c1 + 1].copy()
        abs_coords = [(int(r), int(c)) for r, c in coords]
        couplers, marker_pixels = [], []
        for r, c in abs_coords:
            v = int(grid[r, c])
            rr, cc = int(r - r0), int(c - c0)
            if v in stripe_colors:
                marker_pixels.append((v, rr, cc))
            elif v != 8:
                couplers.append((rr, cc, v))
        trains.append(
            {
                "abs_coords": abs_coords,
                "bbox": bbox,
                "couplers": couplers,
                "marker_pixels": marker_pixels,
            }
        )

    out = grid.copy()
    for t in trains:
        for r, c in t["abs_coords"]:
            out[r, c] = bg

    for t in trains:
        signature = tuple(sorted(color for _, _, color in t["couplers"]))
        assert signature, "Each train must carry at least one colored coupler"
        candidate_groups = groups_by_signature.get(signature)
        assert candidate_groups is not None and len(candidate_groups) == 1, (
            f"Expected one station group for coupler signature {signature}"
        )
        g = candidate_groups[0]
        gset = g["coord_set"]
        matching_regions = [
            rr
            for rr in regions
            if all(rr["r0"] <= r <= rr["r1"] and rr["c0"] <= c <= rr["c1"] for r, c in g["coords"])
        ]
        assert len(matching_regions) == 1, "Each station group must belong to exactly one region"
        reg = matching_regions[0]
        divider_dirs = boundary_dirs_for_region(reg, stripe_colors)
        op = choose_best_op(t, divider_dirs)
        tbbox = apply_transform(t["bbox"], *op)
        t_couplers, (H1, W1) = xform_coords(
            [(r, c) for r, c, _ in t["couplers"]], t["bbox"].shape, *op
        )
        coupler_set = set(t_couplers)
        matching_offsets = {
            (sr - br, sc - bc)
            for br, bc in coupler_set
            for sr, sc in gset
            if all((r + sr - br, c + sc - bc) in gset for r, c in coupler_set)
        }
        assert len(matching_offsets) == 1, "Couplers must align to a unique station-group offset"
        off_r, off_c = next(iter(matching_offsets))
        ok = True
        for r in range(H1):
            for c in range(W1):
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
                if (r, c) in coupler_set:
                    if (rr, cc) not in gset:
                        ok = False
                        break
                elif out[rr, cc] != bg:
                    ok = False
                    break
            if not ok:
                break
        assert ok, "Docked train must fit entirely within its target region"
        for r in range(H1):
            for c in range(W1):
                if int(tbbox[r, c]) != bg and (r, c) not in coupler_set:
                    out[r + off_r, c + off_c] = int(tbbox[r, c])
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

# Train / station patterns: duplicated size-2 stations make color matching necessary.
TRAIN_PATTERNS = [(2, 2), (2, 2, 4)]

# Docked-patch body constraints
BODY_MIN = 5
BODY_MAX = 20
PATCH_H_RANGE = (4, 9)
PATCH_W_RANGE = (3, 8)

# Marker placement
MARKER_DIST_RANGE = (2, 3)
MARKER_JITTER = [-1, 0, 1]
THREE_MARKER_PROB = 0.65

# Growth / pathfinding
BODY_GROWTH_ITERS = 160
PATH_BOUNDS_PAD = 4

# Retry budgets
LAYOUT_RETRIES = 200
DOCK_RETRIES = 20
GENERATE_RETRIES = 300

# Color pool excludes only the train-body color 8.
NON_BODY_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]

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
        pool = NON_BODY_COLORS[:]
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
    if size == 2:
        return {(0, 0), (0, 1)} if random.random() < 0.5 else {(0, 0), (1, 0)}
    if size == 4:
        return {(0, 0), (0, 1), (1, 0), (1, 1)}
    raise ValueError(f"unsupported station size: {size}")


def _choose_signatures(palette, sizes):
    chosen = []
    used = set()
    for size in sizes:
        candidates = [combo for combo in combinations(palette, size) if combo not in used]
        if not candidates:
            return None
        sig = random.choice(candidates)
        used.add(sig)
        chosen.append(sig)
    return chosen


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


def _build_docked_patch(bg, dirs, size, signature):
    coupler_cells = sorted(_station_shape(size))
    coupler_colors = random.sample(list(signature), len(signature))
    colored_couplers = list(zip(coupler_cells, coupler_colors))
    specs = _marker_specs(dirs)
    markers, marker_cells = [], set()
    for color, direction in specs:
        p = _marker_cell(coupler_cells, direction)
        if p in coupler_cells or p in marker_cells:
            return None
        markers.append((p, color, direction))
        marker_cells.add(p)

    blocked = set(coupler_cells) | marker_cells
    contacts = [
        q
        for r, c in coupler_cells
        for dr, dc in DIRS4
        if (q := (r + dr, c + dc)) not in blocked
    ]
    if not contacts:
        return None

    body = {random.choice(contacts)}
    all_seed = set(coupler_cells) | marker_cells | body
    sr0, sc0, sr1, sc1 = _bbox(all_seed)
    bounds = (sr0 - PATH_BOUNDS_PAD, sr1 + PATH_BOUNDS_PAD, sc0 - PATH_BOUNDS_PAD, sc1 + PATH_BOUNDS_PAD)

    for p, _, direction in markers:
        goal = _inward_cell(p, direction)
        if goal in blocked:
            return None
        if not _path_to(body, goal, blocked, bounds):
            return None

    if not _connect_couplers_to_body(coupler_cells, body, blocked, bounds):
        return None

    if len(body) > BODY_MAX:
        return None
    target = random.randint(max(BODY_MIN, len(body)), BODY_MAX)
    _grow_body(body, blocked, target)

    all_cells = body | set(coupler_cells) | marker_cells
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
    for (r, c), color in colored_couplers:
        patch[r - r0, c - c0] = color
        rel_couplers.append((r - r0, c - c0, color))
    return patch, rel_couplers


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
                for item in coupler_rel:
                    pr, pc = item[:2]
                    ar, ac = rr + pr, cc + pc
                    if not (region["r0"] + 2 <= ar <= region["r1"] - 2 and region["c0"] + 2 <= ac <= region["c1"] - 2):
                        ok = False
                        break
            if ok:
                locs.append((rr, cc))
    return locs


def _place_docked_trains(base, bg, regions):
    output = base.copy()
    occupied = output != bg

    sizes = list(random.choice(TRAIN_PATTERNS))
    random.shuffle(sizes)
    stripe_colors = {color for reg in regions for color in reg["dirs"]}
    palette = [c for c in NON_BODY_COLORS if c != bg and c not in stripe_colors]
    if len(palette) < 4:
        return None
    signatures = _choose_signatures(palette, sizes)
    if signatures is None:
        return None

    trains = []
    for size, signature in zip(sizes, signatures):
        for _ in range(DOCK_RETRIES):
            reg = random.choice(regions)
            result = _build_docked_patch(bg, reg["dirs"], size, signature)
            if result is None:
                continue
            patch, coupler_rel = result
            locs = _patch_locations(reg, patch, bg, _dilate(occupied), coupler_rel, True, 0)
            if not locs:
                continue
            r0, c0 = random.choice(locs)
            paste(output, patch, r0, c0, transparent=bg)
            occupied |= _dilate((output != bg) & ~occupied) | (output != bg)
            station_abs = [(r0 + r, c0 + c, color) for r, c, color in coupler_rel]
            trains.append({"patch": patch, "station": station_abs})
            break
        else:
            return None
    return output, trains


def _scatter_trains(base, bg, regions, trains):
    H, W = base.shape
    input_grid = base.copy()
    station_mask = np.zeros((H, W), dtype=bool)
    for t in trains:
        for r, c, color in t["station"]:
            input_grid[r, c] = color
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
    output, trains = placed

    input_grid = _scatter_trains(base, bg, regions, trains)
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
    raise RuntimeError("failed to generate valid 6e4f6532 input")
