# Combo variant of 6e4f6532 (index 22)
# combo_spec: D2/0+D17/2
# Name: 车站形状定向+混合连通性
#
# Design analysis for the combo variant
# ====================================
# Reference-task note:
# - This fusion starts from the original 6e4f6532 layout-and-docking task, keeps
#   the output-first generation strategy, removes divider-color marker logic per
#   D2/0, and replaces the original uniform connectivity assumptions with the
#   D17/2 mixed regime:
#   body-8 components use 4-connectivity, coupler-9 attachment to a body uses
#   8-neighborhood, and isolated station-9 clustering also uses 4-connectivity.
#
# a) Concrete behavior and trace
# - Original behavior:
#   a train is a single non-stripe connected component containing color 8, so
#   attached 9 couplers are discovered by ordinary component labeling. Stations
#   are also 8-connected 9-clusters. Marker colors then choose the train's
#   rotation/flip.
# - Fused behavior:
#   stations are asymmetric 9-polyominoes with sizes 4/5/6 and no train markers
#   exist at all. A train is now reconstructed in two stages:
#   1. label only color-8 pixels with 4-connectivity to get each body;
#   2. attach every non-station 9 that touches exactly one body in the
#      8-neighborhood.
#   Meanwhile station pixels are collected as 9s with no adjacent 8, then split
#   with 4-connectivity. The generator forces the two stations to touch only
#   diagonally, so an 8-connected station read would merge them into one larger
#   blob and destroy the target matching. Orientation is then determined solely
#   by the unique D4 transform whose transformed coupler shape matches the
#   target station shape.
# - Small concrete trace:
#   if two station L-shapes meet at one corner, 4-connectivity yields two
#   separate shapes of sizes 4 and 5; 8-connectivity incorrectly yields one
#   size-9 aggregate. Separately, if a train coupler sits only on a diagonal of
#   the body, 4-neighbor attachment would miss it, but 8-neighbor attachment
#   keeps it part of the train. The fused rule therefore requires both changes.
#
# b) Complexity calibration
# - The generator always emits exactly two trains and two stations. This keeps
#   the diagonal-touching station cue legible and gives repeated evidence for
#   the mixed-connectivity interpretation instead of hiding it among clutter.
# - Station sizes stay in the small set {4,5,6}; each chosen size is unique in a
#   sample, preserving the original "count selects target" stage while making
#   the asymmetric orientation cue visible.
# - Every sample enforces two observable cues:
#   1. the two stations are 4-connected components but a single 8-connected
#      aggregate;
#   2. each train has at least one coupler that is diagonally adjacent to the
#      body and has no orthogonal body neighbor.
#
# c) Hidden-assumption audit against the original solver
# - Original assumption: "train = one labeled non-stripe component containing 8".
#   Broken, because diagonal-only couplers would be disconnected under the
#   default 4-connected component labeling. New solver labels only 8-bodies and
#   attaches couplers afterward via 8-neighborhood.
# - Original assumption: station groups can be labeled with 8-connectivity.
#   Broken, because the generator now makes the two stations touch diagonally on
#   purpose. New solver uses a 4-connected station structuring element.
# - Original assumption: orientation comes from colored markers. Removed. New
#   solver enumerates all 8 dihedral transforms and requires exactly one whose
#   transformed coupler shape equals the target station shape.
# - New assumptions introduced by the fused solver:
#   station sizes are unique, every non-station 9 touches exactly one body in
#   the 8-neighborhood, and each asymmetric station shape has a unique D4 match.
#   The generator enforces all three.
#
# d) Placement legality
# - Generation still works output-first: choose station shapes/positions first,
#   build docked train patches around them, then scatter inverse-transformed
#   trains into input regions.
# - Station pair placement requires:
#   no overlap, no 4-neighbor contact, at least one diagonal contact, and region
#   clearance so the docked body can grow without crossing stripes.
# - Scattered trains keep a one-cell 8-neighborhood buffer from stripes,
#   stations, and one another so couplers cannot be ambiguously assigned.
#
# e) Visual observability
# - A human can see two isolated 9-shapes that visually "kiss" at a corner but
#   do not share an edge; this supports the 4-connected station reading.
# - A human can also see train couplers that touch the body diagonally; this
#   makes the 8-neighbor coupler/body rule visually recoverable.
# - Because there are no markers, the only orientation cue left is the station's
#   own asymmetric shape, which is consistent across examples.
#
# f) Visual cleanliness
# - Limiting the scene to two trains keeps the diagonal-touching station pair
#   readable and prevents the final docked output from becoming crowded.
# - Body growth is bounded to compact patches, and station clearance is stronger
#   than in the source task so trains do not smear into stripes or each other.
#
# g) Change inferability
# - No rule depends on numeric color ordering. Color 8 is always body, color 9 is
#   always station/coupler, and all remaining colors are inert stripe/layout
#   scaffolding.
# - The fused behavior is inferable from a few examples because each sample shows
#   the same paired clues: 4-vs-8 station clustering changes the target shapes,
#   and 8-neighbor coupler attachment changes what counts as part of a train.
#
# Post-change assumption audit
# - `main()` assumes exactly two 4-connected stations with distinct sizes drawn
#   from {4,5,6}; the generator enforces that.
# - `main()` assumes every train body is a 4-connected 8-component and every
#   non-station coupler touches exactly one such body in the 8-neighborhood; the
#   generator enforces spacing so ambiguity cannot occur.
# - `main()` assumes the chosen station shape has exactly one D4 match for the
#   train couplers; the generator uses chiral base shapes with 8 distinct D4
#   images and validates by round-tripping through `main()`.

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg, paste


FOUR_CONN = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
EIGHT_CONN = np.ones((3, 3), dtype=int)
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
DIAGS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
STATION_BASE_SHAPES = {
    4: {(0, 0), (1, 0), (2, 0), (2, 1)},
    5: {(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)},
    6: {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1)},
}


def apply_transform(arr: np.ndarray, flip_ud: bool = False, rot_k: int = 0) -> np.ndarray:
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def transform_coords(coords, shape, flip_ud: bool = False, rot_k: int = 0):
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    out = apply_transform(mask, flip_ud=flip_ud, rot_k=rot_k)
    return set(map(tuple, np.argwhere(out)))


def normalize_coords(coords) -> frozenset[tuple[int, int]]:
    rs = [int(r) for r, _ in coords]
    cs = [int(c) for _, c in coords]
    r0, c0 = min(rs), min(cs)
    return frozenset((int(r) - r0, int(c) - c0) for r, c in coords)


def coords_bbox(coords):
    rs = [int(r) for r, _ in coords]
    cs = [int(c) for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def matching_offset(src_coords, dst_coords):
    src_set = set(src_coords)
    dst_set = set(dst_coords)
    offsets = {
        (dr - sr, dc - sc)
        for sr, sc in src_set
        for dr, dc in dst_set
        if all((r + dr - sr, c + dc - sc) in dst_set for r, c in src_set)
    }
    assert len(offsets) == 1, "Expected a unique alignment offset"
    return next(iter(offsets))


def cheb_adjacent(p, cells) -> bool:
    r, c = p
    return any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in cells)


def orth_adjacent(p, cells) -> bool:
    r, c = p
    return any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in cells)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock train sprites onto isolated station groups by mixing connectivity rules and orienting only by station-shape asymmetry.

    Input structure:
    - A background color fills the grid, while colored border/divider stripes partition the canvas into rectangular regions.
    - Color 8 forms train bodies and must be split with 4-connectivity; color 9 marks both isolated stations and train-attached couplers.
    - Stations are asymmetric 9-polyominoes that may touch other stations diagonally, and train couplers may touch a body only diagonally.

    Rule:
    - Identify station pixels as 9s with no adjacent 8 in the full 8-neighborhood, then cluster those station pixels with 4-connectivity.
    - Identify each train body as a 4-connected color-8 component, then attach every remaining 9 that touches exactly one body in the 8-neighborhood.
    - Match a train to the unique station of the same size, rotate/flip it so the transformed coupler shape exactly matches that station shape, then translate it so the couplers land on the station cells.
    - Erase the scattered trains from their original positions and redraw only the transformed body pixels in the docked location.

    Key insight: The target shapes appear only if station 9s are split by 4-connectivity, while each train is reconstructed only if couplers can attach diagonally to a 4-connected body. Once those mixed-connectivity objects are recovered, the asymmetric station shape alone fixes the orientation.
    """
    grid = grid.copy()
    H, W = grid.shape
    bg = get_bg(grid)

    # This solver reconstructs several implicit structures:
    # - divider stripes (edge + optional internal) that define rectangular regions
    # - stations = isolated 9-pixels, but clustered with 4-connectivity only
    # - trains = 4-connected 8-bodies plus 9-couplers attached by 8-neighborhood
    # For each train, the solver then:
    # - matches station by COUPLER COUNT
    # - chooses the unique D4 transform whose coupler shape matches the station
    # - translates the transformed train so couplers land exactly on the station

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
                if all(
                    grid[r + dr, c + dc] != 8
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if (dr != 0 or dc != 0) and 0 <= r + dr < H and 0 <= c + dc < W
                ):
                    sm[r, c] = True
        lab, n = label(sm, structure=FOUR_CONN)
        groups = []
        for k in range(1, n + 1):
            coords = np.argwhere(lab == k)
            groups.append([(int(r), int(c)) for r, c in coords])
        return groups

    def regions_from_stripes(stripe_mask):
        non = ~stripe_mask
        lab, n = label(non, structure=FOUR_CONN)
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
        new_shape = apply_transform(np.zeros(shape, dtype=bool), flip_ud, rot_k).shape
        return list(result), new_shape

    def choose_unique_op(train, target_group):
        target_sig = normalize_coords(target_group)
        matches = []
        for flip_ud, rot_k in OPS:
            t_body9, _ = xform_coords(train["body9"], train["bbox"].shape, flip_ud, rot_k)
            if normalize_coords(t_body9) == target_sig:
                matches.append((flip_ud, rot_k))
        assert len(matches) == 1, "Station shape must determine a unique orientation"
        return matches[0]

    edge_mask, _, edge_info = find_edge_stripes()
    internal_mask, _ = find_internal_stripes(edge_info)
    stripe_mask = edge_mask | internal_mask
    regions = regions_from_stripes(stripe_mask)

    groups = station_groups()
    groups_by_size = defaultdict(list)
    for g in groups:
        groups_by_size[len(g)].append(g)
    station_set = set(pt for g in groups for pt in g)

    body_lab, n_bodies = label(grid == 8, structure=FOUR_CONN)
    body_components = []
    for k in range(1, n_bodies + 1):
        coords = {(int(r), int(c)) for r, c in np.argwhere(body_lab == k)}
        if coords:
            body_components.append(coords)

    loose_couplers = {(int(r), int(c)) for r, c in np.argwhere(grid == 9) if (int(r), int(c)) not in station_set}
    body_to_couplers = {idx: [] for idx in range(len(body_components))}
    for cp in loose_couplers:
        owners = [idx for idx, body in enumerate(body_components) if cheb_adjacent(cp, body)]
        assert len(owners) == 1, "Each non-station coupler must touch exactly one body by 8-neighborhood"
        body_to_couplers[owners[0]].append(cp)

    trains = []
    for idx, body in enumerate(body_components):
        couplers = body_to_couplers[idx]
        assert couplers, "Each train body must have attached couplers"
        abs_coords = sorted(body | set(couplers))
        r0, c0, r1, c1 = coords_bbox(abs_coords)
        bbox = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        rel_body9 = []
        for r, c in body:
            bbox[r - r0, c - c0] = 8
        for r, c in couplers:
            bbox[r - r0, c - c0] = 9
            rel_body9.append((r - r0, c - c0))
        trains.append({"abs_coords": abs_coords, "bbox": bbox, "body9": rel_body9})

    out = grid.copy()
    for t in trains:
        for r, c in t["abs_coords"]:
            out[r, c] = bg

    assert len(trains) == len(groups), "Expected one target station per train"
    for t in trains:
        n9 = len(t["body9"])
        candidate_groups = groups_by_size.get(n9)
        assert candidate_groups is not None and len(candidate_groups) == 1, (
            f"Expected one station group for coupler count {n9}"
        )
        g = candidate_groups[0]
        gset = set(g)
        matching_regions = [
            rr
            for rr in regions
            if all(rr["r0"] <= r <= rr["r1"] and rr["c0"] <= c <= rr["c1"] for r, c in g)
        ]
        assert len(matching_regions) == 1, "Each station group must belong to exactly one region"
        reg = matching_regions[0]
        op = choose_unique_op(t, g)
        tbbox = apply_transform(t["bbox"], *op)
        t_body9, _ = xform_coords(t["body9"], t["bbox"].shape, *op)
        off_r, off_c = matching_offset(t_body9, gset)
        assert {(r + off_r, c + off_c) for r, c in t_body9} == gset, "Couplers must align onto the station shape"

        h1, w1 = tbbox.shape
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
                if v == 9:
                    if (rr, cc) not in gset or out[rr, cc] != 9:
                        ok = False
                        break
                elif out[rr, cc] != bg:
                    ok = False
                    break
            if not ok:
                break
        assert ok, "Docked train must fit cleanly within its target region"
        for r in range(h1):
            for c in range(w1):
                if int(tbbox[r, c]) == 8:
                    out[r + off_r, c + off_c] = 8
    return out


# --------------- generator ---------------

GRID_H_RANGE = (13, 31)
GRID_W_RANGE = (16, 29)
GRID_MIN_AREA = 300
MIN_REGION_DIM = 12
INTERNAL_STRIPE_THICK = 2
STRIPE_THICK_RANGE = (2, 4)
INTERNAL_PROB = 0.6
TRAIN_COUNT = 2
STATION_SIZES = [4, 5, 6]
STATION_CLEARANCE = 3
BODY_MIN = 5
BODY_MAX = 12
PATCH_H_RANGE = (4, 9)
PATCH_W_RANGE = (4, 8)
BODY_GROWTH_ITERS = 120
PATH_BOUNDS_PAD = 3
LAYOUT_RETRIES = 200
STATION_PAIR_RETRIES = 120
DOCK_RETRIES = 80
GENERATE_RETRIES = 400
COLOR_POOL = list(range(8))


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
    rs = [int(r) for r, _ in coords]
    cs = [int(c) for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _dims(coords):
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def _transform_shape(coords, op):
    r0, c0, r1, c1 = _bbox(coords)
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        arr[int(r - r0), int(c - c0)] = 1
    out = _apply_transform(arr, *op)
    return {(int(r), int(c)) for r, c in np.argwhere(out)}


def _shift(coords, dr, dc):
    return {(int(r + dr), int(c + dc)) for r, c in coords}


def _adjacent8(p, cells):
    r, c = p
    return any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in cells)


def _orth_neighbors(p):
    r, c = p
    return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _dilate8(mask):
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


def _path_to(body, goal, forbidden, bounds):
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
            if p in prev or p in forbidden:
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


def _grow_body(body, couplers, forbidden, target):
    for _ in range(BODY_GROWTH_ITERS):
        actual = body | set(couplers)
        h, w = _dims(actual)
        if len(body) >= target and h >= PATCH_H_RANGE[0] and w >= PATCH_W_RANGE[0]:
            break
        cands = []
        for r, c in body:
            for dr, dc in DIRS4:
                q = (r + dr, c + dc)
                if q in body or q in forbidden:
                    continue
                nh, nw = _dims(actual | {q})
                if nh <= PATCH_H_RANGE[1] and nw <= PATCH_W_RANGE[1]:
                    cands.append(q)
        if not cands or len(body) >= BODY_MAX:
            break
        h, w = _dims(actual)
        if h < PATCH_H_RANGE[0]:
            r0, _, r1, _ = _bbox(actual)
            exp = [q for q in cands if q[0] < r0 or q[0] > r1]
            cands = exp or cands
        elif w < PATCH_W_RANGE[0]:
            _, c0, _, c1 = _bbox(actual)
            exp = [q for q in cands if q[1] < c0 or q[1] > c1]
            cands = exp or cands
        body.add(random.choice(cands))


def _connect_couplers_to_body(couplers, body, forbidden, bounds):
    for cp in couplers:
        if _adjacent8(cp, body):
            continue
        cr, cc = cp
        choices = [
            (cr + dr, cc + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc) and (cr + dr, cc + dc) not in forbidden
        ]
        random.shuffle(choices)
        for q in choices:
            if _path_to(body, q, forbidden, bounds):
                break
        else:
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
        regions = []
        if orient == "v":
            lo = left + MIN_REGION_DIM
            hi = W - right - MIN_REGION_DIM - INTERNAL_STRIPE_THICK
            if lo > hi:
                continue
            c = random.randint(lo, hi)
            grid[top : H - bottom, c : c + INTERNAL_STRIPE_THICK] = internal_col
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
            regions = [
                {"r0": top, "r1": r - 1, "c0": left, "c1": W - right - 1},
                {"r0": r + INTERNAL_STRIPE_THICK, "r1": H - bottom - 1, "c0": left, "c1": W - right - 1},
            ]
        else:
            regions = [inner]

        if all(reg["r1"] - reg["r0"] + 1 >= MIN_REGION_DIM and reg["c1"] - reg["c0"] + 1 >= MIN_REGION_DIM for reg in regions):
            return grid, bg, regions
    return None


def _station_shape(size):
    return _transform_shape(STATION_BASE_SHAPES[size], random.choice(OPS))


def _fits_region(coords, reg, clearance):
    return all(
        reg["r0"] + clearance <= r <= reg["r1"] - clearance and reg["c0"] + clearance <= c <= reg["c1"] - clearance
        for r, c in coords
    )


def _pair_ok(shape_a, shape_b):
    if shape_a & shape_b:
        return False
    diag_touch = False
    for ar, ac in shape_a:
        for br, bc in shape_b:
            dr = abs(ar - br)
            dc = abs(ac - bc)
            if dr == 0 and dc == 0:
                return False
            if dr + dc == 1:
                return False
            if max(dr, dc) == 1 and dr + dc == 2:
                diag_touch = True
    return diag_touch


def _find_station_pair(reg, sizes):
    for _ in range(STATION_PAIR_RETRIES):
        shapes = [(size, _station_shape(size)) for size in sizes]
        size1, shape1 = shapes[0]
        size2, shape2 = shapes[1]
        _, _, r1max, c1max = _bbox(shape1)
        row_choices = range(reg["r0"] + STATION_CLEARANCE, reg["r1"] - STATION_CLEARANCE - r1max + 1)
        col_choices = range(reg["c0"] + STATION_CLEARANCE, reg["c1"] - STATION_CLEARANCE - c1max + 1)
        row_choices = list(row_choices)
        col_choices = list(col_choices)
        if not row_choices or not col_choices:
            continue
        base1 = _shift(shape1, random.choice(row_choices), random.choice(col_choices))
        cells1 = list(base1)
        cells2 = list(shape2)
        random.shuffle(cells1)
        random.shuffle(cells2)
        for _ in range(60):
            a = random.choice(cells1)
            b = random.choice(cells2)
            dr, dc = random.choice(DIAGS)
            base2 = _shift(shape2, a[0] + dr - b[0], a[1] + dc - b[1])
            if not _fits_region(base2, reg, STATION_CLEARANCE):
                continue
            if _pair_ok(base1, base2):
                return [
                    {"size": size1, "station": sorted(base1), "norm": normalize_coords(base1)},
                    {"size": size2, "station": sorted(base2), "norm": normalize_coords(base2)},
                ]
    return None


def _build_docked_patch(bg, couplers_norm):
    couplers = set(couplers_norm)
    coupler_list = list(couplers)
    for _ in range(DOCK_RETRIES):
        diag_cp = random.choice(coupler_list)
        diag_seed_options = []
        for dr, dc in DIAGS:
            q = (diag_cp[0] + dr, diag_cp[1] + dc)
            if q not in couplers:
                diag_seed_options.append(q)
        random.shuffle(diag_seed_options)
        for seed in diag_seed_options:
            preserve_empty = _orth_neighbors(diag_cp) - couplers
            forbidden = set(couplers) | preserve_empty
            if seed in forbidden:
                continue
            body = {seed}
            all_seed = body | set(couplers)
            sr0, sc0, sr1, sc1 = _bbox(all_seed)
            bounds = (sr0 - PATH_BOUNDS_PAD, sr1 + PATH_BOUNDS_PAD, sc0 - PATH_BOUNDS_PAD, sc1 + PATH_BOUNDS_PAD)
            if not _connect_couplers_to_body(couplers, body, forbidden, bounds):
                continue
            if len(body) > BODY_MAX:
                continue
            target = random.randint(max(BODY_MIN, len(body)), BODY_MAX)
            _grow_body(body, couplers, forbidden, target)
            actual = body | set(couplers)
            h, w = _dims(actual)
            diag_only = _adjacent8(diag_cp, body) and not orth_adjacent(diag_cp, body)
            if not diag_only:
                continue
            if not (
                BODY_MIN <= len(body) <= BODY_MAX
                and PATCH_H_RANGE[0] <= h <= PATCH_H_RANGE[1]
                and PATCH_W_RANGE[0] <= w <= PATCH_W_RANGE[1]
            ):
                continue
            r0, c0, _, _ = _bbox(actual)
            patch = np.full((h, w), bg, dtype=int)
            rel_couplers = []
            for r, c in body:
                patch[r - r0, c - c0] = 8
            for r, c in couplers:
                patch[r - r0, c - c0] = 9
                rel_couplers.append((r - r0, c - c0))
            return patch, rel_couplers
    return None


def _validate_patch_placement(patch, reg, canvas, bg, off_r, off_c):
    ph, pw = patch.shape
    for r in range(ph):
        for c in range(pw):
            v = int(patch[r, c])
            if v == bg:
                continue
            rr, cc = off_r + r, off_c + c
            if not (reg["r0"] <= rr <= reg["r1"] and reg["c0"] <= cc <= reg["c1"]):
                return False
            if canvas[rr, cc] != bg:
                return False
    return True


def _patch_locations(region, patch, bg, occupied, margin=0):
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
            if all(not occupied[rr + pr, cc + pc] for pr, pc in coords):
                locs.append((rr, cc))
    return locs


def _mixed_connectivity_cues_present(grid):
    H, W = grid.shape
    station_mask = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 9:
                continue
            if all(
                grid[r + dr, c + dc] != 8
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr != 0 or dc != 0) and 0 <= r + dr < H and 0 <= c + dc < W
            ):
                station_mask[r, c] = True

    station4, n_station4 = label(station_mask, structure=FOUR_CONN)
    _, n_station8 = label(station_mask, structure=EIGHT_CONN)
    if n_station4 != TRAIN_COUNT or n_station8 != 1:
        return False

    body_lab, n_bodies = label(grid == 8, structure=FOUR_CONN)
    if n_bodies != TRAIN_COUNT:
        return False
    bodies = []
    for k in range(1, n_bodies + 1):
        bodies.append({(int(r), int(c)) for r, c in np.argwhere(body_lab == k)})

    station_cells = {(int(r), int(c)) for r, c in np.argwhere(station_mask)}
    diag_only_bodies = set()
    for cp in ((int(r), int(c)) for r, c in np.argwhere(grid == 9) if (int(r), int(c)) not in station_cells):
        owners = [idx for idx, body in enumerate(bodies) if cheb_adjacent(cp, body)]
        if len(owners) != 1:
            return False
        body = bodies[owners[0]]
        if not orth_adjacent(cp, body):
            diag_only_bodies.add(owners[0])
    return len(diag_only_bodies) == TRAIN_COUNT


def _place_docked_trains(base, bg, regions):
    regions = sorted(regions, key=lambda reg: (reg["r1"] - reg["r0"] + 1) * (reg["c1"] - reg["c0"] + 1), reverse=True)
    sizes = random.sample(STATION_SIZES, TRAIN_COUNT)
    for reg in regions:
        for _ in range(STATION_PAIR_RETRIES):
            pair = _find_station_pair(reg, sizes)
            if pair is None:
                continue
            output = base.copy()
            trains = []
            ok = True
            for spec in sorted(pair, key=lambda item: item["size"], reverse=True):
                built = _build_docked_patch(bg, spec["norm"])
                if built is None:
                    ok = False
                    break
                patch, rel_couplers = built
                off_r, off_c = matching_offset(rel_couplers, set(spec["station"]))
                if not _validate_patch_placement(patch, reg, output, bg, off_r, off_c):
                    ok = False
                    break
                paste(output, patch, off_r, off_c, transparent=bg)
                trains.append({"patch": patch, "station": spec["station"], "size": spec["size"]})
            if ok:
                return output, trains
    return None


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
        alt = [op for op in valid_ops[0] if op != (False, 0)]
        if not alt:
            return None
        ops[0] = random.choice(alt)

    forbidden = _dilate8(base != bg) | _dilate8(station_mask)
    for t, op in zip(trains, ops):
        in_patch = _inverse_patch(t["patch"], op)
        regs = regions[:]
        random.shuffle(regs)
        all_locs = []
        for reg in regs:
            locs = _patch_locations(reg, in_patch, bg, forbidden, margin=1)
            all_locs.extend((reg, loc) for loc in locs)
        if not all_locs:
            return None
        _, (r0, c0) = random.choice(all_locs)
        paste(input_grid, in_patch, r0, c0, transparent=bg)
        ph, pw = in_patch.shape
        patch_mask = np.zeros((H, W), dtype=bool)
        patch_mask[r0 : r0 + ph, c0 : c0 + pw] = in_patch != bg
        forbidden |= _dilate8(patch_mask)
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
    if not _mixed_connectivity_cues_present(input_grid):
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
    raise RuntimeError("failed to generate valid 6e4f6532 combo input")


for _size, _shape in STATION_BASE_SHAPES.items():
    assert len({normalize_coords(_transform_shape(_shape, op)) for op in OPS}) == 8, (
        f"station shape size {_size} must have 8 distinct D4 orientations"
    )
