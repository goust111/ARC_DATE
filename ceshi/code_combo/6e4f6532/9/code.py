from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg, paste

# Combo variant of 6e4f6532 (index 9)
# combo_spec: D7/0+D11/1
# Name: 十字四分区+强制跨区对接
#
# Design analysis
# - Concrete behavior and trace:
#   * Base task: a train can start in any region, possibly the same region as the
#     station whose size matches its coupler count. With one internal divider at
#     most, recovering target directions is fairly local.
#   * Here the inner canvas always contains one horizontal and one vertical
#     2-cell divider, so the board is visibly split into four quadrants. Example:
#     a size-2 station sits in the top-left quadrant, while the matching 2-coupler
#     train is scattered into the bottom-right quadrant. The solver must first
#     recover the cross, then identify the train's source quadrant, then reject
#     any same-quadrant size match before orienting and docking the train.
#   * The key delta is not just "more regions"; the source-region identity now
#     participates in matching. A train's coupler count still identifies the
#     station size, but the valid target must also be outside the train's own
#     quadrant.
# - Complexity calibration:
#   * Each sample contains one duplicated station size family, so count alone no
#     longer resolves every match. The ambiguity stays human-readable because the
#     duplicate is limited to a single pair, and the train is always scattered
#     into exactly one of those two quadrants.
#   * Every target region still exposes 3-4 stripe colors, and every docked train
#     always carries both internal-divider colors as markers. This makes the cross
#     visually relevant in every sample rather than leaving it as decorative noise.
# - Hidden assumptions reviewed:
#   * The original internal-stripe scanner assumed a divider could be read as a
#     full constant row or column. That fails at the cross intersection, so the
#     solver first finds the full-width horizontal band, then finds the vertical
#     band while skipping the overwritten crossing rows.
#   * The original matcher only needed "unique station group by size". The combo
#     adds a new assumption: each train occupies exactly one recovered quadrant, and
#     the matching station group of that size is in a different quadrant.
#   * Placement validation still assumes the transformed sprite fits wholly inside
#     one target region and does not overlap stripes or other trains. The generator
#     must explicitly guarantee that when scattering sources and assembling outputs.
# - Placement legality:
#   * Output is still built first. Docked trains are placed inside target quadrants
#     with station clearance and region-bounded fit checks.
#   * Input is produced by inverse-transforming each train and scattering it only
#     into a quadrant different from its target. Stations are protected by dilation,
#     trains keep a one-cell margin from stripes, and retries reject layouts that
#     cannot realize the forced cross-region assignment.
# - Visual observability and cleanliness:
#   * The cross partition is always present and both internal stripe colors are
#     reused as markers, so the four-way frame remains salient in the input/output.
#   * Generated samples require every train to cross quadrants, and the duplicated
#     size pair is cross-swapped so each train starts inside the other same-size
#     station's quadrant. Samples with 3 trains still touch all 4 quadrants across
#     source/target usage. This makes the "must leave its own quadrant" rule
#     visible from a few examples.
#   * Object clutter is controlled by a single duplicated size pair, quadrant-level
#     separation, station clearance from stripes, and dilated non-overlap between
#     scattered trains.

INTERNAL_STRIPE_THICK = 2
MIN_REGION_DIM = 10

GRID_H_RANGE = (13, 31)
GRID_W_RANGE = (16, 29)
GRID_MIN_AREA = 300
MIN_REGION_DIRS = 3
STRIPE_THICK_RANGE = (2, 4)
TRAIN_COUNTS = [2, 3]
SIZE_PATTERNS_BY_COUNT = {
    2: [(1, 1), (2, 2)],
    3: [(1, 1, 2), (1, 1, 4), (2, 2, 1), (2, 2, 4)],
}
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
GENERATE_RETRIES = 300
COLOR_POOL = list(range(8))

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
CROSS_EDGE_SETS = [
    ("top", "bottom", "left", "right"),
    ("top", "bottom", "left"),
    ("top", "bottom", "right"),
    ("top", "left", "right"),
    ("bottom", "left", "right"),
    ("top", "bottom"),
    ("left", "right"),
]


def apply_transform(arr: np.ndarray, flip_ud: bool = False, rot_k: int = 0) -> np.ndarray:
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def transform_coords(coords, shape, flip_ud: bool = False, rot_k: int = 0):
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    out = apply_transform(mask, flip_ud=flip_ud, rot_k=rot_k)
    return set(map(tuple, np.argwhere(out)))


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


def _station_shape(size):
    if size == 1:
        return {(0, 0)}
    if size == 2:
        return {(0, 0), (0, 1)} if random.random() < 0.5 else {(0, 0), (1, 0)}
    return {(0, 0), (0, 1), (1, 0), (1, 1)}


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


def _assign_region_dirs(regions, edge_col, top, bottom, left, right, H, W, internal):
    h_pos, h_col = internal["h"]
    v_pos, v_col = internal["v"]
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
        dirs[h_col] = 2 if reg["r1"] == h_pos - 1 else 0
        dirs[v_col] = 1 if reg["c1"] == v_pos - 1 else 3
        reg["dirs"] = dirs
        reg["internal_colors"] = {h_col, v_col}
        rh = reg["r1"] - reg["r0"] + 1
        rw = reg["c1"] - reg["c0"] + 1
        if rh < MIN_REGION_DIM or rw < MIN_REGION_DIM or not (MIN_REGION_DIRS <= len(dirs) <= 4):
            return False
    return True


def _layout():
    for _ in range(LAYOUT_RETRIES):
        edges = list(random.choice(CROSS_EDGE_SETS))
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
                if ih < min_inner_for_split or iw < min_inner_for_split:
                    continue
                candidates.append((H, W))
        if not candidates:
            continue

        H, W = random.choice(candidates)
        pool = COLOR_POOL[:]
        bg = random.choice(pool)
        pool.remove(bg)
        stripe_cols = random.sample(pool, len(edges) + 2)
        edge_col = dict(zip(edges, stripe_cols[: len(edges)]))
        horiz_col = stripe_cols[len(edges)]
        vert_col = stripe_cols[len(edges) + 1]
        grid = np.full((H, W), bg, dtype=int)

        if top:
            grid[:top, :] = edge_col["top"]
        if bottom:
            grid[H - bottom :, :] = edge_col["bottom"]
        if left:
            grid[top : H - bottom, :left] = edge_col["left"]
        if right:
            grid[top : H - bottom, W - right :] = edge_col["right"]

        r_lo = top + MIN_REGION_DIM
        r_hi = H - bottom - MIN_REGION_DIM - INTERNAL_STRIPE_THICK
        c_lo = left + MIN_REGION_DIM
        c_hi = W - right - MIN_REGION_DIM - INTERNAL_STRIPE_THICK
        if r_lo > r_hi or c_lo > c_hi:
            continue
        r = random.randint(r_lo, r_hi)
        c = random.randint(c_lo, c_hi)
        grid[top : H - bottom, c : c + INTERNAL_STRIPE_THICK] = vert_col
        grid[r : r + INTERNAL_STRIPE_THICK, left : W - right] = horiz_col
        internal = {"h": (r, horiz_col), "v": (c, vert_col)}
        regions = [
            {"name": "tl", "r0": top, "r1": r - 1, "c0": left, "c1": c - 1},
            {"name": "tr", "r0": top, "r1": r - 1, "c0": c + INTERNAL_STRIPE_THICK, "c1": W - right - 1},
            {"name": "bl", "r0": r + INTERNAL_STRIPE_THICK, "r1": H - bottom - 1, "c0": left, "c1": c - 1},
            {"name": "br", "r0": r + INTERNAL_STRIPE_THICK, "r1": H - bottom - 1, "c0": c + INTERNAL_STRIPE_THICK, "c1": W - right - 1},
        ]
        if _assign_region_dirs(regions, edge_col, top, bottom, left, right, H, W, internal):
            return grid, bg, regions
    return None


def _marker_specs(dirs, required_colors):
    required = [(color, dirs[color]) for color in required_colors]
    optional = [item for item in dirs.items() if item[0] not in required_colors]
    if optional and len(dirs) >= 3 and random.random() < THREE_MARKER_PROB:
        return required + [random.choice(optional)]
    return required


def _build_docked_patch(bg, reg, size):
    couplers = _station_shape(size)
    specs = _marker_specs(reg["dirs"], reg["internal_colors"])
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


def _choose_size_pattern():
    n_trains = random.choice(TRAIN_COUNTS)
    return list(random.choice(SIZE_PATTERNS_BY_COUNT[n_trains]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-dock every train onto its matched station, but only across the cross-partitioned quadrants.

    Input structure:
    - The canvas has edge stripes plus one horizontal and one vertical internal stripe, forming
      four rectangular regions with 3-4 colored boundaries each.
    - Color 8 makes train bodies. Color 9 is used both for isolated station clusters and for
      train-attached couplers. Train marker pixels reuse region-stripe colors.

    Rule:
    - Recover station groups as 9-clusters not touching any 8, and recover train components as
      connected non-stripe objects containing 8.
    - A train matches the unique station whose size equals its coupler count and whose region is
      different from the train's current region.
    - Use the target region's stripe-color directions to choose the train's flip/rotation, then
      translate the transformed train so its couplers land exactly on the station cells.

    Key insight:
    - Count still identifies which station shape belongs to a train, but this variant adds a
      second constraint: the train is forbidden to dock inside its own quadrant, so the cross
      partition is part of the matching logic as well as the orientation logic.
    """
    inp = grid
    grid = inp.copy()
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

        horiz = []
        r = r0
        while r + INTERNAL_STRIPE_THICK <= r1:
            band = grid[r : r + INTERNAL_STRIPE_THICK, c0:c1]
            if band.size:
                row_colors = {int(band[k, 0]) for k in range(INTERNAL_STRIPE_THICK)}
                rows_constant = all(
                    (band[k] == band[k, 0]).all() and band[k, 0] != bg for k in range(INTERNAL_STRIPE_THICK)
                )
                if rows_constant and len(row_colors) == 1:
                    horiz.append((r, int(band[0, 0])))
                    r += INTERNAL_STRIPE_THICK
                    continue
            r += 1
        assert len(horiz) == 1, "Expected one horizontal internal stripe"
        hr, h_color = horiz[0]
        stripe[hr : hr + INTERNAL_STRIPE_THICK, c0:c1] = True
        colors.add(h_color)

        keep_rows = [rr for rr in range(r0, r1) if not (hr <= rr < hr + INTERNAL_STRIPE_THICK)]
        vert = []
        c = c0
        while c + INTERNAL_STRIPE_THICK <= c1:
            band = grid[np.ix_(keep_rows, range(c, c + INTERNAL_STRIPE_THICK))]
            if band.size:
                col_colors = {int(band[0, k]) for k in range(INTERNAL_STRIPE_THICK)}
                cols_constant = all(
                    (band[:, k] == band[0, k]).all() and band[0, k] != bg for k in range(INTERNAL_STRIPE_THICK)
                )
                if cols_constant and len(col_colors) == 1:
                    vert.append((c, int(band[0, 0])))
                    c += INTERNAL_STRIPE_THICK
                    continue
            c += 1
        assert len(vert) == 1, "Expected one vertical internal stripe"
        vc, v_color = vert[0]
        stripe[r0:r1, vc : vc + INTERNAL_STRIPE_THICK] = True
        colors.add(v_color)
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

    def regions_from_stripes(stripe_mask):
        non = ~stripe_mask
        lab, n = label(non)
        regs = []
        for k in range(1, n + 1):
            coords = np.argwhere(lab == k)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            if (r1 - r0 + 1) >= MIN_REGION_DIM and (c1 - c0 + 1) >= MIN_REGION_DIM:
                regs.append({"r0": int(r0), "r1": int(r1), "c0": int(c0), "c1": int(c1)})
        regs.sort(key=lambda reg: (reg["r0"], reg["c0"]))
        return regs

    def region_index_of_cells(cells, regs):
        hits = [
            idx
            for idx, reg in enumerate(regs)
            if all(reg["r0"] <= r <= reg["r1"] and reg["c0"] <= c <= reg["c1"] for r, c in cells)
        ]
        assert len(hits) == 1, "Object must belong to exactly one region"
        return hits[0]

    def boundary_dirs_for_region(reg, stripe_colors):
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
        assert train["body9"], "Each train must carry at least one coupler"

        ref_r = sum(r for r, _ in train["body9"]) / len(train["body9"])
        ref_c = sum(c for _, c in train["body9"]) / len(train["body9"])
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
    assert len(regions) == 4, "Cross divider should yield four regions"

    groups = station_groups()
    station_infos = []
    station_set = set()
    for g in groups:
        reg_idx = region_index_of_cells(g, regions)
        station_infos.append({"coords": g, "gset": set(g), "size": len(g), "region_idx": reg_idx})
        station_set.update(g)

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
            if v in stripe_colors:
                marker_pixels.append((v, int(r - r0), int(c - c0)))
        trains.append(
            {
                "abs_coords": abs_coords,
                "bbox": bbox,
                "body9": body9,
                "marker_pixels": marker_pixels,
                "source_region": region_index_of_cells(abs_coords, regions),
            }
        )

    out = grid.copy()
    for t in trains:
        for r, c in t["abs_coords"]:
            out[r, c] = bg

    for t in trains:
        n9 = len(t["body9"])
        assert n9 > 0, "Each train must carry at least one coupler"
        candidates = [
            info for info in station_infos if info["size"] == n9 and info["region_idx"] != t["source_region"]
        ]
        assert len(candidates) == 1, f"Expected one cross-region station group for coupler count {n9}"
        info = candidates[0]
        reg = regions[info["region_idx"]]
        divider_dirs = boundary_dirs_for_region(reg, stripe_colors)
        assert 3 <= len(divider_dirs) <= 4, "Each target region must expose 3-4 boundary stripes"
        op = choose_best_op(t, divider_dirs)
        tbbox = apply_transform(t["bbox"], *op)
        t_body9, (H1, W1) = xform_coords(t["body9"], t["bbox"].shape, *op)
        body9_set = set(t_body9)
        gset = info["gset"]
        matching_offsets = {
            (sr - br, sc - bc)
            for br, bc in body9_set
            for sr, sc in gset
            if all((r + sr - br, c + sc - bc) in gset for r, c in body9_set)
        }
        assert len(matching_offsets) == 1, "Couplers must align to a unique station-group offset"
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
                if stripe_mask[rr, cc] or (out[rr, cc] != bg and out[rr, cc] != 9):
                    ok = False
                    break
            if not ok:
                break
        assert ok, "Docked train must fit entirely within its cross-region target"

        for r in range(H1):
            for c in range(W1):
                v = int(tbbox[r, c])
                if v != bg and v != 9:
                    out[r + off_r, c + off_c] = v
    return out


def _place_docked_trains(base, bg, regions, size_pattern):
    output = base.copy()
    occupied = output != bg

    n_trains = len(size_pattern)
    sizes = list(size_pattern)
    target_regions = random.sample(range(len(regions)), n_trains)
    trains = []
    for size, reg_idx in zip(sizes, target_regions):
        reg = regions[reg_idx]
        for _ in range(DOCK_RETRIES):
            result = _build_docked_patch(bg, reg, size)
            if result is None:
                continue
            patch, coupler_rel = result
            locs = _patch_locations(reg, patch, bg, _dilate(occupied), coupler_rel, True, 0)
            if not locs:
                continue
            r0, c0 = random.choice(locs)
            paste(output, patch, r0, c0, transparent=bg)
            occupied |= _dilate((output != bg) & ~occupied) | (output != bg)
            station_abs = [(r0 + r, c0 + c) for r, c in coupler_rel]
            trains.append(
                {
                    "patch": patch,
                    "size": size,
                    "station": station_abs,
                    "target_region": reg_idx,
                }
            )
            break
        else:
            return None
    return output, trains


def _plan_source_regions(trains, n_regions):
    groups = defaultdict(list)
    for idx, train in enumerate(trains):
        groups[train["size"]].append(idx)

    if not any(len(idxs) == 2 for idxs in groups.values()):
        return None
    if any(len(idxs) > 2 for idxs in groups.values()):
        return None

    source_regions = [None] * len(trains)
    used = set()
    for idxs in groups.values():
        if len(idxs) != 2:
            continue
        a, b = idxs
        source_regions[a] = trains[b]["target_region"]
        source_regions[b] = trains[a]["target_region"]
        used.add(source_regions[a])
        used.add(source_regions[b])

    singleton_idxs = [idxs[0] for idxs in groups.values() if len(idxs) == 1]
    random.shuffle(singleton_idxs)
    for idx in singleton_idxs:
        target = trains[idx]["target_region"]
        candidates = [reg_idx for reg_idx in range(n_regions) if reg_idx != target]
        if not candidates:
            return None
        preferred = [reg_idx for reg_idx in candidates if reg_idx not in used]
        source_regions[idx] = random.choice(preferred or candidates)
        used.add(source_regions[idx])

    return source_regions


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

    source_plan = _plan_source_regions(trains, len(regions))
    if source_plan is None:
        return None

    forbidden = _dilate(base != bg) | _dilate(station_mask)
    used_source_regions = set()
    for t, op, source_region in zip(trains, ops, source_plan):
        in_patch = _inverse_patch(t["patch"], op)
        reg = regions[source_region]
        locs = _patch_locations(reg, in_patch, bg, forbidden, None, False, 1)
        if not locs:
            return None
        r0, c0 = random.choice(locs)
        used_source_regions.add(source_region)
        paste(input_grid, in_patch, r0, c0, transparent=bg)
        ph, pw = in_patch.shape
        patch_mask = np.zeros((H, W), dtype=bool)
        patch_mask[r0 : r0 + ph, c0 : c0 + pw] = in_patch != bg
        forbidden |= _dilate(patch_mask)
        t["source_region"] = source_region

    touched_regions = used_source_regions | {t["target_region"] for t in trains}
    need_touched = 4 if len(trains) == 3 else 2
    if any(t["source_region"] == t["target_region"] for t in trains):
        return None
    if len(touched_regions) < need_touched:
        return None

    return input_grid


def _gen_attempt():
    layout = _layout()
    if layout is None:
        return None
    base, bg, regions = layout

    size_pattern = _choose_size_pattern()
    placed = _place_docked_trains(base, bg, regions, size_pattern)
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
