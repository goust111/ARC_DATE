# Combo variant of 6e4f6532 (index 20)
# combo_spec: D7/0+D8/2+D12/1
# Name: 四分区+干扰项+多对配对

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# Concrete behavior and trace:
# - The base task used 1-2 regions, 2-3 train/station pairs, and no clutter. A
#   component containing 8 was automatically a real train, every isolated 9-cluster
#   was automatically a real station, and a single size match picked the target.
# - Here the inner board is always split by one horizontal and one vertical
#   2-cell divider, so the solver must recover a four-quadrant cross partition.
#   The horizontal divider is painted after the vertical one, so the crossing cells
#   hide the vertical color; this breaks the original "constant full column" stripe
#   scan and requires cross-aware divider recovery.
# - The grid now holds 4-5 real pairings with distinct station sizes from {1,2,3,4,5},
#   plus one fake train (an 8-component with stripe-color markers but no 9 couplers)
#   and one fake station (an isolated 9-cluster of size 6). A concrete trace is:
#   old solver sees six 8-components and six isolated 9-groups and would try to move
#   all 8-components. The combo solver instead:
#   1. recovers the four regions from the cross dividers,
#   2. keeps only 8-components that carry non-station 9 couplers,
#   3. keeps only isolated 9-groups whose size matches exactly one such coupler count,
#   4. uses marker-vs-boundary colors to orient each real train,
#   5. erases only the real trains and docks them onto their matched stations.
# - Sizes 3 and 5 are represented by compact connected clusters, so the expanded
#   pairing set is visible in pixels rather than hidden in metadata.
#
# Complexity calibration:
# - The visual changes are intentionally strong but controlled: there are always four
#   regions, always 4-5 real pairs, always at least one fake train, and always at
#   least one fake station of unmatched size. Each real train always carries both
#   internal-divider colors as markers, so the cross partition actually participates
#   in the orientation rule instead of being decorative.
# - Real station sizes remain unique within a sample, so coupler count still selects
#   a single destination. The decoy station uses size 6 only, avoiding overlap with
#   the real size vocabulary while still looking like a plausible station cluster.
#
# Hidden assumptions audited:
# - "At most one internal stripe" is false; divider detection must recover one full
#   horizontal band and one partial vertical band.
# - "Every 8-component is movable" is false; zero-coupler components are static decoys.
# - "Every isolated 9-cluster is a destination" is false; the size-6 cluster is noise.
# - "Clearing all train-like components is safe" is false; only matched movable trains
#   may be erased, otherwise the fake train disappears incorrectly.
# - "The board is roomy enough for 2-3 objects" is false; the generator must enlarge
#   the canvas and retry budgets so 4-5 docked trains, 4-5 scattered trains, and two
#   static decoys fit without collisions.
# - New solver assumptions introduced here are mirrored by generation: exactly one
#   horizontal divider and one vertical divider, four recovered regions, unique real
#   station sizes, one fake station of size 6, and at least one fake train with no
#   couplers.
#
# Placement legality and cleanliness:
# - Generation stays output-first. It first places the docked real trains in their
#   target quadrants, then injects the static decoys into both input/output, and only
#   after that scatters inverse-transformed real trains back into source regions.
# - All placements use one-cell dilation clearance, station cells keep two-cell
#   clearance from region borders, and fake stations use compact connected hexominoes
#   so the board stays readable instead of becoming a dense 9-colored blob.
#
# Observability and inferability:
# - Across a few examples, a human can see three independent clues that repeat:
#   four-way striped partitioning, many more train/station pairs than the base task,
#   and exactly two objects that never move (one unmatched 9-cluster and one 8-object
#   without couplers). No rule depends on hidden color ordering; marker colors are
#   always copied from visible region boundaries.


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


# ── Tunable constants ────────────────────────────────────────────────
GRID_H_RANGE = (32, 49)
GRID_W_RANGE = (32, 53)
GRID_MIN_AREA = 1050
MIN_REGION_DIM = 12
MIN_REGION_DIRS = 3
INTERNAL_STRIPE_THICK = 2

STRIPE_THICK_RANGE = (2, 4)

TRAIN_COUNTS = [4, 5]
STATION_SIZES = [1, 2, 3, 4, 5]
DECOY_STATION_SIZE = 6

BODY_MIN = 5
BODY_MAX = 20
PATCH_H_RANGE = (4, 9)
PATCH_W_RANGE = (3, 8)

MARKER_DIST_RANGE = (2, 3)
MARKER_JITTER = [-1, 0, 1]
THREE_MARKER_PROB = 0.7

BODY_GROWTH_ITERS = 160
PATH_BOUNDS_PAD = 4

LAYOUT_RETRIES = 320
DOCK_RETRIES = 120
STATIC_DECOY_RETRIES = 80
GENERATE_RETRIES = 900

COLOR_POOL = list(range(8))
CROSS_EDGE_SETS = [
    ("top", "bottom", "left", "right"),
    ("top", "bottom", "left"),
    ("top", "bottom", "right"),
    ("top", "left", "right"),
    ("bottom", "left", "right"),
    ("top", "bottom"),
    ("left", "right"),
]
DECOY_STATION_SHAPES = [
    {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)},
    {(0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)},
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


def _find_cross_internal_stripes(grid: np.ndarray, bg: int, edge_info):
    H, W = grid.shape
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
                (band[k] == band[k, 0]).all() and band[k, 0] != bg
                for k in range(INTERNAL_STRIPE_THICK)
            )
            if rows_constant and len(row_colors) == 1:
                horiz.append((r, int(band[0, 0])))
                r += INTERNAL_STRIPE_THICK
                continue
        r += 1
    assert len(horiz) == 1, "Expected one horizontal divider"
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
                (band[:, k] == band[0, k]).all() and band[0, k] != bg
                for k in range(INTERNAL_STRIPE_THICK)
            )
            if cols_constant and len(col_colors) == 1:
                vert.append((c, int(band[0, 0])))
                c += INTERNAL_STRIPE_THICK
                continue
        c += 1
    assert len(vert) == 1, "Expected one vertical divider"
    vc, v_color = vert[0]
    stripe[r0:r1, vc : vc + INTERNAL_STRIPE_THICK] = True
    colors.add(v_color)
    return stripe, colors


def _regions_from_stripes(stripe_mask: np.ndarray, min_dim: int = MIN_REGION_DIM):
    lab, n = label(~stripe_mask)
    regs = []
    for k in range(1, n + 1):
        coords = np.argwhere(lab == k)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (r1 - r0 + 1) >= min_dim and (c1 - c0 + 1) >= min_dim:
            regs.append({"r0": int(r0), "r1": int(r1), "c0": int(c0), "c1": int(c1)})
    regs.sort(key=lambda reg: (reg["r0"], reg["c0"]))
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
    ref_t = list(transform_coords([(int(round(ref_r)), int(round(ref_c)))], (H0, W0), op[0], op[1]))
    rr_ref, cc_ref = ref_t[0]

    score = 0
    for color, pts in markers_by_color.items():
        mr = sum(r for r, _ in pts) / len(pts)
        mc = sum(c for _, c in pts) / len(pts)
        m_t = list(transform_coords([(int(round(mr)), int(round(mc)))], (H0, W0), op[0], op[1]))
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


def _region_indices_for_coords(regions, coords):
    matches = []
    for idx, reg in enumerate(regions):
        if all(reg["r0"] <= r <= reg["r1"] and reg["c0"] <= c <= reg["c1"] for r, c in coords):
            matches.append(idx)
    return matches


def _collect_move_plan(grid: np.ndarray):
    bg = get_bg(grid)
    edge_mask, edge_colors, edge_info = _find_edge_stripes(grid, bg)
    internal_mask, internal_colors = _find_cross_internal_stripes(grid, bg, edge_info)
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
        matching_regions = _region_indices_for_coords(regions, g)
        if len(matching_regions) != 1:
            continue
        target_region = matching_regions[0]
        divider_dirs = _boundary_dirs_for_region(grid, regions[target_region], stripe_colors)
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
        source_regions = _region_indices_for_coords(regions, t["abs_coords"])
        if len(source_regions) != 1:
            continue
        movable.append(
            {
                "train": t,
                "op": op,
                "score": score,
                "reg": regions[target_region],
                "target_region": target_region,
                "source_region": source_regions[0],
                "offset": next(iter(matching_offsets)),
                "station": g,
            }
        )

    return {
        "bg": bg,
        "stripe_mask": stripe_mask,
        "stripe_colors": stripe_colors,
        "regions": regions,
        "groups": groups,
        "train_components": train_components,
        "movable": movable,
    }


def analyze_combo(grid: np.ndarray) -> dict[str, object]:
    info = _collect_move_plan(grid)
    real_sizes = {len(item["train"]["body9"]) for item in info["movable"]}
    fake_stations = sum(1 for g in info["groups"] if len(g) not in real_sizes)
    fake_trains = sum(1 for t in info["train_components"] if len(t["body9"]) == 0)
    target_regions = {item["target_region"] for item in info["movable"]}
    source_regions = {item["source_region"] for item in info["movable"]}
    cross_moves = sum(1 for item in info["movable"] if item["source_region"] != item["target_region"])
    return {
        "regions": len(info["regions"]),
        "real_trains": len(info["movable"]),
        "fake_trains": fake_trains,
        "fake_stations": fake_stations,
        "cross_region_moves": cross_moves,
        "target_region_count": len(target_regions),
        "source_region_count": len(source_regions),
        "real_sizes": sorted(real_sizes),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-dock the real trains onto matching stations inside a four-quadrant board while leaving decoys fixed.

    Input structure:
    - Colored edge bands plus one horizontal and one vertical internal band split the canvas into four rectangular regions.
    - Color 8 forms train bodies; color 9 is reused for isolated station clusters and for train-attached couplers.
    - There are 4-5 genuine train/station matches, plus static clutter: at least one 8-component with no couplers and at least one isolated 9-cluster of unmatched size.

    Rule:
    - Recover the four regions, then treat isolated 9-clusters as candidate stations and 8-components as candidate trains.
    - Keep only the components that carry attached non-station 9 couplers and whose coupler count matches exactly one isolated 9-cluster.
    - For each such real train, choose the flip/rotation whose stripe-colored markers face the same-color boundaries of the target region, erase the train from its source location, and dock it so its couplers coincide with the target station cells.

    Key insight:
    - The combo still resolves each true pair by “coupler count picks the station, marker colors fix the orientation”, but now that logic must be applied only after filtering out the fake train and fake station and after decoding the cross-shaped divider geometry.
    """
    info = _collect_move_plan(grid)
    bg = info["bg"]
    stripe_mask = info["stripe_mask"]
    regions = info["regions"]
    assert len(regions) == 4, "Expected four cross-partition regions"

    out = grid.copy()
    for item in info["movable"]:
        for r, c in item["train"]["abs_coords"]:
            out[r, c] = bg

    for item in info["movable"]:
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
        assert ok, "Docked train must fit in its target region"
        for r in range(H1):
            for c in range(W1):
                v = int(tbbox[r, c])
                if v != bg and v != 9:
                    out[r + off_r, c + off_c] = v
    return out


def _assign_region_dirs(regions, edge_col, top, bottom, left, right, H, W, internal):
    h_pos, h_col = internal["h"]
    v_pos, v_col = internal["v"]
    for idx, reg in enumerate(regions):
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
        reg["idx"] = idx
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
            {"r0": top, "r1": r - 1, "c0": left, "c1": c - 1},
            {"r0": top, "r1": r - 1, "c0": c + INTERNAL_STRIPE_THICK, "c1": W - right - 1},
            {"r0": r + INTERNAL_STRIPE_THICK, "r1": H - bottom - 1, "c0": left, "c1": c - 1},
            {
                "r0": r + INTERNAL_STRIPE_THICK,
                "r1": H - bottom - 1,
                "c0": c + INTERNAL_STRIPE_THICK,
                "c1": W - right - 1,
            },
        ]
        if _assign_region_dirs(regions, edge_col, top, bottom, left, right, H, W, internal):
            return grid, bg, regions
    return None


def _station_shape(size):
    if size == 1:
        return {(0, 0)}
    if size == 2:
        return {(0, 0), (0, 1)} if random.random() < 0.5 else {(0, 0), (1, 0)}
    if size == 3:
        return {(0, 0), (1, 0), (1, 1)}
    if size == 4:
        return {(0, 0), (0, 1), (1, 0), (1, 1)}
    if size == 5:
        return {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}
    raise ValueError(f"unsupported station size: {size}")


def _marker_specs(dirs, required_colors):
    required = [(color, dirs[color]) for color in required_colors]
    optional = [item for item in dirs.items() if item[0] not in required_colors]
    if optional and len(dirs) >= 3 and random.random() < THREE_MARKER_PROB:
        return required + [random.choice(optional)]
    return required


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


def _render_shape(cells, color, bg):
    r0, c0, r1, c1 = _bbox(cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    rel = []
    for r, c in cells:
        patch[r - r0, c - c0] = color
        rel.append((r - r0, c - c0))
    return patch, rel


def _build_decoy_train_patch(bg, reg):
    for _ in range(DOCK_RETRIES):
        size = random.choice(STATION_SIZES)
        result = _build_docked_patch(bg, reg, size)
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


def _target_region_sequence(n_trains):
    order = list(range(4))
    random.shuffle(order)
    if n_trains == 4:
        return order
    order.append(random.choice(order))
    random.shuffle(order)
    return order


def _place_docked_trains(base, bg, regions):
    output = base.copy()
    occupied = output != bg

    n_trains = random.choice(TRAIN_COUNTS)
    sizes = random.sample(STATION_SIZES, n_trains)
    target_regions = _target_region_sequence(n_trains)
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
            occupied = output != bg
            station_abs = [(r0 + r, c0 + c) for r, c in coupler_rel]
            trains.append(
                {
                    "patch": patch,
                    "station": station_abs,
                    "target_region": reg_idx,
                    "size": size,
                }
            )
            break
        else:
            return None
    if len({t["target_region"] for t in trains}) != 4:
        return None
    return output, trains


def _place_static_decoys(base, output_real, bg, regions):
    decor = base.copy()
    output = output_real.copy()
    occupied = output != bg

    placed_station = False
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
        placed_station = True
        break
    if not placed_station:
        return None

    placed_train = False
    for _ in range(STATIC_DECOY_RETRIES):
        reg = random.choice(regions)
        patch = _build_decoy_train_patch(bg, reg)
        if patch is None:
            continue
        locs = _patch_locations(reg, patch, bg, _dilate(occupied), None, False, 1)
        if not locs:
            continue
        r0, c0 = random.choice(locs)
        paste(decor, patch, r0, c0, transparent=bg)
        paste(output, patch, r0, c0, transparent=bg)
        occupied = output != bg
        placed_train = True
        break
    if not placed_train:
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
    used_source_regions = set()
    moved_across = 0
    for t, op in zip(trains, ops):
        in_patch = _inverse_patch(t["patch"], op)
        order = list(range(len(regions)))
        random.shuffle(order)
        diff_locs = []
        same_locs = []
        for reg_idx in order:
            reg = regions[reg_idx]
            locs = _patch_locations(reg, in_patch, bg, forbidden, None, False, 1)
            bucket = same_locs if reg_idx == t["target_region"] else diff_locs
            bucket.extend((reg_idx, p) for p in locs)
        choices = diff_locs or same_locs
        if not choices:
            return None
        reg_idx, (r0, c0) = random.choice(choices)
        used_source_regions.add(reg_idx)
        if reg_idx != t["target_region"]:
            moved_across += 1
        paste(input_grid, in_patch, r0, c0, transparent=bg)
        ph, pw = in_patch.shape
        patch_mask = np.zeros((H, W), dtype=bool)
        patch_mask[r0 : r0 + ph, c0 : c0 + pw] = in_patch != bg
        forbidden |= _dilate(patch_mask)
        t["source_region"] = reg_idx

    required_cross = 2 if len(trains) == 4 else 3
    if moved_across < required_cross or len(used_source_regions) < 3:
        return None
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

    metrics = analyze_combo(input_grid)
    expected_n = len(trains)
    required_cross = 2 if expected_n == 4 else 3
    if metrics["regions"] != 4:
        return None
    if metrics["real_trains"] != expected_n:
        return None
    if metrics["fake_trains"] < 1 or metrics["fake_stations"] < 1:
        return None
    if metrics["cross_region_moves"] < required_cross:
        return None
    if metrics["target_region_count"] != 4:
        return None
    if DECOY_STATION_SIZE in metrics["real_sizes"]:
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
