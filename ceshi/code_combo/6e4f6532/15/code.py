# Combo variant of 6e4f6532 (index 15)
# combo_spec: D4/3+D10/1
# Name: 同大小车站+耦合器/车站颜色可变
#
# Design analysis
# a) Concrete behavior:
#    In the original task, a train with N couplers could identify its station by
#    the unique station size N. In this combo, two isolated stations can have the
#    same size, and each train/station pair instead shares its own coupler color.
#    A concrete example is two size-2 stations, one color 3 and one color 5:
#    original logic would stop at "size 2", while the combo must continue with
#    "same color as the train couplers", then use stripe-colored markers only to
#    recover the docking orientation inside that chosen region.
# b) Complexity calibration:
#    The generator forces exactly two regions and exactly two trains, with both
#    stations using the same size. That keeps the ambiguity visible in every
#    sample without creating clutter from many overlapping alternatives.
# c) Hidden assumptions audited:
#    The original solver relied on two assumptions that no longer hold:
#    `all stations/couplers are color 9` and `each coupler count maps to one
#    station`. The combo solver therefore detects stations as monochromatic
#    non-bg/non-8/non-stripe blobs, requires each train to carry one coupler
#    color, and matches by color instead of size. Size is still validated by the
#    docking alignment step, but it is not the identifier.
# d) Placement legality:
#    The generator still uses an output-first construction. It places one docked
#    train per region, keeps coupler/station colors distinct from bg and stripe
#    colors, scatters inverse-transformed trains back into regions with padding,
#    and accepts a sample only if `main(input) == output`.
# e) Visual observability:
#    Every sample contains a repeated station size, so a human can see that size
#    alone is insufficient. The deciding cue is directly visible: each station
#    has a distinct color, and the matching train couplers reuse that color.
# f) Visual neatness:
#    Restricting the layout to two large regions and one station per region keeps
#    the repeated-size comparison easy to read. The trains still carry only a few
#    stripe-colored markers, so the colored couplers remain visually prominent.
# g) Change inferability:
#    Color variation is tightly structured rather than free-form: exactly one
#    coupler/station color belongs to each train/station pair, while marker
#    colors still come from the destination region's border stripes. This gives
#    enough repeated evidence across 2-4 examples for a human to infer the rule.

from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import combinations
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg, paste


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
    Dock train sprites onto equal-sized station groups by matching coupler color, then orient via divider-color markers.

    Input structure:
    - Colored edge and internal stripes partition the canvas into two rectangular regions.
    - Color 8 is the train body color.
    - Each isolated station cluster and its matching train couplers share one unique non-stripe color; different pairs use different colors even when station sizes repeat.
    - Train marker pixels reuse boundary-stripe colors, so every train contains body pixels, colored couplers, and stripe-colored orientation cues.

    Rule:
    - Stations are 8-connected monochromatic clusters whose color is neither background nor 8 nor a stripe color, and whose cells have no adjacent 8.
    - Trains are non-stripe connected components containing color 8; their attached non-8/non-stripe pixels are couplers, and each train uses exactly one coupler color.
    - Match each train to the station group of the same color, then rotate/flip so the train's marker colors face the same-color borders of that station's region.
    - Erase the scattered source copy and redraw the transformed train so its couplers land exactly on the station cells.

    Key insight: repeated station sizes make count-based matching ambiguous, so the coupler/station color selects the destination, while stripe-colored markers still determine the unique docking orientation.
    """
    inp = grid
    grid = inp.copy()
    H, W = grid.shape
    bg = get_bg(grid)

    # This puzzle's code is long because it reconstructs several "structures"
    # that are implicit in the grid:
    # - divider stripes (edge + internal) that partition the canvas into regions
    # - station groups = monochromatic non-bg/non-8/non-stripe clusters not adjacent to any body pixel
    # - train sprites = components containing body color 8 plus attached couplers
    # Then, for each train:
    # - identify the target station by COUPLER COLOR
    # - choose orientation by divider-color markers
    # - translate so couplers land exactly onto the matching station cells

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
        groups = []
        for color in range(10):
            if color in (bg, 8) or color in stripe_colors:
                continue
            sm = np.zeros((H, W), dtype=bool)
            for r in range(H):
                for c in range(W):
                    if int(grid[r, c]) != color:
                        continue
                    ok = True
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            rr, cc = r + dr, c + dc
                            if not (0 <= rr < H and 0 <= cc < W):
                                continue
                            if int(grid[rr, cc]) == 8:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        sm[r, c] = True
            struct = np.ones((3, 3), dtype=int)
            lab, n = label(sm, structure=struct)
            for k in range(1, n + 1):
                coords = np.argwhere(lab == k)
                groups.append({
                    "color": int(color),
                    "coords": [(int(r), int(c)) for r, c in coords],
                })
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
        marker_colors = {v for v, _, _ in train["marker_pixels"]}
        if not marker_colors or not marker_colors.issubset(set(divider_dirs)):
            return None
        markers_by_color = defaultdict(list)
        for v, r, c in train["marker_pixels"]:
            markers_by_color[v].append((r, c))
        ref_r = sum(r for r, _ in train["couplers"]) / len(train["couplers"])
        ref_c = sum(c for _, c in train["couplers"]) / len(train["couplers"])
        best, best_score = (False, 0), -1
        for flip_ud in (False, True):
            for k in range(4):
                ref_t = list(
                    transform_coords(
                        [(int(round(ref_r)), int(round(ref_c)))],
                        bbox0.shape,
                        flip_ud,
                        k,
                    )
                )
                rr_ref, cc_ref = ref_t[0]
                score = 0
                for color, pts in markers_by_color.items():
                    mr = sum(r for r, _ in pts) / len(pts)
                    mc = sum(c for _, c in pts) / len(pts)
                    m_t = list(
                        transform_coords(
                            [(int(round(mr)), int(round(mc)))],
                            bbox0.shape,
                            flip_ud,
                            k,
                        )
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

    groups = station_groups()
    groups_by_color = {}
    station_set = set()
    for g in groups:
        matching_regions = [
            reg
            for reg in regions
            if all(reg["r0"] <= r <= reg["r1"] and reg["c0"] <= c <= reg["c1"] for r, c in g["coords"])
        ]
        assert len(matching_regions) == 1, "Each station group must belong to exactly one region"
        assert g["color"] not in groups_by_color, "Each coupler color must map to exactly one station group"
        reg = matching_regions[0]
        groups_by_color[g["color"]] = {
            "coords": g["coords"],
            "region": reg,
            "divider_dirs": boundary_dirs_for_region(reg, stripe_colors),
        }
        station_set.update(g["coords"])

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
        couplers = []
        marker_pixels = []
        for r, c in abs_coords:
            v = int(grid[r, c])
            if v in stripe_colors:
                marker_pixels.append((v, int(r - r0), int(c - c0)))
            elif v != bg and v != 8 and (r, c) not in station_set:
                couplers.append((int(r - r0), int(c - c0), v))
        coupler_colors = {v for _, _, v in couplers}
        assert couplers, "Each train must carry at least one coupler"
        assert len(coupler_colors) == 1, "Each train must use exactly one coupler color"
        trains.append({
            "abs_coords": abs_coords,
            "bbox": bbox,
            "couplers": [(r, c) for r, c, _ in couplers],
            "coupler_color": int(next(iter(coupler_colors))),
            "marker_pixels": marker_pixels,
        })

    out = grid.copy()
    for t in trains:
        for r, c in t["abs_coords"]:
            out[r, c] = bg

    for t in trains:
        info = groups_by_color.get(t["coupler_color"])
        assert info is not None, f"No station group for coupler color {t['coupler_color']}"
        gset = set(info["coords"])
        reg = info["region"]
        assert len(gset) == len(t["couplers"]), "Coupler count must match station size after color pairing"

        op = choose_best_op(t, info["divider_dirs"])
        assert op is not None, "Marker colors must be compatible with the target region"
        tbbox = apply_transform(t["bbox"], *op)
        t_couplers, (H1, W1) = xform_coords(t["couplers"], t["bbox"].shape, *op)
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
                if v == bg or (r, c) in coupler_set:
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
        assert ok, "Docked train must fit entirely within its target region"

        for r in range(H1):
            for c in range(W1):
                v = int(tbbox[r, c])
                if v != bg and (r, c) not in coupler_set:
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

# Internal stripe is forced on so the repeated-size comparison is visually explicit.
INTERNAL_PROB = 1.0

# Train / station counts
TRAIN_COUNTS = [2]
STATION_SIZES = [1, 2, 4]

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

# Color pool for bg, stripes, and variable coupler/station colors (8 is reserved for bodies).
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
        assert use_internal
        orient = random.choice(["v", "h"])
        if orient == "v":
            edges = ["left", "right", random.choice(["top", "bottom"])]
        else:
            edges = ["top", "bottom", random.choice(["left", "right"])]

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
        stripe_cols = random.sample(pool, len(edges) + 1)
        edge_col = dict(zip(edges, stripe_cols[: len(edges)]))
        internal_col = stripe_cols[-1]
        grid = np.full((H, W), bg, dtype=int)

        if top:
            grid[:top, :] = edge_col["top"]
        if bottom:
            grid[H - bottom :, :] = edge_col["bottom"]
        if left:
            grid[top : H - bottom, :left] = edge_col["left"]
        if right:
            grid[top : H - bottom, W - right :] = edge_col["right"]

        regions = []
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
        else:
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

        if len(regions) == 2 and _assign_region_dirs(regions, edge_col, top, bottom, left, right, H, W, internal):
            return grid, bg, regions
    return None


def _station_shape(size):
    if size == 1:
        return {(0, 0)}
    if size == 2:
        return {(0, 0), (0, 1)} if random.random() < 0.5 else {(0, 0), (1, 0)}
    return {(0, 0), (0, 1), (1, 0), (1, 1)}


def _unique_region_colors(regions):
    color_sets = [set(reg["dirs"]) for reg in regions]
    unique_colors = []
    for i, colors in enumerate(color_sets):
        others = set()
        for j, other_colors in enumerate(color_sets):
            if i != j:
                others |= other_colors
        diff = sorted(colors - others)
        if not diff:
            return None
        unique_colors.append(diff)
    return unique_colors


def _marker_specs(dirs, required_colors=()):
    items = list(dirs.items())
    required_colors = set(required_colors)

    def has_required(spec):
        return not required_colors or any(color in required_colors for color, _ in spec)

    pairs = []
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            if (a[1] - b[1]) % 4 in (1, 3) and has_required((a, b)):
                pairs.append((a, b))
    triples = [tuple(spec) for spec in combinations(items, 3) if has_required(spec)]
    if triples and random.random() < THREE_MARKER_PROB:
        return list(random.choice(triples))
    if pairs:
        return list(random.choice(pairs))
    if triples:
        return list(random.choice(triples))
    fallback = [tuple(spec) for spec in combinations(items, min(2, len(items))) if has_required(spec)]
    assert fallback, "region must provide at least one required marker color"
    return list(random.choice(fallback))


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


def _build_docked_patch(bg, dirs, size, coupler_color, required_colors=()):
    couplers = _station_shape(size)
    specs = _marker_specs(dirs, required_colors)
    markers, marker_cells = [], set()
    for color, direction in specs:
        p = _marker_cell(couplers, direction)
        if p in couplers or p in marker_cells:
            return None
        markers.append((p, color, direction))
        marker_cells.add(p)

    blocked = set(couplers) | marker_cells
    contacts = [
        q
        for r, c in couplers
        for dr, dc in DIRS4
        if (q := (r + dr, c + dc)) not in blocked
    ]
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
        patch[r - r0, c - c0] = int(coupler_color)
        rel_couplers.append((r - r0, c - c0))
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
    H, W = base.shape
    output = base.copy()
    occupied = output != bg

    if len(regions) != 2:
        return None
    unique_colors = _unique_region_colors(regions)
    if unique_colors is None:
        return None

    repeated_size = random.choice(STATION_SIZES)
    stripe_colors = {int(color) for reg in regions for color in reg["dirs"]}
    coupler_color_pool = sorted(set(COLOR_POOL) - {int(bg)} - stripe_colors)
    if len(coupler_color_pool) < len(regions):
        return None
    coupler_colors = random.sample(coupler_color_pool, len(regions))

    trains = []
    region_ids = list(range(len(regions)))
    random.shuffle(region_ids)
    for idx, coupler_color in zip(region_ids, coupler_colors):
        reg = regions[idx]
        for _ in range(DOCK_RETRIES):
            result = _build_docked_patch(bg, reg["dirs"], repeated_size, coupler_color, unique_colors[idx])
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
                    "station": station_abs,
                    "coupler_color": int(coupler_color),
                    "station_size": repeated_size,
                }
            )
            break
        else:
            return None
    return output, trains


def _scatter_trains(base, bg, regions, trains):
    H, W = base.shape
    input_grid = base.copy()
    station_mask = np.zeros((H, W), dtype=bool)
    for t in trains:
        for r, c in t["station"]:
            input_grid[r, c] = t["coupler_color"]
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


def _input_station_signature(grid):
    bg = get_bg(grid)
    H, W = grid.shape

    def edge_and_internal_stripes():
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
        r0, r1, c0, c1 = top_h, H - bot_h, left_w, W - right_w
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

    stripe_mask, stripe_colors = edge_and_internal_stripes()
    infos = []
    for color in range(10):
        if color in (bg, 8) or color in stripe_colors:
            continue
        sm = np.zeros((H, W), dtype=bool)
        for r in range(H):
            for c in range(W):
                if int(grid[r, c]) != color:
                    continue
                ok = True
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if not (0 <= rr < H and 0 <= cc < W):
                            continue
                        if int(grid[rr, cc]) == 8:
                            ok = False
                            break
                    if not ok:
                        break
                if ok and not stripe_mask[r, c]:
                    sm[r, c] = True
        lab, n = label(sm, structure=np.ones((3, 3), dtype=int))
        for k in range(1, n + 1):
            infos.append({"color": int(color), "size": int((lab == k).sum())})
    return infos


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

    infos = _input_station_signature(input_grid)
    size_counts = Counter(info["size"] for info in infos)
    if not infos or max(size_counts.values(), default=0) < 2:
        return None
    if len({info["color"] for info in infos}) < 2:
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
