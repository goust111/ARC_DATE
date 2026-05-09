# Combo variant of 6e4f6532 (index 12)
# combo_spec: D1/2+D12/1
# Name: 面积排序匹配+多对配对

from __future__ import annotations

"""
Design analysis for the combo variant: body-area ordering + 4-5 pairings.

a) Concrete behavior trace
   - Source task: each train used its coupler-count to find the one station with the same
     number of 9-cells, then marker colors fixed the rotation/flip.
   - Combo task: 4-5 trains are present, several stations deliberately repeat the same size,
     and trains are paired by ordinal rank instead:
       1. isolate the station groups (9-clusters not touching any 8),
       2. sort trains by the number of body-color 8 cells, descending,
       3. sort stations by station size, descending, breaking ties by visible position,
       4. zip the two ordered lists and dock each train onto its ranked station.
   - Example sketch: if train body areas are [19, 16, 13, 9, 7] and station sizes are
     [5, 5, 3, 2, 2], then the largest train goes to the upper-left size-5 station, the
     next-largest goes to the later size-5 station, then size-3, then the two size-2
     stations in their visual order.

b) Complexity calibration
   - The matching rule stays one-dimensional and visible: compare body area vs station size.
   - Ambiguity is intentional but controlled: repeated station sizes force ranking, while
     distinct train body areas keep the train-side order deterministic.
   - Station sizes are still drawn from the small set {1,2,3,4,5}, so the visual system
     is learnable across 2-4 ARC examples.

c) Hidden-assumption audit
   - Original `len(candidate_groups) == 1` no longer holds; solver must stop assuming
     coupler-count uniqueness.
   - The docking geometry assumption still holds: a train's coupler shape must be congruent
     to its destination station shape after the chosen transform.
   - Equal-sized stations need a stable tie-break, so both solver and generator use the same
     canonical order key based on size first, then top-to-bottom / left-to-right coordinates.
   - Trains must have distinct body areas; generator assigns exact target areas and verifies
     them after placement.

d) Placement legality
   - Generation remains output-first: place fully docked trains first, then scatter inverse
     transforms into the input.
   - Larger canvases and higher retry budgets are required because 4-5 docked patches,
     4-5 scattered patches, and the preserved station groups must all fit without overlap
     or stripe contact.

e) Visual observability
   - A human can count 8-cells per train and 9-cells per station, then rank both lists.
   - Repeated size groups make it obvious that count-matching alone is insufficient.
   - Marker colors still provide the same visible orientation cue as the source task.

f) Visual cleanliness
   - Station shapes for sizes 3 and 5 are compact (L tromino and plus pentomino) to avoid
     long thin artifacts when many objects share the canvas.
   - Clearance from stripes and dilation-based spacing keep trains visually separated.

g) Change inferability
   - No hidden color ordering is introduced; only visible quantities (body area, station
     size, and spatial order among equal-sized stations) control the new pairing.
"""

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg, paste


def apply_transform(arr: np.ndarray, flip_ud: bool = False, rot_k: int = 0) -> np.ndarray:
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def transform_coords(coords, shape, flip_ud: bool = False, rot_k: int = 0):
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    out = apply_transform(mask, flip_ud=flip_ud, rot_k=rot_k)
    return set(map(tuple, np.argwhere(out)))


def _station_order_key(coords) -> tuple:
    pts = tuple(sorted((int(r), int(c)) for r, c in coords))
    return (-len(pts), pts[0][0], pts[0][1], pts)


def _solve(grid: np.ndarray, pairing_mode: str) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    h, w = grid.shape
    bg = int(get_bg(grid))

    def find_edge_stripes():
        stripe = np.zeros((h, w), dtype=bool)
        colors = set()

        top_h = 0
        for r in range(h):
            row = grid[r, :]
            if (row == row[0]).all() and int(row[0]) != bg:
                top_h += 1
            else:
                break
        if top_h:
            stripe[:top_h, :] = True
            colors.add(int(grid[0, 0]))

        bot_h = 0
        for r in range(h - 1, -1, -1):
            row = grid[r, :]
            if (row == row[0]).all() and int(row[0]) != bg:
                bot_h += 1
            else:
                break
        if bot_h:
            stripe[h - bot_h :, :] = True
            colors.add(int(grid[h - 1, 0]))

        inner_r0, inner_r1 = top_h, h - bot_h
        left_w = 0
        for c in range(w):
            col = grid[inner_r0:inner_r1, c]
            if col.size == 0:
                break
            if (col == col[0]).all() and int(col[0]) != bg:
                left_w += 1
            else:
                break
        if left_w:
            stripe[inner_r0:inner_r1, :left_w] = True
            colors.add(int(grid[inner_r0, 0]))

        right_w = 0
        for c in range(w - 1, -1, -1):
            col = grid[inner_r0:inner_r1, c]
            if col.size == 0:
                break
            if (col == col[0]).all() and int(col[0]) != bg:
                right_w += 1
            else:
                break
        if right_w:
            stripe[inner_r0:inner_r1, w - right_w :] = True
            colors.add(int(grid[inner_r0, w - 1]))

        return stripe, colors, (top_h, bot_h, left_w, right_w)

    def find_internal_stripes(edge_info):
        top_h, bot_h, left_w, right_w = edge_info
        r0, r1, c0, c1 = top_h, h - bot_h, left_w, w - right_w
        stripe = np.zeros((h, w), dtype=bool)
        colors = set()

        c = c0
        while c < c1:
            col = grid[r0:r1, c]
            if col.size and (col == col[0]).all() and int(col[0]) != bg:
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
            if row.size and (row == row[0]).all() and int(row[0]) != bg:
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
        sm = np.zeros((h, w), dtype=bool)
        for r in range(h):
            for c in range(w):
                if int(grid[r, c]) != 9:
                    continue
                ok = all(
                    int(grid[r + dr, c + dc]) != 8
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if (dr != 0 or dc != 0) and 0 <= r + dr < h and 0 <= c + dc < w
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
        lab, n = label(~stripe_mask)
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
            mc = Counter(grid[r0 - 1, c0 : c1 + 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 0
        if r1 + 1 < h:
            mc = Counter(grid[r1 + 1, c0 : c1 + 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 2
        if c0 - 1 >= 0:
            mc = Counter(grid[r0 : r1 + 1, c0 - 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 3
        if c1 + 1 < w:
            mc = Counter(grid[r0 : r1 + 1, c1 + 1].tolist()).most_common(1)[0][0]
            if mc in stripe_colors:
                dirs[int(mc)] = 1
        return dirs

    def xform_coords(coords, shape, flip_ud, rot_k):
        result = transform_coords(coords, shape, flip_ud, rot_k)
        new_shape = apply_transform(np.zeros(shape, dtype=int), flip_ud, rot_k).shape
        return list(result), new_shape

    def choose_best_op(train, divider_dirs):
        bbox0 = train["bbox"]
        h0, w0 = bbox0.shape
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
                    transform_coords(
                        [(int(round(ref_r)), int(round(ref_c)))], (h0, w0), flip_ud, k
                    )
                )
                rr_ref, cc_ref = ref_t[0]
                score = 0
                for color, pts in markers_by_color.items():
                    mr = sum(r for r, _ in pts) / len(pts)
                    mc = sum(c for _, c in pts) / len(pts)
                    m_t = list(
                        transform_coords(
                            [(int(round(mr)), int(round(mc)))], (h0, w0), flip_ud, k
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

    raw_groups = station_groups()
    station_set = set(pt for g in raw_groups for pt in g)
    group_infos = []
    for g in raw_groups:
        matching_regions = [
            rr
            for rr in regions
            if all(rr["r0"] <= r <= rr["r1"] and rr["c0"] <= c <= rr["c1"] for r, c in g)
        ]
        assert len(matching_regions) == 1, "Each station group must belong to exactly one region"
        pts = tuple(sorted(g))
        group_infos.append(
            {
                "coords": pts,
                "set": set(pts),
                "size": len(pts),
                "sort_key": _station_order_key(pts),
                "region": matching_regions[0],
            }
        )

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
        body_area = 0
        for r, c in abs_coords:
            v = int(grid[r, c])
            if v == 8:
                body_area += 1
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
                "body_area": body_area,
                "anchor": (int(r0), int(c0)),
            }
        )

    out = grid.copy()
    for t in trains:
        for r, c in t["abs_coords"]:
            out[r, c] = bg

    if pairing_mode == "count":
        groups_by_size = defaultdict(list)
        for g in group_infos:
            groups_by_size[g["size"]].append(g)
        pairings = []
        for t in trains:
            n9 = len(t["body9"])
            assert n9 > 0, "Each train must carry at least one coupler"
            candidate_groups = groups_by_size.get(n9)
            assert candidate_groups is not None and len(candidate_groups) == 1, (
                f"Expected one station group for coupler count {n9}"
            )
            pairings.append((t, candidate_groups[0]))
    else:
        ordered_trains = sorted(
            trains,
            key=lambda t: (
                -int(t["body_area"]),
                int(t["anchor"][0]),
                int(t["anchor"][1]),
                len(t["body9"]),
            ),
        )
        ordered_groups = sorted(group_infos, key=lambda g: g["sort_key"])
        assert len(ordered_trains) == len(ordered_groups), "Train/station counts must match"
        pairings = list(zip(ordered_trains, ordered_groups))

    for t, g in pairings:
        gset = g["set"]
        reg = g["region"]
        divider_dirs = boundary_dirs_for_region(reg, stripe_colors)
        op = choose_best_op(t, divider_dirs)
        tbbox = apply_transform(t["bbox"], *op)
        t_body9, (h1, w1) = xform_coords(t["body9"], t["bbox"].shape, *op)
        body9_set = set(t_body9)
        matching_offsets = {
            (sr - br, sc - bc)
            for br, bc in body9_set
            for sr, sc in gset
            if all((r + sr - br, c + sc - bc) in gset for r, c in body9_set)
        }
        assert len(matching_offsets) == 1, "Couplers must align to a unique station-group offset"
        off_r, off_c = next(iter(matching_offsets))

        ok = True
        for r in range(h1):
            for c in range(w1):
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

        for r in range(h1):
            for c in range(w1):
                v = int(tbbox[r, c])
                if v != bg and v != 9:
                    out[r + off_r, c + off_c] = v

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock 4-5 scattered trains onto 4-5 stations by ranking train body area against station size.

    Input structure:
    - Solid edge and optional internal stripes split the background into one or two rectangular regions.
    - Color 8 cells form train bodies; color 9 appears both as isolated station clusters and as train couplers.
    - Each train also carries 2-3 stripe-colored marker pixels that reference the boundary colors of its target region.

    Rule:
    - Detect station groups as 9-clusters that are not 8-adjacent, and detect trains as connected components containing 8.
    - Sort trains by total count of body-color 8 cells (largest first); sort station groups by their 9-count (largest first), breaking equal-size ties by canonical top-to-bottom / left-to-right order.
    - Pair the two ordered lists, rotate/flip each train so its markers face the matching boundary stripes of that station's region, then translate it so the couplers coincide with the station cells.

    Key insight: Coupler counts are no longer unique when 4-5 pairings share repeated station sizes, so the target is identified by visible rank:
    larger 8-bodies dock to larger 9-stations, while marker colors still determine the docking orientation.
    """
    return _solve(grid, pairing_mode="area")


GRID_H_RANGE = (24, 43)
GRID_W_RANGE = (28, 49)
GRID_MIN_AREA = 800
MIN_REGION_DIM = 16
MIN_REGION_DIRS = 3
INTERNAL_STRIPE_THICK = 2

STRIPE_THICK_RANGE = (2, 4)
INTERNAL_PROB = 0.68

STATION_SIZE_ORDERS = [
    (5, 5, 3, 1),
    (5, 4, 4, 2),
    (5, 3, 3, 2),
    (4, 4, 2, 1),
    (5, 5, 4, 2, 1),
    (5, 5, 3, 3, 1),
    (5, 4, 4, 2, 2),
    (5, 3, 3, 2, 1),
    (4, 4, 3, 2, 1),
    (4, 4, 2, 2, 1),
]
STATION_SIZE_ORDERS_BY_LEN = {
    pair_count: [order for order in STATION_SIZE_ORDERS if len(order) == pair_count]
    for pair_count in (4, 5)
}
BODY_AREA_CHOICES = [7, 9, 11, 13, 15, 17, 19, 20]

BODY_MIN = 5
BODY_MAX = 20
PATCH_H_RANGE = (4, 9)
PATCH_W_RANGE = (3, 8)

MARKER_DIST_RANGE = (2, 3)
MARKER_JITTER = [-1, 0, 1]
THREE_MARKER_PROB = 0.65

BODY_GROWTH_ITERS = 180
PATH_BOUNDS_PAD = 4

LAYOUT_RETRIES = 320
DOCK_RETRIES = 80
GENERATE_RETRIES = 900

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


def _assign_region_dirs(regions, edge_col, top, bottom, left, right, h, w, internal):
    for reg in regions:
        dirs = {}
        if top and reg["r0"] == top:
            dirs[edge_col["top"]] = 0
        if bottom and reg["r1"] == h - bottom - 1:
            dirs[edge_col["bottom"]] = 2
        if left and reg["c0"] == left:
            dirs[edge_col["left"]] = 3
        if right and reg["c1"] == w - right - 1:
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
        for h in range(*GRID_H_RANGE):
            for w in range(*GRID_W_RANGE):
                if h * w < GRID_MIN_AREA:
                    continue
                ih, iw = h - top - bottom, w - left - right
                if ih < MIN_REGION_DIM or iw < MIN_REGION_DIM:
                    continue
                if orient == "v" and iw < min_inner_for_split:
                    continue
                if orient == "h" and ih < min_inner_for_split:
                    continue
                candidates.append((h, w))
        if not candidates:
            continue

        h, w = random.choice(candidates)
        pool = COLOR_POOL[:]
        bg = random.choice(pool)
        pool.remove(bg)
        stripe_cols = random.sample(pool, len(edges) + int(use_internal))
        edge_col = dict(zip(edges, stripe_cols[: len(edges)]))
        internal_col = stripe_cols[-1] if use_internal else None
        grid = np.full((h, w), bg, dtype=int)

        if top:
            grid[:top, :] = edge_col["top"]
        if bottom:
            grid[h - bottom :, :] = edge_col["bottom"]
        if left:
            grid[top : h - bottom, :left] = edge_col["left"]
        if right:
            grid[top : h - bottom, w - right :] = edge_col["right"]

        inner = {"r0": top, "r1": h - bottom - 1, "c0": left, "c1": w - right - 1}
        regions, internal = [], None
        if orient == "v":
            lo = left + MIN_REGION_DIM
            hi = w - right - MIN_REGION_DIM - INTERNAL_STRIPE_THICK
            if lo > hi:
                continue
            c = random.randint(lo, hi)
            grid[top : h - bottom, c : c + INTERNAL_STRIPE_THICK] = internal_col
            internal = ("v", c, internal_col)
            regions = [
                {"r0": top, "r1": h - bottom - 1, "c0": left, "c1": c - 1},
                {
                    "r0": top,
                    "r1": h - bottom - 1,
                    "c0": c + INTERNAL_STRIPE_THICK,
                    "c1": w - right - 1,
                },
            ]
        elif orient == "h":
            lo = top + MIN_REGION_DIM
            hi = h - bottom - MIN_REGION_DIM - INTERNAL_STRIPE_THICK
            if lo > hi:
                continue
            r = random.randint(lo, hi)
            grid[r : r + INTERNAL_STRIPE_THICK, left : w - right] = internal_col
            internal = ("h", r, internal_col)
            regions = [
                {"r0": top, "r1": r - 1, "c0": left, "c1": w - right - 1},
                {
                    "r0": r + INTERNAL_STRIPE_THICK,
                    "r1": h - bottom - 1,
                    "c0": left,
                    "c1": w - right - 1,
                },
            ]
        else:
            regions = [inner]

        if _assign_region_dirs(regions, edge_col, top, bottom, left, right, h, w, internal):
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
        if len(body) >= target:
            break
        all_cells = body | blocked
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

        hh, ww = _dims(all_cells)
        if hh < PATCH_H_RANGE[0]:
            r0, _, r1, _ = _bbox(all_cells)
            exp = [q for q in cands if q[0] < r0 or q[0] > r1]
            cands = exp or cands
        elif ww < PATCH_W_RANGE[0]:
            _, c0, _, c1 = _bbox(all_cells)
            exp = [q for q in cands if q[1] < c0 or q[1] > c1]
            cands = exp or cands
        body.add(random.choice(cands))
    return len(body) == target


def _build_docked_patch(bg, dirs, size, body_target):
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
    if len(body) > body_target or len(body) > BODY_MAX:
        return None
    if not _grow_body(body, blocked, body_target):
        return None

    all_cells = body | set(couplers) | marker_cells
    hh, ww = _dims(all_cells)
    if not (
        BODY_MIN <= len(body) <= BODY_MAX
        and PATCH_H_RANGE[0] <= hh <= PATCH_H_RANGE[1]
        and PATCH_W_RANGE[0] <= ww <= PATCH_W_RANGE[1]
    ):
        return None

    r0, c0, _, _ = _bbox(all_cells)
    patch = np.full((hh, ww), bg, dtype=int)
    for r, c in body:
        patch[r - r0, c - c0] = 8
    for (r, c), color, _ in markers:
        patch[r - r0, c - c0] = color
    rel_couplers = []
    for r, c in couplers:
        patch[r - r0, c - c0] = 9
        rel_couplers.append((r - r0, c - c0))
    return patch, rel_couplers


def _patch_locations(region, patch, bg, occupied, coupler_rel=None, station_clearance=False, margin=0):
    h, w = occupied.shape
    ph, pw = patch.shape
    coords = [(int(r), int(c)) for r, c in np.argwhere(patch != bg)]
    locs = []
    r_start = max(region["r0"] + margin, 0)
    r_stop = min(region["r1"] - margin - ph + 2, h - ph + 1)
    c_start = max(region["c0"] + margin, 0)
    c_stop = min(region["c1"] - margin - pw + 2, w - pw + 1)
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


def _station_key_from_rel(loc, coupler_rel):
    r0, c0 = loc
    return _station_order_key([(r0 + r, c0 + c) for r, c in coupler_rel])


def _place_docked_trains(base, bg, regions, size_order, body_areas):
    output = base.copy()
    occupied = output != bg
    trains = []
    last_key_by_size = {}

    for idx, (size, body_area) in enumerate(zip(size_order, body_areas)):
        for _ in range(DOCK_RETRIES):
            reg = random.choice(regions)
            result = _build_docked_patch(bg, reg["dirs"], size, body_area)
            if result is None:
                continue
            patch, coupler_rel = result
            locs = _patch_locations(reg, patch, bg, _dilate(occupied), coupler_rel, True, 0)
            if not locs:
                continue

            loc_records = []
            prev_key = last_key_by_size.get(size)
            for loc in locs:
                skey = _station_key_from_rel(loc, coupler_rel)
                if prev_key is not None and skey <= prev_key:
                    continue
                loc_records.append((skey, loc))
            if not loc_records:
                continue

            loc_records.sort(key=lambda item: item[0])
            same_size_later = size_order[idx + 1 :].count(size)
            if same_size_later:
                keep = max(1, len(loc_records) - same_size_later)
                skey, (r0, c0) = random.choice(loc_records[:keep])
            else:
                skey, (r0, c0) = random.choice(loc_records)

            paste(output, patch, r0, c0, transparent=bg)
            occupied |= _dilate((output != bg) & ~occupied) | (output != bg)
            station_abs = [(r0 + r, c0 + c) for r, c in coupler_rel]
            trains.append(
                {
                    "patch": patch,
                    "station": station_abs,
                    "body_area": body_area,
                    "size": size,
                }
            )
            last_key_by_size[size] = skey
            break
        else:
            return None

    ordered = sorted(trains, key=lambda t: _station_order_key(t["station"]))
    if [len(t["station"]) for t in ordered] != list(size_order):
        return None
    if [int(t["body_area"]) for t in ordered] != list(body_areas):
        return None
    return output, trains


def _scatter_trains(base, bg, regions, trains):
    h, w = base.shape
    input_grid = base.copy()
    station_mask = np.zeros((h, w), dtype=bool)
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
        patch_mask = np.zeros((h, w), dtype=bool)
        patch_mask[r0 : r0 + ph, c0 : c0 + pw] = in_patch != bg
        forbidden |= _dilate(patch_mask)

    return input_grid


def _count_solver_matches(input_grid, output):
    try:
        return np.array_equal(_solve(input_grid, pairing_mode="count"), output)
    except Exception:
        return False


def _gen_attempt(size_order, body_areas):
    layout = _layout()
    if layout is None:
        return None
    base, bg, regions = layout

    placed = _place_docked_trains(base, bg, regions, size_order, body_areas)
    if placed is None:
        return None
    output, trains = placed

    input_grid = _scatter_trains(base, bg, regions, trains)
    if input_grid is None:
        return None

    counts = Counter(map(int, input_grid.flatten()))
    if counts[bg] != max(counts.values()):
        return None
    if len(trains) not in (4, 5):
        return None
    if len(set(size_order)) == len(size_order):
        return None

    try:
        if not np.array_equal(main(input_grid), output):
            return None
    except Exception:
        return None
    if _count_solver_matches(input_grid, output):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        pair_count = random.choice((4, 5))
        size_order = list(random.choice(STATION_SIZE_ORDERS_BY_LEN[pair_count]))
        body_areas = sorted(random.sample(BODY_AREA_CHOICES, len(size_order)), reverse=True)
        grid = _gen_attempt(size_order, body_areas)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 6e4f6532 combo input")
