# Combo variant of 6e4f6532 (index 23)
# combo_spec: D4/2+D15/2
# Name: 负空间车站+对接后扩展

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg, paste


# Design analysis for the fused variant:
# - Concrete behavior: the original target was an isolated 9-cluster. Here the target
#   is a tiny bg cavity (sizes 1/2/4, same shape family as the couplers) enclosed by a
#   small monochrome frame painted with one of the region stripe colors. A train still
#   matches by coupler count and still orients via stripe-color markers, but the couplers
#   now land on background hole cells instead of pre-existing 9 cells.
# - Why the station frame is cleared: a fully enclosed negative-space hole cannot keep its
#   ring and also host an attached 8-body, so docking must replace the local framed
#   station structure. The solver therefore deletes the matched frame component, paints the
#   transformed train, fills the hole with 9, then adds one extra 9 halo around the 8-body.
# - Complexity calibration: the hole sizes remain the original {1,2,4}, one per train, so
#   matching stays low-branching and visually recoverable. The new halo is always present
#   and produces extra 9 cells beyond the coupler area, making the fused behavior visible.
# - New assumptions and generator support:
#   1. Every cavity is sealed by a single stripe color and is the only small enclosed bg
#      component of that shape family.
#   2. The station frame is not 8-connected to structural border/internal stripes even if
#      it reuses a stripe color.
#   3. Each docked train plus its halo fits inside the target region, and its bbox reserve
#      does not overlap other trains or other station structures.
#   4. Scattered source trains are kept away from station frames so they do not create fake
#      enclosed cavities or ambiguous overlaps.


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
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
ALLOWED_HOLE_SHAPES = {
    frozenset({(0, 0)}),
    frozenset({(0, 0), (0, 1)}),
    frozenset({(0, 0), (1, 0)}),
    frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
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


def _normalize_shape(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0, c0 = min(rs), min(cs)
    return frozenset((int(r - r0), int(c - c0)) for r, c in coords)


def _dilate8(mask: np.ndarray) -> np.ndarray:
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


def _bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _dims(coords):
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock each train onto a framed negative-space station, then grow a 9 halo around the docked body.

    Input structure:
    - bg; colored border/internal stripes split the canvas into rectangular regions.
    - Color 8 is the train body; color 9 appears only on train couplers in the source state.
    - Target stations are not 9-clusters anymore: each one is a tiny bg cavity of size 1, 2, or 4,
      sealed by a small stripe-colored frame inside a region. Stripe-colored marker pixels still sit on trains.

    Rule:
    - Detect the sealed bg cavities and match each train to the unique cavity whose area matches the train's
      coupler count. Use the train markers versus region boundary stripe colors to pick the docking orientation.
    - Erase the scattered train, erase the matched station frame, translate the oriented train so its couplers
      land exactly on the cavity cells, and paint those couplers as 9.
    - After docking, keep the train body/markers and add one extra 9 layer on the bg cells in the 8-neighborhood
      of the docked 8-body, creating an outward halo around the body silhouette.

    Key insight:
    - The target is now negative space: the only small enclosed bg holes act like hidden stations. Coupler count
      finds which hole to use, marker colors still determine the orientation, and the final output is distinguished
      by the extra 9 shell that appears only after docking.
    """
    inp = grid
    grid = inp.copy()
    H, W = grid.shape
    bg = int(get_bg(grid))

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

    def region_for_coords(coords, regions):
        matches = []
        for rr in regions:
            if all(rr["r0"] <= r <= rr["r1"] and rr["c0"] <= c <= rr["c1"] for r, c in coords):
                matches.append(rr)
        assert len(matches) == 1, "Each object must belong to exactly one region"
        return matches[0]

    def bfs_color_component(seed, color):
        q = deque([seed])
        seen = {seed}
        while q:
            r, c = q.popleft()
            for dr, dc in DIRS8:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in seen or int(grid[nr, nc]) != color:
                    continue
                seen.add((nr, nc))
                q.append((nr, nc))
        return seen

    def cavity_stations(stripe_colors, regions):
        bg_mask = grid == bg
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
        lab, n = label(bg_mask, structure=struct)
        stations = []
        for k in range(1, n + 1):
            coords = [(int(r), int(c)) for r, c in np.argwhere(lab == k)]
            if len(coords) not in (1, 2, 4):
                continue
            norm = _normalize_shape(coords)
            if norm not in ALLOWED_HOLE_SHAPES:
                continue
            cset = set(coords)
            boundary_cells = []
            boundary_colors = set()
            sealed = True
            for r, c in coords:
                for dr, dc in DIRS8:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        sealed = False
                        break
                    if (nr, nc) in cset:
                        continue
                    if int(grid[nr, nc]) == bg:
                        sealed = False
                        break
                    boundary_colors.add(int(grid[nr, nc]))
                    boundary_cells.append((nr, nc))
                if not sealed:
                    break
            if not sealed or len(boundary_colors) != 1:
                continue
            frame_color = next(iter(boundary_colors))
            if frame_color not in stripe_colors:
                continue
            seed = next((pt for pt in boundary_cells if int(grid[pt[0], pt[1]]) == frame_color), None)
            assert seed is not None
            frame_coords = bfs_color_component(seed, frame_color)
            if any(stripe_mask[r, c] for r, c in frame_coords):
                continue
            reg = region_for_coords(coords, regions)
            if not all(reg["r0"] <= r <= reg["r1"] and reg["c0"] <= c <= reg["c1"] for r, c in frame_coords):
                continue
            stations.append(
                {
                    "hole": coords,
                    "frame": frame_coords,
                    "frame_color": int(frame_color),
                    "region": reg,
                    "clear_coords": set(coords) | set(frame_coords),
                }
            )
        return stations

    def xform_coords(coords, shape, flip_ud, rot_k):
        result = transform_coords(coords, shape, flip_ud, rot_k)
        new_shape = apply_transform(np.zeros(shape, dtype=bool), flip_ud, rot_k).shape
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
                    transform_coords(
                        [(int(round(ref_r)), int(round(ref_c)))],
                        (H0, W0),
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
                            (H0, W0),
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

    stations = cavity_stations(stripe_colors, regions)
    stations_by_size = defaultdict(list)
    all_station_support = set()
    for st in stations:
        stations_by_size[len(st["hole"])].append(st)
        all_station_support |= st["clear_coords"]

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
        body8, body9, marker_pixels = [], [], []
        for r, c in abs_coords:
            v = int(grid[r, c])
            if v == 8:
                body8.append((int(r - r0), int(c - c0)))
            elif v == 9:
                body9.append((int(r - r0), int(c - c0)))
            if v in stripe_colors:
                marker_pixels.append((v, int(r - r0), int(c - c0)))
        trains.append(
            {
                "abs_coords": abs_coords,
                "bbox": bbox,
                "body8": body8,
                "body9": body9,
                "marker_pixels": marker_pixels,
            }
        )

    out = grid.copy()
    placed_mask = np.zeros((H, W), dtype=bool)
    for t in trains:
        for r, c in t["abs_coords"]:
            out[r, c] = bg

    for t in trains:
        n9 = len(t["body9"])
        assert n9 > 0, "Each train must carry at least one coupler"
        candidate_stations = stations_by_size.get(n9)
        assert candidate_stations is not None and len(candidate_stations) == 1, (
            f"Expected one cavity station for coupler count {n9}"
        )
        st = candidate_stations[0]
        reg = st["region"]
        divider_dirs = boundary_dirs_for_region(reg, stripe_colors)
        op = choose_best_op(t, divider_dirs)
        tbbox = apply_transform(t["bbox"], *op)
        t_body9, (H1, W1) = xform_coords(t["body9"], t["bbox"].shape, *op)
        t_body8, _ = xform_coords(t["body8"], t["bbox"].shape, *op)
        body9_set = set(t_body9)
        body8_set = set(t_body8)
        hole_set = set(st["hole"])

        matching_offsets = {
            (sr - br, sc - bc)
            for br, bc in body9_set
            for sr, sc in hole_set
            if all((r + sr - br, c + sc - bc) in hole_set for r, c in body9_set)
        }
        assert len(matching_offsets) == 1, "Couplers must align to a unique cavity offset"
        off_r, off_c = next(iter(matching_offsets))

        train_abs = {}
        for r in range(H1):
            for c in range(W1):
                v = int(tbbox[r, c])
                if v == bg:
                    continue
                rr, cc = r + off_r, c + off_c
                train_abs[(rr, cc)] = v

        body_abs = {(r + off_r, c + off_c) for r, c in body8_set}
        shell_abs = set()
        for r, c in body_abs:
            for dr, dc in DIRS8:
                q = (r + dr, c + dc)
                if q not in body_abs and q not in train_abs:
                    shell_abs.add(q)

        allowed_station = st["clear_coords"]
        for rr, cc in list(train_abs) + list(shell_abs):
            if not (reg["r0"] <= rr <= reg["r1"] and reg["c0"] <= cc <= reg["c1"]):
                raise AssertionError("Docked train or halo must stay inside its target region")
            if stripe_mask[rr, cc]:
                raise AssertionError("Docked train or halo cannot overlap structural stripes")
            if placed_mask[rr, cc]:
                raise AssertionError("Placed trains cannot overlap")
            if (rr, cc) in all_station_support and (rr, cc) not in allowed_station:
                raise AssertionError("Docked train cannot overlap another station structure")

        for r, c in allowed_station:
            out[r, c] = bg

        for (rr, cc), v in train_abs.items():
            out[rr, cc] = v
            placed_mask[rr, cc] = True
        for rr, cc in shell_abs:
            if out[rr, cc] == bg:
                out[rr, cc] = 9
            placed_mask[rr, cc] = True

    return out


# --------------- generator (fused variant) ---------------

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
DOCK_RETRIES = 30
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


def _adjacent8(p, cells):
    r, c = p
    return any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in cells)


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
    rel_couplers = []
    for r, c in body:
        patch[r - r0, c - c0] = 8
    for (r, c), color, _ in markers:
        patch[r - r0, c - c0] = color
    for r, c in couplers:
        patch[r - r0, c - c0] = 9
        rel_couplers.append((r - r0, c - c0))
    return patch, rel_couplers


def _expand_patch_with_halo(patch, coupler_rel, bg):
    h, w = patch.shape
    out = np.full((h + 2, w + 2), bg, dtype=int)
    out[1 : h + 1, 1 : w + 1] = patch
    body_mask = np.zeros_like(out, dtype=bool)
    body_mask[1 : h + 1, 1 : w + 1] = patch == 8
    occupied = out != bg
    shell = _dilate8(body_mask) & (~body_mask) & (~occupied)
    out[shell] = 9
    shifted = [(int(r + 1), int(c + 1)) for r, c in coupler_rel]
    return out, shifted


def _station_frame_coords(coupler_abs):
    r0, c0, r1, c1 = _bbox(coupler_abs)
    frame = set()
    for r in range(r0 - 1, r1 + 2):
        for c in range(c0 - 1, c1 + 2):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            frame.add((int(r), int(c)))
    return frame


def _bbox_locations(region, shape, blocked, margin=0):
    H, W = blocked.shape
    ph, pw = shape
    locs = []
    r_start = max(region["r0"] + margin, 0)
    r_stop = min(region["r1"] - margin - ph + 2, H - ph + 1)
    c_start = max(region["c0"] + margin, 0)
    c_stop = min(region["c1"] - margin - pw + 2, W - pw + 1)
    for rr in range(r_start, r_stop):
        for cc in range(c_start, c_stop):
            if not blocked[rr : rr + ph, cc : cc + pw].any():
                locs.append((rr, cc))
    return locs


def _place_docked_trains(base, bg, regions):
    H, W = base.shape
    output = base.copy()
    reserved = base != bg

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
            expanded_patch, coupler_rel_out = _expand_patch_with_halo(patch, coupler_rel, bg)
            locs = _bbox_locations(reg, expanded_patch.shape, reserved, 0)
            random.shuffle(locs)
            chosen = None
            for r0, c0 in locs:
                coupler_abs = [(r0 + r, c0 + c) for r, c in coupler_rel_out]
                frame = _station_frame_coords(coupler_abs)
                if not all(reg["r0"] <= r <= reg["r1"] and reg["c0"] <= c <= reg["c1"] for r, c in frame):
                    continue
                chosen = (r0, c0, coupler_abs, frame)
                break
            if chosen is None:
                continue
            r0, c0, coupler_abs, frame = chosen
            paste(output, expanded_patch, r0, c0, transparent=bg)
            ph, pw = expanded_patch.shape
            reserved[r0 : r0 + ph, c0 : c0 + pw] = True
            trains.append(
                {
                    "patch": patch,
                    "output_patch": expanded_patch,
                    "station": coupler_abs,
                    "frame": frame,
                    "frame_color": random.choice(list(reg["dirs"].keys())),
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
        for r, c in t["frame"]:
            input_grid[r, c] = int(t["frame_color"])
            station_mask[r, c] = True
        for r, c in t["station"]:
            input_grid[r, c] = bg
            station_mask[r, c] = True

    valid_ops = []
    for t in trains:
        valid = _valid_ops_for_patch(t["patch"])
        if not valid:
            return None
        valid_ops.append(valid)
    ops = [random.choice(valid) for valid in valid_ops]
    if all(op == (False, 0) for op in ops):
        non_id = [op for valid in valid_ops for op in valid if op != (False, 0)]
        if not non_id:
            return None
        ops[0] = random.choice([op for op in valid_ops[0] if op != (False, 0)] or valid_ops[0])

    forbidden = (base != bg) | _dilate8(station_mask)
    for t, op in zip(trains, ops):
        in_patch = _inverse_patch(t["patch"], op)
        regs = regions[:]
        random.shuffle(regs)
        all_locs = []
        for reg in regs:
            locs = _bbox_locations(reg, in_patch.shape, forbidden, 1)
            all_locs.extend(locs)
        if not all_locs:
            return None
        r0, c0 = random.choice(all_locs)
        paste(input_grid, in_patch, r0, c0, transparent=bg)
        ph, pw = in_patch.shape
        forbidden[r0 : r0 + ph, c0 : c0 + pw] = True

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
    raise RuntimeError("failed to generate valid fused 6e4f6532 input")
