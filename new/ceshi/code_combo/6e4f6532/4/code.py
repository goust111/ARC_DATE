# Combo variant of 6e4f6532 (index 4)
# combo_spec: D1/0+D4/1
# Name: 颜色标签配对+车站携带方向标记
#
# Design analysis for the fused variant:
# a) Concrete behavior:
#    The original task used coupler count to choose the station and stripe-colored
#    train markers to choose orientation. Here those two decisions are moved onto
#    visible target-side cues: a train and its station share one label color for
#    pairing, while each station also carries one repeated-color indicator pixel
#    whose side specifies the docking direction.
# b) Small example trace:
#    Suppose two trains both carry exactly two couplers. In the original task the
#    counts would be ambiguous only if generation failed; in this combo that
#    ambiguity is intentional. Train A might carry label color 3 and Train B label
#    color 6, while two stations of size 2 carry adjacent labels 3 and 6. Solver
#    first matches by label, then reads each station's indicator side to decide
#    which way the matched train must face before its 9-couplers are aligned.
# c) Complexity calibration:
#    Station sizes are deliberately repeated within a sample, so count-only logic
#    cannot work. At the same time the color burden is kept small: one shared
#    indicator color for every station plus one unique label color per train.
# d) Original-assumption audit:
#    The base solver assumed exactly one station per coupler count and at least
#    one stripe-colored marker inside every train. Both assumptions are false now.
#    The fused solver therefore removes count lookup and stripe-marker scoring,
#    replacing them with label lookup and contact-side orientation.
# e) Placement legality:
#    The generator still works output-first. It builds each docked train so the
#    8-body touches the coupler core on exactly one orthogonal side, reserves one
#    station-adjacent indicator cell, reserves one separate station label cell,
#    and only then scatters inverse-transformed source trains.
# f) Visual observability:
#    Humans can now see three distinct target-side cues in each sample: the 9-core
#    station, one shared indicator color, and one matching label color. Because
#    station sizes repeat, the label really matters; because trains have no stripe
#    markers, the station indicator really matters.
# g) Visual cleanliness:
#    Indicator pixels all share one color and never overlap the train patch. Label
#    colors are unique per pair and are limited by the color budget. This keeps
#    the scene segmented instead of mixing many unrelated marker types on trains.
# h) New solver assumptions guaranteed by the generator:
#    - every station has exactly one adjacent indicator pixel of the shared color;
#    - every station has exactly one adjacent label pixel of a unique color;
#    - every train has exactly one attached label pixel of that same unique color;
#    - at least one station size repeats, so the original count rule is invalid;
#    - every train body touches its couplers on exactly one orthogonal side.

from __future__ import annotations

from collections import Counter
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


def _inverse_patch(patch, op):
    for inv in OPS:
        cand = apply_transform(patch, *inv)
        if np.array_equal(apply_transform(cand, *op), patch):
            return cand
    raise ValueError("no inverse transform")


def _valid_ops_for_patch(patch, h_range, w_range):
    hlo, hhi = h_range
    wlo, whi = w_range
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock label-matched trains onto labeled stations, orienting from station-side indicators.

    Input structure:
    - Colored edge/internal stripes still partition the canvas into rectangular regions.
    - Each train is one component containing color-8 body cells, color-9 couplers, and exactly
      one extra non-stripe label pixel; trains no longer carry stripe-colored direction markers.
    - Each station is an isolated 9-cluster with two adjacent helper pixels: one unique label
      color matching exactly one train, and one repeated-color indicator pixel shared by all
      stations in the sample.

    Rule:
    - Detect station groups as 9-clusters whose 8-neighborhood does not touch any 8-body cell.
    - Recover the shared indicator color from the station-side helper pixels, and use the other
      adjacent color as the station's pairing label.
    - Match each train to the station with the same label color, then rotate/flip the train until
      its unique body-to-coupler contact side faces the docking side implied by the station
      indicator. Finally translate so the coupler 9-cells land exactly on the station 9-cells.

    Key insight: pairing and orientation are both decided at the destination. The label color
    chooses which station a train belongs to, and the station's indicator side replaces the old
    stripe-marker cue for how that train must be docked.
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

    def helper_pixels(group):
        extras = {}
        for r, c in group["coords"]:
            for dr, dc in DIRS4:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if (rr, cc) in group["coord_set"] or stripe_mask[rr, cc]:
                    continue
                v = int(grid[rr, cc])
                if v in (bg, 8, 9):
                    continue
                extras[(rr, cc)] = v
        return [(rr, cc, v) for (rr, cc), v in extras.items()]

    def indicator_side(group, indicator_pos):
        rr, cc = indicator_pos
        r0, c0, r1, c1 = _bbox(group["coords"])
        if rr == r0 - 1 and c0 <= cc <= c1:
            return 0
        if rr == r1 + 1 and c0 <= cc <= c1:
            return 2
        if cc == c0 - 1 and r0 <= rr <= r1:
            return 3
        if cc == c1 + 1 and r0 <= rr <= r1:
            return 1
        raise AssertionError("Indicator must sit on one bbox side of its station")

    edge_mask, edge_colors, edge_info = find_edge_stripes()
    internal_mask, internal_colors = find_internal_stripes(edge_info)
    stripe_mask = edge_mask | internal_mask
    stripe_colors = edge_colors | internal_colors
    regions = regions_from_stripes(stripe_mask)

    groups = station_groups()
    all_helpers = [helper_pixels(g) for g in groups]
    helper_color_counts = Counter(v for extras in all_helpers for _, _, v in extras)
    indicator_colors = [color for color, cnt in helper_color_counts.items() if cnt == len(groups)]
    assert len(indicator_colors) == 1, "Expected one shared station-indicator color"
    indicator_color = indicator_colors[0]

    stations_by_label = {}
    station_set = set()
    for group, extras in zip(groups, all_helpers):
        assert len(extras) == 2, "Each station must have exactly one label and one indicator"
        indicator_hits = [(r, c) for r, c, v in extras if v == indicator_color]
        label_hits = [v for _, _, v in extras if v != indicator_color]
        assert len(indicator_hits) == 1 and len(label_hits) == 1, "Station helpers are malformed"
        label_color = int(label_hits[0])
        assert label_color not in stations_by_label, "Station label colors must be unique"
        matching_regions = [
            reg
            for reg in regions
            if all(reg["r0"] <= r <= reg["r1"] and reg["c0"] <= c <= reg["c1"] for r, c in group["coords"])
        ]
        assert len(matching_regions) == 1, "Each station group must belong to exactly one region"
        stations_by_label[label_color] = {
            "group": group["coords"],
            "region": matching_regions[0],
            "dock_dir": _opposite_dir(indicator_side(group, indicator_hits[0])),
        }
        station_set.update(group["coords"])

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
        label_colors = set()
        for r, c in abs_coords:
            v = int(grid[r, c])
            rr, cc = int(r - r0), int(c - c0)
            if v == 9 and (r, c) not in station_set:
                body9.append((rr, cc))
            elif v == 8:
                body8.append((rr, cc))
            elif v not in stripe_colors and v != indicator_color and v != bg:
                label_colors.add(v)
        assert len(label_colors) == 1, "Each train must carry exactly one label color"
        trains.append(
            {
                "abs_coords": abs_coords,
                "bbox": bbox,
                "body9": body9,
                "body8": body8,
                "label_color": next(iter(label_colors)),
            }
        )

    out = grid.copy()
    for train in trains:
        for r, c in train["abs_coords"]:
            out[r, c] = bg

    for train in trains:
        assert train["label_color"] in stations_by_label, "Train label must match some station label"
        target = stations_by_label[train["label_color"]]
        g = target["group"]
        gset = set(g)
        reg = target["region"]
        dock_dir = target["dock_dir"]
        candidate_ops = []
        for op in OPS:
            t_body9 = list(transform_coords(train["body9"], train["bbox"].shape, *op))
            t_body8 = list(transform_coords(train["body8"], train["bbox"].shape, *op))
            if _contact_side(t_body9, t_body8) == dock_dir:
                candidate_ops.append(op)
        assert candidate_ops, "No orientation matches the station indicator"

        placed = False
        for op in candidate_ops:
            tbbox = apply_transform(train["bbox"], *op)
            t_body9, (h1, w1) = xform_coords(train["body9"], train["bbox"].shape, *op)
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
            for r in range(h1):
                for c in range(w1):
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
            for r in range(h1):
                for c in range(w1):
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
INTERNAL_STRIPE_THICK = 2

STRIPE_THICK_RANGE = (2, 4)
INTERNAL_PROB = 0.62

TRAIN_COUNTS = [2, 3]
STATION_SIZES = [1, 2, 4]
TARGET_COMBOS = [(2, 1), (2, 2), (2, 4), (3, 1), (3, 2), (3, 4)]
TARGETED_RETRIES = 80
_combo_cursor = 0

BODY_MIN = 5
BODY_MAX = 20
PATCH_H_RANGE = (4, 9)
PATCH_W_RANGE = (3, 8)
BODY_GROWTH_ITERS = 160

LAYOUT_RETRIES = 200
DOCK_RETRIES = 30
GENERATE_RETRIES = 450

COLOR_POOL = list(range(8))


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
        rh = reg["r1"] - reg["r0"] + 1
        rw = reg["c1"] - reg["c0"] + 1
        if rh < MIN_REGION_DIM or rw < MIN_REGION_DIM or len(dirs) < 3:
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
    # Use a 1x4 / 4x1 station, not a 2x2 block: after scattering, every source-side
    # coupler cell must still sit in the 8-neighborhood of the train body so the
    # solver never mistakes a subset of train couplers for an isolated station.
    return {(0, i) for i in range(4)} if random.random() < 0.5 else {(i, 0) for i in range(4)}


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


def _build_docked_patch(bg, size, dock_dir, train_label_color):
    couplers = _station_shape(size)
    dr, dc = DIRS4[dock_dir]
    contact_band = {(r + dr, c + dc) for r, c in couplers}
    blocked = set(couplers)
    for r, c in couplers:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr != 0 or dc != 0:
                    blocked.add((r + dr, c + dc))
    for cell in contact_band:
        blocked.discard(cell)

    body = set(contact_band)
    target = random.randint(BODY_MIN, BODY_MAX)
    _grow_body(body, blocked, couplers, target)

    occ = body | set(couplers)
    label_candidates = []
    for r, c in body:
        for dr, dc in DIRS4:
            q = (r + dr, c + dc)
            if q in occ:
                continue
            h, w = _dims(occ | {q})
            if PATCH_H_RANGE[0] <= h <= PATCH_H_RANGE[1] and PATCH_W_RANGE[0] <= w <= PATCH_W_RANGE[1]:
                label_candidates.append(q)
    if not label_candidates:
        return None
    label_cell = random.choice(label_candidates)
    occ |= {label_cell}

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
    patch[label_cell[0] - r0, label_cell[1] - c0] = train_label_color

    if _contact_side(coupler_rel, body_rel) != dock_dir:
        return None
    return patch, coupler_rel


def _station_label_candidates(region, station_abs, patch_abs_mask, occupied, indicator_abs):
    gset = set(station_abs)
    cands = set()
    for r, c in station_abs:
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if not (region["r0"] <= rr <= region["r1"] and region["c0"] <= cc <= region["c1"]):
                continue
            if (rr, cc) in gset or (rr, cc) == indicator_abs or patch_abs_mask[rr, cc] or occupied[rr, cc]:
                continue
            cands.add((rr, cc))
    return list(cands)


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


def _place_docked_trains(base, bg, regions, label_pool, arrow_color, n_trains, shared_size=None):
    H, W = base.shape
    output = base.copy()
    occupied = output != bg
    shared_size = random.choice(STATION_SIZES) if shared_size is None else int(shared_size)
    label_colors = random.sample(label_pool, n_trains)

    trains = []
    for label_color in label_colors:
        placed = False
        for _ in range(DOCK_RETRIES):
            reg = random.choice(regions)
            dock_dir = random.choice([0, 1, 2, 3])
            result = _build_docked_patch(bg, shared_size, dock_dir, label_color)
            if result is None:
                continue
            patch, coupler_rel = result
            indicator_rel = _contact_cell(set(coupler_rel), _opposite_dir(dock_dir))
            locs = _patch_locations(reg, patch, bg, _dilate(occupied), coupler_rel, indicator_rel, 0)
            random.shuffle(locs)
            for r0, c0 in locs:
                patch_abs_mask = np.zeros((H, W), dtype=bool)
                ph, pw = patch.shape
                patch_abs_mask[r0 : r0 + ph, c0 : c0 + pw] = patch != bg
                station_abs = [(r0 + r, c0 + c) for r, c in coupler_rel]
                indicator_abs = (r0 + indicator_rel[0], c0 + indicator_rel[1])
                label_cands = _station_label_candidates(reg, station_abs, patch_abs_mask, occupied, indicator_abs)
                if not label_cands:
                    continue
                station_label = random.choice(label_cands)
                paste(output, patch, r0, c0, transparent=bg)
                output[indicator_abs] = arrow_color
                output[station_label] = label_color
                occupied = output != bg
                trains.append(
                    {
                        "patch": patch,
                        "station": station_abs,
                        "indicator": indicator_abs,
                        "station_label": station_label,
                        "label_color": label_color,
                        "size": len(station_abs),
                    }
                )
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return output, trains


def _scatter_trains(base, bg, regions, trains):
    H, W = base.shape
    input_grid = base.copy()
    station_mask = np.zeros((H, W), dtype=bool)
    indicator_mask = np.zeros((H, W), dtype=bool)
    station_label_mask = np.zeros((H, W), dtype=bool)
    for t in trains:
        for r, c in t["station"]:
            input_grid[r, c] = 9
            station_mask[r, c] = True
        ir, ic = t["indicator"]
        input_grid[ir, ic] = t["indicator_color"]
        indicator_mask[ir, ic] = True
        lr, lc = t["station_label"]
        input_grid[lr, lc] = t["label_color"]
        station_label_mask[lr, lc] = True

    valid_ops = []
    for t in trains:
        valid = _valid_ops_for_patch(t["patch"], PATCH_H_RANGE, PATCH_W_RANGE)
        if not valid:
            return None
        valid_ops.append(valid)
    ops = [random.choice(valid) for valid in valid_ops]
    if all(op == (False, 0) for op in ops):
        non_identity = [op for op in valid_ops[0] if op != (False, 0)]
        if not non_identity:
            return None
        ops[0] = random.choice(non_identity)

    forbidden = _dilate(base != bg) | _dilate(station_mask | indicator_mask | station_label_mask)
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


def _combo_metrics(trains, indicator_color):
    station_sizes = [t["size"] for t in trains]
    return {
        "train_count": len(trains),
        "shared_size_repeats": len(station_sizes) - len(set(station_sizes)),
        "label_uniques": len({t["label_color"] for t in trains}),
        "indicator_colors": len({indicator_color}),
    }


def _gen_attempt(preferred_n_trains=None, preferred_size=None):
    layout = _layout()
    if layout is None:
        return None
    base, bg, regions = layout

    stripe_colors = set(int(v) for v in np.unique(base) if int(v) != bg)
    extra_pool = [c for c in COLOR_POOL if c != bg and c not in stripe_colors]
    feasible_counts = [n for n in TRAIN_COUNTS if n + 1 <= len(extra_pool)]
    if not feasible_counts:
        return None
    if preferred_n_trains is not None:
        if preferred_n_trains not in feasible_counts:
            return None
        n_trains = int(preferred_n_trains)
    else:
        n_trains = random.choice(feasible_counts)
    arrow_color = random.choice(extra_pool)
    label_pool = [c for c in extra_pool if c != arrow_color]

    if preferred_size is not None and preferred_size not in STATION_SIZES:
        return None

    placed = _place_docked_trains(
        base,
        bg,
        regions,
        label_pool,
        arrow_color,
        n_trains,
        shared_size=preferred_size,
    )
    if placed is None:
        return None
    output, trains = placed
    for t in trains:
        t["indicator_color"] = arrow_color

    metrics = _combo_metrics(trains, arrow_color)
    if metrics["shared_size_repeats"] < 1:
        return None
    if metrics["label_uniques"] != metrics["train_count"]:
        return None
    if metrics["indicator_colors"] != 1:
        return None

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
    global _combo_cursor

    target_combo = TARGET_COMBOS[_combo_cursor % len(TARGET_COMBOS)]
    _combo_cursor += 1

    for n_trains, size in (target_combo, *[combo for combo in TARGET_COMBOS if combo != target_combo]):
        for _ in range(TARGETED_RETRIES):
            grid = _gen_attempt(preferred_n_trains=n_trains, preferred_size=size)
            if grid is not None:
                return grid

    for _ in range(max(0, GENERATE_RETRIES - TARGETED_RETRIES * len(TARGET_COMBOS))):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 6e4f6532 combo input")
