# Combo variant of 6e4f6532 (index 8)
# combo_spec: D5/2+D6/1
# Name: 保留原位+移动轨迹涂色

from __future__ import annotations

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


def _find_edge_stripes(grid: np.ndarray, bg: int):
    H, W = grid.shape
    stripe = np.zeros((H, W), dtype=bool)
    colors = set()

    top_h = 0
    for r in range(H):
        row = grid[r, :]
        if (row == row[0]).all() and int(row[0]) != int(bg):
            top_h += 1
        else:
            break
    if top_h:
        stripe[:top_h, :] = True
        colors.add(int(grid[0, 0]))

    bot_h = 0
    for r in range(H - 1, -1, -1):
        row = grid[r, :]
        if (row == row[0]).all() and int(row[0]) != int(bg):
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
        if (col == col[0]).all() and int(col[0]) != int(bg):
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
        if (col == col[0]).all() and int(col[0]) != int(bg):
            right_w += 1
        else:
            break
    if right_w:
        stripe[inner_r0:inner_r1, W - right_w :] = True
        colors.add(int(grid[inner_r0, W - 1]))

    return stripe, colors, (top_h, bot_h, left_w, right_w)


def _find_internal_stripes(grid: np.ndarray, bg: int, edge_info):
    H, W = grid.shape
    top_h, bot_h, left_w, right_w = edge_info
    r0, r1, c0, c1 = top_h, H - bot_h, left_w, W - right_w
    stripe = np.zeros((H, W), dtype=bool)
    colors = set()

    c = c0
    while c < c1:
        col = grid[r0:r1, c]
        if col.size and (col == col[0]).all() and int(col[0]) != int(bg):
            color = int(col[0])
            c2 = c
            while c2 + 1 < c1:
                col2 = grid[r0:r1, c2 + 1]
                if col2.size and (col2 == col2[0]).all() and int(col2[0]) == int(color):
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
        if row.size and (row == row[0]).all() and int(row[0]) != int(bg):
            color = int(row[0])
            r2 = r
            while r2 + 1 < r1:
                row2 = grid[r2 + 1, c0:c1]
                if row2.size and (row2 == row2[0]).all() and int(row2[0]) == int(color):
                    r2 += 1
                else:
                    break
            stripe[r : r2 + 1, c0:c1] = True
            colors.add(color)
            r = r2 + 1
        else:
            r += 1

    return stripe, colors


def _station_groups(grid: np.ndarray):
    H, W = grid.shape
    sm = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != 9:
                continue
            ok = all(
                int(grid[r + dr, c + dc]) != 8
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
        coords = np.argwhere(lab == k)
        groups.append([(int(r), int(c)) for r, c in coords])
    return groups


def _regions_from_stripes(stripe_mask: np.ndarray):
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


def _xform_coords(coords, shape, flip_ud, rot_k):
    result = transform_coords(coords, shape, flip_ud, rot_k)
    new_shape = apply_transform(np.zeros(shape, dtype=int), flip_ud, rot_k).shape
    return list(result), new_shape


def _choose_best_op(train, divider_dirs):
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
            ref_t = list(transform_coords([(int(round(ref_r)), int(round(ref_c)))], (H0, W0), flip_ud, k))
            rr_ref, cc_ref = ref_t[0]
            score = 0
            for color, pts in markers_by_color.items():
                mr = sum(r for r, _ in pts) / len(pts)
                mc = sum(c for _, c in pts) / len(pts)
                m_t = list(transform_coords([(int(round(mr)), int(round(mc)))], (H0, W0), flip_ud, k))
                rr_m, cc_m = m_t[0]
                dr, dc = rr_m - rr_ref, cc_m - cc_ref
                dirc = (0 if dr < 0 else 2) if abs(dr) >= abs(dc) else (3 if dc < 0 else 1)
                if dirc == divider_dirs[color]:
                    score += 1
            if score > best_score:
                best_score, best = score, (flip_ud, k)
    return best


def _trace_offsets(delta_r: int, delta_c: int):
    offsets = []
    step_r = 0 if delta_r == 0 else (1 if delta_r > 0 else -1)
    step_c = 0 if delta_c == 0 else (1 if delta_c > 0 else -1)
    for i in range(1, abs(int(delta_r)) + 1):
        offsets.append((i * step_r, 0))
    for j in range(1, abs(int(delta_c)) + 1):
        offsets.append((int(delta_r), j * step_c))
    if offsets:
        offsets.pop()
    return offsets


def _entry_side_from_delta(delta_r: int, delta_c: int) -> int:
    if delta_c > 0:
        return 3
    if delta_c < 0:
        return 1
    if delta_r > 0:
        return 0
    if delta_r < 0:
        return 2
    raise ValueError("zero displacement has no entry side")


def _plan_docking(grid: np.ndarray):
    grid = np.array(grid, dtype=int)
    H, W = grid.shape
    bg = int(get_bg(grid))

    edge_mask, edge_colors, edge_info = _find_edge_stripes(grid, bg)
    internal_mask, internal_colors = _find_internal_stripes(grid, bg, edge_info)
    stripe_mask = edge_mask | internal_mask
    stripe_colors = edge_colors | internal_colors
    regions = _regions_from_stripes(stripe_mask)

    groups = _station_groups(grid)
    groups_by_size = defaultdict(list)
    for g in groups:
        groups_by_size[len(g)].append(g)
    station_set = set(pt for g in groups for pt in g)

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
        body9, marker_pixels, src_body9_abs = [], [], []
        for r, c in abs_coords:
            v = int(grid[r, c])
            if v == 9 and (r, c) not in station_set:
                body9.append((int(r - r0), int(c - c0)))
                src_body9_abs.append((int(r), int(c)))
            if v in stripe_colors:
                marker_pixels.append((v, int(r - r0), int(c - c0)))
        trains.append(
            {
                "abs_coords": abs_coords,
                "abs_set": set(abs_coords),
                "bbox": bbox,
                "body9": body9,
                "src_body9_abs": src_body9_abs,
                "marker_pixels": marker_pixels,
                "src_r0": int(r0),
                "src_c0": int(c0),
            }
        )

    plans = []
    for t in trains:
        n9 = len(t["body9"])
        assert n9 > 0, "Each train must carry at least one coupler"
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
        divider_dirs = _boundary_dirs_for_region(grid, reg, stripe_colors)
        op = _choose_best_op(t, divider_dirs)
        tbbox = apply_transform(t["bbox"], *op)
        t_body9, (H1, W1) = _xform_coords(t["body9"], t["bbox"].shape, *op)
        body9_set = set(t_body9)
        matching_offsets = {
            (sr - br, sc - bc)
            for br, bc in body9_set
            for sr, sc in gset
            if all((r + sr - br, c + sc - bc) in gset for r, c in body9_set)
        }
        assert len(matching_offsets) == 1, "Couplers must align to a unique station-group offset"
        off_r, off_c = next(iter(matching_offsets))
        delta_r = int(off_r - t["src_r0"])
        delta_c = int(off_c - t["src_c0"])
        assert delta_r != 0 or delta_c != 0, "source and target must differ"
        entry_side = _entry_side_from_delta(delta_r, delta_c)
        dir_to_color = {int(v): int(k) for k, v in divider_dirs.items()}
        assert entry_side in dir_to_color, "target region must expose an entry-side stripe"
        plans.append(
            {
                "train": t,
                "station": g,
                "station_set": gset,
                "region": reg,
                "divider_dirs": divider_dirs,
                "dir_to_color": dir_to_color,
                "trail_color": int(dir_to_color[entry_side]),
                "op": op,
                "tbbox": tbbox,
                "target_body9": body9_set,
                "target_h": int(H1),
                "target_w": int(W1),
                "target_r0": int(off_r),
                "target_c0": int(off_c),
                "delta_r": int(delta_r),
                "delta_c": int(delta_c),
            }
        )

    return {
        "grid": grid,
        "H": H,
        "W": W,
        "bg": bg,
        "stripe_mask": stripe_mask,
        "stripe_colors": stripe_colors,
        "regions": regions,
        "groups": groups,
        "station_set": station_set,
        "plans": plans,
    }


def variant_metrics(grid: np.ndarray) -> dict:
    scene = _plan_docking(grid)
    total_trail = 0
    per_train = []
    for plan in scene["plans"]:
        visible = set()
        for dr, dc in _trace_offsets(plan["delta_r"], plan["delta_c"]):
            for r, c in plan["train"]["src_body9_abs"]:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < scene["H"] and 0 <= cc < scene["W"] and (rr, cc) not in plan["train"]["abs_set"]:
                    visible.add((rr, cc))
        total_trail += len(visible)
        per_train.append(
            {
                "trail_cells": len(visible),
                "axis_aligned": int(plan["delta_r"] == 0 or plan["delta_c"] == 0),
                "source_cells": len(plan["train"]["abs_coords"]),
            }
        )
    return {
        "train_count": len(scene["plans"]),
        "total_trail_cells": int(total_trail),
        "axis_aligned_trains": int(sum(p["axis_aligned"] for p in per_train)),
        "per_train": per_train,
    }


# Design analysis for the fused variant:
# - Small example: if a 2-coupler train sits on the left and its matching station of size 2 sits on the right,
#   the original task deletes the left train and redraws a re-oriented copy on the station. Here the left train
#   stays, the right docked copy still appears, and the coupler footprint leaves a colored path between them.
# - The original solver had a strong hidden assumption: clearing the source first meant target placement only had
#   to avoid stripes and other docked copies. That assumption is now false, so the new solver paints onto a copy
#   of the full input and validates docked placement against existing source trains.
# - Full-sprite trails would create heavy self-overlap and visually messy outputs once rotation changes bbox shape.
#   To keep the rule observable, the trail uses the attached coupler footprint only, producing a thin continuous
#   ribbon that clearly links source and destination without hiding either train.
# - Trail color is not arbitrary. It is the stripe color on the destination-region side from which the train
#   approaches, so the path reuses the existing stripe-color code and remains inferable from a few examples.
# - Generator guarantees the new assumptions: source-to-target displacement is axis-aligned, the relevant entry
#   side exists among the target region's boundary colors, and all trail cells stay off stripes, stations, source
#   trains, and docked copies. Unique station sizes continue to keep train-station matching unambiguous.
def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each train onto its matching station while preserving the original train and painting a colored route.

    Input structure:
    - Colored edge/internal stripes partition the canvas into rectangular regions over a dominant background.
    - Color 8 is the train body; color 9 appears both as isolated station clusters and as train-attached couplers.
    - Stripe-colored marker pixels on each train still encode how the docked copy must be rotated/flipped.

    Rule:
    - Stations are 9-clusters not touching any 8; trains are components containing 8.
    - Match a train to the unique station whose size equals its coupler count, orient it by marker-vs-stripe agreement,
      and place a docked copy so its transformed couplers land exactly on the station cells.
    - Unlike the base task, do not erase the source train. Instead, keep it in place and color the intermediate
      coupler positions along the translation path using the stripe color from the side where the train enters
      the destination region.

    Key insight: The old coupler-count and marker-direction logic still identifies the destination and orientation;
    the new visual twist is that docking becomes copy-plus-trail rather than remove-and-move.
    """
    scene = _plan_docking(grid)
    out = scene["grid"].copy()

    for plan in scene["plans"]:
        reg = plan["region"]
        tbbox = plan["tbbox"]
        target_r0 = plan["target_r0"]
        target_c0 = plan["target_c0"]
        H1 = plan["target_h"]
        W1 = plan["target_w"]

        ok = True
        for r in range(H1):
            for c in range(W1):
                v = int(tbbox[r, c])
                if v == scene["bg"] or v == 9:
                    continue
                rr, cc = int(target_r0 + r), int(target_c0 + c)
                if not (reg["r0"] <= rr <= reg["r1"] and reg["c0"] <= cc <= reg["c1"]):
                    ok = False
                    break
                if scene["stripe_mask"][rr, cc] or (int(out[rr, cc]) != int(scene["bg"]) and int(out[rr, cc]) != 9):
                    ok = False
                    break
            if not ok:
                break
        assert ok, "Docked train must fit entirely within its target region"

        trail_color = int(plan["trail_color"])
        for dr, dc in _trace_offsets(plan["delta_r"], plan["delta_c"]):
            for r, c in plan["train"]["src_body9_abs"]:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < scene["H"] and 0 <= cc < scene["W"] and int(out[rr, cc]) == int(scene["bg"]):
                    out[rr, cc] = trail_color

        for r in range(H1):
            for c in range(W1):
                v = int(tbbox[r, c])
                if v != scene["bg"] and v != 9:
                    out[target_r0 + r, target_c0 + c] = v

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

TRAIN_COUNT_CYCLE = (2, 3)
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

LAYOUT_RETRIES = 220
DOCK_RETRIES = 28
GENERATE_RETRIES = 700
MIN_TRAIL_STEPS = 3
MIN_VISIBLE_TRAIL_CELLS = 2
THREE_TRAIN_AREA_BONUS = 96
THREE_TRAIN_DOCK_RETRIES = 42
SCATTER_RETRIES = 18
THREE_TRAIN_SCATTER_RETRIES = 72

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

_train_count_cycle_idx = 0


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


def _next_target_train_count():
    global _train_count_cycle_idx
    target = int(TRAIN_COUNT_CYCLE[_train_count_cycle_idx % len(TRAIN_COUNT_CYCLE)])
    _train_count_cycle_idx += 1
    return target


def _layout(target_train_count):
    for _ in range(LAYOUT_RETRIES):
        if target_train_count >= 3:
            use_internal = False
            orient = None
            edges = ["top", "bottom", "left", "right"]
        elif random.random() < INTERNAL_PROB:
            use_internal = True
            orient = random.choice(["v", "h"])
            if orient == "v":
                edges = ["left", "right", random.choice(["top", "bottom"])]
            else:
                edges = ["top", "bottom", random.choice(["left", "right"])]
        else:
            use_internal = False
            orient = None
            edges = random.sample(["top", "bottom", "left", "right"], random.choice([3, 4]))

        thick = {e: random.randint(*STRIPE_THICK_RANGE) for e in edges}
        top, bottom = thick.get("top", 0), thick.get("bottom", 0)
        left, right = thick.get("left", 0), thick.get("right", 0)

        min_inner_for_split = MIN_REGION_DIM * 2 + INTERNAL_STRIPE_THICK
        min_area = GRID_MIN_AREA + (THREE_TRAIN_AREA_BONUS if target_train_count >= 3 else 0)
        candidates = []
        for H in range(*GRID_H_RANGE):
            for W in range(*GRID_W_RANGE):
                if H * W < min_area:
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

        if _assign_region_dirs(regions, edge_col, top, bottom, left, right, H, W, internal):
            return grid, bg, regions
    return None


def _station_shape(size):
    if size == 1:
        return {(0, 0)}
    if size == 2:
        return {(0, 0), (0, 1)} if random.random() < 0.5 else {(0, 0), (1, 0)}
    return {(0, 0), (0, 1), (1, 0), (1, 1)}


def _marker_specs(dirs, compact=False):
    items = list(dirs.items())
    if not compact and len(items) >= 3 and random.random() < THREE_MARKER_PROB:
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


def _build_docked_patch(bg, dirs, size, compact=False):
    couplers = _station_shape(size)
    specs = _marker_specs(dirs, compact=compact)
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
    body_lo = max(BODY_MIN, len(body))
    body_hi = min(BODY_MAX, BODY_MIN + 6) if compact else BODY_MAX
    target = random.randint(body_lo, max(body_lo, body_hi))
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


def _place_docked_trains(base, bg, regions, target_train_count):
    H, W = base.shape
    output = base.copy()
    occupied = output != bg

    n_trains = int(target_train_count)
    sizes = sorted(random.sample(STATION_SIZES, n_trains), reverse=True)
    dock_retries = THREE_TRAIN_DOCK_RETRIES if n_trains >= 3 else DOCK_RETRIES
    trains = []
    for size in sizes:
        region_order = sorted(
            regions,
            key=lambda reg: (
                -((reg["r1"] - reg["r0"] + 1) * (reg["c1"] - reg["c0"] + 1)),
                random.random(),
            ),
        )
        for _ in range(dock_retries):
            reg = random.choice(region_order[: max(1, min(len(region_order), 2))])
            result = _build_docked_patch(bg, reg["dirs"], size, compact=n_trains >= 3)
            if result is None:
                continue
            patch, coupler_rel = result
            locs = _patch_locations(
                reg,
                patch,
                bg,
                _dilate(occupied),
                coupler_rel,
                True,
                1 if n_trains >= 3 else 0,
            )
            if not locs:
                continue
            r0, c0 = random.choice(locs)
            paste(output, patch, r0, c0, transparent=bg)
            occupied |= _dilate((output != bg) & ~occupied) | (output != bg)
            station_abs = [(r0 + r, c0 + c) for r, c in coupler_rel]
            dock_non9 = {
                (r0 + int(r), c0 + int(c))
                for r, c in np.argwhere((patch != bg) & (patch != 9))
            }
            trains.append(
                {
                    "patch": patch,
                    "station": station_abs,
                    "dock_r0": int(r0),
                    "dock_c0": int(c0),
                    "region": reg,
                    "dock_non9": dock_non9,
                }
            )
            break
        else:
            return None
    return output, trains


def _candidate_scatter_positions(base, bg, regions, forbidden, patch):
    regs = regions[:]
    random.shuffle(regs)
    all_locs = []
    for reg in regs:
        locs = _patch_locations(reg, patch, bg, forbidden, None, False, 0)
        all_locs.extend((reg, p) for p in locs)
    random.shuffle(all_locs)
    return all_locs


def _axis_scatter_positions(region, dock_r0, dock_c0, patch_shape, entry_side):
    ph, pw = patch_shape
    positions = []
    if entry_side == 3:
        if region["r0"] <= dock_r0 and dock_r0 + ph - 1 <= region["r1"]:
            hi = dock_c0 - MIN_TRAIL_STEPS
            for c0 in range(region["c0"], hi + 1):
                if c0 + pw - 1 <= region["c1"]:
                    positions.append((dock_r0, c0))
    elif entry_side == 1:
        if region["r0"] <= dock_r0 and dock_r0 + ph - 1 <= region["r1"]:
            lo = dock_c0 + MIN_TRAIL_STEPS
            for c0 in range(lo, region["c1"] - pw + 2):
                positions.append((dock_r0, c0))
    elif entry_side == 0:
        if region["c0"] <= dock_c0 and dock_c0 + pw - 1 <= region["c1"]:
            hi = dock_r0 - MIN_TRAIL_STEPS
            for r0 in range(region["r0"], hi + 1):
                if r0 + ph - 1 <= region["r1"]:
                    positions.append((r0, dock_c0))
    elif entry_side == 2:
        if region["c0"] <= dock_c0 and dock_c0 + pw - 1 <= region["c1"]:
            lo = dock_r0 + MIN_TRAIL_STEPS
            for r0 in range(lo, region["r1"] - ph + 2):
                positions.append((r0, dock_c0))
    return positions


def _scatter_trains(base, bg, regions, trains):
    H, W = base.shape
    station_mask = np.zeros((H, W), dtype=bool)
    for t in trains:
        for r, c in t["station"]:
            station_mask[r, c] = True
    seeded_input = base.copy()
    seeded_input[station_mask] = 9

    valid_ops = []
    for t in trains:
        valid = _valid_ops_for_patch(t["patch"])
        if not valid:
            return None
        valid_ops.append(valid)
    scatter_retries = THREE_TRAIN_SCATTER_RETRIES if len(trains) >= 3 else SCATTER_RETRIES
    for _ in range(scatter_retries):
        input_grid = seeded_input.copy()
        ops = [random.choice(valid) for valid in valid_ops]
        if all(op == (False, 0) for op in ops):
            non_id = [op for op in valid_ops[0] if op != (False, 0)]
            if not non_id:
                continue
            ops[0] = random.choice(non_id)

        forbidden_input = (base != bg) | _dilate(station_mask)
        output_blocked = (base != bg) | station_mask
        for t in trains:
            for r, c in t["dock_non9"]:
                output_blocked[r, c] = True
        trail_blocked = output_blocked.copy()
        scatter_meta = []
        order = sorted(
            range(len(trains)),
            key=lambda idx: (
                -len(trains[idx]["station"]),
                -int(np.count_nonzero(trains[idx]["patch"] != bg)),
                random.random(),
            ),
        )

        ok_all = True
        for idx in order:
            t = trains[idx]
            op = ops[idx]
            in_patch = _inverse_patch(t["patch"], op)
            rel_nonbg = [(int(r), int(c)) for r, c in np.argwhere(in_patch != bg)]
            rel_body9 = [(int(r), int(c)) for r, c in np.argwhere(in_patch == 9)]
            if not rel_body9:
                ok_all = False
                break

            chosen = None
            candidate_sides = list(set(t["region"]["dirs"].values()))
            random.shuffle(candidate_sides)
            for entry_side in candidate_sides:
                candidates = _axis_scatter_positions(
                    t["region"], t["dock_r0"], t["dock_c0"], in_patch.shape, entry_side
                )
                if entry_side in (1, 2):
                    candidates = list(reversed(candidates))
                if len(trains) >= 3:
                    candidates.sort(
                        key=lambda pos: abs(t["dock_r0"] - pos[0]) + abs(t["dock_c0"] - pos[1]),
                        reverse=True,
                    )
                for r0, c0 in candidates:
                    source_cells = {(r0 + pr, c0 + pc) for pr, pc in rel_nonbg}
                    if any(forbidden_input[r, c] for r, c in source_cells):
                        continue
                    if any(output_blocked[r, c] for r, c in source_cells):
                        continue

                    delta_r = int(t["dock_r0"] - r0)
                    delta_c = int(t["dock_c0"] - c0)
                    trail_cells = set()
                    ok = True
                    for dr, dc in _trace_offsets(delta_r, delta_c):
                        for pr, pc in rel_body9:
                            rr, cc = int(r0 + pr + dr), int(c0 + pc + dc)
                            if not (0 <= rr < H and 0 <= cc < W):
                                ok = False
                                break
                            if base[rr, cc] != bg or station_mask[rr, cc]:
                                ok = False
                                break
                            if trail_blocked[rr, cc] and (rr, cc) not in source_cells:
                                ok = False
                                break
                            if (rr, cc) not in source_cells:
                                trail_cells.add((rr, cc))
                        if not ok:
                            break
                    if not ok or len(trail_cells) < MIN_VISIBLE_TRAIL_CELLS:
                        continue
                    chosen = (r0, c0, trail_cells)
                    break
                if chosen is not None:
                    break

            if chosen is None:
                ok_all = False
                break

            r0, c0, trail_cells = chosen
            paste(input_grid, in_patch, r0, c0, transparent=bg)
            ph, pw = in_patch.shape
            patch_mask = np.zeros((H, W), dtype=bool)
            patch_mask[r0 : r0 + ph, c0 : c0 + pw] = in_patch != bg
            forbidden_input |= _dilate(patch_mask)
            output_blocked |= patch_mask
            for rr, cc in trail_cells:
                trail_blocked[rr, cc] = True
                output_blocked[rr, cc] = True
            scatter_meta.append({"trail_cells": len(trail_cells)})

        if ok_all:
            return input_grid, scatter_meta

    return None


def _gen_attempt(target_train_count):
    layout = _layout(target_train_count)
    if layout is None:
        return None
    base, bg, regions = layout

    placed = _place_docked_trains(base, bg, regions, target_train_count)
    if placed is None:
        return None
    _output, trains = placed

    scattered = _scatter_trains(base, bg, regions, trains)
    if scattered is None:
        return None
    input_grid, scatter_meta = scattered

    counts = Counter(map(int, input_grid.flatten()))
    if counts[bg] != max(counts.values()):
        return None

    try:
        out = main(input_grid)
    except Exception:
        return None

    metrics = variant_metrics(input_grid)
    if metrics["train_count"] != len(trains):
        return None
    if metrics["axis_aligned_trains"] != len(trains):
        return None
    if metrics["total_trail_cells"] < max(MIN_VISIBLE_TRAIL_CELLS * len(trains), 8):
        return None
    if any(int(m["trail_cells"]) < MIN_VISIBLE_TRAIL_CELLS for m in scatter_meta):
        return None
    if np.array_equal(out, input_grid):
        return None

    return input_grid


def generate_input() -> np.ndarray:
    target_train_count = _next_target_train_count()
    retry_schedule = [target_train_count, target_train_count, 5 - target_train_count]
    phase_budget = max(1, GENERATE_RETRIES // len(retry_schedule))
    for desired_count in retry_schedule:
        for _ in range(phase_budget):
            grid = _gen_attempt(desired_count)
            if grid is not None:
                return grid
    for _ in range(GENERATE_RETRIES - phase_budget * len(retry_schedule)):
        grid = _gen_attempt(target_train_count)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 6e4f6532 combo input")
