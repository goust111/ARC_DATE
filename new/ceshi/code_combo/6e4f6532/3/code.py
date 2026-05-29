# Combo variant of 6e4f6532 (index 3)
# combo_spec: D1/4+D2/2
# Name: 洞数匹配+对称轴定向

from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg, paste


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace:
#    Original:
#    - station size was unique, so a train could be matched just by counting attached 9-couplers.
#    - colored marker pixels on the train then picked the final rotation/flip.
#    Fused:
#    - the attached 9-couplers still define the docking footprint, but they are no longer the
#      discriminative matching cue.
#    - instead, the solver counts background holes inside the 8-body and matches that number to the
#      station's 9-count.
#    - orientation is no longer carried by marker colors. The observed symmetry-axis family of the
#      body (horizontal vs vertical) must stay parallel to the destination side's stripe, and the
#      coupler side of the train must face that stripe.
#    Example:
#    - suppose two stations both have size 1. One is placed nearest the top stripe, the other nearest
#      the left stripe. Their sizes are identical, so the old coupler-count rule is ambiguous.
#    - now place two one-hole trains: one horizontally symmetric, one vertically symmetric.
#      The horizontal train can only dock to the top/bottom-side station family; the vertical train
#      can only dock to the left/right-side family. Hole count resolves the size class, symmetry-axis
#      family resolves the duplicated target.
#
# b) Complexity calibration:
#    - repeated station sizes are intentional. If every size stayed unique, then coupler-count and
#      hole-count would collapse to the same matching behavior because docking forces |couplers| to
#      equal |station|.
#    - only two symmetry-axis families are used (horizontal / vertical), and only one duplicated size
#      appears per example. That keeps the rule visible without creating a combinatorial mess.
#
# c) Implicit-assumption audit against the original solver:
#    - "groups_by_size[len(g)] has length 1" is no longer valid. Matching must support repeated sizes.
#    - "marker pixels exist and expose stripe colors" is removed entirely.
#    - "any 90-degree rotation is fair game" is no longer assumed. The generator scatters trains using
#      only axis-preserving transforms, and the solver enforces that the observed symmetry-axis family
#      is preserved when docking.
#    - "coupler count is the target key" is false; the solver now counts holes in the 8-body.
#
# d) Placement legality:
#    - the generator places three docked trains output-first, one per chosen side family, with overlap
#      checks on every placement.
#    - scattered input trains are then placed with a one-cell safety buffer against stripes, stations,
#      and other trains.
#
# e) Visual observability:
#    - hole counts are made obvious by using 1, 2, or 4 clearly separated cavities.
#    - duplicated station sizes force the human to notice the second cue.
#    - the second cue is geometric, not color-coded: station proximity to a specific border stripe
#      determines whether a horizontal-symmetry or vertical-symmetry train must dock there.
#
# f) Visual cleanliness:
#    - the puzzle only uses bg, border stripes, body color 8, and coupler/station color 9.
#    - no extra marker colors are reintroduced, so the train interior holes remain easy to read.
#    - stations are kept near different sides of the single inner region, so docked outputs do not
#      stack into the same visual area.
#
# g) Non-degeneracy target:
#    - every generated example contains a repeated station size split across the two axis families.
#    - therefore "match by coupler count only" is insufficient, while "hole count + symmetry family"
#      is sufficient.


DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
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
AXIS_PRESERVING_OPS = [(False, 0), (False, 2), (True, 0), (True, 2)]


def apply_transform(arr: np.ndarray, flip_ud: bool = False, rot_k: int = 0) -> np.ndarray:
    out = np.flipud(arr) if flip_ud else arr
    return np.rot90(out, int(rot_k))


def transform_coords(coords, shape, flip_ud: bool = False, rot_k: int = 0):
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    out = apply_transform(mask, flip_ud=flip_ud, rot_k=rot_k)
    return set(map(tuple, np.argwhere(out)))


def _xform_coords(coords, shape, flip_ud, rot_k):
    result = transform_coords(coords, shape, flip_ud, rot_k)
    new_shape = apply_transform(np.zeros(shape, dtype=int), flip_ud, rot_k).shape
    return list(result), new_shape


def _bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _dilate(mask: np.ndarray) -> np.ndarray:
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


def _crop_tight(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _count_body_holes(body_mask: np.ndarray) -> int:
    tight = _crop_tight(body_mask)
    empty = ~tight
    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    lab, n = label(empty, structure=struct4)
    holes = 0
    for k in range(1, n + 1):
        pts = np.argwhere(lab == k)
        if not any(r == 0 or c == 0 or r == tight.shape[0] - 1 or c == tight.shape[1] - 1 for r, c in pts):
            holes += 1
    return holes


def _symmetry_axis_kind(body_mask: np.ndarray) -> str:
    tight = _crop_tight(body_mask)
    h_err = int(np.sum(tight != np.flipud(tight)))
    v_err = int(np.sum(tight != np.fliplr(tight)))
    assert min(h_err, v_err) == 0, f"body lost mirror symmetry: h={h_err}, v={v_err}"
    assert h_err != v_err, "body is symmetric in both axes; generator should avoid this"
    return "h" if h_err < v_err else "v"


def _major_dir(from_pts, to_pts) -> int:
    ar = sum(r for r, _ in from_pts) / len(from_pts)
    ac = sum(c for _, c in from_pts) / len(from_pts)
    br = sum(r for r, _ in to_pts) / len(to_pts)
    bc = sum(c for _, c in to_pts) / len(to_pts)
    dr, dc = br - ar, bc - ac
    return (0 if dr < 0 else 2) if abs(dr) >= abs(dc) else (3 if dc < 0 else 1)


def _matching_offsets(body9, station):
    body9_set = set(body9)
    station_set = set(station)
    return {
        (sr - br, sc - bc)
        for br, bc in body9_set
        for sr, sc in station_set
        if all((r + sr - br, c + sc - bc) in station_set for r, c in body9_set)
    }


def _side_distances(coords, region):
    r0, c0, r1, c1 = _bbox(coords)
    return {
        0: r0 - region["r0"],
        1: region["c1"] - c1,
        2: region["r1"] - r1,
        3: c0 - region["c0"],
    }


def _nearest_side(coords, region) -> int:
    dists = _side_distances(coords, region)
    items = sorted(dists.items(), key=lambda kv: kv[1])
    assert items[0][1] < items[1][1], f"nearest side must be unique: {dists}"
    return int(items[0][0])


def _find_edge_stripes(grid: np.ndarray, bg: int):
    H, W = grid.shape
    stripe = np.zeros((H, W), dtype=bool)

    top_h = 0
    for r in range(H):
        row = grid[r, :]
        if (row == row[0]).all() and row[0] != bg:
            top_h += 1
        else:
            break
    if top_h:
        stripe[:top_h, :] = True

    bot_h = 0
    for r in range(H - 1, -1, -1):
        row = grid[r, :]
        if (row == row[0]).all() and row[0] != bg:
            bot_h += 1
        else:
            break
    if bot_h:
        stripe[H - bot_h :, :] = True

    inner_r0, inner_r1 = top_h, H - bot_h
    left_w = 0
    for c in range(W):
        col = grid[inner_r0:inner_r1, c]
        if col.size and (col == col[0]).all() and col[0] != bg:
            left_w += 1
        else:
            break
    if left_w:
        stripe[inner_r0:inner_r1, :left_w] = True

    right_w = 0
    for c in range(W - 1, -1, -1):
        col = grid[inner_r0:inner_r1, c]
        if col.size and (col == col[0]).all() and col[0] != bg:
            right_w += 1
        else:
            break
    if right_w:
        stripe[inner_r0:inner_r1, W - right_w :] = True

    region = {"r0": top_h, "r1": H - bot_h - 1, "c0": left_w, "c1": W - right_w - 1}
    return stripe, region


def _station_groups(grid: np.ndarray):
    H, W = grid.shape
    sm = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 9:
                continue
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and grid[rr, cc] == 8:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                sm[r, c] = True
    lab, n = label(sm, structure=np.ones((3, 3), dtype=int))
    groups = []
    for k in range(1, n + 1):
        coords = np.argwhere(lab == k)
        groups.append([(int(r), int(c)) for r, c in coords])
    return groups


def _extract_trains(grid: np.ndarray, bg: int, stripe_mask: np.ndarray, station_set: set[tuple[int, int]]):
    comp_mask = (grid != bg) & (~stripe_mask)
    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    lab, n = label(comp_mask, structure=struct4)
    trains = []
    for k in range(1, n + 1):
        coords = np.argwhere(lab == k)
        vals = [int(grid[r, c]) for r, c in coords]
        if 8 not in vals:
            continue
        abs_coords = [(int(r), int(c)) for r, c in coords]
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        bbox = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        body = []
        couplers = []
        for r, c in abs_coords:
            if grid[r, c] == 8:
                body.append((int(r - r0), int(c - c0)))
            elif grid[r, c] == 9 and (r, c) not in station_set:
                couplers.append((int(r - r0), int(c - c0)))
        body_mask = bbox == 8
        axis_kind = _symmetry_axis_kind(body_mask)
        hole_count = _count_body_holes(body_mask)
        trains.append(
            {
                "abs_coords": abs_coords,
                "bbox": bbox,
                "body": body,
                "couplers": couplers,
                "axis_kind": axis_kind,
                "hole_count": hole_count,
            }
        )
    return trains


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock mirrored train sprites onto station groups by hole count, then face the nearest border side.

    Input structure:
    - A single inner operating region is framed by thick border stripes of non-bg colors.
    - Color 8 forms each train body; color 9 is used for both isolated station cells and train-attached couplers.
    - Train bodies contain 1, 2, or 4 enclosed holes, and each body is mirror-symmetric along exactly one axis.

    Rule:
    - Stations are 8-connected groups of 9s that do not touch any 8. Their size is the number of 9 cells.
    - For each train, count holes inside the 8-body and match that count to a station size.
    - Read which border side owns that station by finding the station's uniquely nearest stripe side.
    - Reorient the train using only axis-preserving transforms so that its symmetry axis stays parallel to that side and its couplers face that side, then translate the couplers onto the station cells.

    Key insight: topology chooses the size class, but repeated station sizes mean the final target is resolved by geometric symmetry: horizontal-symmetry trains go to top/bottom-side stations, vertical-symmetry trains go to left/right-side stations.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    stripe_mask, region = _find_edge_stripes(g, bg)
    stations = _station_groups(g)
    station_set = set(pt for group in stations for pt in group)
    trains = _extract_trains(g, bg, stripe_mask, station_set)
    assert len(trains) == len(stations) == 3, "generator guarantees three trains and three stations"

    station_info = []
    for idx, group in enumerate(stations):
        side = _nearest_side(group, region)
        axis_kind = "h" if side in (0, 2) else "v"
        station_info.append(
            {
                "idx": idx,
                "coords": group,
                "size": len(group),
                "side": side,
                "axis_kind": axis_kind,
            }
        )

    cleared = g.copy()
    for t in trains:
        for r, c in t["abs_coords"]:
            cleared[r, c] = bg

    candidates = []
    for ti, train in enumerate(trains):
        train_cands = []
        assert train["couplers"], "every train must carry docking couplers"
        for st in station_info:
            if train["hole_count"] != st["size"]:
                continue
            if train["axis_kind"] != st["axis_kind"]:
                continue
            for op in AXIS_PRESERVING_OPS:
                tbbox = apply_transform(train["bbox"], *op)
                t_body = tbbox == 8
                if _symmetry_axis_kind(t_body) != train["axis_kind"]:
                    continue
                t_couplers, (ph, pw) = _xform_coords(train["couplers"], train["bbox"].shape, *op)
                coupler_side = _major_dir([(r, c) for r, c in np.argwhere(t_body)], t_couplers)
                if coupler_side != st["side"]:
                    continue
                offsets = _matching_offsets(t_couplers, st["coords"])
                if len(offsets) != 1:
                    continue
                off_r, off_c = next(iter(offsets))
                paint_cells = []
                ok = True
                for r in range(ph):
                    for c in range(pw):
                        v = int(tbbox[r, c])
                        if v == bg:
                            continue
                        rr, cc = r + off_r, c + off_c
                        if not (region["r0"] <= rr <= region["r1"] and region["c0"] <= cc <= region["c1"]):
                            ok = False
                            break
                        if v != 9:
                            if stripe_mask[rr, cc]:
                                ok = False
                                break
                            if cleared[rr, cc] != bg and cleared[rr, cc] != 9:
                                ok = False
                                break
                            paint_cells.append((rr, cc, v))
                    if not ok:
                        break
                if ok:
                    train_cands.append({"station_idx": st["idx"], "paint": paint_cells})
        assert train_cands, f"no docking candidate for train {ti}"
        candidates.append(train_cands)

    order = sorted(range(len(trains)), key=lambda i: len(candidates[i]))
    used_stations = set()
    assignment = [None] * len(trains)

    def backtrack(pos: int, occupied: set[tuple[int, int]]) -> bool:
        if pos == len(order):
            return True
        ti = order[pos]
        for cand in candidates[ti]:
            if cand["station_idx"] in used_stations:
                continue
            pts = {(r, c) for r, c, _ in cand["paint"]}
            if pts & occupied:
                continue
            used_stations.add(cand["station_idx"])
            assignment[ti] = cand
            if backtrack(pos + 1, occupied | pts):
                return True
            assignment[ti] = None
            used_stations.remove(cand["station_idx"])
        return False

    assert backtrack(0, set()), "expected a unique non-overlapping assignment"

    out = cleared.copy()
    for cand in assignment:
        for r, c, v in cand["paint"]:
            out[r, c] = v
    return out


# --------------- generator ---------------

GRID_H_RANGE = (22, 29)
GRID_W_RANGE = (22, 29)
STRIPE_THICK_RANGE = (2, 4)
GENERATE_RETRIES = 400
DOCK_RETRIES = 80
SCATTER_RETRIES = 80
COLOR_POOL = list(range(8))
SIZE_PATTERNS = [(1, 1, 4), (1, 2, 2), (2, 2, 4)]

BODY_PATTERNS_UP = {
    1: [
        ".####",
        "#####",
        "##.##",
        "#####",
        ".####",
    ],
    2: [
        ".###..",
        "##.##.",
        "######",
        "##.##.",
        ".###..",
    ],
    4: [
        ".####..",
        "##.#.##",
        "######.",
        "##.#.##",
        ".####..",
    ],
}


def _inverse_patch(patch, op):
    flip_ud, rot_k = op
    out = np.rot90(patch, (-int(rot_k)) % 4)
    return np.flipud(out) if flip_ud else out


def _side_to_rot(side: int) -> int:
    if side == 0:
        return 0
    if side == 1:
        return 3
    if side == 2:
        return 2
    return 1


def _canonical_patch_up(size: int, bg: int) -> np.ndarray:
    body_rows = BODY_PATTERNS_UP[size]
    w = len(body_rows[0])
    patch = np.full((len(body_rows) + 1, w), bg, dtype=int)
    if size == 1:
        coupler_cols = [2]
    elif size == 2:
        coupler_cols = [2, 3]
    else:
        coupler_cols = [1, 2, 3, 4]
    for c in coupler_cols:
        patch[0, c] = 9
    for r, row in enumerate(body_rows, start=1):
        for c, ch in enumerate(row):
            if ch == "#":
                patch[r, c] = 8
    return patch


def _build_patch(size: int, side: int, bg: int) -> np.ndarray:
    up = _canonical_patch_up(size, bg)
    return np.rot90(up, _side_to_rot(side))


def _choose_sizes_and_sides():
    sizes = list(random.choice(SIZE_PATTERNS))
    dup = Counter(sizes).most_common(1)[0][0]
    horiz_side = random.choice([0, 2])
    vert_side = random.choice([1, 3])
    remaining_side = random.choice([s for s in range(4) if s not in (horiz_side, vert_side)])
    random.shuffle(sizes)

    dup_indices = [i for i, s in enumerate(sizes) if s == dup]
    non_dup_index = [i for i, s in enumerate(sizes) if s != dup][0]
    sides = [None, None, None]
    random.shuffle(dup_indices)
    sides[dup_indices[0]] = horiz_side
    sides[dup_indices[1]] = vert_side
    sides[non_dup_index] = remaining_side
    return sizes, sides


def _layout():
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    top = random.randint(*STRIPE_THICK_RANGE)
    bottom = random.randint(*STRIPE_THICK_RANGE)
    left = random.randint(*STRIPE_THICK_RANGE)
    right = random.randint(*STRIPE_THICK_RANGE)
    assert H - top - bottom >= 14 and W - left - right >= 14

    pool = COLOR_POOL[:]
    bg = random.choice(pool)
    pool.remove(bg)
    stripe_cols = random.sample(pool, 4)
    grid = np.full((H, W), bg, dtype=int)
    grid[:top, :] = stripe_cols[0]
    grid[H - bottom :, :] = stripe_cols[1]
    grid[top : H - bottom, :left] = stripe_cols[2]
    grid[top : H - bottom, W - right :] = stripe_cols[3]
    region = {"r0": top, "r1": H - bottom - 1, "c0": left, "c1": W - right - 1}
    return grid, bg, region


def _patch_locations(region, patch, bg, occupied, margin=0):
    H, W = occupied.shape
    ph, pw = patch.shape
    locs = []
    r_start = max(region["r0"] + margin, 0)
    r_stop = min(region["r1"] - margin - ph + 2, H - ph + 1)
    c_start = max(region["c0"] + margin, 0)
    c_stop = min(region["c1"] - margin - pw + 2, W - pw + 1)
    for rr in range(r_start, r_stop):
        for cc in range(c_start, c_stop):
            if not occupied[rr : rr + ph, cc : cc + pw].any():
                locs.append((rr, cc))
    return locs


def _docked_locations(region, patch, bg, occupied, side):
    H, W = occupied.shape
    ph, pw = patch.shape
    station_rel = [(int(r), int(c)) for r, c in np.argwhere(patch == 9)]
    locs = []
    for rr in range(region["r0"], region["r1"] - ph + 2):
        for cc in range(region["c0"], region["c1"] - pw + 2):
            if occupied[rr : rr + ph, cc : cc + pw].any():
                continue
            station_abs = [(rr + r, cc + c) for r, c in station_rel]
            dists = _side_distances(station_abs, region)
            items = sorted(dists.items(), key=lambda kv: kv[1])
            if items[0][0] != side or items[0][1] != 2 or items[0][1] >= items[1][1]:
                continue
            locs.append((rr, cc, station_abs))
    return locs


def _place_docked_trains(base, bg, region):
    output = base.copy()
    occupied = output != bg
    sizes, sides = _choose_sizes_and_sides()
    trains = []
    for size, side in zip(sizes, sides):
        patch = _build_patch(size, side, bg)
        locs = _docked_locations(region, patch, bg, _dilate(occupied), side)
        if not locs:
            return None
        r0, c0, station_abs = random.choice(locs)
        paste(output, patch, r0, c0, transparent=bg)
        patch_mask = np.zeros_like(output, dtype=bool)
        ph, pw = patch.shape
        patch_mask[r0 : r0 + ph, c0 : c0 + pw] = patch != bg
        occupied |= patch_mask
        trains.append({"patch": patch, "station": station_abs, "size": size, "side": side})
    return output, trains


def _scatter_trains(base, bg, region, trains):
    H, W = base.shape
    input_grid = base.copy()
    station_mask = np.zeros((H, W), dtype=bool)
    for t in trains:
        for r, c in t["station"]:
            input_grid[r, c] = 9
            station_mask[r, c] = True

    ops = [random.choice(AXIS_PRESERVING_OPS) for _ in trains]
    if all(op == (False, 0) for op in ops):
        ops[0] = random.choice([op for op in AXIS_PRESERVING_OPS if op != (False, 0)])

    forbidden = _dilate(base != bg) | _dilate(station_mask)
    for t, op in zip(trains, ops):
        in_patch = _inverse_patch(t["patch"], op)
        locs = _patch_locations(region, in_patch, bg, forbidden, margin=1)
        if not locs:
            return None
        r0, c0 = random.choice(locs)
        paste(input_grid, in_patch, r0, c0, transparent=bg)
        ph, pw = in_patch.shape
        patch_mask = np.zeros((H, W), dtype=bool)
        patch_mask[r0 : r0 + ph, c0 : c0 + pw] = in_patch != bg
        forbidden |= _dilate(patch_mask)
    return input_grid


def _scene_signature(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    stripe_mask, region = _find_edge_stripes(g, bg)
    stations = _station_groups(g)
    station_set = set(pt for group in stations for pt in group)
    trains = _extract_trains(g, bg, stripe_mask, station_set)
    return {
        "station_sizes": sorted(len(gp) for gp in stations),
        "station_sides": sorted(_nearest_side(gp, region) for gp in stations),
        "train_holes": sorted(t["hole_count"] for t in trains),
        "train_axes": sorted(t["axis_kind"] for t in trains),
    }


def _gen_attempt():
    base, bg, region = _layout()
    placed = _place_docked_trains(base, bg, region)
    if placed is None:
        return None
    output, trains = placed
    input_grid = _scatter_trains(base, bg, region, trains)
    if input_grid is None:
        return None
    try:
        sig = _scene_signature(input_grid)
    except Exception:
        return None
    if max(Counter(sig["station_sizes"]).values()) < 2:
        return None
    if sig["train_holes"] != sig["station_sizes"]:
        return None
    if "h" not in sig["train_axes"] or "v" not in sig["train_axes"]:
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
