# Combo variant of 6e4f6532 (index 11)
# combo_spec: D8/0+D9/0
# Name: 多对一对接+顺序依赖

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg, paste


"""
Design analysis for the fused variant.

1. Concrete behavior shift:
   - Original: each train matched one isolated station cluster by coupler count, so
     every docking job was independent.
   - Here: the visible target is one 8-connected "mega-station" of color 9. Inside
     it, the actual docking lobes are its 4-connected subclusters. Example:
       size-1 dot at (r,c), size-2 domino at (r+1,c+1:r+2), size-4 block at
       (r+3:r+5,c+3:c+5). They touch only diagonally, so humans see one large
       station while the solver can split it into three lobes.
   - The solver must therefore change from "group by whole 9-cluster size" to
     "find the big 8-connected station, split it into 4-connected lobes, and dock
     trains into unused lobes in ascending coupler-count order".

2. Complexity calibration:
   - The station still uses only sizes from {1, 2, 4}; this keeps the matching
     alphabet small and repeated across samples.
   - The mega-station is always clearly visible because all lobes sit inside one
     region and form one diagonal chain, so the D8 change is legible without extra
     legend objects.
   - The orientation signal stays the same as the original task: stripe-colored
     markers on the train indicate which region boundary colors the docked train
     should face. This avoids introducing a second hard-to-see code.

3. Broken original assumptions and how they are repaired:
   - `groups_by_size[len(g)] == 1` is no longer true because one whole station group
     can contain several lobes. The new solver splits station groups into 4-neighbor
     subclusters and matches trains to those instead.
   - "All trains can be placed independently" is no longer assumed. The new solver
     clears every train first, then places them in ascending coupler-count order,
     validating each candidate against the already-docked trains.
   - "One station group belongs to one train" is gone. The solver tracks used lobes
     inside each mega-station separately.

4. Placement legality:
   - The generator builds the final output first. It picks one target region, lays
     down one mega-station with 2-3 diagonal lobes, then docks trains around those
     lobes in ascending size order while checking region bounds, stripe avoidance,
     and non-overlap with already placed trains.
   - Only after a valid docked scene exists are the trains inverse-transformed and
     scattered back into the input.

5. Visual observability:
   - The human-visible cue for D8 is that several trains all move toward one large
     9-cluster instead of several isolated 9-clusters.
   - The human-visible cue for D9 is that those trains crowd the same region and are
     packed around the station in a small-to-large progression. The solver also uses
     current occupancy when validating later placements.

6. New implicit assumptions guaranteed by the generator:
   - Exactly one station group has multiple 4-connected lobes, and every train docks
     into that same group.
   - Each lobe shape is congruent to exactly one train coupler shape after a valid
     rotation/flip.
   - The target region is large enough that sequential docking succeeds without
     touching stripes or leaving the region.
"""


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


def xform_coords(coords, shape, flip_ud, rot_k):
    result = transform_coords(coords, shape, flip_ud, rot_k)
    new_shape = apply_transform(np.zeros(shape, dtype=bool), flip_ud, rot_k).shape
    return sorted(result), new_shape


def bbox_of(coords):
    rs = [int(r) for r, _ in coords]
    cs = [int(c) for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def normalize_coords(coords):
    r0, c0, _, _ = bbox_of(coords)
    return sorted((int(r - r0), int(c - c0)) for r, c in coords)


def exact_matching_offsets(src_coords, dst_coords):
    src = set((int(r), int(c)) for r, c in src_coords)
    dst = set((int(r), int(c)) for r, c in dst_coords)
    offsets = []
    for sr, sc in src:
        for dr, dc in dst:
            off = (int(dr - sr), int(dc - sc))
            moved = {(int(r + off[0]), int(c + off[1])) for r, c in src}
            if moved == dst:
                offsets.append(off)
    return sorted(set(offsets))


def find_edge_stripes(grid, bg):
    H, W = grid.shape
    stripe = np.zeros((H, W), dtype=bool)
    colors = set()

    top_h = 0
    for r in range(H):
        row = grid[r, :]
        if (row == row[0]).all() and int(row[0]) != bg:
            top_h += 1
        else:
            break
    if top_h:
        stripe[:top_h, :] = True
        colors.add(int(grid[0, 0]))

    bot_h = 0
    for r in range(H - 1, -1, -1):
        row = grid[r, :]
        if (row == row[0]).all() and int(row[0]) != bg:
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
        if (col == col[0]).all() and int(col[0]) != bg:
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
        if (col == col[0]).all() and int(col[0]) != bg:
            right_w += 1
        else:
            break
    if right_w:
        stripe[inner_r0:inner_r1, W - right_w :] = True
        colors.add(int(grid[inner_r0, W - 1]))

    return stripe, colors, (top_h, bot_h, left_w, right_w)


def find_internal_stripes(grid, bg, edge_info):
    H, W = grid.shape
    top_h, bot_h, left_w, right_w = edge_info
    r0, r1, c0, c1 = top_h, H - bot_h, left_w, W - right_w
    stripe = np.zeros((H, W), dtype=bool)
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


def regions_from_stripes(stripe_mask):
    non = ~stripe_mask
    lab, n = label(non)
    regs = []
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (int(r1 - r0 + 1) >= 5) and (int(c1 - c0 + 1) >= 5):
            regs.append({"r0": int(r0), "r1": int(r1), "c0": int(c0), "c1": int(c1)})
    return regs


def boundary_dirs_for_region(grid, reg, stripe_colors):
    H, W = grid.shape
    dirs = {}
    r0, r1, c0, c1 = reg["r0"], reg["r1"], reg["c0"], reg["c1"]
    if r0 - 1 >= 0:
        mc = Counter(grid[r0 - 1, c0 : c1 + 1].tolist()).most_common(1)[0][0]
        if int(mc) in stripe_colors:
            dirs[int(mc)] = 0
    if r1 + 1 < H:
        mc = Counter(grid[r1 + 1, c0 : c1 + 1].tolist()).most_common(1)[0][0]
        if int(mc) in stripe_colors:
            dirs[int(mc)] = 2
    if c0 - 1 >= 0:
        mc = Counter(grid[r0 : r1 + 1, c0 - 1].tolist()).most_common(1)[0][0]
        if int(mc) in stripe_colors:
            dirs[int(mc)] = 3
    if c1 + 1 < W:
        mc = Counter(grid[r0 : r1 + 1, c1 + 1].tolist()).most_common(1)[0][0]
        if int(mc) in stripe_colors:
            dirs[int(mc)] = 1
    return dirs


def station_groups(grid):
    H, W = grid.shape
    sm = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != 9:
                continue
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) == 8:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                sm[r, c] = True
    lab, n = label(sm, structure=np.ones((3, 3), dtype=int))
    groups = []
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        groups.append([(int(r), int(c)) for r, c in coords])
    return groups


def split_station_subclusters(group, shape):
    mask = np.zeros(shape, dtype=bool)
    for r, c in group:
        mask[int(r), int(c)] = True
    lab, n = label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int))
    subs = []
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        cells = [(int(r), int(c)) for r, c in coords]
        nr = normalize_coords(cells)
        br0, bc0, _, _ = bbox_of(cells)
        subs.append(
            {
                "cells": cells,
                "norm": nr,
                "size": len(cells),
                "anchor": (int(br0), int(bc0)),
            }
        )
    subs.sort(key=lambda x: (x["size"], x["anchor"][0], x["anchor"][1]))
    return subs


def choose_best_ops(train, divider_dirs):
    bbox0 = train["bbox"]
    H0, W0 = bbox0.shape
    markers_by_color = defaultdict(list)
    for v, r, c in train["marker_pixels"]:
        if int(v) in divider_dirs:
            markers_by_color[int(v)].append((int(r), int(c)))
    assert markers_by_color, "Each train must expose divider-colored markers"
    assert train["body9"], "Each train must carry at least one coupler"

    ref_r = sum(r for r, _ in train["body9"]) / len(train["body9"])
    ref_c = sum(c for _, c in train["body9"]) / len(train["body9"])
    scored = []
    for flip_ud, k in OPS:
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
        scored.append((int(score), (flip_ud, k)))
    scored.sort(key=lambda item: (-item[0], OPS.index(item[1])))
    assert scored[0][0] > 0, "No orientation agrees with marker directions"
    return scored


def extract_trains(grid, bg, stripe_mask, station_set, stripe_colors):
    comp_mask = (grid != bg) & (~stripe_mask)
    lab, n = label(comp_mask)
    trains = []
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        vals = [int(grid[r, c]) for r, c in coords]
        if 8 not in vals:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        bbox = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        abs_coords = [(int(r), int(c)) for r, c in coords]
        body9 = []
        marker_pixels = []
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
                "body9": sorted(body9),
                "body9_norm": normalize_coords(body9),
                "marker_pixels": marker_pixels,
                "source_key": (int(r0), int(c0)),
            }
        )
    return trains


def patch_fits_output(tbbox, bg, region, out, stripe_mask, off_r, off_c):
    H1, W1 = tbbox.shape
    for r in range(H1):
        for c in range(W1):
            v = int(tbbox[r, c])
            if v == bg:
                continue
            rr, cc = r + off_r, c + off_c
            if not (region["r0"] <= rr <= region["r1"] and region["c0"] <= cc <= region["c1"]):
                return False
            if stripe_mask[rr, cc]:
                return False
            if v == 9:
                if int(out[rr, cc]) != 9:
                    return False
            elif int(out[rr, cc]) != bg:
                return False
    return True


def paint_patch(out, tbbox, bg, off_r, off_c):
    H1, W1 = tbbox.shape
    for r in range(H1):
        for c in range(W1):
            v = int(tbbox[r, c])
            if v != bg and v != 9:
                out[r + off_r, c + off_c] = v


def extract_scene(grid):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    edge_mask, edge_colors, edge_info = find_edge_stripes(g, bg)
    internal_mask, internal_colors = find_internal_stripes(g, bg, edge_info)
    stripe_mask = edge_mask | internal_mask
    stripe_colors = edge_colors | internal_colors
    regions = regions_from_stripes(stripe_mask)

    groups = station_groups(g)
    station_set = set(pt for grp in groups for pt in grp)
    stations = []
    for idx, grp in enumerate(groups):
        matches = [
            reg
            for reg in regions
            if all(reg["r0"] <= r <= reg["r1"] and reg["c0"] <= c <= reg["c1"] for r, c in grp)
        ]
        assert len(matches) == 1, "Each station group must belong to exactly one region"
        reg = matches[0]
        subs = split_station_subclusters(grp, g.shape)
        stations.append(
            {
                "station_idx": idx,
                "group": grp,
                "region": reg,
                "divider_dirs": boundary_dirs_for_region(g, reg, stripe_colors),
                "subclusters": subs,
            }
        )

    trains = extract_trains(g, bg, stripe_mask, station_set, stripe_colors)
    return {
        "grid": g,
        "bg": bg,
        "H": H,
        "W": W,
        "stripe_mask": stripe_mask,
        "stripe_colors": stripe_colors,
        "regions": regions,
        "stations": stations,
        "trains": trains,
    }


def _solve(grid: np.ndarray, collect_meta: bool = False):
    scene = extract_scene(grid)
    g = scene["grid"]
    bg = scene["bg"]
    stripe_mask = scene["stripe_mask"]
    stations = scene["stations"]
    trains = scene["trains"]

    out = g.copy()
    for t in trains:
        for r, c in t["abs_coords"]:
            out[int(r), int(c)] = bg

    available = []
    for st in stations:
        for sub_idx, sub in enumerate(st["subclusters"]):
            available.append((st["station_idx"], sub_idx, sub["size"]))

    trains_sorted = sorted(trains, key=lambda t: (len(t["body9"]), t["source_key"][0], t["source_key"][1]))
    used_subclusters = set()
    placement_meta = []

    for order_idx, t in enumerate(trains_sorted):
        n9 = len(t["body9"])
        candidate_records = []
        blocked_checks = 0
        for st in sorted(stations, key=lambda x: (-len(x["subclusters"]), x["station_idx"])):
            ops = choose_best_ops(t, st["divider_dirs"])
            for sub_idx, sub in enumerate(st["subclusters"]):
                if sub["size"] != n9:
                    continue
                if (st["station_idx"], sub_idx) in used_subclusters:
                    continue
                for score, op in ops:
                    t_body9, _ = xform_coords(t["body9"], t["bbox"].shape, *op)
                    if normalize_coords(t_body9) != sub["norm"]:
                        continue
                    offsets = exact_matching_offsets(t_body9, sub["cells"])
                    for off_r, off_c in offsets:
                        tbbox = apply_transform(t["bbox"], *op)
                        fits = patch_fits_output(tbbox, bg, st["region"], out, stripe_mask, off_r, off_c)
                        if fits:
                            candidate_records.append(
                                (
                                    -score,
                                    st["station_idx"],
                                    sub["anchor"][0],
                                    sub["anchor"][1],
                                    OPS.index(op),
                                    tbbox,
                                    off_r,
                                    off_c,
                                    sub_idx,
                                )
                            )
                        else:
                            blocked_checks += 1

        assert candidate_records, f"No valid docking candidate for train with {n9} couplers"
        candidate_records.sort(key=lambda x: x[:5])
        _, station_idx, _, _, _, tbbox, off_r, off_c, sub_idx = candidate_records[0]
        paint_patch(out, tbbox, bg, off_r, off_c)
        used_subclusters.add((station_idx, sub_idx))
        placement_meta.append(
            {
                "order_idx": int(order_idx),
                "couplers": int(n9),
                "station_idx": int(station_idx),
                "sub_idx": int(sub_idx),
                "blocked_candidates": int(blocked_checks),
                "valid_candidates": int(len(candidate_records)),
            }
        )

    if collect_meta:
        mega_station = max(stations, key=lambda st: (len(st["subclusters"]), len(st["group"])))
        return out, {
            "train_count": len(trains_sorted),
            "largest_station_subclusters": len(mega_station["subclusters"]),
            "largest_station_region": mega_station["region"],
            "placements": placement_meta,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock several train sprites onto different lobes of one shared station, using a size-ordered sequence.

    Input structure:
    - The canvas is partitioned by thick colored stripes into 1-2 rectangular regions.
    - Color 8 forms train bodies; color 9 appears both as train couplers and as one large isolated station.
    - That station is 8-connected as a whole, but internally it splits into 2-3 corner-touching 4-connected lobes.
    - Stripe-colored pixels on each train act as orientation markers, just as in the base task.

    Rule:
    - Clear every source train from its starting location.
    - Split the isolated station into 4-connected lobes, then process trains from fewer couplers to more couplers.
    - For each train, choose the rotation/flip whose markers face the matching stripe colors of the target region, and dock the train onto one unused lobe whose 9-shape matches the transformed couplers.
    - Later trains must fit around the already docked earlier ones, so occupied cells in the shared region act as obstacles.

    Key insight:
    - The visible "one big station" is not one destination for one train: it is a bundle of smaller docking lobes.
    - Coupler shape/size selects the lobe, stripe-color markers select the orientation, and ascending coupler count determines the docking order.
    """
    return _solve(np.array(grid, dtype=int))


def analyze_input(grid: np.ndarray) -> dict:
    _, meta = _solve(np.array(grid, dtype=int), collect_meta=True)
    return meta


def _collect_candidate_records(scene, out, train, used_subclusters):
    bg = scene["bg"]
    stripe_mask = scene["stripe_mask"]
    stations = scene["stations"]
    n9 = len(train["body9"])
    records = []
    for st in sorted(stations, key=lambda x: (-len(x["subclusters"]), x["station_idx"])):
        ops = choose_best_ops(train, st["divider_dirs"])
        for sub_idx, sub in enumerate(st["subclusters"]):
            if sub["size"] != n9 or (st["station_idx"], sub_idx) in used_subclusters:
                continue
            for score, op in ops:
                t_body9, _ = xform_coords(train["body9"], train["bbox"].shape, *op)
                if normalize_coords(t_body9) != sub["norm"]:
                    continue
                tbbox = apply_transform(train["bbox"], *op)
                for off_r, off_c in exact_matching_offsets(t_body9, sub["cells"]):
                    if patch_fits_output(tbbox, bg, st["region"], out, stripe_mask, off_r, off_c):
                        records.append(
                            (
                                -score,
                                st["station_idx"],
                                sub["anchor"][0],
                                sub["anchor"][1],
                                OPS.index(op),
                                tbbox,
                                off_r,
                                off_c,
                                sub_idx,
                            )
                        )
    records.sort(key=lambda item: item[:5])
    return records


def _solve_with_order(grid: np.ndarray, reverse: bool = False):
    scene = extract_scene(grid)
    g = scene["grid"]
    bg = scene["bg"]
    trains = scene["trains"]
    out = g.copy()
    for t in trains:
        for r, c in t["abs_coords"]:
            out[int(r), int(c)] = bg

    trains_sorted = sorted(
        trains,
        key=lambda t: (len(t["body9"]), t["source_key"][0], t["source_key"][1]),
        reverse=reverse,
    )
    used_subclusters = set()
    traces = []
    empty_board = out.copy()

    for order_idx, t in enumerate(trains_sorted):
        empty_records = _collect_candidate_records(scene, empty_board, t, used_subclusters)
        current_records = _collect_candidate_records(scene, out, t, used_subclusters)
        traces.append(
            {
                "order_idx": int(order_idx),
                "couplers": int(len(t["body9"])),
                "empty_count": int(len(empty_records)),
                "current_count": int(len(current_records)),
                "empty_top": empty_records[0][:5] if empty_records else None,
                "current_top": current_records[0][:5] if current_records else None,
            }
        )
        if not current_records:
            return False, None, traces
        _, station_idx, _, _, _, tbbox, off_r, off_c, sub_idx = current_records[0]
        paint_patch(out, tbbox, bg, off_r, off_c)
        used_subclusters.add((station_idx, sub_idx))
    return True, out, traces


def _order_dependency_profile(grid: np.ndarray) -> dict:
    ok_fwd, out_fwd, traces_fwd = _solve_with_order(grid, reverse=False)
    ok_rev, _, traces_rev = _solve_with_order(grid, reverse=True)
    later_restricted = False
    top_changed = False
    last_restricted = False
    for trace in traces_fwd[1:]:
        if trace["current_count"] < trace["empty_count"]:
            later_restricted = True
        if trace["current_top"] != trace["empty_top"]:
            top_changed = True
    if len(traces_fwd) >= 2:
        last_restricted = traces_fwd[-1]["current_count"] < traces_fwd[-1]["empty_count"]
    return {
        "forward_ok": bool(ok_fwd),
        "forward_out": out_fwd,
        "reverse_ok": bool(ok_rev),
        "later_restricted": bool(later_restricted),
        "last_restricted": bool(last_restricted),
        "top_changed": bool(top_changed),
        "forward_traces": traces_fwd,
        "reverse_traces": traces_rev,
    }


# --------------- generator ---------------

# Grid / region sizing
GRID_H_RANGE = (18, 34)
GRID_W_RANGE = (20, 33)
GRID_MIN_AREA = 360
MIN_REGION_DIM = 12
MIN_REGION_DIRS = 3
INTERNAL_STRIPE_THICK = 2
STRIPE_THICK_RANGE = (2, 4)
INTERNAL_PROB = 0.58

# Train / station sizing
TRAIN_COUNTS = [2, 3]
TRAIN_SIZE_CHOICES = {
    2: [[1, 2], [1, 4], [2, 4]],
    3: [[1, 2, 4]],
}
BODY_MIN = 5
BODY_MAX = 11
PATCH_H_RANGE = (4, 8)
PATCH_W_RANGE = (3, 7)
STATION_MARGIN = 4

# Marker placement
MARKER_DIST_RANGE = (2, 3)
MARKER_JITTER = [-1, 0, 1]
THREE_MARKER_PROB = 0.7

# Growth / search
BODY_GROWTH_ITERS = 150
PATH_BOUNDS_PAD = 4
LAYOUT_RETRIES = 200
DOCK_RETRIES = 40
GENERATE_RETRIES = 350
PATCH_POOL_SIZE = 10
SCATTER_RETRIES = 18
COLOR_POOL = list(range(8))

_BRANCH_CURSOR = {count: 0 for count in TRAIN_SIZE_CHOICES}
_TRAIN_COUNT_CURSOR = 0


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
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _adjacent8(p, cells):
    r, c = p
    return any(abs(int(r) - int(rr)) <= 1 and abs(int(c) - int(cc)) <= 1 for rr, cc in cells)


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
            p = (int(r + dr), int(c + dc))
            if p in prev or p in blocked:
                continue
            if not (r0 <= p[0] <= r1 and c0 <= p[1] <= c1):
                continue
            prev[p] = (int(r), int(c))
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
        regions = []
        internal = None
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
        if random.random() < 0.5:
            return {(0, 0), (0, 1)}
        return {(0, 0), (1, 0)}
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
        return int(r0 - d), int(random.randint(c0, c1) + jitter)
    if direction == 2:
        return int(r1 + d), int(random.randint(c0, c1) + jitter)
    if direction == 3:
        return int(random.randint(r0, r1) + jitter), int(c0 - d)
    return int(random.randint(r0, r1) + jitter), int(c1 + d)


def _inward_cell(marker, direction):
    dr, dc = DIRS4[(direction + 2) % 4]
    return int(marker[0] + dr), int(marker[1] + dc)


def _connect_couplers_to_body(couplers, body, blocked, bounds):
    for cp in couplers:
        if _adjacent8(cp, body):
            continue
        cr, cc = cp
        choices = [
            (int(cr + dr), int(cc + dc))
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc) and (int(cr + dr), int(cc + dc)) not in blocked
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
                q = (int(r + dr), int(c + dc))
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


def _build_docked_patch(bg, dirs, couplers):
    couplers = set((int(r), int(c)) for r, c in couplers)
    specs = _marker_specs(dirs)
    markers, marker_cells = [], set()
    for color, direction in specs:
        p = _marker_cell(couplers, direction)
        if p in couplers or p in marker_cells:
            return None
        markers.append((p, int(color), int(direction)))
        marker_cells.add(p)

    blocked = set(couplers) | marker_cells
    contacts = [
        q
        for r, c in couplers
        for dr, dc in DIRS4
        if (q := (int(r + dr), int(c + dc))) not in blocked
    ]
    if not contacts:
        return None

    body = {random.choice(contacts)}
    all_seed = set(couplers) | marker_cells | body
    sr0, sc0, sr1, sc1 = _bbox(all_seed)
    bounds = (int(sr0 - PATH_BOUNDS_PAD), int(sr1 + PATH_BOUNDS_PAD), int(sc0 - PATH_BOUNDS_PAD), int(sc1 + PATH_BOUNDS_PAD))

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

    all_cells = body | couplers | marker_cells
    h, w = _dims(all_cells)
    if not (BODY_MIN <= len(body) <= BODY_MAX and PATCH_H_RANGE[0] <= h <= PATCH_H_RANGE[1] and PATCH_W_RANGE[0] <= w <= PATCH_W_RANGE[1]):
        return None

    r0, c0, _, _ = _bbox(all_cells)
    patch = np.full((h, w), bg, dtype=int)
    for r, c in body:
        patch[int(r - r0), int(c - c0)] = 8
    for (r, c), color, _ in markers:
        patch[int(r - r0), int(c - c0)] = int(color)
    rel_couplers = []
    for r, c in couplers:
        patch[int(r - r0), int(c - c0)] = 9
        rel_couplers.append((int(r - r0), int(c - c0)))
    return patch, sorted(rel_couplers)


def _diag_connected_without_edge(candidate, occupied):
    has_diag = False
    for r, c in candidate:
        for rr, cc in occupied:
            dr = abs(int(r - rr))
            dc = abs(int(c - cc))
            if dr + dc == 1:
                return False
            if max(dr, dc) == 1:
                has_diag = True
    return has_diag


def _build_station_cluster(shapes):
    placed = []
    occupied = set()
    for idx, shape in enumerate(shapes):
        shape = set((int(r), int(c)) for r, c in normalize_coords(shape))
        if idx == 0:
            placed.append(shape)
            occupied |= shape
            continue
        _, _, r1, c1 = _bbox(occupied)
        found = None
        for base_r in range(max(0, r1 - 1), r1 + 6):
            for base_c in range(max(0, c1 - 1), c1 + 6):
                cand = {(int(r + base_r), int(c + base_c)) for r, c in shape}
                if cand & occupied:
                    continue
                if _diag_connected_without_edge(cand, occupied):
                    found = cand
                    break
            if found is not None:
                break
        if found is None:
            return None
        placed.append(found)
        occupied |= found

    ur0, uc0, _, _ = _bbox(occupied)
    union = {(int(r - ur0), int(c - uc0)) for r, c in occupied}
    normalized_subs = []
    for sub in placed:
        normalized_subs.append(sorted((int(r - ur0), int(c - uc0)) for r, c in sub))
    order = sorted(
        range(len(normalized_subs)),
        key=lambda i: (len(normalized_subs[i]), bbox_of(normalized_subs[i])[0], bbox_of(normalized_subs[i])[1]),
    )
    return [normalized_subs[i] for i in order], union


def _station_anchor_candidates(region, station_union, margin):
    h, w = _dims(station_union)
    r0, c0, _, _ = _bbox(station_union)
    rel_union = normalize_coords(station_union)
    anchors = []
    for ar in range(region["r0"] + margin, region["r1"] - margin - h + 2):
        for ac in range(region["c0"] + margin, region["c1"] - margin - w + 2):
            anchors.append((int(ar - r0), int(ac - c0), rel_union))
    random.shuffle(anchors)
    return anchors


def _place_station_abs(subclusters_rel, offset):
    off_r, off_c = offset
    abs_subs = []
    union = []
    for sub in subclusters_rel:
        abs_sub = [(int(r + off_r), int(c + off_c)) for r, c in sub]
        abs_subs.append(abs_sub)
        union.extend(abs_sub)
    return abs_subs, union


def _patch_can_land(output, patch, bg, region, off_r, off_c):
    ph, pw = patch.shape
    for r in range(ph):
        for c in range(pw):
            v = int(patch[r, c])
            if v == bg:
                continue
            rr, cc = int(off_r + r), int(off_c + c)
            if not (region["r0"] <= rr <= region["r1"] and region["c0"] <= cc <= region["c1"]):
                return False
            if v == 9:
                if int(output[rr, cc]) != 9:
                    return False
            elif int(output[rr, cc]) != bg:
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
            ok = True
            for pr, pc in coords:
                if occupied[int(rr + pr), int(cc + pc)]:
                    ok = False
                    break
            if ok:
                locs.append((int(rr), int(cc)))
    return locs


def _next_train_count():
    global _TRAIN_COUNT_CURSOR
    count = TRAIN_COUNTS[_TRAIN_COUNT_CURSOR % len(TRAIN_COUNTS)]
    _TRAIN_COUNT_CURSOR += 1
    return int(count)


def _next_size_choice(train_count):
    choices = TRAIN_SIZE_CHOICES[train_count]
    idx = _BRANCH_CURSOR[train_count] % len(choices)
    _BRANCH_CURSOR[train_count] += 1
    return sorted(int(v) for v in choices[idx])


def _candidate_patch_signature(patch, bg, off_r, off_c):
    cells = []
    ph, pw = patch.shape
    for r in range(ph):
        for c in range(pw):
            v = int(patch[r, c])
            if v != bg:
                cells.append((r, c, v))
    return (tuple(cells), int(off_r), int(off_c))


def _occupied_cells_for_patch(patch, bg, off_r, off_c):
    cells = set()
    ph, pw = patch.shape
    for r in range(ph):
        for c in range(pw):
            v = int(patch[r, c])
            if v != bg and v != 9:
                cells.add((int(off_r + r), int(off_c + c)))
    return cells


def _sample_patch_pool(bg, dirs, sub_rel, sub_abs, region, station_board, limit):
    pool = []
    seen = set()
    for _ in range(DOCK_RETRIES * 3):
        built_patch = _build_docked_patch(bg, dirs, sub_rel)
        if built_patch is None:
            continue
        patch, coupler_rel = built_patch
        offsets = exact_matching_offsets(coupler_rel, sub_abs)
        if len(offsets) != 1:
            continue
        off_r, off_c = offsets[0]
        if not _patch_can_land(station_board, patch, bg, region, off_r, off_c):
            continue
        sig = _candidate_patch_signature(patch, bg, off_r, off_c)
        if sig in seen:
            continue
        seen.add(sig)
        pool.append(
            {
                "patch": patch,
                "off_r": int(off_r),
                "off_c": int(off_c),
                "occupied": _occupied_cells_for_patch(patch, bg, off_r, off_c),
            }
        )
        if len(pool) >= limit:
            break
    return pool


def _pool_fit_count(pool, occupied):
    return sum(1 for cand in pool if not (cand["occupied"] & occupied))


def _choose_docked_combo(pools):
    base_counts = [len(pool) for pool in pools]
    best_score = None
    best_choice = None

    def score_choice(choice):
        occupied = set()
        reductions = 0
        reduced_slots = 0
        last_reduced = False
        for idx, cand in enumerate(choice):
            if cand["occupied"] & occupied:
                return None
            if idx + 1 < len(choice):
                next_count = _pool_fit_count(pools[idx + 1], occupied | cand["occupied"])
                if next_count < base_counts[idx + 1]:
                    reductions += base_counts[idx + 1] - next_count
                    reduced_slots += 1
                    if idx + 1 == len(choice) - 1:
                        last_reduced = True
            occupied |= cand["occupied"]
        return (int(last_reduced), int(reduced_slots), int(reductions), -len(occupied))

    def backtrack(idx, occupied, choice):
        nonlocal best_choice, best_score
        if idx == len(pools):
            score = score_choice(choice)
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_choice = list(choice)
            return
        ordered = pools[idx][:]
        random.shuffle(ordered)
        for cand in ordered:
            if cand["occupied"] & occupied:
                continue
            backtrack(idx + 1, occupied | cand["occupied"], choice + [cand])

    backtrack(0, set(), [])
    if best_score is None or best_score[:3] <= (0, 0, 0):
        return None
    return best_choice


def _place_docked_scene(base, bg, regions, sizes):
    target_reg = max(regions, key=lambda reg: ((reg["r1"] - reg["r0"] + 1) * (reg["c1"] - reg["c0"] + 1), reg["r0"], reg["c0"]))
    shapes = [_station_shape(size) for size in sizes]
    built = _build_station_cluster(shapes)
    if built is None:
        return None
    subclusters_rel, station_union_rel = built

    anchors = _station_anchor_candidates(target_reg, station_union_rel, STATION_MARGIN)
    if not anchors:
        return None

    abs_subclusters = None
    station_union_abs = None
    for off_r, off_c, _ in anchors:
        trial_subs, trial_union = _place_station_abs(subclusters_rel, (off_r, off_c))
        if all(target_reg["r0"] <= r <= target_reg["r1"] and target_reg["c0"] <= c <= target_reg["c1"] for r, c in trial_union):
            abs_subclusters = trial_subs
            station_union_abs = trial_union
            break
    if abs_subclusters is None:
        return None

    station_board = base.copy()
    for r, c in station_union_abs:
        station_board[int(r), int(c)] = 9

    pools = []
    for sub_rel, sub_abs in zip(subclusters_rel, abs_subclusters):
        pool = _sample_patch_pool(bg, target_reg["dirs"], sub_rel, sub_abs, target_reg, station_board, PATCH_POOL_SIZE)
        if len(pool) < 2:
            return None
        pools.append(pool)

    chosen = _choose_docked_combo(pools)
    if chosen is None:
        return None

    output = station_board.copy()
    docked = []
    for size, sub_abs, cand in zip(sizes, abs_subclusters, chosen):
        paste(output, cand["patch"], cand["off_r"], cand["off_c"], transparent=bg)
        docked.append(
            {
                "patch": cand["patch"],
                "station": sorted(sub_abs),
                "size": int(size),
            }
        )
    return output, docked, station_union_abs


def _scatter_trains(base, bg, regions, trains, station_union_abs):
    H, W = base.shape
    input_grid = base.copy()
    station_mask = np.zeros((H, W), dtype=bool)
    for r, c in station_union_abs:
        input_grid[int(r), int(c)] = 9
        station_mask[int(r), int(c)] = True

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
    shuffled_regions = regions[:]
    for t, op in zip(trains, ops):
        in_patch = _inverse_patch(t["patch"], op)
        random.shuffle(shuffled_regions)
        all_locs = []
        for reg in shuffled_regions:
            locs = _patch_locations(reg, in_patch, bg, forbidden, margin=1)
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


def _gen_attempt(n_trains, sizes):
    layout = _layout()
    if layout is None:
        return None
    base, bg, regions = layout

    placed = _place_docked_scene(base, bg, regions, sizes)
    if placed is None:
        return None
    output, trains, station_union_abs = placed

    for _ in range(SCATTER_RETRIES):
        input_grid = _scatter_trains(base, bg, regions, trains, station_union_abs)
        if input_grid is None:
            continue
        counts = Counter(map(int, input_grid.flatten()))
        if counts[bg] != max(counts.values()):
            continue
        try:
            solved, meta = _solve(input_grid, collect_meta=True)
            profile = _order_dependency_profile(input_grid)
            if not np.array_equal(solved, output):
                continue
            if meta["largest_station_subclusters"] != n_trains:
                continue
            if len(meta["placements"]) != n_trains:
                continue
            if not profile["forward_ok"] or not np.array_equal(profile["forward_out"], output):
                continue
            if not profile["last_restricted"]:
                continue
            return input_grid
        except Exception:
            continue
    return None


def generate_input() -> np.ndarray:
    n_trains = _next_train_count()
    for _ in range(GENERATE_RETRIES):
        sizes = _next_size_choice(n_trains)
        grid = _gen_attempt(n_trains, sizes)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fused 6e4f6532 input")
