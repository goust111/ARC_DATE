# Combo variant of b9e38dc0 (index 7)
# combo_spec: D2/1+D5/0
# Name: 碰撞停止+辐射碰精灵停止

from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    The scene now always contains at least two same-colored half-enclosure sprites. The fill seed sits in the
#    active sprite, namely the wall component nearest to that seed. The second wall sprite is placed in the
#    radiate direction and serves as one shared collision target.
#    Small example: if the active enclosure opens upward, the original code radiates the wedge until row 0. In
#    this combo, the endpoint rays are traced upward but stop right before the lower border of the upper wall
#    sprite. In the same scene, one or more isolated single pixels sit inside the would-be filled wedge. The
#    original summary says such pixels would ride straight to the edge, erasing a background trail behind them.
#    The fused rule changes that too: each pixel still moves straight in the global radiate direction, but it
#    halts on the last free cell before the same upper wall sprite. Thus one object-to-object distance controls
#    both the wedge height and the mover travel distance.
# b) Complexity calibration:
#    To keep the fusion visible rather than toy-like, the generator does not use a trivial 1-cell stop marker.
#    It places a small same-colored half-enclosure blocker in front of one wedge boundary, so the stop condition
#    is still explained by ordinary wall geometry. At the same time it keeps only 1-3 isolated movers and omits
#    extra shadow obstacles, avoiding a cluttered scene where the new collision logic would be buried under
#    unrelated artifacts.
# c) Implicit-assumption audit:
#    - The source solver chose the fill color by "closest non-wall color". That becomes fragile once extra actor
#      colors are added, so this combo instead identifies the fill as the nearest non-wall component, while the
#      generator keeps movers farther from the wall and keeps the seed blob non-singleton for readability.
#    - The old ray builders assumed only one termination mode: leaving the grid. That assumption is invalid for
#      D5/0, so the combo traces each ray step-by-step and stops when the next cell is another wall component.
#    - The original flood-fill treated every extra non-bg pixel as a blocker. That would wrongly let movers
#      change reachability. Here singleton movers are treated as passable during the fill phase and only move in
#      a later pass, which matches the intended "fill first, then carve a bg trail" behavior.
#    - The mover pass assumes there is a well-defined radiate direction and that the next wall sprite lies ahead
#      on at least one candidate scanline. The generator enforces both and rejects samples where either fused
#      effect would match the edge-baseline output.
# d) Placement legality:
#    The blocker sprite is synthesized from a candidate boundary-ray hit point, then validated as a separate
#    8-connected wall component fully inside the corridor. Movers are chosen only from cells inside the nominal
#    truncated fill zone, with path length >= 2 and distinct scanlines so their trails do not overlap or run
#    into the wall sprites.
# e) Visual observability:
#    Across examples, the visible gap between the two wall sprites changes, and both the wedge truncation and the
#    mover displacement change with it. Because the mover trails are background-colored cuts inside a filled
#    region, the combo reads as one coherent "collision-stop" rule rather than two unrelated hacks.
# f) Visual cleanliness:
#    The palette is kept small: bg, wall, fill, plus 1-2 mover colors. Movers are isolated singletons, the seed
#    stays near the closed end, and the blocker sits cleanly in front of the opening. This prevents ambiguity
#    between structural wall pixels, the fill region, and the mobile singletons.
# g) Change inferability:
#    No hidden color ordering is used. The same blocker geometry repeatedly explains both changes: rays truncate
#    early and isolated pixels fail to reach the border. The generator explicitly rejects degenerate grids where
#    the combo output would equal either the "rays go to edge" baseline or the "movers go to edge" baseline.


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _component_sets(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    return [set((int(r), int(c)) for r, c in pts) for pts in components_mask(mask, connectivity=8)]


def _component_mask(shape: tuple[int, int], comp: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in comp:
        mask[r, c] = True
    return mask


def _comp_neighbors(comp_mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = comp_mask.shape
    out = []
    for rr, cc in neighbors8(r, c):
        if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
            out.append((rr, cc))
    return out


def _outward_dir(comp_mask: np.ndarray, ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
    n1 = _comp_neighbors(comp_mask, *ep)[0]
    d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))  # inward
    d_use = d1
    is_12 = False
    n2s = [p for p in _comp_neighbors(comp_mask, *n1) if p != ep]
    if n2s:
        n2 = n2s[0]
        d2 = (_sgn(n2[0] - n1[0]), _sgn(n2[1] - n1[1]))
        if (d1[0] == 0) ^ (d1[1] == 0):
            if d2[0] != 0 and d2[1] != 0:
                d_use = d2
                is_12 = True
    return (-d_use[0], -d_use[1]), is_12


def _component_ray_info(comp: set[tuple[int, int]], shape: tuple[int, int]) -> dict:
    comp_mask = _component_mask(shape, comp)
    endpoints = sorted([(r, c) for r, c in comp if len(_comp_neighbors(comp_mask, r, c)) == 1])
    assert len(endpoints) >= 2, "expected wall component to have endpoints"
    out_info = [_outward_dir(comp_mask, ep) for ep in endpoints]
    out_dirs = [d for d, _ in out_info]
    tip_12 = [flag for _, flag in out_info]
    dr_sum = sum(d[0] for d in out_dirs)
    dc_sum = sum(d[1] for d in out_dirs)
    rad = (_sgn(dr_sum), _sgn(dc_sum))
    return {
        "comp_mask": comp_mask,
        "endpoints": endpoints,
        "out_dirs": out_dirs,
        "tip_12": tip_12,
        "rad": rad,
    }


def _trace_ray(
    start: tuple[int, int],
    d: tuple[int, int],
    rad: tuple[int, int],
    is_12: bool,
    shape: tuple[int, int],
    stop_mask: np.ndarray | None = None,
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    h, w = shape
    rr, cc = start
    step = 0
    path: list[tuple[int, int]] = []
    while True:
        step += 1
        if not is_12 or step % 2 == 1:
            nr = rr + d[0]
            nc = cc + d[1]
        else:
            if rad[0] != 0:
                nr = rr + d[0]
                nc = cc
            else:
                nr = rr
                nc = cc + d[1]
        if not (0 <= nr < h and 0 <= nc < w):
            return path, None
        if stop_mask is not None and stop_mask[nr, nc]:
            return path, (nr, nc)
        path.append((nr, nc))
        rr, cc = nr, nc


def _trace_line(
    start: tuple[int, int],
    d: tuple[int, int],
    shape: tuple[int, int],
    stop_mask: np.ndarray | None = None,
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    h, w = shape
    rr, cc = start
    path: list[tuple[int, int]] = []
    while True:
        nr = rr + d[0]
        nc = cc + d[1]
        if not (0 <= nr < h and 0 <= nc < w):
            return path, None
        if stop_mask is not None and stop_mask[nr, nc]:
            return path, (nr, nc)
        path.append((nr, nc))
        rr, cc = nr, nc


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape
    bg = get_bg(g)
    vals = [int(v) for v in np.unique(g)]
    non_bg = [v for v in vals if v != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    wall_comps = _component_sets(wall_mask)
    assert wall_comps, "expected at least one wall connected component"

    actor_comps: list[dict] = []
    for color in non_bg:
        if color == wall:
            continue
        for pts in components_mask(g == color, connectivity=4):
            cells = [(int(r), int(c)) for r, c in pts]
            actor_comps.append({"color": color, "cells": cells})
    assert actor_comps, "expected at least one non-wall component"

    wall_points = [p for comp in wall_comps for p in comp]

    def comp_dist(comp_cells: list[tuple[int, int]]) -> tuple[int, int]:
        best = 10**9
        for r, c in comp_cells:
            for wr, wc in wall_points:
                d = abs(r - wr) + abs(c - wc)
                if d < best:
                    best = d
                    if best == 0:
                        return 0, -len(comp_cells)
        return best, -len(comp_cells)

    fill_comp = min(actor_comps, key=lambda rec: comp_dist(rec["cells"]))
    fill = int(fill_comp["color"])
    fill_pos = list(fill_comp["cells"])
    fill_mask = np.zeros((h, w), dtype=bool)
    for r, c in fill_pos:
        fill_mask[r, c] = True

    def wall_comp_dist(comp: set[tuple[int, int]]) -> int:
        best = 10**9
        for r, c in fill_pos:
            for wr, wc in comp:
                d = abs(r - wr) + abs(c - wc)
                if d < best:
                    best = d
                    if best == 0:
                        return 0
        return best

    wall_comp = min(wall_comps, key=lambda comp: (wall_comp_dist(comp), -len(comp)))
    ray_info = _component_ray_info(wall_comp, (h, w))
    comp_mask = ray_info["comp_mask"]
    other_wall_mask = wall_mask & (~comp_mask)

    movers: list[dict] = []
    mover_mask = np.zeros((h, w), dtype=bool)
    for rec in actor_comps:
        if rec is fill_comp:
            continue
        if len(rec["cells"]) == 1:
            r, c = rec["cells"][0]
            movers.append({"pos": (r, c), "color": int(rec["color"])})
            mover_mask[r, c] = True

    return {
        "grid": g,
        "shape": (h, w),
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "fill_pos": fill_pos,
        "fill_mask": fill_mask,
        "wall_mask": wall_mask,
        "wall_comp": wall_comp,
        "comp_mask": comp_mask,
        "other_wall_mask": other_wall_mask,
        "endpoints": ray_info["endpoints"],
        "out_dirs": ray_info["out_dirs"],
        "tip_12": ray_info["tip_12"],
        "rad": ray_info["rad"],
        "movers": sorted(movers, key=lambda rec: (rec["pos"][0], rec["pos"][1], rec["color"])),
        "mover_mask": mover_mask,
    }


def _build_barrier(scene: dict, *, stop_at_other_wall: bool) -> tuple[np.ndarray, list[dict]]:
    barrier = scene["comp_mask"].copy()
    stop_mask = scene["other_wall_mask"] if stop_at_other_wall else None
    ray_meta = []
    for ep, d, is_12 in zip(scene["endpoints"], scene["out_dirs"], scene["tip_12"]):
        path, hit = _trace_ray(ep, d, scene["rad"], is_12, scene["shape"], stop_mask)
        for r, c in path:
            barrier[r, c] = True
        ray_meta.append(
            {
                "start": ep,
                "dir": d,
                "is_12": is_12,
                "cells": path,
                "hit": hit,
            }
        )
    return barrier, ray_meta


def _move_movers(
    out: np.ndarray,
    scene: dict,
    *,
    mover_collision_stop: bool,
) -> list[dict]:
    stop_mask = scene["other_wall_mask"] if mover_collision_stop else None
    mover_meta = []
    for rec in scene["movers"]:
        start = tuple(rec["pos"])
        path, hit = _trace_line(start, scene["rad"], scene["shape"], stop_mask)
        if path:
            out[start] = scene["bg"]
            for cell in path[:-1]:
                out[cell] = scene["bg"]
            out[path[-1]] = rec["color"]
            end = path[-1]
        else:
            end = start
        mover_meta.append(
            {
                "start": start,
                "end": end,
                "color": rec["color"],
                "cells": path,
                "hit": hit,
            }
        )
    return mover_meta


def _solve(
    grid: np.ndarray,
    *,
    stop_at_other_wall: bool,
    mover_collision_stop: bool,
    return_meta: bool = False,
):
    scene = _parse_scene(grid)
    g = scene["grid"]

    barrier, ray_meta = _build_barrier(scene, stop_at_other_wall=stop_at_other_wall)
    passable = (~barrier) & ((g == scene["bg"]) | scene["fill_mask"] | scene["mover_mask"])

    q = scene["fill_pos"][:]
    seen = np.zeros_like(passable, dtype=bool)
    for r, c in q:
        if passable[r, c]:
            seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))

    filled_mask = seen & (g == scene["bg"])
    out = g.copy()
    out[filled_mask] = scene["fill"]
    mover_meta = _move_movers(out, scene, mover_collision_stop=mover_collision_stop)

    if return_meta:
        return out, {
            "scene": scene,
            "ray_meta": ray_meta,
            "mover_meta": mover_meta,
            "filled_mask": filled_mask.copy(),
        }
    return out


def summarize_behavior(grid: np.ndarray) -> dict:
    combo_out, combo = _solve(grid, stop_at_other_wall=True, mover_collision_stop=True, return_meta=True)
    ray_edge_out, ray_edge = _solve(grid, stop_at_other_wall=False, mover_collision_stop=True, return_meta=True)
    mover_edge_out, mover_edge = _solve(grid, stop_at_other_wall=True, mover_collision_stop=False, return_meta=True)
    full_edge_out, full_edge = _solve(grid, stop_at_other_wall=False, mover_collision_stop=False, return_meta=True)

    ray_saved = sum(len(rec["cells"]) for rec in ray_edge["ray_meta"]) - sum(len(rec["cells"]) for rec in combo["ray_meta"])
    mover_saved = sum(len(rec["cells"]) for rec in mover_edge["mover_meta"]) - sum(len(rec["cells"]) for rec in combo["mover_meta"])

    return {
        "combo_out": combo_out,
        "ray_edge_out": ray_edge_out,
        "mover_edge_out": mover_edge_out,
        "full_edge_out": full_edge_out,
        "combo": combo,
        "ray_edge": ray_edge,
        "mover_edge": mover_edge,
        "full_edge": full_edge,
        "ray_saved": int(ray_saved),
        "mover_saved": int(mover_saved),
        "ray_hits": int(sum(rec["hit"] is not None for rec in combo["ray_meta"])),
        "mover_hits": int(sum(rec["hit"] is not None for rec in combo["mover_meta"])),
        "combo_vs_ray_edge": int(np.count_nonzero(combo_out != ray_edge_out)),
        "combo_vs_mover_edge": int(np.count_nonzero(combo_out != mover_edge_out)),
        "combo_vs_full_edge": int(np.count_nonzero(combo_out != full_edge_out)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the active half-enclosure, truncate its radiating wedge on another sprite, and move isolated pixels until that same sprite blocks them.

    Input structure:
    - The most common color is background. The most common non-background color forms two or more separated half-enclosure wall sprites of the same shape family.
    - One small non-wall component sits inside one enclosure and acts as the fill seed. Additional non-wall singleton pixels are isolated movers placed in the outward fill corridor.

    Rule:
    - Pick the wall component nearest the seed as the active enclosure, flood-fill its interior, and let the fill radiate outward through the opening.
    - Trace the two wedge-boundary rays in the opening direction, but stop each ray immediately before it would touch another wall sprite instead of extending it to the grid edge.
    - After the fill is laid down, move every isolated singleton straight along the same radiate direction. Each mover leaves background behind and stops on the last free cell before that same blocking wall sprite.

    Key insight:
    - The distance from the active enclosure to the next wall sprite is the shared control variable. It simultaneously caps the wedge length and caps how far every isolated pixel can travel.
    """
    return _solve(grid, stop_at_other_wall=True, mover_collision_stop=True)


# --------------- generator (combo collision-stop variant) ---------------

# Canvas size bounds
GRID_H_VERT = (14, 18)
GRID_H_HORIZ = (10, 18)
GRID_W = (17, 20)

# Wall geometry
WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 7)
DEPTH_HORIZ = (5, 7)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5

# Placement margins
BLOCKER_OFFSET_MIN = 2
BLOCKER_OFFSET_MAX = 5
BLOCKER_ARM_DEPTH = 3
EDGE_MARGIN = BLOCKER_OFFSET_MAX + 2

# Fill seed
SEED_SIZE_WEIGHTS = [2, 2, 3, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 3

# Movers
N_MOVERS = (1, 3)
MOVER_MIN_STEPS = 2
MOVER_MIN_GAIN = 2
MOVER_WALL_DIST = 2
MOVER_SCANLINE_SEP = 2

# Validation
MIN_NEW_FILL = 10
MIN_RAY_GAIN = 4
MIN_MOVER_GAIN_TOTAL = 2
PALETTE_SIZE = (4, 5)
GEN_RETRIES = 500


def _degree1_pixels(points):
    pts = set(points)
    return [(r, c) for r, c in pts if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1]


def _make_arm(depth, start_v, target_v, extra_steps, tip_12=False):
    sign = 1 if target_v > start_v else -1
    n_side_steps = abs(target_v - start_v)
    chosen_rows = []
    if n_side_steps:
        if tip_12:
            chosen_rows = [2 * (i + 1) for i in range(n_side_steps)]
            extra_steps = 0
        else:
            chosen_rows = [1]
            if n_side_steps > 1:
                chosen_rows += random.sample(range(2, depth + 1), n_side_steps - 1)
            chosen_rows.sort()

    first_row = chosen_rows[0] if chosen_rows else 1
    extra_rows = [r for r in chosen_rows if r != first_row]
    random.shuffle(extra_rows)
    orthogonal_rows = set(extra_rows[:extra_steps])

    pts = [(0, start_v)]
    u, v = 0, start_v
    for row in range(1, depth + 1):
        u += 1
        if row in orthogonal_rows:
            pts.append((u, v))
            v += sign
        elif row in chosen_rows:
            v += sign
        pts.append((u, v))
    return pts


def _build_local_wall(rad, h, w):
    if rad[0]:
        max_depth = min(DEPTH_VERT[1], h - 7)
        if max_depth < DEPTH_VERT[0]:
            return None
        depth = random.randint(DEPTH_VERT[0], max_depth)
        gap = random.randint(*GAP_VERT)
    else:
        max_depth = min(DEPTH_HORIZ[1], w - 7)
        if max_depth < DEPTH_HORIZ[0]:
            return None
        depth = random.randint(DEPTH_HORIZ[0], max_depth)
        gap = random.randint(GAP_HORIZ[0], min(GAP_HORIZ[1], h - 1))

    if gap == 5:
        left_bottom, right_bottom = 2, 3
    elif gap == 6:
        left_bottom, right_bottom = 2, 4
    else:
        left_bottom, right_bottom = random.choice([(2, 5), (3, 4)])

    bridge_extra = max(0, right_bottom - left_bottom - 1)
    base_count = 2 * (depth + 1) + bridge_extra
    max_extra_l = max(0, left_bottom - 1)
    max_extra_r = max(0, gap - right_bottom - 1)
    min_extra = max(0, WALL_PIXELS_MIN - base_count)
    max_extra = min(max_extra_l + max_extra_r, WALL_PIXELS_MAX - base_count)
    if min_extra > max_extra:
        return None

    extra = random.randint(min_extra, max_extra)
    extra_l = random.randint(max(0, extra - max_extra_r), min(max_extra_l, extra))
    extra_r = extra - extra_l

    max_n_side = max(left_bottom, gap - right_bottom)
    tip_12 = depth >= 2 * max_n_side and random.random() < TIP_12_PROB
    left = _make_arm(depth, 0, left_bottom, extra_l, tip_12=tip_12)
    right = _make_arm(depth, gap, right_bottom, extra_r, tip_12=tip_12)
    pts = set(left) | set(right)
    for v in range(left_bottom, right_bottom + 1):
        pts.add((depth, v))

    while len(pts) < WALL_PIXELS_MIN:
        candidates = [
            (u, v)
            for u in range(max(0, depth - 3), depth + 1)
            for v in range(gap + 1)
            if (u, v) not in pts
            and sum(
                (uu, vv) in pts
                for uu in range(u - 1, u + 2)
                for vv in range(v - 1, v + 2)
                if (uu, vv) != (u, v)
            )
            >= 2
        ]
        random.shuffle(candidates)
        added = False
        for p in candidates:
            trial = pts | {p}
            if len(trial) <= WALL_PIXELS_MAX and len(_degree1_pixels(trial)) == 2:
                pts = trial
                added = True
                break
        if not added:
            return None

    if not (WALL_PIXELS_MIN <= len(pts) <= WALL_PIXELS_MAX) or len(_degree1_pixels(pts)) != 2:
        return None
    return pts, depth, gap


def _map_local(u, v, rad, anchor):
    r0, c0 = anchor
    if rad == (-1, 0):
        return r0 + u, c0 + v
    if rad == (1, 0):
        return r0 - u, c0 + v
    if rad == (0, -1):
        return r0 + v, c0 + u
    return r0 + v, c0 - u


def _place_anchor(rad, h, w, depth, gap):
    if rad == (-1, 0):
        lo, hi = EDGE_MARGIN, h - depth - 3
        if lo > hi:
            return None
        return random.randint(lo, hi), random.randint(0, w - gap - 1)
    if rad == (1, 0):
        lo, hi = depth + 2, h - EDGE_MARGIN - 1
        if lo > hi:
            return None
        return random.randint(lo, hi), random.randint(0, w - gap - 1)
    if rad == (0, -1):
        lo, hi = EDGE_MARGIN, w - depth - 3
        if lo > hi:
            return None
        return random.randint(0, h - gap - 1), random.randint(lo, hi)
    lo, hi = depth + 2, w - EDGE_MARGIN - 1
    if lo > hi:
        return None
    return random.randint(0, h - gap - 1), random.randint(lo, hi)


def _local_interior(wall, depth, gap):
    by_row = {}
    for u, v in wall:
        by_row.setdefault(u, []).append(v)
    interior = []
    for u in range(1, depth):
        if u not in by_row or len(by_row[u]) < 2:
            continue
        lo, hi = min(by_row[u]), max(by_row[u])
        for v in range(lo + 1, hi):
            if (u, v) not in wall:
                interior.append((u, v))
    return interior


def _grow_seed_blob(seed, interior_set):
    blob = {seed}
    target_size = random.choice(SEED_SIZE_WEIGHTS)
    frontier = [seed]
    while len(blob) < target_size and frontier:
        p = frontier.pop(random.randrange(len(frontier)))
        u, v = p
        for q in random.sample([(u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)], 4):
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target_size:
                    break
    return blob


def _render_wall(local_wall, rad, anchor):
    pts = set()
    for u, v in local_wall:
        pts.add(_map_local(u, v, rad, anchor))
    return pts


def _grid_dist_to_mask(mask: np.ndarray) -> np.ndarray:
    return distance_transform_cdt(~mask, metric="taxicab")


def _blocker_from_ray_pair(
    p1: tuple[int, int],
    p2: tuple[int, int],
    rad: tuple[int, int],
    shape: tuple[int, int],
) -> set[tuple[int, int]] | None:
    h, w = shape
    if rad[0] != 0:
        if p1[0] != p2[0]:
            return None
        r = p1[0]
        c1, c2 = sorted([p1[1], p2[1]])
        far_r = r + rad[0] * BLOCKER_ARM_DEPTH
        if not (0 <= far_r < h):
            return None
        pts = set()
        for step in range(BLOCKER_ARM_DEPTH + 1):
            rr = r + rad[0] * step
            pts.add((rr, c1))
            pts.add((rr, c2))
        for c in range(c1, c2 + 1):
            pts.add((far_r, c))
        return pts
    if p1[1] != p2[1]:
        return None
    c = p1[1]
    r1, r2 = sorted([p1[0], p2[0]])
    far_c = c + rad[1] * BLOCKER_ARM_DEPTH
    if not (0 <= far_c < w):
        return None
    pts = set()
    for step in range(BLOCKER_ARM_DEPTH + 1):
        cc = c + rad[1] * step
        pts.add((r1, cc))
        pts.add((r2, cc))
    for r in range(r1, r2 + 1):
        pts.add((r, far_c))
    return pts


def _blocker_from_single_ray(
    p: tuple[int, int],
    other_p: tuple[int, int],
    rad: tuple[int, int],
    shape: tuple[int, int],
) -> set[tuple[int, int]] | None:
    h, w = shape
    if rad[0] != 0:
        side = 1 if other_p[1] > p[1] else -1
        cols = [p[1], p[1] + side, p[1] + 2 * side]
        if min(cols) < 0 or max(cols) >= w:
            return None
        far_r = p[0] + rad[0] * 2
        mid_r = p[0] + rad[0]
        if not (0 <= mid_r < h and 0 <= far_r < h):
            return None
        c_lo, c_hi = min(cols[0], cols[2]), max(cols[0], cols[2])
        return {
            (p[0], cols[0]),
            (p[0], cols[2]),
            (mid_r, cols[0]),
            (mid_r, cols[2]),
            *[(far_r, c) for c in range(c_lo, c_hi + 1)],
        }
    side = 1 if other_p[0] > p[0] else -1
    rows = [p[0], p[0] + side, p[0] + 2 * side]
    if min(rows) < 0 or max(rows) >= h:
        return None
    far_c = p[1] + rad[1] * 2
    mid_c = p[1] + rad[1]
    if not (0 <= mid_c < w and 0 <= far_c < w):
        return None
    r_lo, r_hi = min(rows[0], rows[2]), max(rows[0], rows[2])
    return {
        (rows[0], p[1]),
        (rows[2], p[1]),
        (rows[0], mid_c),
        (rows[2], mid_c),
        *[(r, far_c) for r in range(r_lo, r_hi + 1)],
    }


def _try_add_blocker(grid, wall_color, active_pts):
    h, w = grid.shape
    ray_info = _component_ray_info(active_pts, (h, w))
    ray_paths = []
    for ep, d, is_12 in zip(ray_info["endpoints"], ray_info["out_dirs"], ray_info["tip_12"]):
        path, _ = _trace_ray(ep, d, ray_info["rad"], is_12, (h, w), None)
        ray_paths.append(path)
    if len(ray_paths) < 2:
        return None, None

    min_len = min(len(ray_paths[0]), len(ray_paths[1]))
    if min_len <= BLOCKER_OFFSET_MIN + 1:
        return None, None

    bg = get_bg(grid)

    def try_shape(blocker_pts: set[tuple[int, int]] | None):
        if blocker_pts is None:
            return None, None
        if any((r, c) in active_pts for r, c in blocker_pts):
            return None, None
        if any(not (0 <= r < h and 0 <= c < w) for r, c in blocker_pts):
            return None, None
        if any(grid[r, c] != bg for r, c in blocker_pts):
            return None, None

        trial = grid.copy()
        stop_mask = np.zeros((h, w), dtype=bool)
        for r, c in blocker_pts:
            trial[r, c] = wall_color
            stop_mask[r, c] = True
        if len(_component_sets(trial == wall_color)) != 2:
            return None, None

        hits = []
        for ep, d, is_12 in zip(ray_info["endpoints"], ray_info["out_dirs"], ray_info["tip_12"]):
            _, hit = _trace_ray(ep, d, ray_info["rad"], is_12, (h, w), stop_mask)
            hits.append(hit is not None)
        if any(hits):
            return trial, blocker_pts
        return None, None

    idxs = list(range(BLOCKER_OFFSET_MIN - 1, min(min_len - 1, BLOCKER_OFFSET_MAX + 2)))
    random.shuffle(idxs)
    ray_order = [0, 1]
    random.shuffle(ray_order)
    for ridx in ray_order:
        other_idx = 1 - ridx
        for idx in idxs:
            trial, blocker_pts = try_shape(
                _blocker_from_single_ray(
                    ray_paths[ridx][idx],
                    ray_paths[other_idx][min(idx, len(ray_paths[other_idx]) - 1)],
                    ray_info["rad"],
                    (h, w),
                )
            )
            if trial is not None:
                return trial, blocker_pts

    for idx in idxs:
        trial, blocker_pts = try_shape(_blocker_from_ray_pair(ray_paths[0][idx], ray_paths[1][idx], ray_info["rad"], (h, w)))
        if trial is not None:
            return trial, blocker_pts
    return None, None


def _choose_mover_starts(base_grid: np.ndarray, behavior: dict) -> list[tuple[int, int]]:
    scene = behavior["combo"]["scene"]
    filled_mask = behavior["combo"]["filled_mask"]
    wall_dist = _grid_dist_to_mask(scene["wall_mask"])
    starts = []
    candidates = []
    for r, c in zip(*np.where(filled_mask)):
        r = int(r)
        c = int(c)
        if wall_dist[r, c] <= MOVER_WALL_DIST:
            continue
        path, hit = _trace_line((r, c), scene["rad"], scene["shape"], scene["other_wall_mask"])
        if hit is None or len(path) < MOVER_MIN_STEPS:
            continue
        edge_path, _ = _trace_line((r, c), scene["rad"], scene["shape"], None)
        gain = len(edge_path) - len(path)
        if gain < MOVER_MIN_GAIN:
            continue
        candidates.append(((r, c), len(path), gain))

    random.shuffle(candidates)
    candidates.sort(key=lambda rec: (rec[2], rec[1]), reverse=True)

    use_cols = scene["rad"][0] != 0
    used_scanlines: list[int] = []
    for (r, c), _, _ in candidates:
        scanline = c if use_cols else r
        if any(abs(scanline - prev) < MOVER_SCANLINE_SEP for prev in used_scanlines):
            continue
        starts.append((r, c))
        used_scanlines.append(scanline)
        if len(starts) >= random.randint(*N_MOVERS):
            break
    return starts


def _gen_attempt():
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built
    anchor = _place_anchor(rad, h, w, depth, gap)
    if anchor is None:
        return None
    active_pts = _render_wall(local_wall, rad, anchor)

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    mover_colors = palette[3:] or [c for c in range(10) if c not in (bg, wall_color, fill_color)][:2]

    grid = np.full((h, w), bg, dtype=int)
    for r, c in active_pts:
        if not (0 <= r < h and 0 <= c < w):
            return None
        grid[r, c] = wall_color

    interior = _local_interior(local_wall, depth, gap)
    wall_set = set(local_wall)
    adjacent = [
        p
        for p in interior
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    seed_pool = [p for p in adjacent if p[0] >= max(1, depth - SEED_DEPTH_LOOKBACK)]
    if not seed_pool:
        return None

    blob = _grow_seed_blob(random.choice(seed_pool), set(interior))
    for u, v in blob:
        r, c = _map_local(u, v, rad, anchor)
        if grid[r, c] != bg:
            return None
        grid[r, c] = fill_color

    grid, blocker_pts = _try_add_blocker(grid, wall_color, active_pts)
    if grid is None:
        return None

    try:
        behavior = summarize_behavior(grid)
    except Exception:
        return None

    starts = _choose_mover_starts(grid, behavior)
    if not starts:
        return None

    for i, (r, c) in enumerate(starts):
        if grid[r, c] != bg:
            return None
        grid[r, c] = mover_colors[i % len(mover_colors)]

    try:
        behavior = summarize_behavior(grid)
    except Exception:
        return None

    combo_out = behavior["combo_out"]
    new_fill = (combo_out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL:
        return None
    if behavior["ray_hits"] == 0 or behavior["mover_hits"] == 0:
        return None
    if behavior["ray_saved"] < MIN_RAY_GAIN or behavior["mover_saved"] < MIN_MOVER_GAIN_TOTAL:
        return None
    if behavior["combo_vs_ray_edge"] == 0 or behavior["combo_vs_mover_edge"] == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo input")
