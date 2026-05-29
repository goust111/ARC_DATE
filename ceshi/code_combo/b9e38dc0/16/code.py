# Combo variant of b9e38dc0 (index 16)
# combo_spec: D7/3+D16/0
# Name: 障碍吸收变色+多孤立像素

import random
from collections import defaultdict

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis for the fused variant
# a) Concrete behavior:
#    The original task only had one large wall/fill process plus obstacle shadowing. Here the
#    wall and radiating fill stay as stage 1, but stage 2 changes completely: every non-wall,
#    non-fill singleton is a mover, every non-wall, non-fill component of size >= 2 is an
#    obstacle sprite. Example in a downward-radiating scene:
#      - original: a 2-cell obstacle sitting in the filled wedge merely erases fill behind it.
#      - fused: two singleton pixels on the two lanes above that obstacle march downward,
#        carve bg tunnels through the fill, and recolor the same 2-cell obstacle in arrival
#        order; the later one decides the final obstacle color.
#    The visible difference is therefore structural, not cosmetic: output contains bg tunnels
#    and recolored obstacle blocks rather than one-sided fill shadows.
# b) Complexity calibration:
#    Each obstacle sprite is a short 2-cell bar and gets exactly two associated movers. This
#    is enough repetition for humans to notice "one sprite <- multiple isolated pixels" without
#    creating a noisy many-to-many scene. The generator usually places 1-2 such sprites, and
#    may add one extra free mover that reaches the border.
# c) Broken assumptions from the source solver:
#    - "all non-fill extra pixels are blockers" is no longer valid: size-1 and size>=2 now have
#      different semantics.
#    - "obstacles only matter after fill as shadows" is broken: they now participate in color
#      transfer and are never used for scanline shadow deletion.
#    - "the closest non-wall color is the only dynamic color" remains valid only because the
#      generator keeps mover/obstacle sprites farther from the wall than the seed blob.
#    - Flood fill must treat mover cells as passable support cells, otherwise a singleton placed
#      inside the corridor would incorrectly puncture the filled region before it moves.
# d) Placement legality:
#    The generator first builds a valid wall + seed scene, computes the nominal filled domain,
#    then places obstacle bars only on background cells inside that domain. Each mover is placed
#    upstream on a lane that points straight into its target obstacle, with no overlap or
#    out-of-bounds motion. Accepted samples are re-solved and rejected unless the fused metrics
#    confirm the intended behavior.
# e) Visual observability:
#    Humans can see the same corridor logic as the base task, plus repeated local motifs:
#    parallel singleton pixels upstream of a short obstacle bar, bg tunnels in the output,
#    and the obstacle taking one mover's color. With two movers per obstacle, "last arrival
#    wins" can be inferred from distance along the motion axis rather than hidden metadata.
# f) Visual cleanliness:
#    Only one obstacle family is used (small bars), movers are isolated singletons, and their
#    lanes are reserved so different obstacle groups do not overlap. This prevents fill, mover
#    trails, and obstacle bodies from collapsing into one cluttered patch.
# g) Color inference:
#    Colors are few and role-based: wall, fill, one obstacle-body color, and a small mover-color
#    pool. Obstacle recoloring is always anchored by direct geometric alignment between input
#    singleton(s) and the affected obstacle component, so the transferred color can be inferred
#    from position plus motion, not from numeric color ids.


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _axis_value(pos: tuple[int, int], rad: tuple[int, int]) -> int:
    r, c = pos
    dr, dc = rad
    return r * dr + c * dc


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _scene_info(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    vals = [int(v) for v in np.unique(g) if int(v) != bg]
    assert vals, "expected at least one non-background color"

    wall = max(vals, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    wall_comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert wall_comps, "expected at least one wall connected component"

    other_colors = [c for c in vals if c != wall]
    assert other_colors, "expected at least one non-wall color"

    wall_points = [p for comp in wall_comps for p in comp]

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = [(int(r), int(c)) for r, c in zip(*np.where(g == color))]
        best_d = 10**9
        for r, c in pts:
            for wr, wc in wall_points:
                d = abs(r - wr) + abs(c - wc)
                if d < best_d:
                    best_d = d
                    if best_d == 0:
                        return 0, len(pts)
        return best_d, len(pts)

    fill = min(other_colors, key=lambda c: (min_dist_to_wall(c)[0], min_dist_to_wall(c)[1]))
    fill_pos = [(int(r), int(c)) for r, c in zip(*np.where(g == fill))]
    assert fill_pos, "expected at least one fill-seed pixel"

    def comp_min_dist(comp: set[tuple[int, int]]) -> int:
        best_d = 10**9
        for r, c in fill_pos:
            for wr, wc in comp:
                d = abs(r - wr) + abs(c - wc)
                if d < best_d:
                    best_d = d
                    if best_d == 0:
                        return 0
        return best_d

    wall_comp = min(wall_comps, key=lambda comp: (comp_min_dist(comp), -len(comp)))

    comp_mask = np.zeros_like(g, dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True

    def comp_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for rr, cc in neighbors8(r, c):
            if _in_bounds(h, w, rr, cc) and comp_mask[rr, cc]:
                out.append((rr, cc))
        return out

    endpoints = [(r, c) for r, c in wall_comp if len(comp_neighbors(r, c)) == 1]
    assert endpoints, "expected wall component to have endpoints"

    def outward_dir(ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
        n1 = comp_neighbors(*ep)[0]
        d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
        d_use = d1
        is_12 = False
        n2s = [p for p in comp_neighbors(*n1) if p != ep]
        if n2s:
            n2 = n2s[0]
            d2 = (_sgn(n2[0] - n1[0]), _sgn(n2[1] - n1[1]))
            if (d1[0] == 0) ^ (d1[1] == 0) and d2[0] != 0 and d2[1] != 0:
                d_use = d2
                is_12 = True
        return (-d_use[0], -d_use[1]), is_12

    out_info = [outward_dir(ep) for ep in endpoints]
    out_dirs = [d for d, _ in out_info]
    tip_12 = [flag for _, flag in out_info]
    dr_sum = sum(d[0] for d in out_dirs)
    dc_sum = sum(d[1] for d in out_dirs)
    rad = (_sgn(dr_sum), _sgn(dc_sum))
    assert rad != (0, 0), "failed to infer radiate direction"

    barrier = comp_mask.copy()

    def add_ray(start: tuple[int, int], d: tuple[int, int]) -> None:
        rr, cc = start
        while True:
            rr += d[0]
            cc += d[1]
            if not _in_bounds(h, w, rr, cc):
                break
            barrier[rr, cc] = True

    def add_ray_12(start: tuple[int, int], d: tuple[int, int]) -> None:
        rr, cc = start
        step = 0
        while True:
            step += 1
            if step % 2 == 1:
                rr += d[0]
                cc += d[1]
            else:
                if rad[0] != 0:
                    rr += d[0]
                else:
                    cc += d[1]
            if not _in_bounds(h, w, rr, cc):
                break
            barrier[rr, cc] = True

    for ep, d, is_12 in zip(endpoints, out_dirs, tip_12):
        if is_12:
            add_ray_12(ep, d)
        else:
            add_ray(ep, d)

    movers = []
    obstacle_components = []
    obstacle_by_cell: dict[tuple[int, int], int] = {}
    obstacle_cells: dict[int, list[tuple[int, int]]] = {}
    obstacle_input_color: dict[int, int] = {}

    next_obstacle_id = 0
    for color in other_colors:
        if color == fill:
            continue
        for comp in components_mask(g == color, connectivity=8):
            pts = [(int(r), int(c)) for r, c in comp]
            if len(pts) == 1:
                movers.append({"start": pts[0], "color": int(color)})
            else:
                obstacle_components.append({"id": next_obstacle_id, "color": int(color), "cells": pts})
                obstacle_cells[next_obstacle_id] = pts
                obstacle_input_color[next_obstacle_id] = int(color)
                for p in pts:
                    obstacle_by_cell[p] = next_obstacle_id
                next_obstacle_id += 1

    return {
        "grid": g,
        "h": h,
        "w": w,
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "fill_pos": fill_pos,
        "rad": rad,
        "barrier": barrier,
        "wall_mask": comp_mask,
        "movers": movers,
        "obstacle_components": obstacle_components,
        "obstacle_by_cell": obstacle_by_cell,
        "obstacle_cells": obstacle_cells,
        "obstacle_input_color": obstacle_input_color,
    }


def _flood_domain(scene: dict) -> tuple[np.ndarray, np.ndarray]:
    g = scene["grid"]
    h, w = scene["h"], scene["w"]
    bg, fill = scene["bg"], scene["fill"]
    barrier = scene["barrier"]
    fill_pos = list(scene["fill_pos"])

    mover_mask = np.zeros_like(g, dtype=bool)
    for mover in scene["movers"]:
        r, c = mover["start"]
        mover_mask[r, c] = True

    passable = (~barrier) & ((g == bg) | (g == fill) | mover_mask)
    seen = np.zeros_like(passable, dtype=bool)
    q = fill_pos[:]
    for r, c in q:
        if passable[r, c]:
            seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if _in_bounds(h, w, rr, cc) and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))

    filled_bg = seen & (g == bg)
    return seen, filled_bg


def _trace_mover(
    start: tuple[int, int],
    color: int,
    rad: tuple[int, int],
    travel_domain: np.ndarray,
    obstacle_by_cell: dict[tuple[int, int], int],
) -> dict:
    h, w = travel_domain.shape
    dr, dc = rad
    rr, cc = start
    path = [start]
    time = 0

    while True:
        nr, nc = rr + dr, cc + dc
        if not _in_bounds(h, w, nr, nc):
            return {
                "start": start,
                "color": color,
                "time": time,
                "clear": path[:-1],
                "final": path[-1],
                "obstacle_id": None,
            }
        time += 1
        if (nr, nc) in obstacle_by_cell:
            return {
                "start": start,
                "color": color,
                "time": time,
                "clear": path,
                "final": None,
                "obstacle_id": obstacle_by_cell[(nr, nc)],
            }
        if not travel_domain[nr, nc]:
            return {
                "start": start,
                "color": color,
                "time": time - 1,
                "clear": path[:-1],
                "final": path[-1],
                "obstacle_id": None,
            }
        path.append((nr, nc))
        rr, cc = nr, nc


def _solve(grid: np.ndarray, return_info: bool = False):
    scene = _scene_info(grid)
    g = scene["grid"]
    bg = scene["bg"]
    fill = scene["fill"]

    seen, filled_bg = _flood_domain(scene)
    out = np.array(g, copy=True)
    out[filled_bg] = fill

    travel_domain = seen.copy()
    for cells in scene["obstacle_cells"].values():
        for r, c in cells:
            travel_domain[r, c] = False
    for r, c in zip(*np.where(scene["wall_mask"])):
        travel_domain[int(r), int(c)] = False

    mover_results = []
    for mover in sorted(scene["movers"], key=lambda m: (_axis_value(m["start"], scene["rad"]), m["start"][0], m["start"][1], m["color"])):
        mover_results.append(
            _trace_mover(
                mover["start"],
                mover["color"],
                scene["rad"],
                travel_domain,
                scene["obstacle_by_cell"],
            )
        )

    for res in mover_results:
        for r, c in res["clear"]:
            out[r, c] = bg

    for res in mover_results:
        if res["final"] is not None:
            r, c = res["final"]
            out[r, c] = res["color"]

    obstacle_final_color = dict(scene["obstacle_input_color"])
    absorption_events = [res for res in mover_results if res["obstacle_id"] is not None]
    absorption_events.sort(key=lambda e: (e["time"], _axis_value(e["start"], scene["rad"]), e["start"][0], e["start"][1], e["color"]))
    for event in absorption_events:
        obstacle_final_color[event["obstacle_id"]] = event["color"]
        for r, c in scene["obstacle_cells"][event["obstacle_id"]]:
            out[r, c] = event["color"]

    if not return_info:
        return out

    hits_by_obstacle: dict[int, list[dict]] = defaultdict(list)
    for event in absorption_events:
        hits_by_obstacle[event["obstacle_id"]].append(event)

    recolored_obstacles = 0
    for oid, base_color in scene["obstacle_input_color"].items():
        if obstacle_final_color[oid] != base_color:
            recolored_obstacles += 1

    info = {
        "scene": scene,
        "seen": seen,
        "filled_bg": filled_bg,
        "mover_results": mover_results,
        "absorption_events": absorption_events,
        "obstacle_final_color": obstacle_final_color,
        "movers": len(mover_results),
        "absorbed_movers": len(absorption_events),
        "edge_movers": sum(res["final"] is not None for res in mover_results),
        "multi_hit_obstacles": sum(len(events) >= 2 for events in hits_by_obstacle.values()),
        "recolored_obstacles": recolored_obstacles,
        "carved_cells": int(sum(len(res["clear"]) for res in mover_results)),
        "last_wins_obstacles": sum(
            obstacle_final_color[oid] == events[-1]["color"] for oid, events in hits_by_obstacle.items() if len(events) >= 2
        ),
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood the wall enclosure, then let multiple singleton movers carve outward tunnels and recolor any obstacle sprite they hit.

    Input structure:
    - bg is the most common color. The wall is the most frequent non-bg color and forms one open funnel/U-like component whose endpoints define a single cardinal radiate direction.
    - A compact fill seed sits near the closed end. Among the remaining non-wall colors, singleton components are moving pixels and any component of size >= 2 is an obstacle sprite.

    Rule:
    - First recover the original filled region: flood the interior from the seed and continue outward through the opening, bounded by the endpoint rays.
    - Then move every singleton straight along the radiate direction. Cells it vacates become background, so it cuts a bg tunnel through the filled region.
    - If a mover reaches an obstacle component, it is absorbed and that whole obstacle changes to the mover's color. If several movers hit the same obstacle, the one arriving later overwrites earlier recolors. Movers with no obstacle ahead stop at the border.

    Key insight:
    - The wall still determines both the filled corridor and the motion axis. After that, component size separates dynamic singletons from static obstacle sprites, and arrival order along the shared axis explains which mover color survives on each obstacle.
    """
    return _solve(np.asarray(grid, dtype=int), return_info=False)


# --------------- generator ---------------

# Canvas size bounds
GRID_H_VERT = (11, 18)
GRID_H_HORIZ = (8, 18)
GRID_W = (15, 19)

# Wall geometry
WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 10)
DEPTH_HORIZ = (6, 8)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5
OPEN_END_MARGIN = 3

# Fill seed
SEED_SIZE_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 3

# Fusion placement
OBSTACLE_GROUPS = (1, 2)
UPSTREAM_MIN_DIST = 2
UPSTREAM_TARGET_FAR = 4
FREE_MOVER_PROB = 0.55

# Validation
GEN_RETRIES = 400


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
        max_depth = min(DEPTH_VERT[1], h - 6)
        if max_depth < DEPTH_VERT[0]:
            return None
        depth = random.randint(DEPTH_VERT[0], max_depth)
        gap = random.randint(*GAP_VERT)
    else:
        depth = random.randint(DEPTH_HORIZ[0], min(DEPTH_HORIZ[1], w - 6))
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
        return random.randint(OPEN_END_MARGIN, h - depth - 3), random.randint(0, w - gap - 1)
    if rad == (1, 0):
        return random.randint(depth + 2, h - 4), random.randint(0, w - gap - 1)
    if rad == (0, -1):
        return random.randint(0, h - gap - 1), random.randint(OPEN_END_MARGIN, w - depth - 3)
    return random.randint(0, h - gap - 1), random.randint(depth + 2, w - 4)


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


def _upstream_options(
    grid: np.ndarray,
    domain: np.ndarray,
    rad: tuple[int, int],
    cell: tuple[int, int],
    bg: int,
    reserved: set[tuple[int, int]],
) -> list[tuple[int, tuple[int, int]]]:
    dr, dc = rad
    h, w = grid.shape
    rr, cc = cell
    out = []
    dist = 0
    while True:
        rr -= dr
        cc -= dc
        dist += 1
        if not _in_bounds(h, w, rr, cc):
            break
        if not domain[rr, cc]:
            break
        if grid[rr, cc] != bg or (rr, cc) in reserved:
            break
        out.append((dist, (rr, cc)))
    return [item for item in out if item[0] >= UPSTREAM_MIN_DIST]


def _path_cells(start: tuple[int, int], stop_exclusive: tuple[int, int], rad: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = rad
    rr, cc = start
    out = []
    while (rr, cc) != stop_exclusive:
        out.append((rr, cc))
        rr += dr
        cc += dc
    return out


def _choose_group_candidate(
    grid: np.ndarray,
    domain: np.ndarray,
    rad: tuple[int, int],
    bg: int,
    reserved: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
) -> dict | None:
    h, w = grid.shape
    domain_bg = [(int(r), int(c)) for r, c in zip(*np.where(domain & (grid == bg)))]
    if not domain_bg:
        return None
    downstream_cut = int(np.median([_axis_value(p, rad) for p in domain_bg]))
    candidates = []

    if rad[0] != 0:
        for r in range(h):
            for c in range(w - 1):
                cells = [(r, c), (r, c + 1)]
                if min(_axis_value(p, rad) for p in cells) <= downstream_cut:
                    continue
                if any((not domain[rr, cc]) or grid[rr, cc] != bg or (rr, cc) in reserved or (rr, cc) in reserved_paths for rr, cc in cells):
                    continue
                opt_a = _upstream_options(grid, domain, rad, cells[0], bg, reserved | reserved_paths)
                opt_b = _upstream_options(grid, domain, rad, cells[1], bg, reserved | reserved_paths)
                best = None
                for d1, p1 in opt_a:
                    for d2, p2 in opt_b:
                        if d1 == d2:
                            continue
                        score = (max(d1, d2) >= UPSTREAM_TARGET_FAR, abs(d1 - d2), min(d1, d2))
                        if best is None or score > best[0]:
                            best = (score, [(cells[0], d1, p1), (cells[1], d2, p2)])
                if best is None:
                    continue
                candidates.append({"cells": cells, "assignments": best[1], "score": best[0]})
    else:
        for r in range(h - 1):
            for c in range(w):
                cells = [(r, c), (r + 1, c)]
                if min(_axis_value(p, rad) for p in cells) <= downstream_cut:
                    continue
                if any((not domain[rr, cc]) or grid[rr, cc] != bg or (rr, cc) in reserved or (rr, cc) in reserved_paths for rr, cc in cells):
                    continue
                opt_a = _upstream_options(grid, domain, rad, cells[0], bg, reserved | reserved_paths)
                opt_b = _upstream_options(grid, domain, rad, cells[1], bg, reserved | reserved_paths)
                best = None
                for d1, p1 in opt_a:
                    for d2, p2 in opt_b:
                        if d1 == d2:
                            continue
                        score = (max(d1, d2) >= UPSTREAM_TARGET_FAR, abs(d1 - d2), min(d1, d2))
                        if best is None or score > best[0]:
                            best = (score, [(cells[0], d1, p1), (cells[1], d2, p2)])
                if best is None:
                    continue
                candidates.append({"cells": cells, "assignments": best[1], "score": best[0]})

    if not candidates:
        return None
    candidates.sort(key=lambda item: item["score"], reverse=True)
    top = candidates[: min(8, len(candidates))]
    return random.choice(top)


def _place_free_mover(
    grid: np.ndarray,
    domain: np.ndarray,
    rad: tuple[int, int],
    bg: int,
    reserved: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
    color: int,
) -> bool:
    h, w = grid.shape
    dr, dc = rad
    candidates = []
    for r, c in zip(*np.where(domain & (grid == bg))):
        r = int(r)
        c = int(c)
        if (r, c) in reserved or (r, c) in reserved_paths:
            continue
        rr, cc = r, c
        path = []
        ok = True
        while True:
            nr, nc = rr + dr, cc + dc
            if not _in_bounds(h, w, nr, nc):
                break
            if not domain[nr, nc] or grid[nr, nc] != bg or (nr, nc) in reserved or (nr, nc) in reserved_paths:
                ok = False
                break
            path.append((nr, nc))
            rr, cc = nr, nc
        if ok and len(path) >= 2:
            candidates.append(((r, c), path))
    if not candidates:
        return False
    start, path = random.choice(candidates)
    grid[start] = color
    reserved.add(start)
    reserved_paths.update(path)
    return True


def _metrics_for(grid: np.ndarray, out: np.ndarray, info: dict) -> dict:
    changed = int(np.count_nonzero(grid != out))
    final_obstacle_colors = set(info["obstacle_final_color"].values())
    return {
        "changed": changed,
        "movers": int(info["movers"]),
        "absorbed_movers": int(info["absorbed_movers"]),
        "edge_movers": int(info["edge_movers"]),
        "multi_hit_obstacles": int(info["multi_hit_obstacles"]),
        "recolored_obstacles": int(info["recolored_obstacles"]),
        "carved_cells": int(info["carved_cells"]),
        "last_wins_obstacles": int(info["last_wins_obstacles"]),
        "final_obstacle_palette": len(final_obstacle_colors),
    }


def _gen_attempt():
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built
    anchor = _place_anchor(rad, h, w, depth, gap)

    palette = random.sample(range(10), 6)
    bg, wall_color, fill_color, obstacle_color = palette[:4]
    mover_pool = palette[4:]

    grid = np.full((h, w), bg, dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
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

    try:
        _, base_info = _solve(grid, return_info=True)
    except Exception:
        return None

    domain = base_info["seen"]
    reserved = {(int(r), int(c)) for r, c in zip(*np.where(grid != bg))}
    reserved_paths: set[tuple[int, int]] = set()
    group_count = random.randint(*OBSTACLE_GROUPS)
    placed_groups = 0

    for _ in range(group_count):
        candidate = _choose_group_candidate(grid, domain, rad, bg, reserved, reserved_paths)
        if candidate is None:
            break
        cells = candidate["cells"]
        assignments = candidate["assignments"]
        for cell in cells:
            grid[cell] = obstacle_color
            reserved.add(cell)
        start_order = sorted(assignments, key=lambda item: item[1])
        mover_colors = random.sample(mover_pool, 2)
        random.shuffle(mover_colors)
        for idx, (_cell, _dist, pos) in enumerate(start_order):
            grid[pos] = mover_colors[idx]
            reserved.add(pos)
        for cell, _dist, pos in start_order:
            reserved_paths.update(_path_cells(pos, cell, rad))
        placed_groups += 1

    if placed_groups == 0:
        return None

    if random.random() < FREE_MOVER_PROB:
        _place_free_mover(
            grid,
            domain,
            rad,
            bg,
            reserved,
            reserved_paths,
            random.choice(mover_pool),
        )

    try:
        out, info = _solve(grid, return_info=True)
    except Exception:
        return None

    metrics = _metrics_for(grid, out, info)
    if metrics["movers"] < 2:
        return None
    if metrics["absorbed_movers"] < 2:
        return None
    if metrics["multi_hit_obstacles"] < 1:
        return None
    if metrics["recolored_obstacles"] < 1:
        return None
    if metrics["carved_cells"] < 4:
        return None
    if metrics["changed"] < 8:
        return None
    if metrics["last_wins_obstacles"] < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fused b9e38dc0 input")
