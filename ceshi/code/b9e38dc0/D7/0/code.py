# Variant of b9e38dc0
# Dimension: D7 — 障碍物阻挡机制 (obstacle)
# Change: obstacle_blocking——第二步变换中，第一步的精灵（含填充后的区域）作为障碍物阻挡孤立像素的直线路径... → 障碍物导致路径偏转

import heapq
import os
import random
import sys

import numpy as np
from scipy.ndimage import distance_transform_cdt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
SOLVER_DIR = os.path.join(ROOT, "solvers")
if SOLVER_DIR not in sys.path:
    sys.path.append(SOLVER_DIR)

from arc_utils import components_mask, get_bg, neighbors4, neighbors8

# Design analysis
# a) Concrete behavior and trace
#    - Stage 1 stays close to the original task: detect the wall sprite, infer the opening
#      direction `rad`, then flood-fill the enclosed seed region and radiate that fill
#      outward through the opening.
#    - Stage 2 changes completely around D7: every extra non-wall/non-fill pixel is now a
#      mover. It tries to travel toward the same border indicated by `rad`.
#      In the old "blocking" interpretation, a mover would stop when its straight ray hit
#      the stage-1 sprite. In this variant it must keep going: it is allowed to step only
#      in the forward direction or sideways, so it skirts the sprite's boundary and then
#      resumes forward progress until it reaches the target border.
#    - Small trace for `rad = down`: suppose a magenta mover starts above the filled funnel.
#      Old behavior: it would descend until the cell just above the sprite and stop there.
#      New behavior: shortest monotone route goes down, then left along the sprite's edge,
#      then down again to the bottom border. Output therefore relocates the mover to a
#      border cell whose column differs from its start column.
# b) Complexity calibration
#    - The wall/fill geometry and color logic are preserved, so the task keeps the original
#      structural cue. Only the second-stage interaction changes.
#    - Detours are restricted to three move types: forward, side A, side B. This is more
#      complex than the original shadow rule, but still visually learnable because paths
#      are monotone and hug a single large sprite instead of arbitrary mazes.
#    - Generator rejects straight-through cases. At least two movers must actually reroute,
#      so the variant is visible rather than a no-op.
# c) Implicit assumptions audited against the old solver
#    - The old `shadow_process` assumed obstacles were stationary and only erased fill
#      behind them. That assumption is removed; movers now need their own path search.
#    - Fill-color detection still assumes the true seed color is the non-wall color closest
#      to the wall. Generator therefore keeps mover pixels a few cells away from the wall.
#    - Wall detection still assumes wall is the most frequent non-bg color and has one
#      relevant component with two endpoints; generator preserves the original wall ranges.
# d) Placement legality
#    - Movers are placed only on cells that remain background after stage 1, so they do not
#      punch holes into the filled sprite.
#    - Candidate movers must have a valid monotone path around the sprite to the `rad`
#      border, and chosen movers must have distinct starts and distinct destinations.
#    - Separation constraints keep movers visually separate and avoid destination clashes.
# e) Visual observability
#    - The visible invariant is: the large wall+fill sprite is static; isolated pixels on
#      the upstream side reappear on the `rad` border after routing around that sprite.
#    - Because at least two movers reroute per example, a human can compare their start
#      columns/rows against their border landing sites and infer the bypass rule.
# f) Visual cleanliness
#    - Only one large sprite and a handful of singleton movers coexist.
#    - Movers are kept off the stage-1 fill area and spaced apart, so the output does not
#      become cluttered by overlaps or holes.
# g) Color inferability
#    - Colors are not used for sorting or control flow. Mover colors are merely preserved.
#      The rule is geometric: all extra singleton colors behave the same way.
# 9) New assumptions checked against generator
#    - `_find_detour_path` assumes the sprite is a connected forward-facing obstacle that a
#      monotone forward/sideways path can skirt. Generator constructs the original funnel
#      shape and rejects any candidate without such a path.
#    - Destination uniqueness is assumed when painting final mover positions. Generator
#      enforces unique destinations and keeps starts away from the target border.


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _comp_neighbors(comp_mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = comp_mask.shape
    out = []
    for rr, cc in neighbors8(r, c):
        if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
            out.append((rr, cc))
    return out


def _analyze_wall(g: np.ndarray) -> dict:
    h, w = g.shape
    bg = get_bg(g)
    vals = np.unique(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert comps, "expected at least one wall connected component"

    other_colors = [c for c in non_bg if c != wall]
    assert other_colors, "expected at least one non-wall color"

    wall_points = [p for comp in comps for p in comp]

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = list(zip(*np.where(g == color)))
        pts = [(int(r), int(c)) for r, c in pts]
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
    fill_pos = list(zip(*np.where(g == fill)))
    fill_pos = [(int(r), int(c)) for r, c in fill_pos]
    if not fill_pos:
        return {"bg": bg, "wall": wall, "fill": fill, "fill_pos": [], "wall_points": wall_points}

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

    wall_comp = min(comps, key=lambda comp: (comp_min_dist(comp), -len(comp)))
    comp_mask = np.zeros_like(wall_mask, dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True

    endpoints = [(r, c) for r, c in wall_comp if len(_comp_neighbors(comp_mask, r, c)) == 1]
    assert endpoints, "expected wall component to have endpoints"

    def outward_dir(ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
        n1 = _comp_neighbors(comp_mask, *ep)[0]
        d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
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

    out_info = [outward_dir(ep) for ep in endpoints]
    out_dirs = [d for d, _ in out_info]
    tip_12 = [flag for _, flag in out_info]
    rad = (_sgn(sum(d[0] for d in out_dirs)), _sgn(sum(d[1] for d in out_dirs)))

    barrier = comp_mask.copy()

    def add_ray(start: tuple[int, int], d: tuple[int, int]) -> None:
        rr, cc = start
        while True:
            rr += d[0]
            cc += d[1]
            if not (0 <= rr < h and 0 <= cc < w):
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
            if not (0 <= rr < h and 0 <= cc < w):
                break
            barrier[rr, cc] = True

    for ep, d, is_12 in zip(endpoints, out_dirs, tip_12):
        if is_12:
            add_ray_12(ep, d)
        else:
            add_ray(ep, d)

    return {
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "fill_pos": fill_pos,
        "wall_points": wall_points,
        "wall_comp": wall_comp,
        "comp_mask": comp_mask,
        "barrier": barrier,
        "rad": rad,
    }


def _first_stage(grid: np.ndarray) -> dict:
    """
    Compute the original wall/fill transform only, before mover rerouting.
    """
    g = np.array(grid, copy=True)
    info = _analyze_wall(g)
    if not info["fill_pos"]:
        info["out"] = g
        info["filled_mask"] = np.zeros_like(g, dtype=bool)
        info["sprite_mask"] = info["comp_mask"].copy()
        return info

    bg = info["bg"]
    fill = info["fill"]
    barrier = info["barrier"]
    h, w = g.shape

    passable = (~barrier) & ((g == bg) | (g == fill))
    q = info["fill_pos"][:]
    seen = np.zeros_like(passable, dtype=bool)
    for r, c in q:
        if passable[r, c]:
            seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))

    filled_mask = seen & (g == bg)
    out = g.copy()
    out[filled_mask] = fill
    sprite_mask = info["comp_mask"] | (out == fill)

    info["out"] = out
    info["filled_mask"] = filled_mask
    info["sprite_mask"] = sprite_mask
    return info


def _goal_reached(r: int, c: int, rad: tuple[int, int], h: int, w: int) -> bool:
    if rad == (-1, 0):
        return r == 0
    if rad == (1, 0):
        return r == h - 1
    if rad == (0, -1):
        return c == 0
    return c == w - 1


def _detour_moves(rad: tuple[int, int]) -> list[tuple[int, int]]:
    if rad[0] != 0:
        return [rad, (0, -1), (0, 1)]
    return [rad, (-1, 0), (1, 0)]


def _find_detour_path(
    start: tuple[int, int],
    sprite_mask: np.ndarray,
    rad: tuple[int, int],
) -> list[tuple[int, int]] | None:
    h, w = sprite_mask.shape
    sr, sc = start
    if sprite_mask[sr, sc]:
        return None
    moves = _detour_moves(rad)
    start_state = (sr, sc, 0)
    best: dict[tuple[int, int, int], tuple[int, int, int]] = {start_state: (0, 0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int, int], int, int, int]] = [((0, 0, 0), sr, sc, 0)]
    goal_state: tuple[int, int, int] | None = None

    while heap:
        cost, r, c, dprev = heapq.heappop(heap)
        if cost != best.get((r, c, dprev)):
            continue
        if _goal_reached(r, c, rad, h, w) and (r, c) != (sr, sc):
            goal_state = (r, c, dprev)
            break
        for nd, (dr, dc) in enumerate(moves):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if sprite_mask[rr, cc]:
                continue
            ncost = (
                cost[0] + (1 if nd != 0 else 0),
                cost[1] + 1,
                cost[2] + (0 if nd == dprev else 1),
            )
            key = (rr, cc, nd)
            if ncost < best.get(key, (10**9, 10**9, 10**9)):
                best[key] = ncost
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost, rr, cc, nd))

    if goal_state is None:
        return None

    path = []
    cur = goal_state
    while True:
        r, c, _ = cur
        path.append((r, c))
        if (r, c) == (sr, sc):
            break
        cur = prev[cur]
    path.reverse()
    return path


def _straight_stop(start: tuple[int, int], sprite_mask: np.ndarray, rad: tuple[int, int]) -> tuple[int, int]:
    h, w = sprite_mask.shape
    r, c = start
    last = (r, c)
    while True:
        rr, cc = r + rad[0], c + rad[1]
        if not (0 <= rr < h and 0 <= cc < w):
            return last
        if sprite_mask[rr, cc]:
            return last
        r, c = rr, cc
        last = (r, c)


def _ray_hits_sprite(start: tuple[int, int], sprite_mask: np.ndarray, rad: tuple[int, int]) -> bool:
    h, w = sprite_mask.shape
    r, c = start
    while True:
        r += rad[0]
        c += rad[1]
        if not (0 <= r < h and 0 <= c < w):
            return False
        if sprite_mask[r, c]:
            return True


def _path_side_steps(path: list[tuple[int, int]], rad: tuple[int, int]) -> int:
    if len(path) < 2:
        return 0
    return sum(1 for (r1, c1), (r2, c2) in zip(path, path[1:]) if (r2 - r1, c2 - c1) != rad)


def _extra_pixels(g: np.ndarray, bg: int, wall: int, fill: int) -> list[tuple[int, int, int]]:
    coords = np.argwhere((g != bg) & (g != wall) & (g != fill))
    return [(int(r), int(c), int(g[r, c])) for r, c in coords]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the funnel interior, then let isolated pixels advance toward the opening side by skirting the finished sprite when necessary.

    Input structure:
    - The most common color is background. The most frequent non-background color forms a single partial wall with one open side.
    - A small seed of another color sits inside the wall near the closed end. Any remaining non-wall, non-seed pixels are isolated movers.

    Rule:
    - First perform the original fill: flood the interior from the seed and radiate that color outward through the wall opening.
    - Then move each isolated pixel toward the border indicated by the opening direction.
    - If a straight route would hit the stage-1 wall+fill sprite, the pixel is not stopped; it detours sideways along the sprite boundary and continues forward until it reaches that border.

    Key insight: Solve the wall to get the radiating sprite first, then treat that whole sprite as a geometric obstacle and route every extra pixel around it with a forward-biased path.
    """
    g = np.array(grid, copy=True)
    stage1 = _first_stage(g)
    out = stage1["out"].copy()
    bg = stage1["bg"]
    wall = stage1["wall"]
    fill = stage1["fill"]
    rad = stage1["rad"]
    sprite_mask = stage1["sprite_mask"]

    movers = _extra_pixels(g, bg, wall, fill)
    placements: list[tuple[tuple[int, int], int]] = []
    for r, c, color in movers:
        path = _find_detour_path((r, c), sprite_mask, rad)
        if path is None:
            placements.append(((r, c), color))
        else:
            placements.append((path[-1], color))

    for r, c, _ in movers:
        out[r, c] = bg
    for (rr, cc), color in placements:
        out[rr, cc] = color
    return out


# --------------- generator (o5x) ---------------

# -- Configuration constants --------------------------------------------------
GRID_H_VERT = (11, 18)
GRID_H_HORIZ = (8, 18)
GRID_W = (15, 19)

WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 10)
DEPTH_HORIZ = (6, 8)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5

OPEN_END_MARGIN = 3

SEED_SIZE_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 3

MIN_MOVERS = 2
MAX_MOVERS = 5
MOVER_MIN_SEP = 2
MOVER_WALL_DIST = 3
MOVER_END_SEP = 1

MIN_NEW_FILL = 20
PALETTE_SIZE = (4, 5)
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
            ) >= 2
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


def _touches_border(mask):
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _chebyshev_far_enough(pos: tuple[int, int], placed: list[tuple[int, int]], min_sep: int) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= min_sep for rr, cc in placed)


def _candidate_movers(base_grid: np.ndarray, stage1: dict) -> list[dict]:
    bg = stage1["bg"]
    wall = stage1["wall"]
    fill = stage1["fill"]
    rad = stage1["rad"]
    sprite_mask = stage1["sprite_mask"]
    h, w = base_grid.shape

    wall_mask = base_grid == wall
    wall_dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    base_bg = (base_grid == bg) & (stage1["out"] == bg) & (~sprite_mask)
    coords = [(int(r), int(c)) for r, c in zip(*np.where(base_bg & (wall_dist >= MOVER_WALL_DIST)))]
    random.shuffle(coords)

    records = []
    for r, c in coords:
        if _goal_reached(r, c, rad, h, w):
            continue
        if not _ray_hits_sprite((r, c), sprite_mask, rad):
            continue
        path = _find_detour_path((r, c), sprite_mask, rad)
        if path is None:
            continue
        side_steps = _path_side_steps(path, rad)
        if side_steps == 0:
            continue
        end = path[-1]
        if end == _straight_stop((r, c), sprite_mask, rad):
            continue
        records.append(
            {
                "start": (r, c),
                "end": end,
                "side_steps": side_steps,
                "path": path,
                "shift": (end[0] - r, end[1] - c),
            }
        )
    return records


def _pick_movers(records: list[dict]) -> list[dict]:
    if len(records) < MIN_MOVERS:
        return []
    random.shuffle(records)
    chosen = []
    used_starts = []
    used_ends = []
    blocked_cells = set()
    target_n = random.randint(MIN_MOVERS, min(MAX_MOVERS, len(records)))
    for rec in records:
        start = rec["start"]
        end = rec["end"]
        if start in blocked_cells or end in blocked_cells:
            continue
        if not _chebyshev_far_enough(start, used_starts, MOVER_MIN_SEP):
            continue
        if not _chebyshev_far_enough(end, used_ends, MOVER_END_SEP):
            continue
        chosen.append(rec)
        used_starts.append(start)
        used_ends.append(end)
        blocked_cells.add(start)
        blocked_cells.add(end)
        if len(chosen) >= target_n:
            break
    if len(chosen) < MIN_MOVERS:
        return []
    return chosen


def _variant_metrics(grid: np.ndarray) -> dict:
    stage1 = _first_stage(grid)
    movers = _extra_pixels(grid, stage1["bg"], stage1["wall"], stage1["fill"])
    rerouted = 0
    side_steps = []
    horizontal = 0
    vertical = 0
    for r, c, _color in movers:
        path = _find_detour_path((r, c), stage1["sprite_mask"], stage1["rad"])
        if path is None:
            continue
        steps = _path_side_steps(path, stage1["rad"])
        if steps > 0 and path[-1] != _straight_stop((r, c), stage1["sprite_mask"], stage1["rad"]):
            rerouted += 1
            side_steps.append(steps)
            if stage1["rad"][0] != 0:
                horizontal += _sgn(path[-1][1] - c)
            else:
                vertical += _sgn(path[-1][0] - r)
    return {
        "movers": len(movers),
        "rerouted": rerouted,
        "avg_side_steps": float(np.mean(side_steps)) if side_steps else 0.0,
        "horizontal_bias": horizontal,
        "vertical_bias": vertical,
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

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    mover_colors = palette[3:]
    if not mover_colors:
        return None

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
        base_stage1 = _first_stage(grid)
    except Exception:
        return None

    nominal_new_fill = (base_stage1["out"] == fill_color) & (grid == bg)
    if int(nominal_new_fill.sum()) < MIN_NEW_FILL or not _touches_border(nominal_new_fill):
        return None

    records = _candidate_movers(grid, base_stage1)
    chosen = _pick_movers(records)
    if len(chosen) < MIN_MOVERS:
        return None

    for i, rec in enumerate(chosen):
        r, c = rec["start"]
        grid[r, c] = mover_colors[i % len(mover_colors)]

    try:
        out = main(grid)
    except Exception:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
        return None

    metrics = _variant_metrics(grid)
    if metrics["rerouted"] < MIN_MOVERS:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 D7 variant input")
