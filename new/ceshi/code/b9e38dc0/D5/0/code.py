# Variant of b9e38dc0
# Dimension: D5 — 目标/终止条件 (dst)
# Change: 辐射射线延伸到网格边缘终止；孤立像素同样移动到网格边缘... → 辐射到遇到另一个精灵时停止

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis for the D5 variant
# ----------------------------------
# a) Concrete behavior:
#    The input now contains two wall components of the same color. Only the component nearest the fill seed
#    is the active enclosure; the second component sits in the radiate direction and acts as a collision stop.
#    In the original solver the two endpoint rays always continued to the canvas edge, so the filled wedge was
#    bounded by the border. In this variant each endpoint ray is traced step-by-step and halts immediately before
#    the next wall component. Example: if the active enclosure opens upward and a second same-colored enclosure is
#    above it, the old solver keeps the two diagonal rays going to row 0; the new solver stops both rays when the
#    next step would land on the lower border of the upper enclosure. The resulting filled zone is truncated by
#    object-to-object distance rather than by global canvas size.
# b) Complexity calibration:
#    The change is kept narrow: the wall parsing, seed selection, interior flood-fill and obstacle shadows are
#    preserved. Only the ray termination rule changes. To make that visible, the generator always places one extra
#    wall sprite in the radiate corridor and rejects placements where the collision stop would be visually inert.
# c) Implicit-assumption audit:
#    - `wall = most frequent non-bg color` still holds because both enclosures share that color.
#    - `wall_comp = closest component to the fill seed` becomes essential; the extra wall must stay farther away.
#    - The old `add_ray*` helpers assumed the only stopping event was out-of-bounds. That assumption is invalid
#      here, so they are replaced with traced ray walkers that also check for another wall component.
#    - BFS still assumes non-bg/non-fill cells are not passable; the generator therefore keeps the blocking sprite
#      as wall-colored pixels rather than introducing a new collision color.
# d) Placement legality:
#    The blocker is a translated copy of the active wall in the radiate direction. The generator validates that the
#    two wall components remain separate 8-connected components, both rays collide with the blocker, and the new
#    solver output stays inside the canvas without touching the border.
# e) Visual observability:
#    The second wall is a visible object, not a hidden numeric trigger. Samples show the same seed and same active
#    enclosure but with different distances to the blocking sprite, so humans can read ray length from geometry.
# f) Visual cleanliness:
#    Obstacle count is kept modest and candidate placements stay away from wall pixels, which avoids burying the
#    new stop condition under shadow artifacts. The blocker is aligned in the radiate corridor so the interaction
#    is easy to see.
# g) Color inference:
#    No color remapping is introduced. The fill color, wall color and optional obstacle colors keep their original
#    roles; only the fill extent changes because the wall-colored blocking sprite is closer or farther away.


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


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, copy=True)
    h, w = g.shape

    vals = np.unique(g)
    bg = get_bg(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = _component_sets(wall_mask)
    assert comps, "expected at least one wall connected component"

    other_colors = [c for c in non_bg if c != wall]
    assert other_colors, "expected at least one non-wall color"

    wall_points = [p for comp in comps for p in comp]

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
    if not fill_pos:
        return {
            "grid": g,
            "shape": (h, w),
            "bg": bg,
            "wall": wall,
            "fill": fill,
            "fill_pos": [],
        }

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
    ray_info = _component_ray_info(wall_comp, (h, w))
    comp_mask = ray_info["comp_mask"]
    other_wall_mask = wall_mask & (~comp_mask)

    return {
        "grid": g,
        "shape": (h, w),
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "fill_pos": fill_pos,
        "wall_mask": wall_mask,
        "wall_comp": wall_comp,
        "comp_mask": comp_mask,
        "other_wall_mask": other_wall_mask,
        "endpoints": ray_info["endpoints"],
        "out_dirs": ray_info["out_dirs"],
        "tip_12": ray_info["tip_12"],
        "rad": ray_info["rad"],
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


def _solve(grid: np.ndarray, *, stop_at_other_wall: bool, return_meta: bool = False):
    scene = _parse_scene(grid)
    g = scene["grid"]
    if not scene["fill_pos"]:
        return (g, {"scene": scene, "ray_meta": [], "filled_mask": np.zeros_like(g, dtype=bool)}) if return_meta else g

    barrier, ray_meta = _build_barrier(scene, stop_at_other_wall=stop_at_other_wall)
    passable = (~barrier) & ((g == scene["bg"]) | (g == scene["fill"]))

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

    obstacles = (g != scene["bg"]) & (g != scene["fill"]) & (g != scene["wall"])

    def shadow_process(coords: list[tuple[int, int]]) -> None:
        started = False
        blocked = False
        for r, c in coords:
            if filled_mask[r, c]:
                started = True
            if started and obstacles[r, c]:
                blocked = True
                continue
            if blocked and filled_mask[r, c]:
                out[r, c] = scene["bg"]
                filled_mask[r, c] = False

    rad = scene["rad"]
    h, w = g.shape
    if rad != (0, 0):
        dr, dc = rad
        if dr != 0 and dc == 0:
            rows = range(h) if dr > 0 else range(h - 1, -1, -1)
            for c in range(w):
                shadow_process([(r, c) for r in rows])
        elif dc != 0 and dr == 0:
            cols = range(w) if dc > 0 else range(w - 1, -1, -1)
            for r in range(h):
                shadow_process([(r, c) for c in cols])
        else:
            if dr == dc:
                for k in range(-(w - 1), h):
                    coords = [(r, r - k) for r in range(h) if 0 <= r - k < w]
                    coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
                    shadow_process(coords)
            else:
                for k in range(h + w - 1):
                    coords = [(r, k - r) for r in range(h) if 0 <= k - r < w]
                    coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
                    shadow_process(coords)

    if return_meta:
        return out, {"scene": scene, "ray_meta": ray_meta, "filled_mask": filled_mask.copy()}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the region enclosed by one wall sprite, but stop the radiating wedge when it reaches another wall sprite.

    Input structure:
    - Background is the most common color. The wall color is the most common non-background color and may form
      multiple separated partial enclosures of that same color.
    - The fill-colored seed lies inside one enclosure; any extra non-wall colors are static obstacles.

    Rule:
    - Select the wall component nearest the fill seed as the active enclosure, flood-fill its interior, and let the
      fill radiate outward through the opening.
    - The two boundary rays of that radiating wedge advance in the opening direction, but each ray halts as soon as
      its next step would touch another wall component instead of running all the way to the canvas border.
    - Obstacles still cast background-colored shadows behind themselves along the radiate direction.

    Key insight: The visible spacing between the seed-bearing enclosure and the next wall sprite determines where the
    wedge terminates; the global grid edge is no longer the decisive stopping condition.
    """
    return _solve(grid, stop_at_other_wall=True)


# --------------- generator (collision-stop variant) ---------------

# Canvas size bounds
GRID_H_VERT = (14, 18)
GRID_H_HORIZ = (9, 18)
GRID_W = (17, 19)

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
BLOCKER_OFFSET_MAX = 4
EDGE_MARGIN = BLOCKER_OFFSET_MAX + 1

# Fill seed
SEED_SIZE_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 3

# Obstacles
MAX_OBSTACLES = 2
OBSTACLE_IN_FILL_PROB = 0.35
OBSTACLE_MIN_SEP = 2

# Validation
MIN_NEW_FILL = 10
PALETTE_SIZE = (3, 5)
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
        max_depth = min(DEPTH_HORIZ[1], w - EDGE_MARGIN - 3)
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


def _obstacle_fits(pos, placed):
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone):
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    far_from_wall = dist > 1
    eligible = (grid == bg) & far_from_wall
    all_coords = list(zip(*np.where(eligible)))
    fill_coords = [p for p in all_coords if nominal_fill_zone[p[0], p[1]]]
    random.shuffle(all_coords)
    random.shuffle(fill_coords)
    return [(int(r), int(c)) for r, c in all_coords], [(int(r), int(c)) for r, c in fill_coords]


def _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid):
    placed = []
    n_fill = random.randint(0, min(n_obstacles, len(fill_candidates)))
    if fill_candidates and random.random() < OBSTACLE_IN_FILL_PROB:
        n_fill = max(1, n_fill)
    for p in fill_candidates:
        if len(placed) >= n_fill:
            break
        if _obstacle_fits(p, placed):
            placed.append(p)
    placed_set = set(placed)
    for p in candidates:
        if len(placed) >= n_obstacles:
            break
        if p not in placed_set and _obstacle_fits(p, placed):
            placed.append(p)
            placed_set.add(p)
    for i, (r, c) in enumerate(placed):
        grid[r, c] = obstacle_colors[i % len(obstacle_colors)]


def _touches_border(mask):
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _translate_points(points: set[tuple[int, int]], rad: tuple[int, int], offset: int) -> set[tuple[int, int]]:
    dr = rad[0] * offset
    dc = rad[1] * offset
    return {(r + dr, c + dc) for r, c in points}


def _blocker_ok(base_grid: np.ndarray, wall_color: int) -> bool:
    comps = _component_sets(base_grid == wall_color)
    return len(comps) == 2


def _try_add_blocker(grid, wall_color, active_pts, rad):
    h, w = grid.shape
    ray_info = _component_ray_info(active_pts, (h, w))
    bg = get_bg(grid)
    candidate_paths = []
    for ep, d, is_12 in zip(ray_info["endpoints"], ray_info["out_dirs"], ray_info["tip_12"]):
        path, _ = _trace_ray(ep, d, ray_info["rad"], is_12, (h, w), None)
        if path:
            candidate_paths.append(path)

    random.shuffle(candidate_paths)
    for path in candidate_paths:
        lo = max(0, BLOCKER_OFFSET_MIN - 1)
        hi = min(len(path), BLOCKER_OFFSET_MAX + 2)
        idxs = list(range(lo, hi))
        random.shuffle(idxs)
        for idx in idxs:
            r, c = path[idx]
            blocker_pts = {(r, c)}
            if rad[0] != 0:
                for dc in (-1, 1):
                    cc = c + dc
                    if 0 <= cc < w:
                        blocker_pts.add((r, cc))
            else:
                for dr in (-1, 1):
                    rr = r + dr
                    if 0 <= rr < h:
                        blocker_pts.add((rr, c))
            if any((rr, cc) in active_pts for rr, cc in blocker_pts):
                continue
            if any(grid[rr, cc] != bg for rr, cc in blocker_pts):
                continue

            trial = grid.copy()
            for rr, cc in blocker_pts:
                trial[rr, cc] = wall_color
            if not _blocker_ok(trial, wall_color):
                continue

            stop_mask = np.zeros((h, w), dtype=bool)
            for rr, cc in blocker_pts:
                stop_mask[rr, cc] = True
            hits = []
            for ep, d, is_12 in zip(ray_info["endpoints"], ray_info["out_dirs"], ray_info["tip_12"]):
                _, hit = _trace_ray(ep, d, ray_info["rad"], is_12, (h, w), stop_mask)
                hits.append(hit is not None)
            if any(hits):
                return trial, blocker_pts
    return None, None


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

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    obstacle_colors = palette[3:]

    grid = np.full((h, w), bg, dtype=int)
    active_pts = set()
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        active_pts.add((r, c))
        grid[r, c] = wall_color

    grid, blocker_pts = _try_add_blocker(grid, wall_color, active_pts, rad)
    if grid is None:
        return None

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
        nominal_out = main(grid)
        edge_out = _solve(grid, stop_at_other_wall=False)
    except Exception:
        return None
    if np.array_equal(nominal_out, edge_out):
        return None
    nominal_fill_zone = (nominal_out == fill_color) & (grid == bg)

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    if n_obstacles:
        candidates, fill_candidates = _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone)
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        out, meta = _solve(grid, stop_at_other_wall=True, return_meta=True)
        edge_out = _solve(grid, stop_at_other_wall=False)
    except Exception:
        return None

    hits = [rm["hit"] is not None for rm in meta["ray_meta"]]
    if not hits or not any(hits):
        return None
    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL:
        return None
    if np.array_equal(out, edge_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 D5 input")
