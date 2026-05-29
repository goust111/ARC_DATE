# Combo variant of b9e38dc0 (index 37)
# combo_spec: D3/0+D5/0+D12/2
# Name: 周期涂色+碰撞终止+配对相对

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and trace:
#    Start from the original funnel fill: a seed inside a wall enclosure floods
#    its interior and continues outward inside the wedge bounded by the two
#    endpoint rays. Here the scene is specialized into an opposite-facing pair.
#    Example: a seed-bearing left sprite opens to the right, and a matching wall
#    sprite opens back to the left. In the original task the wedge would keep
#    expanding until the grid edge and stay one flat color. In this combo, the
#    wedge is truncated when either boundary ray's next step would touch the
#    facing sprite, and the surviving radiated depths are recolored as AABB:
#    fill, fill, wall, wall, fill, fill, ... . So distance-to-partner controls
#    how many periodic bands fit before the collision stop.
#
# b) Complexity calibration:
#    The pairing rule is kept to one visible layout family: exactly two wall
#    components share the wall color and face each other on one axis. Periodic
#    coloring uses only the seed color and the wall color, with one fixed global
#    period (AABB). This keeps the rule rich enough to be seen but narrow enough
#    to infer from a few examples.
#
# c) Hidden-assumption audit:
#    - The original single-wall assumption is broken, so the solver now parses
#      all wall components, picks the seed-bearing one as active, and explicitly
#      matches it to an opposite-facing partner instead of treating every extra
#      wall pixel as irrelevant.
#    - The old ray builders assumed out-of-bounds was the only stop event. That
#      is replaced by traced ray walking that halts on partner contact.
#    - The original `out[filled_mask] = fill` is no longer sufficient because
#      radiated cells are not monochrome; coloring is delayed until after shadow
#      removal so the periodic bands apply only to cells that truly survive.
#
# d) Placement legality:
#    The generator constructs the pair from one local wall blueprint and maps it
#    twice with opposite radiate directions. The two anchors share the lateral
#    coordinate so their openings visibly face each other, and the front-facing
#    opening distance is chosen only from values that keep both sprites
#    in-bounds and disconnected. The solver then uses that facing front edge as
#    the collision-stop plane for the radiating wedge.
#
# e) Visual observability:
#    A human can read the pairing directly from the input because the two wall
#    sprites have the same geometry and face one another. In the output, the
#    wedge stops in front of the opposing sprite instead of the border, and the
#    truncated corridor shows repeated fill/wall bands whose count changes with
#    the spacing between the pair.
#
# f) Visual neatness:
#    Only one sprite carries the seed, so there is a single transformed region
#    instead of two overlapping radiations. Obstacle count is kept low and
#    obstacle placement avoids wall-adjacent clutter, which leaves the partner
#    interaction visually dominant.
#
# g) Change inferability:
#    No hidden ordering by color id is used. The two recolor values come from
#    structure that is already visible in the input: the seed color and the wall
#    color. The output always contains both colors in the radiated zone, so the
#    periodic rule is supported by repeated local evidence rather than a one-off
#    remapping.
# ---------------------------------------------------------------------------


# -- Configuration constants --------------------------------------------------
GRID_H_VERT = (18, 18)
GRID_H_HORIZ = (8, 13)
GRID_W = (18, 19)

WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (6, 6)
DEPTH_HORIZ = (5, 6)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.45

PAIR_DISTANCE_MIN = 5
PAIR_DISTANCE_MAX = 10
PATTERN_BLOCK = 2
PATTERN_REPEAT_DEPTH = 4

SEED_SIZE_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 3

MAX_OBSTACLES = 2
OBSTACLE_IN_FILL_PROB = 0.35
OBSTACLE_MIN_SEP = 2

MIN_NEW_PAINT = 14
MIN_RADIATE_PIXELS = 10
MIN_WALL_BAND_PIXELS = 4
MIN_EDGE_DIFF = 1
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 600


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _projection(r: int, c: int, d: tuple[int, int]) -> int:
    return int(r) * int(d[0]) + int(c) * int(d[1])


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


def _component_ray_info(comp: set[tuple[int, int]], shape: tuple[int, int]) -> dict:
    comp_mask = _component_mask(shape, comp)
    endpoints = sorted((r, c) for r, c in comp if len(_comp_neighbors(comp_mask, r, c)) == 1)
    assert len(endpoints) >= 2, "expected wall component to have endpoints"
    out_info = [_outward_dir(comp_mask, ep) for ep in endpoints]
    out_dirs = [d for d, _ in out_info]
    tip_12 = [flag for _, flag in out_info]
    dr_sum = sum(d[0] for d in out_dirs)
    dc_sum = sum(d[1] for d in out_dirs)
    rad = (_sgn(dr_sum), _sgn(dc_sum))
    return {
        "comp": comp,
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
    stop_proj: int | None = None,
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
        if stop_proj is not None and _projection(nr, nc, rad) >= stop_proj:
            return path, (nr, nc)
        if stop_mask is not None and stop_mask[nr, nc]:
            return path, (nr, nc)
        path.append((nr, nc))
        rr, cc = nr, nc


def _distance_to_comp(points: list[tuple[int, int]], comp: set[tuple[int, int]]) -> int:
    best_d = 10**9
    for r, c in points:
        for wr, wc in comp:
            d = abs(r - wr) + abs(c - wc)
            if d < best_d:
                best_d = d
                if best_d == 0:
                    return 0
    return best_d


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, copy=True)
    h, w = g.shape
    bg = get_bg(g)
    vals = np.unique(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = _component_sets(wall_mask)
    assert comps, "expected at least one wall component"

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
            "bg": int(bg),
            "wall": int(wall),
            "fill": int(fill),
            "fill_pos": [],
        }

    infos = [_component_ray_info(comp, (h, w)) for comp in comps]

    active_idx = min(
        range(len(infos)),
        key=lambda i: (_distance_to_comp(fill_pos, infos[i]["comp"]), -len(infos[i]["comp"])),
    )
    active = infos[active_idx]
    active_rad = active["rad"]
    open_proj = max(_projection(r, c, active_rad) for r, c in active["endpoints"])
    perp = (active_rad[1], -active_rad[0])
    active_perp = sum(_projection(r, c, perp) for r, c in active["endpoints"]) / len(active["endpoints"])

    partner_idx = None
    if len(infos) > 1:
        def partner_score(i: int) -> tuple[float, ...]:
            if i == active_idx:
                return (9.0, 9.0, 9.0, 9.0)
            info = infos[i]
            opposite = info["rad"] == (-active_rad[0], -active_rad[1])
            ahead = min(_projection(r, c, active_rad) for r, c in info["comp"]) - open_proj
            partner_perp = sum(_projection(r, c, perp) for r, c in info["endpoints"]) / len(info["endpoints"])
            lateral = abs(partner_perp - active_perp)
            return (
                0.0 if opposite else 1.0,
                0.0 if ahead >= 0 else 1.0,
                float(lateral),
                abs(float(ahead)),
            )

        partner_idx = min((i for i in range(len(infos)) if i != active_idx), key=partner_score)

    partner = infos[partner_idx] if partner_idx is not None else None
    partner_mask = partner["comp_mask"] if partner is not None else None

    return {
        "grid": g,
        "shape": (h, w),
        "bg": int(bg),
        "wall": int(wall),
        "fill": int(fill),
        "fill_pos": fill_pos,
        "wall_mask": wall_mask,
        "infos": infos,
        "active": active,
        "partner": partner,
        "partner_mask": partner_mask,
    }


def _build_barrier(scene: dict, *, stop_at_partner: bool) -> tuple[np.ndarray, list[dict]]:
    active = scene["active"]
    barrier = active["comp_mask"].copy()
    stop_mask = scene["partner_mask"] if stop_at_partner else None
    stop_proj = None
    if stop_at_partner and scene["partner"] is not None:
        stop_proj = min(_projection(r, c, active["rad"]) for r, c in scene["partner"]["endpoints"])
    ray_meta = []
    for ep, d, is_12 in zip(active["endpoints"], active["out_dirs"], active["tip_12"]):
        path, hit = _trace_ray(ep, d, active["rad"], is_12, scene["shape"], stop_mask, stop_proj)
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


def _solve_core(
    grid: np.ndarray,
    *,
    stop_at_partner: bool,
    periodic: bool,
    return_meta: bool = False,
):
    scene = _parse_scene(grid)
    g = scene["grid"]
    if not scene["fill_pos"]:
        empty = np.zeros_like(g, dtype=bool)
        meta = {
            "scene": scene,
            "paint_mask": empty.copy(),
            "radiated_mask": empty.copy(),
            "wall_band_mask": empty.copy(),
            "ray_meta": [],
            "depth_map": np.full(g.shape, -1, dtype=int),
        }
        return (g, meta) if return_meta else g

    barrier, ray_meta = _build_barrier(scene, stop_at_partner=stop_at_partner)
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
    rad = scene["active"]["rad"]
    depth_map = np.full(g.shape, -1, dtype=int)
    radiated_mask = np.zeros_like(filled_mask, dtype=bool)
    wall_band_mask = np.zeros_like(filled_mask, dtype=bool)
    stop_depth = None
    if rad != (0, 0):
        open_proj = max(_projection(r, c, rad) for r, c in scene["active"]["endpoints"])
        rows, cols = np.indices(g.shape)
        depth_map = rows * rad[0] + cols * rad[1] - open_proj
        if stop_at_partner and scene["partner"] is not None:
            partner_proj = min(_projection(r, c, rad) for r, c in scene["partner"]["endpoints"])
            stop_depth = partner_proj - open_proj
            filled_mask &= (depth_map < stop_depth) | (depth_map < 0)
    obstacles = (g != scene["bg"]) & (g != scene["fill"]) & (g != scene["wall"])
    h, w = g.shape

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
                filled_mask[r, c] = False

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

    out = g.copy()
    out[filled_mask] = scene["fill"]

    if rad != (0, 0):
        radiated_mask = filled_mask & (depth_map >= 0)
        if periodic:
            wall_band_mask = radiated_mask & (((depth_map // PATTERN_BLOCK) % 2) == 1)
            out[wall_band_mask] = scene["wall"]

    meta = {
        "scene": scene,
        "paint_mask": filled_mask.copy(),
        "radiated_mask": radiated_mask.copy(),
        "wall_band_mask": wall_band_mask.copy(),
        "ray_meta": ray_meta,
        "depth_map": depth_map,
        "ray_hits": [rm["hit"] is not None for rm in ray_meta],
        "stop_depth": stop_depth,
    }
    return (out, meta) if return_meta else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a seed-bearing enclosure, but truncate its radiating wedge against an opposite-facing paired sprite and recolor that wedge periodically.

    Input structure:
    - The background is the most common color. The dominant non-background color forms two separated partial enclosures of the same wall color, arranged as a visible facing pair.
    - One enclosure contains the fill-colored seed blob near its closed end; any remaining non-wall colors are static obstacles.

    Rule:
    - Use the enclosure nearest the seed as the active sprite. Flood-fill its interior and let the fill continue outward through the opening, bounded by the two endpoint rays.
    - Trace each boundary ray step by step and stop it as soon as the next step would cross the opposite-facing partner's front opening edge instead of running to the canvas edge.
    - Keep the trapped interior in the fill color, but recolor the surviving outward radiated cells with a repeating AABB pattern made from fill color and wall color. Obstacles still carve background shadows behind themselves along the active radiate direction.

    Key insight:
    - The partner sprite is not decorative: its facing position sets the exact stopping depth of the wedge, and that finite depth determines how many alternating fill/wall color bands appear before the collision.
    """
    return _solve_core(grid, stop_at_partner=True, periodic=True)


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
        max_depth = min(DEPTH_HORIZ[1], w - 6)
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

    if not (WALL_PIXELS_MIN <= len(pts) <= WALL_PIXELS_MAX):
        return None
    if len(_degree1_pixels(pts)) != 2:
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


def _anchor_candidates(rad, h, w, depth, gap):
    if rad == (-1, 0):
        return [(r, c) for r in range(0, h - depth) for c in range(0, w - gap)]
    if rad == (1, 0):
        return [(r, c) for r in range(depth, h) for c in range(0, w - gap)]
    if rad == (0, -1):
        return [(r, c) for r in range(0, h - gap) for c in range(0, w - depth)]
    return [(r, c) for r in range(0, h - gap) for c in range(depth, w)]


def _place_active_anchor(rad, h, w, depth, gap):
    if rad == (-1, 0):
        rows = range(PATTERN_REPEAT_DEPTH, h - depth)
        cols = range(0, w - gap)
    elif rad == (1, 0):
        rows = range(depth, h - PATTERN_REPEAT_DEPTH)
        cols = range(0, w - gap)
    elif rad == (0, -1):
        rows = range(0, h - gap)
        cols = range(PATTERN_REPEAT_DEPTH, w - depth)
    else:
        rows = range(0, h - gap)
        cols = range(depth, w - PATTERN_REPEAT_DEPTH)

    anchors = [(r, c) for r in rows for c in cols]
    if not anchors:
        return None
    return random.choice(anchors)


def _place_pair_anchors(rad, h, w, depth, gap):
    if rad[0]:
        max_corr = min(PAIR_DISTANCE_MAX, h - 1 - 2 * depth)
    else:
        max_corr = min(PAIR_DISTANCE_MAX, w - 1 - 2 * depth)
    if max_corr < PAIR_DISTANCE_MIN:
        return None
    corridor = random.randint(PAIR_DISTANCE_MIN, max_corr)

    if rad == (0, 1):
        c_lo = depth
        c_hi = w - 1 - corridor - depth
        if c_lo > c_hi or h - gap - 1 < 0:
            return None
        c_active = random.randint(c_lo, c_hi)
        r_common = random.randint(0, h - gap - 1)
        return (r_common, c_active), (r_common, c_active + corridor), corridor
    if rad == (0, -1):
        c_lo = corridor + depth
        c_hi = w - 1 - depth
        if c_lo > c_hi or h - gap - 1 < 0:
            return None
        c_active = random.randint(c_lo, c_hi)
        r_common = random.randint(0, h - gap - 1)
        return (r_common, c_active), (r_common, c_active - corridor), corridor
    if rad == (1, 0):
        r_lo = depth
        r_hi = h - 1 - corridor - depth
        c_options = []
        if corridor <= w - gap - 1:
            c_options.extend(range(corridor, w - gap))
        if w - gap - 1 - corridor >= 0:
            c_options.extend(range(0, w - gap - corridor))
        if r_lo > r_hi or not c_options:
            return None
        r_active = random.randint(r_lo, r_hi)
        c_common = random.choice(c_options)
        return (r_active, c_common), (r_active + corridor, c_common), corridor

    r_lo = corridor + depth
    r_hi = h - 1 - depth
    c_options = []
    if corridor <= w - gap - 1:
        c_options.extend(range(corridor, w - gap))
    if w - gap - 1 - corridor >= 0:
        c_options.extend(range(0, w - gap - corridor))
    if r_lo > r_hi or not c_options:
        return None
    r_active = random.randint(r_lo, r_hi)
    c_common = random.choice(c_options)
    return (r_active, c_common), (r_active - corridor, c_common), corridor


def _render_wall_points(local_wall, rad, anchor):
    return {(_map_local(u, v, rad, anchor)) for u, v in local_wall}


def _find_partner_anchor(local_wall, partner_rad, active_pts, active_info, shape):
    gap = max(v for _, v in local_wall)
    active_rad = active_info["rad"]
    open_proj = max(_projection(r, c, active_rad) for r, c in active_info["endpoints"])
    perp = (active_rad[1], -active_rad[0])
    active_perp = sum(_projection(r, c, perp) for r, c in active_info["endpoints"]) / len(active_info["endpoints"])

    def anchor_for_endpoint(point: tuple[int, int], endpoint_idx: int) -> tuple[int, int]:
        r, c = point
        if partner_rad[0] != 0:
            return (r, c) if endpoint_idx == 0 else (r, c - gap)
        return (r, c) if endpoint_idx == 0 else (r - gap, c)

    best = None
    best_score = None
    ray_paths = []
    for ep, d, is_12 in zip(active_info["endpoints"], active_info["out_dirs"], active_info["tip_12"]):
        path, _ = _trace_ray(ep, d, active_rad, is_12, shape, None)
        ray_paths.append(path)

    candidate_depths = list(range(PATTERN_REPEAT_DEPTH, PAIR_DISTANCE_MAX + 1))
    random.shuffle(candidate_depths)
    for ray_idx, path in enumerate(ray_paths):
        for depth in candidate_depths:
            if depth >= len(path):
                continue
            hit_point = path[depth]
            for endpoint_idx in (ray_idx, 1 - ray_idx):
                anchor = anchor_for_endpoint(hit_point, endpoint_idx)
                partner_pts = _render_wall_points(local_wall, partner_rad, anchor)
                if any(r < 0 or r >= shape[0] or c < 0 or c >= shape[1] for r, c in partner_pts):
                    continue
                if partner_pts & active_pts:
                    continue

                wall_mask = np.zeros(shape, dtype=bool)
                for r, c in active_pts | partner_pts:
                    wall_mask[r, c] = True
                if len(_component_sets(wall_mask)) != 2:
                    continue

                partner_info = _component_ray_info(partner_pts, shape)
                if partner_info["rad"] != partner_rad:
                    continue

                partner_front = min(_projection(r, c, active_rad) for r, c in partner_pts)
                if partner_front - open_proj < 2:
                    continue

                stop_mask = np.zeros(shape, dtype=bool)
                for r, c in partner_pts:
                    stop_mask[r, c] = True

                hits = []
                depths = []
                for ep, d, is_12 in zip(active_info["endpoints"], active_info["out_dirs"], active_info["tip_12"]):
                    ray_path, hit = _trace_ray(ep, d, active_info["rad"], is_12, shape, stop_mask)
                    hits.append(hit is not None)
                    depths.append(len(ray_path))
                if not any(hits):
                    continue
                if max(depths) < PATTERN_REPEAT_DEPTH:
                    continue

                partner_perp = sum(_projection(r, c, perp) for r, c in partner_info["endpoints"]) / len(partner_info["endpoints"])
                lateral = abs(partner_perp - active_perp)
                front_gap = max(0, partner_front - open_proj)
                score = (
                    0 if all(hits) else 1,
                    abs(front_gap - (PATTERN_REPEAT_DEPTH + 1)),
                    abs(depths[ray_idx] - (PATTERN_REPEAT_DEPTH + 1)),
                    lateral,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best = (anchor, partner_info, hits)
    return best


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
    eligible = (grid == bg) & (dist > 1)
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


def _gen_attempt():
    rad = random.choice([(-1, 0), (1, 0)])
    h = random.randint(*GRID_H_VERT)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built
    placed = _place_pair_anchors(rad, h, w, depth, gap)
    if placed is None:
        return None
    active_anchor, partner_anchor, corridor = placed

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    obstacle_colors = palette[3:]

    grid = np.full((h, w), bg, dtype=int)
    active_pts = _render_wall_points(local_wall, rad, active_anchor)
    partner_rad = (-rad[0], -rad[1])
    partner_pts = _render_wall_points(local_wall, partner_rad, partner_anchor)

    for r, c in active_pts:
        grid[r, c] = wall_color

    for r, c in partner_pts:
        if (r, c) in active_pts:
            return None
        grid[r, c] = wall_color

    if len(_component_sets(grid == wall_color)) != 2:
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
        r, c = _map_local(u, v, rad, active_anchor)
        if grid[r, c] != bg:
            return None
        grid[r, c] = fill_color

    try:
        nominal_out, nominal_meta = _solve_core(grid, stop_at_partner=True, periodic=True, return_meta=True)
        edge_out, edge_meta = _solve_core(grid, stop_at_partner=False, periodic=True, return_meta=True)
    except Exception:
        return None

    if scene_partner := nominal_meta["scene"]["partner"]:
        if scene_partner["rad"] != partner_rad:
            return None
    else:
        return None

    if not nominal_meta["ray_hits"] or not any(nominal_meta["ray_hits"]):
        return None

    nominal_fill_zone = nominal_meta["paint_mask"] & (grid == bg)
    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    if n_obstacles:
        candidates, fill_candidates = _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone)
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        out, meta = _solve_core(grid, stop_at_partner=True, periodic=True, return_meta=True)
        edge_out, edge_meta = _solve_core(grid, stop_at_partner=False, periodic=True, return_meta=True)
    except Exception:
        return None

    paint_mask = meta["paint_mask"] & (grid == bg)
    radiated_mask = meta["radiated_mask"] & paint_mask
    wall_band_mask = meta["wall_band_mask"] & paint_mask
    edge_paint_mask = edge_meta["paint_mask"] & (grid == bg)

    if not meta["ray_hits"] or not any(meta["ray_hits"]):
        return None
    if int(paint_mask.sum()) < MIN_NEW_PAINT:
        return None
    if int(radiated_mask.sum()) < MIN_RADIATE_PIXELS:
        return None
    if int(wall_band_mask.sum()) < MIN_WALL_BAND_PIXELS:
        return None
    if int(edge_paint_mask.sum()) - int(paint_mask.sum()) < MIN_EDGE_DIFF:
        return None
    if np.array_equal(out, edge_out):
        return None
    if not np.any((grid == bg) & (out == wall_color)):
        return None
    if not np.any((grid == bg) & (out == fill_color)):
        return None
    if int(meta["depth_map"][radiated_mask].max()) < PATTERN_REPEAT_DEPTH:
        return None

    active_scene = meta["scene"]["active"]
    partner_scene = meta["scene"]["partner"]
    if partner_scene is None:
        return None
    if active_scene["rad"] != rad or partner_scene["rad"] != partner_rad:
        return None

    # Guard against generator degeneration: the pair spacing must matter.
    if corridor < PAIR_DISTANCE_MIN:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo input")
