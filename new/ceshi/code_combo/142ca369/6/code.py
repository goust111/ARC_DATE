# Combo variant of 142ca369 (index 6)
# combo_spec: D8/0+D20/2
# Name: 距离排序发射+先到先得

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

# Design analysis
# spec:
#   fused_rule:
#     - objects stay the same as the base task: 3-cell L arrows plus static wall pixels/segments.
#     - the right-angle vertex is still the ray source and the missing corner still determines
#       the diagonal launch direction.
#     - arrows are no longer independent: they fire from nearer-to-center source to farther-to-center
#       source, using squared Euclidean distance to the grid center on doubled coordinates so both
#       even and odd grid sizes have a visible geometric center.
#     - after one ray finishes, every cell it painted becomes occupied for later rays.
#     - "first come first served" is strict: if a later ray's next diagonal step would enter an
#       occupied ray cell, it terminates immediately. Earlier ray tracks are not reflective walls;
#       only input walls still cause side/corner reflections.
#   concrete_trace:
#     setup:
#       - arrow A starts below-left of the center and points up-right, so it is visually closer
#         to the center.
#       - arrow B starts farther away in the top-left and points down-right toward the same crossing.
#     base_task:
#       - A and B are independent, so both diagonals fully extend; the crossing is just another cell.
#     D20_only style:
#       - if some fixed scan order makes B fire first, A becomes the truncated one.
#     fused variant:
#       - the solver must notice that A is closer to the center, so A fires first.
#       - B then advances until its next step would enter A's painted crossing cell and stops there.
#     visible_difference:
#       - compared with the base task, one tail is missing after the crossing.
#       - compared with a top-left-to-bottom-right order, the opposite ray is missing, so the order
#         cue is genuinely "distance to center", not generic sequencing.
#   complexity_calibration:
#     - keep the original reflection/color-inheritance mechanics for static walls so the task still
#       reads as the same ray-tracing family.
#     - require at least one dynamic ray-ray collision and also require the distance order to disagree
#       with row-major order; otherwise one of the two fused dimensions would be visually invisible.
#     - optionally add one extra later arrow with 0-2 static-wall bounces to keep the scene from
#       collapsing into a trivial "two lines cross" puzzle while avoiding clutter.
#   implicit_assumption_audit:
#     original_loop_order:
#       code: `for start_col, pts in arrows`
#       old assumption: arrow order is irrelevant.
#       fused impact: invalid. Rays now interact, so arrows must be sorted by center distance.
#     original_static_walls_only:
#       code: wall checks only looked at the input wall set.
#       fused impact: partially still valid. Static reflections still only use input walls, but a new
#       diagonal-step collision check against already painted ray cells must happen before advancing.
#     original_write_if_bg:
#       code: `if out[nr, nc] == bg: out[nr, nc] = cur_col`
#       old assumption: an occupied output cell is only bookkeeping.
#       fused impact: invalid for motion. Later rays are forbidden from entering earlier painted cells,
#       so collision must be checked before the write.
#     original_side_bounce_logic:
#       old assumption: any non-bg obstacle relevant to bouncing is an input wall.
#       fused impact: still valid, because the combo spec says occupied ray cells terminate on entry
#       instead of acting like bounce surfaces.
#   placement_legality:
#     - generator reserves all mandatory-pair cells that either ray could occupy under any order, so
#       optional extras cannot accidentally mask the core interaction.
#     - arrow L-shapes, static walls, and planned ray cells are kept disjoint and in-bounds.
#     - wall colors are chosen from colors not used by arrows, preventing same-color 4-connected
#       mergers that could break arrow detection.
#     - all arrow source center distances are unique, because human-visible order should not depend on
#       hidden tiebreaks.
#   observability:
#     - each accepted sample has at least one painted-cell collision, differs from a no-collision
#       simulation, and also differs from a row-major collision simulation.
#     - the nearer arrow is placed near the center while the farther conflicting arrow is visibly
#       earlier in row-major order, so the examples expose the ordering cue.
#   cleanliness:
#     - only one mandatory dynamic collision is enforced.
#     - optional extras are always later in both distance order and row-major order and never cross
#       the reserved mandatory rays, so they add familiar wall behavior without obscuring the core cue.
#   color_inference:
#     - the new rule is geometric rather than a new palette mapping.
#     - wall-color inheritance remains unchanged and is only used on optional static bounces, which
#       gives structural support for any color changes that do appear.
#   new_solver_assumptions_and_generator_guards:
#     - solver assumes every arrow source has a unique center-distance rank.
#       generator guard: reject ties in source distance.
#     - solver assumes dynamic blockers come only from already finished rays.
#       generator guard: optional extras are placed after the mandatory pair in both relevant orders.
#     - solver assumes occupied ray cells terminate only on direct entry, not on side adjacency.
#       generator guard: mandatory interaction is an actual crossing, not an adjacency-only event.


def _is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    pts_set = set(pts)
    miss = [p for p in corners if p not in pts_set]
    if len(miss) != 1:
        return None

    mr, mc = miss[0]
    dr = 1 if mr == r0 else -1
    dc = 1 if mc == c0 else -1

    deg = {p: 0 for p in pts}
    for r, c in pts:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in pts_set:
                deg[(r, c)] += 1
    start = max(deg, key=lambda p: deg[p])
    return (int(dr), int(dc)), (int(start[0]), int(start[1]))


def _center_dist2(r: int, c: int, H: int, W: int) -> int:
    return int((2 * r - (H - 1)) ** 2 + (2 * c - (W - 1)) ** 2)


def _distance_order_key(start: tuple[int, int], H: int, W: int, color: int) -> tuple[int, int, int, int]:
    r, c = start
    return (_center_dist2(r, c, H, W), int(r), int(c), int(color))


def _row_major_order_key(start: tuple[int, int], color: int) -> tuple[int, int, int]:
    r, c = start
    return (int(r), int(c), int(color))


def _build_arrow_pts(start: tuple[int, int], dr: int, dc: int) -> list[tuple[int, int]]:
    sr, sc = start
    mr, mc = sr - dr, sc - dc
    return [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]


def _max_steps_to_edge(r: int, c: int, dr: int, dc: int, H: int, W: int) -> int:
    return min((H - 1 - r) if dr == 1 else r, (W - 1 - c) if dc == 1 else c)


def _straight_path(start: tuple[int, int], direction: tuple[int, int], H: int, W: int) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    path: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < H and 0 <= c < W):
            break
        path.append((int(r), int(c)))
    return path


def _extract_scene(
    g: np.ndarray,
) -> tuple[int, list[dict[str, object]], set[tuple[int, int]], dict[tuple[int, int], int]]:
    H, W = g.shape
    bg = int(get_bg(g))
    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[dict[str, object]] = []

    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if not _is_arrow_pts(pts):
                continue
            info = _arrow_dir_and_start(pts)
            if info is None:
                continue
            direction, start = info
            arrow_cells |= set(pts)
            arrows.append(
                {
                    "color": int(col),
                    "pts": pts,
                    "dir": (int(direction[0]), int(direction[1])),
                    "start": (int(start[0]), int(start[1])),
                    "dist_key": _distance_order_key(start, H, W, int(col)),
                    "row_key": _row_major_order_key(start, int(col)),
                    "dist2": _center_dist2(int(start[0]), int(start[1]), H, W),
                }
            )

    static_walls = {
        (int(r), int(c)): int(g[r, c])
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }
    return bg, arrows, arrow_cells, static_walls


def _simulate(
    grid: np.ndarray,
    *,
    order_mode: str,
    stop_on_painted: bool,
    collect_meta: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, arrows, arrow_cells, static_walls = _extract_scene(g)
    out = g.copy()
    blocked_cells: set[tuple[int, int]] = set()

    if order_mode == "distance":
        arrows_sorted = sorted(arrows, key=lambda a: a["dist_key"])
    elif order_mode == "row_major":
        arrows_sorted = sorted(arrows, key=lambda a: a["row_key"])
    else:
        raise ValueError(f"Unknown order_mode: {order_mode}")

    ray_records: list[dict[str, object]] = []

    for arrow in arrows_sorted:
        start_col = int(arrow["color"])
        dr, dc = arrow["dir"]
        cur = tuple(arrow["start"])
        cur_col = int(start_col)
        seen: set[tuple[tuple[int, int], int, int, int]] = set()
        local_painted: set[tuple[int, int]] = set()
        stop_reason = "exit"

        while True:
            state = (cur, int(dr), int(dc), int(cur_col))
            if state in seen:
                stop_reason = "loop"
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                stop_reason = "exit"
                break

            nxt = (nr, nc)
            if nxt in static_walls:
                wcol = int(static_walls[nxt])
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                stop_reason = "corner_bounce"
                continue

            if nxt in arrow_cells:
                stop_reason = "arrow"
                break

            if stop_on_painted and nxt in blocked_cells:
                stop_reason = "painted_collision"
                break

            cur = nxt
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)
            local_painted.add(cur)
            stop_reason = "exit"

            side = (nr + 1, nc) if int(dr) == 1 else (nr - 1, nc)
            if side in static_walls:
                wcol = int(static_walls[side])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)
                stop_reason = "side_bounce"

            side = (nr, nc + 1) if int(dc) == 1 else (nr, nc - 1)
            if side in static_walls:
                wcol = int(static_walls[side])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)
                stop_reason = "side_bounce"

        blocked_cells |= local_painted
        ray_records.append(
            {
                "start": tuple(arrow["start"]),
                "dir": tuple(arrow["dir"]),
                "color": int(start_col),
                "dist2": int(arrow["dist2"]),
                "painted_count": len(local_painted),
                "stop_reason": stop_reason,
            }
        )

    meta = {
        "ray_records": ray_records,
        "blocked_count": len(blocked_cells),
        "arrow_count": len(arrows_sorted),
    }
    if collect_meta:
        return out, meta
    return out, {}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire nearer-to-center arrows first, and stop later rays on occupied ray cells.

    Static walls keep the original reflection and color-inheritance behavior.
    Dynamic ray-ray interaction is stricter: a later ray terminates when its next
    diagonal step would enter a cell already painted by an earlier ray.
    """
    return _simulate(grid, order_mode="distance", stop_on_painted=True)[0]


def _mandatory_pair_distance_order(
    H: int,
    W: int,
    color_a: int,
    color_b: int,
) -> tuple[list[dict[str, object]], set[tuple[int, int]], set[int], tuple[int, int], int] | None:
    for _ in range(160):
        case = _rng.choice([0, 1])
        cr = _rng.randint(6, H - 7)
        cc = _rng.randint(6, W - 7)
        ta = _rng.randint(2, 4)
        tb = _rng.randint(5, 7)

        if case == 0:
            start_a = (cr + ta, cc - ta)
            dir_a = (-1, 1)
            start_b = (cr - tb, cc - tb)
            dir_b = (1, 1)
            tail_a = min(cr, W - 1 - cc)
            tail_b = min(H - 1 - cr, W - 1 - cc)
        else:
            start_a = (cr + ta, cc + ta)
            dir_a = (-1, -1)
            start_b = (cr - tb, cc + tb)
            dir_b = (1, -1)
            tail_a = min(cr, cc)
            tail_b = min(H - 1 - cr, cc)

        if tail_a < 4 or tail_b < 4:
            continue

        arrow_a = _build_arrow_pts(start_a, *dir_a)
        arrow_b = _build_arrow_pts(start_b, *dir_b)
        if any(not (0 <= r < H and 0 <= c < W) for r, c in arrow_a + arrow_b):
            continue

        path_a = _straight_path(start_a, dir_a, H, W)
        path_b = _straight_path(start_b, dir_b, H, W)
        crossing = (cr, cc)
        if crossing not in path_a or crossing not in path_b:
            continue

        idx_a = path_a.index(crossing)
        idx_b = path_b.index(crossing)
        if idx_a < 1 or idx_b < 2:
            continue
        if len(path_a) - idx_a - 1 < 4 or len(path_b) - idx_b - 1 < 4:
            continue

        set_a = set(path_a)
        set_b = set(path_b)
        if set_a & set_b != {crossing}:
            continue

        cells_a = set(arrow_a)
        cells_b = set(arrow_b)
        if cells_a & cells_b:
            continue
        if (cells_a | cells_b) & (set_a | set_b):
            continue

        dist_a = _center_dist2(int(start_a[0]), int(start_a[1]), H, W)
        dist_b = _center_dist2(int(start_b[0]), int(start_b[1]), H, W)
        if dist_a >= dist_b:
            continue
        if start_b >= start_a:
            continue

        arrows = [
            {
                "start": start_a,
                "dir": dir_a,
                "color": int(color_a),
                "pts": arrow_a,
                "ray_cells": [(int(r), int(c), int(color_a)) for r, c in path_a],
                "dist2": int(dist_a),
            },
            {
                "start": start_b,
                "dir": dir_b,
                "color": int(color_b),
                "pts": arrow_b,
                "ray_cells": [(int(r), int(c), int(color_b)) for r, c in path_b],
                "dist2": int(dist_b),
            },
        ]
        reserved = set_a | set_b
        used_dist2 = {int(dist_a), int(dist_b)}
        return arrows, reserved, used_dist2, start_a, int(dist_b)
    return None


def _plan_extra_arrow(
    H: int,
    W: int,
    color: int,
    wall_palette: list[int],
    occupied_input: set[tuple[int, int]],
    occupied_output: set[tuple[int, int]],
    min_start: tuple[int, int],
    min_dist2: int,
    used_dist2: set[int],
) -> tuple[dict[str, object], set[tuple[int, int]], set[tuple[int, int]], int] | None:
    if not wall_palette:
        return None

    for _ in range(200):
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        n_bounces = _rng.choices([0, 1, 2], weights=[2, 3, 2])[0]

        r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 2)
        c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 2)
        if r_lo > r_hi or c_lo > c_hi:
            continue

        sr = _rng.randint(r_lo, r_hi)
        sc = _rng.randint(c_lo, c_hi)
        start = (int(sr), int(sc))
        dist2 = _center_dist2(int(sr), int(sc), H, W)
        if start <= min_start:
            continue
        if dist2 <= min_dist2 + 6 or dist2 in used_dist2:
            continue

        arrow_pts = _build_arrow_pts(start, dr, dc)
        arrow_set = set(arrow_pts)
        if any(not (0 <= r < H and 0 <= c < W) for r, c in arrow_pts):
            continue
        if any((r, c) in occupied_input or (r, c) in occupied_output for r, c in arrow_pts):
            continue

        cur_r, cur_c = int(sr), int(sc)
        cur_dr, cur_dc = int(dr), int(dc)
        cur_col = int(color)
        ray_cells: list[tuple[int, int, int]] = []
        ray_set: set[tuple[int, int]] = set()
        walls: list[tuple[list[tuple[int, int]], int]] = []
        local_input = set(arrow_set)
        local_output = set(arrow_set)
        ok = True
        bounces_left = int(n_bounces)

        for seg_idx in range(n_bounces + 1):
            max_steps = _max_steps_to_edge(cur_r, cur_c, cur_dr, cur_dc, H, W)
            if seg_idx < n_bounces and bounces_left > 0:
                if max_steps < 4:
                    ok = False
                    break
                seg_len = _rng.randint(3, max(3, max_steps - 1))
            else:
                seg_len = max_steps

            for _step in range(seg_len):
                nr, nc = cur_r + cur_dr, cur_c + cur_dc
                if not (0 <= nr < H and 0 <= nc < W):
                    break
                pos = (int(nr), int(nc))
                if pos in occupied_output or pos in ray_set or pos in arrow_set:
                    ok = False
                    break
                ray_cells.append((int(nr), int(nc), int(cur_col)))
                ray_set.add(pos)
                local_output.add(pos)
                cur_r, cur_c = int(nr), int(nc)
            if not ok:
                break

            if seg_idx >= n_bounces or bounces_left <= 0:
                continue
            bounces_left -= 1

            options: list[tuple[list[tuple[int, int]], str]] = []
            wr, wc = cur_r + cur_dr, cur_c
            if 0 <= wr < H:
                w3 = [(int(wr), int(wc + d)) for d in (-1, 0, 1)]
                if all(0 <= c < W for _, c in w3):
                    options.append((w3, "dr"))
                options.append(([(int(wr), int(wc))], "dr"))

            wr2, wc2 = cur_r, cur_c + cur_dc
            if 0 <= wc2 < W:
                w3v = [(int(wr2 + d), int(wc2)) for d in (-1, 0, 1)]
                if all(0 <= r < H for r, _ in w3v):
                    options.append((w3v, "dc"))
                options.append(([(int(wr2), int(wc2))], "dc"))

            if not options:
                ok = False
                break

            _rng.shuffle(options)
            options.sort(key=lambda item: len(item[0]), reverse=_rng.random() < 0.6)
            chosen = None
            for wpxs, flip in options:
                if any(
                    (r, c) in occupied_input
                    or (r, c) in occupied_output
                    or (r, c) in ray_set
                    or (r, c) in arrow_set
                    or (r, c) in local_input
                    for r, c in wpxs
                ):
                    continue
                chosen = (wpxs, flip)
                break
            if chosen is None:
                ok = False
                break

            wpxs, flip = chosen
            wcol = int(_rng.choice(wall_palette))
            walls.append((wpxs, int(wcol)))
            for cell in wpxs:
                local_input.add(cell)
                local_output.add(cell)

            if ray_cells:
                lr, lc, _ = ray_cells[-1]
                ray_cells[-1] = (int(lr), int(lc), int(wcol))
            cur_col = int(wcol)
            if flip == "dr":
                cur_dr = -int(cur_dr)
            else:
                cur_dc = -int(cur_dc)

        if not ok or len(ray_cells) < 3:
            continue

        return (
            {
                "start": start,
                "dir": (int(dr), int(dc)),
                "color": int(color),
                "pts": arrow_pts,
                "ray_cells": ray_cells,
                "walls": walls,
            },
            local_input,
            local_output,
            int(dist2),
        )
    return None


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    distance_out, distance_meta = _simulate(grid, order_mode="distance", stop_on_painted=True, collect_meta=True)
    no_collision_out = _simulate(grid, order_mode="distance", stop_on_painted=False)[0]
    row_major_out, row_major_meta = _simulate(grid, order_mode="row_major", stop_on_painted=True, collect_meta=True)

    collision_count = sum(1 for rec in distance_meta["ray_records"] if rec["stop_reason"] == "painted_collision")
    changed_vs_no_collision = int(np.count_nonzero(distance_out != no_collision_out))
    changed_vs_row_major = int(np.count_nonzero(distance_out != row_major_out))

    distance_order = [tuple(rec["start"]) for rec in distance_meta["ray_records"]]
    row_major_order = [tuple(rec["start"]) for rec in row_major_meta["ray_records"]]

    dist_values = [int(rec["dist2"]) for rec in distance_meta["ray_records"]]
    return {
        "collision_count": int(collision_count),
        "changed_vs_no_collision": int(changed_vs_no_collision),
        "changed_vs_row_major": int(changed_vs_row_major),
        "order_differs_from_row_major": int(distance_order != row_major_order),
        "unique_source_distances": int(len(dist_values) == len(set(dist_values))),
        "arrow_count": int(distance_meta["arrow_count"]),
    }


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:3]
    wall_palette = palette[3:] if len(palette) > 3 else [9]

    pair = _mandatory_pair_distance_order(H, W, int(arrow_colors[0]), int(arrow_colors[1]))
    if pair is None:
        return None
    arrows, reserved_paths, used_dist2, min_start, min_dist2 = pair

    grid = np.zeros((H, W), dtype=int)
    occupied_input: set[tuple[int, int]] = set()
    occupied_output: set[tuple[int, int]] = set(reserved_paths)

    for arrow in arrows:
        for r, c in arrow["pts"]:
            if (int(r), int(c)) in occupied_input:
                return None
            grid[int(r), int(c)] = int(arrow["color"])
            occupied_input.add((int(r), int(c)))
            occupied_output.add((int(r), int(c)))

    if _rng.random() < 0.65:
        extra = _plan_extra_arrow(
            H=H,
            W=W,
            color=int(arrow_colors[2]),
            wall_palette=wall_palette,
            occupied_input=occupied_input,
            occupied_output=occupied_output,
            min_start=min_start,
            min_dist2=min_dist2,
            used_dist2=used_dist2,
        )
        if extra is not None:
            arrow, local_input, local_output, dist2 = extra
            for r, c in arrow["pts"]:
                grid[int(r), int(c)] = int(arrow["color"])
            for wpxs, wcol in arrow.get("walls", []):
                for r, c in wpxs:
                    grid[int(r), int(c)] = int(wcol)
            occupied_input |= {(int(r), int(c)) for r, c in local_input}
            occupied_output |= {(int(r), int(c)) for r, c in local_output}
            used_dist2.add(int(dist2))

    metrics = _variant_metrics(grid)
    if metrics["collision_count"] < 1:
        return None
    if metrics["changed_vs_no_collision"] < 4:
        return None
    if metrics["changed_vs_row_major"] < 4:
        return None
    if metrics["order_differs_from_row_major"] < 1:
        return None
    if metrics["unique_source_distances"] < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
