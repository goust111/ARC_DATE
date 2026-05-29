# Combo variant of 142ca369 (index 19)
# combo_spec: D8/0+D10/0+D20/2
# Name: 距离排序+墙壁消失+先到先得

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

# Design analysis
# fused_spec:
#   concrete_trace:
#     - Keep the original objects: 3-cell L arrows and 1/3-pixel axis-aligned walls.
#     - Change 1 (D8): arrows do not fire independently. Sort them by squared distance from
#       the ray source (the arrow's right-angle vertex) to the grid center, nearest first.
#     - Change 2 (D10): only static walls cause reflections, but a touched wall pixel is erased
#       from the output and removed from the live wall set for all later simulation.
#     - Change 3 (D20): finished ray tracks become a growing trail set. A later ray does not
#       bounce on that set; it simply stops before entering an already-painted trail cell.
#     - Example: arrow A near the center goes first, side-bounces off a teal wall at (7,8),
#       so output erases that wall pixel and A's later segment turns teal. Arrow B is farther
#       from the center but visually higher/earlier in reading order. B aims toward one teal
#       cell of A's finished trail; under the combo rule B halts one step before that cell.
#       Original 142ca369 would keep the wall visible, would not erase the touched pixel,
#       and B would continue through A's future track because rays were independent.
#   complexity_calibration:
#     - Use exactly 2 arrows so the order cue is legible and the human can compare “near center”
#       against “farther but earlier on the page”.
#     - Force one static-wall bounce on the center-near arrow so wall erasure is always visible.
#     - Force one later trail collision on the farther arrow so the first-come-first-served rule
#       is always visible.
#     - Prefer a 3-pixel wall for the mandatory bounce so the erased center leaves a visible gap.
#   implicit_assumption_audit:
#     - Original `for start_col, pts in arrows:` assumed cross-arrow independence. This is invalid;
#       solver must sort arrows by center distance before tracing.
#     - Original `wall_cells` was static. This is invalid because touched wall pixels disappear.
#       The solver therefore keeps a mutable `live_walls` dict.
#     - Original `if out[nr, nc] == bg: out[nr, nc] = cur_col` was purely rendering. This is no
#       longer enough because later rays must stop before entering earlier trail cells, so the
#       solver checks `settled_trail` before advancing.
#     - Original bounce recolored the ray cell at the reflection point. This is invalid for D10:
#       the ray keeps its pre-bounce cell, while the wall pixel is erased and only subsequent
#       cells inherit the wall color.
#     - Original loop state omitted the wall configuration. This is invalid once walls disappear,
#       so `wall_sig` is part of the visited state.
#     - Multi-change interaction: trails are dynamic blockers but not dynamic reflectors. Only
#       input walls may flip direction; prior trails only terminate a later ray.
#   placement_legality:
#     - Generator plans the near-center arrow output-first: one bounce, one wall, then an exit
#       segment. Its arrow cells, wall pixels, and full output trail are all disjoint.
#     - Generator then back-solves a second arrow whose straight path would next enter one chosen
#       cell from the first trail. The second arrow stops there, so its own painted cells remain
#       disjoint from the first trail.
#     - The second arrow is constrained to be lexicographically earlier than the first source but
#       farther from the center, making the distance sort observable and not confusable with a
#       simple top-left order.
#   visual_observability:
#     - Across examples, input shows a near-center arrow plus a farther arrow. Output shows a wall
#       gap on the near-center arrow's bounce and a truncated far-arrow tail near the first trail.
#     - Because the farther arrow is often visually earlier on the page, humans can infer that the
#       controlling order is not scan order but center distance.
#   cleanliness:
#     - Only one deliberate wall interaction and one deliberate trail collision are required.
#       This keeps the grid sparse and avoids ambiguous multi-layer overlaps.
#   color_inference:
#     - Wall color differs from arrow colors so the post-bounce color change is structurally
#       explained by the nearby erased wall. No hidden color ranking is used.
#   new_solver_assumptions_and_generator_guards:
#     - Solver assumes arrow source distances are orderable; generator rejects center-distance ties.
#     - Solver assumes later rays do not need to use erased wall pixels; generator keeps the second
#       arrow away from the mandatory wall region.
#     - Solver assumes trail collisions are enter-cell collisions, not side-touch bounces; generator
#       makes the second arrow approach a chosen trail cell directly and avoids other wall adjacencies.


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


def _center_order_key(start: tuple[int, int], H: int, W: int, color: int) -> tuple[int, int, int, int]:
    r, c = start
    return (_center_dist2(r, c, H, W), int(r), int(c), int(color))


def _source_order_key(start: tuple[int, int], color: int) -> tuple[int, int, int]:
    return (int(start[0]), int(start[1]), int(color))


def _arrow_points_from_start(sr: int, sc: int, dr: int, dc: int) -> list[tuple[int, int]]:
    mr, mc = sr - dr, sc - dc
    return [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]


def _max_steps_to_edge(r: int, c: int, dr: int, dc: int, H: int, W: int) -> int:
    return min((H - 1 - r) if dr == 1 else r, (W - 1 - c) if dc == 1 else c)


def _detect_scene(g: np.ndarray) -> tuple[int, list[dict[str, object]], set[tuple[int, int]], dict[tuple[int, int], int]]:
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
                    "center_key": _center_order_key(start, H, W, int(col)),
                    "source_key": _source_order_key(start, int(col)),
                }
            )

    live_walls = {
        (int(r), int(c)): int(g[r, c])
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }
    return bg, arrows, arrow_cells, live_walls


def _simulate(
    grid: np.ndarray,
    *,
    order_mode: str = "center",
    stop_on_trail: bool = True,
    collect_meta: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, arrows, arrow_cells, live_walls = _detect_scene(g)
    out = g.copy()
    settled_trail: set[tuple[int, int]] = set()

    if order_mode == "center":
        arrows = sorted(arrows, key=lambda a: a["center_key"])
    elif order_mode == "source":
        arrows = sorted(arrows, key=lambda a: a["source_key"])
    else:
        raise ValueError(f"Unknown order_mode: {order_mode}")

    erased_walls = 0
    per_arrow: list[dict[str, object]] = []

    def erase_wall(cell: tuple[int, int]) -> int:
        nonlocal erased_walls
        erased_walls += 1
        wcol = int(live_walls.pop(cell))
        out[cell[0], cell[1]] = int(bg)
        return wcol

    for arrow in arrows:
        cur = tuple(arrow["start"])
        dr, dc = map(int, arrow["dir"])
        cur_col = int(arrow["color"])
        seen: set[tuple[tuple[int, int], int, int, int, tuple[tuple[int, int], ...]]] = set()
        local_trail: list[tuple[int, int]] = []
        local_trail_set: set[tuple[int, int]] = set()
        wall_hits = 0
        stop_reason = "exit"

        while True:
            wall_sig = tuple(sorted(live_walls))
            state = (cur, int(dr), int(dc), int(cur_col), wall_sig)
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
            if nxt in arrow_cells:
                stop_reason = "arrow"
                break
            if stop_on_trail and nxt in settled_trail:
                stop_reason = "trail_collision"
                break
            if nxt in live_walls:
                cur_col = erase_wall(nxt)
                wall_hits += 1
                dr, dc = -int(dr), -int(dc)
                stop_reason = "corner_bounce"
                continue

            cur = nxt
            if cur not in local_trail_set:
                local_trail.append(cur)
                local_trail_set.add(cur)
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)
            stop_reason = "exit"

            side = (nr + 1, nc) if dr == 1 else (nr - 1, nc)
            if side in live_walls:
                cur_col = erase_wall(side)
                wall_hits += 1
                dr = -int(dr)
                stop_reason = "side_bounce"

            side = (nr, nc + 1) if dc == 1 else (nr, nc - 1)
            if side in live_walls:
                cur_col = erase_wall(side)
                wall_hits += 1
                dc = -int(dc)
                stop_reason = "side_bounce"

        settled_trail |= local_trail_set
        per_arrow.append(
            {
                "start": tuple(arrow["start"]),
                "color": int(arrow["color"]),
                "painted": len(local_trail_set),
                "wall_hits": int(wall_hits),
                "stop_reason": stop_reason,
                "center_key": arrow["center_key"],
                "source_key": arrow["source_key"],
            }
        )

    meta = {
        "arrow_count": len(arrows),
        "erased_walls": int(erased_walls),
        "trail_collisions": int(sum(1 for rec in per_arrow if rec["stop_reason"] == "trail_collision")),
        "per_arrow": per_arrow,
    }
    if collect_meta:
        return out, meta
    return out, {}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire arrows from nearest-to-center to farthest.

    Static walls reflect and then disappear. Finished trails persist only as
    enter-cell blockers, so later rays stop on first contact with earlier paint.
    """
    return _simulate(grid, order_mode="center", stop_on_trail=True)[0]


def _plan_primary_arrow(
    H: int,
    W: int,
    color: int,
    wall_color: int,
) -> dict[str, object] | None:
    for _ in range(500):
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
        c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
        if r_lo > r_hi or c_lo > c_hi:
            continue

        sr = _rng.randint(r_lo, r_hi)
        sc = _rng.randint(c_lo, c_hi)
        dist2 = _center_dist2(sr, sc, H, W)
        if not (8 <= dist2 <= 120):
            continue

        arrow_pts = _arrow_points_from_start(sr, sc, dr, dc)
        arrow_set = set(arrow_pts)
        if any(not (0 <= r < H and 0 <= c < W) for r, c in arrow_pts):
            continue

        max1 = _max_steps_to_edge(sr, sc, dr, dc, H, W)
        if max1 < 8:
            continue
        seg1_len = _rng.randint(3, max1 - 4)

        cur_r, cur_c = int(sr), int(sc)
        ray1: list[tuple[int, int, int]] = []
        ray1_set: set[tuple[int, int]] = set()
        ok = True
        for _step in range(seg1_len):
            nr, nc = cur_r + dr, cur_c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                ok = False
                break
            if (nr, nc) in arrow_set or (nr, nc) in ray1_set:
                ok = False
                break
            ray1.append((int(nr), int(nc), int(color)))
            ray1_set.add((int(nr), int(nc)))
            cur_r, cur_c = int(nr), int(nc)
        if not ok:
            continue

        wall_options: list[tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]] = []

        wr, wc = cur_r + dr, cur_c
        if 0 <= wr < H:
            touched = (int(wr), int(wc))
            w3 = [(int(wr), int(wc + d)) for d in (-1, 0, 1)]
            if all(0 <= c < W for _, c in w3):
                wall_options.append((w3, touched, (-int(dr), int(dc))))
            wall_options.append(([(int(wr), int(wc))], touched, (-int(dr), int(dc))))

        wr2, wc2 = cur_r, cur_c + dc
        if 0 <= wc2 < W:
            touched = (int(wr2), int(wc2))
            w3v = [(int(wr2 + d), int(wc2)) for d in (-1, 0, 1)]
            if all(0 <= r < H for r, _ in w3v):
                wall_options.append((w3v, touched, (int(dr), -int(dc))))
            wall_options.append(([(int(wr2), int(wc2))], touched, (int(dr), -int(dc))))

        _rng.shuffle(wall_options)
        wall_options.sort(key=lambda item: len(item[0]), reverse=True)

        for wall_cells, touched, next_dir in wall_options:
            wall_set = {(int(r), int(c)) for r, c in wall_cells}
            if wall_set & arrow_set:
                continue
            if wall_set & ray1_set:
                continue

            ndr, ndc = next_dir
            max2 = _max_steps_to_edge(cur_r, cur_c, ndr, ndc, H, W)
            if max2 < 5:
                continue

            ray2: list[tuple[int, int, int]] = []
            ray2_set: set[tuple[int, int]] = set()
            rr, cc = int(cur_r), int(cur_c)
            valid = True
            for _step in range(max2):
                nr, nc = rr + ndr, cc + ndc
                if not (0 <= nr < H and 0 <= nc < W):
                    valid = False
                    break
                if (nr, nc) in arrow_set or (nr, nc) in ray1_set or (nr, nc) in wall_set or (nr, nc) in ray2_set:
                    valid = False
                    break
                ray2.append((int(nr), int(nc), int(wall_color)))
                ray2_set.add((int(nr), int(nc)))
                rr, cc = int(nr), int(nc)
            if not valid or len(ray2) < 5:
                continue

            return {
                "color": int(color),
                "start": (int(sr), int(sc)),
                "dir": (int(dr), int(dc)),
                "arrow_pts": arrow_pts,
                "walls": [(wall_cells, int(wall_color), touched)],
                "ray_cells": ray1 + ray2,
                "post_bounce_cells": [(int(r), int(c)) for r, c, _col in ray2],
                "ray_positions": {(int(r), int(c)) for r, c, _col in ray1 + ray2},
                "center_key": _center_order_key((sr, sc), H, W, color),
            }
    return None


def _plan_collision_arrow(
    H: int,
    W: int,
    color: int,
    primary: dict[str, object],
) -> dict[str, object] | None:
    primary_start = tuple(primary["start"])
    primary_dist2 = int(primary["center_key"][0])
    primary_ray = set(primary["ray_positions"])
    primary_input = set(primary["arrow_pts"])
    for wall_cells, _wall_color, _touched in primary["walls"]:
        primary_input.update((int(r), int(c)) for r, c in wall_cells)

    candidates = list(primary["post_bounce_cells"])
    _rng.shuffle(candidates)
    diagonal_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for collision in candidates:
        cr, cc = collision
        if min(cr, cc, H - 1 - cr, W - 1 - cc) < 3:
            continue

        dirs = diagonal_dirs[:]
        _rng.shuffle(dirs)
        for dr, dc in dirs:
            back_limit = _max_steps_to_edge(cr, cc, -dr, -dc, H, W)
            if back_limit < 4:
                continue

            lengths = list(range(4, min(9, back_limit) + 1))
            _rng.shuffle(lengths)
            for steps_to_collision in lengths:
                sr = cr - steps_to_collision * dr
                sc = cc - steps_to_collision * dc
                if not (0 <= sr < H and 0 <= sc < W):
                    continue
                if _center_dist2(sr, sc, H, W) <= primary_dist2:
                    continue
                if (int(sr), int(sc)) >= primary_start:
                    continue

                arrow_pts = _arrow_points_from_start(sr, sc, dr, dc)
                arrow_set = set(arrow_pts)
                if any(not (0 <= r < H and 0 <= c < W) for r, c in arrow_pts):
                    continue
                if arrow_set & primary_input:
                    continue
                if arrow_set & primary_ray:
                    continue

                path: list[tuple[int, int, int]] = []
                path_set: set[tuple[int, int]] = set()
                ok = True
                cur_r, cur_c = int(sr), int(sc)
                for _step in range(steps_to_collision - 1):
                    nr, nc = cur_r + dr, cur_c + dc
                    pos = (int(nr), int(nc))
                    if pos in arrow_set or pos in primary_input or pos in primary_ray or pos in path_set:
                        ok = False
                        break
                    side_r = (int(nr + dr), int(nc))
                    side_c = (int(nr), int(nc + dc))
                    if side_r in primary_input or side_c in primary_input:
                        ok = False
                        break
                    path.append((int(nr), int(nc), int(color)))
                    path_set.add(pos)
                    cur_r, cur_c = int(nr), int(nc)
                if not ok or len(path) < 3:
                    continue

                if (cur_r + dr, cur_c + dc) != (cr, cc):
                    continue

                return {
                    "color": int(color),
                    "start": (int(sr), int(sc)),
                    "dir": (int(dr), int(dc)),
                    "arrow_pts": arrow_pts,
                    "ray_cells": path,
                    "collision_target": (int(cr), int(cc)),
                    "center_key": _center_order_key((sr, sc), H, W, color),
                }
    return None


def _build_planned_output(
    H: int,
    W: int,
    primary: dict[str, object],
    secondary: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)

    for placement in (primary, secondary):
        for r, c in placement["arrow_pts"]:
            grid[r, c] = int(placement["color"])
            out[r, c] = int(placement["color"])

    for r, c, col in primary["ray_cells"]:
        out[r, c] = int(col)
    for r, c, col in secondary["ray_cells"]:
        out[r, c] = int(col)

    for wall_cells, wall_color, touched in primary["walls"]:
        for r, c in wall_cells:
            grid[r, c] = int(wall_color)
            if (int(r), int(c)) != touched:
                out[r, c] = int(wall_color)

    return grid, out


def _combo_metrics(grid: np.ndarray) -> dict[str, int]:
    combo_out, combo_meta = _simulate(grid, order_mode="center", stop_on_trail=True, collect_meta=True)
    no_stop_out, _ = _simulate(grid, order_mode="center", stop_on_trail=False, collect_meta=False)
    source_order_out, _ = _simulate(grid, order_mode="source", stop_on_trail=True, collect_meta=False)

    return {
        "erased_walls": int(combo_meta["erased_walls"]),
        "trail_collisions": int(combo_meta["trail_collisions"]),
        "changed_vs_no_stop": int(np.count_nonzero(combo_out != no_stop_out)),
        "changed_vs_source_order": int(np.count_nonzero(combo_out != source_order_out)),
    }


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_a, arrow_b, wall_color = map(int, palette[:3])

    primary = _plan_primary_arrow(H, W, arrow_a, wall_color)
    if primary is None:
        return None

    secondary = _plan_collision_arrow(H, W, arrow_b, primary)
    if secondary is None:
        return None

    if int(primary["center_key"][0]) >= int(secondary["center_key"][0]):
        return None
    if tuple(secondary["start"]) >= tuple(primary["start"]):
        return None

    grid, planned_out = _build_planned_output(H, W, primary, secondary)
    solved = main(grid)
    if not np.array_equal(solved, planned_out):
        return None

    metrics = _combo_metrics(grid)
    if metrics["erased_walls"] < 1:
        return None
    if metrics["trail_collisions"] < 1:
        return None
    if metrics["changed_vs_no_stop"] < 3:
        return None
    if metrics["changed_vs_source_order"] < 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
