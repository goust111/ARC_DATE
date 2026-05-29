# Variant of 142ca369
# Dimension: D8 — 多步处理与射线顺序 (multi_step)
# Change: multi_bounce——每条射线独立地通过所有可达墙壁逐个反射，多条射线之间无顺序依赖... → 射线按距离排序处理

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

# Design analysis
# spec:
#   reference_order:
#     point: the arrow's right-angle vertex (the actual ray origin visible in the L-shape)
#     score: squared Euclidean distance to the grid center, computed on doubled coordinates:
#            (2*r-(H-1))^2 + (2*c-(W-1))^2
#     tiebreak: row, then col, then color; generation avoids ties so the visual cue is "closer to center fires first"
#   dynamic_rule:
#     original: all rays only bounce off input walls; ray-renders do not affect one another
#     variant: after each ray is finished, every painted ray cell becomes a dynamic wall for later rays only
#     effect: a later ray may side-bounce or corner-bounce off an earlier ray track, adopts that track cell's color,
#             and never paints through it
#   visualized_objects:
#     arrows: unchanged 3-cell L-shapes in 2x2 boxes
#     static_walls: unchanged 1-cell or 3-cell axis-aligned mono-color segments
#     dynamic_walls: not present in input; they appear only implicitly because earlier output ray tracks block later rays
#   concrete_flow_example:
#     original:
#       - arrow A and arrow B are both present in the input
#       - even if B's diagonal path reaches a cell next to A's future ray, nothing happens, because A's ray is not in the input
#     variant:
#       - if A is closer to the center, fire A first and materialize its ray
#       - then B fires; when B reaches a cell adjacent to A's painted track, that painted track behaves exactly like a wall
#       - B flips one axis, recolors the bounce cell with A's track color, and continues
#     difference:
#       - the same input now yields an output where B bends around A's already-painted line instead of passing through empty bg
#   complexity_calibration:
#     - keep the original arrow/wall vocabulary unchanged so the task still reads like "ray tracing with reflections"
#     - force at least two arrows and at least one dynamic interaction, otherwise D8 would be invisible
#     - keep only 2-3 arrows in generation so the order cue stays legible instead of becoming cluttered
#   implicit_assumption_audit:
#     - original `for start_col, pts in arrows:` assumed arrow order does not matter; this is invalid now and must be replaced by sorting
#     - original `wall_cells` was static; this is invalid now because later rays need a larger blocker set including earlier ray pixels
#     - original `if out[nr, nc] == bg: out[nr, nc] = cur_col` was harmless because rays never interacted;
#       now later rays must not overwrite earlier rays, so the solver treats earlier ray cells as blockers before entering them
#     - original side-bounce checks only looked at static walls; they now need to read from "static walls + prior finished rays"
#   placement_legality:
#     - generator keeps a global occupied set over arrows, static walls, and already-planned output ray cells
#     - later arrows may be adjacent to earlier ray cells (to bounce) but may never overlap them
#     - generator validates both `main(input) == planned_output` and ordered-vs-independent output inequality
#   observability:
#     - the center-near arrow is generated first and usually leaves a long visible track
#     - a farther arrow then bends at that track, so the input/output pair exposes that a non-input obstacle appeared in sequence
#   cleanliness_and_color:
#     - dynamic reflections borrow the earlier ray's visible color, giving a structural cue for why the later ray changed color there
#     - generation limits arrow count and spaces start positions by center-distance band to avoid dense ambiguous crossings
#   new_solver_assumptions_and_generator_guards:
#     - solver assumes arrow start distances are orderable; generator uses distinct center-distance scores
#     - solver assumes later rays never need to overwrite prior ray cells; generator blocks overlap and only allows adjacency-based hits
#     - solver assumes dynamic blockers stay fixed during one ray; generator matches that by only using already-finished rays as blockers


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


def _order_key(start: tuple[int, int], H: int, W: int, color: int) -> tuple[int, int, int, int]:
    r, c = start
    return (_center_dist2(r, c, H, W), int(r), int(c), int(color))


def _arrow_points_from_start(sr: int, sc: int, dr: int, dc: int) -> list[tuple[int, int]]:
    mr, mc = sr - dr, sc - dc
    return [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]


def _max_steps_to_edge(r: int, c: int, dr: int, dc: int, H: int, W: int) -> int:
    return min((H - 1 - r) if dr == 1 else r, (W - 1 - c) if dc == 1 else c)


def _trace_single_ray(
    H: int,
    W: int,
    start_col: int,
    start: tuple[int, int],
    direction: tuple[int, int],
    arrow_cells: set[tuple[int, int]],
    static_walls: dict[tuple[int, int], int],
    prior_rays: dict[tuple[int, int], int],
) -> dict[str, object]:
    cur = (int(start[0]), int(start[1]))
    dr, dc = int(direction[0]), int(direction[1])
    cur_col = int(start_col)

    seen: set[tuple[tuple[int, int], int, int, int]] = set()
    trail_order: list[tuple[int, int]] = []
    trail_colors: dict[tuple[int, int], int] = {}
    dynamic_hits = 0
    static_hits = 0
    dynamic_side_hits = 0
    dynamic_corner_hits = 0

    def is_blocker(pos: tuple[int, int]) -> bool:
        return pos in static_walls or pos in prior_rays

    def blocker_color(pos: tuple[int, int]) -> int:
        if pos in static_walls:
            return int(static_walls[pos])
        return int(prior_rays[pos])

    def blocker_is_dynamic(pos: tuple[int, int]) -> bool:
        return pos in prior_rays

    def mark(pos: tuple[int, int], color: int) -> None:
        if pos in arrow_cells or pos in static_walls or pos in prior_rays:
            return
        if pos not in trail_colors:
            trail_order.append(pos)
        trail_colors[pos] = int(color)

    while True:
        state = (cur, int(dr), int(dc), int(cur_col))
        if state in seen:
            break
        seen.add(state)

        nr = int(cur[0] + dr)
        nc = int(cur[1] + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            break

        nxt = (nr, nc)
        if is_blocker(nxt):
            wcol = blocker_color(nxt)
            if blocker_is_dynamic(nxt):
                dynamic_hits += 1
                dynamic_corner_hits += 1
            else:
                static_hits += 1
            mark(cur, wcol)
            cur_col = int(wcol)
            dr, dc = -int(dr), -int(dc)
            continue

        if nxt in arrow_cells:
            break

        cur = nxt
        mark(cur, cur_col)

        side = (nr + 1, nc) if dr == 1 else (nr - 1, nc)
        if is_blocker(side):
            wcol = blocker_color(side)
            if blocker_is_dynamic(side):
                dynamic_hits += 1
                dynamic_side_hits += 1
            else:
                static_hits += 1
            mark(cur, wcol)
            cur_col = int(wcol)
            dr = -int(dr)

        side = (nr, nc + 1) if dc == 1 else (nr, nc - 1)
        if is_blocker(side):
            wcol = blocker_color(side)
            if blocker_is_dynamic(side):
                dynamic_hits += 1
                dynamic_side_hits += 1
            else:
                static_hits += 1
            mark(cur, wcol)
            cur_col = int(wcol)
            dc = -int(dc)

    ray_cells = [(int(r), int(c), int(trail_colors[(r, c)])) for r, c in trail_order]
    return {
        "ray_cells": ray_cells,
        "dynamic_hits": int(dynamic_hits),
        "static_hits": int(static_hits),
        "dynamic_side_hits": int(dynamic_side_hits),
        "dynamic_corner_hits": int(dynamic_corner_hits),
    }


def _detect_arrows(g: np.ndarray, bg: int) -> tuple[list[dict[str, object]], set[tuple[int, int]]]:
    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[dict[str, object]] = []
    H, W = g.shape
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
                    "dir": direction,
                    "start": start,
                    "order_key": _order_key(start, H, W, int(col)),
                }
            )
    return arrows, arrow_cells


def _trace_all(grid: np.ndarray, ordered: bool) -> tuple[np.ndarray, dict[str, object]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrows, arrow_cells = _detect_arrows(g, bg)
    static_walls = {
        (int(r), int(c)): int(g[r, c])
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }

    arrows_sorted = sorted(arrows, key=lambda a: a["order_key"])
    out = g.copy()
    prior_rays: dict[tuple[int, int], int] = {}
    per_arrow: list[dict[str, object]] = []

    for arrow in arrows_sorted:
        ray_info = _trace_single_ray(
            H=H,
            W=W,
            start_col=int(arrow["color"]),
            start=arrow["start"],
            direction=arrow["dir"],
            arrow_cells=arrow_cells,
            static_walls=static_walls,
            prior_rays=prior_rays if ordered else {},
        )
        for r, c, col in ray_info["ray_cells"]:
            out[r, c] = int(col)
        if ordered:
            for r, c, col in ray_info["ray_cells"]:
                prior_rays[(int(r), int(c))] = int(col)
        per_arrow.append(
            {
                "color": int(arrow["color"]),
                "start": arrow["start"],
                "order_key": arrow["order_key"],
                **ray_info,
            }
        )

    meta = {
        "per_arrow": per_arrow,
        "dynamic_hits": int(sum(a["dynamic_hits"] for a in per_arrow)),
        "static_hits": int(sum(a["static_hits"] for a in per_arrow)),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Order arrows by distance from the grid center, then trace them sequentially.

    Static walls still reflect rays exactly as before. The new behavior is that every
    finished ray becomes a dynamic wall for later rays, so ray order changes geometry.
    """
    return _trace_all(grid, ordered=True)[0]


def _plan_static_arrow(
    H: int,
    W: int,
    color: int,
    occupied: set[tuple[int, int]],
    palette: list[int],
    dist_lo: int,
    dist_hi: int,
    require_bounce: bool,
) -> dict[str, object] | None:
    for _ in range(220):
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        n_bounces = _rng.choice([1, 2]) if require_bounce else _rng.choice([0, 1, 1, 2])

        r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
        c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
        if r_lo > r_hi or c_lo > c_hi:
            continue
        sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)
        dist2 = _center_dist2(sr, sc, H, W)
        if not (dist_lo <= dist2 <= dist_hi):
            continue

        arrow_pts = _arrow_points_from_start(sr, sc, dr, dc)
        if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
            continue

        cur_r, cur_c, cur_dr, cur_dc, cur_col = sr, sc, dr, dc, int(color)
        ray_cells: list[tuple[int, int, int]] = []
        walls: list[tuple[list[tuple[int, int]], int]] = []
        ray_set: set[tuple[int, int]] = set()
        arrow_set = set(arrow_pts)
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
                if (nr, nc) in occupied or (nr, nc) in ray_set or (nr, nc) in arrow_set:
                    ok = False
                    break
                ray_cells.append((int(nr), int(nc), int(cur_col)))
                ray_set.add((int(nr), int(nc)))
                cur_r, cur_c = nr, nc
            if not ok:
                break

            if seg_idx >= n_bounces or bounces_left <= 0:
                continue
            bounces_left -= 1

            options: list[tuple[list[tuple[int, int]], str]] = []
            wr, wc = cur_r + cur_dr, cur_c
            if 0 <= wr < H:
                w3 = [(wr, wc + d) for d in (-1, 0, 1)]
                if all(0 <= c < W for _, c in w3):
                    options.append((w3, "dr"))
                options.append(([(wr, wc)], "dr"))

            wr2, wc2 = cur_r, cur_c + cur_dc
            if 0 <= wc2 < W:
                w3v = [(wr2 + d, wc2) for d in (-1, 0, 1)]
                if all(0 <= r < H for r, _ in w3v):
                    options.append((w3v, "dc"))
                options.append(([(wr2, wc2)], "dc"))

            if not options:
                ok = False
                break

            _rng.shuffle(options)
            prefer_3 = _rng.random() < 0.6
            options.sort(key=lambda o: len(o[0]), reverse=prefer_3)
            chosen_wall = None
            for wpxs, flip in options:
                if any((r, c) in occupied or (r, c) in ray_set or (r, c) in arrow_set for r, c in wpxs):
                    continue
                chosen_wall = (wpxs, flip)
                break
            if chosen_wall is None:
                ok = False
                break

            wpxs, flip = chosen_wall
            wcol = int(_rng.choice(palette))
            walls.append((wpxs, wcol))
            if ray_cells:
                lr, lc, _ = ray_cells[-1]
                ray_cells[-1] = (int(lr), int(lc), int(wcol))
            cur_col = int(wcol)
            if flip == "dr":
                cur_dr = -cur_dr
            else:
                cur_dc = -cur_dc

        if not ok or len(ray_cells) < 5:
            continue

        new_cells = set(arrow_pts) | ray_set
        for wpxs, _ in walls:
            new_cells.update(wpxs)
        expected_count = len(arrow_pts) + len(ray_cells) + sum(len(w) for w, _ in walls)
        if len(new_cells) != expected_count or new_cells & occupied:
            continue

        return {
            "color": int(color),
            "start": (int(sr), int(sc)),
            "dir": (int(dr), int(dc)),
            "arrow_pts": arrow_pts,
            "ray_cells": ray_cells,
            "walls": walls,
            "dynamic_hits": 0,
            "order_key": _order_key((sr, sc), H, W, color),
        }
    return None


def _plan_dynamic_arrow(
    H: int,
    W: int,
    color: int,
    occupied: set[tuple[int, int]],
    static_walls: dict[tuple[int, int], int],
    prior_rays: dict[tuple[int, int], int],
    min_dist2: int,
) -> dict[str, object] | None:
    prior_positions = list(prior_rays.keys())
    if not prior_positions:
        return None

    for _ in range(450):
        pr, pc = _rng.choice(prior_positions)
        flip = _rng.choice(["dr", "dc"])
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])

        if flip == "dr":
            bounce_r, bounce_c = pr - dr, pc
        else:
            bounce_r, bounce_c = pr, pc - dc

        if not (0 <= bounce_r < H and 0 <= bounce_c < W):
            continue
        if (bounce_r, bounce_c) in occupied:
            continue

        back_limit = _max_steps_to_edge(bounce_r, bounce_c, -dr, -dc, H, W)
        if back_limit < 2:
            continue
        k = _rng.randint(2, min(7, back_limit))
        sr = bounce_r - k * dr
        sc = bounce_c - k * dc
        dist2 = _center_dist2(sr, sc, H, W)
        if dist2 <= min_dist2:
            continue

        arrow_pts = _arrow_points_from_start(sr, sc, dr, dc)
        arrow_set = set(arrow_pts)
        if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
            continue

        ok = True
        for step in range(1, k + 1):
            pos = (sr + step * dr, sc + step * dc)
            if pos in occupied or pos in arrow_set:
                ok = False
                break
        if not ok:
            continue

        ray_info = _trace_single_ray(
            H=H,
            W=W,
            start_col=int(color),
            start=(int(sr), int(sc)),
            direction=(int(dr), int(dc)),
            arrow_cells=arrow_set,
            static_walls=static_walls,
            prior_rays=prior_rays,
        )
        ray_cells = ray_info["ray_cells"]
        ray_positions = {(r, c) for r, c, _ in ray_cells}
        if ray_info["dynamic_hits"] < 1 or len(ray_cells) < 5:
            continue
        if ray_positions & occupied:
            continue
        if ray_positions & arrow_set:
            continue

        return {
            "color": int(color),
            "start": (int(sr), int(sc)),
            "dir": (int(dr), int(dc)),
            "arrow_pts": arrow_pts,
            "ray_cells": ray_cells,
            "walls": [],
            "dynamic_hits": int(ray_info["dynamic_hits"]),
            "order_key": _order_key((sr, sc), H, W, color),
        }
    return None


def _commit_placement(
    grid: np.ndarray,
    out: np.ndarray,
    occupied: set[tuple[int, int]],
    static_walls: dict[tuple[int, int], int],
    prior_rays: dict[tuple[int, int], int],
    placement: dict[str, object],
) -> None:
    new_cells: set[tuple[int, int]] = set()

    for r, c in placement["arrow_pts"]:
        grid[r, c] = int(placement["color"])
        out[r, c] = int(placement["color"])
        new_cells.add((int(r), int(c)))

    for wpxs, wcol in placement["walls"]:
        for r, c in wpxs:
            grid[r, c] = int(wcol)
            out[r, c] = int(wcol)
            static_walls[(int(r), int(c))] = int(wcol)
            new_cells.add((int(r), int(c)))

    for r, c, col in placement["ray_cells"]:
        out[r, c] = int(col)
        prior_rays[(int(r), int(c))] = int(col)
        new_cells.add((int(r), int(c)))

    occupied |= new_cells


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:3]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    static_walls: dict[tuple[int, int], int] = {}
    prior_rays: dict[tuple[int, int], int] = {}

    first = _plan_static_arrow(
        H=H,
        W=W,
        color=int(arrow_colors[0]),
        occupied=occupied,
        palette=palette,
        dist_lo=0,
        dist_hi=120,
        require_bounce=True,
    )
    if first is None:
        return None
    _commit_placement(grid, out, occupied, static_walls, prior_rays, first)

    second = _plan_dynamic_arrow(
        H=H,
        W=W,
        color=int(arrow_colors[1]),
        occupied=occupied,
        static_walls=static_walls,
        prior_rays=prior_rays,
        min_dist2=int(first["order_key"][0]),
    )
    if second is None or second["order_key"][0] <= first["order_key"][0]:
        return None
    _commit_placement(grid, out, occupied, static_walls, prior_rays, second)

    if _rng.random() < 0.45:
        third = _plan_static_arrow(
            H=H,
            W=W,
            color=int(arrow_colors[2]),
            occupied=occupied,
            palette=palette,
            dist_lo=int(second["order_key"][0]) + 10,
            dist_hi=10**9,
            require_bounce=False,
        )
        if third is not None and third["order_key"][0] > second["order_key"][0]:
            _commit_placement(grid, out, occupied, static_walls, prior_rays, third)

    solved, meta = _trace_all(grid, ordered=True)
    if not np.array_equal(solved, out):
        return None
    independent, _ = _trace_all(grid, ordered=False)
    if np.array_equal(independent, solved):
        return None
    if int(meta["dynamic_hits"]) < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
