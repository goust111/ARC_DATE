from __future__ import annotations

# Variant of 142ca369
# Dimension: D20 — 射线与射线的交互 (multi_step)
# Change: 射线之间无交互——各自独立涂色，后处理的射线可以覆盖先前射线的轨迹...
#   → 射线合流：同色合并，异色阻断
#
# Design analysis for this variant:
# spec_variant:
#   unchanged_core:
#     grid: 18-20 rows x 18-20 cols; bg=0
#     arrows:
#       shape: 3-pixel L in a 2x2 box
#       start: right-angle corner
#       direction: diagonal away from the missing corner
#     walls:
#       shape: 1-pixel or 3-pixel axis-aligned segment
#       role: same as original task; side-touch flips one axis, corner-hit flips both
#       color: ray inherits the wall color at the bounce
#   d20_change:
#     simulation_mode: all rays advance synchronously one step at a time
#     interaction_site: a ray's next center cell touching another ray center/trace
#     same_color:
#       effect: rays merge
#       visible_result: the surviving branch paints a 3-pixel-wide diagonal band
#       width_model: for a segment moving by (dr,dc), paint
#         {(r,c), (r-dr,c+dc), (r+dr,c-dc)}
#       continuation_rule: if several same-color frontiers enter the same center cell
#         on the same tick, the earliest arrow in reading order survives and continues;
#         the others terminate into that merged trunk
#     different_color:
#       effect: mutual blocking
#       visible_result: no contested crossing cell is painted; the involved rays stop
#     generator_requirements:
#       must_show_merge: yes, with a same-color collision that leaves a long thick suffix
#       must_show_block: yes, with a different-color collision that leaves two opposing thin prefixes
#       keep_visual_clarity: interaction regions are spatially separated
#   behavior_trace:
#     original_local_step:
#       - ray advances diagonally
#       - current ray may overwrite previously painted ray cells because there is no ray-ray logic
#     new_local_step:
#       - all active rays first compute their next move using the original wall logic
#       - then collisions are resolved before any paint is committed
#       - same-color collisions keep one active ray and widen the output
#       - different-color collisions cancel the crossing and stop the rays
#   implicit_assumptions_replaced:
#     - original code assumed "out[nr,nc] == bg" was the only case that mattered for ray paint
#       order was therefore harmless; under D20 this is false, so ray paint is tracked explicitly
#     - original generator assumed rays/walls/arrows never overlap in output; under D20 controlled
#       output overlaps are now intentional but only at planned interaction points
#     - original sequential processing let later rays overwrite earlier ones; this variant forbids
#       that by resolving interactions before writing the step
#   legality_notes:
#     - input objects still never overlap: arrows and walls remain disjoint and in-bounds
#     - thick paint is output-only and never changes wall/arrow detection because solver always
#       reads blockers from the input grid, not from the evolving output
#     - the generator chooses collision times/points first, then derives arrow starts, so all
#       required interactions are guaranteed to occur on the solver's synchronous timeline

import sys
import random as _rng

import numpy as np

_ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _ARC_UTILS_DIR not in sys.path:
    sys.path.append(_ARC_UTILS_DIR)

from arc_utils import colors_present, components_mask, get_bg


Coord = tuple[int, int]


class Ray:
    def __init__(self, rid: int, start: Coord, cur: Coord, dr: int, dc: int, color: int, width: int):
        self.rid = int(rid)
        self.start = start
        self.cur = cur
        self.dr = int(dr)
        self.dc = int(dc)
        self.color = int(color)
        self.width = int(width)
        self.alive = True


def _is_arrow_pts(pts: list[Coord]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[Coord]) -> tuple[tuple[int, int], Coord] | None:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
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


def _detect_arrows_and_walls(g: np.ndarray, bg: int):
    arrow_cells: set[Coord] = set()
    arrows: list[tuple[int, Coord, int, int, list[Coord]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if not _is_arrow_pts(pts):
                continue
            info = _arrow_dir_and_start(pts)
            if info is None:
                continue
            (dr, dc), start = info
            arrows.append((int(col), start, int(dr), int(dc), pts))
            arrow_cells |= set(pts)

    arrows.sort(key=lambda item: (item[1][0], item[1][1], item[2], item[3], item[0]))
    wall_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }
    return arrows, arrow_cells, wall_cells


def _footprint(center: Coord, step_dr: int, step_dc: int, width: int, moved: bool) -> list[Coord]:
    if width <= 1 or not moved:
        return [center]
    r, c = center
    return [
        (r, c),
        (r - step_dr, c + step_dc),
        (r + step_dr, c - step_dc),
    ]


def _propose_step(
    ray: Ray,
    g: np.ndarray,
    arrow_cells: set[Coord],
    wall_cells: set[Coord],
    bg: int,
) -> dict | None:
    H, W = g.shape
    cur_r, cur_c = ray.cur
    dr, dc = int(ray.dr), int(ray.dc)
    nr, nc = cur_r + dr, cur_c + dc
    if not (0 <= nr < H and 0 <= nc < W):
        return None

    if (nr, nc) in wall_cells:
        wcol = int(g[nr, nc])
        return {
            "center": (cur_r, cur_c),
            "paint_color": wcol,
            "next_cur": (cur_r, cur_c),
            "next_dr": -dr,
            "next_dc": -dc,
            "next_color": wcol,
            "paint_dr": dr,
            "paint_dc": dc,
            "moved": False,
            "paint_allowed": int(g[cur_r, cur_c]) == int(bg),
        }

    if (nr, nc) in arrow_cells:
        return None

    paint_color = int(ray.color)
    next_dr, next_dc, next_color = dr, dc, int(ray.color)

    if dr == 1 and (nr + 1, nc) in wall_cells:
        wcol = int(g[nr + 1, nc])
        if int(g[nr, nc]) == int(bg):
            paint_color = wcol
        next_color = wcol
        next_dr = -dr
    elif dr == -1 and (nr - 1, nc) in wall_cells:
        wcol = int(g[nr - 1, nc])
        if int(g[nr, nc]) == int(bg):
            paint_color = wcol
        next_color = wcol
        next_dr = -dr

    if dc == 1 and (nr, nc + 1) in wall_cells:
        wcol = int(g[nr, nc + 1])
        if int(g[nr, nc]) == int(bg):
            paint_color = wcol
        next_color = wcol
        next_dc = -dc
    elif dc == -1 and (nr, nc - 1) in wall_cells:
        wcol = int(g[nr, nc - 1])
        if int(g[nr, nc]) == int(bg):
            paint_color = wcol
        next_color = wcol
        next_dc = -dc

    return {
        "center": (nr, nc),
        "paint_color": int(paint_color),
        "next_cur": (nr, nc),
        "next_dr": int(next_dr),
        "next_dc": int(next_dc),
        "next_color": int(next_color),
        "paint_dr": dr,
        "paint_dc": dc,
        "moved": True,
        "paint_allowed": True,
    }


def _simulate(grid: np.ndarray, want_stats: bool = False):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    arrows, arrow_cells, wall_cells = _detect_arrows_and_walls(g, bg)
    out = g.copy()
    input_blockers = {(int(r), int(c)) for r, c in np.argwhere(g != int(bg))}

    rays = [
        Ray(rid=i, start=start, cur=start, dr=dr, dc=dc, color=col, width=1)
        for i, (col, start, dr, dc, _pts) in enumerate(arrows)
    ]
    occupancy: dict[Coord, set[int]] = {}
    stats = {
        "rays_started": len(rays),
        "merge_events": 0,
        "block_events": 0,
        "same_tick_merges": 0,
        "trail_merges": 0,
        "same_tick_blocks": 0,
        "trail_blocks": 0,
        "thick_paint_cells": 0,
        "merged_rays_stopped": 0,
        "blocked_rays_stopped": 0,
    }

    max_ticks = max(1, H * W * 6)
    for _ in range(max_ticks):
        active = [ray for ray in rays if ray.alive]
        if not active:
            break

        proposals: dict[int, dict] = {}
        for ray in active:
            prop = _propose_step(ray, g, arrow_cells, wall_cells, bg)
            if prop is None:
                ray.alive = False
                continue
            proposals[ray.rid] = prop

        if not proposals:
            continue

        cancelled: dict[int, str] = {}
        promote: set[int] = set()
        force_thick_paint: set[int] = set()

        by_center: dict[Coord, list[int]] = {}
        for rid, prop in proposals.items():
            by_center.setdefault(prop["center"], []).append(rid)

        for center, rids in by_center.items():
            if len(rids) <= 1:
                continue
            cols = {int(proposals[rid]["paint_color"]) for rid in rids}
            if len(cols) == 1:
                leader = min(rids)
                promote.add(leader)
                force_thick_paint.add(leader)
                for rid in rids:
                    if rid == leader:
                        continue
                    if rid not in cancelled:
                        cancelled[rid] = "merge_same_tick"
                stats["merge_events"] += 1
                stats["same_tick_merges"] += 1
                stats["merged_rays_stopped"] += len(rids) - 1
            else:
                for rid in rids:
                    if rid not in cancelled:
                        cancelled[rid] = "block_same_tick"
                stats["block_events"] += 1
                stats["same_tick_blocks"] += 1
                stats["blocked_rays_stopped"] += len(rids)

        for rid, prop in proposals.items():
            if rid in cancelled:
                continue
            center = prop["center"]
            owners = occupancy.get(center, set()) - {rid}
            if not owners:
                continue
            same_color = int(out[center]) == int(prop["paint_color"])
            active_owners = {oid for oid in owners if rays[oid].alive}
            if same_color:
                cancelled[rid] = "merge_trail"
                force_thick_paint.add(rid)
                promote |= active_owners
                stats["merge_events"] += 1
                stats["trail_merges"] += 1
                stats["merged_rays_stopped"] += 1
            else:
                cancelled[rid] = "block_trail"
                for oid in active_owners:
                    if oid not in cancelled:
                        cancelled[oid] = "block_trail_owner"
                stats["block_events"] += 1
                stats["trail_blocks"] += 1
                stats["blocked_rays_stopped"] += 1 + len(active_owners)

        for rid in sorted(proposals):
            if rid in cancelled and cancelled[rid] not in {"merge_trail"}:
                continue

            prop = proposals[rid]
            ray = rays[rid]
            width_to_paint = 3 if (ray.width == 3 or rid in promote or rid in force_thick_paint) else 1
            cells = _footprint(prop["center"], prop["paint_dr"], prop["paint_dc"], width_to_paint, prop["moved"])

            for cell in cells:
                rr, cc = cell
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if cell in input_blockers:
                    continue
                prev = int(out[rr, cc])
                if prev not in (int(bg), int(prop["paint_color"])):
                    continue
                out[rr, cc] = int(prop["paint_color"])
                occupancy.setdefault(cell, set()).add(rid)
                if width_to_paint == 3:
                    stats["thick_paint_cells"] += 1

        for rid, prop in proposals.items():
            ray = rays[rid]
            reason = cancelled.get(rid)
            if reason in {"merge_same_tick", "block_same_tick", "block_trail", "block_trail_owner", "merge_trail"}:
                ray.alive = False
                continue
            ray.cur = prop["next_cur"]
            ray.dr = prop["next_dr"]
            ray.dc = prop["next_dc"]
            ray.color = prop["next_color"]
            if ray.width == 1 and rid in promote:
                ray.width = 3

    if want_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, want_stats=False)


def _arrow_points_from_start(start: Coord, dr: int, dc: int) -> list[Coord]:
    sr, sc = start
    mr, mc = sr - dr, sc - dc
    return [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]


def _diag_segment(start: Coord, dr: int, dc: int, steps: int) -> list[Coord]:
    sr, sc = start
    return [(sr + dr * k, sc + dc * k) for k in range(1, steps + 1)]


def _in_bounds(H: int, W: int, cell: Coord) -> bool:
    r, c = cell
    return 0 <= r < H and 0 <= c < W


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    merge_color, block_c1, block_c2, wall_color = palette[:4]

    grid = np.zeros((H, W), dtype=int)
    used_inputs: set[Coord] = set()

    merge_t = _rng.randint(3, 5)
    merge_after = _rng.randint(3, 5)
    block_t = _rng.randint(3, 5)

    p_r_lo = merge_t + 2
    p_r_hi = H - merge_t - merge_after - 3
    p_c_lo = merge_t + 2
    p_c_hi = W - merge_after - 4
    if p_r_lo > p_r_hi or p_c_lo > p_c_hi:
        return None
    p = (_rng.randint(p_r_lo, p_r_hi), _rng.randint(p_c_lo, p_c_hi))

    a_start = (p[0] - merge_t, p[1] - merge_t)
    b_start = (p[0] + merge_t, p[1] - merge_t)
    a_pts = _arrow_points_from_start(a_start, 1, 1)
    b_pts = _arrow_points_from_start(b_start, -1, 1)

    bounce_cell = (p[0] + merge_after, p[1] + merge_after)
    wall = (bounce_cell[0] + 1, bounce_cell[1])
    if not _in_bounds(H, W, wall):
        return None
    tail_dir = (-1, 1)
    tail_len = min(bounce_cell[0], W - 1 - bounce_cell[1])
    if tail_len < 4:
        return None

    q_r_lo = block_t + 2
    q_r_hi = H - block_t - 3
    q_c_lo = max(p[1] + merge_after + 5, 3)
    q_c_hi = W - block_t - 3
    if q_r_lo > q_r_hi or q_c_lo > q_c_hi:
        return None
    q = (_rng.randint(q_r_lo, q_r_hi), _rng.randint(q_c_lo, q_c_hi))

    c_start = (q[0] - block_t, q[1] + block_t)
    d_start = (q[0] + block_t, q[1] + block_t)
    c_pts = _arrow_points_from_start(c_start, 1, -1)
    d_pts = _arrow_points_from_start(d_start, -1, -1)

    all_arrow_cells = a_pts + b_pts + c_pts + d_pts
    if any(not _in_bounds(H, W, cell) for cell in all_arrow_cells):
        return None
    if len(set(all_arrow_cells)) != len(all_arrow_cells):
        return None

    a_future = _diag_segment(p, 1, 1, merge_after)
    a_tail = _diag_segment(bounce_cell, tail_dir[0], tail_dir[1], tail_len)
    b_prefix = _diag_segment(b_start, -1, 1, merge_t - 1)
    c_prefix = _diag_segment(c_start, 1, -1, block_t - 1)
    d_prefix = _diag_segment(d_start, -1, -1, block_t - 1)

    forbidden_centers = set(a_pts + b_pts + c_pts + d_pts + [wall])
    all_centers = set(a_future + a_tail + b_prefix + c_prefix + d_prefix + [p, q, bounce_cell])
    if forbidden_centers & all_centers:
        return None

    # Keep the two interaction regions visually separated.
    if any(abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1 for r1, c1 in a_future + a_tail + [p] for r2, c2 in c_prefix + d_prefix + [q]):
        return None

    placements = [
        (a_pts, merge_color),
        (b_pts, merge_color),
        (c_pts, block_c1),
        (d_pts, block_c2),
        ([wall], wall_color),
    ]
    for cells, color in placements:
        if any(cell in used_inputs for cell in cells):
            return None
        for r, c in cells:
            grid[r, c] = color
            used_inputs.add((r, c))

    out, stats = _simulate(grid, want_stats=True)
    if stats["merge_events"] < 1 or stats["block_events"] < 1:
        return None
    if stats["same_tick_merges"] < 1 or stats["same_tick_blocks"] < 1:
        return None
    if stats["thick_paint_cells"] < 12:
        return None
    if stats["merged_rays_stopped"] < 1 or stats["blocked_rays_stopped"] < 2:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(1200):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 1200 attempts")
