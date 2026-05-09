# Variant of 142ca369
# Dimension: D7 — 墙壁/障碍物形状 (obstacle)
# Change: 单像素或3像素轴对齐线段墙壁，放置在射线对角线路径旁偏移1格的位置... → 移动墙壁（墙壁在反射后消失）

from __future__ import annotations

import random as _rng
import sys

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import colors_present, components_mask, get_bg


# Design analysis for the D7 variant
# ----------------------------------
# spec_delta:
#   obstacle:
#     keep:
#       - walls are still monochrome connected components, either size 1 or a 3-pixel axis-aligned segment
#       - walls still sit one orthogonal cell off a diagonal ray and trigger the same reflection geometry
#     change:
#       - a wall is no longer static scenery
#       - once any ray uses a wall for reflection, the entire connected wall component disappears immediately
#       - disappeared wall cells become background-like: later rays may pass through and paint those cells
#   solver_order:
#     - state now matters, so arrows are processed in row-major order of their right-angle launch cell
#     - generator only creates one intentionally shared wall, and the earlier arrow is placed above/left of the later arrow
#   visible_effect:
#     - outputs remove used walls
#     - at least one later ray passes through a wall cell that existed in the input, so the disappearance is functional, not cosmetic
#
# a) Concrete behavior trace
#   Original:
#     - arrow A bounces off wall W, but W stays in place forever
#     - if arrow B later reaches W, it still bounces or stops there
#   Variant:
#     - arrow A bounces off W and W is erased immediately
#     - arrow B later reaches the same cell and treats it as empty space, painting through it
#   Small example:
#     - W center at (8,8)
#     - A reaches bounce cell (8,7), sees side wall at (8,8), flips dc and W disappears
#     - B's diagonal later steps onto (8,8); in the original task that cell is blocked, here it is paintable
#
# b) Complexity calibration
#   - I keep the original arrow shape and side-bounce geometry intact.
#   - I do not introduce moving directions or multi-step wall motion; "erase after one use" is the minimal stateful D7 change.
#   - To avoid the variant becoming too weak, every generated instance contains one shared wall that is both consumed and later traversed.
#
# c) Implicit-assumption audit against the original main()
#   - Original `wall_cells` was immutable. That breaks here; I replace it with mutable wall-component ownership.
#   - Original loop-state memoization ignored obstacle state. That is unsafe now; I include the alive-wall id tuple.
#   - Original advance only painted cells when `out[nr,nc] == bg`, which is still correct for erased walls because erasure resets them to bg.
#   - Original recolor-on-bounce checked `g[...] == bg`, which is no longer valid for erased-wall cells because input there is non-bg.
#     I instead recolor any non-arrow, non-live-wall bounce cell.
#
# d) Placement legality
#   - Input-time occupancy forbids overlaps among arrows and walls.
#   - Output-time ray occupancy is also kept disjoint, except for the intentional "later ray crosses former wall cell" event.
#   - Shared-wall generation only allows B to overlap the wall center cell, never the wall flanks or arrow cells.
#
# e) Visual observability
#   - The wall visibly vanishes from output.
#   - A second ray paints through a location that was occupied in the input.
#   - Straight filler arrows are allowed, but the disappearing-wall pair is mandatory in every sample.
#
# f) Visual cleanliness
#   - Only one shared wall is created per grid, so the stateful interaction is easy to isolate by eye.
#   - Additional arrows are straight and non-overlapping, which keeps the scene legible.
#
# g) Color inference
#   - Arrow A, arrow B, and the shared wall use distinct colors.
#   - Rays still inherit wall color on bounce exactly as in the base task, so no extra color mapping is introduced.


DIAGS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
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


def _arrow_pts_from_start(start: tuple[int, int], dr: int, dc: int) -> list[tuple[int, int]]:
    sr, sc = start
    mr, mc = sr - dr, sc - dc
    return [
        (int(r), int(c))
        for r in (mr, sr)
        for c in (mc, sc)
        if (int(r), int(c)) != (int(mr), int(mc))
    ]


def _in_bounds(H: int, W: int, cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < H and 0 <= c < W


def _steps_to_edge(H: int, W: int, start: tuple[int, int], dr: int, dc: int) -> int:
    r, c = start
    return min((H - 1 - r) if dr == 1 else r, (W - 1 - c) if dc == 1 else c)


def _ray_cells(start: tuple[int, int], dr: int, dc: int, steps: int) -> list[tuple[int, int]]:
    sr, sc = start
    return [(int(sr + dr * k), int(sc + dc * k)) for k in range(1, steps + 1)]


def _detect_arrows_and_walls(
    g: np.ndarray,
) -> tuple[int, list[dict], set[tuple[int, int]], list[dict], dict[tuple[int, int], int]]:
    bg = int(get_bg(g))
    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[dict] = []

    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if not _is_arrow_pts(pts):
                continue
            info = _arrow_dir_and_start(pts)
            if info is None:
                continue
            (dr, dc), start = info
            arrows.append(
                {
                    "color": int(col),
                    "pts": pts,
                    "start": start,
                    "dir": (int(dr), int(dc)),
                }
            )
            arrow_cells |= set(pts)

    arrows.sort(key=lambda item: (item["start"][0], item["start"][1], item["color"]))

    walls: list[dict] = []
    wall_owner: dict[tuple[int, int], int] = {}
    for col in colors_present(g, exclude={bg}):
        mask = (g == int(col))
        for coords in components_mask(mask, connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if any(p in arrow_cells for p in pts):
                continue
            wid = len(walls)
            cell_set = set(pts)
            walls.append({"id": wid, "color": int(col), "cells": cell_set})
            for cell in cell_set:
                wall_owner[cell] = wid

    return bg, arrows, arrow_cells, walls, wall_owner


def _simulate(grid: np.ndarray, *, return_stats: bool = False):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, arrows, arrow_cells, walls, wall_owner = _detect_arrows_and_walls(g)
    out = g.copy()

    alive = {wall["id"] for wall in walls}
    erased_cells: set[tuple[int, int]] = set()
    stats = {
        "walls_erased": 0,
        "wall_hits": 0,
        "pass_through_erased": 0,
        "corner_hits": 0,
        "side_hits": 0,
    }

    def erase_wall(wid: int) -> int:
        if wid not in alive:
            return int(walls[wid]["color"])
        alive.remove(wid)
        stats["walls_erased"] += 1
        for cell in walls[wid]["cells"]:
            wall_owner.pop(cell, None)
            erased_cells.add(cell)
            out[cell[0], cell[1]] = bg
        return int(walls[wid]["color"])

    for arrow in arrows:
        cur = (int(arrow["start"][0]), int(arrow["start"][1]))
        dr, dc = (int(arrow["dir"][0]), int(arrow["dir"][1]))
        cur_col = int(arrow["color"])
        seen = set()

        while True:
            state = (cur, int(dr), int(dc), int(cur_col), tuple(sorted(alive)))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            hit = wall_owner.get((nr, nc))
            if hit is not None:
                wcol = erase_wall(hit)
                stats["wall_hits"] += 1
                stats["corner_hits"] += 1
                if cur not in arrow_cells and wall_owner.get(cur) is None:
                    out[cur[0], cur[1]] = wcol
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if cur in erased_cells:
                stats["pass_through_erased"] += 1
            if int(out[nr, nc]) == bg:
                out[nr, nc] = int(cur_col)

            adj_hit = None
            if dr == 1:
                adj_hit = wall_owner.get((nr + 1, nc))
            else:
                adj_hit = wall_owner.get((nr - 1, nc))
            if adj_hit is not None:
                wcol = erase_wall(adj_hit)
                stats["wall_hits"] += 1
                stats["side_hits"] += 1
                out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)

            adj_hit = None
            if dc == 1:
                adj_hit = wall_owner.get((nr, nc + 1))
            else:
                adj_hit = wall_owner.get((nr, nc - 1))
            if adj_hit is not None:
                wcol = erase_wall(adj_hit)
                stats["wall_hits"] += 1
                stats["side_hits"] += 1
                out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)

    if return_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from L-shaped arrows. Any wall component used for a reflection
    disappears immediately, so later rays treat those cells as background.
    """
    return _simulate(grid, return_stats=False)


def _pick_wall_cells(
    H: int,
    W: int,
    center: tuple[int, int],
    flip: str,
) -> list[list[tuple[int, int]]]:
    r, c = center
    options: list[list[tuple[int, int]]] = [[(int(r), int(c))]]
    if flip == "dr":
        wall3 = [(int(r), int(c + d)) for d in (-1, 0, 1)]
        if all(_in_bounds(H, W, cell) for cell in wall3):
            options.append(wall3)
    else:
        wall3 = [(int(r + d), int(c)) for d in (-1, 0, 1)]
        if all(_in_bounds(H, W, cell) for cell in wall3):
            options.append(wall3)
    _rng.shuffle(options)
    return options


def _try_shared_pair(
    H: int,
    W: int,
    colors: list[int],
) -> tuple[dict, dict, dict, set[tuple[int, int]], set[tuple[int, int]]] | None:
    color_a, color_b, wall_col = colors[:3]

    for _ in range(500):
        dr_b, dc_b = _rng.choice(DIAGS)
        start_b = (_rng.randrange(H), _rng.randrange(W))
        arrow_b = _arrow_pts_from_start(start_b, dr_b, dc_b)
        if not all(_in_bounds(H, W, cell) for cell in arrow_b):
            continue

        steps_b = _steps_to_edge(H, W, start_b, dr_b, dc_b)
        if steps_b < 8:
            continue
        shared_step = _rng.randint(4, steps_b - 3)
        wall_center = (int(start_b[0] + dr_b * shared_step), int(start_b[1] + dc_b * shared_step))
        path_b = _ray_cells(start_b, dr_b, dc_b, steps_b)
        path_b_set = set(path_b)

        for _ in range(300):
            dr_a, dc_a = _rng.choice(DIAGS)
            flip = _rng.choice(["dr", "dc"])
            if flip == "dr":
                bounce = (int(wall_center[0] - dr_a), int(wall_center[1]))
                after_dir = (-int(dr_a), int(dc_a))
            else:
                bounce = (int(wall_center[0]), int(wall_center[1] - dc_a))
                after_dir = (int(dr_a), -int(dc_a))

            if not _in_bounds(H, W, bounce):
                continue

            max_back = _steps_to_edge(H, W, bounce, -int(dr_a), -int(dc_a))
            if max_back < 4:
                continue
            back_steps = _rng.randint(3, max_back)
            start_a = (int(bounce[0] - dr_a * back_steps), int(bounce[1] - dc_a * back_steps))
            arrow_a = _arrow_pts_from_start(start_a, dr_a, dc_a)
            if not all(_in_bounds(H, W, cell) for cell in arrow_a):
                continue
            if start_a >= start_b:
                continue

            steps_after = _steps_to_edge(H, W, bounce, after_dir[0], after_dir[1])
            if steps_after < 3:
                continue
            pre_a = _ray_cells(start_a, dr_a, dc_a, back_steps)
            post_a = _ray_cells(bounce, after_dir[0], after_dir[1], steps_after)
            path_a_set = set(pre_a) | set(post_a)

            arrow_a_set = set(arrow_a)
            arrow_b_set = set(arrow_b)
            if path_a_set & arrow_a_set:
                continue
            if path_a_set & arrow_b_set:
                continue
            if path_b_set & arrow_a_set:
                continue
            if path_a_set & path_b_set:
                continue
            if arrow_a_set & arrow_b_set:
                continue

            wall_choices = _pick_wall_cells(H, W, wall_center, flip)
            chosen_wall = None
            for wall_cells in wall_choices:
                wall_set = set(wall_cells)
                if wall_set & arrow_a_set:
                    continue
                if wall_set & arrow_b_set:
                    continue
                if wall_set & path_a_set:
                    continue
                if (wall_set - {wall_center}) & path_b_set:
                    continue
                chosen_wall = wall_cells
                break
            if chosen_wall is None:
                continue

            pair_input = arrow_a_set | arrow_b_set | set(chosen_wall)
            pair_rays = path_a_set | path_b_set
            arrow_a_obj = {
                "color": int(color_a),
                "start": start_a,
                "dir": (int(dr_a), int(dc_a)),
                "pts": arrow_a,
            }
            arrow_b_obj = {
                "color": int(color_b),
                "start": start_b,
                "dir": (int(dr_b), int(dc_b)),
                "pts": arrow_b,
            }
            wall_obj = {"color": int(wall_col), "cells": chosen_wall}
            return arrow_a_obj, arrow_b_obj, wall_obj, pair_input, pair_rays

    return None


def _try_straight_arrow(
    H: int,
    W: int,
    color: int,
    occupied_input: set[tuple[int, int]],
    occupied_rays: set[tuple[int, int]],
) -> tuple[dict, set[tuple[int, int]], set[tuple[int, int]]] | None:
    for _ in range(200):
        dr, dc = _rng.choice(DIAGS)
        start = (_rng.randrange(H), _rng.randrange(W))
        pts = _arrow_pts_from_start(start, dr, dc)
        if not all(_in_bounds(H, W, cell) for cell in pts):
            continue
        pts_set = set(pts)
        if pts_set & occupied_input:
            continue

        steps = _steps_to_edge(H, W, start, dr, dc)
        if steps < 4:
            continue
        path = _ray_cells(start, dr, dc, steps)
        path_set = set(path)
        if path_set & occupied_input:
            continue
        if path_set & occupied_rays:
            continue
        if path_set & pts_set:
            continue

        arrow = {"color": int(color), "start": start, "dir": (int(dr), int(dc)), "pts": pts}
        return arrow, pts_set, path_set
    return None


def _paint_input(
    H: int,
    W: int,
    arrows: list[dict],
    walls: list[dict],
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    for arrow in arrows:
        for r, c in arrow["pts"]:
            grid[r, c] = int(arrow["color"])
    for wall in walls:
        for r, c in wall["cells"]:
            grid[r, c] = int(wall["color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_extra = _rng.randint(0, 2)
    palette = list(range(1, 10))
    _rng.shuffle(palette)

    shared = _try_shared_pair(H, W, palette)
    if shared is None:
        return None
    arrow_a, arrow_b, wall, occupied_input, occupied_rays = shared

    arrows = [arrow_a, arrow_b]
    walls = [wall]
    color_idx = 3

    for _ in range(n_extra):
        if color_idx >= len(palette):
            break
        extra = _try_straight_arrow(H, W, palette[color_idx], occupied_input, occupied_rays)
        color_idx += 1
        if extra is None:
            continue
        arrow, pts_set, path_set = extra
        arrows.append(arrow)
        occupied_input |= pts_set
        occupied_rays |= path_set

    grid = _paint_input(H, W, arrows, walls)
    out, stats = _simulate(grid, return_stats=True)
    if stats["walls_erased"] < 1 or stats["pass_through_erased"] < 1:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a valid input for the disappearing-wall variant of 142ca369."""
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
