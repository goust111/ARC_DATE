# Variant of 142ca369
# Dimension: D10 — 反射后变换效果 (transform)
# Change: 反射点重新涂色为墙壁颜色（recolor），射线继承墙壁颜色... → 碰撞检测与射线覆盖
#
# Design analysis
# ---------------
# variant_spec:
#   source_task:
#     id: 409aa875
#     status: solver source not present in the provided workspace; only
#       evaluation JSON was available, so the collision mechanism below is
#       calibrated from the variant brief plus the sample I/O observation that
#       overlaps introduce a dedicated output color instead of ordinary paint.
#   concrete_behavior:
#     input:
#       same object families as the seed task:
#         - monochrome 3-cell L arrows in 2x2 boxes
#         - colored wall pixels / 3-pixel axis-aligned wall segments
#       plus a generator guarantee that at least one pair of rays from distinct
#       arrows geometrically crosses on a background cell.
#     output:
#       - walls and arrows stay as in the input
#       - each ray still travels diagonally, side-bounces on adjacent walls,
#         corner-bounces on direct wall hits, and inherits wall color after a
#         bounce exactly as in the original task
#       - new D10 rule: if two or more distinct rays visit the same background
#         cell, that cell becomes a dedicated collision marker color 9 instead
#         of whichever ray would have painted it first
#   small_trace:
#     original:
#       - arrow A paints blue through cell X
#       - arrow B later also reaches X
#       - because the old solver only painted background cells once, X kept the
#         first painted color and the overlap was visually silent
#     variant:
#       - trace both rays independently against the input walls/arrows
#       - aggregate visits by cell
#       - if X has visits from at least two ray ids, write 9 at X
#     difference:
#       the path geometry is unchanged; only overlap resolution is changed from
#       order-sensitive overwrite / first-hit behavior to explicit collision
#       marking
#   complexity_calibration:
#     - Collision color is fixed to 9 and excluded from all input objects. That
#       keeps the new rule inferable from a few examples; humans only need to
#       notice "crossing rays produce 9".
#     - Samples still contain at least one reflected ray with wall-color
#       inheritance, so the task remains recognizably based on 142ca369 rather
#       than collapsing into a pure line-crossing puzzle.
#     - Collision count is kept small (1-4) so the grid stays readable.
#   implicit_assumption_audit:
#     - Original `if out[nr, nc] == bg: out[nr, nc] = cur_col` assumed ray paths
#       never overlap. That is false here, so output painting is deferred until
#       all rays are traced.
#     - Original bounce recolor of the current cell was safe because no later ray
#       could also need that cell. Here it must feed a per-ray visit map first;
#       collision resolution happens only after all traces are collected.
#     - Original generator rejected any ray overlap. This variant must instead
#       allow overlap between specific ray pairs while still forbidding overlap
#       with arrows and walls.
#   placement_legality:
#     - Intentional collision pairs are built from two straight diagonal rays
#       whose only shared cell is a chosen crossing point.
#     - Additional bouncing arrows are placed with the seed-style planning and
#       are forbidden from using existing reserved ray cells, so accidental extra
#       overlaps stay limited.
#     - Final acceptance uses the solver itself plus metrics: at least one
#       collision cell, at least one bounce, and collision-aware output differs
#       from no-collision rendering on the same input.
#   visual_observability:
#     - Every generated instance contains both:
#         1. a visible diagonal crossing that turns into color 9
#         2. at least one wall-reflected ray segment that changes color
#       This gives the observer a direct before/after clue for the new rule.
#   visual_cleanliness:
#     - Collision color 9 is absent from the input.
#     - Non-collision ray overlaps are rejected; only the intended collision
#       cells can stack multiple trajectories.
#   color_inference:
#     - The only new color transform is "ray-ray overlap => 9".
#       No per-pair lookup table is introduced, avoiding an underconstrained
#       color-mixing problem.
#   new_solver_assumptions:
#     - Input uses background 0 and reserves 9 for collision output only.
#     - Arrows are still monochrome 3-cell L shapes detected per color.
#     - Walls occupy only input foreground cells that are not arrow cells.
#     Generator enforces all three.

from __future__ import annotations

from pathlib import Path
import sys
import random as _rng

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
for _p in (ROOT / "solvers", ROOT / "arc2_solver_select"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

from arc_utils import colors_present, components_mask, get_bg


COLLISION_COLOR = 9


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


def _arrow_pts_from_start(sr: int, sc: int, dr: int, dc: int) -> list[tuple[int, int]]:
    mr, mc = sr - dr, sc - dc
    pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
    return [(int(r), int(c)) for r, c in pts]


def _collect_scene(g: np.ndarray) -> tuple[int, list[dict], set[tuple[int, int]], set[tuple[int, int]]]:
    bg = int(get_bg(g))
    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[dict] = []
    next_id = 0
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if not _is_arrow_pts(pts):
                continue
            info = _arrow_dir_and_start(pts)
            if info is None:
                continue
            direction, start = info
            arrows.append(
                {
                    "id": int(next_id),
                    "color": int(col),
                    "pts": pts,
                    "start": (int(start[0]), int(start[1])),
                    "dir": (int(direction[0]), int(direction[1])),
                }
            )
            next_id += 1
            arrow_cells |= set(pts)
    arrows.sort(key=lambda rec: (rec["color"], rec["start"][0], rec["start"][1], rec["id"]))
    wall_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }
    return bg, arrows, arrow_cells, wall_cells


def _trace_arrow(
    g: np.ndarray,
    bg: int,
    arrow: dict,
    arrow_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
) -> dict:
    H, W = g.shape
    dr, dc = arrow["dir"]
    cur = arrow["start"]
    cur_col = int(arrow["color"])
    seen: set[tuple[tuple[int, int], int, int, int]] = set()
    cell_colors: dict[tuple[int, int], int] = {}
    bounce_cells: set[tuple[int, int]] = set()

    def repaint_if_bg(cell: tuple[int, int], color: int) -> None:
        r, c = cell
        if int(g[r, c]) == int(bg):
            cell_colors[(int(r), int(c))] = int(color)

    while True:
        state = (cur, int(dr), int(dc), int(cur_col))
        if state in seen:
            break
        seen.add(state)

        nr = int(cur[0] + dr)
        nc = int(cur[1] + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            break

        if (nr, nc) in wall_cells:
            wcol = int(g[nr, nc])
            repaint_if_bg(cur, wcol)
            if int(g[cur[0], cur[1]]) == int(bg):
                bounce_cells.add((int(cur[0]), int(cur[1])))
            cur_col = int(wcol)
            dr, dc = -int(dr), -int(dc)
            continue

        if (nr, nc) in arrow_cells:
            break

        cur = (int(nr), int(nc))
        repaint_if_bg(cur, cur_col)

        if int(dr) == 1 and (nr + 1, nc) in wall_cells:
            wcol = int(g[nr + 1, nc])
            repaint_if_bg(cur, wcol)
            bounce_cells.add(cur)
            cur_col = int(wcol)
            dr = -int(dr)
        elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
            wcol = int(g[nr - 1, nc])
            repaint_if_bg(cur, wcol)
            bounce_cells.add(cur)
            cur_col = int(wcol)
            dr = -int(dr)

        if int(dc) == 1 and (nr, nc + 1) in wall_cells:
            wcol = int(g[nr, nc + 1])
            repaint_if_bg(cur, wcol)
            bounce_cells.add(cur)
            cur_col = int(wcol)
            dc = -int(dc)
        elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
            wcol = int(g[nr, nc - 1])
            repaint_if_bg(cur, wcol)
            bounce_cells.add(cur)
            cur_col = int(wcol)
            dc = -int(dc)

    return {
        "id": int(arrow["id"]),
        "color": int(arrow["color"]),
        "cells": cell_colors,
        "bounce_cells": bounce_cells,
    }


def _simulate_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg, arrows, arrow_cells, wall_cells = _collect_scene(g)
    hits: dict[tuple[int, int], dict[int, int]] = {}
    traces: list[dict] = []
    for arrow in arrows:
        trace = _trace_arrow(g, bg, arrow, arrow_cells, wall_cells)
        traces.append(trace)
        for cell, color in trace["cells"].items():
            hits.setdefault(cell, {})[int(trace["id"])] = int(color)

    collision_cells = {cell for cell, ray_map in hits.items() if len(ray_map) >= 2}
    bounce_cells = set().union(*(trace["bounce_cells"] for trace in traces)) if traces else set()
    return {
        "bg": int(bg),
        "arrows": arrows,
        "arrow_cells": arrow_cells,
        "wall_cells": wall_cells,
        "hits": hits,
        "collision_cells": collision_cells,
        "bounce_cells": bounce_cells,
        "traces": traces,
    }


def _render_from_summary(grid: np.ndarray, summary: dict, *, collision: bool) -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    for cell, ray_map in summary["hits"].items():
        r, c = cell
        if collision and len(ray_map) >= 2:
            out[r, c] = int(COLLISION_COLOR)
            continue
        chosen_id = min(ray_map)
        out[r, c] = int(ray_map[chosen_id])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from L-shaped arrows, reflecting off colored wall pixels.

    Variant rule:
    - Reflection geometry and wall-color inheritance stay the same.
    - If two distinct rays visit the same background cell, the output writes the
      dedicated collision color 9 on that cell instead of silently keeping one
      ray's paint.
    """
    summary = _simulate_scene(np.array(grid, dtype=int))
    return _render_from_summary(np.array(grid, dtype=int), summary, collision=True)


def _straight_path_cells(H: int, W: int, sr: int, sc: int, dr: int, dc: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r, c = int(sr), int(sc)
    while True:
        r += int(dr)
        c += int(dc)
        if not (0 <= r < H and 0 <= c < W):
            break
        cells.append((int(r), int(c)))
    return cells


def _sample_collision_ray(
    H: int, W: int, center: tuple[int, int], dr: int, dc: int
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    cr, cc = center
    pre_limit = min(
        (cr - 1) if dr == 1 else (H - 2 - cr),
        (cc - 1) if dc == 1 else (W - 2 - cc),
    )
    post_limit = min(
        (H - 1 - cr) if dr == 1 else cr,
        (W - 1 - cc) if dc == 1 else cc,
    )
    if pre_limit < 3 or post_limit < 3:
        return None
    step_to_center = _rng.randint(3, int(pre_limit))
    sr = int(cr - step_to_center * dr)
    sc = int(cc - step_to_center * dc)
    arrow_pts = _arrow_pts_from_start(sr, sc, dr, dc)
    if any(not (0 <= r < H and 0 <= c < W) for r, c in arrow_pts):
        return None
    path_cells = _straight_path_cells(H, W, sr, sc, dr, dc)
    if not path_cells or center not in path_cells:
        return None
    return sr, sc, arrow_pts, path_cells


def _place_collision_pair(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    color_a: int,
    color_b: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    H, W = grid.shape
    orientations = [((1, 1), (1, -1)), ((-1, 1), (-1, -1))]
    for _ in range(80):
        (dr1, dc1), (dr2, dc2) = _rng.choice(orientations)
        cr = _rng.randint(5, H - 6)
        cc = _rng.randint(5, W - 6)
        center = (int(cr), int(cc))
        if center in reserved:
            continue

        cand1 = _sample_collision_ray(H, W, center, dr1, dc1)
        cand2 = _sample_collision_ray(H, W, center, dr2, dc2)
        if cand1 is None or cand2 is None:
            continue

        _sr1, _sc1, pts1, path1 = cand1
        _sr2, _sc2, pts2, path2 = cand2
        set1 = set(path1)
        set2 = set(path2)
        if (set1 & set2) != {center}:
            continue
        if set(pts1) & set(pts2):
            continue
        if (set(pts1) | set1) & reserved:
            continue
        if (set(pts2) | set2) & reserved:
            continue
        if set(pts1) & set2:
            continue
        if set(pts2) & set1:
            continue

        for r, c in pts1:
            grid[r, c] = int(color_a)
        for r, c in pts2:
            grid[r, c] = int(color_b)
        new_reserved = set(pts1) | set(pts2) | set1 | set2
        return new_reserved, {center}
    return None


def _try_bouncing_arrow(
    H: int,
    W: int,
    reserved: set[tuple[int, int]],
    color: int,
    palette: list[int],
) -> dict | None:
    for _ in range(100):
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        n_bounces = _rng.choices([1, 2, 3], weights=[3, 3, 1])[0]

        r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
        c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
        if r_lo > r_hi or c_lo > c_hi:
            continue
        sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)
        arrow_pts = _arrow_pts_from_start(sr, sc, dr, dc)
        arrow_set = set(arrow_pts)
        if any(not (0 <= r < H and 0 <= c < W) for r, c in arrow_pts):
            continue
        if arrow_set & reserved:
            continue

        cur_r, cur_c = int(sr), int(sc)
        cur_dr, cur_dc = int(dr), int(dc)
        cur_col = int(color)
        ray_cells: list[tuple[int, int, int]] = []
        ray_set: set[tuple[int, int]] = set()
        walls: list[tuple[list[tuple[int, int]], int]] = []
        ok = True

        for seg_idx in range(n_bounces + 1):
            max_steps = min(
                (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                (W - 1 - cur_c) if cur_dc == 1 else cur_c,
            )
            if seg_idx < n_bounces:
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
                cell = (int(nr), int(nc))
                if cell in reserved or cell in ray_set or cell in arrow_set:
                    ok = False
                    break
                ray_cells.append((int(nr), int(nc), int(cur_col)))
                ray_set.add(cell)
                cur_r, cur_c = int(nr), int(nc)
            if not ok:
                break

            if seg_idx >= n_bounces:
                continue

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
            prefer_long = _rng.random() < 0.6
            options.sort(key=lambda item: len(item[0]), reverse=prefer_long)
            choice = None
            for wall_pts, axis in options:
                wall_set = set(wall_pts)
                if wall_set & reserved or wall_set & ray_set or wall_set & arrow_set:
                    continue
                choice = (wall_pts, axis)
                break
            if choice is None:
                ok = False
                break

            wall_pts, axis = choice
            wall_col = int(_rng.choice(palette))
            walls.append((wall_pts, wall_col))
            lr, lc, _old = ray_cells[-1]
            ray_cells[-1] = (int(lr), int(lc), int(wall_col))
            cur_col = int(wall_col)
            if axis == "dr":
                cur_dr = -int(cur_dr)
            else:
                cur_dc = -int(cur_dc)

        if not ok or len(ray_cells) < 4 or not walls:
            continue

        new_reserved = arrow_set | ray_set
        for wall_pts, _wall_col in walls:
            new_reserved.update(wall_pts)
        if new_reserved & reserved:
            continue
        return {
            "arrow_pts": arrow_pts,
            "color": int(color),
            "ray_cells": ray_cells,
            "walls": walls,
            "reserved": new_reserved,
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    if H < 12 or W < 12:
        return None

    grid = np.zeros((H, W), dtype=int)
    reserved: set[tuple[int, int]] = set()
    palette = [c for c in range(1, 9) if c != COLLISION_COLOR]
    _rng.shuffle(palette)

    n_pairs = _rng.choice([1, 2])
    n_bounce_arrows = _rng.choice([1, 2, 2, 3])
    needed_arrow_colors = 2 * n_pairs + n_bounce_arrows
    if needed_arrow_colors > len(palette):
        return None

    arrow_colors = palette[:needed_arrow_colors]
    color_idx = 0

    intended_collision_cells: set[tuple[int, int]] = set()
    for _pair_idx in range(n_pairs):
        placed = _place_collision_pair(
            grid,
            reserved,
            int(arrow_colors[color_idx]),
            int(arrow_colors[color_idx + 1]),
        )
        color_idx += 2
        if placed is None:
            return None
        new_reserved, new_centers = placed
        reserved |= new_reserved
        intended_collision_cells |= new_centers

    for _arrow_idx in range(n_bounce_arrows):
        cand = _try_bouncing_arrow(H, W, reserved, int(arrow_colors[color_idx]), palette)
        color_idx += 1
        if cand is None:
            return None
        for r, c in cand["arrow_pts"]:
            grid[r, c] = int(cand["color"])
        for wall_pts, wall_col in cand["walls"]:
            for r, c in wall_pts:
                grid[r, c] = int(wall_col)
        reserved |= set(cand["reserved"])

    summary = _simulate_scene(grid)
    out = _render_from_summary(grid, summary, collision=True)
    overlay = _render_from_summary(grid, summary, collision=False)

    collision_count = len(summary["collision_cells"])
    bounce_count = len(summary["bounce_cells"])
    if collision_count < len(intended_collision_cells) or collision_count > 4:
        return None
    if bounce_count < 1:
        return None
    if np.array_equal(out, overlay):
        return None
    if not summary["collision_cells"].issuperset(intended_collision_cells):
        return None
    if COLLISION_COLOR in set(int(v) for v in grid.flatten()):
        return None
    if len(summary["arrows"]) != needed_arrow_colors:
        return None
    if not np.any(out == COLLISION_COLOR):
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
