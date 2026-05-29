# Variant of 142ca369
# Dimension: D8 — 多步处理与射线顺序 (multi_step)
# Change: multi_bounce——每条射线独立地通过所有可达墙壁逐个反射，多条射线之间无顺序依赖... → 两阶段处理：先射线后变换

from __future__ import annotations

import random as _rng
import sys
from pathlib import Path

import numpy as np

for _parent in Path(__file__).resolve().parents:
    _cand = _parent / "solvers"
    if _cand.is_dir():
        sys.path.append(str(_cand))
        break

from arc_utils import colors_present, components_mask, get_bg


# Design analysis for the D8 variant
# ----------------------------------
# a) Concrete behavior and trace:
#    - Input objects stay the same: 3-pixel L arrows plus colored wall pixels/segments.
#    - Stage 1 still uses the original per-ray physics: each arrow traces diagonally,
#      side-touch / corner reflections behave the same, and the ray inherits wall color.
#    - The change is only in how multiple rays are combined. Instead of mutating one
#      shared output grid in arrow iteration order, each ray is first simulated on its
#      own layer over the original input.
#    - Stage 2 inspects every background cell touched by rays. If all touching rays
#      paint that cell with the same color, the color is kept. If at least two distinct
#      ray colors meet there, the whole crossing cell is erased back to background.
#    - Small example: suppose a blue ray and a red ray both visit (9,9). In the original
#      sequential solver, whichever arrow is processed first would keep (9,9) because
#      later rays only write when `out[nr,nc] == bg`. In this variant, stage 1 records
#      the multiset {blue, red}; stage 2 sees mixed colors and turns (9,9) back to 0.
#
# b) Complexity calibration:
#    - The post-process has only two observable cases, which is enough for inference:
#      same-color overlap survives, different-color overlap erases.
#    - Generator forces at least one different-color crossing in each sample, so the
#      two-stage effect is visible and does not collapse back to the original task.
#    - Most samples also include at least one bounced ray with walls, preserving the
#      original wall-reflection reasoning instead of reducing the task to plain X lines.
#
# c) Implicit assumption audit against the original `main()`:
#    - `if int(out[nr, nc]) == int(bg): out[nr, nc] = int(cur_col)` assumed inter-ray
#      overlaps never happen. That is invalid now because crossing cells are intentional.
#      Fix: simulate each ray on an independent layer, then combine afterwards.
#    - `for start_col, pts in arrows:` previously mutated one shared `out`, making final
#      colors depend on arrow iteration order when overlaps occur. That order dependence
#      is invalid for a simultaneous two-phase rule. Fix: accumulate per-cell color sets.
#    - Bounce recoloring branches such as `out[nr, nc] = int(wcol)` were only changing
#      the current ray's own bounce point; they remain valid inside a single-ray layer.
#    - Arrow detection per color still assumes arrows are isolated 3-cell components.
#      Generator therefore keeps arrow colors unique and avoids placing same-color walls
#      adjacent to arrow cells.
#
# d) Placement legality:
#    - Structural input cells (arrows and walls) never overlap and never go out of bounds.
#    - Generator allows ray/ray overlap in planned output, but forbids placing any later
#      arrow or wall on an already planned ray cell. This keeps the input unambiguous and
#      ensures solver paths are the intended ones.
#    - Special crossing rays are straight and share exactly one interior cell; extra rays
#      may bounce but their walls are placed using the original side-touch geometry.
#
# e) Visual observability:
#    - Every generated instance contains at least one mixed-color crossing, which shows up
#      as a missing hole on top of otherwise continuous diagonal traces.
#    - Because the hole appears only where two diagonals intersect, humans can infer the
#      post-processing relation from a few examples without needing symbolic labels.
#
# f) Visual cleanliness:
#    - The crossing pair is centered with margins, so the erased cell is not hidden near
#      borders. The generator also rejects samples with too many erased cells.
#    - Allowing only cell-level ray overlap, while keeping input structures disjoint from
#      planned ray cells, prevents unreadable tangles in the input.
#
# g) Color-change inferability:
#    - The only new color transform is background erasure for mixed overlaps.
#    - No arbitrary color mapping is introduced; wall-color inheritance remains exactly the
#      original one, so the new rule is tied to a simple positional cue: crossing cells.
#
# 9) New solver assumptions and generator guarantees:
#    - `_trace_single_ray` assumes arrows/walls are fully determined by the original input
#      and do not change during simulation. Generator respects this by keeping crossings
#      output-only; no new structures are created mid-trace.
#    - The combine step assumes all modified cells come from original background cells.
#      Generator and ray physics satisfy this because arrows/walls are never overwritten.


def _is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[tuple[int, int]]):
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


def _detect_layout(g: np.ndarray):
    bg = int(get_bg(g))
    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)
    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }
    return bg, arrows, arrow_cells, wall_cells


def _trace_single_ray(
    g: np.ndarray,
    bg: int,
    arrow_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    start_col: int,
    pts: list[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    info = _arrow_dir_and_start(pts)
    if info is None:
        return {}

    H, W = g.shape
    (dr, dc), cur = info
    cur_col = int(start_col)
    seen = set()
    layer = np.array(g, copy=True)

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
            if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                layer[int(cur[0]), int(cur[1])] = int(wcol)
            cur_col = int(wcol)
            dr, dc = -int(dr), -int(dc)
            continue

        if (nr, nc) in arrow_cells:
            break

        cur = (int(nr), int(nc))
        if int(layer[nr, nc]) == int(bg):
            layer[nr, nc] = int(cur_col)

        if int(dr) == 1 and (nr + 1, nc) in wall_cells:
            wcol = int(g[nr + 1, nc])
            if int(g[nr, nc]) == int(bg):
                layer[nr, nc] = int(wcol)
            cur_col = int(wcol)
            dr = -int(dr)
        elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
            wcol = int(g[nr - 1, nc])
            if int(g[nr, nc]) == int(bg):
                layer[nr, nc] = int(wcol)
            cur_col = int(wcol)
            dr = -int(dr)

        if int(dc) == 1 and (nr, nc + 1) in wall_cells:
            wcol = int(g[nr, nc + 1])
            if int(g[nr, nc]) == int(bg):
                layer[nr, nc] = int(wcol)
            cur_col = int(wcol)
            dc = -int(dc)
        elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
            wcol = int(g[nr, nc - 1])
            if int(g[nr, nc]) == int(bg):
                layer[nr, nc] = int(wcol)
            cur_col = int(wcol)
            dc = -int(dc)

    painted: dict[tuple[int, int], int] = {}
    changed = (layer != g) & (g == int(bg))
    for r, c in np.argwhere(changed):
        painted[(int(r), int(c))] = int(layer[r, c])
    return painted


def _collect_ray_paints(g: np.ndarray):
    bg, arrows, arrow_cells, wall_cells = _detect_layout(g)
    ray_paints: list[dict[tuple[int, int], int]] = []
    for start_col, pts in arrows:
        ray_paints.append(_trace_single_ray(g, bg, arrow_cells, wall_cells, start_col, pts))
    return bg, arrows, arrow_cells, wall_cells, ray_paints


def _phase1_union(g: np.ndarray, ray_paints: list[dict[tuple[int, int], int]]) -> np.ndarray:
    out = np.array(g, copy=True)
    for painted in ray_paints:
        for (r, c), col in painted.items():
            if int(out[r, c]) == 0:
                out[r, c] = int(col)
    return out


def _overlap_summary(g: np.ndarray):
    bg, arrows, arrow_cells, wall_cells, ray_paints = _collect_ray_paints(g)
    visits: dict[tuple[int, int], list[int]] = {}
    for painted in ray_paints:
        for cell, col in painted.items():
            visits.setdefault(cell, []).append(int(col))
    mixed = {cell for cell, cols in visits.items() if len(set(cols)) >= 2}
    same = {cell for cell, cols in visits.items() if len(cols) >= 2 and len(set(cols)) == 1}
    return {
        "bg": bg,
        "arrows": arrows,
        "arrow_cells": arrow_cells,
        "wall_cells": wall_cells,
        "ray_paints": ray_paints,
        "visits": visits,
        "mixed_cells": mixed,
        "same_cells": same,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Two-stage version of 142ca369.

    Stage 1: trace every arrow independently with the original bounce / color-inherit rule.
    Stage 2: on background cells visited by multiple rays, keep the cell only if all
    visiting rays paint it the same color; otherwise erase the crossing back to bg.
    """
    g = np.array(grid, dtype=int)
    bg, _arrows, _arrow_cells, _wall_cells, ray_paints = _collect_ray_paints(g)

    visits: dict[tuple[int, int], set[int]] = {}
    for painted in ray_paints:
        for cell, col in painted.items():
            visits.setdefault(cell, set()).add(int(col))

    out = np.array(g, copy=True)
    for (r, c), cols in visits.items():
        if len(cols) == 1:
            out[r, c] = next(iter(cols))
        else:
            out[r, c] = int(bg)
    return out


def _arrow_points_from_start(sr: int, sc: int, dr: int, dc: int) -> list[tuple[int, int]]:
    mr, mc = sr - dr, sc - dc
    return [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]


def _in_bounds(H: int, W: int, r: int, c: int) -> bool:
    return 0 <= int(r) < H and 0 <= int(c) < W


def _straight_ray_cells(H: int, W: int, sr: int, sc: int, dr: int, dc: int) -> list[tuple[int, int]]:
    cur_r, cur_c = sr, sc
    out = []
    while True:
        nr, nc = cur_r + dr, cur_c + dc
        if not _in_bounds(H, W, nr, nc):
            break
        out.append((int(nr), int(nc)))
        cur_r, cur_c = nr, nc
    return out


def _plan_crossing_pair(
    H: int,
    W: int,
    colors: list[int],
) -> tuple[list[tuple[int, list[tuple[int, int]]]], list[set[tuple[int, int]]]] | None:
    direction_pairs = [
        ((1, 1), (1, -1)),
        ((1, 1), (-1, 1)),
        ((-1, -1), (1, -1)),
        ((-1, -1), (-1, 1)),
    ]
    for _ in range(120):
        pr = _rng.randint(5, H - 6)
        pc = _rng.randint(5, W - 6)
        (d1, d2) = _rng.choice(direction_pairs)
        ks = [_rng.randint(3, 6), _rng.randint(3, 6)]
        arrow_defs: list[tuple[int, list[tuple[int, int]]]] = []
        ray_sets: list[set[tuple[int, int]]] = []
        structural: set[tuple[int, int]] = set()
        ok = True
        for color, (dr, dc), k in zip(colors, (d1, d2), ks):
            sr = pr - k * dr
            sc = pc - k * dc
            pts = _arrow_points_from_start(sr, sc, dr, dc)
            if any(not _in_bounds(H, W, r, c) for r, c in pts):
                ok = False
                break
            ray_cells = _straight_ray_cells(H, W, sr, sc, dr, dc)
            ray_set = set(ray_cells)
            if (pr, pc) not in ray_set or len(ray_cells) < k + 3:
                ok = False
                break
            if structural & set(pts):
                ok = False
                break
            arrow_defs.append((int(color), pts))
            ray_sets.append(ray_set)
            structural |= set(pts)
        if not ok:
            continue
        if any(set(pts) & other for _, pts in arrow_defs for other in ray_sets):
            continue
        if len(ray_sets[0] & ray_sets[1]) != 1 or (pr, pc) not in (ray_sets[0] & ray_sets[1]):
            continue
        return arrow_defs, ray_sets
    return None


def _plan_random_bouncy_arrow(
    H: int,
    W: int,
    color: int,
    palette: list[int],
    structural_cells: set[tuple[int, int]],
    reserved_ray_cells: set[tuple[int, int]],
):
    for _ in range(120):
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        n_bounces = _rng.choices([1, 2, 3, 4], weights=[3, 4, 2, 1])[0]

        r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
        c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
        if r_lo > r_hi or c_lo > c_hi:
            continue
        sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)
        arrow_pts = _arrow_points_from_start(sr, sc, dr, dc)
        if any(not _in_bounds(H, W, r, c) for r, c in arrow_pts):
            continue
        arrow_set = set(arrow_pts)
        if any((r, c) in structural_cells or (r, c) in reserved_ray_cells for r, c in arrow_pts):
            continue

        cur_r, cur_c, cur_dr, cur_dc, cur_col = sr, sc, dr, dc, color
        ray_cells: list[tuple[int, int, int]] = []
        walls: list[tuple[list[tuple[int, int]], int]] = []
        ray_set: set[tuple[int, int]] = set()
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
                if not _in_bounds(H, W, nr, nc):
                    break
                if (nr, nc) in structural_cells or (nr, nc) in arrow_set or (nr, nc) in ray_set:
                    ok = False
                    break
                ray_cells.append((int(nr), int(nc), int(cur_col)))
                ray_set.add((int(nr), int(nc)))
                cur_r, cur_c = nr, nc
            if not ok:
                break

            if seg_idx >= n_bounces:
                continue

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
            chosen_wall = None
            for wpxs, flip in options:
                if any(
                    (r, c) in structural_cells
                    or (r, c) in reserved_ray_cells
                    or (r, c) in ray_set
                    or (r, c) in arrow_set
                    for r, c in wpxs
                ):
                    continue
                chosen_wall = (wpxs, flip)
                break
            if chosen_wall is None:
                ok = False
                break

            wpxs, flip = chosen_wall
            wcol = _rng.choice([p for p in palette if p != color])
            walls.append((wpxs, int(wcol)))
            if ray_cells:
                lr, lc, _ = ray_cells[-1]
                ray_cells[-1] = (int(lr), int(lc), int(wcol))
            cur_col = int(wcol)
            if flip == "dr":
                cur_dr = -cur_dr
            else:
                cur_dc = -cur_dc

        if not ok or len(ray_cells) < 4 or not walls:
            continue

        wall_cells = {cell for wpxs, _ in walls for cell in wpxs}
        if wall_cells & structural_cells or wall_cells & reserved_ray_cells:
            continue
        if arrow_set & ray_set or arrow_set & wall_cells or ray_set & wall_cells:
            continue
        return {
            "arrow": (int(color), arrow_pts),
            "ray_set": ray_set,
            "walls": walls,
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    palette = list(range(1, 10))
    _rng.shuffle(palette)

    crossing = _plan_crossing_pair(H, W, palette[:2])
    if crossing is None:
        return None
    arrow_defs, base_ray_sets = crossing

    grid = np.zeros((H, W), dtype=int)
    structural_cells: set[tuple[int, int]] = set()
    reserved_ray_cells: set[tuple[int, int]] = set()

    for color, pts in arrow_defs:
        for r, c in pts:
            grid[r, c] = int(color)
            structural_cells.add((r, c))
    for ray_set in base_ray_sets:
        reserved_ray_cells |= set(ray_set)

    n_extra = _rng.randint(1, 2)
    used_colors = {color for color, _ in arrow_defs}
    extra_colors = [col for col in palette if col not in used_colors]
    extra_plans = []
    for color in extra_colors[:n_extra]:
        plan = _plan_random_bouncy_arrow(H, W, color, palette, structural_cells, reserved_ray_cells)
        if plan is None:
            return None
        extra_plans.append(plan)
        acol, apts = plan["arrow"]
        for r, c in apts:
            grid[r, c] = int(acol)
            structural_cells.add((r, c))
        for wpxs, wcol in plan["walls"]:
            for r, c in wpxs:
                grid[r, c] = int(wcol)
                structural_cells.add((r, c))
        reserved_ray_cells |= set(plan["ray_set"])

    summary = _overlap_summary(grid)
    mixed_count = len(summary["mixed_cells"])
    if mixed_count < 1 or mixed_count > 6:
        return None
    if not summary["wall_cells"]:
        return None

    phase1 = _phase1_union(grid, summary["ray_paints"])
    final = main(grid)
    erased = int(np.count_nonzero((phase1 != final) & (final == summary["bg"])))
    if erased < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
