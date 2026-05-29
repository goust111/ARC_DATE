# Combo variant of 142ca369 (index 26)
# combo_spec: D3/3+D10/2
# Name: 仅反射点涂色+碰撞环绕涂色

from __future__ import annotations

import random as _rng
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import colors_present, components_mask, get_bg, neighbors8


# Design analysis for combo D3/3 + D10/2
# =======================================
# 1. Concretization and behavior trace
#    - Original 142ca369 paints every free-flight ray cell, recolors the bounce
#      point to the wall color, then continues in the new color.
#    - D3/3 removes continuous painting. In this combo we take the combo-level
#      wording literally: the ray is invisible during free flight and only the
#      reflection point itself is painted.
#    - D10/2 adds a surround effect at each collision: every 8-neighbor
#      background cell around the whole touched wall component becomes the
#      incoming ray color.
#    - Local example for a side-touch bounce:
#        input:  blue arrow ... invisible diagonal flight ... orange wall
#        original output:
#          - all traversed diagonal cells become blue,
#          - bounce cell becomes orange,
#          - outgoing diagonal cells become orange.
#        combo output:
#          - all free-flight diagonal cells stay background,
#          - only the bounce cell becomes orange,
#          - the wall's surrounding background cells become blue,
#          - outgoing free-flight cells also stay background.
#      The visible result is a colored halo wrapped around the wall plus a single
#      recolored contact pixel, with blank space between arrows and impacts.
#
# 2. Complexity calibration
#    - The combo is visually much sparser than the original, so the generator
#      forces at least one bounce per arrow and keeps invisible segments long
#      enough that the lack of trail is obvious.
#    - The transform remains categorical and local:
#        halo color   = incoming ray color
#        bounce color = wall color
#      This keeps the rule learnable from 2-4 examples.
#    - We keep the original wall vocabulary (1-pixel or 3-pixel axis-aligned
#      walls) so the task still reads as a modified form of the source puzzle.
#
# 3. Implicit assumption audit for main()
#    - Original assumption broken: every visited ray cell should be painted.
#      Not true anymore, so geometry tracing must be separated from painting.
#    - New assumption introduced by surround: a touched wall cell must map to a
#      well-defined wall component. The generator guarantees walls never touch
#      orthogonally, so each wall component is exactly one intended segment.
#    - New precedence rule: the bounce cell is adjacent to the wall and would
#      belong to the halo footprint, but combo semantics require it to remain the
#      special contact mark. Therefore halo painting explicitly excludes the
#      bounce cell.
#    - Sequential side-touch handling from the original solver is preserved. The
#      generator avoids dual side-touch situations at the same ray step, but the
#      solver still behaves coherently if such an input appears.
#
# 4. Placement legality in the generator
#    - Arrows, invisible ray paths, walls, bounce cells, and halo cells are all
#      reserved and pairwise disjoint except for the intended wall-adjacent
#      relation at each bounce.
#    - Candidate walls are kept away from borders so the halo footprint is not
#      clipped and remains easy to see.
#    - Future segments are checked against all reserved cells, so no later path
#      crosses an earlier invisible path or halo.
#
# 5. Visual observability and neatness
#    - Each sample uses 2-4 arrows and 1-3 bounces per arrow to create multiple
#      isolated glow events without overcrowding the board.
#    - Long blank gaps before reflections make the D3 change human-visible.
#    - Clean halo reservation prevents overlap-driven clutter, so the wall,
#      bounce point, and surrounding glow remain visually separable.
#
# 6. Color inferability
#    - The glow color is structurally tied to the incoming arrow/ray color.
#    - The bounce point color is structurally tied to the wall color.
#    - The generator often uses different wall colors from incoming colors so the
#      two roles are directly comparable in the examples.

MIN_BOUNCE_SEG_LEN = 5
MIN_FINAL_SEG_LEN = 3


def _is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[tuple[int, int]]):
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


def _detect_arrows(g: np.ndarray, bg: int):
    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)
    return arrows, arrow_cells


def _wall_data(g: np.ndarray, bg: int, arrow_cells: set[tuple[int, int]]):
    wall_mask = (g != int(bg)).copy()
    for r, c in arrow_cells:
        wall_mask[r, c] = False
    wall_components: list[list[tuple[int, int]]] = []
    wall_component_by_cell: dict[tuple[int, int], int] = {}
    for idx, coords in enumerate(components_mask(wall_mask, connectivity=4)):
        comp = [(int(r), int(c)) for r, c in coords]
        wall_components.append(comp)
        for cell in comp:
            wall_component_by_cell[cell] = idx
    wall_cells = set(wall_component_by_cell)
    return wall_cells, wall_components, wall_component_by_cell


def _paint_surround(
    out: np.ndarray,
    g: np.ndarray,
    bg: int,
    wall_component: list[tuple[int, int]],
    halo_color: int,
    exclude: set[tuple[int, int]] | None = None,
) -> None:
    H, W = g.shape
    blocked = exclude or set()
    wall_set = set(wall_component)
    for wr, wc in wall_component:
        for rr, cc in neighbors8(wr, wc):
            rr = int(rr)
            cc = int(cc)
            cell = (rr, cc)
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if cell in wall_set or cell in blocked:
                continue
            if int(g[rr, cc]) != int(bg):
                continue
            if int(out[rr, cc]) != int(bg):
                continue
            out[rr, cc] = int(halo_color)


def _solve(grid: np.ndarray, *, paint_trail: bool, use_surround: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    arrows, arrow_cells = _detect_arrows(g, bg)
    wall_cells, wall_components, wall_component_by_cell = _wall_data(g, bg, arrow_cells)
    out = g.copy()

    def wall_component(cell: tuple[int, int]) -> list[tuple[int, int]]:
        return wall_components[wall_component_by_cell[cell]]

    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
        cur_col = int(start_col)
        seen: set[tuple[tuple[int, int], int, int, int]] = set()

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
                wcell = (nr, nc)
                wcol = int(g[wcell])
                if use_surround:
                    _paint_surround(out, g, bg, wall_component(wcell), cur_col, exclude={cur})
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if paint_trail and int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            if int(dr) == 1 and (nr + 1, nc) in wall_cells:
                wcell = (nr + 1, nc)
                wcol = int(g[wcell])
                if use_surround:
                    _paint_surround(out, g, bg, wall_component(wcell), cur_col, exclude={cur})
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)
            elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                wcell = (nr - 1, nc)
                wcol = int(g[wcell])
                if use_surround:
                    _paint_surround(out, g, bg, wall_component(wcell), cur_col, exclude={cur})
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)

            if int(dc) == 1 and (nr, nc + 1) in wall_cells:
                wcell = (nr, nc + 1)
                wcol = int(g[wcell])
                if use_surround:
                    _paint_surround(out, g, bg, wall_component(wcell), cur_col, exclude={cur})
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)
            elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                wcell = (nr, nc - 1)
                wcol = int(g[wcell])
                if use_surround:
                    _paint_surround(out, g, bg, wall_component(wcell), cur_col, exclude={cur})
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, paint_trail=False, use_surround=True)


def _original_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, paint_trail=True, use_surround=False)


def _continuous_surround_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, paint_trail=True, use_surround=True)


def _steps_to_edge(r: int, c: int, dr: int, dc: int, H: int, W: int) -> int:
    return min((H - 1 - r) if dr == 1 else r, (W - 1 - c) if dc == 1 else c)


def _wall_halo_cells(
    H: int,
    W: int,
    wall_cells: list[tuple[int, int]],
    exclude: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    halo: set[tuple[int, int]] = set()
    wall_set = set(wall_cells)
    for wr, wc in wall_cells:
        for rr, cc in neighbors8(wr, wc):
            rr = int(rr)
            cc = int(cc)
            cell = (rr, cc)
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if cell in wall_set or cell in exclude:
                continue
            halo.add(cell)
    return halo


def _candidate_wall_options(
    bounce_r: int,
    bounce_c: int,
    cur_dr: int,
    cur_dc: int,
    H: int,
    W: int,
):
    options: list[tuple[list[tuple[int, int]], str]] = []

    wr, wc = bounce_r + cur_dr, bounce_c
    if 1 <= wr < H - 1:
        if 1 <= wc < W - 1:
            options.append(([(wr, wc)], "dr"))
        if 2 <= wc <= W - 3:
            options.append(([(wr, wc - 1), (wr, wc), (wr, wc + 1)], "dr"))

    wr2, wc2 = bounce_r, bounce_c + cur_dc
    if 1 <= wc2 < W - 1:
        if 1 <= wr2 < H - 1:
            options.append(([(wr2, wc2)], "dc"))
        if 2 <= wr2 <= H - 3:
            options.append(([(wr2 - 1, wc2), (wr2, wc2), (wr2 + 1, wc2)], "dc"))
    return options


def _pick_color(palette: list[int], exclude: int) -> int:
    choices = [int(c) for c in palette if int(c) != int(exclude)]
    if not choices:
        return int(exclude)
    return int(_rng.choice(choices))


def _render_plan_output(grid: np.ndarray, plans: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    for plan in plans:
        for bounce in plan["bounces"]:
            br, bc = bounce["bounce_cell"]
            if int(grid[br, bc]) == 0:
                out[br, bc] = int(bounce["wall_color"])
            for r, c in bounce["halo_cells"]:
                if int(grid[r, c]) == 0:
                    out[r, c] = int(bounce["incoming_color"])
    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    combo = main(g)
    original = _original_main(g)
    continuous_surround = _continuous_surround_main(g)
    hidden_flight_cells = int(np.sum((g == bg) & (combo == bg) & (continuous_surround != bg)))
    halo_cells = int(np.sum((g == bg) & (combo != bg) & (original == bg)))
    combo_marks = int(np.sum((g == bg) & (combo != bg)))
    return {
        "hidden_flight_cells": hidden_flight_cells,
        "halo_cells": halo_cells,
        "combo_marks": combo_marks,
    }


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(2, 4)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]

    grid = np.zeros((H, W), dtype=int)
    reserved: set[tuple[int, int]] = set()
    plans: list[dict] = []

    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        n_bounces = _rng.choices([1, 2, 3], weights=[2, 4, 2])[0]

        placed = False
        for _ in range(180):
            r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
            c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)

            mr, mc = sr - dr, sc - dc
            arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
            arrow_set = set(arrow_pts)
            if any(not (0 <= r < H and 0 <= c < W) or (r, c) in reserved for r, c in arrow_pts):
                continue

            cur_r, cur_c = sr, sc
            cur_dr, cur_dc = dr, dc
            cur_col = color
            path: list[tuple[int, int, int]] = []
            ray_set: set[tuple[int, int]] = set()
            wall_set: set[tuple[int, int]] = set()
            halo_set: set[tuple[int, int]] = set()
            bounces: list[dict] = []
            ok = True

            for bounce_idx in range(n_bounces):
                max_steps = _steps_to_edge(cur_r, cur_c, cur_dr, cur_dc, H, W)
                if max_steps < MIN_BOUNCE_SEG_LEN:
                    ok = False
                    break

                lengths = list(range(MIN_BOUNCE_SEG_LEN, max_steps + 1))
                _rng.shuffle(lengths)
                candidates = []
                for seg_len in lengths:
                    seg_cells = [
                        (cur_r + cur_dr * step, cur_c + cur_dc * step)
                        for step in range(1, seg_len + 1)
                    ]
                    seg_set = set(seg_cells)
                    if any(cell in reserved or cell in arrow_set or cell in ray_set or cell in wall_set or cell in halo_set
                           for cell in seg_cells):
                        continue

                    bounce_r, bounce_c = seg_cells[-1]
                    for wall_cells, flip in _candidate_wall_options(bounce_r, bounce_c, cur_dr, cur_dc, H, W):
                        halo = _wall_halo_cells(H, W, wall_cells, exclude={(bounce_r, bounce_c)})
                        if not halo:
                            continue
                        taken = reserved | arrow_set | ray_set | wall_set | halo_set | seg_set
                        if any(cell in taken for cell in wall_cells):
                            continue
                        if any(cell in taken for cell in halo):
                            continue
                        alt_touch = (bounce_r, bounce_c + cur_dc) if flip == "dr" else (bounce_r + cur_dr, bounce_c)
                        if alt_touch in taken:
                            continue

                        next_dr = -cur_dr if flip == "dr" else cur_dr
                        next_dc = -cur_dc if flip == "dc" else cur_dc
                        future_need = MIN_BOUNCE_SEG_LEN if bounce_idx < n_bounces - 1 else MIN_FINAL_SEG_LEN
                        if _steps_to_edge(bounce_r, bounce_c, next_dr, next_dc, H, W) < future_need:
                            continue

                        candidates.append((seg_cells, wall_cells, halo, next_dr, next_dc))

                if not candidates:
                    ok = False
                    break

                _rng.shuffle(candidates)
                candidates.sort(
                    key=lambda item: (len(item[1]), len(item[2])),
                    reverse=(_rng.random() < 0.7),
                )
                seg_cells, wall_cells, halo, next_dr, next_dc = candidates[0]
                wall_color = _pick_color(palette, cur_col)

                for r, c in seg_cells:
                    path.append((int(r), int(c), int(cur_col)))
                    ray_set.add((int(r), int(c)))
                for cell in wall_cells:
                    wall_set.add((int(cell[0]), int(cell[1])))
                for cell in halo:
                    halo_set.add((int(cell[0]), int(cell[1])))

                bounces.append(
                    {
                        "bounce_cell": (int(seg_cells[-1][0]), int(seg_cells[-1][1])),
                        "incoming_color": int(cur_col),
                        "wall_color": int(wall_color),
                        "wall_cells": [(int(r), int(c)) for r, c in wall_cells],
                        "halo_cells": sorted((int(r), int(c)) for r, c in halo),
                    }
                )

                cur_r, cur_c = seg_cells[-1]
                cur_dr, cur_dc = next_dr, next_dc
                cur_col = int(wall_color)

            if not ok:
                continue

            final_steps = _steps_to_edge(cur_r, cur_c, cur_dr, cur_dc, H, W)
            if final_steps < MIN_FINAL_SEG_LEN:
                continue
            final_cells = [
                (cur_r + cur_dr * step, cur_c + cur_dc * step)
                for step in range(1, final_steps + 1)
            ]
            if any(cell in reserved or cell in arrow_set or cell in ray_set or cell in wall_set or cell in halo_set
                   for cell in final_cells):
                continue
            for r, c in final_cells:
                path.append((int(r), int(c), int(cur_col)))
                ray_set.add((int(r), int(c)))

            new_reserved = arrow_set | ray_set | wall_set | halo_set
            expected = len(arrow_set) + len(ray_set) + len(wall_set) + len(halo_set)
            if len(new_reserved) != expected:
                continue
            if new_reserved & reserved:
                continue

            for bounce in bounces:
                wall_cells = bounce["wall_cells"]
                if any(cell in arrow_set or cell in ray_set for cell in wall_cells):
                    ok = False
                    break
                if set(wall_cells) & set(bounce["halo_cells"]):
                    ok = False
                    break
            if not ok:
                continue

            for r, c in arrow_pts:
                grid[r, c] = int(color)
            for bounce in bounces:
                for r, c in bounce["wall_cells"]:
                    grid[r, c] = int(bounce["wall_color"])

            reserved |= new_reserved
            plans.append(
                {
                    "arrow_pts": [(int(r), int(c)) for r, c in arrow_pts],
                    "path": path,
                    "bounces": bounces,
                }
            )
            placed = True
            break

        if not placed:
            return None

    expected = _render_plan_output(grid, plans)
    metrics = _behavior_metrics(grid)
    if metrics["hidden_flight_cells"] <= 0:
        return None
    if metrics["halo_cells"] <= 0:
        return None
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    if np.array_equal(main(grid), _original_main(grid)):
        return None
    if np.array_equal(main(grid), _continuous_surround_main(grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
