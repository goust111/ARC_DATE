# Variant of 142ca369
# Dimension: D10 — 反射后变换效果 (transform)
# Change: 反射点重新涂色为墙壁颜色（recolor），射线继承墙壁颜色... → 碰撞时产生环绕涂色
#
# Design analysis for the D10 variant
# ==================================
# 0. Reference-task status
#    - The prompt asks to read the solver of source task 71e489b6 first.
#    - A repository-wide search in the provided workspace found only the official
#      I/O file `ARC-data/ARC-AGI-2/data/evaluation/71e489b6.json`, not a solver.
#    - Therefore the "surround" mechanism below is calibrated from:
#        1) the explicit variant request text,
#        2) the visible 71e489b6 I/O pattern ("surround" = paint background cells
#           adjacent to a target object),
#        3) the original 142ca369 solver structure.
#
# 1. Concretized behavior and flow impact
#    - Original task:
#      * monochrome 3-cell L arrows fire diagonal rays;
#      * a side-touch wall bounce recolors the bounce cell to wall color;
#      * the ray then inherits the wall color and continues.
#    - New D10 task:
#      * keep the same arrow detection, ray geometry, side-touch / corner bounce
#        logic, and color inheritance;
#      * at each bounce, additionally paint a halo around the entire wall segment:
#        every 8-neighbor background cell of that wall segment is painted in the
#        incoming ray color (the color just before the bounce);
#      * the bounce point still changes to the wall color, so the local pattern is:
#          incoming ray color -> halo around wall
#          wall color         -> wall itself and bounce point
#          inherited wall color -> outgoing ray
#    - Small trace example for one side-touch bounce:
#      * Suppose the ray color is blue and it advances to cell P.
#      * A vertical orange wall segment lies at P's east side, so dc flips.
#      * Original output:
#          - P becomes orange,
#          - outgoing ray is orange,
#          - no other local change.
#      * Variant output:
#          - first, every background cell in the 8-neighborhood of that wall
#            segment becomes blue (incoming color),
#          - then P becomes orange,
#          - outgoing ray is orange.
#      * Difference: wall collisions now leave a visible colored glow that encodes
#        the pre-bounce ray color.
#
# 2. Complexity calibration
#    - The transform stays local and categorical: each bounce has exactly one halo
#      color, namely the incoming ray color. There is no arbitrary palette mapping.
#    - To make the change visible, the generator requires real bounces and reserves
#      empty space around walls so the halo appears as a clean ring/patch.
#    - We still keep wall shapes to the original 1-pixel or 3-pixel axis-aligned
#      segments. This preserves the original object vocabulary.
#
# 3. Implicit-assumption audit against the original main()
#    - Arrow detection per color still works unchanged because the variant does not
#      alter the input vocabulary.
#    - `wall_cells` in the original solver were enough for recoloring, but not for
#      surround. The variant needs wall components so that a 3-pixel wall is
#      surrounded as one object instead of only around the single touched cell.
#    - The original recolor guards `if g[cell] == bg` remain necessary. They prevent
#      recoloring arrow pixels in immediate corner-hit situations.
#    - The original solver only painted empty output cells along the ray. That is
#      still safe because the generator prevents ray/halo overlaps. The surround
#      painter also only writes cells that were background in the input and are
#      still background in the output, so no object is overwritten.
#    - Bounce ordering matters if both dr and dc touches happen at the same ray cell.
#      The original sequential logic is preserved. Each bounce uses the current ray
#      color at that moment, then updates the ray color to the touched wall color.
#
# 4. Placement legality requirements for the generator
#    - Besides the original no-overlap constraints for arrows, rays, and walls, the
#      generator now treats every planned halo cell as reserved output territory.
#    - Future arrows, rays, walls, and later halos may not occupy those cells.
#    - Each chosen wall option is validated with its full halo footprint before it is
#      committed. This guarantees that `main()` can paint the surround effect without
#      collisions or ambiguous overwrites.
#
# 5. Visual observability
#    - A sample with zero bounces would hide the D10 change entirely, so the
#      generator always includes at least one bounce per arrow and rejects any final
#      instance where the variant output equals the original-task output.
#    - The halo color differs from the wall color exactly at collision time, so the
#      observer can infer "incoming color wraps around the obstacle, outgoing color
#      matches the obstacle."
#
# 6. Visual cleanliness
#    - Halos are only placed on previously empty cells and are kept disjoint from
#      all other structures. This avoids cluttered blends that would obscure the
#      arrows or the walls.
#    - Wall options are biased away from borders because border-clipped halos are
#      less legible.
#
# 7. Color-change inferability
#    - The new color rule is local and structurally grounded:
#      * halo color = incoming ray color,
#      * bounce point + outgoing ray color = wall color.
#    - This is inferable from a few examples because the halo is attached directly
#      to the collided wall and uses the immediately preceding ray color.
#
# 8. New assumptions introduced by the variant main(), and how the generator
#    guarantees them
#    - Assumption: each touched wall cell belongs to a well-defined wall component.
#      Guarantee: the generator never lets separate walls touch orthogonally.
#    - Assumption: halo cells that should be painted are background in the input.
#      Guarantee: the generator reserves halo cells and never places objects there.
#    - Assumption: output does not depend on accidental cross-arrow overwrite order.
#      Guarantee: halos, rays, arrows, and walls are pairwise disjoint in output,
#      except for the intended bounce-point recolor.

from __future__ import annotations

import random as _rng
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import colors_present, components_mask, get_bg, neighbors8


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
    wall_components = []
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
    for wr, wc in wall_component:
        for rr, cc in neighbors8(wr, wc):
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if (int(rr), int(cc)) in blocked:
                continue
            if int(g[rr, cc]) != int(bg):
                continue
            if int(out[rr, cc]) != int(bg):
                continue
            out[rr, cc] = int(halo_color)


def _solve(grid: np.ndarray, *, use_surround: bool) -> np.ndarray:
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
        seen = set()

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
                if use_surround:
                    _paint_surround(
                        out,
                        g,
                        bg,
                        wall_component((nr, nc)),
                        cur_col,
                        exclude={cur},
                    )
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
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
    return _solve(grid, use_surround=True)


def _original_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, use_surround=False)


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
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            cell = (rr, cc)
            if cell in wall_set or cell in exclude:
                continue
            halo.add(cell)
    return halo


def _choose_wall_option(
    H: int,
    W: int,
    cur_r: int,
    cur_c: int,
    cur_dr: int,
    cur_dc: int,
    reserved: set[tuple[int, int]],
    arrow_set: set[tuple[int, int]],
    ray_set: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], str, set[tuple[int, int]]] | None:
    options: list[tuple[list[tuple[int, int]], str, set[tuple[int, int]]]] = []

    def maybe_add(wall_cells: list[tuple[int, int]], flip: str, contact: tuple[int, int]) -> None:
        halo = _wall_halo_cells(H, W, wall_cells, exclude={contact})
        if not halo:
            return
        taken = reserved | arrow_set | ray_set
        if any(cell in taken for cell in wall_cells):
            return
        if any(cell in taken for cell in halo):
            return
        if halo & set(wall_cells):
            return
        options.append((wall_cells, flip, halo))

    wr, wc = cur_r + cur_dr, cur_c
    if 1 <= wr < H - 1:
        if 1 <= wc < W - 1:
            maybe_add([(wr, wc)], "dr", (cur_r, cur_c))
        if 1 <= wc - 1 < W - 1 and 1 <= wc + 1 < W - 1:
            maybe_add([(wr, wc - 1), (wr, wc), (wr, wc + 1)], "dr", (cur_r, cur_c))

    wr2, wc2 = cur_r, cur_c + cur_dc
    if 1 <= wc2 < W - 1:
        if 1 <= wr2 < H - 1:
            maybe_add([(wr2, wc2)], "dc", (cur_r, cur_c))
        if 1 <= wr2 - 1 < H - 1 and 1 <= wr2 + 1 < H - 1:
            maybe_add([(wr2 - 1, wc2), (wr2, wc2), (wr2 + 1, wc2)], "dc", (cur_r, cur_c))

    if not options:
        return None

    _rng.shuffle(options)
    options.sort(key=lambda item: (len(item[0]), len(item[2])), reverse=(_rng.random() < 0.7))
    return options[0]


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(1, 5)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = [_rng.choice(palette) for _ in range(n_arrows)]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    reserved: set[tuple[int, int]] = set()
    total_bounces = 0
    total_halo_cells = 0

    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        n_bounces = _rng.choices([1, 2, 3, 4], weights=[3, 4, 2, 1])[0]

        placed = False
        for _ in range(120):
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
            ray_cells: list[tuple[int, int, int]] = []
            ray_set: set[tuple[int, int]] = set()
            walls: list[tuple[list[tuple[int, int]], int]] = []
            halos: list[tuple[set[tuple[int, int]], int]] = []
            ok = True

            for seg_idx in range(n_bounces + 1):
                max_steps = min(
                    (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                    (W - 1 - cur_c) if cur_dc == 1 else cur_c,
                )
                if max_steps <= 0:
                    ok = False
                    break

                if seg_idx < n_bounces:
                    if max_steps < 5:
                        ok = False
                        break
                    seg_len = _rng.randint(3, max(3, max_steps - 2))
                else:
                    seg_len = max_steps

                for _step in range(seg_len):
                    nr, nc = cur_r + cur_dr, cur_c + cur_dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        ok = False
                        break
                    cell = (nr, nc)
                    if cell in reserved or cell in arrow_set or cell in ray_set:
                        ok = False
                        break
                    ray_cells.append((nr, nc, cur_col))
                    ray_set.add(cell)
                    cur_r, cur_c = nr, nc
                if not ok:
                    break

                if seg_idx == n_bounces:
                    continue

                chosen = _choose_wall_option(H, W, cur_r, cur_c, cur_dr, cur_dc, reserved, arrow_set, ray_set)
                if chosen is None:
                    ok = False
                    break
                wall_cells, flip, halo = chosen
                wcol = int(_rng.choice(palette))
                walls.append((wall_cells, wcol))
                halos.append((halo, cur_col))

                if ray_cells:
                    lr, lc, _ = ray_cells[-1]
                    ray_cells[-1] = (lr, lc, wcol)
                cur_col = wcol
                if flip == "dr":
                    cur_dr = -cur_dr
                else:
                    cur_dc = -cur_dc

            if not ok or len(ray_cells) < 3 or not walls:
                continue

            wall_set = set()
            halo_union = set()
            for wall_cells, _wcol in walls:
                for cell in wall_cells:
                    if cell in wall_set:
                        ok = False
                    wall_set.add(cell)
            for halo_cells, _hcol in halos:
                if halo_union & halo_cells:
                    ok = False
                halo_union |= halo_cells
            if not ok:
                continue

            new_output_cells = arrow_set | ray_set | wall_set | halo_union
            expected = len(arrow_set) + len(ray_set) + len(wall_set) + len(halo_union)
            if len(new_output_cells) != expected:
                continue
            if new_output_cells & reserved:
                continue

            for r, c in arrow_pts:
                grid[r, c] = out[r, c] = color
            for r, c, col in ray_cells:
                out[r, c] = col
            for wall_cells, wcol in walls:
                for r, c in wall_cells:
                    grid[r, c] = out[r, c] = wcol
            for halo_cells, hcol in halos:
                for r, c in halo_cells:
                    out[r, c] = hcol

            reserved |= new_output_cells
            total_bounces += len(walls)
            total_halo_cells += len(halo_union)
            placed = True
            break

        if not placed:
            return None

    if total_bounces == 0 or total_halo_cells == 0:
        return None
    if not np.array_equal(main(grid), out):
        return None
    if np.array_equal(main(grid), _original_main(grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
