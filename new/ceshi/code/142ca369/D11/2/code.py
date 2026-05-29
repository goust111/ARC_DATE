# Variant of 142ca369
# Dimension: D11 — 箭头数量与密度 (meta)
# Change: 1-14个箭头，每个独立发射射线，箭头间无交互... → 箭头颜色分组

from __future__ import annotations

import random as _rng
from collections import defaultdict

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

"""
Design analysis for the D11 color-grouping variant
==================================================

Reference-task note
-------------------
- The prompt cites `b5ca7ac4` as inspiration for color grouping.
- This workspace does not contain a local solver file for `b5ca7ac4`; only
  `ARC-data/ARC-AGI-2/data/evaluation/b5ca7ac4.json` is available.
- Therefore the implementation below uses the visible "same color behaves as a
  shared identity" idea as a baseline, but does not import any unseen solver logic.

Variant spec
------------
variant_spec:
  grid: 18-20 rows x 18-20 cols; bg = 0
  arrows:
    shape: same as original task: 3 cells in a 2x2 L-shape
    count: usually 4-5 in generated data
    grouping: arrows with the same input color belong to the same ray group
  walls:
    shape: unchanged: 1-cell or 3-cell axis-aligned segments
    role: unchanged: side-touch reflection and color inheritance
  rays:
    identity: group id = original arrow color; this never changes
    paint_color: starts at arrow color and still changes when walls are hit
    interaction:
      same_group: may occupy / traverse the same trajectory cells
      different_group:
        - if two groups try to enter the same empty cell on the same tick, both stop
          and that cell stays unpainted
        - if a ray tries to enter a cell already traversed by another group on an
          earlier tick, it stops before entering
    other rules: arrow blocking, wall reflection, and wall-color inheritance stay
      as close as possible to the original solver

Concrete behavior tracking
--------------------------
- Original logic: arrow A is traced fully, then arrow B is traced fully. Rays never
  read any "other ray occupied this path cell" state, so every pair of rays behaves
  as if the other did not exist.
- New logic: all arrows advance one diagonal step per tick. We keep a separate
  `ray_owner` map from path-cell -> arrow-group color.
  Example:
    - Same-color pair:
      A(blue) from NW and B(blue) from SW both reach center `p`.
      Because both have group `blue`, both enter `p` and continue past it.
    - Different-color pair:
      A(red) from NE and B(green) from SE both aim at center `q`.
      They propose `q` on the same tick. Because the proposed groups differ,
      both stop, `q` stays background, and neither branch continues.
- The important difference is that "paint color" can change on wall hits, but
  "group identity" must remain the original arrow color. Otherwise a bounced ray
  would silently change teams, which would make the grouping rule unreadable.

Complexity calibration
----------------------
- The generator always includes:
  1. one same-color crossing pair that visibly passes through the intersection
  2. one different-color collision pair that visibly truncates before its would-be
     intersection
- This gives repeated evidence for both branches of the new rule in the same grid.
- Optional extra bounced arrows preserve the original reflection mechanic without
  letting the task collapse into "only line crossing".

Implicit-assumption audit against the original `main()`
------------------------------------------------------
- `for start_col, pts in arrows:` with a full per-arrow trace was safe only because
  rays were independent. In the variant it becomes wrong, because later rays would
  see earlier rays only through accidental paint order rather than through a rule.
- `cur_col` in the original solver was both the current paint color and the only
  identity carried by the ray. That is no longer sufficient. We must track:
    - `group_col`: fixed identity used for collision checks
    - `cur_col`: mutable drawing color after wall hits
- `if out[nr, nc] == bg: out[nr, nc] = cur_col` remains valid for ordinary painting,
  but it can no longer tell us whether a foreign ray has already claimed the cell,
  because two different groups may paint the same wall-derived color. Hence the
  explicit `ray_owner` map.
- Arrow detection still relies on per-color connected components. The generator
  therefore avoids giving walls the same colors as arrows, so an arrow is never
  merged with a same-colored wall component in the input.

Placement legality and solver assumptions guaranteed by the generator
---------------------------------------------------------------------
- Same-group crossing pair:
  two arrows share one output intersection cell and nothing else.
- Different-group collision pair:
  two arrows approach one target cell at equal path length so the collision is
  simultaneous and order-independent.
- Reserved path cells are treated as occupied during generation, so optional extra
  arrows and walls do not accidentally create hidden collisions with the showcase
  pairs.
- Optional bounced arrows use the original side-touch wall placement pattern only;
  the generator does not rely on corner-bounce construction.

Visual observability / neatness
-------------------------------
- The same-color pair forms a full X-like crossing.
- The different-color pair forms two truncated diagonals with a visible gap at the
  blocked meeting point.
- The repeated color on the crossing pair is the only repeated arrow color in most
  samples, which makes the grouping signal legible from 2-4 examples.

New solver assumptions introduced by the variant
------------------------------------------------
- Same-group rays may share cells. The solver resolves this deterministically even
  if multiple same-group rays enter the same cell on one tick. The generator keeps
  those shared cells monochrome by only using unbounced same-group showcase pairs.
- Different-group collisions are defined on trajectory cells, not on wall cells or
  arrow cells. The generator respects that by keeping showcase intersections away
  from walls and source arrows.
"""


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


def _arrow_pts_from_start(sr: int, sc: int, dr: int, dc: int) -> list[tuple[int, int]]:
    mr, mc = sr - dr, sc - dc
    return [(int(r), int(c)) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]


def _detect_arrows_and_walls(grid: np.ndarray) -> tuple[int, list[dict], set[tuple[int, int]]]:
    g = np.array(grid, dtype=int)
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
                    "group_col": int(col),
                    "start_col": int(col),
                    "pts": pts,
                    "start": (int(start[0]), int(start[1])),
                    "dir": (int(dr), int(dc)),
                }
            )
            arrow_cells |= set(pts)

    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }
    return bg, arrows, wall_cells


def _solve_independent(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, arrows, wall_cells = _detect_arrows_and_walls(g)
    arrow_cells = {cell for arrow in arrows for cell in arrow["pts"]}
    out = g.copy()

    for arrow in arrows:
        dr, dc = map(int, arrow["dir"])
        cur = tuple(map(int, arrow["start"]))
        cur_col = int(arrow["start_col"])
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
                wcol = int(g[nr + 1, nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)
            elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                wcol = int(g[nr - 1, nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)

            if int(dc) == 1 and (nr, nc + 1) in wall_cells:
                wcol = int(g[nr, nc + 1])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)
            elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                wcol = int(g[nr, nc - 1])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)

    return out


def _solve_grouped(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, arrows, wall_cells = _detect_arrows_and_walls(g)
    arrow_cells = {cell for arrow in arrows for cell in arrow["pts"]}
    out = g.copy()
    ray_owner: dict[tuple[int, int], int] = {}

    active: list[dict] = []
    for arrow in arrows:
        dr, dc = map(int, arrow["dir"])
        active.append(
            {
                "group_col": int(arrow["group_col"]),
                "cur_col": int(arrow["start_col"]),
                "cur": tuple(map(int, arrow["start"])),
                "dr": int(dr),
                "dc": int(dc),
                "seen": set(),
            }
        )

    while active:
        next_active: list[dict] = []
        proposals: dict[tuple[int, int], list[dict]] = defaultdict(list)

        for state in active:
            cur = tuple(map(int, state["cur"]))
            dr, dc = int(state["dr"]), int(state["dc"])
            cur_col = int(state["cur_col"])
            group_col = int(state["group_col"])
            sig = (cur, int(dr), int(dc), int(cur_col))
            if sig in state["seen"]:
                continue
            state["seen"].add(sig)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                continue

            if (nr, nc) in wall_cells:
                wcol = int(g[nr, nc])
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                    ray_owner[(int(cur[0]), int(cur[1]))] = int(group_col)
                state["cur_col"] = int(wcol)
                state["dr"] = -int(dr)
                state["dc"] = -int(dc)
                next_active.append(state)
                continue

            if (nr, nc) in arrow_cells:
                continue

            proposals[(nr, nc)].append(state)

        for target, entrants in proposals.items():
            owner = ray_owner.get(target)
            surviving = []
            for state in entrants:
                if owner is not None and int(owner) != int(state["group_col"]):
                    continue
                surviving.append(state)
            if not surviving:
                continue

            groups = {int(state["group_col"]) for state in surviving}
            if len(groups) > 1:
                continue

            nr, nc = target
            chosen_color = int(surviving[0]["cur_col"])
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(chosen_color)
            ray_owner[(int(nr), int(nc))] = int(surviving[0]["group_col"])

            for state in surviving:
                dr, dc = int(state["dr"]), int(state["dc"])
                state["cur"] = (int(nr), int(nc))

                if int(dr) == 1 and (nr + 1, nc) in wall_cells:
                    wcol = int(g[nr + 1, nc])
                    if int(g[nr, nc]) == int(bg):
                        out[nr, nc] = int(wcol)
                    ray_owner[(int(nr), int(nc))] = int(state["group_col"])
                    state["cur_col"] = int(wcol)
                    state["dr"] = -int(dr)
                elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                    wcol = int(g[nr - 1, nc])
                    if int(g[nr, nc]) == int(bg):
                        out[nr, nc] = int(wcol)
                    ray_owner[(int(nr), int(nc))] = int(state["group_col"])
                    state["cur_col"] = int(wcol)
                    state["dr"] = -int(dr)

                if int(dc) == 1 and (nr, nc + 1) in wall_cells:
                    wcol = int(g[nr, nc + 1])
                    if int(g[nr, nc]) == int(bg):
                        out[nr, nc] = int(wcol)
                    ray_owner[(int(nr), int(nc))] = int(state["group_col"])
                    state["cur_col"] = int(wcol)
                    state["dc"] = -int(dc)
                elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                    wcol = int(g[nr, nc - 1])
                    if int(g[nr, nc]) == int(bg):
                        out[nr, nc] = int(wcol)
                    ray_owner[(int(nr), int(nc))] = int(state["group_col"])
                    state["cur_col"] = int(wcol)
                    state["dc"] = -int(dc)

                next_active.append(state)

        active = next_active

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant solver for 142ca369.

    Rays still bounce on walls and inherit wall colors, but arrow color now defines
    a persistent collision group:
    - same-group rays may share path cells
    - different-group rays collide and stop
    """
    return _solve_grouped(np.array(grid, dtype=int))


def _trace_full_diagonal(sr: int, sc: int, dr: int, dc: int, H: int, W: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    r, c = int(sr), int(sc)
    while True:
        nr, nc = int(r + dr), int(c + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            break
        out.append((int(nr), int(nc)))
        r, c = int(nr), int(nc)
    return out


def _trace_until_before(
    sr: int,
    sc: int,
    dr: int,
    dc: int,
    stop_cell: tuple[int, int],
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    r, c = int(sr), int(sc)
    while True:
        nr, nc = int(r + dr), int(c + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            break
        if (int(nr), int(nc)) == (int(stop_cell[0]), int(stop_cell[1])):
            break
        out.append((int(nr), int(nc)))
        r, c = int(nr), int(nc)
    return out


def _all_in_bounds(cells: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= int(r) < H and 0 <= int(c) < W for r, c in cells)


def _plan_same_group_crossing(
    H: int,
    W: int,
    color: int,
    occupied: set[tuple[int, int]],
) -> dict | None:
    for _ in range(300):
        L = _rng.randint(4, 6)
        r_lo = max(L + 1, 4)
        r_hi = min(H - L - 2, H // 2 + 1)
        c_lo = L + 1
        c_hi = W - 4
        if r_lo > r_hi or c_lo > c_hi:
            continue

        pr = _rng.randint(r_lo, r_hi)
        pc = _rng.randint(c_lo, c_hi)
        s1 = (pr - L, pc - L)
        s2 = (pr + L, pc - L)
        pts1 = _arrow_pts_from_start(s1[0], s1[1], 1, 1)
        pts2 = _arrow_pts_from_start(s2[0], s2[1], -1, 1)
        if not (_all_in_bounds(pts1, H, W) and _all_in_bounds(pts2, H, W)):
            continue

        ray1 = _trace_full_diagonal(s1[0], s1[1], 1, 1, H, W)
        ray2 = _trace_full_diagonal(s2[0], s2[1], -1, 1, H, W)
        center = (int(pr), int(pc))

        if center not in set(ray1) or center not in set(ray2):
            continue
        if len(set(ray1) & set(ray2)) != 1 or center not in (set(ray1) & set(ray2)):
            continue

        pair_cells = set(pts1) | set(pts2) | set(ray1) | set(ray2)
        if pair_cells & occupied:
            continue
        if (set(pts1) & (set(pts2) | set(ray1) | set(ray2))) or (set(pts2) & (set(ray1) | set(ray2))):
            continue

        return {
            "arrows": [(int(color), pts1), (int(color), pts2)],
            "walls": [],
            "reserved": pair_cells,
            "same_group_shared_cells": 1,
            "diff_group_collision_cells": 0,
        }
    return None


def _plan_diff_group_collision(
    H: int,
    W: int,
    color_a: int,
    color_b: int,
    occupied: set[tuple[int, int]],
) -> dict | None:
    for _ in range(300):
        L = _rng.randint(4, 6)
        r_lo = max(L + 1, H // 2 - 1)
        r_hi = min(H - L - 2, H - 4)
        c_lo = 3
        c_hi = W - L - 2
        if r_lo > r_hi or c_lo > c_hi:
            continue

        qr = _rng.randint(r_lo, r_hi)
        qc = _rng.randint(c_lo, c_hi)
        s1 = (qr - L, qc + L)
        s2 = (qr + L, qc + L)
        pts1 = _arrow_pts_from_start(s1[0], s1[1], 1, -1)
        pts2 = _arrow_pts_from_start(s2[0], s2[1], -1, -1)
        if not (_all_in_bounds(pts1, H, W) and _all_in_bounds(pts2, H, W)):
            continue

        center = (int(qr), int(qc))
        ray1 = _trace_until_before(s1[0], s1[1], 1, -1, center, H, W)
        ray2 = _trace_until_before(s2[0], s2[1], -1, -1, center, H, W)
        if len(ray1) != L - 1 or len(ray2) != L - 1:
            continue

        pair_cells = set(pts1) | set(pts2) | set(ray1) | set(ray2)
        if center in pair_cells:
            continue
        if pair_cells & occupied:
            continue
        if (set(pts1) & (set(pts2) | set(ray1) | set(ray2))) or (set(pts2) & (set(ray1) | set(ray2))):
            continue

        return {
            "arrows": [(int(color_a), pts1), (int(color_b), pts2)],
            "walls": [],
            "reserved": pair_cells,
            "same_group_shared_cells": 0,
            "diff_group_collision_cells": 1,
        }
    return None


def _plan_bounced_arrow(
    H: int,
    W: int,
    arrow_color: int,
    wall_palette: list[int],
    occupied: set[tuple[int, int]],
    bounces: int,
) -> dict | None:
    if not wall_palette:
        return None

    for _ in range(250):
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
        c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
        if r_lo > r_hi or c_lo > c_hi:
            continue
        sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)
        arrow_pts = _arrow_pts_from_start(sr, sc, dr, dc)
        arrow_set = set(arrow_pts)
        if not _all_in_bounds(arrow_pts, H, W) or arrow_set & occupied:
            continue

        cur_r, cur_c, cur_dr, cur_dc = int(sr), int(sc), int(dr), int(dc)
        ray_set: set[tuple[int, int]] = set()
        walls: list[tuple[list[tuple[int, int]], int]] = []
        ok = True

        for seg_idx in range(bounces + 1):
            max_steps = min(
                (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                (W - 1 - cur_c) if cur_dc == 1 else cur_c,
            )
            if seg_idx < bounces:
                if max_steps < 4:
                    ok = False
                    break
                seg_len = _rng.randint(3, max(3, max_steps - 1))
            else:
                seg_len = max_steps

            for _step in range(seg_len):
                nr, nc = int(cur_r + cur_dr), int(cur_c + cur_dc)
                if not (0 <= nr < H and 0 <= nc < W):
                    break
                if (nr, nc) in occupied or (nr, nc) in ray_set or (nr, nc) in arrow_set:
                    ok = False
                    break
                ray_set.add((int(nr), int(nc)))
                cur_r, cur_c = int(nr), int(nc)
            if not ok:
                break

            if seg_idx >= bounces:
                continue

            options: list[tuple[list[tuple[int, int]], str]] = []
            wr, wc = int(cur_r + cur_dr), int(cur_c)
            if 0 <= wr < H:
                w3 = [(int(wr), int(wc + d)) for d in (-1, 0, 1)]
                if all(0 <= cc < W for _, cc in w3):
                    options.append((w3, "dr"))
                options.append(([(int(wr), int(wc))], "dr"))

            wr2, wc2 = int(cur_r), int(cur_c + cur_dc)
            if 0 <= wc2 < W:
                w3v = [(int(wr2 + d), int(wc2)) for d in (-1, 0, 1)]
                if all(0 <= rr < H for rr, _ in w3v):
                    options.append((w3v, "dc"))
                options.append(([(int(wr2), int(wc2))], "dc"))
            if not options:
                ok = False
                break

            _rng.shuffle(options)
            chosen = None
            for cells, flip in options:
                cell_set = set(cells)
                if cell_set & occupied or cell_set & ray_set or cell_set & arrow_set:
                    continue
                chosen = (cells, flip)
                break
            if chosen is None:
                ok = False
                break

            wpxs, flip = chosen
            walls.append((wpxs, int(_rng.choice(wall_palette))))
            if flip == "dr":
                cur_dr = -cur_dr
            else:
                cur_dc = -cur_dc

        if not ok or len(ray_set) < 4:
            continue

        reserved = set(arrow_set) | set(ray_set)
        for wpxs, _ in walls:
            reserved |= set(wpxs)
        if reserved & occupied:
            continue

        return {
            "arrows": [(int(arrow_color), arrow_pts)],
            "walls": walls,
            "reserved": reserved,
            "same_group_shared_cells": 0,
            "diff_group_collision_cells": 0,
        }
    return None


def _apply_plan(grid: np.ndarray, plan: dict) -> None:
    for color, pts in plan["arrows"]:
        for r, c in pts:
            grid[int(r), int(c)] = int(color)
    for wpxs, wcol in plan["walls"]:
        for r, c in wpxs:
            grid[int(r), int(c)] = int(wcol)


def _count_arrows_by_color(grid: np.ndarray) -> dict[int, int]:
    _bg, arrows, _walls = _detect_arrows_and_walls(grid)
    counts: dict[int, int] = defaultdict(int)
    for arrow in arrows:
        counts[int(arrow["group_col"])] += 1
    return dict(counts)


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    palette = list(range(1, 10))
    _rng.shuffle(palette)

    same_group_color = int(palette.pop())
    diff_color_a = int(palette.pop())
    diff_color_b = int(palette.pop())
    extra_arrow_colors: list[int] = []
    if palette and _rng.random() < 0.85:
        extra_arrow_colors.append(int(palette.pop()))

    wall_palette = list(palette)
    occupied: set[tuple[int, int]] = set()
    plans: list[dict] = []

    same_plan = _plan_same_group_crossing(H, W, same_group_color, occupied)
    if same_plan is None:
        return None
    plans.append(same_plan)
    occupied |= set(same_plan["reserved"])

    diff_plan = _plan_diff_group_collision(H, W, diff_color_a, diff_color_b, occupied)
    if diff_plan is None:
        return None
    plans.append(diff_plan)
    occupied |= set(diff_plan["reserved"])

    for color in extra_arrow_colors:
        bounced = _plan_bounced_arrow(
            H=H,
            W=W,
            arrow_color=int(color),
            wall_palette=wall_palette,
            occupied=occupied,
            bounces=_rng.choice([1, 1, 2]),
        )
        if bounced is not None:
            plans.append(bounced)
            occupied |= set(bounced["reserved"])

    grid = np.zeros((H, W), dtype=int)
    for plan in plans:
        _apply_plan(grid, plan)

    out = main(grid)
    base = _solve_independent(grid)
    if np.array_equal(out, base):
        return None

    arrow_counts = _count_arrows_by_color(grid)
    if max(arrow_counts.values(), default=0) < 2:
        return None

    # The showcase pair is constructed to force the new branch; require a visible delta.
    if int(np.count_nonzero(out != base)) < 4:
        return None

    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D11 color-grouping variant."""
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
