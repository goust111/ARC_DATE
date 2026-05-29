# Variant of 142ca369
# Dimension: D15 — 墙壁功能分化 (obstacle)
# Change: 所有墙壁功能相同——都触发侧面反射并传递颜色... → 干扰墙壁（不触发反射的装饰物）

from __future__ import annotations

# Design analysis for the D15 variant
# ===================================
# a) Concrete behavior
#    - The original task treats every non-arrow, non-background pixel as a reflective wall.
#    - This variant splits obstacles into two roles:
#      1. Functional walls: colors that are "announced" by at least one 3-pixel straight wall segment
#         elsewhere in the grid. Any non-arrow component of such a color that is either a single pixel or a
#         3-pixel straight segment is reflective and still transfers color on bounce.
#      2. Decorative distractors: pixels in other colors. They may sit exactly in the side-touch positions
#         where a wall could have caused a bounce, but the ray ignores them and continues straight.
#    - Small trace example:
#      original logic: ray enters (r,c), sees a colored pixel at (r+dr,c), flips dr immediately.
#      D15 logic: if that side pixel uses a decorative-only color, no flip happens; the ray keeps (dr,dc).
#      If later it meets a wall color that also appears in a 3-cell segment, then the bounce happens there.
#
# b) Complexity calibration
#    - The active/inactive distinction is color-based, but not arbitrary: active wall colors are backed by a
#      repeated structural cue (a 3-cell line). This keeps the rule inferable from a few examples.
#    - Decorative pixels stay sparse and isolated so the grid is not cluttered, but every generated sample
#      still contains at least one decorative pixel in a plausible bounce location.
#
# c) Hidden assumptions from the original solver
#    - Original `wall_cells = all non-bg cells except arrows` is no longer valid; that would falsely bounce on
#      decorations.
#    - Side-touch checks and diagonal head-on checks must consult only functional wall cells.
#    - Arrow detection still relies on 3 pixels in a 2x2 box, so the generator must never create decorative
#      components with that shape.
#    - Because decorative pixels are ignored, the generator keeps them off the ray path itself; otherwise the
#      path would visually "pass through" a colored cell while leaving it unchanged, which is harder to read.
#
# d) Placement legality
#    - The generator remains output-first for rays and functional walls.
#    - Functional walls are placed as non-overlapping 3-pixel straight segments adjacent to exactly one bounce.
#    - Decorative pixels are added only after all rays/walls/arrows are fixed, using candidate side-touch
#      locations that are in-bounds and unoccupied.
#    - Decorative colors are disjoint from arrow colors and wall colors, and same-color decorations are kept
#      non-4-connected so they can never accidentally form an active wall signature.
#
# e) Visual observability
#    - Each sample has at least one functional wall segment and at least one decorative distractor.
#    - The generator rejects any sample where the legacy solver (which bounces on every colored obstacle)
#      produces the same output as the new solver, so D15 behavior is guaranteed to matter.
#
# f) Visual cleanliness
#    - Decorative pixels are sparse and isolated.
#    - Only wall colors participate in reflections, so the scene still reads as arrows + rays + some real
#      walls + some irrelevant colored clutter rather than a dense collision field.
#
# g) Color inference
#    - The key color cue is binary and local: colors that appear in 3-cell straight wall segments are
#      reflective; colors that appear only as isolated dots are not.

import random as _rng
import sys
from pathlib import Path

import numpy as np

for _parent in Path(__file__).resolve().parents:
    _candidate = _parent / "solvers" / "arc_utils.py"
    if _candidate.exists():
        if str(_candidate.parent) not in sys.path:
            sys.path.append(str(_candidate.parent))
        break

from arc_utils import colors_present, components_mask, get_bg


def _is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _is_straight_len3(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 3:
        return False
    rs = sorted(r for r, _c in pts)
    cs = sorted(c for _r, c in pts)
    return (len(set(rs)) == 1 and cs == list(range(cs[0], cs[0] + 3))) or (
        len(set(cs)) == 1 and rs == list(range(rs[0], rs[0] + 3))
    )


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


def _classify_scene(g: np.ndarray):
    bg = int(get_bg(g))
    arrows, arrow_cells = _detect_arrows(g, bg)

    functional_colors: set[int] = set()
    components_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    for col in colors_present(g, exclude={bg}):
        mask = g == int(col)
        if arrow_cells:
            for r, c in arrow_cells:
                mask[r, c] = False
        comps: list[list[tuple[int, int]]] = []
        for coords in components_mask(mask, connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if pts:
                comps.append(pts)
                if _is_straight_len3(pts):
                    functional_colors.add(int(col))
        components_by_color[int(col)] = comps

    wall_cells: set[tuple[int, int]] = set()
    decorative_cells: set[tuple[int, int]] = set()
    for col, comps in components_by_color.items():
        for pts in comps:
            cell_set = set(pts)
            if col in functional_colors and (len(pts) == 1 or _is_straight_len3(pts)):
                wall_cells |= cell_set
            else:
                decorative_cells |= cell_set

    return bg, arrows, arrow_cells, wall_cells, decorative_cells, functional_colors


def _solve(grid: np.ndarray, *, variant: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, arrows, arrow_cells, wall_cells, _decorative_cells, _functional_colors = _classify_scene(g)
    if not variant:
        wall_cells = {
            (int(r), int(c))
            for r, c in np.argwhere(g != int(bg))
            if (int(r), int(c)) not in arrow_cells
        }

    out = g.copy()

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Only obstacle colors backed by at least one 3-pixel straight wall segment are reflective.
    - Other colored obstacle pixels are decorative distractors and do not trigger any bounce.
    """
    return _solve(grid, variant=True)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, variant=False)


def _plan_arrow(
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    arrow_color: int,
    active_wall_colors: list[int],
    n_bounces: int,
):
    dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])

    for _ in range(90):
        r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
        c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
        if r_lo > r_hi or c_lo > c_hi:
            continue
        sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)

        mr, mc = sr - dr, sc - dc
        arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
        arrow_set = set(arrow_pts)
        if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
            continue

        cur_r, cur_c = sr, sc
        cur_dr, cur_dc = dr, dc
        cur_col = int(arrow_color)
        ray_cells: list[tuple[int, int, int]] = []
        step_info: list[tuple[int, int, int, int, bool]] = []
        walls: list[tuple[list[tuple[int, int]], int]] = []
        ray_set: set[tuple[int, int]] = set()
        wall_set: set[tuple[int, int]] = set()
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
                if max_steps < 4:
                    ok = False
                    break
                seg_len = _rng.randint(3, max_steps - 1)
            else:
                seg_len = max_steps

            for _step in range(seg_len):
                nr, nc = cur_r + cur_dr, cur_c + cur_dc
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                if (
                    (nr, nc) in occupied
                    or (nr, nc) in arrow_set
                    or (nr, nc) in ray_set
                    or (nr, nc) in wall_set
                ):
                    ok = False
                    break
                ray_cells.append((nr, nc, cur_col))
                step_info.append((nr, nc, cur_dr, cur_dc, False))
                ray_set.add((nr, nc))
                cur_r, cur_c = nr, nc
            if not ok:
                break

            if seg_idx >= n_bounces:
                continue

            options: list[tuple[list[tuple[int, int]], str]] = []
            wr, wc = cur_r + cur_dr, cur_c
            w3h = [(wr, wc + d) for d in (-1, 0, 1)]
            if all(0 <= r < H and 0 <= c < W for r, c in w3h):
                options.append((w3h, "dr"))

            wr2, wc2 = cur_r, cur_c + cur_dc
            w3v = [(wr2 + d, wc2) for d in (-1, 0, 1)]
            if all(0 <= r < H and 0 <= c < W for r, c in w3v):
                options.append((w3v, "dc"))

            _rng.shuffle(options)
            chosen_wall = None
            for wpxs, flip in options:
                if any(
                    (r, c) in occupied or (r, c) in arrow_set or (r, c) in ray_set or (r, c) in wall_set
                    for r, c in wpxs
                ):
                    continue
                chosen_wall = (wpxs, flip)
                break
            if chosen_wall is None:
                ok = False
                break

            wpxs, flip = chosen_wall
            wcol = int(_rng.choice(active_wall_colors))
            walls.append((wpxs, wcol))
            wall_set.update(wpxs)

            lr, lc, _ = ray_cells[-1]
            ray_cells[-1] = (lr, lc, wcol)
            step_info[-1] = (lr, lc, step_info[-1][2], step_info[-1][3], True)
            cur_col = wcol
            if flip == "dr":
                cur_dr = -cur_dr
            else:
                cur_dc = -cur_dc

        if not ok or len(ray_cells) < 3 or not walls:
            continue

        all_cells = set(arrow_pts) | ray_set | wall_set
        if all_cells & occupied:
            continue

        candidate_decorations: list[tuple[int, int]] = []
        blocked = occupied | all_cells
        for r, c, step_dr, step_dc, bounced in step_info:
            if bounced:
                continue
            for rr, cc in ((r + step_dr, c), (r, c + step_dc)):
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in blocked:
                    candidate_decorations.append((int(rr), int(cc)))
        if not candidate_decorations:
            continue

        dedup_candidates = list(dict.fromkeys(candidate_decorations))
        return {
            "arrow_pts": arrow_pts,
            "ray_cells": ray_cells,
            "walls": walls,
            "occupied": all_cells,
            "candidates": dedup_candidates,
        }

    return None


def _can_place_decoration(
    pos: tuple[int, int],
    color: int,
    occupied: set[tuple[int, int]],
    by_color: dict[int, set[tuple[int, int]]],
) -> bool:
    if pos in occupied:
        return False
    r, c = pos
    same = by_color.setdefault(int(color), set())
    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if (rr, cc) in same:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(1, 5)
    n_active_colors = _rng.randint(1, 2)
    n_deco_colors = _rng.randint(1, 2)
    if n_arrows + n_active_colors + n_deco_colors > 9:
        return None

    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]
    active_wall_colors = palette[n_arrows : n_arrows + n_active_colors]
    deco_colors = palette[n_arrows + n_active_colors : n_arrows + n_active_colors + n_deco_colors]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    decoration_candidates: list[tuple[int, int]] = []
    total_bounces = 0

    for ai, arrow_color in enumerate(arrow_colors):
        if ai == 0:
            n_bounces = _rng.choices([1, 2, 3], weights=[3, 3, 1])[0]
        else:
            n_bounces = _rng.choices([0, 1, 2, 3], weights=[2, 3, 2, 1])[0]

        plan = _plan_arrow(H, W, occupied, int(arrow_color), active_wall_colors, int(n_bounces))
        if plan is None:
            return None

        total_bounces += int(n_bounces)
        for r, c in plan["arrow_pts"]:
            grid[r, c] = out[r, c] = int(arrow_color)
        for r, c, col in plan["ray_cells"]:
            out[r, c] = int(col)
        for wpxs, wcol in plan["walls"]:
            for r, c in wpxs:
                grid[r, c] = out[r, c] = int(wcol)

        occupied |= plan["occupied"]
        decoration_candidates.extend(plan["candidates"])

    if total_bounces == 0:
        return None

    candidate_pool = list(dict.fromkeys(decoration_candidates))
    _rng.shuffle(candidate_pool)
    if not candidate_pool:
        return None

    target_n = min(len(candidate_pool), _rng.randint(2, 4))
    target_n = max(1, target_n)
    placed_by_color: dict[int, set[tuple[int, int]]] = {}
    placed_decorations: list[tuple[int, int, int]] = []

    for pos in candidate_pool:
        colors = deco_colors[:]
        _rng.shuffle(colors)
        chosen_color = None
        for deco_col in colors:
            if _can_place_decoration(pos, int(deco_col), occupied, placed_by_color):
                chosen_color = int(deco_col)
                break
        if chosen_color is None:
            continue
        r, c = pos
        grid[r, c] = out[r, c] = chosen_color
        occupied.add(pos)
        placed_by_color.setdefault(chosen_color, set()).add(pos)
        placed_decorations.append((r, c, chosen_color))
        if len(placed_decorations) >= target_n:
            break

    if not placed_decorations:
        return None

    new_out = main(grid)
    if not np.array_equal(new_out, out):
        return None
    if np.array_equal(_legacy_main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D15 variant of task 142ca369."""
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
