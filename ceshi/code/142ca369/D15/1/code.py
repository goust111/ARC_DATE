# Variant of 142ca369
# Dimension: D15 — 墙壁功能分化 (obstacle)
# Change: 所有墙壁功能相同——都触发侧面反射并传递颜色... → 加速墙与减速墙
#
# Design analysis
# ---------------
# spec:
#   concrete_variant:
#     walls:
#       fast_wall_color: 8
#       slow_wall_color: 2
#       meaning:
#         - color 8 wall: reflect as usual, inherit color 8, and set future ray step length to 2
#         - color 2 wall: reflect as usual, inherit color 2, and set future ray step length to 1
#       why_fixed_colors:
#         - D15 says wall function is distinguished by color; using exactly two repeated wall colors
#           makes the mapping human-inferable from 2-4 examples.
#         - Arrow colors deliberately exclude 2 and 8, so those colors read as wall-function markers
#           rather than ambiguous "sometimes arrow, sometimes wall".
#     rays:
#       start_step: 1
#       accelerated_motion:
#         - after a fast-wall bounce, future diagonal motion jumps by 2 cells each move
#         - the skipped middle cell is not painted ("jump one cell while painting")
#       slowed_motion:
#         - after a slow-wall bounce, future diagonal motion returns to the original step 1
#     preserved_parts:
#       - arrow detection and firing direction stay unchanged
#       - side-touch reflection still flips exactly one axis
#       - corner reflection still reverses both axes
#       - bounce cell still adopts the wall color
#       - rays still stop on exit or when landing on another arrow
#   behavior_trace:
#     original_local_case:
#       - arrow color 4 moves with step 1: (5,5) -> (6,6) -> (7,7)
#       - wall at (8,7) side-touches the cell (7,7), so dr flips and color becomes wall color
#       - afterwards the ray continues step 1: (6,8) -> (5,9) -> ...
#     variant_local_case:
#       - same first part, but suppose the wall color is 8 (fast wall)
#       - bounce at (7,7) still recolors that cell to 8 and flips dr
#       - afterwards motion uses step 2: (5,9) -> (3,11) -> ...
#       - cells (6,8) and (4,10) stay background, so the painted diagonal visibly becomes sparse
#     difference:
#       - original logic assumes every post-bounce segment fills consecutive diagonal cells
#       - D15 requires some post-bounce segments to skip every other diagonal cell, controlled only by wall color
#   complexity_calibration:
#     - Keeping only two wall-function colors avoids too many hidden color-function mappings.
#     - Repeating both wall colors in the same sample gives direct support for the rule.
#     - A featured arrow is forced to encounter both wall types so the contrast is visible within one grid.
#   implicit_assumption_audit:
#     - Original solver stored state as (pos, dr, dc, color); this is no longer sufficient because the same
#       ray state with different step lengths would evolve differently. The new state must include step_len.
#     - Original solver advanced by exactly one diagonal cell per loop. Every next-cell calculation that used
#       cur + dr / cur + dc now depends on cur + step_len * dr / dc.
#     - Original generator assumed every traversed cell was painted; that is no longer true for accelerated
#       segments, so it now reserves skipped midpoint cells to prevent other objects from crossing invisible jumps.
#     - Original side-touch code could harmlessly update only color and direction because step length was constant.
#       In D15, every bounce must also update step_len from wall color.
#   placement_legality:
#     - Generator remains output-first: it plans the full visible ray path, the wall colors, and the hidden
#       skipped midpoint cells before committing the input.
#     - Arrows, visible ray cells, walls, and hidden midpoint cells are all kept disjoint during planning.
#     - Wall placement still uses the original side-touch geometry: a wall is one orthogonal cell off the
#       bounce point and flips only the touched axis.
#   visual_observability:
#     - Each generated grid contains at least one fast wall and at least one slow wall.
#     - The featured arrow always has both a fast segment and a slow segment, so the human can compare
#       sparse diagonal painting against dense diagonal painting in one sample.
#   visual_cleanliness:
#     - Arrow count is kept moderate and hidden midpoint cells are reserved to avoid cluttering crossings.
#     - Wall colors are reused consistently, reducing palette noise.

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

FAST_WALL_COLOR = 8
SLOW_WALL_COLOR = 2
WALL_FUNCTION_COLORS = (FAST_WALL_COLOR, SLOW_WALL_COLOR)
ARROW_PALETTE = [1, 3, 4, 5, 6, 7, 9]


def _wall_step(color: int) -> int:
    return 2 if int(color) == FAST_WALL_COLOR else 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from L-shaped arrows.

    D15 change:
    - wall color 8 = fast wall, so after that bounce the ray advances by 2 cells per move
    - wall color 2 = slow wall, so after that bounce the ray advances by 1 cell per move
    - skipped midpoint cells on fast segments are not painted
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
        if len(pts) != 3:
            return False
        rs = [r for r, _c in pts]
        cs = [c for _r, c in pts]
        return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)

    def arrow_dir_and_start(pts: list[tuple[int, int]]):
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

    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)

    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()

    def recolor_contact(r: int, c: int, wall_color: int) -> None:
        if int(g[r, c]) == int(bg):
            out[r, c] = int(wall_color)

    for start_col, pts in arrows:
        info = arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
        cur_col = int(start_col)
        step_len = 1
        seen = set()

        while True:
            state = (cur, int(dr), int(dc), int(cur_col), int(step_len))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + step_len * dr)
            nc = int(cur[1] + step_len * dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in wall_cells:
                wcol = int(g[nr, nc])
                recolor_contact(int(cur[0]), int(cur[1]), wcol)
                cur_col = int(wcol)
                step_len = _wall_step(wcol)
                dr, dc = -int(dr), -int(dc)
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            if int(dr) == 1 and (nr + 1, nc) in wall_cells:
                wcol = int(g[nr + 1, nc])
                recolor_contact(nr, nc, wcol)
                cur_col = int(wcol)
                step_len = _wall_step(wcol)
                dr = -int(dr)
            elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                wcol = int(g[nr - 1, nc])
                recolor_contact(nr, nc, wcol)
                cur_col = int(wcol)
                step_len = _wall_step(wcol)
                dr = -int(dr)

            if int(dc) == 1 and (nr, nc + 1) in wall_cells:
                wcol = int(g[nr, nc + 1])
                recolor_contact(nr, nc, wcol)
                cur_col = int(wcol)
                step_len = _wall_step(wcol)
                dc = -int(dc)
            elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                wcol = int(g[nr, nc - 1])
                recolor_contact(nr, nc, wcol)
                cur_col = int(wcol)
                step_len = _wall_step(wcol)
                dc = -int(dc)

    return out


def _max_moves(H: int, W: int, r: int, c: int, dr: int, dc: int, step_len: int) -> int:
    row_room = (H - 1 - r) // step_len if dr == 1 else r // step_len
    col_room = (W - 1 - c) // step_len if dc == 1 else c // step_len
    return int(min(row_room, col_room))


def _segment_min_len(is_initial: bool, step_len: int) -> int:
    if is_initial:
        return 3
    return 2 if step_len == 2 else 3


def _sample_bounce_colors(featured: bool) -> list[int]:
    if featured:
        return list(_rng.choice([
            [FAST_WALL_COLOR, SLOW_WALL_COLOR],
            [SLOW_WALL_COLOR, FAST_WALL_COLOR],
            [FAST_WALL_COLOR, SLOW_WALL_COLOR, FAST_WALL_COLOR],
        ]))
    n_bounces = _rng.choices([0, 1, 2], weights=[1, 3, 2])[0]
    return [_rng.choice(WALL_FUNCTION_COLORS) for _ in range(n_bounces)]


def _pick_wall_options(
    H: int,
    W: int,
    cur_r: int,
    cur_c: int,
    cur_dr: int,
    cur_dc: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[list[tuple[int, int]], str]]:
    options: list[tuple[list[tuple[int, int]], str]] = []

    wr, wc = cur_r + cur_dr, cur_c
    if 0 <= wr < H:
        one = [(wr, wc)]
        if all(0 <= c < W and (rr, cc) not in blocked for rr, cc in one for c in [cc]):
            options.append((one, "dr"))
        w3 = [(wr, wc + d) for d in (-1, 0, 1)]
        if all(0 <= c < W and (rr, cc) not in blocked for rr, cc in w3 for c in [cc]):
            options.append((w3, "dr"))

    wr2, wc2 = cur_r, cur_c + cur_dc
    if 0 <= wc2 < W:
        one = [(wr2, wc2)]
        if all(0 <= r < H and (rr, cc) not in blocked for rr, cc in one for r in [rr]):
            options.append((one, "dc"))
        w3v = [(wr2 + d, wc2) for d in (-1, 0, 1)]
        if all(0 <= r < H and (rr, cc) not in blocked for rr, cc in w3v for r in [rr]):
            options.append((w3v, "dc"))

    return options


def _plan_arrow(
    H: int,
    W: int,
    arrow_color: int,
    global_blocked: set[tuple[int, int]],
    *,
    featured: bool,
):
    bounce_colors = _sample_bounce_colors(featured)

    for _ in range(120):
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
        c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
        if r_lo > r_hi or c_lo > c_hi:
            continue
        sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)

        mr, mc = sr - dr, sc - dc
        arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
        if any(not (0 <= r < H and 0 <= c < W) or (r, c) in global_blocked for r, c in arrow_pts):
            continue

        local_blocked = set(arrow_pts)
        cur_r, cur_c = sr, sc
        cur_dr, cur_dc = dr, dc
        cur_col = int(arrow_color)
        step_len = 1
        ray_cells: list[tuple[int, int, int]] = []
        walls: list[tuple[list[tuple[int, int]], int]] = []
        visible_cells: set[tuple[int, int]] = set(arrow_pts)
        fast_landings = 0

        ok = True
        for seg_idx in range(len(bounce_colors) + 1):
            max_moves = _max_moves(H, W, cur_r, cur_c, cur_dr, cur_dc, step_len)
            min_len = _segment_min_len(seg_idx == 0, step_len)
            if max_moves < min_len:
                ok = False
                break
            seg_len = _rng.randint(min_len, max_moves)

            start_of_segment = len(ray_cells)
            for _move in range(seg_len):
                step_cells = []
                for jump in range(1, step_len + 1):
                    rr = cur_r + jump * cur_dr
                    cc = cur_c + jump * cur_dc
                    if not (0 <= rr < H and 0 <= cc < W):
                        ok = False
                        break
                    if (rr, cc) in global_blocked or (rr, cc) in local_blocked:
                        ok = False
                        break
                    step_cells.append((int(rr), int(cc)))
                if not ok:
                    break

                for hidden in step_cells[:-1]:
                    local_blocked.add(hidden)
                nr, nc = step_cells[-1]
                local_blocked.add((nr, nc))
                visible_cells.add((nr, nc))
                ray_cells.append((nr, nc, cur_col))
                cur_r, cur_c = nr, nc

            if not ok:
                break

            if step_len == 2:
                fast_landings += len(ray_cells) - start_of_segment

            if seg_idx == len(bounce_colors):
                continue

            options = _pick_wall_options(H, W, cur_r, cur_c, cur_dr, cur_dc, global_blocked | local_blocked)
            if not options:
                ok = False
                break
            _rng.shuffle(options)
            prefer_long = _rng.random() < 0.6
            options.sort(key=lambda item: len(item[0]), reverse=prefer_long)

            wcol = int(bounce_colors[seg_idx])
            chosen = options[0]
            wpxs, flip = chosen
            walls.append((wpxs, wcol))
            for cell in wpxs:
                local_blocked.add(cell)
                visible_cells.add(cell)

            last_r, last_c, _old = ray_cells[-1]
            ray_cells[-1] = (last_r, last_c, wcol)
            cur_col = int(wcol)
            step_len = _wall_step(wcol)
            if flip == "dr":
                cur_dr = -cur_dr
            else:
                cur_dc = -cur_dc

        if not ok or len(ray_cells) < 4:
            continue

        if featured:
            wall_colors = {int(col) for _wpxs, col in walls}
            if wall_colors != set(WALL_FUNCTION_COLORS):
                continue
            if fast_landings < 2:
                continue

        return {
            "arrow_pts": arrow_pts,
            "ray_cells": ray_cells,
            "walls": walls,
            "blocked": local_blocked,
            "visible": visible_cells,
        }

    return None


def _gen_attempt() -> np.ndarray | None:
    """
    Output-first generator for the D15 variant.

    It guarantees:
    - at least one fast wall and one slow wall per sample
    - at least one visibly accelerated segment with multiple painted landings
    """
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(1, 4)
    arrow_colors = _rng.sample(ARROW_PALETTE, n_arrows)

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    blocked: set[tuple[int, int]] = set()
    wall_color_counts = {FAST_WALL_COLOR: 0, SLOW_WALL_COLOR: 0}

    for ai, color in enumerate(arrow_colors):
        planned = _plan_arrow(H, W, int(color), blocked, featured=(ai == 0))
        if planned is None:
            return None

        for r, c in planned["arrow_pts"]:
            grid[r, c] = out[r, c] = int(color)
        for r, c, col in planned["ray_cells"]:
            out[r, c] = int(col)
        for wpxs, wcol in planned["walls"]:
            wall_color_counts[int(wcol)] += 1
            for r, c in wpxs:
                grid[r, c] = out[r, c] = int(wcol)

        blocked |= planned["blocked"]

    if wall_color_counts[FAST_WALL_COLOR] == 0 or wall_color_counts[SLOW_WALL_COLOR] == 0:
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D15 variant of task 142ca369."""
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
