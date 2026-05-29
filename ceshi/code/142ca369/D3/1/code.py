# Variant of 142ca369
# Dimension: D3 — 轨迹涂色模式 (trajectory)
# Change: 连续轨迹（continuous_trail）——射线经过的每个背景格子都被涂色，形成连续的彩色线条... → 间歇跳跃轨迹

from __future__ import annotations

"""
Design analysis for the D3 variant (continuous trail -> intermittent skip).

variant_spec:
  core_change:
    - Arrows, wall detection, reflection geometry, and color inheritance stay the same.
    - The ray still moves one diagonal cell at a time through the full geometric path.
    - Only the coloring cadence changes: with skip_gap = 1, the first visited bg cell is painted,
      the second visited bg cell is left blank, the third is painted, and so on.
    - Therefore some physically visited cells are invisible in the output, yet they still matter for
      wall bounces, color changes, and collision legality.
  concrete_example:
    - Suppose an arrow starts at (5,5) and travels (dr,dc)=(1,1).
    - Original behavior over three steps: (6,6),(7,7),(8,8) are all painted.
    - Variant behavior: step 1 paints (6,6), step 2 traverses (7,7) but leaves it bg,
      step 3 paints (8,8).
    - If a side wall causes a bounce at step 2, the ray still reflects there and adopts the
      wall color, but (7,7) remains blank because D3 only changes trail rendering, not motion.
      The new color appears from the next painted step onward.
  complexity_calibration:
    - The skip cadence is fixed for the whole task family (every other visited cell). This avoids
      an extra hidden degree of freedom that would be hard to infer from 2-4 ARC examples.
    - To keep the change visible, generated rays must include at least two skipped visited cells
      overall and at least four visited cells total, so outputs do not collapse back to the
      original continuous look.
  implicit_assumption_audit:
    - Original solver assumption: every advance onto a bg cell should paint it.
      This is false now; painting must depend on the move index parity.
    - Original solver loop-state key used only (position, direction, color). With intermittent
      painting, revisiting the same geometric state on a different parity would produce a different
      future output, so the step phase modulo the paint period must also be tracked.
    - Original generator treated "visited by the ray" and "painted in the output" as the same set.
      This is false now; skipped cells must still be reserved from overlaps because the solver
      traverses them and can bounce from them.
    - Original bounce recoloring wrote the bounce cell whenever it was bg in the input.
      In the variant this remains valid only on paint steps; on skipped steps the bounce cell stays bg.
  placement_legality:
    - All traversed cells, painted or skipped, are reserved in ray_set.
    - Walls remain one cell off the diagonal in the direction of travel and are adjacent to exactly
      one traversed cell at the moment of bounce.
    - Arrows, walls, painted trail cells, and skipped traversed cells never overlap.
  visual_observability:
    - Keeping the L-shaped arrows and mono-color walls unchanged preserves the original task cue.
    - The dashed output is directly visible because each accepted sample contains hidden traversed
      cells; comparing input/output reveals that the ray does not fill every diagonal cell.
    - Bounce/color inheritance is still inferable because walls remain the only non-arrow colored
      objects, and later painted cells switch to wall colors exactly as before.
  cleanliness:
    - The variant does not add new markers or symbols; it only sparsifies the trail.
    - Reserving skipped cells avoids accidental clutter from overlaps that would make the hidden
      path ambiguous or unreadable.
  post_change_solver_assumptions:
    - The solver now assumes a fixed paint period of 2 for this task family.
    - The generator guarantees no ray revisits and no object occupies skipped traversed cells,
      so the solver's overwrite behavior on bg cells remains safe.
"""

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg


SKIP_GAP = 1
TRAIL_PERIOD = SKIP_GAP + 1


def _should_paint(step_idx: int) -> bool:
    """Paint the 1st, 3rd, 5th, ... visited cells for a dashed diagonal trail."""
    return int(step_idx) % int(TRAIL_PERIOD) == 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from triangular arrows, reflecting off colored wall pixels.

    Variant rule:
    - The ray still advances cell-by-cell along the same reflected diagonal path as in 142ca369.
    - Only every other visited background cell is painted, yielding a dashed trail.
    - Reflection and color inheritance still occur on skipped cells.
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
        start = max(deg, key=lambda p: deg[p])  # right-angle corner
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
        (int(r), int(c)) for r, c in np.argwhere(g != int(bg)) if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()

    for start_col, pts in arrows:
        info = arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
        cur_col = int(start_col)
        step_idx = 0
        seen: set[tuple[tuple[int, int], int, int, int, int]] = set()

        while True:
            phase = int(step_idx) % int(TRAIL_PERIOD)
            state = (cur, int(dr), int(dc), int(cur_col), phase)
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            # Diagonal hit into a wall: corner reflection.
            if (nr, nc) in wall_cells:
                wcol = int(g[nr, nc])
                if _should_paint(step_idx) and int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            step_idx += 1
            paint_now = _should_paint(step_idx)
            if paint_now and int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            # Side-touch bounces still happen on every traversed cell, including skipped ones.
            if int(dr) == 1 and (nr + 1, nc) in wall_cells:
                wcol = int(g[nr + 1, nc])
                if paint_now and int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)
            elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                wcol = int(g[nr - 1, nc])
                if paint_now and int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)

            if int(dc) == 1 and (nr, nc + 1) in wall_cells:
                wcol = int(g[nr, nc + 1])
                if paint_now and int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)
            elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                wcol = int(g[nr, nc - 1])
                if paint_now and int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)

    return out


def _gen_attempt() -> np.ndarray | None:
    """Plan dashed ray paths output-first, then derive the corresponding input."""
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(1, 7)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    total_hidden_steps = 0

    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        n_bounces = _rng.choices([0, 1, 2, 3, 4], weights=[2, 3, 3, 2, 1])[0]

        placed = False
        for _ in range(80):
            r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
            c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)

            mr, mc = sr - dr, sc - dc
            arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
            if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
                continue

            cur_r, cur_c, cur_dr, cur_dc, cur_col = sr, sc, dr, dc, color
            step_idx = 0
            trail_steps: list[list[int | bool]] = []
            walls: list[tuple[list[tuple[int, int]], int]] = []
            ray_set: set[tuple[int, int]] = set()
            arrow_set = set(arrow_pts)
            ok = True
            hidden_steps = 0
            painted_steps = 0
            bounces_left = n_bounces

            for seg_idx in range(n_bounces + 1):
                max_steps = min(
                    (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                    (W - 1 - cur_c) if cur_dc == 1 else cur_c,
                )
                if seg_idx < n_bounces and bounces_left > 0:
                    if max_steps < 4:
                        bounces_left = 0
                        seg_len = max_steps
                    else:
                        seg_len = _rng.randint(3, max(3, max_steps - 1))
                else:
                    seg_len = max_steps

                for _ in range(seg_len):
                    nr, nc = cur_r + cur_dr, cur_c + cur_dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        break
                    if (nr, nc) in occupied or (nr, nc) in ray_set or (nr, nc) in arrow_set:
                        ok = False
                        break
                    step_idx += 1
                    paint_now = _should_paint(step_idx)
                    trail_steps.append([int(nr), int(nc), int(cur_col), bool(paint_now)])
                    if paint_now:
                        painted_steps += 1
                    else:
                        hidden_steps += 1
                    ray_set.add((int(nr), int(nc)))
                    cur_r, cur_c = int(nr), int(nc)
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

                # If the bounce point is one of the painted steps, it adopts the wall color.
                if trail_steps and bool(trail_steps[-1][3]):
                    trail_steps[-1][2] = int(wcol)
                cur_col = int(wcol)
                if flip == "dr":
                    cur_dr = -cur_dr
                else:
                    cur_dc = -cur_dc

            if not ok or len(ray_set) < 4 or hidden_steps < 2 or painted_steps < 2:
                continue

            new_cells = set(arrow_pts) | ray_set
            for wpxs, _ in walls:
                new_cells.update(wpxs)
            expected_count = len(arrow_pts) + len(ray_set) + sum(len(w) for w, _ in walls)
            if len(new_cells) != expected_count or new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = color
                out[r, c] = color
            for r, c, col, paint_now in trail_steps:
                if bool(paint_now):
                    out[int(r), int(c)] = int(col)
            for wpxs, wcol in walls:
                for r, c in wpxs:
                    grid[r, c] = wcol
                    out[r, c] = wcol

            occupied |= new_cells
            total_hidden_steps += hidden_steps
            placed = True
            break

        if not placed:
            return None

    if total_hidden_steps < 2:
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the dashed-trail variant."""
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
