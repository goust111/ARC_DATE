# Variant of 142ca369
# Dimension: D3 — 轨迹涂色模式 (trajectory)
# Change: 连续轨迹（continuous_trail）——射线经过的每个背景格子都被涂色，形成连续的彩色线条... → 仅在反射点周围涂色

from __future__ import annotations

from pathlib import Path
import random as _rng
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import colors_present, components_mask, get_bg


# Design analysis for the D3 variant.
#
# variant_spec:
#   core_change:
#     old: every background cell on the diagonal ray is painted continuously
#     new: only short marker windows around reflections are painted
#   concretization:
#     reflection_marker_radius: 2
#     painted_window_per_bounce:
#       - the 2 cells immediately before the bounce keep the incoming ray color
#       - the bounce cell is recolored to the wall color
#       - the 2 cells immediately after the bounce use the new wall color
#     unpainted_free_flight:
#       - all other ray cells remain background
#   behavior_trace:
#     original_local_flow: a segment of length 9 ending at a wall paints all 9 cells
#     variant_local_flow:
#       - cells 1..6 on that segment stay background
#       - cells 7..8 use the incoming color
#       - cell 9 (bounce point) uses the wall color
#       - the next 2 cells after the turn use the wall color
#     visible_difference:
#       - original output shows long diagonal stripes
#       - variant output shows isolated 5-cell elbows around walls
#   complexity_calibration:
#     - radius is fixed at 2, so the rule stays learnable
#     - generator forces at least 1 bounce per arrow and keeps bounce-to-bounce
#       diagonal distance >= 6, so marker windows do not merge back into a
#       continuous trail
#     - final post-bounce segment length is forced to be >= 4 when possible so
#       the short outgoing marker is visible and still followed by blank flight
#   implicit_assumption_audit:
#     - the original solver safely painted cells eagerly because every traversed
#       ray cell belonged in the output; that assumption is false here
#     - therefore main() must separate geometric tracing from output painting
#     - arrow detection, wall detection, reflection direction changes, and color
#       inheritance remain valid and are kept unchanged in spirit
#     - the new solver assumes generator inputs do not create dual side-touch
#       bounces on the same step; generator enforces one planned wall per bounce
#   placement_legality:
#     - arrows, invisible ray paths, and walls are still pairwise non-overlapping
#     - planned wall centers are checked so no unrelated occupied cell sits in the
#       alternative side-touch slot, avoiding accidental extra bounces
#     - all planned paths are validated with an independent planned-output
#       renderer plus main(input) == expected_output
#   observability:
#     - each grid contains 2-6 arrows, each with 1-4 bounces
#     - long invisible gaps before/after markers make the D3 change obvious from
#       only a few examples
#   visual_neatness:
#     - invisible trajectories are still reserved in the generator, so visible
#       markers from different arrows do not interpenetrate
#   color_inference:
#     - incoming marker cells preserve the pre-bounce color
#     - the center and outgoing marker cells use the wall color, so the wall to
#       reflected-color relation is still directly visible to a human solver

MARK_RADIUS = 2
MIN_BOUNCE_SEG_LEN = 6
MIN_FINAL_SEG_LEN = 4


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


def _paint_bg(out: np.ndarray, g: np.ndarray, r: int, c: int, color: int, bg: int) -> None:
    if int(g[r, c]) == int(bg):
        out[r, c] = int(color)


def _trace_marker_ops(
    g: np.ndarray,
    arrow_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    start_col: int,
    pts: list[tuple[int, int]],
):
    H, W = g.shape
    info = _arrow_dir_and_start(pts)
    if info is None:
        return []
    (dr, dc), cur = info
    cur_col = int(start_col)
    recent: list[tuple[int, int, int]] = []
    ops: list[tuple[int, int, int]] = []
    seen: set[tuple[tuple[int, int], int, int, int]] = set()
    pending_after = 0
    pending_color = int(cur_col)

    def mark_before_and_bounce(bounce_pos: tuple[int, int], wall_color: int) -> None:
        before = recent[-(MARK_RADIUS + 1) : -1]
        for rr, cc, old_col in before:
            ops.append((int(rr), int(cc), int(old_col)))
        ops.append((int(bounce_pos[0]), int(bounce_pos[1]), int(wall_color)))

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
            wall_color = int(g[nr, nc])
            if recent and recent[-1][:2] == cur:
                mark_before_and_bounce(cur, wall_color)
                recent[-1] = (int(cur[0]), int(cur[1]), int(wall_color))
            cur_col = int(wall_color)
            pending_after = MARK_RADIUS
            pending_color = int(wall_color)
            dr, dc = -int(dr), -int(dc)
            continue

        if (nr, nc) in arrow_cells:
            break

        cur = (int(nr), int(nc))
        recent.append((int(nr), int(nc), int(cur_col)))
        if len(recent) > MARK_RADIUS + 1:
            recent.pop(0)

        if pending_after > 0:
            ops.append((int(nr), int(nc), int(pending_color)))
            pending_after -= 1

        bounce_axis = None
        wall_color = None
        if int(dr) == 1 and (nr + 1, nc) in wall_cells:
            bounce_axis = "dr"
            wall_color = int(g[nr + 1, nc])
        elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
            bounce_axis = "dr"
            wall_color = int(g[nr - 1, nc])
        elif int(dc) == 1 and (nr, nc + 1) in wall_cells:
            bounce_axis = "dc"
            wall_color = int(g[nr, nc + 1])
        elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
            bounce_axis = "dc"
            wall_color = int(g[nr, nc - 1])

        if bounce_axis is None or wall_color is None:
            continue

        mark_before_and_bounce(cur, wall_color)
        cur_col = int(wall_color)
        recent[-1] = (int(cur[0]), int(cur[1]), int(wall_color))
        pending_after = MARK_RADIUS
        pending_color = int(wall_color)
        if bounce_axis == "dr":
            dr = -int(dr)
        else:
            dc = -int(dc)

    return ops


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    arrows, arrow_cells = _detect_arrows(g, bg)
    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c)) for r, c in np.argwhere(g != int(bg)) if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()
    for start_col, pts in arrows:
        for r, c, color in _trace_marker_ops(g, arrow_cells, wall_cells, start_col, pts):
            _paint_bg(out, g, int(r), int(c), int(color), bg)
    return out


def _steps_to_edge(r: int, c: int, dr: int, dc: int, H: int, W: int) -> int:
    return min((H - 1 - r) if dr == 1 else r, (W - 1 - c) if dc == 1 else c)


def _wall_options(
    cur_r: int,
    cur_c: int,
    cur_dr: int,
    cur_dc: int,
    H: int,
    W: int,
):
    options: list[tuple[list[tuple[int, int]], str, tuple[int, int]]] = []

    wr, wc = cur_r + cur_dr, cur_c
    if 0 <= wr < H:
        if 0 <= wc - 1 and wc + 1 < W:
            options.append(([(wr, wc - 1), (wr, wc), (wr, wc + 1)], "dr", (wr, wc)))
        options.append(([(wr, wc)], "dr", (wr, wc)))

    wr2, wc2 = cur_r, cur_c + cur_dc
    if 0 <= wc2 < W:
        if 0 <= wr2 - 1 and wr2 + 1 < H:
            options.append(([(wr2 - 1, wc2), (wr2, wc2), (wr2 + 1, wc2)], "dc", (wr2, wc2)))
        options.append(([(wr2, wc2)], "dc", (wr2, wc2)))
    return options


def _render_plan_output(grid: np.ndarray, plans: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    for plan in plans:
        path = plan["path"]
        for bounce_idx, wall_color in plan["bounces"]:
            for idx in range(max(0, bounce_idx - MARK_RADIUS), bounce_idx):
                r, c, col = path[idx]
                if int(grid[r, c]) == 0:
                    out[r, c] = int(col)
            br, bc, _ = path[bounce_idx]
            if int(grid[br, bc]) == 0:
                out[br, bc] = int(wall_color)
            for idx in range(bounce_idx + 1, min(len(path), bounce_idx + 1 + MARK_RADIUS)):
                r, c, _ = path[idx]
                if int(grid[r, c]) == 0:
                    out[r, c] = int(wall_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(2, 6)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]

    grid = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    plans: list[dict] = []

    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        n_bounces = _rng.choices([1, 2, 3, 4], weights=[2, 4, 3, 1])[0]

        placed = False
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
            cur_col = color
            ray_set: set[tuple[int, int]] = set()
            wall_set: set[tuple[int, int]] = set()
            path: list[tuple[int, int, int]] = []
            walls: list[tuple[list[tuple[int, int]], int]] = []
            bounces: list[tuple[int, int]] = []
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
                    seg_cells = [(cur_r + cur_dr * step, cur_c + cur_dc * step) for step in range(1, seg_len + 1)]
                    if any(cell in occupied or cell in ray_set or cell in arrow_set or cell in wall_set for cell in seg_cells):
                        continue

                    bounce_r, bounce_c = seg_cells[-1]
                    for wall_cells, flip, touch_center in _wall_options(bounce_r, bounce_c, cur_dr, cur_dc, H, W):
                        if any(cell in occupied or cell in ray_set or cell in arrow_set or cell in wall_set for cell in wall_cells):
                            continue
                        alt_touch = (bounce_r, bounce_c + cur_dc) if flip == "dr" else (bounce_r + cur_dr, bounce_c)
                        if alt_touch in occupied or alt_touch in wall_set or alt_touch in arrow_set:
                            continue

                        new_dr = -cur_dr if flip == "dr" else cur_dr
                        new_dc = -cur_dc if flip == "dc" else cur_dc
                        future_need = MIN_BOUNCE_SEG_LEN if bounce_idx < n_bounces - 1 else MIN_FINAL_SEG_LEN
                        if _steps_to_edge(bounce_r, bounce_c, new_dr, new_dc, H, W) < future_need:
                            continue
                        candidates.append((seg_cells, wall_cells, flip, new_dr, new_dc))

                if not candidates:
                    ok = False
                    break

                seg_cells, wall_cells, flip, next_dr, next_dc = _rng.choice(candidates)
                for r, c in seg_cells:
                    path.append((int(r), int(c), int(cur_col)))
                    ray_set.add((int(r), int(c)))
                wall_color = int(_rng.choice(palette))
                walls.append(([(int(r), int(c)) for r, c in wall_cells], wall_color))
                wall_set.update((int(r), int(c)) for r, c in wall_cells)
                bounces.append((len(path) - 1, wall_color))
                cur_r, cur_c = seg_cells[-1]
                cur_dr, cur_dc = next_dr, next_dc
                cur_col = wall_color

            if not ok:
                continue

            final_steps = _steps_to_edge(cur_r, cur_c, cur_dr, cur_dc, H, W)
            if final_steps < MIN_FINAL_SEG_LEN:
                continue
            final_cells = [(cur_r + cur_dr * step, cur_c + cur_dc * step) for step in range(1, final_steps + 1)]
            if any(cell in occupied or cell in ray_set or cell in arrow_set or cell in wall_set for cell in final_cells):
                continue
            for r, c in final_cells:
                path.append((int(r), int(c), int(cur_col)))
                ray_set.add((int(r), int(c)))

            new_cells = set(arrow_pts) | ray_set | wall_set
            expected_count = len(arrow_pts) + len(ray_set) + len(wall_set)
            if len(new_cells) != expected_count or new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = color
            for wpxs, wcol in walls:
                for r, c in wpxs:
                    grid[r, c] = wcol

            occupied |= new_cells
            plans.append(
                {
                    "arrow_pts": arrow_pts,
                    "path": path,
                    "bounces": bounces,
                    "walls": walls,
                }
            )
            placed = True
            break

        if not placed:
            return None

    expected = _render_plan_output(grid, plans)
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
