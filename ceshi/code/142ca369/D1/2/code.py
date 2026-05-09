# Variant of 142ca369
# Dimension: D1 — 射线路径方向 (path)
# Change: 对角线射线（diagonal_ray）——箭头沿45度对角线方向发射射线，每步同时移动一行一列... → 先直后弯路径

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

# Design analysis for the D1 variant
# ----------------------------------
# spec:
#   preserved_objects:
#     grid: 18-20 rows x 18-20 cols; bg = 0
#     arrows:
#       shape: 3 pixels in a 2x2 bbox with one corner missing
#       direction_code:
#         missing corner still determines the signed pair (dr, dc)
#         the right-angle vertex is still the ray origin
#     walls:
#       shape: single pixel or 3-pixel horizontal line
#       role: walls are the only non-arrow blockers and still donate color
#   path_variant:
#     global_branch_choice:
#       choose one branch globally for the dataset: every ray starts straight first
#       reason: if some arrows start straight and others start diagonal, an extra cue
#         would be needed; the original arrow shape does not encode that bit
#     straight_mode:
#       exact_motion: advance one cell vertically by (dr, 0)
#       side_touch: after landing on (r, c), if a wall exists at (r, c + dc),
#         the current cell is recolored to the wall color, dc flips sign, and the
#         ray switches to diagonal mode
#     diagonal_mode:
#       exact_motion: advance one cell diagonally by (dr, dc)
#       side_touch: after landing on (r, c), if a wall exists at (r + dr, c),
#         the current cell is recolored to the wall color, dr flips sign, and the
#         ray switches back to straight mode
#     head_on_hits:
#       not used by the generator, but handled defensively
#       straight head-on wall: adopt wall color, flip dr, switch to diagonal
#       diagonal head-on wall: adopt wall color, flip both axes, switch to straight
#   concrete_behavior_trace:
#     original_at_key_step:
#       arrow with (dr, dc) = (1, 1) goes (r, c) -> (r+1, c+1) -> (r+2, c+2) ...
#       a side wall at (r+2, c+3) flips dc while remaining diagonal
#     variant_at_same_stage:
#       same arrow first goes straight: (r, c) -> (r+1, c) -> (r+2, c) ...
#       wall at (r+2, c+1) recolors (r+2, c), flips dc, and next mode is diagonal:
#       (r+2, c) -> (r+3, c-1) -> (r+4, c-2) ...
#       the next wall is placed at (r+5, c-2), so the ray flips dr and returns to
#       vertical motion
#     visible_difference:
#       the path is no longer symmetric diagonal zigzag; it alternates long vertical
#       runs and slanted runs, producing a clear sawtooth silhouette
#   complexity_calibration:
#     keep:
#       arrow detection, color inheritance, wall-as-input-only objects, output-first
#         generation, no-overlap policy
#     adapt:
#       require at least one arrow with >=2 bounces so the mode toggle is actually
#         visible in a typical sample, not just a single bend
#       restrict 3-pixel walls to horizontal segments; this avoids accidental early
#         triggers next to vertical straight segments
#   implicit_assumption_audit:
#     original_solver_dependencies_broken_by_variant:
#       - next cell was always (cur_r + dr, cur_c + dc); no longer true in straight mode
#       - side-touch checks used both axes every step; now the relevant side depends on mode
#       - flipping dr/dc alone used to preserve motion type; now motion type itself toggles
#       - generator assumed every segment consumed both row and col slack equally; vertical
#         segments now consume only row slack and need different wall placement
#     new_solver_assumptions_and_generator_guarantees:
#       - arrows are still isolated 3-cell monochrome L-shapes
#       - planned side walls sit exactly at the cells the solver checks for the active mode
#       - no wall is placed so that it would trigger an earlier unplanned bounce
#       - rays, arrows, and walls remain pairwise non-overlapping in the full output
#   placement_legality:
#     arrows use the original 2x2 bbox construction
#     vertical segments never step onto occupied cells
#     diagonal segments never step onto occupied cells
#     each planned bounce places a side wall wholly in-bounds and off the ray cells
#     the final candidate is accepted only if main(input) exactly matches the planned output
#   visual_observability:
#     every sample contains at least one 2+ bounce ray, so humans can see
#       straight -> diagonal -> straight alternation
#     wall colors still mark the bend points, which makes the mode switches readable
#   cleanliness:
#     only three element types remain on the board: arrows, isolated/horizontal walls,
#       and the eventual painted trace
#     generator rejects crowded placements rather than stacking cues in the same area


STRAIGHT = "straight"
DIAGONAL = "diagonal"


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


def _detect_scene(g: np.ndarray) -> tuple[int, list[tuple[int, list[tuple[int, int]]]], set[tuple[int, int]], set[tuple[int, int]]]:
    bg = int(get_bg(g))
    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)
    wall_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }
    return bg, arrows, arrow_cells, wall_cells


def _trace_variant(grid: np.ndarray, *, collect_stats: bool = False):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, arrows, arrow_cells, wall_cells = _detect_scene(g)
    out = g.copy()
    stats: list[dict[str, int | str | tuple[int, int]]] = []

    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
        cur_col = int(start_col)
        mode = STRAIGHT
        seen: set[tuple[tuple[int, int], int, int, int, str]] = set()
        straight_steps = 0
        diagonal_steps = 0
        bounces = 0

        while True:
            state = (cur, int(dr), int(dc), int(cur_col), mode)
            if state in seen:
                break
            seen.add(state)

            step_dr = int(dr)
            step_dc = 0 if mode == STRAIGHT else int(dc)
            nr = int(cur[0] + step_dr)
            nc = int(cur[1] + step_dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in wall_cells:
                wcol = int(g[nr, nc])
                if int(g[int(cur[0]), int(cur[1])]) == bg:
                    out[int(cur[0]), int(cur[1])] = wcol
                cur_col = wcol
                bounces += 1
                if mode == STRAIGHT:
                    dr = -int(dr)
                    mode = DIAGONAL
                else:
                    dr, dc = -int(dr), -int(dc)
                    mode = STRAIGHT
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (nr, nc)
            if int(out[nr, nc]) == bg:
                out[nr, nc] = int(cur_col)
            if mode == STRAIGHT:
                straight_steps += 1
                side = (nr, int(nc + dc))
                if 0 <= side[0] < H and 0 <= side[1] < W and side in wall_cells:
                    wcol = int(g[side[0], side[1]])
                    if int(g[nr, nc]) == bg:
                        out[nr, nc] = wcol
                    cur_col = wcol
                    dc = -int(dc)
                    mode = DIAGONAL
                    bounces += 1
            else:
                diagonal_steps += 1
                side = (int(nr + dr), nc)
                if 0 <= side[0] < H and 0 <= side[1] < W and side in wall_cells:
                    wcol = int(g[side[0], side[1]])
                    if int(g[nr, nc]) == bg:
                        out[nr, nc] = wcol
                    cur_col = wcol
                    dr = -int(dr)
                    mode = STRAIGHT
                    bounces += 1

        if collect_stats:
            stats.append(
                {
                    "start": cur,
                    "bounces": bounces,
                    "straight_steps": straight_steps,
                    "diagonal_steps": diagonal_steps,
                }
            )

    if collect_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace alternating vertical and diagonal rays from L-shaped arrows.

    Input:
    - background grid with monochrome 3-pixel L-shaped arrows
    - non-arrow wall pixels / short wall segments

    Variant rule:
    - The missing corner still sets the signed pair (dr, dc), and the right-angle
      corner is still the ray origin.
    - Every ray starts in straight mode and moves vertically by (dr, 0).
    - A side wall at (r, c + dc) recolors the current ray cell, flips dc, and
      switches the ray into diagonal mode.
    - In diagonal mode the ray moves by (dr, dc).
    - A side wall at (r + dr, c) recolors the current ray cell, flips dr, and
      switches the ray back into straight mode.
    - On each bounce the ray inherits the wall color.
    - Rays stop at the boundary or when stepping into another arrow.
    """
    return _trace_variant(grid, collect_stats=False)


def _max_steps(H: int, W: int, r: int, c: int, dr: int, dc: int, mode: str) -> int:
    if mode == STRAIGHT:
        return (H - 1 - r) if dr == 1 else r
    return min((H - 1 - r) if dr == 1 else r, (W - 1 - c) if dc == 1 else c)


def _choose_wall_shape(center_r: int, center_c: int, H: int, W: int) -> list[list[tuple[int, int]]]:
    options: list[list[tuple[int, int]]] = [[(center_r, center_c)]]
    h3 = [(center_r, center_c - 1), (center_r, center_c), (center_r, center_c + 1)]
    if all(0 <= rr < H and 0 <= cc < W for rr, cc in h3):
        options.append(h3)
    _rng.shuffle(options)
    return options


def _planned_output_from_parts(
    H: int,
    W: int,
    placements: list[tuple[list[tuple[int, int]], list[tuple[int, int, int]], list[tuple[list[tuple[int, int]], int]]]],
    arrow_colors: list[int],
) -> np.ndarray:
    out = np.zeros((H, W), dtype=int)
    for color, (arrow_pts, ray_cells, walls) in zip(arrow_colors, placements, strict=True):
        for r, c in arrow_pts:
            out[r, c] = color
        for r, c, col in ray_cells:
            out[r, c] = col
        for wpxs, wcol in walls:
            for r, c in wpxs:
                out[r, c] = wcol
    return out


def _gen_attempt() -> np.ndarray | None:
    """
    Output-first generator for the straight-then-diagonal path variant.

    Each arrow plans an alternating sequence of:
    - straight vertical segment
    - diagonal segment
    - straight vertical segment
    - ...

    and places a side wall at each segment end. The candidate is accepted only if
    the solver reproduces the planned output exactly.
    """
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(1, 6)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]

    grid = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    committed_walls: set[tuple[int, int]] = set()
    placements: list[tuple[list[tuple[int, int]], list[tuple[int, int, int]], list[tuple[list[tuple[int, int]], int]]]] = []
    must_have_multi_toggle = _rng.randrange(n_arrows)

    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        if ai == must_have_multi_toggle:
            n_bounces = _rng.choice([2, 3, 4])
        else:
            n_bounces = _rng.choices([1, 2, 3, 4], weights=[2, 4, 3, 1])[0]

        placed = False
        for _ in range(160):
            r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
            c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)

            mr, mc = sr - dr, sc - dc
            arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
            arrow_set = set(arrow_pts)
            if any((r, c) in occupied or not (0 <= r < H and 0 <= c < W) for r, c in arrow_pts):
                continue

            cur_r, cur_c = sr, sc
            cur_dr, cur_dc = dr, dc
            cur_col = color
            mode = STRAIGHT
            ray_cells: list[tuple[int, int, int]] = []
            ray_set: set[tuple[int, int]] = set()
            walls: list[tuple[list[tuple[int, int]], int]] = []
            local_walls: set[tuple[int, int]] = set()
            ok = True

            for seg_idx in range(n_bounces + 1):
                max_steps = _max_steps(H, W, cur_r, cur_c, cur_dr, cur_dc, mode)
                if max_steps <= 0:
                    ok = False
                    break

                is_last = seg_idx == n_bounces
                lower = 2 if max_steps >= 2 else 1
                upper = max_steps

                if not is_last:
                    if mode == STRAIGHT:
                        upper = min(upper, (H - 2 - cur_r) if cur_dr == 1 else (cur_r - 1))
                    if upper < lower:
                        ok = False
                        break
                    seg_len = _rng.randint(lower, upper)
                else:
                    seg_len = upper

                for step_idx in range(seg_len):
                    step_dr = cur_dr
                    step_dc = 0 if mode == STRAIGHT else cur_dc
                    nr, nc = cur_r + step_dr, cur_c + step_dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        ok = False
                        break
                    if (nr, nc) in occupied or (nr, nc) in ray_set or (nr, nc) in arrow_set or (nr, nc) in local_walls:
                        ok = False
                        break

                    wall_view = committed_walls | local_walls
                    if mode == STRAIGHT:
                        side = (nr, nc + cur_dc)
                        if 0 <= side[0] < H and 0 <= side[1] < W and side in wall_view:
                            ok = False
                            break
                    else:
                        side = (nr + cur_dr, nc)
                        if 0 <= side[0] < H and 0 <= side[1] < W and side in wall_view:
                            ok = False
                            break

                    ray_cells.append((nr, nc, cur_col))
                    ray_set.add((nr, nc))
                    cur_r, cur_c = nr, nc

                if not ok:
                    break

                if is_last:
                    continue

                if mode == STRAIGHT:
                    center = (cur_r, cur_c + cur_dc)
                else:
                    center = (cur_r + cur_dr, cur_c)
                if not (0 <= center[0] < H and 0 <= center[1] < W):
                    ok = False
                    break

                chosen_wall = None
                for wpxs in _choose_wall_shape(center[0], center[1], H, W):
                    if any((r, c) in occupied or (r, c) in ray_set or (r, c) in arrow_set or (r, c) in local_walls for r, c in wpxs):
                        continue
                    chosen_wall = wpxs
                    break
                if chosen_wall is None:
                    ok = False
                    break

                wcol = int(_rng.choice(palette))
                walls.append((chosen_wall, wcol))
                local_walls.update(chosen_wall)
                if ray_cells:
                    lr, lc, _ = ray_cells[-1]
                    ray_cells[-1] = (lr, lc, wcol)
                cur_col = wcol
                if mode == STRAIGHT:
                    cur_dc = -cur_dc
                    mode = DIAGONAL
                else:
                    cur_dr = -cur_dr
                    mode = STRAIGHT

            if not ok or len(ray_cells) < 4:
                continue

            new_cells = set(arrow_pts) | ray_set | local_walls
            expected_count = len(arrow_pts) + len(ray_cells) + sum(len(w) for w, _ in walls)
            if len(new_cells) != expected_count:
                continue
            if new_cells & occupied:
                continue

            placements.append((arrow_pts, ray_cells, walls))
            for r, c in arrow_pts:
                grid[r, c] = color
            for wpxs, wcol in walls:
                for r, c in wpxs:
                    grid[r, c] = wcol
            occupied |= new_cells
            committed_walls |= local_walls
            placed = True
            break

        if not placed:
            return None

    out = _planned_output_from_parts(H, W, placements, arrow_colors)
    solved = main(grid)
    if not np.array_equal(solved, out):
        return None

    _solved_out, stats = _trace_variant(grid, collect_stats=True)
    if not stats:
        return None
    if max(int(s["bounces"]) for s in stats) < 2:
        return None
    if sum(int(s["straight_steps"]) > 0 and int(s["diagonal_steps"]) > 0 for s in stats) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D1 path-variant of 142ca369."""
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
