# Combo variant of 142ca369 (index 24)
# combo_spec: D1/0+D13/0
# Name: 直线射线+正面阻挡墙壁

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

"""
Design analysis for the combo variant
=====================================

combo:
  task: 142ca369
  combo_spec: D1/0 + D13/0
  name: straight_ray + frontal_blocking_wall

a) concretization_and_trace:
  preserved_input_language:
    arrows:
      same 3-cell monochrome L-shapes in a 2x2 bbox.
      The missing corner still defines a signed pair (dr, dc).
      The ray still starts at the right-angle vertex.
    walls:
      now only single pixels, and each planned bounce wall sits directly on the
      active ray path instead of one cell to the side.
  fused_motion_rule:
    - The signed pair (dr, dc) is still extracted from the arrow.
    - Every ray starts in horizontal mode and advances by (0, dc).
      reason: the arrow shape itself does not encode whether to start horizontal
      or vertical, so this must be a global convention.
    - On a frontal hit, the ray does not enter the wall cell.
      Instead it:
        1. recolors the current ray cell to the wall color,
        2. adopts that wall color for future painting,
        3. flips both latent signs: (dr, dc) -> (-dr, -dc),
        4. toggles to the other axis.
    - Therefore a horizontal segment hitting a wall turns into a vertical
      segment in direction (-dr, 0), and a vertical segment hitting a wall turns
      into a horizontal segment in direction (0, -dc).
  concrete_trace:
    original_task:
      arrow gives (dr, dc) = (1, 1)
      path: (r,c) -> (r+1,c+1) -> ...
      side wall one cell below/aside flips only one component and keeps the ray
      in a diagonal regime.
    combo_variant:
      same arrow gives latent (dr, dc) = (1, 1)
      start horizontal:
        (r,c) -> (r,c+1) -> (r,c+2)
      wall placed at (r,c+3)
      bounce:
        current cell (r,c+2) becomes wall color
        latent pair becomes (-1,-1)
        active axis changes to vertical
      next segment:
        (r-1,c+2) -> (r-2,c+2) -> ...
    visible_difference:
      the original makes diagonal zigzags driven by side-touch adjacency.
      the combo makes axis-aligned corridor turns driven by on-path blockers.

b) complexity_calibration:
  - If we kept side-touch checks together with D13 frontal walls, many inputs
    would admit two plausible wall semantics. The combo removes side-touch
    behavior entirely.
  - If all rays were allowed to have 0 or 1 bounce, many samples would look like
    a plain straight line. The generator therefore forces at least one arrow to
    have >= 2 frontal bounces so the axis toggling is visible.
  - Walls are restricted to single pixels. A 3-pixel line centered on the path
    would visually resemble a barrier object rather than a point blocker and
    would introduce extra unused pixels on adjacent lanes.
  - The global "horizontal-first" convention limits freedom to one dataset-wide
    rule rather than adding a hidden per-arrow bit not encoded by the input.

c) hidden_assumption_audit_against_original_main:
  - `nr, nc = cur + (dr, dc)`:
      broken. The combo no longer advances diagonally on every step.
      Replacement: use an explicit active axis and step by either (0, dc) or
      (dr, 0).
  - `if (nr, nc) in wall_cells: dr, dc = -dr, -dc`:
      still partly valid. Frontal walls are still on the next path cell and they
      still flip both signs, but they now also toggle the active axis.
  - arrow blockers:
      unlike pure D13 diagonal foldback, this combo does not require passing back
      through the source arrow. So arrow cells can remain hard blockers.
  - side-touch checks against `(nr+dr, nc)` and `(nr, nc+dc)`:
      invalid. They depended on the old off-path obstacle geometry and must be
      removed completely.
  - original generator's slack logic:
      invalid. Horizontal and vertical segments consume row/col budget
      separately, so planning must reason about the active axis explicitly.

d) placement_legality:
  - Arrow cells, ray cells, and wall cells are all reserved before committing an
    arrow.
  - For a bounced segment of length L, the next cell on the active axis is the
    wall and must be in-bounds, unused, and not an arrow cell.
  - Each future segment must also remain disjoint from all earlier ray cells and
    walls so the solver cannot trigger an unplanned early bounce.
  - The final no-bounce segment runs straight to the border; every painted cell
    on that run must be free.

e) visual_observability:
  - In input, the key clue is that blocker pixels lie exactly at the end of a
    straight corridor implied by the arrow direction.
  - In output, at least one arrow visibly uses both axes and turns multiple
    times. This separates the combo from both the original diagonal task and a
    degenerate "just draw one straight line" simplification.
  - Color inheritance stays visible because each bounce recolors the turning cell
    and the following segment extends with the new color.

f) visual_cleanliness:
  - Only three visual object types remain: L-arrows, isolated wall pixels, and
    axis-aligned ray traces.
  - The generator rejects overlaps and self-intersections, so different arrows do
    not create ambiguous crossings or mixed-color joints.
  - Grid size and density stay close to the original task, which keeps the board
    readable.

g) color_change_inference:
  - The ray always starts in the arrow color and only changes when it frontally
    hits a wall.
  - Wall colors are biased to differ from the incoming ray color whenever
    possible, making the inheritance rule visible from a few examples.
  - The recolored pivot cell and the continuation segment provide two coupled
    cues for the same color transition, which is stronger than a hidden lookup.

new_solver_assumptions_and_generator_guarantees:
  - assumption: all arrows use the same horizontal-first convention.
    guarantee: generator always plans rays that way.
  - assumption: every non-arrow colored pixel is a frontal wall, not a side wall.
    guarantee: generator places walls only on planned hit cells and nowhere else.
  - assumption: arrow cells are blockers for every ray.
    guarantee: generator never places a path that would need to traverse any
      arrow cell after launch.
  - assumption: rays do not self-intersect.
    guarantee: generator backtracks on any candidate whose planned cells repeat.
"""

HORIZONTAL = "horizontal"
VERTICAL = "vertical"


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


def _find_arrows(g: np.ndarray, bg: int) -> tuple[list[dict[str, object]], set[tuple[int, int]]]:
    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[dict[str, object]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if not _is_arrow_pts(pts):
                continue
            info = _arrow_dir_and_start(pts)
            if info is None:
                continue
            (dr, dc), start = info
            cells = set(pts)
            arrows.append(
                {
                    "color": int(col),
                    "cells": cells,
                    "dir": (int(dr), int(dc)),
                    "start": (int(start[0]), int(start[1])),
                }
            )
            arrow_cells |= cells
    return arrows, arrow_cells


def _step_delta(axis: str, dr: int, dc: int) -> tuple[int, int]:
    if axis == HORIZONTAL:
        return 0, int(dc)
    return int(dr), 0


def _steps_to_edge(H: int, W: int, r: int, c: int, axis: str, dr: int, dc: int) -> int:
    if axis == HORIZONTAL:
        return (W - 1 - c) if dc == 1 else c
    return (H - 1 - r) if dr == 1 else r


def _trace_variant(grid: np.ndarray, *, collect_stats: bool = False):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    arrows, arrow_cells = _find_arrows(g, bg)
    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()
    stats: list[dict[str, int | tuple[int, int] | str]] = []

    for arrow in arrows:
        dr, dc = arrow["dir"]  # type: ignore[assignment]
        cur = arrow["start"]  # type: ignore[assignment]
        cur_col = int(arrow["color"])
        axis = HORIZONTAL
        seen: set[tuple[tuple[int, int], int, int, int, str]] = set()
        horizontal_steps = 0
        vertical_steps = 0
        bounces = 0

        while True:
            cur = (int(cur[0]), int(cur[1]))
            state = (cur, int(dr), int(dc), int(cur_col), axis)
            if state in seen:
                break
            seen.add(state)

            step_r, step_c = _step_delta(axis, int(dr), int(dc))
            nr = int(cur[0] + step_r)
            nc = int(cur[1] + step_c)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in wall_cells:
                wcol = int(g[nr, nc])
                if int(g[cur[0], cur[1]]) == int(bg):
                    out[cur[0], cur[1]] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                axis = VERTICAL if axis == HORIZONTAL else HORIZONTAL
                bounces += 1
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)
            if axis == HORIZONTAL:
                horizontal_steps += 1
            else:
                vertical_steps += 1

        if collect_stats:
            stats.append(
                {
                    "start": (int(arrow["start"][0]), int(arrow["start"][1])),  # type: ignore[index]
                    "axis0": axis,
                    "bounces": int(bounces),
                    "horizontal_steps": int(horizontal_steps),
                    "vertical_steps": int(vertical_steps),
                }
            )

    if collect_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Detect each monochrome 3-cell L-arrow as in the original task.
    - The missing corner still yields a latent signed pair (dr, dc).
    - Every ray starts in horizontal mode and moves by (0, dc).
    - A wall directly on the next ray cell is a frontal blocker:
      the ray recolors the current cell to the wall color, adopts that color,
      flips both signs, and switches to the other axis.
    - Hence horizontal motion bounces into vertical motion, and vertical motion
      bounces into horizontal motion.
    - Rays stop when they leave the grid or hit any arrow cell.
    """
    return _trace_variant(grid, collect_stats=False)


def _arrow_points_from_start(sr: int, sc: int, dr: int, dc: int) -> list[tuple[int, int]]:
    mr, mc = sr - dr, sc - dc
    return [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]


def _pick_wall_color(cur_color: int, palette: list[int]) -> int:
    choices = [c for c in palette if c != cur_color]
    if choices:
        return int(_rng.choice(choices))
    return int(cur_color)


def _build_colored_ray(
    arrow_color: int,
    segments: list[list[tuple[int, int]]],
    wall_colors: list[int],
) -> list[tuple[int, int, int]]:
    colored: list[tuple[int, int, int]] = []
    cur_color = int(arrow_color)
    for seg_idx, seg in enumerate(segments):
        for r, c in seg:
            colored.append((int(r), int(c), int(cur_color)))
        if seg_idx < len(wall_colors) and seg:
            wcol = int(wall_colors[seg_idx])
            lr, lc, _ = colored[-1]
            colored[-1] = (int(lr), int(lc), int(wcol))
            cur_color = int(wcol)
    return colored


def _plan_segments(
    H: int,
    W: int,
    cur: tuple[int, int],
    dr: int,
    dc: int,
    axis: str,
    bounces_left: int,
    blocked: set[tuple[int, int]],
    ray_seen: set[tuple[int, int]],
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    max_steps = _steps_to_edge(H, W, int(cur[0]), int(cur[1]), axis, int(dr), int(dc))
    step_r, step_c = _step_delta(axis, int(dr), int(dc))

    if bounces_left == 0:
        if max_steps < 3:
            return None
        seg: list[tuple[int, int]] = []
        r, c = int(cur[0]), int(cur[1])
        for _ in range(int(max_steps)):
            r += int(step_r)
            c += int(step_c)
            cell = (int(r), int(c))
            if cell in blocked or cell in ray_seen:
                return None
            seg.append(cell)
        return [seg], []

    min_seg = 2
    if max_steps < min_seg + 1:
        return None

    candidates = list(range(min_seg, int(max_steps)))
    _rng.shuffle(candidates)
    for seg_len in candidates:
        seg: list[tuple[int, int]] = []
        r, c = int(cur[0]), int(cur[1])
        ok = True
        for _ in range(int(seg_len)):
            r += int(step_r)
            c += int(step_c)
            cell = (int(r), int(c))
            if cell in blocked or cell in ray_seen:
                ok = False
                break
            seg.append(cell)
        if not ok or not seg:
            continue

        wall = (int(r + step_r), int(c + step_c))
        if not (0 <= wall[0] < H and 0 <= wall[1] < W):
            continue
        if wall in blocked or wall in ray_seen or wall in seg:
            continue

        next_axis = VERTICAL if axis == HORIZONTAL else HORIZONTAL
        next_blocked = set(blocked)
        next_blocked.add(wall)
        next_ray_seen = set(ray_seen)
        next_ray_seen.update(seg)
        rest = _plan_segments(
            H,
            W,
            seg[-1],
            -int(dr),
            -int(dc),
            next_axis,
            int(bounces_left) - 1,
            next_blocked,
            next_ray_seen,
        )
        if rest is None:
            continue
        rest_segments, rest_walls = rest
        return [seg] + rest_segments, [wall] + rest_walls

    return None


def _gen_attempt() -> np.ndarray | None:
    """
    Output-first generator for the orthogonal frontal-bounce combo.

    Each arrow plans a sequence of axis-aligned segments. Every bounce is caused
    by a wall pixel directly on the path; the solver then flips both latent signs
    and toggles to the other axis. Candidates are accepted only if:
    - all placements are disjoint,
    - at least one arrow has >= 2 bounces,
    - `main(input)` reproduces the planned output exactly.
    """
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(1, 6)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = [int(c) for c in palette[:n_arrows]]

    grid = np.zeros((H, W), dtype=int)
    planned_out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    has_multi_bounce = False
    force_multi_idx = _rng.randrange(n_arrows)

    for ai, color in enumerate(arrow_colors):
        placed = False
        for _ in range(220):
            dr = int(_rng.choice([-1, 1]))
            dc = int(_rng.choice([-1, 1]))
            if ai == force_multi_idx:
                n_bounces = int(_rng.choice([2, 3, 4]))
            else:
                n_bounces = int(_rng.choices([0, 1, 2, 3], weights=[1, 3, 3, 1])[0])

            r_lo, r_hi = (1 if dr == 1 else 0), (H - 1 if dr == 1 else H - 2)
            c_lo, c_hi = (1 if dc == 1 else 0), (W - 1 if dc == 1 else W - 2)
            if r_lo > r_hi or c_lo > c_hi:
                continue

            sr = int(_rng.randint(r_lo, r_hi))
            sc = int(_rng.randint(c_lo, c_hi))
            arrow_pts = _arrow_points_from_start(sr, sc, dr, dc)
            arrow_set = set(arrow_pts)
            if any(not (0 <= r < H and 0 <= c < W) for r, c in arrow_pts):
                continue
            if arrow_set & occupied:
                continue

            geometry = _plan_segments(
                H,
                W,
                (sr, sc),
                dr,
                dc,
                HORIZONTAL,
                n_bounces,
                occupied | arrow_set,
                set(),
            )
            if geometry is None:
                continue
            segments, walls = geometry
            if len(walls) != n_bounces:
                continue

            ray_cells_plain = [cell for seg in segments for cell in seg]
            ray_set = set(ray_cells_plain)
            wall_set = set(walls)
            if len(ray_set) != len(ray_cells_plain):
                continue
            if arrow_set & ray_set or arrow_set & wall_set or ray_set & wall_set:
                continue

            wall_colors = []
            cur_color = int(color)
            for _wall in walls:
                wcol = _pick_wall_color(cur_color, palette)
                wall_colors.append(int(wcol))
                cur_color = int(wcol)
            colored_ray = _build_colored_ray(int(color), segments, wall_colors)
            if len(colored_ray) < 4:
                continue

            new_cells = set(arrow_pts) | ray_set | wall_set
            if new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = int(color)
                planned_out[r, c] = int(color)
            for r, c, col in colored_ray:
                planned_out[r, c] = int(col)
            for (r, c), wcol in zip(walls, wall_colors, strict=True):
                grid[r, c] = int(wcol)
                planned_out[r, c] = int(wcol)

            occupied |= new_cells
            has_multi_bounce = has_multi_bounce or (n_bounces >= 2)
            placed = True
            break

        if not placed:
            return None

    if not has_multi_bounce:
        return None

    solved = main(grid)
    if not np.array_equal(solved, planned_out):
        return None
    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D1/0 + D13/0 combo variant."""
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate valid combo variant input after 400 attempts")
