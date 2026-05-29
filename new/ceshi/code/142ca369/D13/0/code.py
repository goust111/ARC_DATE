# Variant of 142ca369
# Dimension: D13 — 墙壁与射线的空间关系 (obstacle)
# Change: 墙壁放置在射线对角线路径旁偏移1格处（侧面接触），不直接在射线路径上... → 墙壁直接在射线路径上（正面阻挡）

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

"""
Design analysis for the D13 variant
===================================

source_task:
  value: None
  note: No separate inspiration task was provided, so the design is derived only
    from the original solver/generator/spec and the requested D13 flip.

a) concretization_and_trace:
  input_objects:
    arrows:
      same as original: monochrome 3-cell L-shapes in a 2x2 box.
      The missing corner still defines the initial diagonal firing direction.
    walls:
      changed: walls are now single pixels placed directly on the ray's diagonal
      line, not one cell to the side.
  output_effect:
    - The ray travels diagonally in the arrow color.
    - When the next diagonal cell is a wall pixel, the ray does not enter that
      cell; instead it "frontally" bounces by reversing both direction
      components and adopting the wall color.
    - To make the bounce visible, the ray is allowed to pass through its own
      source arrow's right-angle vertex on the way back, then continue out on
      the opposite side of the same diagonal.
    - Other arrows still block the ray.
  concrete_trace:
    original_side_touch:
      start at (5,5), dir=(1,1), wall at (6,5).
      Ray advances to (6,6), sees a side wall below, flips only dr, then keeps
      zigzagging forward on a new diagonal.
    variant_head_on:
      start at (5,5), dir=(1,1), wall at (8,8).
      Ray paints (6,6),(7,7), sees wall in the next diagonal step, recolors
      current cell to the wall color, reverses to (-1,-1), passes back through
      the source vertex, and then extends toward the opposite border.
    difference:
      The old geometry is a forward zigzag produced by lateral contacts. The
      new geometry is an on-line foldback around a frontal blocker.

b) complexity_calibration:
  - If the solver only changed the bounce test but still treated every arrow
    cell as a hard blocker, the ray would hit the wall, return to its own
    source, and stop immediately. That would make most bounced outputs look
    almost identical to a truncated straight ray, which is too weak visually.
  - Therefore the variant keeps the original arrow language but relaxes exactly
    one assumption: the ray may traverse its own arrow cells, while foreign
    arrows remain blockers.
  - Wall shapes are reduced to single pixels. Keeping 3-pixel side-adjacent
    bars would mix frontal and lateral cues in the same example, obscuring the
    D13 change.
  - Each generated grid is forced to contain at least one bounced arrow, and
    the bounced ray must have visible extension on the return side.

c) hidden_assumptions_in_original_main:
  - `if (nr, nc) in wall_cells: ... dr, dc = -dr, -dc`:
    This branch already implements head-on reversal, so it is reusable.
  - `if (nr, nc) in arrow_cells: break`:
    This relied on the old task's empirical side-touch regime, where rays never
    needed to cross their own source. In the variant this becomes wrong because
    frontal bounce naturally sends the ray back through the source vertex.
  - `if out[nr, nc] == bg: out[nr, nc] = cur_col`:
    Safe to keep because the generator uses at most one frontal wall per arrow.
    The return path revisits forward cells but does not need to recolor them for
    the foldback to be visible; visibility comes from the newly painted cells on
    the opposite side of the source.
  - Side-touch checks against `(nr+dr, nc)` and `(nr, nc+dc)`:
    These depend on walls being offset one cell from the diagonal. Under the
    variant that assumption is intentionally false, so those branches must be
    removed.

d) placement_legality:
  - Arrows, walls, and all cells that will be painted by rays are reserved
    during generation; later objects cannot overlap them.
  - A bounced arrow is only accepted if:
    1. there are at least 2 free diagonal cells before the wall,
    2. there are at least 2 free diagonal cells behind the source so the
       post-bounce continuation is visible,
    3. the wall cell and every painted ray cell are in-bounds and unoccupied.
  - A straight non-bouncing arrow is only accepted if it has a visible run to
    the boundary and does not intersect any reserved cells.
  - Because `main()` still blocks on foreign arrows, the generator forbids ray
    paths from crossing other arrows' cells.

e) visual_observability:
  - In input, the crucial clue is that the wall lies directly on the same
    diagonal that the arrow points along.
  - In output, a bounced example visibly grows on the opposite side of the
    source arrow after meeting that on-line wall. This is the main evidence that
    the obstacle causes a 180-degree turn rather than a side deflection.
  - At least one arrow per generated grid exhibits this behavior, so the
    variant cannot collapse to "all rays are straight".

f) visual_cleanliness:
  - Single-pixel walls keep the diagonal relation obvious.
  - Ray paths are non-overlapping across arrows, which avoids ambiguous color
    mixing and makes each arrow-wall interaction readable as an independent
    motif.
  - The grid size remains 18-20 in both dimensions, matching the original
    density regime.

g) color_change_inference:
  - The ray still inherits the wall's color at the bounce, exactly as in the
    original task.
  - To keep this inferable, wall colors are biased to differ from the launching
    arrow color when possible.
  - Because the bounced ray continues onto newly empty cells on the opposite
    side, the inherited color produces a distinct extension rather than being
    hidden inside already-painted cells.

new_solver_assumptions_and_generator_guarantees:
  - assumption: any on-path wall should be interpreted as a frontal blocker.
    guarantee: generator places walls only on planned diagonal hit cells.
  - assumption: only the current arrow's own cells may be traversed.
    guarantee: generator never puts a second arrow on a planned ray path.
  - assumption: a single frontal wall per arrow is enough to show the new rule.
    guarantee: generator enforces at least one such wall per grid and gives that
      arrow visible pre-wall and post-source segments.
"""


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


def _find_arrows(g: np.ndarray, bg: int):
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
                    "pts": pts,
                    "cells": cells,
                    "dir": (int(dr), int(dc)),
                    "start": (int(start[0]), int(start[1])),
                }
            )
            arrow_cells |= cells
    return arrows, arrow_cells


def _steps_to_edge(H: int, W: int, r: int, c: int, dr: int, dc: int) -> int:
    row_steps = (H - 1 - r) if dr == 1 else r
    col_steps = (W - 1 - c) if dc == 1 else c
    return int(min(row_steps, col_steps))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Each L-arrow fires diagonally away from its missing corner.
    - Walls sit directly on diagonal ray cells, so every wall interaction is a
      head-on collision.
    - On a hit the ray adopts the wall color and reverses both axes.
    - The ray may pass back through its own source arrow, but stops at other
      arrows or when it exits the grid.
    """
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

    for arrow in arrows:
        dr, dc = arrow["dir"]  # type: ignore[assignment]
        cur = arrow["start"]  # type: ignore[assignment]
        own_cells = arrow["cells"]  # type: ignore[assignment]
        cur_col = int(arrow["color"])
        seen: set[tuple[tuple[int, int], int, int, int]] = set()

        while True:
            cur = (int(cur[0]), int(cur[1]))
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
                if int(g[cur[0], cur[1]]) == int(bg):
                    out[cur[0], cur[1]] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                continue

            if (nr, nc) in arrow_cells and (nr, nc) not in own_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

    return out


def _arrow_points_from_start(sr: int, sc: int, dr: int, dc: int) -> list[tuple[int, int]]:
    mr, mc = sr - dr, sc - dc
    return [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]


def _ray_cells_no_bounce(H: int, W: int, sr: int, sc: int, dr: int, dc: int) -> list[tuple[int, int]]:
    steps = _steps_to_edge(H, W, sr, sc, dr, dc)
    return [(sr + i * dr, sc + i * dc) for i in range(1, steps + 1)]


def _ray_cells_one_bounce(
    H: int, W: int, sr: int, sc: int, dr: int, dc: int, wall_step: int
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    ahead = [(sr + i * dr, sc + i * dc) for i in range(1, wall_step)]
    back_steps = _steps_to_edge(H, W, sr, sc, -dr, -dc)
    behind = [(sr - i * dr, sc - i * dc) for i in range(1, back_steps + 1)]
    wall = (sr + wall_step * dr, sc + wall_step * dc)
    return ahead + behind, wall


def _pick_wall_color(arrow_color: int, palette: list[int]) -> int:
    choices = [c for c in palette if c != arrow_color]
    if choices:
        return int(_rng.choice(choices))
    return int(arrow_color)


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(1, 7)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]

    grid = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    has_bounced = False

    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        force_bounce = ai == n_arrows - 1 and not has_bounced
        placed = False

        for _ in range(160):
            dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
            r_lo, r_hi = (1 if dr == 1 else 0), (H - 1 if dr == 1 else H - 2)
            c_lo, c_hi = (1 if dc == 1 else 0), (W - 1 if dc == 1 else W - 2)
            if r_lo > r_hi or c_lo > c_hi:
                continue

            sr = _rng.randint(r_lo, r_hi)
            sc = _rng.randint(c_lo, c_hi)
            arrow_pts = _arrow_points_from_start(sr, sc, dr, dc)
            if any(not (0 <= r < H and 0 <= c < W) for r, c in arrow_pts):
                continue
            arrow_set = set(arrow_pts)
            if any((r, c) in occupied for r, c in arrow_pts):
                continue

            ahead_steps = _steps_to_edge(H, W, sr, sc, dr, dc)
            behind_steps = _steps_to_edge(H, W, sr, sc, -dr, -dc)
            want_bounce = force_bounce or (_rng.random() < 0.75)

            wall_cells: list[tuple[int, int]] = []
            if want_bounce:
                if ahead_steps < 3 or behind_steps < 2:
                    continue
                steps = list(range(3, ahead_steps + 1))
                _rng.shuffle(steps)
                chosen = None
                for wall_step in steps:
                    ray_cells, wall = _ray_cells_one_bounce(H, W, sr, sc, dr, dc, wall_step)
                    cell_set = set(ray_cells)
                    if len(cell_set) != len(ray_cells):
                        continue
                    reserved = arrow_set | cell_set | {wall}
                    if reserved & occupied:
                        continue
                    chosen = (ray_cells, wall)
                    break
                if chosen is None:
                    continue
                ray_cells, wall = chosen
                wall_cells = [wall]
            else:
                ray_cells = _ray_cells_no_bounce(H, W, sr, sc, dr, dc)
                if len(ray_cells) < 4:
                    continue
                reserved = arrow_set | set(ray_cells)
                if reserved & occupied:
                    continue

            reserved = arrow_set | set(ray_cells) | set(wall_cells)
            if len(reserved) != len(arrow_set) + len(set(ray_cells)) + len(set(wall_cells)):
                continue

            for r, c in arrow_pts:
                grid[r, c] = color
            for r, c in wall_cells:
                grid[r, c] = _pick_wall_color(color, palette)

            occupied |= reserved
            has_bounced = has_bounced or bool(wall_cells)
            placed = True
            break

        if not placed:
            return None

    if not has_bounced:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D13 head-on-wall variant."""
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate valid input after 300 attempts")
