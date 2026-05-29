# Combo variant of 142ca369 (index 11)
# combo_spec: D13/0+D15/0
# Name: 正面阻挡墙壁+功能分化

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

"""
Design analysis for combo D13/0 + D15/0
=======================================

a) concretization_and_trace
---------------------------
- Original task: the ray advances diagonally, then checks side-adjacent wall
  pixels. A wall flips only one axis and the path keeps zigzagging forward.
- This combo removes side-touch geometry entirely. Every meaningful wall now
  sits directly on the diagonal path, so the next diagonal step is always the
  collision test.
- Wall semantics are split by component size:
  1. 1-pixel wall = frontal reflector. The ray does not enter that cell; it
     adopts the wall color and reverses both axes.
  2. 3-pixel axis-aligned wall = frontal permeable wall. The center pixel lies
     on the diagonal; the ray enters that cell, adopts the wall color, and
     keeps the same direction through it.
- Concrete trace:
  Example arrow starts at (5,5), dir=(1,1).
  * If a 1-pixel wall is at (9,9), the ray paints (6,6),(7,7); on trying to
    step into (9,9) it recolors (8,8) to the wall color, turns to (-1,-1),
    crosses its own source vertex, and extends toward the opposite border.
  * If instead a 3-pixel horizontal wall is centered at (9,9), the ray paints
    (6,6),(7,7),(8,8), steps into wall cell (9,9), changes color there, and
    continues to (10,10),(11,11),... in the new color.
- The visible difference is strong: one wall makes the path fold back along the
  same diagonal, while the other lets it continue straight after recoloring.

b) complexity_calibration
-------------------------
- Only two wall categories are used, keyed by an obvious visual cue:
  single pixel vs 3-pixel bar. No hidden color table or orientation-specific
  function split is introduced.
- Every generated grid contains both categories, so the fused behavior cannot
  collapse into a pure D13 or pure D15 sample.
- The generator uses exactly two active arrows, one for each wall type. This is
  enough to show the contrast clearly without clutter from many overlapping ray
  histories.

c) hidden_assumptions_in_original_main
--------------------------------------
- Original branch `if (nr, nc) in wall_cells` already encoded head-on reversal,
  but it assumed every wall behaves identically. The combo must first classify
  wall components before deciding reflect vs pass-through.
- Original `if (nr, nc) in arrow_cells: break` becomes too strict because a
  frontal reflector naturally sends the ray back through its own source. The
  combo therefore allows traversal through the current arrow's own cells while
  still blocking on foreign arrows.
- Original side-touch checks against `(nr+dr, nc)` and `(nr, nc+dc)` are no
  longer valid: they depend on walls living one cell off the diagonal. Under
  this combo that placement is forbidden, so those branches are removed.
- Classification now assumes non-arrow components are either:
  * size 1 -> reflect wall
  * size 3 axis-aligned line -> pass wall
  The generator guarantees only these two wall shapes.

d) placement_legality
---------------------
- Input objects do not overlap except for the intended semantic fact that the
  center cell of a 3-pixel pass wall lies on the ray's future diagonal path.
- The reflector case reserves all painted cells before and after the bounce, the
  wall cell itself, and the arrow cells, so later objects cannot collide.
- The permeable wall reserves the 3-pixel bar plus the pre-wall and post-wall
  diagonal runs. The ray is allowed to go through the wall center only because
  that is the defined behavior of the combo.
- Global rotate/flip augmentation is applied after a valid base layout is built,
  preserving non-overlap and keeping the task visually varied.

e) visual_observability
-----------------------
- In input, the human can directly see that one obstacle is a single pixel on
  the arrow's diagonal while the other is a 3-pixel bar whose center sits on
  the diagonal.
- In output, the single pixel causes a clear foldback onto the opposite side of
  the source arrow; the 3-pixel wall causes a straight continuation beyond the
  wall in the wall's color.
- Because both behaviors appear in the same sample, the size->function mapping
  is inferable from only a few examples.

f) visual_cleanliness
---------------------
- The generator keeps the two arrows on separated diagonal bands before applying
  symmetry, so ray histories and walls remain easy to parse.
- Colors are assigned distinctly across the two arrows and two walls, avoiding
  accidental 4-connectivity mergers that would make object parsing ambiguous.
- Grid sizes remain within the original 18-20 regime.

g) color-change inference
-------------------------
- Color transfer remains structurally simple: the ray always adopts the color of
  the wall it interacts with.
- Reflect and pass walls both recolor the ray, so the only new question for the
  solver is whether direction changes. This keeps color reasoning orthogonal to
  geometry reasoning.

h) quantitative anti-degeneracy target
--------------------------------------
- Each generated grid must contain:
  * exactly one 1-pixel frontal reflector hit with visible post-bounce cells;
  * exactly one 3-pixel frontal permeable hit with visible post-wall cells.
- These grids should disagree with a degenerate "all on-path walls reflect"
  solver because the 3-pixel wall continues forward instead of folding back.

i) new solver assumptions and generator guarantees
--------------------------------------------------
- assumption: the current arrow's own cells may be traversed after a frontal
  reflection. guarantee: the generator uses this only for the reflector arrow.
- assumption: 3-pixel pass walls are axis-aligned and centered on the hit cell.
  guarantee: the generator places only horizontal or vertical 3-bars.
- assumption: arrows, walls, and foreign-ray paths do not create ambiguous
  same-color 4-connected components. guarantee: all active objects use distinct
  colors and the layout is collision-checked before acceptance.
"""

Coord = tuple[int, int]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_arrow_pts(pts: list[Coord]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[Coord]) -> tuple[tuple[int, int], Coord] | None:
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
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in pts_set:
                deg[(r, c)] += 1
    start = max(deg, key=lambda p: deg[p])
    return (int(dr), int(dc)), (int(start[0]), int(start[1]))


def _classify_wall_component(pts: list[Coord]) -> str:
    if len(pts) == 1:
        return "reflect"
    if len(pts) == 3:
        rs = {r for r, _c in pts}
        cs = {c for _r, c in pts}
        if len(rs) == 1 or len(cs) == 1:
            return "pass"
    return "reflect"


def _detect_scene(g: np.ndarray, bg: int):
    arrow_cells: set[Coord] = set()
    arrow_sigs: set[tuple[Coord, ...]] = set()
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
            sig = tuple(sorted(pts))
            cells = set(pts)
            arrows.append(
                {
                    "color": int(col),
                    "cells": cells,
                    "dir": (int(dr), int(dc)),
                    "start": (int(start[0]), int(start[1])),
                }
            )
            arrow_sigs.add(sig)
            arrow_cells |= cells

    wall_kind: dict[Coord, str] = {}
    wall_color: dict[Coord, int] = {}
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if tuple(sorted(pts)) in arrow_sigs:
                continue
            kind = _classify_wall_component(pts)
            for p in pts:
                wall_kind[p] = kind
                wall_color[p] = int(col)

    return arrows, arrow_cells, wall_kind, wall_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Walls are placed directly on the ray's diagonal path.
    - 1-pixel walls are frontal reflectors: adopt wall color and reverse both
      axes without entering the wall cell.
    - 3-pixel axis-aligned walls are frontal pass walls: step into the wall
      center, adopt wall color, and continue in the same direction.
    - Rays may cross their own source arrow after a frontal reflection, but
      still stop on foreign arrows or when leaving the grid.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrows, arrow_cells, wall_kind, wall_color = _detect_scene(g, bg)
    wall_cells = set(wall_kind)
    out = g.copy()

    for arrow in arrows:
        dr, dc = arrow["dir"]  # type: ignore[assignment]
        cur = arrow["start"]  # type: ignore[assignment]
        own_cells = arrow["cells"]  # type: ignore[assignment]
        cur_col = int(arrow["color"])
        seen: set[tuple[Coord, int, int, int]] = set()

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

            if (nr, nc) in arrow_cells and (nr, nc) not in own_cells:
                break

            if (nr, nc) in wall_cells:
                wcol = int(wall_color[(nr, nc)])
                if wall_kind[(nr, nc)] == "pass":
                    cur = (int(nr), int(nc))
                    cur_col = int(wcol)
                    continue
                if int(g[cur[0], cur[1]]) == int(bg):
                    out[cur[0], cur[1]] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                continue

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

    return out


def _steps_to_edge(H: int, W: int, r: int, c: int, dr: int, dc: int) -> int:
    row_steps = (H - 1 - r) if dr == 1 else r
    col_steps = (W - 1 - c) if dc == 1 else c
    return int(min(row_steps, col_steps))


def _make_arrow(start: Coord, dr: int, dc: int) -> list[Coord]:
    sr, sc = start
    mr, mc = sr - dr, sc - dc
    return [(int(r), int(c)) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]


def _apply_symmetry(arr: np.ndarray, rot_k: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(arr, int(rot_k))
    if flip_lr:
        out = np.fliplr(out)
    return np.array(out, dtype=int)


def _reflect_paths(H: int, W: int, start: Coord, dr: int, dc: int, wall_step: int):
    sr, sc = start
    before = [(int(sr + i * dr), int(sc + i * dc)) for i in range(1, wall_step)]
    wall = (int(sr + wall_step * dr), int(sc + wall_step * dc))
    back_steps = _steps_to_edge(H, W, sr, sc, -dr, -dc)
    behind = [(int(sr - i * dr), int(sc - i * dc)) for i in range(1, back_steps + 1)]
    return before, wall, behind


def _pass_paths(H: int, W: int, start: Coord, dr: int, dc: int, wall_step: int):
    sr, sc = start
    before = [(int(sr + i * dr), int(sc + i * dc)) for i in range(1, wall_step)]
    center = (int(sr + wall_step * dr), int(sc + wall_step * dc))
    ahead_steps = _steps_to_edge(H, W, center[0], center[1], dr, dc)
    after = [(int(center[0] + i * dr), int(center[1] + i * dc)) for i in range(1, ahead_steps + 1)]
    return before, center, after


def _pass_wall(center: Coord, orient: str) -> list[Coord]:
    r, c = center
    if orient == "h":
        return [(int(r), int(c - 1)), (int(r), int(c)), (int(r), int(c + 1))]
    return [(int(r - 1), int(c)), (int(r), int(c)), (int(r + 1), int(c))]


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow1_col, arrow2_col, wall1_col, wall2_col = [int(x) for x in palette[:4]]

    start1 = (_rng.randint(4, 6), _rng.randint(4, 6))
    start2 = (_rng.randint(H - 5, H - 3), _rng.randint(2, 4))
    wall_step1 = _rng.randint(3, 6)
    wall_step2 = _rng.randint(3, 5)
    pass_orient = _rng.choice(["h", "v"])

    arrow1 = _make_arrow(start1, 1, 1)
    arrow2 = _make_arrow(start2, -1, 1)
    before1, wall1, behind1 = _reflect_paths(H, W, start1, 1, 1, wall_step1)
    before2, center2, after2 = _pass_paths(H, W, start2, -1, 1, wall_step2)
    wall2 = _pass_wall(center2, pass_orient)

    if len(before1) < 2 or len(behind1) < 2 or len(before2) < 2 or len(after2) < 2:
        return None

    input_cells = set(arrow1) | set(arrow2) | {wall1} | set(wall2)
    if len(input_cells) != len(arrow1) + len(arrow2) + 1 + len(set(wall2)):
        return None

    reflect_ray = set(before1) | set(behind1)
    pass_ray = set(before2) | set(after2)
    if reflect_ray & input_cells:
        return None
    if pass_ray & input_cells:
        return None
    if reflect_ray & pass_ray:
        return None

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)

    for r, c in arrow1:
        grid[r, c] = out[r, c] = int(arrow1_col)
    for r, c in arrow2:
        grid[r, c] = out[r, c] = int(arrow2_col)
    grid[wall1] = out[wall1] = int(wall1_col)
    for r, c in wall2:
        grid[r, c] = out[r, c] = int(wall2_col)

    for r, c in before1[:-1]:
        out[r, c] = int(arrow1_col)
    out[before1[-1]] = int(wall1_col)
    for r, c in behind1:
        out[r, c] = int(wall1_col)

    for r, c in before2:
        out[r, c] = int(arrow2_col)
    for r, c in after2:
        out[r, c] = int(wall2_col)

    rot_k = _rng.randint(0, 3)
    flip_lr = bool(_rng.randint(0, 1))
    grid_t = _apply_symmetry(grid, rot_k, flip_lr)
    out_t = _apply_symmetry(out, rot_k, flip_lr)

    if np.array_equal(grid_t, out_t):
        return None
    if not np.array_equal(main(grid_t), out_t):
        return None
    return grid_t


def generate_input() -> np.ndarray:
    """Generate a random valid input for combo D13/0 + D15/0."""
    for _ in range(200):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 200 attempts")
