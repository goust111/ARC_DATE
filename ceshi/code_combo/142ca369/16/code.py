# Combo variant of 142ca369 (index 16)
# combo_spec: D7/0+D13/0+D15/0
# Name: L/T形墙壁+正面阻挡+功能分化

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
------------------------------
- Arrows stay the same as the source task: a 3-cell L inside a 2x2 box. The
  missing corner still determines the diagonal firing direction, and the ray
  starts at the right-angle vertex.
- Walls are now 4-cell polyomino components only: either an L tetromino or a T
  tetromino. Unlike the source task, each wall sits directly on the ray's
  diagonal path, so the next diagonal step may land on a wall component.
- The two wall families have different functions:
  1. L wall = reflective blocker. The ray does not enter the hit cell. Instead,
     the current ray cell adopts the wall color, and exactly one axis flips.
     Which axis flips depends on the L orientation at the hit cell:
       if the L extends forward along the ray's row direction, flip dc;
       if the L extends forward along the ray's column direction, flip dr.
  2. T wall = permeable color changer. The ray moves into the wall cell, adopts
     the wall color, and keeps the same direction. The generator keeps the hit
     cell as the only diagonal cell of that T component, so the ray visibly
     passes through the structure rather than bouncing repeatedly inside it.
- Small trace for dir=(+1,+1):
  source task:
    side wall at (r,c+1) only changes one axis after the ray first advances to
    (r,c); geometry is a lateral zigzag.
  combo task, L wall:
    wall hit cell is the next diagonal cell h=(r+1,c+1). If that L continues to
    h+(0,+1), it flips dr and leaves from (r,c) toward (-1,+1). If it instead
    continues to h+(+1,0), it flips dc and leaves toward (+1,-1).
  combo task, T wall:
    wall hit cell is again h=(r+1,c+1), but now the ray enters h, turns to the
    T color, and continues to (r+2,c+2).
- The visible difference from the source is therefore twofold: walls lie on the
  ray diagonal itself, and wall shape now determines whether the ray reflects or
  passes through.

b) Complexity calibration
-------------------------
- Only two wall vocabularies are allowed: L and T. This is enough to make shape
  recognition necessary without creating a large case table.
- L walls only have two semantic orientations from the solver's viewpoint:
  "forward-row arm present" or "forward-col arm present". That keeps the new
  reflection rule human-inferable.
- T walls all share one function, so the human only needs to infer one shape ->
  one behavior mapping for pass-through.
- Every generated grid contains at least one L event and at least one T event.
  Without that guarantee, the fusion could degenerate into a pure D13 or pure
  D15 subset that would not expose the combined rule.

c) Implicit assumptions broken from the original main()
-------------------------------------------------------
- Original assumption: every non-arrow pixel can be treated as an undifferentiated
  wall cell set.
  Broken: behavior now depends on 4-connected component shape, so the solver must
  classify wall components before tracing rays.
- Original assumption: a diagonal wall hit always means 180-degree reversal.
  Broken: L hits now produce single-axis reflection, while T hits do not reflect
  at all.
- Original assumption: side-touch checks against (nr+dr,nc) and (nr,nc+dc) are
  the main bounce logic.
  Broken: in this combo, the critical interaction is the next diagonal cell
  itself, not an offset side cell. Side-touch logic is removed entirely.
- Original assumption: a ray never needs to step onto a wall cell.
  Broken: T walls are intentionally permeable, so the solver must allow
  `cur = (nr,nc)` when the hit component is T.

d) Placement legality
---------------------
- Arrows and wall components never overlap in the input.
- For L events, only the wall hit cell lies on the ray diagonal; the reflected
- path is planned from the pre-hit ray cell, so the ray never enters the wall.
- For T events, exactly one wall cell lies on the ray diagonal, and the next
  diagonal step is guaranteed to leave the T component.
- The generator keeps the two arrows far apart and rejects any accidental overlap
  between planned ray cells, wall cells, and other arrows.
- Each object gets its own color, preventing same-color 4-neighbor mergers from
  confusing arrow/wall component detection.

e) Visual observability
-----------------------
- In the input, the decisive clue is that the walls are not stray side pixels;
  they are compact L/T polyominoes sitting directly in front of the arrow path.
- In the output, the L wall causes a turn exactly at the cell before the wall,
  while the T wall lets the ray continue straight but in the wall color.
- With one L example and one T example in every grid, a human can contrast
  "same on-path placement, different shape, different behavior".

f) Visual cleanliness
---------------------
- The generator uses a sparse 20x20 layout with two arrows placed in separate
  regions and only one planned wall event per arrow.
- Wall colors differ from arrow colors, and objects are separated enough that the
  L and T silhouettes remain easy to inspect.

g) Color-change inferability
----------------------------
- Color inheritance is still local: the ray adopts the color of the wall
  component it just interacted with.
- Because each grid includes both a reflecting L wall and a permeable T wall with
  distinct colors, the human can see that color change is shared across both wall
  types while the geometric effect differs by shape.

h) Quantitative fusion check target
-----------------------------------
- The generator is designed so that every sample has:
  1. at least one L wall component,
  2. at least one T wall component,
  3. at least one single-axis reflection event,
  4. at least one straight pass-through color-change event.
- Comparing the combo solver with the original source-task solver on the same
  generated inputs should usually produce different outputs because the original
  solver would treat these on-path wall hits as generic reversals.

i) New solver assumptions and generator guarantees
--------------------------------------------------
- Assumption: non-arrow components are only 4-cell L or T tetrominoes.
  Guarantee: the generator emits only those shapes.
- Assumption: an L hit cell has exactly one forward-adjacent arm among
  `(h+dr,hc)` and `(hr,hc+dc)`, which determines the reflected axis.
  Guarantee: the generator only uses L templates that satisfy that condition.
- Assumption: T walls are entered through their center hit cell, and the next
  diagonal step leaves the component.
  Guarantee: the generator places T templates with the hit cell at the degree-3
  center, so only that center lies on the ray diagonal.
"""

Coord = tuple[int, int]


def _normalize_shape(cells: set[Coord]) -> tuple[Coord, ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _transform_shape(cells: set[Coord], rot: int, reflect: bool) -> set[Coord]:
    out = set(cells)
    if reflect:
        out = {(r, -c) for r, c in out}
    for _ in range(rot % 4):
        out = {(c, -r) for r, c in out}
    return set(_normalize_shape(out))


def _all_orientations(base: set[Coord], *, allow_reflect: bool) -> set[tuple[Coord, ...]]:
    out: set[tuple[Coord, ...]] = set()
    for rot in range(4):
        out.add(_normalize_shape(_transform_shape(base, rot, False)))
        if allow_reflect:
            out.add(_normalize_shape(_transform_shape(base, rot, True)))
    return out


_BASE_L4 = {(0, 0), (0, 1), (0, 2), (1, 2)}
_BASE_T4 = {(0, 1), (1, 0), (1, 1), (1, 2)}
_L_SHAPES = _all_orientations(_BASE_L4, allow_reflect=True)
_T_SHAPES = _all_orientations(_BASE_T4, allow_reflect=False)


def _neighbors4(r: int, c: int) -> tuple[Coord, ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_arrow_pts(pts: list[Coord]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[Coord]) -> tuple[tuple[int, int], Coord] | None:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    pts_set = set(pts)
    missing = [p for p in corners if p not in pts_set]
    if len(missing) != 1:
        return None

    mr, mc = missing[0]
    dr = 1 if mr == r0 else -1
    dc = 1 if mc == c0 else -1

    deg = {p: 0 for p in pts}
    for r, c in pts:
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in pts_set:
                deg[(r, c)] += 1
    start = max(deg, key=lambda p: deg[p])
    return (int(dr), int(dc)), (int(start[0]), int(start[1]))


def _classify_wall_component(pts: list[Coord]) -> str | None:
    if len(pts) != 4:
        return None
    norm = _normalize_shape(set(pts))
    if norm in _L_SHAPES:
        return "L"
    if norm in _T_SHAPES:
        return "T"
    return None


def _detect_scene(g: np.ndarray, bg: int):
    arrow_cells: set[Coord] = set()
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

    wall_id_by_cell: dict[Coord, int] = {}
    wall_kind: dict[int, str] = {}
    wall_color: dict[int, int] = {}
    wall_cells_by_id: dict[int, set[Coord]] = {}
    wid = 0
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            cells = set(pts)
            if cells & arrow_cells:
                continue
            kind = _classify_wall_component(pts)
            if kind is None:
                continue
            wall_kind[wid] = kind
            wall_color[wid] = int(col)
            wall_cells_by_id[wid] = cells
            for cell in cells:
                wall_id_by_cell[cell] = wid
            wid += 1

    return arrows, arrow_cells, wall_id_by_cell, wall_kind, wall_color, wall_cells_by_id


def _l_reflection_axis(wall_cells: set[Coord], hit: Coord, dr: int, dc: int) -> str:
    hr, hc = hit
    row_forward = (int(hr + dr), int(hc))
    col_forward = (int(hr), int(hc + dc))
    row_back = (int(hr - dr), int(hc))
    col_back = (int(hr), int(hc - dc))

    if row_forward in wall_cells and col_forward not in wall_cells:
        return "dc"
    if col_forward in wall_cells and row_forward not in wall_cells:
        return "dr"
    if row_back in wall_cells and col_back not in wall_cells:
        return "dc"
    if col_back in wall_cells and row_back not in wall_cells:
        return "dr"
    return "both"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - L/T polyomino walls lie directly on the ray diagonal.
    - Hitting an L wall blocks the next step, adopts the wall color, and reflects
      by flipping exactly one axis based on the L orientation.
    - Hitting a T wall lets the ray step into the wall cell, adopt the wall
      color, and continue straight through.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrows, arrow_cells, wall_id_by_cell, wall_kind, wall_color, wall_cells_by_id = _detect_scene(g, bg)
    out = g.copy()

    for arrow in arrows:
        dr, dc = arrow["dir"]  # type: ignore[assignment]
        cur = arrow["start"]  # type: ignore[assignment]
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

            if (nr, nc) in arrow_cells:
                break

            if (nr, nc) in wall_id_by_cell:
                wid = wall_id_by_cell[(nr, nc)]
                wcol = int(wall_color[wid])
                if wall_kind[wid] == "T":
                    cur = (int(nr), int(nc))
                    cur_col = int(wcol)
                    continue

                if int(g[cur[0], cur[1]]) == int(bg):
                    out[cur[0], cur[1]] = int(wcol)
                cur_col = int(wcol)
                axis = _l_reflection_axis(wall_cells_by_id[wid], (int(nr), int(nc)), int(dr), int(dc))
                if axis in {"dr", "both"}:
                    dr = -int(dr)
                if axis in {"dc", "both"}:
                    dc = -int(dc)
                continue

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

    return out


def _make_arrow(start: Coord, dr: int, dc: int) -> list[Coord]:
    sr, sc = start
    mr, mc = sr - dr, sc - dc
    return [(int(r), int(c)) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]


def _to_edge(start: Coord, dr: int, dc: int, H: int, W: int) -> list[Coord]:
    r, c = start
    out: list[Coord] = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < H and 0 <= c < W):
            return out
        out.append((int(r), int(c)))


def _apply_symmetry(arr: np.ndarray, rot_k: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(arr, int(rot_k))
    if flip_lr:
        out = np.fliplr(out)
    return np.array(out, dtype=int)


def _l_wall_cells(hit: Coord, dr: int, dc: int, flip_axis: str) -> list[Coord]:
    hr, hc = hit
    if flip_axis == "dr":
        rel = [(0, 0), (0, dc), (0, 2 * dc), (dr, 2 * dc)]
    else:
        rel = [(0, 0), (dr, 0), (2 * dr, 0), (2 * dr, dc)]
    return [(int(hr + rr), int(hc + cc)) for rr, cc in rel]


def _t_wall_cells(hit: Coord, dr: int, dc: int, missing_arm: str) -> list[Coord]:
    hr, hc = hit
    arms = {
        "frow": [(-dr, 0), (0, -dc), (0, dc)],
        "brow": [(dr, 0), (0, -dc), (0, dc)],
        "fcol": [(-dr, 0), (dr, 0), (0, -dc)],
        "bcol": [(-dr, 0), (dr, 0), (0, dc)],
    }[missing_arm]
    return [(int(hr), int(hc))] + [(int(hr + rr), int(hc + cc)) for rr, cc in arms]


def _build_reflect_event(
    start: Coord,
    dr: int,
    dc: int,
    pre_steps: int,
    flip_axis: str,
    arrow_col: int,
    wall_col: int,
    H: int,
    W: int,
):
    arrow_pts = _make_arrow(start, dr, dc)
    pre_cells = [(int(start[0] + i * dr), int(start[1] + i * dc)) for i in range(1, pre_steps + 1)]
    pre_hit = pre_cells[-1]
    hit = (int(pre_hit[0] + dr), int(pre_hit[1] + dc))
    wall_cells = _l_wall_cells(hit, dr, dc, flip_axis)
    next_dr, next_dc = (-dr, dc) if flip_axis == "dr" else (dr, -dc)
    post_cells = _to_edge(pre_hit, next_dr, next_dc, H, W)

    output_cells: list[tuple[int, int, int]] = []
    for r, c in pre_cells[:-1]:
        output_cells.append((int(r), int(c), int(arrow_col)))
    output_cells.append((int(pre_hit[0]), int(pre_hit[1]), int(wall_col)))
    for r, c in post_cells:
        output_cells.append((int(r), int(c), int(wall_col)))

    return {
        "arrow": arrow_pts,
        "wall": wall_cells,
        "output": output_cells,
        "painted": pre_cells + post_cells,
        "input_reserved": set(arrow_pts) | set(wall_cells),
    }


def _build_pass_event(
    start: Coord,
    dr: int,
    dc: int,
    wall_step: int,
    missing_arm: str,
    arrow_col: int,
    wall_col: int,
    H: int,
    W: int,
):
    arrow_pts = _make_arrow(start, dr, dc)
    hit = (int(start[0] + wall_step * dr), int(start[1] + wall_step * dc))
    before_cells = [(int(start[0] + i * dr), int(start[1] + i * dc)) for i in range(1, wall_step)]
    wall_cells = _t_wall_cells(hit, dr, dc, missing_arm)
    after_cells = _to_edge(hit, dr, dc, H, W)

    output_cells: list[tuple[int, int, int]] = []
    for r, c in before_cells:
        output_cells.append((int(r), int(c), int(arrow_col)))
    for r, c in after_cells:
        output_cells.append((int(r), int(c), int(wall_col)))

    return {
        "arrow": arrow_pts,
        "wall": wall_cells,
        "output": output_cells,
        "painted": before_cells + [hit] + after_cells,
        "input_reserved": set(arrow_pts) | set(wall_cells),
    }


def _gen_attempt() -> np.ndarray | None:
    H = W = 20
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow1_col, wall1_col, arrow2_col, wall2_col = [int(x) for x in palette[:4]]

    reflect_flip = _rng.choice(["dr", "dc"])
    reflect_steps = _rng.randint(3, 5)
    pass_steps = _rng.randint(3, 5)
    t_missing = _rng.choice(["frow", "brow", "fcol", "bcol"])

    reflect = _build_reflect_event(
        start=(4, 4),
        dr=1,
        dc=1,
        pre_steps=reflect_steps,
        flip_axis=reflect_flip,
        arrow_col=arrow1_col,
        wall_col=wall1_col,
        H=H,
        W=W,
    )
    permeable = _build_pass_event(
        start=(16, 3),
        dr=-1,
        dc=1,
        wall_step=pass_steps,
        missing_arm=t_missing,
        arrow_col=arrow2_col,
        wall_col=wall2_col,
        H=H,
        W=W,
    )

    all_input = reflect["input_reserved"] | permeable["input_reserved"]
    if len(all_input) != len(reflect["input_reserved"]) + len(permeable["input_reserved"]):
        return None

    reflect_painted = set(reflect["painted"])
    pass_painted = set(permeable["painted"])
    if reflect_painted & permeable["input_reserved"]:
        return None
    if (pass_painted - set(permeable["wall"])) & reflect["input_reserved"]:
        return None
    if reflect_painted & (pass_painted - {tuple(permeable["wall"][0])}):
        return None

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)

    for r, c in reflect["arrow"]:
        grid[r, c] = out[r, c] = int(arrow1_col)
    for r, c in reflect["wall"]:
        grid[r, c] = out[r, c] = int(wall1_col)
    for r, c in permeable["arrow"]:
        grid[r, c] = out[r, c] = int(arrow2_col)
    for r, c in permeable["wall"]:
        grid[r, c] = out[r, c] = int(wall2_col)

    for r, c, col in reflect["output"]:
        if grid[r, c] == 0:
            out[r, c] = int(col)
        else:
            out[r, c] = int(grid[r, c])
    for r, c, col in permeable["output"]:
        if grid[r, c] == 0:
            out[r, c] = int(col)
        else:
            out[r, c] = int(grid[r, c])

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
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
