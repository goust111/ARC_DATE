# Combo variant of 4a21e3da (index 34)
# combo_spec: D4/1+D7/0+D8/0
# Name: 碰撞停止+迭代处理+三控制像素

from __future__ import annotations

"""
Design analysis for the fused variant
=====================================

a) Concreteization and behavior trace
  Source task:
    one vertical split and/or one horizontal split partitions the centered sprite;
    every fg pixel on a split line stays in place; off-line pieces slide to corners.

  Fused task:
    always use 3 control pixels in a 2+1 layout:
      - either 2 vertical controls (top+bottom) plus 1 horizontal control (left or right)
      - or 2 horizontal controls (left+right) plus 1 vertical control (top or bottom)
    The three control lines still define a full 2x3 or 3x2 partition of the sprite, so a human
    sees 6 conceptual regions. The duplicated-axis middle stripe/row vanishes, and the four outer
    corner regions slide to the four grid corners.
    The D4/1 change is fused into ray drawing itself: a ray does not extend to the farthest aligned
    fg witness if a perpendicular control line is encountered first. This matters visually because
    the generator deliberately leaves split-line fg beyond the collision point, so the output shows
    "the cut line exists but the painted ray stopped early".

  Small vertical example:
    controls = top@c1, bottom@c2, left@r with c1 < c2
    original 3-line-without-collision paint:
      - top ray on c1 paints down to the lowest fg on c1
      - bottom ray on c2 paints up to the highest fg on c2
      - left ray on r paints right to the farthest fg on r
    fused paint:
      - top ray on c1 stops at row r
      - bottom ray on c2 stops at row r
      - left ray on r stops at column c1
      - fg on rows/cols r,c1,c2 still stays in place everywhere on those split lines
    partition:
      top/bottom x left/middle/right gives 6 regions
      left/right columns survive to the four corners; middle column vanishes
    key visible difference:
      the collision stop does not change which conceptual region a piece belongs to, but it changes
      how much ctrl-colored paint appears before the still-present split-line fg.

b) Complexity calibration
  - The rule is visibly different from the seed task in every accepted sample because all samples use
    3 controls and at least one ray is collision-shortened.
  - The layout only has two families (2 vertical + 1 horizontal, or the transpose), so the human does
    not need to infer many unrelated cases.
  - The duplicated-axis middle band is always the disappearing part. That keeps the 6-region logic
    compact and repeatedly supported across examples.

c) Implicit-assumption audit of the original main()
  - Original assumption: one scalar `c_split` and one scalar `r_split` are enough.
    Broken by 3 controls. The new solver stores either two split columns + one split row, or the
    transpose.
  - Original assumption: binary regions (`L/R`, `T/B`) are sufficient.
    Broken by 3 controls. The new solver uses `L/M/R` or `T/M/B`.
  - Original assumption: ray stop depends only on farthest aligned fg.
    Broken by D4/1. The new solver compares the farthest fg stop against perpendicular control lines
    and truncates to the nearest collision.
  - Original assumption: the ray-painted segment and the cut classification are identical.
    No longer true here by design: full split lines still define region membership, but painted rays
    can be shorter. The generator exploits this so the new stop rule is observable rather than hidden.

d) Placement legality guaranteed by the generator
  - Sprite bbox stays strictly inside the grid with margin >= 4 on all sides.
  - Every split line passes through the bbox interior, and both duplicated-axis outer bands plus the
    middle band/row have positive size.
  - Corner pieces are generated inside disjoint source boxes, never on split lines.
  - Split-line fg is added separately and used as the connectivity bridge, so the whole input sprite
    remains one 4-connected component.

e) Visual observability
  - The output always shows 3 border controls and 3 rays.
  - At least one ray is visibly shorter than the farthest aligned split-line fg, so collision stop is
    inferable from examples instead of being a silent internal detail.
  - The four outer regions move to the four corners while one or two middle regions disappear.

f) Visual cleanliness
  - Only the original 3 colors are used.
  - The centered sprite and the split lines are sparse and axis-aligned, so the control color, frozen
    split-line fg, and moved corner pieces remain easy to distinguish.

g) New solver assumptions and matching generator guarantees
  - `main()` assumes exactly 3 border control pixels in a 2+1 opposite-edge family.
    `generate_input()` constructs only those layouts.
  - `main()` assumes each moving connected component lies fully inside one conceptual region label.
    The generator places each piece inside a single source box and adds split-line fg separately.
  - `main()` assumes collision shortening is visible on the split lines.
    The generator forces aligned fg beyond each collision point, and rejects samples whose collision
    output matches the no-collision baseline.
"""

import random
from collections import deque

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    for _ in range(int(target_px) - 1):
        neighbors = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if (nr, nc) not in cells:
                    neighbors.add((nr, nc))
        valid = []
        for nr, nc in neighbors:
            test = cells | {(nr, nc)}
            hs = [r for r, _ in test]
            ws = [c for _, c in test]
            if max(hs) - min(hs) + 1 <= int(max_h) and max(ws) - min(ws) + 1 <= int(max_w):
                valid.append((nr, nc))
        if not valid:
            break
        cells.add(random.choice(valid))
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return frozenset((int(r - min(rs)), int(c - min(cs))) for r, c in cells)


def _edge_of_ctrl(H: int, W: int, r: int, c: int) -> str:
    if int(r) == 0:
        return "top"
    if int(r) == int(H - 1):
        return "bottom"
    if int(c) == 0:
        return "left"
    if int(c) == int(W - 1):
        return "right"
    raise AssertionError("control pixel must lie on the border")


def _detect_roles(g: np.ndarray) -> tuple[int, int, int, list[tuple[int, int]]]:
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})
    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(int(bg))
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != int(ctrl)))
    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(ctrl))]
    return int(bg), int(fg), int(ctrl), ctrl_pts


def _ctrl_info(H: int, W: int, r: int, c: int) -> dict[str, int | str]:
    edge = _edge_of_ctrl(H, W, r, c)
    if edge in ("top", "bottom"):
        return {"edge": edge, "axis": "v", "line": int(c), "r": int(r), "c": int(c)}
    return {"edge": edge, "axis": "h", "line": int(r), "r": int(r), "c": int(c)}


def _parse_layout(ctrl_pts: list[tuple[int, int]], H: int, W: int) -> dict[str, object]:
    infos = [_ctrl_info(H, W, r, c) for r, c in ctrl_pts]
    vertical = [info for info in infos if str(info["axis"]) == "v"]
    horizontal = [info for info in infos if str(info["axis"]) == "h"]
    assert len(infos) == 3

    if len(vertical) == 2 and len(horizontal) == 1:
        v_edges = {str(info["edge"]) for info in vertical}
        assert v_edges == {"top", "bottom"}
        h_edge = str(horizontal[0]["edge"])
        assert h_edge in {"left", "right"}
        return {
            "mode": "vertical_dup",
            "ctrl_infos": infos,
            "ctrl_edges": {"top", "bottom", h_edge},
            "c_splits": sorted(int(info["line"]) for info in vertical),
            "r_split": int(horizontal[0]["line"]),
            "singleton_edge": h_edge,
        }

    if len(horizontal) == 2 and len(vertical) == 1:
        h_edges = {str(info["edge"]) for info in horizontal}
        assert h_edges == {"left", "right"}
        v_edge = str(vertical[0]["edge"])
        assert v_edge in {"top", "bottom"}
        return {
            "mode": "horizontal_dup",
            "ctrl_infos": infos,
            "ctrl_edges": {"left", "right", v_edge},
            "r_splits": sorted(int(info["line"]) for info in horizontal),
            "c_split": int(vertical[0]["line"]),
            "singleton_edge": v_edge,
        }

    raise AssertionError("expected a 2+1 three-control layout")


def _split_cells(shape: tuple[int, int], layout: dict[str, object]) -> set[tuple[int, int]]:
    H, W = shape
    cells: set[tuple[int, int]] = set()
    if str(layout["mode"]) == "vertical_dup":
        c0, c1 = [int(x) for x in layout["c_splits"]]
        r = int(layout["r_split"])
        for rr in range(H):
            cells.add((int(rr), int(c0)))
            cells.add((int(rr), int(c1)))
        for cc in range(W):
            cells.add((int(r), int(cc)))
    else:
        r0, r1 = [int(x) for x in layout["r_splits"]]
        c = int(layout["c_split"])
        for cc in range(W):
            cells.add((int(r0), int(cc)))
            cells.add((int(r1), int(cc)))
        for rr in range(H):
            cells.add((int(rr), int(c)))
    return cells


def _region_label(r: int, c: int, layout: dict[str, object]) -> tuple[str, str]:
    if str(layout["mode"]) == "vertical_dup":
        c0, c1 = [int(x) for x in layout["c_splits"]]
        rs = int(layout["r_split"])
        hs = "T" if int(r) < rs else "B"
        if int(c) < c0:
            vs = "L"
        elif int(c) < c1:
            vs = "M"
        else:
            vs = "R"
        return hs, vs

    r0, r1 = [int(x) for x in layout["r_splits"]]
    cs = int(layout["c_split"])
    if int(r) < r0:
        hs = "T"
    elif int(r) < r1:
        hs = "M"
    else:
        hs = "B"
    vs = "L" if int(c) < cs else "R"
    return hs, vs


def _keep_and_dest(label: tuple[str, str], layout: dict[str, object]) -> tuple[bool, tuple[str, str] | None]:
    hs, vs = label
    if str(layout["mode"]) == "vertical_dup":
        if vs == "M":
            return False, None
        return True, (str(hs), "L" if vs == "L" else "R")

    if hs == "M":
        return False, None
    return True, ("T" if hs == "T" else "B", str(vs))


def _components(points: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    pts = set((int(r), int(c)) for r, c in points)
    comps: list[list[tuple[int, int]]] = []
    while pts:
        seed = next(iter(pts))
        q = deque([seed])
        pts.remove(seed)
        comp = [seed]
        while q:
            r, c = q.popleft()
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if (nr, nc) in pts:
                    pts.remove((nr, nc))
                    q.append((nr, nc))
                    comp.append((nr, nc))
        comps.append(comp)
    return comps


def _farthest_fg_stop(g: np.ndarray, fg: int, info: dict[str, int | str]) -> int | None:
    H, W = g.shape
    axis = str(info["axis"])
    edge = str(info["edge"])
    line = int(info["line"])
    if axis == "v":
        rows = [int(rr) for rr in np.argwhere(g[:, line] == int(fg)).flatten()]
        if not rows:
            return None
        return int(max(rows) if edge == "top" else min(rows))
    cols = [int(cc) for cc in np.argwhere(g[line, :] == int(fg)).flatten()]
    if not cols:
        return None
    return int(max(cols) if edge == "left" else min(cols))


def _collision_stop(g: np.ndarray, fg: int, info: dict[str, int | str], layout: dict[str, object], *, use_collision: bool) -> int | None:
    far = _farthest_fg_stop(g, fg, info)
    if far is None or not use_collision:
        return far

    axis = str(info["axis"])
    edge = str(info["edge"])
    if axis == "v":
        src = int(info["r"])
        step = 1 if edge == "top" else -1
        if str(layout["mode"]) == "vertical_dup":
            candidates = [int(layout["r_split"])]
        else:
            candidates = [int(x) for x in layout["r_splits"]]
    else:
        src = int(info["c"])
        step = 1 if edge == "left" else -1
        if str(layout["mode"]) == "vertical_dup":
            candidates = [int(x) for x in layout["c_splits"]]
        else:
            candidates = [int(layout["c_split"])]

    valid = []
    for cand in candidates:
        if step == 1 and int(src) <= int(cand) <= int(far):
            valid.append(int(cand))
        if step == -1 and int(src) >= int(cand) >= int(far):
            valid.append(int(cand))
    if not valid:
        return int(far)
    return int(min(valid) if step == 1 else max(valid))


def _ray_cells(info: dict[str, int | str], stop: int | None) -> list[tuple[int, int]]:
    if stop is None:
        return []
    axis = str(info["axis"])
    edge = str(info["edge"])
    out: list[tuple[int, int]] = []
    if axis == "v":
        src = int(info["r"])
        line = int(info["line"])
        step = 1 if edge == "top" else -1
        for rr in range(int(src), int(stop) + step, step):
            out.append((int(rr), int(line)))
    else:
        src = int(info["c"])
        line = int(info["line"])
        step = 1 if edge == "left" else -1
        for cc in range(int(src), int(stop) + step, step):
            out.append((int(line), int(cc)))
    return out


def _slide_points(out: np.ndarray, pts: list[tuple[int, int]], fg: int, dh: str, dv: str) -> None:
    H, W = out.shape
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    rmin, rmax = int(min(rs)), int(max(rs))
    cmin, cmax = int(min(cs)), int(max(cs))
    dr = int(0 - rmin) if dh == "T" else int((H - 1) - rmax)
    dc = int(0 - cmin) if dv == "L" else int((W - 1) - cmax)
    for r, c in pts:
        out[int(r + dr), int(c + dc)] = int(fg)


def _solve_simultaneous(grid: np.ndarray, *, use_collision: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, fg, ctrl, ctrl_pts = _detect_roles(g)
    layout = _parse_layout(ctrl_pts, H, W)
    split = _split_cells((H, W), layout)

    out = np.full((H, W), int(bg), dtype=int)

    for info in layout["ctrl_infos"]:
        stop = _collision_stop(g, fg, info, layout, use_collision=use_collision)
        for rr, cc in _ray_cells(info, stop):
            if int(g[rr, cc]) == int(bg):
                out[rr, cc] = int(ctrl)

    for r, c in ctrl_pts:
        out[int(r), int(c)] = int(ctrl)

    move_fg: set[tuple[int, int]] = set()
    for r, c in np.argwhere(g == int(fg)):
        pt = (int(r), int(c))
        if pt in split:
            out[int(r), int(c)] = int(fg)
        else:
            move_fg.add(pt)

    for comp in _components(move_fg):
        labels = {_region_label(r, c, layout) for r, c in comp}
        assert len(labels) == 1
        label = next(iter(labels))
        keep, dest = _keep_and_dest(label, layout)
        if not keep:
            continue
        dh, dv = dest
        _slide_points(out, comp, fg, dh, dv)

    return out


def _solve_no_collision(grid: np.ndarray) -> np.ndarray:
    return _solve_simultaneous(np.array(grid, dtype=int), use_collision=False)


def _iter_order_key(info: dict[str, int | str]) -> tuple[int, int]:
    edge = str(info["edge"])
    if edge == "top":
        return (0, 0)
    if edge == "bottom":
        return (0, 1)
    if edge == "left":
        return (1, 0)
    return (1, 1)


def _farthest_fg_stop_from_set(
    current_fg: set[tuple[int, int]],
    info: dict[str, int | str],
) -> int | None:
    axis = str(info["axis"])
    edge = str(info["edge"])
    line = int(info["line"])
    if axis == "v":
        rows = [int(r) for r, c in current_fg if int(c) == int(line)]
        if not rows:
            return None
        return int(max(rows) if edge == "top" else min(rows))
    cols = [int(c) for r, c in current_fg if int(r) == int(line)]
    if not cols:
        return None
    return int(max(cols) if edge == "left" else min(cols))


def _collision_stop_from_set(
    current_fg: set[tuple[int, int]],
    info: dict[str, int | str],
    blocked: set[tuple[int, int]],
    *,
    use_collision: bool,
) -> int | None:
    far = _farthest_fg_stop_from_set(current_fg, info)
    if far is None or not use_collision or not blocked:
        return far
    path = _ray_cells(info, far)
    for rr, cc in path[1:]:
        if (int(rr), int(cc)) in blocked:
            return int(rr) if str(info["axis"]) == "v" else int(cc)
    return int(far)


def _apply_iter_step(
    current_fg: set[tuple[int, int]],
    info: dict[str, int | str],
    H: int,
    W: int,
    blocked: set[tuple[int, int]],
    *,
    use_collision: bool,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    axis = str(info["axis"])
    edge = str(info["edge"])
    line = int(info["line"])
    stop = _collision_stop_from_set(current_fg, info, blocked, use_collision=use_collision)
    ray_cells = set(_ray_cells(info, stop))

    keep_line: set[tuple[int, int]] = set()
    groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in current_fg:
        if axis == "v" and int(c) == int(line):
            keep_line.add((int(r), int(c)))
            continue
        if axis == "h" and int(r) == int(line):
            keep_line.add((int(r), int(c)))
            continue
        if axis == "v":
            key = ("A", "L" if int(c) < int(line) else "R")
        else:
            key = ("T" if int(r) < int(line) else "B", "A")
        groups.setdefault(key, []).append((int(r), int(c)))

    nxt = set(keep_line)
    for (hs, vs), pts in groups.items():
        if edge == "top":
            dh, dv = "T", str(vs)
        elif edge == "bottom":
            dh, dv = "B", str(vs)
        elif edge == "left":
            dh, dv = str(hs), "L"
        else:
            dh, dv = str(hs), "R"
        slid = np.full((H, W), -1, dtype=int)
        _slide_points(slid, pts, 1, dh, dv)
        nxt |= {(int(r), int(c)) for r, c in np.argwhere(slid == 1)}

    return nxt, ray_cells


def _solve_iterative(grid: np.ndarray, *, use_collision: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, fg, ctrl, ctrl_pts = _detect_roles(g)
    ordered_ctrls = sorted((_ctrl_info(H, W, r, c) for r, c in ctrl_pts), key=_iter_order_key)
    current_fg = {(int(r), int(c)) for r, c in np.argwhere(g == int(fg))}

    blocked: set[tuple[int, int]] = set()
    all_rays: set[tuple[int, int]] = set()
    for info in ordered_ctrls:
        current_fg, new_rays = _apply_iter_step(current_fg, info, H, W, blocked, use_collision=use_collision)
        blocked |= set(new_rays)
        all_rays |= set(new_rays)

    out = np.full((H, W), int(bg), dtype=int)
    for r, c in all_rays:
        if int(g[int(r), int(c)]) == int(bg):
            out[int(r), int(c)] = int(ctrl)
    for r, c in current_fg:
        out[int(r), int(c)] = int(fg)
    for r, c in ctrl_pts:
        out[int(r), int(c)] = int(ctrl)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_iterative(np.array(grid, dtype=int), use_collision=True)


def _place_shape_in_box(
    shape: frozenset[tuple[int, int]],
    box: tuple[int, int, int, int],
    *,
    anchor_r: str,
    anchor_c: str,
) -> set[tuple[int, int]]:
    r0, r1, c0, c1 = [int(x) for x in box]
    ph = int(max(r for r, _ in shape) + 1)
    pw = int(max(c for _, c in shape) + 1)
    box_h = int(r1 - r0 + 1)
    box_w = int(c1 - c0 + 1)
    if ph > box_h or pw > box_w:
        raise ValueError("shape does not fit the box")

    if anchor_r == "top":
        base_r = int(r0)
    elif anchor_r == "bottom":
        base_r = int(r1 - ph + 1)
    else:
        base_r = int(r0 + (box_h - ph) // 2)

    if anchor_c == "left":
        base_c = int(c0)
    elif anchor_c == "right":
        base_c = int(c1 - pw + 1)
    else:
        base_c = int(c0 + (box_w - pw) // 2)

    return {(int(base_r + r), int(base_c + c)) for r, c in shape}


def _region_boxes(
    sprite_r0: int,
    sprite_c0: int,
    sprite_h: int,
    sprite_w: int,
    layout: dict[str, object],
) -> dict[str, tuple[int, int, int, int]]:
    r1 = int(sprite_r0 + sprite_h - 1)
    c1 = int(sprite_c0 + sprite_w - 1)
    boxes: dict[str, tuple[int, int, int, int]] = {}
    if str(layout["mode"]) == "vertical_dup":
        c0, c2 = [int(x) for x in layout["c_splits"]]
        rs = int(layout["r_split"])
        boxes["TL"] = (int(sprite_r0), int(rs - 1), int(sprite_c0), int(c0 - 1))
        boxes["TM"] = (int(sprite_r0), int(rs - 1), int(c0 + 1), int(c2 - 1))
        boxes["TR"] = (int(sprite_r0), int(rs - 1), int(c2 + 1), int(c1))
        boxes["BL"] = (int(rs + 1), int(r1), int(sprite_c0), int(c0 - 1))
        boxes["BM"] = (int(rs + 1), int(r1), int(c0 + 1), int(c2 - 1))
        boxes["BR"] = (int(rs + 1), int(r1), int(c2 + 1), int(c1))
    else:
        r0, r2 = [int(x) for x in layout["r_splits"]]
        cs = int(layout["c_split"])
        boxes["TL"] = (int(sprite_r0), int(r0 - 1), int(sprite_c0), int(cs - 1))
        boxes["TR"] = (int(sprite_r0), int(r0 - 1), int(cs + 1), int(c1))
        boxes["ML"] = (int(r0 + 1), int(r2 - 1), int(sprite_c0), int(cs - 1))
        boxes["MR"] = (int(r0 + 1), int(r2 - 1), int(cs + 1), int(c1))
        boxes["BL"] = (int(r2 + 1), int(r1), int(sprite_c0), int(cs - 1))
        boxes["BR"] = (int(r2 + 1), int(r1), int(cs + 1), int(c1))
    return boxes


def _box_area(box: tuple[int, int, int, int]) -> int:
    r0, r1, c0, c1 = [int(x) for x in box]
    if r1 < r0 or c1 < c0:
        return 0
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _make_split_fg(
    sprite_r0: int,
    sprite_c0: int,
    sprite_h: int,
    sprite_w: int,
    layout: dict[str, object],
) -> set[tuple[int, int]]:
    r1 = int(sprite_r0 + sprite_h - 1)
    c1 = int(sprite_c0 + sprite_w - 1)
    pts: set[tuple[int, int]] = set()

    if str(layout["mode"]) == "vertical_dup":
        c0, c2 = [int(x) for x in layout["c_splits"]]
        rs = int(layout["r_split"])
        for cc in range(int(c0), int(c2) + 1):
            pts.add((int(rs), int(cc)))
        for rr in [int(rs - 1), int(rs), int(rs + 1), int(rs + 3)]:
            if int(sprite_r0) <= int(rr) <= int(r1):
                pts.add((int(rr), int(c0)))
        for rr in [int(rs - 3), int(rs - 1), int(rs), int(rs + 1)]:
            if int(sprite_r0) <= int(rr) <= int(r1):
                pts.add((int(rr), int(c2)))
    else:
        r0, r2 = [int(x) for x in layout["r_splits"]]
        cs = int(layout["c_split"])
        for rr in range(int(r0), int(r2) + 1):
            pts.add((int(rr), int(cs)))
        for cc in [int(cs - 1), int(cs), int(cs + 1), int(cs + 3)]:
            if int(sprite_c0) <= int(cc) <= int(c1):
                pts.add((int(r0), int(cc)))
        for cc in [int(cs - 3), int(cs - 1), int(cs), int(cs + 1)]:
            if int(sprite_c0) <= int(cc) <= int(c1):
                pts.add((int(r2), int(cc)))

    return pts


def _input_anchor(label: str) -> tuple[str, str]:
    row_part = label[0]
    col_part = label[1]
    if row_part == "T":
        ar = "bottom"
    elif row_part == "B":
        ar = "top"
    else:
        ar = "center"

    if col_part == "L":
        ac = "right"
    elif col_part == "R":
        ac = "left"
    else:
        ac = "center"
    return ar, ac


def _layout_for_generator(H: int, W: int) -> tuple[int, int, int, int, dict[str, object]]:
    sprite_h = 8
    sprite_w = random.randint(9, 10)
    sprite_r0 = random.randint(4, H - 5 - sprite_h + 1)
    sprite_c0 = random.randint(4, W - 5 - sprite_w + 1)
    local_r = random.randint(3, 4)
    local_c0 = random.randint(2, 3)
    local_c1 = random.randint(local_c0 + 3, sprite_w - 3)
    singleton_edge = random.choice(["left", "right"])
    layout = {
        "mode": "vertical_dup",
        "c_splits": [int(sprite_c0 + local_c0), int(sprite_c0 + local_c1)],
        "r_split": int(sprite_r0 + local_r),
        "singleton_edge": singleton_edge,
        "ctrl_edges": {"top", "bottom", singleton_edge},
    }
    return sprite_r0, sprite_c0, sprite_h, sprite_w, layout


def _make_ctrl_pts(H: int, W: int, layout: dict[str, object]) -> list[tuple[int, int]]:
    if str(layout["mode"]) == "vertical_dup":
        c0, c1 = [int(x) for x in layout["c_splits"]]
        r = int(layout["r_split"])
        pts = [(0, int(c0)), (int(H - 1), int(c1))]
        if str(layout["singleton_edge"]) == "left":
            pts.append((int(r), 0))
        else:
            pts.append((int(r), int(W - 1)))
        return pts

    r0, r1 = [int(x) for x in layout["r_splits"]]
    c = int(layout["c_split"])
    pts = [(int(r0), 0), (int(r1), int(W - 1))]
    if str(layout["singleton_edge"]) == "top":
        pts.append((0, int(c)))
    else:
        pts.append((int(H - 1), int(c)))
    return pts


def _collision_metrics(inp: np.ndarray) -> dict[str, int]:
    g = np.array(inp, dtype=int)
    H, W = g.shape
    bg, fg, _ctrl, ctrl_pts = _detect_roles(g)
    _ = bg
    layout = _parse_layout(ctrl_pts, H, W)

    shortened = 0
    total_gap = 0
    for info in layout["ctrl_infos"]:
        far = _collision_stop(g, fg, info, layout, use_collision=False)
        stop = _collision_stop(g, fg, info, layout, use_collision=True)
        if far is None or stop is None:
            continue
        if int(far) != int(stop):
            shortened += 1
            total_gap += abs(int(far) - int(stop))

    out_collision = _solve_simultaneous(g, use_collision=True)
    out_open = _solve_simultaneous(g, use_collision=False)
    diff_cells = int(np.count_nonzero(out_collision != out_open))
    return {"shortened": int(shortened), "gap": int(total_gap), "diff_cells": int(diff_cells)}


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(18, 20)
    bg, fg, ctrl_color = 1, 7, 2

    sprite_r0, sprite_c0, sprite_h, sprite_w, layout = _layout_for_generator(H, W)
    boxes = _region_boxes(sprite_r0, sprite_c0, sprite_h, sprite_w, layout)
    split_fg = _make_split_fg(sprite_r0, sprite_c0, sprite_h, sprite_w, layout)

    kept_labels = ["TL", "TR", "BL", "BR"]

    input_pieces: dict[str, set[tuple[int, int]]] = {}
    for lab in kept_labels:
        box = boxes[lab]
        area = _box_area(box)
        if area < 4:
            return None
        r0, r1, c0, c1 = box
        target_px = random.randint(3, min(6, area))
        shape = _grow_piece(int(r1 - r0 + 1), int(c1 - c0 + 1), target_px)
        ar, ac = _input_anchor(lab)
        pts = _place_shape_in_box(shape, box, anchor_r=ar, anchor_c=ac)
        if pts & split_fg:
            return None
        input_pieces[lab] = pts

    vanish_pts: set[tuple[int, int]] = set()
    if str(layout["mode"]) == "vertical_dup":
        c0, c1 = [int(x) for x in layout["c_splits"]]
        rs = int(layout["r_split"])
        for cc in range(int(c0 + 1), int(c1)):
            if int(sprite_r0) <= int(rs - 1) <= int(sprite_r0 + sprite_h - 1):
                vanish_pts.add((int(rs - 1), int(cc)))
            if int(sprite_r0) <= int(rs + 1) <= int(sprite_r0 + sprite_h - 1):
                vanish_pts.add((int(rs + 1), int(cc)))
    else:
        r0, r1 = [int(x) for x in layout["r_splits"]]
        cs = int(layout["c_split"])
        for rr in range(int(r0 + 1), int(r1)):
            if int(sprite_c0) <= int(cs - 1) <= int(sprite_c0 + sprite_w - 1):
                vanish_pts.add((int(rr), int(cs - 1)))
            if int(sprite_c0) <= int(cs + 1) <= int(sprite_c0 + sprite_w - 1):
                vanish_pts.add((int(rr), int(cs + 1)))

    combined_fg = set(split_fg) | vanish_pts
    for pts in input_pieces.values():
        if combined_fg & pts:
            return None
        combined_fg |= pts

    if not combined_fg:
        return None

    fg_arr = np.array(list(combined_fg), dtype=int)
    bbox_h = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    bbox_w = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if bbox_h < 7 or bbox_h > 10 or bbox_w < 7 or bbox_w > 10:
        return None

    density = len(combined_fg) / float(bbox_h * bbox_w)
    if not (0.39 <= density <= 0.60):
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in combined_fg:
        mask[int(r), int(c)] = 1
    if not is_contiguous(mask):
        return None

    ctrl_pts = _make_ctrl_pts(H, W, layout)
    if any(pt in combined_fg for pt in ctrl_pts):
        return None

    inp = np.full((H, W), int(bg), dtype=int)
    for r, c in combined_fg:
        inp[int(r), int(c)] = int(fg)
    for r, c in ctrl_pts:
        inp[int(r), int(c)] = int(ctrl_color)

    try:
        seq_out = main(inp)
        sim_out = _solve_simultaneous(inp, use_collision=True)
    except Exception:
        return None

    metrics = _collision_metrics(inp)
    if metrics["shortened"] < 2:
        return None
    if metrics["gap"] < 2 or metrics["diff_cells"] < 2:
        return None

    if np.array_equal(seq_out, _solve_no_collision(inp)):
        return None
    if np.array_equal(seq_out, sim_out):
        return None
    if int(np.count_nonzero(seq_out != sim_out)) < 6:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a non-degenerate fused sample after 400 attempts")
