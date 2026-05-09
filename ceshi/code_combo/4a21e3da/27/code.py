# Combo variant of 4a21e3da (index 27)
# combo_spec: D7/0+D9/0
# Name: 迭代逐条处理+分割线像素也移动

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


# Design analysis for the combo variant
# =====================================
# a) concreteization + trace:
#    - Keep the original vocabulary: one centered fg sprite, bg field, and 2 border ctrl pixels.
#    - The crucial fusion is "time order + copied cut-line pixels":
#      step 1 uses the vertical ctrl first; fg on that split column is not kept in place, but copied
#      into both left/right regions before those regions slide to the top or bottom corners.
#      step 2 then uses the horizontal ctrl on that moved layout; its split-row fg is again copied into
#      both top/bottom regions before those regions slide toward the left or right side.
#    - Small example:
#      original/D7-only step 1: left piece -> TL, right piece -> TR, split-column fg stays in the center.
#      combo step 1: left piece gets a copy of the split-column motif on its inner boundary, and the right
#      piece gets the same copy on its inner boundary; these extra boundary pixels can now lie on step 2's
#      split row.
#      D7-only step 2 therefore cuts a weaker row pattern than the combo step 2. The combo's second cut is
#      altered by the copied step-1 motif, which is the intended cross-step interaction.
# b) complexity calibration:
#    - The generator always uses exactly 2 perpendicular controls. With 1 control, D7 is invisible.
#    - It fixes the first ctrl to the top edge and the second to the left/right edge so the sequential
#      order is visually stable across examples instead of hidden behind too many cases.
#    - The sprite is still one connected fg component with the original scale regime (18-20 square grid,
#      7x8..9-ish sprite, 3 colors only), so the new difficulty comes from the fused state transition.
# c) implicit-assumption audit:
#    - Original solver assumption "all cuts are simultaneous" is invalid: the second ray must inspect the
#      fg produced by the first move, not the input fg.
#    - Original/D7 assumption "split-line fg stays put" is invalid: per-step split-line fg must be removed
#      from the center line and copied into every adjacent moving region.
#    - Original grouping by static input quadrants is invalid: after step 1 the geometry changes, so step 2
#      must rebuild its top/bottom groups from the intermediate state.
#    - Original generator assumption "visibility comes for free once both controls exist" is too weak here:
#      we must guarantee a copied vertical split pixel lands exactly on the second split row after step 1,
#      otherwise the D9 part may not affect step 2 at all.
# d) placement legality:
#    - Left/right subpieces are generated inside disjoint rectangles on either side of the vertical split.
#    - A dedicated split-column pixel is chosen so that after the top-slide of step 1 it lands on row
#      `r_split`; this is within bounds because the sprite starts at row 4 and has height 7.
#    - Connectivity, bbox, density, and border margins are checked on the final input sprite.
# e) visual observability:
#    - Examples visibly show two phases: first the centered sprite becomes two top corner pieces, then those
#      are cut again along the border-controlled row and shift sideways.
#    - The D9 contribution is also visible because the stage-1 pieces gain an inner "copied seam" not
#      present in D7-only outputs, and that seam changes the second cut.
# f) visual cleanliness:
#    - Only 3 colors are used and rays are still intermittent ctrl lines on bg cells only.
#    - The copied seam is attached to corner pieces rather than left in the center, so the output remains
#      readable instead of stacking multiple fg layers on the original split line.
# g) inferability:
#    - No hidden color ordering or numeric-color rule is used. The only stable cue is geometry: border ctrl
#      position determines cut orientation and move direction, and the copied split motif repeats structurally.
#
# New solver assumptions and generator guarantees:
# - `main()` assumes exactly one fg color and one ctrl color distinct from bg; the generator uses {1,2,7}.
# - `main()` assumes at most one vertical and one horizontal ctrl; the generator always uses exactly one of each.
# - `main()` assumes step 2 has visible fg on its split row in the intermediate state; the generator enforces
#   this and additionally checks that some of that row comes specifically from copied step-1 split pixels.
# - `main()` assumes each region that receives copied split pixels also has its own non-split fg anchor; the
#   generator builds both side pieces first, so copied pixels never create a floating piece by themselves.


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _detect_roles(g: np.ndarray) -> tuple[int, int, int, list[tuple[int, int]]]:
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})
    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(int(bg))
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != int(ctrl)))
    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(ctrl))]
    return bg, fg, ctrl, ctrl_pts


def _ctrl_info(r: int, c: int, H: int, W: int) -> dict[str, int | str]:
    if int(r) == 0:
        return {"edge": "top", "axis": "v", "line": int(c), "r": int(r), "c": int(c)}
    if int(r) == int(H - 1):
        return {"edge": "bottom", "axis": "v", "line": int(c), "r": int(r), "c": int(c)}
    if int(c) == 0:
        return {"edge": "left", "axis": "h", "line": int(r), "r": int(r), "c": int(c)}
    if int(c) == int(W - 1):
        return {"edge": "right", "axis": "h", "line": int(r), "r": int(r), "c": int(c)}
    raise AssertionError("control pixel must be on the border")


def _ctrl_order_key(info: dict[str, int | str]) -> tuple[int, int]:
    edge = str(info["edge"])
    if edge == "top":
        return (0, 0)
    if edge == "bottom":
        return (0, 1)
    if edge == "left":
        return (1, 0)
    return (1, 1)


def _move_region_to_corner(pts: list[tuple[int, int]], dh: str, dv: str, H: int, W: int) -> set[tuple[int, int]]:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    rmin, rmax = int(min(rs)), int(max(rs))
    cmin, cmax = int(min(cs)), int(max(cs))
    dr = int(0 - rmin) if dh == "T" else int((H - 1) - rmax)
    dc = int(0 - cmin) if dv == "L" else int((W - 1) - cmax)
    return {(int(r + dr), int(c + dc)) for r, c in pts}


def _ray_cells_for_step(
    current_fg: set[tuple[int, int]],
    info: dict[str, int | str],
    H: int,
    W: int,
) -> set[tuple[int, int]]:
    edge = str(info["edge"])
    axis = str(info["axis"])
    line = int(info["line"])
    ray_cells: set[tuple[int, int]] = set()

    if axis == "v":
        rows = sorted(int(r) for r, c in current_fg if int(c) == int(line))
        if not rows:
            return ray_cells
        stop = int(max(rows)) if edge == "top" else int(min(rows))
        rng = range(0, int(stop) + 1) if edge == "top" else range(int(H - 1), int(stop) - 1, -1)
        for rr in rng:
            if (int(rr), int(line)) not in current_fg:
                ray_cells.add((int(rr), int(line)))
    else:
        cols = sorted(int(c) for r, c in current_fg if int(r) == int(line))
        if not cols:
            return ray_cells
        stop = int(max(cols)) if edge == "left" else int(min(cols))
        rng = range(0, int(stop) + 1) if edge == "left" else range(int(W - 1), int(stop) - 1, -1)
        for cc in rng:
            if (int(line), int(cc)) not in current_fg:
                ray_cells.add((int(line), int(cc)))

    return ray_cells


def _apply_step_combo(
    current_fg: set[tuple[int, int]],
    info: dict[str, int | str],
    H: int,
    W: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    edge = str(info["edge"])
    axis = str(info["axis"])
    line = int(info["line"])
    ray_cells = _ray_cells_for_step(current_fg, info, H, W)

    if axis == "v":
        groups: dict[str, list[tuple[int, int]]] = {"L": [], "R": []}
        split_pixels: list[tuple[int, int]] = []
        for r, c in current_fg:
            if int(c) == int(line):
                split_pixels.append((int(r), int(c)))
            elif int(c) < int(line):
                groups["L"].append((int(r), int(c)))
            else:
                groups["R"].append((int(r), int(c)))

        nxt: set[tuple[int, int]] = set()
        dh = "T" if edge == "top" else "B"
        for side, pts in groups.items():
            if not pts:
                continue
            piece = pts + split_pixels
            nxt |= _move_region_to_corner(piece, dh, side, H, W)
        return nxt, ray_cells

    groups = {"T": [], "B": []}
    split_pixels = []
    for r, c in current_fg:
        if int(r) == int(line):
            split_pixels.append((int(r), int(c)))
        elif int(r) < int(line):
            groups["T"].append((int(r), int(c)))
        else:
            groups["B"].append((int(r), int(c)))

    nxt = set()
    dv = "L" if edge == "left" else "R"
    for side, pts in groups.items():
        if not pts:
            continue
        piece = pts + split_pixels
        nxt |= _move_region_to_corner(piece, side, dv, H, W)
    return nxt, ray_cells


def _apply_step_keep_line(
    current_fg: set[tuple[int, int]],
    info: dict[str, int | str],
    H: int,
    W: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    edge = str(info["edge"])
    axis = str(info["axis"])
    line = int(info["line"])
    ray_cells = _ray_cells_for_step(current_fg, info, H, W)

    if axis == "v":
        groups = {"L": [], "R": []}
        keep_line: set[tuple[int, int]] = set()
        for r, c in current_fg:
            if int(c) == int(line):
                keep_line.add((int(r), int(c)))
            elif int(c) < int(line):
                groups["L"].append((int(r), int(c)))
            else:
                groups["R"].append((int(r), int(c)))
        nxt = set(keep_line)
        dh = "T" if edge == "top" else "B"
        for side, pts in groups.items():
            if pts:
                nxt |= _move_region_to_corner(pts, dh, side, H, W)
        return nxt, ray_cells

    groups = {"T": [], "B": []}
    keep_line = set()
    for r, c in current_fg:
        if int(r) == int(line):
            keep_line.add((int(r), int(c)))
        elif int(r) < int(line):
            groups["T"].append((int(r), int(c)))
        else:
            groups["B"].append((int(r), int(c)))
    nxt = set(keep_line)
    dv = "L" if edge == "left" else "R"
    for side, pts in groups.items():
        if pts:
            nxt |= _move_region_to_corner(pts, side, dv, H, W)
    return nxt, ray_cells


def _simulate_steps(
    current_fg: set[tuple[int, int]],
    ordered_ctrls: list[dict[str, int | str]],
    H: int,
    W: int,
    *,
    keep_line: bool,
    capture_states: bool = False,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], list[set[tuple[int, int]]]]:
    fg_now = set((int(r), int(c)) for r, c in current_fg)
    ray_cells: set[tuple[int, int]] = set()
    states: list[set[tuple[int, int]]] = [set(fg_now)] if capture_states else []

    for info in ordered_ctrls:
        if keep_line:
            fg_now, new_rays = _apply_step_keep_line(fg_now, info, H, W)
        else:
            fg_now, new_rays = _apply_step_combo(fg_now, info, H, W)
        ray_cells |= new_rays
        if capture_states:
            states.append(set(fg_now))

    return fg_now, ray_cells, states


def _render_state(
    H: int,
    W: int,
    bg: int,
    fg: int,
    ctrl: int,
    ctrl_pts: list[tuple[int, int]],
    fg_pts: set[tuple[int, int]],
    ray_cells: set[tuple[int, int]],
) -> np.ndarray:
    out = np.full((H, W), int(bg), dtype=int)
    for r, c in ray_cells:
        out[int(r), int(c)] = int(ctrl)
    for r, c in fg_pts:
        out[int(r), int(c)] = int(fg)
    for r, c in ctrl_pts:
        out[int(r), int(c)] = int(ctrl)
    return out


def _is_split_fg(r: int, c: int, r_split: int | None, c_split: int | None) -> bool:
    return (c_split is not None and int(c) == int(c_split)) or (r_split is not None and int(r) == int(r_split))


def _region_of_pixel(r: int, c: int, r_split: int | None, c_split: int | None) -> tuple[str, str]:
    hs = "A"
    vs = "A"
    if r_split is not None:
        hs = "T" if int(r) < int(r_split) else "B"
    if c_split is not None:
        vs = "L" if int(c) < int(c_split) else "R"
    return hs, vs


def _adjacent_regions_for_split_pixel(
    r: int,
    c: int,
    r_split: int | None,
    c_split: int | None,
) -> list[tuple[str, str]]:
    if r_split is None:
        hs_opts = ["A"]
    elif int(r) < int(r_split):
        hs_opts = ["T"]
    elif int(r) > int(r_split):
        hs_opts = ["B"]
    else:
        hs_opts = ["T", "B"]

    if c_split is None:
        vs_opts = ["A"]
    elif int(c) < int(c_split):
        vs_opts = ["L"]
    elif int(c) > int(c_split):
        vs_opts = ["R"]
    else:
        vs_opts = ["L", "R"]

    return [(hs, vs) for hs in hs_opts for vs in vs_opts]


def _touches_edges(hs: str, vs: str) -> set[str]:
    te: set[str] = set()
    if hs in ("T", "A"):
        te.add("top")
    if hs in ("B", "A"):
        te.add("bottom")
    if vs in ("L", "A"):
        te.add("left")
    if vs in ("R", "A"):
        te.add("right")
    return te


def _pick_dest(hs: str, vs: str, E: set[str], r_split: int | None, c_split: int | None) -> tuple[str, str]:
    if r_split is not None:
        dh = hs
    elif ("top" in E) and ("bottom" not in E):
        dh = "T"
    elif ("bottom" in E) and ("top" not in E):
        dh = "B"
    else:
        dh = "T"

    if c_split is not None:
        dv = vs
    elif ("left" in E) and ("right" not in E):
        dv = "L"
    elif ("right" in E) and ("left" not in E):
        dv = "R"
    else:
        dv = "L"

    return dh, dv


def _solve_simultaneous_split_copy(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, fg, ctrl, ctrl_pts = _detect_roles(g)

    E: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    ctrl_infos = []
    for r, c in ctrl_pts:
        info = _ctrl_info(r, c, H, W)
        ctrl_infos.append(info)
        E.add(str(info["edge"]))
        if str(info["axis"]) == "v":
            c_split = int(info["line"])
        else:
            r_split = int(info["line"])

    out = np.full((H, W), int(bg), dtype=int)
    for info in ctrl_infos:
        for r, c in _ray_cells_for_step({(int(rr), int(cc)) for rr, cc in np.argwhere(g == int(fg))}, info, H, W):
            out[int(r), int(c)] = int(ctrl)
    for r, c in ctrl_pts:
        out[int(r), int(c)] = int(ctrl)

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    split_pixels: list[tuple[int, int]] = []
    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if _is_split_fg(r, c, r_split, c_split):
            split_pixels.append((r, c))
        else:
            reg = _region_of_pixel(r, c, r_split, c_split)
            regions.setdefault(reg, []).append((r, c))

    split_copies: dict[tuple[str, str], list[tuple[int, int]]] = {reg: [] for reg in regions}
    for r, c in split_pixels:
        for reg in _adjacent_regions_for_split_pixel(r, c, r_split, c_split):
            if reg in split_copies:
                split_copies[reg].append((r, c))

    for reg, pts in regions.items():
        hs, vs = reg
        if not (_touches_edges(hs, vs) & E):
            continue
        piece = pts + split_copies.get(reg, [])
        moved = _move_region_to_corner(piece, *_pick_dest(hs, vs, E, r_split, c_split), H, W)
        for r, c in moved:
            out[int(r), int(c)] = int(fg)

    return out


def _solve_sequential_keep_line(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, fg, ctrl, ctrl_pts = _detect_roles(g)
    ordered_ctrls = sorted((_ctrl_info(r, c, H, W) for r, c in ctrl_pts), key=_ctrl_order_key)
    current_fg = {(int(r), int(c)) for r, c in np.argwhere(g == int(fg))}
    final_fg, ray_cells, _ = _simulate_steps(current_fg, ordered_ctrls, H, W, keep_line=True, capture_states=False)
    return _render_state(H, W, bg, fg, ctrl, ctrl_pts, final_fg, ray_cells)


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, fg, ctrl, ctrl_pts = _detect_roles(g)
    ordered_ctrls = sorted((_ctrl_info(r, c, H, W) for r, c in ctrl_pts), key=_ctrl_order_key)
    current_fg = {(int(r), int(c)) for r, c in np.argwhere(g == int(fg))}

    combo_fg, combo_rays, combo_states = _simulate_steps(current_fg, ordered_ctrls, H, W, keep_line=False, capture_states=True)
    keep_fg, keep_rays, keep_states = _simulate_steps(current_fg, ordered_ctrls, H, W, keep_line=True, capture_states=True)
    combo_out = _render_state(H, W, bg, fg, ctrl, ctrl_pts, combo_fg, combo_rays)
    keep_out = _render_state(H, W, bg, fg, ctrl, ctrl_pts, keep_fg, keep_rays)
    sim_out = _solve_simultaneous_split_copy(g)

    h_infos = [info for info in ordered_ctrls if str(info["axis"]) == "h"]
    row = int(h_infos[0]["line"]) if h_infos else -1
    combo_row = {int(c) for r, c in combo_states[1] if int(r) == int(row)} if len(combo_states) > 1 else set()
    keep_row = {int(c) for r, c in keep_states[1] if int(r) == int(row)} if len(keep_states) > 1 else set()

    return {
        "stage1_extra_on_second_line": len(combo_row - keep_row),
        "final_diff_vs_d7_only": int(np.sum(combo_out != keep_out)),
        "final_diff_vs_d9_only": int(np.sum(combo_out != sim_out)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply the control rays sequentially.

    Each step uses the current fg layout:
    1. draw the current ray up to the farthest fg on that line,
    2. remove fg on that split line from the center,
    3. copy those split-line fg pixels into every adjacent moving region,
    4. move the augmented regions to the edge-determined corners/sides.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, fg, ctrl, ctrl_pts = _detect_roles(g)
    ordered_ctrls = sorted((_ctrl_info(r, c, H, W) for r, c in ctrl_pts), key=_ctrl_order_key)
    current_fg = {(int(r), int(c)) for r, c in np.argwhere(g == int(fg))}
    final_fg, ray_cells, _ = _simulate_steps(current_fg, ordered_ctrls, H, W, keep_line=False, capture_states=False)
    return _render_state(H, W, bg, fg, ctrl, ctrl_pts, final_fg, ray_cells)


def _make_side_piece(h: int, w: int, target_px: int, anchor_side: str) -> frozenset[tuple[int, int]]:
    anchor_col = 0 if anchor_side == "left" else int(w - 1)
    cells = {(int(r), int(anchor_col)) for r in range(int(h))}
    bridge_row = random.randint(0, int(h) - 1)
    outer_col = int(w - 1 - anchor_col)
    for cc in range(min(int(anchor_col), int(outer_col)), max(int(anchor_col), int(outer_col)) + 1):
        cells.add((int(bridge_row), int(cc)))

    while len(cells) < int(target_px):
        frontier = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < int(h) and 0 <= nc < int(w) and (int(nr), int(nc)) not in cells:
                    frontier.add((int(nr), int(nc)))
        if not frontier:
            break
        cands = list(frontier)
        random.shuffle(cands)
        cands.sort(key=lambda p: abs(int(p[1]) - int(anchor_col)))
        pick = random.choice(cands[: min(4, len(cands))])
        cells.add((int(pick[0]), int(pick[1])))

    return frozenset((int(r), int(c)) for r, c in cells)


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(18, 20)
    bg, fg, ctrl = 1, 7, 2

    h_edge = random.choice(["left", "right"])
    piece_h = 7
    left_w, right_w = random.choice([(3, 4), (4, 3), (4, 4)])

    c_lo = max(int(4 + left_w), int(W * 0.45))
    c_hi = min(int(W - 5 - right_w), int(W * 0.60))
    if c_lo > c_hi:
        return None
    c_split = random.randint(int(c_lo), int(c_hi))

    sprite_r0 = 4
    r_split = random.choice([4, 5])
    ctrl_pts = [(0, int(c_split)), (int(r_split), 0 if h_edge == "left" else int(W - 1))]

    left_target = random.randint(piece_h + 3, min(piece_h + 7, piece_h * left_w))
    right_target = random.randint(piece_h + 3, min(piece_h + 7, piece_h * right_w))
    left_piece = _make_side_piece(piece_h, left_w, left_target, "right")
    right_piece = _make_side_piece(piece_h, right_w, right_target, "left")

    sprite_c0 = int(c_split - left_w)
    left_pts = {(int(sprite_r0 + r), int(sprite_c0 + c)) for r, c in left_piece}
    right_pts = {(int(sprite_r0 + r), int(c_split + 1 + c)) for r, c in right_piece}

    split_rows_local = {int(r_split)}
    desired_n_split = random.randint(3, 4)
    while len(split_rows_local) < int(desired_n_split):
        split_rows_local.add(random.randint(0, piece_h - 1))
    split_pixels = {(int(sprite_r0 + rr), int(c_split)) for rr in split_rows_local}

    combined = set(left_pts) | set(right_pts) | set(split_pixels)
    if not combined:
        return None

    fg_arr = np.array(sorted(combined))
    bbox_h = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    bbox_w = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if bbox_h < 5 or bbox_h > 8 or bbox_w < 8 or bbox_w > 10:
        return None

    density = len(combined) / float(bbox_h * bbox_w)
    if density < 0.39 or density > 0.60:
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in combined:
        if r < 4 or r > H - 5 or c < 4 or c > W - 5:
            return None
        mask[int(r), int(c)] = 1
    if not is_contiguous(mask):
        return None

    inp = np.full((H, W), int(bg), dtype=int)
    for r, c in combined:
        inp[int(r), int(c)] = int(fg)
    for r, c in ctrl_pts:
        inp[int(r), int(c)] = int(ctrl)

    try:
        _bg, _fg, _ctrl, found_ctrl = _detect_roles(inp)
        ordered_ctrls = sorted((_ctrl_info(r, c, H, W) for r, c in found_ctrl), key=_ctrl_order_key)
        if len(ordered_ctrls) != 2:
            return None
        if str(ordered_ctrls[0]["axis"]) != "v" or str(ordered_ctrls[1]["axis"]) != "h":
            return None

        current_fg = {(int(r), int(c)) for r, c in np.argwhere(inp == int(fg))}
        combo_final, combo_rays, combo_states = _simulate_steps(
            current_fg, ordered_ctrls, H, W, keep_line=False, capture_states=True
        )
        keep_final, keep_rays, keep_states = _simulate_steps(
            current_fg, ordered_ctrls, H, W, keep_line=True, capture_states=True
        )
        if len(combo_states) != 3 or len(keep_states) != 3:
            return None

        combo_row = {int(c) for r, c in combo_states[1] if int(r) == int(r_split)}
        keep_row = {int(c) for r, c in keep_states[1] if int(r) == int(r_split)}
        extra_cols = combo_row - keep_row
        if int(left_w) not in extra_cols:
            return None
        if int(W - right_w - 1) not in extra_cols:
            return None
        if not any(int(c) < int(left_w) for c in combo_row):
            return None
        if not any(int(c) > int(W - right_w - 1) for c in combo_row):
            return None

        combo_out = _render_state(H, W, bg, fg, ctrl, ctrl_pts, combo_final, combo_rays)
        keep_out = _render_state(H, W, bg, fg, ctrl, ctrl_pts, keep_final, keep_rays)
        sim_out = _solve_simultaneous_split_copy(inp)

        if np.array_equal(combo_out, keep_out):
            return None
        if np.array_equal(combo_out, sim_out):
            return None
        if int(np.sum(combo_out != keep_out)) < 6:
            return None
        if int(np.sum(combo_out != sim_out)) < 6:
            return None

        metrics = _fusion_metrics(inp)
        if metrics["stage1_extra_on_second_line"] < 2:
            return None
        if metrics["final_diff_vs_d7_only"] < 6 or metrics["final_diff_vs_d9_only"] < 6:
            return None
    except Exception:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid combo input after 400 attempts")
