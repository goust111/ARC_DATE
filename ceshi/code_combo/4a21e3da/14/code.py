from __future__ import annotations

# Combo variant of 4a21e3da (index 14)
# combo_spec: D7/0+D8/0
# Name: 迭代逐条处理+三控制像素

import random
import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


# Design analysis for the combo variant
# -------------------------------------
# concreteization:
# - The visual vocabulary stays close to the source task: one centered-ish fg sprite, one ctrl color used
#   only on the border and along intermittent rays, and a uniform bg.
# - D8/0 fixes the meta-structure to exactly 3 control pixels. This combo uses 2 vertical controls
#   (top and bottom) plus 1 horizontal control (left), yielding 3 cut lines: columns c1 < c2 and row r.
#   Those lines define 6 geometric regions in the input sprite: T/L, T/M, T/R, B/L, B/M, B/R.
# - D7/0 changes time semantics. The 3 controls are not interpreted simultaneously. They are applied in a
#   deterministic order inferred from edge positions: top vertical first, then bottom vertical, then left
#   horizontal. Every step reads the fg produced by the previous step.
#
# small trace:
# - Original 4a21e3da with two controls is "static": read all split lines from the input once, classify
#   every fg pixel once, and move each region directly to its destination corner.
# - This combo is "dynamic":
#   1. top control at c1 cuts the centered sprite into left vs right and moves those two pieces to the
#      top corners while c1-line fg stays in place;
#   2. bottom control at c2 then cuts that intermediate layout. It sees the piece already moved by step 1,
#      so the second cut line acts on a different geometry than the input;
#   3. left control at row r finally cuts the stage-2 layout and moves upper/lower parts to the left
#      corners.
# - The generator forces a concrete dependency chain:
#   1. the stage-1 right piece lands on c2, so step 2 would be impossible to reason about from the input
#      alone;
#   2. the stage-2 bottom pieces cross row r, so step 3 acts on moved corner pieces rather than only on
#      the original centered sprite.
#
# complexity calibration:
# - Exactly 3 controls keeps the new D8 structure obvious while avoiding too many free cases for a
#   2-4-example ARC task.
# - The geometry is intentionally restricted: fixed edge pattern (top, bottom, left), fixed control order,
#   fixed sprite bbox size 7x10, and fixed line offsets inside the bbox. That makes the multi-step rule
#   learnable while still allowing random irregular pieces.
# - The sprite remains one connected fg component and still uses only 3 colors. The extra complexity is
#   from temporal state changes, not from extra object types or hidden color encodings.
#
# implicit-assumption audit against the original solver:
# - Original code assumes at most 1 vertical and 1 horizontal split. That breaks here because there are
#   2 vertical controls. The combo solver therefore handles each control independently and never stores a
#   single global c_split / r_split.
# - Original code draws rays from the input grid only. That is wrong for D7/0 because later rays must stop
#   at fg created by earlier moves. Here step 2 and step 3 both rely on that.
# - Original code groups fg once from the input. That also breaks: each step must regroup the current fg by
#   the active cut line only.
# - Original code's "fg on split line stays in place" remains valid, but now it is a per-step rule on the
#   current state, not a one-time property of the input.
#
# placement legality:
# - The generator uses a fixed 7x10 sprite bbox fully inside a 4-cell border margin.
# - The top cut column is the third bbox column, the bottom cut column is the eighth bbox column, and the
#   left-control row is the sixth bbox row. This leaves non-empty left/middle/right bands and ensures the
#   input visibly contains 6 potential regions.
# - A full fg bridge on the control row keeps the input sprite 4-connected. Vertical band spines make the
#   later corner pieces tall enough that step 2 reaches c2 and step 3 reaches row r.
#
# visual observability / cleanliness:
# - All 3 control pixels are on different border positions and the 3 rays are long. The output visibly
#   reflects all of them.
# - The generator rejects samples unless stage 1 creates multiple fg pixels on c2, stage 2 creates fg on
#   row r at both left and right extremes, and step 3 changes the stage-2 layout. Those checks prevent
#   accidental collapse back to a simpler 1-step or 2-step rule.
# - The alternative order top -> left -> bottom is also simulated and rejected if it matches the intended
#   output. This keeps the ordering information meaningful to a solver.
#
# new solver assumptions and generator guarantees:
# - `main()` assumes the only non-bg border color is the control color; the generator uses fixed colors
#   {bg, ctrl, fg} = {1, 2, 7}.
# - `main()` assumes every control lies on the border and that edge order alone is enough to infer the
#   intended sequence. The generator always emits top, bottom, and left controls.
# - `main()` assumes later split lines actually intersect the fg state they are meant to process. The
#   generator verifies that from the simulated intermediate states before accepting a sample.


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


def _apply_step(
    current_fg: set[tuple[int, int]],
    info: dict[str, int | str],
    H: int,
    W: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    edge = str(info["edge"])
    axis = str(info["axis"])
    line = int(info["line"])

    groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
    keep_line: set[tuple[int, int]] = set()
    ray_cells: set[tuple[int, int]] = set()

    if axis == "v":
        rows = sorted(int(r) for r, c in current_fg if int(c) == int(line))
        if rows:
            stop = int(max(rows)) if edge == "top" else int(min(rows))
            rng = range(0, int(stop) + 1) if edge == "top" else range(int(H - 1), int(stop) - 1, -1)
            for rr in rng:
                if (int(rr), int(line)) not in current_fg:
                    ray_cells.add((int(rr), int(line)))

        for r, c in current_fg:
            if int(c) == int(line):
                keep_line.add((int(r), int(c)))
            else:
                groups.setdefault(("A", "L" if int(c) < int(line) else "R"), []).append((int(r), int(c)))
    else:
        cols = sorted(int(c) for r, c in current_fg if int(r) == int(line))
        if cols:
            stop = int(max(cols)) if edge == "left" else int(min(cols))
            rng = range(0, int(stop) + 1) if edge == "left" else range(int(W - 1), int(stop) - 1, -1)
            for cc in rng:
                if (int(line), int(cc)) not in current_fg:
                    ray_cells.add((int(line), int(cc)))

        for r, c in current_fg:
            if int(r) == int(line):
                keep_line.add((int(r), int(c)))
            else:
                groups.setdefault(("T" if int(r) < int(line) else "B", "A"), []).append((int(r), int(c)))

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
        nxt |= _move_region_to_corner(pts, dh, dv, H, W)

    return nxt, ray_cells


def _simulate_iterative(
    current_fg: set[tuple[int, int]],
    ordered_ctrls: list[dict[str, int | str]],
    H: int,
    W: int,
    *,
    capture_states: bool = False,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], list[set[tuple[int, int]]]]:
    fg_now = set((int(r), int(c)) for r, c in current_fg)
    ray_cells: set[tuple[int, int]] = set()
    states: list[set[tuple[int, int]]] = [set(fg_now)] if capture_states else []

    for info in ordered_ctrls:
        fg_now, new_rays = _apply_step(fg_now, info, H, W)
        ray_cells |= new_rays
        if capture_states:
            states.append(set(fg_now))

    return fg_now, ray_cells, states


def _render_output(
    H: int,
    W: int,
    bg: int,
    fg: int,
    ctrl: int,
    ctrl_pts: list[tuple[int, int]],
    final_fg: set[tuple[int, int]],
    ray_cells: set[tuple[int, int]],
) -> np.ndarray:
    out = np.full((H, W), int(bg), dtype=int)
    for r, c in ray_cells:
        out[int(r), int(c)] = int(ctrl)
    for r, c in final_fg:
        out[int(r), int(c)] = int(fg)
    for r, c in ctrl_pts:
        out[int(r), int(c)] = int(ctrl)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply the 3 border controls sequentially.

    Order is inferred from edge positions: top, then bottom, then left/right.
    Each step repeats the original single-ray logic on the current fg state:
    keep split-line fg in place, move one side or half to the active edge's corners,
    and let that intermediate state feed the next ray.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, fg, ctrl, ctrl_pts = _detect_roles(g)
    ordered_ctrls = sorted((_ctrl_info(r, c, H, W) for r, c in ctrl_pts), key=_ctrl_order_key)
    current_fg = {(int(r), int(c)) for r, c in np.argwhere(g == int(fg))}
    final_fg, ray_cells, _states = _simulate_iterative(current_fg, ordered_ctrls, H, W)
    return _render_output(H, W, bg, fg, ctrl, ctrl_pts, final_fg, ray_cells)


def _sample_rows(h: int, k: int, *, exclude: set[int] | None = None) -> list[int]:
    ex = exclude or set()
    pool = [int(r) for r in range(int(h)) if int(r) not in ex]
    random.shuffle(pool)
    return pool[: int(min(k, len(pool)))]


def _build_left_band(h: int, row_cut: int) -> set[tuple[int, int]]:
    cells = {(int(r), 1) for r in range(int(h))}
    for r in _sample_rows(h, random.randint(2, 3), exclude={int(row_cut)}):
        cells.add((int(r), 0))
    return cells


def _build_middle_band(h: int, row_cut: int) -> set[tuple[int, int]]:
    cells = {(int(r), 0) for r in range(int(h))}
    rows1 = _sample_rows(h, random.randint(1, 2), exclude={int(row_cut)})
    for r in rows1:
        cells.add((int(r), 1))
    rows2 = random.sample(rows1, k=random.randint(1, len(rows1))) if rows1 else []
    for r in rows2:
        cells.add((int(r), 2))
    rows3 = random.sample(rows2, k=random.randint(0, len(rows2))) if rows2 else []
    for r in rows3:
        cells.add((int(r), 3))
    return cells


def _build_right_band(h: int, row_cut: int) -> set[tuple[int, int]]:
    cells = {(int(r), 0) for r in range(int(h))}
    for r in _sample_rows(h, random.randint(2, 3), exclude={int(row_cut)}):
        cells.add((int(r), 1))
    return cells


def _place_local(cells: set[tuple[int, int]], r0: int, c0: int) -> set[tuple[int, int]]:
    return {(int(r0 + r), int(c0 + c)) for r, c in cells}


def _simulate_grid(
    inp: np.ndarray,
    ordered_ctrls: list[dict[str, int | str]] | None = None,
    *,
    capture_states: bool = False,
) -> tuple[np.ndarray, list[set[tuple[int, int]]], list[dict[str, int | str]], set[tuple[int, int]]]:
    g = np.array(inp, dtype=int)
    H, W = g.shape
    bg, fg, ctrl, ctrl_pts = _detect_roles(g)
    infos = ordered_ctrls or sorted((_ctrl_info(r, c, H, W) for r, c in ctrl_pts), key=_ctrl_order_key)
    current_fg = {(int(r), int(c)) for r, c in np.argwhere(g == int(fg))}
    final_fg, ray_cells, states = _simulate_iterative(current_fg, infos, H, W, capture_states=capture_states)
    out = _render_output(H, W, bg, fg, ctrl, ctrl_pts, final_fg, ray_cells)
    return out, states, infos, final_fg


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(18, 20)
    bg, fg, ctrl = 1, 7, 2

    sprite_h = 7
    sprite_w = 10
    sprite_r0 = int(H - 11)
    sprite_c0 = int(W - 14)
    row_cut = 5
    c1_local = 2
    c2_local = 7

    r_split = int(sprite_r0 + row_cut)
    c1 = int(sprite_c0 + c1_local)
    c2 = int(sprite_c0 + c2_local)

    ctrl_pts = [(0, int(c1)), (int(H - 1), int(c2)), (int(r_split), 0)]

    row_bridge = {(int(row_cut), int(c)) for c in range(int(sprite_w))}
    split1_rows = [0, int(row_cut)] + _sample_rows(sprite_h, 1, exclude={0, int(row_cut)})
    split2_rows = [int(row_cut)] + _sample_rows(sprite_h, 2, exclude={int(row_cut)})
    split1 = {(int(r), int(c1_local)) for r in split1_rows}
    split2 = {(int(r), int(c2_local)) for r in split2_rows}

    left_band = _build_left_band(sprite_h, row_cut)
    middle_band = _place_local(_build_middle_band(sprite_h, row_cut), 0, 3)
    right_band = _place_local(_build_right_band(sprite_h, row_cut), 0, 8)

    local_fg = row_bridge | split1 | split2 | left_band | middle_band | right_band
    global_fg = _place_local(local_fg, sprite_r0, sprite_c0)

    bbox_rs = [r for r, _ in global_fg]
    bbox_cs = [c for _, c in global_fg]
    if int(max(bbox_rs) - min(bbox_rs) + 1) != int(sprite_h):
        return None
    if int(max(bbox_cs) - min(bbox_cs) + 1) != int(sprite_w):
        return None

    if min(bbox_rs) < 4 or max(bbox_rs) > H - 5 or min(bbox_cs) < 4 or max(bbox_cs) > W - 5:
        return None

    density = len(global_fg) / float(sprite_h * sprite_w)
    if density < 0.39 or density > 0.60:
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in global_fg:
        mask[int(r), int(c)] = 1
    if not is_contiguous(mask):
        return None

    inp = np.full((H, W), int(bg), dtype=int)
    for r, c in global_fg:
        inp[int(r), int(c)] = int(fg)
    for r, c in ctrl_pts:
        inp[int(r), int(c)] = int(ctrl)

    try:
        seq_out, states, infos, final_fg = _simulate_grid(inp, capture_states=True)
        if len(infos) != 3:
            return None
        if [str(info["edge"]) for info in infos] != ["top", "bottom", "left"]:
            return None
        if len(states) != 4:
            return None

        stage0, stage1, stage2, stage3 = states
        if stage3 != final_fg:
            return None

        stage1_on_c2 = {(r, c) for r, c in stage1 if int(c) == int(c2)}
        if len(stage1_on_c2) < 4:
            return None
        if len(stage1_on_c2 - {(r, c) for r, c in stage0 if int(c) == int(c2)}) < 3:
            return None

        stage2_on_row = {(r, c) for r, c in stage2 if int(r) == int(r_split)}
        if not any(int(c) <= int(c1) for _, c in stage2_on_row):
            return None
        if not any(int(c) >= W - 2 for _, c in stage2_on_row):
            return None
        if len(stage2_on_row - {(r, c) for r, c in stage1 if int(r) == int(r_split)}) < 2:
            return None

        if stage2 == stage3:
            return None

        alt_infos = [infos[0], infos[2], infos[1]]
        alt_out, _alt_states, _alt_infos, _alt_final_fg = _simulate_grid(inp, ordered_ctrls=alt_infos, capture_states=False)
        if np.array_equal(seq_out, alt_out):
            return None

        first_two_fg, _rays_two, _states_two = _simulate_iterative(stage0, infos[:2], H, W, capture_states=False)
        if first_two_fg == stage3:
            return None
    except Exception:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid combo input after 300 attempts")
