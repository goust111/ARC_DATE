from __future__ import annotations

# Variant of 4a21e3da
# Dimension: D7 — 多步处理序列 (multi_step)
# Change: 两步：先用射线切割精灵，再将片段移到角落... → 迭代切割——逐条射线依次处理

import random
import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


# Design analysis for the D7 variant
# ----------------------------------
# concreteization:
# - Keep the original visual vocabulary: one fg sprite, bg field, and 2 edge control pixels.
# - Make the rule change entirely about time order. The vertical-edge control is always applied first.
#   It cuts the centered sprite into left/right regions and moves those regions to the top or bottom
#   corners. Only then is the horizontal-edge control applied to that intermediate arrangement.
# - The second ray therefore cuts already-moved corner pieces rather than the original centered sprite.
#   This is the key visible difference from the original simultaneous 2-ray version.
# - To make the second step observable, the generator forces the first moved pieces to cross the second
#   control row. After step 1, that row contains fg at both extreme sides, so step 2 visibly acts on
#   both corner pieces.
#
# small trace:
# - Original 2-control logic:
#   centered sprite + vertical split + horizontal split -> 4 quadrants computed from the original sprite
#   at once -> three quadrants move directly to their final corners.
# - New D7 logic:
#   step 1 (vertical): centered sprite -> left/right halves move to top/bottom corners, while split-line
#   fg stays near the original center;
#   step 2 (horizontal): that new layout is cut again along the horizontal control row -> upper/lower
#   regions of the moved layout are shifted again toward the left/right control side.
# - Difference:
#   the second cut depends on the intermediate state. The same input therefore produces a different
#   output from the original simultaneous solver, and the generator explicitly rejects degenerate cases
#   where both outputs would match.
#
# complexity calibration:
# - The variant always uses 2 controls. With only 1 control, sequential and original behavior are the
#   same, so D7 would be invisible.
# - Keep the original low-color, single-sprite setup so the new complexity comes from step order only.
# - Preserve the original “region as a whole slides to a corner” semantics for each individual ray.
#   This avoids adding a new component-level rule that would change more than D7.
#
# implicit-assumption audit against the original solver:
# - Original code assumes one vertical split and one horizontal split can be read once and then applied
#   simultaneously. In D7 that assumption breaks: the second cut must read the fg produced by the first
#   cut, not the input fg.
# - Original code groups fg once by static quadrants of the input. In D7 those groups must be rebuilt
#   after each step because the first move changes positions before the second cut.
# - Original ray drawing reads the input grid `g`. In D7 the stop position for the second ray must be
#   computed from the intermediate fg state, otherwise the ray length is wrong.
# - Original keep-in-place logic for split-line fg is still valid, but now it must run per step on the
#   current fg set.
#
# placement legality:
# - Input fg stays at least 4 cells from every border, so role detection remains unambiguous.
# - The sprite is constructed from a left piece, split-line connector pixels, and a right piece, and is
#   checked for 4-connectivity.
# - The sprite height is fixed at 7 so the first-step corner pieces are tall enough to intersect the
#   second control row while the original input still respects the 4-cell border margin.
#
# visual observability / cleanliness:
# - The second control row is chosen so it intersects the original sprite and the first-step corner
#   pieces. This makes both control pixels visibly relevant from the examples.
# - A non-degeneracy check requires the step-1 state to have fg on the second control row at both the
#   left and right extremes, ensuring the second ray cuts across the moved layout rather than only the
#   static center split pixels.
# - Another check rejects cases where the sequential output accidentally equals the original
#   simultaneous output.
#
# new solver assumptions and generator guarantees:
# - `main()` assumes there is at least one border control color and exactly one fg color distinct from it.
#   The generator uses fixed colors {bg, ctrl, fg} = {1, 2, 7}.
# - `main()` assumes the sequential order is visually derivable: vertical-edge control first, then
#   horizontal-edge control. The generator always emits exactly one of each.
# - `main()` assumes the second step has fg on its split row to cut. The generator verifies this from the
#   actual step-1 state before accepting a sample.


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
            if edge == "top":
                rng = range(0, int(stop) + 1)
            else:
                rng = range(int(H - 1), int(stop) - 1, -1)
            for rr in rng:
                if (int(rr), int(line)) not in current_fg:
                    ray_cells.add((int(rr), int(line)))

        for r, c in current_fg:
            if int(c) == int(line):
                keep_line.add((int(r), int(c)))
            else:
                key = ("A", "L" if int(c) < int(line) else "R")
                groups.setdefault(key, []).append((int(r), int(c)))
    else:
        cols = sorted(int(c) for r, c in current_fg if int(r) == int(line))
        if cols:
            stop = int(max(cols)) if edge == "left" else int(min(cols))
            if edge == "left":
                rng = range(0, int(stop) + 1)
            else:
                rng = range(int(W - 1), int(stop) - 1, -1)
            for cc in rng:
                if (int(line), int(cc)) not in current_fg:
                    ray_cells.add((int(line), int(cc)))

        for r, c in current_fg:
            if int(r) == int(line):
                keep_line.add((int(r), int(c)))
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sequentially apply edge-control rays: vertical control first, then horizontal control.

    Each ray repeats the original single-ray logic on the current fg state:
    cut by one split line, keep fg on that line in place, and slide the remaining
    left/right or top/bottom regions to the corresponding corners.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, fg, ctrl, ctrl_pts = _detect_roles(g)

    ordered_ctrls = sorted((_ctrl_info(r, c, H, W) for r, c in ctrl_pts), key=_ctrl_order_key)
    current_fg = {(int(r), int(c)) for r, c in np.argwhere(g == int(fg))}
    final_fg, ray_cells, _states = _simulate_iterative(current_fg, ordered_ctrls, H, W)

    out = np.full((H, W), int(bg), dtype=int)
    for r, c in ray_cells:
        out[int(r), int(c)] = int(ctrl)
    for r, c in final_fg:
        out[int(r), int(c)] = int(fg)
    for r, c in ctrl_pts:
        out[int(r), int(c)] = int(ctrl)
    return out


def _solve_simultaneous_baseline(grid: np.ndarray) -> np.ndarray:
    """Original 4a21e3da behavior, used only to reject sequentially-degenerate samples."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, fg, ctrl, ctrl_pts = _detect_roles(g)

    E: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for r, c in ctrl_pts:
        info = _ctrl_info(r, c, H, W)
        E.add(str(info["edge"]))
        if str(info["axis"]) == "v":
            c_split = int(info["line"])
        else:
            r_split = int(info["line"])

    out = np.full((H, W), int(bg), dtype=int)

    for r, c in ctrl_pts:
        info = _ctrl_info(r, c, H, W)
        edge = str(info["edge"])
        line = int(info["line"])
        if str(info["axis"]) == "v":
            rows = [int(rr) for rr in np.argwhere(g[:, int(line)] == int(fg)).flatten()]
            if rows:
                stop = int(max(rows)) if edge == "top" else int(min(rows))
                if edge == "top":
                    rng = range(0, int(stop) + 1)
                else:
                    rng = range(int(H - 1), int(stop) - 1, -1)
                for rr in rng:
                    if int(g[int(rr), int(line)]) == int(bg):
                        out[int(rr), int(line)] = int(ctrl)
        else:
            cols = [int(cc) for cc in np.argwhere(g[int(line), :] == int(fg)).flatten()]
            if cols:
                stop = int(max(cols)) if edge == "left" else int(min(cols))
                if edge == "left":
                    rng = range(0, int(stop) + 1)
                else:
                    rng = range(int(W - 1), int(stop) - 1, -1)
                for cc in rng:
                    if int(g[int(line), int(cc)]) == int(bg):
                        out[int(line), int(cc)] = int(ctrl)

    for r, c in ctrl_pts:
        out[int(r), int(c)] = int(ctrl)

    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (c_split is not None and int(c) == int(c_split)) or (r_split is not None and int(r) == int(r_split)):
            out[int(r), int(c)] = int(fg)

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (c_split is not None and int(c) == int(c_split)) or (r_split is not None and int(r) == int(r_split)):
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if int(r) < int(r_split) else "B"
        if c_split is not None:
            vs = "L" if int(c) < int(c_split) else "R"
        regions.setdefault((hs, vs), []).append((int(r), int(c)))

    def touches_edges(hs: str, vs: str) -> set[str]:
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

    def pick_dest(hs: str, vs: str) -> tuple[str, str]:
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
        return str(dh), str(dv)

    for (hs, vs), pts in regions.items():
        if not (touches_edges(hs, vs) & E):
            continue
        dh, dv = pick_dest(hs, vs)
        for r, c in _move_region_to_corner(pts, dh, dv, H, W):
            out[int(r), int(c)] = int(fg)

    return out


def _make_side_piece(h: int, w: int, target_px: int, anchor_side: str) -> frozenset[tuple[int, int]]:
    anchor_col = 0 if anchor_side == "left" else int(w - 1)
    cells = {(int(r), int(anchor_col)) for r in range(int(h))}

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

    if len({c for _, c in cells}) == 1 and int(w) >= 2:
        row = random.randint(0, int(h) - 1)
        extra_col = 1 if int(anchor_col) == 0 else int(w - 2)
        cells.add((int(row), int(extra_col)))

    return frozenset((int(r), int(c)) for r, c in cells)


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(18, 20)
    bg, fg, ctrl = 1, 7, 2

    piece_h = 7
    left_w = random.choice([3, 4])
    right_w = random.choice([3, 4])
    if int(left_w + right_w) < 7:
        return None

    v_edge = random.choice(["top", "bottom"])
    h_edge = random.choice(["left", "right"])

    c_lo = max(int(4 + left_w), int(W * 0.4))
    c_hi = min(int(W - 5 - right_w), int(W * 0.6))
    if c_lo > c_hi:
        return None
    c_split = random.randint(int(c_lo), int(c_hi))

    if v_edge == "top":
        sprite_r0 = 4
        cut_local = random.choice([4, 5])
        r_split = int(cut_local)
        ctrl_v = (0, int(c_split))
    else:
        sprite_r0 = int(H - 5 - piece_h + 1)
        cut_local = random.choice([1, 2])
        r_split = int(H - piece_h + cut_local)
        ctrl_v = (int(H - 1), int(c_split))

    ctrl_h = (int(r_split), 0 if h_edge == "left" else int(W - 1))
    ctrl_pts = [ctrl_v, ctrl_h]

    left_target = random.randint(piece_h + 4, min(piece_h + 7, piece_h * left_w))
    right_target = random.randint(piece_h + 4, min(piece_h + 7, piece_h * right_w))
    left_piece = _make_side_piece(piece_h, left_w, left_target, "right")
    right_piece = _make_side_piece(piece_h, right_w, right_target, "left")

    sprite_c0 = int(c_split - left_w)
    left_pts = {(int(sprite_r0 + r), int(sprite_c0 + c)) for r, c in left_piece}
    right_pts = {(int(sprite_r0 + r), int(c_split + 1 + c)) for r, c in right_piece}

    cut_input_local = int(r_split - sprite_r0)
    if not (0 <= cut_input_local < piece_h):
        return None

    split_rows = {int(cut_input_local)}
    desired_n_split = random.choice([3, 4])
    while len(split_rows) < int(desired_n_split):
        split_rows.add(random.randint(0, piece_h - 1))
    split_pixels = {(int(sprite_r0 + rr), int(c_split)) for rr in split_rows}

    combined = left_pts | right_pts | split_pixels

    sprite_w = int(left_w + 1 + right_w)
    if sprite_w < 8 or sprite_w > 10:
        return None
    if piece_h < 5 or piece_h > 8:
        return None

    density = len(combined) / float(piece_h * sprite_w)
    if density < 0.39 or density > 0.60:
        return None

    if not combined:
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
        final_fg, _ray_cells, states = _simulate_iterative(current_fg, ordered_ctrls, H, W, capture_states=True)
        if len(states) != 3:
            return None

        stage1_fg = states[1]
        row_cols = [int(c) for r, c in stage1_fg if int(r) == int(r_split)]
        if not row_cols:
            return None
        if not any(int(c) < int(left_w) for c in row_cols):
            return None
        if not any(int(c) > int(W - 1 - right_w) for c in row_cols):
            return None

        seq_out = main(inp)
        sim_out = _solve_simultaneous_baseline(inp)
        if np.array_equal(seq_out, sim_out):
            return None
        if int(np.sum(seq_out != sim_out)) < 6:
            return None

        if not final_fg:
            return None
    except Exception:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid D7 input after 400 attempts")
