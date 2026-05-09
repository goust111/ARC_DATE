# Combo variant of 20270e3b (index 29)
# combo_spec: D7/2+D12/0+D13/0
# Name: 自由放置基准裁剪重叠
"""
Design analysis
===============

a) Concrete behavior / trace
   - Original task:
       1. detect the base sprite anchored at the input top-left
       2. shift the plug so its horizontal ctrl bar sits one row above the base ctrl bar
       3. allow overlap only at the connector seam
       4. tight-crop the merged union
   - This combo:
       1. docking translation is unchanged
       2. plug may extend above, left, right, or below the base bbox after docking
       3. inside the base bbox, plug overwrites base even on body-body overlap
       4. output is only the base sprite's original bbox window
   - Micro-trace:
       suppose the base bbox is 6x7, the aligned plug spans rows -1..6 and cols -1..8,
       and some of its stem overlaps a base rib at rows 3..4. The original solver would
       fail because the plug extends above/left and overlaps the body away from the seam.
       This combo instead keeps only rows 0..5 / cols 0..6, discards the rest, and lets
       the in-window plug pixels overwrite the base before ctrl->obj recoloring.

b) Complexity calibration
   - If the generator only clipped the plug on the right/bottom, D12 would not actually
     matter. To make the combo observable, every accepted sample has at least one clipped
     plug pixel above or left of the base window.
   - If the generator only allowed deeper overlap without any plug-only in-window pixels,
     the output could look like an ordinary base frame. Every accepted sample therefore
     also has deep body overlap plus extra visible plug pixels inside the window.
   - The shape family stays small: one framed base, one branchy plug, one horizontal
     connector length shared by both sprites.

c) Hidden assumptions review
   - Component parsing and connector matching from the original task still hold.
   - The old merged-bbox rendering assumption is invalid because D7 fixes the output
     window to the base bbox only.
   - The old assertion "plug must not extend above/left of base" is invalid because D12
     explicitly removes that restriction.
   - The old assertion "overlap only at ctrl seam" is invalid because D13 allows body
     overlap with plug priority.
   - New main() assumptions:
       * exactly two 8-connected foreground components
       * both components have non-empty, shape-matching horizontal ctrl bars
       * the base component is still the one whose bbox is closest to the input origin
       * base ctrl free side faces down; plug ctrl free side faces up
     The generator enforces all of these.

d) Placement legality
   - Generation is output-first: build a docked scene in base-window coordinates plus
     small margins, then crop base and plug separately and place them apart on the input.
   - Input sprites never overlap and are separated by at least one blank row or column,
     so label_mask(..., connectivity=8) still sees exactly two components.
   - Plug spill outside the base window is allowed in the docked scene, but those pixels
     are intentionally clipped from the output.

e) Visual observability
   - In every accepted sample:
       * some docked plug pixels lie outside the base window
       * at least one of those clipped pixels is above or left of the base window
       * the visible in-window plug also forms plug-only structure below the seam
   - Across examples, the human can see that the plug is docked the same way as before,
     yet the output footprint never expands beyond the base rectangle.

f) Visual cleanliness
   - The base is a sparse frame with 1-2 internal ribs; the plug is a stem with a few
     short branches and an optional crown above the connector. This keeps the scene legible
     despite overlap and clipping.
   - Output colors remain only bg=1 and obj=4 after connector recoloring.

g) Color inference
   - No new color mapping is introduced. ctrl=7 still disappears into obj=4 after docking.
   - Because the combo changes geometry rather than palette, few examples are enough for
     a human to infer the rule.
"""

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste


BG = 1
OBJ = 4
CTRL = 7


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _norm_conn(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [int(r) for r, _c in conn_pts]
    cs = [int(c) for _r, c in conn_pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (int(r0), int(c0))


def _crop_non_bg(grid: np.ndarray, bg: int = BG) -> np.ndarray | None:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _add_h(mask: np.ndarray, r: int, c0: int, c1: int) -> None:
    c0, c1 = sorted((int(c0), int(c1)))
    mask[int(r), c0 : c1 + 1] = True


def _add_v(mask: np.ndarray, c: int, r0: int, r1: int) -> None:
    r0, r1 = sorted((int(r0), int(r1)))
    mask[r0 : r1 + 1, int(c)] = True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock the plug onto the base, clip to the base bbox, and allow plug overwrite.

    The docking translation is the same as in the original task, but two old constraints
    are removed:
    - the docked plug may extend above or left of the base bbox
    - body-body overlap is allowed and resolves in favor of the plug
    """
    g = np.array(grid, dtype=int)

    lab, n = label_mask(g != int(BG), connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        comps.append({(int(r), int(c)) for r, c in np.argwhere(lab == int(k))})
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = set(comps[0]), set(comps[1])

    def conn(sprite: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(r, c) for (r, c) in sprite if int(g[int(r), int(c)]) == int(CTRL)}

    connA = conn(A)
    connB = conn(B)
    assert connA and connB
    shpA, _ = _norm_conn(connA)
    shpB, _ = _norm_conn(connB)
    assert shpA == shpB

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, connD = A, connA
        plug, connP = B, connB
    else:
        base, connD = B, connB
        plug, connP = A, connA

    base_body = base - connD
    plug_body = plug - connP
    for r, c in connD:
        assert (int(r + 1), int(c)) not in base_body
    for r, c in connP:
        assert (int(r - 1), int(c)) not in plug_body

    _sp, (pr0, pc0) = _norm_conn(connP)
    _sd, (dr0, dc0) = _norm_conn(connD)
    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    br0, bc0, br1, bc1 = _bbox_of_pts(base)
    assert int(br0) == 0 and int(bc0) == 0
    out = np.full((int(br1 - br0 + 1), int(bc1 - bc0 + 1)), int(BG), dtype=int)

    for r, c in base:
        out[int(r - br0), int(c - bc0)] = int(g[int(r), int(c)])

    for r, c in plug_shift:
        if not (int(br0) <= int(r) <= int(br1) and int(bc0) <= int(c) <= int(bc1)):
            continue
        out[int(r - br0), int(c - bc0)] = int(g[int(r - tr), int(c - tc)])

    out[out == int(CTRL)] = int(OBJ)
    return out


def _build_base() -> tuple[np.ndarray, int, list[int]] | None:
    bh = _rng.randint(5, 7)
    bw = _rng.randint(5, 9)
    conn_len = _rng.randint(1, min(3, bw - 2))

    conn_row_max = min(3, bh - 4)
    if conn_row_max < 1:
        return None
    conn_row = _rng.randint(1, conn_row_max)

    conn_c0_min = 1
    conn_c0_max = bw - conn_len - 2
    if conn_c0_max < conn_c0_min:
        return None
    conn_c0 = _rng.randint(conn_c0_min, conn_c0_max)
    conn_cols = list(range(conn_c0, conn_c0 + conn_len))

    base_mask = np.zeros((bh, bw), dtype=bool)
    _add_h(base_mask, 0, 0, bw - 1)
    _add_h(base_mask, bh - 1, 0, bw - 1)
    _add_v(base_mask, 0, 0, bh - 1)
    _add_v(base_mask, bw - 1, 0, bh - 1)

    for c in conn_cols:
        _add_v(base_mask, c, 0, conn_row)

    rib_rows = [conn_row + 2]
    if conn_row + 3 <= bh - 2 and _rng.random() < 0.65:
        rib_rows.append(_rng.randint(conn_row + 3, bh - 2))
    rib_rows = sorted(set(rib_rows))
    for rr in rib_rows:
        rib_end = _rng.randint(conn_cols[-1], bw - 1)
        _add_h(base_mask, rr, 0, rib_end)

    if conn_row + 2 <= bh - 2 and _rng.random() < 0.55:
        pillar_col = _rng.randint(conn_cols[-1], bw - 2)
        _add_v(base_mask, pillar_col, conn_row + 2, bh - 1)

    if _rng.random() < 0.35:
        roof_col = _rng.randint(max(conn_cols[-1] + 1, 1), bw - 1)
        _add_v(base_mask, roof_col, 0, _rng.randint(1, min(conn_row, bh - 2)))

    for c in conn_cols:
        base_mask[conn_row + 1, c] = False

    base_grid = np.full((bh, bw), int(BG), dtype=int)
    base_grid[base_mask] = int(OBJ)
    for c in conn_cols:
        base_grid[conn_row, c] = int(CTRL)

    if not is_contiguous(base_grid != int(BG), background=0, connectivity=8):
        return None
    return base_grid, int(conn_row), conn_cols


def _build_plug(
    bh: int, bw: int, conn_row: int, conn_cols: list[int]
) -> tuple[np.ndarray, dict[str, int]] | None:
    top_extra = _rng.randint(0, 1)
    left_extra = _rng.randint(0, 1)
    if top_extra == 0 and left_extra == 0:
        if _rng.random() < 0.5:
            top_extra = 1
        else:
            left_extra = 1
    right_extra = _rng.randint(0, 2)
    bottom_extra = _rng.randint(0, 1)

    ext_h = int(top_extra + bh + bottom_extra)
    ext_w = int(left_extra + bw + right_extra)
    plug_mask = np.zeros((ext_h, ext_w), dtype=bool)

    prow = int(top_extra + conn_row - 1)
    gconn = [int(left_extra + c) for c in conn_cols]
    conn_start, conn_end = int(gconn[0]), int(gconn[-1])

    stem_bottom_min = int(top_extra + conn_row + 3)
    stem_bottom_max = int(ext_h - 1)
    if stem_bottom_min > stem_bottom_max:
        return None
    stem_bottom = _rng.randint(stem_bottom_min, stem_bottom_max)

    for c in gconn:
        plug_mask[prow, c] = True
        plug_mask[prow + 1 : stem_bottom + 1, c] = True

    branch_rows = {int(prow + 1), int(stem_bottom)}
    if stem_bottom - (prow + 1) >= 1:
        branch_rows.add(_rng.randint(prow + 1, stem_bottom))
    for rr in sorted(branch_rows):
        if left_extra > 0 and _rng.random() < 0.8:
            left_end = 0
        else:
            left_end = max(0, conn_start - _rng.randint(0, 2))
        if right_extra > 0 and _rng.random() < 0.7:
            right_end = ext_w - 1
        else:
            right_end = min(ext_w - 1, conn_end + _rng.randint(0, 2))
        _add_h(plug_mask, rr, left_end, right_end)

    if left_extra > 0:
        rr = _rng.choice(sorted(branch_rows))
        _add_h(plug_mask, rr, 0, conn_start)

    if top_extra > 0:
        side_choices: list[int] = []
        if conn_start > 0:
            side_choices.append(conn_start - 1)
        if conn_end < ext_w - 1:
            side_choices.append(conn_end + 1)
        if not side_choices:
            return None
        side_col = int(_rng.choice(side_choices))
        _add_v(plug_mask, side_col, 0, prow)
        cap_left = max(0, side_col - _rng.randint(0, 1 + left_extra))
        cap_right = min(ext_w - 1, side_col + _rng.randint(0, 1 + right_extra))
        _add_h(plug_mask, 0, cap_left, cap_right)
        for c in gconn:
            plug_mask[prow - 1, c] = False

    if bottom_extra > 0 and stem_bottom < ext_h - 1:
        side_col = _rng.randint(max(0, conn_start - 1), min(ext_w - 1, conn_end + 1))
        _add_v(plug_mask, side_col, stem_bottom, ext_h - 1)

    for c in gconn:
        if prow == 0:
            continue
        if plug_mask[prow - 1, c]:
            return None

    plug_grid = np.full((ext_h, ext_w), int(BG), dtype=int)
    plug_grid[plug_mask] = int(OBJ)
    for c in gconn:
        plug_grid[prow, c] = int(CTRL)

    if not is_contiguous(plug_grid != int(BG), background=0, connectivity=8):
        return None
    for c in gconn:
        if int(plug_grid[prow + 1, c]) != int(OBJ):
            return None

    pixel_count = int(np.sum(plug_grid != int(BG)))
    if pixel_count < 5 or pixel_count > 29:
        return None

    metrics = {
        "top_extra": int(top_extra),
        "left_extra": int(left_extra),
        "right_extra": int(right_extra),
        "bottom_extra": int(bottom_extra),
    }
    return plug_grid, metrics


def _render_window(base_grid: np.ndarray, plug_grid: np.ndarray, conn_row: int, conn_cols: list[int]) -> tuple[np.ndarray, dict[str, int]]:
    bh, bw = map(int, base_grid.shape)
    conn_c0 = int(conn_cols[0])

    plug_pts = np.argwhere(plug_grid != int(BG))
    conn_pts = np.argwhere(plug_grid == int(CTRL))
    assert plug_pts.size > 0 and conn_pts.size > 0
    pr0 = int(conn_pts[:, 0].min())
    pc0 = int(conn_pts[:, 1].min())
    tr = int((conn_row - 1) - pr0)
    tc = int(conn_c0 - pc0)

    out = np.array(base_grid, copy=True)
    deep_overlap = 0
    plug_only_inside = 0
    outside_total = 0
    outside_top = 0
    outside_left = 0
    outside_right = 0
    outside_bottom = 0

    for r, c in plug_pts:
        rr = int(r + tr)
        cc = int(c + tc)
        if 0 <= rr < bh and 0 <= cc < bw:
            if int(base_grid[rr, cc]) == int(OBJ) and rr > int(conn_row):
                deep_overlap += 1
            if int(base_grid[rr, cc]) == int(BG):
                plug_only_inside += 1
            out[rr, cc] = int(plug_grid[r, c])
        else:
            outside_total += 1
            if rr < 0:
                outside_top += 1
            if cc < 0:
                outside_left += 1
            if rr >= bh:
                outside_bottom += 1
            if cc >= bw:
                outside_right += 1

    out[out == int(CTRL)] = int(OBJ)
    metrics = {
        "deep_overlap": int(deep_overlap),
        "plug_only_inside": int(plug_only_inside),
        "outside_total": int(outside_total),
        "outside_top": int(outside_top),
        "outside_left": int(outside_left),
        "outside_right": int(outside_right),
        "outside_bottom": int(outside_bottom),
    }
    return out, metrics


def _build_example() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]] | None:
    built = _build_base()
    if built is None:
        return None
    base_grid, conn_row, conn_cols = built
    bh, bw = map(int, base_grid.shape)

    plug_grid, extra = _build_plug(bh, bw, conn_row, conn_cols) or (None, None)
    if plug_grid is None or extra is None:
        return None

    expected, metrics = _render_window(base_grid, plug_grid, conn_row, conn_cols)
    metrics.update(extra)

    if metrics["outside_total"] <= 0:
        return None
    if metrics["outside_top"] + metrics["outside_left"] <= 0:
        return None
    if metrics["deep_overlap"] <= 0:
        return None
    if metrics["plug_only_inside"] <= 0:
        return None

    base_sprite = _crop_non_bg(base_grid, BG)
    plug_sprite = _crop_non_bg(plug_grid, BG)
    if base_sprite is None or plug_sprite is None:
        return None

    if base_sprite.shape != base_grid.shape:
        return None
    if int(np.sum(base_sprite != int(BG))) < 5:
        return None

    return base_sprite, plug_sprite, expected, metrics


def _place_input(base_sprite: np.ndarray, plug_sprite: np.ndarray) -> np.ndarray | None:
    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)

    layouts: list[str] = []
    if bw + 1 + pw <= 13 and max(bh, ph) <= 11:
        layouts.append("horizontal")
    if bh + 1 + ph <= 11 and max(bw, pw) <= 13:
        layouts.append("vertical")
    if not layouts:
        return None

    layout = _rng.choice(layouts)
    if layout == "horizontal":
        canvas_h = _rng.randint(max(7, bh, ph), 11)
        canvas_w = _rng.randint(max(3, bw + 1 + pw), 13)
        pr = _rng.randint(0, canvas_h - ph)
        pc = _rng.randint(bw + 1, canvas_w - pw)
    else:
        canvas_h = _rng.randint(max(7, bh + 1 + ph), 11)
        canvas_w = _rng.randint(max(3, bw, pw), 13)
        pr = _rng.randint(bh + 1, canvas_h - ph)
        pc = _rng.randint(0, canvas_w - pw)

    canvas = np.full((canvas_h, canvas_w), int(BG), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=int(BG))
    paste(canvas, plug_sprite, pr, pc, transparent=int(BG))

    lab, n = label_mask(canvas != int(BG), connectivity=8)
    if int(n) != 2:
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    built = _build_example()
    if built is None:
        return None
    base_sprite, plug_sprite, expected, _metrics = built

    canvas = _place_input(base_sprite, plug_sprite)
    if canvas is None:
        return None

    try:
        result = main(canvas)
    except Exception:
        return None
    if not np.array_equal(result, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 500 attempts")
