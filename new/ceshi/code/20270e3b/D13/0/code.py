# Variant of 20270e3b
# Dimension: D13 — 重叠处理规则 (meta)
# Change: 对接时仅允许在连接器区域重叠（ctrl遇到obj），其他位置不允许重叠
#   → 允许部分body重叠（叠加优先级）：plug 的 body 可以与 base 的 body 重叠，且 plug 覆盖 base。
"""
Design analysis for D13
======================

a) Concrete behavior / trace
   - The original task docks the plug by aligning the two horizontal ctrl bars so the
     plug connector row lands exactly one row above the base connector row.
   - In the original solver, after this translation, any overlap outside the 2-row
     connector seam is rejected.
   - In this variant the translation is unchanged, but deeper overlap is legal:
     plug body pixels may continue downward into the base body. When that happens,
     plug is still stamped last, so plug wins on overlap.
   - Small example:
     original: plug stem may overlap base ctrl at row r, but if it also reaches a
     base rib at row r+2 the assert fails.
     variant: the same stem is allowed to pass through that rib; output keeps the
     union silhouette with the plug's lower pixels still visible below the rib.

b) Complexity calibration
   - If generation kept the original random non-overlapping shapes and merely removed
     the assert, many samples would visually look identical to the base task.
   - To make D13 observable, the generator now constructs assembled states with:
       1. at least one deep body-body overlap row below the connector seam
       2. at least one plug-only extension below/around that overlap
     This produces tighter nesting rather than simple seam-joining.

c) Hidden assumptions in original main()
   - Component parsing / connector matching / base selection remain valid.
   - The critical old assumption was:
       if cur != bg and val != bg: assert cur == ctrl or val == ctrl
     That line encoded "only connector-zone overlap exists".
   - In D13 that premise is false, because body-body overlap is intentional. The
     render order itself is still correct, so the fix is to keep base-first then
     always let plug overwrite.
   - The orientation checks remain useful: base connector still has free side down,
     plug connector still has free side up.

d) Placement legality
   - Input still contains exactly two non-overlapping components.
   - Generator works output-first: construct aligned base/plug in assembled coords,
     crop each sprite separately, then place base at (0,0) and place plug elsewhere
     on a larger canvas with guaranteed separation.
   - Assembled plug is constrained to never extend above or left of base, preserving
     the original "base stays at output top-left after crop" invariant.

e) Visual observability
   - Every accepted sample has deep_overlap > 0, meaning the new rule is actually
     exercised. The plug also contributes additional non-overlapping body cells below
     that overlap so the nesting is visible in the output shape.

f) Visual cleanliness
   - The shape family is still sparse and thin-walled: a base chamber/frame plus a
     hanging plug stem with small branches. Only two colors appear in the output
     after ctrl recoloring, so the merged result remains legible.

g) Color inference
   - Colors are unchanged from the original task: bg=1, obj=4, ctrl=7.
   - The only color event is still "ctrl disappears into obj after docking", which
     remains easy to infer from a few examples.

Main() assumptions introduced by the variant
--------------------------------------------
- Plug overwrite is unconditional on overlap; therefore inputs must keep the same
  connector alignment/orientation rules so the translated plug position is unique.
- The generator guarantees:
    * exactly two 8-connected foreground components in the input
    * matching horizontal connectors
    * base anchored at the input top-left
    * plug not extending above/left of base after docking
    * at least one deep body-body overlap cell below the seam
"""

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste


def main(grid: np.ndarray) -> np.ndarray:
    """Dock the plug onto the base, allowing plug body to overwrite base body."""
    g = np.array(grid, dtype=int)
    bg, obj, ctrl = 1, 4, 7

    lab, n = label_mask(g != bg, connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        comps.append({(int(r), int(c)) for r, c in np.argwhere(lab == int(k))})
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = set(comps[0]), set(comps[1])

    def bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
        coords = np.array(list(pts), dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        return int(r0), int(c0), int(r1), int(c1)

    def conn(sprite: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(r, c) for (r, c) in sprite if int(g[int(r), int(c)]) == int(ctrl)}

    def norm(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
        rs = [r for r, _c in conn_pts]
        cs = [c for _r, c in conn_pts]
        r0, c0 = int(min(rs)), int(min(cs))
        shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
        return shape, (int(r0), int(c0))

    connA = conn(A)
    connB = conn(B)
    assert connA and connB
    shpA, _ = norm(connA)
    shpB, _ = norm(connB)
    assert shpA == shpB

    r0a, c0a, _r1a, _c1a = bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = bbox_of_pts(B)
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

    _sp, (pr0, pc0) = norm(connP)
    _sd, (dr0, dc0) = norm(connD)
    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    pts_all = base | plug_shift
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = bbox_coords(coords_all)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))

    base_coords = np.array(list(base), dtype=int)
    br0, bc0 = int(base_coords[:, 0].min()), int(base_coords[:, 1].min())
    assert rr0 == br0 and cc0 == bc0
    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)

    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(g[int(r), int(c)])

    # D13 change: after docking, plug may overlap the base body as well.
    # Rendering order defines the priority: plug overwrites base.
    for r, c in plug_shift:
        val = int(g[int(r - tr), int(c - tc)])
        out[int(r - rr0), int(c - cc0)] = int(val)

    out[out == int(ctrl)] = int(obj)
    nz = np.argwhere(out != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _add_h(mask: np.ndarray, r: int, c0: int, c1: int) -> None:
    c0, c1 = sorted((int(c0), int(c1)))
    mask[int(r), c0 : c1 + 1] = True


def _add_v(mask: np.ndarray, c: int, r0: int, r1: int) -> None:
    r0, r1 = sorted((int(r0), int(r1)))
    mask[r0 : r1 + 1, int(c)] = True


def _crop(grid: np.ndarray, bg: int) -> np.ndarray | None:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _render_assembled(base_grid: np.ndarray, plug_grid: np.ndarray, bg: int, obj: int, ctrl: int) -> np.ndarray:
    out = np.array(base_grid, copy=True)
    mask = plug_grid != int(bg)
    out[mask] = plug_grid[mask]
    out[out == int(ctrl)] = int(obj)
    nz = np.argwhere(out != int(bg))
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _interaction_metrics(base_grid: np.ndarray, plug_grid: np.ndarray, conn_row: int) -> dict[str, int]:
    bg, obj = 1, 4
    base_obj = base_grid == int(obj)
    plug_fg = plug_grid != int(bg)
    deep_rows = np.arange(base_grid.shape[0])[:, None] > int(conn_row)
    seam_rows = np.arange(base_grid.shape[0])[:, None] == int(conn_row)
    deep_overlap = int(np.sum(plug_fg & base_obj & deep_rows))
    seam_overlap = int(np.sum(plug_fg & seam_rows))
    plug_only_below = int(np.sum(plug_fg & (~base_obj) & deep_rows))
    return {
        "deep_overlap": deep_overlap,
        "seam_overlap": seam_overlap,
        "plug_only_below": plug_only_below,
    }


def _build_base_and_plug():
    bg, obj, ctrl = 1, 4, 7

    out_h = _rng.randint(5, 8)
    out_w = _rng.randint(5, 10)
    conn_len = _rng.randint(1, min(3, out_w - 2))

    conn_row_max = min(3, out_h - 4)
    if conn_row_max < 1:
        return None
    conn_row = _rng.randint(1, conn_row_max)

    right_wall_min = max(conn_len + 2, 3)
    if right_wall_min > out_w - 1:
        return None
    right_wall = _rng.randint(right_wall_min, out_w - 1)

    conn_c0_min = 1
    conn_c0_max = right_wall - conn_len - 1
    if conn_c0_max < conn_c0_min:
        return None
    conn_c0 = _rng.randint(conn_c0_min, conn_c0_max)
    conn_cols = list(range(conn_c0, conn_c0 + conn_len))

    base_bottom = _rng.randint(conn_row + 2, out_h - 1)
    roof_end = _rng.randint(right_wall, out_w - 1)

    base_mask = np.zeros((out_h, out_w), dtype=bool)
    _add_h(base_mask, 0, 0, roof_end)
    _add_v(base_mask, 0, 0, base_bottom)
    _add_v(base_mask, right_wall, 0, base_bottom)
    _add_h(base_mask, base_bottom, 0, right_wall)
    for c in conn_cols:
        _add_v(base_mask, c, 0, conn_row)

    rib_rows = [conn_row + 2]
    if base_bottom - conn_row >= 4 and _rng.random() < 0.6:
        rib_rows.append(_rng.randint(conn_row + 3, base_bottom - 1))
    rib_rows = sorted(set(rib_rows))
    for rr in rib_rows:
        rib_end = _rng.randint(conn_c0 + conn_len - 1, right_wall)
        _add_h(base_mask, rr, 0, rib_end)

    if base_bottom - 1 > conn_row and _rng.random() < 0.5:
        pillar_col = _rng.randint(conn_c0 + conn_len, right_wall)
        _add_v(base_mask, pillar_col, conn_row + 2, base_bottom)

    if roof_end > right_wall and _rng.random() < 0.5:
        _add_v(base_mask, roof_end, 0, _rng.randint(1, min(base_bottom, out_h - 1)))

    for c in conn_cols:
        if conn_row + 1 < out_h:
            base_mask[conn_row + 1, c] = False

    base_grid = np.full((out_h, out_w), int(bg), dtype=int)
    base_grid[base_mask] = int(obj)
    for c in conn_cols:
        base_grid[conn_row, c] = int(ctrl)

    if not is_contiguous(base_grid != int(bg), background=0, connectivity=8):
        return None

    plug_mask = np.zeros((out_h, out_w), dtype=bool)
    prow = conn_row - 1
    stem_bottom = _rng.randint(conn_row + 3, out_h - 1)

    for c in conn_cols:
        plug_mask[prow, c] = True
        plug_mask[conn_row, c] = True
        plug_mask[conn_row + 1 : stem_bottom + 1, c] = True

    branch_rows = {conn_row + 1, stem_bottom}
    if stem_bottom - conn_row >= 2:
        branch_rows.add(_rng.randint(conn_row + 1, stem_bottom))
    for rr in sorted(branch_rows):
        left = _rng.randint(0, min(2, conn_c0))
        right = _rng.randint(0, min(2, out_w - 1 - (conn_c0 + conn_len - 1)))
        _add_h(plug_mask, rr, conn_c0 - left, conn_c0 + conn_len - 1 + right)

    if _rng.random() < 0.6:
        side = -1 if conn_c0 > 0 and (_rng.random() < 0.5 or conn_c0 + conn_len >= out_w - 1) else 1
        nub_row = _rng.randint(conn_row + 1, stem_bottom)
        if side < 0:
            nub_c0 = max(0, conn_c0 - _rng.randint(1, min(2, conn_c0)))
            _add_h(plug_mask, nub_row, nub_c0, conn_c0)
        else:
            nub_c1 = min(out_w - 1, conn_c0 + conn_len - 1 + _rng.randint(1, min(2, out_w - conn_c0 - conn_len)))
            _add_h(plug_mask, nub_row, conn_c0 + conn_len - 1, nub_c1)

    if np.any(plug_mask[:prow, :]):
        return None

    plug_grid = np.full((out_h, out_w), int(bg), dtype=int)
    plug_grid[plug_mask] = int(obj)
    for c in conn_cols:
        plug_grid[prow, c] = int(ctrl)

    if not is_contiguous(plug_grid != int(bg), background=0, connectivity=8):
        return None

    for c in conn_cols:
        if plug_grid[conn_row, c] != int(obj):
            return None

    metrics = _interaction_metrics(base_grid, plug_grid, conn_row)
    if metrics["deep_overlap"] <= 0:
        return None
    if metrics["plug_only_below"] <= 0:
        return None

    base_sprite = _crop(base_grid, bg)
    plug_sprite = _crop(plug_grid, bg)
    expected = _render_assembled(base_grid, plug_grid, bg, obj, ctrl)
    if base_sprite is None or plug_sprite is None:
        return None
    if int(np.sum(base_sprite != int(bg))) < 5 or int(np.sum(plug_sprite != int(bg))) < 5:
        return None

    return base_sprite, plug_sprite, expected, metrics


def _place_input(base_sprite: np.ndarray, plug_sprite: np.ndarray) -> np.ndarray | None:
    bg = 1
    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape

    layouts = []
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

    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=bg)
    paste(canvas, plug_sprite, pr, pc, transparent=bg)
    lab, n = label_mask(canvas != int(bg), connectivity=8)
    if int(n) != 2:
        return None
    return canvas


def _gen_attempt():
    built = _build_base_and_plug()
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


def generate_input():
    for _ in range(400):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 400 attempts")
