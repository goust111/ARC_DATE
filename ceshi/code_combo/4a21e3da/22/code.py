# Combo variant of 4a21e3da (index 22)
# combo_spec: D4/2+D13/1
# Name: 射线到外接矩形停止+面积筛选

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
  Base task:
    - Border control pixels define one vertical split and/or one horizontal
      split.
    - Original rays are drawn from the border to the farthest foreground pixel
      on that control line.
    - The split partitions the centered sprite into 2 or 4 rectangles.
    - Original keep rule: regions touching a control edge survive.

  Combo behavior:
    - D4/2 changes only the visible ray extent: the ray stops at the input
      sprite bounding-box edge, not at the farthest foreground pixel on the
      control line.
    - D13/1 changes the keep rule: after partitioning by the split line(s),
      keep the largest N rectangular regions, where N is the number of control
      pixels.

  Small trace:
    - Suppose a top control defines column c_split and the input sprite bbox is
      rows 5..10. Split-line foreground exists at rows 6, 8, 9 on c_split.
    - Original ray would paint ctrl on bg cells from row 0 down to row 9.
    - This combo paints ctrl only from row 0 down to row 5, because row 5 is
      the bbox entry row.
    - If a left control also exists, and the four partition rectangles are
      TL=2x3, TR=2x5, BL=3x3, BR=3x5, the kept set is the largest N=2 areas:
      BR and TR. The original edge-touch rule for {top,left} would instead keep
      TL, TR, BL.
    - The solver must therefore infer both: shorter rays are governed by the
      whole sprite bbox, and the kept pieces are chosen by partition area.

  Key interaction:
    - The shorter rays no longer visually reach the deep split-line pixels, so
      the full partition geometry is not directly exposed by the ctrl color.
      Area selection still depends on the bbox-defined partition, which creates
      genuine interaction between D4 and D13 instead of two unrelated edits.

b) Complexity calibration
  - Every region is populated, so area ranking is observable.
  - Region pieces span their whole partition bbox via sparse L-shaped pieces,
    making width and height legible without equating fg-count with area.
  - The generator rejects ties at the keep/drop boundary and prefers 2-control
    cases, because those give the clearest multi-way comparisons.
  - The generator also rejects cases where bbox-stop rays would look identical
    to the original farthest-stop rays, so the D4 change stays visible.

c) Implicit assumptions that change from the seed
  - The original ray drawer assumed "stop at farthest fg on the line". That is
    obsolete. The combo solver instead computes the input fg bbox first and uses
    its edge as the stop coordinate.
  - The original keep decision used `touches control edge`. That is obsolete.
    The combo solver must rank partition rectangle areas and keep only N.
  - The original solver did not need strict region-area ordering. The generator
    now enforces strict ordering around the keep/drop threshold.
  - The solver now assumes the fg bbox is meaningful for ray termination. The
    generator guarantees a single centered connected sprite with margin >= 4,
    so the bbox is stable and visually recoverable.

d) Placement legality
  - Grid remains 18-20 square with colors {1,2,7}.
  - Control pixels remain on distinct edges; with 2 controls they are
    perpendicular, so there is at most one horizontal and one vertical split.
  - The sprite bbox remains 5-8 high and 8-10 wide with margin >= 4.
  - Each region piece is placed strictly inside its own partition rectangle and
    never crosses a split line.
  - After sliding, surviving pieces still fit in their destination corner and
    do not overflow.

e) Visual observability
  - Humans can read partition size because each partition contains a sparse
    shape whose bbox equals the full partition rectangle.
  - Rays clearly stop at the outer bbox edge rather than at the deepest split
    pixel; the difference is visible as extra blank gap inside the bbox.
  - The selected regions differ from the original edge-touch rule, so the area
    filter is actually needed.

f) Visual cleanliness
  - Only three colors are used.
  - Each region contains one connected sparse piece; split-line fg is limited
    to 2-4 cells per split.
  - The ctrl rays remain single-pixel straight lines and stay outside the
    sprite interior, so clutter is limited.

g) Color inference
  - No color remapping is introduced. Colors keep the same semantic roles as in
    the seed task.
"""


REGION_ORDER = {
    ("T", "L"): 0,
    ("T", "R"): 1,
    ("B", "L"): 2,
    ("B", "R"): 3,
    ("A", "L"): 4,
    ("A", "R"): 5,
    ("T", "A"): 6,
    ("B", "A"): 7,
}


def _all_regions(has_h: bool, has_v: bool) -> list[tuple[str, str]]:
    if has_h and has_v:
        return [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    if has_v:
        return [("A", "L"), ("A", "R")]
    return [("T", "A"), ("B", "A")]


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


def _pick_dest(
    hs: str,
    vs: str,
    *,
    r_split: int | None,
    c_split: int | None,
    edges: set[str],
) -> tuple[str, str]:
    if r_split is not None:
        dh = hs
    elif ("top" in edges) and ("bottom" not in edges):
        dh = "T"
    elif ("bottom" in edges) and ("top" not in edges):
        dh = "B"
    else:
        dh = "T"

    if c_split is not None:
        dv = vs
    elif ("left" in edges) and ("right" not in edges):
        dv = "L"
    elif ("right" in edges) and ("left" not in edges):
        dv = "R"
    else:
        dv = "L"
    return dh, dv


def _region_area(
    hs: str,
    vs: str,
    *,
    sprite_r0: int,
    sprite_r1: int,
    sprite_c0: int,
    sprite_c1: int,
    r_split: int | None,
    c_split: int | None,
) -> int:
    if r_split is None:
        h = sprite_r1 - sprite_r0 + 1
    elif hs == "T":
        h = r_split - sprite_r0
    else:
        h = sprite_r1 - r_split

    if c_split is None:
        w = sprite_c1 - sprite_c0 + 1
    elif vs == "L":
        w = c_split - sprite_c0
    else:
        w = sprite_c1 - c_split
    return int(h * w)


def _select_largest_regions(
    present_regions: list[tuple[str, str]],
    *,
    num_keep: int,
    sprite_r0: int,
    sprite_r1: int,
    sprite_c0: int,
    sprite_c1: int,
    r_split: int | None,
    c_split: int | None,
) -> set[tuple[str, str]]:
    scored = []
    for hs, vs in present_regions:
        area = _region_area(
            hs,
            vs,
            sprite_r0=sprite_r0,
            sprite_r1=sprite_r1,
            sprite_c0=sprite_c0,
            sprite_c1=sprite_c1,
            r_split=r_split,
            c_split=c_split,
        )
        scored.append((-area, REGION_ORDER[(hs, vs)], (hs, vs)))
    scored.sort()
    return {reg for _, _, reg in scored[: min(num_keep, len(scored))]}


def _infer_structure(
    g: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], set[str], int | None, int | None]:
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})

    border_cols = (
        set(map(int, g[0, :]))
        | set(map(int, g[-1, :]))
        | set(map(int, g[:, 0]))
        | set(map(int, g[:, -1]))
    )
    border_cols.discard(bg)
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != ctrl))

    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == ctrl)]
    edges: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for r, c in ctrl_pts:
        if r == 0:
            edges.add("top")
            c_split = c
        elif r == g.shape[0] - 1:
            edges.add("bottom")
            c_split = c
        elif c == 0:
            edges.add("left")
            r_split = r
        elif c == g.shape[1] - 1:
            edges.add("right")
            r_split = r
    return bg, fg, ctrl_pts, edges, r_split, c_split


def _paint_ray_segment(
    out: np.ndarray,
    g: np.ndarray,
    *,
    fg: int,
    ctrl: int,
    bg: int,
    ctrl_pt: tuple[int, int],
    stop: int,
) -> None:
    H, W = g.shape
    r, c = ctrl_pt
    if r == 0:
        for rr in range(0, stop + 1):
            if int(g[rr, c]) == bg:
                out[rr, c] = ctrl
    elif r == H - 1:
        for rr in range(H - 1, stop - 1, -1):
            if int(g[rr, c]) == bg:
                out[rr, c] = ctrl
    elif c == 0:
        for cc in range(0, stop + 1):
            if int(g[r, cc]) == bg:
                out[r, cc] = ctrl
    elif c == W - 1:
        for cc in range(W - 1, stop - 1, -1):
            if int(g[r, cc]) == bg:
                out[r, cc] = ctrl
    out[r, c] = ctrl


def _draw_bbox_rays(
    out: np.ndarray,
    g: np.ndarray,
    ctrl_pts: list[tuple[int, int]],
    *,
    fg: int,
    ctrl: int,
    bg: int,
    sprite_r0: int,
    sprite_r1: int,
    sprite_c0: int,
    sprite_c1: int,
) -> None:
    for r, c in ctrl_pts:
        if r == 0:
            _paint_ray_segment(out, g, fg=fg, ctrl=ctrl, bg=bg, ctrl_pt=(r, c), stop=sprite_r0)
        elif r == g.shape[0] - 1:
            _paint_ray_segment(out, g, fg=fg, ctrl=ctrl, bg=bg, ctrl_pt=(r, c), stop=sprite_r1)
        elif c == 0:
            _paint_ray_segment(out, g, fg=fg, ctrl=ctrl, bg=bg, ctrl_pt=(r, c), stop=sprite_c0)
        elif c == g.shape[1] - 1:
            _paint_ray_segment(out, g, fg=fg, ctrl=ctrl, bg=bg, ctrl_pt=(r, c), stop=sprite_c1)


def _draw_farthest_rays(
    out: np.ndarray,
    g: np.ndarray,
    ctrl_pts: list[tuple[int, int]],
    *,
    fg: int,
    ctrl: int,
    bg: int,
) -> None:
    H, W = g.shape
    for r, c in ctrl_pts:
        if r == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                _paint_ray_segment(out, g, fg=fg, ctrl=ctrl, bg=bg, ctrl_pt=(r, c), stop=max(rows))
        elif r == H - 1:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                _paint_ray_segment(out, g, fg=fg, ctrl=ctrl, bg=bg, ctrl_pt=(r, c), stop=min(rows))
        elif c == 0:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                _paint_ray_segment(out, g, fg=fg, ctrl=ctrl, bg=bg, ctrl_pt=(r, c), stop=max(cols))
        elif c == W - 1:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                _paint_ray_segment(out, g, fg=fg, ctrl=ctrl, bg=bg, ctrl_pt=(r, c), stop=min(cols))


def _ray_cells(grid: np.ndarray, *, mode: str) -> set[tuple[int, int]]:
    g = np.array(grid, dtype=int)
    bg, fg, ctrl_pts, _edges, _r_split, _c_split = _infer_structure(g)
    ctrl = int(g[ctrl_pts[0]])
    fg_coords = np.argwhere(g == fg)
    sprite_r0 = int(fg_coords[:, 0].min())
    sprite_r1 = int(fg_coords[:, 0].max())
    sprite_c0 = int(fg_coords[:, 1].min())
    sprite_c1 = int(fg_coords[:, 1].max())
    canvas = np.full(g.shape, bg, dtype=int)
    if mode == "bbox":
        _draw_bbox_rays(
            canvas,
            g,
            ctrl_pts,
            fg=fg,
            ctrl=ctrl,
            bg=bg,
            sprite_r0=sprite_r0,
            sprite_r1=sprite_r1,
            sprite_c0=sprite_c0,
            sprite_c1=sprite_c1,
        )
    else:
        _draw_farthest_rays(canvas, g, ctrl_pts, fg=fg, ctrl=ctrl, bg=bg)
    return {
        (int(r), int(c))
        for r, c in np.argwhere(canvas == ctrl)
        if (int(r), int(c)) not in ctrl_pts
    }


def variant_diagnostics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify the two intended combo signals on one generated input.

    Returns:
      - ray_reduction: how many ctrl-colored ray cells disappear relative to
        the seed farthest-stop ray rule.
      - selection_diff: 1 iff largest-N selection differs from original
        edge-touch selection.
    """
    g = np.array(grid, dtype=int)
    _bg, fg, ctrl_pts, edges, r_split, c_split = _infer_structure(g)
    fg_coords = np.argwhere(g == fg)
    sprite_r0 = int(fg_coords[:, 0].min())
    sprite_r1 = int(fg_coords[:, 0].max())
    sprite_c0 = int(fg_coords[:, 1].min())
    sprite_c1 = int(fg_coords[:, 1].max())

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in fg_coords:
        r = int(r)
        c = int(c)
        if (c_split is not None and c == c_split) or (r_split is not None and r == r_split):
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if r < r_split else "B"
        if c_split is not None:
            vs = "L" if c < c_split else "R"
        regions.setdefault((hs, vs), []).append((r, c))

    selected = _select_largest_regions(
        list(regions.keys()),
        num_keep=len(ctrl_pts),
        sprite_r0=sprite_r0,
        sprite_r1=sprite_r1,
        sprite_c0=sprite_c0,
        sprite_c1=sprite_c1,
        r_split=r_split,
        c_split=c_split,
    )
    original_selected = {reg for reg in regions if _touches_edges(reg[0], reg[1]) & edges}
    return {
        "ray_reduction": len(_ray_cells(g, mode="farthest") - _ray_cells(g, mode="bbox")),
        "selection_diff": int(selected != original_selected),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
      1. Compute the input sprite bbox.
      2. Draw each control ray only up to the bbox edge on its path.
      3. Preserve foreground pixels on split line(s).
      4. Partition remaining fg by split side and keep only the largest N
         rectangles, where N is the number of control pixels.
      5. Slide each kept region to its destination corner using the seed
         translation rule; discard the rest.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, fg, ctrl_pts, edges, r_split, c_split = _infer_structure(g)
    ctrl = int(g[ctrl_pts[0]])

    fg_coords = np.argwhere(g == fg)
    sprite_r0 = int(fg_coords[:, 0].min())
    sprite_r1 = int(fg_coords[:, 0].max())
    sprite_c0 = int(fg_coords[:, 1].min())
    sprite_c1 = int(fg_coords[:, 1].max())

    out = np.full((H, W), bg, dtype=int)
    _draw_bbox_rays(
        out,
        g,
        ctrl_pts,
        fg=fg,
        ctrl=ctrl,
        bg=bg,
        sprite_r0=sprite_r0,
        sprite_r1=sprite_r1,
        sprite_c0=sprite_c0,
        sprite_c1=sprite_c1,
    )

    for r, c in fg_coords:
        r = int(r)
        c = int(c)
        if (c_split is not None and c == c_split) or (r_split is not None and r == r_split):
            out[r, c] = fg

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in fg_coords:
        r = int(r)
        c = int(c)
        if (c_split is not None and c == c_split) or (r_split is not None and r == r_split):
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if r < r_split else "B"
        if c_split is not None:
            vs = "L" if c < c_split else "R"
        regions.setdefault((hs, vs), []).append((r, c))

    selected = _select_largest_regions(
        list(regions.keys()),
        num_keep=len(ctrl_pts),
        sprite_r0=sprite_r0,
        sprite_r1=sprite_r1,
        sprite_c0=sprite_c0,
        sprite_c1=sprite_c1,
        r_split=r_split,
        c_split=c_split,
    )

    for reg, pts in regions.items():
        if reg not in selected:
            continue
        dh, dv = _pick_dest(reg[0], reg[1], r_split=r_split, c_split=c_split, edges=edges)
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        dr = -min(rs) if dh == "T" else (H - 1 - max(rs))
        dc = -min(cs) if dv == "L" else (W - 1 - max(cs))
        for r, c in pts:
            out[r + dr, c + dc] = fg

    return out


def _make_spanning_piece(
    hs: str,
    vs: str,
    *,
    sub_h: int,
    sub_w: int,
    has_h: bool,
    has_v: bool,
    edges: set[str],
) -> frozenset[tuple[int, int]]:
    if has_h:
        row_toward_split = sub_h - 1 if hs == "T" else 0
    else:
        row_toward_split = 0 if ("top" in edges and "bottom" not in edges) else sub_h - 1

    if has_v:
        col_toward_split = sub_w - 1 if vs == "L" else 0
    else:
        col_toward_split = 0 if ("left" in edges and "right" not in edges) else sub_w - 1

    cells = {(row_toward_split, c) for c in range(sub_w)} | {
        (r, col_toward_split) for r in range(sub_h)
    }

    extras = []
    for r in range(sub_h):
        for c in range(sub_w):
            if (r, c) in cells:
                continue
            if any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in cells):
                extras.append((r, c))
    random.shuffle(extras)
    for r, c in extras[: random.randint(0, min(3, len(extras)))]:
        cells.add((r, c))
    return frozenset(cells)


def _layout_regions(
    *,
    sprite_r0: int,
    sprite_c0: int,
    sprite_h: int,
    sprite_w: int,
    r_split: int | None,
    c_split: int | None,
    has_h: bool,
    has_v: bool,
) -> dict[tuple[str, str], tuple[int, int, int, int]]:
    sprite_r1 = sprite_r0 + sprite_h - 1
    sprite_c1 = sprite_c0 + sprite_w - 1

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = sprite_r0 if hs in ("T", "A") else int(r_split + 1)
        r1 = (int(r_split - 1) if hs == "T" else sprite_r1) if has_h else sprite_r1
        c0 = sprite_c0 if vs in ("L", "A") else int(c_split + 1)
        c1 = (int(c_split - 1) if vs == "L" else sprite_c1) if has_v else sprite_c1
        return r0, r1, c0, c1

    return {reg: sub_bbox(*reg) for reg in _all_regions(has_h, has_v)}


def _expected_output(
    *,
    inp: np.ndarray,
    H: int,
    W: int,
    bg: int,
    fg: int,
    ctrl_color: int,
    ctrl_pts: list[tuple[int, int]],
    edges: set[str],
    r_split: int | None,
    c_split: int | None,
    selected_regions: set[tuple[str, str]],
    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]],
    split_pixels: set[tuple[int, int]],
    sprite_r0: int,
    sprite_r1: int,
    sprite_c0: int,
    sprite_c1: int,
) -> np.ndarray:
    out = np.full((H, W), bg, dtype=int)
    for reg in selected_regions:
        pts = input_pieces[reg]
        dh, dv = _pick_dest(reg[0], reg[1], r_split=r_split, c_split=c_split, edges=edges)
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        dr = -min(rs) if dh == "T" else (H - 1 - max(rs))
        dc = -min(cs) if dv == "L" else (W - 1 - max(cs))
        for r, c in pts:
            out[r + dr, c + dc] = fg
    for r, c in split_pixels:
        out[r, c] = fg
    _draw_bbox_rays(
        out,
        inp,
        ctrl_pts,
        fg=fg,
        ctrl=ctrl_color,
        bg=bg,
        sprite_r0=sprite_r0,
        sprite_r1=sprite_r1,
        sprite_c0=sprite_c0,
        sprite_c1=sprite_c1,
    )
    return out


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(18, 20)
    bg, fg, ctrl_color = 1, 7, 2

    num_ctrl = random.choices([1, 2], weights=[1, 4])[0]
    if num_ctrl == 1:
        ctrl_edges = [random.choice(["top", "bottom", "left", "right"])]
    else:
        ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]

    edges = set(ctrl_edges)
    has_v = any(edge in {"top", "bottom"} for edge in ctrl_edges)
    has_h = any(edge in {"left", "right"} for edge in ctrl_edges)

    if has_h:
        top_h = random.randint(2, 5)
        bottom_h = random.randint(2, 5)
        sprite_h = top_h + bottom_h + 1
        if not (5 <= sprite_h <= 8):
            return None
    else:
        top_h = bottom_h = None
        sprite_h = random.randint(5, 8)

    if has_v:
        left_w = random.randint(3, 6)
        right_w = random.randint(3, 6)
        sprite_w = left_w + right_w + 1
        if not (8 <= sprite_w <= 10):
            return None
    else:
        left_w = right_w = None
        sprite_w = random.randint(8, 10)

    max_r0 = H - sprite_h - 4
    max_c0 = W - sprite_w - 4
    if max_r0 < 4 or max_c0 < 4:
        return None
    sprite_r0 = random.randint(4, max_r0)
    sprite_c0 = random.randint(4, max_c0)
    sprite_r1 = sprite_r0 + sprite_h - 1
    sprite_c1 = sprite_c0 + sprite_w - 1

    r_split = sprite_r0 + top_h if has_h else None
    c_split = sprite_c0 + left_w if has_v else None

    ctrl_pts = []
    for edge in ctrl_edges:
        if edge == "top":
            ctrl_pts.append((0, int(c_split)))
        elif edge == "bottom":
            ctrl_pts.append((H - 1, int(c_split)))
        elif edge == "left":
            ctrl_pts.append((int(r_split), 0))
        elif edge == "right":
            ctrl_pts.append((int(r_split), W - 1))

    regions = _all_regions(has_h, has_v)
    region_boxes = _layout_regions(
        sprite_r0=sprite_r0,
        sprite_c0=sprite_c0,
        sprite_h=sprite_h,
        sprite_w=sprite_w,
        r_split=r_split,
        c_split=c_split,
        has_h=has_h,
        has_v=has_v,
    )

    areas = {
        reg: _region_area(
            reg[0],
            reg[1],
            sprite_r0=sprite_r0,
            sprite_r1=sprite_r1,
            sprite_c0=sprite_c0,
            sprite_c1=sprite_c1,
            r_split=r_split,
            c_split=c_split,
        )
        for reg in regions
    }
    sorted_areas = sorted(areas.values(), reverse=True)
    if len(sorted_areas) > num_ctrl and sorted_areas[num_ctrl - 1] <= sorted_areas[num_ctrl]:
        return None
    if len(sorted_areas) == 2 and sorted_areas[0] - sorted_areas[1] < 2:
        return None
    if len(sorted_areas) > 2 and sorted_areas[num_ctrl - 1] - sorted_areas[num_ctrl] < 2:
        return None

    selected_regions = _select_largest_regions(
        regions,
        num_keep=num_ctrl,
        sprite_r0=sprite_r0,
        sprite_r1=sprite_r1,
        sprite_c0=sprite_c0,
        sprite_c1=sprite_c1,
        r_split=r_split,
        c_split=c_split,
    )
    original_regions = {reg for reg in regions if _touches_edges(reg[0], reg[1]) & edges}
    if selected_regions == original_regions:
        return None

    split_pixels: set[tuple[int, int]] = set()
    if has_v:
        rows = list(range(sprite_r0, sprite_r1 + 1))
        if has_h:
            rows.remove(int(r_split))
        if "top" in edges:
            rows = [r for r in rows if r > sprite_r0]
        if "bottom" in edges:
            rows = [r for r in rows if r < sprite_r1]
        if len(rows) < 2:
            return None
        n = random.randint(2, min(4, len(rows)))
        for r in random.sample(rows, n):
            split_pixels.add((r, int(c_split)))
    if has_h:
        cols = list(range(sprite_c0, sprite_c1 + 1))
        if has_v:
            cols.remove(int(c_split))
        if "left" in edges:
            cols = [c for c in cols if c > sprite_c0]
        if "right" in edges:
            cols = [c for c in cols if c < sprite_c1]
        if len(cols) < 2:
            return None
        n = random.randint(2, min(4, len(cols)))
        for c in random.sample(cols, n):
            split_pixels.add((int(r_split), c))

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    all_fg = set(split_pixels)
    for reg in regions:
        r0, r1, c0, c1 = region_boxes[reg]
        sub_h = r1 - r0 + 1
        sub_w = c1 - c0 + 1
        if sub_h <= 0 or sub_w <= 0:
            return None
        local_piece = _make_spanning_piece(
            reg[0],
            reg[1],
            sub_h=sub_h,
            sub_w=sub_w,
            has_h=has_h,
            has_v=has_v,
            edges=edges,
        )
        pts = frozenset((r0 + r, c0 + c) for r, c in local_piece)
        for r, c in pts:
            if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                return None
            if has_v and ((reg[1] == "L" and c >= int(c_split)) or (reg[1] == "R" and c <= int(c_split))):
                return None
            if has_h and ((reg[0] == "T" and r >= int(r_split)) or (reg[0] == "B" and r <= int(r_split))):
                return None
        input_pieces[reg] = pts
        all_fg |= set(pts)

    fg_arr = np.array(list(all_fg))
    bbox_h = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    bbox_w = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if bbox_h != sprite_h or bbox_w != sprite_w:
        return None

    density = len(all_fg) / float(sprite_h * sprite_w)
    if not (0.39 <= density <= 0.60):
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in all_fg:
        mask[r, c] = 1
    if not is_contiguous(mask):
        return None

    inp = np.full((H, W), bg, dtype=int)
    for r, c in all_fg:
        inp[r, c] = fg
    for r, c in ctrl_pts:
        inp[r, c] = ctrl_color

    bbox_ray_cells = _ray_cells(inp, mode="bbox")
    farthest_ray_cells = _ray_cells(inp, mode="farthest")
    if not (farthest_ray_cells - bbox_ray_cells):
        return None

    out = _expected_output(
        inp=inp,
        H=H,
        W=W,
        bg=bg,
        fg=fg,
        ctrl_color=ctrl_color,
        ctrl_pts=ctrl_pts,
        edges=edges,
        r_split=r_split,
        c_split=c_split,
        selected_regions=selected_regions,
        input_pieces=input_pieces,
        split_pixels=split_pixels,
        sprite_r0=sprite_r0,
        sprite_r1=sprite_r1,
        sprite_c0=sprite_c0,
        sprite_c1=sprite_c1,
    )
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 600 attempts")
