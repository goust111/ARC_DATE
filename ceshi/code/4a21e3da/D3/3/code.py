# Variant of 4a21e3da
# Dimension: D3 — 源对象类型 (src)
# Change: control_pixels — 网格边缘的1-2个红色控制像素作为射线源... → 箭头形状指示方向

"""
Design analysis for the D3 variant
==================================

a) Concrete visual change and behavior trace
- The original task used 1 or 2 isolated red border pixels as ray emitters. In this variant,
  each emitter is replaced by a 3-cell L-shaped arrow of the same control color.
- Each arrow is a 2x2 box with one missing corner, touching exactly one grid edge:
  - top edge / ray downward:    {(0,s), (0,s+1), (1,s)}      shape `┌`
  - bottom edge / ray upward:   {(H-1,s-1), (H-1,s), (H-2,s)} shape `┘`
  - left edge / ray rightward:  {(s-1,0), (s,0), (s,1)}      shape `└`
  - right edge / ray leftward:  {(s,W-2), (s,W-1), (s+1,W-1)} shape `┐`
- The launch row/col is still a single split line exactly as in the original task; the arrow only
  changes how that split line is encoded. Example:
  - Original: top control pixel at (0, 9) means c_split = 9.
  - Variant: ctrl cells {(0,9), (0,10), (1,9)} must first be decoded as a top arrow,
    which still implies c_split = 9. The later ray drawing and piece motion are unchanged.

b) Complexity calibration
- The variant adds one inference step but keeps only four arrow orientations, one per cardinal
  direction. This is enough to make the change visible while keeping the rule learnable from
  a few examples.
- I keep the rest of the task intact: same grid sizes, same sprite generation, same split-line
  semantics, same kept/vanishing regions. That avoids accidental extra dimensions.

c) Audit of original solver assumptions that break here
- `ctrl_pts = np.argwhere(g == ctrl)` was valid only when every control-colored cell was an
  independent emitter. In the variant, that would incorrectly create three emitters per arrow.
- The original edge-detection loop assumed every ctrl cell directly determines a split line.
  Now only one border-aligned cell within each L-shape is the launch anchor.
- The original output restoration only needed to redraw isolated control pixels. Here the full
  3-cell arrow must persist in output, so the solver must remember all cells in each component.

d) Placement legality constraints for the generator
- Every arrow is a 4-connected ctrl component of size exactly 3, bbox exactly 2x2, and touches
  exactly one edge. Generator avoids corners so no arrow touches two edges.
- Split coordinates still lie near the center (40-60%) and the sprite still stays at least
  4 cells away from the border, so arrows never overlap the sprite or split-line foreground.
- When two arrows exist, they remain perpendicular exactly as in the original task:
  one vertical split (top or bottom) and one horizontal split (left or right).

e) Visual observability
- The change is directly visible in input and output: the control marker is now a small corner
  arrow instead of a dot, and its orientation predicts where the red ray will appear in output.
- Because the output ray reveals the launch row/col, a human can align each arrow shape with the
  direction it controls across examples.

f) Visual cleanliness
- Arrows are only 3 cells and live on the border strip plus one adjacent interior cell, while
  the sprite keeps a 4-cell margin. This keeps the control marker visually separate from the
  orange object and from the corner-slid output pieces.

g) Color inference
- No new color logic is introduced. The task still uses only bg, fg, ctrl. The variant changes
  shape encoding only, which is fully visible and does not rely on hidden color ordering.

New solver assumptions introduced by this variant
-------------------------------------------------
- All ctrl-colored cells in the input belong to 1 or 2 arrow components, each size 3.
- Allowed normalized arrow shapes are exactly the four listed above.
- A vertical arrow uniquely determines one split column; a horizontal arrow uniquely determines
  one split row. The generator guarantees there is at most one of each.
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


TOP_ARROW = frozenset({(0, 0), (0, 1), (1, 0)})
RIGHT_ARROW = frozenset({(0, 0), (0, 1), (1, 1)})
LEFT_ARROW = frozenset({(0, 0), (1, 0), (1, 1)})
BOTTOM_ARROW = frozenset({(0, 1), (1, 0), (1, 1)})


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    comps: list[list[tuple[int, int]]] = []
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((rr, cc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _decode_arrow(comp: list[tuple[int, int]], H: int, W: int) -> dict[str, object]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    norm = frozenset((r - rmin, c - cmin) for r, c in comp)
    touches = []
    if rmin == 0:
        touches.append("top")
    if rmax == H - 1:
        touches.append("bottom")
    if cmin == 0:
        touches.append("left")
    if cmax == W - 1:
        touches.append("right")
    assert len(comp) == 3 and (rmax - rmin) == 1 and (cmax - cmin) == 1
    assert len(touches) == 1

    edge = touches[0]
    if norm == TOP_ARROW:
        assert edge == "top"
        launch = (0, cmin)
    elif norm == BOTTOM_ARROW:
        assert edge == "bottom"
        launch = (H - 1, cmax)
    elif norm == LEFT_ARROW:
        assert edge == "left"
        launch = (rmax, 0)
    elif norm == RIGHT_ARROW:
        assert edge == "right"
        launch = (rmin, W - 1)
    else:
        raise AssertionError("Unexpected control-shape orientation")

    return {"edge": edge, "launch": launch, "cells": tuple(sorted(comp))}


def _find_arrows(g: np.ndarray, ctrl: int) -> list[dict[str, object]]:
    comps = _components4(g == int(ctrl))
    arrows = [_decode_arrow(comp, *g.shape) for comp in comps]
    arrows.sort(key=lambda a: (str(a["edge"]), tuple(a["launch"])))
    return arrows


def _paint_ray(scan_grid: np.ndarray, out: np.ndarray, edge: str, launch: tuple[int, int], fg: int, ctrl: int, bg: int) -> None:
    H, W = scan_grid.shape
    r0, c0 = launch
    if edge == "top":
        rows = [int(rr) for rr in np.flatnonzero(scan_grid[:, c0] == int(fg))]
        if rows:
            stop = max(rows)
            for rr in range(0, stop + 1):
                if int(scan_grid[rr, c0]) == int(bg):
                    out[rr, c0] = int(ctrl)
    elif edge == "bottom":
        rows = [int(rr) for rr in np.flatnonzero(scan_grid[:, c0] == int(fg))]
        if rows:
            stop = min(rows)
            for rr in range(H - 1, stop - 1, -1):
                if int(scan_grid[rr, c0]) == int(bg):
                    out[rr, c0] = int(ctrl)
    elif edge == "left":
        cols = [int(cc) for cc in np.flatnonzero(scan_grid[r0, :] == int(fg))]
        if cols:
            stop = max(cols)
            for cc in range(0, stop + 1):
                if int(scan_grid[r0, cc]) == int(bg):
                    out[r0, cc] = int(ctrl)
    elif edge == "right":
        cols = [int(cc) for cc in np.flatnonzero(scan_grid[r0, :] == int(fg))]
        if cols:
            stop = min(cols)
            for cc in range(W - 1, stop - 1, -1):
                if int(scan_grid[r0, cc]) == int(bg):
                    out[r0, cc] = int(ctrl)
    else:
        raise ValueError(edge)


def _make_arrow(edge: str, split: int, H: int, W: int) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    if edge == "top":
        return {(0, split), (0, split + 1), (1, split)}, (0, split)
    if edge == "bottom":
        return {(H - 1, split - 1), (H - 1, split), (H - 2, split)}, (H - 1, split)
    if edge == "left":
        return {(split - 1, 0), (split, 0), (split, 1)}, (split, 0)
    if edge == "right":
        return {(split, W - 2), (split, W - 1), (split + 1, W - 1)}, (split, W - 1)
    raise ValueError(edge)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut a sprite along border-arrow rays and slide pieces to grid corners.

    Input structure:
    - bg, fg (main object, ~centered), ctrl (1-2 L-shaped arrows on grid edges).

    Rule:
    - Decode each 3-cell ctrl component as an inward-pointing arrow and recover its launch row/col.
    - Draw rays inward from those decoded launch lines, painting only bg cells up to the farthest
      fg pixel on the same row/col (intermittent).
    - Rays partition the grid; fg on the split line stays in place.
    - Cut pieces in regions touching an arrow's border edge slide to the corresponding corner;
      others vanish.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    nonbg = colors_present(g, exclude={bg})
    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(int(bg))
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != int(ctrl)))

    arrows = _find_arrows(g, ctrl)

    E: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for arrow in arrows:
        edge = str(arrow["edge"])
        launch_r, launch_c = tuple(arrow["launch"])
        E.add(edge)
        if edge in ("top", "bottom"):
            assert c_split is None or c_split == int(launch_c)
            c_split = int(launch_c)
        else:
            assert r_split is None or r_split == int(launch_r)
            r_split = int(launch_r)

    out = np.full((H, W), int(bg), dtype=int)

    for arrow in arrows:
        _paint_ray(g, out, str(arrow["edge"]), tuple(arrow["launch"]), fg, ctrl, bg)

    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[int(r), int(c)] = int(ctrl)

    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (c_split is not None and c == int(c_split)) or (r_split is not None and r == int(r_split)):
            out[r, c] = int(fg)

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (c_split is not None and c == int(c_split)) or (r_split is not None and r == int(r_split)):
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if r < int(r_split) else "B"
        if c_split is not None:
            vs = "L" if c < int(c_split) else "R"
        regions.setdefault((hs, vs), []).append((r, c))

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
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        rmin, rmax = int(min(rs)), int(max(rs))
        cmin, cmax = int(min(cs)), int(max(cs))
        dr = int(0 - rmin) if dh == "T" else int((H - 1) - rmax)
        dc = int(0 - cmin) if dv == "L" else int((W - 1) - cmax)

        for r, c in pts:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(fg)

    return out


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    for _ in range(target_px - 1):
        neighbors = set()
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in cells:
                    neighbors.add((nr, nc))
        valid = [
            (nr, nc)
            for nr, nc in neighbors
            if max(r for r, _ in cells | {(nr, nc)}) - min(r for r, _ in cells | {(nr, nc)}) + 1 <= max_h
            and max(c for _, c in cells | {(nr, nc)}) - min(c for _, c in cells | {(nr, nc)}) + 1 <= max_w
        ]
        if not valid:
            break
        cells.add(random.choice(valid))
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return frozenset((r - min(rs), c - min(cs)) for r, c in cells)


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2

    num_ctrl = random.choices([1, 2], weights=[1, 4])[0]
    if num_ctrl == 1:
        ctrl_edges = [random.choice(["top", "bottom", "left", "right"])]
    else:
        ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]

    split_lo_w = max(1, int(W * 0.4))
    split_hi_w = min(W - 2, int(W * 0.6))
    split_lo_h = max(1, int(H * 0.4))
    split_hi_h = min(H - 2, int(H * 0.6))

    c_split = r_split = None
    arrows: list[dict[str, object]] = []
    E = set(ctrl_edges)
    has_v = has_h = False

    for edge in ctrl_edges:
        if edge in ("top", "bottom"):
            c_split = random.randint(split_lo_w, split_hi_w)
            cells, launch = _make_arrow(edge, c_split, H, W)
            arrows.append({"edge": edge, "cells": cells, "launch": launch})
            has_v = True
        else:
            r_split = random.randint(split_lo_h, split_hi_h)
            cells, launch = _make_arrow(edge, r_split, H, W)
            arrows.append({"edge": edge, "cells": cells, "launch": launch})
            has_h = True

    def pick_dest(hs: str, vs: str) -> tuple[str, str]:
        if r_split is not None:
            dh = hs
        elif "top" in E and "bottom" not in E:
            dh = "T"
        elif "bottom" in E and "top" not in E:
            dh = "B"
        else:
            dh = "T"
        if c_split is not None:
            dv = vs
        elif "left" in E and "right" not in E:
            dv = "L"
        elif "right" in E and "left" not in E:
            dv = "R"
        else:
            dv = "L"
        return dh, dv

    def touches_E(hs: str, vs: str) -> bool:
        te = set()
        if hs in ("T", "A"):
            te.add("top")
        if hs in ("B", "A"):
            te.add("bottom")
        if vs in ("L", "A"):
            te.add("left")
        if vs in ("R", "A"):
            te.add("right")
        return bool(te & E)

    if has_h and has_v:
        all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    elif has_v:
        all_regions = [("A", "L"), ("A", "R")]
    else:
        all_regions = [("T", "A"), ("B", "A")]

    kept = [reg for reg in all_regions if touches_E(*reg)]
    vanish = [reg for reg in all_regions if reg not in kept]

    sprite_h = random.randint(5, 8)
    sprite_w = random.randint(8, 10)

    if has_v:
        frac = random.uniform(0.4, 0.6)
        sprite_c0 = c_split - int(frac * sprite_w)
        sprite_c0 = max(4, min(sprite_c0, W - 5 - sprite_w + 1))
        if sprite_c0 > W - 5 - sprite_w + 1:
            return None
        if c_split <= sprite_c0 or c_split >= sprite_c0 + sprite_w - 1:
            return None
    else:
        hi = W - 5 - sprite_w + 1
        if hi < 4:
            return None
        sprite_c0 = random.randint(4, hi)

    if has_h:
        frac = random.uniform(0.4, 0.6)
        sprite_r0 = r_split - int(frac * sprite_h)
        sprite_r0 = max(4, min(sprite_r0, H - 5 - sprite_h + 1))
        if sprite_r0 > H - 5 - sprite_h + 1:
            return None
        if r_split <= sprite_r0 or r_split >= sprite_r0 + sprite_h - 1:
            return None
    else:
        hi = H - 5 - sprite_h + 1
        if hi < 4:
            return None
        sprite_r0 = random.randint(4, hi)

    sprite_r1 = sprite_r0 + sprite_h - 1
    sprite_c1 = sprite_c0 + sprite_w - 1

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = sprite_r0 if hs in ("T", "A") else r_split + 1
        r1 = (r_split - 1 if hs == "T" else sprite_r1) if has_h else sprite_r1
        c0 = sprite_c0 if vs in ("L", "A") else c_split + 1
        c1 = (c_split - 1 if vs == "L" else sprite_c1) if has_v else sprite_c1
        return r0, r1, c0, c1

    for reg in kept + vanish:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None

    split_pixels: set[tuple[int, int]] = set()
    if has_v:
        n = random.randint(2, 4)
        rows = [r for r in range(sprite_r0, sprite_r1 + 1) if r != r_split]
        if len(rows) < n:
            return None
        for r in random.sample(rows, n):
            split_pixels.add((r, c_split))
    if has_h:
        n = random.randint(2, 4)
        cols = [c for c in range(sprite_c0, sprite_c1 + 1) if c != c_split]
        if len(cols) < n:
            return None
        for c in random.sample(cols, n):
            split_pixels.add((r_split, c))

    n_split = len(split_pixels)
    area = sprite_h * sprite_w
    target_density = random.uniform(0.39, 0.60)
    target_total = int(target_density * area)

    n_kept = len(kept)
    n_vanish = len(vanish)
    budget_pieces = target_total - n_split
    if n_vanish > 0:
        vanish_budget = random.randint(3, min(7, max(3, budget_pieces // 3)))
        budget_pieces -= vanish_budget
    else:
        vanish_budget = 0

    if budget_pieces < n_kept * 3:
        return None

    piece_budgets: list[int] = []
    remaining = budget_pieces
    for i in range(n_kept):
        r0, r1, c0, c1 = sub_bbox(*kept[i])
        max_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if i == n_kept - 1:
            px = remaining
        else:
            avg = remaining // (n_kept - i)
            lo_px = max(3, avg - 2)
            hi_px = min(max_area, avg + 2)
            if lo_px > hi_px:
                return None
            px = random.randint(lo_px, hi_px)
        px = max(3, min(px, max_area, remaining - (n_kept - i - 1) * 3))
        piece_budgets.append(px)
        remaining -= px

    if remaining != 0 or any(b < 3 for b in piece_budgets):
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}

    for idx, reg in enumerate(kept):
        hs, vs = reg
        dh, dv = pick_dest(hs, vs)
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = r1 - r0 + 1
        sub_w = c1 - c0 + 1
        target_px = piece_budgets[idx]

        piece = _grow_piece(sub_h, sub_w, target_px)
        if len(piece) < 3:
            return None

        ph = max(r for r, _ in piece) + 1
        pw = max(c for _, c in piece) + 1

        if has_h:
            inp_r0 = (r1 - ph + 1) if hs == "T" else r0
        else:
            split_rows = sorted(r for r, c in split_pixels if c == c_split)
            if split_rows:
                mid = (split_rows[0] + split_rows[-1]) // 2
                inp_r0 = max(r0, min(mid - ph // 2, r1 - ph + 1))
            else:
                inp_r0 = r0

        if has_v:
            inp_c0 = (c1 - pw + 1) if vs == "L" else c0
        else:
            split_cols = sorted(c for r, c in split_pixels if r == r_split)
            if split_cols:
                mid = (split_cols[0] + split_cols[-1]) // 2
                inp_c0 = max(c0, min(mid - pw // 2, c1 - pw + 1))
            else:
                inp_c0 = c0

        inp_pts = frozenset((r + inp_r0, c + inp_c0) for r, c in piece)

        for r, c in inp_pts:
            if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                return None
            if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                return None
            if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                return None

        input_pieces[reg] = inp_pts

        inp_rs = [r for r, _ in inp_pts]
        inp_cs = [c for _, c in inp_pts]
        dr = -min(inp_rs) if dh == "T" else (H - 1 - max(inp_rs))
        dc = -min(inp_cs) if dv == "L" else (W - 1 - max(inp_cs))
        out_pts = frozenset((r + dr, c + dc) for r, c in inp_pts)

        for r, c in out_pts:
            if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                return None
            if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                return None
        output_pieces[reg] = out_pts

    all_inp_fg: set[tuple[int, int]] = set()
    for pts in input_pieces.values():
        all_inp_fg |= set(pts)
    all_inp_fg |= split_pixels
    if not all_inp_fg:
        return None

    fg_arr = np.array(list(all_inp_fg))
    ah = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    aw = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if ah < 5 or ah > 8 or aw < 8 or aw > 10:
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in all_inp_fg:
        mask[r, c] = 1
    if not is_contiguous(mask):
        return None

    vanish_fg: set[tuple[int, int]] = set()
    if vanish:
        for vreg in vanish:
            vhs, vvs = vreg
            vr0, vr1, vc0, vc1 = sub_bbox(vhs, vvs)
            vh, vw = vr1 - vr0 + 1, vc1 - vc0 + 1
            if vh < 1 or vw < 1:
                return None
            max_px = min(vanish_budget, vh * vw)
            if max_px < 3:
                return None
            piece = _grow_piece(vh, vw, max_px)
            if len(piece) < 3:
                return None
            pph = max(r for r, _ in piece) + 1
            ppw = max(c for _, c in piece) + 1
            vr_off = (vr1 - pph + 1) if vhs == "T" else vr0
            vc_off = (vc1 - ppw + 1) if vvs == "L" else vc0
            vpts = {(r + vr_off, c + vc_off) for r, c in piece}
            for r, c in vpts:
                if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                    return None
                if (r, c) in all_inp_fg:
                    return None
            vanish_fg = vpts

    combined = all_inp_fg | vanish_fg
    mask2 = np.zeros((H, W), dtype=int)
    for r, c in combined:
        mask2[r, c] = 1
    if not is_contiguous(mask2):
        return None

    inp = np.full((H, W), bg, dtype=int)
    for r, c in combined:
        inp[r, c] = fg
    for arrow in arrows:
        for r, c in arrow["cells"]:
            inp[r, c] = ctrl_color

    out = np.full((H, W), bg, dtype=int)
    for pts in output_pieces.values():
        for r, c in pts:
            out[r, c] = fg
    for r, c in split_pixels:
        out[r, c] = fg

    for arrow in arrows:
        _paint_ray(out, out, str(arrow["edge"]), tuple(arrow["launch"]), fg, ctrl_color, bg)
    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[r, c] = ctrl_color

    fg_all = np.array(list(combined))
    final_h = int(fg_all[:, 0].max() - fg_all[:, 0].min() + 1)
    final_w = int(fg_all[:, 1].max() - fg_all[:, 1].min() + 1)
    actual_density = len(combined) / (final_h * final_w)
    if actual_density < 0.38 or actual_density > 0.62:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
