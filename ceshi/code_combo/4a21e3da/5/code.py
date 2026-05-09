# Combo variant of 4a21e3da (index 5)
# combo_spec: D2/1+D12/0
# Name: 颜色循环射线+多色精灵

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


# Design analysis for the fused variant.
#
# 1. Concrete behavior:
#    Original task has one fg color and monochrome ctrl rays. Here the sprite uses
#    2-3 foreground colors, while every split-line pixel uses one designated anchor
#    color. Rays still skip foreground and only paint background, but now their
#    painted bg cells alternate [anchor, ctrl, anchor, ctrl, ...] when read from
#    the border inward; the border control pixel itself stays ctrl.
#    Small trace: if a top ctrl at col c sees split-line fg at rows 6 and 9, then
#    original output paints bg cells on rows 1..5 and 7..8 all ctrl. New output
#    paints those same bg cells as anchor/ctrl/anchor/... while the split-line fg
#    cells stay anchor. This makes the ray visibly "belong" to the same color that
#    remains on the cut line.
#
# 2. Complexity calibration:
#    The stripe color is not chosen by hidden numeric ordering. It is the visible
#    color shared by the cut-line pixels. That keeps the rule inferable from a few
#    samples. Sprite colors are capped at 2-3 so the grid stays readable.
#
# 3. Original hidden assumptions that break:
#    - `fg = next(nonbg != ctrl)` assumed exactly one foreground color.
#    - Stage 1 wrote only `ctrl` on every ray cell.
#    - Stage 2 rewrote every surviving pixel as one flat fg color.
#    - Generator painted the whole sprite with one color.
#    The new solver therefore treats "foreground" as every non-bg/non-ctrl pixel,
#    infers the anchor color from split-line pixels, and always copies original
#    pixel colors during split-line preservation and region translation.
#
# 4. Placement legality:
#    Geometry still follows the original output-first construction, so pieces stay
#    inside their legal region before and after translation. The generator also
#    keeps the sprite 4 cells away from the border and verifies contiguity.
#
# 5. Visual observability:
#    Every sample contains both at least one multicolor moved piece and at least
#    one anchor-colored stripe cell on a ray. This prevents degeneration back to
#    the original single-color / single-ray semantics.


SPRITE_PALETTE = [3, 4, 5, 6, 7, 8, 9]


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


def _pick_dest(hs: str, vs: str, edges: set[str], r_split: int | None, c_split: int | None) -> tuple[str, str]:
    if r_split is not None:
        dh = hs
    elif "top" in edges and "bottom" not in edges:
        dh = "T"
    elif "bottom" in edges and "top" not in edges:
        dh = "B"
    else:
        dh = "T"

    if c_split is not None:
        dv = vs
    elif "left" in edges and "right" not in edges:
        dv = "L"
    elif "right" in edges and "left" not in edges:
        dv = "R"
    else:
        dv = "L"
    return dh, dv


def _infer_structure(grid: np.ndarray) -> tuple[int, int, np.ndarray, list[tuple[int, int]], set[str], int | None, int | None, int]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    border_vals = [int(x) for x in g[0, :]] + [int(x) for x in g[h - 1, :]] + [int(x) for x in g[:, 0]] + [int(x) for x in g[:, w - 1]]
    border_nonbg = [x for x in border_vals if x != bg]
    assert border_nonbg
    ctrl = int(Counter(border_nonbg).most_common(1)[0][0])
    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == ctrl)]

    edges: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for r, c in ctrl_pts:
        if r == 0:
            edges.add("top")
            c_split = c
        elif r == h - 1:
            edges.add("bottom")
            c_split = c
        elif c == 0:
            edges.add("left")
            r_split = r
        elif c == w - 1:
            edges.add("right")
            r_split = r

    fg_mask = (g != bg) & (g != ctrl)
    split_mask = np.zeros_like(fg_mask, dtype=bool)
    if c_split is not None:
        split_mask[:, c_split] |= fg_mask[:, c_split]
    if r_split is not None:
        split_mask[r_split, :] |= fg_mask[r_split, :]

    split_cols = [int(g[r, c]) for r, c in np.argwhere(split_mask)]
    if split_cols:
        anchor = int(Counter(split_cols).most_common(1)[0][0])
    else:
        fg_cols = [int(x) for x in g[fg_mask].tolist()]
        assert fg_cols
        anchor = int(Counter(fg_cols).most_common(1)[0][0])

    return bg, ctrl, fg_mask, ctrl_pts, edges, c_split, r_split, anchor


def _paint_striped_ray(
    out: np.ndarray,
    src: np.ndarray,
    bg: int,
    ctrl: int,
    anchor: int,
    cells: list[tuple[int, int]],
) -> None:
    painted = 0
    for r, c in cells:
        if int(src[r, c]) != bg:
            continue
        out[r, c] = anchor if painted % 2 == 0 else ctrl
        painted += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut a multicolor sprite along border-control rays and slide kept pieces to corners.

    Fused rule:
    - The sprite uses 2-3 foreground colors.
    - Foreground pixels on split lines remain in place and keep their colors.
    - Rays still skip foreground, but painted bg cells alternate anchor/ctrl colors.
    - Non-split sprite regions touching a control edge slide to the corresponding
      corner by translation only, so every pixel keeps its original color.
    - Regions not touching any control edge vanish.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg, ctrl, fg_mask, ctrl_pts, edges, c_split, r_split, anchor = _infer_structure(g)

    out = np.full((h, w), bg, dtype=int)

    for r, c in ctrl_pts:
        if r == 0:
            rows = [int(rr) for rr in np.argwhere(fg_mask[:, c]).flatten()]
            if rows:
                stop = int(max(rows))
                _paint_striped_ray(out, g, bg, ctrl, anchor, [(rr, c) for rr in range(0, stop + 1)])
        elif r == h - 1:
            rows = [int(rr) for rr in np.argwhere(fg_mask[:, c]).flatten()]
            if rows:
                stop = int(min(rows))
                _paint_striped_ray(out, g, bg, ctrl, anchor, [(rr, c) for rr in range(h - 1, stop - 1, -1)])
        elif c == 0:
            cols = [int(cc) for cc in np.argwhere(fg_mask[r, :]).flatten()]
            if cols:
                stop = int(max(cols))
                _paint_striped_ray(out, g, bg, ctrl, anchor, [(r, cc) for cc in range(0, stop + 1)])
        elif c == w - 1:
            cols = [int(cc) for cc in np.argwhere(fg_mask[r, :]).flatten()]
            if cols:
                stop = int(min(cols))
                _paint_striped_ray(out, g, bg, ctrl, anchor, [(r, cc) for cc in range(w - 1, stop - 1, -1)])

    for r, c in ctrl_pts:
        out[r, c] = ctrl

    split_pts: set[tuple[int, int]] = set()
    for r, c in np.argwhere(fg_mask):
        r = int(r)
        c = int(c)
        if (c_split is not None and c == c_split) or (r_split is not None and r == r_split):
            split_pts.add((r, c))
            out[r, c] = int(g[r, c])

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(fg_mask):
        r = int(r)
        c = int(c)
        if (r, c) in split_pts:
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if r < r_split else "B"
        if c_split is not None:
            vs = "L" if c < c_split else "R"
        regions.setdefault((hs, vs), []).append((r, c))

    for (hs, vs), pts in regions.items():
        if not (_touches_edges(hs, vs) & edges):
            continue
        dh, dv = _pick_dest(hs, vs, edges, r_split, c_split)
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        rmin, rmax = int(min(rs)), int(max(rs))
        cmin, cmax = int(min(cs)), int(max(cs))
        dr = -rmin if dh == "T" else (h - 1 - rmax)
        dc = -cmin if dv == "L" else (w - 1 - cmax)
        for r, c in pts:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(g[r, c])

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


def _assign_piece_colors(
    pts: set[tuple[int, int]] | frozenset[tuple[int, int]],
    palette: list[int],
    *,
    offset: int,
) -> dict[tuple[int, int], int]:
    ordered = sorted(pts, key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]))
    out: dict[tuple[int, int], int] = {}
    k = len(palette)
    for idx, cell in enumerate(ordered):
        out[cell] = int(palette[(idx + offset) % k])
    if len(ordered) >= 2 and k >= 2:
        out[ordered[0]] = int(palette[offset % k])
        out[ordered[1]] = int(palette[(offset + 1) % k])
    if len(ordered) >= 3 and k >= 3:
        out[ordered[2]] = int(palette[(offset + 2) % k])
    return out


def _fusion_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    bg, ctrl, fg_mask, ctrl_pts, _edges, c_split, r_split, anchor = _infer_structure(inp)
    sprite_colors = sorted(colors_present(inp, exclude={bg, ctrl}))

    split_pts = {
        (int(r), int(c))
        for r, c in np.argwhere(fg_mask)
        if (c_split is not None and int(c) == c_split) or (r_split is not None and int(r) == r_split)
    }

    ray_bg_cells: list[tuple[int, int]] = []
    for r, c in ctrl_pts:
        if r == 0:
            rows = [int(rr) for rr in np.argwhere(fg_mask[:, c]).flatten()]
            stop = int(max(rows))
            ray_bg_cells.extend((rr, c) for rr in range(0, stop + 1) if int(inp[rr, c]) == bg)
        elif r == inp.shape[0] - 1:
            rows = [int(rr) for rr in np.argwhere(fg_mask[:, c]).flatten()]
            stop = int(min(rows))
            ray_bg_cells.extend((rr, c) for rr in range(inp.shape[0] - 1, stop - 1, -1) if int(inp[rr, c]) == bg)
        elif c == 0:
            cols = [int(cc) for cc in np.argwhere(fg_mask[r, :]).flatten()]
            stop = int(max(cols))
            ray_bg_cells.extend((r, cc) for cc in range(0, stop + 1) if int(inp[r, cc]) == bg)
        elif c == inp.shape[1] - 1:
            cols = [int(cc) for cc in np.argwhere(fg_mask[r, :]).flatten()]
            stop = int(min(cols))
            ray_bg_cells.extend((r, cc) for cc in range(inp.shape[1] - 1, stop - 1, -1) if int(inp[r, cc]) == bg)

    alt_ray = sum(1 for r, c in ray_bg_cells if int(out[r, c]) == anchor)

    moved_multicolor = 0
    h, w = inp.shape
    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(fg_mask):
        r = int(r)
        c = int(c)
        if (r, c) in split_pts:
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if r < r_split else "B"
        if c_split is not None:
            vs = "L" if c < c_split else "R"
        regions.setdefault((hs, vs), []).append((r, c))

    for (hs, vs), pts in regions.items():
        if not (_touches_edges(hs, vs) & _edges):
            continue
        dh, dv = _pick_dest(hs, vs, _edges, r_split, c_split)
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)
        dr = -rmin if dh == "T" else (h - 1 - rmax)
        dc = -cmin if dv == "L" else (w - 1 - cmax)
        out_colors = {int(out[r + dr, c + dc]) for r, c in pts}
        if len(out_colors) >= 2:
            moved_multicolor += 1

    return {
        "sprite_colors": len(sprite_colors),
        "anchor_ray_cells": alt_ray,
        "moved_multicolor_regions": moved_multicolor,
    }


def _gen_attempt() -> np.ndarray | None:
    s = random.randint(18, 20)
    h = w = s
    bg = 1
    ctrl_color = 2

    num_ctrl = random.choices([1, 2], weights=[1, 4])[0]
    if num_ctrl == 1:
        ctrl_edges = [random.choice(["top", "bottom", "left", "right"])]
    else:
        ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]

    c_split = None
    r_split = None
    ctrl_pts: list[tuple[int, int]] = []
    edges = set(ctrl_edges)
    has_v = False
    has_h = False

    for edge in ctrl_edges:
        if edge in ("top", "bottom"):
            c_split = random.randint(int(w * 0.4), int(w * 0.6))
            ctrl_pts.append((0 if edge == "top" else h - 1, c_split))
            has_v = True
        else:
            r_split = random.randint(int(h * 0.4), int(h * 0.6))
            ctrl_pts.append((r_split, 0 if edge == "left" else w - 1))
            has_h = True

    if has_h and has_v:
        all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    elif has_v:
        all_regions = [("A", "L"), ("A", "R")]
    else:
        all_regions = [("T", "A"), ("B", "A")]

    kept = [reg for reg in all_regions if _touches_edges(*reg) & edges]
    vanish = [reg for reg in all_regions if reg not in kept]

    sprite_h = random.randint(5, 8)
    sprite_w = random.randint(8, 10)

    if has_v:
        frac = random.uniform(0.4, 0.6)
        sprite_c0 = c_split - int(frac * sprite_w)
        sprite_c0 = max(4, min(sprite_c0, w - 5 - sprite_w + 1))
        if sprite_c0 > w - 5 - sprite_w + 1:
            return None
        if c_split <= sprite_c0 or c_split >= sprite_c0 + sprite_w - 1:
            return None
    else:
        hi = w - 5 - sprite_w + 1
        if hi < 4:
            return None
        sprite_c0 = random.randint(4, hi)

    if has_h:
        frac = random.uniform(0.4, 0.6)
        sprite_r0 = r_split - int(frac * sprite_h)
        sprite_r0 = max(4, min(sprite_r0, h - 5 - sprite_h + 1))
        if sprite_r0 > h - 5 - sprite_h + 1:
            return None
        if r_split <= sprite_r0 or r_split >= sprite_r0 + sprite_h - 1:
            return None
    else:
        hi = h - 5 - sprite_h + 1
        if hi < 4:
            return None
        sprite_r0 = random.randint(4, hi)

    sprite_r1 = sprite_r0 + sprite_h - 1
    sprite_c1 = sprite_c0 + sprite_w - 1

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = sprite_r0 if hs in ("T", "A") else int(r_split) + 1
        r1 = (int(r_split) - 1 if hs == "T" else sprite_r1) if has_h else sprite_r1
        c0 = sprite_c0 if vs in ("L", "A") else int(c_split) + 1
        c1 = (int(c_split) - 1 if vs == "L" else sprite_c1) if has_v else sprite_c1
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
            split_pixels.add((r, int(c_split)))
    if has_h:
        n = random.randint(2, 4)
        cols = [c for c in range(sprite_c0, sprite_c1 + 1) if c != c_split]
        if len(cols) < n:
            return None
        for c in random.sample(cols, n):
            split_pixels.add((int(r_split), c))

    area = sprite_h * sprite_w
    target_density = random.uniform(0.39, 0.60)
    target_total = int(target_density * area)
    budget_pieces = target_total - len(split_pixels)

    if vanish:
        vanish_budget = random.randint(3, min(7, max(3, budget_pieces // 3)))
        budget_pieces -= vanish_budget
    else:
        vanish_budget = 0

    if budget_pieces < len(kept) * 3:
        return None

    piece_budgets: list[int] = []
    remaining = budget_pieces
    for idx, reg in enumerate(kept):
        r0, r1, c0, c1 = sub_bbox(*reg)
        max_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if idx == len(kept) - 1:
            px = remaining
        else:
            avg = remaining // (len(kept) - idx)
            lo_px = max(3, avg - 2)
            hi_px = min(max_area, avg + 2)
            if lo_px > hi_px:
                return None
            px = random.randint(lo_px, hi_px)
        px = max(3, min(px, max_area, remaining - (len(kept) - idx - 1) * 3))
        piece_budgets.append(px)
        remaining -= px
    if remaining != 0 or any(b < 3 for b in piece_budgets):
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}

    for idx, reg in enumerate(kept):
        hs, vs = reg
        dh, dv = _pick_dest(hs, vs, edges, r_split, c_split)
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = r1 - r0 + 1
        sub_w = c1 - c0 + 1
        piece = _grow_piece(sub_h, sub_w, piece_budgets[idx])
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
            if r < 4 or r > h - 5 or c < 4 or c > w - 5:
                return None
            if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                return None
            if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                return None
        input_pieces[reg] = inp_pts

        inp_rs = [r for r, _ in inp_pts]
        inp_cs = [c for _, c in inp_pts]
        dr = -min(inp_rs) if dh == "T" else (h - 1 - max(inp_rs))
        dc = -min(inp_cs) if dv == "L" else (w - 1 - max(inp_cs))
        out_pts = frozenset((r + dr, c + dc) for r, c in inp_pts)
        for r, c in out_pts:
            if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                return None
            if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                return None
        output_pieces[reg] = out_pts

    all_inp_fg = set(split_pixels)
    for pts in input_pieces.values():
        all_inp_fg |= set(pts)
    if not all_inp_fg:
        return None

    arr = np.array(list(all_inp_fg))
    bb_h = int(arr[:, 0].max() - arr[:, 0].min() + 1)
    bb_w = int(arr[:, 1].max() - arr[:, 1].min() + 1)
    if bb_h < 5 or bb_h > 8 or bb_w < 8 or bb_w > 10:
        return None

    mask = np.zeros((h, w), dtype=int)
    for r, c in all_inp_fg:
        mask[r, c] = 1
    if not is_contiguous(mask):
        return None

    vanish_fg: set[tuple[int, int]] = set()
    if vanish:
        for vreg in vanish:
            vr0, vr1, vc0, vc1 = sub_bbox(*vreg)
            vh = vr1 - vr0 + 1
            vw = vc1 - vc0 + 1
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
            vhs, vvs = vreg
            vr_off = (vr1 - pph + 1) if vhs == "T" else vr0
            vc_off = (vc1 - ppw + 1) if vvs == "L" else vc0
            vpts = {(r + vr_off, c + vc_off) for r, c in piece}
            for r, c in vpts:
                if r < 4 or r > h - 5 or c < 4 or c > w - 5:
                    return None
                if (r, c) in all_inp_fg:
                    return None
            vanish_fg |= vpts

    combined = all_inp_fg | vanish_fg
    mask2 = np.zeros((h, w), dtype=int)
    for r, c in combined:
        mask2[r, c] = 1
    if not is_contiguous(mask2):
        return None

    num_sprite_colors = random.choices([2, 3], weights=[4, 1])[0]
    sprite_colors = random.sample(SPRITE_PALETTE, num_sprite_colors)
    anchor = sprite_colors[0]

    color_map: dict[tuple[int, int], int] = {}
    for cell in split_pixels:
        color_map[cell] = anchor

    kept_multicolor = 0
    for idx, reg in enumerate(kept):
        piece_colors = _assign_piece_colors(input_pieces[reg], sprite_colors, offset=idx)
        color_map.update(piece_colors)
        if len(set(piece_colors.values())) >= 2:
            kept_multicolor += 1

    if kept_multicolor == 0:
        return None

    if vanish_fg:
        vanish_colors = _assign_piece_colors(vanish_fg, sprite_colors, offset=len(kept))
        color_map.update(vanish_colors)

    if len(set(color_map.values())) < 2:
        return None

    inp = np.full((h, w), bg, dtype=int)
    for r, c in combined:
        inp[r, c] = int(color_map[(r, c)])
    for r, c in ctrl_pts:
        inp[r, c] = ctrl_color

    fg_all = np.array(list(combined))
    final_h = int(fg_all[:, 0].max() - fg_all[:, 0].min() + 1)
    final_w = int(fg_all[:, 1].max() - fg_all[:, 1].min() + 1)
    actual_density = len(combined) / (final_h * final_w)
    if actual_density < 0.38 or actual_density > 0.62:
        return None

    out = main(inp)
    metrics = _fusion_metrics(inp, out)
    if metrics["sprite_colors"] < 2:
        return None
    if metrics["anchor_ray_cells"] < 1:
        return None
    if metrics["moved_multicolor_regions"] < 1:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
