# Variant of 4a21e3da
# Dimension: D3 — 源对象类型 (src)
# Change: control_pixels — 网格边缘的1-2个红色控制像素作为射线源... → 边缘色条作为射线源
#
# Design analysis
# a) Concretization and behavior trace:
#    - The old task used one border pixel per ray source. This variant replaces each source
#      with a short ctrl-colored bar of length 2 or 3 that extends inward from the border
#      along the future ray axis. Top/bottom bars are vertical; left/right bars are horizontal.
#    - The split line is still the row/col containing that bar. The output ray continues
#      farther inward on the same line, painting only bg cells until the farthest fg pixel.
#    - The new information carried by the source object is the bar length:
#      vertical-source len 2 -> prefer left regions; len 3 -> prefer right regions.
#      horizontal-source len 2 -> prefer top regions; len 3 -> prefer bottom regions.
#    - Region motion is intentionally kept close to the original: once a region survives,
#      it still slides to the same corner dictated by its geometric quadrant and the source edge.
#    - Concrete trace for a top bar:
#      old: one top pixel at column c keeps both left/right halves and sends them to top corners.
#      new: a vertical bar at column c of len 2 or 3 still cuts at column c, but only the
#      length-chosen half survives; the other half disappears. With two bars, the vanished
#      quadrant is the one rejected by both length choices.
# b) Complexity calibration:
#    - Only two bar lengths are used. This keeps the rule learnable while still making the
#      variant visually different from the original.
#    - In two-bar cases the output still usually contains three moved pieces plus split-line
#      pixels, so the task does not collapse into a trivial single-fragment puzzle.
# c) Implicit assumption audit against the original main():
#    - Original code assumed every ctrl cell was an independent source pixel; that breaks here.
#      The new solver must group ctrl cells into 1-2 connected bar components and infer edge,
#      split line, and encoded side from each component as a whole.
#    - Original "kept if region touches a control edge" no longer holds. Kept/vanish now
#      depends on side labels selected by bar lengths.
#    - The ray drawing stage can stay mostly unchanged because the bar lies on the same line;
#      only source parsing changes.
# d) Placement legality:
#    - Bars are length 2-3, so keeping sprite margins >= 4 still prevents overlap.
#    - The generator keeps one vertical split at most and one horizontal split at most, so
#      bar parsing is unambiguous and moved pieces still fit on their side of the split.
# e) Visual observability:
#    - Humans can directly see the source changed from a dot to a short inward strip.
#    - The only non-geometric state carried by the strip is length 2 vs 3, which is a visible,
#      repeated binary cue.
# f) Visual cleanliness:
#    - Bars live on otherwise empty margins, rays stay on a single row/col, and the sprite
#      remains centered. The grid does not accumulate overlapping annotation systems.
# g) Color inference:
#    - No new color mapping is introduced. The same ctrl color marks both source bars and
#      emitted rays, so the variant stays within the original 3-color regime.
# 9) New solver assumptions checked against the generator:
#    - Every ctrl component is a straight contiguous 2-3 cell bar anchored to exactly one edge.
#    - There is at most one vertical-source bar and at most one horizontal-source bar.
#    - The sprite still contains fg on every active split line, so each ray has a stopping point.

from __future__ import annotations

import random
import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    for _ in range(int(target_px) - 1):
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


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not bool(mask[r, c]) or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
                    if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bar_cells(edge: str, split: int, length: int, H: int, W: int) -> list[tuple[int, int]]:
    if edge == "top":
        return [(r, split) for r in range(length)]
    if edge == "bottom":
        return [(H - 1 - r, split) for r in range(length)]
    if edge == "left":
        return [(split, c) for c in range(length)]
    if edge == "right":
        return [(split, W - 1 - c) for c in range(length)]
    raise ValueError(edge)


def _parse_bars(g: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    H, W = g.shape
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})
    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(bg)
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != ctrl))

    bars: list[dict[str, object]] = []
    for comp in _components(g == ctrl):
        rs = sorted({int(r) for r, _ in comp})
        cs = sorted({int(c) for _, c in comp})
        comp_set = set(comp)
        edge: str
        split: int
        if len(cs) == 1 and rs[0] == 0:
            edge = "top"
            split = int(cs[0])
            expect = set((r, split) for r in range(len(rs)))
        elif len(cs) == 1 and rs[-1] == H - 1:
            edge = "bottom"
            split = int(cs[0])
            expect = set((H - 1 - r, split) for r in range(len(rs)))
        elif len(rs) == 1 and cs[0] == 0:
            edge = "left"
            split = int(rs[0])
            expect = set((split, c) for c in range(len(cs)))
        elif len(rs) == 1 and cs[-1] == W - 1:
            edge = "right"
            split = int(rs[0])
            expect = set((split, W - 1 - c) for c in range(len(cs)))
        else:
            raise AssertionError("Control component is not an edge-anchored straight bar")

        assert comp_set == expect
        length = len(comp)
        assert length in (2, 3)
        keep_h = None
        keep_v = None
        if edge in ("top", "bottom"):
            keep_v = "L" if length == 2 else "R"
        else:
            keep_h = "T" if length == 2 else "B"
        bars.append(
            {
                "edge": edge,
                "split": int(split),
                "length": int(length),
                "cells": list(expect),
                "keep_h": keep_h,
                "keep_v": keep_v,
            }
        )

    assert 1 <= len(bars) <= 2
    assert sum(bar["edge"] in ("top", "bottom") for bar in bars) <= 1
    assert sum(bar["edge"] in ("left", "right") for bar in bars) <= 1
    return ctrl, fg, bars


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    ctrl, fg, bars = _parse_bars(g)

    E: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    keep_v: str | None = None
    keep_h: str | None = None
    for bar in bars:
        edge = str(bar["edge"])
        E.add(edge)
        if edge in ("top", "bottom"):
            c_split = int(bar["split"])
            keep_v = str(bar["keep_v"])
        else:
            r_split = int(bar["split"])
            keep_h = str(bar["keep_h"])

    out = np.full((H, W), bg, dtype=int)

    for bar in bars:
        edge = str(bar["edge"])
        split = int(bar["split"])
        if edge == "top":
            rows = [int(rr) for rr in np.argwhere(g[:, split] == fg).flatten()]
            if rows:
                stop = int(max(rows))
                for rr in range(0, stop + 1):
                    if int(g[rr, split]) == bg:
                        out[rr, split] = ctrl
        elif edge == "bottom":
            rows = [int(rr) for rr in np.argwhere(g[:, split] == fg).flatten()]
            if rows:
                stop = int(min(rows))
                for rr in range(H - 1, stop - 1, -1):
                    if int(g[rr, split]) == bg:
                        out[rr, split] = ctrl
        elif edge == "left":
            cols = [int(cc) for cc in np.argwhere(g[split, :] == fg).flatten()]
            if cols:
                stop = int(max(cols))
                for cc in range(0, stop + 1):
                    if int(g[split, cc]) == bg:
                        out[split, cc] = ctrl
        else:
            cols = [int(cc) for cc in np.argwhere(g[split, :] == fg).flatten()]
            if cols:
                stop = int(min(cols))
                for cc in range(W - 1, stop - 1, -1):
                    if int(g[split, cc]) == bg:
                        out[split, cc] = ctrl

    for bar in bars:
        for r, c in bar["cells"]:
            out[int(r), int(c)] = ctrl

    for r, c in np.argwhere(g == fg):
        rr = int(r)
        cc = int(c)
        if (c_split is not None and cc == c_split) or (r_split is not None and rr == r_split):
            out[rr, cc] = fg

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == fg):
        rr = int(r)
        cc = int(c)
        if (c_split is not None and cc == c_split) or (r_split is not None and rr == r_split):
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if rr < r_split else "B"
        if c_split is not None:
            vs = "L" if cc < c_split else "R"
        regions.setdefault((hs, vs), []).append((rr, cc))

    def keep_region(hs: str, vs: str) -> bool:
        keep = False
        if keep_v is not None and vs == keep_v:
            keep = True
        if keep_h is not None and hs == keep_h:
            keep = True
        return keep

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
        if not keep_region(hs, vs):
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
                out[rr, cc] = fg

    return out


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2

    num_ctrl = random.choices([1, 2], weights=[1, 4])[0]
    if num_ctrl == 1:
        ctrl_edges = [random.choice(["top", "bottom", "left", "right"])]
    else:
        ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]

    bar_specs: list[dict[str, int | str]] = []
    E = set(ctrl_edges)
    c_split = None
    r_split = None
    has_v = False
    has_h = False

    lengths = [random.choice([2, 3]) for _ in ctrl_edges]
    if len(lengths) == 2 and lengths[0] == lengths[1] and random.random() < 0.7:
        lengths[1] = 5 - lengths[0]

    for edge, length in zip(ctrl_edges, lengths):
        if edge in ("top", "bottom"):
            c_split = random.randint(int(W * 0.4), int(W * 0.6))
            has_v = True
            bar_specs.append({"edge": edge, "length": int(length), "split": int(c_split)})
        else:
            r_split = random.randint(int(H * 0.4), int(H * 0.6))
            has_h = True
            bar_specs.append({"edge": edge, "length": int(length), "split": int(r_split)})

    keep_v = None
    keep_h = None
    for bar in bar_specs:
        if bar["edge"] in ("top", "bottom"):
            keep_v = "L" if int(bar["length"]) == 2 else "R"
        else:
            keep_h = "T" if int(bar["length"]) == 2 else "B"

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

    def region_kept(hs: str, vs: str) -> bool:
        keep = False
        if keep_v is not None and vs == keep_v:
            keep = True
        if keep_h is not None and hs == keep_h:
            keep = True
        return keep

    if has_h and has_v:
        all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    elif has_v:
        all_regions = [("A", "L"), ("A", "R")]
    else:
        all_regions = [("T", "A"), ("B", "A")]

    kept = [reg for reg in all_regions if region_kept(*reg)]
    vanish = [reg for reg in all_regions if reg not in kept]
    if not kept or not vanish:
        return None

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
    n_split = len(split_pixels)
    budget_pieces = target_total - n_split
    if budget_pieces < len(kept) * 3 + 3:
        return None

    vanish_budget = random.randint(3, min(7, max(3, budget_pieces // 3)))
    budget_pieces -= vanish_budget
    if budget_pieces < len(kept) * 3:
        return None

    piece_budgets: list[int] = []
    remaining = budget_pieces
    for i, reg in enumerate(kept):
        r0, r1, c0, c1 = sub_bbox(*reg)
        max_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if i == len(kept) - 1:
            px = remaining
        else:
            avg = remaining // (len(kept) - i)
            lo_px = max(3, avg - 2)
            hi_px = min(max_area, avg + 2)
            if lo_px > hi_px:
                return None
            px = random.randint(lo_px, hi_px)
        px = max(3, min(px, max_area, remaining - (len(kept) - i - 1) * 3))
        piece_budgets.append(px)
        remaining -= px
    if remaining != 0 or any(px < 3 for px in piece_budgets):
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
            split_rows = sorted(r for r, c in split_pixels if has_v and c == c_split)
            if split_rows:
                mid = (split_rows[0] + split_rows[-1]) // 2
                inp_r0 = max(r0, min(mid - ph // 2, r1 - ph + 1))
            else:
                inp_r0 = r0

        if has_v:
            inp_c0 = (c1 - pw + 1) if vs == "L" else c0
        else:
            split_cols = sorted(c for r, c in split_pixels if has_h and r == r_split)
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
    if len(vanish) != 1:
        return None
    vhs, vvs = vanish[0]
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
    vanish_fg = {(r + vr_off, c + vc_off) for r, c in piece}
    for r, c in vanish_fg:
        if r < 4 or r > H - 5 or c < 4 or c > W - 5:
            return None
        if (r, c) in all_inp_fg:
            return None

    combined = all_inp_fg | vanish_fg
    mask2 = np.zeros((H, W), dtype=int)
    for r, c in combined:
        mask2[r, c] = 1
    if not is_contiguous(mask2):
        return None

    inp = np.full((H, W), bg, dtype=int)
    for r, c in combined:
        inp[r, c] = fg
    for bar in bar_specs:
        for r, c in _bar_cells(str(bar["edge"]), int(bar["split"]), int(bar["length"]), H, W):
            if inp[r, c] != bg:
                return None
            inp[r, c] = ctrl_color

    out = np.full((H, W), bg, dtype=int)
    for pts in output_pieces.values():
        for r, c in pts:
            out[r, c] = fg
    for r, c in split_pixels:
        out[r, c] = fg

    for bar in bar_specs:
        edge = str(bar["edge"])
        split = int(bar["split"])
        if edge == "top":
            fg_rows = [r for r in range(H) if out[r, split] == fg]
            if fg_rows:
                for r in range(max(fg_rows) + 1):
                    if out[r, split] == bg:
                        out[r, split] = ctrl_color
        elif edge == "bottom":
            fg_rows = [r for r in range(H) if out[r, split] == fg]
            if fg_rows:
                for r in range(H - 1, min(fg_rows) - 1, -1):
                    if out[r, split] == bg:
                        out[r, split] = ctrl_color
        elif edge == "left":
            fg_cols = [c for c in range(W) if out[split, c] == fg]
            if fg_cols:
                for c in range(max(fg_cols) + 1):
                    if out[split, c] == bg:
                        out[split, c] = ctrl_color
        else:
            fg_cols = [c for c in range(W) if out[split, c] == fg]
            if fg_cols:
                for c in range(W - 1, min(fg_cols) - 1, -1):
                    if out[split, c] == bg:
                        out[split, c] = ctrl_color

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
