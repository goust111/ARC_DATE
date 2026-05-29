# Combo variant of 4a21e3da (index 39)
# combo_spec: D2/1+D5/0
# Name: 颜色循环射线+未接壤区域着色

from __future__ import annotations

import random
import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior trace
# - Keep the original geometry: one connected central fg sprite, one vertical split and one
#   horizontal split defined by two border ctrl pixels, and 2-4 fg anchor pixels on each split line.
# - Original task:
#   * rays paint every reachable bg cell on the split row/col in solid ctrl color;
#   * regions touching a control edge slide to their matching corner;
#   * the unique region touching neither control edge disappears.
# - This fused variant changes both highlighted branches in a coordinated way:
#   * ray paint on bg cells alternates [alt, ctrl, alt, ctrl, ...] when read inward after the
#     border ctrl pixel, so each ray becomes a 2-color stripe;
#   * the detached non-touching region no longer disappears; it stays in place and recolors to the
#     same alt color used by the ray stripes.
# - Example with controls on top and right:
#   * regions are TL, TR, BL, BR;
#   * TL/TR/BR still slide to their same-side corners;
#   * BL stays near the split intersection and turns alt;
#   * the top-down ray and right-left ray both show ctrl/alt striping on background cells.
#   The shared alt color ties "ray secondary color" to "detached-region recolor" visually.
#
# b) Complexity calibration
# - The generator always uses two perpendicular controls so the recolor branch is visible in every
#   sample; otherwise D5/0 would disappear on 1-control cases.
# - Only one extra output color is introduced. That keeps the color rule compact enough for 2-4 ARC
#   examples: humans see the same new color both on ray stripes and on the preserved detached piece.
# - Detached pieces are forced to have at least 4 cells, and each ray has at least one alt stripe
#   cell, so the fused change does not collapse into a barely visible single-pixel effect.
#
# c) Hidden assumptions from the original solver that break, and how the combo repairs them
# - Original stage 1 assumed "every painted ray cell is ctrl"; broken by D2/1. We replace it with a
#   dedicated striped-ray painter that only counts bg cells, preserving the original intermittent
#   skip over fg cells.
# - Original transform branch used `continue` for non-touching regions; broken by D5/0. We now write
#   those cells back in place using the alt color.
# - Original color inference `fg = only non-bg non-border color` still holds because the new alt
#   color appears only in outputs, never in generated inputs.
# - Original ray-end logic looked only for input fg on the split line. That remains valid because
#   the detached piece is generated strictly away from both split lines, so recoloring it never
#   changes the ray endpoint.
#
# d) Placement legality
# - Output-first generation is preserved. Kept pieces are generated inside their legal quadrant and
#   translated toward corners. The detached piece is generated inside the unique non-touching
#   quadrant and stays there in the output.
# - The sprite bbox stays 5-8 by 8-10 with margin >= 4 from every border, so striped rays always
#   have room before reaching the sprite and never need clipping.
# - The combined input sprite (kept pieces + split pixels + detached piece) must remain a single
#   4-connected component; candidates failing that are rejected.
#
# e) Visual observability
# - A human observer sees two repeated clues across samples:
#   * every ray alternates between ctrl and the same secondary color;
#   * the formerly "missing" quadrant persists in exactly that same secondary color.
# - The generator rejects cases where the combo output matches the D5-only baseline
#   (solid recolor but no ray striping) or the D2-only baseline (striped rays but detached piece
#   erased). This guarantees that both components are behaviorally active.
#
# f) Visual cleanliness
# - The new alt color is used in only two structured places: ray background cells and the detached
#   piece. No extra overlays, labels, or numeric-color tricks are introduced.
# - Detached cells never lie on split lines, and moved corner pieces slide away from the split
#   center, so the recolored piece stays visually separated from the corner pieces.
#
# g) Color inferability
# - The combo uses exactly four semantic colors in the output: bg, ctrl, fg, alt.
# - The alt color has a single repeated meaning: "secondary ray stripe / preserved detached region".
#   That shared role is structural and repeated, so the mapping is inferable from a few examples.
#
# New solver assumptions introduced by the combo
# - Generated inputs always contain exactly one vertical and one horizontal control.
# - At least one fg pixel exists on each split line.
# - The detached region never touches the split row or split column.
# - The chosen alt color is distinct from bg/fg/ctrl.


ALT_COLOR = 3


def _pick_alt(bg: int, fg: int, ctrl: int) -> int:
    preferred = [ALT_COLOR, 4, 5, 6, 8, 9, 0]
    for cand in preferred:
        if int(cand) not in {int(bg), int(fg), int(ctrl)}:
            return int(cand)
    raise AssertionError("no available alternate color")


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
    return str(dh), str(dv)


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, int, int, int, int, list[tuple[int, int]], set[str], int, int]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})

    border_cols = set(map(int, g[0, :])) | set(map(int, g[h - 1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, w - 1]))
    border_cols.discard(int(bg))
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != int(ctrl)))
    alt = _pick_alt(bg, fg, ctrl)

    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(ctrl))]
    edges: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for r, c in ctrl_pts:
        if int(r) == 0:
            edges.add("top")
            c_split = int(c)
        elif int(r) == int(h - 1):
            edges.add("bottom")
            c_split = int(c)
        elif int(c) == 0:
            edges.add("left")
            r_split = int(r)
        elif int(c) == int(w - 1):
            edges.add("right")
            r_split = int(r)

    assert c_split is not None and r_split is not None
    return g, bg, fg, ctrl, alt, ctrl_pts, edges, int(r_split), int(c_split)


def _paint_cycle_on_bg(
    out: np.ndarray,
    src: np.ndarray,
    bg: int,
    ctrl: int,
    alt: int,
    cells: list[tuple[int, int]],
) -> None:
    painted = 0
    for r, c in cells:
        if int(src[int(r), int(c)]) != int(bg):
            continue
        out[int(r), int(c)] = int(alt if painted % 2 == 0 else ctrl)
        painted += 1


def _region_key(r: int, c: int, r_split: int, c_split: int) -> tuple[str, str]:
    hs = "T" if int(r) < int(r_split) else "B"
    vs = "L" if int(c) < int(c_split) else "R"
    return str(hs), str(vs)


def _solve_core(grid: np.ndarray, *, striped_rays: bool, recolor_detached: bool) -> np.ndarray:
    g, bg, fg, ctrl, alt, ctrl_pts, edges, r_split, c_split = _parse_input(grid)
    h, w = g.shape
    out = np.full((h, w), int(bg), dtype=int)

    for r, c in ctrl_pts:
        if int(r) == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, int(c)] == int(fg)).flatten()]
            if rows:
                stop = int(max(rows))
                cells = [(int(rr), int(c)) for rr in range(0, int(stop) + 1)]
                if striped_rays:
                    _paint_cycle_on_bg(out, g, bg, ctrl, alt, cells)
                else:
                    for rr, cc in cells:
                        if int(g[int(rr), int(cc)]) == int(bg):
                            out[int(rr), int(cc)] = int(ctrl)
        elif int(r) == int(h - 1):
            rows = [int(rr) for rr in np.argwhere(g[:, int(c)] == int(fg)).flatten()]
            if rows:
                stop = int(min(rows))
                cells = [(int(rr), int(c)) for rr in range(int(h - 1), int(stop) - 1, -1)]
                if striped_rays:
                    _paint_cycle_on_bg(out, g, bg, ctrl, alt, cells)
                else:
                    for rr, cc in cells:
                        if int(g[int(rr), int(cc)]) == int(bg):
                            out[int(rr), int(cc)] = int(ctrl)
        elif int(c) == 0:
            cols = [int(cc) for cc in np.argwhere(g[int(r), :] == int(fg)).flatten()]
            if cols:
                stop = int(max(cols))
                cells = [(int(r), int(cc)) for cc in range(0, int(stop) + 1)]
                if striped_rays:
                    _paint_cycle_on_bg(out, g, bg, ctrl, alt, cells)
                else:
                    for rr, cc in cells:
                        if int(g[int(rr), int(cc)]) == int(bg):
                            out[int(rr), int(cc)] = int(ctrl)
        elif int(c) == int(w - 1):
            cols = [int(cc) for cc in np.argwhere(g[int(r), :] == int(fg)).flatten()]
            if cols:
                stop = int(min(cols))
                cells = [(int(r), int(cc)) for cc in range(int(w - 1), int(stop) - 1, -1)]
                if striped_rays:
                    _paint_cycle_on_bg(out, g, bg, ctrl, alt, cells)
                else:
                    for rr, cc in cells:
                        if int(g[int(rr), int(cc)]) == int(bg):
                            out[int(rr), int(cc)] = int(ctrl)

    for r, c in ctrl_pts:
        out[int(r), int(c)] = int(ctrl)

    split_pts: set[tuple[int, int]] = set()
    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if int(r) == int(r_split) or int(c) == int(c_split):
            split_pts.add((int(r), int(c)))
            out[int(r), int(c)] = int(fg)

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (int(r), int(c)) in split_pts:
            continue
        regions.setdefault(_region_key(r, c, r_split, c_split), []).append((int(r), int(c)))

    for (hs, vs), pts in regions.items():
        if not (_touches_edges(hs, vs) & edges):
            if recolor_detached:
                for r, c in pts:
                    out[int(r), int(c)] = int(alt)
            continue

        dh, dv = _pick_dest(hs, vs, edges, r_split, c_split)
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        rmin, rmax = int(min(rs)), int(max(rs))
        cmin, cmax = int(min(cs)), int(max(cs))
        dr = int(0 - rmin) if dh == "T" else int((h - 1) - rmax)
        dc = int(0 - cmin) if dv == "L" else int((w - 1) - cmax)

        for r, c in pts:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[int(rr), int(cc)] = int(fg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut a sprite with striped border rays.

    Fused rule:
    - rays still skip fg and only paint bg cells, but alternate between the ctrl color and a fixed
      secondary color;
    - pieces touching a control edge still slide to the matching corner;
    - the unique non-touching piece stays in place and recolors to that same secondary color.
    """
    return _solve_core(grid, striped_rays=True, recolor_detached=True)


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    for _ in range(int(target_px) - 1):
        neighbors = set()
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = int(r + dr), int(c + dc)
                if (nr, nc) not in cells:
                    neighbors.add((int(nr), int(nc)))
        valid = [
            (int(nr), int(nc))
            for nr, nc in neighbors
            if max(r for r, _ in cells | {(int(nr), int(nc))}) - min(r for r, _ in cells | {(int(nr), int(nc))}) + 1 <= int(max_h)
            and max(c for _, c in cells | {(int(nr), int(nc))}) - min(c for _, c in cells | {(int(nr), int(nc))}) + 1 <= int(max_w)
        ]
        if not valid:
            break
        cells.add(random.choice(valid))
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return frozenset((int(r - min(rs)), int(c - min(cs))) for r, c in cells)


def _piece_dims(piece: frozenset[tuple[int, int]]) -> tuple[int, int]:
    rs = [r for r, _ in piece]
    cs = [c for _, c in piece]
    return int(max(rs) - min(rs) + 1), int(max(cs) - min(cs) + 1)


def _grow_piece_with_retry(
    max_h: int,
    max_w: int,
    target_px: int,
    *,
    min_h: int = 1,
    min_w: int = 1,
    tries: int = 40,
) -> frozenset[tuple[int, int]] | None:
    best: frozenset[tuple[int, int]] | None = None
    for _ in range(int(tries)):
        piece = _grow_piece(int(max_h), int(max_w), int(target_px))
        ph, pw = _piece_dims(piece)
        if len(piece) >= int(target_px) and ph >= int(min_h) and pw >= int(min_w):
            return piece
        if best is None or len(piece) > len(best):
            best = piece
    return best


def _ray_bg_sequences(inp: np.ndarray) -> list[list[tuple[int, int]]]:
    g, bg, fg, _ctrl, _alt, ctrl_pts, _edges, _r_split, _c_split = _parse_input(inp)
    h, w = g.shape
    seqs: list[list[tuple[int, int]]] = []
    for r, c in ctrl_pts:
        cells: list[tuple[int, int]]
        if int(r) == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, int(c)] == int(fg)).flatten()]
            stop = int(max(rows))
            cells = [(int(rr), int(c)) for rr in range(0, int(stop) + 1) if int(g[int(rr), int(c)]) == int(bg)]
        elif int(r) == int(h - 1):
            rows = [int(rr) for rr in np.argwhere(g[:, int(c)] == int(fg)).flatten()]
            stop = int(min(rows))
            cells = [(int(rr), int(c)) for rr in range(int(h - 1), int(stop) - 1, -1) if int(g[int(rr), int(c)]) == int(bg)]
        elif int(c) == 0:
            cols = [int(cc) for cc in np.argwhere(g[int(r), :] == int(fg)).flatten()]
            stop = int(max(cols))
            cells = [(int(r), int(cc)) for cc in range(0, int(stop) + 1) if int(g[int(r), int(cc)]) == int(bg)]
        else:
            cols = [int(cc) for cc in np.argwhere(g[int(r), :] == int(fg)).flatten()]
            stop = int(min(cols))
            cells = [(int(r), int(cc)) for cc in range(int(w - 1), int(stop) - 1, -1) if int(g[int(r), int(cc)]) == int(bg)]
        seqs.append(cells)
    return seqs


def _fusion_metrics(inp: np.ndarray, combo: np.ndarray) -> dict[str, int]:
    _g, bg, fg, ctrl, alt, _ctrl_pts, _edges, _r_split, _c_split = _parse_input(inp)
    ray_sequences = _ray_bg_sequences(inp)
    ray_alt_cells = sum(1 for seq in ray_sequences for r, c in seq if int(combo[int(r), int(c)]) == int(alt))
    rays_with_alt = sum(1 for seq in ray_sequences if any(int(combo[int(r), int(c)]) == int(alt) for r, c in seq))
    recolor_cells = int(np.count_nonzero((inp == int(fg)) & (combo == int(alt))))
    d5_only = _solve_core(inp, striped_rays=False, recolor_detached=True)
    d2_only = _solve_core(inp, striped_rays=True, recolor_detached=False)
    base = _solve_core(inp, striped_rays=False, recolor_detached=False)
    return {
        "ray_alt_cells": int(ray_alt_cells),
        "rays_with_alt": int(rays_with_alt),
        "recolor_cells": int(recolor_cells),
        "diff_vs_d5_only": int(np.count_nonzero(combo != d5_only)),
        "diff_vs_d2_only": int(np.count_nonzero(combo != d2_only)),
        "diff_vs_base": int(np.count_nonzero(combo != base)),
        "alt_total": int(np.count_nonzero(combo == int(alt))),
        "ctrl": int(ctrl),
        "bg": int(bg),
    }


def _bridge_regions(edges: set[str]) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str], tuple[str, str]]:
    top_bottom = "T" if "top" in edges else "B"
    left_right = "L" if "left" in edges else "R"
    bridge = (top_bottom, left_right)
    side_v = (top_bottom, "R" if left_right == "L" else "L")
    side_h = ("B" if top_bottom == "T" else "T", left_right)
    detached = ("B" if top_bottom == "T" else "T", "R" if left_right == "L" else "L")
    return bridge, side_v, side_h, detached


def _pick_unique_positions(
    preferred_regs: list[tuple[str, str]],
    candidate_map: dict[tuple[str, str], list[int]],
    *,
    target_count: int,
) -> list[int] | None:
    picked: list[int] = []
    used: set[int] = set()
    for reg in preferred_regs:
        vals = [int(v) for v in candidate_map.get(reg, []) if int(v) not in used]
        if not vals:
            vals = [int(v) for v in candidate_map.get(reg, [])]
        if not vals:
            return None
        choice = int(vals[len(vals) // 2])
        picked.append(choice)
        used.add(choice)

    pool = sorted({int(v) for vals in candidate_map.values() for v in vals if int(v) not in used})
    while len(picked) < int(target_count) and pool:
        choice = int(pool.pop(len(pool) // 2))
        picked.append(choice)
        used.add(choice)
        pool = [int(v) for v in pool if int(v) not in used]

    if len(set(picked)) < min(2, int(target_count)):
        return None
    return [int(v) for v in picked[:target_count]]


def _gen_attempt() -> np.ndarray | None:
    s = random.randint(18, 20)
    h = w = s
    bg, fg, ctrl = 1, 7, 2
    alt = _pick_alt(bg, fg, ctrl)

    ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]
    edges = set(ctrl_edges)

    c_split: int | None = None
    r_split: int | None = None
    ctrl_pts: list[tuple[int, int]] = []
    for edge in ctrl_edges:
        if edge in ("top", "bottom"):
            c_split = random.randint(int(w * 0.4), int(w * 0.6))
            ctrl_pts.append((0 if edge == "top" else int(h - 1), int(c_split)))
        else:
            r_split = random.randint(int(h * 0.4), int(h * 0.6))
            ctrl_pts.append((int(r_split), 0 if edge == "left" else int(w - 1)))

    assert c_split is not None and r_split is not None

    all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    kept = [reg for reg in all_regions if _touches_edges(*reg) & edges]
    detached = [reg for reg in all_regions if reg not in kept]
    if len(kept) != 3 or len(detached) != 1:
        return None

    sprite_h = random.randint(5, 8)
    sprite_w = random.randint(8, 10)

    frac_c = random.uniform(0.4, 0.6)
    sprite_c0 = int(c_split) - int(frac_c * sprite_w)
    sprite_c0 = max(4, min(sprite_c0, int(w - 5 - sprite_w + 1)))
    if sprite_c0 > int(w - 5 - sprite_w + 1):
        return None
    if int(c_split) <= sprite_c0 or int(c_split) >= sprite_c0 + sprite_w - 1:
        return None

    frac_r = random.uniform(0.4, 0.6)
    sprite_r0 = int(r_split) - int(frac_r * sprite_h)
    sprite_r0 = max(4, min(sprite_r0, int(h - 5 - sprite_h + 1)))
    if sprite_r0 > int(h - 5 - sprite_h + 1):
        return None
    if int(r_split) <= sprite_r0 or int(r_split) >= sprite_r0 + sprite_h - 1:
        return None

    sprite_r1 = int(sprite_r0 + sprite_h - 1)
    sprite_c1 = int(sprite_c0 + sprite_w - 1)

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = int(sprite_r0) if hs == "T" else int(r_split) + 1
        r1 = int(r_split) - 1 if hs == "T" else int(sprite_r1)
        c0 = int(sprite_c0) if vs == "L" else int(c_split) + 1
        c1 = int(c_split) - 1 if vs == "L" else int(sprite_c1)
        return int(r0), int(r1), int(c0), int(c1)

    for reg in all_regions:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None
        if int((r1 - r0 + 1) * (c1 - c0 + 1)) < 3:
            return None

    n_v = 3
    n_h = 3
    n_split = int(n_v + n_h)
    area = int(sprite_h * sprite_w)
    min_piece_total = int(n_split + 3 * len(kept) + 4)
    target_total = max(int(min_piece_total), int(random.uniform(0.50, 0.60) * area))
    if target_total >= area:
        target_total = int(area - 1)
    if target_total < min_piece_total:
        return None

    budget_pieces = int(target_total - n_split)
    max_detached = int(min(8, budget_pieces - 3 * len(kept)))
    if max_detached < 4:
        return None
    detached_budget = int(random.randint(4, max_detached))
    budget_pieces -= detached_budget

    kept_budgets: list[int] = []
    remaining = int(budget_pieces)
    for idx, reg in enumerate(kept):
        r0, r1, c0, c1 = sub_bbox(*reg)
        max_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        min_left = int((len(kept) - idx - 1) * 3)
        if idx == len(kept) - 1:
            px = int(remaining)
        else:
            avg = int(remaining // (len(kept) - idx))
            lo_px = int(max(3, avg - 2))
            hi_px = int(min(max_area, avg + 2, remaining - min_left))
            if lo_px > hi_px:
                return None
            px = int(random.randint(lo_px, hi_px))
        px = int(max(3, min(px, max_area, remaining - min_left)))
        kept_budgets.append(int(px))
        remaining -= int(px)
    if remaining != 0 or any(px < 3 for px in kept_budgets):
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    for idx, reg in enumerate(kept):
        hs, vs = reg
        dh, dv = _pick_dest(hs, vs, edges, r_split, c_split)
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = int(r1 - r0 + 1)
        sub_w = int(c1 - c0 + 1)
        target_px = int(kept_budgets[idx])

        piece = _grow_piece_with_retry(sub_h, sub_w, target_px, min_h=1, min_w=2)
        if piece is None or len(piece) < target_px:
            return None

        ph, pw = _piece_dims(piece)
        inp_r0 = int(r1 - ph + 1) if hs == "T" else int(r0)
        inp_c0 = int(c1 - pw + 1) if vs == "L" else int(c0)
        inp_pts = frozenset((int(r + inp_r0), int(c + inp_c0)) for r, c in piece)

        for r, c in inp_pts:
            if r < 4 or r > int(h - 5) or c < 4 or c > int(w - 5):
                return None
            if (vs == "L" and c >= int(c_split)) or (vs == "R" and c <= int(c_split)):
                return None
            if (hs == "T" and r >= int(r_split)) or (hs == "B" and r <= int(r_split)):
                return None
        input_pieces[reg] = inp_pts

        inp_rs = [r for r, _ in inp_pts]
        inp_cs = [c for _, c in inp_pts]
        dr = int(-min(inp_rs)) if dh == "T" else int(h - 1 - max(inp_rs))
        dc = int(-min(inp_cs)) if dv == "L" else int(w - 1 - max(inp_cs))
        out_pts = frozenset((int(r + dr), int(c + dc)) for r, c in inp_pts)
        for r, c in out_pts:
            if (vs == "L" and c >= int(c_split)) or (vs == "R" and c <= int(c_split)):
                return None
            if (hs == "T" and r >= int(r_split)) or (hs == "B" and r <= int(r_split)):
                return None
    all_inp_fg: set[tuple[int, int]] = set()
    for pts in input_pieces.values():
        all_inp_fg |= set(pts)
    if not all_inp_fg:
        return None

    detached_reg = detached[0]
    dr0, dr1, dc0, dc1 = sub_bbox(*detached_reg)
    det_h = int(dr1 - dr0 + 1)
    det_w = int(dc1 - dc0 + 1)
    if det_h < 1 or det_w < 1:
        return None
    if det_h * det_w < detached_budget:
        return None

    detached_piece = _grow_piece_with_retry(det_h, det_w, detached_budget, min_h=2, min_w=2)
    if detached_piece is None or len(detached_piece) < detached_budget:
        return None

    dph, dpw = _piece_dims(detached_piece)
    dhs, dvs = detached_reg
    det_r_off = int(dr1 - dph + 1) if dhs == "T" else int(dr0)
    det_c_off = int(dc1 - dpw + 1) if dvs == "L" else int(dc0)
    detached_fg = {(int(r + det_r_off), int(c + det_c_off)) for r, c in detached_piece}

    for r, c in detached_fg:
        if r < 4 or r > int(h - 5) or c < 4 or c > int(w - 5):
            return None
        if (r, c) in all_inp_fg:
            return None
        if (dvs == "L" and c >= int(c_split)) or (dvs == "R" and c <= int(c_split)):
            return None
        if (dhs == "T" and r >= int(r_split)) or (dhs == "B" and r <= int(r_split)):
            return None
        if int(r) == int(r_split) or int(c) == int(c_split):
            return None

    region_cells: dict[tuple[str, str], set[tuple[int, int]]] = {reg: set(pts) for reg, pts in input_pieces.items()}
    region_cells[detached_reg] = set(detached_fg)
    bridge_reg, side_v_reg, side_h_reg, detached_reg2 = _bridge_regions(edges)
    if detached_reg2 != detached_reg:
        return None

    v_candidates: dict[tuple[str, str], list[int]] = {}
    h_candidates: dict[tuple[str, str], list[int]] = {}
    for reg, pts in region_cells.items():
        hs, vs = reg
        want_col = int(c_split - 1) if vs == "L" else int(c_split + 1)
        want_row = int(r_split - 1) if hs == "T" else int(r_split + 1)
        v_candidates[reg] = sorted({int(r) for r, c in pts if int(c) == int(want_col)})
        h_candidates[reg] = sorted({int(c) for r, c in pts if int(r) == int(want_row)})
        if not v_candidates[reg] or not h_candidates[reg]:
            return None

    v_rows = _pick_unique_positions([bridge_reg, side_v_reg, detached_reg], v_candidates, target_count=n_v)
    h_cols = _pick_unique_positions([bridge_reg, side_h_reg, detached_reg], h_candidates, target_count=n_h)
    if v_rows is None or h_cols is None:
        return None

    split_pixels: set[tuple[int, int]] = {(int(r), int(c_split)) for r in v_rows} | {(int(r_split), int(c)) for c in h_cols}

    combined = all_inp_fg | detached_fg | split_pixels
    fg_arr = np.array(list(combined), dtype=int)
    ah = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    aw = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if ah < 5 or ah > 8 or aw < 8 or aw > 10:
        return None

    mask = np.zeros((int(h), int(w)), dtype=int)
    for r, c in combined:
        mask[int(r), int(c)] = 1
    if not is_contiguous(mask):
        return None

    actual_density = len(combined) / float(ah * aw)
    if actual_density < 0.38 or actual_density > 0.62:
        return None

    inp = np.full((int(h), int(w)), int(bg), dtype=int)
    for r, c in combined:
        inp[int(r), int(c)] = int(fg)
    for cr, cc in ctrl_pts:
        inp[int(cr), int(cc)] = int(ctrl)

    combo = main(inp)

    metrics = _fusion_metrics(inp, combo)
    if metrics["rays_with_alt"] < len(ctrl_pts):
        return None
    if metrics["ray_alt_cells"] < len(ctrl_pts):
        return None
    if metrics["recolor_cells"] < int(detached_budget):
        return None
    if metrics["diff_vs_d5_only"] <= 0:
        return None
    if metrics["diff_vs_d2_only"] <= 0:
        return None
    if metrics["diff_vs_base"] <= metrics["diff_vs_d5_only"]:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
