# Combo variant of 4a21e3da (index 18)
# combo_spec: D12/1+D16/0
# Name: 区域按象限着色+消失区域作为干扰项

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


"""
Design analysis for the combo D12/1 + D16/0.

a) Concrete behavior and trace
   Base task:
   - A monochrome orange sprite sits near the center.
   - One vertical split and one horizontal split are induced by two red border controls.
   - Orange cells on the split row/col stay in place.
   - Each off-line region touching a control edge slides to its matching corner.
   - The unique non-touching quadrant would originally disappear.

   Combo task:
   - Keep the same input structure: bg=1, ctrl=2, sprite=7.
   - After cutting, each moved region is recolored by its destination corner:
       TL -> 3, TR -> 4, BL -> 6, BR -> 8.
   - The non-touching quadrant no longer disappears; it stays at its original coordinates in orange.
   - Split-line pixels also stay orange, so the output contains two orange semantics:
       1) stationary split pixels on the cut lines
       2) stationary distractor pixels in the non-touching quadrant

   Example trace for controls on top + left:
   - TL, TR, BL touch a control edge, so they slide to TL/TR/BL corners and recolor to 3/4/6.
   - BR touches neither top nor left, so instead of vanishing it stays in the center as orange.
   - The solver must distinguish "moved and recolored" from "stationary and still orange".

b) Complexity calibration
   - The generator is fixed to the two-control regime so every sample visibly contains both combo effects.
   - Only four destination colors are used, one per corner, so the mapping is repeated and learnable.
   - The static distractor is kept at 3-7 pixels: large enough to notice, small enough not to dominate.

c) Implicit assumption audit against the base solver
   - Border non-bg color still uniquely identifies the control color because recolored regions occur only in output.
   - Input sprite is still monochrome orange, so fg detection from input remains valid.
   - The base assumption "non-touching region disappears" is now false; solver must write it back in place.
   - The base assumption "all moved regions stay orange" is also false; moved writeout must use corner colors.
   - Rays are still drawn only through background cells and still rely only on split-line orange cells.
   - Static distractor cells must avoid split row/col, otherwise they would be confused with split-line cells.

d) Placement legality
   - Generator keeps one vertical and one horizontal split only.
   - Kept regions are placed strictly on their side of the split, and their corner-slid outputs remain on that side.
   - The static distractor is generated only inside the unique non-touching quadrant, never on the split row/col.
   - Input union of kept pieces + split pixels + distractor is forced to be contiguous.
   - Output can be disconnected, which is intended.

e) Visual observability
   - Every sample shows three recolored corner pieces plus one orange central distractor.
   - The distractor remains near the original sprite location while moved regions jump to corners, making the
     "moved vs stationary" distinction visible from a few examples.
   - Because the kept regions go to different corners in different samples, the position->color mapping is inferable.

f) Visual cleanliness
   - Only one semantic layer occupies each cell in output: background, red ray, orange stationary pixel, or one
     recolored moved region.
   - Margins and original bbox limits from the base task are preserved so the center distractor and corner pieces
     stay well separated.

g) New solver assumptions and generator guarantees
   - Solver assumes a unique static quadrant exists; generator guarantees two perpendicular controls.
   - Solver assumes destination corner colors are determined structurally by target corner, not by input color id.
   - Solver assumes no recolored moved piece lands on a split line; generator enforces the inherited side constraints.
"""


_CORNER_COLORS: dict[tuple[str, str], int] = {
    ("T", "L"): 3,
    ("T", "R"): 4,
    ("B", "L"): 6,
    ("B", "R"): 8,
}


def _corner_color(h_side: str, v_side: str) -> int:
    return int(_CORNER_COLORS[(str(h_side), str(v_side))])


def _detect_structure(
    g: np.ndarray,
) -> tuple[int, int, int, int, int, list[tuple[int, int]], set[str], int | None, int | None]:
    H, W = g.shape
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

    E: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for r, c in ctrl_pts:
        if r == 0:
            E.add("top")
            c_split = c
        elif r == H - 1:
            E.add("bottom")
            c_split = c
        elif c == 0:
            E.add("left")
            r_split = r
        elif c == W - 1:
            E.add("right")
            r_split = r

    assert r_split is not None and c_split is not None
    return H, W, bg, fg, ctrl, ctrl_pts, E, r_split, c_split


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

    return str(dh), str(dv)


def _draw_rays(
    out: np.ndarray,
    g: np.ndarray,
    bg: int,
    fg: int,
    ctrl: int,
    ctrl_pts: list[tuple[int, int]],
) -> None:
    H, W = g.shape
    for r, c in ctrl_pts:
        if r == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                stop = int(max(rows))
                for rr in range(0, stop + 1):
                    if g[rr, c] == bg:
                        out[rr, c] = ctrl
        elif r == H - 1:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                stop = int(min(rows))
                for rr in range(H - 1, stop - 1, -1):
                    if g[rr, c] == bg:
                        out[rr, c] = ctrl
        elif c == 0:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                stop = int(max(cols))
                for cc in range(0, stop + 1):
                    if g[r, cc] == bg:
                        out[r, cc] = ctrl
        elif c == W - 1:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                stop = int(min(cols))
                for cc in range(W - 1, stop - 1, -1):
                    if g[r, cc] == bg:
                        out[r, cc] = ctrl

    for r, c in ctrl_pts:
        out[r, c] = ctrl


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - touching regions slide to corners and recolor by destination corner
    - split-line pixels stay orange in place
    - the unique non-touching quadrant stays in place as orange distractor
    """
    g = np.array(grid, dtype=int)
    H, W, bg, fg, ctrl, ctrl_pts, E, r_split, c_split = _detect_structure(g)

    out = np.full((H, W), bg, dtype=int)
    _draw_rays(out, g, bg, fg, ctrl, ctrl_pts)

    for r, c in np.argwhere(g == fg):
        r = int(r)
        c = int(c)
        if c == c_split or r == r_split:
            out[r, c] = fg

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == fg):
        r = int(r)
        c = int(c)
        if c == c_split or r == r_split:
            continue

        hs = "T" if r < r_split else "B"
        vs = "L" if c < c_split else "R"
        regions.setdefault((hs, vs), []).append((r, c))

    for (hs, vs), pts in regions.items():
        if not (_touches_edges(hs, vs) & E):
            for r, c in pts:
                out[r, c] = fg
            continue

        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        moved_color = _corner_color(dh, dv)
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        rmin, rmax = int(min(rs)), int(max(rs))
        cmin, cmax = int(min(cs)), int(max(cs))
        dr = -rmin if dh == "T" else (H - 1) - rmax
        dc = -cmin if dv == "L" else (W - 1) - cmax

        for r, c in pts:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = moved_color

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


def _behavior_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    H, W, bg, fg, ctrl, ctrl_pts, E, r_split, c_split = _detect_structure(inp)
    moved_colors = {3, 4, 6, 8}
    static_orange_off_split = 0
    split_orange = 0
    for r in range(H):
        for c in range(W):
            if out[r, c] != fg:
                continue
            if r == r_split or c == c_split:
                split_orange += 1
            else:
                static_orange_off_split += 1
    visible_moved_colors = sum(int(np.any(out == col)) for col in moved_colors)
    return {
        "static_orange_off_split": int(static_orange_off_split),
        "split_orange": int(split_orange),
        "visible_moved_colors": int(visible_moved_colors),
        "num_ctrl": int(len(ctrl_pts)),
    }


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2

    ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]
    E = set(ctrl_edges)
    c_split = random.randint(int(W * 0.4), int(W * 0.6))
    r_split = random.randint(int(H * 0.4), int(H * 0.6))
    ctrl_pts = [
        (0 if ctrl_edges[0] == "top" else H - 1, c_split),
        (r_split, 0 if ctrl_edges[1] == "left" else W - 1),
    ]

    all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    kept = [reg for reg in all_regions if _touches_edges(*reg) & E]
    static_regs = [reg for reg in all_regions if reg not in kept]
    if len(static_regs) != 1 or len(kept) != 3:
        return None

    sprite_h = random.randint(5, 8)
    sprite_w = random.randint(8, 10)

    frac_c = random.uniform(0.4, 0.6)
    sprite_c0 = c_split - int(frac_c * sprite_w)
    sprite_c0 = max(4, min(sprite_c0, W - 5 - sprite_w + 1))
    if sprite_c0 > W - 5 - sprite_w + 1:
        return None
    if c_split <= sprite_c0 or c_split >= sprite_c0 + sprite_w - 1:
        return None

    frac_r = random.uniform(0.4, 0.6)
    sprite_r0 = r_split - int(frac_r * sprite_h)
    sprite_r0 = max(4, min(sprite_r0, H - 5 - sprite_h + 1))
    if sprite_r0 > H - 5 - sprite_h + 1:
        return None
    if r_split <= sprite_r0 or r_split >= sprite_r0 + sprite_h - 1:
        return None

    sprite_r1 = sprite_r0 + sprite_h - 1
    sprite_c1 = sprite_c0 + sprite_w - 1

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = sprite_r0 if hs == "T" else r_split + 1
        r1 = r_split - 1 if hs == "T" else sprite_r1
        c0 = sprite_c0 if vs == "L" else c_split + 1
        c1 = c_split - 1 if vs == "L" else sprite_c1
        return r0, r1, c0, c1

    for reg in all_regions:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None

    split_pixels: set[tuple[int, int]] = set()
    n = random.randint(2, 4)
    rows = [r for r in range(sprite_r0, sprite_r1 + 1) if r != r_split]
    if len(rows) < n:
        return None
    for r in random.sample(rows, n):
        split_pixels.add((r, c_split))

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
    budget_pieces = target_total - n_split

    static_budget = random.randint(3, min(7, max(3, budget_pieces // 3)))
    budget_pieces -= static_budget
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
    output_pieces: dict[tuple[str, str], tuple[str, str, frozenset[tuple[int, int]]]] = {}

    for idx, reg in enumerate(kept):
        hs, vs = reg
        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = r1 - r0 + 1
        sub_w = c1 - c0 + 1
        piece = _grow_piece(sub_h, sub_w, piece_budgets[idx])
        if len(piece) < 3:
            return None

        ph = max(r for r, _ in piece) + 1
        pw = max(c for _, c in piece) + 1
        inp_r0 = (r1 - ph + 1) if hs == "T" else r0
        inp_c0 = (c1 - pw + 1) if vs == "L" else c0
        inp_pts = frozenset((r + inp_r0, c + inp_c0) for r, c in piece)

        for r, c in inp_pts:
            if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                return None
            if (hs == "T" and r >= r_split) or (hs == "B" and r <= r_split):
                return None
            if (vs == "L" and c >= c_split) or (vs == "R" and c <= c_split):
                return None

        input_pieces[reg] = inp_pts

        inp_rs = [r for r, _ in inp_pts]
        inp_cs = [c for _, c in inp_pts]
        dr = -min(inp_rs) if dh == "T" else (H - 1 - max(inp_rs))
        dc = -min(inp_cs) if dv == "L" else (W - 1 - max(inp_cs))
        out_pts = frozenset((r + dr, c + dc) for r, c in inp_pts)
        for r, c in out_pts:
            if (hs == "T" and r >= r_split) or (hs == "B" and r <= r_split):
                return None
            if (vs == "L" and c >= c_split) or (vs == "R" and c <= c_split):
                return None
        output_pieces[reg] = (dh, dv, out_pts)

    all_inp_fg = set().union(*input_pieces.values()) if input_pieces else set()
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

    static_reg = static_regs[0]
    vhs, vvs = static_reg
    vr0, vr1, vc0, vc1 = sub_bbox(vhs, vvs)
    vh, vw = vr1 - vr0 + 1, vc1 - vc0 + 1
    max_px = min(static_budget, vh * vw)
    if max_px < 3:
        return None
    static_piece = _grow_piece(vh, vw, max_px)
    if len(static_piece) < 3:
        return None

    pph = max(r for r, _ in static_piece) + 1
    ppw = max(c for _, c in static_piece) + 1
    vr_off = (vr1 - pph + 1) if vhs == "T" else vr0
    vc_off = (vc1 - ppw + 1) if vvs == "L" else vc0
    static_fg = {(r + vr_off, c + vc_off) for r, c in static_piece}
    for r, c in static_fg:
        if r < 4 or r > H - 5 or c < 4 or c > W - 5:
            return None
        if (r, c) in all_inp_fg:
            return None
        if r == r_split or c == c_split:
            return None
    if len(static_fg) < 3:
        return None

    combined = all_inp_fg | static_fg
    mask2 = np.zeros((H, W), dtype=int)
    for r, c in combined:
        mask2[r, c] = 1
    if not is_contiguous(mask2):
        return None

    inp = np.full((H, W), bg, dtype=int)
    for r, c in combined:
        inp[r, c] = fg
    for cr, cc in ctrl_pts:
        inp[cr, cc] = ctrl_color

    out = np.full((H, W), bg, dtype=int)
    for dh, dv, pts in output_pieces.values():
        piece_color = _corner_color(dh, dv)
        for r, c in pts:
            out[r, c] = piece_color
    for r, c in split_pixels:
        out[r, c] = fg
    for r, c in static_fg:
        out[r, c] = fg

    for cr, cc in ctrl_pts:
        if cr == 0:
            fg_rows = [r for r in range(H) if out[r, cc] == fg]
            if fg_rows:
                for r in range(max(fg_rows) + 1):
                    if out[r, cc] == bg:
                        out[r, cc] = ctrl_color
        elif cr == H - 1:
            fg_rows = [r for r in range(H) if out[r, cc] == fg]
            if fg_rows:
                for r in range(H - 1, min(fg_rows) - 1, -1):
                    if out[r, cc] == bg:
                        out[r, cc] = ctrl_color
        elif cc == 0:
            fg_cols = [c for c in range(W) if out[cr, c] == fg]
            if fg_cols:
                for c in range(max(fg_cols) + 1):
                    if out[cr, c] == bg:
                        out[cr, c] = ctrl_color
        elif cc == W - 1:
            fg_cols = [c for c in range(W) if out[cr, c] == fg]
            if fg_cols:
                for c in range(W - 1, min(fg_cols) - 1, -1):
                    if out[cr, c] == bg:
                        out[cr, c] = ctrl_color
    for cr, cc in ctrl_pts:
        out[cr, cc] = ctrl_color

    fg_all = np.array(list(combined))
    final_h = int(fg_all[:, 0].max() - fg_all[:, 0].min() + 1)
    final_w = int(fg_all[:, 1].max() - fg_all[:, 1].min() + 1)
    actual_density = len(combined) / (final_h * final_w)
    if actual_density < 0.38 or actual_density > 0.62:
        return None

    metrics = _behavior_metrics(inp, out)
    if metrics["num_ctrl"] != 2:
        return None
    if metrics["static_orange_off_split"] < 3:
        return None
    if metrics["visible_moved_colors"] < 3:
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
