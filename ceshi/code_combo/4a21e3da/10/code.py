# Combo variant of 4a21e3da (index 10)
# combo_spec: D5/0+D9/2
# Name: 未接壤区域着色+分割线像素变色

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


# Design analysis for the combo variant
# -------------------------------------
# concrete_trace:
#   - Keep the original geometry: one vertical split and one horizontal split,
#     induced by two border control pixels on perpendicular edges.
#   - Example with controls on top and right:
#     regions are TL / TR / BL / BR.
#     Original task:
#       * regions touching a control edge slide outward to their matching corner
#       * the unique non-touching region vanishes
#       * fg lying exactly on the split row/col stays fg in place
#     Combo task:
#       * touching regions still slide to corners unchanged
#       * the unique non-touching region no longer vanishes; it stays put and
#         recolors to a fixed extra color
#       * split-line fg no longer stays fg; it recolors to the control color,
#         so the cut becomes a continuous visible control-colored divider
#   - Visible output therefore always contains three foreground-like roles:
#       * moved fg pieces in corners
#       * stationary recolored detached piece near the split intersection
#       * ctrl-colored split markers on the cut line itself
# complexity_calibration:
#   - The D5 behavior only exists when there is a non-touching quadrant, so the
#     generator always uses exactly two perpendicular controls.
#   - Recolor uses one fixed target color to keep inference tractable from a few
#     examples; no per-instance color permutation is introduced.
#   - Each split keeps 2-3 anchor pixels so the ctrl-colored divider is obvious
#     in every sample, and the detached piece is forced to have at least 4 cells
#     so the recolor branch is also obvious.
# implicit_assumption_audit_for_main:
#   - Color inference still assumes the only non-bg border color is ctrl, and
#     the remaining non-bg input color is fg. Safe because recolor appears only
#     in outputs, never inputs.
#   - Ray drawing still uses input fg on the split lines to decide where the ray
#     stops. This remains valid because the input is unchanged.
#   - The old "split fg remain fg" block is no longer valid; it must write ctrl.
#   - The old "non-touching regions continue" block encoded erasure. In the
#     combo, that branch writes recolor cells in place instead.
#   - Because split-line fg turns into ctrl in the output, a generator cannot
#     reconstruct ray length from output fg anymore. It must use the known
#     split_pixels anchors directly, otherwise rays would be too short.
# placement_legality:
#   - Detached region is generated entirely inside the unique non-touching
#     quadrant, never on the split row/col and never on the border.
#   - Corner-slid kept pieces remain in their own half/quadrant, so they cannot
#     overwrite the ctrl divider or the stationary recolored piece.
#   - Input union of kept pieces + split pixels + detached piece must remain a
#     single 4-connected sprite, matching the base task.
# observability_and_cleanliness:
#   - Every generated sample exhibits both combo effects simultaneously:
#       * at least one recolored detached quadrant
#       * at least one ctrl-colored split marker line segment
#   - The three output roles occupy separated spatial zones: corners, center
#     detached quadrant, and split lines. This keeps the scene readable.
# new_main_assumptions_backed_by_generator:
#   - Exactly one vertical and one horizontal split are present.
#   - Each split line has at least one fg anchor pixel in the input.
#   - The detached region is unique and non-empty.
#   - Recolor target differs from bg, fg, and ctrl.


RECOLOR_COLOR = 3


def _pick_recolor(bg: int, fg: int, ctrl: int) -> int:
    if RECOLOR_COLOR not in {int(bg), int(fg), int(ctrl)}:
        return int(RECOLOR_COLOR)
    for cand in [4, 5, 6, 8, 9, 0]:
        if cand not in {int(bg), int(fg), int(ctrl)}:
            return int(cand)
    raise AssertionError("No recolor color available")


def _on_split(r: int, c: int, r_split: int | None, c_split: int | None) -> bool:
    return (c_split is not None and int(c) == int(c_split)) or (r_split is not None and int(r) == int(r_split))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut a sprite along border-control rays, slide touching pieces to corners,
    recolor the non-touching piece in place, and recolor split-line fg to ctrl.
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
    recolor = _pick_recolor(bg, fg, ctrl)

    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(ctrl))]

    E: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for r, c in ctrl_pts:
        if int(r) == 0:
            E.add("top")
            c_split = int(c)
        elif int(r) == int(H - 1):
            E.add("bottom")
            c_split = int(c)
        elif int(c) == 0:
            E.add("left")
            r_split = int(r)
        elif int(c) == int(W - 1):
            E.add("right")
            r_split = int(r)

    out = np.full((H, W), int(bg), dtype=int)

    for r, c in ctrl_pts:
        if int(r) == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, int(c)] == int(fg)).flatten()]
            if rows:
                stop = int(max(rows))
                for rr in range(0, int(stop) + 1):
                    if int(g[int(rr), int(c)]) == int(bg):
                        out[int(rr), int(c)] = int(ctrl)
        elif int(r) == int(H - 1):
            rows = [int(rr) for rr in np.argwhere(g[:, int(c)] == int(fg)).flatten()]
            if rows:
                stop = int(min(rows))
                for rr in range(int(H - 1), int(stop) - 1, -1):
                    if int(g[int(rr), int(c)]) == int(bg):
                        out[int(rr), int(c)] = int(ctrl)
        elif int(c) == 0:
            cols = [int(cc) for cc in np.argwhere(g[int(r), :] == int(fg)).flatten()]
            if cols:
                stop = int(max(cols))
                for cc in range(0, int(stop) + 1):
                    if int(g[int(r), int(cc)]) == int(bg):
                        out[int(r), int(cc)] = int(ctrl)
        elif int(c) == int(W - 1):
            cols = [int(cc) for cc in np.argwhere(g[int(r), :] == int(fg)).flatten()]
            if cols:
                stop = int(min(cols))
                for cc in range(int(W - 1), int(stop) - 1, -1):
                    if int(g[int(r), int(cc)]) == int(bg):
                        out[int(r), int(cc)] = int(ctrl)

    for r, c in ctrl_pts:
        out[int(r), int(c)] = int(ctrl)

    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if _on_split(r, c, r_split, c_split):
            out[int(r), int(c)] = int(ctrl)

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if _on_split(r, c, r_split, c_split):
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
            for r, c in pts:
                out[int(r), int(c)] = int(recolor)
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
                out[int(rr), int(cc)] = int(fg)

    return out


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    for _ in range(int(target_px) - 1):
        neighbors = set()
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = int(r + dr), int(c + dc)
                if (nr, nc) not in cells:
                    neighbors.add((nr, nc))
        valid = [
            (nr, nc)
            for nr, nc in neighbors
            if max(r for r, _ in cells | {(nr, nc)}) - min(r for r, _ in cells | {(nr, nc)}) + 1 <= int(max_h)
            and max(c for _, c in cells | {(nr, nc)}) - min(c for _, c in cells | {(nr, nc)}) + 1 <= int(max_w)
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


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2
    recolor = _pick_recolor(bg, fg, ctrl_color)

    ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]

    c_split = r_split = None
    ctrl_pts: list[tuple[int, int]] = []
    E = set(ctrl_edges)

    for edge in ctrl_edges:
        if edge in ("top", "bottom"):
            c_split = random.randint(int(W * 0.4), int(W * 0.6))
            ctrl_pts.append((0 if edge == "top" else H - 1, int(c_split)))
        else:
            r_split = random.randint(int(H * 0.4), int(H * 0.6))
            ctrl_pts.append((int(r_split), 0 if edge == "left" else W - 1))

    assert c_split is not None and r_split is not None

    def touches_E(hs: str, vs: str) -> bool:
        te = set()
        if hs == "T":
            te.add("top")
        if hs == "B":
            te.add("bottom")
        if vs == "L":
            te.add("left")
        if vs == "R":
            te.add("right")
        return bool(te & E)

    all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    kept = [reg for reg in all_regions if touches_E(*reg)]
    detached = [reg for reg in all_regions if reg not in kept]
    if len(kept) != 3 or len(detached) != 1:
        return None

    sprite_h = random.randint(5, 8)
    sprite_w = random.randint(8, 10)

    frac_c = random.uniform(0.4, 0.6)
    sprite_c0 = int(c_split) - int(frac_c * sprite_w)
    sprite_c0 = max(4, min(sprite_c0, W - 5 - sprite_w + 1))
    if sprite_c0 > W - 5 - sprite_w + 1:
        return None
    if int(c_split) <= sprite_c0 or int(c_split) >= sprite_c0 + sprite_w - 1:
        return None

    frac_r = random.uniform(0.4, 0.6)
    sprite_r0 = int(r_split) - int(frac_r * sprite_h)
    sprite_r0 = max(4, min(sprite_r0, H - 5 - sprite_h + 1))
    if sprite_r0 > H - 5 - sprite_h + 1:
        return None
    if int(r_split) <= sprite_r0 or int(r_split) >= sprite_r0 + sprite_h - 1:
        return None

    sprite_r1 = sprite_r0 + sprite_h - 1
    sprite_c1 = sprite_c0 + sprite_w - 1

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = sprite_r0 if hs == "T" else int(r_split) + 1
        r1 = int(r_split) - 1 if hs == "T" else sprite_r1
        c0 = sprite_c0 if vs == "L" else int(c_split) + 1
        c1 = int(c_split) - 1 if vs == "L" else sprite_c1
        return int(r0), int(r1), int(c0), int(c1)

    for reg in all_regions:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0 or (r1 - r0 + 1) * (c1 - c0 + 1) < 3:
            return None

    split_pixels: set[tuple[int, int]] = set()
    n_v = random.randint(2, 3)
    rows = [r for r in range(sprite_r0, sprite_r1 + 1) if r != int(r_split)]
    if len(rows) < n_v:
        return None
    for r in random.sample(rows, n_v):
        split_pixels.add((int(r), int(c_split)))

    n_h = random.randint(2, 3)
    cols = [c for c in range(sprite_c0, sprite_c1 + 1) if c != int(c_split)]
    if len(cols) < n_h:
        return None
    for c in random.sample(cols, n_h):
        split_pixels.add((int(r_split), int(c)))

    n_split = len(split_pixels)
    area = sprite_h * sprite_w
    min_total = n_split + 3 * len(kept) + 4
    target_total = max(min_total, int(random.uniform(0.50, 0.60) * area))
    if target_total >= area:
        target_total = area - 1
    if target_total < min_total:
        return None

    budget_pieces = target_total - n_split
    max_detached = min(8, budget_pieces - 3 * len(kept))
    if max_detached < 4:
        return None
    detached_budget = random.randint(4, max_detached)
    budget_pieces -= detached_budget

    kept_budgets: list[int] = []
    remaining = budget_pieces
    for idx, reg in enumerate(kept):
        r0, r1, c0, c1 = sub_bbox(*reg)
        max_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        min_left = (len(kept) - idx - 1) * 3
        if idx == len(kept) - 1:
            px = remaining
        else:
            avg = remaining // (len(kept) - idx)
            lo_px = max(3, avg - 2)
            hi_px = min(max_area, avg + 2, remaining - min_left)
            if lo_px > hi_px:
                return None
            px = random.randint(lo_px, hi_px)
        px = max(3, min(px, max_area, remaining - min_left))
        kept_budgets.append(int(px))
        remaining -= int(px)

    if remaining != 0 or any(px < 3 for px in kept_budgets):
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}

    for idx, reg in enumerate(kept):
        hs, vs = reg
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = r1 - r0 + 1
        sub_w = c1 - c0 + 1
        target_px = kept_budgets[idx]

        piece = _grow_piece_with_retry(sub_h, sub_w, target_px, min_h=1, min_w=2)
        if piece is None or len(piece) < target_px:
            return None

        ph, pw = _piece_dims(piece)
        inp_r0 = (r1 - ph + 1) if hs == "T" else r0
        inp_c0 = (c1 - pw + 1) if vs == "L" else c0
        inp_pts = frozenset((int(r + inp_r0), int(c + inp_c0)) for r, c in piece)

        for r, c in inp_pts:
            if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                return None
            if (vs == "L" and c >= int(c_split)) or (vs == "R" and c <= int(c_split)):
                return None
            if (hs == "T" and r >= int(r_split)) or (hs == "B" and r <= int(r_split)):
                return None

        input_pieces[reg] = inp_pts

        inp_rs = [r for r, _ in inp_pts]
        inp_cs = [c for _, c in inp_pts]
        dr = -min(inp_rs) if hs == "T" else (H - 1 - max(inp_rs))
        dc = -min(inp_cs) if vs == "L" else (W - 1 - max(inp_cs))
        out_pts = frozenset((int(r + dr), int(c + dc)) for r, c in inp_pts)

        for r, c in out_pts:
            if (vs == "L" and c >= int(c_split)) or (vs == "R" and c <= int(c_split)):
                return None
            if (hs == "T" and r >= int(r_split)) or (hs == "B" and r <= int(r_split)):
                return None
        output_pieces[reg] = out_pts

    all_inp_fg: set[tuple[int, int]] = set(split_pixels)
    for pts in input_pieces.values():
        all_inp_fg |= set(pts)
    if not all_inp_fg:
        return None

    det_reg = detached[0]
    dr0, dr1, dc0, dc1 = sub_bbox(*det_reg)
    det_h = dr1 - dr0 + 1
    det_w = dc1 - dc0 + 1
    if det_h * det_w < detached_budget:
        return None

    det_piece = _grow_piece_with_retry(det_h, det_w, detached_budget, min_h=2, min_w=2)
    if det_piece is None or len(det_piece) < detached_budget:
        return None

    dph, dpw = _piece_dims(det_piece)
    dhs, dvs = det_reg
    det_r_off = (dr1 - dph + 1) if dhs == "T" else dr0
    det_c_off = (dc1 - dpw + 1) if dvs == "L" else dc0
    detached_fg = {(int(r + det_r_off), int(c + det_c_off)) for r, c in det_piece}

    for r, c in detached_fg:
        if r < 4 or r > H - 5 or c < 4 or c > W - 5:
            return None
        if (r, c) in all_inp_fg:
            return None
        if _on_split(r, c, int(r_split), int(c_split)):
            return None
        if (dvs == "L" and c >= int(c_split)) or (dvs == "R" and c <= int(c_split)):
            return None
        if (dhs == "T" and r >= int(r_split)) or (dhs == "B" and r <= int(r_split)):
            return None

    combined = all_inp_fg | detached_fg
    fg_arr = np.array(list(combined))
    ah = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    aw = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if ah < 5 or ah > 8 or aw < 8 or aw > 10:
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in combined:
        mask[int(r), int(c)] = 1
    if not is_contiguous(mask):
        return None

    inp = np.full((H, W), int(bg), dtype=int)
    for r, c in combined:
        inp[int(r), int(c)] = int(fg)
    for cr, cc in ctrl_pts:
        inp[int(cr), int(cc)] = int(ctrl_color)

    out = np.full((H, W), int(bg), dtype=int)
    for pts in output_pieces.values():
        for r, c in pts:
            out[int(r), int(c)] = int(fg)
    for r, c in detached_fg:
        out[int(r), int(c)] = int(recolor)

    for cr, cc in ctrl_pts:
        if cr == 0:
            rows = [r for r, c in split_pixels if c == cc]
            if rows:
                for r in range(max(rows) + 1):
                    if out[int(r), int(cc)] == int(bg):
                        out[int(r), int(cc)] = int(ctrl_color)
        elif cr == H - 1:
            rows = [r for r, c in split_pixels if c == cc]
            if rows:
                for r in range(H - 1, min(rows) - 1, -1):
                    if out[int(r), int(cc)] == int(bg):
                        out[int(r), int(cc)] = int(ctrl_color)
        elif cc == 0:
            cols = [c for r, c in split_pixels if r == cr]
            if cols:
                for c in range(max(cols) + 1):
                    if out[int(cr), int(c)] == int(bg):
                        out[int(cr), int(c)] = int(ctrl_color)
        elif cc == W - 1:
            cols = [c for r, c in split_pixels if r == cr]
            if cols:
                for c in range(W - 1, min(cols) - 1, -1):
                    if out[int(cr), int(c)] == int(bg):
                        out[int(cr), int(c)] = int(ctrl_color)

    for r, c in split_pixels:
        out[int(r), int(c)] = int(ctrl_color)
    for cr, cc in ctrl_pts:
        out[int(cr), int(cc)] = int(ctrl_color)

    actual_density = len(combined) / float(ah * aw)
    if actual_density < 0.38 or actual_density > 0.62:
        return None

    if not np.any(out == int(recolor)):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
