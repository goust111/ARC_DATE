# Combo variant of 4a21e3da (index 13)
# combo_spec: D6/0+D14/1
# Name: 裁剪到有效区域+矩形网格
#
# Design analysis for the fused variant
# a) Concretization / behavior trace
#    - Base task: a centered fg sprite is cut by one border-origin ray (or one
#      vertical plus one horizontal ray). Fg on the split line stays in place;
#      off-line fg is partitioned into rectangular regions. Regions touching a
#      control edge slide to the corresponding corner; non-touching regions vanish.
#    - This combo keeps that mechanics but changes two stages together:
#      1. The input canvas is now obviously rectangular instead of square.
#         Examples are around 12-14 x 19-22 or the transposed tall version.
#         The sprite bbox is also asymmetric: wide sprites on wide boards and
#         tall sprites on tall boards. This makes the split visibly asymmetric.
#      2. After the usual corner-slide output is assembled, delete every outer
#         all-bg row/column and keep only the minimal non-bg rectangle.
#    - Concrete trace: on a 12x20 board with a top control, the split is a
#      vertical column through a short-wide central sprite. Left/right pieces
#      slide to the top-left / top-right corners, split-line fg stays central,
#      then the blank bottom rows are cropped away. On a 20x12 board with a
#      left control, top/bottom pieces slide to the top-left / bottom-left
#      corners and the blank right-side columns are cropped away.
# b) Complexity calibration
#    - Two-control cases were intentionally removed from the generator. When the
#      original task uses both a vertical and a horizontal split, surviving
#      pieces typically touch all four borders, so tight-crop often becomes
#      invisible. Restricting generation to one control keeps the crop rule
#      observable while preserving the base reasoning chain.
#    - Variety is still sufficient: four control-edge orientations remain, and
#      the board can be wide or tall. The same fused rule repeats across those
#      low-cardinality cases, so a human can infer it from a few examples.
# c) Hidden assumptions audit
#    - `main()` already uses `H, W = grid.shape`; no square-only arithmetic is
#      present, so the cut and slide logic itself survives rectangular boards.
#    - Border-color detection assumes the control color is the only non-bg color
#      on the outer frame. The generator keeps the fg sprite strictly away from
#      the frame, so that assumption still holds.
#    - Region classification assumes at most one vertical and one horizontal
#      split. The generator still emits exactly one control, so that assumption
#      remains stronger than needed.
#    - Tight crop is appended after the full uncropped output is built. This is
#      safe because control/fg inference and all translations happen in input
#      coordinates before cropping is applied.
# d) Placement legality
#    - The generator builds a single connected sprite by forcing both kept
#      pieces to touch one contiguous split-line segment. Every placement is
#      checked for bounds, overlap, bbox shape, density, and connectivity.
# e) Visual observability
#    - Every generated input is non-square, and every accepted output is smaller
#      than the input along at least one axis. That makes both D14 and D6
#      visible from the pixel grids alone.
# f) Visual cleanliness
#    - Only the original three colors are used. The sprite stays separated from
#      the border except for the control pixel, so rays, split-line fg, and the
#      slid corner pieces remain easy to parse.
# g) Inferability of changes
#    - No new color mapping is introduced. The only new inference burden is:
#      "same cut-and-slide rule as before, but on a rectangular board, then crop
#      away all empty margins in the output." That is structurally supported by
#      repeated geometry rather than hidden numeric/color conventions.

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


def _tight_crop_nonbg(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != int(bg))
    if coords.size == 0:
        return np.array(grid, dtype=int, copy=True)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.array(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int, copy=True)


def _solve_uncropped(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    nonbg = colors_present(g, exclude={bg})
    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(int(bg))
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != int(ctrl)))

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


def main(grid: np.ndarray) -> np.ndarray:
    uncropped = _solve_uncropped(grid)
    bg = int(get_bg(uncropped))
    return _tight_crop_nonbg(uncropped, bg)


def _grow_piece(max_h: int, max_w: int, target_px: int):
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


def _piece_dims(piece: frozenset[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in piece) + 1, max(c for _, c in piece) + 1


def _bbox_dims(points: set[tuple[int, int]]) -> tuple[int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return int(max(rs) - min(rs) + 1), int(max(cs) - min(cs) + 1)


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


def _pick_dest(E: set[str], r_split: int | None, c_split: int | None, hs: str, vs: str) -> tuple[str, str]:
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


def _slide_to_corner(pts: set[tuple[int, int]], H: int, W: int, dh: str, dv: str) -> set[tuple[int, int]]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    dr = -rmin if dh == "T" else (H - 1 - rmax)
    dc = -cmin if dv == "L" else (W - 1 - cmax)
    return {(r + dr, c + dc) for r, c in pts}


def _make_piece(max_h: int, max_w: int, target_px: int, min_h: int = 1, min_w: int = 1):
    for _ in range(48):
        piece = _grow_piece(max_h, max_w, target_px)
        if len(piece) != target_px:
            continue
        ph, pw = _piece_dims(piece)
        if ph >= min_h and pw >= min_w:
            return piece
    return None


def _choose_rect_dims() -> tuple[int, int]:
    if random.random() < 0.5:
        return random.randint(12, 14), random.randint(19, 22)
    return random.randint(19, 22), random.randint(12, 14)


def _choose_piece_budgets(total: int, cap_a: int, cap_b: int) -> tuple[int, int] | None:
    if total < 6 or cap_a < 3 or cap_b < 3:
        return None
    ideal_a = int(round(total * cap_a / (cap_a + cap_b)))
    lo = max(3, ideal_a - 3, total - cap_b)
    hi = min(cap_a, ideal_a + 3, total - 3)
    if lo > hi:
        return None
    a = random.randint(lo, hi)
    b = total - a
    if 3 <= a <= cap_a and 3 <= b <= cap_b:
        return a, b
    return None


def _gen_attempt():
    H, W = _choose_rect_dims()
    bg, fg, ctrl_color = 1, 7, 2

    ctrl_edge = random.choice(["top", "bottom", "left", "right"])
    E = {ctrl_edge}
    c_split: int | None = None
    r_split: int | None = None
    if ctrl_edge in ("top", "bottom"):
        c_split = random.randint(int(W * 0.35), int(W * 0.65))
        ctrl_pt = (0 if ctrl_edge == "top" else H - 1, c_split)
    else:
        r_split = random.randint(int(H * 0.35), int(H * 0.65))
        ctrl_pt = (r_split, 0 if ctrl_edge == "left" else W - 1)

    if H < W:
        sprite_h = random.randint(4, 5)
        sprite_w = random.randint(8, 11)
        target_h_range = (4, 6)
        target_w_range = (8, 11)
    else:
        sprite_h = random.randint(8, 11)
        sprite_w = random.randint(4, 5)
        target_h_range = (8, 11)
        target_w_range = (4, 6)

    margin_r = 2 if H <= 14 else 4
    margin_c = 2 if W <= 14 else 4

    if c_split is not None:
        c0_lo = margin_c
        c0_hi = W - margin_c - sprite_w
        if c0_lo > c0_hi:
            return None
        frac = random.uniform(0.35, 0.65)
        sprite_c0 = c_split - int(round(frac * (sprite_w - 1)))
        sprite_c0 = max(c0_lo, min(sprite_c0, c0_hi))
        sprite_c1 = sprite_c0 + sprite_w - 1
        left_w = c_split - sprite_c0
        right_w = sprite_c1 - c_split
        if left_w < 2 or right_w < 2:
            return None

        r0_lo = margin_r
        r0_hi = H - margin_r - sprite_h
        if r0_lo > r0_hi:
            return None
        center_r0 = (H - sprite_h) // 2
        sprite_r0 = max(r0_lo, min(center_r0 + random.randint(-1, 1), r0_hi))
        sprite_r1 = sprite_r0 + sprite_h - 1
    else:
        r0_lo = margin_r
        r0_hi = H - margin_r - sprite_h
        if r0_lo > r0_hi:
            return None
        frac = random.uniform(0.35, 0.65)
        sprite_r0 = r_split - int(round(frac * (sprite_h - 1)))
        sprite_r0 = max(r0_lo, min(sprite_r0, r0_hi))
        sprite_r1 = sprite_r0 + sprite_h - 1
        top_h = r_split - sprite_r0
        bot_h = sprite_r1 - r_split
        if top_h < 2 or bot_h < 2:
            return None

        c0_lo = margin_c
        c0_hi = W - margin_c - sprite_w
        if c0_lo > c0_hi:
            return None
        center_c0 = (W - sprite_w) // 2
        sprite_c0 = max(c0_lo, min(center_c0 + random.randint(-1, 1), c0_hi))
        sprite_c1 = sprite_c0 + sprite_w - 1

    split_len = random.randint(2, min(4, sprite_h if c_split is not None else sprite_w))
    area = sprite_h * sprite_w
    target_total = int(round(random.uniform(0.40, 0.58) * area))
    budget_pieces = target_total - split_len
    if budget_pieces < 6:
        return None

    input_pieces: dict[tuple[str, str], set[tuple[int, int]]] = {}

    if c_split is not None:
        left_w = c_split - sprite_c0
        right_w = sprite_c1 - c_split
        caps = (sprite_h * left_w, sprite_h * right_w)
        budgets = _choose_piece_budgets(budget_pieces, caps[0], caps[1])
        if budgets is None:
            return None
        left_px, right_px = budgets

        left_piece = _make_piece(sprite_h, left_w, left_px, min_h=max(2, min(sprite_h, target_h_range[0])), min_w=max(2, left_w - 1))
        right_piece = _make_piece(sprite_h, right_w, right_px, min_h=max(2, min(sprite_h, target_h_range[0])), min_w=max(2, right_w - 1))
        if left_piece is None or right_piece is None:
            return None

        lph, lpw = _piece_dims(left_piece)
        rph, rpw = _piece_dims(right_piece)
        left_touch_rows = sorted(r for r, c in left_piece if c == lpw - 1)
        right_touch_rows = sorted(r for r, c in right_piece if c == 0)
        left_key = random.choice(left_touch_rows)
        right_key = random.choice(right_touch_rows)

        link_lo = max(sprite_r0 + left_key, sprite_r0 + right_key)
        link_hi = min(sprite_r1 - lph + 1 + left_key, sprite_r1 - rph + 1 + right_key)
        if link_lo > link_hi:
            return None
        link_r = random.randint(link_lo, link_hi)

        left_r0 = link_r - left_key
        right_r0 = link_r - right_key
        left_c0 = c_split - lpw
        right_c0 = c_split + 1
        left_pts = {(r + left_r0, c + left_c0) for r, c in left_piece}
        right_pts = {(r + right_r0, c + right_c0) for r, c in right_piece}

        seg_lo = max(sprite_r0, link_r - split_len + 1)
        seg_hi = min(link_r, sprite_r1 - split_len + 1)
        if seg_lo > seg_hi:
            return None
        seg_r0 = random.randint(seg_lo, seg_hi)
        split_pixels = {(r, c_split) for r in range(seg_r0, seg_r0 + split_len)}

        input_pieces[("A", "L")] = left_pts
        input_pieces[("A", "R")] = right_pts
    else:
        top_h = r_split - sprite_r0
        bot_h = sprite_r1 - r_split
        caps = (top_h * sprite_w, bot_h * sprite_w)
        budgets = _choose_piece_budgets(budget_pieces, caps[0], caps[1])
        if budgets is None:
            return None
        top_px, bot_px = budgets

        top_piece = _make_piece(top_h, sprite_w, top_px, min_h=max(2, top_h - 1), min_w=max(2, min(sprite_w, target_w_range[0])))
        bot_piece = _make_piece(bot_h, sprite_w, bot_px, min_h=max(2, bot_h - 1), min_w=max(2, min(sprite_w, target_w_range[0])))
        if top_piece is None or bot_piece is None:
            return None

        tph, tpw = _piece_dims(top_piece)
        bph, bpw = _piece_dims(bot_piece)
        top_touch_cols = sorted(c for r, c in top_piece if r == tph - 1)
        bot_touch_cols = sorted(c for r, c in bot_piece if r == 0)
        top_key = random.choice(top_touch_cols)
        bot_key = random.choice(bot_touch_cols)

        link_lo = max(sprite_c0 + top_key, sprite_c0 + bot_key)
        link_hi = min(sprite_c1 - tpw + 1 + top_key, sprite_c1 - bpw + 1 + bot_key)
        if link_lo > link_hi:
            return None
        link_c = random.randint(link_lo, link_hi)

        top_r0 = r_split - tph
        bot_r0 = r_split + 1
        top_c0 = link_c - top_key
        bot_c0 = link_c - bot_key
        top_pts = {(r + top_r0, c + top_c0) for r, c in top_piece}
        bot_pts = {(r + bot_r0, c + bot_c0) for r, c in bot_piece}

        seg_lo = max(sprite_c0, link_c - split_len + 1)
        seg_hi = min(link_c, sprite_c1 - split_len + 1)
        if seg_lo > seg_hi:
            return None
        seg_c0 = random.randint(seg_lo, seg_hi)
        split_pixels = {(r_split, c) for c in range(seg_c0, seg_c0 + split_len)}

        input_pieces[("T", "A")] = top_pts
        input_pieces[("B", "A")] = bot_pts

    combined = set(split_pixels)
    for pts in input_pieces.values():
        if combined & pts:
            return None
        combined |= pts

    if not combined:
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in combined:
        if not (0 <= r < H and 0 <= c < W):
            return None
        mask[r, c] = 1
    if not is_contiguous(mask):
        return None

    bb_h, bb_w = _bbox_dims(combined)
    if not (target_h_range[0] <= bb_h <= target_h_range[1] and target_w_range[0] <= bb_w <= target_w_range[1]):
        return None
    if not ((W > H and bb_w >= bb_h + 3) or (H > W and bb_h >= bb_w + 3)):
        return None

    density = len(combined) / float(bb_h * bb_w)
    if not (0.38 <= density <= 0.62):
        return None

    inp = np.full((H, W), bg, dtype=int)
    for r, c in combined:
        inp[r, c] = fg
    inp[int(ctrl_pt[0]), int(ctrl_pt[1])] = ctrl_color

    uncropped = _solve_uncropped(inp)
    cropped = _tight_crop_nonbg(uncropped, bg)

    if cropped.shape == inp.shape:
        return None
    if H == W:
        return None

    if not np.array_equal(main(inp), cropped):
        return None

    # Ensure the generator truly exposes the combo rather than degenerating
    # back to the original square/no-crop behavior.
    if uncropped.shape == cropped.shape:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
