# Combo variant of 4a21e3da (index 25)
# combo_spec: D5/0+D12/1
# Name: 未接壤区域着色+区域按象限着色

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    Start from the base task: one centered connected fg sprite is cut by one vertical and one horizontal
#    intermittent ray emitted from border control pixels. Split-line fg cells stay in place, and every other
#    fg cell belongs to one of four quadrants relative to the split intersection.
#    Example with controls on top and left:
#    - base task:
#      TL, TR, BL touch a control edge, so they slide to TL/TR/BL corners and keep fg color;
#      BR touches neither control edge, so it vanishes.
#    - D5/0 alone:
#      TL/TR/BL still slide, but BR stays at its original central location and is recolored.
#    - D12/1 alone:
#      TL/TR/BL still slide, but each moved region gets a corner-dependent color instead of uniform fg;
#      BR still vanishes.
#    - this combo:
#      TL/TR/BL slide and recolor by destination corner, while BR stays in place and recolors to one fixed
#      "detached-region" color. The result is a full four-region color partition map plus the unchanged
#      split-line fg pixels and ctrl rays.
# b) Complexity calibration:
#    The interesting branch only exists when both a vertical and a horizontal cut are present, so the
#    generator always uses exactly two perpendicular controls. That yields exactly 4 regions, 3 moved
#    regions, and 1 stationary detached region in every training example. Moved colors come from the fixed
#    corner mapping TL->3, TR->4, BL->6, BR->8 (matching D12/1), and the detached region uses one fixed
#    extra color (prefer 5). This keeps the palette small enough to infer from 2-4 examples while still
#    making both sub-rules simultaneously visible.
# c) Implicit-assumption audit against the original main():
#    - Control-color inference from border non-bg cells still works because all new recolor outputs appear
#      only in the output, never on the input border.
#    - Input still contains exactly one fg color for the sprite, so "fg = other non-bg non-ctrl color" is
#      still valid.
#    - Ray drawing originally depended only on fg cells that lie on the split lines. This remains valid:
#      the stationary detached region is generated strictly away from both split lines, so it cannot change
#      ray endpoints.
#    - The original non-touching branch used `continue` to erase. That assumption is intentionally broken:
#      the combo solver now writes those cells back in place with the detached color.
#    - The original moved branch assumed all moved regions used the same color. That is also broken: the
#      combo solver must determine each moved region's destination corner and paint it with the mapped color.
# d) Placement legality requirements for the generator:
#    - Each quadrant bbox must exist and have room for at least a 3-cell connected piece.
#    - The three moved pieces stay strictly on their own sides of the split before and after sliding.
#    - The detached piece stays strictly inside the unique non-touching quadrant, off both split lines.
#    - The reconstructed input sprite (moved pieces pulled inward + split pixels + detached piece) must be
#      one 4-connected fg component, stay inside a 5-8 x 8-10 bbox, and keep a >=4 cell margin to borders.
# e) Visual observability:
#    Every generated sample simultaneously shows:
#    - three corner-slid regions with three different destination colors;
#    - one central stationary region with a fourth region color;
#    - orange split-line residue and red rays.
#    A human can compare multiple examples and see that position relative to the cuts controls both whether
#    a piece moves and which color it becomes.
# f) Visual cleanliness:
#    No cell carries multiple semantics: bg, ctrl ray, stationary split-line fg, moved-region color, or
#    detached-region color. The moved pieces slide outward, so they cannot cover the detached region; the
#    split-line fg cells remain in the center and are kept in the original fg color to make the cut readable.
# g) Color-rule inferability:
#    The moved-color rule is structural: destination corner decides color, not raw color ids in the input.
#    The detached recolor is also structural: "the only region that does not touch a control edge" becomes
#    the fixed detached color. Because the generator randomizes which two edges contain controls, different
#    absolute quadrants play the detached role across examples, which helps isolate the true rule.
# h) Anti-degeneracy choice:
#    To avoid samples collapsing into D5-only or D12-only behavior, the generator forces the detached region
#    to be non-empty and all three moved regions to be non-empty. That guarantees both:
#    - at least one stationary recolored region exists;
#    - at least three distinct corner colors exist.
# i) New solver assumptions introduced here and how generation guarantees them:
#    - The detached recolor must be different from bg/fg/ctrl and from all corner colors. `_pick_stationary`
#      enforces this.
#    - Each moved region must have a well-defined destination corner. Two perpendicular controls guarantee it.
#    - Split-line pixels must exist on both cuts so the rays are observable. Generator always places 2-3 fg
#      cells on each split line.


_CORNER_COLORS: dict[tuple[str, str], int] = {
    ("T", "L"): 3,
    ("T", "R"): 4,
    ("B", "L"): 6,
    ("B", "R"): 8,
}

_STATIONARY_COLOR_PREFERENCE = [5, 9, 0]


def _corner_color(h_side: str, v_side: str) -> int:
    return int(_CORNER_COLORS[(str(h_side), str(v_side))])


def _pick_stationary_color(bg: int, fg: int, ctrl: int) -> int:
    blocked = {int(bg), int(fg), int(ctrl), *map(int, _CORNER_COLORS.values())}
    for cand in _STATIONARY_COLOR_PREFERENCE:
        if int(cand) not in blocked:
            return int(cand)
    for cand in range(10):
        if int(cand) not in blocked:
            return int(cand)
    raise AssertionError("No stationary recolor available")


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
    return str(dh), str(dv)


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut a sprite along border-control rays.

    Combo behavior:
    - touching regions slide to their destination corners and recolor by destination quadrant;
    - the unique non-touching region does not vanish and instead stays in place with a fixed new color;
    - split-line fg pixels remain fg, rays remain ctrl-colored.
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
    stationary = _pick_stationary_color(bg, fg, ctrl)

    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(ctrl))]

    edges: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for r, c in ctrl_pts:
        if int(r) == 0:
            edges.add("top")
            c_split = int(c)
        elif int(r) == int(H - 1):
            edges.add("bottom")
            c_split = int(c)
        elif int(c) == 0:
            edges.add("left")
            r_split = int(r)
        elif int(c) == int(W - 1):
            edges.add("right")
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

    for (hs, vs), pts in regions.items():
        if not (_touches_edges(hs, vs) & edges):
            for r, c in pts:
                out[int(r), int(c)] = int(stationary)
            continue

        dh, dv = _pick_dest(hs, vs, edges, r_split, c_split)
        moved_color = _corner_color(dh, dv)
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
                out[int(rr), int(cc)] = int(moved_color)

    return out


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2
    stationary = _pick_stationary_color(bg, fg, ctrl_color)

    ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]
    edges = set(ctrl_edges)

    c_split = random.randint(int(W * 0.4), int(W * 0.6))
    r_split = random.randint(int(H * 0.4), int(H * 0.6))
    ctrl_pts = [
        (0 if ctrl_edges[0] == "top" else H - 1, int(c_split)),
        (int(r_split), 0 if ctrl_edges[1] == "left" else W - 1),
    ]

    all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    kept = [reg for reg in all_regions if _touches_edges(*reg) & edges]
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
        if r1 < r0 or c1 < c0:
            return None
        if (r1 - r0 + 1) * (c1 - c0 + 1) < 3:
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
    min_piece_total = n_split + 3 * len(kept) + 4
    target_total = max(min_piece_total, int(random.uniform(0.50, 0.60) * area))
    if target_total >= area:
        target_total = area - 1
    if target_total < min_piece_total:
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
    output_pieces: dict[tuple[str, str], tuple[str, str, frozenset[tuple[int, int]]]] = {}

    for idx, reg in enumerate(kept):
        hs, vs = reg
        dh, dv = _pick_dest(hs, vs, edges, int(r_split), int(c_split))
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
        dr = -min(inp_rs) if dh == "T" else (H - 1 - max(inp_rs))
        dc = -min(inp_cs) if dv == "L" else (W - 1 - max(inp_cs))
        out_pts = frozenset((int(r + dr), int(c + dc)) for r, c in inp_pts)

        for r, c in out_pts:
            if (vs == "L" and c >= int(c_split)) or (vs == "R" and c <= int(c_split)):
                return None
            if (hs == "T" and r >= int(r_split)) or (hs == "B" and r <= int(r_split)):
                return None
        output_pieces[reg] = (dh, dv, out_pts)

    all_inp_fg: set[tuple[int, int]] = set()
    for pts in input_pieces.values():
        all_inp_fg |= set(pts)
    all_inp_fg |= split_pixels
    if not all_inp_fg:
        return None

    detached_reg = detached[0]
    dr0, dr1, dc0, dc1 = sub_bbox(*detached_reg)
    det_h = dr1 - dr0 + 1
    det_w = dc1 - dc0 + 1
    if det_h < 1 or det_w < 1 or det_h * det_w < detached_budget:
        return None

    detached_piece = _grow_piece_with_retry(det_h, det_w, detached_budget, min_h=2, min_w=2)
    if detached_piece is None or len(detached_piece) < detached_budget:
        return None

    dph, dpw = _piece_dims(detached_piece)
    dhs, dvs = detached_reg
    det_r_off = (dr1 - dph + 1) if dhs == "T" else dr0
    det_c_off = (dc1 - dpw + 1) if dvs == "L" else dc0
    detached_fg = {(int(r + det_r_off), int(c + det_c_off)) for r, c in detached_piece}

    for r, c in detached_fg:
        if r < 4 or r > H - 5 or c < 4 or c > W - 5:
            return None
        if (r, c) in all_inp_fg:
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
    visible_corner_colors: set[int] = set()
    for dh, dv, pts in output_pieces.values():
        piece_color = _corner_color(dh, dv)
        visible_corner_colors.add(int(piece_color))
        for r, c in pts:
            out[int(r), int(c)] = int(piece_color)
    for r, c in split_pixels:
        out[int(r), int(c)] = int(fg)
    for r, c in detached_fg:
        out[int(r), int(c)] = int(stationary)

    split_rows = [r for r, c in split_pixels if int(c) == int(c_split)]
    split_cols = [c for r, c in split_pixels if int(r) == int(r_split)]
    for cr, cc in ctrl_pts:
        if cr == 0 and split_rows:
            for r in range(max(split_rows) + 1):
                if out[int(r), int(cc)] == int(bg):
                    out[int(r), int(cc)] = int(ctrl_color)
        elif cr == H - 1 and split_rows:
            for r in range(H - 1, min(split_rows) - 1, -1):
                if out[int(r), int(cc)] == int(bg):
                    out[int(r), int(cc)] = int(ctrl_color)
        elif cc == 0 and split_cols:
            for c in range(max(split_cols) + 1):
                if out[int(cr), int(c)] == int(bg):
                    out[int(cr), int(c)] = int(ctrl_color)
        elif cc == W - 1 and split_cols:
            for c in range(W - 1, min(split_cols) - 1, -1):
                if out[int(cr), int(c)] == int(bg):
                    out[int(cr), int(c)] = int(ctrl_color)
    for cr, cc in ctrl_pts:
        out[int(cr), int(cc)] = int(ctrl_color)

    actual_density = len(combined) / float(ah * aw)
    if actual_density < 0.38 or actual_density > 0.62:
        return None
    if len(visible_corner_colors) != 3:
        return None
    if not np.any(out == int(stationary)):
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
