# Variant of 4a21e3da
# Dimension: D5 — 切割后片段的变换规则 (transform)
# Change: 与控制边接壤的区域滑动到对应角落；不接壤的区域消失（erase）... → 片段按大小排序重新分配角落

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous

# Design analysis for this D5 variant
# -----------------------------------
# source_task: None.
#
# a) Concrete behavior and trace
#    Input still contains one connected fg sprite near the center, plus 1-2 control pixels on the border.
#    Control rays are unchanged: they paint bg cells inward along one row/col, skipping fg, until the farthest
#    fg on that split line. Split-line fg pixels stay fixed in the output.
#    The only changed stage is what happens to the cut pieces that survive.
#    Example with top+left controls:
#    - Original: TL / TR / BL survive and slide to TL / TR / BL respectively; BR disappears.
#    - Variant: if the three surviving pieces have sizes 9, 6, 4, then after sorting by size descending they
#      slide to TL, TR, BR. The smallest surviving piece no longer stays in its source-side corner.
#    This makes the changed rule visible in the output without altering the cut geometry itself.
#
# b) Complexity calibration
#    The new rule only works if piece sizes are visibly different. The generator therefore enforces pairwise
#    distinct surviving piece sizes with gaps of at least 2 pixels, biases toward 2-control cases so that
#    many examples show 3 surviving pieces rather than only 2, and avoids the one 2-control edge pairing
#    whose occupied-corner set can accidentally look like the original task too often.
#
# c) Review of old assumptions
#    The original main() assumed "destination corner = source region corner" via pick_dest(hs, vs). That is
#    exactly the D5 property that no longer holds, so that mapping must be removed.
#    The original generator also validated output pieces against staying on their original side of the split.
#    That check is now invalid, because a right-side piece may legitimately end up at TL or BR.
#    A new subtle assumption appears: if two surviving pieces have equal size, "sort by size" is ambiguous.
#    The solver uses a deterministic tie-break for robustness, but the generator avoids ties entirely.
#
# d) Placement legality
#    Kept and vanishing pieces are still generated inside their original input subregions so the input sprite can
#    remain contiguous around the split lines. In the output, pieces are only translated, never rotated/flipped.
#    Piece bounding boxes are capped to compact 5x5-ish shapes with <= 12 cells, which keeps corner placements
#    safely separated from one another and from the central split-line fg.
#
# e) Visual observability
#    Humans can count relative piece sizes directly from the input. The output then places the largest survivor
#    at TL and the smallest at BR. For 3 survivors, the middle-sized one goes to TR; for 2 survivors, the two
#    occupied corners are TL and BR. This produces a visible departure from the original "same-side corner" rule.
#
# f) Visual cleanliness
#    No extra colors or markers are introduced. Rays stay central, pieces stay compact, and corner placements are
#    sparse, so the resulting grids remain easy to parse by eye.
#
# g) Color inference
#    Colors are unchanged from the base task: bg, fg, ctrl. The variant depends only on visible pixel counts, not
#    on hidden color ordering or any numeric property of color ids.
#
# New solver assumptions audited against the generator
#    - There is exactly one fg color and one ctrl color on the border.
#    - There is at most one vertical split and at most one horizontal split.
#    - Every surviving region has a non-empty piece; vanishing regions may be empty in single-control cases.
#    - Surviving piece sizes are distinct in generated data, so size ranking is observable.
#    - Corner placements for 2 or 3 surviving pieces follow a fixed ladder: [TL, BR] or [TL, TR, BR].


def _corner_rank_sequence(n: int) -> list[tuple[str, str]]:
    if n <= 0:
        return []
    if n == 1:
        return [("T", "L")]
    if n == 2:
        return [("T", "L"), ("B", "R")]
    if n == 3:
        return [("T", "L"), ("T", "R"), ("B", "R")]
    return [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")][:n]


def _slide_points_to_corner(
    pts: list[tuple[int, int]] | frozenset[tuple[int, int]],
    H: int,
    W: int,
    dh: str,
    dv: str,
) -> frozenset[tuple[int, int]]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    rmin, rmax = int(min(rs)), int(max(rs))
    cmin, cmax = int(min(cs)), int(max(cs))
    dr = int(0 - rmin) if dh == "T" else int((H - 1) - rmax)
    dc = int(0 - cmin) if dv == "L" else int((W - 1) - cmax)
    return frozenset((int(r + dr), int(c + dc)) for r, c in pts)


def _region_side(r: int, c: int, r_split: int | None, c_split: int | None) -> tuple[str, str]:
    hs = "A"
    vs = "A"
    if r_split is not None:
        hs = "T" if int(r) < int(r_split) else "B"
    if c_split is not None:
        vs = "L" if int(c) < int(c_split) else "R"
    return hs, vs


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut a sprite along border-control rays and slide the surviving pieces by size rank.

    Unchanged from the base task:
    - border control pixels define one vertical and/or one horizontal split;
    - rays are drawn inward on bg cells only;
    - fg pixels on split line(s) stay in place;
    - regions not touching any control edge disappear.

    Changed:
    - surviving pieces are sorted by pixel count and reassigned to global corners:
      largest -> TL, smallest -> BR, intermediate ranks fill the fixed corner ladder.
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

    split_line_pts: set[tuple[int, int]] = set()
    for r, c in np.argwhere(g == int(fg)):
        rr = int(r)
        cc = int(c)
        if (c_split is not None and int(cc) == int(c_split)) or (r_split is not None and int(rr) == int(r_split)):
            out[int(rr), int(cc)] = int(fg)
            split_line_pts.add((int(rr), int(cc)))

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == int(fg)):
        rr = int(r)
        cc = int(c)
        if (rr, cc) in split_line_pts:
            continue
        hs, vs = _region_side(rr, cc, r_split, c_split)
        regions.setdefault((hs, vs), []).append((rr, cc))

    survivors: list[tuple[tuple[str, str], list[tuple[int, int]]]] = []
    for reg, pts in regions.items():
        if _touches_edges(*reg) & E:
            survivors.append((reg, pts))

    # Tie-breakers make behavior deterministic on arbitrary inputs, but generated tasks avoid equal sizes.
    survivors.sort(
        key=lambda item: (
            -len(item[1]),
            min(r for r, _ in item[1]),
            min(c for _, c in item[1]),
            item[0],
        )
    )

    for (_reg, pts), (dh, dv) in zip(survivors, _corner_rank_sequence(len(survivors))):
        for rr, cc in _slide_points_to_corner(pts, H, W, dh, dv):
            if 0 <= rr < H and 0 <= cc < W:
                out[int(rr), int(cc)] = int(fg)

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


def _compact_piece_cap(sub_h: int, sub_w: int) -> int:
    return int(min(sub_h * sub_w, min(sub_h, 5) * min(sub_w, 5), 12))


def _sample_piece(target_px: int, sub_h: int, sub_w: int) -> frozenset[tuple[int, int]] | None:
    max_h = int(min(sub_h, 5))
    max_w = int(min(sub_w, 5))
    if target_px > max_h * max_w:
        return None
    for _ in range(40):
        piece = _grow_piece(max_h, max_w, target_px)
        if len(piece) != target_px:
            continue
        ph = max(r for r, _ in piece) + 1
        pw = max(c for _, c in piece) + 1
        if ph >= 2 and pw >= 2:
            return piece
    return None


def _strict_size_gap(values: tuple[int, ...]) -> bool:
    ranked = sorted(values, reverse=True)
    return all(ranked[i] - ranked[i + 1] >= 2 for i in range(len(ranked) - 1))


def _choose_piece_sizes(
    caps: list[int],
    non_split_min: int,
    non_split_max: int,
    vanish_cap: int,
) -> tuple[list[int], int] | None:
    candidates: list[tuple[list[int], int]] = []
    if len(caps) == 2:
        for a in range(3, caps[0] + 1):
            for b in range(3, caps[1] + 1):
                sizes = (a, b)
                if not _strict_size_gap(sizes):
                    continue
                kept_sum = a + b
                if vanish_cap == 0:
                    if non_split_min <= kept_sum <= non_split_max:
                        candidates.append(([a, b], 0))
                else:
                    lo = max(3, non_split_min - kept_sum)
                    hi = min(vanish_cap, 6, non_split_max - kept_sum)
                    for vanish_size in range(lo, hi + 1):
                        candidates.append(([a, b], int(vanish_size)))
    elif len(caps) == 3:
        for a in range(3, caps[0] + 1):
            for b in range(3, caps[1] + 1):
                for c in range(3, caps[2] + 1):
                    sizes = (a, b, c)
                    if len(set(sizes)) != 3 or not _strict_size_gap(sizes):
                        continue
                    kept_sum = a + b + c
                    if vanish_cap == 0:
                        if non_split_min <= kept_sum <= non_split_max:
                            candidates.append(([a, b, c], 0))
                    else:
                        lo = max(3, non_split_min - kept_sum)
                        hi = min(vanish_cap, 6, non_split_max - kept_sum)
                        for vanish_size in range(lo, hi + 1):
                            candidates.append(([a, b, c], int(vanish_size)))
    if not candidates:
        return None
    return random.choice(candidates)


def _region_sort_key(pts: frozenset[tuple[int, int]]) -> tuple[int, int, int]:
    return (-len(pts), min(r for r, _ in pts), min(c for _, c in pts))


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2

    num_ctrl = random.choices([1, 2], weights=[1, 5])[0]
    if num_ctrl == 1:
        ctrl_edges = [random.choice(["top", "bottom", "left", "right"])]
    else:
        ctrl_edges = random.choice(
            [
                ["top", "left"],
                ["bottom", "left"],
                ["bottom", "right"],
            ]
        )

    c_split = r_split = None
    ctrl_pts: list[tuple[int, int]] = []
    E = set(ctrl_edges)
    has_v = has_h = False

    for edge in ctrl_edges:
        if edge in ("top", "bottom"):
            c_split = random.randint(int(W * 0.4), int(W * 0.6))
            ctrl_pts.append((0 if edge == "top" else H - 1, c_split))
            has_v = True
        else:
            r_split = random.randint(int(H * 0.4), int(H * 0.6))
            ctrl_pts.append((r_split, 0 if edge == "left" else W - 1))
            has_h = True

    def touches_E(hs: str, vs: str) -> bool:
        return bool(_touches_edges(hs, vs) & E)

    if has_h and has_v:
        all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    elif has_v:
        all_regions = [("A", "L"), ("A", "R")]
    else:
        all_regions = [("T", "A"), ("B", "A")]

    kept = [r for r in all_regions if touches_E(*r)]
    vanish = [r for r in all_regions if r not in kept]

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
        r0 = sprite_r0 if hs in ("T", "A") else int(r_split + 1)
        r1 = (int(r_split - 1) if hs == "T" else sprite_r1) if has_h else sprite_r1
        c0 = sprite_c0 if vs in ("L", "A") else int(c_split + 1)
        c1 = (int(c_split - 1) if vs == "L" else sprite_c1) if has_v else sprite_c1
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
    density_min = max(1, int(np.ceil(0.40 * area)))
    density_max = max(density_min, int(np.floor(0.58 * area)))
    non_split_min = density_min - len(split_pixels)
    non_split_max = density_max - len(split_pixels)
    if non_split_max < len(kept) * 3:
        return None

    kept_caps: list[int] = []
    for reg in kept:
        r0, r1, c0, c1 = sub_bbox(*reg)
        kept_caps.append(_compact_piece_cap(r1 - r0 + 1, c1 - c0 + 1))
    if any(cap < 3 for cap in kept_caps):
        return None

    vanish_cap = 0
    if vanish:
        vr0, vr1, vc0, vc1 = sub_bbox(*vanish[0])
        vanish_cap = min(_compact_piece_cap(vr1 - vr0 + 1, vc1 - vc0 + 1), 7)
        if vanish_cap < 3:
            return None

    size_plan = _choose_piece_sizes(kept_caps, non_split_min, non_split_max, vanish_cap)
    if size_plan is None:
        return None
    piece_sizes, vanish_budget = size_plan

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    all_inp_fg = set(split_pixels)

    for reg, target_px in zip(kept, piece_sizes):
        hs, vs = reg
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = r1 - r0 + 1
        sub_w = c1 - c0 + 1
        piece = _sample_piece(target_px, sub_h, sub_w)
        if piece is None:
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

        inp_pts = frozenset((int(r + inp_r0), int(c + inp_c0)) for r, c in piece)
        for r, c in inp_pts:
            if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                return None
            if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                return None
            if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                return None
            if (r, c) in all_inp_fg:
                return None

        input_pieces[reg] = inp_pts
        all_inp_fg |= set(inp_pts)

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
        vhs, vvs = vanish[0]
        vr0, vr1, vc0, vc1 = sub_bbox(vhs, vvs)
        vh = vr1 - vr0 + 1
        vw = vc1 - vc0 + 1
        vanish_piece = _sample_piece(vanish_budget, vh, vw)
        if vanish_piece is None:
            return None
        pph = max(r for r, _ in vanish_piece) + 1
        ppw = max(c for _, c in vanish_piece) + 1
        vr_off = (vr1 - pph + 1) if vhs == "T" else vr0
        vc_off = (vc1 - ppw + 1) if vvs == "L" else vc0
        vpts = {(int(r + vr_off), int(c + vc_off)) for r, c in vanish_piece}
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

    ranked_regs = sorted(kept, key=lambda reg: _region_sort_key(input_pieces[reg]))
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    occupied_out: set[tuple[int, int]] = set(split_pixels)
    for reg, (dh, dv) in zip(ranked_regs, _corner_rank_sequence(len(ranked_regs))):
        out_pts = _slide_points_to_corner(input_pieces[reg], H, W, dh, dv)
        if occupied_out & set(out_pts):
            return None
        occupied_out |= set(out_pts)
        output_pieces[reg] = out_pts

    inp = np.full((H, W), bg, dtype=int)
    for r, c in combined:
        inp[r, c] = fg
    for cr, cc in ctrl_pts:
        inp[cr, cc] = ctrl_color

    out = np.full((H, W), bg, dtype=int)
    for pts in output_pieces.values():
        for r, c in pts:
            out[r, c] = fg
    for r, c in split_pixels:
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
    if actual_density < 0.39 or actual_density > 0.60:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
