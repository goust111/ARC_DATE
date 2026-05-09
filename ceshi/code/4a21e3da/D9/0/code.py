# Variant of 4a21e3da
# Dimension: D9 — 分割线上前景像素的处理 (meta)
# Change: 分割线上的前景像素保留在原位不移动... → 分割线像素也参与移动

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous

# Design analysis for D9
# ======================
# variant_spec:
#   unchanged_core:
#     - grid remains 18-20 square, colors remain {bg, fg, ctrl}
#     - 1-2 border ctrl pixels define one vertical split and optionally one horizontal split
#     - ctrl rays are still drawn first from the input and still only paint background cells
#     - non-adjacent regions still vanish; adjacent regions still slide to their corresponding corners
#   d9_change:
#     - original behavior: every fg pixel on c_split / r_split is painted back in-place before pieces move
#     - new behavior: split-line fg is not kept at the center
#     - instead, each split-line fg pixel is copied into every geometrically adjacent region, and each moved
#       region carries its own copy when translated to a corner
#   concrete_pixel_behavior:
#     - vertical-only split: each split pixel belongs to both (A,L) and (A,R)
#     - horizontal-only split: each split pixel belongs to both (T,A) and (B,A)
#     - two-split case:
#         - a vertical split pixel above r_split belongs to (T,L) and (T,R)
#         - a vertical split pixel below r_split belongs to (B,L) and (B,R)
#         - a horizontal split pixel left of c_split belongs to (T,L) and (B,L)
#         - a horizontal split pixel right of c_split belongs to (T,R) and (B,R)
#         - an intersection pixel, if it existed, would belong to all four quadrants
#   small_trace:
#     - original: TL piece excludes split pixels; the split pixels remain in the center after the cut
#     - D9: TL piece is augmented with the split pixels adjacent to TL, then the whole augmented piece slides
#       to the TL corner; TR/BL/BR receive their own copies in the same way
#     - observable difference: the central split line becomes fg-empty after movement, while the corner pieces
#       gain repeated boundary motifs copied from the split line
#   complexity_calibration:
#     - the variant is only visible if split pixels are numerous enough and touch multiple regions, so the
#       generator forces 3-5 split pixels on one-split tasks and both sides of each split on two-split tasks
#     - the generator also forces every region piece to touch the split via boundary seeds; otherwise the copy
#       rule would be hard to infer visually and the solver would rely on hidden assumptions
#   implicit_assumption_audit:
#     - original `for split fg: out[r,c]=fg` depended on "split fg stays put"; this must be removed
#     - original region grouping skipped split fg entirely and never re-attached it; D9 requires a second pass
#       that redistributes split fg copies back into each region before computing the move translation
#     - original generator drew rays from the output fg distribution; that happened to work only because split
#       fg remained on the line. Under D9 the rays must still come from the input geometry, so generator
#       validation is done via `main(inp)` instead of reconstructing rays from the final fg positions
#   placement_legality:
#     - region seeds are built on the split-adjacent boundary plus short outward arms, guaranteeing connectivity,
#       non-overlap across subregions, and enough spread for the sprite bbox to remain visible
#     - each region occupies only its own rectangular sub-bbox, so pieces cannot overlap and solver labels remain
#       unambiguous
#   visual_readability:
#     - the output has only three visual layers: bg, ctrl rays, fg pieces
#     - central split fg is removed, which makes the copied corner-boundary motifs stand out instead of stacking
#       fg on top of the ray line


def _layout_info(g: np.ndarray):
    H, W = g.shape
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})

    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
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

    return H, W, bg, fg, ctrl, ctrl_pts, E, r_split, c_split


def _is_split_fg(r: int, c: int, r_split: int | None, c_split: int | None) -> bool:
    return (c_split is not None and c == c_split) or (r_split is not None and r == r_split)


def _region_of_pixel(r: int, c: int, r_split: int | None, c_split: int | None) -> tuple[str, str]:
    hs = "A"
    vs = "A"
    if r_split is not None:
        hs = "T" if r < r_split else "B"
    if c_split is not None:
        vs = "L" if c < c_split else "R"
    return hs, vs


def _all_regions(r_split: int | None, c_split: int | None) -> list[tuple[str, str]]:
    if r_split is not None and c_split is not None:
        return [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    if c_split is not None:
        return [("A", "L"), ("A", "R")]
    return [("T", "A"), ("B", "A")]


def _adjacent_regions_for_split_pixel(
    r: int, c: int, r_split: int | None, c_split: int | None
) -> list[tuple[str, str]]:
    if r_split is None:
        hs_opts = ["A"]
    elif r < r_split:
        hs_opts = ["T"]
    elif r > r_split:
        hs_opts = ["B"]
    else:
        hs_opts = ["T", "B"]

    if c_split is None:
        vs_opts = ["A"]
    elif c < c_split:
        vs_opts = ["L"]
    elif c > c_split:
        vs_opts = ["R"]
    else:
        vs_opts = ["L", "R"]

    return [(hs, vs) for hs in hs_opts for vs in vs_opts]


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
    hs: str, vs: str, E: set[str], r_split: int | None, c_split: int | None
) -> tuple[str, str]:
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

    return dh, dv


def main(grid: np.ndarray) -> np.ndarray:
    """
    D9 variant of 4a21e3da.

    Rays are unchanged, but split-line fg no longer stays in place. Instead each split-line fg
    pixel is duplicated into every adjacent region, and each kept region moves together with
    its assigned split-pixel copies.
    """
    g = np.array(grid, dtype=int)
    H, W, bg, fg, ctrl, ctrl_pts, E, r_split, c_split = _layout_info(g)

    out = np.full((H, W), bg, dtype=int)

    # Stage 1 is still based on the input geometry: ctrl rays are drawn before any fg movement.
    for r, c in ctrl_pts:
        if r == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                stop = int(max(rows))
                for rr in range(0, stop + 1):
                    if int(g[rr, c]) == bg:
                        out[rr, c] = ctrl
        elif r == H - 1:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                stop = int(min(rows))
                for rr in range(H - 1, stop - 1, -1):
                    if int(g[rr, c]) == bg:
                        out[rr, c] = ctrl
        elif c == 0:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                stop = int(max(cols))
                for cc in range(0, stop + 1):
                    if int(g[r, cc]) == bg:
                        out[r, cc] = ctrl
        elif c == W - 1:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                stop = int(min(cols))
                for cc in range(W - 1, stop - 1, -1):
                    if int(g[r, cc]) == bg:
                        out[r, cc] = ctrl

    for r, c in ctrl_pts:
        out[r, c] = ctrl

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    split_pixels: list[tuple[int, int]] = []
    for r, c in np.argwhere(g == fg):
        r = int(r)
        c = int(c)
        if _is_split_fg(r, c, r_split, c_split):
            split_pixels.append((r, c))
            continue
        reg = _region_of_pixel(r, c, r_split, c_split)
        regions.setdefault(reg, []).append((r, c))

    split_copies: dict[tuple[str, str], list[tuple[int, int]]] = {reg: [] for reg in regions}
    for r, c in split_pixels:
        for reg in _adjacent_regions_for_split_pixel(r, c, r_split, c_split):
            if reg in split_copies:
                split_copies[reg].append((r, c))

    for reg, pts in regions.items():
        hs, vs = reg
        if not (_touches_edges(hs, vs) & E):
            continue

        piece = pts + split_copies.get(reg, [])
        rs = [r for r, _ in piece]
        cs = [c for _, c in piece]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        dr = -rmin if dh == "T" else (H - 1 - rmax)
        dc = -cmin if dv == "L" else (W - 1 - cmax)

        for r, c in piece:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = fg

    return out


def _grow_from_seeds(
    max_h: int, max_w: int, target_px: int, seeds: set[tuple[int, int]]
) -> frozenset[tuple[int, int]]:
    cells = set(seeds)
    if target_px < len(cells):
        return frozenset()
    while len(cells) < target_px:
        neighbors = set()
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < max_h and 0 <= nc < max_w and (nr, nc) not in cells:
                    neighbors.add((nr, nc))
        if not neighbors:
            break
        cells.add(random.choice(sorted(neighbors)))
    return frozenset(cells)


def _sub_bbox(
    hs: str,
    vs: str,
    sprite_r0: int,
    sprite_r1: int,
    sprite_c0: int,
    sprite_c1: int,
    r_split: int | None,
    c_split: int | None,
) -> tuple[int, int, int, int]:
    r0 = sprite_r0 if hs in ("T", "A") else int(r_split) + 1
    r1 = (int(r_split) - 1 if hs == "T" else sprite_r1) if r_split is not None else sprite_r1
    c0 = sprite_c0 if vs in ("L", "A") else int(c_split) + 1
    c1 = (int(c_split) - 1 if vs == "L" else sprite_c1) if c_split is not None else sprite_c1
    return r0, r1, c0, c1


def _select_split_pixels(
    sprite_r0: int,
    sprite_r1: int,
    sprite_c0: int,
    sprite_c1: int,
    r_split: int | None,
    c_split: int | None,
) -> tuple[set[tuple[int, int]], list[int], list[int]]:
    split_pixels: set[tuple[int, int]] = set()
    vertical_rows: list[int] = []
    horizontal_cols: list[int] = []

    if c_split is not None and r_split is None:
        rows = list(range(sprite_r0, sprite_r1 + 1))
        n = random.randint(3, min(5, len(rows)))
        vertical_rows = sorted(random.sample(rows, n))
        split_pixels |= {(r, c_split) for r in vertical_rows}
    elif r_split is not None and c_split is None:
        cols = list(range(sprite_c0, sprite_c1 + 1))
        n = random.randint(3, min(5, len(cols)))
        horizontal_cols = sorted(random.sample(cols, n))
        split_pixels |= {(r_split, c) for c in horizontal_cols}
    else:
        top_rows = list(range(sprite_r0, int(r_split)))
        bot_rows = list(range(int(r_split) + 1, sprite_r1 + 1))
        left_cols = list(range(sprite_c0, int(c_split)))
        right_cols = list(range(int(c_split) + 1, sprite_c1 + 1))
        if not top_rows or not bot_rows or not left_cols or not right_cols:
            return set(), [], []

        n_top = random.randint(1, min(2, len(top_rows)))
        n_bot = random.randint(1, min(2, len(bot_rows)))
        n_left = random.randint(1, min(2, len(left_cols)))
        n_right = random.randint(1, min(2, len(right_cols)))

        vertical_rows = sorted(random.sample(top_rows, n_top) + random.sample(bot_rows, n_bot))
        horizontal_cols = sorted(random.sample(left_cols, n_left) + random.sample(right_cols, n_right))
        split_pixels |= {(r, int(c_split)) for r in vertical_rows}
        split_pixels |= {(int(r_split), c) for c in horizontal_cols}

    return split_pixels, vertical_rows, horizontal_cols


def _build_region_seed(
    hs: str,
    vs: str,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    r_split: int | None,
    c_split: int | None,
    vertical_rows: list[int],
    horizontal_cols: list[int],
) -> set[tuple[int, int]] | None:
    seeds: set[tuple[int, int]] = set()
    boundary_col: int | None = None
    boundary_row: int | None = None
    relevant_rows: list[int] = []
    relevant_cols: list[int] = []

    if c_split is not None and vs != "A":
        boundary_col = int(c_split) - 1 if vs == "L" else int(c_split) + 1
        relevant_rows = [
            rr
            for rr in vertical_rows
            if hs == "A" or (hs == "T" and rr < int(r_split)) or (hs == "B" and rr > int(r_split))
        ]
        if not relevant_rows:
            return None
        lo = min(relevant_rows)
        hi = max(relevant_rows)
        if r_split is not None and hs != "A":
            elbow_row = int(r_split) - 1 if hs == "T" else int(r_split) + 1
            lo = min(lo, elbow_row)
            hi = max(hi, elbow_row)
        for rr in range(lo, hi + 1):
            seeds.add((rr, boundary_col))

    if r_split is not None and hs != "A":
        boundary_row = int(r_split) - 1 if hs == "T" else int(r_split) + 1
        relevant_cols = [
            cc
            for cc in horizontal_cols
            if vs == "A" or (vs == "L" and cc < int(c_split)) or (vs == "R" and cc > int(c_split))
        ]
        if not relevant_cols:
            return None
        lo = min(relevant_cols)
        hi = max(relevant_cols)
        if c_split is not None and vs != "A":
            elbow_col = int(c_split) - 1 if vs == "L" else int(c_split) + 1
            lo = min(lo, elbow_col)
            hi = max(hi, elbow_col)
        for cc in range(lo, hi + 1):
            seeds.add((boundary_row, cc))

    if not seeds:
        return None

    if boundary_col is not None:
        anchor_row = (
            sorted(relevant_rows)[len(relevant_rows) // 2]
            if relevant_rows
            else (boundary_row if boundary_row is not None else (r0 + r1) // 2)
        )
        outer_col = c0 if vs == "L" else c1
        for cc in range(min(outer_col, boundary_col), max(outer_col, boundary_col) + 1):
            seeds.add((anchor_row, cc))

    if boundary_row is not None:
        anchor_col = (
            sorted(relevant_cols)[len(relevant_cols) // 2]
            if relevant_cols
            else (boundary_col if boundary_col is not None else (c0 + c1) // 2)
        )
        outer_row = r0 if hs == "T" else r1
        for rr in range(min(outer_row, boundary_row), max(outer_row, boundary_row) + 1):
            seeds.add((rr, anchor_col))

    if any(rr < r0 or rr > r1 or cc < c0 or cc > c1 for rr, cc in seeds):
        return None

    return seeds


def _old_main(grid: np.ndarray) -> np.ndarray:
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

    out = np.full((H, W), bg, dtype=int)

    for r, c in ctrl_pts:
        if r == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                for rr in range(0, max(rows) + 1):
                    if int(g[rr, c]) == bg:
                        out[rr, c] = ctrl
        elif r == H - 1:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                for rr in range(H - 1, min(rows) - 1, -1):
                    if int(g[rr, c]) == bg:
                        out[rr, c] = ctrl
        elif c == 0:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                for cc in range(0, max(cols) + 1):
                    if int(g[r, cc]) == bg:
                        out[r, cc] = ctrl
        elif c == W - 1:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                for cc in range(W - 1, min(cols) - 1, -1):
                    if int(g[r, cc]) == bg:
                        out[r, cc] = ctrl

    for r, c in ctrl_pts:
        out[r, c] = ctrl

    for r, c in np.argwhere(g == fg):
        r = int(r)
        c = int(c)
        if _is_split_fg(r, c, r_split, c_split):
            out[r, c] = fg

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == fg):
        r = int(r)
        c = int(c)
        if _is_split_fg(r, c, r_split, c_split):
            continue
        regions.setdefault(_region_of_pixel(r, c, r_split, c_split), []).append((r, c))

    for reg, pts in regions.items():
        hs, vs = reg
        if not (_touches_edges(hs, vs) & E):
            continue
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)
        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        dr = -rmin if dh == "T" else (H - 1 - rmax)
        dc = -cmin if dv == "L" else (W - 1 - cmax)
        for r, c in pts:
            out[r + dr, c + dc] = fg
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

    regions = _all_regions(r_split, c_split)
    kept = [reg for reg in regions if _touches_edges(*reg) & E]

    sprite_h = random.randint(5, 8)
    sprite_w = random.randint(8, 10)

    if c_split is not None:
        frac = random.uniform(0.4, 0.6)
        sprite_c0 = int(c_split) - int(frac * sprite_w)
        sprite_c0 = max(4, min(sprite_c0, W - 5 - sprite_w + 1))
        if sprite_c0 > W - 5 - sprite_w + 1:
            return None
        if int(c_split) <= sprite_c0 or int(c_split) >= sprite_c0 + sprite_w - 1:
            return None
    else:
        hi = W - 5 - sprite_w + 1
        if hi < 4:
            return None
        sprite_c0 = random.randint(4, hi)

    if r_split is not None:
        frac = random.uniform(0.4, 0.6)
        sprite_r0 = int(r_split) - int(frac * sprite_h)
        sprite_r0 = max(4, min(sprite_r0, H - 5 - sprite_h + 1))
        if sprite_r0 > H - 5 - sprite_h + 1:
            return None
        if int(r_split) <= sprite_r0 or int(r_split) >= sprite_r0 + sprite_h - 1:
            return None
    else:
        hi = H - 5 - sprite_h + 1
        if hi < 4:
            return None
        sprite_r0 = random.randint(4, hi)

    sprite_r1 = sprite_r0 + sprite_h - 1
    sprite_c1 = sprite_c0 + sprite_w - 1

    split_pixels, vertical_rows, horizontal_cols = _select_split_pixels(
        sprite_r0, sprite_r1, sprite_c0, sprite_c1, r_split, c_split
    )
    if not split_pixels:
        return None

    seeds_by_region: dict[tuple[str, str], set[tuple[int, int]]] = {}
    min_sizes: dict[tuple[str, str], int] = {}
    max_sizes: dict[tuple[str, str], int] = {}
    for reg in regions:
        hs, vs = reg
        r0, r1, c0, c1 = _sub_bbox(hs, vs, sprite_r0, sprite_r1, sprite_c0, sprite_c1, r_split, c_split)
        if r1 < r0 or c1 < c0:
            return None
        seed = _build_region_seed(hs, vs, r0, r1, c0, c1, r_split, c_split, vertical_rows, horizontal_cols)
        if not seed:
            return None
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if len(seed) > area:
            return None
        seeds_by_region[reg] = seed
        min_sizes[reg] = max(3, len(seed))
        max_sizes[reg] = area

    area = sprite_h * sprite_w
    target_total = random.randint(int(np.ceil(0.42 * area)), int(np.floor(0.58 * area)))
    piece_budget = target_total - len(split_pixels)
    min_sum = sum(min_sizes.values())
    if piece_budget < min_sum:
        return None

    piece_sizes = dict(min_sizes)
    extra = piece_budget - min_sum
    while extra > 0:
        choices = [reg for reg in regions if piece_sizes[reg] < max_sizes[reg]]
        if not choices:
            return None
        weights = [3 if reg in kept else 2 for reg in choices]
        reg = random.choices(choices, weights=weights)[0]
        piece_sizes[reg] += 1
        extra -= 1

    region_pts: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for reg in regions:
        hs, vs = reg
        r0, r1, c0, c1 = _sub_bbox(hs, vs, sprite_r0, sprite_r1, sprite_c0, sprite_c1, r_split, c_split)
        local_seed = {(r - r0, c - c0) for r, c in seeds_by_region[reg]}
        piece = _grow_from_seeds(r1 - r0 + 1, c1 - c0 + 1, piece_sizes[reg], local_seed)
        if len(piece) != piece_sizes[reg]:
            return None
        region_pts[reg] = {(r + r0, c + c0) for r, c in piece}

    combined = set(split_pixels)
    for pts in region_pts.values():
        combined |= pts

    mask = np.zeros((H, W), dtype=int)
    for r, c in combined:
        mask[r, c] = 1
    if not is_contiguous(mask):
        return None

    fg_arr = np.array(sorted(combined))
    bbox_h = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    bbox_w = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if bbox_h < 5 or bbox_h > 8 or bbox_w < 8 or bbox_w > 10:
        return None

    density = len(combined) / float(bbox_h * bbox_w)
    if density < 0.38 or density > 0.62:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for r, c in combined:
        inp[r, c] = fg
    for r, c in ctrl_pts:
        inp[r, c] = ctrl_color

    out = main(inp)

    # Variant-specific input filter: original split positions must not remain as fg in the output.
    if any(int(out[r, c]) == fg for r, c in split_pixels):
        return None

    # Guard against degeneration back to the original task behavior.
    if np.array_equal(out, _old_main(inp)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
