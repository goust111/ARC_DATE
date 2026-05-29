# Combo variant of 4a21e3da (index 29)
# combo_spec: D5/1+D16/1
# Name: 镜像翻转+消失区域移到中心

from __future__ import annotations

import random
import sys

import numpy as np

try:
    from arc_utils import colors_present, get_bg, is_contiguous
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import colors_present, get_bg, is_contiguous


"""
Design analysis for combo D5/1 + D16/1
======================================

a) concretization_and_trace:
  original:
    - input: one connected fg sprite near the center, plus 1-2 ctrl pixels on the
      border. A top/bottom ctrl defines a vertical split column; a left/right ctrl
      defines a horizontal split row.
    - output: draw intermittent ctrl rays inward; fg on split lines stays fixed;
      regions touching a control edge slide unchanged to the corresponding outer
      corner; the unique non-touching quadrant under two controls disappears.
  combo:
    - kept regions still go to outer corners, but first they are mirrored inside
      their own bbox. For corner motion this means flipping vertically and
      horizontally before anchoring to the destination corner.
    - the old vanishing quadrant no longer disappears. It keeps its original
      orientation and is translated into the split intersection's matching inner
      corner, one cell away from both split lines.
  small_trace:
    - suppose controls are on `top` and `left`, so the four regions are TL, TR,
      BL, BR and BR is the only region not touching either control edge.
    - original: TL/TR/BL slide unchanged to outer corners, BR is skipped.
    - combo: TL/TR/BL are mirrored-then-cornered, while BR is preserved and moved
      near `(r_split + 1, c_split + 1)`.
    - visible difference: output now shows "outer mirrored fragments" plus one
      "inner preserved fragment", so the solver must separate two target rules:
      corner+flip versus center+no-flip.

b) complexity_calibration:
  - I keep the original control vocabulary and split geometry: at most one vertical
    split and one horizontal split, with exactly one non-touching quadrant when two
    controls exist.
  - To make the fusion visible, the generator intentionally emits only two-control
    samples. That keeps the center-preservation rule present in every example while
    still staying inside the source task family.
  - Reflection is deterministic: surviving outer pieces are mirrored on both bbox
    axes before corner placement. Center pieces are never mirrored. This low-branch
    structure is easy to spot across a few ARC examples.

c) implicit_assumption_review_of_original_main:
  - Original Stage 1 ray logic only depends on border controls and fg extent on the
    split lines. That assumption still holds, so the ray code stays unchanged.
  - Original split-line preservation assumed those fg pixels are excluded from later
    region motion. That is still valid.
  - Original Stage 2 preserved local offsets by pure translation. That fails for
    kept regions because mirrored motion must reverse local row/column order inside
    the region bbox before anchoring to a corner.
  - Original `if not (touches_edges(...) & E): continue` encoded the assumption that
    the non-touching quadrant vanishes. That assumption is false in the combo and is
    replaced by an explicit center-inner-corner target.
  - New solver assumption: center relocation only happens when both `r_split` and
    `c_split` exist. The generator guarantees this by always creating two controls.

d) placement_legality:
  - Grid size remains 18-20 square. Sprite bbox stays inside the original 5-8 by
    8-10 envelope and keeps margin >= 4 from the grid border.
  - Each quadrant receives at least 3 fg cells and is generated strictly on its own
    side of both split lines. Input placement keeps every piece flush toward the
    split so the whole sprite can be connected through the split-line fg pixels.
  - Output legality is checked region by region: every target set must remain inside
    the grid, stay on the correct side of the split lines, avoid split-line fg, and
    avoid overlap with other output pieces.

e) visual_observability:
  - Every generated sample includes two controls, so the center-preserved region is
    always present and cannot be missed.
  - The generator rejects kept pieces whose mirrored-corner output equals the legacy
    slide-only output, preventing accidental degeneration to the base/D16-only task.
  - Humans see a stable pattern: three outer corner fragments change silhouette,
    while one center-adjacent fragment keeps its silhouette. Two or three examples
    are enough to infer the dual rule.

f) visual_cleanliness:
  - Colors remain in the original roles: bg, fg, ctrl only.
  - Rays live on split lines and corners live at the frame, while the preserved
    center fragment sits one cell away from the split intersection. This separation
    avoids clutter and makes region identities easy to read.

g) color_inference:
  - No color remapping is introduced. All new behavior is geometric, so there is no
    hidden dependence on color values or non-visual ordering.

h) new_solver_assumptions_and_generator_guarantees:
  - Assumption: exactly one vertical split and one horizontal split exist for combo
    examples. Guarantee: generator always picks one top/bottom control and one
    left/right control.
  - Assumption: the center-preserved region is unique. Guarantee: with two
    perpendicular controls there is exactly one quadrant whose edge-touch set is
    disjoint from the control-edge set.
  - Assumption: mirrored motion should be visible. Guarantee: each kept region is
    rejected if mirrored-corner placement matches legacy slide-only placement, and
    the total mirrored-vs-legacy symmetric-difference is forced above a threshold.
"""


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


def _pick_corner_dest(hs: str, vs: str, E: set[str], r_split: int | None, c_split: int | None) -> tuple[str, str]:
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


def _normalize_points(pts: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> tuple[frozenset[tuple[int, int]], int, int]:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    rmin, cmin = int(min(rs)), int(min(cs))
    shape = frozenset((int(r - rmin), int(c - cmin)) for r, c in pts)
    h = int(max(r for r, _ in shape) + 1)
    w = int(max(c for _, c in shape) + 1)
    return shape, h, w


def _flip_shape(
    shape: frozenset[tuple[int, int]],
    h: int,
    w: int,
    *,
    flip_v: bool,
    flip_h: bool,
) -> frozenset[tuple[int, int]]:
    out = set()
    for r, c in shape:
        rr = int(h - 1 - r) if flip_v else int(r)
        cc = int(w - 1 - c) if flip_h else int(c)
        out.add((int(rr), int(cc)))
    return frozenset(out)


def _place_shape_in_corner(
    shape: frozenset[tuple[int, int]],
    h: int,
    w: int,
    dh: str,
    dv: str,
    H: int,
    W: int,
) -> frozenset[tuple[int, int]]:
    r0 = 0 if dh == "T" else int(H - h)
    c0 = 0 if dv == "L" else int(W - w)
    return frozenset((int(r0 + r), int(c0 + c)) for r, c in shape)


def _mirror_then_corner(
    pts: set[tuple[int, int]] | frozenset[tuple[int, int]],
    dh: str,
    dv: str,
    H: int,
    W: int,
) -> frozenset[tuple[int, int]]:
    shape, h, w = _normalize_points(pts)
    mirrored = _flip_shape(shape, h, w, flip_v=(dh in ("T", "B")), flip_h=(dv in ("L", "R")))
    return _place_shape_in_corner(mirrored, h, w, dh, dv, H, W)


def _slide_only_corner(
    pts: set[tuple[int, int]] | frozenset[tuple[int, int]],
    dh: str,
    dv: str,
    H: int,
    W: int,
) -> frozenset[tuple[int, int]]:
    shape, h, w = _normalize_points(pts)
    return _place_shape_in_corner(shape, h, w, dh, dv, H, W)


def _translate_to_center_inner_corner(
    pts: list[tuple[int, int]] | set[tuple[int, int]] | frozenset[tuple[int, int]],
    hs: str,
    vs: str,
    r_split: int,
    c_split: int,
) -> frozenset[tuple[int, int]]:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    rmin, rmax = int(min(rs)), int(max(rs))
    cmin, cmax = int(min(cs)), int(max(cs))
    dr = int((r_split - 1) - rmax) if hs == "T" else int((r_split + 1) - rmin)
    dc = int((c_split - 1) - cmax) if vs == "L" else int((c_split + 1) - cmin)
    return frozenset((int(r + dr), int(c + dc)) for r, c in pts)


def _target_points_for_region(
    pts: list[tuple[int, int]] | set[tuple[int, int]] | frozenset[tuple[int, int]],
    hs: str,
    vs: str,
    E: set[str],
    H: int,
    W: int,
    r_split: int | None,
    c_split: int | None,
) -> frozenset[tuple[int, int]]:
    if _touches_edges(hs, vs) & E:
        dh, dv = _pick_corner_dest(hs, vs, E, r_split, c_split)
        return _mirror_then_corner(frozenset(pts), dh, dv, H, W)

    assert r_split is not None and c_split is not None
    return _translate_to_center_inner_corner(frozenset(pts), hs, vs, r_split, c_split)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut a sprite along border-control rays. Regions touching control edges are
    mirrored into outer corners; the unique non-touching quadrant is preserved
    without mirroring and moved to the split intersection's inner corner.
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
        for rr, cc in _target_points_for_region(pts, hs, vs, E, H, W, r_split, c_split):
            if 0 <= int(rr) < H and 0 <= int(cc) < W:
                out[int(rr), int(cc)] = int(fg)

    return out


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    for _ in range(int(target_px) - 1):
        neighbors = set()
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = int(r + dr), int(c + dc)
                if (int(nr), int(nc)) not in cells:
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
    rs = [int(r) for r, _ in cells]
    cs = [int(c) for _, c in cells]
    return frozenset((int(r - min(rs)), int(c - min(cs))) for r, c in cells)


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2

    ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]
    E = set(ctrl_edges)
    has_v = has_h = True

    c_split = random.randint(int(W * 0.4), int(W * 0.6))
    r_split = random.randint(int(H * 0.4), int(H * 0.6))
    ctrl_pts = [
        (0 if ctrl_edges[0] == "top" else int(H - 1), int(c_split)),
        (int(r_split), 0 if ctrl_edges[1] == "left" else int(W - 1)),
    ]

    all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    kept = [reg for reg in all_regions if _touches_edges(*reg) & E]
    center_regs = [reg for reg in all_regions if reg not in kept]
    if len(center_regs) != 1:
        return None
    center_reg = center_regs[0]

    sprite_h = random.randint(6, 8)
    sprite_w = random.randint(8, 10)

    frac_h = random.uniform(0.4, 0.6)
    frac_w = random.uniform(0.4, 0.6)
    sprite_r0 = int(r_split - int(frac_h * sprite_h))
    sprite_c0 = int(c_split - int(frac_w * sprite_w))
    sprite_r0 = max(4, min(int(sprite_r0), int(H - 5 - sprite_h + 1)))
    sprite_c0 = max(4, min(int(sprite_c0), int(W - 5 - sprite_w + 1)))
    if sprite_r0 > int(H - 5 - sprite_h + 1) or sprite_c0 > int(W - 5 - sprite_w + 1):
        return None
    if int(r_split) <= int(sprite_r0) or int(r_split) >= int(sprite_r0 + sprite_h - 1):
        return None
    if int(c_split) <= int(sprite_c0) or int(c_split) >= int(sprite_c0 + sprite_w - 1):
        return None

    sprite_r1 = int(sprite_r0 + sprite_h - 1)
    sprite_c1 = int(sprite_c0 + sprite_w - 1)

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = int(sprite_r0 if hs == "T" else r_split + 1)
        r1 = int(r_split - 1 if hs == "T" else sprite_r1)
        c0 = int(sprite_c0 if vs == "L" else c_split + 1)
        c1 = int(c_split - 1 if vs == "L" else sprite_c1)
        return int(r0), int(r1), int(c0), int(c1)

    for reg in all_regions:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None
        if (int(r1 - r0 + 1) * int(c1 - c0 + 1)) < 3:
            return None

    split_pixels: set[tuple[int, int]] = set()
    n_v = random.randint(2, 4)
    rows = [int(r) for r in range(int(sprite_r0), int(sprite_r1) + 1) if int(r) != int(r_split)]
    if len(rows) < n_v:
        return None
    for r in random.sample(rows, n_v):
        split_pixels.add((int(r), int(c_split)))

    n_h = random.randint(2, 4)
    cols = [int(c) for c in range(int(sprite_c0), int(sprite_c1) + 1) if int(c) != int(c_split)]
    if len(cols) < n_h:
        return None
    for c in random.sample(cols, n_h):
        split_pixels.add((int(r_split), int(c)))

    n_split = len(split_pixels)
    area = int(sprite_h * sprite_w)
    target_density = random.uniform(0.44, 0.58)
    target_total = int(target_density * area)
    budget_pieces = int(target_total - n_split)

    if budget_pieces < int(len(all_regions) * 3):
        return None

    piece_budgets: list[int] = []
    remaining = int(budget_pieces)
    for i, reg in enumerate(all_regions):
        r0, r1, c0, c1 = sub_bbox(*reg)
        max_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        remaining_slots = int(len(all_regions) - i - 1)
        min_rest = int(remaining_slots * 3)
        hi_px = int(min(max_area, remaining - min_rest))
        if hi_px < 3:
            return None
        if i == int(len(all_regions) - 1):
            px = int(remaining)
        else:
            avg = int(remaining // (len(all_regions) - i))
            lo_px = int(max(3, avg - 2))
            hi_band = int(min(hi_px, avg + 2))
            if lo_px > hi_band:
                lo_px = 3
                hi_band = hi_px
            if lo_px > hi_band:
                return None
            px = random.randint(lo_px, hi_band)
        if px > max_area:
            return None
        piece_budgets.append(int(px))
        remaining -= int(px)

    if remaining != 0 or any(int(b) < 3 for b in piece_budgets):
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    legacy_output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}

    for idx, reg in enumerate(all_regions):
        hs, vs = reg
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = int(r1 - r0 + 1)
        sub_w = int(c1 - c0 + 1)
        target_px = int(piece_budgets[idx])

        piece = _grow_piece(sub_h, sub_w, target_px)
        if len(piece) < 3:
            return None

        ph = int(max(r for r, _ in piece) + 1)
        pw = int(max(c for _, c in piece) + 1)
        if ph < 2 or pw < 2:
            return None

        inp_r0 = int((r1 - ph + 1) if hs == "T" else r0)
        inp_c0 = int((c1 - pw + 1) if vs == "L" else c0)
        inp_pts = frozenset((int(r + inp_r0), int(c + inp_c0)) for r, c in piece)

        for r, c in inp_pts:
            if int(r) < 4 or int(r) > int(H - 5) or int(c) < 4 or int(c) > int(W - 5):
                return None
            if (vs == "L" and int(c) >= int(c_split)) or (vs == "R" and int(c) <= int(c_split)):
                return None
            if (hs == "T" and int(r) >= int(r_split)) or (hs == "B" and int(r) <= int(r_split)):
                return None

        input_pieces[reg] = inp_pts

        out_pts = _target_points_for_region(inp_pts, hs, vs, E, H, W, r_split, c_split)
        if reg in kept:
            dh, dv = _pick_corner_dest(hs, vs, E, r_split, c_split)
            legacy_pts = _slide_only_corner(inp_pts, dh, dv, H, W)
            if out_pts == legacy_pts:
                return None
            legacy_output_pieces[reg] = legacy_pts

        for r, c in out_pts:
            if not (0 <= int(r) < H and 0 <= int(c) < W):
                return None
            if (vs == "L" and int(c) >= int(c_split)) or (vs == "R" and int(c) <= int(c_split)):
                return None
            if (hs == "T" and int(r) >= int(r_split)) or (hs == "B" and int(r) <= int(r_split)):
                return None
            if (int(r), int(c)) in split_pixels:
                return None
        output_pieces[reg] = out_pts

    variant_delta = sum(len(output_pieces[reg] ^ legacy_output_pieces[reg]) for reg in kept)
    if variant_delta < 6:
        return None

    center_pts = output_pieces[center_reg]
    if len(center_pts) < 3:
        return None

    all_inp_fg: set[tuple[int, int]] = set(split_pixels)
    for pts in input_pieces.values():
        all_inp_fg |= set(pts)

    fg_arr = np.array(list(all_inp_fg))
    ah = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    aw = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if ah < 5 or ah > 8 or aw < 8 or aw > 10:
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in all_inp_fg:
        mask[int(r), int(c)] = 1
    if not is_contiguous(mask):
        return None

    inp = np.full((H, W), int(bg), dtype=int)
    for r, c in all_inp_fg:
        inp[int(r), int(c)] = int(fg)
    for cr, cc in ctrl_pts:
        inp[int(cr), int(cc)] = int(ctrl_color)

    out = np.full((H, W), int(bg), dtype=int)
    occupied_out: set[tuple[int, int]] = set()
    for reg in all_regions:
        pts = output_pieces[reg]
        if occupied_out & pts:
            return None
        occupied_out |= set(pts)
        for r, c in pts:
            out[int(r), int(c)] = int(fg)
    for r, c in split_pixels:
        out[int(r), int(c)] = int(fg)

    for cr, cc in ctrl_pts:
        if int(cr) == 0:
            fg_rows = [int(r) for r in range(H) if int(out[int(r), int(cc)]) == int(fg)]
            if fg_rows:
                for r in range(int(max(fg_rows)) + 1):
                    if int(out[int(r), int(cc)]) == int(bg):
                        out[int(r), int(cc)] = int(ctrl_color)
        elif int(cr) == int(H - 1):
            fg_rows = [int(r) for r in range(H) if int(out[int(r), int(cc)]) == int(fg)]
            if fg_rows:
                for r in range(int(H - 1), int(min(fg_rows)) - 1, -1):
                    if int(out[int(r), int(cc)]) == int(bg):
                        out[int(r), int(cc)] = int(ctrl_color)
        elif int(cc) == 0:
            fg_cols = [int(c) for c in range(W) if int(out[int(cr), int(c)]) == int(fg)]
            if fg_cols:
                for c in range(int(max(fg_cols)) + 1):
                    if int(out[int(cr), int(c)]) == int(bg):
                        out[int(cr), int(c)] = int(ctrl_color)
        elif int(cc) == int(W - 1):
            fg_cols = [int(c) for c in range(W) if int(out[int(cr), int(c)]) == int(fg)]
            if fg_cols:
                for c in range(int(W - 1), int(min(fg_cols)) - 1, -1):
                    if int(out[int(cr), int(c)]) == int(bg):
                        out[int(cr), int(c)] = int(ctrl_color)
    for cr, cc in ctrl_pts:
        out[int(cr), int(cc)] = int(ctrl_color)

    actual_density = len(all_inp_fg) / float(ah * aw)
    if actual_density < 0.38 or actual_density > 0.62:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 600 attempts")
