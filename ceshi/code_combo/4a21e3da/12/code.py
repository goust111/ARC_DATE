# Combo variant of 4a21e3da (index 12)
# combo_spec: D5/1+D17/0
# Name: 镜像翻转+颜色编码指令
#
# Design analysis for the fused variant
# a) Concretization / behavior trace
#    - The board keeps the original family: one connected orange sprite near the center,
#      one vertical split and one horizontal split implied by two border control pixels,
#      and intermittent control-color rays drawn only in the output.
#    - The fusion changes Stage 2 in a coupled way. Edge position still decides which
#      ray is emitted and which rectangular regions survive. Control color now decides
#      the mirror axis used when a surviving region slides to its corner:
#        red(2)  -> horizontal mirror inside the region's own bbox
#        teal(8) -> vertical mirror inside the region's own bbox
#    - A region can touch one control edge or both. It accumulates instructions from all
#      touched control edges before corner placement. Example with a red top control and
#      a teal left control:
#        top-right region touches only top  -> horizontal flip only
#        bottom-left region touches only left -> vertical flip only
#        top-left region touches both -> horizontal + vertical flip
#        bottom-right touches neither -> vanishes
#      The original task would slide these three kept pieces unchanged; the fused task
#      changes their silhouettes according to the touched control colors.
# b) Complexity calibration
#    - Only two instruction colors are used, and every generated sample contains both.
#      This keeps the legend small while giving repeated evidence for the mapping.
#    - Each sample deliberately contains three surviving regions with three transform
#      profiles: horizontal-only, vertical-only, and both. That is enough support for
#      humans to infer the color-to-axis rule from a few examples.
# c) Hidden assumptions audit
#    - Original solver assumption: exactly one border control color. Broken now because
#      the border contains red and teal controls.
#    - Original solver assumption: all surviving regions use the same transform. Broken
#      now because transform depends on which control-colored edges the region touches.
#    - Original Stage 2 preserved local offsets by pure translation. Broken now because
#      local row/col order may be reversed before placement.
#    - Original generator assumption: one shared control color for all rays. Broken now
#      because each ray must retain its own color so the instruction stays visible.
#    - New solver assumption: all border non-background cells are control markers and
#      all interior non-background cells are sprite pixels. The generator preserves the
#      original >=4 margin so no sprite pixel appears on the border.
# d) Placement legality
#    - Generated pieces stay inside their pre-cut sub-boxes in the input.
#    - Output placement uses the same bbox dimensions as the input piece because mirrors
#      do not change width/height; the generator checks each transformed piece remains on
#      the correct side of both split lines after corner placement.
#    - The vanishing quadrant still contains an input-only fragment so the pre-cut sprite
#      remains a single connected component.
# e) Visual observability
#    - A red-only region and a teal-only region appear in every sample, so the two colors
#      are not just decorative; each visibly produces a different mirror axis.
#    - The generator rejects any kept piece whose required mirror would look identical to
#      simple slide-only behavior, preventing accidental collapse back to the base task.
# f) Visual cleanliness
#    - The palette stays small: blue background(1), orange sprite(7), red control(2),
#      teal control(8). Rays reuse the control color, moved pieces remain orange, and the
#      split-line pixels stay in the center, so layers remain easy to parse.
# g) Color-rule inferability
#    - There are only two semantic colors and both repeat within each board as border
#      markers plus rays. The only changing factor between otherwise similar kept regions
#      is which colored control edge they touch, making the legend recoverable by sight.
# h) Quantitative anti-degeneration plan
#    - For generated inputs, compare the fused output against a slide-only baseline on the
#      same input. The metric `delta_vs_slide` must be positive for every sample; if not,
#      the generator has produced a case where color-coded mirrors are not actually visible.
# i) New solver assumptions checked against the generator
#    - There is exactly one vertical split and one horizontal split.
#    - The two control colors are always one red and one teal.
#    - Every surviving region's required mirror is visibly non-trivial.

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous

HFLIP_CTRL = 2
VFLIP_CTRL = 8


def _normalize_piece(
    cells: set[tuple[int, int]] | frozenset[tuple[int, int]],
) -> tuple[frozenset[tuple[int, int]], int, int]:
    rs = [int(r) for r, _ in cells]
    cs = [int(c) for _, c in cells]
    r0 = int(min(rs))
    c0 = int(min(cs))
    piece = frozenset((int(r - r0), int(c - c0)) for r, c in cells)
    h = int(max(r for r, _ in piece) + 1)
    w = int(max(c for _, c in piece) + 1)
    return piece, h, w


def _flip_piece(
    piece: set[tuple[int, int]] | frozenset[tuple[int, int]],
    *,
    flip_v: bool,
    flip_h: bool,
) -> frozenset[tuple[int, int]]:
    norm, h, w = _normalize_piece(piece)
    out = set()
    for r, c in norm:
        rr = int(h - 1 - r) if flip_v else int(r)
        cc = int(w - 1 - c) if flip_h else int(c)
        out.add((int(rr), int(cc)))
    return frozenset(out)


def _flip_is_visible(
    piece: set[tuple[int, int]] | frozenset[tuple[int, int]],
    *,
    flip_v: bool,
    flip_h: bool,
) -> bool:
    norm, _h, _w = _normalize_piece(piece)
    return _flip_piece(norm, flip_v=flip_v, flip_h=flip_h) != norm


def _place_shape_in_corner(
    piece: set[tuple[int, int]] | frozenset[tuple[int, int]],
    H: int,
    W: int,
    dh: str,
    dv: str,
) -> frozenset[tuple[int, int]]:
    norm, h, w = _normalize_piece(piece)
    r0 = 0 if dh == "T" else int(H - h)
    c0 = 0 if dv == "L" else int(W - w)
    return frozenset((int(r0 + r), int(c0 + c)) for r, c in norm)


def _transform_piece_to_corner(
    piece: set[tuple[int, int]] | frozenset[tuple[int, int]],
    H: int,
    W: int,
    dh: str,
    dv: str,
    *,
    flip_v: bool,
    flip_h: bool,
) -> frozenset[tuple[int, int]]:
    flipped = _flip_piece(piece, flip_v=flip_v, flip_h=flip_h)
    return _place_shape_in_corner(flipped, H, W, dh, dv)


def _slide_only_corner(
    piece: set[tuple[int, int]] | frozenset[tuple[int, int]],
    H: int,
    W: int,
    dh: str,
    dv: str,
) -> frozenset[tuple[int, int]]:
    return _place_shape_in_corner(piece, H, W, dh, dv)


def _piece_hw(piece: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> tuple[int, int]:
    _norm, h, w = _normalize_piece(piece)
    return int(h), int(w)


def _edge_of(r: int, c: int, H: int, W: int) -> str:
    if int(r) == 0:
        return "top"
    if int(r) == int(H - 1):
        return "bottom"
    if int(c) == 0:
        return "left"
    if int(c) == int(W - 1):
        return "right"
    raise ValueError("control point is not on the border")


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
    return str(dh), str(dv)


def _region_tags_for_pixel(r: int, c: int, r_split: int | None, c_split: int | None) -> tuple[str, str]:
    hs = "A"
    vs = "A"
    if r_split is not None:
        hs = "T" if int(r) < int(r_split) else "B"
    if c_split is not None:
        vs = "L" if int(c) < int(c_split) else "R"
    return hs, vs


def _region_flip_flags(hs: str, vs: str, E: set[str], edge_to_color: dict[str, int]) -> tuple[bool, bool]:
    touched = _touches_edges(hs, vs) & E
    flip_h = any(int(edge_to_color[edge]) == int(HFLIP_CTRL) for edge in touched)
    flip_v = any(int(edge_to_color[edge]) == int(VFLIP_CTRL) for edge in touched)
    return bool(flip_v), bool(flip_h)


def _draw_rays(
    g: np.ndarray,
    out: np.ndarray,
    ctrl_pts: list[tuple[int, int, int]],
    fg: int,
    bg: int,
) -> None:
    H, W = g.shape
    for r, c, color in ctrl_pts:
        if int(r) == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, int(c)] == int(fg)).flatten()]
            if rows:
                stop = int(max(rows))
                for rr in range(0, int(stop) + 1):
                    if int(g[int(rr), int(c)]) == int(bg):
                        out[int(rr), int(c)] = int(color)
        elif int(r) == int(H - 1):
            rows = [int(rr) for rr in np.argwhere(g[:, int(c)] == int(fg)).flatten()]
            if rows:
                stop = int(min(rows))
                for rr in range(int(H - 1), int(stop) - 1, -1):
                    if int(g[int(rr), int(c)]) == int(bg):
                        out[int(rr), int(c)] = int(color)
        elif int(c) == 0:
            cols = [int(cc) for cc in np.argwhere(g[int(r), :] == int(fg)).flatten()]
            if cols:
                stop = int(max(cols))
                for cc in range(0, int(stop) + 1):
                    if int(g[int(r), int(cc)]) == int(bg):
                        out[int(r), int(cc)] = int(color)
        elif int(c) == int(W - 1):
            cols = [int(cc) for cc in np.argwhere(g[int(r), :] == int(fg)).flatten()]
            if cols:
                stop = int(min(cols))
                for cc in range(int(W - 1), int(stop) - 1, -1):
                    if int(g[int(r), int(cc)]) == int(bg):
                        out[int(r), int(cc)] = int(color)
        out[int(r), int(c)] = int(color)


def _parse_input(grid: np.ndarray) -> tuple[int, int, int, int | None, int | None, set[str], dict[str, int], list[tuple[int, int, int]]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})

    ctrl_pts: list[tuple[int, int, int]] = []
    border_nonbg: set[int] = set()
    for r in range(H):
        for c in range(W):
            if int(g[r, c]) == int(bg):
                continue
            if int(r) in (0, H - 1) or int(c) in (0, W - 1):
                ctrl_pts.append((int(r), int(c), int(g[r, c])))
                border_nonbg.add(int(g[r, c]))

    assert len(ctrl_pts) == 2
    assert border_nonbg == {int(HFLIP_CTRL), int(VFLIP_CTRL)}
    fg_candidates = [int(color) for color in nonbg if int(color) not in border_nonbg]
    assert len(fg_candidates) == 1
    fg = int(fg_candidates[0])

    E: set[str] = set()
    edge_to_color: dict[str, int] = {}
    c_split: int | None = None
    r_split: int | None = None
    for r, c, color in ctrl_pts:
        edge = _edge_of(r, c, H, W)
        E.add(edge)
        edge_to_color[edge] = int(color)
        if edge in ("top", "bottom"):
            c_split = int(c)
        else:
            r_split = int(r)

    assert c_split is not None and r_split is not None
    return bg, fg, H, r_split, c_split, E, edge_to_color, ctrl_pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Two border controls define one vertical split and one horizontal split.
    - Rays keep the color of their originating control.
    - Surviving regions still move to the corner implied by the cut geometry.
    - Red-touched regions mirror horizontally before corner placement.
    - Teal-touched regions mirror vertically before corner placement.
    - A region touching both colors applies both mirrors; untouched regions vanish.
    """
    g = np.array(grid, dtype=int)
    bg, fg, H, r_split, c_split, E, edge_to_color, ctrl_pts = _parse_input(g)
    W = int(g.shape[1])

    out = np.full((H, W), int(bg), dtype=int)
    _draw_rays(g, out, ctrl_pts, fg, bg)

    for r, c in np.argwhere(g == int(fg)):
        rr = int(r)
        cc = int(c)
        if int(rr) == int(r_split) or int(cc) == int(c_split):
            out[int(rr), int(cc)] = int(fg)

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == int(fg)):
        rr = int(r)
        cc = int(c)
        if int(rr) == int(r_split) or int(cc) == int(c_split):
            continue
        hs, vs = _region_tags_for_pixel(rr, cc, r_split, c_split)
        regions.setdefault((hs, vs), []).append((int(rr), int(cc)))

    for (hs, vs), pts in regions.items():
        touched = _touches_edges(hs, vs) & E
        if not touched:
            continue
        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        flip_v, flip_h = _region_flip_flags(hs, vs, E, edge_to_color)
        out_pts = _transform_piece_to_corner(pts, H, W, dh, dv, flip_v=flip_v, flip_h=flip_h)
        for rr, cc in out_pts:
            out[int(rr), int(cc)] = int(fg)

    return out


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    for _ in range(int(target_px) - 1):
        neighbors = set()
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr = int(r + dr)
                nc = int(c + dc)
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
    norm, _h, _w = _normalize_piece(cells)
    return norm


def _output_caps(
    hs: str,
    vs: str,
    H: int,
    W: int,
    r_split: int,
    c_split: int,
) -> tuple[int, int]:
    cap_h = int(r_split) if hs == "T" else int(H - 1 - r_split)
    cap_w = int(c_split) if vs == "L" else int(W - 1 - c_split)
    return int(cap_h), int(cap_w)


def _build_split_pixels(
    sprite_r0: int,
    sprite_r1: int,
    sprite_c0: int,
    sprite_c1: int,
    r_split: int,
    c_split: int,
) -> set[tuple[int, int]]:
    split_pixels: set[tuple[int, int]] = set()

    top_rows = [int(r) for r in range(int(sprite_r0), int(r_split))]
    bottom_rows = [int(r) for r in range(int(r_split + 1), int(sprite_r1 + 1))]
    left_cols = [int(c) for c in range(int(sprite_c0), int(c_split))]
    right_cols = [int(c) for c in range(int(c_split + 1), int(sprite_c1 + 1))]

    if not top_rows or not bottom_rows or not left_cols or not right_cols:
        return set()

    split_pixels.add((random.choice(top_rows), int(c_split)))
    split_pixels.add((random.choice(bottom_rows), int(c_split)))
    split_pixels.add((int(r_split), random.choice(left_cols)))
    split_pixels.add((int(r_split), random.choice(right_cols)))

    extra_v_pool = [int(r) for r in top_rows + bottom_rows if (int(r), int(c_split)) not in split_pixels]
    extra_h_pool = [int(c) for c in left_cols + right_cols if (int(r_split), int(c)) not in split_pixels]
    if extra_v_pool and random.random() < 0.7:
        split_pixels.add((random.choice(extra_v_pool), int(c_split)))
    if extra_h_pool and random.random() < 0.7:
        split_pixels.add((int(r_split), random.choice(extra_h_pool)))
    return split_pixels


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg = 1, 7

    vertical_edge = random.choice(["top", "bottom"])
    horizontal_edge = random.choice(["left", "right"])
    ctrl_edges = [vertical_edge, horizontal_edge]
    if random.random() < 0.5:
        ctrl_colors = [HFLIP_CTRL, VFLIP_CTRL]
    else:
        ctrl_colors = [VFLIP_CTRL, HFLIP_CTRL]

    c_split = random.randint(int(W * 0.4), int(W * 0.6))
    r_split = random.randint(int(H * 0.4), int(H * 0.6))

    ctrl_pts: list[tuple[int, int, int]] = []
    edge_to_color: dict[str, int] = {}
    for edge, color in zip(ctrl_edges, ctrl_colors):
        if edge == "top":
            ctrl_pts.append((0, int(c_split), int(color)))
        elif edge == "bottom":
            ctrl_pts.append((int(H - 1), int(c_split), int(color)))
        elif edge == "left":
            ctrl_pts.append((int(r_split), 0, int(color)))
        else:
            ctrl_pts.append((int(r_split), int(W - 1), int(color)))
        edge_to_color[edge] = int(color)

    E = set(ctrl_edges)
    all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    kept = [reg for reg in all_regions if _touches_edges(*reg) & E]
    vanish = [reg for reg in all_regions if reg not in kept]
    assert len(kept) == 3 and len(vanish) == 1

    sprite_h = random.randint(6, 8)
    sprite_w = random.randint(8, 10)

    frac_c = random.uniform(0.4, 0.6)
    sprite_c0 = int(c_split - int(frac_c * sprite_w))
    sprite_c0 = max(4, min(sprite_c0, int(W - 5 - sprite_w + 1)))
    if sprite_c0 > int(W - 5 - sprite_w + 1):
        return None
    if not (int(sprite_c0) < int(c_split) < int(sprite_c0 + sprite_w - 1)):
        return None

    frac_r = random.uniform(0.4, 0.6)
    sprite_r0 = int(r_split - int(frac_r * sprite_h))
    sprite_r0 = max(4, min(sprite_r0, int(H - 5 - sprite_h + 1)))
    if sprite_r0 > int(H - 5 - sprite_h + 1):
        return None
    if not (int(sprite_r0) < int(r_split) < int(sprite_r0 + sprite_h - 1)):
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

    split_pixels = _build_split_pixels(sprite_r0, sprite_r1, sprite_c0, sprite_c1, r_split, c_split)
    if len(split_pixels) < 4:
        return None

    n_split = len(split_pixels)
    area = int(sprite_h * sprite_w)
    target_lo = max(int(0.46 * area), int(n_split + 15))
    target_hi = int(0.60 * area)
    if target_lo > target_hi:
        return None
    target_total = random.randint(int(target_lo), int(target_hi))

    vanish_budget_hi = min(5, int(target_total - n_split - 12))
    if vanish_budget_hi < 3:
        return None
    vanish_budget = random.randint(3, int(vanish_budget_hi))
    budget_pieces = int(target_total - n_split - vanish_budget)
    if budget_pieces < 12:
        return None

    piece_budgets: list[int] = []
    remaining = int(budget_pieces)
    for idx, reg in enumerate(kept):
        hs, vs = reg
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = int(r1 - r0 + 1)
        sub_w = int(c1 - c0 + 1)
        cap_h, cap_w = _output_caps(hs, vs, H, W, r_split, c_split)
        max_area = int(min(sub_h * sub_w, cap_h * cap_w))
        min_px = 4
        if max_area < min_px:
            return None
        slots_left = int(len(kept) - idx - 1)
        hi_px = min(max_area, int(remaining - slots_left * min_px))
        lo_px = min_px
        if idx < len(kept) - 1:
            avg = int(remaining // (len(kept) - idx))
            lo_px = max(lo_px, int(avg - 2))
            hi_px = min(hi_px, int(avg + 2))
        if lo_px > hi_px:
            return None
        px = hi_px if idx == len(kept) - 1 else random.randint(int(lo_px), int(hi_px))
        piece_budgets.append(int(px))
        remaining -= int(px)

    if remaining != 0:
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    slide_only_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}

    for idx, reg in enumerate(kept):
        hs, vs = reg
        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        flip_v, flip_h = _region_flip_flags(hs, vs, E, edge_to_color)
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = int(r1 - r0 + 1)
        sub_w = int(c1 - c0 + 1)
        cap_h, cap_w = _output_caps(hs, vs, H, W, r_split, c_split)
        target_px = int(piece_budgets[idx])

        placed = False
        for _ in range(80):
            piece = _grow_piece(sub_h, sub_w, target_px)
            if len(piece) != int(target_px):
                continue
            ph, pw = _piece_hw(piece)
            if ph > sub_h or pw > sub_w or ph > cap_h or pw > cap_w:
                continue
            if flip_h and pw < 2:
                continue
            if flip_v and ph < 2:
                continue
            if not _flip_is_visible(piece, flip_v=flip_v, flip_h=flip_h):
                continue

            inp_r0 = int(r1 - ph + 1) if hs == "T" else int(r0)
            inp_c0 = int(c1 - pw + 1) if vs == "L" else int(c0)
            inp_pts = frozenset((int(r + inp_r0), int(c + inp_c0)) for r, c in piece)

            ok = True
            for r, c in inp_pts:
                if r < 4 or r > int(H - 5) or c < 4 or c > int(W - 5):
                    ok = False
                    break
                if hs == "T" and r >= int(r_split):
                    ok = False
                    break
                if hs == "B" and r <= int(r_split):
                    ok = False
                    break
                if vs == "L" and c >= int(c_split):
                    ok = False
                    break
                if vs == "R" and c <= int(c_split):
                    ok = False
                    break
            if not ok:
                continue

            out_pts = _transform_piece_to_corner(inp_pts, H, W, dh, dv, flip_v=flip_v, flip_h=flip_h)
            slide_pts = _slide_only_corner(inp_pts, H, W, dh, dv)
            if out_pts == slide_pts:
                continue

            for r, c in out_pts:
                if hs == "T" and r >= int(r_split):
                    ok = False
                    break
                if hs == "B" and r <= int(r_split):
                    ok = False
                    break
                if vs == "L" and c >= int(c_split):
                    ok = False
                    break
                if vs == "R" and c <= int(c_split):
                    ok = False
                    break
            if not ok:
                continue

            input_pieces[reg] = inp_pts
            output_pieces[reg] = out_pts
            slide_only_pieces[reg] = slide_pts
            placed = True
            break

        if not placed:
            return None

    all_inp_fg: set[tuple[int, int]] = set()
    for pts in input_pieces.values():
        all_inp_fg |= set(pts)
    all_inp_fg |= set(split_pixels)
    if not all_inp_fg:
        return None

    fg_arr = np.array(list(all_inp_fg))
    bbox_h = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    bbox_w = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if bbox_h < 6 or bbox_h > 8 or bbox_w < 8 or bbox_w > 10:
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in all_inp_fg:
        mask[int(r), int(c)] = 1
    if not is_contiguous(mask):
        return None

    vanish_fg: set[tuple[int, int]] = set()
    vhs, vvs = vanish[0]
    vr0, vr1, vc0, vc1 = sub_bbox(vhs, vvs)
    vh = int(vr1 - vr0 + 1)
    vw = int(vc1 - vc0 + 1)
    if vh < 2 or vw < 2:
        return None
    for _ in range(60):
        piece = _grow_piece(vh, vw, int(vanish_budget))
        if len(piece) != int(vanish_budget):
            continue
        pph, ppw = _piece_hw(piece)
        vr_off = int(vr1 - pph + 1) if vhs == "T" else int(vr0)
        vc_off = int(vc1 - ppw + 1) if vvs == "L" else int(vc0)
        vpts = {(int(r + vr_off), int(c + vc_off)) for r, c in piece}
        ok = True
        for r, c in vpts:
            if r < 4 or r > int(H - 5) or c < 4 or c > int(W - 5):
                ok = False
                break
            if (int(r), int(c)) in all_inp_fg:
                ok = False
                break
        if not ok:
            continue
        vanish_fg = vpts
        break
    if not vanish_fg:
        return None

    combined = set(all_inp_fg) | set(vanish_fg)
    mask2 = np.zeros((H, W), dtype=int)
    for r, c in combined:
        mask2[int(r), int(c)] = 1
    if not is_contiguous(mask2):
        return None

    inp = np.full((H, W), int(bg), dtype=int)
    for r, c in combined:
        inp[int(r), int(c)] = int(fg)
    for r, c, color in ctrl_pts:
        inp[int(r), int(c)] = int(color)

    out = np.full((H, W), int(bg), dtype=int)
    for pts in output_pieces.values():
        for r, c in pts:
            out[int(r), int(c)] = int(fg)
    for r, c in split_pixels:
        out[int(r), int(c)] = int(fg)
    _draw_rays(inp, out, ctrl_pts, fg, bg)

    density = len(combined) / float(bbox_h * bbox_w)
    if density < 0.39 or density > 0.62:
        return None

    slide_only = np.full((H, W), int(bg), dtype=int)
    for pts in slide_only_pieces.values():
        for r, c in pts:
            slide_only[int(r), int(c)] = int(fg)
    for r, c in split_pixels:
        slide_only[int(r), int(c)] = int(fg)
    _draw_rays(inp, slide_only, ctrl_pts, fg, bg)
    if np.array_equal(out, slide_only):
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


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Helper for external validation scripts.

    Returns counts for the three kept-region transform profiles plus the pixel-level
    symmetric-difference size against a slide-only baseline.
    """
    g = np.array(grid, dtype=int)
    bg, fg, H, r_split, c_split, E, edge_to_color, _ctrl_pts = _parse_input(g)
    W = int(g.shape[1])

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == int(fg)):
        rr = int(r)
        cc = int(c)
        if int(rr) == int(r_split) or int(cc) == int(c_split):
            continue
        hs, vs = _region_tags_for_pixel(rr, cc, r_split, c_split)
        regions.setdefault((hs, vs), []).append((int(rr), int(cc)))

    h_only = 0
    v_only = 0
    hv = 0
    delta_vs_slide = 0
    for (hs, vs), pts in regions.items():
        touched = _touches_edges(hs, vs) & E
        if not touched:
            continue
        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        flip_v, flip_h = _region_flip_flags(hs, vs, E, edge_to_color)
        if flip_h and flip_v:
            hv += 1
        elif flip_h:
            h_only += 1
        elif flip_v:
            v_only += 1
        out_pts = _transform_piece_to_corner(pts, H, W, dh, dv, flip_v=flip_v, flip_h=flip_h)
        slide_pts = _slide_only_corner(pts, H, W, dh, dv)
        delta_vs_slide += int(len(out_pts ^ slide_pts))

    return {
        "h_only_regions": int(h_only),
        "v_only_regions": int(v_only),
        "hv_regions": int(hv),
        "delta_vs_slide": int(delta_vs_slide),
    }
