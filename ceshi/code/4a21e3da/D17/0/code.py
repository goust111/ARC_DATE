# Variant of 4a21e3da
# Dimension: D17 — 控制像素边缘位置含义 (symbol_instructions)
# Change: 控制像素所在边缘隐式决定射线方向（顶部→向下，左边→向右等）和区域保留规则... → 控制像素颜色编码指令
#
# requested_reference: 58490d8a
# reference_note:
# - The workspace does not contain a readable solver file for `58490d8a` under the
#   allowed paths, so the design below only uses its evaluation JSON as inspiration
#   for "color carries instruction".
#
# design_analysis:
#   a_concretization_and_trace:
#   - The input still contains one connected orange sprite (`fg=7`) near the center and
#     one or two border control pixels on perpendicular edges.
#   - The geometric meaning of edge position is preserved only for locating the cut line
#     and shooting the ray inward. The new symbolic meaning is in the control color:
#       red (`2`)  -> regions touching that control edge slide to their corner unchanged
#       teal (`8`) -> regions touching that control edge rotate 90 deg clockwise, then slide
#   - Rays are colored with the same control color, so the instruction is visible both at
#     the border pixel and along the emitted ray.
#   - Concrete trace with a top control at column 9:
#       old task: top edge implies a vertical split; both kept left/right regions slide up
#       new task, red top control: identical geometry, both kept regions slide up unchanged
#       new task, teal top control: same split line, but both kept regions rotate first
#     With two controls, each region checks the control edges it touches; if any touching
#     control is teal, that region rotates before corner placement.
#   b_complexity_calibration:
#   - Only two instruction colors are used. This keeps the legend small and learnable.
#   - Mixed-color two-control cases are allowed, which creates both transformed and
#     untransformed regions in the same sample without introducing extra operation types.
#   c_old_solver_assumption_audit:
#   - `ctrl = sorted(border_cols)[0]` assumed a single control color on the border.
#   - `fg = next(c for c in nonbg if c != ctrl)` assumed exactly one non-control color.
#   - `ctrl_pts = argwhere(g == ctrl)` missed any second control color entirely.
#   - Stage 1 painted every ray with one shared control color.
#   - Stage 2 always translated pieces; that was harmless before because all kept regions
#     used the same motion type.
#   - The generator also assumed one shared `ctrl_color` and derived output by translation
#     only, so it would silently erase the new color-coded instruction semantics.
#   d_placement_legality:
#   - Border non-background cells are reserved for control pixels only; the sprite keeps
#     a margin >= 4 from every edge, so solver-side color role detection stays unambiguous.
#   - Every kept piece is generated inside its pre-cut sub-box, then its output placement
#     is checked against the split-line side constraints after the optional rotation.
#   - Vanishing pieces remain input-only and are still checked for contiguity with the
#     centered sprite.
#   e_visual_observability:
#   - Blue/teal control edges must actually change visible shape orientation, so any piece
#     that will be rotated is forced to be non-invariant under a 90 deg rotation.
#   - Mixed two-control cases are common, producing one red-only region and one blue-affected
#     region in the same board, which makes the color/function mapping learnable from sight.
#   f_visual_cleanliness:
#   - The board still uses only bg, fg, and at most two control colors. Rays stay on their
#     own split line; moved pieces live in corners; split-line fg stays near the center.
#   g_color_rule_inference:
#   - The only semantic color variation is `2` vs `8`, and both appear as rays plus border
#     markers. No hidden numeric ordering is used; the human-visible hue difference is the
#     entire cue.
#   h_variant_metric_plan:
#   - After implementation, measure per-sample `blue_kept_region_count` and compare the new
#     solver against the original solver on the same generated inputs. For every sample with
#     `blue_kept_region_count > 0`, the outputs should differ; otherwise the generator would
#     be collapsing back to the original behavior.
#   i_new_solver_assumptions:
#   - At most one vertical split and one horizontal split still hold.
#   - Any border non-background color is a control color; the only interior non-background
#     color is the sprite color.
#   - Teal (`8`) is the only rotating instruction color, and the generator guarantees that
#     every teal-affected piece has a visibly distinct clockwise rotation.

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous

MOVE_CTRL = 2
ROTATE_CTRL = 8


def _normalize_piece(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0 = min(rs)
    c0 = min(cs)
    return frozenset((int(r - r0), int(c - c0)) for r, c in cells)


def _piece_hw(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> tuple[int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return int(max(rs) + 1), int(max(cs) + 1)


def _rotate_piece_cw(
    cells: set[tuple[int, int]] | frozenset[tuple[int, int]], turns: int = 1
) -> frozenset[tuple[int, int]]:
    piece = _normalize_piece(cells)
    for _ in range(int(turns) % 4):
        height, _width = _piece_hw(piece)
        piece = frozenset((int(c), int(height - 1 - r)) for r, c in piece)
        piece = _normalize_piece(piece)
    return piece


def _rotation_is_visible(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    piece = _normalize_piece(cells)
    return _rotate_piece_cw(piece) != piece


def _edge_of(r: int, c: int, H: int, W: int) -> str:
    if int(r) == 0:
        return "top"
    if int(r) == int(H - 1):
        return "bottom"
    if int(c) == 0:
        return "left"
    if int(c) == int(W - 1):
        return "right"
    raise ValueError("control point is not on a border")


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


def _place_piece_to_corner(
    pts: list[tuple[int, int]] | set[tuple[int, int]] | frozenset[tuple[int, int]],
    H: int,
    W: int,
    dh: str,
    dv: str,
    rotate: bool,
) -> frozenset[tuple[int, int]]:
    local = _normalize_piece(set((int(r), int(c)) for r, c in pts))
    if rotate:
        local = _rotate_piece_cw(local)
    ph, pw = _piece_hw(local)
    base_r = 0 if dh == "T" else int(H - ph)
    base_c = 0 if dv == "L" else int(W - pw)
    return frozenset((int(base_r + r), int(base_c + c)) for r, c in local)


def _region_tags_for_pixel(r: int, c: int, r_split: int | None, c_split: int | None) -> tuple[str, str]:
    hs = "A"
    vs = "A"
    if r_split is not None:
        hs = "T" if int(r) < int(r_split) else "B"
    if c_split is not None:
        vs = "L" if int(c) < int(c_split) else "R"
    return hs, vs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Border control pixels still define inward rays and cut lines.
    - Control color now encodes the kept-region action:
      red(2)=slide unchanged, teal(8)=rotate CW then slide.
    - If a kept region touches multiple control edges, teal dominates.
    """
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

    assert ctrl_pts
    fg_candidates = [int(c) for c in nonbg if int(c) not in border_nonbg]
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

    out = np.full((H, W), int(bg), dtype=int)

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

    for r, c, color in ctrl_pts:
        out[int(r), int(c)] = int(color)

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
        hs, vs = _region_tags_for_pixel(r, c, r_split, c_split)
        regions.setdefault((hs, vs), []).append((int(r), int(c)))

    for (hs, vs), pts in regions.items():
        touched = _touches_edges(hs, vs) & E
        if not touched:
            continue

        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        rotate = any(int(edge_to_color[edge]) == int(ROTATE_CTRL) for edge in touched)
        out_pts = _place_piece_to_corner(pts, H, W, dh, dv, rotate)
        for rr, cc in out_pts:
            if 0 <= rr < H and 0 <= cc < W:
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
    return _normalize_piece(cells)


def _output_caps(
    hs: str,
    vs: str,
    H: int,
    W: int,
    has_h: bool,
    has_v: bool,
    r_split: int | None,
    c_split: int | None,
) -> tuple[int, int]:
    if has_h:
        assert r_split is not None
        cap_h = int(r_split) if hs == "T" else int(H - 1 - r_split)
    else:
        cap_h = int(H)

    if has_v:
        assert c_split is not None
        cap_w = int(c_split) if vs == "L" else int(W - 1 - c_split)
    else:
        cap_w = int(W)
    return cap_h, cap_w


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg = 1, 7

    num_ctrl = random.choices([1, 2], weights=[1, 4])[0]
    if num_ctrl == 1:
        ctrl_edges = [random.choice(["top", "bottom", "left", "right"])]
    else:
        ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]

    if num_ctrl == 1:
        ctrl_colors = [random.choice([MOVE_CTRL, ROTATE_CTRL])]
    else:
        ctrl_colors = random.choices(
            [
                [MOVE_CTRL, MOVE_CTRL],
                [ROTATE_CTRL, ROTATE_CTRL],
                [MOVE_CTRL, ROTATE_CTRL],
                [ROTATE_CTRL, MOVE_CTRL],
            ],
            weights=[1, 1, 2, 2],
        )[0]

    c_split = None
    r_split = None
    ctrl_pts: list[tuple[int, int, int]] = []
    edge_to_color: dict[str, int] = {}
    E = set(ctrl_edges)
    has_v = False
    has_h = False

    for edge, color in zip(ctrl_edges, ctrl_colors):
        if edge in ("top", "bottom"):
            c_split = random.randint(int(W * 0.4), int(W * 0.6))
            ctrl_pts.append((0 if edge == "top" else H - 1, c_split, int(color)))
            has_v = True
        else:
            r_split = random.randint(int(H * 0.4), int(H * 0.6))
            ctrl_pts.append((r_split, 0 if edge == "left" else W - 1, int(color)))
            has_h = True
        edge_to_color[edge] = int(color)

    def region_rotates(hs: str, vs: str) -> bool:
        touched = _touches_edges(hs, vs) & E
        return any(int(edge_to_color[edge]) == int(ROTATE_CTRL) for edge in touched)

    if has_h and has_v:
        all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    elif has_v:
        all_regions = [("A", "L"), ("A", "R")]
    else:
        all_regions = [("T", "A"), ("B", "A")]

    kept = [reg for reg in all_regions if _touches_edges(*reg) & E]
    vanish = [reg for reg in all_regions if reg not in kept]

    sprite_h = random.randint(5, 8)
    sprite_w = random.randint(8, 10)

    if has_v:
        assert c_split is not None
        frac = random.uniform(0.4, 0.6)
        sprite_c0 = int(c_split - int(frac * sprite_w))
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
        assert r_split is not None
        frac = random.uniform(0.4, 0.6)
        sprite_r0 = int(r_split - int(frac * sprite_h))
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

    sprite_r1 = int(sprite_r0 + sprite_h - 1)
    sprite_c1 = int(sprite_c0 + sprite_w - 1)

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = int(sprite_r0 if hs in ("T", "A") else r_split + 1)
        r1 = int((r_split - 1 if hs == "T" else sprite_r1) if has_h else sprite_r1)
        c0 = int(sprite_c0 if vs in ("L", "A") else c_split + 1)
        c1 = int((c_split - 1 if vs == "L" else sprite_c1) if has_v else sprite_c1)
        return r0, r1, c0, c1

    for reg in kept + vanish:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None

    split_pixels: set[tuple[int, int]] = set()
    if has_v:
        assert c_split is not None
        n = random.randint(2, 4)
        rows = [r for r in range(sprite_r0, sprite_r1 + 1) if r != r_split]
        if len(rows) < n:
            return None
        for r in random.sample(rows, n):
            split_pixels.add((int(r), int(c_split)))
    if has_h:
        assert r_split is not None
        n = random.randint(2, 4)
        cols = [c for c in range(sprite_c0, sprite_c1 + 1) if c != c_split]
        if len(cols) < n:
            return None
        for c in random.sample(cols, n):
            split_pixels.add((int(r_split), int(c)))

    n_split = len(split_pixels)
    area = sprite_h * sprite_w
    target_density = random.uniform(0.39, 0.60)
    target_total = int(target_density * area)

    n_kept = len(kept)
    n_vanish = len(vanish)
    budget_pieces = int(target_total - n_split)
    if n_vanish > 0:
        vanish_budget = random.randint(3, min(7, max(3, budget_pieces // 3)))
        budget_pieces -= int(vanish_budget)
    else:
        vanish_budget = 0

    if budget_pieces < n_kept * 3:
        return None

    piece_budgets: list[int] = []
    remaining = int(budget_pieces)
    for idx, reg in enumerate(kept):
        hs, vs = reg
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = int(r1 - r0 + 1)
        sub_w = int(c1 - c0 + 1)
        cap_h, cap_w = _output_caps(hs, vs, H, W, has_h, has_v, r_split, c_split)
        max_area = int(min(sub_h * sub_w, cap_h * cap_w))
        if idx == n_kept - 1:
            px = remaining
        else:
            avg = remaining // (n_kept - idx)
            lo_px = max(3, avg - 2)
            hi_px = min(max_area, avg + 2)
            if lo_px > hi_px:
                return None
            px = random.randint(lo_px, hi_px)
        px = max(3, min(px, max_area, remaining - (n_kept - idx - 1) * 3))
        piece_budgets.append(int(px))
        remaining -= int(px)

    if remaining != 0 or any(b < 3 for b in piece_budgets):
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}

    for idx, reg in enumerate(kept):
        hs, vs = reg
        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = int(r1 - r0 + 1)
        sub_w = int(c1 - c0 + 1)
        target_px = int(piece_budgets[idx])
        rotate = region_rotates(hs, vs)
        cap_h, cap_w = _output_caps(hs, vs, H, W, has_h, has_v, r_split, c_split)

        placed = False
        for _ in range(40):
            piece = _grow_piece(sub_h, sub_w, target_px)
            if len(piece) != target_px:
                continue
            if rotate and not _rotation_is_visible(piece):
                continue

            ph, pw = _piece_hw(piece)
            out_shape = _rotate_piece_cw(piece) if rotate else piece
            oh, ow = _piece_hw(out_shape)
            if ph > sub_h or pw > sub_w or oh > cap_h or ow > cap_w:
                continue

            if has_h:
                inp_r0 = int((r1 - ph + 1) if hs == "T" else r0)
            else:
                split_rows = sorted(r for r, c in split_pixels if c == c_split)
                if split_rows:
                    mid = (split_rows[0] + split_rows[-1]) // 2
                    inp_r0 = int(max(r0, min(mid - ph // 2, r1 - ph + 1)))
                else:
                    inp_r0 = int(r0)

            if has_v:
                inp_c0 = int((c1 - pw + 1) if vs == "L" else c0)
            else:
                split_cols = sorted(c for r, c in split_pixels if r == r_split)
                if split_cols:
                    mid = (split_cols[0] + split_cols[-1]) // 2
                    inp_c0 = int(max(c0, min(mid - pw // 2, c1 - pw + 1)))
                else:
                    inp_c0 = int(c0)

            inp_pts = frozenset((int(r + inp_r0), int(c + inp_c0)) for r, c in piece)
            ok = True
            for r, c in inp_pts:
                if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                    ok = False
                    break
                if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                    ok = False
                    break
                if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                    ok = False
                    break
            if not ok:
                continue

            out_pts = _place_piece_to_corner(inp_pts, H, W, dh, dv, rotate)
            for r, c in out_pts:
                if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                    ok = False
                    break
                if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                    ok = False
                    break
            if not ok:
                continue

            input_pieces[reg] = inp_pts
            output_pieces[reg] = out_pts
            placed = True
            break

        if not placed:
            return None

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
        mask[int(r), int(c)] = 1
    if not is_contiguous(mask):
        return None

    vanish_fg: set[tuple[int, int]] = set()
    if vanish:
        for vreg in vanish:
            vhs, vvs = vreg
            vr0, vr1, vc0, vc1 = sub_bbox(vhs, vvs)
            vh = int(vr1 - vr0 + 1)
            vw = int(vc1 - vc0 + 1)
            if vh < 1 or vw < 1:
                return None
            max_px = min(vanish_budget, vh * vw)
            if max_px < 3:
                return None
            piece = _grow_piece(vh, vw, max_px)
            if len(piece) < 3:
                return None
            pph, ppw = _piece_hw(piece)
            vr_off = int((vr1 - pph + 1) if vhs == "T" else vr0)
            vc_off = int((vc1 - ppw + 1) if vvs == "L" else vc0)
            vpts = {(int(r + vr_off), int(c + vc_off)) for r, c in piece}
            for r, c in vpts:
                if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                    return None
                if (r, c) in all_inp_fg:
                    return None
            vanish_fg |= vpts

    combined = all_inp_fg | vanish_fg
    mask2 = np.zeros((H, W), dtype=int)
    for r, c in combined:
        mask2[int(r), int(c)] = 1
    if not is_contiguous(mask2):
        return None

    inp = np.full((H, W), bg, dtype=int)
    for r, c in combined:
        inp[int(r), int(c)] = int(fg)
    for cr, cc, color in ctrl_pts:
        inp[int(cr), int(cc)] = int(color)

    out = np.full((H, W), bg, dtype=int)
    for pts in output_pieces.values():
        for r, c in pts:
            out[int(r), int(c)] = int(fg)
    for r, c in split_pixels:
        out[int(r), int(c)] = int(fg)

    for cr, cc, color in ctrl_pts:
        if cr == 0:
            fg_rows = [r for r in range(H) if out[r, cc] == fg]
            if fg_rows:
                for r in range(max(fg_rows) + 1):
                    if out[int(r), int(cc)] == bg:
                        out[int(r), int(cc)] = int(color)
        elif cr == H - 1:
            fg_rows = [r for r in range(H) if out[r, cc] == fg]
            if fg_rows:
                for r in range(H - 1, min(fg_rows) - 1, -1):
                    if out[int(r), int(cc)] == bg:
                        out[int(r), int(cc)] = int(color)
        elif cc == 0:
            fg_cols = [c for c in range(W) if out[cr, c] == fg]
            if fg_cols:
                for c in range(max(fg_cols) + 1):
                    if out[int(cr), int(c)] == bg:
                        out[int(cr), int(c)] = int(color)
        elif cc == W - 1:
            fg_cols = [c for c in range(W) if out[cr, c] == fg]
            if fg_cols:
                for c in range(W - 1, min(fg_cols) - 1, -1):
                    if out[int(cr), int(c)] == bg:
                        out[int(cr), int(c)] = int(color)
    for cr, cc, color in ctrl_pts:
        out[int(cr), int(cc)] = int(color)

    fg_all = np.array(list(combined))
    final_h = int(fg_all[:, 0].max() - fg_all[:, 0].min() + 1)
    final_w = int(fg_all[:, 1].max() - fg_all[:, 1].min() + 1)
    actual_density = len(combined) / float(final_h * final_w)
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
