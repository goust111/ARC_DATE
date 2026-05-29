# Variant of 4a21e3da
# Dimension: D5 — 切割后片段的变换规则 (transform)
# Change: 与控制边接壤的区域滑动到对应角落；不接壤的区域消失（erase）... → 片段旋转后滑入角落

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous

# ---------------------------------------------------------------------------
# Design analysis for the D5 variant
# ---------------------------------------------------------------------------
#
# spec_like_variant:
#   invariant_parts:
#     grid: 18-20 x 18-20, square
#     colors: {1:bg, 7:fg sprite, 2:ctrl ray}
#     ctrl_pixels:
#       count: 1 or 2
#       placement: border only; if 2 then one vertical split + one horizontal split
#     sprite:
#       centered with >= 4-cell margin
#       single connected fg component in input
#       split lines pass through sprite interior
#       fg on split line(s) stays in place in output
#     ray:
#       draw ctrl-colored intermittent rays on bg cells only, from each ctrl pixel to the farthest fg on that line
#     erase_rule:
#       any cut region that touches none of the active control edges still vanishes
#
#   D5_core_change:
#     original: kept regions translate to their destination corner with orientation preserved
#     variant: kept regions first rotate, then translate to their destination corner
#     corner_selection: unchanged from the original task
#     rotation_rule:
#       1. Determine the destination corner (TL/TR/BL/BR) exactly as in the original task.
#       2. Count how many active control edges are also edges of that destination corner.
#          - one matching active edge  -> quarter-turn
#          - two matching active edges -> half-turn
#       3. Quarter-turn direction is the visible "toward that corner" direction:
#          - top->TL = 90 CCW, top->TR = 90 CW
#          - bottom->BL = 90 CW, bottom->BR = 90 CCW
#          - left->TL = 90 CW, left->BL = 90 CCW
#          - right->TR = 90 CCW, right->BR = 90 CW
#          - if both the vertical and horizontal active edges match the destination corner,
#            their effects compose into 180 degrees.
#
#   concrete_trace:
#     input_example:
#       controls: top at column c_split, left at row r_split
#       kept regions: TL, TR, BL
#       vanished region: BR
#     original_stage2:
#       TL piece slides unchanged to top-left corner
#       TR piece slides unchanged to top-right corner
#       BL piece slides unchanged to bottom-left corner
#     variant_stage2:
#       TL piece rotates 180 and then moves to top-left
#       TR piece rotates 90 CW and then moves to top-right
#       BL piece rotates 90 CCW and then moves to bottom-left
#     visible_difference:
#       irregular monochrome pieces change silhouette at the corners instead of only changing position
#
#   complexity_calibration:
#     kept piece rotations are restricted to {90 CW, 90 CCW, 180}; no flips and no color remapping
#     the destination corner logic stays identical to the original task, so only one reasoning axis changes
#     generator rejects rotation-invariant pieces under the required rotation, preventing visual degeneration
#     two-control cases are kept common so the 180-degree branch appears regularly
#
#   hidden_assumption_audit_on_main:
#     - Border color detection still assumes ctrl is the only non-bg color on the border.
#       Generator keeps fg >= 4 cells away from every edge, so this remains true.
#     - Split inference still assumes at most one vertical and one horizontal split.
#       Generator still places 1 or 2 perpendicular ctrl pixels only.
#     - Region grouping still treats each side/quadrant as a single piece container.
#       Generator still creates at most one connected fg piece per region.
#     - The new transform now depends on local piece orientation, so using absolute translation deltas
#       from the original code would be wrong; the solver must crop each region to its bbox, rotate locally,
#       then place the rotated bbox flush to the corner.
#     - Rotated kept pieces must remain on their side of the split lines; generator enforces this by
#       rejecting shapes whose rotated bbox would cross the relevant split line.
#
#   placement_legality_and_cleanliness:
#     generator keeps output corner pieces in disjoint corners and input pieces in disjoint split regions
#     split-line fg is still sparse (2-4 per split), so rays / split pixels / corner pieces remain visually separable
#     no color-change rule is introduced; observability comes only from shape rotation
#
#   new_main_assumptions_review:
#     - _rotation_k assumes every kept region's destination corner shares at least one active control edge.
#       Generator keeps the original corner-selection logic, so that is guaranteed.
#     - rotated corner placement assumes piece bbox is the correct transform primitive.
#       Generator constructs one connected piece per region, so bbox-crop + rotate preserves the intended sprite.
#     - overwriting in output is safe only when corner pieces are disjoint.
#       Generator rejects any rotated piece that would cross a split line, so opposite-corner overlap cannot happen.


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
    else:
        if ("top" in E) and ("bottom" not in E):
            dh = "T"
        elif ("bottom" in E) and ("top" not in E):
            dh = "B"
        else:
            dh = "T"
    if c_split is not None:
        dv = vs
    else:
        if ("left" in E) and ("right" not in E):
            dv = "L"
        elif ("right" in E) and ("left" not in E):
            dv = "R"
        else:
            dv = "L"
    return str(dh), str(dv)


def _rotation_k(dh: str, dv: str, E: set[str]) -> int:
    v_match = (dh == "T" and "top" in E) or (dh == "B" and "bottom" in E)
    h_match = (dv == "L" and "left" in E) or (dv == "R" and "right" in E)
    count = int(v_match) + int(h_match)
    if count == 2:
        return 2
    assert count == 1
    if v_match:
        if dh == "T":
            return 1 if dv == "L" else 3
        return 3 if dv == "L" else 1
    if dv == "L":
        return 3 if dh == "T" else 1
    return 1 if dh == "T" else 3


def _rotate_local_mask(pts: list[tuple[int, int]] | set[tuple[int, int]], krot: int) -> np.ndarray:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    rmin, cmin = int(min(rs)), int(min(cs))
    h = int(max(rs) - rmin + 1)
    w = int(max(cs) - cmin + 1)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in pts:
        mask[int(r - rmin), int(c - cmin)] = True
    return np.rot90(mask, int(krot))


def _place_rotated_to_corner(
    pts: list[tuple[int, int]] | set[tuple[int, int]],
    krot: int,
    dh: str,
    dv: str,
    H: int,
    W: int,
) -> set[tuple[int, int]]:
    rot_mask = _rotate_local_mask(pts, int(krot))
    ph, pw = map(int, rot_mask.shape)
    base_r = 0 if dh == "T" else int(H - ph)
    base_c = 0 if dv == "L" else int(W - pw)
    return {
        (int(base_r + r), int(base_c + c))
        for r, c in np.argwhere(rot_mask)
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut a sprite along border-control rays and send kept pieces to corners.

    Variant rule:
    - Piece retention / erasure is unchanged from 4a21e3da.
    - Every kept piece is rotated locally by 90/180 degrees according to the active
      control-edge -> destination-corner relation, then the rotated bbox is placed
      flush into the destination corner.
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
        if not (_touches_edges(hs, vs) & E):
            continue
        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        krot = _rotation_k(dh, dv, E)
        for rr, cc in _place_rotated_to_corner(pts, int(krot), dh, dv, int(H), int(W)):
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


def _mask_from_piece(piece: frozenset[tuple[int, int]]) -> np.ndarray:
    ph = int(max(r for r, _ in piece) + 1)
    pw = int(max(c for _, c in piece) + 1)
    mask = np.zeros((ph, pw), dtype=bool)
    for r, c in piece:
        mask[int(r), int(c)] = True
    return mask


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
    has_v = has_h = False

    for edge in ctrl_edges:
        if edge in ("top", "bottom"):
            c_split = random.randint(int(W * 0.4), int(W * 0.6))
            ctrl_pts.append((0 if edge == "top" else H - 1, int(c_split)))
            has_v = True
        else:
            r_split = random.randint(int(H * 0.4), int(H * 0.6))
            ctrl_pts.append((int(r_split), 0 if edge == "left" else W - 1))
            has_h = True

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
        frac = random.uniform(0.4, 0.6)
        sprite_c0 = int(c_split - int(frac * sprite_w))
        sprite_c0 = max(4, min(int(sprite_c0), int(W - 5 - sprite_w + 1)))
        if sprite_c0 > int(W - 5 - sprite_w + 1):
            return None
        if int(c_split) <= int(sprite_c0) or int(c_split) >= int(sprite_c0 + sprite_w - 1):
            return None
    else:
        hi = int(W - 5 - sprite_w + 1)
        if hi < 4:
            return None
        sprite_c0 = random.randint(4, hi)

    if has_h:
        frac = random.uniform(0.4, 0.6)
        sprite_r0 = int(r_split - int(frac * sprite_h))
        sprite_r0 = max(4, min(int(sprite_r0), int(H - 5 - sprite_h + 1)))
        if sprite_r0 > int(H - 5 - sprite_h + 1):
            return None
        if int(r_split) <= int(sprite_r0) or int(r_split) >= int(sprite_r0 + sprite_h - 1):
            return None
    else:
        hi = int(H - 5 - sprite_h + 1)
        if hi < 4:
            return None
        sprite_r0 = random.randint(4, hi)

    sprite_r1 = int(sprite_r0 + sprite_h - 1)
    sprite_c1 = int(sprite_c0 + sprite_w - 1)

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = int(sprite_r0 if hs in ("T", "A") else int(r_split) + 1)
        r1 = int((int(r_split) - 1 if hs == "T" else sprite_r1) if has_h else sprite_r1)
        c0 = int(sprite_c0 if vs in ("L", "A") else int(c_split) + 1)
        c1 = int((int(c_split) - 1 if vs == "L" else sprite_c1) if has_v else sprite_c1)
        return int(r0), int(r1), int(c0), int(c1)

    for reg in kept + vanish:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None

    split_pixels: set[tuple[int, int]] = set()
    if has_v:
        rows = [
            r
            for r in range(int(sprite_r0), int(sprite_r1) + 1)
            if (r_split is None or r != int(r_split))
        ]
        n = random.randint(2, 4)
        if len(rows) < n:
            return None
        for r in random.sample(rows, n):
            split_pixels.add((int(r), int(c_split)))
    if has_h:
        cols = [
            c
            for c in range(int(sprite_c0), int(sprite_c1) + 1)
            if (c_split is None or c != int(c_split))
        ]
        n = random.randint(2, 4)
        if len(cols) < n:
            return None
        for c in random.sample(cols, n):
            split_pixels.add((int(r_split), int(c)))

    n_split = len(split_pixels)
    area = int(sprite_h * sprite_w)
    target_total = int(random.uniform(0.39, 0.60) * area)

    n_kept = len(kept)
    budget_pieces = int(target_total - n_split)
    if budget_pieces < n_kept * 3:
        return None

    if vanish:
        max_vanish = max(3, min(7, budget_pieces // 3))
        if max_vanish < 3:
            return None
        vanish_budget = random.randint(3, max_vanish)
        budget_pieces -= int(vanish_budget)
    else:
        vanish_budget = 0

    if budget_pieces < n_kept * 3:
        return None

    piece_budgets: list[int] = []
    remaining = int(budget_pieces)
    for i, reg in enumerate(kept):
        r0, r1, c0, c1 = sub_bbox(*reg)
        max_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if i == n_kept - 1:
            px = remaining
        else:
            avg = remaining // (n_kept - i)
            lo_px = max(3, int(avg - 2))
            hi_px = min(max_area, int(avg + 2))
            if lo_px > hi_px:
                return None
            px = random.randint(lo_px, hi_px)
        px = max(3, min(px, max_area, remaining - (n_kept - i - 1) * 3))
        piece_budgets.append(int(px))
        remaining -= int(px)

    if remaining != 0 or any(int(b) < 3 for b in piece_budgets):
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], set[tuple[int, int]]] = {}

    def output_caps(dh: str, dv: str) -> tuple[int, int]:
        out_h_cap = int(H if not has_h else (int(r_split) if dh == "T" else H - int(r_split) - 1))
        out_w_cap = int(W if not has_v else (int(c_split) if dv == "L" else W - int(c_split) - 1))
        return int(out_h_cap), int(out_w_cap)

    for idx, reg in enumerate(kept):
        hs, vs = reg
        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        krot = _rotation_k(dh, dv, E)
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = int(r1 - r0 + 1)
        sub_w = int(c1 - c0 + 1)
        target_px = int(piece_budgets[idx])
        out_h_cap, out_w_cap = output_caps(dh, dv)

        accepted = False
        for _piece_try in range(80):
            piece = _grow_piece(sub_h, sub_w, target_px)
            if len(piece) < 3:
                continue
            mask = _mask_from_piece(piece)
            ph, pw = map(int, mask.shape)
            if ph < 2 or pw < 2:
                continue
            if np.array_equal(mask, np.rot90(mask, int(krot))):
                continue

            rot_h, rot_w = map(int, np.rot90(mask, int(krot)).shape)
            if rot_h > out_h_cap or rot_w > out_w_cap:
                continue

            if has_h:
                inp_r0 = int(r1 - ph + 1) if hs == "T" else int(r0)
            else:
                split_rows = sorted(r for r, c in split_pixels if c == int(c_split))
                if split_rows:
                    mid = int((split_rows[0] + split_rows[-1]) // 2)
                    inp_r0 = max(int(r0), min(int(mid - ph // 2), int(r1 - ph + 1)))
                else:
                    inp_r0 = int(r0)

            if has_v:
                inp_c0 = int(c1 - pw + 1) if vs == "L" else int(c0)
            else:
                split_cols = sorted(c for r, c in split_pixels if r == int(r_split))
                if split_cols:
                    mid = int((split_cols[0] + split_cols[-1]) // 2)
                    inp_c0 = max(int(c0), min(int(mid - pw // 2), int(c1 - pw + 1)))
                else:
                    inp_c0 = int(c0)

            inp_pts = frozenset((int(r + inp_r0), int(c + inp_c0)) for r, c in piece)

            ok = True
            for r, c in inp_pts:
                if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                    ok = False
                    break
                if has_v and ((vs == "L" and c >= int(c_split)) or (vs == "R" and c <= int(c_split))):
                    ok = False
                    break
                if has_h and ((hs == "T" and r >= int(r_split)) or (hs == "B" and r <= int(r_split))):
                    ok = False
                    break
            if not ok:
                continue

            out_pts = _place_rotated_to_corner(inp_pts, int(krot), dh, dv, int(H), int(W))
            for r, c in out_pts:
                if has_v and ((dv == "L" and c >= int(c_split)) or (dv == "R" and c <= int(c_split))):
                    ok = False
                    break
                if has_h and ((dh == "T" and r >= int(r_split)) or (dh == "B" and r <= int(r_split))):
                    ok = False
                    break
            if not ok:
                continue

            input_pieces[reg] = inp_pts
            output_pieces[reg] = out_pts
            accepted = True
            break

        if not accepted:
            return None

    all_inp_fg: set[tuple[int, int]] = set(split_pixels)
    for pts in input_pieces.values():
        if all_inp_fg & set(pts):
            return None
        all_inp_fg |= set(pts)
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
            vh, vw = int(vr1 - vr0 + 1), int(vc1 - vc0 + 1)
            if vh < 1 or vw < 1:
                return None
            max_px = min(int(vanish_budget), int(vh * vw))
            if max_px < 3:
                return None

            accepted = False
            for _piece_try in range(60):
                piece = _grow_piece(vh, vw, max_px)
                if len(piece) < 3:
                    continue
                pph = int(max(r for r, _ in piece) + 1)
                ppw = int(max(c for _, c in piece) + 1)
                vr_off = int(vr1 - pph + 1) if vhs == "T" else int(vr0)
                vc_off = int(vc1 - ppw + 1) if vvs == "L" else int(vc0)
                vpts = {(int(r + vr_off), int(c + vc_off)) for r, c in piece}
                ok = True
                for r, c in vpts:
                    if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                        ok = False
                        break
                    if (r, c) in all_inp_fg:
                        ok = False
                        break
                if not ok:
                    continue
                vanish_fg = vpts
                accepted = True
                break
            if not accepted:
                return None

    combined = all_inp_fg | vanish_fg
    mask2 = np.zeros((H, W), dtype=int)
    for r, c in combined:
        mask2[int(r), int(c)] = 1
    if not is_contiguous(mask2):
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
    for r, c in split_pixels:
        out[int(r), int(c)] = int(fg)

    for cr, cc in ctrl_pts:
        if cr == 0:
            fg_rows = [r for r in range(H) if out[int(r), int(cc)] == int(fg)]
            if fg_rows:
                for r in range(int(max(fg_rows)) + 1):
                    if out[int(r), int(cc)] == int(bg):
                        out[int(r), int(cc)] = int(ctrl_color)
        elif cr == H - 1:
            fg_rows = [r for r in range(H) if out[int(r), int(cc)] == int(fg)]
            if fg_rows:
                for r in range(H - 1, int(min(fg_rows)) - 1, -1):
                    if out[int(r), int(cc)] == int(bg):
                        out[int(r), int(cc)] = int(ctrl_color)
        elif cc == 0:
            fg_cols = [c for c in range(W) if out[int(cr), int(c)] == int(fg)]
            if fg_cols:
                for c in range(int(max(fg_cols)) + 1):
                    if out[int(cr), int(c)] == int(bg):
                        out[int(cr), int(c)] = int(ctrl_color)
        elif cc == W - 1:
            fg_cols = [c for c in range(W) if out[int(cr), int(c)] == int(fg)]
            if fg_cols:
                for c in range(W - 1, int(min(fg_cols)) - 1, -1):
                    if out[int(cr), int(c)] == int(bg):
                        out[int(cr), int(c)] = int(ctrl_color)
    for cr, cc in ctrl_pts:
        out[int(cr), int(cc)] = int(ctrl_color)

    fg_all = np.array(list(combined))
    final_h = int(fg_all[:, 0].max() - fg_all[:, 0].min() + 1)
    final_w = int(fg_all[:, 1].max() - fg_all[:, 1].min() + 1)
    actual_density = len(combined) / float(final_h * final_w)
    if actual_density < 0.38 or actual_density > 0.62:
        return None

    if np.array_equal(main(inp), out) is False:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
