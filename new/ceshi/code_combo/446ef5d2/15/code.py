# Combo variant of 446ef5d2 (index 15)
# combo_spec: D4/3+D5/1
# Name: 镜像输出配合移动轨迹
#
# Design analysis for the fused variant
# ====================================
# a) Concrete behavior and trace
# - Original task:
#   1. detect the 3-cell L marker and infer which rectangle corner is pinned;
#   2. reassemble the scattered rectangular pieces by translation only;
#   3. paste the assembled rectangle at the marker corner with no extra marks.
# - Fused task:
#   1. keep the same marker parsing and rectangle reassembly;
#   2. reflect the fully assembled rectangle as a whole, using the marker corner to choose
#      the axis exactly as in D4/3:
#        * `TL` / `BL` -> left-right reflection (`np.fliplr`)
#        * `TR` / `BR` -> up-down reflection (`np.flipud`)
#   3. for every scattered input piece, compute where that piece's placement box lands after the
#      whole-rectangle reflection, then draw a 1-pixel-wide monochrome L-shaped trail from the
#      source box center to the reflected target box center;
#   4. paste the reflected rectangle last, so the final rectangle stays crisp while the external
#      parts of the trails remain visible.
# - Small trace:
#   suppose a piece originally sits at input box `(2:4, 10:13)`, the rectangle is reconstructed,
#   and under a `TL` marker its local placement box `(r=1,c=6,h=2,w=3)` reflects to local box
#   `(1, W-9, 2, 3)`. The output first paints a marker-colored L-path from the source-box center
#   to that reflected target-box center, then pastes the reflected whole rectangle over the target.
#   Compared with the base task, both the rectangle content and the inferred movement endpoints
#   change together.
#
# b) Complexity calibration
# - The assembly search remains unchanged: pieces are still translated only and boundary matching
#   still uses `rect_col` as a wildcard.
# - The extra reasoning is visible but low-branching:
#   one 2-way axis choice from the marker corner, and one fixed trail rule using the same axis to
#   choose whether the L-path moves horizontally first or vertically first.
# - Trails use exactly one existing color (the marker color), so no new color mapping puzzle is
#   introduced.
#
# c) Implicit-assumption audit
# - The original `assemble(...)` logic assumes the pieces themselves are never rotated/reflected.
#   That still holds: reflection is applied only after the whole rectangle is reconstructed.
# - The original output code assumed the assembled array itself was final. That assumption breaks
#   here, so `main()` must now:
#   1. keep the per-piece placement coordinates returned by backtracking,
#   2. reflect those placement boxes,
#   3. render trails from input boxes to reflected target boxes,
#   4. paste the reflected rectangle last.
# - The source-rectangle decomposition still assumes every non-marker object in the input is a
#   solid rectangle. The generator keeps that invariant.
# - The new trail renderer assumes each piece has a unique source rectangle and a unique placement
#   box in the solved tiling. The generator guarantees exact rectangular tilings, so that remains
#   true even when some piece contents are similar.
#
# d) Placement legality
# - Input legality remains the same for pieces: non-overlapping scattered rectangles with padding.
# - Output legality is preserved because reflection keeps the assembled rectangle size fixed.
# - Trail paths are center-to-center L-shapes, so every trail stays in-bounds automatically once
#   both endpoints are in-bounds. The final rectangle is pasted after the trails, preventing noisy
#   overdraw inside the destination box.
#
# e) Visual observability
# - The generator rejects rectangles whose reflection changes too few cells, so D4 is visible.
# - It also rejects samples whose visible external trail cells are too few, so D5 is visible.
# - Using marker color for trails makes the extra output structure easy to spot without inventing a
#   fresh output-only color.
#
# f) Visual cleanliness
# - Only two output layers exist: marker-colored trails and the reflected rectangle.
# - Trails are single-pixel L-lines, not filled swept corridors, which keeps the picture readable.
# - Pasting the rectangle after the trails avoids corrupting the piece patterns at the destination.
#
# g) Color inferability
# - No numeric-color rule is used. The marker's visible orientation chooses the reflection axis, and
#   the same marker color reappears as the universal trajectory color.
#
# h) Anti-degeneration guarantees used by the generator
# - `reflection_diff`: number of cells changed by reflecting the rectangle.
# - `visible_trail_cells`: number of trail-colored cells that remain visible outside the final
#   reflected rectangle.
# - `pieces_with_visible_trail`: how many pieces contribute at least two visible external trail
#   cells.
# - A candidate is rejected unless all three are large enough, preventing collapse back to either
#   the base task or the reflection-only variant.

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import ndimage

from arc_utils import STRUCT8, paste


def _reflect_axis_for_corner(corner: str) -> str:
    """Left-corner cases mirror left-right; right-corner cases mirror up-down."""
    return "vertical" if corner in ("TL", "BL") else "horizontal"


def _reflect_rect(rect: np.ndarray, corner: str) -> np.ndarray:
    axis = _reflect_axis_for_corner(corner)
    if axis == "vertical":
        return np.fliplr(rect)
    return np.flipud(rect)


def _top_left_from_corner(mr: int, mc: int, H: int, W: int, corner: str) -> tuple[int, int]:
    if corner == "TL":
        return int(mr), int(mc)
    if corner == "TR":
        return int(mr), int(mc - W + 1)
    if corner == "BL":
        return int(mr - H + 1), int(mc)
    return int(mr - H + 1), int(mc - W + 1)


def _mirror_local_box(box: tuple[int, int, int, int], H: int, W: int, corner: str) -> tuple[int, int, int, int]:
    r, c, h, w = map(int, box)
    axis = _reflect_axis_for_corner(corner)
    if axis == "vertical":
        return int(r), int(W - c - w), int(h), int(w)
    return int(H - r - h), int(c), int(h), int(w)


def _box_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r, c, h, w = map(int, box)
    return int(r + h // 2), int(c + w // 2)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar, ac, ah, aw = map(int, a)
    br, bc, bh, bw = map(int, b)
    return not (ar + ah <= br or br + bh <= ar or ac + aw <= bc or bc + bw <= ac)


def _line_coords(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, a)
    r1, c1 = map(int, b)
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _trail_points_for_boxes(
    src_box: tuple[int, int, int, int], dst_box: tuple[int, int, int, int], corner: str
) -> list[tuple[int, int]]:
    src = _box_center(src_box)
    dst = _box_center(dst_box)
    sr, sc = src
    dr, dc = dst
    axis = _reflect_axis_for_corner(corner)
    if axis == "vertical":
        bend = (sr, dc)
    else:
        bend = (dr, sc)
    pts = _line_coords(src, bend)
    tail = _line_coords(bend, dst)
    if tail:
        pts.extend(tail[1:])
    return pts


def _find_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], str]:
    cnt = Counter(map(int, g.flatten()))
    mark = None
    missing = None
    mb = None
    for col in sorted(c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3):
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(col))]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0) == 1 and (c1 - c0) == 1:
            all4 = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            miss = list(all4 - set(pts))
            if len(miss) == 1:
                mark = int(col)
                missing = miss[0]
                mb = (r0, c0, r1, c1)
                break
    assert not (mark is None or missing is None or mb is None)
    r0, c0, r1, c1 = mb
    if missing == (r0, c0):
        corner = "BR"
    elif missing == (r0, c1):
        corner = "BL"
    elif missing == (r1, c0):
        corner = "TR"
    else:
        corner = "TL"
    return int(mark), (int(missing[0]), int(missing[1])), corner


def _decompose_rectangles(g: np.ndarray, bg: int, mark: int) -> list[tuple[int, int, int, int, np.ndarray]]:
    GH, GW = g.shape
    occ = (g != int(bg)) & (g != int(mark))
    used = np.zeros_like(occ, dtype=bool)
    rects: list[tuple[int, int, int, int, np.ndarray]] = []
    while True:
        idx = np.argwhere(occ & (~used))
        if idx.size == 0:
            break
        rr, cc = map(int, idx[0])
        w = 0
        while cc + w < GW and bool(occ[rr, cc + w]) and (not bool(used[rr, cc + w])):
            w += 1
        h = 0
        while rr + h < GH and bool((occ[rr + h, cc : cc + w] & (~used[rr + h, cc : cc + w])).all()):
            h += 1
        used[rr : rr + h, cc : cc + w] = True
        rects.append((rr, cc, h, w, g[rr : rr + h, cc : cc + w].copy()))
    return rects


def _assemble_rectangle(
    rects: list[tuple[int, int, int, int, np.ndarray]],
    mr: int,
    mc: int,
    corner: str,
    GH: int,
    GW: int,
) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int], int]:
    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))

    allpix: list[int] = []
    for _r, _c, _h, _w, a in rects:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    anchor_idx = None
    for i, (rr, cc, h, w, _a) in enumerate(rects):
        if rr <= mr < rr + h and cc <= mc < cc + w:
            anchor_idx = i
            break

    pieces = [a for _r, _c, _h, _w, a in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]

    dims: list[tuple[int, int]] = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh == 0:
            dims.append((hh, total_area // hh))
            if hh * hh != total_area:
                dims.append((total_area // hh, hh))

    def fits(H: int, W: int) -> bool:
        tr, tc = _top_left_from_corner(mr, mc, H, W, corner)
        return 0 <= tr and 0 <= tc and (tr + H) <= GH and (tc + W) <= GW

    dims = [(int(H), int(W)) for H, W in dims if fits(int(H), int(W))]
    dims = sorted(dims, key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    def compat(v1: np.ndarray, v2: np.ndarray) -> bool:
        v1 = np.array(v1, dtype=int)
        v2 = np.array(v2, dtype=int)
        m = (v1 != int(rect_col)) & (v2 != int(rect_col))
        return bool((v1[m] == v2[m]).all())

    def assemble(H: int, W: int) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
        occ2 = np.full((H, W), -1, dtype=int)
        usedp = [False] * len(pieces)
        placements: list[tuple[int, int] | None] = [None] * len(pieces)

        def boundary_ok(a: np.ndarray, r: int, c: int) -> bool:
            h, w = a.shape
            if r == 0 and not np.all(a[0, :] == int(rect_col)):
                return False
            if (r + h) == H and not np.all(a[-1, :] == int(rect_col)):
                return False
            if c == 0 and not np.all(a[:, 0] == int(rect_col)):
                return False
            if (c + w) == W and not np.all(a[:, -1] == int(rect_col)):
                return False
            return True

        def neighbor_ok(a: np.ndarray, r: int, c: int) -> bool:
            h, w = a.shape
            if c > 0:
                left = occ2[r : r + h, c - 1]
                m = left != -1
                if m.any() and (not compat(left[m], a[:, 0][m])):
                    return False
            if (c + w) < W:
                right = occ2[r : r + h, c + w]
                m = right != -1
                if m.any() and (not compat(right[m], a[:, -1][m])):
                    return False
            if r > 0:
                up = occ2[r - 1, c : c + w]
                m = up != -1
                if m.any() and (not compat(up[m], a[0, :][m])):
                    return False
            if (r + h) < H:
                down = occ2[r + h, c : c + w]
                m = down != -1
                if m.any() and (not compat(down[m], a[-1, :][m])):
                    return False
            return True

        def can_place(i: int, r: int, c: int) -> bool:
            a = pieces[i]
            h, w = a.shape
            if (r + h) > H or (c + w) > W:
                return False
            if (occ2[r : r + h, c : c + w] != -1).any():
                return False
            if not boundary_ok(a, r, c):
                return False
            if not neighbor_ok(a, r, c):
                return False
            return True

        def place(i: int, r: int, c: int) -> None:
            a = pieces[i]
            h, w = a.shape
            occ2[r : r + h, c : c + w] = a
            usedp[i] = True
            placements[i] = (int(r), int(c))

        def unplace(i: int, r: int, c: int) -> None:
            a = pieces[i]
            h, w = a.shape
            occ2[r : r + h, c : c + w] = -1
            usedp[i] = False
            placements[i] = None

        if anchor_idx is not None:
            rr, cc, h0, w0, _ = rects[anchor_idx]
            corners = {
                (rr, cc): "TL",
                (rr, cc + w0 - 1): "TR",
                (rr + h0 - 1, cc): "BL",
                (rr + h0 - 1, cc + w0 - 1): "BR",
            }
            if (mr, mc) in corners and corners[(mr, mc)] == corner:
                ah, aw = ph[anchor_idx], pw[anchor_idx]
                if corner == "TL":
                    ar0, ac0 = 0, 0
                elif corner == "TR":
                    ar0, ac0 = 0, W - aw
                elif corner == "BL":
                    ar0, ac0 = H - ah, 0
                else:
                    ar0, ac0 = H - ah, W - aw
                if can_place(anchor_idx, ar0, ac0):
                    place(anchor_idx, ar0, ac0)

        order = sorted(range(len(pieces)), key=lambda i: (-ph[i] * pw[i], -ph[i], -pw[i], i))

        def next_empty() -> tuple[int, int] | None:
            idx = np.argwhere(occ2 == -1)
            if idx.size == 0:
                return None
            return tuple(map(int, idx[0]))

        solved = False

        def bt() -> None:
            nonlocal solved
            nxt = next_empty()
            if nxt is None:
                solved = True
                return
            r, c = nxt
            for i in order:
                if usedp[i]:
                    continue
                if can_place(i, r, c):
                    place(i, r, c)
                    bt()
                    if solved:
                        return
                    unplace(i, r, c)

        bt()
        if not solved:
            return None
        final_places = [p for p in placements]
        assert all(p is not None for p in final_places)
        return occ2, [(int(r), int(c)) for r, c in final_places if r is not None and c is not None]

    for H, W in dims:
        result = assemble(H, W)
        if result is not None:
            ass, placements = result
            return ass, placements, (int(H), int(W)), int(rect_col)
    raise AssertionError("assemble failed")


def _render_output(
    GH: int,
    GW: int,
    bg: int,
    final_rect: np.ndarray,
    top_left: tuple[int, int],
    corner: str,
    trail_col: int,
    source_boxes: list[tuple[int, int, int, int]],
    target_boxes: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, dict]:
    out = np.full((GH, GW), int(bg), dtype=int)
    trail_mask = np.zeros((GH, GW), dtype=bool)
    tr, tc = map(int, top_left)
    H, W = final_rect.shape
    rect_mask = np.zeros((GH, GW), dtype=bool)
    rect_mask[tr : tr + H, tc : tc + W] = True

    per_piece_visible: list[int] = []
    for src_box, dst_box in zip(source_boxes, target_boxes):
        pts = _trail_points_for_boxes(src_box, dst_box, corner)
        visible = 0
        for r, c in pts:
            if 0 <= r < GH and 0 <= c < GW:
                trail_mask[r, c] = True
                out[r, c] = int(trail_col)
                if not rect_mask[r, c]:
                    visible += 1
        per_piece_visible.append(int(visible))

    out[tr : tr + H, tc : tc + W] = final_rect
    stats = {
        "visible_trail_cells": int(np.count_nonzero(trail_mask & (~rect_mask))),
        "pieces_with_visible_trail": int(sum(v >= 2 for v in per_piece_visible)),
        "per_piece_visible_trail": per_piece_visible,
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the rectangle, reflect it according to the marker corner, and draw
    marker-colored L-shaped trails from each source piece to its reflected target box.
    """
    g = np.array(grid, dtype=int)
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    mark, missing, corner = _find_marker(g, bg)
    rects = _decompose_rectangles(g, bg, mark)

    mr, mc = map(int, missing)
    ass, placements, (H, W), _rect_col = _assemble_rectangle(rects, mr, mc, corner, GH, GW)
    final_rect = _reflect_rect(ass, corner)
    top_left = _top_left_from_corner(mr, mc, H, W, corner)
    tr, tc = map(int, top_left)

    source_boxes: list[tuple[int, int, int, int]] = []
    target_boxes: list[tuple[int, int, int, int]] = []
    for (src_r, src_c, h, w, _a), (pr, pc) in zip(rects, placements):
        source_boxes.append((int(src_r), int(src_c), int(h), int(w)))
        lr, lc, lh, lw = _mirror_local_box((pr, pc, h, w), H, W, corner)
        target_boxes.append((int(tr + lr), int(tc + lc), int(lh), int(lw)))

    out, _stats = _render_output(GH, GW, bg, final_rect, top_left, corner, mark, source_boxes, target_boxes)
    return out


def _candidate_locations(canvas: np.ndarray, patch: np.ndarray, bg: int, padding: int = 1) -> list[tuple[int, int]]:
    H, W = canvas.shape
    h, w = patch.shape
    occ = canvas != int(bg)
    if padding > 0:
        occ = ndimage.binary_dilation(occ, structure=STRUCT8, iterations=padding)
    mask = patch != int(bg)
    cands: list[tuple[int, int]] = []
    for r0 in range(0, H - h + 1):
        for c0 in range(0, W - w + 1):
            if not bool((occ[r0 : r0 + h, c0 : c0 + w] & mask).any()):
                cands.append((int(r0), int(c0)))
    return cands


def _pick_scatter_location(
    canvas: np.ndarray,
    patch: np.ndarray,
    bg: int,
    target_box: tuple[int, int, int, int],
    output_box: tuple[int, int, int, int],
    rng,
) -> tuple[int, int]:
    h, w = patch.shape
    cands = _candidate_locations(canvas, patch, bg, padding=1)
    if not cands:
        raise ValueError("no free location")

    scored: list[tuple[float, int, int]] = []
    target_center = _box_center(target_box)
    for r0, c0 in cands:
        src_box = (int(r0), int(c0), int(h), int(w))
        src_center = _box_center(src_box)
        dist = abs(src_center[0] - target_center[0]) + abs(src_center[1] - target_center[1])
        score = float(dist)
        if _boxes_overlap(src_box, target_box):
            score -= 100.0
        if _boxes_overlap(src_box, output_box):
            score -= 20.0
        score += float(rng.random()) * 0.25
        scored.append((score, int(r0), int(c0)))

    scored.sort(reverse=True)
    topk = scored[: min(12, len(scored))]
    _score, rr, cc = topk[rng.randrange(len(topk))]
    return int(rr), int(cc)


def _scatter_pieces(
    GH: int,
    GW: int,
    bg: int,
    marker_col: int,
    marker_cells: list[tuple[int, int]],
    piece_arrays: list[np.ndarray],
    target_boxes: list[tuple[int, int, int, int]],
    output_box: tuple[int, int, int, int],
    rng,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    n = len(piece_arrays)
    order = sorted(range(n), key=lambda i: (-piece_arrays[i].shape[0] * piece_arrays[i].shape[1], rng.random()))
    for _attempt in range(18):
        canvas = np.full((GH, GW), int(bg), dtype=int)
        for r, c in marker_cells:
            canvas[r, c] = int(marker_col)

        source_boxes: list[tuple[int, int, int, int] | None] = [None] * n
        ok = True
        for idx in order:
            patch = piece_arrays[idx]
            try:
                r0, c0 = _pick_scatter_location(canvas, patch, bg, target_boxes[idx], output_box, rng)
            except ValueError:
                ok = False
                break
            paste(canvas, patch, r0, c0)
            source_boxes[idx] = (int(r0), int(c0), int(patch.shape[0]), int(patch.shape[1]))

        if ok and all(b is not None for b in source_boxes):
            return canvas, [(int(r), int(c), int(h), int(w)) for r, c, h, w in source_boxes if r is not None]
    return None


def _gen_attempt() -> np.ndarray | None:
    import random as _r

    GH = _r.randint(12, 26)
    GW = _r.randint(12, 26)

    palette = list(range(10))
    _r.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    n_interior = _r.randint(1, 3)
    interior_cols = palette[3 : 3 + n_interior]

    corner = _r.choice(["TL", "TR", "BL", "BR"])

    min_side, max_side = 5, min(18, min(GH, GW) - 3)
    if max_side < min_side:
        return None
    RH = _r.randint(min_side, max_side)
    RW = _r.randint(min_side, max_side)
    if RH * RW > int(0.42 * GH * GW):
        return None

    if corner == "TL":
        mr_min, mr_max = 2, GH - RH
        mc_min, mc_max = 2, GW - RW
    elif corner == "TR":
        mr_min, mr_max = 2, GH - RH
        mc_min, mc_max = RW - 1, GW - 3
    elif corner == "BL":
        mr_min, mr_max = RH - 1, GH - 3
        mc_min, mc_max = 2, GW - RW
    else:
        mr_min, mr_max = RH - 1, GH - 3
        mc_min, mc_max = RW - 1, GW - 3

    if mr_min > mr_max or mc_min > mc_max:
        return None
    mr = _r.randint(mr_min, mr_max)
    mc = _r.randint(mc_min, mc_max)

    rect = np.full((RH, RW), rect_col, dtype=int)
    iH, iW = RH - 2, RW - 2
    if iH < 1 or iW < 1:
        return None

    interior = np.full((iH, iW), rect_col, dtype=int)
    target_fill = _r.uniform(0.35, 0.85)
    target_pixels = max(1, int(target_fill * iH * iW))
    blob_occupied = np.zeros((iH, iW), dtype=bool)

    for ci, ic in enumerate(interior_cols):
        if ci == len(interior_cols) - 1:
            budget = target_pixels - int(blob_occupied.sum())
        else:
            budget = max(1, target_pixels // len(interior_cols))
        if budget <= 0:
            budget = 1

        attempts = 0
        seed = None
        while attempts < 200:
            sr, sc = _r.randint(0, iH - 1), _r.randint(0, iW - 1)
            if blob_occupied[sr, sc]:
                attempts += 1
                continue
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < iH and 0 <= nc < iW and blob_occupied[nr, nc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                seed = (sr, sc)
                break
            attempts += 1
        if seed is None:
            continue

        blob = {seed}
        frontier = [seed]
        while len(blob) < budget and frontier:
            _r.shuffle(frontier)
            nf = []
            for fr, fc in frontier:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = fr + dr, fc + dc
                    if (nr, nc) in blob:
                        continue
                    if nr < 0 or nr >= iH or nc < 0 or nc >= iW:
                        continue
                    if blob_occupied[nr, nc]:
                        continue
                    conflict = False
                    for dr2 in (-1, 0, 1):
                        for dc2 in (-1, 0, 1):
                            if dr2 == 0 and dc2 == 0:
                                continue
                            ar, ac = nr + dr2, nc + dc2
                            if 0 <= ar < iH and 0 <= ac < iW:
                                if blob_occupied[ar, ac] and (ar, ac) not in blob:
                                    conflict = True
                                    break
                        if conflict:
                            break
                    if conflict:
                        continue
                    blob.add((nr, nc))
                    nf.append((nr, nc))
                    if len(blob) >= budget:
                        break
                if len(blob) >= budget:
                    break
            frontier = nf

        for br, bc in blob:
            interior[br, bc] = ic
            blob_occupied[br, bc] = True

    non_rc_count = int(np.sum(interior != rect_col))
    fill_ratio = non_rc_count / (iH * iW) if iH * iW > 0 else 0.0
    if fill_ratio < 0.30 or fill_ratio > 0.90:
        return None
    rect[1:-1, 1:-1] = interior

    reflected = _reflect_rect(rect, corner)
    reflection_diff = int(np.sum(rect != reflected))
    if reflection_diff < max(6, int(0.08 * RH * RW)):
        return None

    def _rand_splits(total: int, min_s: int = 2, max_s: int = 7) -> list[int] | None:
        eff_max = min(max_s, max(min_s, _r.randint(min_s, max_s)))
        segs: list[int] = []
        rem = total
        while rem > 0:
            if rem <= eff_max:
                if rem < min_s:
                    return None
                segs.append(rem)
                rem = 0
            else:
                hi = min(eff_max, rem - min_s)
                if min_s > hi:
                    return None
                s = _r.randint(min_s, hi)
                segs.append(s)
                rem -= s
        return segs

    row_segs = _rand_splits(RH)
    col_segs = _rand_splits(RW)
    if row_segs is None or col_segs is None:
        return None

    row_starts: list[int] = []
    s = 0
    for rs in row_segs:
        row_starts.append(s)
        s += rs
    col_starts: list[int] = []
    s = 0
    for cs in col_segs:
        col_starts.append(s)
        s += cs

    nr, nc_grid = len(row_segs), len(col_segs)
    cell_id = [[i * nc_grid + j for j in range(nc_grid)] for i in range(nr)]
    n_cells = nr * nc_grid
    if n_cells < 4:
        return None

    def _get_piece_bbox(pid: int) -> tuple[int, int, int, int] | None:
        cells = [(i, j) for i in range(nr) for j in range(nc_grid) if cell_id[i][j] == pid]
        if not cells:
            return None
        ris = [c[0] for c in cells]
        cis = [c[1] for c in cells]
        return min(ris), min(cis), max(ris), max(cis)

    def _pixel_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
        h = sum(row_segs[i] for i in range(r0i, r1i + 1))
        w = sum(col_segs[j] for j in range(c0i, c1i + 1))
        return h, w

    target_pieces = _r.randint(4, min(12, n_cells))
    merges = []
    for i in range(nr):
        for j in range(nc_grid):
            if j + 1 < nc_grid:
                merges.append(("h", i, j))
            if i + 1 < nr:
                merges.append(("v", i, j))
    _r.shuffle(merges)

    def _count_pieces() -> int:
        return len(set(cell_id[i][j] for i in range(nr) for j in range(nc_grid)))

    for direction, mi, mj in merges:
        if _count_pieces() <= target_pieces:
            break
        if direction == "h":
            pid1 = cell_id[mi][mj]
            pid2 = cell_id[mi][mj + 1]
        else:
            pid1 = cell_id[mi][mj]
            pid2 = cell_id[mi + 1][mj]
        if pid1 == pid2:
            continue
        bb1 = _get_piece_bbox(pid1)
        bb2 = _get_piece_bbox(pid2)
        if bb1 is None or bb2 is None:
            continue
        mr0 = min(bb1[0], bb2[0])
        mc0 = min(bb1[1], bb2[1])
        mr1 = max(bb1[2], bb2[2])
        mc1 = max(bb1[3], bb2[3])
        ph, pw = _pixel_size(mr0, mc0, mr1, mc1)
        if ph > 7 or pw > 7:
            continue
        ok = True
        for ii in range(mr0, mr1 + 1):
            for jj in range(mc0, mc1 + 1):
                if cell_id[ii][jj] != pid1 and cell_id[ii][jj] != pid2:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for ii in range(nr):
            for jj in range(nc_grid):
                if cell_id[ii][jj] == pid2:
                    cell_id[ii][jj] = pid1

    pieces: list[tuple[int, int, int, int]] = []
    seen = set()
    for i in range(nr):
        for j in range(nc_grid):
            pid = cell_id[i][j]
            if pid in seen:
                continue
            seen.add(pid)
            bb = _get_piece_bbox(pid)
            assert bb is not None
            r0i, c0i, r1i, c1i = bb
            pr = row_starts[r0i]
            pc = col_starts[c0i]
            ph = sum(row_segs[ii] for ii in range(r0i, r1i + 1))
            pw = sum(col_segs[jj] for jj in range(c0i, c1i + 1))
            pieces.append((pr, pc, ph, pw))

    if len(pieces) < 4 or len(pieces) > 12:
        return None

    for pr, pc, ph, pw in pieces:
        if ph < 2 or pw < 2 or ph > 7 or pw > 7:
            return None
        piece = rect[pr : pr + ph, pc : pc + pw]
        nrc = int(np.sum(piece != rect_col))
        fill = nrc / (ph * pw)
        if fill < 0.10 or fill > 0.60:
            return None
        has_non_rc_edge = False
        for edge in [piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]]:
            if np.any(edge != rect_col):
                has_non_rc_edge = True
                break
        if not has_non_rc_edge:
            return None
        if pr == 0 and not np.all(piece[0, :] == rect_col):
            return None
        if pr + ph == RH and not np.all(piece[-1, :] == rect_col):
            return None
        if pc == 0 and not np.all(piece[:, 0] == rect_col):
            return None
        if pc + pw == RW and not np.all(piece[:, -1] == rect_col):
            return None

    if corner == "TL":
        anchor_r, anchor_c = 0, 0
    elif corner == "TR":
        anchor_r, anchor_c = 0, RW - 1
    elif corner == "BL":
        anchor_r, anchor_c = RH - 1, 0
    else:
        anchor_r, anchor_c = RH - 1, RW - 1

    anchor_idx = None
    for i, (pr, pc, ph, pw) in enumerate(pieces):
        if pr <= anchor_r < pr + ph and pc <= anchor_c < pc + pw:
            anchor_idx = i
            break
    if anchor_idx is None:
        return None

    apr, apc, aph, apw = pieces[anchor_idx]
    if corner == "TL" and not (apr == 0 and apc == 0):
        return None
    if corner == "TR" and not (apr == 0 and apc + apw == RW):
        return None
    if corner == "BL" and not (apr + aph == RH and apc == 0):
        return None
    if corner == "BR" and not (apr + aph == RH and apc + apw == RW):
        return None

    if corner == "TL":
        marker_cells = [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    elif corner == "TR":
        marker_cells = [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    elif corner == "BL":
        marker_cells = [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    else:
        marker_cells = [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]

    for cr, cc in marker_cells:
        if cr < 0 or cr >= GH or cc < 0 or cc >= GW:
            return None

    tr, tc = _top_left_from_corner(mr, mc, RH, RW, corner)
    output_box = (int(tr), int(tc), int(RH), int(RW))

    piece_arrays = [rect[pr : pr + ph, pc : pc + pw].copy() for pr, pc, ph, pw in pieces]
    target_boxes: list[tuple[int, int, int, int]] = []
    for pr, pc, ph, pw in pieces:
        lr, lc, lh, lw = _mirror_local_box((pr, pc, ph, pw), RH, RW, corner)
        target_boxes.append((int(tr + lr), int(tc + lc), int(lh), int(lw)))

    scattered = _scatter_pieces(GH, GW, bg, marker_col, marker_cells, piece_arrays, target_boxes, output_box, _r)
    if scattered is None:
        return None
    canvas, source_boxes = scattered

    expected_out, trail_stats = _render_output(
        GH, GW, bg, reflected, (tr, tc), corner, marker_col, source_boxes, target_boxes
    )

    if trail_stats["visible_trail_cells"] < max(8, len(pieces) + 2):
        return None
    if trail_stats["pieces_with_visible_trail"] < max(2, len(pieces) // 3):
        return None

    try:
        actual_out = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual_out, expected_out):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
