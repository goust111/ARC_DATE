# Combo variant of 446ef5d2 (index 6)
# combo_spec: D5/2+D19/1
# Name: 原位轮廓线配合背景色变化

from __future__ import annotations

from collections import Counter
import random

import numpy as np


"""
Design analysis for the fused variant
=====================================

Concrete behavior trace
-----------------------
Original task:
- Input background is a dominant color `bg`.
- Rectangular fragments are scattered on `bg`.
- A 3-cell L marker of color `mark` indicates which corner of the final rectangle is pinned.
- Output keeps the same `bg` and only shows the assembled rectangle.

Fused variant:
- The rectangle assembly logic is unchanged.
- After fragments "move" to the assembled rectangle, each fragment's original bounding box is
  redrawn as a hollow rectangle at its old location.
- The output background does not stay `bg`; it changes to the marker color `mark`.
- To make the two dimensions genuinely interact, the hollow outlines are drawn in the *old*
  background color `bg`. This means the old background, invisible in the input, becomes a
  visible line color only because the new background is `mark`.

Small example:
- Suppose input bg is 8, marker color is 4, and one scattered 3x4 piece sits at rows 1..3,
  cols 6..9.
- Original output: keep background 8, erase the source piece completely, show only the final
  assembled rectangle.
- New output: fill empty space with 4, draw a hollow 3x4 frame of color 8 at rows 1..3,
  cols 6..9, and place the assembled rectangle at the marker corner. The L marker disappears
  naturally because the whole background becomes 4.

Complexity calibration
----------------------
- The background-change rule uses one stable source: the L-marker color. This is visually
  strong and repeated in every example, so it is inferable from a few I/O pairs.
- The outline rule uses no extra arbitrary color. The old background becomes the outline
  color, which avoids introducing a weakly-motivated new mapping.
- Pieces are still axis-aligned rectangles, so source outlines are clean hollow boxes rather
  than noisy traces. This keeps the added complexity legible.

Implicit-assumption audit against the original solver
-----------------------------------------------------
- The original parsing assumes that every non-bg, non-marker connected rectangle is a source
  fragment. This still holds because the input itself is unchanged.
- The original solver assumes `bg` is the most common color in the input. Still true.
- The original solver assumes the L-marker color appears exactly 3 times and forms a 2x2-minus-1
  pattern. Still true.
- The original solver assumes pieces do not overlap and are rectangular. Still true.
- New output rendering adds assumptions not present before:
  1. Source outlines should not overlap the assembled target rectangle, otherwise two output
     objects would compete for the same cells.
  2. Source outlines should remain visually separated from one another.
- The generator therefore reserves the final rectangle footprint while scattering pieces and
  keeps the existing >=1 cell spacing between scattered pieces.

Placement legality and observability
------------------------------------
- The generator uses output-first construction, then scatters pieces only in cells outside the
  reserved final rectangle footprint.
- Candidate locations are checked with a 1-cell padding margin, so pieces do not touch each
  other and do not touch the reserved final rectangle.
- Because the output background is `mark` and outlines are `bg`, the fused behavior is
  non-degenerate: if we reverted to the original rendering, both the background field and the
  source-fragment footprints would visibly change.
"""


def _find_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], str]:
    cnt = Counter(map(int, g.flatten()))
    for col in sorted(c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3):
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(col))]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0) != 1 or (c1 - c0) != 1:
            continue
        all4 = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        miss = list(all4 - set(pts))
        if len(miss) != 1:
            continue
        missing = miss[0]
        if missing == (r0, c0):
            corner = "BR"
        elif missing == (r0, c1):
            corner = "BL"
        elif missing == (r1, c0):
            corner = "TR"
        else:
            corner = "TL"
        return int(col), (int(missing[0]), int(missing[1])), corner
    raise AssertionError("marker not found")


def _extract_rects(g: np.ndarray, bg: int, mark: int) -> list[tuple[int, int, int, int, np.ndarray]]:
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


def _top_left_from_corner(mr: int, mc: int, H: int, W: int, corner: str) -> tuple[int, int]:
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - W + 1
    if corner == "BL":
        return mr - H + 1, mc
    return mr - H + 1, mc - W + 1


def _assemble_from_rects(
    rects: list[tuple[int, int, int, int, np.ndarray]],
    missing: tuple[int, int],
    corner: str,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int], int]:
    GH, GW = map(int, grid_shape)
    mr, mc = map(int, missing)
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

    dims = []
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
        m = (v1 != int(rect_col)) & (v2 != int(rect_col))
        return bool((v1[m] == v2[m]).all())

    def assemble(H: int, W: int) -> np.ndarray | None:
        occ2 = np.full((H, W), -1, dtype=int)
        usedp = [False] * len(pieces)

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

        def unplace(i: int, r: int, c: int) -> None:
            a = pieces[i]
            h, w = a.shape
            occ2[r : r + h, c : c + w] = -1
            usedp[i] = False

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
        return occ2 if solved else None

    for H, W in dims:
        ass = assemble(H, W)
        if ass is not None:
            return ass, (int(H), int(W)), rect_col
    raise AssertionError("assembly failed")


def _paint_hollow_box(out: np.ndarray, rr: int, cc: int, h: int, w: int, color: int) -> None:
    out[rr, cc : cc + w] = int(color)
    out[rr + h - 1, cc : cc + w] = int(color)
    out[rr : rr + h, cc] = int(color)
    out[rr : rr + h, cc + w - 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, missing, corner = _find_marker(g, bg)
    rects = _extract_rects(g, bg, mark)
    ass, (H, W), _rect_col = _assemble_from_rects(rects, missing, corner, (GH, GW))
    tr, tc = _top_left_from_corner(int(missing[0]), int(missing[1]), H, W, corner)

    out_bg = int(mark)
    out = np.full((GH, GW), out_bg, dtype=int)

    for rr, cc, h, w, _a in rects:
        _paint_hollow_box(out, int(rr), int(cc), int(h), int(w), int(bg))

    out[tr : tr + H, tc : tc + W] = ass
    return out


def _choose_rectangle_dims(GH: int, GW: int) -> tuple[int, int] | None:
    max_side = min(13, GH - 4, GW - 4)
    if max_side < 5:
        return None
    for _ in range(80):
        RH = random.randint(5, max_side)
        RW = random.randint(5, max_side)
        area = RH * RW
        if area <= int(0.30 * GH * GW):
            return RH, RW
    return None


def _grow_blobs(iH: int, iW: int, rect_col: int, interior_cols: list[int]) -> np.ndarray | None:
    interior = np.full((iH, iW), rect_col, dtype=int)
    target_fill = random.uniform(0.35, 0.75)
    target_pixels = max(1, int(target_fill * iH * iW))
    blob_occupied = np.zeros((iH, iW), dtype=bool)

    for ci, ic in enumerate(interior_cols):
        if ci == len(interior_cols) - 1:
            budget = target_pixels - int(blob_occupied.sum())
        else:
            budget = max(1, target_pixels // len(interior_cols))
        budget = max(1, budget)

        seed = None
        for _ in range(200):
            sr, sc = random.randint(0, iH - 1), random.randint(0, iW - 1)
            if blob_occupied[sr, sc]:
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
        if seed is None:
            continue

        blob = {seed}
        frontier = [seed]
        while len(blob) < budget and frontier:
            random.shuffle(frontier)
            nf = []
            for fr, fc in frontier:
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr, nc = fr + dr, fc + dc
                    if not (0 <= nr < iH and 0 <= nc < iW):
                        continue
                    if (nr, nc) in blob or blob_occupied[nr, nc]:
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

    fill_ratio = float(np.mean(interior != rect_col))
    if fill_ratio < 0.30 or fill_ratio > 0.85:
        return None
    return interior


def _rand_splits(total: int, min_s: int = 2, max_s: int = 7) -> list[int] | None:
    eff_max = min(max_s, max(min_s, random.randint(min_s, max_s)))
    segs = []
    rem = total
    while rem > 0:
        if rem <= eff_max:
            if rem < min_s:
                return None
            segs.append(rem)
            break
        hi = min(eff_max, rem - min_s)
        if min_s > hi:
            return None
        s = random.randint(min_s, hi)
        segs.append(s)
        rem -= s
    return segs


def _partition_rectangle(rect: np.ndarray) -> list[tuple[int, int, int, int]] | None:
    RH, RW = rect.shape
    row_segs = _rand_splits(RH)
    col_segs = _rand_splits(RW)
    if row_segs is None or col_segs is None:
        return None

    row_starts = []
    s = 0
    for rs in row_segs:
        row_starts.append(s)
        s += rs
    col_starts = []
    s = 0
    for cs in col_segs:
        col_starts.append(s)
        s += cs

    nr, nc_grid = len(row_segs), len(col_segs)
    cell_id = [[i * nc_grid + j for j in range(nc_grid)] for i in range(nr)]
    n_cells = nr * nc_grid
    if n_cells < 4:
        return None

    def get_piece_bbox(pid: int) -> tuple[int, int, int, int] | None:
        cells = [(i, j) for i in range(nr) for j in range(nc_grid) if cell_id[i][j] == pid]
        if not cells:
            return None
        ri = [c[0] for c in cells]
        ci = [c[1] for c in cells]
        return min(ri), min(ci), max(ri), max(ci)

    def pixel_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
        h = sum(row_segs[i] for i in range(r0i, r1i + 1))
        w = sum(col_segs[j] for j in range(c0i, c1i + 1))
        return h, w

    target_pieces = random.randint(4, min(10, n_cells))
    merges = []
    for i in range(nr):
        for j in range(nc_grid):
            if j + 1 < nc_grid:
                merges.append(("h", i, j))
            if i + 1 < nr:
                merges.append(("v", i, j))
    random.shuffle(merges)

    def count_pieces() -> int:
        return len(set(cell_id[i][j] for i in range(nr) for j in range(nc_grid)))

    for direction, mi, mj in merges:
        if count_pieces() <= target_pieces:
            break
        pid1 = cell_id[mi][mj]
        pid2 = cell_id[mi][mj + 1] if direction == "h" else cell_id[mi + 1][mj]
        if pid1 == pid2:
            continue
        bb1 = get_piece_bbox(pid1)
        bb2 = get_piece_bbox(pid2)
        if bb1 is None or bb2 is None:
            continue
        mr0, mc0 = min(bb1[0], bb2[0]), min(bb1[1], bb2[1])
        mr1, mc1 = max(bb1[2], bb2[2]), max(bb1[3], bb2[3])
        ph, pw = pixel_size(mr0, mc0, mr1, mc1)
        if ph > 7 or pw > 7:
            continue
        ok = True
        for ii in range(mr0, mr1 + 1):
            for jj in range(mc0, mc1 + 1):
                if cell_id[ii][jj] not in (pid1, pid2):
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

    pieces = []
    seen = set()
    for i in range(nr):
        for j in range(nc_grid):
            pid = cell_id[i][j]
            if pid in seen:
                continue
            seen.add(pid)
            bb = get_piece_bbox(pid)
            if bb is None:
                return None
            r0i, c0i, r1i, c1i = bb
            pr = row_starts[r0i]
            pc = col_starts[c0i]
            ph = sum(row_segs[ii] for ii in range(r0i, r1i + 1))
            pw = sum(col_segs[jj] for jj in range(c0i, c1i + 1))
            pieces.append((pr, pc, ph, pw))

    return pieces if 4 <= len(pieces) <= 10 else None


def _piece_checks(
    rect: np.ndarray,
    pieces: list[tuple[int, int, int, int]],
    rect_col: int,
    corner: str,
) -> int | None:
    RH, RW = rect.shape
    for pr, pc, ph, pw in pieces:
        if ph < 2 or pw < 2 or ph > 7 or pw > 7:
            return None
        piece = rect[pr : pr + ph, pc : pc + pw]
        fill = float(np.mean(piece != rect_col))
        if fill < 0.10 or fill > 0.60:
            return None
        if not any(np.any(edge != rect_col) for edge in [piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]]):
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
    return anchor_idx


def _marker_cells(corner: str, mr: int, mc: int) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _reserve_and_place(
    GH: int,
    GW: int,
    pieces: list[np.ndarray],
    marker_cells: list[tuple[int, int]],
    reserved_rect: tuple[int, int, int, int],
) -> list[tuple[int, int]] | None:
    bg_canvas = np.zeros((GH, GW), dtype=bool)
    r0, c0, r1, c1 = reserved_rect
    bg_canvas[r0 : r1 + 1, c0 : c1 + 1] = True
    for r, c in marker_cells:
        bg_canvas[r, c] = True

    placements: list[tuple[int, int] | None] = [None] * len(pieces)
    order = list(range(len(pieces)))
    random.shuffle(order)
    for idx in order:
        pa = pieces[idx]
        h, w = pa.shape
        candidates = []
        for rr in range(0, GH - h + 1):
            for cc in range(0, GW - w + 1):
                rr0 = max(0, rr - 1)
                cc0 = max(0, cc - 1)
                rr1 = min(GH, rr + h + 1)
                cc1 = min(GW, cc + w + 1)
                if bg_canvas[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((rr, cc))
        if not candidates:
            return None
        rr, cc = random.choice(candidates)
        bg_canvas[rr : rr + h, cc : cc + w] = True
        placements[idx] = (rr, cc)
    if any(p is None for p in placements):
        return None
    return [(int(p[0]), int(p[1])) for p in placements if p is not None]


def _render_expected(
    GH: int,
    GW: int,
    bg: int,
    marker_col: int,
    rect: np.ndarray,
    top_left: tuple[int, int],
    scattered_boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = np.full((GH, GW), int(marker_col), dtype=int)
    for rr, cc, h, w in scattered_boxes:
        _paint_hollow_box(out, rr, cc, h, w, int(bg))
    tr, tc = top_left
    RH, RW = rect.shape
    out[tr : tr + RH, tc : tc + RW] = rect
    return out


def _gen_attempt() -> np.ndarray | None:
    GH = random.randint(16, 26)
    GW = random.randint(16, 26)

    dims = _choose_rectangle_dims(GH, GW)
    if dims is None:
        return None
    RH, RW = dims

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    n_interior = random.randint(1, 3)
    interior_cols = palette[3 : 3 + n_interior]

    corner = random.choice(["TL", "TR", "BL", "BR"])
    if corner == "TL":
        mr_min, mr_max = 2, GH - RH - 2
        mc_min, mc_max = 2, GW - RW - 2
    elif corner == "TR":
        mr_min, mr_max = 2, GH - RH - 2
        mc_min, mc_max = RW + 1, GW - 3
    elif corner == "BL":
        mr_min, mr_max = RH + 1, GH - 3
        mc_min, mc_max = 2, GW - RW - 2
    else:
        mr_min, mr_max = RH + 1, GH - 3
        mc_min, mc_max = RW + 1, GW - 3
    if mr_min > mr_max or mc_min > mc_max:
        return None
    mr = random.randint(mr_min, mr_max)
    mc = random.randint(mc_min, mc_max)

    rect = np.full((RH, RW), rect_col, dtype=int)
    interior = _grow_blobs(RH - 2, RW - 2, rect_col, interior_cols)
    if interior is None:
        return None
    rect[1:-1, 1:-1] = interior

    pieces = _partition_rectangle(rect)
    if pieces is None:
        return None
    anchor_idx = _piece_checks(rect, pieces, rect_col, corner)
    if anchor_idx is None:
        return None

    marker_cells = _marker_cells(corner, mr, mc)
    for r, c in marker_cells:
        if not (0 <= r < GH and 0 <= c < GW):
            return None

    tr, tc = _top_left_from_corner(mr, mc, RH, RW, corner)
    reserved_rect = (tr, tc, tr + RH - 1, tc + RW - 1)

    piece_arrays = [rect[pr : pr + ph, pc : pc + pw].copy() for pr, pc, ph, pw in pieces]
    placements = _reserve_and_place(GH, GW, piece_arrays, marker_cells, reserved_rect)
    if placements is None:
        return None

    canvas = np.full((GH, GW), bg, dtype=int)
    for r, c in marker_cells:
        canvas[r, c] = marker_col

    scattered_boxes: list[tuple[int, int, int, int]] = []
    for pa, (rr, cc) in zip(piece_arrays, placements):
        h, w = pa.shape
        canvas[rr : rr + h, cc : cc + w] = pa
        scattered_boxes.append((rr, cc, h, w))

    expected = _render_expected(GH, GW, bg, marker_col, rect, (tr, tc), scattered_boxes)
    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
