# Variant of 446ef5d2
# Dimension: D14 — 碎片的空间分布规则 (meta)
# Change: 碎片随机散布在背景上，彼此不重叠，间距至少 1 个背景像素... → 碎片按特定顺序排列

from __future__ import annotations

from collections import Counter
import random

import numpy as np


# Design analysis for the D14 variant
# -----------------------------------
# a) Pixel-level concretization:
#    The original input uses the same rectangular pieces, but scatters them randomly.
#    In this variant the pieces are still separate axis-aligned rectangles, yet they are
#    displayed in an ordered "catalog": pieces sharing the same final top edge are aligned
#    in the same display row, and pieces sharing the same final left edge are aligned in
#    the same display column. This gives a visible clue about their final row/column order.
#    Example: if the final rectangle has pieces A(top-left), B(top-right), C(bottom-left),
#    D(bottom-right), the input no longer places A/B/C/D arbitrarily; it places them as
#    a 2x2 table in that same order, with background gaps between entries.
#
#    Original flow at the placement stage:
#      final rectangle -> cut into pieces -> shuffle order -> random free locations
#    Variant flow:
#      final rectangle -> cut into pieces -> compute each piece's final top-left row/col
#      rank -> place them into display rows/cols preserving that rank order.
#
# b) Complexity calibration:
#    If all pieces sat in a single row or a single column, the new rule would be weak.
#    The generator therefore requires at least 2 distinct display rows and 2 distinct
#    display columns. Piece count stays in the original broad range, and colors / matching
#    rules remain unchanged, so the novelty is the spatial regularity rather than a new
#    matching mechanism.
#
# c) Implicit assumptions in the original solver that depended on random scatter:
#    - Piece extraction assumes each visible piece is an isolated solid rectangle.
#      This still holds because the ordered layout keeps background gutters.
#    - The backtracking order was area-based and ignored input position.
#      In the variant that would waste the new signal, so main() now derives display
#      row/column ranks from input positions and uses them as monotone constraints.
#    - The old "anchor piece at marker corner" detection looked for a piece occupying
#      the marker's missing cell in the input, which was effectively dead code. Here the
#      useful anchor is instead inferred from the ordered display: the corner piece is the
#      extremal piece in row-rank/col-rank for the marker corner.
#
# d) Placement legality:
#    Pieces remain non-overlapping and separated by explicit gutters. The marker's 2x2
#    neighborhood is kept away from the ordered display block, so the marker is visually
#    distinct and piece extraction sees only the intended rectangles.
#
# e) Visual observability:
#    Across 2-4 examples, humans can compare input and output and notice that the input
#    already lists the fragments in top-to-bottom / left-to-right final order. The solver
#    still needs edge-color matching to recover exact dimensions, so the puzzle does not
#    collapse into trivial copying.
#
# f) Visual cleanliness:
#    The ordered display is a small grid of separated rectangles. No overlays are used,
#    and the marker remains a separate 3-cell L. This avoids the noisy look that a
#    partially overlapping "exploded view" would create.
#
# g) Color reasoning:
#    No new color mapping is introduced. Colors retain their original roles:
#    bg = background, rect_col = wildcard border color, marker_col = 3-cell L marker,
#    interior colors = matching structure on piece boundaries.


def _find_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], tuple[int, int, int, int]]:
    cnt = Counter(map(int, g.flatten()))
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
                return int(col), miss[0], (r0, c0, r1, c1)
    raise AssertionError("marker not found")


def _corner_from_missing(missing: tuple[int, int], mb: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = mb
    if missing == (r0, c0):
        return "BR"
    if missing == (r0, c1):
        return "BL"
    if missing == (r1, c0):
        return "TR"
    return "TL"


def _extract_rectangles(g: np.ndarray, bg: int, mark: int) -> list[tuple[int, int, int, int, np.ndarray]]:
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


def _compat(v1: np.ndarray, v2: np.ndarray, rect_col: int) -> bool:
    v1 = np.array(v1, dtype=int)
    v2 = np.array(v2, dtype=int)
    m = (v1 != int(rect_col)) & (v2 != int(rect_col))
    return bool((v1[m] == v2[m]).all())


def _layout_ranks(rects: list[tuple[int, int, int, int, np.ndarray]]) -> tuple[list[int], list[int]]:
    row_vals = sorted({int(rr) for rr, _cc, _h, _w, _a in rects})
    col_vals = sorted({int(cc) for _rr, cc, _h, _w, _a in rects})
    row_rank = {v: i for i, v in enumerate(row_vals)}
    col_rank = {v: i for i, v in enumerate(col_vals)}
    return [row_rank[int(rr)] for rr, _cc, _h, _w, _a in rects], [col_rank[int(cc)] for _rr, cc, _h, _w, _a in rects]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble ordered rectangular pieces into a rectangle at an L-marker corner.

    Variant-specific observation:
    - Visible pieces are no longer randomly scattered. Their input positions preserve the
      top-left row/column order they will have in the final rectangle.
    - main() still validates placements by edge matching with rect_col as wildcard, but
      now also uses the inferred input row/column ranks as monotone constraints.
    """
    g = np.array(grid, dtype=int)
    GH, GW = g.shape

    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, missing, mb = _find_marker(g, bg)
    corner = _corner_from_missing(missing, mb)

    rects = _extract_rectangles(g, bg, mark)
    assert rects
    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))

    allpix: list[int] = []
    for _r, _c, _h, _w, a in rects:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    pieces = [a for _r, _c, _h, _w, a in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]
    row_rank, col_rank = _layout_ranks(rects)

    mr, mc = map(int, missing)

    dims = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh == 0:
            dims.append((hh, total_area // hh))
            if hh * hh != total_area:
                dims.append((total_area // hh, hh))

    def fits(H: int, W: int) -> bool:
        if corner == "TL":
            tr, tc = mr, mc
        elif corner == "TR":
            tr, tc = mr, mc - W + 1
        elif corner == "BL":
            tr, tc = mr - H + 1, mc
        else:
            tr, tc = mr - H + 1, mc - W + 1
        return 0 <= tr and 0 <= tc and (tr + H) <= GH and (tc + W) <= GW

    dims = [(int(H), int(W)) for H, W in dims if fits(int(H), int(W))]
    dims = sorted(dims, key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    def assemble(H: int, W: int) -> np.ndarray | None:
        occ2 = np.full((H, W), -1, dtype=int)
        usedp = [False] * len(pieces)
        placed_pos: list[tuple[int, int] | None] = [None] * len(pieces)

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
                if m.any() and (not _compat(left[m], a[:, 0][m], rect_col)):
                    return False
            if (c + w) < W:
                right = occ2[r : r + h, c + w]
                m = right != -1
                if m.any() and (not _compat(right[m], a[:, -1][m], rect_col)):
                    return False
            if r > 0:
                up = occ2[r - 1, c : c + w]
                m = up != -1
                if m.any() and (not _compat(up[m], a[0, :][m], rect_col)):
                    return False
            if (r + h) < H:
                down = occ2[r + h, c : c + w]
                m = down != -1
                if m.any() and (not _compat(down[m], a[-1, :][m], rect_col)):
                    return False
            return True

        def order_ok(i: int, r: int, c: int) -> bool:
            for j, pos in enumerate(placed_pos):
                if pos is None:
                    continue
                rj, cj = pos
                if row_rank[i] < row_rank[j] and r > rj:
                    return False
                if row_rank[i] > row_rank[j] and r < rj:
                    return False
                if row_rank[i] == row_rank[j] and r != rj:
                    return False
                if col_rank[i] < col_rank[j] and c > cj:
                    return False
                if col_rank[i] > col_rank[j] and c < cj:
                    return False
                if col_rank[i] == col_rank[j] and c != cj:
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
            if not order_ok(i, r, c):
                return False
            return True

        def place(i: int, r: int, c: int) -> None:
            a = pieces[i]
            h, w = a.shape
            occ2[r : r + h, c : c + w] = a
            usedp[i] = True
            placed_pos[i] = (r, c)

        def unplace(i: int, r: int, c: int) -> None:
            a = pieces[i]
            h, w = a.shape
            occ2[r : r + h, c : c + w] = -1
            usedp[i] = False
            placed_pos[i] = None

        # Variant-specific anchor: use the extremal display ranks to infer the corner piece.
        target_rr = min(row_rank) if corner in ("TL", "TR") else max(row_rank)
        target_cc = min(col_rank) if corner in ("TL", "BL") else max(col_rank)
        anchor_candidates = [i for i in range(len(pieces)) if row_rank[i] == target_rr and col_rank[i] == target_cc]
        for anchor_idx in anchor_candidates:
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
                break

        order = sorted(
            range(len(pieces)),
            key=lambda i: (row_rank[i], col_rank[i], -(ph[i] * pw[i]), -ph[i], -pw[i], i),
        )

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

    ass = None
    used_dim = None
    for H, W in dims:
        ass = assemble(H, W)
        if ass is not None:
            used_dim = (H, W)
            break

    assert not (ass is None or used_dim is None)

    H, W = used_dim
    if corner == "TL":
        tr, tc = mr, mc
    elif corner == "TR":
        tr, tc = mr, mc - W + 1
    elif corner == "BL":
        tr, tc = mr - H + 1, mc
    else:
        tr, tc = mr - H + 1, mc - W + 1

    out = np.full((GH, GW), int(bg), dtype=int)
    out[tr : tr + H, tc : tc + W] = ass
    return out


def _rand_splits(total: int, min_s: int = 2, max_s: int = 7, min_parts: int = 2, max_parts: int = 4) -> list[int] | None:
    feasible = []
    for k in range(min_parts, max_parts + 1):
        if k * min_s <= total <= k * max_s:
            feasible.append(k)
    if not feasible:
        return None
    k = random.choice(feasible)
    segs = []
    rem = total
    for i in range(k):
        rem_parts = k - i - 1
        lo = max(min_s, rem - rem_parts * max_s)
        hi = min(max_s, rem - rem_parts * min_s)
        if lo > hi:
            return None
        val = random.randint(lo, hi)
        segs.append(val)
        rem -= val
    return segs


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _marker_cells(corner: str, mr: int, mc: int) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _gen_attempt() -> np.ndarray | None:
    # --- colors ---
    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    n_interior = random.randint(1, 3)
    interior_cols = palette[3 : 3 + n_interior]

    # --- build assembled rectangle first ---
    RH = random.randint(6, 14)
    RW = random.randint(6, 14)
    rect = np.full((RH, RW), rect_col, dtype=int)
    iH, iW = RH - 2, RW - 2
    if iH < 1 or iW < 1:
        return None
    interior = np.full((iH, iW), rect_col, dtype=int)

    target_fill = random.uniform(0.35, 0.80)
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
            nxt_frontier = []
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
                    nxt_frontier.append((nr, nc))
                    if len(blob) >= budget:
                        break
                if len(blob) >= budget:
                    break
            frontier = nxt_frontier

        for br, bc in blob:
            interior[br, bc] = ic
            blob_occupied[br, bc] = True

    fill_ratio = int(np.sum(interior != rect_col)) / (iH * iW)
    if fill_ratio < 0.30 or fill_ratio > 0.90:
        return None
    rect[1:-1, 1:-1] = interior

    # --- partition rectangle into rectangular pieces ---
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

    def piece_bbox(pid: int) -> tuple[int, int, int, int] | None:
        cells = [(i, j) for i in range(nr) for j in range(nc_grid) if cell_id[i][j] == pid]
        if not cells:
            return None
        ri = [r for r, _c in cells]
        ci = [c for _r, c in cells]
        return min(ri), min(ci), max(ri), max(ci)

    def pixel_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
        h = sum(row_segs[i] for i in range(r0i, r1i + 1))
        w = sum(col_segs[j] for j in range(c0i, c1i + 1))
        return h, w

    def count_pieces() -> int:
        return len(set(cell_id[i][j] for i in range(nr) for j in range(nc_grid)))

    target_pieces = random.randint(4, min(10, n_cells))
    merges = []
    for i in range(nr):
        for j in range(nc_grid):
            if j + 1 < nc_grid:
                merges.append(("h", i, j))
            if i + 1 < nr:
                merges.append(("v", i, j))
    random.shuffle(merges)

    for direction, mi, mj in merges:
        if count_pieces() <= target_pieces:
            break
        if direction == "h":
            pid1 = cell_id[mi][mj]
            pid2 = cell_id[mi][mj + 1]
        else:
            pid1 = cell_id[mi][mj]
            pid2 = cell_id[mi + 1][mj]
        if pid1 == pid2:
            continue
        bb1 = piece_bbox(pid1)
        bb2 = piece_bbox(pid2)
        if bb1 is None or bb2 is None:
            continue
        mr0 = min(bb1[0], bb2[0])
        mc0 = min(bb1[1], bb2[1])
        mr1 = max(bb1[2], bb2[2])
        mc1 = max(bb1[3], bb2[3])
        ph, pw = pixel_size(mr0, mc0, mr1, mc1)
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

    pieces = []
    seen = set()
    for i in range(nr):
        for j in range(nc_grid):
            pid = cell_id[i][j]
            if pid in seen:
                continue
            seen.add(pid)
            bb = piece_bbox(pid)
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
        fill = int(np.sum(piece != rect_col)) / (ph * pw)
        if fill < 0.10 or fill > 0.60:
            return None
        has_non_rc_edge = any(np.any(edge != rect_col) for edge in [piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]])
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

    top_rows = sorted({pr for pr, _pc, _ph, _pw in pieces})
    left_cols = sorted({pc for _pr, pc, _ph, _pw in pieces})
    if len(top_rows) < 2 or len(left_cols) < 2:
        return None
    row_rank = {v: i for i, v in enumerate(top_rows)}
    col_rank = {v: i for i, v in enumerate(left_cols)}

    gap = 1
    row_heights = [0] * len(top_rows)
    col_widths = [0] * len(left_cols)
    for pr, pc, ph, pw in pieces:
        row_heights[row_rank[pr]] = max(row_heights[row_rank[pr]], ph)
        col_widths[col_rank[pc]] = max(col_widths[col_rank[pc]], pw)
    AH = sum(row_heights) + gap * (len(row_heights) - 1)
    AW = sum(col_widths) + gap * (len(col_widths) - 1)

    baseH = max(10, max(AH, RH) + 4)
    baseW = max(10, max(AW, RW) + 4)
    if baseH > 26 or baseW > 26:
        return None
    GH = random.randint(baseH, 26)
    GW = random.randint(baseW, 26)

    row_pos = []
    s = 0
    for idx, h in enumerate(row_heights):
        row_pos.append(s)
        s += h + (gap if idx + 1 < len(row_heights) else 0)
    col_pos = []
    s = 0
    for idx, w in enumerate(col_widths):
        col_pos.append(s)
        s += w + (gap if idx + 1 < len(col_widths) else 0)

    arranged_block = np.full((AH, AW), bg, dtype=int)
    for pr, pc, ph, pw in pieces:
        rr = row_pos[row_rank[pr]]
        cc = col_pos[col_rank[pc]]
        arranged_block[rr : rr + ph, cc : cc + pw] = rect[pr : pr + ph, pc : pc + pw]

    arr_positions = [(r, c) for r in range(GH - AH + 1) for c in range(GW - AW + 1)]
    random.shuffle(arr_positions)

    corner = random.choice(["TL", "TR", "BL", "BR"])
    marker_candidates = []
    if corner == "TL":
        mr_range = range(2, GH - RH + 1)
        mc_range = range(2, GW - RW + 1)
    elif corner == "TR":
        mr_range = range(2, GH - RH + 1)
        mc_range = range(RW - 1, GW - 2)
    elif corner == "BL":
        mr_range = range(RH - 1, GH - 2)
        mc_range = range(2, GW - RW + 1)
    else:
        mr_range = range(RH - 1, GH - 2)
        mc_range = range(RW - 1, GW - 2)

    for mr in mr_range:
        for mc in mc_range:
            marker_candidates.append((mr, mc))
    random.shuffle(marker_candidates)

    chosen = None
    for arr_r0, arr_c0 in arr_positions:
        arr_bbox = (arr_r0 - 1, arr_c0 - 1, arr_r0 + AH, arr_c0 + AW)
        for mr, mc in marker_candidates:
            marker_box = (min(r for r, _c in _marker_cells(corner, mr, mc)), min(c for _r, c in _marker_cells(corner, mr, mc)),
                          max(r for r, _c in _marker_cells(corner, mr, mc)), max(c for _r, c in _marker_cells(corner, mr, mc)))
            if _bbox_intersects(arr_bbox, marker_box):
                continue
            chosen = (arr_r0, arr_c0, mr, mc)
            break
        if chosen is not None:
            break
    if chosen is None:
        return None

    arr_r0, arr_c0, mr, mc = chosen
    canvas = np.full((GH, GW), bg, dtype=int)
    canvas[arr_r0 : arr_r0 + AH, arr_c0 : arr_c0 + AW] = arranged_block
    for cr, cc in _marker_cells(corner, mr, mc):
        if cr < 0 or cr >= GH or cc < 0 or cc >= GW:
            return None
        if canvas[cr, cc] != bg:
            return None
        canvas[cr, cc] = marker_col

    if corner == "TL":
        tr, tc = mr, mc
    elif corner == "TR":
        tr, tc = mr, mc - RW + 1
    elif corner == "BL":
        tr, tc = mr - RH + 1, mc
    else:
        tr, tc = mr - RH + 1, mc - RW + 1

    expected_out = np.full((GH, GW), bg, dtype=int)
    expected_out[tr : tr + RH, tc : tc + RW] = rect

    try:
        actual_out = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual_out, expected_out):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
