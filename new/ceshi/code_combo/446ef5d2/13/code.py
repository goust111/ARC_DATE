# Combo variant of 446ef5d2 (index 13)
# combo_spec: D7/0+D14/0
# Name: 有序排列碎片配合外向内拼接

from __future__ import annotations

from collections import Counter
import random

import numpy as np


# Design analysis for the combo variant
# -------------------------------------
# a) Pixel-level concretization and trace:
#    Original task:
#      - Build a target rectangle with rect_col border and irregular interior colors.
#      - Cut it into rectangular pieces.
#      - Scatter those pieces randomly on the background.
#      - Solver extracts rectangles and uses generic backtracking with edge matching.
#    Combo task:
#      - The same kind of rectangular pieces are still extracted as solid rectangles.
#      - But the input no longer scatters them randomly. Instead, it displays them as an
#        ordered catalog: pieces that start on the same final top edge share a display row,
#        and pieces that start on the same final left edge share a display column.
#      - The solver must also reason in layers: first place pieces whose final top-left
#        belongs to the outer ring of the piece-rank grid, then move one ring inward.
#    Small concrete trace:
#      - Suppose the final rectangle is partitioned into a 3x3 grid of piece start ranks.
#      - The input shows those pieces as a 3x3 catalog with background gutters.
#      - Outer-ring pieces are exactly the pieces whose rank row or rank col is extremal.
#      - The center-start piece is visibly in the center of the catalog and is only used
#        after the outer ring is placed. This changes the search logic from "any piece
#        anywhere" to "read row/col order from the input, then peel the assembly onion".
#
# b) Complexity calibration:
#    The ordered layout alone would be too weak if all pieces lived in one row/column.
#    The generator therefore requires at least 3 distinct piece-start rows and 3 distinct
#    piece-start columns, so at least two rank-layers exist. It also requires at least one
#    inner piece (layer >= 1). This preserves ambiguity at the edge-matching level while
#    making the fusion behavior visually observable.
#
# c) Implicit assumption audit against the original solver:
#    - Rectangle extraction by scanning non-bg/non-marker cells still works because the
#      catalog uses background gutters and pieces remain axis-aligned rectangles.
#    - The original anchor detection looked for a piece physically covering the marker's
#      missing cell in the input. That assumption is broken here because the catalog is
#      spatially separate from the marker. We replace it with an ordered-layout anchor:
#      the corner piece is the unique extremal row-rank/col-rank piece for the marker
#      corner.
#    - The original search selected the lexicographically first empty output cell. That is
#      incompatible with outer-to-inner assembly because after placing part of the outer
#      ring, the earliest empty pixel may already lie in the interior while some outer-ring
#      piece is still unplaced. We therefore switch to piece-driven search: pick the next
#      piece from the shallowest remaining rank-layer, enumerate legal placements, and
#      place the most constrained piece first.
#    - D7's prose mentions prioritizing pieces with rect_col-only exposed borders. In this
#      task family the original generator already forbids fully unconstrained "all four
#      edges rect_col" pieces. The combo therefore operationalizes the priority as:
#      outer-layer pieces first, and inside a layer prefer pieces touching more rectangle
#      borders (corners before sides, sides before inner-start pieces).
#
# d) Placement legality:
#    The generator builds the output rectangle first, cuts it into legal rectangular pieces,
#    then lays the catalog block and the 3-cell L-marker on disjoint background regions.
#    The catalog block has explicit gutters, marker cells stay in bounds, and the final
#    rectangle implied by the marker always fits inside the grid.
#
# e) Visual observability:
#    Across examples, a human can see that the input already lists fragments in final
#    row/column order, because top-row fragments appear in the display's top row, left-edge
#    fragments appear in the display's left column, and inner fragments appear toward the
#    center. The output is still not a trivial copy because exact assembly depends on edge
#    matching and the marker-corner placement.
#
# f) Visual cleanliness:
#    The combo avoids overlays entirely. There are only three object types on the input:
#    background, one catalog block of separated rectangles, and one isolated 3-cell L
#    marker. This keeps the examples legible.
#
# g) Inferability of the rule:
#    No new color mapping is introduced. Colors keep their original semantics. The added
#    signals are geometric and therefore directly visible: ordered display rows/columns and
#    a nested ring structure over those ranks.
# 
# h) Variant-specific metric used after testing:
#    - `ordered_row_count = number of distinct visible piece top coordinates`
#    - `ordered_col_count = number of distinct visible piece left coordinates`
#    - `max_rank_layer = max(min(row_rank, col_rank, ...))` over visible pieces
#    - `inner_piece_count = number of pieces with rank-layer >= 1`
#    - `cross_layer_piece_count = number of generated pieces whose latent cell bbox spans
#      more than one cell-ring`
#    - The generator is rejected unless ordered_row_count >= 3, ordered_col_count >= 3,
#      `max_rank_layer >= 1`, `inner_piece_count >= 1`, and
#      `cross_layer_piece_count == 0`, so the fusion behavior is not merely a search-order
#      artifact.


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


def _rank_layers(row_rank: list[int], col_rank: list[int]) -> list[int]:
    max_rr = max(row_rank)
    max_cc = max(col_rank)
    return [
        min(rr, cc, max_rr - rr, max_cc - cc)
        for rr, cc in zip(row_rank, col_rank)
    ]


def _cell_ring_index(r0: int, c0: int, r1: int, c1: int, nr: int, nc: int) -> int:
    return min(int(r0), int(c0), int(nr - 1 - r1), int(nc - 1 - c1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble an ordered catalog of rectangular pieces into a rectangle at an L-marker corner.

    New assumptions introduced by the combo and guaranteed by generate_input():
    - Visible input rectangles are laid out in a rank-preserving catalog with gutters.
    - There are at least 3 distinct piece-start rows and 3 distinct piece-start columns.
    - At least one piece belongs to an inner rank-layer, so outer-to-inner reasoning matters.
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
    piece_layer = _rank_layers(row_rank, col_rank)

    min_rr, max_rr = min(row_rank), max(row_rank)
    min_cc, max_cc = min(col_rank), max(col_rank)
    border_degree = [
        int(rr == min_rr) + int(rr == max_rr) + int(cc == min_cc) + int(cc == max_cc)
        for rr, cc in zip(row_rank, col_rank)
    ]

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
                if row_rank[i] < row_rank[j] and r >= rj:
                    return False
                if row_rank[i] > row_rank[j] and r <= rj:
                    return False
                if row_rank[i] == row_rank[j] and r != rj:
                    return False
                if col_rank[i] < col_rank[j] and c >= cj:
                    return False
                if col_rank[i] > col_rank[j] and c <= cj:
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

        def candidate_positions(i: int) -> list[tuple[int, int]]:
            h, w = ph[i], pw[i]
            r_lo, r_hi = 0, H - h
            c_lo, c_hi = 0, W - w

            if row_rank[i] == min_rr:
                r_lo = r_hi = 0
            elif row_rank[i] == max_rr:
                r_lo = r_hi = H - h
            for j, pos in enumerate(placed_pos):
                if pos is None:
                    continue
                rj, _cj = pos
                if row_rank[i] == row_rank[j]:
                    r_lo = max(r_lo, rj)
                    r_hi = min(r_hi, rj)
                elif row_rank[i] < row_rank[j]:
                    r_hi = min(r_hi, rj - 1)
                else:
                    r_lo = max(r_lo, rj + 1)

            if col_rank[i] == min_cc:
                c_lo = c_hi = 0
            elif col_rank[i] == max_cc:
                c_lo = c_hi = W - w
            for j, pos in enumerate(placed_pos):
                if pos is None:
                    continue
                _rj, cj = pos
                if col_rank[i] == col_rank[j]:
                    c_lo = max(c_lo, cj)
                    c_hi = min(c_hi, cj)
                elif col_rank[i] < col_rank[j]:
                    c_hi = min(c_hi, cj - 1)
                else:
                    c_lo = max(c_lo, cj + 1)

            if r_lo > r_hi or c_lo > c_hi:
                return []

            out: list[tuple[int, int]] = []
            for r in range(r_lo, r_hi + 1):
                for c in range(c_lo, c_hi + 1):
                    if can_place(i, r, c):
                        out.append((r, c))
            return out

        # Ordered-layout anchor: the corner piece is the unique extremal rank pair.
        target_rr = min_rr if corner in ("TL", "TR") else max_rr
        target_cc = min_cc if corner in ("TL", "BL") else max_cc
        anchor_candidates = [i for i in range(len(pieces)) if row_rank[i] == target_rr and col_rank[i] == target_cc]
        assert len(anchor_candidates) == 1
        anchor_idx = anchor_candidates[0]
        ah, aw = ph[anchor_idx], pw[anchor_idx]
        if corner == "TL":
            ar0, ac0 = 0, 0
        elif corner == "TR":
            ar0, ac0 = 0, W - aw
        elif corner == "BL":
            ar0, ac0 = H - ah, 0
        else:
            ar0, ac0 = H - ah, W - aw
        if not can_place(anchor_idx, ar0, ac0):
            return None
        place(anchor_idx, ar0, ac0)

        priority = sorted(
            range(len(pieces)),
            key=lambda i: (
                piece_layer[i],
                -border_degree[i],
                row_rank[i],
                col_rank[i],
                -(ph[i] * pw[i]),
                -ph[i],
                -pw[i],
                i,
            ),
        )

        solved = False

        def bt() -> None:
            nonlocal solved
            if all(usedp):
                solved = True
                return

            current_layer = min(piece_layer[i] for i in range(len(pieces)) if not usedp[i])
            choice_i = None
            choice_pos: list[tuple[int, int]] | None = None

            for i in priority:
                if usedp[i] or piece_layer[i] != current_layer:
                    continue
                pos = candidate_positions(i)
                if not pos:
                    return
                if choice_pos is None or len(pos) < len(choice_pos):
                    choice_i = i
                    choice_pos = pos
                if choice_pos is not None and len(choice_pos) == 1:
                    break

            assert choice_i is not None and choice_pos is not None
            for r, c in choice_pos:
                place(choice_i, r, c)
                bt()
                if solved:
                    return
                unplace(choice_i, r, c)

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


def _rand_splits(
    total: int,
    min_s: int = 2,
    max_s: int = 6,
    min_parts: int = 3,
    max_parts: int = 4,
) -> list[int] | None:
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
    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    n_interior = random.randint(1, 3)
    interior_cols = palette[3 : 3 + n_interior]

    RH = random.randint(7, 14)
    RW = random.randint(7, 14)
    rect = np.full((RH, RW), rect_col, dtype=int)
    iH, iW = RH - 2, RW - 2
    if iH < 1 or iW < 1:
        return None

    interior = np.full((iH, iW), rect_col, dtype=int)
    target_fill = random.uniform(0.38, 0.82)
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

    row_segs = _rand_splits(RH, min_parts=3, max_parts=4)
    col_segs = _rand_splits(RW, min_parts=3, max_parts=4)
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

    def single_ring_bbox(r0i: int, c0i: int, r1i: int, c1i: int) -> bool:
        ring = _cell_ring_index(r0i, c0i, r1i, c1i, nr, nc_grid)
        for ii in range(r0i, r1i + 1):
            for jj in range(c0i, c1i + 1):
                if _cell_ring_index(ii, jj, ii, jj, nr, nc_grid) != ring:
                    return False
        return True

    def count_pieces() -> int:
        return len(set(cell_id[i][j] for i in range(nr) for j in range(nc_grid)))

    target_pieces = random.randint(max(5, n_cells // 2), min(10, n_cells))
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
        if not single_ring_bbox(mr0, mc0, mr1, mc1):
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
    latent_layers = []
    cross_layer_piece_count = 0
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
            latent_layers.append(_cell_ring_index(r0i, c0i, r1i, c1i, nr, nc_grid))
            if not single_ring_bbox(r0i, c0i, r1i, c1i):
                cross_layer_piece_count += 1

    if len(pieces) < 5 or len(pieces) > 12:
        return None

    top_rows = sorted({pr for pr, _pc, _ph, _pw in pieces})
    left_cols = sorted({pc for _pr, pc, _ph, _pw in pieces})
    if len(top_rows) < 3 or len(left_cols) < 3:
        return None
    if set(top_rows) != set(row_starts) or set(left_cols) != set(col_starts):
        return None

    row_rank_map = {v: i for i, v in enumerate(top_rows)}
    col_rank_map = {v: i for i, v in enumerate(left_cols)}
    layers = [
        min(
            row_rank_map[pr],
            col_rank_map[pc],
            len(top_rows) - 1 - row_rank_map[pr],
            len(left_cols) - 1 - col_rank_map[pc],
        )
        for pr, pc, _ph, _pw in pieces
    ]
    if any(int(a) != int(b) for a, b in zip(layers, latent_layers)):
        return None
    if max(layers) < 1:
        return None
    if sum(int(layer == 0) for layer in layers) < 4:
        return None
    if cross_layer_piece_count != 0:
        return None

    for pr, pc, ph, pw in pieces:
        if ph < 2 or pw < 2 or ph > 7 or pw > 7:
            return None
        piece = rect[pr : pr + ph, pc : pc + pw]
        fill = int(np.sum(piece != rect_col)) / (ph * pw)
        if fill < 0.10 or fill > 0.60:
            return None
        has_non_rc_edge = any(
            np.any(edge != rect_col)
            for edge in [piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]]
        )
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

    gap = 1
    row_heights = [0] * len(top_rows)
    col_widths = [0] * len(left_cols)
    for pr, pc, ph, pw in pieces:
        row_heights[row_rank_map[pr]] = max(row_heights[row_rank_map[pr]], ph)
        col_widths[col_rank_map[pc]] = max(col_widths[col_rank_map[pc]], pw)
    AH = sum(row_heights) + gap * (len(row_heights) - 1)
    AW = sum(col_widths) + gap * (len(col_widths) - 1)

    baseH = max(12, max(AH, RH) + 5)
    baseW = max(12, max(AW, RW) + 5)
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
        rr = row_pos[row_rank_map[pr]]
        cc = col_pos[col_rank_map[pc]]
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
            m_cells = _marker_cells(corner, mr, mc)
            marker_box = (
                min(r for r, _c in m_cells),
                min(c for _r, c in m_cells),
                max(r for r, _c in m_cells),
                max(c for _r, c in m_cells),
            )
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
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 600 attempts")
