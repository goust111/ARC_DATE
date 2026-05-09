# Variant of 446ef5d2
# Dimension: D6 — 输出网格形状 (grid_shape)
# Change: 输出与输入网格尺寸相同，组装矩形放置在标记角对应位置，其余区域填充背景色
#   -> 碎片打包排列输出
#
# Design analysis
# ===============
# Reference baseline:
# - I read the original solver at `/home/lixinyu/snap/new/arc2_solver_select/446ef5d2.py`.
# - I also read the packed-grid source solver at
#   `/home/lixinyu/snap/new_code/solvers/arc2/58f5dbd5.py`.
# - The reusable idea from 58f5dbd5 is not its object logic but its output layout:
#   preserve row/column order, insert 1-cell gaps, and add a 1-cell outer bg border.
#
# a) Concretization and behavior trace
# - Input stays the same as the original task: scattered rectangular puzzle pieces plus a
#   3-cell L-marker whose missing cell indicates the rectangle corner.
# - The solver still has to infer the assembled rectangle by edge-color compatibility with
#   `rect_col` as wildcard.
# - Original output:
#   reassemble the whole rectangle and place it back into the input-sized canvas at the
#   marker corner.
# - Variant output:
#   do not paste the solved rectangle back into the original location. Instead, once the
#   solver knows each piece's position `(r, c)` inside the solved rectangle, collect the
#   pieces into a new compact grid:
#   - group pieces by solved top row and solved left column,
#   - keep those row/column orders,
#   - assign each row a height equal to the largest piece starting on that row,
#   - assign each column a width equal to the largest piece starting on that column,
#   - place every piece at the top-left of its packed cell,
#   - leave 1 bg cell between packed rows/columns and add a 1-cell bg border.
# - Small trace:
#   if the solved rectangle contains pieces at solved top-lefts
#   `(0,0),(0,4),(3,0),(3,2)`, the original task would output one large merged rectangle.
#   The variant instead makes packed rows `{0,3}` and packed cols `{0,2,4}`, then lays
#   the four original piece sprites into that packed table with 1-cell bg gutters.
#
# b) Complexity calibration
# - I keep the original matching logic, piece count, color logic, and no-rotation rule.
# - The new rule is visually stronger than a mere relocation because the output shape now
#   changes and the rectangle is visibly not re-merged.
# - To avoid degenerate samples where the packed layout collapses into a single row or
#   single column and becomes too close to the original task, the generator requires at
#   least 2 distinct packed row groups and 2 distinct packed column groups.
#
# c) Hidden-assumption audit of the original `main()`
# - Marker detection depends on a non-bg color appearing exactly 3 times in a 2x2 L.
#   This remains valid.
# - Rectangle extraction depends on every scattered piece being an axis-aligned filled
#   rectangle with at least 1 bg cell around it. This remains valid.
# - Rectangle-dimension search depends on the true solved rectangle fitting the original
#   input grid at the marker corner. This remains valid because the input construction is
#   unchanged.
# - The original final lines assumed the solved rectangle itself should be written into an
#   input-sized canvas. That assumption is exactly what D6 changes; those lines must be
#   replaced by packed-layout rendering.
# - The new packed renderer assumes that solved top-left rows/cols define the ordering
#   classes. The generator must therefore build the target output from the same solved
#   coordinates, not from scattered coordinates or color ids.
#
# d) Placement legality
# - Input generation still scatters pieces onto a larger bg canvas with `padding=1`, so
#   scattered pieces never touch and the rectangle extraction logic is safe.
# - The packed output uses deterministic row/column offsets derived from packed row heights
#   and packed column widths, so pieces never overlap and always stay in-bounds.
#
# e) Visual observability
# - The output shape changes from "same as input" to "compact packed table", which is
#   obvious across 2-3 examples.
# - The gaps and outer border are visible structure cues. Humans can see that the pieces
#   are shown separately, in solved order, rather than merged into one rectangle.
#
# f) Visual cleanliness
# - No overlays are introduced: pieces are copied intact into disjoint packed cells over a
#   plain bg canvas.
# - The marker disappears from the output, as in the original task; only the packed pieces
#   remain.
#
# g) Color inferability
# - No new color remapping is added. Colors keep the original semantics: `rect_col`
#   remains the wildcard border color inside the reasoning process, and packed output
#   simply reuses piece pixels verbatim.
#
# h) Variant-specific metric used after testing
# - `shape_changed = output.shape != input.shape`
# - `packed_gap_count = number of all-bg rows or cols separating occupied packed groups`
# - The generator is rejected unless packed row groups >= 2, packed col groups >= 2, and
#   `shape_changed` is true, ensuring the D6 behavior is actually exercised.
#
# i) New implicit assumptions after modification
# - `_pack_piece_layout` assumes each packed row/column key comes from solved piece
#   top-lefts; generator guarantees this because it derives expected outputs from the same
#   solved piece coordinates used before scattering.
# - The packed layout may contain empty meta-cells when a piece spans multiple latent split
#   bands in the assembled rectangle. That is intentional and visually acceptable because
#   the rule is "show correct row/column order without merging", not "reconstruct exact
#   adjacency lengths".

from __future__ import annotations

from collections import Counter
import numpy as np

from arc_utils import paste, random_free_location


def _find_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], str]:
    cnt = Counter(map(int, g.flatten()))
    mark = None
    missing = None
    mb = None
    for col in sorted([c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3]):
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
                missing = (int(miss[0][0]), int(miss[0][1]))
                mb = (int(r0), int(c0), int(r1), int(c1))
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
    return int(mark), missing, corner


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


def _pack_piece_layout(
    placements: list[tuple[int, int]],
    pieces: list[np.ndarray],
    bg: int,
    *,
    gap: int = 1,
    border: int = 1,
) -> np.ndarray:
    row_keys = sorted({int(r) for r, _c in placements})
    col_keys = sorted({int(c) for _r, c in placements})
    assert row_keys and col_keys

    row_heights = {
        int(rk): max(int(p.shape[0]) for (r, _c), p in zip(placements, pieces) if int(r) == int(rk))
        for rk in row_keys
    }
    col_widths = {
        int(ck): max(int(p.shape[1]) for (_r, c), p in zip(placements, pieces) if int(c) == int(ck))
        for ck in col_keys
    }

    row_offsets: dict[int, int] = {}
    cur = int(border)
    for rk in row_keys:
        row_offsets[int(rk)] = int(cur)
        cur += int(row_heights[int(rk)]) + int(gap)
    out_h = int(cur - gap + border)

    col_offsets: dict[int, int] = {}
    cur = int(border)
    for ck in col_keys:
        col_offsets[int(ck)] = int(cur)
        cur += int(col_widths[int(ck)]) + int(gap)
    out_w = int(cur - gap + border)

    out = np.full((out_h, out_w), int(bg), dtype=int)
    for (r, c), piece in zip(placements, pieces):
        rr = int(row_offsets[int(r)])
        cc = int(col_offsets[int(c)])
        ph, pw = map(int, piece.shape)
        assert not bool((out[rr : rr + ph, cc : cc + pw] != int(bg)).any())
        out[rr : rr + ph, cc : cc + pw] = piece
    return out


def _solve_piece_layout(grid: np.ndarray) -> tuple[int, list[np.ndarray], list[tuple[int, int]]]:
    g = np.array(grid, dtype=int)
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    mark, missing, corner = _find_marker(g, bg)
    rects = _extract_rects(g, bg, mark)
    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))

    allpix: list[int] = []
    for _r, _c, _h, _w, a in rects:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    mr, mc = map(int, missing)

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

    def compat(v1: np.ndarray, v2: np.ndarray) -> bool:
        a1 = np.array(v1, dtype=int)
        a2 = np.array(v2, dtype=int)
        m = (a1 != int(rect_col)) & (a2 != int(rect_col))
        return bool((a1[m] == a2[m]).all())

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
        assert all(p is not None for p in placements)
        final_places = []
        for pos in placements:
            assert pos is not None
            final_places.append((int(pos[0]), int(pos[1])))
        return occ2, final_places

    solved_grid = None
    solved_places = None
    for H, W in dims:
        assembled = assemble(int(H), int(W))
        if assembled is not None:
            solved_grid, solved_places = assembled
            break

    assert not (solved_grid is None or solved_places is None)
    return int(bg), pieces, solved_places


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered rectangles logically, then pack the solved pieces into a new grid.

    The reasoning stage is unchanged from the original task: infer the rectangle using
    edge-color matching with `rect_col` as wildcard and the L-marker as corner hint.
    The output stage is the D6 variant: return a compact packed grid of the solved pieces
    in row/column order, with 1-cell gaps and a 1-cell outer bg border.
    """
    bg, pieces, placements = _solve_piece_layout(grid)
    return _pack_piece_layout(placements, pieces, bg, gap=1, border=1)


def _gen_attempt() -> np.ndarray | None:
    import random as _r

    GH = _r.randint(10, 26)
    GW = _r.randint(10, 26)

    palette = list(range(10))
    _r.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    n_interior = _r.randint(1, 3)
    interior_cols = palette[3 : 3 + n_interior]

    corner = _r.choice(["TL", "TR", "BL", "BR"])

    min_side, max_side = 5, min(22, min(GH, GW) - 3)
    if max_side < min_side:
        return None
    RH = _r.randint(min_side, max_side)
    RW = _r.randint(min_side, max_side)
    if _r.random() < 0.5 and max_side >= 8:
        RH = _r.randint(max(6, min_side), max_side)
        RW = _r.randint(max(6, min_side), max_side)

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
    fill_ratio = non_rc_count / (iH * iW) if iH * iW > 0 else 0
    if fill_ratio < 0.30 or fill_ratio > 0.90:
        return None

    rect[1:-1, 1:-1] = interior

    def _rand_splits(total: int, min_s: int = 2, max_s: int = 7) -> list[int] | None:
        eff_max = min(max_s, max(min_s, _r.randint(min_s, max_s)))
        segs = []
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
    if len(row_segs) < 2 or len(col_segs) < 2:
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

    def _get_piece_bbox(pid: int) -> tuple[int, int, int, int] | None:
        cells = [(i, j) for i in range(nr) for j in range(nc_grid) if cell_id[i][j] == pid]
        if not cells:
            return None
        ri = [c[0] for c in cells]
        ci = [c[1] for c in cells]
        return min(ri), min(ci), max(ri), max(ci)

    def _pixel_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
        h = sum(row_segs[i] for i in range(r0i, r1i + 1))
        w = sum(col_segs[j] for j in range(c0i, c1i + 1))
        return h, w

    if n_cells < 4:
        return None
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

    pieces = []
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

    row_keys = sorted({int(pr) for pr, _pc, _ph, _pw in pieces})
    col_keys = sorted({int(pc) for _pr, pc, _ph, _pw in pieces})
    if len(row_keys) < 2 or len(col_keys) < 2:
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

    canvas = np.full((GH, GW), bg, dtype=int)
    for cr, cc in marker_cells:
        canvas[cr, cc] = marker_col

    piece_arrays = [rect[pr : pr + ph, pc : pc + pw].copy() for pr, pc, ph, pw in pieces]

    order = list(range(len(piece_arrays)))
    _r.shuffle(order)
    for idx in order:
        pa = piece_arrays[idx]
        try:
            r0, c0 = random_free_location(canvas, pa, transparent=bg, padding=1)
            paste(canvas, pa, r0, c0)
        except ValueError:
            return None

    placements = [(int(pr), int(pc)) for pr, pc, _ph, _pw in pieces]
    expected_out = _pack_piece_layout(placements, piece_arrays, bg, gap=1, border=1)
    if expected_out.shape == canvas.shape:
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
