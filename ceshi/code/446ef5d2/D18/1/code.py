# Variant of 446ef5d2
# Dimension: D18 — 碎片切割方式 (meta)
# Change: 矩形被切割成网格状（行列划分后可合并相邻格），所有碎片都是矩形且恰好拼成完整矩形... → 重叠切割（碎片有共享像素）

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import paste, random_free_location


# Design analysis for the D18 overlap-cut variant
# ------------------------------------------------
# spec_delta:
#   what_changes:
#     - Pieces are still scattered as solid rectangles in the input.
#     - In the hidden assembled rectangle, adjacent pieces no longer meet with a gapless
#       edge-to-edge tiling. Instead, they share exactly one boundary row or boundary
#       column. The shared strip must be pixel-identical because it represents the same
#       pixels duplicated into both pieces.
#     - Therefore sum(piece areas) > assembled rectangle area whenever there is at least
#       one internal cut.
#   concrete_pixel_model:
#     - Use closed intervals for the cut grid: if column coordinates are [0, 3, 6], then
#       the left cell is cols 0..3 and the right cell is cols 3..6. They share col 3.
#     - Merged cells remain rectangles because a merged piece is just the bbox spanning
#       several consecutive closed intervals.
#   small_trace:
#     - Old task: two 3x4 pieces placed side by side use offsets (0, 0) and (0, 4),
#       total area 12 + 12 = 24, final rectangle area also 24.
#     - New task: the same visible content is cut with a 1-column overlap, so the second
#       piece sits at (0, 3). The shared column must match exactly. Total piece area is
#       still 24, but the final rectangle is only 3x7 = 21.
#     - This breaks the old solver assumption "factor pairs of total piece area are the
#       rectangle dimensions" and also breaks its "placement cells cannot overlap" rule.
#   complexity_calibration:
#     - Keep 4-10 pieces and piece side lengths 2-7 so the task still looks like the
#       original family, but require every internal shared strip to contain at least one
#       non-rect_col pixel so the overlap mechanism is visible and informative.
#     - Restrict overlaps to width/height exactly 1, never thicker, to keep the rule
#       simple and visually legible.
#   implicit_assumption_audit:
#     - Original `total_area = sum(h*w)` is no longer the final rectangle area.
#     - Original `can_place` rejected any overlap; this must become "overlap allowed iff
#       colors agree exactly".
#     - Original neighbor checks only compared touching outer edges with rect_col as a
#       wildcard. Here the duplicated strip is a real shared region, so exact equality is
#       the correct invariant.
#     - Original dimension search iterated factor pairs of area. Here dimensions are
#       recovered from the bbox of the placed pieces after overlap-aware backtracking.
#   legality_and_cleanliness:
#     - Generator still scatters pieces with >= 1 background cell gap, so input objects
#       remain separable by rectangle extraction.
#     - The marker logic is unchanged: the missing cell still pins one corner of the
#       assembled rectangle.
#     - All outer rectangle borders stay rect_col, so the original border cue remains.
#   observability:
#     - The overlap is observable because several scattered pieces show colored pixels on
#       their outermost rows/cols that later coincide in the output instead of merely
#       touching. With 2-4 examples, a human can see that duplicated colored edge strips
#       collapse into one strip in the assembled rectangle.
#   new_solver_assumptions:
#     - Every true adjacency is a 1-row or 1-column overlap.
#     - The corner piece exists and contains the marker-indicated rectangle corner at the
#       matching corner of that piece.
#     - Every internal shared strip has at least one non-rect_col pixel, reducing false
#       pairings; the generator enforces this.


def _find_marker(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int], str]:
    cnt = Counter(map(int, grid.flatten()))
    for col in sorted(c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3):
        pts = [(int(r), int(c)) for r, c in np.argwhere(grid == int(col))]
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


def _extract_rectangles(grid: np.ndarray, bg: int, marker: int) -> list[tuple[int, int, int, int, np.ndarray]]:
    occ = (grid != int(bg)) & (grid != int(marker))
    used = np.zeros_like(occ, dtype=bool)
    rects: list[tuple[int, int, int, int, np.ndarray]] = []
    gh, gw = grid.shape
    while True:
        idx = np.argwhere(occ & (~used))
        if idx.size == 0:
            break
        rr, cc = map(int, idx[0])
        w = 0
        while cc + w < gw and bool(occ[rr, cc + w]) and (not bool(used[rr, cc + w])):
            w += 1
        h = 0
        while rr + h < gh and bool((occ[rr + h, cc : cc + w] & (~used[rr + h, cc : cc + w])).all()):
            h += 1
        used[rr : rr + h, cc : cc + w] = True
        rects.append((rr, cc, h, w, grid[rr : rr + h, cc : cc + w].copy()))
    return rects


def _all_piece_pixels(rects: list[tuple[int, int, int, int, np.ndarray]]) -> list[int]:
    vals: list[int] = []
    for _r, _c, _h, _w, arr in rects:
        vals.extend(map(int, arr.flatten()))
    return vals


def _build_relations(pieces: list[np.ndarray]) -> list[list[list[tuple[int, int]]]]:
    n = len(pieces)
    rel = [[[] for _ in range(n)] for _ in range(n)]
    for i, a in enumerate(pieces):
        hi, wi = map(int, a.shape)
        for j, b in enumerate(pieces):
            if i == j:
                continue
            hj, wj = map(int, b.shape)
            offsets: set[tuple[int, int]] = set()

            dc = wi - 1
            for dr in range(-(hj - 1), hi):
                top = max(0, dr)
                bottom = min(hi, dr + hj)
                if bottom <= top:
                    continue
                bj0 = top - dr
                if np.array_equal(a[top:bottom, wi - 1], b[bj0 : bj0 + (bottom - top), 0]):
                    offsets.add((int(dr), int(dc)))

            dc = -(wj - 1)
            for dr in range(-(hj - 1), hi):
                top = max(0, dr)
                bottom = min(hi, dr + hj)
                if bottom <= top:
                    continue
                bj0 = top - dr
                if np.array_equal(a[top:bottom, 0], b[bj0 : bj0 + (bottom - top), wj - 1]):
                    offsets.add((int(dr), int(dc)))

            dr = hi - 1
            for dc2 in range(-(wj - 1), wi):
                left = max(0, dc2)
                right = min(wi, dc2 + wj)
                if right <= left:
                    continue
                bj0 = left - dc2
                if np.array_equal(a[hi - 1, left:right], b[0, bj0 : bj0 + (right - left)]):
                    offsets.add((int(dr), int(dc2)))

            dr = -(hj - 1)
            for dc2 in range(-(wj - 1), wi):
                left = max(0, dc2)
                right = min(wi, dc2 + wj)
                if right <= left:
                    continue
                bj0 = left - dc2
                if np.array_equal(a[0, left:right], b[hj - 1, bj0 : bj0 + (right - left)]):
                    offsets.add((int(dr), int(dc2)))

            rel[i][j] = sorted(offsets)
    return rel


def _anchor_position(corner: str, mr: int, mc: int, h: int, w: int) -> tuple[int, int]:
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - w + 1
    if corner == "BL":
        return mr - h + 1, mc
    return mr - h + 1, mc - w + 1


def _anchor_ok(piece: np.ndarray, rect_col: int, corner: str) -> bool:
    if corner == "TL":
        return bool(np.all(piece[0, :] == rect_col) and np.all(piece[:, 0] == rect_col) and piece[0, 0] == rect_col)
    if corner == "TR":
        return bool(np.all(piece[0, :] == rect_col) and np.all(piece[:, -1] == rect_col) and piece[0, -1] == rect_col)
    if corner == "BL":
        return bool(np.all(piece[-1, :] == rect_col) and np.all(piece[:, 0] == rect_col) and piece[-1, 0] == rect_col)
    return bool(np.all(piece[-1, :] == rect_col) and np.all(piece[:, -1] == rect_col) and piece[-1, -1] == rect_col)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    gh, gw = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    marker, (mr, mc), corner = _find_marker(g, bg)
    rects = _extract_rectangles(g, bg, marker)
    assert rects

    pieces = [arr for _r, _c, _h, _w, arr in rects]
    rect_col = int(Counter(_all_piece_pixels(rects)).most_common(1)[0][0])
    rel = _build_relations(pieces)
    n = len(pieces)
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]

    occ = np.full((gh, gw), -1, dtype=int)
    cover = np.zeros((gh, gw), dtype=int)
    used = [False] * n
    pos: dict[int, tuple[int, int]] = {}

    def directional_ok(r: int, c: int, h: int, w: int) -> bool:
        if r < 0 or c < 0 or r + h > gh or c + w > gw:
            return False
        if corner == "TL":
            return r >= mr and c >= mc
        if corner == "TR":
            return r >= mr and (c + w - 1) <= mc
        if corner == "BL":
            return (r + h - 1) <= mr and c >= mc
        return (r + h - 1) <= mr and (c + w - 1) <= mc

    def can_place(i: int, r: int, c: int) -> bool:
        a = pieces[i]
        h, w = a.shape
        if not directional_ok(int(r), int(c), int(h), int(w)):
            return False
        view = occ[r : r + h, c : c + w]
        return not bool(((view != -1) & (view != a)).any())

    def place(i: int, r: int, c: int) -> None:
        a = pieces[i]
        h, w = a.shape
        occ_view = occ[r : r + h, c : c + w]
        cov_view = cover[r : r + h, c : c + w]
        occ_view[cov_view == 0] = a[cov_view == 0]
        cov_view[:] = cov_view + 1
        used[i] = True
        pos[i] = (int(r), int(c))

    def unplace(i: int) -> None:
        r, c = pos.pop(i)
        h, w = ph[i], pw[i]
        occ_view = occ[r : r + h, c : c + w]
        cov_view = cover[r : r + h, c : c + w]
        cov_view[:] = cov_view - 1
        occ_view[cov_view == 0] = -1
        used[i] = False

    def partial_bbox_ok() -> bool:
        placed = [(idx, pos[idx][0], pos[idx][1], ph[idx], pw[idx]) for idx in pos]
        if not placed:
            return True
        r0 = min(r for _i, r, _c, _h, _w in placed)
        c0 = min(c for _i, _r, c, _h, _w in placed)
        r1 = max(r + h - 1 for _i, r, _c, h, _w in placed)
        c1 = max(c + w - 1 for _i, _r, c, _h, w in placed)
        if corner == "TL" and (r0 != mr or c0 != mc):
            return False
        if corner == "TR" and (r0 != mr or c1 != mc):
            return False
        if corner == "BL" and (r1 != mr or c0 != mc):
            return False
        if corner == "BR" and (r1 != mr or c1 != mc):
            return False
        return (r1 - r0 + 1) <= gh and (c1 - c0 + 1) <= gw

    def final_ok() -> bool:
        mask = cover > 0
        coords = np.argwhere(mask)
        if coords.size == 0:
            return False
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        if corner == "TL" and (r0, c0) != (mr, mc):
            return False
        if corner == "TR" and (r0, c1) != (mr, mc):
            return False
        if corner == "BL" and (r1, c0) != (mr, mc):
            return False
        if corner == "BR" and (r1, c1) != (mr, mc):
            return False
        submask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if not bool(submask.all()):
            return False
        sub = occ[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(sub[0, :] == rect_col):
            return False
        if not np.all(sub[-1, :] == rect_col):
            return False
        if not np.all(sub[:, 0] == rect_col):
            return False
        if not np.all(sub[:, -1] == rect_col):
            return False
        return True

    def frontier_options() -> tuple[int | None, list[tuple[int, int]]]:
        best_i = None
        best_positions: list[tuple[int, int]] = []
        for j in range(n):
            if used[j]:
                continue
            options: set[tuple[int, int]] = set()
            for i, (ri, ci) in pos.items():
                for dr, dc in rel[i][j]:
                    rr = int(ri + dr)
                    cc = int(ci + dc)
                    if can_place(j, rr, cc):
                        options.add((rr, cc))
            if not options:
                continue
            ordered = sorted(
                options,
                key=lambda rc: (
                    -sum(
                        max(0, min(rc[0] + ph[j], pos[k][0] + ph[k]) - max(rc[0], pos[k][0]))
                        * max(0, min(rc[1] + pw[j], pos[k][1] + pw[k]) - max(rc[1], pos[k][1]))
                        for k in pos
                    ),
                    rc[0],
                    rc[1],
                ),
            )
            if best_i is None or len(ordered) < len(best_positions):
                best_i = j
                best_positions = ordered
        return best_i, best_positions

    solved = False
    out = np.full((gh, gw), bg, dtype=int)

    def bt() -> None:
        nonlocal solved, out
        if solved:
            return
        if all(used):
            if final_ok():
                out = np.full((gh, gw), bg, dtype=int)
                mask = cover > 0
                out[mask] = occ[mask]
                solved = True
            return
        j, candidates = frontier_options()
        if j is None:
            return
        for r, c in candidates:
            place(j, r, c)
            if partial_bbox_ok():
                bt()
            if solved:
                return
            unplace(j)

    anchors = [i for i, p in enumerate(pieces) if _anchor_ok(p, rect_col, corner)]
    anchors = sorted(anchors, key=lambda i: (-ph[i] * pw[i], -ph[i], -pw[i], i))
    assert anchors
    for anchor in anchors:
        ar, ac = _anchor_position(corner, mr, mc, ph[anchor], pw[anchor])
        if not can_place(anchor, ar, ac):
            continue
        place(anchor, ar, ac)
        if partial_bbox_ok():
            bt()
        if solved:
            break
        unplace(anchor)

    assert solved
    return out


def _rand_diffs(total: int, rng, min_d: int = 2, max_d: int = 4) -> list[int] | None:
    diffs: list[int] = []
    rem = int(total)
    while rem > 0:
        if min_d <= rem <= max_d:
            diffs.append(rem)
            rem = 0
            continue
        hi = min(max_d, rem - min_d)
        if hi < min_d:
            return None
        d = rng.randint(min_d, hi)
        diffs.append(int(d))
        rem -= int(d)
    return diffs


def _piece_bbox_from_cells(row_coords: list[int], col_coords: list[int], r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int, int, int]:
    pr = int(row_coords[r0i])
    pc = int(col_coords[c0i])
    ph = int(row_coords[r1i + 1] - row_coords[r0i] + 1)
    pw = int(col_coords[c1i + 1] - col_coords[c0i] + 1)
    return pr, pc, ph, pw


def _collect_shared_strips(
    rect: np.ndarray,
    pieces: list[tuple[int, int, int, int, tuple[int, int, int, int]]],
    rect_col: int,
) -> list[tuple[str, tuple[int, int, int]]]:
    strips: list[tuple[str, tuple[int, int, int]]] = []
    n = len(pieces)
    for i in range(n):
        pr, pc, ph, pw, _bboxi = pieces[i]
        for j in range(i + 1, n):
            qr, qc, qh, qw, _bboxj = pieces[j]
            if pc + pw - 1 == qc or qc + qw - 1 == pc:
                top = max(pr, qr)
                bot = min(pr + ph, qr + qh)
                if bot - top >= 2:
                    col = qc if pc + pw - 1 == qc else pc
                    vals = rect[top:bot, col]
                    if np.any(vals != rect_col):
                        strips.append(("v", (top, bot, int(col))))
            if pr + ph - 1 == qr or qr + qh - 1 == pr:
                left = max(pc, qc)
                right = min(pc + pw, qc + qw)
                if right - left >= 2:
                    row = qr if pr + ph - 1 == qr else pr
                    vals = rect[row, left:right]
                    if np.any(vals != rect_col):
                        strips.append(("h", (left, right, int(row))))
    return strips


def _force_overlap_signal(
    rect: np.ndarray,
    pieces: list[tuple[int, int, int, int, tuple[int, int, int, int]]],
    rect_col: int,
    interior_cols: list[int],
    rng,
) -> None:
    if not interior_cols:
        return
    n = len(pieces)
    for i in range(n):
        pr, pc, ph, pw, _bboxi = pieces[i]
        for j in range(i + 1, n):
            qr, qc, qh, qw, _bboxj = pieces[j]
            if pc + pw - 1 == qc or qc + qw - 1 == pc:
                top = max(pr, qr)
                bot = min(pr + ph, qr + qh)
                if bot - top >= 2:
                    col = qc if pc + pw - 1 == qc else pc
                    if np.all(rect[top:bot, col] == rect_col):
                        length = bot - top
                        run = min(length, rng.randint(2, min(4, length)))
                        start = rng.randint(top, bot - run)
                        rect[start : start + run, col] = interior_cols[rng.randint(0, len(interior_cols) - 1)]
            if pr + ph - 1 == qr or qr + qh - 1 == pr:
                left = max(pc, qc)
                right = min(pc + pw, qc + qw)
                if right - left >= 2:
                    row = qr if pr + ph - 1 == qr else pr
                    if np.all(rect[row, left:right] == rect_col):
                        length = right - left
                        run = min(length, rng.randint(2, min(4, length)))
                        start = rng.randint(left, right - run)
                        rect[row, start : start + run] = interior_cols[rng.randint(0, len(interior_cols) - 1)]


def _gen_attempt() -> np.ndarray | None:
    import random as _r

    gh = _r.randint(12, 26)
    gw = _r.randint(12, 26)

    palette = list(range(10))
    _r.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    n_interior = _r.randint(1, 3)
    interior_cols = palette[3 : 3 + n_interior]

    corner = _r.choice(["TL", "TR", "BL", "BR"])

    min_side, max_side = 8, min(20, min(gh, gw) - 3)
    if max_side < min_side:
        return None
    rh = _r.randint(min_side, max_side)
    rw = _r.randint(min_side, max_side)

    if corner == "TL":
        mr_min, mr_max = 2, gh - rh
        mc_min, mc_max = 2, gw - rw
    elif corner == "TR":
        mr_min, mr_max = 2, gh - rh
        mc_min, mc_max = rw - 1, gw - 3
    elif corner == "BL":
        mr_min, mr_max = rh - 1, gh - 3
        mc_min, mc_max = 2, gw - rw
    else:
        mr_min, mr_max = rh - 1, gh - 3
        mc_min, mc_max = rw - 1, gw - 3
    if mr_min > mr_max or mc_min > mc_max:
        return None
    mr = _r.randint(mr_min, mr_max)
    mc = _r.randint(mc_min, mc_max)

    rect = np.full((rh, rw), rect_col, dtype=int)
    ih, iw = rh - 2, rw - 2
    if ih < 2 or iw < 2:
        return None
    interior = np.full((ih, iw), rect_col, dtype=int)

    target_fill = _r.uniform(0.35, 0.78)
    target_pixels = max(1, int(target_fill * ih * iw))
    blob_occupied = np.zeros((ih, iw), dtype=bool)

    for ci, ic in enumerate(interior_cols):
        if ci == len(interior_cols) - 1:
            budget = target_pixels - int(blob_occupied.sum())
        else:
            budget = max(1, target_pixels // len(interior_cols))
        budget = max(1, budget)

        attempts = 0
        seed = None
        while attempts < 200:
            sr, sc = _r.randint(0, ih - 1), _r.randint(0, iw - 1)
            if blob_occupied[sr, sc]:
                attempts += 1
                continue
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < ih and 0 <= nc < iw and blob_occupied[nr, nc]:
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
            new_frontier = []
            for fr, fc in frontier:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = fr + dr, fc + dc
                    if (nr, nc) in blob or nr < 0 or nr >= ih or nc < 0 or nc >= iw or blob_occupied[nr, nc]:
                        continue
                    conflict = False
                    for dr2 in (-1, 0, 1):
                        for dc2 in (-1, 0, 1):
                            if dr2 == 0 and dc2 == 0:
                                continue
                            ar, ac = nr + dr2, nc + dc2
                            if 0 <= ar < ih and 0 <= ac < iw and blob_occupied[ar, ac] and (ar, ac) not in blob:
                                conflict = True
                                break
                        if conflict:
                            break
                    if conflict:
                        continue
                    blob.add((nr, nc))
                    new_frontier.append((nr, nc))
                    if len(blob) >= budget:
                        break
                if len(blob) >= budget:
                    break
            frontier = new_frontier

        for br, bc in blob:
            interior[br, bc] = ic
            blob_occupied[br, bc] = True

    fill_ratio = float(np.mean(interior != rect_col))
    if fill_ratio < 0.28 or fill_ratio > 0.85:
        return None
    rect[1:-1, 1:-1] = interior

    row_diffs = _rand_diffs(rh - 1, _r)
    col_diffs = _rand_diffs(rw - 1, _r)
    if row_diffs is None or col_diffs is None:
        return None

    row_coords = [0]
    for d in row_diffs:
        row_coords.append(row_coords[-1] + int(d))
    col_coords = [0]
    for d in col_diffs:
        col_coords.append(col_coords[-1] + int(d))
    nr = len(row_coords) - 1
    nc = len(col_coords) - 1
    if nr * nc < 4:
        return None

    cell_id = [[i * nc + j for j in range(nc)] for i in range(nr)]
    target_pieces = _r.randint(4, min(10, nr * nc))

    def get_bbox(pid: int) -> tuple[int, int, int, int] | None:
        cells = [(i, j) for i in range(nr) for j in range(nc) if cell_id[i][j] == pid]
        if not cells:
            return None
        rs = [x for x, _ in cells]
        cs = [y for _, y in cells]
        return min(rs), min(cs), max(rs), max(cs)

    def piece_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
        ph = row_coords[r1i + 1] - row_coords[r0i] + 1
        pw = col_coords[c1i + 1] - col_coords[c0i] + 1
        return int(ph), int(pw)

    def count_pieces() -> int:
        return len({cell_id[i][j] for i in range(nr) for j in range(nc)})

    merges = []
    for i in range(nr):
        for j in range(nc):
            if j + 1 < nc:
                merges.append(("h", i, j))
            if i + 1 < nr:
                merges.append(("v", i, j))
    _r.shuffle(merges)

    for direction, mi, mj in merges:
        if count_pieces() <= target_pieces:
            break
        pid1 = cell_id[mi][mj]
        pid2 = cell_id[mi][mj + 1] if direction == "h" else cell_id[mi + 1][mj]
        if pid1 == pid2:
            continue
        bb1 = get_bbox(pid1)
        bb2 = get_bbox(pid2)
        if bb1 is None or bb2 is None:
            continue
        mr0 = min(bb1[0], bb2[0])
        mc0 = min(bb1[1], bb2[1])
        mr1 = max(bb1[2], bb2[2])
        mc1 = max(bb1[3], bb2[3])
        ph, pw = piece_size(mr0, mc0, mr1, mc1)
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
            for jj in range(nc):
                if cell_id[ii][jj] == pid2:
                    cell_id[ii][jj] = pid1

    pieces_meta: list[tuple[int, int, int, int, tuple[int, int, int, int]]] = []
    seen = set()
    for i in range(nr):
        for j in range(nc):
            pid = cell_id[i][j]
            if pid in seen:
                continue
            seen.add(pid)
            bb = get_bbox(pid)
            assert bb is not None
            pr, pc, ph, pw = _piece_bbox_from_cells(row_coords, col_coords, bb[0], bb[1], bb[2], bb[3])
            pieces_meta.append((pr, pc, ph, pw, bb))

    if len(pieces_meta) < 4 or len(pieces_meta) > 10:
        return None

    _force_overlap_signal(rect, pieces_meta, rect_col, interior_cols, _r)
    shared_strips = _collect_shared_strips(rect, pieces_meta, rect_col)
    if len(shared_strips) < max(3, len(pieces_meta) - 1):
        return None

    piece_arrays = []
    for pr, pc, ph, pw, _bb in pieces_meta:
        if ph < 2 or pw < 2 or ph > 7 or pw > 7:
            return None
        piece = rect[pr : pr + ph, pc : pc + pw].copy()
        nrc = int(np.sum(piece != rect_col))
        fill = nrc / float(ph * pw)
        if fill < 0.08 or fill > 0.65:
            return None
        if pr == 0 and not np.all(piece[0, :] == rect_col):
            return None
        if pr + ph == rh and not np.all(piece[-1, :] == rect_col):
            return None
        if pc == 0 and not np.all(piece[:, 0] == rect_col):
            return None
        if pc + pw == rw and not np.all(piece[:, -1] == rect_col):
            return None
        has_signal_edge = False
        if pr > 0 and np.any(piece[0, :] != rect_col):
            has_signal_edge = True
        if pr + ph < rh and np.any(piece[-1, :] != rect_col):
            has_signal_edge = True
        if pc > 0 and np.any(piece[:, 0] != rect_col):
            has_signal_edge = True
        if pc + pw < rw and np.any(piece[:, -1] != rect_col):
            has_signal_edge = True
        if not has_signal_edge:
            return None
        piece_arrays.append(piece)

    if corner == "TL":
        anchor_cell = (0, 0)
    elif corner == "TR":
        anchor_cell = (0, rw - 1)
    elif corner == "BL":
        anchor_cell = (rh - 1, 0)
    else:
        anchor_cell = (rh - 1, rw - 1)
    anchor_hits = 0
    for pr, pc, ph, pw, _bb in pieces_meta:
        if pr <= anchor_cell[0] < pr + ph and pc <= anchor_cell[1] < pc + pw:
            anchor_hits += 1
    if anchor_hits != 1:
        return None

    if corner == "TL":
        marker_cells = [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
        tr, tc = mr, mc
    elif corner == "TR":
        marker_cells = [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
        tr, tc = mr, mc - rw + 1
    elif corner == "BL":
        marker_cells = [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
        tr, tc = mr - rh + 1, mc
    else:
        marker_cells = [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]
        tr, tc = mr - rh + 1, mc - rw + 1
    for cr, cc in marker_cells:
        if cr < 0 or cr >= gh or cc < 0 or cc >= gw:
            return None

    canvas = np.full((gh, gw), bg, dtype=int)
    for cr, cc in marker_cells:
        canvas[cr, cc] = marker_col

    order = list(range(len(piece_arrays)))
    _r.shuffle(order)
    for idx in order:
        piece = piece_arrays[idx]
        try:
            r0, c0 = random_free_location(canvas, piece, transparent=bg, padding=1)
        except ValueError:
            return None
        paste(canvas, piece, r0, c0)

    expected = np.full((gh, gw), bg, dtype=int)
    expected[tr : tr + rh, tc : tc + rw] = rect

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
