# Variant of 446ef5d2
# Dimension: D8 — 标记系统的形式 (symbol_instructions)
# Change: 3 像素 L 形标记（2x2 缺一角），使用独特颜色（仅出现 3 次），缺失角指示矩形的一个顶点
#   → 十字形标记指示中心
#
# Design analysis
# ===============
# source_task:
# - None. No extra reference solver exists for this variant, so the baseline is the original 446ef5d2 code/spec only.
#
# a) Concretization and behavior trace
# - Old task:
#   The unique instruction symbol is a 3-cell L in a 2x2 box. Its missing cell directly gives one rectangle corner.
#   The solver therefore knows a corner coordinate before testing area factor pairs.
# - New task:
#   The unique instruction symbol is a 5-cell plus sign in a 3x3 box:
#       . X .
#       X X X
#       . X .
#   Its center cell denotes the center of the final assembled rectangle, not a corner.
#   To keep the center visually and logically exact on a pixel grid, the final rectangle dimensions are restricted
#   to odd x odd. The solver now uses:
#     1. plus-center coordinate
#     2. total scattered-piece area
#     3. odd factor pairs that fit around that center
#   to recover the rectangle bbox. The rectangle is then assembled exactly as in the original task.
# - Small trace:
#   Suppose the plus center is at (8, 11) and scattered piece area is 63.
#   Old logic would look for a corner and try placements anchored there; that premise no longer exists.
#   New logic enumerates odd factor pairs of 63, e.g. (7, 9), (9, 7), (3, 21), ...
#   For each candidate (H, W), top-left must be (8 - H//2, 11 - W//2). Only candidates that fit the grid survive.
#   The backtracking assembler then checks which size admits a full tiling with boundary/matching constraints.
#   The practical generator guarantee is stronger: only one odd factor pair is allowed to fit for the chosen center,
#   so the center instruction is visually meaningful and operationally decisive.
#
# b) Complexity calibration
# - If even dimensions were allowed, the "center" would lie between pixels and the plus marker would underspecify the
#   rectangle position. Therefore odd x odd dimensions are necessary, not decorative.
# - If multiple odd factor pairs fit around the same center, the marker would become visually weaker because examples
#   would not clearly show how size is recovered. The generator filters those cases out.
# - The puzzle is not simplified into "just center the pieces": edge matching with rect_col as wildcard is unchanged.
#   Only the anchor inference moved from corner-based to center-based.
#
# c) Old implicit assumptions that break under the variant
# - Original lines 27-58: assume a non-bg color appears exactly 3 times and forms an L; invalid now.
# - Original lines 86-117 and 255-263: convert a missing L-cell into a corner and derive top-left from that corner;
#   invalid now because the symbol gives the center instead.
# - Original lines 88-93 and 193-213: try to locate/pre-place an anchor piece containing that corner cell;
#   invalid now because the center marker is not tied to a visible corner cell in any scattered piece.
# - The rectangle decomposition, rect_col inference, wildcard edge compatibility, and backtracking tiling remain valid.
#
# d) Placement legality
# - The plus marker is placed on background and never overlaps scattered pieces in the input.
# - Pieces are still scattered with >=1 background padding from each other and from the marker.
# - The final rectangle is centered on the marker center and may cover the marker area in the output; this is intended.
#   The marker is an instruction symbol, not rectangle material, exactly analogous to the old marker disappearing in output.
#
# e) Visual observability
# - Across examples, the distinctive 5-cell plus is easy to spot and clearly no longer sits near a corner.
# - Because the output rectangle is centered on that plus, a human can compare 2-4 examples and infer that the plus
#   indicates the middle of the destination, while the pieces still determine content.
# - Restricting to odd x odd rectangles makes the "center" literal and visible.
#
# f) Visual cleanliness
# - The marker has its own unique color and 5-cell symmetric shape; pieces are rectangular blocks separated by background.
# - `random_free_location(..., padding=1)` ensures no piece touches the marker, so the symbol remains legible.
#
# g) Color inferability
# - No new color mapping is introduced.
# - As in the original, `rect_col` is inferred structurally as the dominant piece color and wildcard border color.
# - The generator additionally rejects cases where a non-marker piece color appears exactly 5 times, preventing accidental
#   confusion with the 5-cell marker detection rule.
#
# 9) New implicit assumptions introduced by the modified solver, and how the generator guarantees them
# - Assumption: the marker is the only non-bg color with total count 5 that forms a plus in a 3x3 bbox.
#   Guarantee: marker color is unique; generator rejects piece colors with total count 5.
# - Assumption: the rectangle has odd height and odd width, so the center is a cell.
#   Guarantee: generator samples only odd dimensions.
# - Assumption: the marker center plus total piece area leaves at least one fitting odd factor pair.
#   Guarantee: generator validates fit before scattering.
# - Assumption: ambiguous center-based sizes are undesirable.
#   Guarantee: generator keeps only cases with exactly one fitting odd factor pair for that marker center.

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import paste, random_free_location


def _find_plus_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int]]:
    cnt = Counter(map(int, g.flatten()))
    for col in sorted(c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 5):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g == int(col))}
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0) != 2 or (c1 - c0) != 2:
            continue
        center = (r0 + 1, c0 + 1)
        want = {
            center,
            (center[0] - 1, center[1]),
            (center[0] + 1, center[1]),
            (center[0], center[1] - 1),
            (center[0], center[1] + 1),
        }
        if pts == want:
            return int(col), center
    raise AssertionError("plus marker not found")


def _extract_rectangles(g: np.ndarray, occ: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
    GH, GW = g.shape
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


def _top_left_from_center(center: tuple[int, int], H: int, W: int) -> tuple[int, int]:
    cr, cc = map(int, center)
    return cr - H // 2, cc - W // 2


def _candidate_dims(total_area: int, center: tuple[int, int], GH: int, GW: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh != 0:
            continue
        for H, W in [(hh, total_area // hh), (total_area // hh, hh)]:
            H = int(H)
            W = int(W)
            if H % 2 == 0 or W % 2 == 0 or H < 5 or W < 5:
                continue
            tr, tc = _top_left_from_center(center, H, W)
            if 0 <= tr and 0 <= tc and (tr + H) <= GH and (tc + W) <= GW:
                dims.append((H, W))
    dims = sorted(set(dims), key=lambda t: (abs(t[0] - t[1]), -t[0], -t[1]))
    return dims


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into a rectangle centered at a 5-cell plus marker.

    Input structure:
    - bg = most common color.
    - Non-bg rectangular pieces are scattered on the grid.
    - A unique 5-pixel plus marker (center + 4-neighbors) indicates the center of the final rectangle.

    Rule:
    - rect_col = most common color across all piece pixels; all outer rectangle edges must be rect_col.
    - Translate pieces (no rotation) to tile the rectangle; touching edges must match, with rect_col as wildcard.
    - The final rectangle has odd height and odd width, so the plus center is the exact rectangle center.
    """
    g = np.array(grid, dtype=int)
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    mark, center = _find_plus_marker(g, bg)

    occ = (g != int(bg)) & (g != int(mark))
    rects = _extract_rectangles(g, occ)
    assert rects

    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))

    allpix: list[int] = []
    for _r, _c, _h, _w, a in rects:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    pieces = [a for _r, _c, _h, _w, a in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]

    dims = _candidate_dims(total_area, center, GH, GW)
    assert dims

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

        order = sorted(range(len(pieces)), key=lambda i: (-ph[i] * pw[i], -ph[i], -pw[i], i))
        solved = False

        def next_empty() -> tuple[int, int] | None:
            idx = np.argwhere(occ2 == -1)
            if idx.size == 0:
                return None
            return tuple(map(int, idx[0]))

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

    assert ass is not None and used_dim is not None

    H, W = used_dim
    tr, tc = _top_left_from_center(center, H, W)
    out = np.full((GH, GW), int(bg), dtype=int)
    out[tr : tr + H, tc : tc + W] = ass
    return out


def _center_fit_dims(total_area: int, GH: int, GW: int, cr: int, cc: int) -> list[tuple[int, int]]:
    return _candidate_dims(total_area, (cr, cc), GH, GW)


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

    max_side = min(21, GH, GW)
    odd_sides = [s for s in range(5, max_side + 1, 2)]
    if not odd_sides:
        return None

    RH = _r.choice(odd_sides)
    RW = _r.choice(odd_sides)
    if _r.random() < 0.5:
        large_odd = [s for s in odd_sides if s >= 7]
        if large_odd:
            RH = _r.choice(large_odd)
            RW = _r.choice(large_odd)

    total_area = RH * RW

    center_choices = [
        (cr, cc)
        for cr in range(RH // 2, GH - RH // 2)
        for cc in range(RW // 2, GW - RW // 2)
    ]
    _r.shuffle(center_choices)

    center = None
    for cr, cc in center_choices:
        if _center_fit_dims(total_area, GH, GW, cr, cc) == [(RH, RW)]:
            center = (cr, cc)
            break
    if center is None:
        return None

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

    rect[1:-1, 1:-1] = interior

    rc, cc = RH // 2, RW // 2
    for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        rect[rc + dr, cc + dc] = rect_col

    non_rc_count = int(np.sum(rect[1:-1, 1:-1] != rect_col))
    fill_ratio = non_rc_count / (iH * iW) if iH * iW > 0 else 0.0
    if fill_ratio < 0.30 or fill_ratio > 0.90:
        return None

    rect_counts = Counter(map(int, rect.flatten()))
    for col, n in rect_counts.items():
        if int(col) != int(bg) and int(col) != int(rect_col) and int(n) == 5:
            return None

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

    canvas = np.full((GH, GW), bg, dtype=int)

    cr, cc = center
    marker_cells = [(cr, cc), (cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)]
    for mr, mc in marker_cells:
        if mr < 0 or mr >= GH or mc < 0 or mc >= GW:
            return None
        canvas[mr, mc] = marker_col

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

    tr, tc = _top_left_from_center(center, RH, RW)
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
