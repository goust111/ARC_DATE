# Combo variant of 446ef5d2 (index 8)
# combo_spec: D8/2+D13/1
# Name: 十字标记指示中心
#
# Design analysis
# ===============
# source_task:
# - Baseline is the original 446ef5d2 solver/spec.
# - The only reference implementation provided is the single-variant D8/2 code, which already
#   captures the core shift from corner-anchoring to center-anchoring. D13/1 is not an extra
#   surface feature; it reinforces that the old "corner type is directly encoded by the marker"
#   meta-assumption is removed and must be replaced by "infer size from area, then infer corners
#   from the center".
#
# a) Concretization and behavior trace
# - Old task:
#   A unique 3-cell L marker sat next to one rectangle corner. The missing cell of that 2x2 box
#   directly named the corner cell of the assembled rectangle, so the solver could anchor the
#   final bbox before any tiling search.
# - New fused task:
#   The marker is a 5-cell plus:
#       . X .
#       X X X
#       . X .
#   and its center cell marks the rectangle center, not a corner. The solver must:
#   1. detect the plus center,
#   2. sum scattered piece areas,
#   3. enumerate odd factor pairs of that area that can be centered there,
#   4. recover the four corner coordinates from the chosen (H, W),
#   5. tile the rectangle exactly as before.
# - Small trace:
#   Suppose the plus center is at (8, 11) and total piece area is 63.
#   The old logic would ask "which corner did the L point at?" That question is now invalid.
#   The fused logic instead tests odd factor pairs of 63 such as (7, 9), (9, 7), (3, 21), ...
#   Only candidates with top-left = (8 - H//2, 11 - W//2) inside the grid survive.
#   After a valid size is chosen, corners become:
#     TL = (8 - H//2, 11 - W//2)
#     TR = (8 - H//2, 11 + W//2)
#     BL = (8 + H//2, 11 - W//2)
#     BR = (8 + H//2, 11 + W//2)
#   This is the intended D8/2 + D13/1 fusion: the symbol no longer encodes any corner identity,
#   and corner coordinates are reconstructed only after size inference from area.
#
# b) Complexity calibration
# - Odd x odd dimensions are required. Otherwise a single cell could not be the exact rectangle
#   center, and the plus would underspecify the bbox.
# - The generator keeps only cases where the center and total area admit exactly one fitting odd
#   factor pair. Without this, the center cue would be weaker and examples would be less legible.
# - The task is not reduced to mere recentering: the wildcard edge-matching/backtracking logic from
#   the original task is preserved. Only the localization mechanism changed.
#
# c) Implicit assumption review against the original main()
# - Original marker detection assumed "count == 3" and L geometry. That is invalid and replaced by
#   "count == 5" plus geometry.
# - Original code derived a corner label (TL/TR/BL/BR) from the L's missing cell. That entire branch
#   disappears because the plus no longer encodes corner type.
# - Original candidate-dimension filtering used a known corner anchor. That must be replaced by
#   center-based bbox reconstruction.
# - Original optional anchor-piece preplacement relied on the visible corner cell being inside one
#   scattered piece at a matching corner. That premise is gone, so the fused solver removes it.
# - Rectangle extraction, rect_col inference, boundary checks, wildcard compatibility, and the
#   backtracking tiler remain valid and are reused.
#
# d) Placement legality
# - The plus marker is always painted on background cells, never overlapping scattered pieces.
# - Pieces are still scattered with at least one background cell of padding, which also keeps the
#   marker visually isolated.
# - The output rectangle may cover the marker's former location. This is intended: the marker is an
#   instruction symbol in the input only.
#
# e) Visual observability
# - Across examples, the 5-cell plus is obviously a different symbol than the old corner L and it is
#   clearly located in the middle region of the final rectangle rather than at a corner.
# - Because the rectangle is reconstructed symmetrically around that plus and its size comes from the
#   pieces' total area, a human can infer "center marker + area-sized assembly" from a few examples.
#
# f) Visual cleanliness
# - The marker uses its own unique color and symmetric 3x3 footprint.
# - Pieces remain axis-aligned rectangles separated by background, so the grid is not cluttered.
#
# g) Inferability of changes
# - No hidden color ordering is used. `rect_col` is still inferred structurally as the dominant piece
#   color and wildcard border color.
# - To keep marker detection unambiguous, the generator rejects cases where any non-marker piece color
#   appears exactly 5 times in total; otherwise the solver might have multiple count-5 candidates.
#
# 9) New solver assumptions introduced here, and generator guarantees
# - Assumption: exactly one non-bg color with total count 5 forms a plus. Guaranteed by unique marker
#   color and by rejecting non-marker total-count-5 colors.
# - Assumption: final rectangle dimensions are odd and at least 5. Guaranteed by generator sampling.
# - Assumption: center + area yields at least one fitting odd factor pair. Guaranteed before scatter.
# - Assumption: center + area should determine the bbox uniquely in generated tasks. Guaranteed by
#   keeping only samples with exactly one fitting odd factor pair.

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


def _extract_rectangles(
    g: np.ndarray, occ: np.ndarray
) -> list[tuple[int, int, int, int, np.ndarray]]:
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


def _corners_from_center(
    center: tuple[int, int], H: int, W: int
) -> dict[str, tuple[int, int]]:
    cr, cc = map(int, center)
    hh = H // 2
    hw = W // 2
    return {
        "TL": (cr - hh, cc - hw),
        "TR": (cr - hh, cc + hw),
        "BL": (cr + hh, cc - hw),
        "BR": (cr + hh, cc + hw),
    }


def _candidate_dims(total_area: int, center: tuple[int, int], GH: int, GW: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh != 0:
            continue
        for H, W in ((hh, total_area // hh), (total_area // hh, hh)):
            H = int(H)
            W = int(W)
            if H < 5 or W < 5 or H % 2 == 0 or W % 2 == 0:
                continue
            corners = _corners_from_center(center, H, W)
            tr, tc = corners["TL"]
            br, bc = corners["BR"]
            if 0 <= tr and 0 <= tc and br < GH and bc < GW:
                dims.append((H, W))
    return sorted(set(dims), key=lambda t: (abs(t[0] - t[1]), -t[0], -t[1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into a rectangle centered at a 5-cell plus marker.

    The fused rule is:
    - the plus gives the center cell of the final rectangle,
    - the total scattered-piece area constrains the rectangle dimensions,
    - the chosen dimensions determine all four corners,
    - pieces tile that rectangle with the original wildcard edge-matching rule.
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
        mask = (v1 != int(rect_col)) & (v2 != int(rect_col))
        return bool((v1[mask] == v2[mask]).all())

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
                mask = left != -1
                if mask.any() and (not compat(left[mask], a[:, 0][mask])):
                    return False
            if (c + w) < W:
                right = occ2[r : r + h, c + w]
                mask = right != -1
                if mask.any() and (not compat(right[mask], a[:, -1][mask])):
                    return False
            if r > 0:
                up = occ2[r - 1, c : c + w]
                mask = up != -1
                if mask.any() and (not compat(up[mask], a[0, :][mask])):
                    return False
            if (r + h) < H:
                down = occ2[r + h, c : c + w]
                mask = down != -1
                if mask.any() and (not compat(down[mask], a[-1, :][mask])):
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
    top_left = _corners_from_center(center, H, W)["TL"]
    out = np.full((GH, GW), int(bg), dtype=int)
    tr, tc = top_left
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
        larger = [s for s in odd_sides if s >= 7]
        if larger:
            RH = _r.choice(larger)
            RW = _r.choice(larger)

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
        budget = max(1, budget)

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
            new_frontier = []
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
                    new_frontier.append((nr, nc))
                    if len(blob) >= budget:
                        break
                if len(blob) >= budget:
                    break
            frontier = new_frontier

        for br, bc in blob:
            interior[br, bc] = ic
            blob_occupied[br, bc] = True

    rect[1:-1, 1:-1] = interior

    non_rc_count = int(np.sum(interior != rect_col))
    fill_ratio = non_rc_count / (iH * iW)
    if fill_ratio < 0.30 or fill_ratio > 0.90:
        return None

    rect_counts = Counter(map(int, rect.flatten()))
    for col, n in rect_counts.items():
        if int(col) != int(rect_col) and int(n) == 5:
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
    acc = 0
    for rs in row_segs:
        row_starts.append(acc)
        acc += rs

    col_starts = []
    acc = 0
    for cs in col_segs:
        col_starts.append(acc)
        acc += cs

    nr = len(row_segs)
    nc_grid = len(col_segs)
    cell_id = [[i * nc_grid + j for j in range(nc_grid)] for i in range(nr)]
    n_cells = nr * nc_grid
    if n_cells < 4:
        return None

    def _get_piece_bbox(pid: int) -> tuple[int, int, int, int] | None:
        cells = [(i, j) for i in range(nr) for j in range(nc_grid) if cell_id[i][j] == pid]
        if not cells:
            return None
        ri = [r for r, _ in cells]
        ci = [c for _, c in cells]
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

    top_left = _corners_from_center(center, RH, RW)["TL"]
    expected_out = np.full((GH, GW), bg, dtype=int)
    tr, tc = top_left
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
