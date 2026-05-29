# Variant of 446ef5d2
# Dimension: D15 — 碎片内部填充率约束 (meta)
# Change: 每块碎片 15-55% 的面积为非 rect_col 像素；整体矩形内部（除外框外）35-85% 为 interior_... → 极低填充率（碎片内大部分是 rect_col）
#
# Design analysis for this variant
# --------------------------------
# a) Concrete behavior trace
#    - The core scene is unchanged: scattered rectangular pieces, one 3-cell L-marker,
#      and an output that reassembles those pieces into a single rectangle anchored at
#      the marker gap.
#    - The visible change is that pieces now carry only a few colored pixels on their
#      internal seams. Most of every piece is rect_col. To keep the sparse clue system
#      readable and solvable, every true internal seam contributes exactly one aligned
#      colored pair. In the assembled rectangle these appear as isolated seam dots rather
#      than dense blobs.
#    - Small trace:
#      original: two neighboring pieces may share many colored cells on a seam, so
#      `compat()` gets several hard constraints.
#      variant: the same seam may share only one colored cell; all other seam cells are
#      rect_col wildcards. The solver therefore cannot rely on dense local evidence and
#      must prefer placements that preserve actual seam clues.
# b) Complexity calibration
#    - If we only lowered fill without changing generation, many seams would become fully
#      wildcard and multiple assemblies would satisfy the old solver.
#    - To avoid that degeneration, the generator uses low-degree partitions (`2xN` or
#      `Nx2`) and guarantees one real clue per internal seam. This keeps the puzzle hard
#      but not ambiguous: clues are sparse, repeated, and visually observable.
# c) Implicit assumption review against the original main()
#    - `bg = most common color in grid`: with low fill, rect_col can dominate globally if
#      the rectangle is too large. The generator now enforces rectangle area small enough
#      that background remains globally most common.
#    - `rect_col = most common across piece pixels`: this remains valid and becomes even
#      stronger under low fill.
#    - `compat()` treating rect_col as wildcard: still valid, but by itself too weak. The
#      new solver therefore does not accept completed tilings whose internal seams contain
#      zero true clue matches.
#    - The original backtracking used "first empty cell" and returned the first valid
#      assembly. Under sparse clues this can lock onto a weakly-constrained arrangement.
#      The new solver uses MRV-style position choice and scores/validates seam clues.
# d) Placement legality
#    - Pieces are still solid rectangles, scattered with >=1 background gap.
#    - Marker placement follows the original corner mapping and is checked for bounds.
#    - The final rectangle is placed exactly at the marker gap and validated by calling
#      `main()` on each generated sample.
# e) Visual observability
#    - Across 2-4 ARC examples a human can notice that the assembled rectangle is mostly
#      rect_col and only tiny seam-aligned pixels carry information.
#    - Two interior colors are used by seam orientation when available, giving repeated
#      visible evidence instead of hidden numeric logic.
# f) Visual cleanliness
#    - Sparse clue pixels are placed away from seam endpoints, so they do not pile up at
#      four-way corners. This keeps pieces readable despite the stronger wildcard effect.
# g) Color inference
#    - Colors are not ordered by numeric value. The generator only uses 1-2 interior
#      colors, repeated structurally on seams, so a human can infer that equal colored
#      seam dots must line up across matching pieces.
#
# New solver-side assumptions introduced by the variant
# ----------------------------------------------------
# - Every true internal seam in generated data has at least one aligned non-rect_col
#   match. `generate_input()` guarantees this explicitly.
# - Piece fill is driven almost entirely by seam-clue count, so partitions are restricted
#   to shapes where degree/area stays inside 5-15%.

from __future__ import annotations

from collections import Counter
import math
import random

import numpy as np

from arc_utils import paste, random_free_location


def _find_marker(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int], tuple[int, int, int, int], str]:
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
        return int(col), (int(missing[0]), int(missing[1])), (r0, c0, r1, c1), corner
    raise AssertionError("marker not found")


def _extract_rectangles(grid: np.ndarray, bg: int, mark: int) -> list[tuple[int, int, int, int, np.ndarray]]:
    occ = (grid != int(bg)) & (grid != int(mark))
    used = np.zeros_like(occ, dtype=bool)
    gh, gw = grid.shape
    rects: list[tuple[int, int, int, int, np.ndarray]] = []
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


def _clue_count(v1: np.ndarray, v2: np.ndarray, rect_col: int) -> int:
    a = np.asarray(v1, dtype=int)
    b = np.asarray(v2, dtype=int)
    m = (a != int(rect_col)) & (b != int(rect_col)) & (a == b)
    return int(m.sum())


def _compat(v1: np.ndarray, v2: np.ndarray, rect_col: int) -> bool:
    a = np.asarray(v1, dtype=int)
    b = np.asarray(v2, dtype=int)
    m = (a != int(rect_col)) & (b != int(rect_col))
    return bool((a[m] == b[m]).all())


def _edge_info(piece: np.ndarray, rect_col: int) -> int:
    top = int(np.sum(piece[0, :] != int(rect_col)))
    bottom = int(np.sum(piece[-1, :] != int(rect_col)))
    left = int(np.sum(piece[:, 0] != int(rect_col)))
    right = int(np.sum(piece[:, -1] != int(rect_col)))
    return top + bottom + left + right


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    gh, gw = g.shape

    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, missing, _bbox, corner = _find_marker(g, bg)
    rects = _extract_rectangles(g, bg, mark)
    assert rects

    allpix: list[int] = []
    for _r, _c, _h, _w, arr in rects:
        allpix.extend(map(int, arr.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))
    mr, mc = missing

    pieces = [arr for _r, _c, _h, _w, arr in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]
    pareas = [int(h) * int(w) for h, w in zip(ph, pw)]
    pedge = [_edge_info(p, rect_col) for p in pieces]

    dims: list[tuple[int, int]] = []
    for hh in range(1, int(math.isqrt(total_area)) + 1):
        if total_area % hh != 0:
            continue
        dims.append((hh, total_area // hh))
        if hh * hh != total_area:
            dims.append((total_area // hh, hh))

    def fits(hh: int, ww: int) -> bool:
        if corner == "TL":
            tr, tc = mr, mc
        elif corner == "TR":
            tr, tc = mr, mc - ww + 1
        elif corner == "BL":
            tr, tc = mr - hh + 1, mc
        else:
            tr, tc = mr - hh + 1, mc - ww + 1
        return 0 <= tr and 0 <= tc and tr + hh <= gh and tc + ww <= gw

    dims = [(int(hh), int(ww)) for hh, ww in dims if fits(int(hh), int(ww))]
    dims.sort(key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    anchor_idx = None
    for i, (rr, cc, h, w, _a) in enumerate(rects):
        if rr <= mr < rr + h and cc <= mc < cc + w:
            anchor_idx = i
            break

    def assemble(hh: int, ww: int) -> tuple[np.ndarray | None, int]:
        occ2 = np.full((hh, ww), -1, dtype=int)
        pid2 = np.full((hh, ww), -1, dtype=int)
        usedp = [False] * len(pieces)
        placements: list[tuple[int, int] | None] = [None] * len(pieces)
        best_grid: np.ndarray | None = None
        best_score = -1

        def boundary_ok(arr: np.ndarray, r: int, c: int) -> bool:
            h, w = arr.shape
            if r == 0 and not np.all(arr[0, :] == int(rect_col)):
                return False
            if r + h == hh and not np.all(arr[-1, :] == int(rect_col)):
                return False
            if c == 0 and not np.all(arr[:, 0] == int(rect_col)):
                return False
            if c + w == ww and not np.all(arr[:, -1] == int(rect_col)):
                return False
            return True

        def neighbor_ok(arr: np.ndarray, r: int, c: int) -> tuple[bool, int]:
            h, w = arr.shape
            gain = 0

            def check_segment(existing_cols: np.ndarray, existing_ids: np.ndarray, new_cols: np.ndarray) -> tuple[bool, int]:
                m = existing_cols != -1
                if m.any() and (not _compat(existing_cols[m], new_cols[m], rect_col)):
                    return False, 0
                seg_gain = _clue_count(existing_cols[m], new_cols[m], rect_col) if m.any() else 0
                if np.all(existing_ids != -1):
                    uniq = np.unique(existing_ids)
                    if len(uniq) == 1 and seg_gain == 0:
                        return False, 0
                return True, seg_gain

            if c > 0:
                ok, seg = check_segment(occ2[r : r + h, c - 1], pid2[r : r + h, c - 1], arr[:, 0])
                if not ok:
                    return False, 0
                gain += seg
            if c + w < ww:
                ok, seg = check_segment(occ2[r : r + h, c + w], pid2[r : r + h, c + w], arr[:, -1])
                if not ok:
                    return False, 0
                gain += seg
            if r > 0:
                ok, seg = check_segment(occ2[r - 1, c : c + w], pid2[r - 1, c : c + w], arr[0, :])
                if not ok:
                    return False, 0
                gain += seg
            if r + h < hh:
                ok, seg = check_segment(occ2[r + h, c : c + w], pid2[r + h, c : c + w], arr[-1, :])
                if not ok:
                    return False, 0
                gain += seg
            return True, gain

        def eval_place(i: int, r: int, c: int) -> tuple[bool, int]:
            arr = pieces[i]
            h, w = arr.shape
            if r + h > hh or c + w > ww:
                return False, 0
            if (occ2[r : r + h, c : c + w] != -1).any():
                return False, 0
            if not boundary_ok(arr, r, c):
                return False, 0
            return neighbor_ok(arr, r, c)

        def place(i: int, r: int, c: int) -> None:
            arr = pieces[i]
            h, w = arr.shape
            occ2[r : r + h, c : c + w] = arr
            pid2[r : r + h, c : c + w] = i
            usedp[i] = True
            placements[i] = (r, c)

        def unplace(i: int, r: int, c: int) -> None:
            arr = pieces[i]
            h, w = arr.shape
            occ2[r : r + h, c : c + w] = -1
            pid2[r : r + h, c : c + w] = -1
            usedp[i] = False
            placements[i] = None

        def score_completed() -> tuple[bool, int]:
            total = 0
            for i, pos in enumerate(placements):
                assert pos is not None
                r, c = pos
                arr = pieces[i]
                h, w = arr.shape
                if c + w < ww:
                    nbr = int(pid2[r, c + w])
                    if nbr == -1 or not np.all(pid2[r : r + h, c + w] == nbr):
                        return False, 0
                    seam = occ2[r : r + h, c + w]
                    sc = _clue_count(arr[:, -1], seam, rect_col)
                    if sc == 0:
                        return False, 0
                    total += sc
                if r + h < hh:
                    nbr = int(pid2[r + h, c])
                    if nbr == -1 or not np.all(pid2[r + h, c : c + w] == nbr):
                        return False, 0
                    seam = occ2[r + h, c : c + w]
                    sc = _clue_count(arr[-1, :], seam, rect_col)
                    if sc == 0:
                        return False, 0
                    total += sc
            return True, total

        def select_next() -> tuple[tuple[int, int] | None, list[tuple[int, int]]]:
            empties = np.argwhere(occ2 == -1)
            if empties.size == 0:
                return None, []
            best_pos = None
            best_candidates: list[tuple[int, int]] | None = None
            best_key = None
            for r, c in map(tuple, empties):
                r = int(r)
                c = int(c)
                if r > 0 and occ2[r - 1, c] == -1:
                    continue
                if c > 0 and occ2[r, c - 1] == -1:
                    continue
                cand: list[tuple[int, int, int, int]] = []
                for i in range(len(pieces)):
                    if usedp[i]:
                        continue
                    ok, gain = eval_place(i, r, c)
                    if ok:
                        cand.append((i, gain, pedge[i], pareas[i]))
                key = (len(cand), r, c)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pos = (r, c)
                    best_candidates = cand
                if best_key is not None and best_key[0] == 0:
                    break
            if best_pos is None or best_candidates is None:
                return None, []
            best_candidates.sort(key=lambda t: (-t[1], -t[2], -t[3], t[0]))
            return best_pos, [(i, gain) for i, gain, _edge, _area in best_candidates]

        def bt(current_score: int) -> None:
            nonlocal best_grid, best_score
            pos, candidates = select_next()
            if pos is None:
                ok, score = score_completed()
                if ok and score > best_score:
                    best_score = score
                    best_grid = occ2.copy()
                return
            if not candidates:
                return
            future_bound = current_score + sum(pedge[i] for i in range(len(pieces)) if not usedp[i])
            if future_bound <= best_score:
                return
            r, c = pos
            for i, gain in candidates:
                place(i, r, c)
                bt(current_score + gain)
                unplace(i, r, c)

        if anchor_idx is not None:
            rr, cc, h0, w0, _ = rects[anchor_idx]
            corners = {
                (rr, cc): "TL",
                (rr, cc + w0 - 1): "TR",
                (rr + h0 - 1, cc): "BL",
                (rr + h0 - 1, cc + w0 - 1): "BR",
            }
            if (mr, mc) in corners and corners[(mr, mc)] == corner:
                if corner == "TL":
                    ar0, ac0 = 0, 0
                elif corner == "TR":
                    ar0, ac0 = 0, ww - w0
                elif corner == "BL":
                    ar0, ac0 = hh - h0, 0
                else:
                    ar0, ac0 = hh - h0, ww - w0
                ok, _gain = eval_place(anchor_idx, ar0, ac0)
                if ok:
                    place(anchor_idx, ar0, ac0)

        bt(0)
        return best_grid, best_score

    best_ass = None
    best_dim = None
    best_score = -1
    for hh, ww in dims:
        ass, score = assemble(hh, ww)
        if ass is not None and score > best_score:
            best_ass = ass
            best_dim = (hh, ww)
            best_score = score

    assert best_ass is not None and best_dim is not None
    hh, ww = best_dim
    if corner == "TL":
        tr, tc = mr, mc
    elif corner == "TR":
        tr, tc = mr, mc - ww + 1
    elif corner == "BL":
        tr, tc = mr - hh + 1, mc
    else:
        tr, tc = mr - hh + 1, mc - ww + 1

    out = np.full((gh, gw), int(bg), dtype=int)
    out[tr : tr + hh, tc : tc + ww] = best_ass
    return out


def _piece_degree(piece: tuple[int, int, int, int], pieces: list[tuple[int, int, int, int]]) -> int:
    pr, pc, ph, pw = piece
    deg = 0
    for qr, qc, qh, qw in pieces:
        if (pr, pc, ph, pw) == (qr, qc, qh, qw):
            continue
        if pr == qr + qh and not (pc + pw <= qc or qc + qw <= pc):
            deg += 1
        elif qr == pr + ph and not (pc + pw <= qc or qc + qw <= pc):
            deg += 1
        elif pc == qc + qw and not (pr + ph <= qr or qr + qh <= pr):
            deg += 1
        elif qc == pc + pw and not (pr + ph <= qr or qr + qh <= pr):
            deg += 1
    return deg


def _build_partition(rng: random.Random) -> tuple[int, int, list[tuple[int, int, int, int]], list[int]]:
    if rng.random() < 0.5:
        n_rows, n_cols = 2, rng.randint(2, 4)
    else:
        n_rows, n_cols = rng.randint(2, 4), 2

    row_segs = [rng.randint(4, 7) for _ in range(n_rows)]
    col_segs = [rng.randint(4, 7) for _ in range(n_cols)]
    rh = int(sum(row_segs))
    rw = int(sum(col_segs))

    row_starts: list[int] = []
    acc = 0
    for s in row_segs:
        row_starts.append(acc)
        acc += s
    col_starts: list[int] = []
    acc = 0
    for s in col_segs:
        col_starts.append(acc)
        acc += s

    pieces: list[tuple[int, int, int, int]] = []
    for i, pr in enumerate(row_starts):
        for j, pc in enumerate(col_starts):
            pieces.append((pr, pc, row_segs[i], col_segs[j]))

    degrees = [_piece_degree(p, pieces) for p in pieces]
    for (pr, pc, ph, pw), deg in zip(pieces, degrees):
        area = ph * pw
        fill = deg / area
        if fill < 0.05 or fill > 0.15:
            raise ValueError("partition does not satisfy low-fill seam budget")

    return rh, rw, pieces, degrees


def _paint_sparse_clues(
    rect: np.ndarray,
    pieces: list[tuple[int, int, int, int]],
    interior_cols: list[int],
    rng: random.Random,
) -> None:
    vertical_col = int(interior_cols[0])
    horizontal_col = int(interior_cols[-1])

    by_top_left = {(pr, pc): (pr, pc, ph, pw) for pr, pc, ph, pw in pieces}

    for pr, pc, ph, pw in pieces:
        right_piece = by_top_left.get((pr, pc + pw))
        if right_piece is not None:
            offset = rng.randint(1, ph - 2)
            rr = pr + offset
            rect[rr, pc + pw - 1] = vertical_col
            rect[rr, pc + pw] = vertical_col
        down_piece = by_top_left.get((pr + ph, pc))
        if down_piece is not None:
            offset = rng.randint(1, pw - 2)
            cc = pc + offset
            rect[pr + ph - 1, cc] = horizontal_col
            rect[pr + ph, cc] = horizontal_col


def _marker_cells(corner: str, mr: int, mc: int) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _assembled_top_left(corner: str, mr: int, mc: int, rh: int, rw: int) -> tuple[int, int]:
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - rw + 1
    if corner == "BL":
        return mr - rh + 1, mc
    return mr - rh + 1, mc - rw + 1


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    try:
        rh, rw, pieces, degrees = _build_partition(rng)
    except ValueError:
        return None

    min_gh = max(rh + 3, 18)
    min_gw = max(rw + 3, 18)
    if min_gh > 26 or min_gw > 26:
        return None

    gh = rng.randint(min_gh, 26)
    gw = rng.randint(min_gw, 26)

    palette = list(range(10))
    rng.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    n_interior = 2 if rng.random() < 0.75 else 1
    interior_cols = palette[3 : 3 + n_interior]

    rect = np.full((rh, rw), rect_col, dtype=int)
    _paint_sparse_clues(rect, pieces, interior_cols, rng)

    corner = rng.choice(["TL", "TR", "BL", "BR"])
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
    mr = rng.randint(mr_min, mr_max)
    mc = rng.randint(mc_min, mc_max)
    marker_cells = _marker_cells(corner, mr, mc)
    if any(r < 0 or r >= gh or c < 0 or c >= gw for r, c in marker_cells):
        return None

    piece_arrays = [rect[pr : pr + ph, pc : pc + pw].copy() for pr, pc, ph, pw in pieces]

    for piece, deg in zip(piece_arrays, degrees):
        ph, pw = piece.shape
        fill = float(np.sum(piece != rect_col)) / float(ph * pw)
        if fill < 0.05 or fill > 0.15:
            return None
        edge_has_color = any(
            np.any(edge != rect_col)
            for edge in (piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1])
        )
        if deg <= 0 or not edge_has_color:
            return None

    interior_area = max(1, (rh - 2) * (rw - 2))
    interior_fill = float(np.sum(rect[1:-1, 1:-1] != rect_col)) / float(interior_area)
    if interior_fill < 0.05 or interior_fill > 0.20:
        return None

    canvas = np.full((gh, gw), bg, dtype=int)
    for r, c in marker_cells:
        canvas[r, c] = marker_col

    order = list(range(len(piece_arrays)))
    rng.shuffle(order)
    for idx in order:
        patch = piece_arrays[idx]
        try:
            r0, c0 = random_free_location(canvas, patch, transparent=bg, padding=1)
        except ValueError:
            return None
        paste(canvas, patch, r0, c0)

    bg_count = int(np.sum(canvas == bg))
    rect_count = int(np.sum(canvas == rect_col))
    if bg_count <= rect_count:
        return None

    out = np.full((gh, gw), bg, dtype=int)
    tr, tc = _assembled_top_left(corner, mr, mc, rh, rw)
    out[tr : tr + rh, tc : tc + rw] = rect

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
