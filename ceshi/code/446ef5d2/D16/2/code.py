# Variant of 446ef5d2
# Dimension: D16 — 锚定碎片的特殊性 (meta)
# Change: 恰好一块碎片包含标记缺失角对应的矩形顶点，该碎片作为锚定碎片优先放置在角落... → 多个候选锚定碎片

from __future__ import annotations

from collections import Counter
import numpy as np

from arc_utils import paste, random_free_location


# Design analysis for the D16 variant
# ----------------------------------
# a) Concrete behavior and trace
#    Original behavior: the solver looked for a single "special" piece and, if found,
#    pre-placed it at the marker-indicated rectangle corner before backtracking.
#    In the original code that specialness was encoded by a unique geometric coincidence:
#    one scattered piece happened to occupy the marker's missing-grid coordinate and also
#    had the matching local corner there. After that pre-placement, the remaining search
#    was straightforward edge matching.
#    Variant behavior: several pieces visibly have the same local rect_col L-corner in the
#    marker-required orientation. Any of them could look like the corner piece to a human.
#    The solver therefore cannot trust a single anchor inference; it must try multiple
#    candidate anchors, place one at the corner, and validate by completing the full
#    rectangle. Example for corner=TL:
#    - old logic: find one piece that is somehow tied to (mr, mc), put it at (0, 0)
#    - new logic: collect every piece whose top edge and left edge are all rect_col,
#      try each one at (0, 0), and keep only the one that allows a full tiling
#    - difference: the anchor step becomes a controlled search branch instead of a direct
#      deterministic pick.
#
# b) Complexity calibration
#    The variant is kept narrow: only the anchor-selection mechanism changes. We do not
#    add rotations, flips, or new object types. To make the change visually observable,
#    the generator guarantees at least two candidate corner-looking pieces. At the same
#    time, the number of candidates is kept small (typically 2-3), so the task remains
#    about trying a few plausible anchors rather than facing unconstrained combinatorics.
#
# c) Implicit-assumption audit against the original main()
#    - Marker detection still assumes one color appears exactly 3 times as a 2x2-minus-1
#      L. This remains valid and is preserved.
#    - Rectangle decomposition still assumes every non-bg non-marker object is an axis-
#      aligned rectangle and that objects are bg-separated. This remains valid and is
#      preserved by the generator.
#    - The old anchor_idx scan depended on the old uniqueness assumption. In this variant
#      it is invalid: several pieces can satisfy the visual corner condition, and the
#      scattered coordinates no longer identify the true one. That code path is replaced
#      with candidate enumeration by local edge content.
#    - The old "pre-place at most one anchor, otherwise solve without an anchor" pattern
#      was harmless before because a unique anchor often collapsed the search. In this
#      variant, skipping explicit anchor branching would ignore the requested change, so
#      assemble() is now called once per candidate anchor (plus a final fallback).
#
# d) Placement legality
#    The generator still builds the final rectangle first, partitions it into rectangles,
#    then scatters those rectangles with >=1 bg-cell padding. Extra candidate anchors are
#    created by forcing two incident edges of selected pieces to rect_col in the target
#    corner orientation. Because rect_col is already the wildcard match color, this never
#    creates illegal edge mismatches by itself. The generator revalidates every piece and
#    the whole puzzle with main(), so only legal placements survive.
#
# e) Visual observability
#    Across input/output examples, the user can see several fragments with the same kind
#    of rect_col corner that "could" fit the L-marker. Only one completes the edge colors
#    consistently. That is visible from pixels alone and does not depend on color ids,
#    ordering by numeric values, or hidden metadata.
#
# f) Visual cleanliness
#    No new marks are overlaid on pieces. We only alter existing piece borders so that
#    some internal pieces gain a clean rect_col L corner. Objects remain separated by bg,
#    and the only explicit marker is still the original 3-cell L, so the scene stays easy
#    to parse.
#
# g) Color-inference safety
#    Colors keep the original semantics: bg, one marker color, one rect_col wildcard/
#    border color, and 1-3 interior colors. No new color mapping rule is introduced.
#
# New solver assumptions introduced by the variant
#    - At least one piece has the required corner-oriented rect_col L; the generator
#      guarantees this because the true corner piece always does.
#    - Usually multiple such pieces exist; the generator explicitly enforces >=2.
#    - The correct anchor choice is the one whose forced placement allows a complete
#      tiling under the unchanged edge-compatibility rule; the generator validates this.


def _corner_from_missing(missing: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    if missing == (r0, c0):
        return "BR"
    if missing == (r0, c1):
        return "BL"
    if missing == (r1, c0):
        return "TR"
    return "TL"


def _top_left_from_corner(corner: str, mr: int, mc: int, H: int, W: int) -> tuple[int, int]:
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - W + 1
    if corner == "BL":
        return mr - H + 1, mc
    return mr - H + 1, mc - W + 1


def _find_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], str]:
    cnt = Counter(map(int, g.flatten()))
    for col in sorted([c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3]):
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(col))]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0) != 1 or (c1 - c0) != 1:
            continue
        all4 = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        miss = list(all4 - set(pts))
        if len(miss) == 1:
            missing = miss[0]
            return int(col), missing, _corner_from_missing(missing, (r0, c0, r1, c1))
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


def _is_corner_candidate(piece: np.ndarray, corner: str, rect_col: int) -> bool:
    if corner == "TL":
        return bool(np.all(piece[0, :] == int(rect_col)) and np.all(piece[:, 0] == int(rect_col)))
    if corner == "TR":
        return bool(np.all(piece[0, :] == int(rect_col)) and np.all(piece[:, -1] == int(rect_col)))
    if corner == "BL":
        return bool(np.all(piece[-1, :] == int(rect_col)) and np.all(piece[:, 0] == int(rect_col)))
    return bool(np.all(piece[-1, :] == int(rect_col)) and np.all(piece[:, -1] == int(rect_col)))


def _force_candidate_corner(
    rect: np.ndarray,
    piece_box: tuple[int, int, int, int],
    corner: str,
    rect_col: int,
) -> None:
    pr, pc, ph, pw = piece_box
    if corner == "TL":
        rect[pr, pc : pc + pw] = rect_col
        rect[pr : pr + ph, pc] = rect_col
    elif corner == "TR":
        rect[pr, pc : pc + pw] = rect_col
        rect[pr : pr + ph, pc + pw - 1] = rect_col
    elif corner == "BL":
        rect[pr + ph - 1, pc : pc + pw] = rect_col
        rect[pr : pr + ph, pc] = rect_col
    else:
        rect[pr + ph - 1, pc : pc + pw] = rect_col
        rect[pr : pr + ph, pc + pw - 1] = rect_col


def _piece_stats(piece: np.ndarray, rect_col: int) -> tuple[float, bool]:
    ph, pw = piece.shape
    fill = float(np.sum(piece != rect_col)) / float(ph * pw)
    has_non_rc_edge = False
    for edge in [piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]]:
        if np.any(edge != rect_col):
            has_non_rc_edge = True
            break
    return fill, has_non_rc_edge


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into a rectangle at an L-marker corner.

    Variant D16:
    - Multiple pieces may look like plausible corner anchors because they have the
      required rect_col L-corner in the marker-specified orientation.
    - The solver must try candidate anchors and keep the one that allows a full tiling.
    """
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

    pieces = [a for _r, _c, _h, _w, a in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]
    mr, mc = map(int, missing)

    dims = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh == 0:
            dims.append((hh, total_area // hh))
            if hh * hh != total_area:
                dims.append((total_area // hh, hh))

    def fits(H: int, W: int) -> bool:
        tr, tc = _top_left_from_corner(corner, mr, mc, H, W)
        return 0 <= tr and 0 <= tc and (tr + H) <= GH and (tc + W) <= GW

    dims = [(int(H), int(W)) for H, W in dims if fits(int(H), int(W))]
    dims = sorted(dims, key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    def compat(v1: np.ndarray, v2: np.ndarray) -> bool:
        m = (v1 != int(rect_col)) & (v2 != int(rect_col))
        return bool((v1[m] == v2[m]).all())

    candidate_anchors = [i for i, piece in enumerate(pieces) if _is_corner_candidate(piece, corner, rect_col)]
    anchor_trials: list[int | None] = candidate_anchors + [None]

    def assemble(H: int, W: int, anchor_idx: int | None) -> np.ndarray | None:
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
            a = pieces[anchor_idx]
            ah, aw = a.shape
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

    ass = None
    used_dim = None
    for H, W in dims:
        seen_trials: set[int | None] = set()
        for anchor_idx in anchor_trials:
            if anchor_idx in seen_trials:
                continue
            seen_trials.add(anchor_idx)
            ass = assemble(H, W, anchor_idx)
            if ass is not None:
                used_dim = (H, W)
                break
        if ass is not None:
            break

    assert not (ass is None or used_dim is None)

    H, W = used_dim
    tr, tc = _top_left_from_corner(corner, mr, mc, H, W)
    out = np.full((GH, GW), int(bg), dtype=int)
    out[tr : tr + H, tc : tc + W] = ass
    return out


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
                            if 0 <= ar < iH and 0 <= ac < iW and blob_occupied[ar, ac] and (ar, ac) not in blob:
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

    fill_ratio = float(np.sum(interior != rect_col)) / float(iH * iW)
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
        ph0, pw0 = _pixel_size(mr0, mc0, mr1, mc1)
        if ph0 > 7 or pw0 > 7:
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
            if bb is None:
                return None
            r0i, c0i, r1i, c1i = bb
            pr = row_starts[r0i]
            pc = col_starts[c0i]
            ph0 = sum(row_segs[ii] for ii in range(r0i, r1i + 1))
            pw0 = sum(col_segs[jj] for jj in range(c0i, c1i + 1))
            pieces.append((pr, pc, ph0, pw0))

    if len(pieces) < 4 or len(pieces) > 12:
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
    for i, (pr, pc, ph0, pw0) in enumerate(pieces):
        if pr <= anchor_r < pr + ph0 and pc <= anchor_c < pc + pw0:
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

    candidate_indices = {anchor_idx}
    decoy_pool = []
    for i, box in enumerate(pieces):
        if i == anchor_idx:
            continue
        pr, pc, ph0, pw0 = box
        if ph0 < 2 or pw0 < 2:
            continue
        piece = rect[pr : pr + ph0, pc : pc + pw0]
        before_fill, _ = _piece_stats(piece, rect_col)
        if before_fill < 0.18:
            continue
        shadow = piece.copy()
        if corner == "TL":
            shadow[0, :] = rect_col
            shadow[:, 0] = rect_col
        elif corner == "TR":
            shadow[0, :] = rect_col
            shadow[:, -1] = rect_col
        elif corner == "BL":
            shadow[-1, :] = rect_col
            shadow[:, 0] = rect_col
        else:
            shadow[-1, :] = rect_col
            shadow[:, -1] = rect_col
        after_fill, has_non_rc_edge = _piece_stats(shadow, rect_col)
        if after_fill < 0.10 or not has_non_rc_edge:
            continue
        changed = int(np.sum(piece != shadow))
        decoy_pool.append((changed, -ph0 * pw0, i))

    if not decoy_pool:
        return None

    _r.shuffle(decoy_pool)
    decoy_pool.sort()
    target_candidates = min(len(decoy_pool) + 1, _r.choice([2, 2, 3]))
    for _changed, _neg_area, i in decoy_pool[: max(1, target_candidates - 1)]:
        candidate_indices.add(i)

    if len(candidate_indices) < 2:
        return None

    for i in candidate_indices:
        _force_candidate_corner(rect, pieces[i], corner, rect_col)

    piece_arrays = [rect[pr : pr + ph0, pc : pc + pw0].copy() for pr, pc, ph0, pw0 in pieces]

    actual_candidates = [i for i, piece in enumerate(piece_arrays) if _is_corner_candidate(piece, corner, rect_col)]
    if anchor_idx not in actual_candidates or len(actual_candidates) < 2:
        return None

    for i, (pr, pc, ph0, pw0) in enumerate(pieces):
        piece = piece_arrays[i]
        if ph0 < 2 or pw0 < 2 or ph0 > 7 or pw0 > 7:
            return None
        fill, has_non_rc_edge = _piece_stats(piece, rect_col)
        if fill < 0.10 or fill > 0.60:
            return None
        if not has_non_rc_edge:
            return None
        if pr == 0 and not np.all(piece[0, :] == rect_col):
            return None
        if pr + ph0 == RH and not np.all(piece[-1, :] == rect_col):
            return None
        if pc == 0 and not np.all(piece[:, 0] == rect_col):
            return None
        if pc + pw0 == RW and not np.all(piece[:, -1] == rect_col):
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

    order = list(range(len(piece_arrays)))
    _r.shuffle(order)
    for idx in order:
        pa = piece_arrays[idx]
        try:
            r0, c0 = random_free_location(canvas, pa, transparent=bg, padding=1)
            paste(canvas, pa, r0, c0)
        except ValueError:
            return None

    tr, tc = _top_left_from_corner(corner, mr, mc, RH, RW)
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
