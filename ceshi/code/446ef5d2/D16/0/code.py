# Variant of 446ef5d2
# Dimension: D16 — 锚定碎片的特殊性 (meta)
# Change: 恰好一块碎片包含标记缺失角对应的矩形顶点，该碎片作为锚定碎片优先放置在角落... → 无锚定碎片——所有碎片平等

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import paste


"""
Design analysis for D16 (no anchor piece)
=========================================

source_task:
  present: false
  action: skipped

a) concretization_and_trace:
  original_visual:
    - The marker is a 3-cell L whose missing cell indicates one corner of the target rectangle.
    - In the original design, one scattered piece could sit exactly against that missing cell,
      so the solver got a privileged "start here" clue.
  variant_visual:
    - The marker still indicates the rectangle corner.
    - No scattered piece is allowed to occupy or hug that marker corner area.
    - The marker is visually isolated by background, so it is only a locator, not a piece selector.
    - Several pieces can have the required rect_col corner-edges, so the solver must choose.
  flow_delta_example:
    - Original path: detect marker -> identify the one piece geometrically touching the marker gap
      -> pre-place that exact piece at the assembled corner -> backtrack the rest.
    - Variant path: detect marker -> compute rectangle size candidates -> for each size, try every
      piece whose two corner-facing outer edges are fully rect_col as the first corner piece -> backtrack.
    - Difference: the first placement is now a search branch, not a direct deduction.

b) complexity_calibration:
  - Removing the anchor without any other constraint would be too weak if the generator still often
    happened to leave one obvious candidate. To keep the change observable, the generator rejects
    samples unless at least two scattered pieces are corner-eligible for the marked corner.
  - The rest of the task stays unchanged: same rectangle assembly, same wildcard border color,
    same rectangular piece tiling, no rotations.

c) implicit_assumption_audit_on_original_main:
  - The old `anchor_idx` scan assumes a piece may be recognized from its current location relative
    to the marker. In this variant that assumption is false by construction.
  - The old pre-placement block assumes the first corner placement can be forced from input geometry.
    That is exactly the behavior we must remove.
  - Boundary matching, rectangle dimension search, and wildcard edge compatibility do not depend on
    anchor uniqueness and remain valid.

d) placement_legality:
  - Pieces are still scattered as non-overlapping rectangles with >=1 background gap.
  - A forbidden zone covering the 2x2 marker box plus a 1-cell moat prevents any piece from acting
    as an implicit anchor or visually touching the marker.
  - The output remains a single assembled rectangle pinned to the marker's missing cell.

e) visual_observability:
  - Across 2-4 examples, the human-visible difference is that the marker stands alone instead of
    already touching one specific piece.
  - Because at least two pieces are corner-eligible, the solver really has to infer assembly rather
    than read off a unique start piece from the input.

f) visual_cleanliness:
  - The moat around the marker keeps the marker region uncluttered.
  - Pieces remain axis-aligned rectangles with background spacing, so object boundaries stay clear.

g) color_inference:
  - No new color rule is introduced.
  - Colors keep their original roles: bg, marker, rect_col wildcard border, and 1-3 interior colors.

new_solver_assumptions_and_generator_guarantees:
  - Assumption: at least one piece can occupy the marked corner for the true rectangle size.
    Guarantee: the real corner piece always exists.
  - Assumption: trying several corner-eligible first pieces is tractable.
    Guarantee: piece count and sizes remain in the original 4-12 / 2-7 ranges.
  - Assumption: the marker region itself contains no anchor clue.
    Guarantee: scattering forbids overlap with the marker moat, so the old anchor relation never appears.
"""


def _find_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], str, tuple[int, int, int, int]]:
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
        if len(miss) != 1:
            continue
        missing = tuple(map(int, miss[0]))
        if missing == (r0, c0):
            corner = "BR"
        elif missing == (r0, c1):
            corner = "BL"
        elif missing == (r1, c0):
            corner = "TR"
        else:
            corner = "TL"
        return int(col), missing, corner, (r0, c0, r1, c1)
    raise AssertionError("marker not found")


def _extract_rects(g: np.ndarray, occ: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
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


def _top_left_from_corner_cell(corner: str, mr: int, mc: int, H: int, W: int) -> tuple[int, int]:
    if corner == "TL":
        return int(mr), int(mc)
    if corner == "TR":
        return int(mr), int(mc - W + 1)
    if corner == "BL":
        return int(mr - H + 1), int(mc)
    return int(mr - H + 1), int(mc - W + 1)


def _corner_origin(corner: str, H: int, W: int, h: int, w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, W - w
    if corner == "BL":
        return H - h, 0
    return H - h, W - w


def _piece_corner_candidate(piece: np.ndarray, corner: str, rect_col: int) -> bool:
    if corner == "TL":
        return bool(np.all(piece[0, :] == rect_col) and np.all(piece[:, 0] == rect_col))
    if corner == "TR":
        return bool(np.all(piece[0, :] == rect_col) and np.all(piece[:, -1] == rect_col))
    if corner == "BL":
        return bool(np.all(piece[-1, :] == rect_col) and np.all(piece[:, 0] == rect_col))
    return bool(np.all(piece[-1, :] == rect_col) and np.all(piece[:, -1] == rect_col))


def _marker_cells_from_corner(corner: str, mr: int, mc: int) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into a rectangle at an L-marker corner.

    D16 variant:
    - The marker identifies the rectangle corner only.
    - No scattered piece is privileged by current placement.
    - The solver must try all pieces that can legally occupy the marked corner.
    """
    g = np.array(grid, dtype=int)
    GH, GW = g.shape

    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, missing, corner, _mb = _find_marker(g, bg)

    occ = (g != int(bg)) & (g != int(mark))
    rects = _extract_rects(g, occ)
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
        tr, tc = _top_left_from_corner_cell(corner, mr, mc, H, W)
        return 0 <= tr and 0 <= tc and (tr + H) <= GH and (tc + W) <= GW

    dims = [(int(H), int(W)) for H, W in dims if fits(int(H), int(W))]
    dims = sorted(dims, key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    def compat(v1: np.ndarray, v2: np.ndarray) -> bool:
        v1 = np.array(v1, dtype=int)
        v2 = np.array(v2, dtype=int)
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
            if r < 0 or c < 0 or (r + h) > H or (c + w) > W:
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

        def next_empty() -> tuple[int, int] | None:
            idx = np.argwhere(occ2 == -1)
            if idx.size == 0:
                return None
            return tuple(map(int, idx[0]))

        solved = False

        def bt(started: bool) -> None:
            nonlocal solved
            if not started:
                for i in order:
                    r0, c0 = _corner_origin(corner, H, W, ph[i], pw[i])
                    if can_place(i, r0, c0):
                        place(i, r0, c0)
                        bt(True)
                        if solved:
                            return
                        unplace(i, r0, c0)
                return

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
                    bt(True)
                    if solved:
                        return
                    unplace(i, r, c)

        bt(False)
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
    tr, tc = _top_left_from_corner_cell(corner, mr, mc, H, W)
    out = np.full((GH, GW), int(bg), dtype=int)
    out[tr : tr + H, tc : tc + W] = ass
    return out


def _rand_splits(total: int, rng: random.Random, min_s: int = 2, max_s: int = 7) -> list[int] | None:
    eff_max = min(max_s, max(min_s, rng.randint(min_s, max_s)))
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
            s = rng.randint(min_s, hi)
            segs.append(s)
            rem -= s
    return segs


def _count_corner_candidates(piece_arrays: list[np.ndarray], corner: str, rect_col: int) -> int:
    return sum(1 for piece in piece_arrays if _piece_corner_candidate(piece, corner, rect_col))


def _random_free_location_with_forbidden(
    canvas: np.ndarray,
    patch: np.ndarray,
    *,
    bg: int,
    forbidden: np.ndarray,
    rng: random.Random,
    padding: int = 1,
) -> tuple[int, int]:
    H, W = canvas.shape
    h, w = patch.shape
    candidates: list[tuple[int, int]] = []
    occ = canvas != bg
    for r0 in range(0, H - h + 1):
        for c0 in range(0, W - w + 1):
            rr0 = max(0, r0 - padding)
            rr1 = min(H, r0 + h + padding)
            cc0 = max(0, c0 - padding)
            cc1 = min(W, c0 + w + padding)
            if bool(occ[rr0:rr1, cc0:cc1].any()):
                continue
            if bool(forbidden[r0 : r0 + h, c0 : c0 + w].any()):
                continue
            candidates.append((r0, c0))
    if not candidates:
        raise ValueError("no free location with forbidden zone")
    return rng.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    GH = rng.randint(10, 26)
    GW = rng.randint(10, 26)

    palette = list(range(10))
    rng.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    n_interior = rng.randint(1, 3)
    interior_cols = palette[3 : 3 + n_interior]

    corner = rng.choice(["TL", "TR", "BL", "BR"])

    min_side, max_side = 5, min(22, min(GH, GW) - 3)
    if max_side < min_side:
        return None
    RH = rng.randint(min_side, max_side)
    RW = rng.randint(min_side, max_side)
    if rng.random() < 0.5 and max_side >= 8:
        RH = rng.randint(max(6, min_side), max_side)
        RW = rng.randint(max(6, min_side), max_side)

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
    mr = rng.randint(mr_min, mr_max)
    mc = rng.randint(mc_min, mc_max)

    rect = np.full((RH, RW), rect_col, dtype=int)
    iH, iW = RH - 2, RW - 2
    if iH < 1 or iW < 1:
        return None

    interior = np.full((iH, iW), rect_col, dtype=int)
    target_fill = rng.uniform(0.35, 0.85)
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
            sr, sc = rng.randint(0, iH - 1), rng.randint(0, iW - 1)
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
            rng.shuffle(frontier)
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
    fill_ratio = non_rc_count / (iH * iW)
    if fill_ratio < 0.30 or fill_ratio > 0.90:
        return None
    rect[1:-1, 1:-1] = interior

    row_segs = _rand_splits(RH, rng)
    col_segs = _rand_splits(RW, rng)
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

    target_pieces = rng.randint(4, min(12, n_cells))
    merges = []
    for i in range(nr):
        for j in range(nc_grid):
            if j + 1 < nc_grid:
                merges.append(("h", i, j))
            if i + 1 < nr:
                merges.append(("v", i, j))
    rng.shuffle(merges)

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
            bb = get_piece_bbox(pid)
            assert bb is not None
            r0i, c0i, r1i, c1i = bb
            pr = row_starts[r0i]
            pc = col_starts[c0i]
            ph = sum(row_segs[ii] for ii in range(r0i, r1i + 1))
            pw = sum(col_segs[jj] for jj in range(c0i, c1i + 1))
            pieces.append((pr, pc, ph, pw))

    if len(pieces) < 4 or len(pieces) > 12:
        return None

    piece_arrays = []
    for pr, pc, ph, pw in pieces:
        if ph < 2 or pw < 2 or ph > 7 or pw > 7:
            return None
        piece = rect[pr:pr + ph, pc:pc + pw].copy()
        nrc = int(np.sum(piece != rect_col))
        fill = nrc / (ph * pw)
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
        piece_arrays.append(piece)

    # D16 visibility guard: the marker should not imply a unique starting piece.
    if _count_corner_candidates(piece_arrays, corner, rect_col) < 2:
        return None

    marker_cells = _marker_cells_from_corner(corner, mr, mc)
    for cr, cc in marker_cells:
        if cr < 0 or cr >= GH or cc < 0 or cc >= GW:
            return None

    canvas = np.full((GH, GW), bg, dtype=int)
    for cr, cc in marker_cells:
        canvas[cr, cc] = marker_col

    # The 2x2 marker box plus a 1-cell moat must stay empty of pieces.
    forbidden = np.zeros((GH, GW), dtype=bool)
    br0 = max(0, mr - 2)
    br1 = min(GH, mr + 3)
    bc0 = max(0, mc - 2)
    bc1 = min(GW, mc + 3)
    forbidden[br0:br1, bc0:bc1] = True

    order = list(range(len(piece_arrays)))
    rng.shuffle(order)
    for idx in order:
        pa = piece_arrays[idx]
        try:
            r0, c0 = _random_free_location_with_forbidden(
                canvas,
                pa,
                bg=bg,
                forbidden=forbidden,
                rng=rng,
                padding=1,
            )
            paste(canvas, pa, r0, c0)
        except ValueError:
            return None

    tr, tc = _top_left_from_corner_cell(corner, mr, mc, RH, RW)
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
