# Variant of 446ef5d2
# Dimension: D14 — 碎片的空间分布规则 (meta)
# Change: 碎片随机散布在背景上，彼此不重叠，间距至少 1 个背景像素... → 碎片可以相邻（零间距）

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, paste, random_free_location


# Design analysis for the D14 variant
# -----------------------------------
# a) Concrete behavior:
#    The original task shows separated rectangular fragments on the background.
#    Here at least one pair of fragments is allowed to touch edge-to-edge with
#    zero background gap. To keep the D14 change visible but not turn the task
#    into a completely different segmentation puzzle, generation forces exactly
#    one 4-connected "touching cluster" made of two rectangles whose union is
#    non-rectangular. All other fragments remain isolated, and the marker logic,
#    colors, rectangle assembly rule, and wildcard rect_col rule stay unchanged.
#    Small trace:
#    - Original: two scattered 3x3 pieces must have at least one bg cell between
#      them, so the solver can scan non-bg cells and each piece is already a
#      separate rectangular component.
#    - Variant: one 3x3 piece may touch another 2x4 piece along part of one side,
#      producing one 4-connected L/step-shaped occupied component. The solver
#      must first split that non-rectangular component back into two rectangles,
#      then run the original rectangle-assembly search.
#    - Difference: the assembly stage is unchanged; only piece extraction from
#      the input canvas changes.
# b) Complexity calibration:
#    Allowing arbitrary flush contacts between many pieces would create hidden
#    seams and frequently merge pieces into larger perfect rectangles, which is
#    a qualitatively harder object-partition problem and no longer a focused D14
#    perturbation. This variant therefore uses a single touching pair whose union
#    is visibly non-rectangular. The effect is easy to notice in 2-4 examples,
#    but still requires connected-component reasoning beyond the original task.
# c) Implicit assumptions in the original main():
#    - `occ` + greedy width/height growth assumed every piece is surrounded by bg.
#      This fails once two pieces touch, because the greedy rectangle can absorb
#      pixels from both pieces.
#    - `anchor_idx` lookup over those greedy rectangles is only valid if the
#      decomposition is correct.
#    - The backtracking assembler itself does not depend on the spacing rule; it
#      only needs the correct list of rectangular piece arrays.
#    New solver behavior:
#    - Label 4-connected occupied components.
#    - A rectangular component is still a single piece.
#    - A non-rectangular component is assumed to be exactly two rectangles; the
#      solver enumerates rectangle-removal splits whose remainder is also a full
#      rectangle, then tries assembly on each partition option.
# d) Placement legality:
#    Generator explicitly places the anchor piece at the marker gap corner, then
#    places one composite 2-piece cluster as a single patch, then places all
#    remaining pieces with padding=1 so no accidental extra touching occurs.
#    This guarantees no overlap, no out-of-bounds placement, and exactly one
#    zero-gap touching event.
# e) Visual observability:
#    The touching cluster is a 4-connected non-rectangular mass, so a human can
#    see that the variant differs from the original "all pieces separated" rule.
#    The cluster is not allowed to collapse into a bigger rectangle, because that
#    would hide the D14 change rather than reveal it.
# f) Visual cleanliness:
#    Only one touching cluster is introduced; the rest of the board remains clean.
#    The marker stays isolated from non-anchor pieces, so the L marker is still
#    easy to spot and still clearly indicates the assembly corner.
# g) Color inferability:
#    No extra color rule is introduced. Colors keep the original semantics, so
#    D14 is the only conceptual change.
# Assumption review added by the new solver:
#    - Any non-rectangular occupied 4-component consists of exactly two touching
#      rectangles. The generator guarantees this by constructing exactly one such
#      cluster and preventing any accidental additional contacts.


def _find_marker(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int], tuple[int, int, int, int], str]:
    cnt = Counter(map(int, grid.flatten()))
    for col in sorted([c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3]):
        pts = [(int(r), int(c)) for r, c in np.argwhere(grid == int(col))]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0) == 1 and (c1 - c0) == 1:
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
            return int(col), missing, (r0, c0, r1, c1), corner
    raise AssertionError("marker not found")


def _piece_valid(piece: np.ndarray, rect_col: int) -> bool:
    h, w = map(int, piece.shape)
    if h < 2 or w < 2 or h > 7 or w > 7:
        return False
    nrc = int(np.sum(piece != int(rect_col)))
    fill = nrc / (h * w)
    if fill < 0.10 or fill > 0.60:
        return False
    for edge in (piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]):
        if np.any(edge != int(rect_col)):
            return True
    return False


def _rect_sum(ps: np.ndarray, r0: int, c0: int, h: int, w: int) -> int:
    r1 = r0 + h
    c1 = c0 + w
    return int(ps[r1, c1] - ps[r0, c1] - ps[r1, c0] + ps[r0, c0])


def _component_partition_options(
    grid: np.ndarray, occ: np.ndarray, rect_col: int
) -> list[list[tuple[int, int, int, int, np.ndarray]]]:
    coords = np.argwhere(occ)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    sub_occ = occ[r0 : r1 + 1, c0 : c1 + 1]
    sub_grid = grid[r0 : r1 + 1, c0 : c1 + 1]
    area = int(sub_occ.sum())
    Hc, Wc = sub_occ.shape

    if area == Hc * Wc:
        piece = sub_grid.copy()
        if _piece_valid(piece, rect_col):
            return [[(r0, c0, Hc, Wc, piece)]]
        return []

    ps = np.pad(sub_occ.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    options = []
    seen = set()

    for rr in range(Hc):
        for cc in range(Wc):
            if not bool(sub_occ[rr, cc]):
                continue
            for h in range(2, min(7, Hc - rr) + 1):
                for w in range(2, min(7, Wc - cc) + 1):
                    a1 = h * w
                    if a1 >= area:
                        continue
                    if _rect_sum(ps, rr, cc, h, w) != a1:
                        continue
                    rem = sub_occ.copy()
                    rem[rr : rr + h, cc : cc + w] = False
                    rem_area = int(rem.sum())
                    if rem_area < 4:
                        continue
                    rem_coords = np.argwhere(rem)
                    rr0, cc0 = rem_coords.min(axis=0)
                    rr1, cc1 = rem_coords.max(axis=0)
                    rr0 = int(rr0)
                    cc0 = int(cc0)
                    rr1 = int(rr1)
                    cc1 = int(cc1)
                    hh = rr1 - rr0 + 1
                    ww = cc1 - cc0 + 1
                    if rem_area != hh * ww:
                        continue
                    if hh < 2 or ww < 2 or hh > 7 or ww > 7:
                        continue
                    piece1 = sub_grid[rr : rr + h, cc : cc + w].copy()
                    piece2 = sub_grid[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
                    if not (_piece_valid(piece1, rect_col) and _piece_valid(piece2, rect_col)):
                        continue
                    rects = [
                        (r0 + rr, c0 + cc, h, w, piece1),
                        (r0 + rr0, c0 + cc0, hh, ww, piece2),
                    ]
                    rects = sorted(rects, key=lambda t: (t[0], t[1], t[2], t[3]))
                    key = tuple((r, c, h0, w0) for r, c, h0, w0, _ in rects)
                    if key in seen:
                        continue
                    seen.add(key)
                    options.append(rects)

    return options


def _extract_piece_options(
    grid: np.ndarray, bg: int, mark: int, rect_col: int
) -> list[list[tuple[int, int, int, int, np.ndarray]]]:
    occ = (grid != int(bg)) & (grid != int(mark))
    lab, n = label(occ.astype(int), structure=STRUCT4)
    per_component = []
    for k in range(1, int(n) + 1):
        comp = lab == k
        opts = _component_partition_options(grid, comp, rect_col)
        if not opts:
            return []
        per_component.append(opts)
    return [list(itertools.chain.from_iterable(choice)) for choice in itertools.product(*per_component)]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, missing, _mb, corner = _find_marker(g, bg)
    mr, mc = map(int, missing)

    occ_pixels = g[(g != int(bg)) & (g != int(mark))]
    rect_col = int(Counter(map(int, occ_pixels.flatten())).most_common(1)[0][0])

    piece_options = _extract_piece_options(g, bg, mark, rect_col)
    assert piece_options

    dims_cache: dict[int, list[tuple[int, int]]] = {}

    def compat(v1, v2):
        v1 = np.array(v1, dtype=int)
        v2 = np.array(v2, dtype=int)
        m = (v1 != int(rect_col)) & (v2 != int(rect_col))
        return bool((v1[m] == v2[m]).all())

    def assemble(rects: list[tuple[int, int, int, int, np.ndarray]], H: int, W: int):
        pieces = [a for _r, _c, _h, _w, a in rects]
        ph = [int(p.shape[0]) for p in pieces]
        pw = [int(p.shape[1]) for p in pieces]
        occ2 = np.full((H, W), -1, dtype=int)
        usedp = [False] * len(pieces)

        anchor_idx = None
        for i, (rr, cc, h, w, _a) in enumerate(rects):
            if rr <= mr < rr + h and cc <= mc < cc + w:
                anchor_idx = i
                break

        def boundary_ok(a, r, c):
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

        def neighbor_ok(a, r, c):
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

        def can_place(i, r, c):
            a = pieces[i]
            h, w = a.shape
            if (r + h) > H or (c + w) > W:
                return False
            if (occ2[r : r + h, c : c + w] != -1).any():
                return False
            return boundary_ok(a, r, c) and neighbor_ok(a, r, c)

        def place(i, r, c):
            a = pieces[i]
            h, w = a.shape
            occ2[r : r + h, c : c + w] = a
            usedp[i] = True

        def unplace(i, r, c):
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

        def next_empty():
            idx = np.argwhere(occ2 == -1)
            if idx.size == 0:
                return None
            return tuple(map(int, idx[0]))

        solved = False

        def bt():
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

    for rects in piece_options:
        total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))
        if total_area not in dims_cache:
            dims = []
            for hh in range(1, int(total_area**0.5) + 1):
                if total_area % hh == 0:
                    dims.append((hh, total_area // hh))
                    if hh * hh != total_area:
                        dims.append((total_area // hh, hh))

            def fits(H, W):
                if corner == "TL":
                    tr, tc = mr, mc
                elif corner == "TR":
                    tr, tc = mr, mc - W + 1
                elif corner == "BL":
                    tr, tc = mr - H + 1, mc
                else:
                    tr, tc = mr - H + 1, mc - W + 1
                return 0 <= tr and 0 <= tc and (tr + H) <= GH and (tc + W) <= GW

            dims_cache[total_area] = sorted(
                [(int(H), int(W)) for H, W in dims if fits(int(H), int(W))],
                key=lambda t: (-t[0] * t[1], -t[0], -t[1]),
            )

        for H, W in dims_cache[total_area]:
            ass = assemble(rects, H, W)
            if ass is None:
                continue
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

    raise AssertionError("assembly failed")


def _rand_splits(total: int, min_s: int = 2, max_s: int = 7) -> list[int] | None:
    eff_max = min(max_s, max(min_s, random.randint(min_s, max_s)))
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
            s = random.randint(min_s, hi)
            segs.append(s)
            rem -= s
    return segs


def _build_rect(RH: int, RW: int, rect_col: int, interior_cols: list[int]) -> np.ndarray | None:
    rect = np.full((RH, RW), rect_col, dtype=int)
    iH, iW = RH - 2, RW - 2
    if iH < 1 or iW < 1:
        return None

    interior = np.full((iH, iW), rect_col, dtype=int)
    target_fill = random.uniform(0.35, 0.85)
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
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = fr + dr, fc + dc
                    if (nr, nc) in blob or nr < 0 or nr >= iH or nc < 0 or nc >= iW:
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

    fill_ratio = float(np.mean(interior != rect_col))
    if fill_ratio < 0.30 or fill_ratio > 0.90:
        return None

    rect[1:-1, 1:-1] = interior
    return rect


def _partition_rect(rect: np.ndarray, rect_col: int) -> list[tuple[int, int, int, int]] | None:
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

    target_pieces = random.randint(4, min(12, n_cells))

    def get_piece_bbox(pid):
        cells = [(i, j) for i in range(nr) for j in range(nc_grid) if cell_id[i][j] == pid]
        if not cells:
            return None
        ri = [c[0] for c in cells]
        ci = [c[1] for c in cells]
        return min(ri), min(ci), max(ri), max(ci)

    def pixel_size(r0i, c0i, r1i, c1i):
        h = sum(row_segs[i] for i in range(r0i, r1i + 1))
        w = sum(col_segs[j] for j in range(c0i, c1i + 1))
        return h, w

    merges = []
    for i in range(nr):
        for j in range(nc_grid):
            if j + 1 < nc_grid:
                merges.append(("h", i, j))
            if i + 1 < nr:
                merges.append(("v", i, j))
    random.shuffle(merges)

    def count_pieces():
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
            r0i, c0i, r1i, c1i = bb
            pr = row_starts[r0i]
            pc = col_starts[c0i]
            ph = sum(row_segs[ii] for ii in range(r0i, r1i + 1))
            pw = sum(col_segs[jj] for jj in range(c0i, c1i + 1))
            piece = rect[pr : pr + ph, pc : pc + pw]
            if not _piece_valid(piece, rect_col):
                return None
            if pr == 0 and not np.all(piece[0, :] == rect_col):
                return None
            if pr + ph == RH and not np.all(piece[-1, :] == rect_col):
                return None
            if pc == 0 and not np.all(piece[:, 0] == rect_col):
                return None
            if pc + pw == RW and not np.all(piece[:, -1] == rect_col):
                return None
            pieces.append((pr, pc, ph, pw))

    if len(pieces) < 4 or len(pieces) > 12:
        return None
    return pieces


def _marker_cells(corner: str, mr: int, mc: int) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _anchor_top_left(corner: str, mr: int, mc: int, h: int, w: int) -> tuple[int, int]:
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - w + 1
    if corner == "BL":
        return mr - h + 1, mc
    return mr - h + 1, mc - w + 1


def _cluster_variants(a: np.ndarray, b: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]]:
    h1, w1 = map(int, a.shape)
    h2, w2 = map(int, b.shape)
    variants = []

    def add_variant(r2: int, c2: int):
        min_r = min(0, r2)
        min_c = min(0, c2)
        r1n, c1n = -min_r, -min_c
        r2n, c2n = r2 - min_r, c2 - min_c
        H = max(r1n + h1, r2n + h2)
        W = max(c1n + w1, c2n + w2)
        canvas = np.full((H, W), -1, dtype=int)
        canvas[r1n : r1n + h1, c1n : c1n + w1] = a
        overlap = canvas[r2n : r2n + h2, c2n : c2n + w2] != -1
        if overlap.any():
            return
        canvas[r2n : r2n + h2, c2n : c2n + w2] = b
        occ = canvas != -1
        if int(occ.sum()) != h1 * w1 + h2 * w2:
            return
        rr = np.argwhere(occ)
        bh = int(rr[:, 0].max() - rr[:, 0].min() + 1)
        bw = int(rr[:, 1].max() - rr[:, 1].min() + 1)
        if bh * bw == int(occ.sum()):
            return
        variants.append((canvas, (r1n, c1n), (r2n, c2n)))

    for c2 in range(-(w2 - 1), w1):
        left = max(0, c2)
        right = min(w1, c2 + w2)
        if left < right:
            add_variant(h1, c2)
            add_variant(-h2, c2)

    for r2 in range(-(h2 - 1), h1):
        top = max(0, r2)
        bottom = min(h1, r2 + h2)
        if top < bottom:
            add_variant(r2, w1)
            add_variant(r2, -w2)

    uniq = []
    seen = set()
    for canvas, p1, p2 in variants:
        key = (canvas.shape, p1, p2, tuple(map(int, canvas.flatten())))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((canvas, p1, p2))
    return uniq


def _gen_attempt() -> np.ndarray | None:
    GH = random.randint(10, 26)
    GW = random.randint(10, 26)

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    n_interior = random.randint(1, 3)
    interior_cols = palette[3 : 3 + n_interior]

    corner = random.choice(["TL", "TR", "BL", "BR"])
    min_side, max_side = 5, min(22, min(GH, GW) - 3)
    if max_side < min_side:
        return None
    RH = random.randint(min_side, max_side)
    RW = random.randint(min_side, max_side)
    if random.random() < 0.5 and max_side >= 8:
        RH = random.randint(max(6, min_side), max_side)
        RW = random.randint(max(6, min_side), max_side)

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
    mr = random.randint(mr_min, mr_max)
    mc = random.randint(mc_min, mc_max)

    rect = _build_rect(RH, RW, rect_col, interior_cols)
    if rect is None:
        return None

    pieces = _partition_rect(rect, rect_col)
    if pieces is None:
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

    marker_cells = _marker_cells(corner, mr, mc)
    for cr, cc in marker_cells:
        if cr < 0 or cr >= GH or cc < 0 or cc >= GW:
            return None

    piece_arrays = [rect[pr : pr + ph, pc : pc + pw].copy() for pr, pc, ph, pw in pieces]

    remaining = [i for i in range(len(piece_arrays)) if i != anchor_idx]
    if len(remaining) < 2:
        return None
    pair_candidates = remaining[:]
    random.shuffle(pair_candidates)
    pair = None
    cluster_patch = None
    for i in pair_candidates:
        for j in pair_candidates:
            if i == j:
                continue
            variants = _cluster_variants(piece_arrays[i], piece_arrays[j])
            if variants:
                pair = (i, j)
                cluster_patch = random.choice(variants)
                break
        if pair is not None:
            break
    if pair is None or cluster_patch is None:
        return None

    canvas = np.full((GH, GW), bg, dtype=int)
    for cr, cc in marker_cells:
        canvas[cr, cc] = marker_col

    anchor_piece = piece_arrays[anchor_idx]
    ar0, ac0 = _anchor_top_left(corner, mr, mc, aph, apw)
    if ar0 < 0 or ac0 < 0 or ar0 + aph > GH or ac0 + apw > GW:
        return None
    if np.any(canvas[ar0 : ar0 + aph, ac0 : ac0 + apw] != bg):
        return None
    paste(canvas, anchor_piece, ar0, ac0)

    cluster_arr, _p1, _p2 = cluster_patch
    cluster_patch_bg = cluster_arr.copy()
    cluster_patch_bg[cluster_patch_bg == -1] = bg
    try:
        cr0, cc0 = random_free_location(canvas, cluster_patch_bg, transparent=bg, padding=1)
    except ValueError:
        return None
    paste(canvas, cluster_patch_bg, cr0, cc0)

    pair_set = {pair[0], pair[1], anchor_idx}
    rest = [i for i in range(len(piece_arrays)) if i not in pair_set]
    random.shuffle(rest)
    for idx in rest:
        pa = piece_arrays[idx]
        try:
            r0, c0 = random_free_location(canvas, pa, transparent=bg, padding=1)
        except ValueError:
            return None
        paste(canvas, pa, r0, c0)

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

    occ = (canvas != int(bg)) & (canvas != int(marker_col))
    lab, n = label(occ.astype(int), structure=STRUCT4)
    nonrect_components = 0
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == k)
        rr0, cc0 = coords.min(axis=0)
        rr1, cc1 = coords.max(axis=0)
        area = int(coords.shape[0])
        if area != int((rr1 - rr0 + 1) * (cc1 - cc0 + 1)):
            nonrect_components += 1
    if nonrect_components != 1:
        return None

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
