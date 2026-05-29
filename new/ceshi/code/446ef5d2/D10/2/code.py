# Variant of 446ef5d2
# Dimension: D10 — 矩形外边缘约束 (meta)
# Change: 组装矩形的四条外边必须全部由 rect_col 组成（纯色边框）... → 仅两条相邻外边为 rect_col

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import paste, random_free_location


# Design analysis for the D10 variant
# -----------------------------------
# spec_variant:
#   core_change:
#     original: all four outer edges of the assembled rectangle are fully rect_col
#     variant: only the two outer edges adjacent to the marker corner are forced to be rect_col
#     concretization:
#       - TL marker corner  -> top and left edges are fully rect_col
#       - TR marker corner  -> top and right edges are fully rect_col
#       - BL marker corner  -> bottom and left edges are fully rect_col
#       - BR marker corner  -> bottom and right edges are fully rect_col
#       - the other two edges are intentionally allowed to contain colored pixels
#   visibility_guardrail:
#     - If the generator merely "allows" colored pixels on the free edges, many samples would still
#       accidentally look like the original task.
#     - To make D10 observable and to reduce ambiguity during assembly, the generator forces both
#       unconstrained outer edges to contain at least one non-rect_col segment.
#     - The solver therefore uses a completion check that requires both unconstrained edges to be
#       non-uniform (not all rect_col). This is a deliberate tightening used only to keep the
#       variant visible and avoid regression to the original four-edge case.
#   behavior_trace:
#     small_example:
#       - Suppose the marker indicates corner TL.
#       - Original logic: any candidate assembly with a colored pixel on the bottom edge fails
#         immediately because boundary_ok requires all four borders to be rect_col.
#       - Variant logic: bottom/right are now free edges, so a colored strip on the bottom or
#         right boundary is legal, while top/left must stay pure rect_col.
#       - Visual difference: the solved rectangle still attaches to the marker corner, but only
#         the two edges touching that corner are clean rect_col borders; the opposite sides can
#         show interior colors leaking to the boundary.
#   complexity_calibration:
#     - The free edges are not made fully arbitrary. They only gain short colored bands, so the
#       rectangle still looks like a coherent assembly puzzle rather than a noisy mosaic.
#     - Corner-dependent asymmetry is repeated across every sample, so humans can infer that the
#       marker does more than pin location: it also identifies which two borders must stay pure.
#   implicit_assumption_review_main:
#     - Marker detection still assumes a unique 3-cell L color. Unchanged and still guaranteed.
#     - Piece extraction still assumes scattered objects are axis-aligned rectangles. Generator
#       preserves that exactly.
#     - rect_col inference as the most common piece color still works because constrained borders
#       plus much of the interior remain rect_col-dominant.
#     - boundary_ok in the original code depended on "all four sides pure". This is the key line
#       that must change: each side check now depends on the marker-derived constrained-side set.
#     - The original anchor pre-placement branch is effectively optional because scattered pieces
#       usually do not cover the marker's missing cell in input coordinates. It remains harmless.
#     - With weaker border constraints, wrong factor pairs become more plausible. To counter that,
#       the solver only accepts completed assemblies whose two unconstrained outer edges both
#       contain some non-rect_col pixels, matching the generator's construction.
#   placement_legality:
#     - Colored bands are painted only on unconstrained sides and never overwrite cells that lie
#       on constrained sides, including the mixed corners shared by a constrained and free edge.
#     - Pieces are still scattered with >=1 background padding via random_free_location.
#     - The output rectangle placement remains corner-anchored exactly as in the original task.
#   visual_cleanliness:
#     - The marker stays separate from pieces.
#     - Pieces remain rectangular blocks, so the puzzle structure is still easy to parse.
#     - Free-edge color bands are contiguous strips, not isolated random pixels, keeping the
#       contrast against the constrained borders visually legible.
#   new_solver_assumptions_to_match_generator:
#     - Both unconstrained outer edges contain at least one non-rect_col pixel.
#     - All constrained edges are fully rect_col.
#     - No rotation or reflection of pieces is used.


def _corner_side_sets(corner: str) -> tuple[set[str], set[str]]:
    if corner == "TL":
        return {"top", "left"}, {"bottom", "right"}
    if corner == "TR":
        return {"top", "right"}, {"bottom", "left"}
    if corner == "BL":
        return {"bottom", "left"}, {"top", "right"}
    return {"bottom", "right"}, {"top", "left"}


def _edge_values(arr: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return arr[0, :]
    if side == "bottom":
        return arr[-1, :]
    if side == "left":
        return arr[:, 0]
    return arr[:, -1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into a rectangle at an L-marker corner.

    Variant rule:
    - rect_col remains the wildcard color for interior edge matching.
    - Only the two outer edges adjacent to the marker corner must be fully rect_col.
    - The opposite two outer edges are required by this generator to expose some non-rect_col
      pixels, so the asymmetry is visible and can guide candidate selection.
    """
    g = np.array(grid, dtype=int)
    GH, GW = g.shape

    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

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
                missing = miss[0]
                mb = (r0, c0, r1, c1)
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

    pure_sides, free_sides = _corner_side_sets(corner)

    occ = (g != int(bg)) & (g != int(mark))
    used = np.zeros_like(occ, dtype=bool)
    rects = []
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

    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))

    allpix = []
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

    def compat(v1, v2) -> bool:
        v1 = np.array(v1, dtype=int)
        v2 = np.array(v2, dtype=int)
        m = (v1 != int(rect_col)) & (v2 != int(rect_col))
        return bool((v1[m] == v2[m]).all())

    def edge_score(arr: np.ndarray) -> int:
        score = 0
        for side in free_sides:
            score += int(np.sum(_edge_values(arr, side) != int(rect_col)))
        return score

    def assemble(H: int, W: int) -> np.ndarray | None:
        occ2 = np.full((H, W), -1, dtype=int)
        usedp = [False] * len(pieces)

        def boundary_ok(a: np.ndarray, r: int, c: int) -> bool:
            h, w = a.shape
            if "top" in pure_sides and r == 0 and not np.all(a[0, :] == int(rect_col)):
                return False
            if "bottom" in pure_sides and (r + h) == H and not np.all(a[-1, :] == int(rect_col)):
                return False
            if "left" in pure_sides and c == 0 and not np.all(a[:, 0] == int(rect_col)):
                return False
            if "right" in pure_sides and (c + w) == W and not np.all(a[:, -1] == int(rect_col)):
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

        def final_edges_ok() -> bool:
            if np.any(occ2 == -1):
                return False
            for side in pure_sides:
                if not np.all(_edge_values(occ2, side) == int(rect_col)):
                    return False
            for side in free_sides:
                if np.all(_edge_values(occ2, side) == int(rect_col)):
                    return False
            return True

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

        def bt() -> None:
            nonlocal solved
            nxt = next_empty()
            if nxt is None:
                if final_edges_ok():
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
    best_score = -1
    for H, W in dims:
        cand = assemble(H, W)
        if cand is None:
            continue
        score = edge_score(cand)
        if score > best_score:
            best_score = score
            ass = cand
            used_dim = (H, W)

    assert not (ass is None or used_dim is None)

    H, W = used_dim
    if corner == "TL":
        top_left = (mr, mc)
    elif corner == "TR":
        top_left = (mr, mc - W + 1)
    elif corner == "BL":
        top_left = (mr - H + 1, mc)
    else:
        top_left = (mr - H + 1, mc - W + 1)

    out = np.full((GH, GW), int(bg), dtype=int)
    tr, tc = map(int, top_left)
    out[tr : tr + H, tc : tc + W] = ass
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _paint_band(
    rect: np.ndarray,
    reserved: np.ndarray,
    side: str,
    color: int,
    pure_sides: set[str],
    rng,
) -> bool:
    RH, RW = rect.shape
    if side in ("top", "bottom"):
        usable = RW - (1 if "left" in pure_sides else 0) - (1 if "right" in pure_sides else 0)
        if usable < 2:
            return False
        length = rng.randint(2, min(5, usable))
        depth = rng.randint(2, min(3, RH - 1))
        c_min = 1 if "left" in pure_sides else 0
        c_max = RW - length - (1 if "right" in pure_sides else 0)
        if c_min > c_max:
            return False
        starts = list(range(c_min, c_max + 1))
        rng.shuffle(starts)
        for start in starts:
            cols = slice(start, start + length)
            rows = slice(0, depth) if side == "top" else slice(RH - depth, RH)
            if reserved[rows, cols].any():
                continue
            rect[rows, cols] = int(color)
            reserved[rows, cols] = True
            return True
        return False

    usable = RH - (1 if "top" in pure_sides else 0) - (1 if "bottom" in pure_sides else 0)
    if usable < 2:
        return False
    length = rng.randint(2, min(5, usable))
    depth = rng.randint(2, min(3, RW - 1))
    r_min = 1 if "top" in pure_sides else 0
    r_max = RH - length - (1 if "bottom" in pure_sides else 0)
    if r_min > r_max:
        return False
    starts = list(range(r_min, r_max + 1))
    rng.shuffle(starts)
    for start in starts:
        rows = slice(start, start + length)
        cols = slice(0, depth) if side == "left" else slice(RW - depth, RW)
        if reserved[rows, cols].any():
            continue
        rect[rows, cols] = int(color)
        reserved[rows, cols] = True
        return True
    return False


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
    pure_sides, free_sides = _corner_side_sets(corner)

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

    target_fill = _r.uniform(0.35, 0.80)
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

    reserved = np.zeros((RH, RW), dtype=bool)
    for side in free_sides:
        color = _r.choice(interior_cols)
        if not _paint_band(rect, reserved, side, color, pure_sides, _r):
            return None

    for side in pure_sides:
        if not np.all(_edge_values(rect, side) == rect_col):
            return None
    for side in free_sides:
        if np.all(_edge_values(rect, side) == rect_col):
            return None

    def _rand_splits(total: int, min_s: int = 2, max_s: int = 7):
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

    def _get_piece_bbox(pid: int):
        cells = [(i, j) for i in range(nr) for j in range(nc_grid) if cell_id[i][j] == pid]
        if not cells:
            return None
        ri = [c[0] for c in cells]
        ci = [c[1] for c in cells]
        return min(ri), min(ci), max(ri), max(ci)

    def _pixel_size(r0i: int, c0i: int, r1i: int, c1i: int):
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

    pieces = []
    seen = set()
    for i in range(nr):
        for j in range(nc_grid):
            pid = cell_id[i][j]
            if pid in seen:
                continue
            seen.add(pid)
            bb = _get_piece_bbox(pid)
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
        if fill < 0.10 or fill > 0.65:
            return None
        has_non_rc_edge = False
        for edge in [piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]]:
            if np.any(edge != rect_col):
                has_non_rc_edge = True
                break
        if not has_non_rc_edge:
            return None
        if pr == 0 and "top" in pure_sides and not np.all(piece[0, :] == rect_col):
            return None
        if pr + ph == RH and "bottom" in pure_sides and not np.all(piece[-1, :] == rect_col):
            return None
        if pc == 0 and "left" in pure_sides and not np.all(piece[:, 0] == rect_col):
            return None
        if pc + pw == RW and "right" in pure_sides and not np.all(piece[:, -1] == rect_col):
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

    piece_arrays = []
    for pr, pc, ph, pw in pieces:
        piece_arrays.append(rect[pr : pr + ph, pc : pc + pw].copy())

    order = list(range(len(piece_arrays)))
    _r.shuffle(order)
    for idx in order:
        pa = piece_arrays[idx]
        try:
            r0, c0 = random_free_location(canvas, pa, transparent=bg, padding=1)
            paste(canvas, pa, r0, c0)
        except ValueError:
            return None

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
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
