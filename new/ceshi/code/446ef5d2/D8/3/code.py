from __future__ import annotations

# Variant of 446ef5d2
# Dimension: D8 — 标记系统的形式 (symbol_instructions)
# Change: 3 像素 L 形标记（2x2 缺一角），使用独特颜色（仅出现 3 次），缺失角指示矩形的一个顶点... → 边线标记指示两条边
#
# Design analysis for the D8 variant
# ==================================
# a) Concrete behavior trace
# - Original marker: one 3-cell L shape, and the empty cell in its 2x2 bbox is the
#   target rectangle corner. Dimensions are recovered indirectly by factor-pair search.
# - New marker: the marker color forms exactly two disconnected 1-cell-thick line
#   segments, one horizontal and one vertical. They are outside the future rectangle
#   but collinear with one horizontal edge and one vertical edge of that rectangle.
# - The horizontal segment length equals the target width W.
# - The vertical segment length equals the target height H.
# - The solver extends the horizontal row and vertical column mentally: their
#   intersection is the corner cell. Example for a TL corner:
#   marker cells on row r at columns [c-W, ..., c-1] and on column c at rows
#   [r-H, ..., r-1]. Their supporting lines intersect at (r, c), which becomes the
#   rectangle's top-left cell, and the segment lengths directly give (H, W).
# - The rest of the task is unchanged: scattered rectangular pieces are translated
#   without rotation and assembled so touching edges match, with rect_col as wildcard.
#
# b) Complexity calibration
# - The marker is still a single distinct color, but visually much larger than the
#   original 3-pixel L and clearly decomposes into one horizontal and one vertical cue.
# - Lengths are not free-form metadata: they repeat the rectangle side lengths, so the
#   clue is strong and human-visible in 2-4 examples.
# - Side lengths are kept in a moderate range (5-11) so the line segments remain
#   "short" while still being easy to count and different from the original marker.
#
# c) Old solver assumptions that break under the variant
# - The original loop searched for a non-bg color appearing exactly 3 times and forming
#   a 2x2-minus-1 L. That no longer holds.
# - The original `missing` cell logic mapped the absent L-corner to TL/TR/BL/BR. Here
#   there is no missing pixel; corner inference comes from relative placement of the two
#   line segments.
# - The original dimension search enumerated factor pairs because the marker gave only a
#   corner. In this variant the marker lengths fix H and W directly, so factor search is
#   unnecessary and would reintroduce ambiguity.
# - The original rectangle decomposition ignored connectivity and simply peeled maximal
#   rectangles from the mask. For the new marker detector we need a stricter test:
#   removing the true marker color must leave only full rectangular connected components.
#
# d) Placement legality guaranteed by the generator
# - The rectangle is created first, then partitioned into rectangular pieces as before.
# - The chosen corner cell leaves room for both the rectangle and the two marker
#   segments on the exterior side of that corner.
# - Pieces are scattered with padding=1 after the marker is drawn, so pieces never
#   overlap the marker and remain visually separable.
#
# e) Visual observability
# - Humans can now read two facts directly from the input: which row/column become the
#   rectangle edges, and how long each side must be. This is visible from raw geometry,
#   not color ids or hidden ordering.
# - The four corner cases remain active (TL/TR/BL/BR), so the rule is not collapsed to
#   one fixed orientation.
#
# f) Visual cleanliness
# - Marker segments are 1-cell thick and disconnected from each other and from the
#   scattered pieces. They do not overlap any piece pixels.
# - Because pieces are at least 2x2 while marker segments are thin lines, the two object
#   classes stay easy to tell apart.
#
# g) Color inferability
# - No new color mapping is introduced. The distinct marker color is structural only.
#
# Post-change assumption review
# - `main()` now assumes exactly one non-bg color can be removed so that all remaining
#   non-bg cells split into full rectangular 4-connected components; `generate_input()`
#   enforces this by reserving a unique marker color and scattering rectangular pieces.
# - `main()` assumes the marker components are exactly one horizontal contiguous segment
#   and one vertical contiguous segment with lengths >= 5; `generate_input()` uses that
#   exact construction.
# - `main()` assumes the line lengths equal the assembled rectangle height/width; the
#   generator encodes the sides that way and validates with `main(input)`.

from collections import Counter
import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, paste, random_free_location


def _extract_rectangles(mask: np.ndarray, grid: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
    """Return full-rectangle connected components as (r, c, h, w, arr)."""
    lab, ncomp = label(mask.astype(int), structure=STRUCT4)
    rects: list[tuple[int, int, int, int, np.ndarray]] = []
    for k in range(1, ncomp + 1):
        pts = np.argwhere(lab == k)
        if pts.size == 0:
            continue
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if int(pts.shape[0]) != h * w:
            raise ValueError("non-rectangular component")
        rects.append((r0, c0, h, w, grid[r0 : r1 + 1, c0 : c1 + 1].copy()))
    return rects


def _parse_line_component(pts: np.ndarray) -> tuple[str, int, int, int]:
    """Return ('H', fixed_row, c0, c1) or ('V', fixed_col, r0, r1)."""
    rows = sorted(set(int(r) for r in pts[:, 0]))
    cols = sorted(set(int(c) for c in pts[:, 1]))
    if len(rows) == 1:
        c0, c1 = cols[0], cols[-1]
        if len(cols) != c1 - c0 + 1 or int(pts.shape[0]) != len(cols):
            raise ValueError("horizontal component is not contiguous")
        return ("H", rows[0], c0, c1)
    if len(cols) == 1:
        r0, r1 = rows[0], rows[-1]
        if len(rows) != r1 - r0 + 1 or int(pts.shape[0]) != len(rows):
            raise ValueError("vertical component is not contiguous")
        return ("V", cols[0], r0, r1)
    raise ValueError("component is not a 1-cell-thick straight segment")


def _find_marker(grid: np.ndarray, bg: int) -> tuple[int, str, tuple[int, int], tuple[int, int]]:
    """
    Find the unique edge-line marker.

    Returns:
        marker_color, corner_name, corner_coord, (H, W)
    """
    counts = Counter(map(int, grid.flatten()))
    candidates: list[tuple[int, str, tuple[int, int], tuple[int, int]]] = []

    for col in sorted(int(c) for c in counts if int(c) != int(bg)):
        mask = grid == int(col)
        total = int(mask.sum())
        if total < 10:
            continue

        lab, ncomp = label(mask.astype(int), structure=STRUCT4)
        if ncomp != 2:
            continue

        parts = []
        ok = True
        for k in range(1, ncomp + 1):
            pts = np.argwhere(lab == k)
            try:
                parts.append(_parse_line_component(pts))
            except ValueError:
                ok = False
                break
        if not ok:
            continue

        if {p[0] for p in parts} != {"H", "V"}:
            continue
        horiz = next(p for p in parts if p[0] == "H")
        vert = next(p for p in parts if p[0] == "V")

        hr, hc0, hc1 = horiz[1], horiz[2], horiz[3]
        vc, vr0, vr1 = vert[1], vert[2], vert[3]
        H = vr1 - vr0 + 1
        W = hc1 - hc0 + 1
        if H < 5 or W < 5:
            continue
        if total != H + W:
            continue

        corner = None
        if hc1 < vc and vr1 < hr:
            corner = "TL"
        elif hc0 > vc and vr1 < hr:
            corner = "TR"
        elif hc1 < vc and vr0 > hr:
            corner = "BL"
        elif hc0 > vc and vr0 > hr:
            corner = "BR"
        if corner is None:
            continue

        # Removing the true marker must leave only rectangular scattered pieces.
        occ = (grid != int(bg)) & (grid != int(col))
        try:
            rects = _extract_rectangles(occ, grid)
        except ValueError:
            continue
        if len(rects) < 4:
            continue
        if any(h < 2 or w < 2 for _r, _c, h, w, _a in rects):
            continue

        candidates.append((int(col), corner, (int(hr), int(vc)), (int(H), int(W))))

    if len(candidates) != 1:
        raise ValueError("edge-line marker not uniquely identified")
    return candidates[0]


def _corner_to_top_left(corner: str, mr: int, mc: int, H: int, W: int) -> tuple[int, int]:
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - W + 1
    if corner == "BL":
        return mr - H + 1, mc
    return mr - H + 1, mc - W + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into a rectangle at an edge-line marker corner.

    Marker rule:
    - A unique marker color forms exactly two disconnected line segments.
    - One horizontal segment is collinear with a target horizontal rectangle edge.
    - One vertical segment is collinear with a target vertical rectangle edge.
    - Their implied intersection is the rectangle corner.
    - Segment lengths equal the rectangle width/height.
    """
    g = np.array(grid, dtype=int)
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    mark, corner, (mr, mc), (H, W) = _find_marker(g, bg)
    tr, tc = _corner_to_top_left(corner, mr, mc, H, W)
    if not (0 <= tr and 0 <= tc and tr + H <= GH and tc + W <= GW):
        raise ValueError("marker-implied rectangle is out of bounds")

    occ = (g != int(bg)) & (g != int(mark))
    rects = _extract_rectangles(occ, g)
    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))
    if total_area != H * W:
        raise ValueError("marker lengths do not match total piece area")

    allpix: list[int] = []
    for _r, _c, _h, _w, a in rects:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    pieces = [a for _r, _c, _h, _w, a in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]

    def compat(v1: np.ndarray, v2: np.ndarray) -> bool:
        v1 = np.array(v1, dtype=int)
        v2 = np.array(v2, dtype=int)
        m = (v1 != int(rect_col)) & (v2 != int(rect_col))
        return bool((v1[m] == v2[m]).all())

    def assemble() -> np.ndarray | None:
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

    ass = assemble()
    if ass is None:
        raise ValueError("piece assembly failed")

    out = np.full((GH, GW), int(bg), dtype=int)
    out[tr : tr + H, tc : tc + W] = ass
    return out


def _marker_segments(corner: str, mr: int, mc: int, H: int, W: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if corner == "TL":
        horiz = [(mr, c) for c in range(mc - W, mc)]
        vert = [(r, mc) for r in range(mr - H, mr)]
    elif corner == "TR":
        horiz = [(mr, c) for c in range(mc + 1, mc + W + 1)]
        vert = [(r, mc) for r in range(mr - H, mr)]
    elif corner == "BL":
        horiz = [(mr, c) for c in range(mc - W, mc)]
        vert = [(r, mc) for r in range(mr + 1, mr + H + 1)]
    else:  # BR
        horiz = [(mr, c) for c in range(mc + 1, mc + W + 1)]
        vert = [(r, mc) for r in range(mr + 1, mr + H + 1)]
    return horiz, vert


def _pick_corner_cell(corner: str, GH: int, GW: int, RH: int, RW: int, rnd) -> tuple[int, int] | None:
    if corner == "TL":
        r_lo, r_hi = RH + 1, GH - RH - 1
        c_lo, c_hi = RW + 1, GW - RW - 1
    elif corner == "TR":
        r_lo, r_hi = RH + 1, GH - RH - 1
        c_lo, c_hi = RW, GW - RW - 2
    elif corner == "BL":
        r_lo, r_hi = RH, GH - RH - 2
        c_lo, c_hi = RW + 1, GW - RW - 1
    else:  # BR
        r_lo, r_hi = RH, GH - RH - 2
        c_lo, c_hi = RW, GW - RW - 2
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return rnd.randint(r_lo, r_hi), rnd.randint(c_lo, c_hi)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _gen_attempt() -> np.ndarray | None:
    import random as _r

    GH = _r.randint(12, 26)
    GW = _r.randint(12, 26)

    palette = list(range(10))
    _r.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    n_interior = _r.randint(1, 3)
    interior_cols = palette[3 : 3 + n_interior]

    max_rh = min(11, (GH - 2) // 2)
    max_rw = min(11, (GW - 2) // 2)
    min_side = 5
    if max_rh < min_side or max_rw < min_side:
        return None

    RH = _r.randint(min_side, max_rh)
    RW = _r.randint(min_side, max_rw)
    if _r.random() < 0.5:
        RH = _r.randint(max(min_side, max_rh - 3), max_rh)
        RW = _r.randint(max(min_side, max_rw - 3), max_rw)

    corner = _r.choice(["TL", "TR", "BL", "BR"])
    corner_cell = _pick_corner_cell(corner, GH, GW, RH, RW, _r)
    if corner_cell is None:
        return None
    mr, mc = corner_cell

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
    fill_ratio = non_rc_count / (iH * iW) if iH * iW > 0 else 0.0
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
    acc = 0
    for rs in row_segs:
        row_starts.append(acc)
        acc += rs
    col_starts = []
    acc = 0
    for cs in col_segs:
        col_starts.append(acc)
        acc += cs

    nr, nc_grid = len(row_segs), len(col_segs)
    cell_id = [[i * nc_grid + j for j in range(nc_grid)] for i in range(nr)]
    n_cells = nr * nc_grid
    if n_cells < 4:
        return None

    def _get_piece_bbox(pid: int) -> tuple[int, int, int, int] | None:
        cells = [(i, j) for i in range(nr) for j in range(nc_grid) if cell_id[i][j] == pid]
        if not cells:
            return None
        ri = [ij[0] for ij in cells]
        ci = [ij[1] for ij in cells]
        return min(ri), min(ci), max(ri), max(ci)

    def _pixel_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
        h = sum(row_segs[i] for i in range(r0i, r1i + 1))
        w = sum(col_segs[j] for j in range(c0i, c1i + 1))
        return h, w

    target_pieces = _r.randint(4, min(10, n_cells))
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
            if bb is None:
                return None
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

    horiz_cells, vert_cells = _marker_segments(corner, mr, mc, RH, RW)
    for cr, cc in horiz_cells + vert_cells:
        if not (0 <= cr < GH and 0 <= cc < GW):
            return None

    canvas = np.full((GH, GW), bg, dtype=int)
    for cr, cc in horiz_cells + vert_cells:
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

    tr, tc = _corner_to_top_left(corner, mr, mc, RH, RW)
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
