# Combo variant of 446ef5d2 (index 14)
# combo_spec: D1/5+D18/1
# Name: 连接器匹配配合重叠切割
#
# Design analysis for the fused variant
# =====================================
# a) Concrete behavior / behavior trace
#    - Original task:
#      * scattered rectangles are translated without overlap;
#      * final rectangle area equals the sum of piece areas;
#      * neighboring edges are checked by whole-edge compatibility with rect_col wildcard.
#    - D1/5 contribution:
#      * each internal side carries one dedicated connector pixel;
#      * matching is keyed by connector color + offset instead of dense whole-edge agreement.
#    - D18/1 contribution:
#      * adjacent pieces share exactly one row or one column in the assembled state;
#      * therefore total piece area is larger than the visible rectangle area.
#    - Fused pixel model used here:
#      * pieces come from a closed-interval grid partition, so neighboring cells overlap by
#        one shared row/column and diagonal cells may meet at a single shared corner pixel;
#      * every internal shared strip contains exactly one connector pixel placed away from
#        strip endpoints, so it is visually "inside the overlap" rather than on a corner;
#      * non-connector overlap pixels must still agree exactly because they are duplicated
#        copies of the same hidden rectangle strip.
#    - Small trace:
#      * Suppose two 4x4 pieces overlap by one column. Old logic would try offsets differing
#        by 4 columns and forbid overlap. The fused logic uses offset 3, allows the 1-column
#        overlap, and checks that the connector on the shared column aligns.
#      * If another candidate piece shares the same size but its connector sits one row lower,
#        the exact same 1-column overlap is rejected because the connector global coordinate
#        does not match, even though the boundary shape type is the same.
#
# b) Complexity calibration
#    - The generator uses 2x2, 2x3, 3x2, or 3x3 overlapping grids: enough repeated seams to
#      make connector reuse observable, but still small enough for reliable backtracking.
#    - Two connector colors are repeated across multiple seams. This avoids a degenerate
#      "every seam has a unique color so assembly is trivial" regime.
#    - Ordinary interior colors are also painted onto shared strips, so overlap is visible as
#      duplicated colored edge rows/columns rather than as an invisible bookkeeping detail.
#
# c) Implicit assumption audit against the original main()
#    - `total_area = sum(h*w)` no longer equals final rectangle area. The solver therefore does
#      not enumerate factor pairs from area at all; it reconstructs the rectangle by anchored
#      overlap-aware placement and then reads the occupied bbox.
#    - `can_place` may no longer reject all overlap. It must allow overlap iff the overlapping
#      pixels are exactly equal and the overlap geometry is a 1-row strip, 1-column strip, or
#      a derived 1x1 diagonal corner.
#    - Whole-edge wildcard compatibility is no longer the placement signal. Instead, candidate
#      pair offsets are generated from connector descriptors, and exact overlap equality acts
#      only as the physical consistency check.
#    - Pre-placing the anchor by area-factor dimensions is invalid. The anchor is now placed
#      directly at the marker corner and the rest of the rectangle grows outward from it.
#
# d) Placement legality
#    - Generator places scattered pieces with >=1 background padding and never touches marker cells.
#    - In the hidden assembled rectangle, every overlap is width/height exactly 1 by construction.
#    - Connectors are never placed on strip endpoints, so they do not collide with crossings of
#      horizontal and vertical overlap seams.
#
# e) Visual observability
#    - A human sees edge-only connector dots repeated on many scattered pieces.
#    - The output visibly collapses duplicated colored edge strips into single shared rows/cols,
#      so the "overlap cut" part is directly observable.
#    - Because connector colors never appear in piece interiors, the connector role is inferable
#      from a few examples instead of being hidden among ordinary decoration colors.
#
# f) Visual cleanliness
#    - The generator avoids T-junction side splits by using an unmerged grid. Each internal side
#      has at most one true opposite neighbor, which keeps connector semantics crisp.
#    - Shared-strip signals are sparse and short, so seam structure is visible without flooding
#      the rectangle with unrelated edge noise.
#
# g) Inferability of the changed rule
#    - Connector colors are a small repeated palette reserved for edges only.
#    - Interior colors are reused both in true interiors and on some shared strips, providing a
#      visible overlap cue while still remaining distinct from connector colors.
#    - The rule inferred from examples is stable: "find the rectangle corner from the L marker,
#      then assemble pieces so connector dots align on shared 1-pixel overlap strips."
#
# New solver assumptions introduced here
# --------------------------------------
# - Pieces are axis-aligned rectangles extracted cleanly from the scattered input.
# - Every non-boundary side has exactly one connector; boundary sides have none and stay rect_col.
# - Overlaps are only 1 row / 1 column / derived 1x1 diagonal corner. No thicker overlaps occur.
# - Connector colors are edge-only colors. The generator enforces all of these assumptions.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import paste, random_free_location


SIDES = ("T", "B", "L", "R")
OPPOSITE = {"T": "B", "B": "T", "L": "R", "R": "L"}


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
        miss = list({(r0, c0), (r0, c1), (r1, c0), (r1, c1)} - set(pts))
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


def _edge_values(a: np.ndarray, side: str) -> np.ndarray:
    if side == "T":
        return a[0, :]
    if side == "B":
        return a[-1, :]
    if side == "L":
        return a[:, 0]
    return a[:, -1]


def _connector_desc(a: np.ndarray, connector_colors: set[int]) -> dict[str, tuple[int, int] | None]:
    desc: dict[str, tuple[int, int] | None] = {}
    for side in SIDES:
        edge = np.array(_edge_values(a, side), dtype=int)
        idx = [int(k) for k, v in enumerate(edge) if int(v) in connector_colors]
        assert len(idx) <= 1
        desc[side] = (idx[0], int(edge[idx[0]])) if idx else None
    return desc


def _all_piece_pixels(rects: list[tuple[int, int, int, int, np.ndarray]]) -> list[int]:
    vals: list[int] = []
    for _r, _c, _h, _w, arr in rects:
        vals.extend(map(int, arr.flatten()))
    return vals


def _collect_meta(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    gh, gw = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    marker, missing, corner = _find_marker(g, bg)
    rects = _extract_rectangles(g, bg, marker)
    assert rects
    pieces = [arr for _r, _c, _h, _w, arr in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]
    rect_col = int(Counter(_all_piece_pixels(rects)).most_common(1)[0][0])

    edge_colors: set[int] = set()
    interior_colors: set[int] = set()
    for a in pieces:
        edge_colors.update(map(int, a[0, :]))
        edge_colors.update(map(int, a[-1, :]))
        edge_colors.update(map(int, a[:, 0]))
        edge_colors.update(map(int, a[:, -1]))
        if a.shape[0] > 2 and a.shape[1] > 2:
            interior_colors.update(map(int, a[1:-1, 1:-1].flatten()))
    edge_colors.discard(int(rect_col))
    interior_colors.discard(int(rect_col))
    connector_colors = {int(c) for c in edge_colors if int(c) not in interior_colors}
    assert connector_colors
    descs = [_connector_desc(a, connector_colors) for a in pieces]
    return {
        "g": g,
        "gh": gh,
        "gw": gw,
        "bg": bg,
        "marker": marker,
        "missing": missing,
        "corner": corner,
        "rects": rects,
        "pieces": pieces,
        "ph": ph,
        "pw": pw,
        "rect_col": rect_col,
        "connector_colors": connector_colors,
        "descs": descs,
    }


def _overlap_bounds(
    r1: int,
    c1: int,
    h1: int,
    w1: int,
    r2: int,
    c2: int,
    h2: int,
    w2: int,
) -> tuple[int, int, int, int] | None:
    rr0 = max(int(r1), int(r2))
    rr1 = min(int(r1 + h1), int(r2 + h2))
    cc0 = max(int(c1), int(c2))
    cc1 = min(int(c1 + w1), int(c2 + w2))
    if rr0 >= rr1 or cc0 >= cc1:
        return None
    return int(rr0), int(rr1), int(cc0), int(cc1)


def _overlap_matches(a: np.ndarray, b: np.ndarray, dr: int, dc: int) -> bool:
    h1, w1 = map(int, a.shape)
    h2, w2 = map(int, b.shape)
    ov = _overlap_bounds(0, 0, h1, w1, dr, dc, h2, w2)
    if ov is None:
        return False
    rr0, rr1, cc0, cc1 = ov
    av = a[rr0:rr1, cc0:cc1]
    bv = b[rr0 - dr : rr1 - dr, cc0 - dc : cc1 - dc]
    return bool(np.array_equal(av, bv))


def _connector_global(desc: tuple[int, int] | None, side: str, r: int, c: int, h: int, w: int) -> tuple[int, int, int] | None:
    if desc is None:
        return None
    off, col = desc
    if side == "T":
        return int(r), int(c + off), int(col)
    if side == "B":
        return int(r + h - 1), int(c + off), int(col)
    if side == "L":
        return int(r + off), int(c), int(col)
    return int(r + off), int(c + w - 1), int(col)


def _direct_overlap_sides(
    r1: int,
    c1: int,
    h1: int,
    w1: int,
    r2: int,
    c2: int,
    h2: int,
    w2: int,
) -> tuple[str, str] | None:
    ov = _overlap_bounds(r1, c1, h1, w1, r2, c2, h2, w2)
    if ov is None:
        return None
    rr0, rr1, cc0, cc1 = ov
    inter_h = rr1 - rr0
    inter_w = cc1 - cc0
    if inter_h > 1 and inter_w > 1:
        return None
    if inter_w == 1 and inter_h >= 2:
        side1 = "L" if cc0 == c1 else "R" if cc1 == c1 + w1 else None
        side2 = "L" if cc0 == c2 else "R" if cc1 == c2 + w2 else None
        if side1 is None or side2 is None or OPPOSITE[side1] != side2:
            return None
        return side1, side2
    if inter_h == 1 and inter_w >= 2:
        side1 = "T" if rr0 == r1 else "B" if rr1 == r1 + h1 else None
        side2 = "T" if rr0 == r2 else "B" if rr1 == r2 + h2 else None
        if side1 is None or side2 is None or OPPOSITE[side1] != side2:
            return None
        return side1, side2
    return None


def _relation_valid(
    a: np.ndarray,
    b: np.ndarray,
    desc_a: dict[str, tuple[int, int] | None],
    desc_b: dict[str, tuple[int, int] | None],
    side_a: str,
    dr: int,
    dc: int,
) -> bool:
    side_b = OPPOSITE[side_a]
    da = desc_a[side_a]
    db = desc_b[side_b]
    if da is None or db is None or da[1] != db[1]:
        return False
    if not _overlap_matches(a, b, dr, dc):
        return False
    sides = _direct_overlap_sides(0, 0, a.shape[0], a.shape[1], dr, dc, b.shape[0], b.shape[1])
    if sides != (side_a, side_b):
        return False
    ga = _connector_global(da, side_a, 0, 0, a.shape[0], a.shape[1])
    gb = _connector_global(db, side_b, dr, dc, b.shape[0], b.shape[1])
    return ga == gb


def _build_relations(
    pieces: list[np.ndarray],
    descs: list[dict[str, tuple[int, int] | None]],
) -> list[list[list[tuple[int, int]]]]:
    n = len(pieces)
    rel = [[[] for _ in range(n)] for _ in range(n)]
    for i, a in enumerate(pieces):
        hi, wi = map(int, a.shape)
        for j, b in enumerate(pieces):
            if i == j:
                continue
            hj, wj = map(int, b.shape)
            offsets: set[tuple[int, int]] = set()

            d = descs[i]["R"]
            e = descs[j]["L"]
            if d is not None and e is not None and d[1] == e[1]:
                dr = int(d[0] - e[0])
                dc = int(wi - 1)
                if _relation_valid(a, b, descs[i], descs[j], "R", dr, dc):
                    offsets.add((dr, dc))

            d = descs[i]["L"]
            e = descs[j]["R"]
            if d is not None and e is not None and d[1] == e[1]:
                dr = int(d[0] - e[0])
                dc = int(-(wj - 1))
                if _relation_valid(a, b, descs[i], descs[j], "L", dr, dc):
                    offsets.add((dr, dc))

            d = descs[i]["B"]
            e = descs[j]["T"]
            if d is not None and e is not None and d[1] == e[1]:
                dr = int(hi - 1)
                dc = int(d[0] - e[0])
                if _relation_valid(a, b, descs[i], descs[j], "B", dr, dc):
                    offsets.add((dr, dc))

            d = descs[i]["T"]
            e = descs[j]["B"]
            if d is not None and e is not None and d[1] == e[1]:
                dr = int(-(hj - 1))
                dc = int(d[0] - e[0])
                if _relation_valid(a, b, descs[i], descs[j], "T", dr, dc):
                    offsets.add((dr, dc))

            rel[i][j] = sorted(offsets)
    return rel


def _anchor_position(corner: str, mr: int, mc: int, h: int, w: int) -> tuple[int, int]:
    if corner == "TL":
        return int(mr), int(mc)
    if corner == "TR":
        return int(mr), int(mc - w + 1)
    if corner == "BL":
        return int(mr - h + 1), int(mc)
    return int(mr - h + 1), int(mc - w + 1)


def _anchor_ok(
    piece: np.ndarray,
    rect_col: int,
    corner: str,
    desc: dict[str, tuple[int, int] | None],
) -> bool:
    if corner == "TL":
        return bool(np.all(piece[0, :] == rect_col) and np.all(piece[:, 0] == rect_col) and desc["T"] is None and desc["L"] is None)
    if corner == "TR":
        return bool(np.all(piece[0, :] == rect_col) and np.all(piece[:, -1] == rect_col) and desc["T"] is None and desc["R"] is None)
    if corner == "BL":
        return bool(np.all(piece[-1, :] == rect_col) and np.all(piece[:, 0] == rect_col) and desc["B"] is None and desc["L"] is None)
    return bool(np.all(piece[-1, :] == rect_col) and np.all(piece[:, -1] == rect_col) and desc["B"] is None and desc["R"] is None)


def main(grid: np.ndarray) -> np.ndarray:
    meta = _collect_meta(grid)
    g = meta["g"]
    gh, gw = meta["gh"], meta["gw"]
    bg = meta["bg"]
    mr, mc = map(int, meta["missing"])
    corner = meta["corner"]
    pieces = meta["pieces"]
    ph = meta["ph"]
    pw = meta["pw"]
    rect_col = meta["rect_col"]
    descs = meta["descs"]
    rel = _build_relations(pieces, descs)
    n = len(pieces)

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
            return r >= mr and c + w - 1 <= mc
        if corner == "BL":
            return r + h - 1 <= mr and c >= mc
        return r + h - 1 <= mr and c + w - 1 <= mc

    def can_place(i: int, r: int, c: int) -> bool:
        a = pieces[i]
        h, w = map(int, a.shape)
        if not directional_ok(int(r), int(c), h, w):
            return False
        if not pos:
            return True
        touches = 0
        for j, (rj, cj) in pos.items():
            b = pieces[j]
            hj, wj = map(int, b.shape)
            ov = _overlap_bounds(r, c, h, w, rj, cj, hj, wj)
            if ov is None:
                continue
            rr0, rr1, cc0, cc1 = ov
            inter_h = rr1 - rr0
            inter_w = cc1 - cc0
            if inter_h > 1 and inter_w > 1:
                return False
            av = a[rr0 - r : rr1 - r, cc0 - c : cc1 - c]
            bv = b[rr0 - rj : rr1 - rj, cc0 - cj : cc1 - cj]
            if not np.array_equal(av, bv):
                return False
            if inter_h == 1 or inter_w == 1:
                touches += 1
            sides = _direct_overlap_sides(r, c, h, w, rj, cj, hj, wj)
            if sides is None:
                # Only a 1x1 diagonal corner overlap may skip connector checks.
                if not (inter_h == 1 and inter_w == 1):
                    return False
                continue
            side_i, side_j = sides
            di = descs[i][side_i]
            dj = descs[j][side_j]
            if di is None or dj is None or di[1] != dj[1]:
                return False
            gi = _connector_global(di, side_i, r, c, h, w)
            gj = _connector_global(dj, side_j, rj, cj, hj, wj)
            if gi != gj:
                return False
        return touches > 0 or not pos

    def place(i: int, r: int, c: int) -> None:
        a = pieces[i]
        h, w = map(int, a.shape)
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
        if not pos:
            return True
        placed = [(idx, pos[idx][0], pos[idx][1], ph[idx], pw[idx]) for idx in pos]
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
        return True

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

    anchors = [i for i, p in enumerate(pieces) if _anchor_ok(p, rect_col, corner, descs[i])]
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


def _split_fixed_parts(total: int, parts: int, rng: random.Random, min_d: int = 2, max_d: int = 5) -> list[int] | None:
    rem = int(total)
    out: list[int] = []
    for idx in range(parts):
        left = parts - idx - 1
        lo = max(min_d, rem - left * max_d)
        hi = min(max_d, rem - left * min_d)
        if lo > hi:
            return None
        val = rng.randint(lo, hi)
        out.append(int(val))
        rem -= int(val)
    return out if rem == 0 else None


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
        return int(mr), int(mc)
    if corner == "TR":
        return int(mr), int(mc - rw + 1)
    if corner == "BL":
        return int(mr - rh + 1), int(mc)
    return int(mr - rh + 1), int(mc - rw + 1)


def _grow_blob(available: np.ndarray, target: int, rng: random.Random) -> set[tuple[int, int]]:
    h, w = available.shape
    spots = [(int(r), int(c)) for r, c in np.argwhere(available)]
    if not spots:
        return set()
    sr, sc = rng.choice(spots)
    blob = {(sr, sc)}
    frontier = [(sr, sc)]
    while len(blob) < target and frontier:
        fr, fc = rng.choice(frontier)
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = fr + dr, fc + dc
            if 0 <= nr < h and 0 <= nc < w and available[nr, nc] and (nr, nc) not in blob:
                nbrs.append((nr, nc))
        if not nbrs:
            frontier.remove((fr, fc))
            continue
        nr, nc = rng.choice(nbrs)
        blob.add((int(nr), int(nc)))
        frontier.append((int(nr), int(nc)))
    return blob


def _paint_interior(rect: np.ndarray, row_coords: list[int], col_coords: list[int], interior_cols: list[int], rng: random.Random) -> bool:
    rh, rw = rect.shape
    ih, iw = rh - 2, rw - 2
    if ih < 2 or iw < 2:
        return False
    interior_mask = np.ones((ih, iw), dtype=bool)
    total = ih * iw
    target_fill = rng.uniform(0.30, 0.68)
    target_pixels = max(4, int(target_fill * total))
    color_budgets = [target_pixels // len(interior_cols)] * len(interior_cols)
    for k in range(target_pixels - sum(color_budgets)):
        color_budgets[k % len(color_budgets)] += 1
    used_any = set()
    for col, budget in zip(interior_cols, color_budgets):
        budget = max(2, budget)
        blob = _grow_blob(interior_mask, budget, rng)
        for br, bc in blob:
            rect[1 + br, 1 + bc] = int(col)
            interior_mask[br, bc] = False
            used_any.add(int(col))
    # Guarantee each interior color appears away from piece edges as well.
    for col in interior_cols:
        if int(col) in used_any:
            continue
        rr = rng.randint(1, rh - 2)
        cc = rng.randint(1, rw - 2)
        rect[rr, cc] = int(col)
    return True


def _paint_seam_signal(
    rect: np.ndarray,
    fixed_axis: int,
    start: int,
    end: int,
    interior_cols: list[int],
    rng: random.Random,
    horizontal: bool,
    reserved: set[int],
) -> None:
    slots = [p for p in range(start + 1, end) if p not in reserved]
    if not slots:
        return
    col = int(rng.choice(interior_cols))
    run_len = min(len(slots), rng.randint(1, min(3, len(slots))))
    start_idx = rng.randint(0, len(slots) - run_len)
    run = slots[start_idx : start_idx + run_len]
    for pos in run:
        if horizontal:
            rect[fixed_axis, pos] = col
        else:
            rect[pos, fixed_axis] = col


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    palette = list(range(10))
    rng.shuffle(palette)
    bg = int(palette[0])
    rect_col = int(palette[1])
    marker_col = int(palette[2])
    connector_cols = [int(palette[3]), int(palette[4])]
    n_interior = rng.randint(1, 2)
    interior_cols = [int(x) for x in palette[5 : 5 + n_interior]]

    nr = rng.choice([2, 3])
    nc = rng.choice([2, 3])
    if nr * nc < 4:
        return None

    row_diffs = _split_fixed_parts(rng.randint(6, 14), nr, rng)
    col_diffs = _split_fixed_parts(rng.randint(6, 14), nc, rng)
    if row_diffs is None or col_diffs is None:
        return None
    rh = int(sum(row_diffs) + 1)
    rw = int(sum(col_diffs) + 1)
    if rh < 7 or rw < 7 or rh > 18 or rw > 18:
        return None

    row_coords = [0]
    for d in row_diffs:
        row_coords.append(int(row_coords[-1] + d))
    col_coords = [0]
    for d in col_diffs:
        col_coords.append(int(col_coords[-1] + d))

    gh_min = max(12, rh + 4)
    gw_min = max(12, rw + 4)
    if gh_min > 26 or gw_min > 26:
        return None
    gh = rng.randint(gh_min, 26)
    gw = rng.randint(gw_min, 26)

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

    rect = np.full((rh, rw), rect_col, dtype=int)
    if not _paint_interior(rect, row_coords, col_coords, interior_cols, rng):
        return None

    seam_count = 0
    for i in range(nr):
        for j in range(nc - 1):
            col = int(col_coords[j + 1])
            r0, r1 = int(row_coords[i]), int(row_coords[i + 1])
            length = r1 - r0 + 1
            if length < 3:
                return None
            conn_r = rng.randint(r0 + 1, r1 - 1)
            rect[conn_r, col] = int(connector_cols[(i + j) % len(connector_cols)])
            _paint_seam_signal(rect, col, r0, r1, interior_cols, rng, False, {conn_r})
            seam_count += 1

    for i in range(nr - 1):
        for j in range(nc):
            row = int(row_coords[i + 1])
            c0, c1 = int(col_coords[j]), int(col_coords[j + 1])
            length = c1 - c0 + 1
            if length < 3:
                return None
            conn_c = rng.randint(c0 + 1, c1 - 1)
            rect[row, conn_c] = int(connector_cols[(i + j + 1) % len(connector_cols)])
            _paint_seam_signal(rect, row, c0, c1, interior_cols, rng, True, {conn_c})
            seam_count += 1

    pieces_meta: list[tuple[int, int, int, int, int, int]] = []
    piece_arrays: list[np.ndarray] = []
    for i in range(nr):
        for j in range(nc):
            pr = int(row_coords[i])
            pc = int(col_coords[j])
            ph = int(row_coords[i + 1] - row_coords[i] + 1)
            pw = int(col_coords[j + 1] - col_coords[j] + 1)
            if ph < 3 or pw < 3 or ph > 7 or pw > 7:
                return None
            arr = rect[pr : pr + ph, pc : pc + pw].copy()
            pieces_meta.append((i, j, pr, pc, ph, pw))
            piece_arrays.append(arr)

    total_piece_area = int(sum(int(a.shape[0] * a.shape[1]) for a in piece_arrays))
    if total_piece_area <= rh * rw:
        return None
    if gh * gw < int(2.6 * total_piece_area):
        return None

    connector_set = set(connector_cols)
    all_piece_pixels = []
    for arr in piece_arrays:
        all_piece_pixels.extend(map(int, arr.flatten()))
    if Counter(all_piece_pixels).most_common(1)[0][0] != rect_col:
        return None

    for idx, (i, j, _pr, _pc, ph, pw) in enumerate(pieces_meta):
        piece = piece_arrays[idx]
        desc = _connector_desc(piece, connector_set)
        expected = {
            "T": None if i == 0 else "present",
            "B": None if i == nr - 1 else "present",
            "L": None if j == 0 else "present",
            "R": None if j == nc - 1 else "present",
        }
        for side, need in expected.items():
            if need is None:
                if desc[side] is not None:
                    return None
                edge = _edge_values(piece, side)
                if not np.all(edge == rect_col):
                    return None
            else:
                if desc[side] is None:
                    return None
        if np.any(piece[1:-1, 1:-1] == connector_cols[0]) or np.any(piece[1:-1, 1:-1] == connector_cols[1]):
            return None
        fill = float(np.mean(piece != rect_col))
        if fill < 0.05 or fill > 0.65:
            return None
        if np.sum(piece != rect_col) < 3:
            return None

    # Every seam must be visibly signaled by at least one non-rect_col pixel.
    signal_hits = 0
    for i in range(nr):
        for j in range(nc - 1):
            col = int(col_coords[j + 1])
            r0, r1 = int(row_coords[i]), int(row_coords[i + 1])
            vals = rect[r0 : r1 + 1, col]
            if np.any(vals != rect_col):
                signal_hits += 1
    for i in range(nr - 1):
        for j in range(nc):
            row = int(row_coords[i + 1])
            c0, c1 = int(col_coords[j]), int(col_coords[j + 1])
            vals = rect[row, c0 : c1 + 1]
            if np.any(vals != rect_col):
                signal_hits += 1
    if signal_hits != seam_count:
        return None

    canvas = np.full((gh, gw), bg, dtype=int)
    for cr, cc in _marker_cells(corner, mr, mc):
        if cr < 0 or cr >= gh or cc < 0 or cc >= gw:
            return None
        canvas[cr, cc] = int(marker_col)

    order = sorted(range(len(piece_arrays)), key=lambda k: -(piece_arrays[k].shape[0] * piece_arrays[k].shape[1]))
    rng.shuffle(order)
    for idx in order:
        arr = piece_arrays[idx]
        try:
            r0, c0 = random_free_location(canvas, arr, transparent=bg, padding=1)
        except ValueError:
            return None
        paste(canvas, arr, int(r0), int(c0))

    tr, tc = _assembled_top_left(corner, mr, mc, rh, rw)
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
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
