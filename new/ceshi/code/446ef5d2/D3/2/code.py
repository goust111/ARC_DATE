# Variant of 446ef5d2
# Dimension: D3 — 目标结构的形式 (dst)
# Change: 以 L 形标记的缺失角为起点，碎片拼接成一个完整矩形，矩形四条外边全为 rect_col... → 多个独立矩形目标

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, paste, random_free_location


# Design analysis for the D3 variant
# ----------------------------------
# spec_variant:
#   core_change:
#     from: "all scattered pieces belong to one assembled rectangle anchored by one L-marker"
#     to: "there are two independent assembled rectangles, each anchored by its own L-marker"
#   concretization:
#     - The input contains exactly two isolated 3-cell L markers.
#     - Each marker color is also the unique interior color of one target rectangle.
#     - All pieces still preserve original behavior locally:
#       rectangular fragments, no rotation, common rect_col border/wildcard,
#       and shared-edge matching only constrains non-rect_col cells.
#     - The new global step is grouping:
#       pieces must first be partitioned into two sets before each set is assembled
#       around the marker of the same color.
#   behavior_trace:
#     - Original solver path:
#       find one 3-cell marker by color-count==3 ->
#       remove that color from the occupancy mask ->
#       treat every remaining piece as one pool ->
#       find one rectangle dimension and assemble once.
#     - Variant solver path:
#       find every 4-connected component of size 3 that forms a 2x2-minus-1 L ->
#       keep their exact cells as markers even though marker colors also appear inside pieces ->
#       decompose the remaining occupied pixels into piece rectangles ->
#       infer rect_col globally ->
#       group pieces by their only non-rect_col color ->
#       assemble one rectangle per marker/group pair and paste both into the output.
#     - Difference:
#       the old "one marker / one piece pool / one rectangle" assumption is invalid in three places,
#       so marker detection, occupancy masking, and assembly loop all need to change.
#   complexity_calibration:
#     - Targets are fixed at two. This makes the D3 change visually obvious in 2-3 examples
#       without exploding the search space.
#     - Each target uses one dedicated interior color. Grouping is therefore visible and stable,
#       but still non-trivial because the solver must separate piece pools before assembly.
#     - rect_col remains shared across both targets to preserve the original wildcard edge logic.
#   implicit_assumption_audit_main:
#     - Original marker search depended on "marker color appears exactly 3 times".
#       This no longer holds because the marker color also appears in its pieces.
#       Replacement: detect marker by connected-component geometry, not total count.
#     - Original occupancy mask excluded one marker color globally.
#       That would wrongly erase piece interiors in the variant.
#       Replacement: exclude only the exact marker cells.
#     - Original total_area summed all pieces because there was one target.
#       Variant requires a separate total area per group.
#     - Original output wrote one assembled rectangle once.
#       Variant must assemble and paste two disjoint rectangles.
#   placement_legality:
#     - Generator places the two final rectangles and their markers first, with at least a
#       one-cell moat between targets for visual clarity and to prevent output overlap.
#     - Scattered pieces are then placed with padding=1 from everything already on the canvas,
#       which keeps markers isolated as size-3 connected components.
#   visual_observability:
#     - A human sees two L markers and two color families of pieces.
#       In the output, each family rebuilds at the marker of the same color.
#       This is inferable from a few examples without relying on color ids as numbers.
#   neatness:
#     - Only three visual object types coexist: background, two isolated L markers, and separated pieces.
#       No overlays or tags are added on top of pieces.
#   color_reasoning:
#     - No hidden color permutation is used.
#       Marker color == target interior color is the visible grouping cue.


def _top_left_from_marker(mr: int, mc: int, corner: str, H: int, W: int) -> tuple[int, int]:
    if corner == "TL":
        return int(mr), int(mc)
    if corner == "TR":
        return int(mr), int(mc - W + 1)
    if corner == "BL":
        return int(mr - H + 1), int(mc)
    return int(mr - H + 1), int(mc - W + 1)


def _missing_from_top_left(tr: int, tc: int, H: int, W: int, corner: str) -> tuple[int, int]:
    if corner == "TL":
        return int(tr), int(tc)
    if corner == "TR":
        return int(tr), int(tc + W - 1)
    if corner == "BL":
        return int(tr + H - 1), int(tc)
    return int(tr + H - 1), int(tc + W - 1)


def _marker_cells_from_missing(mr: int, mc: int, corner: str) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _find_markers(g: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    markers: list[dict] = []
    marker_mask = np.zeros_like(g, dtype=bool)
    for col in sorted(int(c) for c in np.unique(g) if int(c) != int(bg)):
        labs, nlab = label(g == int(col), STRUCT4)
        for lab in range(1, int(nlab) + 1):
            pts = np.argwhere(labs == lab)
            if int(pts.shape[0]) != 3:
                continue
            coords = {(int(r), int(c)) for r, c in pts}
            rs = [r for r, _ in coords]
            cs = [c for _, c in coords]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            if (r1 - r0) != 1 or (c1 - c0) != 1:
                continue
            miss = list({(r0, c0), (r0, c1), (r1, c0), (r1, c1)} - coords)
            if len(miss) != 1:
                continue
            missing = tuple(miss[0])
            if missing == (r0, c0):
                corner = "BR"
            elif missing == (r0, c1):
                corner = "BL"
            elif missing == (r1, c0):
                corner = "TR"
            else:
                corner = "TL"
            markers.append(
                {
                    "color": int(col),
                    "cells": sorted(coords),
                    "missing": (int(missing[0]), int(missing[1])),
                    "corner": corner,
                }
            )
            for r, c in coords:
                marker_mask[r, c] = True
    return markers, marker_mask


def _extract_rectangles(g: np.ndarray, bg: int, marker_mask: np.ndarray) -> list[np.ndarray]:
    GH, GW = g.shape
    occ = (g != int(bg)) & (~marker_mask)
    used = np.zeros_like(occ, dtype=bool)
    rects: list[np.ndarray] = []
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
        rects.append(g[rr : rr + h, cc : cc + w].copy())
    return rects


def _compat(v1: np.ndarray, v2: np.ndarray, rect_col: int) -> bool:
    vv1 = np.array(v1, dtype=int)
    vv2 = np.array(v2, dtype=int)
    m = (vv1 != int(rect_col)) & (vv2 != int(rect_col))
    return bool((vv1[m] == vv2[m]).all())


def _assemble_group(
    pieces: list[np.ndarray], rect_col: int, marker: dict, grid_shape: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int]]:
    GH, GW = map(int, grid_shape)
    mr, mc = map(int, marker["missing"])
    corner = str(marker["corner"])
    total_area = int(sum(int(a.shape[0]) * int(a.shape[1]) for a in pieces))

    dims: list[tuple[int, int]] = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh == 0:
            dims.append((hh, total_area // hh))
            if hh * hh != total_area:
                dims.append((total_area // hh, hh))

    def fits(H: int, W: int) -> bool:
        tr, tc = _top_left_from_marker(mr, mc, corner, H, W)
        return 0 <= tr and 0 <= tc and (tr + H) <= GH and (tc + W) <= GW

    dims = [(int(H), int(W)) for H, W in dims if fits(int(H), int(W))]
    dims = sorted(dims, key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]

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
                if m.any() and (not _compat(left[m], a[:, 0][m], rect_col)):
                    return False
            if (c + w) < W:
                right = occ2[r : r + h, c + w]
                m = right != -1
                if m.any() and (not _compat(right[m], a[:, -1][m], rect_col)):
                    return False
            if r > 0:
                up = occ2[r - 1, c : c + w]
                m = up != -1
                if m.any() and (not _compat(up[m], a[0, :][m], rect_col)):
                    return False
            if (r + h) < H:
                down = occ2[r + h, c : c + w]
                m = down != -1
                if m.any() and (not _compat(down[m], a[-1, :][m], rect_col)):
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

    ass = None
    used_dim = None
    for H, W in dims:
        ass = assemble(H, W)
        if ass is not None:
            used_dim = (int(H), int(W))
            break

    assert ass is not None and used_dim is not None
    top_left = _top_left_from_marker(mr, mc, corner, used_dim[0], used_dim[1])
    return ass, top_left


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    markers, marker_mask = _find_markers(g, bg)
    assert len(markers) >= 2

    rects = _extract_rectangles(g, bg, marker_mask)
    assert rects

    allpix: list[int] = []
    for a in rects:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    groups: dict[int, list[np.ndarray]] = defaultdict(list)
    for a in rects:
        cols = sorted({int(v) for v in a.flatten()} - {int(rect_col)})
        assert len(cols) == 1
        groups[int(cols[0])].append(a)

    marker_by_color = {int(m["color"]): m for m in markers}
    assert len(marker_by_color) == len(markers)
    assert set(groups.keys()) == set(marker_by_color.keys())

    out = np.full((GH, GW), int(bg), dtype=int)
    filled = np.zeros((GH, GW), dtype=bool)
    for col in sorted(marker_by_color):
        ass, (tr, tc) = _assemble_group(groups[col], rect_col, marker_by_color[col], (GH, GW))
        H, W = ass.shape
        assert not filled[tr : tr + H, tc : tc + W].any()
        out[tr : tr + H, tc : tc + W] = ass
        filled[tr : tr + H, tc : tc + W] = True
    return out


def _rand_splits(total: int, rng, min_s: int = 2, max_s: int = 5) -> list[int] | None:
    segs: list[int] = []
    rem = int(total)
    eff_max = min(int(max_s), max(int(min_s), rng.randint(min_s, max_s)))
    while rem > 0:
        if rem <= eff_max:
            if rem < min_s:
                return None
            segs.append(int(rem))
            rem = 0
        else:
            hi = min(eff_max, rem - min_s)
            if min_s > hi:
                return None
            s = rng.randint(min_s, hi)
            segs.append(int(s))
            rem -= int(s)
    return segs


def _build_target_rect(RH: int, RW: int, rect_col: int, accent: int, rng) -> np.ndarray | None:
    iH, iW = RH - 2, RW - 2
    if iH < 1 or iW < 1:
        return None
    for _ in range(120):
        interior = np.full((iH, iW), int(accent), dtype=int)
        hole_count = rng.randint(1, 4)
        for _hole in range(hole_count):
            hh = rng.randint(1, max(1, iH // 2))
            ww = rng.randint(1, max(1, iW // 2))
            rr = rng.randint(0, iH - hh)
            cc = rng.randint(0, iW - ww)
            interior[rr : rr + hh, cc : cc + ww] = int(rect_col)
        if rng.random() < 0.7 and iH >= 2:
            rr = rng.randint(0, iH - 1)
            span = rng.randint(max(1, iW // 3), iW)
            cc = rng.randint(0, iW - span)
            interior[rr, cc : cc + span] = int(rect_col)
        if rng.random() < 0.7 and iW >= 2:
            cc = rng.randint(0, iW - 1)
            span = rng.randint(max(1, iH // 3), iH)
            rr = rng.randint(0, iH - span)
            interior[rr : rr + span, cc] = int(rect_col)

        fill = float(np.mean(interior == int(accent)))
        if not (0.35 <= fill <= 0.85):
            continue
        if not np.any(interior == int(rect_col)):
            continue
        rect = np.full((RH, RW), int(rect_col), dtype=int)
        rect[1:-1, 1:-1] = interior
        return rect
    return None


def _has_marker_like_component(piece: np.ndarray, accent: int) -> bool:
    labs, nlab = label(piece == int(accent), STRUCT4)
    for lab in range(1, int(nlab) + 1):
        pts = np.argwhere(labs == lab)
        if int(pts.shape[0]) != 3:
            continue
        rs = pts[:, 0]
        cs = pts[:, 1]
        if int(rs.max() - rs.min()) == 1 and int(cs.max() - cs.min()) == 1:
            return True
    return False


def _partition_rect(rect: np.ndarray, rect_col: int, accent: int, rng) -> list[np.ndarray] | None:
    RH, RW = rect.shape
    for _ in range(200):
        row_segs = _rand_splits(RH, rng, min_s=2, max_s=min(5, RH))
        col_segs = _rand_splits(RW, rng, min_s=2, max_s=min(5, RW))
        if row_segs is None or col_segs is None:
            continue

        row_starts: list[int] = []
        acc = 0
        for rs in row_segs:
            row_starts.append(acc)
            acc += int(rs)
        col_starts: list[int] = []
        acc = 0
        for cs in col_segs:
            col_starts.append(acc)
            acc += int(cs)

        nr = len(row_segs)
        nc = len(col_segs)
        cell_id = [[i * nc + j for j in range(nc)] for i in range(nr)]
        n_cells = nr * nc
        if n_cells < 3:
            continue
        target_pieces = rng.randint(3, min(6, n_cells))

        def get_piece_bbox(pid: int) -> tuple[int, int, int, int] | None:
            cells = [(i, j) for i in range(nr) for j in range(nc) if cell_id[i][j] == pid]
            if not cells:
                return None
            ris = [c[0] for c in cells]
            cis = [c[1] for c in cells]
            return min(ris), min(cis), max(ris), max(cis)

        def pixel_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
            h = sum(int(row_segs[i]) for i in range(r0i, r1i + 1))
            w = sum(int(col_segs[j]) for j in range(c0i, c1i + 1))
            return int(h), int(w)

        merges: list[tuple[str, int, int]] = []
        for i in range(nr):
            for j in range(nc):
                if j + 1 < nc:
                    merges.append(("h", i, j))
                if i + 1 < nr:
                    merges.append(("v", i, j))
        rng.shuffle(merges)

        def count_pieces() -> int:
            return len({cell_id[i][j] for i in range(nr) for j in range(nc)})

        for direction, mi, mj in merges:
            if count_pieces() <= target_pieces:
                break
            if direction == "h":
                pid1 = cell_id[mi][mj]
                pid2 = cell_id[mi][mj + 1]
            else:
                pid1 = cell_id[mi][mj]
                pid2 = cell_id[mi + 1][mj]
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
                for jj in range(nc):
                    if cell_id[ii][jj] == pid2:
                        cell_id[ii][jj] = pid1

        pieces: list[np.ndarray] = []
        seen: set[int] = set()
        valid = True
        for i in range(nr):
            for j in range(nc):
                pid = cell_id[i][j]
                if pid in seen:
                    continue
                seen.add(pid)
                bb = get_piece_bbox(pid)
                assert bb is not None
                r0i, c0i, r1i, c1i = bb
                pr = row_starts[r0i]
                pc = col_starts[c0i]
                ph = sum(int(row_segs[ii]) for ii in range(r0i, r1i + 1))
                pw = sum(int(col_segs[jj]) for jj in range(c0i, c1i + 1))
                piece = rect[pr : pr + ph, pc : pc + pw].copy()

                if ph < 2 or pw < 2 or ph > 7 or pw > 7:
                    valid = False
                    break
                n_accent = int(np.sum(piece == int(accent)))
                if n_accent == 0:
                    valid = False
                    break
                fill = n_accent / float(ph * pw)
                if fill < 0.10 or fill > 0.85:
                    valid = False
                    break
                if pr == 0 and not np.all(piece[0, :] == int(rect_col)):
                    valid = False
                    break
                if pr + ph == RH and not np.all(piece[-1, :] == int(rect_col)):
                    valid = False
                    break
                if pc == 0 and not np.all(piece[:, 0] == int(rect_col)):
                    valid = False
                    break
                if pc + pw == RW and not np.all(piece[:, -1] == int(rect_col)):
                    valid = False
                    break
                if not any(np.any(edge == int(accent)) for edge in [piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]]):
                    valid = False
                    break
                if {int(v) for v in piece.flatten()} - {int(rect_col), int(accent)}:
                    valid = False
                    break
                if _has_marker_like_component(piece, accent):
                    valid = False
                    break
                pieces.append(piece)
            if not valid:
                break

        if valid and 3 <= len(pieces) <= 6:
            return pieces
    return None


def _clear_with_padding(mask: np.ndarray, occ: np.ndarray, pad: int = 1) -> bool:
    H, W = occ.shape
    for r, c in np.argwhere(mask):
        r0 = max(0, int(r) - pad)
        r1 = min(H, int(r) + pad + 1)
        c0 = max(0, int(c) - pad)
        c1 = min(W, int(c) + pad + 1)
        if occ[r0:r1, c0:c1].any():
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    import random as _r

    GH = _r.randint(16, 26)
    GW = _r.randint(16, 26)

    palette = list(range(10))
    _r.shuffle(palette)
    bg = int(palette[0])
    rect_col = int(palette[1])
    accents = [int(palette[2]), int(palette[3])]

    targets: list[dict] = []
    for accent in accents:
        RH = _r.randint(5, min(10, GH - 4))
        RW = _r.randint(5, min(10, GW - 4))
        corner = _r.choice(["TL", "TR", "BL", "BR"])
        rect = _build_target_rect(RH, RW, rect_col, accent, _r)
        if rect is None:
            return None
        pieces = _partition_rect(rect, rect_col, accent, _r)
        if pieces is None:
            return None
        targets.append(
            {
                "accent": accent,
                "corner": corner,
                "RH": RH,
                "RW": RW,
                "rect": rect,
                "pieces": pieces,
            }
        )

    # Place the two final rectangles and markers first.
    occ = np.zeros((GH, GW), dtype=bool)
    for target in sorted(targets, key=lambda t: -(int(t["RH"]) * int(t["RW"]))):
        placed = False
        RH = int(target["RH"])
        RW = int(target["RW"])
        corner = str(target["corner"])
        for _ in range(300):
            tr = _r.randint(0, GH - RH)
            tc = _r.randint(0, GW - RW)
            mr, mc = _missing_from_top_left(tr, tc, RH, RW, corner)
            marker_cells = _marker_cells_from_missing(mr, mc, corner)
            if not all(0 <= r < GH and 0 <= c < GW for r, c in marker_cells):
                continue
            footprint = np.zeros((GH, GW), dtype=bool)
            footprint[tr : tr + RH, tc : tc + RW] = True
            for r, c in marker_cells:
                footprint[r, c] = True
            if not _clear_with_padding(footprint, occ, pad=1):
                continue
            occ |= footprint
            target["top_left"] = (int(tr), int(tc))
            target["missing"] = (int(mr), int(mc))
            target["marker_cells"] = marker_cells
            placed = True
            break
        if not placed:
            return None

    canvas = np.full((GH, GW), int(bg), dtype=int)
    expected_out = np.full((GH, GW), int(bg), dtype=int)

    for target in targets:
        tr, tc = map(int, target["top_left"])
        RH = int(target["RH"])
        RW = int(target["RW"])
        expected_out[tr : tr + RH, tc : tc + RW] = target["rect"]
        for r, c in target["marker_cells"]:
            canvas[int(r), int(c)] = int(target["accent"])

    piece_pool: list[np.ndarray] = []
    for target in targets:
        piece_pool.extend([p.copy() for p in target["pieces"]])
    _r.shuffle(piece_pool)

    for piece in piece_pool:
        try:
            r0, c0 = random_free_location(canvas, piece, transparent=bg, padding=1)
        except ValueError:
            return None
        paste(canvas, piece, r0, c0)

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
