# Combo variant of 446ef5d2 (index 5)
# combo_spec: D4/2+D5/0
# Name: 保留源碎片并擦除标记
#
# Design analysis for the fused variant
# ====================================
# a) Concrete behavior / trace
#    - Original task:
#      input = scattered rectangular pieces + 3-cell L marker.
#      output = only the assembled rectangle at the marker corner; all scattered pieces disappear.
#    - Fused task:
#      input stays the same.
#      output keeps every scattered source piece exactly where it was, also pastes the assembled rectangle
#      at the marker-indicated corner, and finally erases the 3 marker pixels back to background.
#    - Small trace:
#      Suppose one blue/yellow piece sits at top-left of the input, another sits at bottom-right, and the
#      marker gap indicates a target corner near the center. The original output removes both scattered pieces
#      and shows only the assembled rectangle near the center. The fused output still shows the two scattered
#      pieces at their old positions, adds the assembled rectangle near the center, and the L marker itself is
#      gone. This is "copy and assemble", not "move and assemble".
#
# b) Complexity calibration
#    - If generator allowed source pieces to overlap the assembled rectangle area, the keep-source signal would
#      become visually muddy because some "preserved" pixels would be hidden under the pasted rectangle.
#      The generator therefore reserves the future assembly rectangle area as empty background in the input.
#    - The task still keeps the original rectangle-partition freedom, 4-12 pieces, and wildcard border color.
#      The only changed branch is the output composition stage, which is visually obvious across examples.
#
# c) Original implicit assumptions that break
#    - Old finalization assumed `out` starts as all background and only the assembled rectangle matters.
#      That is invalid now because scattered pieces must survive into the output.
#    - Old generator could scatter pieces anywhere, including the future assembly area, because output discarded
#      all sources. That becomes a visual-conflict risk now, so the generator must reserve the target area.
#    - Marker pixels used to disappear only because the whole output canvas was rebuilt from background. In this
#      fused variant we start from the input, so marker erasure must be explicit.
#    - The assembly search itself still assumes:
#      * exactly one 3-cell L marker color,
#      * non-marker foreground decomposes into full rectangles,
#      * touching piece edges match with `rect_col` as wildcard.
#      Those assumptions remain valid and are preserved by the generator.
#
# d) Placement legality
#    - Generator places the marker first, reserves the future assembled rectangle footprint, then scatters pieces
#      with `padding=1`, so pieces do not overlap each other, the marker, or the future rectangle area.
#    - The reserved footprint is cleared back to background before returning the input, so the input still shows
#      only marker + scattered pieces.
#
# e) Visual observability
#    - A human sees the same scattered pieces in input and output, plus one new assembled rectangle at the marker.
#    - The marker is visibly instructional only: it exists in input but is absent in output.
#    - Because source pieces remain visible, the difference from the original "move away" behavior is immediate in
#      2-3 examples without relying on any invisible numeric property.
#
# f) Visual cleanliness
#    - Reserving the target rectangle keeps the assembled copy spatially separate from the source clutter.
#    - The only overlay in output is the assembled rectangle itself; there is no source/target pile-up in the
#      intended generated samples.
#
# g) Quantitative anti-degeneracy target
#    - For every accepted generated sample:
#      * the number of preserved source-piece pixels at their original coordinates is strictly positive,
#      * the number of marker-colored pixels in the output is zero.
#    - Because the target area is reserved, the fused output differs from the original baseline output by exactly
#      the kept scattered pieces, making the keep-source behavior non-degenerate.
#
# h) New solver assumptions introduced by this variant
#    - Main now assumes only that preserving sources is done by copying the full input first, then erasing the
#      true marker cells, then pasting the assembled rectangle. This is robust even if a non-generated test case
#      lets the assembled rectangle overlap a source piece: the assembled rectangle simply paints over it.
#    - The generator supports the cleaner stronger regime where no such overlap occurs.

from __future__ import annotations

from collections import Counter
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "solvers"))
from arc_utils import paste, random_free_location


def _marker_cells(corner: str, mr: int, mc: int) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _top_left_from_marker(mr: int, mc: int, corner: str, H: int, W: int) -> tuple[int, int]:
    if corner == "TL":
        return int(mr), int(mc)
    if corner == "TR":
        return int(mr), int(mc - W + 1)
    if corner == "BL":
        return int(mr - H + 1), int(mc)
    return int(mr - H + 1), int(mc - W + 1)


def _find_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], tuple[int, int, int, int], str]:
    cnt = Counter(map(int, g.flatten()))
    for col in sorted(c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3):
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
        missing = miss[0]
        if missing == (r0, c0):
            corner = "BR"
        elif missing == (r0, c1):
            corner = "BL"
        elif missing == (r1, c0):
            corner = "TR"
        else:
            corner = "TL"
        return int(col), (int(missing[0]), int(missing[1])), (int(r0), int(c0), int(r1), int(c1)), corner
    raise AssertionError("marker not found")


def _extract_rectangles(g: np.ndarray, bg: int, marker_col: int) -> list[tuple[int, int, int, int, np.ndarray]]:
    GH, GW = g.shape
    occ = (g != int(bg)) & (g != int(marker_col))
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


def _rect_color(rects: list[tuple[int, int, int, int, np.ndarray]]) -> int:
    allpix: list[int] = []
    for _r, _c, _h, _w, a in rects:
        allpix.extend(map(int, a.flatten()))
    return int(Counter(allpix).most_common(1)[0][0])


def _assemble_rectangle(
    rects: list[tuple[int, int, int, int, np.ndarray]],
    rect_col: int,
    missing: tuple[int, int],
    corner: str,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    GH, GW = map(int, grid_shape)
    mr, mc = map(int, missing)
    pieces = [a for _r, _c, _h, _w, a in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]
    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))

    anchor_idx = None
    for i, (rr, cc, h, w, _a) in enumerate(rects):
        if rr <= mr < rr + h and cc <= mc < cc + w:
            anchor_idx = i
            break

    dims: list[tuple[int, int]] = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh != 0:
            continue
        dims.append((hh, total_area // hh))
        if hh * hh != total_area:
            dims.append((total_area // hh, hh))

    def fits(H: int, W: int) -> bool:
        tr, tc = _top_left_from_marker(mr, mc, corner, H, W)
        return 0 <= tr and 0 <= tc and (tr + H) <= GH and (tc + W) <= GW

    dims = [(int(H), int(W)) for H, W in dims if fits(int(H), int(W))]
    dims.sort(key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    def compat(v1: np.ndarray, v2: np.ndarray) -> bool:
        a1 = np.array(v1, dtype=int)
        a2 = np.array(v2, dtype=int)
        m = (a1 != int(rect_col)) & (a2 != int(rect_col))
        return bool((a1[m] == a2[m]).all())

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
                if not can_place(i, r, c):
                    continue
                place(i, r, c)
                bt()
                if solved:
                    return
                unplace(i, r, c)

        bt()
        return occ2 if solved else None

    for H, W in dims:
        ass = assemble(H, W)
        if ass is None:
            continue
        return ass, _top_left_from_marker(mr, mc, corner, H, W)
    raise AssertionError("assembly failed")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered rectangles at the marker corner, but keep the scattered source
    pieces in place and erase the L marker from the final output.
    """
    g = np.array(grid, dtype=int)
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    marker_col, missing, _bbox, corner = _find_marker(g, bg)
    rects = _extract_rectangles(g, bg, marker_col)
    assert rects
    rect_col = _rect_color(rects)
    assembled, (tr, tc) = _assemble_rectangle(rects, rect_col, missing, corner, g.shape)

    out = g.copy()
    mr, mc = map(int, missing)
    for r, c in _marker_cells(corner, mr, mc):
        out[int(r), int(c)] = int(bg)
    paste(out, assembled, int(tr), int(tc))
    return out


def _forms_l_marker(points: np.ndarray) -> bool:
    if int(points.shape[0]) != 3:
        return False
    rs = points[:, 0]
    cs = points[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())
    if (r1 - r0) != 1 or (c1 - c0) != 1:
        return False
    all4 = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    return len(all4 - {tuple(map(int, p)) for p in points}) == 1


def _expected_output(
    canvas: np.ndarray,
    rect: np.ndarray,
    top_left: tuple[int, int],
    marker_cells: list[tuple[int, int]],
    bg: int,
) -> np.ndarray:
    out = np.array(canvas, dtype=int).copy()
    for r, c in marker_cells:
        out[int(r), int(c)] = int(bg)
    paste(out, rect, int(top_left[0]), int(top_left[1]))
    return out


def _gen_attempt() -> np.ndarray | None:
    # --- grid size ---
    GH = random.randint(10, 26)
    GW = random.randint(10, 26)

    # --- colors ---
    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    rect_col = int(palette[1])
    marker_col = int(palette[2])
    n_interior = random.randint(1, 3)
    interior_cols = [int(c) for c in palette[3 : 3 + n_interior]]

    # --- corner type and marker position ---
    corner = random.choice(["TL", "TR", "BL", "BR"])

    # --- assembled rectangle dimensions ---
    min_side, max_side = 5, min(22, min(GH, GW) - 3)
    if max_side < min_side:
        return None
    RH = random.randint(min_side, max_side)
    RW = random.randint(min_side, max_side)
    if random.random() < 0.5 and max_side >= 8:
        RH = random.randint(max(6, min_side), max_side)
        RW = random.randint(max(6, min_side), max_side)

    # --- marker placement ---
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
    marker_cells = _marker_cells(corner, mr, mc)
    if not all(0 <= r < GH and 0 <= c < GW for r, c in marker_cells):
        return None
    tr, tc = _top_left_from_marker(mr, mc, corner, RH, RW)

    # --- build assembled rectangle ---
    rect = np.full((RH, RW), int(rect_col), dtype=int)
    iH, iW = RH - 2, RW - 2
    if iH < 1 or iW < 1:
        return None
    interior = np.full((iH, iW), int(rect_col), dtype=int)

    target_fill = random.uniform(0.35, 0.85)
    target_pixels = max(1, int(target_fill * iH * iW))
    blob_occupied = np.zeros((iH, iW), dtype=bool)

    for ci, ic in enumerate(interior_cols):
        if ci == len(interior_cols) - 1:
            budget = target_pixels - int(blob_occupied.sum())
        else:
            budget = max(1, target_pixels // len(interior_cols))
        budget = max(1, int(budget))

        attempts = 0
        seed = None
        while attempts < 200:
            sr, sc = random.randint(0, iH - 1), random.randint(0, iW - 1)
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
            random.shuffle(frontier)
            nf: list[tuple[int, int]] = []
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
            interior[int(br), int(bc)] = int(ic)
            blob_occupied[int(br), int(bc)] = True

    non_rc_count = int(np.sum(interior != int(rect_col)))
    fill_ratio = non_rc_count / (iH * iW)
    if fill_ratio < 0.30 or fill_ratio > 0.90:
        return None
    rect[1:-1, 1:-1] = interior

    # --- partition rectangle into rectangular pieces ---
    def _rand_splits(total: int, min_s: int = 2, max_s: int = 7) -> list[int] | None:
        eff_max = min(max_s, max(min_s, random.randint(min_s, max_s)))
        segs: list[int] = []
        rem = int(total)
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

    row_segs = _rand_splits(RH)
    col_segs = _rand_splits(RW)
    if row_segs is None or col_segs is None:
        return None

    row_starts: list[int] = []
    s = 0
    for rs in row_segs:
        row_starts.append(s)
        s += rs
    col_starts: list[int] = []
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
        ris = [cell[0] for cell in cells]
        cis = [cell[1] for cell in cells]
        return min(ris), min(cis), max(ris), max(cis)

    def _pixel_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
        h = sum(row_segs[i] for i in range(r0i, r1i + 1))
        w = sum(col_segs[j] for j in range(c0i, c1i + 1))
        return int(h), int(w)

    def _count_pieces() -> int:
        return len(set(cell_id[i][j] for i in range(nr) for j in range(nc_grid)))

    target_pieces = random.randint(4, min(12, n_cells))
    merges: list[tuple[str, int, int]] = []
    for i in range(nr):
        for j in range(nc_grid):
            if j + 1 < nc_grid:
                merges.append(("h", i, j))
            if i + 1 < nr:
                merges.append(("v", i, j))
    random.shuffle(merges)

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
    seen: set[int] = set()
    for i in range(nr):
        for j in range(nc_grid):
            pid = int(cell_id[i][j])
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
            pieces.append((int(pr), int(pc), int(ph), int(pw)))

    if len(pieces) < 4 or len(pieces) > 12:
        return None

    for pr, pc, ph, pw in pieces:
        if ph < 2 or pw < 2 or ph > 7 or pw > 7:
            return None
        piece = rect[pr : pr + ph, pc : pc + pw]
        nrc = int(np.sum(piece != int(rect_col)))
        fill = nrc / (ph * pw)
        if fill < 0.10 or fill > 0.60:
            return None
        has_non_rc_edge = False
        for edge in [piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]]:
            if np.any(edge != int(rect_col)):
                has_non_rc_edge = True
                break
        if not has_non_rc_edge:
            return None
        if pr == 0 and not np.all(piece[0, :] == int(rect_col)):
            return None
        if pr + ph == RH and not np.all(piece[-1, :] == int(rect_col)):
            return None
        if pc == 0 and not np.all(piece[:, 0] == int(rect_col)):
            return None
        if pc + pw == RW and not np.all(piece[:, -1] == int(rect_col)):
            return None

    # --- scatter pieces while reserving the future assembled rectangle area ---
    reserve = -1
    canvas = np.full((GH, GW), int(bg), dtype=int)
    canvas[tr : tr + RH, tc : tc + RW] = int(reserve)
    for r, c in marker_cells:
        canvas[int(r), int(c)] = int(marker_col)

    piece_arrays = [rect[pr : pr + ph, pc : pc + pw].copy() for pr, pc, ph, pw in pieces]
    order = list(range(len(piece_arrays)))
    random.shuffle(order)
    for idx in order:
        pa = piece_arrays[idx]
        try:
            r0, c0 = random_free_location(canvas, pa, transparent=int(bg), padding=1)
        except ValueError:
            return None
        paste(canvas, pa, int(r0), int(c0))

    canvas[canvas == int(reserve)] = int(bg)

    # Reject accidental extra marker-like colors.
    for col, n in Counter(map(int, canvas.flatten())).items():
        if int(col) in {int(bg), int(marker_col)} or int(n) != 3:
            continue
        if _forms_l_marker(np.argwhere(canvas == int(col))):
            return None

    expected_out = _expected_output(canvas, rect, (tr, tc), marker_cells, bg)
    kept_source_pixels = int(np.sum((canvas != int(bg)) & (canvas != int(marker_col))))
    if kept_source_pixels <= 0:
        return None
    if int(np.sum(expected_out == int(marker_col))) != 0:
        return None

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
