# Variant of 446ef5d2
# Dimension: D19 — 背景与前景的关系 (meta)
# Change: 背景色是网格中最常见的颜色，碎片散布在背景上，输出中背景保持不变... → 背景带有噪声图案

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import label

_HERE = Path(__file__).resolve()
for _parent in [_HERE.parent, *_HERE.parents]:
    _solver_dir = _parent / "solvers"
    if (_solver_dir / "arc_utils.py").exists():
        if str(_solver_dir) not in sys.path:
            sys.path.append(str(_solver_dir))
        break

from arc_utils import STRUCT4, paste, random_free_location

"""
Design analysis for the D19 variant
==================================

a) Concrete behavior and trace
YAML-style summary:
  background:
    old: one solid color, detected as global majority
    new: a 2-color checkerboard covering the whole grid
    consequence: foreground is no longer "all non-bg-color cells";
      it is "all cells that disagree with the inferred checkerboard value at that coordinate"
  visible effect:
    input: scattered rectangular fragments and the 3-cell L marker sit on top of an exposed checkerboard
    output: the assembled rectangle is pasted onto the same checkerboard instead of a flat canvas
  tiny trace:
    old solver:
      - pick most common color 8 as bg
      - treat every non-8 cell as foreground
    new solver:
      - infer checkerboard colors, e.g. parity-0 -> 8, parity-1 -> 5
      - reconstruct the full expected checkerboard
      - a cell colored 5 on the "other parity" is background, not fragment
      - only cells differing from that reconstructed checkerboard become foreground

b) Complexity calibration
  - The pattern is fixed to a checkerboard, not an arbitrary texture family. That keeps the new
    reasoning focused on segmentation instead of turning the task into open-ended texture inference.
  - Two distinct background colors make the change visually obvious.
  - Rectangle area is capped so enough background remains exposed; otherwise one checkerboard phase
    might stop being the majority on its parity class and segmentation would become brittle.

c) Implicit assumptions in the original main() that break here
  - `bg = global majority color`: invalid, because the checkerboard has two background colors.
  - `occ = (g != bg) & (g != mark)`: invalid, because one of the two checkerboard colors would be
    misclassified as foreground everywhere.
  - `out = full(bg)`: invalid, because output must preserve the patterned background.
  The new code replaces these with checkerboard inference, anomaly masking against that pattern,
  and output initialized from the reconstructed checkerboard.

d) Placement legality
  - The generator still uses rectangular non-overlapping pieces with a one-cell gap.
  - Because the visible canvas is never "empty" anymore, placement is checked on a separate binary
    occupancy canvas; the checkerboard itself is just visual background.
  - Marker cells are inserted before scattering pieces, and piece placement uses padding so no piece
    touches the marker or another piece.

e) Visual observability
  - Across samples, the eye can immediately see that many background pixels of two colors survive
    unchanged from input to output, while the fragments alone are rearranged.
  - The segmentation burden is visible because one of the background colors can look superficially
    similar to fragment colors in frequency, yet it repeats globally with checkerboard regularity.

f) Cleanliness
  - Only two background colors are used, so the bottom layer remains readable.
  - Pieces do not reuse either background color; this prevents background/piece ambiguity inside
    the segmentation step.

g) Color inference
  - No rule depends on numeric color ids. The two background colors are inferred from repeated
    positional structure (checkerboard parity), not from color ranking beyond local frequency on
    each parity class.

New solver assumptions introduced by the variant
  - The background is always a 2-color checkerboard aligned to global grid parity.
  - Piece and marker colors are distinct from both checkerboard colors.
  - Enough checkerboard remains exposed that, on each parity class, the true background color is
    the strongest support signal. The generator enforces this by limiting total piece area.
"""


def _checkerboard(shape: tuple[int, int], even_col: int, odd_col: int) -> np.ndarray:
    rr, cc = np.indices(shape)
    return np.where(((rr + cc) & 1) == 0, int(even_col), int(odd_col)).astype(int)


def _infer_checkerboard_colors(g: np.ndarray) -> tuple[int, int]:
    counts = [Counter(), Counter()]
    for (r, c), v in np.ndenumerate(g):
        counts[(int(r) + int(c)) & 1][int(v)] += 1

    colors = sorted({int(v) for v in g.flatten()})
    best_pair = None
    best_score = -1
    for even_col in colors:
        for odd_col in colors:
            if even_col == odd_col:
                continue
            score = counts[0][even_col] + counts[1][odd_col]
            if score > best_score:
                best_score = score
                best_pair = (int(even_col), int(odd_col))

    assert best_pair is not None
    return best_pair


def _component_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
    labs, nlab = label(mask.astype(int), structure=STRUCT4)
    comps = []
    for lab in range(1, int(nlab) + 1):
        pts = np.argwhere(labs == lab)
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        comps.append((r0, c0, r1, c1, pts))
    comps.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return comps


def _find_marker(g: np.ndarray, fg_mask: np.ndarray) -> tuple[int, tuple[int, int], str, np.ndarray]:
    marker_mask = np.zeros_like(fg_mask, dtype=bool)
    info = None
    for r0, c0, r1, c1, pts in _component_boxes(fg_mask):
        if len(pts) != 3:
            continue
        cols = {int(g[int(r), int(c)]) for r, c in pts}
        if len(cols) != 1:
            continue
        if (r1 - r0) != 1 or (c1 - c0) != 1:
            continue
        all4 = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        miss = list(all4 - {tuple(map(int, p)) for p in pts})
        if len(miss) != 1:
            continue
        info = (int(next(iter(cols))), tuple(map(int, miss[0])), (r0, c0, r1, c1), pts)
        break

    assert info is not None
    mark, missing, (r0, c0, r1, c1), pts = info
    for r, c in pts:
        marker_mask[int(r), int(c)] = True

    if missing == (r0, c0):
        corner = "BR"
    elif missing == (r0, c1):
        corner = "BL"
    elif missing == (r1, c0):
        corner = "TR"
    else:
        corner = "TL"
    return mark, missing, corner, marker_mask


def _extract_rectangles(g: np.ndarray, occ_mask: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
    rects = []
    for r0, c0, r1, c1, pts in _component_boxes(occ_mask):
        box = occ_mask[r0 : r1 + 1, c0 : c1 + 1]
        assert int(len(pts)) == int(box.size)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        rects.append((r0, c0, h, w, g[r0 : r1 + 1, c0 : c1 + 1].copy()))
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into a rectangle at an L-marker corner.

    Variant-specific change:
    - The background is a 2-color checkerboard rather than a flat color.
    - The solver must first reconstruct that checkerboard, then treat deviations from it as
      foreground fragments / marker pixels.
    """
    g = np.array(grid, dtype=int)
    GH, GW = g.shape

    bg_even, bg_odd = _infer_checkerboard_colors(g)
    bg_pattern = _checkerboard((GH, GW), bg_even, bg_odd)

    fg_mask = g != bg_pattern
    mark, missing, corner, marker_mask = _find_marker(g, fg_mask)

    occ = fg_mask & (~marker_mask)
    rects = _extract_rectangles(g, occ)
    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))

    allpix: list[int] = []
    for _r, _c, _h, _w, a in rects:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    mr, mc = map(int, missing)
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
            used_dim = (H, W)
            break

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

    out = bg_pattern.copy()
    tr, tc = map(int, top_left)
    out[tr : tr + H, tc : tc + W] = ass
    return out


def _gen_attempt() -> np.ndarray | None:
    import random as _r

    GH = _r.randint(10, 26)
    GW = _r.randint(10, 26)

    palette = list(range(10))
    _r.shuffle(palette)
    bg_even = palette[0]
    bg_odd = palette[1]
    rect_col = palette[2]
    marker_col = palette[3]
    n_interior = _r.randint(1, 3)
    interior_cols = palette[4 : 4 + n_interior]

    corner = _r.choice(["TL", "TR", "BL", "BR"])

    max_side = min(16, min(GH, GW) - 3)
    dims = []
    for RH in range(5, max_side + 1):
        for RW in range(5, max_side + 1):
            area = RH * RW
            if area < 30:
                continue
            if area > int(0.38 * GH * GW):
                continue
            dims.append((RH, RW))
    if not dims:
        return None
    dims.sort(key=lambda t: (t[0] * t[1], t[0], t[1]))
    RH, RW = _r.choice(dims[len(dims) // 3 :])

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

    bg_pattern = _checkerboard((GH, GW), bg_even, bg_odd)
    canvas = bg_pattern.copy()
    occ_canvas = np.zeros((GH, GW), dtype=int)

    for cr, cc in marker_cells:
        canvas[cr, cc] = marker_col
        occ_canvas[cr, cc] = 1

    piece_arrays = [rect[pr : pr + ph, pc : pc + pw].copy() for pr, pc, ph, pw in pieces]
    order = list(range(len(piece_arrays)))
    _r.shuffle(order)
    for idx in order:
        pa = piece_arrays[idx]
        try:
            r0, c0 = random_free_location(
                occ_canvas,
                np.ones(pa.shape, dtype=int),
                transparent=0,
                padding=1,
            )
        except ValueError:
            return None
        paste(canvas, pa, r0, c0)
        occ_canvas[r0 : r0 + pa.shape[0], c0 : c0 + pa.shape[1]] = 1

    if corner == "TL":
        tr, tc = mr, mc
    elif corner == "TR":
        tr, tc = mr, mc - RW + 1
    elif corner == "BL":
        tr, tc = mr - RH + 1, mc
    else:
        tr, tc = mr - RH + 1, mc - RW + 1

    expected_out = bg_pattern.copy()
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
