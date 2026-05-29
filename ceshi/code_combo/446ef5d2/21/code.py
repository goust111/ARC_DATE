# Combo variant of 446ef5d2 (index 21)
# combo_spec: D1/3+D4/3+D8/0
# Name: 翻转匹配+镜像输出+单像素标记
#
# Design analysis
# ---------------
# a) Concrete behavior and trace
# - Input keeps the original "scattered rectangular fragments on background" layout,
#   but the old 3-cell L marker is replaced by one isolated marker pixel.
# - The marker pixel does not encode direction by local shape anymore. Its quadrant
#   relative to the whole grid center encodes which rectangle corner it denotes:
#   TL / TR / BL / BR.
# - Pieces are still axis-aligned rectangles, but each one may need either no change,
#   a left-right flip, or an up-down flip before it participates in the assembly.
# - After the rectangle has been reconstructed, the whole rectangle is mirrored once
#   more before writing the output:
#     left-side corners (TL, BL) -> left-right reflection
#     right-side corners (TR, BR) -> up-down reflection
# - Small trace:
#   1. Original task: find L-marker gap -> place canonical assembled rectangle there.
#   2. This combo: find singleton marker -> infer corner from quadrant -> flip some
#      pieces during edge matching -> assemble canonical rectangle -> reflect whole
#      rectangle according to that corner -> paste reflected result at the same box.
#   3. Difference chain:
#      parsing changes (singleton marker + quadrant),
#      assembly search changes (piece flips allowed),
#      output changes (whole-rectangle reflection).
#
# b) Complexity calibration
# - The marker system has only 4 cases, one per quadrant, so humans can infer it from
#   a few examples.
# - The reflection stage has only 2 axes, keyed by corner side, so it is visible but
#   not combinatorially messy.
# - The generator rejects samples unless:
#   1. at least two scattered pieces are truly mirrored in the input, and
#   2. reflecting the assembled rectangle changes many cells.
#   This prevents silent collapse back to either the original task or a weak single
#   variant where the extra degrees of freedom are decorative.
#
# c) Implicit-assumption audit against the original solver
# - "Marker is a 3-cell L of a color appearing exactly 3 times" is invalid now.
#   We must instead find the unique singleton non-bg connected component.
# - "The missing L corner is the anchor cell" is invalid now. The marker pixel itself
#   is the rectangle corner cell, and its quadrant supplies the corner label.
# - "Each piece has one fixed raster" is invalid under D1/3. The assembly search must
#   try each piece under {identity, fliplr, flipud}, deduplicated for symmetric cases.
# - "Assembled rectangle is already the final output" is invalid under D4/3. We must
#   reflect the finished rectangle before placing it in the output grid.
# - The original anchor pre-placement shortcut is unsafe here because the singleton
#   marker occupies the corner cell and scattered pieces never overlap it. The combo
#   solver therefore relies entirely on the backtracking tiler.
#
# d) Placement legality
# - The generator chooses a rectangle first, then chooses a marker location in the
#   corresponding quadrant with room for the rectangle and away from the center lines.
# - Pieces are scattered with `random_free_location(..., padding=1)`, so they do not
#   overlap each other or the marker.
# - Output placement stays legal because reflection preserves the rectangle bbox.
#
# e) Visual observability
# - The marker change is visible because the input now contains one isolated pixel
#   instead of an L shape.
# - The quadrant rule is visible because the output rectangle expands from different
#   corners depending on where the marker sits.
# - The flip rule is visible because asymmetric piece edge signatures appear mirrored
#   in the scattered input, yet the final rectangle restores a coherent pattern.
# - The reflection rule is visible because the produced rectangle is a mirrored version
#   of the canonical assembly; the generator rejects near-symmetric cases.
#
# f) Visual cleanliness
# - No overlays are stacked on top of pieces. The singleton marker is isolated, pieces
#   remain separated by background, and the final output contains only one rectangle.
# - The combo therefore changes reasoning in three stages without making the scene
#   visually cluttered.
#
# g) New solver assumptions introduced here and how the generator guarantees them
# - Exactly one singleton non-bg component exists: the generator uses one 1x1 marker
#   and keeps all pieces at least 2x2.
# - Needed flips are among {identity, fliplr, flipud}: the generator scatters pieces
#   using only those transforms.
# - The reflection is non-trivial: the generator measures cell differences between the
#   canonical rectangle and its reflected version and rejects weak cases.
# - The flip degree of freedom matters: the generator checks that a no-flip solver does
#   not reproduce the expected combo output on accepted samples.

from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, paste, random_free_location


LAST_GEN_INFO: dict[str, object] = {}


def _corner_candidates_from_quadrant(mr: int, mc: int, gh: int, gw: int) -> list[str]:
    if gh % 2 == 1 and mr == gh // 2:
        row_tags = ["T", "B"]
    else:
        row_tags = ["T"] if mr < (gh / 2.0) else ["B"]

    if gw % 2 == 1 and mc == gw // 2:
        col_tags = ["L", "R"]
    else:
        col_tags = ["L"] if mc < (gw / 2.0) else ["R"]

    return [rt + ct for rt in row_tags for ct in col_tags]


def _top_left_from_corner(mr: int, mc: int, h: int, w: int, corner: str) -> tuple[int, int]:
    if corner == "TL":
        return int(mr), int(mc)
    if corner == "TR":
        return int(mr), int(mc - w + 1)
    if corner == "BL":
        return int(mr - h + 1), int(mc)
    return int(mr - h + 1), int(mc - w + 1)


def _reflect_rect(rect: np.ndarray, corner: str) -> np.ndarray:
    if corner in ("TL", "BL"):
        return np.fliplr(rect)
    return np.flipud(rect)


def _find_singleton_marker(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    non_bg = g != int(bg)
    lab, nlab = label(non_bg.astype(int), structure=STRUCT4)
    marker = None
    for k in range(1, int(nlab) + 1):
        pts = np.argwhere(lab == k)
        if pts.shape[0] != 1:
            continue
        rr, cc = map(int, pts[0])
        if marker is not None:
            raise AssertionError("expected exactly one singleton marker")
        marker = (int(g[rr, cc]), rr, cc)
    if marker is None:
        raise AssertionError("singleton marker not found")
    return marker


def _extract_rectangles(g: np.ndarray, bg: int, mr: int, mc: int) -> list[tuple[int, int, int, int, np.ndarray]]:
    gh, gw = g.shape
    occ = g != int(bg)
    occ[mr, mc] = False
    used = np.zeros_like(occ, dtype=bool)
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
        rects.append((rr, cc, h, w, g[rr : rr + h, cc : cc + w].copy()))
    return rects


def _unique_piece_variants(piece: np.ndarray, allow_flip: bool) -> list[np.ndarray]:
    raw = [np.array(piece, dtype=int)]
    if allow_flip:
        raw.extend([np.fliplr(piece), np.flipud(piece)])
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    for arr in raw:
        key = (tuple(map(int, arr.shape)), tuple(map(int, arr.flatten())))
        if key in seen:
            continue
        seen.add(key)
        out.append(np.array(arr, dtype=int))
    return out


def _render_output(
    rect: np.ndarray,
    gh: int,
    gw: int,
    bg: int,
    mr: int,
    mc: int,
    corner: str,
    *,
    apply_reflection: bool,
) -> np.ndarray:
    h, w = map(int, rect.shape)
    out = np.full((gh, gw), int(bg), dtype=int)
    placed = _reflect_rect(rect, corner) if apply_reflection else rect
    tr, tc = _top_left_from_corner(mr, mc, h, w, corner)
    out[tr : tr + h, tc : tc + w] = placed
    return out


def _solve_core(grid: np.ndarray, *, allow_flip: bool, apply_reflection: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    gh, gw = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    _mark, mr, mc = _find_singleton_marker(g, bg)
    corner_candidates = _corner_candidates_from_quadrant(mr, mc, gh, gw)

    rects = _extract_rectangles(g, bg, mr, mc)
    if not rects:
        raise AssertionError("no pieces found")

    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))
    allpix: list[int] = []
    for _r, _c, _h, _w, a in rects:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    pieces = [a for _r, _c, _h, _w, a in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]
    variants = [_unique_piece_variants(p, allow_flip=allow_flip) for p in pieces]

    dims: list[tuple[int, int]] = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh != 0:
            continue
        dims.append((hh, total_area // hh))
        if hh * hh != total_area:
            dims.append((total_area // hh, hh))
    dims = sorted(dims, key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    def compat(v1: np.ndarray, v2: np.ndarray) -> bool:
        vv1 = np.array(v1, dtype=int)
        vv2 = np.array(v2, dtype=int)
        m = (vv1 != int(rect_col)) & (vv2 != int(rect_col))
        return bool((vv1[m] == vv2[m]).all())

    def assemble(h_big: int, w_big: int) -> np.ndarray | None:
        occ2 = np.full((h_big, w_big), -1, dtype=int)
        usedp = [False] * len(pieces)

        def boundary_ok(a: np.ndarray, r: int, c: int) -> bool:
            h, w = map(int, a.shape)
            if r == 0 and not bool(np.all(a[0, :] == int(rect_col))):
                return False
            if (r + h) == h_big and not bool(np.all(a[-1, :] == int(rect_col))):
                return False
            if c == 0 and not bool(np.all(a[:, 0] == int(rect_col))):
                return False
            if (c + w) == w_big and not bool(np.all(a[:, -1] == int(rect_col))):
                return False
            return True

        def neighbor_ok(a: np.ndarray, r: int, c: int) -> bool:
            h, w = map(int, a.shape)
            if c > 0:
                left = occ2[r : r + h, c - 1]
                m = left != -1
                if m.any() and (not compat(left[m], a[:, 0][m])):
                    return False
            if (c + w) < w_big:
                right = occ2[r : r + h, c + w]
                m = right != -1
                if m.any() and (not compat(right[m], a[:, -1][m])):
                    return False
            if r > 0:
                up = occ2[r - 1, c : c + w]
                m = up != -1
                if m.any() and (not compat(up[m], a[0, :][m])):
                    return False
            if (r + h) < h_big:
                down = occ2[r + h, c : c + w]
                m = down != -1
                if m.any() and (not compat(down[m], a[-1, :][m])):
                    return False
            return True

        def can_place(a: np.ndarray, r: int, c: int) -> bool:
            h, w = map(int, a.shape)
            if (r + h) > h_big or (c + w) > w_big:
                return False
            if bool((occ2[r : r + h, c : c + w] != -1).any()):
                return False
            if not boundary_ok(a, r, c):
                return False
            if not neighbor_ok(a, r, c):
                return False
            return True

        def place(i: int, a: np.ndarray, r: int, c: int) -> None:
            h, w = map(int, a.shape)
            occ2[r : r + h, c : c + w] = a
            usedp[i] = True

        def unplace(i: int, a: np.ndarray, r: int, c: int) -> None:
            h, w = map(int, a.shape)
            occ2[r : r + h, c : c + w] = -1
            usedp[i] = False

        order = sorted(
            range(len(pieces)),
            key=lambda i: (len(variants[i]), -(ph[i] * pw[i]), -ph[i], -pw[i], i),
        )

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
                for a in variants[i]:
                    if not can_place(a, r, c):
                        continue
                    place(i, a, r, c)
                    bt()
                    if solved:
                        return
                    unplace(i, a, r, c)

        bt()
        return occ2 if solved else None

    for corner in corner_candidates:
        valid_dims = []
        for h_big, w_big in dims:
            tr, tc = _top_left_from_corner(mr, mc, h_big, w_big, corner)
            if 0 <= tr and 0 <= tc and (tr + h_big) <= gh and (tc + w_big) <= gw:
                valid_dims.append((int(h_big), int(w_big)))
        for h_big, w_big in valid_dims:
            ass = assemble(h_big, w_big)
            if ass is None:
                continue
            return _render_output(
                ass,
                gh,
                gw,
                bg,
                mr,
                mc,
                corner,
                apply_reflection=apply_reflection,
            )

    raise AssertionError("assembly failed")


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_core(grid, allow_flip=True, apply_reflection=True)


def _solve_without_flip(grid: np.ndarray) -> np.ndarray:
    return _solve_core(grid, allow_flip=False, apply_reflection=True)


def _solve_without_reflection(grid: np.ndarray) -> np.ndarray:
    return _solve_core(grid, allow_flip=True, apply_reflection=False)


def _rand_splits(total: int, rng: random.Random, min_s: int = 2, max_s: int = 7) -> list[int] | None:
    eff_max = min(max_s, max(min_s, rng.randint(min_s, max_s)))
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
            s = rng.randint(min_s, hi)
            segs.append(s)
            rem -= s
    return segs


def _build_rect(
    rh: int,
    rw: int,
    rect_col: int,
    interior_cols: list[int],
    rng: random.Random,
) -> np.ndarray | None:
    rect = np.full((rh, rw), int(rect_col), dtype=int)
    ih, iw = rh - 2, rw - 2
    if ih < 1 or iw < 1:
        return None

    interior = np.full((ih, iw), int(rect_col), dtype=int)
    target_fill = rng.uniform(0.38, 0.82)
    target_pixels = max(1, int(target_fill * ih * iw))
    blob_occupied = np.zeros((ih, iw), dtype=bool)

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
            sr, sc = rng.randint(0, ih - 1), rng.randint(0, iw - 1)
            if blob_occupied[sr, sc]:
                attempts += 1
                continue
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < ih and 0 <= nc < iw and blob_occupied[nr, nc]:
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
                    if nr < 0 or nr >= ih or nc < 0 or nc >= iw:
                        continue
                    if blob_occupied[nr, nc]:
                        continue
                    conflict = False
                    for dr2 in (-1, 0, 1):
                        for dc2 in (-1, 0, 1):
                            if dr2 == 0 and dc2 == 0:
                                continue
                            ar, ac = nr + dr2, nc + dc2
                            if 0 <= ar < ih and 0 <= ac < iw and blob_occupied[ar, ac] and (ar, ac) not in blob:
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
            interior[br, bc] = int(ic)
            blob_occupied[br, bc] = True

    fill_ratio = float(np.mean(interior != int(rect_col))) if interior.size else 0.0
    if fill_ratio < 0.30 or fill_ratio > 0.90:
        return None
    rect[1:-1, 1:-1] = interior
    return rect


def _partition_rect(
    rect: np.ndarray,
    rh: int,
    rw: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    row_segs = _rand_splits(rh, rng)
    col_segs = _rand_splits(rw, rng)
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

    target_pieces = rng.randint(4, min(10, n_cells))
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
            if bb is None:
                return None
            r0i, c0i, r1i, c1i = bb
            pr = row_starts[r0i]
            pc = col_starts[c0i]
            ph = sum(row_segs[ii] for ii in range(r0i, r1i + 1))
            pw = sum(col_segs[jj] for jj in range(c0i, c1i + 1))
            pieces.append((pr, pc, ph, pw))
    return pieces


def _piece_has_non_rc_edge(piece: np.ndarray, rect_col: int) -> bool:
    for edge in [piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]]:
        if bool(np.any(edge != int(rect_col))):
            return True
    return False


def _scatter_transform_options(piece: np.ndarray) -> list[tuple[str, np.ndarray]]:
    opts = [("id", np.array(piece, dtype=int))]
    h = np.fliplr(piece)
    if not bool(np.array_equal(h, piece)):
        opts.append(("h", np.array(h, dtype=int)))
    v = np.flipud(piece)
    if not bool(np.array_equal(v, piece)):
        opts.append(("v", np.array(v, dtype=int)))
    return opts


def _pick_marker_position(
    corner: str,
    gh: int,
    gw: int,
    rh: int,
    rw: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    top_max = gh // 2 - 1
    bottom_min = gh // 2 + (gh % 2)
    left_max = gw // 2 - 1
    right_min = gw // 2 + (gw % 2)

    if corner == "TL":
        mr_min, mr_max = 1, min(gh - rh - 1, top_max)
        mc_min, mc_max = 1, min(gw - rw - 1, left_max)
    elif corner == "TR":
        mr_min, mr_max = 1, min(gh - rh - 1, top_max)
        mc_min, mc_max = max(rw, right_min), gw - 2
    elif corner == "BL":
        mr_min, mr_max = max(rh, bottom_min), gh - 2
        mc_min, mc_max = 1, min(gw - rw - 1, left_max)
    else:
        mr_min, mr_max = max(rh, bottom_min), gh - 2
        mc_min, mc_max = max(rw, right_min), gw - 2

    if mr_min > mr_max or mc_min > mc_max:
        return None
    return rng.randint(mr_min, mr_max), rng.randint(mc_min, mc_max)


def _gen_attempt() -> np.ndarray | None:
    global LAST_GEN_INFO

    rng = random.Random()

    gh = rng.randint(12, 24)
    gw = rng.randint(12, 24)

    palette = list(range(10))
    rng.shuffle(palette)
    bg = int(palette[0])
    rect_col = int(palette[1])
    marker_col = int(palette[2])
    n_interior = rng.randint(1, 3)
    interior_cols = [int(x) for x in palette[3 : 3 + n_interior]]

    corner = rng.choice(["TL", "TR", "BL", "BR"])

    min_side = 5
    max_side = min(14, min(gh, gw) - 4)
    if max_side < min_side:
        return None
    rh = rng.randint(min_side, max_side)
    rw = rng.randint(min_side, max_side)
    if rng.random() < 0.45 and max_side >= 8:
        rh = rng.randint(max(6, min_side), max_side)
        rw = rng.randint(max(6, min_side), max_side)

    marker_pos = _pick_marker_position(corner, gh, gw, rh, rw, rng)
    if marker_pos is None:
        return None
    mr, mc = marker_pos

    rect = _build_rect(rh, rw, rect_col, interior_cols, rng)
    if rect is None:
        return None

    reflected_rect = _reflect_rect(rect, corner)
    reflection_diff = int(np.sum(rect != reflected_rect))
    if reflection_diff < max(6, int(0.10 * rh * rw)):
        return None

    pieces = _partition_rect(rect, rh, rw, rng)
    if pieces is None or len(pieces) < 4 or len(pieces) > 10:
        return None

    piece_arrays: list[np.ndarray] = []
    transformable_idxs: list[int] = []
    for i, (pr, pc, ph, pw) in enumerate(pieces):
        if ph < 2 or pw < 2 or ph > 7 or pw > 7:
            return None
        piece = rect[pr : pr + ph, pc : pc + pw].copy()
        piece_arrays.append(piece)
        fill = float(np.mean(piece != int(rect_col)))
        if fill < 0.10 or fill > 0.60:
            return None
        if not _piece_has_non_rc_edge(piece, rect_col):
            return None
        if pr == 0 and not bool(np.all(piece[0, :] == int(rect_col))):
            return None
        if pr + ph == rh and not bool(np.all(piece[-1, :] == int(rect_col))):
            return None
        if pc == 0 and not bool(np.all(piece[:, 0] == int(rect_col))):
            return None
        if pc + pw == rw and not bool(np.all(piece[:, -1] == int(rect_col))):
            return None
        if len(_scatter_transform_options(piece)) > 1:
            transformable_idxs.append(i)

    if len(transformable_idxs) < 2:
        return None

    forced = rng.sample(transformable_idxs, rng.randint(2, min(4, len(transformable_idxs))))
    chosen_piece_arrays: list[np.ndarray] = []
    flipped_count = 0
    h_count = 0
    v_count = 0
    for i, piece in enumerate(piece_arrays):
        opts = _scatter_transform_options(piece)
        if i in forced:
            non_id = [(tag, arr) for tag, arr in opts if tag != "id"]
            if not non_id:
                return None
            tag, arr = rng.choice(non_id)
        else:
            pool = opts if rng.random() < 0.35 else [opts[0]]
            tag, arr = rng.choice(pool)
        chosen_piece_arrays.append(arr.copy())
        if tag != "id":
            flipped_count += 1
            if tag == "h":
                h_count += 1
            elif tag == "v":
                v_count += 1

    if flipped_count < 2:
        return None

    canvas = np.full((gh, gw), int(bg), dtype=int)
    canvas[mr, mc] = int(marker_col)

    order = list(range(len(chosen_piece_arrays)))
    rng.shuffle(order)
    for idx in order:
        pa = chosen_piece_arrays[idx]
        try:
            r0, c0 = random_free_location(canvas, pa, transparent=int(bg), padding=1)
        except ValueError:
            return None
        paste(canvas, pa, r0, c0)

    expected_out = _render_output(
        rect,
        gh,
        gw,
        bg,
        mr,
        mc,
        corner,
        apply_reflection=True,
    )

    try:
        actual_out = main(canvas)
    except Exception:
        return None
    if not bool(np.array_equal(actual_out, expected_out)):
        return None

    try:
        noflip_out = _solve_without_flip(canvas)
        if bool(np.array_equal(noflip_out, expected_out)):
            return None
    except Exception:
        pass

    try:
        unreflected_out = _solve_without_reflection(canvas)
    except Exception:
        return None
    if bool(np.array_equal(unreflected_out, expected_out)):
        return None

    combo_delta = int(np.sum(unreflected_out != expected_out))
    if combo_delta < max(6, int(0.10 * rh * rw)):
        return None

    LAST_GEN_INFO = {
        "corner": corner,
        "grid_shape": (gh, gw),
        "rect_shape": (rh, rw),
        "marker": (mr, mc),
        "piece_count": len(piece_arrays),
        "transformable_piece_count": len(transformable_idxs),
        "flipped_piece_count": flipped_count,
        "horizontal_flips": h_count,
        "vertical_flips": v_count,
        "reflection_diff": reflection_diff,
        "combo_delta": combo_delta,
    }
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
