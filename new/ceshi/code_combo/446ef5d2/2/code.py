from __future__ import annotations

# Combo variant of 446ef5d2 (index 2)
# combo_spec: D1/3+D8/0
# Name: 翻转匹配配合简化标记
#
# Design analysis
# - Concrete behavior and trace:
#   The original task used a 3-pixel L marker whose missing cell directly gave the
#   rectangle corner, and every scattered rectangular piece had one fixed raster.
#   This combo changes both of those steps:
#   1. The marker is now a single isolated pixel. Its quadrant relative to the whole
#      grid center encodes which rectangle corner it denotes:
#      TL quadrant -> rectangle grows down-right from the marker,
#      TR quadrant -> down-left,
#      BL quadrant -> up-right,
#      BR quadrant -> up-left.
#   2. Each scattered piece may be used as-is, flipped left-right, or flipped
#      top-bottom before placement. Rotations are still forbidden.
#   Example trace: suppose a 3x4 piece in the final rectangle should sit near the
#   top edge, with a distinctive colored stripe on its bottom border. In the input,
#   that same piece may be vertically flipped, so the stripe appears on its top
#   border instead. Under the old solver, the edge signatures no longer align. Under
#   this combo solver, the singleton marker first fixes the rectangle corner via its
#   quadrant, then backtracking may mirror the piece to recover the correct border.
# - Complexity calibration:
#   The marker is visually simpler than the old L, so quadrant coding must be obvious
#   across samples. The generator therefore keeps the marker off center lines and
#   spreads corner choices across all four quadrants. To keep the flip dimension
#   active instead of decorative, the generator rejects cases unless at least two
#   scattered pieces are genuinely mirrored and a no-flip solver no longer reproduces
#   the canonical output.
# - Implicit-assumption audit against the original main():
#   The old "find a color that appears exactly 3 times in a 2x2-minus-1 shape" is
#   invalid; we must instead find the unique 1-cell non-background connected
#   component.
#   The old `missing` cell doubled as the future rectangle corner. That disappears:
#   now the marker cell itself is the corner cell.
#   The old anchor-piece shortcut assumed some input piece already occupied that
#   corner cell. That is false here because the marker occupies it in the input, so
#   the shortcut is removed entirely.
#   The old `pieces[i]` fixed-raster assumption is also invalid. Boundary checks and
#   neighbor matching must be applied to transformed piece rasters, not the observed
#   raster alone.
# - Placement legality:
#   The generator still uses output-first construction, partitions the final rectangle
#   into axis-aligned rectangles, then scatters them with at least one background cell
#   of separation. The singleton marker is placed first. Pieces are only mirrored,
#   never rotated, so they remain rectangular and can still be scattered safely.
# - Visual observability:
#   Human solvers can see two repeated cues from only a few examples:
#   1. the lone marker pixel's quadrant predicts the rectangle expansion direction;
#   2. some pieces have asymmetric edge/interior patterns whose handedness must be
#      mirrored before assembly. The visual change is therefore structural rather than
#      numeric: no color-id ordering is used.
# - Visual cleanliness:
#   Pieces remain separated by background; the marker is a single isolated pixel; and
#   every piece is at least 2x2, so the marker cannot be mistaken for a piece.
# - New solver assumptions introduced here and guaranteed by the generator:
#   1. Exactly one singleton non-background component exists, and it is the marker.
#   2. Any needed mirror is recoverable by one of {identity, fliplr, flipud}.
#   3. The marker is not placed on the grid's center row or center column, so the
#      quadrant-to-corner mapping is unambiguous on generated samples.
#   4. The canonical target remains one axis-aligned rectangle whose border is all
#      `rect_col`, with piece matching still driven by non-`rect_col` edge agreement.

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, paste, random_free_location


def _corner_from_quadrant(mr: int, mc: int, gh: int, gw: int) -> list[str]:
    if gh % 2 == 1 and mr == gh // 2:
        row_tags = ["T", "B"]
    else:
        row_tags = ["T"] if mr < (gh / 2.0) else ["B"]

    if gw % 2 == 1 and mc == gw // 2:
        col_tags = ["L", "R"]
    else:
        col_tags = ["L"] if mc < (gw / 2.0) else ["R"]

    out = []
    for rt in row_tags:
        for ct in col_tags:
            out.append(rt + ct)
    return out


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


def _top_left_for(corner: str, mr: int, mc: int, h: int, w: int) -> tuple[int, int]:
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - w + 1
    if corner == "BL":
        return mr - h + 1, mc
    return mr - h + 1, mc - w + 1


def _solve_core(grid: np.ndarray, allow_flip: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    gh, gw = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    _mark, mr, mc = _find_singleton_marker(g, bg)
    corner_candidates = _corner_from_quadrant(mr, mc, gh, gw)

    rects = _extract_rectangles(g, bg, mr, mc)
    if not rects:
        raise AssertionError("no pieces")

    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))
    allpix: list[int] = []
    for _r, _c, _h, _w, a in rects:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    pieces = [a for _r, _c, _h, _w, a in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]
    variants = [_unique_piece_variants(p, allow_flip=allow_flip) for p in pieces]

    dims = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh != 0:
            continue
        dims.append((hh, total_area // hh))
        if hh * hh != total_area:
            dims.append((total_area // hh, hh))
    dims = sorted(dims, key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    def fits(corner: str, h: int, w: int) -> bool:
        tr, tc = _top_left_for(corner, mr, mc, h, w)
        return 0 <= tr and 0 <= tc and (tr + h) <= gh and (tc + w) <= gw

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

    ass = None
    used_dim = None
    used_corner = None
    for corner in corner_candidates:
        valid_dims = [(int(hh), int(ww)) for hh, ww in dims if fits(corner, int(hh), int(ww))]
        for h_big, w_big in valid_dims:
            ass = assemble(h_big, w_big)
            if ass is not None:
                used_dim = (h_big, w_big)
                used_corner = corner
                break
        if ass is not None:
            break

    if ass is None or used_dim is None or used_corner is None:
        raise AssertionError("assembly failed")

    tr, tc = _top_left_for(used_corner, mr, mc, used_dim[0], used_dim[1])
    out = np.full((gh, gw), int(bg), dtype=int)
    out[tr : tr + used_dim[0], tc : tc + used_dim[1]] = ass
    return out


def _solve_without_flip(grid: np.ndarray) -> np.ndarray:
    return _solve_core(grid, allow_flip=False)


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_core(grid, allow_flip=True)


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


def _build_rect(rh: int, rw: int, rect_col: int, interior_cols: list[int], rng: random.Random) -> np.ndarray | None:
    rect = np.full((rh, rw), int(rect_col), dtype=int)
    i_h, i_w = rh - 2, rw - 2
    if i_h < 1 or i_w < 1:
        return None

    interior = np.full((i_h, i_w), int(rect_col), dtype=int)
    target_fill = rng.uniform(0.35, 0.85)
    target_pixels = max(1, int(target_fill * i_h * i_w))
    blob_occupied = np.zeros((i_h, i_w), dtype=bool)

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
            sr, sc = rng.randint(0, i_h - 1), rng.randint(0, i_w - 1)
            if blob_occupied[sr, sc]:
                attempts += 1
                continue
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < i_h and 0 <= nc < i_w and blob_occupied[nr, nc]:
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
                    if nr < 0 or nr >= i_h or nc < 0 or nc >= i_w:
                        continue
                    if blob_occupied[nr, nc]:
                        continue
                    conflict = False
                    for dr2 in (-1, 0, 1):
                        for dc2 in (-1, 0, 1):
                            if dr2 == 0 and dc2 == 0:
                                continue
                            ar, ac = nr + dr2, nc + dc2
                            if 0 <= ar < i_h and 0 <= ac < i_w:
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
            interior[br, bc] = int(ic)
            blob_occupied[br, bc] = True

    fill_ratio = float(np.mean(interior != int(rect_col))) if interior.size else 0.0
    if fill_ratio < 0.30 or fill_ratio > 0.90:
        return None
    rect[1:-1, 1:-1] = interior
    return rect


def _partition_rect(rect: np.ndarray, rh: int, rw: int, rng: random.Random) -> list[tuple[int, int, int, int]] | None:
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


def _pick_marker_position(corner: str, gh: int, gw: int, rh: int, rw: int, rng: random.Random) -> tuple[int, int] | None:
    top_max = gh // 2 - 1
    bottom_min = gh // 2 + (gh % 2)
    left_max = gw // 2 - 1
    right_min = gw // 2 + (gw % 2)

    if corner == "TL":
        mr_min, mr_max = 0, min(gh - rh, top_max)
        mc_min, mc_max = 0, min(gw - rw, left_max)
    elif corner == "TR":
        mr_min, mr_max = 0, min(gh - rh, top_max)
        mc_min, mc_max = max(rw - 1, right_min), gw - 1
    elif corner == "BL":
        mr_min, mr_max = max(rh - 1, bottom_min), gh - 1
        mc_min, mc_max = 0, min(gw - rw, left_max)
    else:
        mr_min, mr_max = max(rh - 1, bottom_min), gh - 1
        mc_min, mc_max = max(rw - 1, right_min), gw - 1

    if mr_min > mr_max or mc_min > mc_max:
        return None
    return rng.randint(mr_min, mr_max), rng.randint(mc_min, mc_max)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    gh = rng.randint(10, 26)
    gw = rng.randint(10, 26)

    palette = list(range(10))
    rng.shuffle(palette)
    bg = int(palette[0])
    rect_col = int(palette[1])
    marker_col = int(palette[2])
    n_interior = rng.randint(1, 3)
    interior_cols = [int(x) for x in palette[3 : 3 + n_interior]]

    corner = rng.choice(["TL", "TR", "BL", "BR"])

    min_side = 5
    max_side = min(20, min(gh, gw) - 1)
    if max_side < min_side:
        return None
    rh = rng.randint(min_side, max_side)
    rw = rng.randint(min_side, max_side)
    if rng.random() < 0.5 and max_side >= 8:
        rh = rng.randint(max(6, min_side), max_side)
        rw = rng.randint(max(6, min_side), max_side)

    marker_pos = _pick_marker_position(corner, gh, gw, rh, rw, rng)
    if marker_pos is None:
        return None
    mr, mc = marker_pos

    rect = _build_rect(rh, rw, rect_col, interior_cols, rng)
    if rect is None:
        return None

    pieces = _partition_rect(rect, rh, rw, rng)
    if pieces is None or len(pieces) < 4 or len(pieces) > 10:
        return None

    piece_arrays = []
    transformable_idxs = []
    for i, (pr, pc, ph, pw) in enumerate(pieces):
        if ph < 2 or pw < 2 or ph > 7 or pw > 7:
            return None
        piece = rect[pr : pr + ph, pc : pc + pw].copy()
        piece_arrays.append(piece)
        fill = int(np.sum(piece != int(rect_col))) / (ph * pw)
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

    tr, tc = _top_left_for(corner, mr, mc, rh, rw)
    expected_out = np.full((gh, gw), int(bg), dtype=int)
    expected_out[tr : tr + rh, tc : tc + rw] = rect

    forced = rng.sample(transformable_idxs, rng.randint(2, min(4, len(transformable_idxs))))
    chosen_piece_arrays = []
    flipped_count = 0
    for i, piece in enumerate(piece_arrays):
        opts = _scatter_transform_options(piece)
        if i in forced:
            non_id = [(tag, arr) for tag, arr in opts if tag != "id"]
            if not non_id:
                return None
            tag, arr = rng.choice(non_id)
        else:
            if rng.random() < 0.35:
                tag, arr = rng.choice(opts)
            else:
                tag, arr = opts[0]
        chosen_piece_arrays.append(arr.copy())
        if tag != "id":
            flipped_count += 1
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
            paste(canvas, pa, r0, c0)
        except ValueError:
            return None

    try:
        actual_out = main(canvas)
    except Exception:
        return None
    if not bool(np.array_equal(actual_out, expected_out)):
        return None

    try:
        no_flip_out = _solve_without_flip(canvas)
        if bool(np.array_equal(no_flip_out, expected_out)):
            return None
    except Exception:
        pass

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
