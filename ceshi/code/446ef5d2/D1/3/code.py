from __future__ import annotations

# Variant of 446ef5d2
# Dimension: D1 — 边缘匹配条件 (match_cond)
# Change: 相邻碎片的接触边上，非 rect_col 像素位置和颜色必须一致；rect_col 充当通配符可匹配任何颜色... → 翻转后边缘匹配
#
# Design analysis
# - Concrete behavior:
#   Input still contains one 3-cell L marker plus several scattered rectangular pieces.
#   Output is still a single assembled rectangle whose corner is pinned by the marker gap.
#   The only rule change is that each piece may need a mirror before translation:
#   identity, horizontal flip, or vertical flip. Rotations are still forbidden.
#   Example trace:
#   a 3x4 piece in the assembled rectangle may have top edge [2,1,2,2] and bottom
#   edge [2,3,2,2]. After a vertical flip in the input, those two edge signatures swap.
#   Under the original solver, that scattered piece can only be used as-is, so the edge
#   would meet the wrong neighbor. Under this variant solver, the piece is flipped back
#   before placement, restoring the canonical rectangle.
# - Complexity calibration:
#   A pure flip variant is only visible if pieces are not mirror-symmetric. The generator
#   therefore rejects puzzles unless at least two scattered pieces are actually mirrored
#   in the input and the original no-flip baseline solver can no longer recover the
#   canonical output. This keeps the new degree of freedom active instead of decorative.
# - Implicit-assumption audit against the original main():
#   The old `pieces = [a ...]` plus `can_place(i,r,c)` assumed one fixed raster per piece.
#   That is invalid now; we must enumerate per-piece transform candidates.
#   The old anchor-piece pre-placement shortcut also depended on a fixed relation between
#   the observed piece corner and the final assembled corner. With flips, that shortcut
#   becomes much less reliable and is removed; correctness comes from the backtracking
#   search itself.
#   `boundary_ok` and `neighbor_ok` remain valid, but they must inspect the transformed
#   raster, not the original raster.
# - Placement legality:
#   The generator still builds the final rectangle first, partitions it into axis-aligned
#   rectangles, then scatters them with at least one background cell of separation.
#   Marker cells are placed first, and `random_free_location(..., padding=1)` prevents
#   overlaps or touching. All scattered pieces remain rectangular because flips preserve
#   the bbox.
# - Visual observability:
#   Human solvers can see that some fragments have asymmetric colored edge signatures
#   whose handedness differs from their location in the final rectangle. Because the
#   output rectangle is not mirrored as a whole and rotations never occur, the repeated
#   explanatory pattern across samples is "mirror certain fragments, then assemble".
# - Visual cleanliness:
#   No extra markers or overlays are added. The scene still contains one marker and
#   separated rectangular pieces, so the new signal is localized to the fragment content
#   instead of adding clutter.
# - Color inference:
#   Color semantics are unchanged. `rect_col` is still the dominant border/wildcard color,
#   and all other colors are ordinary piece content. No ordering depends on numeric color
#   ids, only on visible structure.
# - New solver assumptions introduced here:
#   1. Every required mirrored piece is recoverable by one of {identity, fliplr, flipud}.
#      The generator guarantees this by only scattering with those transforms.
#   2. The canonical assembly is still an axis-aligned rectangle with unchanged dimensions.
#      The generator preserves the original output-first rectangle construction.
#   3. The flip degree of freedom is actually necessary. The generator verifies that the
#      no-flip baseline solver fails or returns a different output on every accepted sample.

from collections import Counter
import random
import numpy as np

from arc_utils import paste, random_free_location


def _find_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], tuple[int, int, int, int]]:
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
        if len(miss) == 1:
            return int(col), miss[0], (r0, c0, r1, c1)
    raise AssertionError("marker not found")


def _missing_to_corner(missing: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    if missing == (r0, c0):
        return "BR"
    if missing == (r0, c1):
        return "BL"
    if missing == (r1, c0):
        return "TR"
    return "TL"


def _extract_rectangles(g: np.ndarray, bg: int, mark: int) -> list[tuple[int, int, int, int, np.ndarray]]:
    GH, GW = g.shape
    occ = (g != int(bg)) & (g != int(mark))
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


def _solve_core(grid: np.ndarray, allow_flip: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, missing, mb = _find_marker(g, bg)
    corner = _missing_to_corner(missing, mb)

    rects = _extract_rectangles(g, bg, mark)
    if not rects:
        raise AssertionError("no pieces")
    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))
    allpix: list[int] = []
    for _r, _c, _h, _w, a in rects:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    mr, mc = map(int, missing)
    pieces = [a for _r, _c, _h, _w, a in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]
    variants = [_unique_piece_variants(p, allow_flip=allow_flip) for p in pieces]

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
            h, w = map(int, a.shape)
            if r == 0 and not bool(np.all(a[0, :] == int(rect_col))):
                return False
            if (r + h) == H and not bool(np.all(a[-1, :] == int(rect_col))):
                return False
            if c == 0 and not bool(np.all(a[:, 0] == int(rect_col))):
                return False
            if (c + w) == W and not bool(np.all(a[:, -1] == int(rect_col))):
                return False
            return True

        def neighbor_ok(a: np.ndarray, r: int, c: int) -> bool:
            h, w = map(int, a.shape)
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

        def can_place(a: np.ndarray, r: int, c: int) -> bool:
            h, w = map(int, a.shape)
            if (r + h) > H or (c + w) > W:
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
    for H, W in dims:
        ass = assemble(H, W)
        if ass is not None:
            used_dim = (H, W)
            break

    if ass is None or used_dim is None:
        raise AssertionError("assembly failed")

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


def _build_rect(RH: int, RW: int, rect_col: int, interior_cols: list[int], rng: random.Random) -> np.ndarray | None:
    rect = np.full((RH, RW), int(rect_col), dtype=int)
    iH, iW = RH - 2, RW - 2
    if iH < 1 or iW < 1:
        return None
    interior = np.full((iH, iW), int(rect_col), dtype=int)
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
            interior[br, bc] = int(ic)
            blob_occupied[br, bc] = True

    non_rc_count = int(np.sum(interior != int(rect_col)))
    fill_ratio = non_rc_count / (iH * iW)
    if fill_ratio < 0.30 or fill_ratio > 0.90:
        return None
    rect[1:-1, 1:-1] = interior
    return rect


def _partition_rect(
    rect: np.ndarray, RH: int, RW: int, rng: random.Random
) -> list[tuple[int, int, int, int]] | None:
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


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    GH = rng.randint(10, 26)
    GW = rng.randint(10, 26)

    palette = list(range(10))
    rng.shuffle(palette)
    bg = int(palette[0])
    rect_col = int(palette[1])
    marker_col = int(palette[2])
    n_interior = rng.randint(1, 3)
    interior_cols = [int(x) for x in palette[3 : 3 + n_interior]]

    corner = rng.choice(["TL", "TR", "BL", "BR"])

    min_side, max_side = 5, min(20, min(GH, GW) - 3)
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

    rect = _build_rect(RH, RW, rect_col, interior_cols, rng)
    if rect is None:
        return None

    pieces = _partition_rect(rect, RH, RW, rng)
    if pieces is None or len(pieces) < 4 or len(pieces) > 10:
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
    piece_arrays = []
    transformable_idxs = []
    for i, (pr, pc, ph, pw) in enumerate(pieces):
        if ph < 2 or pw < 2 or ph > 7 or pw > 7:
            return None
        piece = rect[pr : pr + ph, pc : pc + pw].copy()
        piece_arrays.append(piece)
        nrc = int(np.sum(piece != int(rect_col)))
        fill = nrc / (ph * pw)
        if fill < 0.10 or fill > 0.60:
            return None
        if not _piece_has_non_rc_edge(piece, rect_col):
            return None
        if pr == 0 and not bool(np.all(piece[0, :] == int(rect_col))):
            return None
        if pr + ph == RH and not bool(np.all(piece[-1, :] == int(rect_col))):
            return None
        if pc == 0 and not bool(np.all(piece[:, 0] == int(rect_col))):
            return None
        if pc + pw == RW and not bool(np.all(piece[:, -1] == int(rect_col))):
            return None
        if pr <= anchor_r < pr + ph and pc <= anchor_c < pc + pw:
            anchor_idx = i
        if len(_scatter_transform_options(piece)) > 1:
            transformable_idxs.append(i)
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

    if len(transformable_idxs) < 2:
        return None

    if corner == "TL":
        marker_cells = [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
        tr, tc = mr, mc
    elif corner == "TR":
        marker_cells = [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
        tr, tc = mr, mc - RW + 1
    elif corner == "BL":
        marker_cells = [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
        tr, tc = mr - RH + 1, mc
    else:
        marker_cells = [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]
        tr, tc = mr - RH + 1, mc - RW + 1
    for cr, cc in marker_cells:
        if cr < 0 or cr >= GH or cc < 0 or cc >= GW:
            return None

    expected_out = np.full((GH, GW), int(bg), dtype=int)
    expected_out[tr : tr + RH, tc : tc + RW] = rect

    forced = rng.sample(transformable_idxs, rng.randint(2, min(4, len(transformable_idxs))))
    chosen_piece_arrays = []
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
            tag, arr = rng.choice(opts if rng.random() < 0.35 else [opts[0]])
        chosen_piece_arrays.append(arr.copy())
        if tag != "id":
            flipped_count += 1
            if tag == "h":
                h_count += 1
            elif tag == "v":
                v_count += 1
    if flipped_count < 2 or (h_count == 0 and v_count == 0):
        return None

    canvas = np.full((GH, GW), int(bg), dtype=int)
    for cr, cc in marker_cells:
        canvas[cr, cc] = int(marker_col)

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
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
