# Variant of 446ef5d2
# Dimension: D19 — 背景与前景的关系 (meta)
# Change: 背景色是网格中最常见的颜色，碎片散布在背景上，输出中背景保持不变... → 背景色与 rect_col 相同
#
# Design analysis for the D19 variant
# -----------------------------------
# spec:
#   variant_core:
#     - bg == rect_col. The assembled rectangle's outer border and every "wildcard"
#       cell use the same color as the ambient background.
#     - In the input, the scattered fragments therefore lose the rect_col parts of
#       their silhouette. Only non-bg interior pixels remain visible.
#   concrete_behavior:
#     - Original task: each fragment is a visible solid rectangle, because every
#       rect_col cell is still distinguishable from the background.
#     - Variant: each fragment is still conceptually rectangular, but the parts that
#       lie on the rectangle's outer border disappear into the background. The solver
#       must recover fragments from colored interior evidence instead of from explicit
#       non-bg rectangles.
#   small_trace:
#     - Suppose a top-edge piece in the assembled rectangle is 4x5. In the original
#       task, all 20 cells are visible in the input.
#     - Here its top row is rect_col==bg, so that row disappears when scattered.
#       The visible input only shows the 3x5 inner core. Solver behavior changes from
#       "take non-bg bbox as the piece" to "take the colored connected component,
#       recover its rectangular core bbox, assemble those cores, then add back the
#       global 1-cell rect_col border in the output".
#   complexity_calibration:
#     - To make the D19 change visible but still solvable, generator forces every
#       visible core to have:
#         1. a 4-connected non-bg mask,
#         2. non-bg pixels touching all four sides of the visible core bbox,
#         3. at least some bg holes, so the input does not degrade into obvious full
#            solid rectangles.
#     - This keeps core extraction unambiguous while still hiding the true outer
#       border of each full piece.
#   hidden_assumption_audit:
#     - Original `occ = (g != bg) & (g != mark)` assumed every piece cell is non-bg.
#       That is false now: many genuine piece cells equal bg.
#     - Original rectangle decomposition by scanning `occ` rows/cols assumed those
#       non-bg cells themselves form complete rectangles. False in the variant.
#     - Original `rect_col = most common color across piece pixels` no longer works,
#       because rect_col pixels are merged into the background and never appear as
#       recoverable piece pixels. In the variant, `rect_col` is simply `bg`.
#     - Original boundary checks required outer placed pieces to expose rect_col
#       edges. After stripping the 1-cell outer border, the assembled object is now
#       the inner rectangle, so those checks must be removed.
#   legality_and_generation:
#     - Generator builds the assembled rectangle as: 1-cell bg/rect_col frame +
#       interior tiled by rectangular visible cores.
#     - Input scattering reserves the full hidden piece bbox, then pastes only the
#       visible core at the proper offset, so conceptual full pieces neither overlap
#       nor touch.
#   visual_observability:
#     - Across samples, the human-visible difference is immediate: scattered pieces
#       are no longer clear rectangles. Their outlines are incomplete and must be
#       inferred from connected colored interiors and the marker corner.
#   solver_new_assumptions:
#     - Each visible fragment corresponds to exactly one 4-connected colored core.
#     - The colored core touches all four sides of its own bbox, so bbox recovery is
#       stable.
#     - Adding a 1-cell border around the assembled inner rectangle recreates the
#       hidden rect_col/bkg frame. Generator explicitly guarantees these assumptions.

from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, paste


def _find_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], tuple[int, int, int, int]]:
    cnt = Counter(map(int, g.flatten()))
    for col in sorted(c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3):
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(col))]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0) == 1 and (c1 - c0) == 1:
            all4 = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            miss = list(all4 - set(pts))
            if len(miss) == 1:
                return int(col), miss[0], (r0, c0, r1, c1)
    raise AssertionError("marker not found")


def _corner_from_missing(missing: tuple[int, int], mb: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = mb
    if missing == (r0, c0):
        return "BR"
    if missing == (r0, c1):
        return "BL"
    if missing == (r1, c0):
        return "TR"
    return "TL"


def _full_top_left(corner: str, mr: int, mc: int, full_h: int, full_w: int) -> tuple[int, int]:
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - full_w + 1
    if corner == "BL":
        return mr - full_h + 1, mc
    return mr - full_h + 1, mc - full_w + 1


def _marker_cells(corner: str, mr: int, mc: int) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _extract_visible_core_rects(g: np.ndarray, bg: int, mark: int) -> list[np.ndarray]:
    vis = (g != int(bg)) & (g != int(mark))
    lab, nlab = label(vis, structure=STRUCT4)
    assert nlab > 0
    pieces: list[np.ndarray] = []
    for k in range(1, int(nlab) + 1):
        pts = np.argwhere(lab == k)
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        mask = arr != int(bg)
        assert mask[0, :].any()
        assert mask[-1, :].any()
        assert mask[:, 0].any()
        assert mask[:, -1].any()
        pieces.append(arr)
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    D19 variant solver.

    The visible scattered objects are the inner cores of the original rectangular
    pieces. Those cores tile the assembled rectangle after removing its 1-cell
    rect_col border, and rect_col itself is equal to the background color.
    """
    g = np.array(grid, dtype=int)
    GH, GW = g.shape

    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    rect_col = bg

    mark, missing, mb = _find_marker(g, bg)
    corner = _corner_from_missing(missing, mb)
    mr, mc = map(int, missing)

    pieces = _extract_visible_core_rects(g, bg, mark)
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]
    total_area = int(sum(h * w for h, w in zip(ph, pw)))

    dims: list[tuple[int, int]] = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh == 0:
            dims.append((hh, total_area // hh))
            if hh * hh != total_area:
                dims.append((total_area // hh, hh))

    def fits(inner_h: int, inner_w: int) -> bool:
        full_h = inner_h + 2
        full_w = inner_w + 2
        tr, tc = _full_top_left(corner, mr, mc, full_h, full_w)
        return 0 <= tr and 0 <= tc and tr + full_h <= GH and tc + full_w <= GW

    dims = [(int(h), int(w)) for h, w in dims if fits(int(h), int(w))]
    dims.sort(key=lambda x: (-x[0] * x[1], -x[0], -x[1]))

    def compat(v1: np.ndarray, v2: np.ndarray) -> bool:
        v1 = np.array(v1, dtype=int)
        v2 = np.array(v2, dtype=int)
        mask = (v1 != int(rect_col)) & (v2 != int(rect_col))
        return bool((v1[mask] == v2[mask]).all())

    def assemble(inner_h: int, inner_w: int) -> np.ndarray | None:
        occ = np.full((inner_h, inner_w), -1, dtype=int)
        used = [False] * len(pieces)
        order = sorted(range(len(pieces)), key=lambda i: (-ph[i] * pw[i], -ph[i], -pw[i], i))
        solved = False

        def neighbor_ok(a: np.ndarray, r: int, c: int) -> bool:
            h, w = a.shape
            if c > 0:
                left = occ[r : r + h, c - 1]
                m = left != -1
                if m.any() and not compat(left[m], a[:, 0][m]):
                    return False
            if c + w < inner_w:
                right = occ[r : r + h, c + w]
                m = right != -1
                if m.any() and not compat(right[m], a[:, -1][m]):
                    return False
            if r > 0:
                up = occ[r - 1, c : c + w]
                m = up != -1
                if m.any() and not compat(up[m], a[0, :][m]):
                    return False
            if r + h < inner_h:
                down = occ[r + h, c : c + w]
                m = down != -1
                if m.any() and not compat(down[m], a[-1, :][m]):
                    return False
            return True

        def can_place(i: int, r: int, c: int) -> bool:
            a = pieces[i]
            h, w = a.shape
            if r + h > inner_h or c + w > inner_w:
                return False
            if (occ[r : r + h, c : c + w] != -1).any():
                return False
            return neighbor_ok(a, r, c)

        def place(i: int, r: int, c: int) -> None:
            a = pieces[i]
            h, w = a.shape
            occ[r : r + h, c : c + w] = a
            used[i] = True

        def unplace(i: int, r: int, c: int) -> None:
            a = pieces[i]
            h, w = a.shape
            occ[r : r + h, c : c + w] = -1
            used[i] = False

        def next_empty() -> tuple[int, int] | None:
            idx = np.argwhere(occ == -1)
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
                if used[i]:
                    continue
                if can_place(i, r, c):
                    place(i, r, c)
                    bt()
                    if solved:
                        return
                    unplace(i, r, c)

        bt()
        return occ if solved else None

    ass = None
    used_dim = None
    for inner_h, inner_w in dims:
        ass = assemble(inner_h, inner_w)
        if ass is not None:
            used_dim = (inner_h, inner_w)
            break

    assert ass is not None and used_dim is not None
    inner_h, inner_w = used_dim
    tr, tc = _full_top_left(corner, mr, mc, inner_h + 2, inner_w + 2)
    out = np.full((GH, GW), int(bg), dtype=int)
    out[tr + 1 : tr + 1 + inner_h, tc + 1 : tc + 1 + inner_w] = ass
    return out


def _rand_splits(total: int, rng: random.Random, min_s: int = 2, max_s: int = 5) -> list[int] | None:
    segs: list[int] = []
    rem = int(total)
    while rem > 0:
        if rem < min_s:
            return None
        if rem <= max_s:
            segs.append(rem)
            rem = 0
            continue
        hi = min(max_s, rem - min_s)
        if hi < min_s:
            return None
        segs.append(rng.randint(min_s, hi))
        rem -= segs[-1]
    return segs


def _partition_inner(ih: int, iw: int, rng: random.Random) -> tuple[list[tuple[int, int, int, int]], np.ndarray] | None:
    row_segs = _rand_splits(ih, rng)
    col_segs = _rand_splits(iw, rng)
    if row_segs is None or col_segs is None:
        return None

    row_starts: list[int] = []
    col_starts: list[int] = []
    s = 0
    for x in row_segs:
        row_starts.append(s)
        s += x
    s = 0
    for x in col_segs:
        col_starts.append(s)
        s += x

    nr = len(row_segs)
    nc = len(col_segs)
    n_cells = nr * nc
    if n_cells < 4:
        return None

    cell_id = [[i * nc + j for j in range(nc)] for i in range(nr)]

    def get_bbox(pid: int) -> tuple[int, int, int, int] | None:
        cells = [(i, j) for i in range(nr) for j in range(nc) if cell_id[i][j] == pid]
        if not cells:
            return None
        rs = [x for x, _ in cells]
        cs = [x for _, x in cells]
        return min(rs), min(cs), max(rs), max(cs)

    def pixel_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
        h = sum(row_segs[i] for i in range(r0i, r1i + 1))
        w = sum(col_segs[j] for j in range(c0i, c1i + 1))
        return h, w

    target_pieces = rng.randint(4, min(9, n_cells))
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
        bb1 = get_bbox(pid1)
        bb2 = get_bbox(pid2)
        if bb1 is None or bb2 is None:
            continue
        mr0 = min(bb1[0], bb2[0])
        mc0 = min(bb1[1], bb2[1])
        mr1 = max(bb1[2], bb2[2])
        mc1 = max(bb1[3], bb2[3])
        ph, pw = pixel_size(mr0, mc0, mr1, mc1)
        if ph > 5 or pw > 5:
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

    pieces: list[tuple[int, int, int, int]] = []
    piece_map = np.full((ih, iw), -1, dtype=int)
    seen: set[int] = set()
    for i in range(nr):
        for j in range(nc):
            pid = cell_id[i][j]
            if pid in seen:
                continue
            seen.add(pid)
            bb = get_bbox(pid)
            assert bb is not None
            r0i, c0i, r1i, c1i = bb
            pr = row_starts[r0i]
            pc = col_starts[c0i]
            ph = sum(row_segs[ii] for ii in range(r0i, r1i + 1))
            pw = sum(col_segs[jj] for jj in range(c0i, c1i + 1))
            if ph < 2 or pw < 2 or ph > 5 or pw > 5:
                return None
            idx = len(pieces)
            piece_map[pr : pr + ph, pc : pc + pw] = idx
            pieces.append((pr, pc, ph, pw))

    if len(pieces) < 4 or len(pieces) > 9:
        return None
    return pieces, piece_map


def _grow_connected_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    total = h * w
    mask = np.zeros((h, w), dtype=bool)
    rr = rng.randrange(h)
    cc = rng.randrange(w)
    mask[rr, :] = True
    mask[:, cc] = True
    frontier = [(r, c) for r, c in np.argwhere(mask)]
    base = int(mask.sum())
    upper = max(base, min(total - 1, int(round(0.78 * total))))
    target = rng.randint(base, upper) if upper >= base else base
    while int(mask.sum()) < target:
        r, c = frontier[rng.randrange(len(frontier))]
        opts: list[tuple[int, int]] = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                opts.append((nr, nc))
        if not opts:
            if int(mask.sum()) == total:
                break
            frontier = [(r0, c0) for r0, c0 in np.argwhere(mask)]
            continue
        nr, nc = opts[rng.randrange(len(opts))]
        mask[nr, nc] = True
        frontier.append((nr, nc))
    return mask


def _make_piece_core(h: int, w: int, bg: int, interior_cols: list[int], rng: random.Random) -> np.ndarray:
    arr = np.full((h, w), int(bg), dtype=int)
    mask = _grow_connected_mask(h, w, rng)
    main_col = int(rng.choice(interior_cols))
    arr[mask] = main_col

    if len(interior_cols) >= 2 and int(mask.sum()) >= 5 and rng.random() < 0.8:
        accent = int(rng.choice([c for c in interior_cols if int(c) != main_col]))
        pts = [tuple(map(int, x)) for x in np.argwhere(mask)]
        seed = pts[rng.randrange(len(pts))]
        chosen = {seed}
        frontier = [seed]
        budget = rng.randint(1, max(1, min(len(pts) // 3, 6)))
        while frontier and len(chosen) < budget:
            r, c = frontier.pop(rng.randrange(len(frontier)))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in chosen:
                    chosen.add((nr, nc))
                    frontier.append((nr, nc))
                    if len(chosen) >= budget:
                        break
        for r, c in chosen:
            arr[r, c] = accent
    return arr


def _harmonize_boundaries(inner: np.ndarray, piece_map: np.ndarray, bg: int, rng: random.Random) -> None:
    ih, iw = inner.shape
    for _ in range(8):
        changed = False
        for r in range(ih):
            for c in range(iw - 1):
                if piece_map[r, c] == piece_map[r, c + 1]:
                    continue
                a = int(inner[r, c])
                b = int(inner[r, c + 1])
                if a != bg and b != bg and a != b:
                    if rng.random() < 0.5:
                        inner[r, c + 1] = a
                    else:
                        inner[r, c] = b
                    changed = True
        for r in range(ih - 1):
            for c in range(iw):
                if piece_map[r, c] == piece_map[r + 1, c]:
                    continue
                a = int(inner[r, c])
                b = int(inner[r + 1, c])
                if a != bg and b != bg and a != b:
                    if rng.random() < 0.5:
                        inner[r + 1, c] = a
                    else:
                        inner[r, c] = b
                    changed = True
        if not changed:
            return


def _piece_valid(arr: np.ndarray, bg: int) -> bool:
    mask = arr != int(bg)
    if not mask.any():
        return False
    if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
        return False
    fill = float(mask.mean())
    if fill < 0.25 or fill > 0.85:
        return False
    if (~mask).sum() == 0:
        return False
    lab, nlab = label(mask, structure=STRUCT4)
    return int(nlab) == 1


def _scatter_piece(
    canvas: np.ndarray,
    occ: np.ndarray,
    core: np.ndarray,
    full_h: int,
    full_w: int,
    top_off: int,
    left_off: int,
    bg: int,
    rng: random.Random,
) -> bool:
    gh, gw = canvas.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(0, gh - full_h + 1):
        for c0 in range(0, gw - full_w + 1):
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(gh, r0 + full_h + 1)
            cc1 = min(gw, c0 + full_w + 1)
            if occ[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((r0, c0))
    if not candidates:
        return False
    r0, c0 = candidates[rng.randrange(len(candidates))]
    occ[r0 : r0 + full_h, c0 : c0 + full_w] = True
    paste(canvas, core, r0 + top_off, c0 + left_off, transparent=bg)
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    GH = rng.randint(10, 26)
    GW = rng.randint(10, 26)

    palette = list(range(10))
    rng.shuffle(palette)
    bg = palette[0]
    rect_col = bg
    marker_col = palette[1]
    n_interior = rng.randint(1, 3)
    interior_cols = palette[2 : 2 + n_interior]

    max_full_h = min(18, GH - 2)
    max_full_w = min(18, GW - 2)
    if max_full_h < 6 or max_full_w < 6:
        return None

    for _ in range(20):
        RH = rng.randint(6, max_full_h)
        RW = rng.randint(6, max_full_w)
        if RH * RW <= int(0.38 * GH * GW):
            break
    else:
        return None

    IH = RH - 2
    IW = RW - 2

    part = _partition_inner(IH, IW, rng)
    if part is None:
        return None
    pieces, piece_map = part

    full_specs: list[tuple[int, int, int, int, int, int]] = []
    for pr, pc, ph, pw in pieces:
        top_strip = 1 if pr == 0 else 0
        bottom_strip = 1 if pr + ph == IH else 0
        left_strip = 1 if pc == 0 else 0
        right_strip = 1 if pc + pw == IW else 0
        full_h = ph + top_strip + bottom_strip
        full_w = pw + left_strip + right_strip
        if full_h > 7 or full_w > 7:
            return None
        full_specs.append((full_h, full_w, top_strip, left_strip, bottom_strip, right_strip))

    inner = np.full((IH, IW), int(bg), dtype=int)
    for pr, pc, ph, pw in pieces:
        core = _make_piece_core(ph, pw, bg, interior_cols, rng)
        inner[pr : pr + ph, pc : pc + pw] = core

    _harmonize_boundaries(inner, piece_map, bg, rng)

    used_colors = {int(x) for x in np.unique(inner) if int(x) != int(bg)}
    if not used_colors:
        return None

    for idx, (pr, pc, ph, pw) in enumerate(pieces):
        arr = inner[pr : pr + ph, pc : pc + pw]
        if not _piece_valid(arr, bg):
            return None
        if not np.array_equal(np.argwhere(piece_map == idx).min(axis=0), np.array([pr, pc])):
            return None

    corner = rng.choice(["TL", "TR", "BL", "BR"])
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

    marker_cells = _marker_cells(corner, mr, mc)
    for r, c in marker_cells:
        if not (0 <= r < GH and 0 <= c < GW):
            return None

    canvas = np.full((GH, GW), int(bg), dtype=int)
    occ = np.zeros((GH, GW), dtype=bool)
    for r, c in marker_cells:
        canvas[r, c] = int(marker_col)
        occ[r, c] = True

    order = list(range(len(pieces)))
    rng.shuffle(order)
    for idx in order:
        pr, pc, ph, pw = pieces[idx]
        full_h, full_w, top_strip, left_strip, _bottom_strip, _right_strip = full_specs[idx]
        core = inner[pr : pr + ph, pc : pc + pw].copy()
        if not _scatter_piece(canvas, occ, core, full_h, full_w, top_strip, left_strip, bg, rng):
            return None

    expected = np.full((GH, GW), int(bg), dtype=int)
    tr, tc = _full_top_left(corner, mr, mc, RH, RW)
    expected[tr + 1 : tr + 1 + IH, tc + 1 : tc + 1 + IW] = inner

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
