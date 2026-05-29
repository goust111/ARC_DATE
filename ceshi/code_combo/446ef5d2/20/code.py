# Combo variant of 446ef5d2 (index 20)
# combo_spec: D14/1+D19/2
# Name: 背景色与 rect_col 相同配合碎片零间距

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, paste


# Design analysis for the fused combo
# -----------------------------------
# a) Concrete behavior and trace:
#    Original task:
#    - Each piece is a fully visible rectangle on a distinct background.
#    - Pieces are separated by at least one background cell, so connected-component
#      extraction already gives the piece list.
#    Fused task:
#    - rect_col is changed to be exactly the background color. Any rect_col cell on
#      a scattered piece therefore disappears into the canvas.
#    - The solver can only see each piece's colored "core": an irregular 4-connected
#      pattern inside a rectangular core bbox. For pieces on the assembled border,
#      the hidden outer strip is gone as well.
#    - On top of that, one pair of scattered pieces is placed with zero gap between
#      their hidden full-piece rectangles, and their visible colored cores are forced
#      to 4-connect across that contact seam. The input now contains one merged
#      colored component that must be split back into two rectangular core bboxes.
#    Small trace:
#    - Original: two 3x4 pieces show up as two explicit 3x4 rectangles.
#    - D19 alone: those pieces may only show 3x3 and 2x4 irregular colored cores.
#    - D14+D19: those two hidden full boxes may touch flush, so the two irregular
#      cores merge into one colored component with no visible rectangle border. The
#      solver must infer two rectangular core bboxes from internal colored support.
# b) Complexity calibration:
#    - Allowing arbitrary many touching groups would explode partition ambiguity.
#      This combo keeps exactly one touching cluster made of exactly two pieces.
#    - All other pieces stay isolated with padding=1 from every other object, so the
#      fusion is visible but still stable.
#    - Core rectangles are kept at side length 2-5, which keeps the split search
#      tractable and avoids giant ambiguous components.
# c) Hidden assumption audit against the original main():
#    - `occ = (g != bg) & (g != mark)` no longer equals "all piece cells", because
#      rect_col cells are hidden as background. It now only reveals colored cores.
#    - Greedy rectangle extraction from occupied pixels is invalid twice over:
#      visible cores are irregular with holes, and one pair of cores can touch.
#    - `rect_col = most common piece color` fails, because rect_col is not visible
#      inside extracted cores. In this combo `rect_col` is simply `bg`.
#    - Border checks on assembled pieces also fail: the visible cores tile the inner
#      rectangle, not the full rectangle with the hidden 1-cell rect_col frame.
#    New solver assumptions introduced here:
#    - Every visible component is either one valid rectangular core bbox, or one
#      touching cluster of exactly two such core bboxes.
#    - A valid core has a 4-connected colored mask, touches all four sides of its
#      bbox, and contains some background holes. The generator guarantees this.
#    - The two-piece touching cluster has non-rectangular union of core bboxes and
#      contains at least one cross-seam colored 4-neighbor contact. The generator
#      explicitly filters for that.
# d) Placement legality:
#    - Generator scatters rectangular full-piece bboxes, not just visible pixels, so
#      hidden border strips cannot overlap silently.
#    - The touching pair is placed as one reserved cluster bbox; all remaining
#      pieces are placed with padding=1 from the marker and from each other.
# e) Visual observability:
#    - Across examples, the background and rect_col are identical, so the assembled
#      rectangle outline is invisible in the output except for the colored interior.
#    - In the input, one merged colored component is visibly more complex than a
#      single fragment; humans can see that segmentation must come from internal
#      colored structure rather than explicit rectangular silhouettes.
# f) Visual cleanliness:
#    - Only one touching cluster is introduced.
#    - The marker remains isolated by placement padding, so it does not visually
#      fuse with scattered fragments.
# g) Color inferability:
#    - No hidden color ordering is used. The only color rule is structural:
#      background == rect_col, while the other 1-3 colors form the informative
#      interior/core patterns and drive boundary matching.


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


def _core_valid(arr: np.ndarray, bg: int) -> bool:
    h, w = map(int, arr.shape)
    if h < 2 or w < 2 or h > 5 or w > 5:
        return False
    mask = arr != int(bg)
    if not mask.any():
        return False
    if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
        return False
    fill = float(mask.mean())
    if fill < 0.28 or fill > 0.88:
        return False
    if np.all(mask):
        return False
    lab, nlab = label(mask.astype(int), structure=STRUCT4)
    return int(nlab) == 1


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return not (ar + ah <= br or br + bh <= ar or ac + aw <= bc or bc + bw <= ac)


def _has_cross_contact(
    arr: np.ndarray,
    bg: int,
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    owner = np.full(arr.shape, -1, dtype=int)
    owner[ar : ar + ah, ac : ac + aw] = 0
    owner[br : br + bh, bc : bc + bw] = np.where(
        owner[br : br + bh, bc : bc + bw] == -1,
        1,
        owner[br : br + bh, bc : bc + bw],
    )
    mask = arr != int(bg)
    H, W = arr.shape
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or owner[r, c] != 0:
                continue
            if r + 1 < H and mask[r + 1, c] and owner[r + 1, c] == 1:
                return True
            if c + 1 < W and mask[r, c + 1] and owner[r, c + 1] == 1:
                return True
            if r - 1 >= 0 and mask[r - 1, c] and owner[r - 1, c] == 1:
                return True
            if c - 1 >= 0 and mask[r, c - 1] and owner[r, c - 1] == 1:
                return True
    return False


def _component_partition_options(
    g: np.ndarray, comp: np.ndarray, bg: int
) -> list[list[tuple[int, int, int, int, np.ndarray]]]:
    pts = np.argwhere(comp)
    r0, c0 = map(int, pts.min(axis=0))
    r1, c1 = map(int, pts.max(axis=0))
    sub_mask = comp[r0 : r1 + 1, c0 : c1 + 1]
    sub_grid = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    Hc, Wc = sub_grid.shape

    options: list[list[tuple[int, int, int, int, np.ndarray]]] = []
    if _core_valid(sub_grid, bg):
        options.append([(r0, c0, Hc, Wc, sub_grid.copy())])

    seen = set()
    for rr in range(Hc):
        for cc in range(Wc):
            for h in range(2, min(5, Hc - rr) + 1):
                for w in range(2, min(5, Wc - cc) + 1):
                    rect1 = (rr, cc, h, w)
                    arr1 = sub_grid[rr : rr + h, cc : cc + w]
                    if not _core_valid(arr1, bg):
                        continue

                    rem_mask = sub_mask.copy()
                    rem_mask[rr : rr + h, cc : cc + w] = False
                    if not rem_mask.any():
                        continue

                    pts2 = np.argwhere(rem_mask)
                    rr0, cc0 = map(int, pts2.min(axis=0))
                    rr1, cc1 = map(int, pts2.max(axis=0))
                    hh = rr1 - rr0 + 1
                    ww = cc1 - cc0 + 1
                    rect2 = (rr0, cc0, hh, ww)
                    if hh < 2 or ww < 2 or hh > 5 or ww > 5:
                        continue
                    if _rectangles_overlap(rect1, rect2):
                        continue
                    if h * w + hh * ww >= (max(rr + h, rr0 + hh) - min(rr, rr0)) * (
                        max(cc + w, cc0 + ww) - min(cc, cc0)
                    ):
                        continue

                    arr2 = sub_grid[rr0 : rr0 + hh, cc0 : cc0 + ww]
                    if not _core_valid(arr2, bg):
                        continue

                    cover = np.zeros_like(sub_mask, dtype=bool)
                    cover[rr : rr + h, cc : cc + w] = True
                    cover[rr0 : rr0 + hh, cc0 : cc0 + ww] = True
                    if np.any(sub_mask & (~cover)):
                        continue
                    if not _has_cross_contact(sub_grid, bg, rect1, rect2):
                        continue

                    rects = [
                        (r0 + rr, c0 + cc, h, w, arr1.copy()),
                        (r0 + rr0, c0 + cc0, hh, ww, arr2.copy()),
                    ]
                    rects.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
                    key = tuple((r, c, h0, w0) for r, c, h0, w0, _ in rects)
                    if key in seen:
                        continue
                    seen.add(key)
                    options.append(rects)

    return options


def _extract_piece_options(
    g: np.ndarray, bg: int, mark: int
) -> list[list[tuple[int, int, int, int, np.ndarray]]]:
    vis = (g != int(bg)) & (g != int(mark))
    lab, nlab = label(vis.astype(int), structure=STRUCT4)
    if int(nlab) <= 0:
        return []

    per_component: list[list[list[tuple[int, int, int, int, np.ndarray]]]] = []
    for k in range(1, int(nlab) + 1):
        comp = lab == k
        opts = _component_partition_options(g, comp, bg)
        if not opts:
            return []
        per_component.append(opts)

    return [list(itertools.chain.from_iterable(choice)) for choice in itertools.product(*per_component)]


def _assemble_inner(
    pieces: list[np.ndarray], bg: int, inner_h: int, inner_w: int
) -> np.ndarray | None:
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]
    occ = np.full((inner_h, inner_w), -1, dtype=int)
    used = [False] * len(pieces)
    order = sorted(range(len(pieces)), key=lambda i: (-ph[i] * pw[i], -ph[i], -pw[i], i))
    solved = False

    def compat(v1: np.ndarray, v2: np.ndarray) -> bool:
        mask = (v1 != int(bg)) & (v2 != int(bg))
        return bool((v1[mask] == v2[mask]).all())

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


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    GH, GW = g.shape

    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    rect_col = bg
    mark, missing, mb = _find_marker(g, bg)
    corner = _corner_from_missing(missing, mb)
    mr, mc = map(int, missing)

    piece_options = _extract_piece_options(g, bg, mark)
    assert piece_options

    dims_cache: dict[int, list[tuple[int, int]]] = {}
    for rects in piece_options:
        pieces = [a for _r, _c, _h, _w, a in rects]
        total_area = int(sum(int(a.shape[0]) * int(a.shape[1]) for a in pieces))
        if total_area not in dims_cache:
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

            dims_cache[total_area] = sorted(
                [(int(h), int(w)) for h, w in dims if fits(int(h), int(w))],
                key=lambda x: (-x[0] * x[1], -x[0], -x[1]),
            )

        for inner_h, inner_w in dims_cache[total_area]:
            ass = _assemble_inner(pieces, rect_col, inner_h, inner_w)
            if ass is None:
                continue
            tr, tc = _full_top_left(corner, mr, mc, inner_h + 2, inner_w + 2)
            out = np.full((GH, GW), int(bg), dtype=int)
            out[tr + 1 : tr + 1 + inner_h, tc + 1 : tc + 1 + inner_w] = ass
            return out

    raise AssertionError("assembly failed")


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


def _partition_inner(
    ih: int, iw: int, rng: random.Random
) -> tuple[list[tuple[int, int, int, int]], np.ndarray] | None:
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

    target_pieces = rng.randint(4, min(8, n_cells))
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

    if len(pieces) < 4 or len(pieces) > 8:
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
        accent_choices = [c for c in interior_cols if int(c) != main_col]
        accent = int(rng.choice(accent_choices))
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


def _scatter_bbox_patch(
    canvas: np.ndarray,
    occ: np.ndarray,
    patch: np.ndarray,
    bg: int,
    rng: random.Random,
    padding: int,
) -> bool:
    gh, gw = canvas.shape
    ph, pw = patch.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(0, gh - ph + 1):
        for c0 in range(0, gw - pw + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(gh, r0 + ph + padding)
            cc1 = min(gw, c0 + pw + padding)
            if occ[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((r0, c0))
    if not candidates:
        return False
    r0, c0 = candidates[rng.randrange(len(candidates))]
    occ[r0 : r0 + ph, c0 : c0 + pw] = True
    paste(canvas, patch, r0, c0, transparent=bg)
    return True


def _cluster_variants(
    a: np.ndarray, b: np.ndarray, bg: int
) -> list[np.ndarray]:
    h1, w1 = map(int, a.shape)
    h2, w2 = map(int, b.shape)
    variants: list[np.ndarray] = []
    seen = set()

    def add_variant(r2: int, c2: int) -> None:
        min_r = min(0, r2)
        min_c = min(0, c2)
        r1n, c1n = -min_r, -min_c
        r2n, c2n = r2 - min_r, c2 - min_c
        H = max(r1n + h1, r2n + h2)
        W = max(c1n + w1, c2n + w2)
        if h1 * w1 + h2 * w2 >= H * W:
            return

        canvas = np.full((H, W), int(bg), dtype=int)
        paste(canvas, a, r1n, c1n, transparent=bg)
        paste(canvas, b, r2n, c2n, transparent=bg)

        vis = canvas != int(bg)
        lab, nlab = label(vis.astype(int), structure=STRUCT4)
        if int(nlab) != 1:
            return
        if not _has_cross_contact(canvas, bg, (r1n, c1n, h1, w1), (r2n, c2n, h2, w2)):
            return

        key = (canvas.shape, tuple(map(int, canvas.flatten())))
        if key in seen:
            return
        seen.add(key)
        variants.append(canvas)

    for c2 in range(-(w2 - 1), w1):
        left = max(0, c2)
        right = min(w1, c2 + w2)
        if left < right:
            add_variant(h1, c2)
            add_variant(-h2, c2)

    for r2 in range(-(h2 - 1), h1):
        top = max(0, r2)
        bottom = min(h1, r2 + h2)
        if top < bottom:
            add_variant(r2, w1)
            add_variant(r2, -w2)

    return variants


def _gen_attempt_with_meta() -> tuple[np.ndarray, dict[str, int]] | None:
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
        if RH * RW <= int(0.42 * GH * GW):
            break
    else:
        return None

    IH = RH - 2
    IW = RW - 2

    part = _partition_inner(IH, IW, rng)
    if part is None:
        return None
    pieces, piece_map = part

    full_specs: list[tuple[int, int, int, int]] = []
    for pr, pc, ph, pw in pieces:
        top_strip = 1 if pr == 0 else 0
        bottom_strip = 1 if pr + ph == IH else 0
        left_strip = 1 if pc == 0 else 0
        right_strip = 1 if pc + pw == IW else 0
        full_h = ph + top_strip + bottom_strip
        full_w = pw + left_strip + right_strip
        if full_h > 7 or full_w > 7:
            return None
        full_specs.append((full_h, full_w, top_strip, left_strip))

    inner = np.full((IH, IW), int(bg), dtype=int)
    for pr, pc, ph, pw in pieces:
        core = _make_piece_core(ph, pw, bg, interior_cols, rng)
        inner[pr : pr + ph, pc : pc + pw] = core

    _harmonize_boundaries(inner, piece_map, bg, rng)

    full_patches: list[np.ndarray] = []
    for idx, (pr, pc, ph, pw) in enumerate(pieces):
        arr = inner[pr : pr + ph, pc : pc + pw].copy()
        if not _core_valid(arr, bg):
            return None
        full_h, full_w, top_strip, left_strip = full_specs[idx]
        patch = np.full((full_h, full_w), int(bg), dtype=int)
        paste(patch, arr, top_strip, left_strip, transparent=bg)
        full_patches.append(patch)

    pair = None
    cluster_patch = None
    pair_indices = list(range(len(full_patches)))
    rng.shuffle(pair_indices)
    for i in pair_indices:
        for j in pair_indices:
            if i == j:
                continue
            variants = _cluster_variants(full_patches[i], full_patches[j], bg)
            if variants:
                pair = (i, j)
                cluster_patch = rng.choice(variants)
                break
        if pair is not None:
            break
    if pair is None or cluster_patch is None:
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

    if not _scatter_bbox_patch(canvas, occ, cluster_patch, bg, rng, padding=1):
        return None

    rest = [i for i in range(len(full_patches)) if i not in set(pair)]
    rng.shuffle(rest)
    for idx in rest:
        if not _scatter_bbox_patch(canvas, occ, full_patches[idx], bg, rng, padding=1):
            return None

    expected = np.full((GH, GW), int(bg), dtype=int)
    tr, tc = _full_top_left(corner, mr, mc, RH, RW)
    expected[tr + 1 : tr + 1 + IH, tc + 1 : tc + 1 + IW] = inner

    vis = (canvas != int(bg)) & (canvas != int(marker_col))
    lab, nlab = label(vis.astype(int), structure=STRUCT4)
    if int(nlab) >= len(pieces):
        return None
    comp_sizes = sorted(int((lab == k).sum()) for k in range(1, int(nlab) + 1))
    if not comp_sizes:
        return None

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    meta = {
        "piece_count": len(pieces),
        "visible_component_count": int(nlab),
        "touch_cluster_count": 1,
    }
    return canvas, meta


def _gen_attempt() -> np.ndarray | None:
    result = _gen_attempt_with_meta()
    if result is None:
        return None
    return result[0]


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
