# Combo variant of 446ef5d2 (index 16)
# combo_spec: D9/1+D17/0
# Name: 多种方向通配色配合无分离约束

from __future__ import annotations

# Design analysis for the fused variant
# ====================================
# a) Concrete behavior and trace
# - Original task:
#   * one global `rect_col` is both the outer-border color and the wildcard on every seam.
#   * interior accent colors are 8-separated, so dense color tangles almost never appear.
#   * Example: a vertical seam compares [rect, 4, rect] vs [7, 4, 7]; it is valid because
#     only the middle 4 must match when `rect_col` is treated as the universal wildcard.
# - Fused variant:
#   * there are 2-3 wildcard colors, but each is directional:
#       - H-role colors are wild only on left/right adjacency (vertical seams).
#       - V-role colors are wild only on top/bottom adjacency (horizontal seams).
#   * different interior colors may touch freely; generation intentionally creates many
#     4-neighbor contacts between distinct interior colors.
#   * Example:
#       vertical seam: [H, 4, H] vs [2, 5, 2] is valid at the H positions, but 4 vs 5
#       is still a real mismatch unless one side is H there.
#       horizontal seam: [V, 3, V] over [1, 6, 1] is valid at the V positions, but the
#       same V color does not help on a vertical seam.
# - To avoid seam-corner conflicts, only non-corner edge cells carry directional wildcards.
#   Every grid-vertex corner is forced to be an exact shared color across touching pieces.
#
# b) Complexity calibration
# - The task keeps the original scatter -> reconstruct anchored rectangle structure.
# - To make the fused rule visible, the assembled rectangle is partitioned into a 2-4 by
#   2-4 grid of rectangular pieces. This produces repeated vertical and horizontal seams,
#   so each directional wildcard color gets multiple visible uses in one sample.
# - I restrict wildcard colors to 2 or 3 total and interior colors to 2 or 3 total. That
#   keeps the rule inferable from 2-4 examples instead of becoming a color soup.
#
# c) Hidden-assumption audit against the original `main()`
# - `rect_col = most_common(piece pixels)` is invalid: there is no single wildcard now.
# - `compat(v1, v2)` using one color for every seam is invalid. Vertical seams and
#   horizontal seams need different wildcard sets.
# - `boundary_ok()` requiring one border color is invalid. Top/bottom borders must use
#   V-role colors, left/right borders must use H-role colors, and the four rectangle
#   corners may use either because they lie at an orientation junction.
# - Storing only one compatibility rule for all adjacencies is invalid. The solver must
#   distinguish left/right vs top/bottom comparisons.
# - The old generator's 8-separation assumption is deliberately removed: dense mixed-color
#   interiors are not noise here, they are evidence that wildcard directionality, not color
#   isolation, is what explains the seams.
#
# d) Placement legality
# - Pieces remain axis-aligned rectangles, scattered without overlap and with >=1 background
#   cell of padding.
# - The L-marker logic is unchanged: its missing cell still pins one corner of the final
#   rectangle.
# - The generator validates each sample by running the new solver and comparing against the
#   constructed output rectangle.
#
# e) Visual observability
# - In training pairs, a human can see that left/right rectangle borders use one set of
#   colors while top/bottom borders use another set.
# - Dense interior mosaics include many side-adjacent color changes, so the old
#   "separated blobs" explanation is visibly wrong.
# - Repeated seam segments show the same orientation-specific permissiveness several times
#   in one puzzle, which is crucial for inferring the direction rule from few examples.
#
# f) Visual cleanliness
# - There are still only three object layers: background, marker, scattered rectangular
#   pieces. No overlaps or auxiliary annotations are introduced.
# - Exact-match corner pixels at grid vertices keep the assembled rectangle legible even
#   though seam interiors use many directional wildcard cells.
#
# g) Color inferability
# - No rule depends on numeric color ids. Roles are inferred from visible behavior:
#   a color repeatedly appearing on left/right borders and vertical seam masks is H-role;
#   a color repeatedly appearing on top/bottom borders and horizontal seam masks is V-role.
# - Interior colors are intentionally reused in dense local patterns so they are not
#   mistaken for wildcard colors.

from collections import Counter
from itertools import combinations
import numpy as np

from arc_utils import paste, random_free_location


def _find_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], str]:
    cnt = Counter(map(int, g.flatten()))
    for col, n in sorted(cnt.items()):
        if int(col) == int(bg) or int(n) != 3:
            continue
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(col))]
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
        return int(col), missing, corner
    raise AssertionError("marker not found")


def _extract_rectangles(g: np.ndarray, occ: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
    GH, GW = g.shape
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


def _top_left_from_corner(mr: int, mc: int, H: int, W: int, corner: str) -> tuple[int, int]:
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - W + 1
    if corner == "BL":
        return mr - H + 1, mc
    return mr - H + 1, mc - W + 1


def _edge_orientation_counts(pieces: list[np.ndarray]) -> tuple[Counter, Counter]:
    lr = Counter()
    tb = Counter()
    for a in pieces:
        h, w = a.shape
        if h > 2:
            for v in map(int, a[1:-1, 0]):
                lr[v] += 1
            for v in map(int, a[1:-1, -1]):
                lr[v] += 1
        if w > 2:
            for v in map(int, a[0, 1:-1]):
                tb[v] += 1
            for v in map(int, a[-1, 1:-1]):
                tb[v] += 1
    return lr, tb


def _role_maps(pieces: list[np.ndarray]) -> list[tuple[set[int], set[int]]]:
    colors = sorted({int(v) for p in pieces for v in p.flatten()})
    lr, tb = _edge_orientation_counts(pieces)
    h_cands = [c for c in colors if lr[c] > 0]
    v_cands = [c for c in colors if tb[c] > 0]
    scored: list[tuple[int, tuple[set[int], set[int]]]] = []
    seen = set()
    for total in (2, 3):
        for hsz in range(1, total):
            vsz = total - hsz
            for hs in combinations(h_cands, hsz):
                hset = set(map(int, hs))
                rem_v = [c for c in v_cands if c not in hset]
                for vs in combinations(rem_v, vsz):
                    vset = set(map(int, vs))
                    key = (tuple(sorted(hset)), tuple(sorted(vset)))
                    if key in seen:
                        continue
                    seen.add(key)
                    score = sum(lr[c] - tb[c] for c in hset) + sum(tb[c] - lr[c] for c in vset)
                    scored.append((int(score), (hset, vset)))
    scored.sort(key=lambda x: (-x[0], len(x[1][0]) + len(x[1][1]), sorted(x[1][0]), sorted(x[1][1])))
    return [rv for _score, rv in scored]


def _solve_fixed_roles(
    g: np.ndarray,
    hset: set[int],
    vset: set[int],
    *,
    return_meta: bool = False,
) -> tuple[np.ndarray, dict] | np.ndarray | None:
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, missing, corner = _find_marker(g, bg)
    occ = (g != int(bg)) & (g != int(mark))
    rects = _extract_rectangles(g, occ)
    if not rects:
        return None

    pieces = [a for _r, _c, _h, _w, a in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]
    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))
    mr, mc = map(int, missing)

    anchor_idx = None
    for i, (rr, cc, h, w, _a) in enumerate(rects):
        if rr <= mr < rr + h and cc <= mc < cc + w:
            anchor_idx = i
            break

    dims = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh != 0:
            continue
        ww = total_area // hh
        dims.append((int(hh), int(ww)))
        if hh != ww:
            dims.append((int(ww), int(hh)))

    def fits(H: int, W: int) -> bool:
        tr, tc = _top_left_from_corner(mr, mc, H, W, corner)
        return 0 <= tr and 0 <= tc and (tr + H) <= GH and (tc + W) <= GW

    dims = [(H, W) for H, W in dims if fits(H, W)]
    dims.sort(key=lambda t: (-min(t), -max(t)))

    def is_h(col: int) -> bool:
        return int(col) in hset

    def is_v(col: int) -> bool:
        return int(col) in vset

    def vertical_compat(a: np.ndarray, b: np.ndarray) -> bool:
        for x, y in zip(map(int, a), map(int, b)):
            if x == y or is_h(x) or is_h(y):
                continue
            return False
        return True

    def horizontal_compat(a: np.ndarray, b: np.ndarray) -> bool:
        for x, y in zip(map(int, a), map(int, b)):
            if x == y or is_v(x) or is_v(y):
                continue
            return False
        return True

    def boundary_ok(a: np.ndarray, r0: int, c0: int, H: int, W: int) -> bool:
        h, w = a.shape
        for rr in range(h):
            for cc in range(w):
                gr = r0 + rr
                gc = c0 + cc
                col = int(a[rr, cc])
                need_h = gc == 0 or gc == W - 1
                need_v = gr == 0 or gr == H - 1
                if not need_h and not need_v:
                    continue
                if need_h and need_v:
                    if not (is_h(col) or is_v(col)):
                        return False
                elif need_h:
                    if not is_h(col):
                        return False
                else:
                    if not is_v(col):
                        return False
        return True

    def assemble(H: int, W: int) -> tuple[np.ndarray, np.ndarray] | None:
        occ2 = np.full((H, W), -1, dtype=int)
        owner = np.full((H, W), -1, dtype=int)
        usedp = [False] * len(pieces)

        def neighbor_ok(a: np.ndarray, r: int, c: int) -> bool:
            h, w = a.shape
            if c > 0:
                left = occ2[r : r + h, c - 1]
                m = left != -1
                if m.any() and (not vertical_compat(left[m], a[:, 0][m])):
                    return False
            if (c + w) < W:
                right = occ2[r : r + h, c + w]
                m = right != -1
                if m.any() and (not vertical_compat(right[m], a[:, -1][m])):
                    return False
            if r > 0:
                up = occ2[r - 1, c : c + w]
                m = up != -1
                if m.any() and (not horizontal_compat(up[m], a[0, :][m])):
                    return False
            if (r + h) < H:
                down = occ2[r + h, c : c + w]
                m = down != -1
                if m.any() and (not horizontal_compat(down[m], a[-1, :][m])):
                    return False
            return True

        def can_place(i: int, r: int, c: int) -> bool:
            a = pieces[i]
            h, w = a.shape
            if (r + h) > H or (c + w) > W:
                return False
            if (owner[r : r + h, c : c + w] != -1).any():
                return False
            if not boundary_ok(a, r, c, H, W):
                return False
            if not neighbor_ok(a, r, c):
                return False
            return True

        def place(i: int, r: int, c: int) -> None:
            a = pieces[i]
            h, w = a.shape
            occ2[r : r + h, c : c + w] = a
            owner[r : r + h, c : c + w] = int(i)
            usedp[i] = True

        def unplace(i: int, r: int, c: int) -> None:
            a = pieces[i]
            h, w = a.shape
            occ2[r : r + h, c : c + w] = -1
            owner[r : r + h, c : c + w] = -1
            usedp[i] = False

        if anchor_idx is not None:
            rr, cc, h0, w0, _ = rects[anchor_idx]
            piece_corner = {
                (rr, cc): "TL",
                (rr, cc + w0 - 1): "TR",
                (rr + h0 - 1, cc): "BL",
                (rr + h0 - 1, cc + w0 - 1): "BR",
            }
            if (mr, mc) in piece_corner and piece_corner[(mr, mc)] == corner:
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

        order = sorted(
            range(len(pieces)),
            key=lambda i: (-(ph[i] * pw[i]), -abs(ph[i] - pw[i]), -ph[i], -pw[i], i),
        )

        def next_empty() -> tuple[int, int] | None:
            idx = np.argwhere(owner == -1)
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
        if not solved:
            return None
        return occ2.copy(), owner.copy()

    for H, W in dims:
        assembled = assemble(H, W)
        if assembled is None:
            continue
        ass, owner = assembled
        tr, tc = _top_left_from_corner(mr, mc, H, W, corner)
        out = np.full((GH, GW), int(bg), dtype=int)
        out[tr : tr + H, tc : tc + W] = ass
        if return_meta:
            meta = {
                "hset": set(map(int, hset)),
                "vset": set(map(int, vset)),
                "assembled": ass,
                "owner": owner,
                "top_left": (int(tr), int(tc)),
                "dims": (int(H), int(W)),
            }
            return out, meta
        return out
    return None


def _solve_core(g: np.ndarray, *, return_meta: bool = False) -> tuple[np.ndarray, dict] | np.ndarray:
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, _missing, _corner = _find_marker(g, bg)
    occ = (g != int(bg)) & (g != int(mark))
    rects = _extract_rectangles(g, occ)
    pieces = [a for _r, _c, _h, _w, a in rects]
    for hset, vset in _role_maps(pieces):
        res = _solve_fixed_roles(g, hset, vset, return_meta=return_meta)
        if res is not None:
            return res
    raise AssertionError("no directional-role solution found")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into a rectangle at an L-marker corner.

    Fused rule:
    - 2-3 colors are directional wildcards.
    - H-role colors act as wildcard only on left/right adjacency and the left/right outer border.
    - V-role colors act as wildcard only on top/bottom adjacency and the top/bottom outer border.
    - Interior colors can touch arbitrarily; no blob-separation heuristic is available.
    """
    g = np.array(grid, dtype=int)
    return _solve_core(g, return_meta=False)  # type: ignore[return-value]


def _rand_segments(total_parts: int, rng, lo: int = 3, hi: int = 5) -> list[int]:
    return [int(rng.randint(lo, hi)) for _ in range(total_parts)]


def _dense_piece_pattern(h: int, w: int, interior_cols: list[int], rng) -> np.ndarray:
    n = len(interior_cols)
    a = np.zeros((h, w), dtype=int)
    mode = int(rng.randint(0, 3))
    off = int(rng.randint(0, n))
    ax = int(rng.randint(1, 3))
    ay = int(rng.randint(1, 3))
    for r in range(h):
        for c in range(w):
            if mode == 0:
                idx = (off + ax * r + ay * c + ((r + c) % 2)) % n
            elif mode == 1:
                idx = (off + (r // 2) + ay * c + (r % 2)) % n
            else:
                idx = (off + ax * r + (c // 2) + (c % 2)) % n
            a[r, c] = int(interior_cols[int(idx)])
    return a


def _assemble_rect(piece_grid: list[list[np.ndarray]]) -> np.ndarray:
    return np.vstack([np.hstack(row) for row in piece_grid])


def _apply_vertex_constraints(
    piece_grid: list[list[np.ndarray]],
    row_segs: list[int],
    col_segs: list[int],
    h_cols: list[int],
    v_cols: list[int],
    interior_cols: list[int],
    rng,
) -> None:
    nr = len(row_segs)
    nc = len(col_segs)
    for vr in range(nr + 1):
        for vc in range(nc + 1):
            if (vr in (0, nr)) and (vc in (0, nc)):
                col = int(rng.choice(h_cols + v_cols))
            elif vr in (0, nr):
                col = int(rng.choice(v_cols))
            elif vc in (0, nc):
                col = int(rng.choice(h_cols))
            else:
                col = int(rng.choice(interior_cols))
            if vr > 0 and vc > 0:
                piece_grid[vr - 1][vc - 1][-1, -1] = col
            if vr > 0 and vc < nc:
                piece_grid[vr - 1][vc][-1, 0] = col
            if vr < nr and vc > 0:
                piece_grid[vr][vc - 1][0, -1] = col
            if vr < nr and vc < nc:
                piece_grid[vr][vc][0, 0] = col


def _decorate_boundaries(
    piece_grid: list[list[np.ndarray]],
    row_segs: list[int],
    col_segs: list[int],
    h_cols: list[int],
    v_cols: list[int],
    interior_cols: list[int],
    rng,
) -> None:
    nr = len(row_segs)
    nc = len(col_segs)

    for j in range(nc):
        top = piece_grid[0][j]
        if top.shape[1] > 2:
            for cc in range(1, top.shape[1] - 1):
                top[0, cc] = int(rng.choice(v_cols))
        bottom = piece_grid[nr - 1][j]
        if bottom.shape[1] > 2:
            for cc in range(1, bottom.shape[1] - 1):
                bottom[-1, cc] = int(rng.choice(v_cols))

    for i in range(nr):
        left = piece_grid[i][0]
        if left.shape[0] > 2:
            for rr in range(1, left.shape[0] - 1):
                left[rr, 0] = int(rng.choice(h_cols))
        right = piece_grid[i][nc - 1]
        if right.shape[0] > 2:
            for rr in range(1, right.shape[0] - 1):
                right[rr, -1] = int(rng.choice(h_cols))

    for i in range(nr):
        for j in range(nc - 1):
            left = piece_grid[i][j]
            right = piece_grid[i][j + 1]
            rows = list(range(1, left.shape[0] - 1))
            if not rows:
                continue
            forced_wild = int(rng.choice(rows))
            forced_exact = int(rng.choice(rows))
            for rr in rows:
                if rr == forced_wild:
                    mode = int(rng.randint(0, 2))
                elif rr == forced_exact:
                    mode = 2
                else:
                    mode = int(rng.choice([0, 1, 2]))
                if mode == 0:
                    left[rr, -1] = int(rng.choice(h_cols))
                    right[rr, 0] = int(rng.choice(interior_cols))
                elif mode == 1:
                    left[rr, -1] = int(rng.choice(interior_cols))
                    right[rr, 0] = int(rng.choice(h_cols))
                else:
                    col = int(rng.choice(interior_cols))
                    left[rr, -1] = col
                    right[rr, 0] = col

    for i in range(nr - 1):
        for j in range(nc):
            up = piece_grid[i][j]
            down = piece_grid[i + 1][j]
            cols = list(range(1, up.shape[1] - 1))
            if not cols:
                continue
            forced_wild = int(rng.choice(cols))
            forced_exact = int(rng.choice(cols))
            for cc in cols:
                if cc == forced_wild:
                    mode = int(rng.randint(0, 2))
                elif cc == forced_exact:
                    mode = 2
                else:
                    mode = int(rng.choice([0, 1, 2]))
                if mode == 0:
                    up[-1, cc] = int(rng.choice(v_cols))
                    down[0, cc] = int(rng.choice(interior_cols))
                elif mode == 1:
                    up[-1, cc] = int(rng.choice(interior_cols))
                    down[0, cc] = int(rng.choice(v_cols))
                else:
                    col = int(rng.choice(interior_cols))
                    up[-1, cc] = col
                    down[0, cc] = col


def _corner_marker_cells(corner: str, mr: int, mc: int) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _single_wildcard_degenerate(canvas: np.ndarray, expected_out: np.ndarray) -> bool:
    g = np.array(canvas, dtype=int)
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, _missing, _corner = _find_marker(g, bg)
    colors = sorted({int(v) for v in g.flatten() if int(v) not in (bg, mark)})
    for col in colors:
        out = _solve_fixed_roles(g, {int(col)}, {int(col)}, return_meta=False)
        if out is not None and np.array_equal(out, expected_out):
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    import random as _r

    palette = list(range(10))
    _r.shuffle(palette)
    bg = int(palette[0])
    marker_col = int(palette[1])

    h_count, v_count = map(int, _r.choice([(1, 2), (2, 1)]))
    total_wild = int(h_count + v_count)
    h_cols = list(map(int, palette[2 : 2 + h_count]))
    v_cols = list(map(int, palette[2 + h_count : 2 + total_wild]))
    n_interior = int(_r.choice([2, 3]))
    interior_start = 2 + total_wild
    interior_cols = list(map(int, palette[interior_start : interior_start + n_interior]))
    if len(set([bg, marker_col] + h_cols + v_cols + interior_cols)) != 2 + total_wild + n_interior:
        return None

    nr = int(_r.choice([2, 3, 4]))
    nc = int(_r.choice([2, 3, 4]))
    if nr * nc < 4 or nr * nc > 12:
        return None
    row_segs = _rand_segments(nr, _r)
    col_segs = _rand_segments(nc, _r)
    RH = int(sum(row_segs))
    RW = int(sum(col_segs))
    if RH > 16 or RW > 16 or RH * RW > 110:
        return None

    GH_min = max(14, RH + 6)
    GW_min = max(14, RW + 6)
    if GH_min > 26 or GW_min > 26:
        return None
    GH = int(_r.randint(GH_min, 26))
    GW = int(_r.randint(GW_min, 26))

    corner = _r.choice(["TL", "TR", "BL", "BR"])
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
    mr = int(_r.randint(mr_min, mr_max))
    mc = int(_r.randint(mc_min, mc_max))

    piece_grid: list[list[np.ndarray]] = []
    for h in row_segs:
        row = []
        for w in col_segs:
            row.append(_dense_piece_pattern(h, w, interior_cols, _r))
        piece_grid.append(row)

    _apply_vertex_constraints(piece_grid, row_segs, col_segs, h_cols, v_cols, interior_cols, _r)
    _decorate_boundaries(piece_grid, row_segs, col_segs, h_cols, v_cols, interior_cols, _r)
    rect = _assemble_rect(piece_grid)

    piece_arrays = [piece_grid[i][j].copy() for i in range(nr) for j in range(nc)]
    for pa in piece_arrays:
        h, w = pa.shape
        if h < 3 or w < 3 or h > 7 or w > 7:
            return None
        interior_mask = np.ones_like(pa, dtype=bool)
        interior_mask[0, :] = False
        interior_mask[-1, :] = False
        interior_mask[:, 0] = False
        interior_mask[:, -1] = False
        if interior_mask.any():
            uniq_inside = set(map(int, pa[interior_mask]))
            if len(uniq_inside) < 2:
                return None

    marker_cells = _corner_marker_cells(corner, mr, mc)
    for cr, cc in marker_cells:
        if not (0 <= cr < GH and 0 <= cc < GW):
            return None

    canvas = np.full((GH, GW), int(bg), dtype=int)
    for cr, cc in marker_cells:
        canvas[cr, cc] = int(marker_col)

    order = list(range(len(piece_arrays)))
    _r.shuffle(order)
    for idx in order:
        pa = piece_arrays[idx]
        try:
            r0, c0 = random_free_location(canvas, pa, transparent=bg, padding=1)
        except ValueError:
            return None
        paste(canvas, pa, r0, c0)

    tr, tc = _top_left_from_corner(mr, mc, RH, RW, corner)
    expected_out = np.full((GH, GW), int(bg), dtype=int)
    expected_out[tr : tr + RH, tc : tc + RW] = rect

    try:
        actual_out = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual_out, expected_out):
        return None

    if _single_wildcard_degenerate(canvas, expected_out):
        return None

    metrics = _fusion_metrics(canvas)
    if metrics["wild_colors"] != total_wild:
        return None
    if metrics["h_required"] <= 0 or metrics["v_required"] <= 0:
        return None
    if metrics["mixed_interior_4adj"] <= 0:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    solved = _solve_core(np.array(grid, dtype=int), return_meta=True)
    out, meta = solved  # type: ignore[misc]
    _ = out
    ass = np.array(meta["assembled"], dtype=int)
    owner = np.array(meta["owner"], dtype=int)
    hset = set(map(int, meta["hset"]))
    vset = set(map(int, meta["vset"]))
    h_required = 0
    v_required = 0
    mixed_interior_4adj = 0
    H, W = ass.shape
    for r in range(H):
        for c in range(W - 1):
            a = int(ass[r, c])
            b = int(ass[r, c + 1])
            if owner[r, c] != owner[r, c + 1] and a != b and (a in hset or b in hset):
                h_required += 1
            if a != b and a not in hset and a not in vset and b not in hset and b not in vset:
                mixed_interior_4adj += 1
    for r in range(H - 1):
        for c in range(W):
            a = int(ass[r, c])
            b = int(ass[r + 1, c])
            if owner[r, c] != owner[r + 1, c] and a != b and (a in vset or b in vset):
                v_required += 1
            if a != b and a not in hset and a not in vset and b not in hset and b not in vset:
                mixed_interior_4adj += 1
    return {
        "h_required": int(h_required),
        "v_required": int(v_required),
        "mixed_interior_4adj": int(mixed_interior_4adj),
        "wild_colors": int(len(hset) + len(vset)),
    }
