from __future__ import annotations

# Variant of 446ef5d2
# Dimension: D9 — rect_col（通配色/边框色）的角色 (meta)
# Change: rect_col 是碎片中最常见的颜色，在边缘匹配时充当通配符，同时构成组装矩形的四条外边... → 通配色由碎片位置决定
#
# Design analysis
# ===============
# a) Concrete behavior and trace
# - The original task has one global rect_col. Every piece uses that same color as both
#   its dominant background and the wildcard on matching edges, so the assembled outer
#   border is monochrome.
# - In this variant, each piece has its own local wildcard color: the most common color
#   inside that piece. Different pieces can use different dominant colors. When two pieces
#   touch, a boundary position is compatible if either side shows its own local wildcard,
#   or both sides show the same non-wildcard color. The assembled outer border is therefore
#   a rectangle whose border is split into piece-colored segments rather than a single color.
# - Small trace:
#   original: left edge [2,7,2], right edge [2,7,2], global rect_col=2 -> compatible.
#   variant: left piece wildcard=2, right piece wildcard=8. Edges [2,7,2] and [8,7,8]
#   are still compatible, because the middle 7 matches and the other positions are local
#   wildcards on each side. But [2,7,2] and [8,6,8] are not compatible because 7 vs 6 are
#   both non-wildcards.
#
# b) Complexity calibration
# - I keep the original "scatter rectangles -> reconstruct anchored rectangle" structure.
# - To make the D9 change visible without adding unrelated complexity, the generator uses
#   2-3 reusable wildcard colors across 4-12 pieces. This produces repeated evidence that
#   wildcard identification is local, while avoiding a visually noisy 8-10 color puzzle.
# - Shared seams carry striped accent patterns so the pieces do not become trivially
#   interchangeable once the global wildcard is removed.
#
# c) Hidden-assumption audit against the original main()
# - `rect_col = Counter(allpix).most_common(1)[0][0]`: invalid now; there is no single
#   wildcard for all pieces.
# - `compat(v1, v2)` using one global rect_col: invalid now. Compatibility must be checked
#   per cell using each placed neighbor's piece id and that piece's own wildcard.
# - `boundary_ok(...)` requiring every outer edge to equal the same rect_col: invalid now.
#   The correct condition is "the edge equals that particular piece's wildcard".
# - Storing only colors in the assembly grid was harmless before because one wildcard was
#   global. It breaks now because the same color value can be wildcard for one piece but a
#   constrained accent for another. The new solver stores both pixel colors and owner ids.
#
# d) Placement legality
# - Pieces remain axis-aligned rectangles with >=1 background cell of separation when
#   scattered.
# - The marker logic is unchanged: the missing corner of the L-marker pins one rectangle
#   corner, and one anchor piece contains that corner at its matching piece corner.
# - The generator validates every produced input by running `main(input)` and comparing the
#   result to the constructed output rectangle.
#
# e) Visual observability
# - Humans can see that different scattered pieces have different dominant colors.
# - In the output, the border is no longer monochrome; it switches color at piece boundaries.
# - Matching seams still show repeated accent stripes, so the rule is inferable from a few
#   examples instead of relying on hidden color ids.
#
# f) Visual cleanliness
# - Only three visible layers remain: background, marker, and rectangular pieces.
# - Accent patterns are limited to seam stripes and occasional tiny interior motifs, so the
#   rectangle stays readable instead of turning into dense multicolor noise.
#
# g) Color inferability
# - Colors are not used by numeric order. The only color-dependent rule is visible frequency
#   within each piece: the piece's most common color is its wildcard.
# - The wildcard palette is intentionally small and reused, so examples provide repeated
#   support for the local-wildcard concept.

from collections import Counter
import numpy as np

from arc_utils import paste, random_free_location


def _find_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], tuple[int, int, int, int], str]:
    cnt = Counter(map(int, g.flatten()))
    mark = None
    missing = None
    mb = None
    for col in sorted([c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3]):
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(col))]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0) == 1 and (c1 - c0) == 1:
            all4 = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            miss = list(all4 - set(pts))
            if len(miss) == 1:
                mark = int(col)
                missing = miss[0]
                mb = (r0, c0, r1, c1)
                break
    assert not (mark is None or missing is None or mb is None)

    r0, c0, r1, c1 = mb
    if missing == (r0, c0):
        corner = "BR"
    elif missing == (r0, c1):
        corner = "BL"
    elif missing == (r1, c0):
        corner = "TR"
    else:
        corner = "TL"
    return mark, missing, mb, corner


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into an anchored rectangle.

    Variant rule:
    - Each piece has its own wildcard color: the most common color inside that piece.
    - Outer borders of the final rectangle are piece-local wildcard segments.
    - At touching boundaries, only positions where both sides show non-wildcard colors must
      match exactly.
    """
    g = np.array(grid, dtype=int)
    GH, GW = g.shape

    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, missing, _mb, corner = _find_marker(g, bg)

    occ = (g != int(bg)) & (g != int(mark))
    rects = _extract_rectangles(g, occ)
    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))

    mr, mc = map(int, missing)
    pieces = [a for _r, _c, _h, _w, a in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]
    piece_wild = [int(Counter(map(int, p.flatten())).most_common(1)[0][0]) for p in pieces]
    nonwild_score = [int(np.sum(p != piece_wild[i])) for i, p in enumerate(pieces)]

    anchor_idx = None
    for i, (rr, cc, h, w, _a) in enumerate(rects):
        if rr <= mr < rr + h and cc <= mc < cc + w:
            anchor_idx = i
            break

    dims: list[tuple[int, int]] = []
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

    def edge_compat(other_vals: np.ndarray, other_ids: np.ndarray, cur_vals: np.ndarray, cur_wild: int) -> bool:
        for ov, oid, cv in zip(map(int, other_vals), map(int, other_ids), map(int, cur_vals)):
            if oid == -1:
                continue
            other_wild = piece_wild[int(oid)]
            if ov == other_wild or cv == int(cur_wild) or ov == cv:
                continue
            return False
        return True

    def assemble(H: int, W: int) -> np.ndarray | None:
        occ2 = np.full((H, W), -1, dtype=int)
        owner = np.full((H, W), -1, dtype=int)
        usedp = [False] * len(pieces)

        def boundary_ok(i: int, r: int, c: int) -> bool:
            a = pieces[i]
            h, w = a.shape
            wild = piece_wild[i]
            if r == 0 and not np.all(a[0, :] == int(wild)):
                return False
            if (r + h) == H and not np.all(a[-1, :] == int(wild)):
                return False
            if c == 0 and not np.all(a[:, 0] == int(wild)):
                return False
            if (c + w) == W and not np.all(a[:, -1] == int(wild)):
                return False
            return True

        def neighbor_ok(i: int, r: int, c: int) -> bool:
            a = pieces[i]
            h, w = a.shape
            wild = piece_wild[i]
            if c > 0:
                left_ids = owner[r : r + h, c - 1]
                m = left_ids != -1
                if m.any() and (not edge_compat(occ2[r : r + h, c - 1][m], left_ids[m], a[:, 0][m], wild)):
                    return False
            if (c + w) < W:
                right_ids = owner[r : r + h, c + w]
                m = right_ids != -1
                if m.any() and (not edge_compat(occ2[r : r + h, c + w][m], right_ids[m], a[:, -1][m], wild)):
                    return False
            if r > 0:
                up_ids = owner[r - 1, c : c + w]
                m = up_ids != -1
                if m.any() and (not edge_compat(occ2[r - 1, c : c + w][m], up_ids[m], a[0, :][m], wild)):
                    return False
            if (r + h) < H:
                down_ids = owner[r + h, c : c + w]
                m = down_ids != -1
                if m.any() and (not edge_compat(occ2[r + h, c : c + w][m], down_ids[m], a[-1, :][m], wild)):
                    return False
            return True

        def can_place(i: int, r: int, c: int) -> bool:
            a = pieces[i]
            h, w = a.shape
            if (r + h) > H or (c + w) > W:
                return False
            if (owner[r : r + h, c : c + w] != -1).any():
                return False
            if not boundary_ok(i, r, c):
                return False
            if not neighbor_ok(i, r, c):
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

        order = sorted(
            range(len(pieces)),
            key=lambda i: (-nonwild_score[i], -(ph[i] * pw[i]), -ph[i], -pw[i], i),
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
        tr, tc = mr, mc
    elif corner == "TR":
        tr, tc = mr, mc - W + 1
    elif corner == "BL":
        tr, tc = mr - H + 1, mc
    else:
        tr, tc = mr - H + 1, mc - W + 1

    out = np.full((GH, GW), int(bg), dtype=int)
    out[tr : tr + H, tc : tc + W] = ass
    return out


def _rand_splits(total: int, rng, min_s: int = 2, max_s: int = 6) -> list[int] | None:
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


def _partition_rectangle(RH: int, RW: int, rng):
    row_segs = _rand_splits(RH, rng)
    col_segs = _rand_splits(RW, rng)
    if row_segs is None or col_segs is None:
        return None

    row_starts: list[int] = []
    col_starts: list[int] = []
    s = 0
    for rs in row_segs:
        row_starts.append(s)
        s += rs
    s = 0
    for cs in col_segs:
        col_starts.append(s)
        s += cs

    nr = len(row_segs)
    nc = len(col_segs)
    cell_id = [[i * nc + j for j in range(nc)] for i in range(nr)]
    n_cells = nr * nc
    if n_cells < 4:
        return None
    target_pieces = rng.randint(4, min(10, n_cells))

    def get_piece_bbox(pid: int):
        cells = [(i, j) for i in range(nr) for j in range(nc) if cell_id[i][j] == pid]
        if not cells:
            return None
        ris = [t[0] for t in cells]
        cjs = [t[1] for t in cells]
        return min(ris), min(cjs), max(ris), max(cjs)

    def pixel_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
        h = sum(row_segs[i] for i in range(r0i, r1i + 1))
        w = sum(col_segs[j] for j in range(c0i, c1i + 1))
        return h, w

    merges = []
    for i in range(nr):
        for j in range(nc):
            if j + 1 < nc:
                merges.append(("h", i, j))
            if i + 1 < nr:
                merges.append(("v", i, j))
    rng.shuffle(merges)

    def count_pieces() -> int:
        return len(set(cell_id[i][j] for i in range(nr) for j in range(nc)))

    for direction, mi, mj in merges:
        if count_pieces() <= target_pieces:
            break
        pid1 = cell_id[mi][mj]
        pid2 = cell_id[mi][mj + 1] if direction == "h" else cell_id[mi + 1][mj]
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

    pieces = []
    seen = set()
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
            ph = sum(row_segs[ii] for ii in range(r0i, r1i + 1))
            pw = sum(col_segs[jj] for jj in range(c0i, c1i + 1))
            pieces.append(
                {
                    "pr": pr,
                    "pc": pc,
                    "ph": ph,
                    "pw": pw,
                    "r0i": r0i,
                    "c0i": c0i,
                    "r1i": r1i,
                    "c1i": c1i,
                }
            )

    if len(pieces) < 4 or len(pieces) > 10:
        return None
    for p in pieces:
        if p["ph"] < 2 or p["pw"] < 2 or p["ph"] > 7 or p["pw"] > 7:
            return None
    return row_segs, col_segs, pieces


def _edge_positions(length: int, start: int, period: int, phase: int) -> list[int]:
    pos = [i for i in range(length) if ((start + i + phase) % period) == 0]
    if not pos:
        pos = [length // 2]
    return pos


def _build_piece_arrays(RH: int, RW: int, pieces, wild_cols: list[int], accent_cols: list[int], rng):
    piece_arrays = []
    piece_wilds = []
    for _ in pieces:
        piece_wilds.append(int(wild_cols[rng.randrange(len(wild_cols))]))
    if len(set(piece_wilds)) < 2:
        return None

    for i, p in enumerate(pieces):
        piece_arrays.append(np.full((p["ph"], p["pw"]), piece_wilds[i], dtype=int))

    v_sig = {}
    h_sig = {}

    def get_v_sig(key: int) -> tuple[int, int, int]:
        if key not in v_sig:
            period = rng.choice([2, 3])
            v_sig[key] = (
                int(accent_cols[key % len(accent_cols)]),
                period,
                rng.randrange(period),
            )
        return v_sig[key]

    def get_h_sig(key: int) -> tuple[int, int, int]:
        if key not in h_sig:
            period = rng.choice([2, 3])
            h_sig[key] = (
                int(accent_cols[(key + 1) % len(accent_cols)]),
                period,
                rng.randrange(period),
            )
        return h_sig[key]

    for i in range(len(pieces)):
        pi = pieces[i]
        for j in range(i + 1, len(pieces)):
            pj = pieces[j]
            if pi["pc"] + pi["pw"] == pj["pc"] or pj["pc"] + pj["pw"] == pi["pc"]:
                if pi["pc"] + pi["pw"] == pj["pc"]:
                    left_idx, right_idx = i, j
                    left, right = pi, pj
                else:
                    left_idx, right_idx = j, i
                    left, right = pj, pi
                r0 = max(left["pr"], right["pr"])
                r1 = min(left["pr"] + left["ph"], right["pr"] + right["ph"])
                if r0 < r1:
                    length = r1 - r0
                    color, period, phase = get_v_sig(left["c1i"])
                    pos = _edge_positions(length, r0, period, phase)
                    la = piece_arrays[left_idx]
                    ra = piece_arrays[right_idx]
                    for off in pos:
                        lr = r0 - left["pr"] + off
                        rr = r0 - right["pr"] + off
                        la[lr, -1] = int(color)
                        ra[rr, 0] = int(color)
            if pi["pr"] + pi["ph"] == pj["pr"] or pj["pr"] + pj["ph"] == pi["pr"]:
                if pi["pr"] + pi["ph"] == pj["pr"]:
                    top_idx, bot_idx = i, j
                    top, bot = pi, pj
                else:
                    top_idx, bot_idx = j, i
                    top, bot = pj, pi
                c0 = max(top["pc"], bot["pc"])
                c1 = min(top["pc"] + top["pw"], bot["pc"] + bot["pw"])
                if c0 < c1:
                    length = c1 - c0
                    color, period, phase = get_h_sig(top["r1i"])
                    pos = _edge_positions(length, c0, period, phase)
                    ta = piece_arrays[top_idx]
                    ba = piece_arrays[bot_idx]
                    for off in pos:
                        tc = c0 - top["pc"] + off
                        bc = c0 - bot["pc"] + off
                        ta[-1, tc] = int(color)
                        ba[0, bc] = int(color)

    for i, p in enumerate(pieces):
        a = piece_arrays[i]
        base = piece_wilds[i]
        if p["ph"] >= 4 and p["pw"] >= 4 and rng.random() < 0.65:
            ih = rng.randint(1, min(2, p["ph"] - 2))
            iw = rng.randint(1, min(2, p["pw"] - 2))
            ir = rng.randint(1, p["ph"] - ih - 1)
            ic = rng.randint(1, p["pw"] - iw - 1)
            acol = int(accent_cols[(i + p["ph"] + p["pw"]) % len(accent_cols)])
            a[ir : ir + ih, ic : ic + iw] = acol
        if Counter(map(int, a.flatten())).most_common(1)[0][0] != int(base):
            return None
        if not any(np.any(edge != int(base)) for edge in (a[0, :], a[-1, :], a[:, 0], a[:, -1])):
            return None

    rect = np.full((RH, RW), -1, dtype=int)
    for i, p in enumerate(pieces):
        pr, pc, ph, pw = p["pr"], p["pc"], p["ph"], p["pw"]
        rect[pr : pr + ph, pc : pc + pw] = piece_arrays[i]
    assert np.all(rect != -1)
    return rect, piece_arrays, piece_wilds


def _gen_attempt() -> np.ndarray | None:
    import random as _r

    GH = _r.randint(10, 26)
    GW = _r.randint(10, 26)

    palette = list(range(10))
    _r.shuffle(palette)
    bg = int(palette[0])
    marker_col = int(palette[1])
    n_wild = _r.randint(2, 3)
    n_accent = _r.randint(1, 2)
    wild_cols = [int(c) for c in palette[2 : 2 + n_wild]]
    accent_cols = [int(c) for c in palette[2 + n_wild : 2 + n_wild + n_accent]]

    corner = _r.choice(["TL", "TR", "BL", "BR"])

    max_side = min(18, min(GH, GW) - 4)
    if max_side < 6:
        return None
    RH = _r.randint(6, max_side)
    RW = _r.randint(6, max_side)

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

    part = _partition_rectangle(RH, RW, _r)
    if part is None:
        return None
    _row_segs, _col_segs, pieces = part

    built = _build_piece_arrays(RH, RW, pieces, wild_cols, accent_cols, _r)
    if built is None:
        return None
    rect, piece_arrays, piece_wilds = built

    # Outer border of each border piece must be its own wildcard color.
    for i, p in enumerate(pieces):
        a = piece_arrays[i]
        wild = piece_wilds[i]
        if p["pr"] == 0 and not np.all(a[0, :] == int(wild)):
            return None
        if p["pr"] + p["ph"] == RH and not np.all(a[-1, :] == int(wild)):
            return None
        if p["pc"] == 0 and not np.all(a[:, 0] == int(wild)):
            return None
        if p["pc"] + p["pw"] == RW and not np.all(a[:, -1] == int(wild)):
            return None

    anchor_r = 0 if corner in ("TL", "TR") else RH - 1
    anchor_c = 0 if corner in ("TL", "BL") else RW - 1
    anchor_idx = None
    for i, p in enumerate(pieces):
        if p["pr"] <= anchor_r < p["pr"] + p["ph"] and p["pc"] <= anchor_c < p["pc"] + p["pw"]:
            anchor_idx = i
            break
    if anchor_idx is None:
        return None
    ap = pieces[anchor_idx]
    if corner == "TL" and not (ap["pr"] == 0 and ap["pc"] == 0):
        return None
    if corner == "TR" and not (ap["pr"] == 0 and ap["pc"] + ap["pw"] == RW):
        return None
    if corner == "BL" and not (ap["pr"] + ap["ph"] == RH and ap["pc"] == 0):
        return None
    if corner == "BR" and not (ap["pr"] + ap["ph"] == RH and ap["pc"] + ap["pw"] == RW):
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

    canvas = np.full((GH, GW), bg, dtype=int)
    for cr, cc in marker_cells:
        canvas[cr, cc] = int(marker_col)

    order = list(range(len(piece_arrays)))
    _r.shuffle(order)
    for idx in order:
        pa = piece_arrays[idx]
        try:
            r0, c0 = random_free_location(canvas, pa, transparent=bg, padding=1)
            paste(canvas, pa, r0, c0)
        except ValueError:
            return None

    expected_out = np.full((GH, GW), bg, dtype=int)
    expected_out[tr : tr + RH, tc : tc + RW] = rect

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
