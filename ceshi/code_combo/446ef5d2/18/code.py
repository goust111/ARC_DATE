# Combo variant of 446ef5d2 (index 18)
# combo_spec: D1/5+D9/2
# Name: 连接器匹配配合碎片自有通配色
#
# Design analysis
# ===============
# a) Concrete behavior / behavior trace
# - Visual object model:
#   * Input still contains scattered rectangular pieces plus a 3-cell L marker.
#   * Output is still one axis-aligned rectangle whose corner is pinned by the marker gap.
#   * Two changes are fused:
#     1. Matching no longer uses whole-edge compatibility. Each INTERNAL side has exactly one
#        connector dot. Two pieces may touch iff opposite sides have the same connector color
#        at the same offset.
#     2. There is no global rect_col anymore. Each piece has its own wildcard color: the most
#        common color inside that piece. Outer rectangle borders are therefore piece-colored
#        segments, not one global monochrome frame.
# - Small trace on a vertical seam:
#   * Original task: left edge [2, 7, 2], right edge [2, 7, 2], global rect_col=2 -> match.
#   * D9-only local-wild idea: left wildcard=2, right wildcard=8. Then [2, 7, 2] can still
#     match [8, 7, 8] because only non-wild cells must agree.
#   * Fused rule here: left edge [2, 4, 6, 2], right edge [8, 4, 5, 8]. The shared connector
#     is the 4 at offset 1. The 6 vs 5 mismatch at offset 2 is deliberate and must be ignored.
#     So whole-edge matching fails, local-wild whole-edge matching also fails, but connector
#     matching succeeds. This is the intended behavioral difference.
#
# b) Complexity calibration
# - The generator uses a clean 2x2 / 2x3 / 3x2 rectangular grid partition. This keeps the
#   connector semantics visible: each side has at most one neighbor and therefore at most one
#   connector.
# - Wildcard colors are chosen from a tiny palette of two reusable colors. This gives repeated
#   evidence that wildcard identity is local to each piece rather than globally fixed.
# - Connector colors are also chosen from a tiny repeated palette of two edge-only colors.
#   That keeps the "connector" concept visible instead of hiding it behind one-off colors.
# - Every real seam gets one extra non-wild mismatch point. This prevents degeneration back to
#   any whole-edge compatibility rule while staying visually simple.
#
# c) Hidden-assumption audit against the original main()
# - `rect_col = Counter(allpix).most_common(1)[0][0]`: invalid. There is no single wildcard.
#   The solver now computes `piece_wild[i]` separately for every piece.
# - `compat(v1, v2)` comparing whole edges with one rect_col wildcard: invalid. True seams in
#   this combo intentionally contain a non-wild mismatch, so only connector descriptors matter.
# - `boundary_ok(...)` requiring one global border color: invalid. Each border segment must equal
#   that specific piece's wildcard.
# - Keeping only colors in the assembly board is insufficient if wildcard meaning is local.
#   The solver stores both pixel colors and owner ids so it can reason about full-side contact.
# - The original solver allowed partial side overlap as long as compared pixels were compatible.
#   That would blur connector semantics. The fused solver therefore assumes grid-style tiling:
#   if two pieces touch, they share a full side and that side has exactly one connector.
#
# d) Placement legality
# - The generator places the assembled rectangle first, partitions it into rectangle pieces, then
#   scatters those pieces with >=1 background padding using `random_free_location`.
# - Connectors and mismatch dots are only written onto INTERNAL sides, never onto outer borders
#   or side corners, so the final border remains a clean piece-local wildcard frame.
# - Marker cells are placed first and pieces are scattered around them, preventing overlap.
#
# e) Visual observability
# - Humans can see that different pieces have different dominant colors.
# - They can also see small repeated connector dots on internal sides and notice that some true
#   matching seams contain another visible color disagreement elsewhere on the same side.
# - Across 2-4 examples, the plausible rule is therefore: "identify each piece's own base color,
#   then align pieces by edge connector dots rather than by full edge stripes."
#
# f) Visual cleanliness
# - The grid contains only three visually distinct layers: background, marker, and rectangular
#   pieces. Within pieces, non-wild content is sparse: small interior motifs plus one connector
#   and one mismatch dot per internal side.
# - Using only two wildcard colors, two connector colors, and two accent colors keeps the scene
#   legible instead of turning into dense color noise.
#
# g) Change inferability
# - Color ids are never interpreted numerically. The only color-dependent rules are visible:
#   * a piece's most common color is its wildcard;
#   * connector colors appear only as edge dots;
#   * accent colors appear in small interior motifs and at deliberate seam mismatches.
# - Reuse of both wildcard colors and both connector colors across pieces gives enough repeated
#   structure for a human to infer the rule from a small number of examples.
#
# New solver assumptions introduced here
# --------------------------------------
# - Every piece is at least 4x4, so its wildcard also appears in the strict interior. This makes
#   connector colors inferable as edge-only colors, and the generator guarantees this.
# - Final tiling is a clean row/column grid. If two pieces touch, they touch along one whole side.
# - Every internal side has exactly one connector and one additional non-wild mismatch point.
# - Connector colors never appear in any piece interior. Accent colors may appear both in interiors
#   and on mismatch points. The generator validates these assumptions by running `main()`.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import paste, random_free_location


SIDES = ("T", "B", "L", "R")


def _find_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], str]:
    cnt = Counter(map(int, g.flatten()))
    mark = None
    missing = None
    bbox = None
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
                missing = (int(miss[0][0]), int(miss[0][1]))
                bbox = (int(r0), int(c0), int(r1), int(c1))
                break
    assert not (mark is None or missing is None or bbox is None)

    r0, c0, r1, c1 = bbox
    if missing == (r0, c0):
        corner = "BR"
    elif missing == (r0, c1):
        corner = "BL"
    elif missing == (r1, c0):
        corner = "TR"
    else:
        corner = "TL"
    return int(mark), missing, corner


def _extract_rects(g: np.ndarray, bg: int, mark: int) -> list[tuple[int, int, int, int, np.ndarray]]:
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


def _edge_values(a: np.ndarray, side: str) -> np.ndarray:
    if side == "T":
        return a[0, :]
    if side == "B":
        return a[-1, :]
    if side == "L":
        return a[:, 0]
    return a[:, -1]


def _opposite(side: str) -> str:
    return {"T": "B", "B": "T", "L": "R", "R": "L"}[side]


def _connector_desc(a: np.ndarray, connector_colors: set[int]) -> dict[str, tuple[int, int] | None]:
    desc: dict[str, tuple[int, int] | None] = {}
    for side in SIDES:
        edge = np.array(_edge_values(a, side), dtype=int)
        idx = [int(k) for k, v in enumerate(edge) if int(v) in connector_colors]
        assert len(idx) <= 1
        desc[side] = (idx[0], int(edge[idx[0]])) if idx else None
    return desc


def _collect_meta(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, missing, corner = _find_marker(g, bg)
    rects = _extract_rects(g, bg, mark)
    pieces = [a for _r, _c, _h, _w, a in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]
    piece_wild = [int(Counter(map(int, p.flatten())).most_common(1)[0][0]) for p in pieces]

    edge_colors: set[int] = set()
    interior_colors: set[int] = set()
    for a, wild in zip(pieces, piece_wild):
        edge_colors.update(map(int, a[0, :]))
        edge_colors.update(map(int, a[-1, :]))
        edge_colors.update(map(int, a[:, 0]))
        edge_colors.update(map(int, a[:, -1]))
        interior_colors.update(map(int, a[1:-1, 1:-1].flatten()))
        interior_colors.add(int(wild))
    connector_colors = {int(c) for c in edge_colors if int(c) not in interior_colors}
    assert connector_colors

    descs = [_connector_desc(a, connector_colors) for a in pieces]
    return {
        "g": g,
        "GH": GH,
        "GW": GW,
        "bg": bg,
        "mark": mark,
        "missing": missing,
        "corner": corner,
        "rects": rects,
        "pieces": pieces,
        "ph": ph,
        "pw": pw,
        "piece_wild": piece_wild,
        "connector_colors": connector_colors,
        "descs": descs,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into a marker-anchored rectangle.

    Fused rule:
    - Each piece has its own wildcard color: its most common color.
    - Outer rectangle borders are piece-local wildcard segments.
    - Each internal side has exactly one connector pixel; opposite connectors must align in
      both color and offset. All other seam pixels are ignored.
    """
    meta = _collect_meta(grid)
    GH, GW = meta["GH"], meta["GW"]
    bg = meta["bg"]
    missing = meta["missing"]
    corner = meta["corner"]
    rects = meta["rects"]
    pieces = meta["pieces"]
    ph = meta["ph"]
    pw = meta["pw"]
    piece_wild = meta["piece_wild"]
    descs = meta["descs"]

    total_area = int(sum(int(h) * int(w) for h, w in zip(ph, pw)))
    mr, mc = map(int, missing)

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

    def assemble(H: int, W: int) -> np.ndarray | None:
        color_occ = np.full((H, W), -1, dtype=int)
        pid_occ = np.full((H, W), -1, dtype=int)
        usedp = [False] * len(pieces)
        placed: dict[int, tuple[int, int]] = {}

        def at_boundary(side: str, r: int, c: int, h: int, w: int) -> bool:
            if side == "T":
                return r == 0
            if side == "B":
                return (r + h) == H
            if side == "L":
                return c == 0
            return (c + w) == W

        def full_neighbor(side: str, r: int, c: int, h: int, w: int) -> int | None:
            if side == "L":
                if c == 0:
                    return None
                seg = pid_occ[r : r + h, c - 1]
                if np.all(seg == -1):
                    return None
                if np.any(seg == -1):
                    return -2
                ids = set(map(int, seg))
                if len(ids) != 1:
                    return -2
                j = ids.pop()
                jr, jc = placed[j]
                jh, jw = pieces[j].shape
                return int(j) if (jr == r and jh == h and jc + jw == c) else -2
            if side == "R":
                if (c + w) == W:
                    return None
                seg = pid_occ[r : r + h, c + w]
                if np.all(seg == -1):
                    return None
                if np.any(seg == -1):
                    return -2
                ids = set(map(int, seg))
                if len(ids) != 1:
                    return -2
                j = ids.pop()
                jr, jc = placed[j]
                jh, jw = pieces[j].shape
                return int(j) if (jr == r and jh == h and c + w == jc) else -2
            if side == "T":
                if r == 0:
                    return None
                seg = pid_occ[r - 1, c : c + w]
                if np.all(seg == -1):
                    return None
                if np.any(seg == -1):
                    return -2
                ids = set(map(int, seg))
                if len(ids) != 1:
                    return -2
                j = ids.pop()
                jr, jc = placed[j]
                jh, jw = pieces[j].shape
                return int(j) if (jc == c and jw == w and jr + jh == r) else -2
            if (r + h) == H:
                return None
            seg = pid_occ[r + h, c : c + w]
            if np.all(seg == -1):
                return None
            if np.any(seg == -1):
                return -2
            ids = set(map(int, seg))
            if len(ids) != 1:
                return -2
            j = ids.pop()
            jr, jc = placed[j]
            jh, jw = pieces[j].shape
            return int(j) if (jc == c and jw == w and r + h == jr) else -2

        def side_ok(i: int, side: str, r: int, c: int) -> bool:
            a = pieces[i]
            h, w = a.shape
            edge = np.array(_edge_values(a, side), dtype=int)
            desc = descs[i][side]
            wild = piece_wild[i]
            if at_boundary(side, r, c, h, w):
                return desc is None and bool(np.all(edge == int(wild)))
            if desc is None:
                return False
            j = full_neighbor(side, r, c, h, w)
            if j == -2:
                return False
            if j is None:
                return True
            od = descs[j][_opposite(side)]
            return od is not None and od == desc

        def can_place(i: int, r: int, c: int) -> bool:
            a = pieces[i]
            h, w = a.shape
            if (r + h) > H or (c + w) > W:
                return False
            if (pid_occ[r : r + h, c : c + w] != -1).any():
                return False
            return (
                side_ok(i, "T", r, c)
                and side_ok(i, "B", r, c)
                and side_ok(i, "L", r, c)
                and side_ok(i, "R", r, c)
            )

        def place(i: int, r: int, c: int) -> None:
            a = pieces[i]
            h, w = a.shape
            color_occ[r : r + h, c : c + w] = a
            pid_occ[r : r + h, c : c + w] = int(i)
            usedp[i] = True
            placed[i] = (int(r), int(c))

        def unplace(i: int, r: int, c: int) -> None:
            a = pieces[i]
            h, w = a.shape
            color_occ[r : r + h, c : c + w] = -1
            pid_occ[r : r + h, c : c + w] = -1
            usedp[i] = False
            del placed[i]

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
            key=lambda i: (
                descs[i]["T"] is not None,
                descs[i]["L"] is not None,
                -ph[i] * pw[i],
                -ph[i],
                -pw[i],
                i,
            ),
        )

        def next_empty() -> tuple[int, int] | None:
            idx = np.argwhere(pid_occ == -1)
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
        return color_occ if solved else None

    ass = None
    used_dim = None
    for H, W in dims:
        ass = assemble(H, W)
        if ass is not None:
            used_dim = (int(H), int(W))
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


def _assembled_top_left(corner: str, mr: int, mc: int, RH: int, RW: int) -> tuple[int, int]:
    if corner == "TL":
        return int(mr), int(mc)
    if corner == "TR":
        return int(mr), int(mc - RW + 1)
    if corner == "BL":
        return int(mr - RH + 1), int(mc)
    return int(mr - RH + 1), int(mc - RW + 1)


def _marker_cells(corner: str, mr: int, mc: int) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _choose_grid_shape(rng: random.Random) -> tuple[int, int]:
    return rng.choice([(2, 2), (2, 3), (3, 2)])


def _segment_lengths(n: int, rng: random.Random) -> list[int]:
    lengths = [rng.randint(4, 6) for _ in range(n)]
    if len(set(lengths)) == 1:
        lengths[rng.randrange(n)] = 4 if lengths[0] != 4 else 6
    return lengths


def _row_col_starts(segs: list[int]) -> list[int]:
    starts: list[int] = []
    s = 0
    for v in segs:
        starts.append(int(s))
        s += int(v)
    return starts


def _safe_two_positions(length: int, rng: random.Random) -> tuple[int, int]:
    pos = list(range(1, length - 1))
    assert len(pos) >= 2
    rng.shuffle(pos)
    return int(pos[0]), int(pos[1])


def _paint_interior_motif(a: np.ndarray, accent_cols: list[int], idx: int, rng: random.Random) -> None:
    h, w = a.shape
    ah = rng.randint(1, min(2, h - 2))
    aw = rng.randint(1, min(2, w - 2))
    ar = rng.randint(1, h - ah - 1)
    ac = rng.randint(1, w - aw - 1)
    acol = int(accent_cols[idx % len(accent_cols)])
    a[ar : ar + ah, ac : ac + aw] = int(acol)
    if rng.random() < 0.5:
        br = rng.randint(1, h - 2)
        bc = rng.randint(1, w - 2)
        bcol = int(accent_cols[(idx + 1) % len(accent_cols)])
        a[br, bc] = int(bcol)


def _build_piece_arrays(
    row_segs: list[int],
    col_segs: list[int],
    wild_cols: list[int],
    connector_cols: list[int],
    accent_cols: list[int],
    rng: random.Random,
) -> tuple[np.ndarray, list[np.ndarray], list[int], dict[str, int]] | None:
    nr, nc = len(row_segs), len(col_segs)
    RH, RW = int(sum(row_segs)), int(sum(col_segs))
    row_starts = _row_col_starts(row_segs)
    col_starts = _row_col_starts(col_segs)

    piece_arrays: list[np.ndarray] = []
    piece_wilds: list[int] = []
    for i in range(nr):
        for j in range(nc):
            wild = int(wild_cols[(i + 2 * j + (i * j)) % len(wild_cols)])
            piece_wilds.append(wild)
            a = np.full((row_segs[i], col_segs[j]), int(wild), dtype=int)
            _paint_interior_motif(a, accent_cols, i * nc + j, rng)
            piece_arrays.append(a)

    if len(set(piece_wilds)) < 2:
        return None

    connector_pairs = 0
    mismatch_pairs = 0

    def idx_of(i: int, j: int) -> int:
        return i * nc + j

    for i in range(nr):
        for j in range(nc - 1):
            li = idx_of(i, j)
            ri = idx_of(i, j + 1)
            left = piece_arrays[li]
            right = piece_arrays[ri]
            pos_conn, pos_mis = _safe_two_positions(left.shape[0], rng)
            conn_col = int(connector_cols[(i + j) % len(connector_cols)])
            left[pos_conn, -1] = int(conn_col)
            right[pos_conn, 0] = int(conn_col)
            left[pos_mis, -1] = int(accent_cols[(i + j) % len(accent_cols)])
            right[pos_mis, 0] = int(accent_cols[(i + j + 1) % len(accent_cols)])
            connector_pairs += 1
            mismatch_pairs += 1

    for i in range(nr - 1):
        for j in range(nc):
            ti = idx_of(i, j)
            bi = idx_of(i + 1, j)
            top = piece_arrays[ti]
            bot = piece_arrays[bi]
            pos_conn, pos_mis = _safe_two_positions(top.shape[1], rng)
            conn_col = int(connector_cols[(i + j + 1) % len(connector_cols)])
            top[-1, pos_conn] = int(conn_col)
            bot[0, pos_conn] = int(conn_col)
            top[-1, pos_mis] = int(accent_cols[(i + j + 1) % len(accent_cols)])
            bot[0, pos_mis] = int(accent_cols[(i + j) % len(accent_cols)])
            connector_pairs += 1
            mismatch_pairs += 1

    rect = np.full((RH, RW), -1, dtype=int)
    for i in range(nr):
        for j in range(nc):
            idx = idx_of(i, j)
            pr, pc = row_starts[i], col_starts[j]
            a = piece_arrays[idx]
            rect[pr : pr + a.shape[0], pc : pc + a.shape[1]] = a

    if np.any(rect == -1):
        return None

    for idx, a in enumerate(piece_arrays):
        wild = piece_wilds[idx]
        if Counter(map(int, a.flatten())).most_common(1)[0][0] != int(wild):
            return None
        if not np.any(a[1:-1, 1:-1] == int(wild)):
            return None

    metrics = {
        "connector_pairs": int(connector_pairs),
        "mismatch_pairs": int(mismatch_pairs),
        "distinct_wilds": int(len(set(piece_wilds))),
    }
    return rect, piece_arrays, piece_wilds, metrics


def _piece_specs_from_grid(row_segs: list[int], col_segs: list[int]) -> list[dict[str, int]]:
    nr, nc = len(row_segs), len(col_segs)
    row_starts = _row_col_starts(row_segs)
    col_starts = _row_col_starts(col_segs)
    specs: list[dict[str, int]] = []
    for i in range(nr):
        for j in range(nc):
            specs.append(
                {
                    "pr": int(row_starts[i]),
                    "pc": int(col_starts[j]),
                    "ph": int(row_segs[i]),
                    "pw": int(col_segs[j]),
                    "gi": int(i),
                    "gj": int(j),
                }
            )
    return specs


def _validate_piece_layout(
    piece_arrays: list[np.ndarray],
    piece_specs: list[dict[str, int]],
    piece_wilds: list[int],
    connector_cols: set[int],
    RH: int,
    RW: int,
) -> bool:
    for spec, a, wild in zip(piece_specs, piece_arrays, piece_wilds):
        pr, pc, ph, pw = spec["pr"], spec["pc"], spec["ph"], spec["pw"]
        if not (4 <= ph <= 6 and 4 <= pw <= 6):
            return False
        if pr == 0 and not np.all(a[0, :] == int(wild)):
            return False
        if pr + ph == RH and not np.all(a[-1, :] == int(wild)):
            return False
        if pc == 0 and not np.all(a[:, 0] == int(wild)):
            return False
        if pc + pw == RW and not np.all(a[:, -1] == int(wild)):
            return False
        desc = _connector_desc(a, connector_cols)
        expect = {
            "T": None if pr == 0 else "present",
            "B": None if pr + ph == RH else "present",
            "L": None if pc == 0 else "present",
            "R": None if pc + pw == RW else "present",
        }
        for side, need in expect.items():
            if need is None and desc[side] is not None:
                return False
            if need == "present" and desc[side] is None:
                return False
    return True


def _connector_mismatch_metric(
    row_segs: list[int],
    col_segs: list[int],
    piece_arrays: list[np.ndarray],
    piece_wilds: list[int],
    connector_cols: set[int],
) -> dict[str, int]:
    nr, nc = len(row_segs), len(col_segs)
    connector_ok = 0
    nonwild_mismatch = 0

    def idx_of(i: int, j: int) -> int:
        return i * nc + j

    for i in range(nr):
        for j in range(nc - 1):
            left = piece_arrays[idx_of(i, j)]
            right = piece_arrays[idx_of(i, j + 1)]
            lw = int(piece_wilds[idx_of(i, j)])
            rw = int(piece_wilds[idx_of(i, j + 1)])
            for a, b in zip(map(int, left[:, -1]), map(int, right[:, 0])):
                if a in connector_cols and b in connector_cols and a == b:
                    connector_ok += 1
                elif a != int(lw) and b != int(rw) and a != b:
                    nonwild_mismatch += 1
        # no else

    for i in range(nr - 1):
        for j in range(nc):
            top = piece_arrays[idx_of(i, j)]
            bot = piece_arrays[idx_of(i + 1, j)]
            tw = int(piece_wilds[idx_of(i, j)])
            bw = int(piece_wilds[idx_of(i + 1, j)])
            for a, b in zip(map(int, top[-1, :]), map(int, bot[0, :])):
                if a in connector_cols and b in connector_cols and a == b:
                    connector_ok += 1
                elif a != int(tw) and b != int(bw) and a != b:
                    nonwild_mismatch += 1

    return {
        "connector_ok": int(connector_ok),
        "nonwild_mismatch": int(nonwild_mismatch),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    palette = list(range(10))
    rng.shuffle(palette)
    bg = int(palette[0])
    marker_col = int(palette[1])
    wild_cols = [int(palette[2]), int(palette[3])]
    connector_cols = [int(palette[4]), int(palette[5])]
    accent_cols = [int(palette[6]), int(palette[7])]

    nr, nc = _choose_grid_shape(rng)
    row_segs = _segment_lengths(nr, rng)
    col_segs = _segment_lengths(nc, rng)
    RH, RW = int(sum(row_segs)), int(sum(col_segs))

    GH_min = max(10, RH + 4)
    GW_min = max(10, RW + 4)
    if GH_min > 26 or GW_min > 26:
        return None
    GH = rng.randint(GH_min, 26)
    GW = rng.randint(GW_min, 26)
    if GH * GW < int(2.1 * RH * RW):
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

    built = _build_piece_arrays(row_segs, col_segs, wild_cols, connector_cols, accent_cols, rng)
    if built is None:
        return None
    rect, piece_arrays, piece_wilds, metrics = built
    piece_specs = _piece_specs_from_grid(row_segs, col_segs)

    if not _validate_piece_layout(piece_arrays, piece_specs, piece_wilds, set(connector_cols), RH, RW):
        return None

    seam_metric = _connector_mismatch_metric(
        row_segs,
        col_segs,
        piece_arrays,
        piece_wilds,
        set(connector_cols),
    )
    if seam_metric["connector_ok"] != metrics["connector_pairs"]:
        return None
    if seam_metric["nonwild_mismatch"] < metrics["mismatch_pairs"]:
        return None

    anchor_r = 0 if corner in ("TL", "TR") else RH - 1
    anchor_c = 0 if corner in ("TL", "BL") else RW - 1
    anchor_idx = None
    for i, spec in enumerate(piece_specs):
        if spec["pr"] <= anchor_r < spec["pr"] + spec["ph"] and spec["pc"] <= anchor_c < spec["pc"] + spec["pw"]:
            anchor_idx = i
            break
    if anchor_idx is None:
        return None
    ap = piece_specs[anchor_idx]
    if corner == "TL" and not (ap["pr"] == 0 and ap["pc"] == 0):
        return None
    if corner == "TR" and not (ap["pr"] == 0 and ap["pc"] + ap["pw"] == RW):
        return None
    if corner == "BL" and not (ap["pr"] + ap["ph"] == RH and ap["pc"] == 0):
        return None
    if corner == "BR" and not (ap["pr"] + ap["ph"] == RH and ap["pc"] + ap["pw"] == RW):
        return None

    marker_cells = _marker_cells(corner, mr, mc)
    for cr, cc in marker_cells:
        if cr < 0 or cr >= GH or cc < 0 or cc >= GW:
            return None

    canvas = np.full((GH, GW), int(bg), dtype=int)
    for cr, cc in marker_cells:
        canvas[cr, cc] = int(marker_col)

    order = list(range(len(piece_arrays)))
    rng.shuffle(order)
    for idx in order:
        pa = piece_arrays[idx]
        try:
            r0, c0 = random_free_location(canvas, pa, transparent=bg, padding=1)
            paste(canvas, pa, int(r0), int(c0))
        except ValueError:
            return None

    tr, tc = _assembled_top_left(corner, mr, mc, RH, RW)
    expected_out = np.full((GH, GW), int(bg), dtype=int)
    expected_out[tr : tr + RH, tc : tc + RW] = rect

    try:
        actual_out = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual_out, expected_out):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
