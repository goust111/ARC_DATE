# Variant of 446ef5d2
# Dimension: D1 — 边缘匹配条件 (match_cond)
# Change: 相邻碎片的接触边上，非 rect_col 像素位置和颜色必须一致；rect_col 充当通配符可匹配任何颜色
#   -> 连接器匹配：每条内侧边只有一个专用 connector 像素需要对齐，其他接触边像素不再要求兼容。
#
# Design analysis for this variant
# =================================
# a) Concrete behavior / behavior trace
#    - Visual object model:
#      * Input still contains scattered rectangular pieces on a bg field plus a 3-cell L marker.
#      * The output is still one axis-aligned rectangle whose corner is pinned by the marker gap.
#      * The change is only about how two neighboring pieces are recognized:
#        every INTERNAL side of a piece has exactly one connector pixel, located away from corners,
#        and the touching piece has the same connector color at the same offset on the opposite side.
#      * Non-connector edge pixels are allowed to disagree. To make this visible, the generator adds
#        one deliberate mismatch per true seam using ordinary interior colors.
#    - Small trace:
#      * Old logic on a vertical seam compared two whole edge vectors with rect_col wildcard.
#        If both sides had [rect_col, blue, rect_col] and [rect_col, red, rect_col], the match failed.
#      * New logic only reads the dedicated connector pixel, e.g. row offset 1 with color yellow.
#        The blue/red mismatch elsewhere on the same seam is ignored, so the same two pieces now match.
#      * Therefore the changed behavior lands exactly in neighbor compatibility during assembly and in
#        input construction of each seam.
#
# b) Complexity calibration
#    - Using only one connector color would make the "color" part of the rule vacuous.
#      This variant uses two connector colors, repeated across many seams.
#    - Allowing arbitrary partial-edge contacts would require multiple connectors on one side and would
#      obscure the rule. The generator therefore uses a clean row/column grid partition so each side has
#      at most one neighbor and one connector.
#    - Pieces keep local interior structure so the puzzle is not reduced to plain colored blocks.
#
# c) Old implicit assumptions that break here
#    - Original compat(v1, v2): assumes whole-edge equality on overlapping non-rect_col positions.
#      This is false now because true seams intentionally contain mismatching non-connector pixels.
#    - Original neighbor_ok: assumes partial side contacts are acceptable and edge colors alone are enough.
#      In this variant that would blur connector semantics, so the solver now requires full-side contact
#      with exactly one opposite piece.
#    - Original generator inherits piece boundaries from one continuous rectangle pattern.
#      Here boundaries are edited after partition to inject connector pixels and deliberate mismatches.
#
# d) Placement legality
#    - Pieces are still scattered with >=1 bg padding and never overlap the marker.
#    - In the assembled rectangle, every outer border cell stays rect_col.
#    - Connectors are never placed on border-corner cells; they sit on safe interior edge positions.
#    - Every non-boundary side has exactly one connector, so the solver can require that property.
#
# e) Visual observability
#    - Across examples, humans can see small repeated "connector dots" on piece edges and also see that
#      some true neighboring edges do NOT form continuous color bands anymore. This makes the rule change
#      visible from a few demonstrations instead of hidden in implementation details.
#
# f) Visual cleanliness
#    - The partition is a tidy rectangular grid, pieces are separated when scattered, and connector colors
#      are reserved for edge-only dots. Ordinary interior colors may touch seams, but only as sparse
#      distractors, so the connector remains easy to spot.
#
# g) Color inference
#    - Connector colors are a tiny repeated palette and appear only on edges, never in piece interiors.
#      Ordinary interior colors appear inside pieces as well, so a human (and the solver) can distinguish
#      "edge-only connector colors" from general decoration colors.
#
# New solver assumptions introduced by this variant
# -------------------------------------------------
# - Final tiling is a full row/column grid: if two pieces touch, they share a whole side.
# - Every internal side has exactly one connector; every outer side has none and is all rect_col.
# - Connector colors are exactly the colors that appear on piece edges but nowhere in piece interiors.
# - The generator guarantees those assumptions before validating with main().

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
                missing = miss[0]
                bbox = (r0, c0, r1, c1)
                break
    assert not (mark is None or missing is None or bbox is None)

    r0, c0, _r1, c1 = bbox
    if missing == (r0, c0):
        corner = "BR"
    elif missing == (r0, c1):
        corner = "BL"
    elif missing == (_r1, c0):
        corner = "TR"
    else:
        corner = "TL"
    return int(mark), (int(missing[0]), int(missing[1])), corner


def _extract_rects(g: np.ndarray, bg: int, mark: int):
    GH, GW = g.shape
    occ = (g != int(bg)) & (g != int(mark))
    used = np.zeros_like(occ, dtype=bool)
    rects = []
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
        if idx:
            desc[side] = (idx[0], int(edge[idx[0]]))
        else:
            desc[side] = None
    return desc


def _collect_meta(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, missing, corner = _find_marker(g, bg)
    rects = _extract_rects(g, bg, mark)
    pieces = [a for _r, _c, _h, _w, a in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]

    allpix = []
    for a in pieces:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    edge_colors: set[int] = set()
    interior_colors: set[int] = set()
    for a in pieces:
        h, w = a.shape
        edge_colors.update(map(int, a[0, :]))
        edge_colors.update(map(int, a[-1, :]))
        edge_colors.update(map(int, a[:, 0]))
        edge_colors.update(map(int, a[:, -1]))
        if h > 2 and w > 2:
            interior_colors.update(map(int, a[1:-1, 1:-1].flatten()))
    edge_colors.discard(int(rect_col))
    interior_colors.discard(int(rect_col))
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
        "rect_col": rect_col,
        "connector_colors": connector_colors,
        "descs": descs,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into a rectangle at an L-marker corner.

    Variant rule:
    - Each internal side has exactly one connector pixel.
    - Two touching pieces match iff opposite sides have the same connector color at the same offset.
    - Other edge pixels are ignored for matching.
    """
    meta = _collect_meta(grid)
    g = meta["g"]
    GH, GW = meta["GH"], meta["GW"]
    bg = meta["bg"]
    missing = meta["missing"]
    corner = meta["corner"]
    pieces = meta["pieces"]
    ph = meta["ph"]
    pw = meta["pw"]
    rect_col = meta["rect_col"]
    descs = meta["descs"]

    total_area = int(sum(int(h) * int(w) for h, w in zip(ph, pw)))
    mr, mc = map(int, missing)

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

    def assemble(H: int, W: int):
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
                if not (jr == r and jh == h and jc + jw == c):
                    return -2
                return int(j)
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
                if not (jr == r and jh == h and c + w == jc):
                    return -2
                return int(j)
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
                if not (jc == c and jw == w and jr + jh == r):
                    return -2
                return int(j)
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
            if not (jc == c and jw == w and r + h == jr):
                return -2
            return int(j)

        def side_ok(i: int, side: str, r: int, c: int) -> bool:
            a = pieces[i]
            h, w = a.shape
            edge = np.array(_edge_values(a, side), dtype=int)
            desc = descs[i][side]
            if at_boundary(side, r, c, h, w):
                return desc is None and bool(np.all(edge == int(rect_col)))

            # In this variant all non-boundary sides must expose one connector.
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

        def next_empty():
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


def _grow_blob(mask: np.ndarray, target: int, rng: random.Random) -> set[tuple[int, int]]:
    h, w = mask.shape
    sr = rng.randint(0, h - 1)
    sc = rng.randint(0, w - 1)
    blob = {(sr, sc)}
    frontier = [(sr, sc)]
    while len(blob) < target and frontier:
        fr, fc = rng.choice(frontier)
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = fr + dr, fc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blob:
                nbrs.append((nr, nc))
        if not nbrs:
            frontier.remove((fr, fc))
            continue
        nr, nc = rng.choice(nbrs)
        blob.add((nr, nc))
        frontier.append((nr, nc))
    return blob


def _paint_piece(piece: np.ndarray, interior_cols: list[int], rng: random.Random) -> set[int]:
    h, w = piece.shape
    ih, iw = h - 2, w - 2
    used: set[int] = set()
    interior_mask = np.zeros((ih, iw), dtype=bool)
    n_blobs = rng.randint(1, min(2, len(interior_cols)))
    for _ in range(n_blobs):
        col = int(rng.choice(interior_cols))
        max_target = max(2, min(ih * iw, (ih * iw) // 3 + 1))
        target = rng.randint(2, max_target)
        blob = _grow_blob(interior_mask, target, rng)
        for br, bc in blob:
            piece[1 + br, 1 + bc] = int(col)
            interior_mask[br, bc] = True
            used.add(int(col))
    return used


def _safe_edge_positions(length: int) -> list[int]:
    return list(range(1, length - 1))


def _pick_two_positions(length: int, rng: random.Random) -> tuple[int, int]:
    pos = _safe_edge_positions(length)
    assert len(pos) >= 2
    rng.shuffle(pos)
    return int(pos[0]), int(pos[1])


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


def _seam_metric_from_rect(rect: np.ndarray, row_segs: list[int], col_segs: list[int], connector_cols: set[int]) -> dict[str, int]:
    mismatch_pairs = 0
    connector_pairs = 0
    r0 = 0
    for rs in row_segs:
        c0 = 0
        for cs in col_segs[:-1]:
            seam_left = rect[r0 : r0 + rs, c0 + cs - 1]
            seam_right = rect[r0 : r0 + rs, c0 + cs]
            for a, b in zip(seam_left.tolist(), seam_right.tolist()):
                if int(a) in connector_cols and int(b) in connector_cols and int(a) == int(b):
                    connector_pairs += 1
                elif int(a) != int(b):
                    mismatch_pairs += 1
            c0 += cs
        r0 += rs
    c0 = 0
    for cs in col_segs:
        r0 = 0
        for rs in row_segs[:-1]:
            seam_top = rect[r0 + rs - 1, c0 : c0 + cs]
            seam_bot = rect[r0 + rs, c0 : c0 + cs]
            for a, b in zip(seam_top.tolist(), seam_bot.tolist()):
                if int(a) in connector_cols and int(b) in connector_cols and int(a) == int(b):
                    connector_pairs += 1
                elif int(a) != int(b):
                    mismatch_pairs += 1
            r0 += rs
        c0 += cs
    return {"connector_pairs": int(connector_pairs), "mismatch_pairs": int(mismatch_pairs)}


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    palette = list(range(10))
    rng.shuffle(palette)
    bg = int(palette[0])
    rect_col = int(palette[1])
    marker_col = int(palette[2])
    connector_cols = [int(palette[3]), int(palette[4])]
    interior_cols = [int(palette[5]), int(palette[6])]

    nr = rng.randint(2, 3)
    nc = rng.randint(2, 3)
    row_pool = [4, 5, 6]
    col_pool = [4, 5, 6]
    rng.shuffle(row_pool)
    rng.shuffle(col_pool)
    row_segs = sorted(row_pool[:nr], reverse=rng.random() < 0.5)
    col_segs = sorted(col_pool[:nc], reverse=rng.random() < 0.5)
    RH = int(sum(row_segs))
    RW = int(sum(col_segs))

    GH_min = max(10, RH + 4)
    GW_min = max(10, RW + 4)
    if GH_min > 26 or GW_min > 26:
        return None

    GH = rng.randint(GH_min, 26)
    GW = rng.randint(GW_min, 26)
    if GH * GW < int(2.2 * RH * RW):
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

    pieces: list[list[np.ndarray]] = []
    used_interior: set[int] = set()
    for rs in row_segs:
        row = []
        for cs in col_segs:
            piece = np.full((rs, cs), int(rect_col), dtype=int)
            used_interior.update(_paint_piece(piece, interior_cols, rng))
            row.append(piece)
        pieces.append(row)

    for col in interior_cols:
        if int(col) not in used_interior:
            i = rng.randrange(nr)
            j = rng.randrange(nc)
            rs, cs = pieces[i][j].shape
            pieces[i][j][rs // 2, cs // 2] = int(col)
            used_interior.add(int(col))

    seam_count = 0
    for i in range(nr):
        for j in range(nc - 1):
            left = pieces[i][j]
            right = pieces[i][j + 1]
            rs = left.shape[0]
            conn_pos, mismatch_pos = _pick_two_positions(rs, rng)
            conn_col = int(connector_cols[(i + j) % len(connector_cols)])
            left[conn_pos, -1] = conn_col
            right[conn_pos, 0] = conn_col
            a_col = int(interior_cols[(i + j) % len(interior_cols)])
            b_col = int(interior_cols[(i + j + 1) % len(interior_cols)])
            if a_col == b_col:
                b_col = int(interior_cols[(i + j + 2) % len(interior_cols)])
            left[mismatch_pos, -1] = a_col
            right[mismatch_pos, 0] = b_col
            seam_count += 1

    for i in range(nr - 1):
        for j in range(nc):
            top = pieces[i][j]
            bot = pieces[i + 1][j]
            cs = top.shape[1]
            conn_pos, mismatch_pos = _pick_two_positions(cs, rng)
            conn_col = int(connector_cols[(i + j + 1) % len(connector_cols)])
            top[-1, conn_pos] = conn_col
            bot[0, conn_pos] = conn_col
            a_col = int(interior_cols[(i + j + 1) % len(interior_cols)])
            b_col = int(interior_cols[(i + j) % len(interior_cols)])
            if a_col == b_col:
                b_col = int(interior_cols[(i + j + 2) % len(interior_cols)])
            top[-1, mismatch_pos] = a_col
            bot[0, mismatch_pos] = b_col
            seam_count += 1

    rect = np.full((RH, RW), int(rect_col), dtype=int)
    piece_arrays = []
    row_starts = []
    acc = 0
    for rs in row_segs:
        row_starts.append(acc)
        acc += rs
    col_starts = []
    acc = 0
    for cs in col_segs:
        col_starts.append(acc)
        acc += cs

    for i, rs in enumerate(row_segs):
        for j, cs in enumerate(col_segs):
            pr, pc = row_starts[i], col_starts[j]
            rect[pr : pr + rs, pc : pc + cs] = pieces[i][j]
            piece_arrays.append(pieces[i][j].copy())

    piece_count = nr * nc
    if not (4 <= piece_count <= 12):
        return None

    all_piece_pixels = []
    for pa in piece_arrays:
        all_piece_pixels.extend(map(int, pa.flatten()))
    if Counter(all_piece_pixels).most_common(1)[0][0] != int(rect_col):
        return None

    if Counter(map(int, np.full((GH, GW), int(bg)).flatten())).most_common(1)[0][0] != int(bg):
        return None

    # Validate piece-side assumptions.
    connector_set = set(map(int, connector_cols))
    for i in range(nr):
        for j in range(nc):
            piece = pieces[i][j]
            h, w = piece.shape
            top = piece[0, :]
            bottom = piece[-1, :]
            left = piece[:, 0]
            right = piece[:, -1]
            if i == 0 and not np.all(top == int(rect_col)):
                return None
            if i == nr - 1 and not np.all(bottom == int(rect_col)):
                return None
            if j == 0 and not np.all(left == int(rect_col)):
                return None
            if j == nc - 1 and not np.all(right == int(rect_col)):
                return None
            side_desc = _connector_desc(piece, connector_set)
            expected = {
                "T": None if i == 0 else "present",
                "B": None if i == nr - 1 else "present",
                "L": None if j == 0 else "present",
                "R": None if j == nc - 1 else "present",
            }
            for side, need in expected.items():
                if need is None and side_desc[side] is not None:
                    return None
                if need == "present" and side_desc[side] is None:
                    return None
            non_rect = int(np.sum(piece != int(rect_col)))
            if non_rect < 3:
                return None

    metric = _seam_metric_from_rect(rect, row_segs, col_segs, connector_set)
    if metric["connector_pairs"] != seam_count:
        return None
    if metric["mismatch_pairs"] < seam_count:
        return None

    canvas = np.full((GH, GW), int(bg), dtype=int)
    for cr, cc in _marker_cells(corner, mr, mc):
        if cr < 0 or cr >= GH or cc < 0 or cc >= GW:
            return None
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
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
