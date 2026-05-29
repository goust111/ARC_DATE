# Variant of 446ef5d2
# Dimension: D2 — 源对象的形状与结构 (src)
# Change: 散布在背景上的矩形碎片，每块 2-7 格宽高，包含 rect_col 和 interior_col 像素，至少一条边含非... → 碎片尺寸统一

from __future__ import annotations

from collections import Counter
from itertools import product

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, paste, random_free_location


# Design analysis for the D2 variant
# ----------------------------------
# a) Concrete behavior:
#    The assembled object is still one axis-aligned rectangle pinned to the L-marker's
#    missing cell, and `rect_col` is still the wildcard/border color. The change is
#    that the rectangle is partitioned into a regular slot grid, e.g. 3x4 slots of
#    4x5 pieces, so every scattered fragment has the same shape. Position can no
#    longer be inferred from size. Instead, every internal shared edge carries a
#    color signature: the two corner cells of that edge stay `rect_col`, while the
#    middle cells use interior colors. Two adjacent pieces are compatible only when
#    these edge signatures match. Example: in the original task, a 2x6 strip and a
#    5x3 block might be easy to place due to unique size. In this variant, both are
#    impossible because every piece is, say, 4x4, so the solver must place pieces
#    by matching edge signatures and by checking which edges are blank outer-border
#    edges versus informative internal edges.
# b) Complexity calibration:
#    Piece sizes are restricted to 4-5 in each dimension and the generator always
#    uses 3 interior colors. This is deliberate: internal edge signatures keep
#    `rect_col` at the endpoints to avoid corner conflicts, leaving 2-3 middle cells
#    available for exact matching. With 3 colors that gives 9 or 27 distinct
#    signatures per orientation, enough for 4-12 same-size pieces without forcing
#    duplicate/ambiguous joins.
# c) Original implicit assumptions that break or become irrelevant:
#    - The original solver sorted/placed pieces partly via differing heights/widths.
#      That cue disappears, so the new solver first asserts that all extracted
#      rectangles share one shape and then searches over a slot grid of piece
#      positions.
#    - The original search filled the first empty pixel of the target rectangle.
#      With equal-size pieces that is weaker than searching over piece slots; a slot
#      search makes the new structural constraint explicit.
#    - The original generator split the rectangle into variable rectangles, so
#      "piece dimensions" were meaningful clues. Here the generator instead samples a
#      slot lattice and writes distinct edge signatures onto equal-size pieces.
# d) Placement legality:
#    Pieces are still rectangular connected components, scattered without overlap and
#    with at least one background cell of padding. The marker is placed first and the
#    assembled rectangle is chosen so it fits the canvas at the indicated corner.
# e) Visual observability:
#    Across examples, the user can see that all fragments are the same size while the
#    output remains a single rectangle. The only visible way to recover order is:
#    blank edges must face the outside border, and colored edge signatures must line
#    up. This makes the D2 change obvious in 2-3 examples.
# f) Visual cleanliness:
#    Edge signatures never fight at piece corners because signature endpoints are
#    always `rect_col`. That keeps each piece readable as a tidy rectangle instead of
#    a noisy overlap of row/column constraints.
# g) Color inference:
#    No numeric color ordering is used. Interior colors only participate as visible
#    equality classes along edges; `rect_col` remains identifiable as the border and
#    wildcard color because it dominates piece pixels and forms every outer edge.
# h) New solver assumptions and generator guarantees:
#    - Solver assumption: every scattered non-marker object is a filled rectangle.
#      Generator guarantee: every piece is emitted as a solid equal-size rectangle.
#    - Solver assumption: all pieces share one shape.
#      Generator guarantee: the rectangle is partitioned by a fixed slot size.
#    - Solver assumption: internal slot edges carry at least one non-rect pixel, so
#      blank-edge patterns identify border slots.
#      Generator guarantee: every internal boundary signature colors all middle edge
#      cells, while outer rectangle edges stay fully `rect_col`.


def _find_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], str]:
    cnt = Counter(map(int, g.flatten()))
    for col in sorted([c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3]):
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(col))]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0) != 1 or (c1 - c0) != 1:
            continue
        all4 = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        miss = list(all4 - set(pts))
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
        return int(col), (int(missing[0]), int(missing[1])), corner
    raise AssertionError("marker not found")


def _top_left_from_corner(mr: int, mc: int, H: int, W: int, corner: str) -> tuple[int, int]:
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - W + 1
    if corner == "BL":
        return mr - H + 1, mc
    return mr - H + 1, mc - W + 1


def _compat(v1: np.ndarray, v2: np.ndarray, rect_col: int) -> bool:
    m = (v1 != int(rect_col)) & (v2 != int(rect_col))
    return bool((v1[m] == v2[m]).all())


def _extract_pieces(g: np.ndarray, bg: int, marker_col: int) -> list[np.ndarray]:
    occ = ((g != int(bg)) & (g != int(marker_col))).astype(int)
    lbl, n = label(occ, structure=STRUCT4)
    pieces: list[np.ndarray] = []
    for idx in range(1, int(n) + 1):
        coords = np.argwhere(lbl == idx)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        block = lbl[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(block == idx):
            raise AssertionError("non-rectangular piece")
        pieces.append(g[r0 : r1 + 1, c0 : c1 + 1].copy())
    if not pieces:
        raise AssertionError("no pieces found")
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble equal-size scattered pieces into a rectangle anchored by the L-marker.

    Compared with the source task, piece geometry no longer helps: every piece has
    the same shape, so reconstruction is done on a piece-slot grid using blank outer
    edges plus color-signature matching along internal edges.
    """
    g = np.array(grid, dtype=int)
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    marker_col, (mr, mc), corner = _find_marker(g, bg)
    pieces = _extract_pieces(g, bg, marker_col)

    shape_set = {tuple(map(int, p.shape)) for p in pieces}
    assert len(shape_set) == 1
    ph, pw = next(iter(shape_set))

    allpix: list[int] = []
    for p in pieces:
        allpix.extend(map(int, p.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    n_pieces = len(pieces)
    dims: list[tuple[int, int]] = []
    for slot_r in range(1, n_pieces + 1):
        if n_pieces % slot_r != 0:
            continue
        slot_c = n_pieces // slot_r
        H = slot_r * ph
        W = slot_c * pw
        tr, tc = _top_left_from_corner(mr, mc, H, W, corner)
        if 0 <= tr and 0 <= tc and (tr + H) <= GH and (tc + W) <= GW:
            dims.append((slot_r, slot_c))
    dims.sort(key=lambda rc: (abs(rc[0] - rc[1]), -rc[0] * rc[1], -rc[0], -rc[1]))

    edge_info = []
    for p in pieces:
        top = p[0, :].copy()
        bottom = p[-1, :].copy()
        left = p[:, 0].copy()
        right = p[:, -1].copy()
        edge_info.append(
            {
                "arr": p,
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "top_blank": bool(np.all(top == int(rect_col))),
                "bottom_blank": bool(np.all(bottom == int(rect_col))),
                "left_blank": bool(np.all(left == int(rect_col))),
                "right_blank": bool(np.all(right == int(rect_col))),
            }
        )

    def assemble(slot_r: int, slot_c: int) -> np.ndarray | None:
        placed = [[-1 for _ in range(slot_c)] for _ in range(slot_r)]
        used = [False] * n_pieces

        def fits_piece(i: int, sr: int, sc: int) -> bool:
            info = edge_info[i]
            if info["top_blank"] != (sr == 0):
                return False
            if info["bottom_blank"] != (sr == slot_r - 1):
                return False
            if info["left_blank"] != (sc == 0):
                return False
            if info["right_blank"] != (sc == slot_c - 1):
                return False

            up = placed[sr - 1][sc] if sr > 0 else -1
            if up != -1 and not _compat(edge_info[up]["bottom"], info["top"], rect_col):
                return False
            down = placed[sr + 1][sc] if sr + 1 < slot_r else -1
            if down != -1 and not _compat(info["bottom"], edge_info[down]["top"], rect_col):
                return False
            left = placed[sr][sc - 1] if sc > 0 else -1
            if left != -1 and not _compat(edge_info[left]["right"], info["left"], rect_col):
                return False
            right = placed[sr][sc + 1] if sc + 1 < slot_c else -1
            if right != -1 and not _compat(info["right"], edge_info[right]["left"], rect_col):
                return False
            return True

        def choose_slot() -> tuple[int, int, list[int]] | None:
            best: tuple[int, int, list[int]] | None = None
            for sr in range(slot_r):
                for sc in range(slot_c):
                    if placed[sr][sc] != -1:
                        continue
                    cand = [i for i in range(n_pieces) if (not used[i]) and fits_piece(i, sr, sc)]
                    if not cand:
                        return sr, sc, []
                    if best is None or len(cand) < len(best[2]):
                        best = (sr, sc, cand)
            return best

        solved = False

        def bt() -> None:
            nonlocal solved
            choice = choose_slot()
            if choice is None:
                solved = True
                return
            sr, sc, cand = choice
            if not cand:
                return
            cand.sort(
                key=lambda i: (
                    -int(np.sum(edge_info[i]["top"] != rect_col)),
                    -int(np.sum(edge_info[i]["bottom"] != rect_col)),
                    -int(np.sum(edge_info[i]["left"] != rect_col)),
                    -int(np.sum(edge_info[i]["right"] != rect_col)),
                    i,
                )
            )
            for i in cand:
                used[i] = True
                placed[sr][sc] = i
                bt()
                if solved:
                    return
                placed[sr][sc] = -1
                used[i] = False

        bt()
        if not solved:
            return None

        ass = np.full((slot_r * ph, slot_c * pw), int(rect_col), dtype=int)
        for sr in range(slot_r):
            for sc in range(slot_c):
                idx = placed[sr][sc]
                ass[sr * ph : (sr + 1) * ph, sc * pw : (sc + 1) * pw] = edge_info[idx]["arr"]
        return ass

    assembled = None
    used_dims = None
    for slot_r, slot_c in dims:
        assembled = assemble(slot_r, slot_c)
        if assembled is not None:
            used_dims = (slot_r, slot_c)
            break

    assert assembled is not None and used_dims is not None
    H = used_dims[0] * ph
    W = used_dims[1] * pw
    tr, tc = _top_left_from_corner(mr, mc, H, W, corner)
    out = np.full((GH, GW), int(bg), dtype=int)
    out[tr : tr + H, tc : tc + W] = assembled
    return out


def _signature_pool(length: int, rect_col: int, colors: list[int]) -> list[np.ndarray]:
    mid = length - 2
    assert mid >= 2
    out = []
    for vals in product(colors, repeat=mid):
        out.append(np.array((rect_col, *vals, rect_col), dtype=int))
    return out


def _build_rect_from_piece_grid(
    piece_h: int,
    piece_w: int,
    slot_r: int,
    slot_c: int,
    rect_col: int,
    interior_cols: list[int],
    rng,
) -> tuple[np.ndarray, list[np.ndarray]]:
    pieces = [[np.full((piece_h, piece_w), rect_col, dtype=int) for _ in range(slot_c)] for _ in range(slot_r)]

    vertical_needed = slot_r * (slot_c - 1)
    horizontal_needed = (slot_r - 1) * slot_c
    v_pool = _signature_pool(piece_h, rect_col, interior_cols)
    h_pool = _signature_pool(piece_w, rect_col, interior_cols)
    if vertical_needed > len(v_pool) or horizontal_needed > len(h_pool):
        raise AssertionError("not enough signatures")
    rng.shuffle(v_pool)
    rng.shuffle(h_pool)

    v_it = iter(v_pool[:vertical_needed])
    h_it = iter(h_pool[:horizontal_needed])
    for sr in range(slot_r):
        for sc in range(slot_c - 1):
            sig = next(v_it)
            pieces[sr][sc][:, -1] = sig
            pieces[sr][sc + 1][:, 0] = sig
    for sr in range(slot_r - 1):
        for sc in range(slot_c):
            sig = next(h_it)
            pieces[sr][sc][-1, :] = sig
            pieces[sr + 1][sc][0, :] = sig

    piece_arrays = [pieces[sr][sc].copy() for sr in range(slot_r) for sc in range(slot_c)]
    seen = {tuple(map(int, p.flatten())) for p in piece_arrays}
    if len(seen) != len(piece_arrays):
        raise AssertionError("duplicate piece patterns")

    rect = np.full((slot_r * piece_h, slot_c * piece_w), rect_col, dtype=int)
    for sr in range(slot_r):
        for sc in range(slot_c):
            rect[sr * piece_h : (sr + 1) * piece_h, sc * piece_w : (sc + 1) * piece_w] = pieces[sr][sc]
    return rect, piece_arrays


def _gen_attempt(return_meta: bool = False):
    import random as _r

    rng = _r
    palette = list(range(10))
    rng.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    interior_cols = palette[3:6]

    piece_h = rng.randint(4, 5)
    piece_w = rng.randint(4, 5)
    slot_choices = [(r, c) for r in (2, 3) for c in (2, 3, 4) if 4 <= r * c <= 12]
    rng.shuffle(slot_choices)

    picked = None
    for slot_r, slot_c in slot_choices:
        RH = slot_r * piece_h
        RW = slot_c * piece_w
        if RH > 18 or RW > 18:
            continue
        if RH * RW > 220:
            continue
        picked = (slot_r, slot_c, RH, RW)
        break
    if picked is None:
        return None
    slot_r, slot_c, RH, RW = picked

    GH_min = max(10, RH + 4)
    GW_min = max(10, RW + 4)
    GH_max = min(26, RH + 10)
    GW_max = min(26, RW + 10)
    if GH_min > GH_max or GW_min > GW_max:
        return None
    GH = rng.randint(GH_min, GH_max)
    GW = rng.randint(GW_min, GW_max)

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

    try:
        rect, piece_arrays = _build_rect_from_piece_grid(
            piece_h, piece_w, slot_r, slot_c, rect_col, interior_cols, rng
        )
    except AssertionError:
        return None

    marker_cells: list[tuple[int, int]]
    if corner == "TL":
        marker_cells = [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    elif corner == "TR":
        marker_cells = [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    elif corner == "BL":
        marker_cells = [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    else:
        marker_cells = [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]
    for rr, cc in marker_cells:
        if not (0 <= rr < GH and 0 <= cc < GW):
            return None

    canvas = np.full((GH, GW), bg, dtype=int)
    for rr, cc in marker_cells:
        canvas[rr, cc] = marker_col

    order = list(range(len(piece_arrays)))
    rng.shuffle(order)
    for idx in order:
        pa = piece_arrays[idx]
        try:
            r0, c0 = random_free_location(canvas, pa, transparent=bg, padding=1)
            paste(canvas, pa, r0, c0)
        except ValueError:
            return None

    tr, tc = _top_left_from_corner(mr, mc, RH, RW, corner)
    expected = np.full((GH, GW), bg, dtype=int)
    expected[tr : tr + RH, tc : tc + RW] = rect

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    if return_meta:
        blank_patterns = []
        signal_edges = []
        for p in piece_arrays:
            blank_patterns.append(
                (
                    bool(np.all(p[0, :] == rect_col)),
                    bool(np.all(p[:, -1] == rect_col)),
                    bool(np.all(p[-1, :] == rect_col)),
                    bool(np.all(p[:, 0] == rect_col)),
                )
            )
            signal_edges.append(
                int(np.any(p[0, :] != rect_col))
                + int(np.any(p[:, -1] != rect_col))
                + int(np.any(p[-1, :] != rect_col))
                + int(np.any(p[:, 0] != rect_col))
            )
        return canvas, {
            "piece_shape": (piece_h, piece_w),
            "slot_shape": (slot_r, slot_c),
            "piece_count": len(piece_arrays),
            "blank_pattern_count": len(set(blank_patterns)),
            "min_signal_edges": min(signal_edges),
            "max_signal_edges": max(signal_edges),
        }
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt(return_meta=False)
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
