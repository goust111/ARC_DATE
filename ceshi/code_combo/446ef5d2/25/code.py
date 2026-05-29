# Combo variant of 446ef5d2 (index 25)
# combo_spec: D7/1+D11/1+D14/0
# Name: 有序排列大量小碎片行优先拼接

from __future__ import annotations

from collections import Counter
import random

import numpy as np


# Design analysis for the fused variant
# -------------------------------------
# a) Pixel-level concretization:
#    Original task flow:
#      build one rectangle -> cut into a handful of rectangles -> scatter them randomly
#      -> solver backtracks from a corner piece and tests many placements.
#    Fused flow:
#      build one rectangle -> cut it into a dense grid of 15/16/18/20 small rectangles
#      (heights 2-3, widths fixed at 3) -> place those pieces in the input as an ordered
#      catalog with the same row/column structure as the final rectangle.
#      The visible clue is therefore:
#        "input row 0 contains final row 0 pieces, input row 1 contains final row 1 pieces,
#         and within each input row the pieces already appear left-to-right."
#    Concrete toy trace:
#      If the final tiling is a 3x5 piece grid, the input also shows a 3x5 catalog.
#      Piece (0,0) is the first visible piece, piece (0,1) is immediately to its right,
#      ..., piece (2,4) is the last. Reconstruction is no longer "find any next piece";
#      it is "read row 0 left->right, then row 1 left->right, then row 2 left->right".
#
# b) Complexity calibration:
#    The piece count is raised to 15-20, which makes unconstrained search unnecessary and
#    visually distinct from the seed task. To keep the puzzle solvable by humans, the input
#    order is informative, but the solver still must use edge compatibility because the
#    marker only gives the output anchor, not the rectangle dimensions or pixel content.
#    Small pieces remain informative because each carries at least one boundary color cue.
#
# c) Implicit-assumption audit:
#    - The original anchor-piece heuristic (piece covering the marker corner in the input)
#      is invalid once pieces live in a separate ordered catalog. It is removed.
#    - The original factor-pair search over rectangle dimensions is also unnecessary: the
#      ordered catalog reveals the row partition and column partition directly.
#    - Rectangle extraction still assumes each visible object is a solid axis-aligned
#      rectangle. The generator preserves this by separating pieces with 1-cell gutters.
#    - Row-first assembly only checks already-placed neighbors (left and above). The
#      generator therefore plants matching cues exactly along row chains, plus one vertical
#      link from the start of each row to the row above, so the intended strategy is real.
#
# d) Placement legality:
#    The generator uses an output-first approach, arranges pieces in a non-overlapping
#    catalog block, and keeps the 3-cell marker away from that block. The output rectangle
#    still fits fully inside the grid at the marker-indicated corner.
#
# e) Visual observability:
#    Across a few examples a human can see three repeated facts:
#    - there are many more pieces than before,
#    - the pieces are tiny and displayed in a table-like order,
#    - reconstruction must proceed row by row because row starts are the only places with
#      a top-edge cue while the rest of each row chains horizontally.
#    No hidden numeric ordering such as color-id sorting is used.
#
# f) Visual cleanliness:
#    The input has two clean regions only: one ordered catalog of disjoint rectangles and
#    one L marker. No overlaps, no partial transparency, no mixed object types in the same
#    area.
#
# g) Color reasoning:
#    Colors keep their original roles. `rect_col` remains the wildcard/border color and is
#    still the most frequent color across piece pixels. One seam color carries most of the
#    edge matches; 1-2 accent colors may appear inside some 3x3 pieces, but they do not
#    alter the assembly rule.


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


def _extract_rectangles(g: np.ndarray, bg: int, mark: int) -> list[tuple[int, int, int, int, np.ndarray]]:
    gh, gw = g.shape
    occ = (g != int(bg)) & (g != int(mark))
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


def _compat(v1: np.ndarray, v2: np.ndarray, rect_col: int) -> bool:
    v1 = np.array(v1, dtype=int)
    v2 = np.array(v2, dtype=int)
    mask = (v1 != int(rect_col)) & (v2 != int(rect_col))
    return bool((v1[mask] == v2[mask]).all())


def _top_left_from_corner(corner: str, mr: int, mc: int, h: int, w: int) -> tuple[int, int]:
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - w + 1
    if corner == "BL":
        return mr - h + 1, mc
    return mr - h + 1, mc - w + 1


def _marker_cells(corner: str, mr: int, mc: int) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _catalog_metrics(rects: list[tuple[int, int, int, int, np.ndarray]]) -> dict[str, int]:
    row_vals = sorted({int(rr) for rr, _cc, _h, _w, _a in rects})
    col_vals = sorted({int(cc) for _rr, cc, _h, _w, _a in rects})
    piece_count = len(rects)
    complete = int(len(row_vals) * len(col_vals) == piece_count)
    small = int(all(2 <= int(h) <= 3 and 2 <= int(w) <= 3 for _r, _c, h, w, _a in rects))
    return {
        "piece_count": piece_count,
        "row_count": len(row_vals),
        "col_count": len(col_vals),
        "complete_catalog": complete,
        "all_small": small,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble an ordered catalog of many small fragments into one rectangle.

    Fusion-specific rule:
    - Input positions already encode final row/column order.
    - Pieces are placed strictly row-major.
    - When placing a piece, only the left neighbor and the above neighbor matter.
    """
    g = np.array(grid, dtype=int)
    gh, gw = g.shape

    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, missing, mb = _find_marker(g, bg)
    corner = _corner_from_missing(missing, mb)

    rects = _extract_rectangles(g, bg, mark)
    assert rects

    metrics = _catalog_metrics(rects)
    assert 13 <= metrics["piece_count"] <= 20
    assert metrics["complete_catalog"] == 1
    assert metrics["all_small"] == 1

    allpix: list[int] = []
    for _r, _c, _h, _w, arr in rects:
        allpix.extend(map(int, arr.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    row_vals = sorted({int(rr) for rr, _cc, _h, _w, _a in rects})
    col_vals = sorted({int(cc) for _rr, cc, _h, _w, _a in rects})
    row_rank = {v: i for i, v in enumerate(row_vals)}
    col_rank = {v: i for i, v in enumerate(col_vals)}

    n_rows = len(row_vals)
    n_cols = len(col_vals)
    catalog: dict[tuple[int, int], np.ndarray] = {}
    row_heights: list[int | None] = [None] * n_rows
    col_widths: list[int | None] = [None] * n_cols
    for rr, cc, h, w, arr in rects:
        key = (row_rank[int(rr)], col_rank[int(cc)])
        assert key not in catalog
        catalog[key] = arr
        if row_heights[key[0]] is None:
            row_heights[key[0]] = int(h)
        else:
            assert row_heights[key[0]] == int(h)
        if col_widths[key[1]] is None:
            col_widths[key[1]] = int(w)
        else:
            assert col_widths[key[1]] == int(w)

    assert all(v is not None for v in row_heights)
    assert all(v is not None for v in col_widths)
    row_heights2 = [int(v) for v in row_heights if v is not None]
    col_widths2 = [int(v) for v in col_widths if v is not None]

    row_pos: list[int] = []
    acc = 0
    for h in row_heights2:
        row_pos.append(acc)
        acc += h
    col_pos: list[int] = []
    acc = 0
    for w in col_widths2:
        col_pos.append(acc)
        acc += w

    h_total = sum(row_heights2)
    w_total = sum(col_widths2)
    mr, mc = map(int, missing)
    tr, tc = _top_left_from_corner(corner, mr, mc, h_total, w_total)
    assert 0 <= tr and 0 <= tc and tr + h_total <= gh and tc + w_total <= gw

    assembled = np.full((h_total, w_total), -1, dtype=int)

    for rr in range(n_rows):
        for cc in range(n_cols):
            piece = catalog[(rr, cc)]
            h, w = map(int, piece.shape)
            assert h == row_heights2[rr]
            assert w == col_widths2[cc]
            r0 = row_pos[rr]
            c0 = col_pos[cc]

            if rr == 0:
                assert np.all(piece[0, :] == int(rect_col))
            if rr == n_rows - 1:
                assert np.all(piece[-1, :] == int(rect_col))
            if cc == 0:
                assert np.all(piece[:, 0] == int(rect_col))
            if cc == n_cols - 1:
                assert np.all(piece[:, -1] == int(rect_col))

            if cc > 0:
                left = assembled[r0 : r0 + h, c0 - 1]
                assert (left != -1).all()
                assert _compat(left, piece[:, 0], rect_col)
            if rr > 0:
                up = assembled[r0 - 1, c0 : c0 + w]
                assert (up != -1).all()
                assert _compat(up, piece[0, :], rect_col)

            assembled[r0 : r0 + h, c0 : c0 + w] = piece

    out = np.full((gh, gw), int(bg), dtype=int)
    out[tr : tr + h_total, tc : tc + w_total] = assembled
    return out


def _pick_piece_grid() -> tuple[int, int]:
    return random.choice([(3, 5), (3, 6), (4, 4), (4, 5), (5, 3), (5, 4), (6, 3)])


def _build_rect_and_pieces(
    n_rows: int,
    n_cols: int,
    row_heights: list[int],
    col_widths: list[int],
    rect_col: int,
    seam_col: int,
    accent_cols: list[int],
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[np.ndarray]]:
    row_starts: list[int] = []
    acc = 0
    for h in row_heights:
        row_starts.append(acc)
        acc += h
    col_starts: list[int] = []
    acc = 0
    for w in col_widths:
        col_starts.append(acc)
        acc += w

    rh = sum(row_heights)
    rw = sum(col_widths)
    rect = np.full((rh, rw), rect_col, dtype=int)

    pieces_meta: list[tuple[int, int, int, int]] = []
    piece_arrays: list[np.ndarray] = []

    for rr in range(n_rows):
        for cc in range(n_cols):
            h = row_heights[rr]
            w = col_widths[cc]
            arr = np.full((h, w), rect_col, dtype=int)

            # Horizontal chain: every piece (except the last in a row) exports one cue to
            # its right neighbor; every non-first piece receives the matching cue on the left.
            row_idx = h - 1 if rr < n_rows - 1 else 0
            if cc > 0:
                arr[row_idx, 0] = seam_col
            if cc + 1 < n_cols:
                arr[row_idx, w - 1] = seam_col

            # Vertical row-link: only the first piece of each new row connects upward.
            if cc == 0 and rr > 0:
                arr[0, w - 1] = seam_col
            if cc == 0 and rr + 1 < n_rows:
                arr[h - 1, w - 1] = seam_col

            seam_count = int(np.sum(arr != rect_col))
            if h == 3 and accent_cols and seam_count <= 2 and random.random() < 0.55:
                arr[1, 1] = random.choice(accent_cols)

            pr = row_starts[rr]
            pc = col_starts[cc]
            rect[pr : pr + h, pc : pc + w] = arr
            pieces_meta.append((pr, pc, h, w))
            piece_arrays.append(arr.copy())

    return rect, pieces_meta, piece_arrays


def _gen_attempt() -> np.ndarray | None:
    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    seam_col = palette[3]
    accent_cols = palette[4 : 4 + random.randint(1, 2)]

    n_rows, n_cols = _pick_piece_grid()
    row_heights = [random.choice([2, 3]) for _ in range(n_rows)]
    if all(h == 3 for h in row_heights) and random.random() < 0.7:
        row_heights[random.randrange(n_rows)] = 2
    col_widths = [3] * n_cols

    rect, pieces_meta, piece_arrays = _build_rect_and_pieces(
        n_rows=n_rows,
        n_cols=n_cols,
        row_heights=row_heights,
        col_widths=col_widths,
        rect_col=rect_col,
        seam_col=seam_col,
        accent_cols=accent_cols,
    )

    rh, rw = rect.shape
    piece_count = len(piece_arrays)
    if not (13 <= piece_count <= 20):
        return None

    allpix = list(map(int, rect.flatten()))
    if Counter(allpix).most_common(1)[0][0] != rect_col:
        return None

    gap = 1
    ah = rh + gap * (n_rows - 1)
    aw = rw + gap * (n_cols - 1)

    base_h = max(10, ah + 2, rh + 2)
    base_w = max(10, aw + 2, rw + 2)
    if base_h > 26 or base_w > 26:
        return None
    gh = random.randint(base_h, 26)
    gw = random.randint(base_w, 26)

    row_pos: list[int] = []
    acc = 0
    for idx, h in enumerate(row_heights):
        row_pos.append(acc)
        acc += h + (gap if idx + 1 < n_rows else 0)
    col_pos: list[int] = []
    acc = 0
    for idx, w in enumerate(col_widths):
        col_pos.append(acc)
        acc += w + (gap if idx + 1 < n_cols else 0)

    arranged_block = np.full((ah, aw), bg, dtype=int)
    for idx, (pr, pc, h, w) in enumerate(pieces_meta):
        _ = pr, pc  # piece order already matches final row-major order
        r_slot = idx // n_cols
        c_slot = idx % n_cols
        r0 = row_pos[r_slot]
        c0 = col_pos[c_slot]
        arranged_block[r0 : r0 + h, c0 : c0 + w] = piece_arrays[idx]

    corner = random.choice(["TL", "TR", "BL", "BR"])
    if corner == "TL":
        mr_range = range(2, gh - rh + 1)
        mc_range = range(2, gw - rw + 1)
    elif corner == "TR":
        mr_range = range(2, gh - rh + 1)
        mc_range = range(rw - 1, gw - 2)
    elif corner == "BL":
        mr_range = range(rh - 1, gh - 2)
        mc_range = range(2, gw - rw + 1)
    else:
        mr_range = range(rh - 1, gh - 2)
        mc_range = range(rw - 1, gw - 2)

    marker_candidates = [(mr, mc) for mr in mr_range for mc in mc_range]
    random.shuffle(marker_candidates)
    arr_positions = [(r0, c0) for r0 in range(gh - ah + 1) for c0 in range(gw - aw + 1)]
    random.shuffle(arr_positions)

    chosen: tuple[int, int, int, int] | None = None
    for arr_r0, arr_c0 in arr_positions:
        arr_bbox = (arr_r0 - 1, arr_c0 - 1, arr_r0 + ah, arr_c0 + aw)
        for mr, mc in marker_candidates:
            cells = _marker_cells(corner, mr, mc)
            if any(r < 0 or r >= gh or c < 0 or c >= gw for r, c in cells):
                continue
            marker_bbox = (
                min(r for r, _c in cells),
                min(c for _r, c in cells),
                max(r for r, _c in cells),
                max(c for _r, c in cells),
            )
            if _bbox_intersects(arr_bbox, marker_bbox):
                continue
            chosen = (arr_r0, arr_c0, mr, mc)
            break
        if chosen is not None:
            break
    if chosen is None:
        return None

    arr_r0, arr_c0, mr, mc = chosen
    canvas = np.full((gh, gw), bg, dtype=int)
    canvas[arr_r0 : arr_r0 + ah, arr_c0 : arr_c0 + aw] = arranged_block
    for r, c in _marker_cells(corner, mr, mc):
        if canvas[r, c] != bg:
            return None
        canvas[r, c] = marker_col

    tr, tc = _top_left_from_corner(corner, mr, mc, rh, rw)
    expected_out = np.full((gh, gw), bg, dtype=int)
    expected_out[tr : tr + rh, tc : tc + rw] = rect

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
