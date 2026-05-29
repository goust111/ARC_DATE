# Combo variant of 446ef5d2 (index 24)
# combo_spec: D6/2+D7/3+D15/2
# Name: 碎片打包排列+填充率编码顺序+链式拼接
#
# Design analysis
# ===============
# Reference baseline:
# - Original solver: `/home/lixinyu/snap/new/arc2_solver_select/446ef5d2.py`
# - D6 packed-output reference: `/home/lixinyu/snap/new/ceshi/code/446ef5d2/D6/2/code.py`
# - Original task spec embedded in the prompt.
#
# a) Concretization and behavior trace
# - Original task:
#   scattered rectangular pieces + one 3-cell L-marker -> reconstruct one solid rectangle
#   at the marker corner.
# - This combo keeps the same input objects, but changes the target representation.
# - New intended reasoning chain:
#   1. infer the corner from the marker and recover `rect_col`,
#   2. infer which piece is first by lowest fill ratio,
#   3. greedily place pieces in strictly increasing fill-ratio order,
#      always attaching the next piece to the current chain frontier with a real seam clue,
#   4. output does not merge pieces back; it shows the ordered pieces in a compact
#      row-major packing with 1-cell gaps and a 1-cell outer border.
# - Small trace on a 2x3 latent piece grid:
#   original output would be one 10x15 rectangle.
#   combo output instead lists the 6 solved pieces in fill order
#   `p0,p1,p2 / p3,p4,p5`, separated by background gaps.
#   The solver still needs edge matching to recover the chain, but the final image is an
#   ordered catalogue of pieces rather than the merged rectangle.
#
# b) Complexity calibration
# - If fill ratio alone determined the answer, the task would collapse into pure sorting.
#   To avoid that, the generator also builds a real chain of seam clues: each consecutive
#   pair in the hidden assembly shares a colored seam, while non-consecutive contacts are
#   rect_col-only wildcards.
# - If we kept dense seam clues everywhere, the D15 signal would be visually drowned out.
#   Therefore each seam contributes only one visible clue cell, while the monotone fill
#   ladder is created by extra interior colored cells.
# - To make "fill ratio order" human-visible, all generated pieces have the same area and
#   strictly increasing colored-pixel counts. Humans can compare density directly instead
#   of having to estimate ratios across very different sizes.
#
# c) Hidden-assumption audit of the original `main()`
# - Original marker detection ("a non-bg color appearing exactly 3 times in a 2x2 L")
#   still holds.
# - Original rectangle extraction assumes scattered objects are axis-aligned solid
#   rectangles with >=1 background gap. Generator still guarantees that.
# - Original solver searched all piece placements with backtracking and returned the first
#   complete tiling. That assumption is deliberately broken: here `main()` assumes the
#   hidden assembly admits a chain order with strictly increasing fill ratios and no
#   backtracking is needed. `generate_input()` constructs exactly such cases.
# - Original output-writing logic assumed the solved rectangle should be pasted back into
#   the input-sized canvas. D6 invalidates that; final rendering is now a packed sequence.
# - New solver assumption:
#   the lowest-fill piece is the anchor piece at the marker corner, and each later piece
#   has a unique best greedy attachment with positive seam-clue overlap against the
#   immediately previous piece. Generator enforces both.
#
# d) Placement legality
# - Input generation is output-first in the sense that the hidden solved chain is built
#   first, then scattered with `padding=1`.
# - The marker is placed with the original corner mapping and bounded so the full hidden
#   rectangle would fit.
# - Packed output uses deterministic row/column cell sizes derived from piece shapes, so
#   there is no overlap and no out-of-bounds write.
#
# e) Visual observability
# - Across 2-4 examples a human can see:
#   1. outputs are not the original rectangle; they are separated packed pieces,
#   2. the packed reading order follows increasing fill density,
#   3. the pieces still preserve seam-colored edges, indicating they come from a hidden
#      chain assembly rather than arbitrary sorting.
# - No ordering step depends on raw color ids or other non-visual metadata.
#
# f) Visual cleanliness
# - Marker is only an input hint and disappears in the output.
# - Pieces remain intact rectangles and are separated by 1-cell background gutters, so the
#   packed output stays readable.
# - Extra fill-ratio cells are placed away from borders except the true chain seams, so
#   seam evidence is not smeared over the whole piece.
#
# g) Color inferability
# - There is no color remapping. `rect_col` remains the wildcard border color and the
#   interior colors simply encode visible clue pixels.
# - The generator uses only 2 interior colors, repeated structurally on chain seams and
#   interior bonus dots, so humans are not asked to infer arbitrary numeric color rules.
#
# h) Variant-specific metrics used after testing
# - `shape_changed = output.shape != input.shape`
# - `ordered_fills_strict = colored counts of packed pieces are strictly increasing`
# - `component_count = number of packed non-bg rectangles = number of input pieces`
# - `chain_clue_min =` for input pieces sorted by fill ratio, the minimum best seam-clue
#   score across consecutive pairs; this should stay positive if the D7 chain really
#   contributes and the task has not degenerated into plain density sorting.
# - Degeneration check:
#   if packed pieces were merged back or fill order ceased to matter, either
#   `shape_changed` or `ordered_fills_strict` would fail.
#
# i) New implicit assumptions after modification
# - All pieces in generated data have equal area; this makes fill-ratio ranking observable.
# - Consecutive chain pieces share exactly one seam clue cell; non-chain adjacencies are
#   fully wildcard. The greedy solver depends on that asymmetry.
# - The packed renderer assumes row-major display of the solved chain order. Generator
#   builds expected outputs with the same rule.

from __future__ import annotations

from collections import Counter
import math
import random

import numpy as np

from arc_utils import paste, random_free_location


def _find_marker(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int], str]:
    cnt = Counter(map(int, grid.flatten()))
    for col in sorted(c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3):
        pts = [(int(r), int(c)) for r, c in np.argwhere(grid == int(col))]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0) != 1 or (c1 - c0) != 1:
            continue
        all4 = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        missing = list(all4 - set(pts))
        if len(missing) != 1:
            continue
        miss = missing[0]
        if miss == (r0, c0):
            corner = "BR"
        elif miss == (r0, c1):
            corner = "BL"
        elif miss == (r1, c0):
            corner = "TR"
        else:
            corner = "TL"
        return int(col), (int(miss[0]), int(miss[1])), corner
    raise AssertionError("marker not found")


def _extract_rectangles(
    grid: np.ndarray,
    bg: int,
    ignored_color: int | None = None,
) -> list[tuple[int, int, int, int, np.ndarray]]:
    gh, gw = grid.shape
    occ = grid != int(bg)
    if ignored_color is not None:
        occ &= grid != int(ignored_color)
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
        rects.append((rr, cc, h, w, grid[rr : rr + h, cc : cc + w].copy()))
    return rects


def _compat(v1: np.ndarray, v2: np.ndarray, rect_col: int) -> bool:
    a = np.asarray(v1, dtype=int)
    b = np.asarray(v2, dtype=int)
    m = (a != int(rect_col)) & (b != int(rect_col))
    return bool((a[m] == b[m]).all())


def _clue_count(v1: np.ndarray, v2: np.ndarray, rect_col: int) -> int:
    a = np.asarray(v1, dtype=int)
    b = np.asarray(v2, dtype=int)
    m = (a != int(rect_col)) & (b != int(rect_col)) & (a == b)
    return int(np.sum(m))


def _non_rect_count(piece: np.ndarray, rect_col: int) -> int:
    return int(np.sum(piece != int(rect_col)))


def _fill_order(pieces: list[np.ndarray], rect_col: int) -> list[int]:
    numer = [_non_rect_count(piece, rect_col) for piece in pieces]
    denom = [int(piece.shape[0] * piece.shape[1]) for piece in pieces]
    return sorted(range(len(pieces)), key=lambda i: (numer[i] / denom[i], numer[i], denom[i], i))


def _anchor_pos(corner: str, h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - pw
    if corner == "BL":
        return h - ph, 0
    return h - ph, w - pw


def _pack_ordered_pieces(
    ordered_pieces: list[np.ndarray],
    bg: int,
    *,
    gap: int = 1,
    border: int = 1,
) -> np.ndarray:
    n = len(ordered_pieces)
    cols = max(2, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))

    row_heights = []
    for r in range(rows):
        start = r * cols
        stop = min((r + 1) * cols, n)
        row_heights.append(max(int(ordered_pieces[i].shape[0]) for i in range(start, stop)))

    col_widths = []
    for c in range(cols):
        present = [i for i in range(c, n, cols)]
        col_widths.append(max(int(ordered_pieces[i].shape[1]) for i in present))

    out_h = 2 * int(border) + sum(row_heights) + int(gap) * (rows - 1)
    out_w = 2 * int(border) + sum(col_widths) + int(gap) * (cols - 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    row_offsets = []
    cur = int(border)
    for h in row_heights:
        row_offsets.append(cur)
        cur += int(h) + int(gap)

    col_offsets = []
    cur = int(border)
    for w in col_widths:
        col_offsets.append(cur)
        cur += int(w) + int(gap)

    for idx, piece in enumerate(ordered_pieces):
        rr = idx // cols
        cc = idx % cols
        r0 = row_offsets[rr]
        c0 = col_offsets[cc]
        ph, pw = map(int, piece.shape)
        out[r0 : r0 + ph, c0 : c0 + pw] = piece
    return out


def _best_pair_clue(piece_a: np.ndarray, piece_b: np.ndarray, rect_col: int) -> int:
    best = 0
    ha, wa = map(int, piece_a.shape)
    hb, wb = map(int, piece_b.shape)
    if ha == hb:
        edges = [
            (piece_a[:, -1], piece_b[:, 0]),
            (piece_a[:, 0], piece_b[:, -1]),
        ]
        for left, right in edges:
            if _compat(left, right, rect_col):
                best = max(best, _clue_count(left, right, rect_col))
    if wa == wb:
        edges = [
            (piece_a[-1, :], piece_b[0, :]),
            (piece_a[0, :], piece_b[-1, :]),
        ]
        for top, bottom in edges:
            if _compat(top, bottom, rect_col):
                best = max(best, _clue_count(top, bottom, rect_col))
    return int(best)


def _solve_chain(
    grid: np.ndarray,
) -> tuple[int, int, list[np.ndarray], list[int], tuple[int, int]]:
    g = np.array(grid, dtype=int)
    gh, gw = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, missing, corner = _find_marker(g, bg)

    rects = _extract_rectangles(g, bg, ignored_color=mark)
    assert rects
    pieces = [arr for _r, _c, _h, _w, arr in rects]
    total_area = int(sum(arr.shape[0] * arr.shape[1] for arr in pieces))

    allpix: list[int] = []
    for piece in pieces:
        allpix.extend(map(int, piece.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    order = _fill_order(pieces, rect_col)
    anchor_idx = order[0]
    mr, mc = missing

    dims: list[tuple[int, int]] = []
    for hh in range(1, int(math.isqrt(total_area)) + 1):
        if total_area % hh != 0:
            continue
        dims.append((hh, total_area // hh))
        if hh * hh != total_area:
            dims.append((total_area // hh, hh))

    def fits(hh: int, ww: int) -> bool:
        if corner == "TL":
            tr, tc = mr, mc
        elif corner == "TR":
            tr, tc = mr, mc - ww + 1
        elif corner == "BL":
            tr, tc = mr - hh + 1, mc
        else:
            tr, tc = mr - hh + 1, mc - ww + 1
        return 0 <= tr and 0 <= tc and tr + hh <= gh and tc + ww <= gw

    dims = [(int(hh), int(ww)) for hh, ww in dims if fits(int(hh), int(ww))]
    dims.sort(key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    def boundary_ok(piece: np.ndarray, r: int, c: int, hh: int, ww: int) -> bool:
        ph, pw = piece.shape
        if r == 0 and not np.all(piece[0, :] == int(rect_col)):
            return False
        if r + ph == hh and not np.all(piece[-1, :] == int(rect_col)):
            return False
        if c == 0 and not np.all(piece[:, 0] == int(rect_col)):
            return False
        if c + pw == ww and not np.all(piece[:, -1] == int(rect_col)):
            return False
        return True

    def try_dim(hh: int, ww: int) -> list[tuple[int, int]] | None:
        occ = np.full((hh, ww), -1, dtype=int)
        pid = np.full((hh, ww), -1, dtype=int)
        placements: list[tuple[int, int] | None] = [None] * len(pieces)

        def place(piece_idx: int, r: int, c: int) -> None:
            piece = pieces[piece_idx]
            ph, pw = piece.shape
            occ[r : r + ph, c : c + pw] = piece
            pid[r : r + ph, c : c + pw] = int(piece_idx)
            placements[piece_idx] = (int(r), int(c))

        apiece = pieces[anchor_idx]
        aph, apw = apiece.shape
        ar, ac = _anchor_pos(corner, hh, ww, int(aph), int(apw))
        if not boundary_ok(apiece, ar, ac, hh, ww):
            return None
        place(anchor_idx, int(ar), int(ac))
        last_idx = anchor_idx

        for piece_idx in order[1:]:
            piece = pieces[piece_idx]
            ph, pw = piece.shape
            best: tuple[int, int, int, int, int] | None = None
            best_pos: tuple[int, int] | None = None

            for r in range(hh - ph + 1):
                for c in range(ww - pw + 1):
                    if (occ[r : r + ph, c : c + pw] != -1).any():
                        continue
                    if not boundary_ok(piece, r, c, hh, ww):
                        continue

                    total_touch = 0
                    total_clue = 0
                    last_touch = 0
                    last_clue = 0
                    ok = True

                    def scan(existing_cols: np.ndarray, existing_ids: np.ndarray, new_cols: np.ndarray) -> None:
                        nonlocal total_touch, total_clue, last_touch, last_clue, ok
                        m = existing_cols != -1
                        if (not ok) or (not m.any()):
                            return
                        if not _compat(existing_cols[m], new_cols[m], rect_col):
                            ok = False
                            return
                        total_touch += int(np.sum(m))
                        total_clue += _clue_count(existing_cols[m], new_cols[m], rect_col)
                        last_mask = m & (existing_ids == int(last_idx))
                        if last_mask.any():
                            last_touch += int(np.sum(last_mask))
                            last_clue += _clue_count(existing_cols[last_mask], new_cols[last_mask], rect_col)

                    if c > 0:
                        scan(occ[r : r + ph, c - 1], pid[r : r + ph, c - 1], piece[:, 0])
                    if c + pw < ww:
                        scan(occ[r : r + ph, c + pw], pid[r : r + ph, c + pw], piece[:, -1])
                    if r > 0:
                        scan(occ[r - 1, c : c + pw], pid[r - 1, c : c + pw], piece[0, :])
                    if r + ph < hh:
                        scan(occ[r + ph, c : c + pw], pid[r + ph, c : c + pw], piece[-1, :])
                    if (not ok) or total_touch == 0 or last_clue <= 0:
                        continue

                    score = (int(last_clue), int(total_clue), int(last_touch), int(total_touch), -int(r), -int(c))
                    if best is None or score > best:
                        best = score
                        best_pos = (int(r), int(c))

            if best_pos is None or best is None:
                return None
            place(piece_idx, int(best_pos[0]), int(best_pos[1]))
            last_idx = int(piece_idx)

        if (occ == -1).any():
            return None

        final_places: list[tuple[int, int]] = []
        for pos in placements:
            if pos is None:
                return None
            final_places.append((int(pos[0]), int(pos[1])))
        return final_places

    solved_places = None
    solved_dim = None
    for hh, ww in dims:
        placements = try_dim(int(hh), int(ww))
        if placements is not None:
            solved_places = placements
            solved_dim = (int(hh), int(ww))
            break

    assert solved_places is not None and solved_dim is not None
    return int(bg), int(rect_col), pieces, order, solved_dim


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve the hidden rectangle by fill-ratio-guided chain assembly, then display the
    ordered pieces in a compact packed grid.
    """
    bg, _rect_col, pieces, order, _dim = _solve_chain(grid)
    ordered_pieces = [pieces[i] for i in order]
    return _pack_ordered_pieces(ordered_pieces, bg, gap=1, border=1)


def _snake_order(nr: int, nc: int, corner: str) -> list[tuple[int, int]]:
    if corner in ("TL", "TR"):
        rows = list(range(nr))
    else:
        rows = list(range(nr - 1, -1, -1))
    left_start = corner in ("TL", "BL")

    order: list[tuple[int, int]] = []
    for ridx, r in enumerate(rows):
        if left_start:
            cols = list(range(nc)) if ridx % 2 == 0 else list(range(nc - 1, -1, -1))
        else:
            cols = list(range(nc - 1, -1, -1)) if ridx % 2 == 0 else list(range(nc))
        for c in cols:
            order.append((int(r), int(c)))
    return order


def _marker_cells(corner: str, mr: int, mc: int) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _gen_attempt() -> np.ndarray | None:
    rnd = random.Random()

    palette = list(range(10))
    rnd.shuffle(palette)
    bg = int(palette[0])
    rect_col = int(palette[1])
    marker_col = int(palette[2])
    interior_cols = [int(palette[3]), int(palette[4])]

    corner = rnd.choice(["TL", "TR", "BL", "BR"])
    nr, nc = rnd.choice([(2, 2), (2, 3), (3, 2), (3, 3)])
    ph = rnd.randint(5, 6)
    pw = rnd.randint(5, 6)
    piece_area = int(ph * pw)
    n_pieces = int(nr * nc)

    min_fill = 3
    max_fill = min(piece_area - 2, int(math.floor(0.52 * piece_area)))
    if max_fill - min_fill + 1 < n_pieces:
        return None
    fill_counts = [min_fill + i for i in range(n_pieces)]

    rh = int(nr * ph)
    rw = int(nc * pw)
    min_gh = max(rh + 4, int(math.ceil(math.sqrt(2 * piece_area * n_pieces + 40))))
    min_gw = max(rw + 4, int(math.ceil(math.sqrt(2 * piece_area * n_pieces + 40))))
    if min_gh > 26 or min_gw > 26:
        return None
    gh = rnd.randint(min_gh, 26)
    gw = rnd.randint(min_gw, 26)
    if gh * gw <= 2 * piece_area * n_pieces + 20:
        return None

    if corner == "TL":
        mr_min, mr_max = 2, gh - rh
        mc_min, mc_max = 2, gw - rw
    elif corner == "TR":
        mr_min, mr_max = 2, gh - rh
        mc_min, mc_max = rw - 1, gw - 3
    elif corner == "BL":
        mr_min, mr_max = rh - 1, gh - 3
        mc_min, mc_max = 2, gw - rw
    else:
        mr_min, mr_max = rh - 1, gh - 3
        mc_min, mc_max = rw - 1, gw - 3
    if mr_min > mr_max or mc_min > mc_max:
        return None
    mr = rnd.randint(mr_min, mr_max)
    mc = rnd.randint(mc_min, mc_max)

    pieces = [np.full((ph, pw), rect_col, dtype=int) for _ in range(n_pieces)]
    order = _snake_order(nr, nc, corner)
    order_index = {cell: i for i, cell in enumerate(order)}

    seam_used: list[set[tuple[int, int]]] = [set() for _ in range(n_pieces)]

    for step in range(n_pieces - 1):
        ar, ac = order[step]
        br, bc = order[step + 1]
        a_idx = int(order_index[(ar, ac)])
        b_idx = int(order_index[(br, bc)])
        seam_color = int(interior_cols[step % len(interior_cols)])

        if ar == br:
            edge_row = 1 + (step % (ph - 2))
            if bc == ac + 1:
                pieces[a_idx][edge_row, pw - 1] = seam_color
                pieces[b_idx][edge_row, 0] = seam_color
                seam_used[a_idx].add((edge_row, pw - 1))
                seam_used[b_idx].add((edge_row, 0))
            else:
                pieces[a_idx][edge_row, 0] = seam_color
                pieces[b_idx][edge_row, pw - 1] = seam_color
                seam_used[a_idx].add((edge_row, 0))
                seam_used[b_idx].add((edge_row, pw - 1))
        else:
            edge_col = 1 + (step % (pw - 2))
            if br == ar + 1:
                pieces[a_idx][ph - 1, edge_col] = seam_color
                pieces[b_idx][0, edge_col] = seam_color
                seam_used[a_idx].add((ph - 1, edge_col))
                seam_used[b_idx].add((0, edge_col))
            else:
                pieces[a_idx][0, edge_col] = seam_color
                pieces[b_idx][ph - 1, edge_col] = seam_color
                seam_used[a_idx].add((0, edge_col))
                seam_used[b_idx].add((ph - 1, edge_col))

    interior_coords = [(r, c) for r in range(1, ph - 1) for c in range(1, pw - 1)]
    for idx, piece in enumerate(pieces):
        current = _non_rect_count(piece, rect_col)
        need = int(fill_counts[idx] - current)
        if need < 0:
            return None
        available = [(r, c) for r, c in interior_coords if piece[r, c] == int(rect_col)]
        if need > len(available):
            return None
        for k in range(need):
            rr, cc = available[k]
            piece[rr, cc] = int(interior_cols[(idx + k) % len(interior_cols)])

    for idx, piece in enumerate(pieces):
        fill = _non_rect_count(piece, rect_col) / piece_area
        if fill < 0.10 or fill > 0.60:
            return None
        if not any(np.any(edge != int(rect_col)) for edge in [piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]]):
            return None

    canvas = np.full((gh, gw), bg, dtype=int)
    for cr, cc in _marker_cells(corner, mr, mc):
        if cr < 0 or cr >= gh or cc < 0 or cc >= gw:
            return None
        canvas[cr, cc] = marker_col

    scatter_order = list(range(n_pieces))
    rnd.shuffle(scatter_order)
    for idx in scatter_order:
        piece = pieces[idx]
        try:
            r0, c0 = random_free_location(canvas, piece, transparent=bg, padding=1)
        except ValueError:
            return None
        paste(canvas, piece, int(r0), int(c0))

    expected = _pack_ordered_pieces(pieces, bg, gap=1, border=1)
    if expected.shape == canvas.shape:
        return None

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    metrics = variant_metrics(canvas)
    if (not metrics["shape_changed"]) or (not metrics["ordered_fills_strict"]) or metrics["component_count"] != n_pieces:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")


def variant_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    inp = np.asarray(grid, dtype=int)
    in_bg = int(Counter(map(int, inp.flatten())).most_common(1)[0][0])
    mark, _missing, _corner = _find_marker(inp, in_bg)
    in_rects = _extract_rectangles(inp, in_bg, ignored_color=mark)
    in_pieces = [arr for _r, _c, _h, _w, arr in in_rects]
    input_allpix: list[int] = []
    for arr in in_pieces:
        input_allpix.extend(map(int, arr.flatten()))
    input_rect_col = int(Counter(input_allpix).most_common(1)[0][0])
    input_order = _fill_order(in_pieces, input_rect_col)
    chain_clues = [
        _best_pair_clue(in_pieces[input_order[i]], in_pieces[input_order[i + 1]], input_rect_col)
        for i in range(len(input_order) - 1)
    ]

    out = main(inp)
    rects = _extract_rectangles(out, in_bg, ignored_color=None)
    allpix: list[int] = []
    for _r, _c, _h, _w, arr in rects:
        allpix.extend(map(int, arr.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])
    ordered = sorted(rects, key=lambda t: (int(t[0]), int(t[1])))
    fills = [_non_rect_count(arr, rect_col) for _r, _c, _h, _w, arr in ordered]
    return {
        "shape_changed": bool(out.shape != grid.shape),
        "ordered_fills_strict": bool(all(a < b for a, b in zip(fills, fills[1:]))),
        "component_count": int(len(rects)),
        "fill_span": int(fills[-1] - fills[0]) if fills else 0,
        "chain_clue_min": int(min(chain_clues)) if chain_clues else 0,
        "chain_clue_sum": int(sum(chain_clues)),
    }
