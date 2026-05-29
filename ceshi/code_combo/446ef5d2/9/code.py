# Combo variant of 446ef5d2 (index 9)
# combo_spec: D10/0+D16/0
# Name: 取消外边框约束配合无锚定碎片

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import paste


"""
design_analysis:
  combo_core:
    - D10 removes the old "assembled outer border is all rect_col" rule.
      So a solver can no longer classify an edge as exterior by checking whether
      it is pure rect_col.
    - D16 removes the old anchor-piece privilege.
      So a solver can no longer identify a unique starting fragment from the
      piece currently touching the marker gap.
    - The fused task should therefore behave like a pure edge-matching search:
      the marker only gives the rectangle corner position in the canvas, while
      the piece arrangement itself must be recovered from compatible edge colors.

  concretization_and_trace:
    - Original task:
      1. Detect marker and missing cell.
      2. Use the piece currently containing that corner cell as the anchor.
      3. Use "outer border must be all rect_col" to prune placements hard.
    - Fused task:
      1. Detect marker and missing cell.
      2. Ignore current piece-marker proximity entirely; generator leaves a moat
         around the marker so no piece acts as an implicit anchor.
      3. Search for a tiling using only edge compatibility plus rectangle size.
         Outer edges may contain accent pixels, so boundary placement cannot use
         the old all-rect_col test.
    - Small example:
      - Suppose the assembled rectangle is a 3x3 grid of pieces.
      - In the original task, the corner piece is obvious from marker contact,
        and any piece placed on the outer top edge must have a pure rect_col top.
      - In the fused task, the marker stands alone, several pieces can locally
        serve as the marked corner candidate, and top edges also carry accent
        signatures. The solver must try placements and verify which edges can be
        completed by mates and which arrangement tiles the rectangle exactly.

  complexity_calibration:
    - To keep the task solvable after removing both clues, the generator uses a
      regular 2-4 by 2-4 rectangular grid partition with no merges.
    - This creates a strong but still visual structure:
      every internal side mates with exactly one full opposite side of equal
      length, while outer sides also carry signatures and are not globally
      recognizable by color purity.
    - Row heights and column widths are chosen distinct within each sample.
      This makes the full assembly stable without introducing hidden numeric
      ordering rules.

  implicit_assumption_review:
    - Old assumption broken:
      `boundary_ok(... np.all(edge == rect_col))` is invalid under D10 and must
      be removed.
    - Old assumption broken:
      pre-placing an anchor piece from marker geometry is invalid under D16 and
      must be removed.
    - New solver assumption:
      internal adjacencies are full-edge matches between equal-length sides.
      This is guaranteed by the generator's regular grid partition.
    - New solver assumption:
      if a side is not on the rectangle boundary, it needs a full-edge mate.
      The generator guarantees that each internal side has exactly one true mate,
      but many outer sides also have decoy-compatible signatures so "no mate =>
      outer" is no longer available.

  placement_legality:
    - Pieces are scattered with a 1-cell background gap.
    - The marker keeps a 1-cell moat around its 2x2 box, so no piece overlaps or
      visually hugs it.
    - The assembled rectangle always fits fully inside the grid and the output is
      background everywhere else.

  visual_observability:
    - The outer border of the final rectangle visibly contains accent pixels on
      all four sides, so the D10 change is directly observable.
    - The marker sits isolated in background, so the D16 change is also visible:
      the marker marks location only, not a specific fragment.
    - Because multiple pieces are locally compatible with the marked corner
      directions, the input does not reveal a unique start piece.

  visual_cleanliness:
    - Edge accents are placed at two fixed non-corner cells per side, avoiding
      corner overlaps.
    - Interior decoration stays sparse and inside piece interiors.
    - A marker moat prevents clutter near the only explicit locator.

  color_inference:
    - Colors keep simple roles: bg, marker, rect_col wildcard, and three accent
      colors reused on edges and sparse interiors.
    - No invisible ordering by color id is used; only visible edge agreement and
      rectangle tiling matter.
"""


def _find_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], str, tuple[int, int, int, int]]:
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
        if len(miss) != 1:
            continue
        missing = tuple(map(int, miss[0]))
        if missing == (r0, c0):
            corner = "BR"
        elif missing == (r0, c1):
            corner = "BL"
        elif missing == (r1, c0):
            corner = "TR"
        else:
            corner = "TL"
        return int(col), missing, corner, (r0, c0, r1, c1)
    raise AssertionError("marker not found")


def _extract_rects(g: np.ndarray, occ: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
    gh, gw = g.shape
    used = np.zeros_like(occ, dtype=bool)
    rects: list[tuple[int, int, int, int, np.ndarray]] = []
    while True:
        idx = np.argwhere(occ & (~used))
        if idx.size == 0:
            return rects
        rr, cc = map(int, idx[0])
        w = 0
        while cc + w < gw and bool(occ[rr, cc + w]) and (not bool(used[rr, cc + w])):
            w += 1
        h = 0
        while rr + h < gh and bool((occ[rr + h, cc : cc + w] & (~used[rr + h, cc : cc + w])).all()):
            h += 1
        used[rr : rr + h, cc : cc + w] = True
        rects.append((rr, cc, h, w, g[rr : rr + h, cc : cc + w].copy()))


def _top_left_from_corner_cell(corner: str, mr: int, mc: int, H: int, W: int) -> tuple[int, int]:
    if corner == "TL":
        return int(mr), int(mc)
    if corner == "TR":
        return int(mr), int(mc - W + 1)
    if corner == "BL":
        return int(mr - H + 1), int(mc)
    return int(mr - H + 1), int(mc - W + 1)


def _marker_cells_from_corner(corner: str, mr: int, mc: int) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _compat(v1: np.ndarray, v2: np.ndarray, rect_col: int) -> bool:
    a = np.array(v1, dtype=int)
    b = np.array(v2, dtype=int)
    m = (a != int(rect_col)) & (b != int(rect_col))
    return bool((a[m] == b[m]).all())


def _build_mate_lists(
    pieces: list[np.ndarray], rect_col: int
) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]]:
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]
    top = [[] for _ in pieces]
    bottom = [[] for _ in pieces]
    left = [[] for _ in pieces]
    right = [[] for _ in pieces]
    for i, a in enumerate(pieces):
        for j, b in enumerate(pieces):
            if i == j:
                continue
            if pw[i] == pw[j]:
                if _compat(a[0, :], b[-1, :], rect_col):
                    top[i].append(j)
                if _compat(a[-1, :], b[0, :], rect_col):
                    bottom[i].append(j)
            if ph[i] == ph[j]:
                if _compat(a[:, 0], b[:, -1], rect_col):
                    left[i].append(j)
                if _compat(a[:, -1], b[:, 0], rect_col):
                    right[i].append(j)
    return top, bottom, left, right


def _sig_positions(length: int) -> tuple[int, int]:
    return 1, length - 2


def _paint_h_edge(arr: np.ndarray, row: int, c0: int, width: int, code: tuple[int, int]) -> None:
    p0, p1 = _sig_positions(width)
    arr[row, c0 + p0] = int(code[0])
    arr[row, c0 + p1] = int(code[1])


def _paint_v_edge(arr: np.ndarray, r0: int, col: int, height: int, code: tuple[int, int]) -> None:
    p0, p1 = _sig_positions(height)
    arr[r0 + p0, col] = int(code[0])
    arr[r0 + p1, col] = int(code[1])


def _decorate_piece_interior(
    rect: np.ndarray,
    piece: tuple[int, int, int, int],
    accent_colors: list[int],
    rng: random.Random,
) -> None:
    pr, pc, ph, pw = piece
    cells = [(r, c) for r in range(pr + 1, pr + ph - 1) for c in range(pc + 1, pc + pw - 1)]
    if not cells:
        return
    rng.shuffle(cells)
    color = int(rng.choice(accent_colors))
    rect[cells[0]] = color
    if len(cells) >= 4 and rng.random() < 0.7:
        rect[cells[1]] = color
    if len(cells) >= 9 and rng.random() < 0.35:
        rect[cells[2]] = int(rng.choice(accent_colors))


def _rand_unique_segments(total_max: int, count: int, lo: int, hi: int, rng: random.Random) -> list[int] | None:
    vals = list(range(lo, hi + 1))
    if count > len(vals):
        return None
    for _ in range(120):
        segs = rng.sample(vals, count)
        rng.shuffle(segs)
        if sum(segs) <= total_max:
            return segs
    return None


def _random_free_location_with_forbidden(
    canvas: np.ndarray,
    patch: np.ndarray,
    *,
    bg: int,
    forbidden: np.ndarray,
    rng: random.Random,
    padding: int = 1,
) -> tuple[int, int]:
    H, W = canvas.shape
    h, w = patch.shape
    occ = canvas != int(bg)
    candidates: list[tuple[int, int]] = []
    for r0 in range(0, H - h + 1):
        for c0 in range(0, W - w + 1):
            rr0 = max(0, r0 - padding)
            rr1 = min(H, r0 + h + padding)
            cc0 = max(0, c0 - padding)
            cc1 = min(W, c0 + w + padding)
            if bool(occ[rr0:rr1, cc0:cc1].any()):
                continue
            if bool(forbidden[r0 : r0 + h, c0 : c0 + w].any()):
                continue
            candidates.append((r0, c0))
    if not candidates:
        raise ValueError("no free location with forbidden zone")
    return rng.choice(candidates)


def _count_marked_corner_candidates(
    pieces: list[np.ndarray], rect_col: int, corner: str, n_rows: int, n_cols: int
) -> int:
    top, bottom, left, right = _build_mate_lists(pieces, rect_col)
    need_top = n_rows > 1 and corner in ("BL", "BR")
    need_bottom = n_rows > 1 and corner in ("TL", "TR")
    need_left = n_cols > 1 and corner in ("TR", "BR")
    need_right = n_cols > 1 and corner in ("TL", "BL")
    total = 0
    for i in range(len(pieces)):
        if need_top and not top[i]:
            continue
        if need_bottom and not bottom[i]:
            continue
        if need_left and not left[i]:
            continue
        if need_right and not right[i]:
            continue
        total += 1
    return total


def _forced_outer_hits(
    pieces: list[np.ndarray], rect_col: int, row_of_piece: list[int], col_of_piece: list[int], n_rows: int, n_cols: int
) -> int:
    top, bottom, left, right = _build_mate_lists(pieces, rect_col)
    hits = 0
    for i in range(len(pieces)):
        if row_of_piece[i] == 0 and not top[i]:
            hits += 1
        if row_of_piece[i] == n_rows - 1 and not bottom[i]:
            hits += 1
        if col_of_piece[i] == 0 and not left[i]:
            hits += 1
        if col_of_piece[i] == n_cols - 1 and not right[i]:
            hits += 1
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into a rectangle at an L-marker corner.

    Combo behavior:
    - Outer edges may contain non-rect_col pixels.
    - No scattered piece is privileged by current location near the marker.
    - The solver therefore performs pure edge-compatible rectangle tiling.
    """
    g = np.array(grid, dtype=int)
    GH, GW = g.shape

    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, missing, corner, _mb = _find_marker(g, bg)

    occ = (g != int(bg)) & (g != int(mark))
    rects = _extract_rects(g, occ)
    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))

    allpix: list[int] = []
    for _r, _c, _h, _w, a in rects:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    pieces = [a for _r, _c, _h, _w, a in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]
    area = [ph[i] * pw[i] for i in range(len(pieces))]
    top_mates, bottom_mates, left_mates, right_mates = _build_mate_lists(pieces, rect_col)
    mate_score = [len(top_mates[i]) + len(bottom_mates[i]) + len(left_mates[i]) + len(right_mates[i]) for i in range(len(pieces))]

    mr, mc = map(int, missing)
    dims: list[tuple[int, int]] = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh == 0:
            dims.append((hh, total_area // hh))
            if hh * hh != total_area:
                dims.append((total_area // hh, hh))

    def fits(H: int, W: int) -> bool:
        tr, tc = _top_left_from_corner_cell(corner, mr, mc, H, W)
        return 0 <= tr and 0 <= tc and (tr + H) <= GH and (tc + W) <= GW

    dims = [(int(H), int(W)) for H, W in dims if fits(int(H), int(W))]
    dims = sorted(dims, key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    def assemble(H: int, W: int) -> np.ndarray | None:
        occ2 = np.full((H, W), -1, dtype=int)
        usedp = [False] * len(pieces)

        def has_unused(indices: list[int]) -> bool:
            return any(not usedp[j] for j in indices)

        def can_place(i: int, r: int, c: int) -> bool:
            a = pieces[i]
            h, w = a.shape
            if r < 0 or c < 0 or (r + h) > H or (c + w) > W:
                return False
            if (occ2[r : r + h, c : c + w] != -1).any():
                return False

            if r > 0:
                up = occ2[r - 1, c : c + w]
                if np.any(up == -1):
                    return False
                if not _compat(up, a[0, :], rect_col):
                    return False
            elif h < H and not has_unused(bottom_mates[i]):
                return False

            if c > 0:
                left = occ2[r : r + h, c - 1]
                if np.any(left == -1):
                    return False
                if not _compat(left, a[:, 0], rect_col):
                    return False
            elif w < W and not has_unused(right_mates[i]):
                return False

            if (r + h) < H:
                down = occ2[r + h, c : c + w]
                if np.any(down != -1):
                    if np.any(down == -1):
                        return False
                    if not _compat(down, a[-1, :], rect_col):
                        return False
                elif not has_unused(bottom_mates[i]):
                    return False

            if (c + w) < W:
                right = occ2[r : r + h, c + w]
                if np.any(right != -1):
                    if np.any(right == -1):
                        return False
                    if not _compat(right, a[:, -1], rect_col):
                        return False
                elif not has_unused(right_mates[i]):
                    return False

            return True

        def place(i: int, r: int, c: int) -> None:
            a = pieces[i]
            h, w = a.shape
            occ2[r : r + h, c : c + w] = a
            usedp[i] = True

        def unplace(i: int, r: int, c: int) -> None:
            a = pieces[i]
            h, w = a.shape
            occ2[r : r + h, c : c + w] = -1
            usedp[i] = False

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
            candidates: list[tuple[int, int]] = []
            for i in range(len(pieces)):
                if usedp[i]:
                    continue
                if can_place(i, r, c):
                    score = mate_score[i] * 100 - area[i]
                    candidates.append((score, i))
            candidates.sort()
            for _score, i in candidates:
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

    assert ass is not None and used_dim is not None

    H, W = used_dim
    tr, tc = _top_left_from_corner_cell(corner, mr, mc, H, W)
    out = np.full((GH, GW), int(bg), dtype=int)
    out[tr : tr + H, tc : tc + W] = ass
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    valid_grids = [(r, c) for r in range(2, 5) for c in range(2, 5) if 4 <= r * c <= 12]
    n_rows, n_cols = rng.choice(valid_grids)

    row_segs = _rand_unique_segments(22, n_rows, 4, 7, rng)
    col_segs = _rand_unique_segments(22, n_cols, 4, 7, rng)
    if row_segs is None or col_segs is None:
        return None

    RH = sum(row_segs)
    RW = sum(col_segs)
    if RH > 19 or RW > 19:
        return None

    GH = RH + rng.randint(6, 26 - RH) if RH <= 20 else RH + rng.randint(4, 26 - RH)
    GW = RW + rng.randint(6, 26 - RW) if RW <= 20 else RW + rng.randint(4, 26 - RW)
    if GH > 26 or GW > 26:
        return None

    palette = list(range(10))
    rng.shuffle(palette)
    bg = int(palette[0])
    rect_col = int(palette[1])
    marker_col = int(palette[2])
    accent_colors = [int(x) for x in palette[3:6]]
    codes = [(a, b) for a in accent_colors for b in accent_colors]

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

    rect = np.full((RH, RW), rect_col, dtype=int)
    row_starts: list[int] = []
    col_starts: list[int] = []
    s = 0
    for seg in row_segs:
        row_starts.append(s)
        s += seg
    s = 0
    for seg in col_segs:
        col_starts.append(s)
        s += seg

    pieces_meta: list[tuple[int, int, int, int]] = []
    row_of_piece: list[int] = []
    col_of_piece: list[int] = []
    for i, ph in enumerate(row_segs):
        for j, pw in enumerate(col_segs):
            pieces_meta.append((row_starts[i], col_starts[j], ph, pw))
            row_of_piece.append(i)
            col_of_piece.append(j)

    for i, ph in enumerate(row_segs):
        if ph < 4:
            return None
        if (n_cols - 1) > len(codes):
            return None
        seams = rng.sample(codes, n_cols - 1)
        left_outer = rng.choice(seams)
        right_outer = rng.choice(seams)
        lines = [left_outer] + seams + [right_outer]
        r0 = row_starts[i]
        for j, pw in enumerate(col_segs):
            c0 = col_starts[j]
            _paint_v_edge(rect, r0, c0, ph, lines[j])
            _paint_v_edge(rect, r0, c0 + pw - 1, ph, lines[j + 1])

    for j, pw in enumerate(col_segs):
        if pw < 4:
            return None
        if (n_rows - 1) > len(codes):
            return None
        seams = rng.sample(codes, n_rows - 1)
        top_outer = rng.choice(seams)
        bottom_outer = rng.choice(seams)
        lines = [top_outer] + seams + [bottom_outer]
        c0 = col_starts[j]
        for i, ph in enumerate(row_segs):
            r0 = row_starts[i]
            _paint_h_edge(rect, r0, c0, pw, lines[i])
            _paint_h_edge(rect, r0 + ph - 1, c0, pw, lines[i + 1])

    for piece in pieces_meta:
        _decorate_piece_interior(rect, piece, accent_colors, rng)

    piece_arrays: list[np.ndarray] = []
    for pr, pc, ph, pw in pieces_meta:
        pa = rect[pr : pr + ph, pc : pc + pw].copy()
        piece_arrays.append(pa)

    allpix: list[int] = []
    for pa in piece_arrays:
        allpix.extend(map(int, pa.flatten()))
    if Counter(allpix).most_common(1)[0][0] != rect_col:
        return None

    for pa in piece_arrays:
        ph, pw = pa.shape
        fill = float(np.sum(pa != rect_col)) / float(ph * pw)
        if fill < 0.10 or fill > 0.60:
            return None
        if not (
            np.any(pa[0, :] != rect_col)
            or np.any(pa[-1, :] != rect_col)
            or np.any(pa[:, 0] != rect_col)
            or np.any(pa[:, -1] != rect_col)
        ):
            return None

    if not np.any(rect[0, :] != rect_col):
        return None
    if not np.any(rect[-1, :] != rect_col):
        return None
    if not np.any(rect[:, 0] != rect_col):
        return None
    if not np.any(rect[:, -1] != rect_col):
        return None

    corner_candidates = _count_marked_corner_candidates(piece_arrays, rect_col, corner, n_rows, n_cols)
    if corner_candidates < 2:
        return None

    forced_hits = _forced_outer_hits(piece_arrays, rect_col, row_of_piece, col_of_piece, n_rows, n_cols)
    if forced_hits != 0:
        return None

    marker_cells = _marker_cells_from_corner(corner, mr, mc)
    for cr, cc in marker_cells:
        if cr < 0 or cr >= GH or cc < 0 or cc >= GW:
            return None

    canvas = np.full((GH, GW), bg, dtype=int)
    for cr, cc in marker_cells:
        canvas[cr, cc] = marker_col

    forbidden = np.zeros((GH, GW), dtype=bool)
    br0 = max(0, mr - 2)
    br1 = min(GH, mr + 3)
    bc0 = max(0, mc - 2)
    bc1 = min(GW, mc + 3)
    forbidden[br0:br1, bc0:bc1] = True

    order = list(range(len(piece_arrays)))
    rng.shuffle(order)
    for idx in order:
        pa = piece_arrays[idx]
        try:
            r0, c0 = _random_free_location_with_forbidden(
                canvas,
                pa,
                bg=bg,
                forbidden=forbidden,
                rng=rng,
                padding=1,
            )
            paste(canvas, pa, r0, c0)
        except ValueError:
            return None

    tr, tc = _top_left_from_corner_cell(corner, mr, mc, RH, RW)
    expected = np.full((GH, GW), bg, dtype=int)
    expected[tr : tr + RH, tc : tc + RW] = rect

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
