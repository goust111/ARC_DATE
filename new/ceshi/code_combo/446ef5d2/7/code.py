# Combo variant of 446ef5d2 (index 7)
# combo_spec: D7/1+D11/1
# Name: 行优先拼接配合大量小碎片

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import paste, random_free_location


"""
Design analysis for the fused variant
=====================================

a) Concrete behavior trace
--------------------------
Original task:
- Rectangular fragments are scattered on the background.
- A 3-cell L marker points to one corner of the final rectangle.
- The solver can conceptually grow the assembly from the anchored corner and use generic
  backtracking over 4-12 pieces of mixed sizes.

Fused variant:
- The final rectangle is partitioned into a dense grid of 13-20 *small* rectangular pieces.
- Every piece is a 3x3 tile. This stays inside the requested 2-3 side-length regime while
  making the high piece count reliable to generate and easy to read.
- Pieces must be placed in strict row-major order in the assembled rectangle:
  top row left->right, then next row, and so on.
- Each tile carries matching information only on its top/left/right/bottom edge midpoints.
  Because the solver always fills the next row-major slot, a candidate tile only needs to
  match the already placed tile on the left and the already placed tile above.

Small example in piece coordinates:
- Suppose the hidden rectangle is a 5x3 tile grid, so 15 scattered 3x3 pieces.
- Original logic would still treat this as a generic jigsaw and search over many placements.
- New logic first infers `n_cols = #pieces whose top edge is all rect_col` and
  `n_rows = #pieces whose left edge is all rect_col`, then fills slots
  (0,0), (0,1), ..., (0,2), (1,0), ...
- The marker still decides where the finished rectangle lands on the canvas, but it no longer
  determines the assembly order.

b) Complexity calibration
-------------------------
- The piece count is forced to 13-20, so generic free placement is much more expensive than
  in the base task.
- To avoid making generation brittle, the pieces are uniformly small 3x3 tiles. This keeps
  the visual effect of "many small fragments" strong without introducing noisy geometry.
- Row-major structure is not a hidden implementation trick: border pieces are recognizable
  because only top-row tiles have a fully rect_col top edge, only left-column tiles have a
  fully rect_col left edge, etc. Repeated evidence across many tiles makes the structure
  inferable from a few examples.

c) Implicit-assumption audit against the original solver
--------------------------------------------------------
- Original assumption: pieces may have varied rectangle sizes and the anchor piece is useful.
  No longer true. The new solver does not rely on anchor pre-placement at all.
- Original assumption: candidate rectangle dimensions come from area factor pairs.
  Replaced by a stronger assumption: all pieces have the same small size, and border-edge
  counts reveal the row/column grid directly.
- Original assumption: generic backtracking may need to inspect all four neighbors.
  Replaced by row-major placement, so only left/up can already exist at placement time.
- New assumptions introduced here:
  1. All fragments are identical in size (3x3).
  2. Every internal adjacency has a non-rect_col midpoint, so internal edges are never
     mistaken for outer rectangle borders.
  3. The hidden rectangle is a full tile grid with 4-6 rows and 3-4 columns, yielding
     13-20 pieces total.
- The generator explicitly enforces all three assumptions.

d) Placement legality
---------------------
- The generator constructs the final rectangle first, then scatters the tiles onto the
  background with `random_free_location(..., padding=1)`.
- This guarantees no overlap, no out-of-bounds placement, and at least one background cell
  between scattered pieces.
- The L marker is placed before scattering, so pieces cannot overwrite it.

e) Visual observability
-----------------------
- Humans can see that there are many more pieces than in the base task.
- Each tile is tiny and carries one or more colored edge-midpoint clues, producing a dense
  "micro-jigsaw" appearance instead of a few chunky blocks.
- Because border tiles have blank outer edges while internal edges carry colored midpoints,
  the notion of rows and columns is supported by repeated local evidence.

f) Visual cleanliness
---------------------
- Tiles remain axis-aligned rectangles and are separated by background when scattered, so the
  input does not become visually tangled.
- The output is still a single clean rectangle on the same canvas, preserving the original
  task's readability.

g) Degeneracy check
-------------------
- The fused behavior is quantified by two generator-side invariants:
  1. piece count is always in 13-20;
  2. border-edge counts decode a row-major tile grid (`top-border count = n_cols`,
     `left-border count = n_rows`).
- This means the "many small pieces" dimension is directly visible in the input, and the
  "row-major" dimension is encoded in the piece-edge structure rather than being only an
  internal search optimization.
"""


PIECE_H = 3
PIECE_W = 3
GRID_CHOICES = [(4, 4), (5, 3), (6, 3), (5, 4)]


def _find_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], str]:
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


def _extract_rects(g: np.ndarray, bg: int, mark: int) -> list[np.ndarray]:
    gh, gw = g.shape
    occ = (g != int(bg)) & (g != int(mark))
    used = np.zeros_like(occ, dtype=bool)
    rects: list[np.ndarray] = []
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
        rects.append(g[rr : rr + h, cc : cc + w].copy())
    return rects


def _top_left_from_corner(mr: int, mc: int, h: int, w: int, corner: str) -> tuple[int, int]:
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - w + 1
    if corner == "BL":
        return mr - h + 1, mc
    return mr - h + 1, mc - w + 1


def _compat(v1: np.ndarray, v2: np.ndarray, rect_col: int) -> bool:
    m = (v1 != int(rect_col)) & (v2 != int(rect_col))
    return bool((v1[m] == v2[m]).all())


def _is_outer_edge(vec: np.ndarray, rect_col: int) -> bool:
    return bool(np.all(vec == int(rect_col)))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    gh, gw = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    mark, missing, corner = _find_marker(g, bg)
    pieces = _extract_rects(g, bg, mark)
    assert len(pieces) >= 13
    assert all(p.shape == pieces[0].shape for p in pieces)
    ph, pw = map(int, pieces[0].shape)
    assert ph == PIECE_H and pw == PIECE_W

    allpix: list[int] = []
    for piece in pieces:
        allpix.extend(map(int, piece.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    top_flags = [_is_outer_edge(piece[0, :], rect_col) for piece in pieces]
    bottom_flags = [_is_outer_edge(piece[-1, :], rect_col) for piece in pieces]
    left_flags = [_is_outer_edge(piece[:, 0], rect_col) for piece in pieces]
    right_flags = [_is_outer_edge(piece[:, -1], rect_col) for piece in pieces]

    n_cols = sum(top_flags)
    n_rows = sum(left_flags)
    assert n_rows * n_cols == len(pieces)
    assert sum(bottom_flags) == n_cols
    assert sum(right_flags) == n_rows

    top_edges = [piece[0, :].copy() for piece in pieces]
    bottom_edges = [piece[-1, :].copy() for piece in pieces]
    left_edges = [piece[:, 0].copy() for piece in pieces]
    right_edges = [piece[:, -1].copy() for piece in pieces]

    slots = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    board = [[-1] * n_cols for _ in range(n_rows)]
    used = [False] * len(pieces)
    solved = False

    def matches_slot(i: int, r: int, c: int) -> bool:
        if top_flags[i] != (r == 0):
            return False
        if bottom_flags[i] != (r == n_rows - 1):
            return False
        if left_flags[i] != (c == 0):
            return False
        if right_flags[i] != (c == n_cols - 1):
            return False
        if c > 0:
            j = board[r][c - 1]
            if j == -1 or not _compat(right_edges[j], left_edges[i], rect_col):
                return False
        if r > 0:
            j = board[r - 1][c]
            if j == -1 or not _compat(bottom_edges[j], top_edges[i], rect_col):
                return False
        return True

    def bt(pos: int) -> None:
        nonlocal solved
        if pos == len(slots):
            solved = True
            return
        r, c = slots[pos]
        for i in range(len(pieces)):
            if used[i] or not matches_slot(i, r, c):
                continue
            used[i] = True
            board[r][c] = i
            bt(pos + 1)
            if solved:
                return
            board[r][c] = -1
            used[i] = False

    bt(0)
    assert solved

    rect_h = n_rows * ph
    rect_w = n_cols * pw
    top_left = _top_left_from_corner(int(missing[0]), int(missing[1]), rect_h, rect_w, corner)
    tr, tc = map(int, top_left)
    assert 0 <= tr and 0 <= tc and tr + rect_h <= gh and tc + rect_w <= gw

    rect = np.full((rect_h, rect_w), int(rect_col), dtype=int)
    for r in range(n_rows):
        for c in range(n_cols):
            idx = board[r][c]
            rect[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw] = pieces[idx]

    out = np.full((gh, gw), int(bg), dtype=int)
    out[tr : tr + rect_h, tc : tc + rect_w] = rect
    return out


def _marker_cells(corner: str, mr: int, mc: int) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _build_piece(
    rect_col: int,
    top: int,
    bottom: int,
    left: int,
    right: int,
) -> np.ndarray:
    piece = np.full((PIECE_H, PIECE_W), int(rect_col), dtype=int)
    piece[0, 1] = int(top)
    piece[2, 1] = int(bottom)
    piece[1, 0] = int(left)
    piece[1, 2] = int(right)
    center = rect_col
    for col in (top, left, right, bottom):
        if col != rect_col:
            center = col
            break
    piece[1, 1] = int(center)
    return piece


def _gen_attempt() -> np.ndarray | None:
    gh = random.randint(24, 26)
    gw = random.randint(24, 26)

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    n_interior = random.randint(2, 3)
    interior_cols = palette[3 : 3 + n_interior]

    n_rows, n_cols = random.choice(GRID_CHOICES)
    piece_count = n_rows * n_cols
    rect_h = n_rows * PIECE_H
    rect_w = n_cols * PIECE_W

    if rect_h + 2 > gh or rect_w + 2 > gw:
        return None

    corner = random.choice(["TL", "TR", "BL", "BR"])

    if corner == "TL":
        mr_min, mr_max = 2, gh - rect_h
        mc_min, mc_max = 2, gw - rect_w
    elif corner == "TR":
        mr_min, mr_max = 2, gh - rect_h
        mc_min, mc_max = rect_w - 1, gw - 3
    elif corner == "BL":
        mr_min, mr_max = rect_h - 1, gh - 3
        mc_min, mc_max = 2, gw - rect_w
    else:
        mr_min, mr_max = rect_h - 1, gh - 3
        mc_min, mc_max = rect_w - 1, gw - 3
    if mr_min > mr_max or mc_min > mc_max:
        return None
    mr = random.randint(mr_min, mr_max)
    mc = random.randint(mc_min, mc_max)

    vertical_codes = [
        [random.choice(interior_cols) for _ in range(n_cols - 1)]
        for _ in range(n_rows)
    ]
    horizontal_codes = [
        [random.choice(interior_cols) for _ in range(n_cols)]
        for _ in range(n_rows - 1)
    ]

    pieces: list[np.ndarray] = []
    rect = np.full((rect_h, rect_w), int(rect_col), dtype=int)
    for r in range(n_rows):
        for c in range(n_cols):
            top = rect_col if r == 0 else horizontal_codes[r - 1][c]
            bottom = rect_col if r == n_rows - 1 else horizontal_codes[r][c]
            left = rect_col if c == 0 else vertical_codes[r][c - 1]
            right = rect_col if c == n_cols - 1 else vertical_codes[r][c]
            piece = _build_piece(rect_col, top, bottom, left, right)
            pieces.append(piece)
            rect[r * PIECE_H : (r + 1) * PIECE_H, c * PIECE_W : (c + 1) * PIECE_W] = piece

    if not (13 <= len(pieces) <= 20):
        return None
    if any(piece.shape != (PIECE_H, PIECE_W) for piece in pieces):
        return None

    # Decode the row/column counts from border edges exactly as the solver will.
    decoded_cols = sum(int(np.all(piece[0, :] == rect_col)) for piece in pieces)
    decoded_rows = sum(int(np.all(piece[:, 0] == rect_col)) for piece in pieces)
    if decoded_rows != n_rows or decoded_cols != n_cols:
        return None

    marker_cells = _marker_cells(corner, mr, mc)
    for cr, cc in marker_cells:
        if cr < 0 or cr >= gh or cc < 0 or cc >= gw:
            return None

    canvas = np.full((gh, gw), int(bg), dtype=int)
    for cr, cc in marker_cells:
        canvas[cr, cc] = int(marker_col)

    order = list(range(piece_count))
    random.shuffle(order)
    for idx in order:
        piece = pieces[idx]
        try:
            r0, c0 = random_free_location(canvas, piece, transparent=bg, padding=1)
        except ValueError:
            return None
        paste(canvas, piece, r0, c0)

    tr, tc = _top_left_from_corner(mr, mc, rect_h, rect_w, corner)
    expected = np.full((gh, gw), int(bg), dtype=int)
    expected[tr : tr + rect_h, tc : tc + rect_w] = rect

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 400 attempts")
