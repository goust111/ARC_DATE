# Variant of e8686506
# Dimension: D16 — 碎片间的空间关系约束 (meta)
# Change: 碎片在输入中的相对位置不编码拼合信息——拼合完全由边缘几何关系决定... → 输入中碎片的相对位置暗示拼合位置

import random
from collections import Counter
from math import gcd

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# -----------------------------------------------------------------------------
# Design analysis for the D16 variant
#
# a) Concrete behavior + trace
# Original task: non-frame pieces are scattered arbitrarily; only the LR-symmetric
# frame and edge-compatibility constraints tell the solver where pieces belong.
# Variant: non-frame pieces are no longer arbitrary. Their input bbox top-left
# row/column order is an "exploded layout" of the final assembly:
# - if piece A is above piece B in the finished rectangle, A also appears above B
#   in the input;
# - if two pieces start on the same output row/column, they share the same input
#   row/column band.
# Small trace:
# - Original scatter might place a top-row bar below a bottom-row singleton.
# - Variant scatter assigns both to sparse bands derived from their output bbox
#   top-left coordinates, so the bar stays above and the singleton stays below.
# - Solver therefore adds pairwise row/column order constraints instead of relying
#   on geometry alone.
#
# b) Complexity calibration
# The variant is intentionally stronger than the original but not trivial:
# - frame detection, mirror symmetry, anchor requirements, and no-same-color-touch
#   remain active;
# - only non-frame piece placement gains a layout clue.
# The clue is discrete and repeated across many pairs of pieces, which makes it
# visually observable without introducing fragile exact-coordinate arithmetic.
#
# c) Implicit-assumption audit
# Original main() assumed scattered positions were irrelevant after extracting
# connected components. That is no longer true for non-frame pieces.
# The main lines that depended on the old assumption were:
# - grouping identical pieces into types and discarding per-instance positions;
# - solving only by local propagation over shape/color/frame constraints.
# In the variant we keep per-instance non-frame pieces, preserve their input bbox
# origins, and enforce pairwise row/column comparisons during search.
# Frame handling remains valid because the generator still places the frame as one
# aligned HxW outline group with the same relative geometry as before.
#
# d) Placement legality
# Generator-side sparse layout uses monotone row/column bands built from output
# bbox top-left coordinates. Band height/width is the max patch size in that band,
# plus one blank separator cell, so pieces never overlap and keep at least one-cell
# spacing. Frame and arrangement are placed in disjoint canvas regions.
#
# e) Visual observability
# A human now sees two signals:
# - the original outline/frame cue;
# - a clearly ordered non-frame arrangement where top/bottom/left/right relations
#   already resemble the final assembly.
# This is visible from 2-3 examples because it is repeated across many pieces, not
# hidden in color ids or arbitrary per-sample mappings.
#
# f) Visual cleanliness
# The non-frame arrangement is laid out in sparse bands instead of random clutter.
# Pieces are separated by blank rows/columns and are kept away from the frame by a
# fixed gap, so no region mixes too many cues at once.
#
# g) Color inference
# No new color rule is introduced. Colors keep their original role; the D16 change
# is entirely spatial.
#
# h) Variant metric
# For validation we measure the fraction of non-frame piece pairs whose input
# row-order sign and column-order sign match the corresponding output pair. The
# intended variant score is 1.0 for every generated sample.
#
# i) New main() assumptions and generator guarantees
# New assumption: non-frame input pieces preserve output bbox top-left partial
# order. Generator guarantees this by assigning each piece to monotone sparse row
# and column bands keyed by its true output bbox top-left coordinates.
# -----------------------------------------------------------------------------


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            h = n // w
            out.append((h, w))
    return out


def _sign(x: int) -> int:
    return -1 if x < 0 else (1 if x > 0 else 0)


def _build_requirements(
    H: int,
    W: int,
    comps_by_color: dict[int, list[np.ndarray]],
) -> np.ndarray:
    """
    Build per-cell required color constraints (req[r,c] == -1 means unconstrained).

    Heuristics supported by training examples:
      - If there is a color with exactly 4 singleton components, those are the four corners.
      - If there are full horizontal bars of length W-2 in the input, use them as the top/bottom edges
        between the corners (ordered by their vertical position in the input).
      - If there is a color with exactly two full vertical line components (width=1) in the input,
        place them on the left/right edges centered vertically.
    """
    req = np.full((H, W), -1, dtype=int)

    corner_colors = [
        col
        for col, comps in comps_by_color.items()
        if len(comps) == 4 and all(len(p) == 1 for p in comps)
    ]
    assert len(corner_colors) <= 1, "corner singleton color must be unique when present"
    corner_color = corner_colors[0] if corner_colors else None
    if corner_color is not None:
        req[0, 0] = corner_color
        req[0, W - 1] = corner_color
        req[H - 1, 0] = corner_color
        req[H - 1, W - 1] = corner_color

    bar_colors: list[tuple[int, int]] = []
    if W >= 3 and corner_color is not None:
        for col, comps in comps_by_color.items():
            for pts in comps:
                r0, c0, r1, c1 = bbox_coords(pts)
                hh, ww = r1 - r0 + 1, c1 - c0 + 1
                if hh == 1 and ww == (W - 2) and len(pts) == ww:
                    bar_colors.append((r0, col))
                    break
        if bar_colors:
            assert len(bar_colors) <= 2, "horizontal edge bar colors must be top/bottom only"
            bar_colors.sort()
            top_col = bar_colors[0][1]
            bot_col = bar_colors[-1][1]
            req[0, 1 : W - 1] = top_col
            req[H - 1, 1 : W - 1] = bot_col

    vertical_line_candidates = []
    for col, comps in comps_by_color.items():
        if len(comps) != 2:
            continue
        sigs = [_piece_sig(pts) for pts in comps]
        (h0, w0, coords0), (h1, w1, coords1) = sigs[0], sigs[1]
        if w0 == 1 and w1 == 1 and h0 == h1 and coords0 == coords1 and h0 > 1:
            vertical_line_candidates.append((col, h0))
    assert len(vertical_line_candidates) <= 1, "paired vertical edge-line color must be unique when present"
    if vertical_line_candidates:
        col, L = vertical_line_candidates[0]
        top = (H - L) // 2
        assert 0 <= top and top + L <= H, "paired vertical edge lines must fit in the output height"
        req[top : top + L, 0] = col
        req[top : top + L, W - 1] = col

    return req


def _find_frame_parts(g: np.ndarray, bg: int, H: int, W: int) -> list[tuple[int, np.ndarray]]:
    comps8_all: list[tuple[int, np.ndarray]] = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

    def _is_lr_symmetric(parts: list[tuple[int, np.ndarray]], h: int, w: int) -> bool:
        all_pts = np.concatenate([p for _, p in parts])
        r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        bmp = np.zeros((h, w), dtype=bool)
        for _col, pts in parts:
            for r, c in pts:
                bmp[int(r) - r0, int(c) - c0] = True
        return np.array_equal(bmp, bmp[:, ::-1])

    frame_parts: list[tuple[int, np.ndarray]] = []
    for i, (c1, p1) in enumerate(comps8_all):
        r0, c0, r1, c1b = bbox_coords(p1)
        if r1 - r0 + 1 == H and c1b - c0 + 1 == W and _is_lr_symmetric([(c1, p1)], H, W):
            frame_parts = [(c1, p1)]
            break
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W:
                if _is_lr_symmetric([(c1, p1), (c2, p2)], H, W):
                    frame_parts = [(c1, p1), (c2, p2)]
                    break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 LR-symmetric pieces (8-conn) whose union bbox equals output dims"
    return frame_parts


def _frame_req_and_mask(
    shape: tuple[int, int],
    H: int,
    W: int,
    frame_parts: list[tuple[int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    req = np.full((H, W), -1, dtype=int)
    frame_mask = np.zeros(shape, dtype=bool)
    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in frame_parts:
        for r, c in pts:
            rr, cc = int(r), int(c)
            frame_mask[rr, cc] = True
            or_, oc = rr - fr0, cc - fc0
            if 0 <= or_ < H and 0 <= oc < W:
                req[or_, oc] = int(col)
    return req, frame_mask


def _extract_non_frame_pieces(
    g: np.ndarray,
    bg: int,
    frame_mask: np.ndarray,
) -> tuple[list[dict], dict[int, list[np.ndarray]], dict[int, int], int]:
    pieces: list[dict] = []
    comps_by_color: dict[int, list[np.ndarray]] = {}
    color_piece_count: dict[int, int] = {}
    total_area = 0

    for col in sorted(set(map(int, np.unique(g)))):
        if col == bg:
            continue
        comps = components_mask(g == col, connectivity=4)
        comps_by_color[col] = comps
        color_piece_count[col] = len(comps)
        for pts in comps:
            total_area += int(len(pts))
            if all(bool(frame_mask[int(r), int(c)]) for r, c in pts):
                continue
            r0, c0, _r1, _c1 = bbox_coords(pts)
            h, w, coords = _piece_sig(pts)
            pieces.append(
                {
                    "col": int(col),
                    "coords": coords,
                    "h": int(h),
                    "w": int(w),
                    "in_r": int(r0),
                    "in_c": int(c0),
                }
            )
    return pieces, comps_by_color, color_piece_count, total_area


def _relation_mats(pieces: list[dict]) -> tuple[list[list[int]], list[list[int]]]:
    n = len(pieces)
    row_rel = [[0] * n for _ in range(n)]
    col_rel = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            row_rel[i][j] = _sign(int(pieces[i]["in_r"]) - int(pieces[j]["in_r"]))
            col_rel[i][j] = _sign(int(pieces[i]["in_c"]) - int(pieces[j]["in_c"]))
    return row_rel, col_rel


def _static_candidates(
    piece: dict,
    H: int,
    W: int,
    req: np.ndarray,
    fixed_board: np.ndarray,
    disallow_touch: set[int],
) -> list[dict]:
    col = int(piece["col"])
    coords = piece["coords"]
    out = []
    is_singleton = len(coords) == 1
    h = int(piece["h"])
    w = int(piece["w"])
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            cells = [(r0 + dr, c0 + dc) for dr, dc in coords]
            ok = True
            for r, c in cells:
                if fixed_board[r, c] != -1:
                    ok = False
                    break
                if req[r, c] != -1 and int(req[r, c]) != col:
                    ok = False
                    break
                if is_singleton and not (r == 0 or r == H - 1 or c == 0 or c == W - 1):
                    ok = False
                    break
                mc = W - 1 - c
                mv = fixed_board[r, mc]
                if mv != -1 and int(mv) != col:
                    ok = False
                    break
                if col in disallow_touch:
                    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= rr < H and 0 <= cc < W and fixed_board[rr, cc] == col:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                out.append({"r0": int(r0), "c0": int(c0), "cells": tuple(cells)})
    return out


def _candidate_ok(
    idx: int,
    cand: dict,
    pieces: list[dict],
    board: np.ndarray,
    placed: dict[int, dict],
    row_rel: list[list[int]],
    col_rel: list[list[int]],
    disallow_touch: set[int],
    W: int,
) -> bool:
    col = int(pieces[idx]["col"])
    for j, other in placed.items():
        if _sign(int(cand["r0"]) - int(other["r0"])) != row_rel[idx][j]:
            return False
        if _sign(int(cand["c0"]) - int(other["c0"])) != col_rel[idx][j]:
            return False

    for r, c in cand["cells"]:
        if board[r, c] != -1:
            return False
        mc = W - 1 - c
        mv = board[r, mc]
        if mv != -1 and int(mv) != col:
            return False
        if col in disallow_touch:
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < board.shape[0] and 0 <= cc < board.shape[1] and board[rr, cc] == col:
                    return False
    return True


def _place(board: np.ndarray, cand: dict, col: int) -> list[tuple[int, int]]:
    changed = []
    for r, c in cand["cells"]:
        board[r, c] = int(col)
        changed.append((int(r), int(c)))
    return changed


def _unplace(board: np.ndarray, changed: list[tuple[int, int]]) -> None:
    for r, c in changed:
        board[r, c] = -1


def _solve_with_layout(
    pieces: list[dict],
    H: int,
    W: int,
    req: np.ndarray,
    fixed_board: np.ndarray,
    disallow_touch: set[int],
) -> np.ndarray:
    n = len(pieces)
    row_rel, col_rel = _relation_mats(pieces)
    cand_lists = [
        _static_candidates(piece, H, W, req, fixed_board, disallow_touch)
        for piece in pieces
    ]
    assert all(cands for cands in cand_lists), "a piece has no valid placement under static constraints"

    board = fixed_board.copy()
    placed: dict[int, dict] = {}
    remaining = set(range(n))

    def feasible(i: int) -> list[dict]:
        return [
            cand
            for cand in cand_lists[i]
            if _candidate_ok(i, cand, pieces, board, placed, row_rel, col_rel, disallow_touch, W)
        ]

    def search() -> bool:
        if not remaining:
            return True

        best_i = None
        best_opts: list[dict] | None = None
        for i in sorted(remaining):
            opts = feasible(i)
            if not opts:
                return False
            if best_opts is None or len(opts) < len(best_opts):
                best_i = i
                best_opts = opts
                if len(best_opts) == 1:
                    break

        assert best_i is not None and best_opts is not None
        remaining.remove(best_i)
        try:
            for cand in best_opts:
                changed = _place(board, cand, int(pieces[best_i]["col"]))
                placed[best_i] = cand
                ok = True
                for j in remaining:
                    if not feasible(j):
                        ok = False
                        break
                if ok and search():
                    return True
                del placed[best_i]
                _unplace(board, changed)
            return False
        finally:
            remaining.add(best_i)

    assert search(), "layout-constrained assembly failed"
    assert np.array_equal(board, board[:, ::-1]), "assembled board must be LR-symmetric"
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble an exploded set of fragments into a left-right symmetric rectangle.

    Input structure:
    - bg is the most common color; one or two LR-symmetric frame pieces still appear as an aligned HxW outline.
    - All remaining fragments are disconnected, and their sparse input layout preserves the same above/below and left/right relations they have in the final assembly.

    Rule:
    - Infer the output rectangle size from total non-background area.
    - Lock the frame and any corner/bar/side-line anchors exactly as in the base task.
    - Place every remaining piece so shape/color constraints hold, the rectangle stays LR-symmetric, repeated same-color pieces do not 4-touch, and the pairwise row/column order of piece bbox origins matches the input layout.

    Key insight:
    - The non-frame pieces form an exploded arrangement of the solved picture, so the input already tells you which pieces belong above, below, left, or right of each other; frame geometry then turns those ordinal clues into a unique dense packing.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    dims_from_area = []
    total_area_all = int(np.count_nonzero(g != bg))
    for h, w in _factors(total_area_all):
        if h >= w:
            dims_from_area.append((h, w))
    assert dims_from_area, "at least one portrait factor pair must exist for the total piece area"
    best_gap = min(abs(h - w) for h, w in dims_from_area)
    best_dims = [(h, w) for h, w in dims_from_area if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    frame_parts = _find_frame_parts(g, bg, H, W)
    frame_req, frame_mask = _frame_req_and_mask(g.shape, H, W, frame_parts)

    pieces, comps_by_color, color_piece_count, total_area = _extract_non_frame_pieces(g, bg, frame_mask)
    assert total_area == total_area_all, "piece-area accounting must match non-background area"

    req = _build_requirements(H, W, comps_by_color)
    frame_cells = np.argwhere(frame_req != -1)
    for r, c in frame_cells:
        req[int(r), int(c)] = int(frame_req[int(r), int(c)])

    fixed_board = np.full((H, W), -1, dtype=int)
    for r, c in frame_cells:
        fixed_board[int(r), int(c)] = int(frame_req[int(r), int(c)])

    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}
    sol = _solve_with_layout(pieces, H, W, req, fixed_board, disallow_touch)
    return sol.astype(int)


# --------------- generator (D16 variant) ---------------

# Output dimensions
_AREA_MIN = 25
_AREA_MAX = 50
_WIDTH_REQUIRED = 5
_HEIGHT_MAX = 10

# Colors
_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_TWO_FRAME_MIN_H = 6
_TWO_FRAME_MIN_COLORS = 5
_TWO_FRAME_PROB = 0.35
_CENTER_CAP_PROB = 0.5

# Piece constraints
_MAX_PIECE_AREA = 6
_MAX_CHUNK_SIZE = 6
_PIECE_COUNT_MIN = 9
_PIECE_COUNT_MAX = 17

# Layout / canvas
_BAND_GAP = 1
_FRAME_LAYOUT_SEP = 3
_CANVAS_MARGIN_MIN = 1
_CANVAS_MARGIN_MAX = 3

# Retry
_GEN_MAX_RETRIES = 300


def _valid_dims() -> list[tuple[int, int]]:
    results = []
    for area in range(_AREA_MIN, _AREA_MAX + 1):
        dims = [(area // w, w) for w in range(1, area + 1) if area % w == 0 and area // w >= w]
        best_gap = min(abs(h - w) for h, w in dims)
        best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
        if len(best) == 1:
            h, w = best[0]
            if w == _WIDTH_REQUIRED and h <= _HEIGHT_MAX:
                results.append((h, w))
    return results


_VALID_DIMS = _valid_dims()
_DIM_WEIGHTS = [h * w for h, w in _VALID_DIMS]


def _zigzag_frame(H: int) -> tuple[set[tuple[int, int]], int, int]:
    cells = set()
    start_c = random.randint(0, 1)
    c = start_c
    for r in range(H):
        cells.add((r, c))
        if r == H - 1:
            break
        move = random.choice(["down", "horiz", "diag"])
        if move == "down":
            pass
        elif move == "horiz":
            cells.add((r, 1 - c))
            c = 1 - c
        else:
            c = 1 - c
    return cells, start_c, c


def _fill_holes(
    grid: np.ndarray,
    H: int,
    W: int,
    frame1: int,
    frame2: int,
    split: int,
    fill_pool: list[int],
) -> None:
    holes = components_mask(grid == -1, connectivity=4)
    prev_col = None
    for hole in holes:
        cells = sorted((int(r), int(c)) for r, c in hole)
        chunks = [cells[i : i + _MAX_CHUNK_SIZE] for i in range(0, len(cells), _MAX_CHUNK_SIZE)]
        for chunk in chunks:
            fc_set = {frame1 if r < split else frame2 for r, _ in chunk}
            avoid = fc_set | ({prev_col} if prev_col is not None else set())
            opts = [f for f in fill_pool if f not in avoid]
            if not opts:
                opts = [f for f in fill_pool if f not in fc_set] or fill_pool
            col = random.choice(opts)
            for r, c in chunk:
                grid[r, c] = col
                grid[r, W - 1 - c] = col
            prev_col = col


def _build_output(H: int, W: int) -> tuple[np.ndarray | None, set[int]]:
    half = W // 2
    n_colors = random.randint(_N_COLORS_MIN, _N_COLORS_MAX)
    palette = list(range(n_colors))
    random.shuffle(palette)

    use_two_frames = (
        H >= _TWO_FRAME_MIN_H
        and n_colors >= _TWO_FRAME_MIN_COLORS
        and random.random() < _TWO_FRAME_PROB
    )
    ci = 0
    frame1 = palette[ci]
    ci += 1
    frame2 = frame1
    if use_two_frames:
        frame2 = palette[ci]
        ci += 1
    fill_pool = palette[ci:]
    if not fill_pool:
        return None, set()

    frame_half, start_c, end_c = _zigzag_frame(H)
    split = H // 2 if frame1 != frame2 else H
    grid = np.full((H, W), -1, dtype=int)

    for r, c in frame_half:
        fc = frame1 if r < split else frame2
        grid[r, c] = fc
        grid[r, W - 1 - c] = fc

    if start_c == 1 and random.random() < _CENTER_CAP_PROB:
        grid[0, half] = frame1
    if end_c == 1 and random.random() < _CENTER_CAP_PROB:
        grid[H - 1, half] = frame2

    _fill_holes(grid, H, W, frame1, frame2, split, fill_pool)
    if np.any(grid < 0):
        return None, set()

    frame_set = {frame1}
    if frame2 != frame1:
        frame_set.add(frame2)
    return grid, frame_set


def _make_output() -> tuple[np.ndarray | None, int, set[int]]:
    H, W = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    out, abstract_fcs = _build_output(H, W)
    if out is None:
        return None, 0, set()
    bg = random.randint(0, 9)
    abstract = sorted(np.unique(out))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, set()
    cmap = {int(a): available[i] for i, a in enumerate(abstract)}
    real_fcs = {cmap[a] for a in abstract_fcs}
    remapped = np.vectorize(lambda x: cmap[int(x)])(out)
    return remapped.astype(int), bg, real_fcs


def _extract_non_frame_patches(out: np.ndarray, bg: int, fcs: set[int]) -> list[dict]:
    patches = []
    for col in sorted(map(int, np.unique(out))):
        if col in fcs:
            continue
        for pts in components_mask(out == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = col
            patches.append(
                {
                    "patch": patch,
                    "col": int(col),
                    "out_r": int(r0),
                    "out_c": int(c0),
                    "h": int(patch.shape[0]),
                    "w": int(patch.shape[1]),
                    "area": int((patch != bg).sum()),
                }
            )
    random.shuffle(patches)
    patches.sort(key=lambda p: (-int(p["area"]), int(p["out_r"]), int(p["out_c"])))
    return patches


def _pairwise_layout_score(pieces: list[dict]) -> float:
    if len(pieces) < 2:
        return 1.0
    total = 0
    good = 0
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            total += 2
            if _sign(int(pieces[i]["in_r"]) - int(pieces[j]["in_r"])) == _sign(
                int(pieces[i]["out_r"]) - int(pieces[j]["out_r"])
            ):
                good += 1
            if _sign(int(pieces[i]["in_c"]) - int(pieces[j]["in_c"])) == _sign(
                int(pieces[i]["out_c"]) - int(pieces[j]["out_c"])
            ):
                good += 1
    return good / total


def _make_arrangement_patch(pieces: list[dict], bg: int) -> tuple[np.ndarray, list[dict]]:
    row_vals = sorted({int(p["out_r"]) for p in pieces})
    col_vals = sorted({int(p["out_c"]) for p in pieces})
    row_h = {rv: max(int(p["h"]) for p in pieces if int(p["out_r"]) == rv) for rv in row_vals}
    col_w = {cv: max(int(p["w"]) for p in pieces if int(p["out_c"]) == cv) for cv in col_vals}

    row_map: dict[int, int] = {}
    col_map: dict[int, int] = {}
    cur = 0
    for rv in row_vals:
        row_map[int(rv)] = int(cur)
        cur += int(row_h[rv]) + _BAND_GAP
    arr_h = max(row_map[int(rv)] + int(row_h[rv]) for rv in row_vals)

    cur = 0
    for cv in col_vals:
        col_map[int(cv)] = int(cur)
        cur += int(col_w[cv]) + _BAND_GAP
    arr_w = max(col_map[int(cv)] + int(col_w[cv]) for cv in col_vals)

    patch = np.full((arr_h, arr_w), bg, dtype=int)
    placed = []
    for piece in pieces:
        rr = int(row_map[int(piece["out_r"])])
        cc = int(col_map[int(piece["out_c"])])
        patch_slice = patch[rr : rr + int(piece["h"]), cc : cc + int(piece["w"])]
        assert not np.any((patch_slice != bg) & (piece["patch"] != bg)), "arrangement pieces must not overlap"
        paste(patch, piece["patch"], rr, cc, transparent=bg)
        q = dict(piece)
        q["local_r"] = int(rr)
        q["local_c"] = int(cc)
        placed.append(q)
    return patch, placed


def _scatter(out: np.ndarray, bg: int, fcs: set[int]) -> tuple[np.ndarray, float, str]:
    oh, ow = out.shape
    frame_patch = np.full((oh, ow), bg, dtype=int)
    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in fcs:
        frame_mask |= out == int(col)
    frame_patch[frame_mask] = out[frame_mask]

    non_frame = _extract_non_frame_patches(out, bg, fcs)
    arr_patch, arranged = _make_arrangement_patch(non_frame, bg)
    ah, aw = arr_patch.shape

    margin = random.randint(_CANVAS_MARGIN_MIN, _CANVAS_MARGIN_MAX)
    layout_mode = random.choice(["right", "below"])
    if layout_mode == "right":
        canvas_h = max(oh, ah) + 2 * margin
        canvas_w = ow + _FRAME_LAYOUT_SEP + aw + 2 * margin
        frame_r = random.randint(margin, canvas_h - margin - oh)
        arr_r = random.randint(margin, canvas_h - margin - ah)
        frame_c = margin
        arr_c = frame_c + ow + _FRAME_LAYOUT_SEP
    else:
        canvas_h = oh + _FRAME_LAYOUT_SEP + ah + 2 * margin
        canvas_w = max(ow, aw) + 2 * margin
        frame_r = margin
        arr_r = frame_r + oh + _FRAME_LAYOUT_SEP
        frame_c = random.randint(margin, canvas_w - margin - ow)
        arr_c = random.randint(margin, canvas_w - margin - aw)

    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    paste(canvas, frame_patch, frame_r, frame_c, transparent=bg)
    paste(canvas, arr_patch, arr_r, arr_c, transparent=bg)

    for piece in arranged:
        piece["in_r"] = int(arr_r + int(piece["local_r"]))
        piece["in_c"] = int(arr_c + int(piece["local_c"]))
    score = _pairwise_layout_score(arranged)
    return canvas, float(score), str(layout_mode)


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _gen_attempt_with_meta() -> tuple[np.ndarray | None, np.ndarray | None, float, str]:
    out, bg, fcs = _make_output()
    if out is None:
        return None, None, 0.0, ""
    if not np.array_equal(out, out[:, ::-1]):
        return None, None, 0.0, ""
    if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
        return None, None, 0.0, ""
    pc = _piece_count(out)
    if not (_PIECE_COUNT_MIN <= pc <= _PIECE_COUNT_MAX):
        return None, None, 0.0, ""
    if any(
        len(pts) > _MAX_PIECE_AREA
        for col in np.unique(out)
        for pts in components_mask(out == col, connectivity=4)
    ):
        return None, None, 0.0, ""
    try:
        inp, score, mode = _scatter(out, bg, fcs)
        if score < 1.0:
            return None, None, float(score), mode
        got = main(inp)
    except Exception:
        return None, None, 0.0, ""
    if np.array_equal(got, out):
        return inp, out, float(score), mode
    return None, None, float(score), mode


def _gen_attempt() -> np.ndarray | None:
    inp, _out, _score, _mode = _gen_attempt_with_meta()
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 D16 input")
