# Combo variant of 7b3084d4 (index 6)
# combo_spec: D4/3+D13/0
# Name: 镜像翻转拼合+镜像对称输出

import random
from collections import deque

import numpy as np


BG = 0
ANCHOR_COLOR = 7
MARKER_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 8, 9]
CANVAS_H = 20
CANVAS_W = 20
MAX_ATTEMPTS = 500
TL, TR, BL, BR = 0, 1, 2, 3


# Design analysis for the fused variant
# a) Concrete behavior:
#    Original 7b3084d4 extracts four scattered pieces, rotates the anchor until
#    the lone gray tag reaches output (0,0), and accepts any exact rectangle.
#    Here the visible target is stricter: after packing, the color layout must
#    be left-right mirror-symmetric, while the gray marker still stays at the
#    top-left as a corner tag.  Because a literal singleton tag at (0,0) would
#    break exact color symmetry on an even-width board, the solver checks mirror
#    symmetry after temporarily replacing the tag with the anchor's own color.
#    Example: if the top-left anchor cell is tagged gray and the top-right cell
#    is anchor-colored, the underlying colored tiling can still be symmetric.
#    The anchor is also no longer rotation-only: the generator feeds it through
#    a reflected-then-rotated pose that pure rotations cannot recover from.
# b) Complexity calibration:
#    To make the mirror rule visible, the output uses repeated colors across the
#    mirrored pair of top pieces and across the mirrored pair of bottom pieces.
#    That breaks the original solver's "one color = one piece" shortcut, but is
#    still easy for a human to parse because the four pieces remain separated in
#    the input.  The ownership layout keeps only a few symmetric tab controls:
#    one mirrored horizontal pair and one mirrored center-row pair, so samples
#    are varied without becoming visually arbitrary.
# c) Broken implicit assumptions from the original solver:
#    - Grouping by global color is invalid once mirrored pieces reuse colors.
#    - Rotating the anchor is insufficient; dihedral search is required.
#    - Any packed rectangle is no longer acceptable; the final assembly must
#      satisfy a mirror check after removing the gray tag from the comparison.
# d) Placement legality:
#    The generator is output-first, extracts four connected pieces, transforms
#    them into tight patches, and scatters those patches on a 20x20 canvas with
#    at least one-cell padding around occupied pixels, so pieces never overlap
#    or touch.
# e) Visual observability:
#    Across examples, solvers and humans can see two stable cues together:
#    repeated colors on mirrored left/right pieces, and anchor poses that are
#    visibly mirrored rather than just rotated.  Both cues survive in only 2-4
#    examples because the generator rejects cases where the anchor shape would
#    be recoverable by rotation alone.
# f) Visual cleanliness:
#    No distractors are added, pieces are well separated, and the only special
#    marker is the lone gray tag.  Symmetry is expressed through large colored
#    regions rather than hidden metadata.
# g) Inferability:
#    Color changes are structural rather than arbitrary: top-left/right share
#    the anchor color, bottom-left/right share one secondary color, and the
#    mirror axis is always vertical through the center of the assembled square.


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield r + dr, c + dc


def _bg_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _normalize(cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    min_r = min(r for r, _, _ in cells)
    min_c = min(c for _, c, _ in cells)
    out = [(r - min_r, c - min_c, col) for r, c, col in cells]
    out.sort()
    return out


def _rotate_cells(cells: list[tuple[int, int, int]], k: int) -> list[tuple[int, int, int]]:
    pts = list(cells)
    for _ in range(k % 4):
        pts = [(c, -r, col) for r, c, col in pts]
    return _normalize(pts)


def _reflect_cells(cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    return _normalize([(r, -c, col) for r, c, col in cells])


def _variants(cells: list[tuple[int, int, int]], allow_flip: bool = False) -> list[list[tuple[int, int, int]]]:
    bases = [cells]
    if allow_flip:
        bases.append(_reflect_cells(cells))
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    outs: list[list[tuple[int, int, int]]] = []
    for base in bases:
        for k in range(4):
            v = _rotate_cells(base, k)
            key = tuple(v)
            if key not in seen:
                seen.add(key)
                outs.append(v)
    return outs


def _bbox(cells: list[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _, _ in cells]
    cs = [c for _, c, _ in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _marker_point(cells: list[tuple[int, int, int]]) -> tuple[int, int] | None:
    pts = [(r, c) for r, c, col in cells if col == MARKER_COLOR]
    if len(pts) == 1:
        return pts[0]
    return None


def _components(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for sr, sc in np.argwhere(g != bg):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        abs_cells: list[tuple[int, int, int]] = []
        while q:
            r, c = q.popleft()
            abs_cells.append((r, c, int(g[r, c])))
            for nr, nc in _neighbors8(r, c):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] != bg:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(
            {
                "cells": _normalize(abs_cells),
                "abs": abs_cells,
                "bbox": _bbox(abs_cells),
            }
        )
    return comps, bg


def _anchor_index(comps: list[dict]) -> int | None:
    for i, comp in enumerate(comps):
        if _marker_point(comp["cells"]) is not None:
            return i
    return None


def _dims(area: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for h in range(1, int(area**0.5) + 1):
        if area % h == 0:
            w = area // h
            out.append((h, w))
            if h != w:
                out.append((w, h))
    out.sort(key=lambda hw: (0 if hw[0] == hw[1] else 1, abs(hw[0] - hw[1]), hw[0], hw[1]))
    return out


def _placeable(board: np.ndarray, placement: list[tuple[int, int, int]]) -> bool:
    h, w = board.shape
    for r, c, _ in placement:
        if not (0 <= r < h and 0 <= c < w) or board[r, c] != -1:
            return False
    return True


def _apply(board: np.ndarray, placement: list[tuple[int, int, int]], val: int | None = None) -> None:
    for r, c, col in placement:
        board[r, c] = col if val is None else val


def _all_placements(var: list[tuple[int, int, int]], H: int, W: int) -> list[list[tuple[int, int, int]]]:
    ph = max(r for r, _, _ in var) + 1
    pw = max(c for _, c, _ in var) + 1
    outs: list[list[tuple[int, int, int]]] = []
    for tr in range(H - ph + 1):
        for tc in range(W - pw + 1):
            outs.append([(r + tr, c + tc, col) for r, c, col in var])
    return outs


def _effective_symmetry_board(board: np.ndarray) -> np.ndarray | None:
    pts = np.argwhere(board == MARKER_COLOR)
    if len(pts) != 1:
        return None
    mr, mc = map(int, pts[0])
    neigh = {
        int(board[nr, nc])
        for nr, nc in _neighbors8(mr, mc)
        if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1] and board[nr, nc] not in (-1, BG, MARKER_COLOR)
    }
    if len(neigh) != 1:
        return None
    out = board.copy()
    out[mr, mc] = next(iter(neigh))
    return out


def _is_lr_symmetric(board: np.ndarray) -> bool:
    effective = _effective_symmetry_board(board)
    if effective is None or effective.shape[1] % 2 != 0:
        return False
    return np.array_equal(effective, effective[:, ::-1])


def _solve_packing(comps: list[dict], H: int, W: int) -> np.ndarray | None:
    anchor_idx = _anchor_index(comps)
    if anchor_idx is None or W % 2 != 0:
        return None

    board = np.full((H, W), -1, dtype=int)
    piece_vars = [_variants(comp["cells"], allow_flip=(i == anchor_idx)) for i, comp in enumerate(comps)]
    remaining = [i for i in range(len(comps)) if i != anchor_idx]

    for var in piece_vars[anchor_idx]:
        marker = _marker_point(var)
        if marker is None:
            continue
        mr, mc = marker
        anchor_place = [(r - mr, c - mc, col) for r, c, col in var]
        if not _placeable(board, anchor_place):
            continue
        _apply(board, anchor_place)

        def backtrack(rem: list[int]) -> bool:
            if not rem:
                return _is_lr_symmetric(board)

            best_i = None
            best_opts = None
            for i in rem:
                opts = []
                for piece_var in piece_vars[i]:
                    for placement in _all_placements(piece_var, H, W):
                        if _placeable(board, placement):
                            opts.append(placement)
                if not opts:
                    return False
                if best_opts is None or len(opts) < len(best_opts):
                    best_i, best_opts = i, opts
                    if len(best_opts) == 1:
                        break

            assert best_i is not None and best_opts is not None
            next_rem = [x for x in rem if x != best_i]
            for placement in best_opts:
                _apply(board, placement)
                if backtrack(next_rem):
                    return True
                _apply(board, placement, -1)
            return False

        if backtrack(remaining):
            return board.copy()
        _apply(board, anchor_place, -1)

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble four separated puzzle pieces into a square whose colored layout is mirror-symmetric, while preserving the tagged anchor corner.

    Input structure:
    - Background is the dominant color 0. Four disconnected 8-connected pieces are scattered on the canvas.
    - One piece is the anchor: it contains a single gray marker pixel (color 5) attached at one corner, and mirrored left/right output pairs may reuse the same fill color.

    Rule:
    - Extract pieces by connected components rather than by color, because mirrored partner pieces can share colors.
    - Search the anchor in the full flip+rotation space so its marker can be moved to output position (0,0); rotate the other pieces while packing them into one square with no gaps.
    - Among complete packings, keep only the one whose assembled color layout is left-right mirror-symmetric after treating the gray marker as a tag on top of the anchor color.

    Key insight:
    - The gray pixel still fixes the anchor corner, but it is no longer enough to rotate pieces blindly: the anchor may need a reflection, and the repeated colors across mirrored piece pairs reveal that the finished square must balance around its vertical midline.
    """
    g = np.asarray(grid, dtype=int)
    comps, _bg = _components(g)
    assert comps, "input must contain at least one non-background component"

    area = sum(len(comp["cells"]) for comp in comps)
    side = int(round(area**0.5))
    candidate_dims = [(side, side)] if side * side == area and side % 2 == 0 else _dims(area)
    for H, W in candidate_dims:
        solved = _solve_packing(comps, H, W)
        if solved is not None:
            return solved.astype(int)
    raise AssertionError("no symmetric packing found")


def _is_contiguous(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _build_symmetric_ownership(N: int) -> np.ndarray | None:
    S = 2 * N
    for _ in range(200):
        own = np.zeros((S, S), dtype=int)
        own[:N, :N] = TL
        own[:N, N:] = TR
        own[N:, :N] = BL
        own[N:, N:] = BR

        p = random.randint(1, N - 1)
        h_depth = random.randint(1, N - 1)
        h_dir = random.choice(["up", "down"])
        vt_depth = random.randint(1, N - 1)
        vb_depth = random.randint(1, N - 1)

        for col, up_owner, down_owner in ((p, BL, TL), (S - 1 - p, BR, TR)):
            if h_dir == "up":
                own[N - h_depth:N, col] = up_owner
            else:
                own[N:N + h_depth, col] = down_owner

        own[p, N - vt_depth:N] = TR
        own[p, N:N + vt_depth] = TL
        own[S - 1 - p, N - vb_depth:N] = BR
        own[S - 1 - p, N:N + vb_depth] = BL

        if all(_is_contiguous(own == q) and int(np.sum(own == q)) > 0 for q in (TL, TR, BL, BR)):
            return own
    return None


def _extract_piece(own: np.ndarray, label: int, color: int, marker_abs: tuple[int, int] | None = None) -> list[tuple[int, int, int]]:
    rows, cols = np.where(own == label)
    min_r, min_c = int(rows.min()), int(cols.min())
    cells = [(int(r) - min_r, int(c) - min_c, color) for r, c in zip(rows, cols)]
    if marker_abs is not None:
        mr, mc = marker_abs
        local_marker = (mr - min_r, mc - min_c)
        cells = [(r, c, MARKER_COLOR if (r, c) == local_marker else col) for r, c, col in cells]
    return _normalize(cells)


def _cells_to_patch(cells: list[tuple[int, int, int]]) -> np.ndarray:
    h = max(r for r, _, _ in cells) + 1
    w = max(c for _, c, _ in cells) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c, col in cells:
        patch[r, c] = col
    return patch


def _scatter_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=int)
    occupied = np.zeros((CANVAS_H, CANVAS_W), dtype=bool)
    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        patch = patches[idx]
        h, w = patch.shape
        mask = patch != 0
        placed = False
        for _ in range(300):
            r0 = random.randint(0, CANVAS_H - h)
            c0 = random.randint(0, CANVAS_W - w)
            if np.any(occupied[r0:r0 + h, c0:c0 + w] & mask):
                continue
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(CANVAS_H, r0 + h + 1)
            cc1 = min(CANVAS_W, c0 + w + 1)
            if np.any(occupied[rr0:rr1, cc0:cc1]):
                continue
            canvas[r0:r0 + h, c0:c0 + w][mask] = patch[mask]
            occupied[r0:r0 + h, c0:c0 + w][mask] = True
            placed = True
            break
        if not placed:
            return None
    return canvas


def _reflection_only_variants(cells: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    rot_keys = {tuple(v) for v in _variants(cells, allow_flip=False)}
    outs = [v for v in _variants(cells, allow_flip=True) if tuple(v) not in rot_keys]
    return outs


def _build_target() -> tuple[np.ndarray, list[list[tuple[int, int, int]]]] | None:
    N = random.randint(3, 7)
    own = _build_symmetric_ownership(N)
    if own is None:
        return None

    lower_color = random.choice(COLOR_POOL)
    output = np.zeros((2 * N, 2 * N), dtype=int)
    output[(own == TL) | (own == TR)] = ANCHOR_COLOR
    output[(own == BL) | (own == BR)] = lower_color
    output[0, 0] = MARKER_COLOR

    anchor = _extract_piece(own, TL, ANCHOR_COLOR, marker_abs=(0, 0))
    if not _reflection_only_variants(anchor):
        return None

    pieces = [
        anchor,
        _extract_piece(own, TR, ANCHOR_COLOR),
        _extract_piece(own, BL, lower_color),
        _extract_piece(own, BR, lower_color),
    ]
    return output, pieces


def _transform_pieces_for_input(pieces: list[list[tuple[int, int, int]]]) -> list[list[tuple[int, int, int]]]:
    anchor_choices = _reflection_only_variants(pieces[0])
    assert anchor_choices, "anchor must require reflection in this combo generator"
    transformed = [random.choice(anchor_choices)]
    for cells in pieces[1:]:
        transformed.append(random.choice(_variants(cells, allow_flip=False)))
    return transformed


def _gen_attempt() -> np.ndarray | None:
    built = _build_target()
    if built is None:
        return None
    expected, pieces = built
    transformed = _transform_pieces_for_input(pieces)
    canvas = _scatter_patches([_cells_to_patch(cells) for cells in transformed])
    if canvas is None:
        return None
    try:
        out = main(canvas)
    except Exception:
        return None
    if not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
