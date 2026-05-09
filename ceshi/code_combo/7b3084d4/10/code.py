# Combo variant of 7b3084d4 (index 10)
# combo_spec: D6/0+D7/0+D10/2
# Name: 标记定序逐步拼装+部分旋转约束
#
# Design analysis:
# - Concreteization: the original task has one anchor marker and free rotation for
#   every non-anchor piece. This combo keeps the same 4-piece rectangle assembly,
#   but now three pieces carry singleton markers with visible order colors
#   5 < 6 < 8, while the fourth piece has no marker. The assembly order is fixed:
#   marker-5 anchor -> marker-6 piece -> marker-8 piece -> unmarked piece.
#   Only marked pieces may rotate in the input; the unmarked piece must keep its
#   original orientation. The output still shows the final assembled rectangle.
# - Small example trace: in a 6x6 target, the top-left anchor (marker 5) is placed
#   first at output corner (0,0). The top-right piece (marker 6) must be attached
#   to the anchor's exposed right edge. The bottom-right piece (marker 8) is then
#   attached to the structure created so far, and the bottom-left unmarked piece
#   fills the remaining gap without rotation. Compared with the original solver,
#   the search is no longer "pick any piece, try any rotation"; it is
#   "follow marker order, and only rotate marked pieces".
# - Complexity calibration: to keep the visual rule readable from a few examples,
#   the marker palette is restricted to three repeated singleton colors
#   {5, 6, 8}; body colors never reuse these colors. The unmarked piece is always
#   unique and intentionally asymmetric, so "fixed orientation" is visible rather
#   than a no-op on a rotationally symmetric shape.
# - Hidden assumptions audit: the original code assumed a single singleton marker
#   identifies the anchor, and all remaining pieces are monochrome objects that
#   can be rotated freely. That breaks here. The new solver therefore extracts
#   8-connected components, detects per-component singleton marker colors, sorts
#   pieces by marker order, and assigns a rotation mode per piece
#   ("rot" for marked, "fixed" for unmarked).
# - Placement legality: the generator is still output-first. It constructs a valid
#   2N x 2N tiling, extracts four non-overlapping components, randomly rotates
#   only the marked pieces, and scatters them on a 20x20 canvas with at least one
#   background cell of padding, so pieces never overlap or touch in the input.
# - Visual observability and neatness: the extra markers are embedded directly in
#   the pieces, not floating outside them. Their positions are stable corners of
#   three canonical pieces, so the order signal is easy to spot. The bottom-left
#   piece is always the only unmarked one, which makes the rotation restriction
#   visible without introducing a second hidden coding scheme.
# - Fusion-specific guardrails: the generator rejects samples where the fixed
#   unmarked piece is rotationally symmetric, and rejects samples where none of
#   the marked pieces is actually rotated in the input. It also enforces that the
#   marker-8 piece depends on the marker-6 piece in the intended clockwise chain
#   (top-left -> top-right -> bottom-right -> bottom-left), so the staged
#   attachment behavior is present in the generated family rather than only in the
#   solver implementation.

from __future__ import annotations

import math
import random
from collections import Counter, deque

import numpy as np

from arc_utils import get_bg as _bg_color, is_contiguous, neighbors8, random_free_location


BG = 0
ANCHOR_COLOR = 7
MARKER_SEQUENCE = (5, 6, 8)
MARKER_SET = set(MARKER_SEQUENCE)
COLOR_POOL = [1, 2, 3, 4, 9]
CANVAS_H, CANVAS_W = 20, 20
PLACEMENT_PADDING = 1
MAX_ATTEMPTS = 400
TAB_NAMES = ["H_left", "H_right", "V_top", "V_bot"]

TL, TR, BL, BR = 0, 1, 2, 3
LABEL_ORDER = (TL, TR, BR, BL)
MARKER_FOR_LABEL = {TL: 5, TR: 6, BR: 8}


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


def _variants(cells: list[tuple[int, int, int]], mode: str) -> list[list[tuple[int, int, int]]]:
    if mode == "fixed":
        return [_normalize(cells)]
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    outs: list[list[tuple[int, int, int]]] = []
    for k in range(4):
        var = _rotate_cells(cells, k)
        key = tuple(var)
        if key not in seen:
            seen.add(key)
            outs.append(var)
    return outs


def _cells_to_patch(cells: list[tuple[int, int, int]]) -> np.ndarray:
    h = max(r for r, _, _ in cells) + 1
    w = max(c for _, c, _ in cells) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c, col in cells:
        patch[r, c] = col
    return patch


def _components(g: np.ndarray) -> list[list[tuple[int, int, int]]]:
    bg = _bg_color(g)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int, int]]] = []
    for sr, sc in np.argwhere(g != bg):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        cells_abs: list[tuple[int, int, int]] = []
        while q:
            r, c = q.popleft()
            cells_abs.append((r, c, int(g[r, c])))
            for nr, nc in neighbors8(r, c):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] != bg:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(_normalize(cells_abs))
    return comps


def _piece_marker(cells: list[tuple[int, int, int]], color_counts: Counter[int]) -> int | None:
    colors = {col for _, _, col in cells}
    for marker in MARKER_SEQUENCE:
        if marker in colors and color_counts[marker] == 1:
            return marker
    return None


def _extract_pieces(g: np.ndarray) -> list[dict]:
    comps = _components(np.asarray(g, dtype=int))
    assert len(comps) == 4, "input must contain exactly four separated pieces"
    color_counts = Counter(map(int, np.asarray(g, dtype=int).ravel()))
    pieces = []
    for cells in comps:
        marker = _piece_marker(cells, color_counts)
        pieces.append(
            {
                "cells": cells,
                "marker_color": marker,
            }
        )

    marked = [p["marker_color"] for p in pieces if p["marker_color"] is not None]
    unmarked = [p for p in pieces if p["marker_color"] is None]
    assert sorted(marked) == list(MARKER_SEQUENCE), "pieces must carry markers 5, 6, and 8"
    assert len(unmarked) == 1, "exactly one piece must be unmarked"
    return pieces


def _dims(area: int) -> list[tuple[int, int]]:
    side = math.isqrt(area)
    if side * side == area:
        return [(side, side)]
    out = []
    for h in range(1, side + 1):
        if area % h == 0:
            w = area // h
            out.append((h, w))
            if h != w:
                out.append((w, h))
    out.sort(key=lambda t: (abs(t[0] - t[1]), t[0], t[1]))
    return out


def _placeable(board: np.ndarray, placement: list[tuple[int, int, int]]) -> bool:
    h, w = board.shape
    for r, c, _ in placement:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if board[r, c] != -1:
            return False
    return True


def _apply(board: np.ndarray, placement: list[tuple[int, int, int]], val: int | None = None) -> None:
    for r, c, col in placement:
        board[r, c] = col if val is None else val


def _touches_structure(board: np.ndarray, placement: list[tuple[int, int, int]]) -> bool:
    for r, c, _ in placement:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1] and board[nr, nc] != -1:
                return True
    return False


def _touches_owner(owner_board: np.ndarray, placement: list[tuple[int, int, int]], owner_id: int) -> bool:
    for r, c, _ in placement:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < owner_board.shape[0] and 0 <= nc < owner_board.shape[1]:
                if owner_board[nr, nc] == owner_id:
                    return True
    return False


def _placement_bbox(placement: list[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _, _ in placement]
    cs = [c for _, c, _ in placement]
    return min(rs), min(cs), max(rs), max(cs)


def _piece_order(pieces: list[dict]) -> list[int]:
    marked = sorted(
        [i for i, p in enumerate(pieces) if p["marker_color"] is not None],
        key=lambda i: pieces[i]["marker_color"],
    )
    unmarked = [i for i, p in enumerate(pieces) if p["marker_color"] is None]
    return marked + unmarked


def _anchor_placement(
    variant: list[tuple[int, int, int]], marker_color: int
) -> list[tuple[int, int, int]] | None:
    pts = [(r, c) for r, c, col in variant if col == marker_color]
    if len(pts) != 1:
        return None
    mr, mc = pts[0]
    placed = [(r - mr, c - mc, col) for r, c, col in variant]
    return placed


def _placements_for_piece(
    board: np.ndarray,
    owner_board: np.ndarray,
    variant: list[tuple[int, int, int]],
    *,
    marker_color: int | None,
    first_piece: bool,
    must_touch_owner: int | None,
) -> list[list[tuple[int, int, int]]]:
    h, w = board.shape
    outs: list[list[tuple[int, int, int]]] = []
    if first_piece:
        assert marker_color is not None
        placed = _anchor_placement(variant, marker_color)
        if placed is not None and _placeable(board, placed):
            outs.append(placed)
        return outs

    ph = max(r for r, _, _ in variant) + 1
    pw = max(c for _, c, _ in variant) + 1
    for tr in range(h - ph + 1):
        for tc in range(w - pw + 1):
            placed = [(r + tr, c + tc, col) for r, c, col in variant]
            if not _placeable(board, placed):
                continue
            if must_touch_owner is not None:
                if not _touches_owner(owner_board, placed, must_touch_owner):
                    continue
            elif not _touches_structure(board, placed):
                continue
            outs.append(placed)
    outs.sort(key=lambda p: _placement_bbox(p))
    return outs


def _solve_packing(pieces: list[dict], H: int, W: int) -> np.ndarray | None:
    board = np.full((H, W), -1, dtype=int)
    owner_board = np.full((H, W), -1, dtype=int)
    order = _piece_order(pieces)
    piece_vars = []
    for p in pieces:
        mode = "rot" if p["marker_color"] is not None else "fixed"
        piece_vars.append(_variants(p["cells"], mode))

    def backtrack(step: int) -> bool:
        if step == len(order):
            return not np.any(board == -1)

        pi = order[step]
        placements = []
        for var in piece_vars[pi]:
            placements.extend(
                _placements_for_piece(
                    board,
                    owner_board,
                    var,
                    marker_color=pieces[pi]["marker_color"],
                    first_piece=(step == 0),
                    must_touch_owner=None if step == 0 else order[step - 1],
                )
            )
        if not placements:
            return False

        seen: set[tuple[tuple[int, int, int], ...]] = set()
        unique_placements = []
        for placement in placements:
            key = tuple(sorted(placement))
            if key not in seen:
                seen.add(key)
                unique_placements.append(placement)

        for placement in unique_placements:
            _apply(board, placement)
            _apply(owner_board, placement, pi)
            if backtrack(step + 1):
                return True
            _apply(board, placement, -1)
            _apply(owner_board, placement, -1)
        return False

    if backtrack(0):
        return board
    return None


def _solve(grid: np.ndarray) -> np.ndarray:
    pieces = _extract_pieces(grid)
    area = sum(len(p["cells"]) for p in pieces)
    for H, W in _dims(area):
        solved = _solve_packing(pieces, H, W)
        if solved is not None:
            return solved.astype(int)
    raise AssertionError("ordered packing failed for all candidate dimensions")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble four scattered puzzle pieces into one rectangle using marker order and selective rotation.

    Input structure:
    - bg is the most common color. There are exactly four 8-separated non-bg pieces on the canvas.
    - Three pieces each contain one singleton marker pixel with colors 5, 6, and 8.
    - The remaining piece has no marker pixel. Markers stay embedded in their pieces rather than floating nearby.

    Rule:
    - Extract the four connected pieces and sort them by marker color: 5 first, then 6, then 8, and finally the unmarked piece.
    - Place the marker-5 piece first so its marker lands at output coordinate (0,0).
    - Append one piece at a time in that order; piece 6 must touch piece 5, piece 8 must touch piece 6, and the unmarked piece must touch piece 8.
    - Marked pieces may be rotated by quarter turns when searching for their fit, but the unmarked piece must keep its original orientation.

    Key insight:
    - The singleton marker colors no longer just identify an anchor; together they encode the assembly sequence, and the absence of a marker encodes the "do not rotate" exception.
    """
    return _solve(np.asarray(grid, dtype=int))


def _build_ownership(N: int, p: int, tab_specs: dict[str, tuple[int, str] | None]) -> np.ndarray | None:
    S = 2 * N
    own = np.zeros((S, S), dtype=int)
    own[:N, :N] = TL
    own[:N, N:] = TR
    own[N:, :N] = BL
    own[N:, N:] = BR

    for kind, col, up_owner, down_owner in [
        ("H_left", p, BL, TL),
        ("H_right", S - 1 - p, BR, TR),
    ]:
        spec = tab_specs.get(kind)
        if spec is None:
            continue
        depth, direction = spec
        if direction == "up":
            own[N - depth : N, col] = up_owner
        else:
            own[N : N + depth, col] = down_owner

    for kind, row, left_owner, right_owner in [
        ("V_top", p, TR, TL),
        ("V_bot", S - 1 - p, BR, BL),
    ]:
        spec = tab_specs.get(kind)
        if spec is None:
            continue
        depth, direction = spec
        if direction == "left":
            own[row, N - depth : N] = left_owner
        else:
            own[row, N : N + depth] = right_owner

    for q in (TL, TR, BL, BR):
        mask = (own == q).astype(int)
        if mask.sum() == 0:
            return None
        if not is_contiguous(mask, background=0, connectivity=8):
            return None
    return own


def _random_tab_specs(N: int) -> dict[str, tuple[int, str] | None]:
    specs: dict[str, tuple[int, str] | None] = {}
    for tn in TAB_NAMES:
        depth = random.randint(1, N - 1)
        direction = random.choice(["up", "down"] if tn.startswith("H") else ["left", "right"])
        specs[tn] = (depth, direction)
    if N == 2:
        specs[random.choice(TAB_NAMES)] = None
    return specs


def _labels_touch(own: np.ndarray, a: int, b: int) -> bool:
    h, w = own.shape
    for r in range(h):
        for c in range(w):
            if own[r, c] != a:
                continue
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and own[nr, nc] == b:
                    return True
    return False


def _extract_piece(
    own: np.ndarray,
    label: int,
    body_color: int,
    marker_map: dict[tuple[int, int], int],
) -> list[tuple[int, int, int]]:
    rows, cols = np.where(own == label)
    min_r, min_c = int(rows.min()), int(cols.min())
    cells = []
    for r, c in zip(rows, cols):
        r, c = int(r), int(c)
        col = marker_map.get((r, c), body_color)
        cells.append((r - min_r, c - min_c, int(col)))
    cells.sort()
    return cells


def _build_case() -> tuple[np.ndarray, list[tuple[int, list[tuple[int, int, int]]]], dict]:
    N = random.randint(3, 7)
    p = random.randint(1, N - 1)
    own = _build_ownership(N, p, _random_tab_specs(N))
    if own is None:
        raise ValueError("ownership generation failed")

    assert _labels_touch(own, TL, TR)
    assert _labels_touch(own, TR, BR)
    assert _labels_touch(own, BR, BL)
    if own[0, 0] != TL or own[0, N] != TR or own[N, N] != BR:
        raise ValueError("marker anchor cells must stay on their intended pieces")
    if _labels_touch(own, TL, BR):
        raise ValueError("marker-8 piece must not attach directly to the anchor")
    if _labels_touch(own, TR, BL):
        raise ValueError("unmarked piece must not shortcut directly to marker-6")

    pool = COLOR_POOL.copy()
    random.shuffle(pool)
    colors = {TL: ANCHOR_COLOR, TR: pool[0], BL: pool[1], BR: pool[2]}

    output = np.zeros_like(own, dtype=int)
    for label, col in colors.items():
        output[own == label] = col

    marker_map = {(0, 0): 5, (0, N): 6, (N, N): 8}
    for pos, col in marker_map.items():
        output[pos] = col

    pieces = []
    for label in LABEL_ORDER:
        pieces.append((label, _extract_piece(own, label, colors[label], marker_map)))

    fixed_piece = dict(pieces)[BL]
    if len(_variants(fixed_piece, "rot")) == 1:
        raise ValueError("unmarked fixed piece is rotationally symmetric")

    meta = {
        "N": N,
        "rotatable_markers": list(MARKER_SEQUENCE),
        "fixed_label": BL,
        "fixed_piece_variants": len(_variants(fixed_piece, "rot")),
        "br_depends_on_tr": True,
    }
    return output, pieces, meta


def _transform_pieces(
    pieces: list[tuple[int, list[tuple[int, int, int]]]]
) -> tuple[list[list[tuple[int, int, int]]], dict]:
    transformed = []
    rotated_marked_count = 0
    for label, cells in pieces:
        mode = "rot" if label in MARKER_FOR_LABEL else "fixed"
        variants = _variants(cells, mode)
        chosen = random.choice(variants)
        if label in MARKER_FOR_LABEL and tuple(chosen) != tuple(_normalize(cells)):
            rotated_marked_count += 1
        transformed.append(chosen)
    return transformed, {"rotated_marked_count": rotated_marked_count}


def _scatter_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=int)
    order = list(range(len(patches)))
    random.shuffle(order)
    for i in order:
        patch = patches[i]
        try:
            r0, c0 = random_free_location(
                canvas,
                patch,
                transparent=0,
                padding=PLACEMENT_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        region = canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        mask = patch != 0
        region[mask] = patch[mask]
    return canvas


def _gen_attempt(return_meta: bool = False):
    try:
        output, labeled_pieces, meta = _build_case()
    except ValueError:
        return (None, None) if return_meta else None

    transformed, rot_meta = _transform_pieces(labeled_pieces)
    if rot_meta["rotated_marked_count"] == 0:
        return (None, None) if return_meta else None

    patches = [_cells_to_patch(cells) for cells in transformed]
    canvas = _scatter_patches(patches)
    if canvas is None:
        return (None, None) if return_meta else None
    try:
        _extract_pieces(canvas)
    except Exception:
        return (None, None) if return_meta else None

    try:
        solved = main(canvas)
    except Exception:
        return (None, None) if return_meta else None
    if not np.array_equal(solved, output):
        return (None, None) if return_meta else None

    metrics = {
        **meta,
        **rot_meta,
        "marked_count": 3,
        "unmarked_count": 1,
    }
    if return_meta:
        return canvas, metrics
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
