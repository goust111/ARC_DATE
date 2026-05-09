# Combo variant of 7b3084d4 (index 19)
# combo_spec: D3/1+D8/1
# Name: L/U形输出+无tab简单分割
#
# Design analysis
# a) Concrete trace:
#    Original: 4 quadrant-like pieces with tabs must tile a full rectangle, so
#    the solver can scan factor pairs of the total area and rely on interlocks.
#    Fused variant: the 4 pieces are plain rectangles.  The assembled output is
#    not filled; it is a tight bounding box containing either:
#    - an L-shape with a missing bottom-right corner, or
#    - a U-shape with a missing top-middle notch.
#    Example L:
#      anchor 2x3, top-mid 2x2, top-right 2x4, bottom-left 3x5
#      Output is 5x9, with a 3x4 zero corner at bottom-right.
#    Example U:
#      anchor 2x3, top-right 2x4, bottom-left 3x3, bottom-right 3x6
#      Output is 5x9, with a 2x2 zero notch at top-middle.
#    The marker still rotates with the anchor piece and must land at output
#    cell (0,0).  The difference is that the solver now matches a small family
#    of non-rectangular templates instead of searching only full rectangles.
#
# b) Complexity calibration:
#    Using only two silhouettes keeps the rule observable from a few examples,
#    while still breaking the original "tile a rectangle" assumption.  The
#    dimensions are sampled from small visible ranges and the top/bottom heights
#    are forced different, which prevents L/U cases from collapsing into the
#    same degenerate geometry.
#
# c) Hidden-assumption review:
#    - Original `_dims(area)` assumed the output had no holes; invalid now.
#    - Original packing ignored empty cells inside the tight bbox; invalid now.
#    - Original color extraction grouped one piece per color; fragile once the
#      anchor uses a marker color.  This variant extracts 8-connected pieces.
#    - Original uniqueness came from tabs.  Here uniqueness comes from template
#      equations between rectangle sizes plus the fixed anchor corner.
#
# d) Placement legality:
#    Generator builds the output first, extracts four rectangular patches,
#    rotates them independently, then scatters them on a 20x20 canvas with at
#    least one-cell padding so pieces never overlap or touch.
#
# e) Visual observability:
#    The zero notch/corner is visible in the output, so the non-rectangular
#    target can be inferred directly by eye.  Piece colors remain few and
#    stable: anchor is 7, marker is 5, the other three pieces use distinct
#    colors from a small pool.
#
# f) Visual cleanliness:
#    There are only four components, each a solid rectangle.  No tabs, no
#    internal decoration beyond the marker, and the scatter step keeps clear
#    separation between objects.
#
# g) New solver assumptions and generator guarantees:
#    - Exactly four 8-connected pieces exist.
#    - Every piece is a filled rectangle in its tight bbox.
#    - Exactly one piece contains the singleton marker color 5.
#    - A valid sample matches one of the two templates after rotating pieces.
#    `generate_input()` constructs every sample to satisfy these assumptions and
#    rejects any sample for which `main()` does not reproduce the target.

import itertools
import random
from collections import deque

import numpy as np


BG = 0
ANCHOR_COLOR = 7
MARKER_COLOR = 5
OTHER_COLORS = [1, 2, 3, 4, 6, 8, 9]
CANVAS_H = 20
CANVAS_W = 20
MAX_ATTEMPTS = 500


def _bg_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield r + dr, c + dc


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


def _variants(cells: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    outs: list[list[tuple[int, int, int]]] = []
    for k in range(4):
        rot = _rotate_cells(cells, k)
        key = tuple(rot)
        if key not in seen:
            seen.add(key)
            outs.append(rot)
    return outs


def _cells_to_patch(cells: list[tuple[int, int, int]]) -> np.ndarray:
    h = max(r for r, _, _ in cells) + 1
    w = max(c for _, c, _ in cells) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c, col in cells:
        patch[r, c] = col
    return patch


def _is_filled_rectangle(cells: list[tuple[int, int, int]]) -> bool:
    h = max(r for r, _, _ in cells) + 1
    w = max(c for _, c, _ in cells) + 1
    return len(cells) == h * w


def _marker_pos(cells: list[tuple[int, int, int]]) -> tuple[int, int] | None:
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
        abs_cells = []
        while q:
            r, c = q.popleft()
            abs_cells.append((r, c, int(g[r, c])))
            for nr, nc in _neighbors8(r, c):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] != bg:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        local = _normalize(abs_cells)
        comps.append({"cells": local})
    return comps, bg


def _variant_records(cells: list[tuple[int, int, int]], marker_tl: bool = False) -> list[dict]:
    out = []
    for var in _variants(cells):
        marker = _marker_pos(var)
        if marker_tl and marker != (0, 0):
            continue
        patch = _cells_to_patch(var)
        out.append(
            {
                "cells": var,
                "patch": patch,
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
            }
        )
    return out


def _place_patch(board: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != 0
    board[top : top + patch.shape[0], left : left + patch.shape[1]][mask] = patch[mask]


def _board_from_l(anchor: dict, top_mid: dict, top_right: dict, bottom_left: dict) -> np.ndarray | None:
    if not (anchor["h"] == top_mid["h"] == top_right["h"]):
        return None
    if bottom_left["w"] != anchor["w"] + top_mid["w"]:
        return None
    h_top = anchor["h"]
    H = h_top + bottom_left["h"]
    W = bottom_left["w"] + top_right["w"]
    board = np.zeros((H, W), dtype=int)
    _place_patch(board, anchor["patch"], 0, 0)
    _place_patch(board, top_mid["patch"], 0, anchor["w"])
    _place_patch(board, top_right["patch"], 0, anchor["w"] + top_mid["w"])
    _place_patch(board, bottom_left["patch"], h_top, 0)
    if np.all(board != 0):
        return None
    return board


def _board_from_u(anchor: dict, top_right: dict, bottom_left: dict, bottom_band: dict) -> np.ndarray | None:
    if anchor["h"] != top_right["h"]:
        return None
    if anchor["w"] != bottom_left["w"]:
        return None
    if bottom_left["h"] != bottom_band["h"]:
        return None
    gap_w = bottom_band["w"] - top_right["w"]
    if gap_w <= 0:
        return None
    h_top = anchor["h"]
    H = h_top + bottom_left["h"]
    W = anchor["w"] + gap_w + top_right["w"]
    board = np.zeros((H, W), dtype=int)
    _place_patch(board, anchor["patch"], 0, 0)
    _place_patch(board, top_right["patch"], 0, anchor["w"] + gap_w)
    _place_patch(board, bottom_left["patch"], h_top, 0)
    _place_patch(board, bottom_band["patch"], h_top, anchor["w"])
    if np.all(board != 0):
        return None
    return board


def _solve_templates(comps: list[dict]) -> np.ndarray:
    assert len(comps) == 4, "input must contain exactly four scattered pieces"
    for comp in comps:
        assert _is_filled_rectangle(comp["cells"]), "each piece must be a solid rectangle"

    anchor_idx = next((i for i, comp in enumerate(comps) if _marker_pos(comp["cells"]) is not None), None)
    assert anchor_idx is not None, "one piece must contain the singleton marker"

    anchor_variants = _variant_records(comps[anchor_idx]["cells"], marker_tl=True)
    assert anchor_variants, "anchor must be rotatable so the marker reaches the top-left corner"

    other_idxs = [i for i in range(len(comps)) if i != anchor_idx]
    other_variant_sets = {i: _variant_records(comps[i]["cells"]) for i in other_idxs}
    for i in other_idxs:
        assert other_variant_sets[i], "each non-anchor piece must have at least one orientation"

    for anchor in anchor_variants:
        for perm in itertools.permutations(other_idxs):
            vars1 = other_variant_sets[perm[0]]
            vars2 = other_variant_sets[perm[1]]
            vars3 = other_variant_sets[perm[2]]
            for rec1 in vars1:
                for rec2 in vars2:
                    for rec3 in vars3:
                        board = _board_from_l(anchor, rec1, rec2, rec3)
                        if board is not None:
                            return board
                        board = _board_from_u(anchor, rec1, rec2, rec3)
                        if board is not None:
                            return board

    raise AssertionError("pieces do not match the supported L/U rectangular-piece templates")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble four scattered rectangular pieces into a tight L- or U-shaped silhouette.

    Input structure:
    - Background is the most common color.
    - Exactly four 8-connected pieces are scattered on the canvas and separated by gaps.
    - One piece is the anchor: it is a filled rectangle whose corner cell is replaced by
      a unique marker pixel of color 5.

    Rule:
    - Extract the four pieces by connectivity, not by color.
    - Rotate the anchor so its marker sits at output position (0,0).
    - Rotate the other three rectangles as needed and fit all four pieces into one of two
      non-rectangular templates: an L with a missing bottom-right corner, or a U with a
      missing top-middle notch.
    - Return the tight bounding box of that assembly, leaving the missing corner/notch as 0.

    Key insight:
    - Without tabs, the decisive evidence is no longer geometric interlocking.  The marker
      fixes the anchor corner, and the rectangle dimensions reveal whether the final layout
      is the L template or the U template.
    """
    comps, _bg = _components(np.asarray(grid, dtype=int))
    assert comps, "input must contain at least one non-background piece"
    return _solve_templates(comps).astype(int)


def _rect_cells(h: int, w: int, color: int, marker: bool = False) -> list[tuple[int, int, int]]:
    cells = []
    for r in range(h):
        for c in range(w):
            col = MARKER_COLOR if marker and (r, c) == (0, 0) else color
            cells.append((r, c, col))
    return cells


def _build_l_output(dim_vals: tuple[int, int, int, int, int], colors: list[int]) -> tuple[np.ndarray, list[list[tuple[int, int, int]]]]:
    h_top, h_bottom, w_a, w_b, w_c = dim_vals
    H = h_top + h_bottom
    W = w_a + w_b + w_c
    out = np.zeros((H, W), dtype=int)
    out[:h_top, :w_a] = ANCHOR_COLOR
    out[:h_top, w_a : w_a + w_b] = colors[0]
    out[:h_top, w_a + w_b :] = colors[1]
    out[h_top:, : w_a + w_b] = colors[2]
    out[0, 0] = MARKER_COLOR
    pieces = [
        _rect_cells(h_top, w_a, ANCHOR_COLOR, marker=True),
        _rect_cells(h_top, w_b, colors[0]),
        _rect_cells(h_top, w_c, colors[1]),
        _rect_cells(h_bottom, w_a + w_b, colors[2]),
    ]
    return out, pieces


def _build_u_output(dim_vals: tuple[int, int, int, int, int], colors: list[int]) -> tuple[np.ndarray, list[list[tuple[int, int, int]]]]:
    h_top, h_bottom, w_left, w_gap, w_right = dim_vals
    H = h_top + h_bottom
    W = w_left + w_gap + w_right
    out = np.zeros((H, W), dtype=int)
    out[:h_top, :w_left] = ANCHOR_COLOR
    out[:h_top, w_left + w_gap :] = colors[0]
    out[h_top:, :w_left] = colors[1]
    out[h_top:, w_left:] = colors[2]
    out[0, 0] = MARKER_COLOR
    pieces = [
        _rect_cells(h_top, w_left, ANCHOR_COLOR, marker=True),
        _rect_cells(h_top, w_right, colors[0]),
        _rect_cells(h_bottom, w_left, colors[1]),
        _rect_cells(h_bottom, w_gap + w_right, colors[2]),
    ]
    return out, pieces


def _sample_dims(shape_kind: str) -> tuple[int, int, int, int, int]:
    vals = [2, 3, 4, 5]
    if shape_kind == "L":
        h_top, h_bottom = random.sample(vals, 2)
        w_a, w_b, w_c = random.sample(vals, 3)
        return h_top, h_bottom, w_a, w_b, w_c
    h_top, h_bottom = random.sample(vals, 2)
    w_left, w_gap, w_right = random.sample(vals, 3)
    return h_top, h_bottom, w_left, w_gap, w_right


def _transform_pieces(pieces: list[list[tuple[int, int, int]]]) -> list[list[tuple[int, int, int]]]:
    return [random.choice(_variants(cells)) for cells in pieces]


def _scatter_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=int)
    occupied = np.zeros((CANVAS_H, CANVAS_W), dtype=bool)
    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        patch = patches[idx]
        h, w = patch.shape
        placed = False
        for _ in range(300):
            r0 = random.randint(0, CANVAS_H - h)
            c0 = random.randint(0, CANVAS_W - w)
            rr0 = max(0, r0 - 1)
            rr1 = min(CANVAS_H, r0 + h + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(CANVAS_W, c0 + w + 1)
            if np.any(occupied[rr0:rr1, cc0:cc1]):
                continue
            mask = patch != 0
            region = canvas[r0 : r0 + h, c0 : c0 + w]
            if np.any(region[mask] != 0):
                continue
            region[mask] = patch[mask]
            occupied[r0 : r0 + h, c0 : c0 + w][mask] = True
            placed = True
            break
        if not placed:
            return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    shape_kind = random.choice(["L", "U"])
    colors = random.sample(OTHER_COLORS, 3)
    dims = _sample_dims(shape_kind)
    if shape_kind == "L":
        expected, pieces = _build_l_output(dims, colors)
    else:
        expected, pieces = _build_u_output(dims, colors)

    transformed = _transform_pieces(pieces)
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
