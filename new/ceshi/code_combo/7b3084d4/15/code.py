# Combo variant of 7b3084d4 (index 15)
# combo_spec: D9/0+D11/0+D12/2
# Name: 空间关系排列+最大面积锚定+颜色编码象限
#
# Design analysis
# a) Concrete behavior:
# - Original: a singleton grey marker identifies the anchor, pieces are randomly
#   scattered, and the solver backtracks over rectangle packings.
# - Here: there is no marker at all. The four tabbed pieces are arranged in a
#   visible 2x2 meta-layout on the 20x20 canvas. The unique largest-area piece
#   is the anchor. Colors 1/2/3/4 encode TL/TR/BL/BR. Example trace:
#   input top-left slot holds color 2, top-right holds color 1 and is largest,
#   bottom-left holds color 4, bottom-right holds color 3. The solver first
#   reads this ordered layout, then uses area to confirm the anchor, then uses
#   the color code to assign output quadrants, and finally searches only over
#   rotations while each piece is placed flush to its designated corner.
# b) Complexity calibration:
# - The pieces still come from the original tabbed quadrant construction, so the
#   geometry is nontrivial and rotations matter.
# - The layout rule has one repeated, sample-visible pattern instead of a large
#   family of hidden permutations; this keeps the relation learnable from a few
#   examples.
# - Using only colors 1..4 avoids an overlarge palette while still making the
#   quadrant code immediately observable.
# c) Implicit-assumption audit:
# - Any logic that looked for a singleton marker is removed.
# - Any logic that assumed random scattering is replaced by explicit component
#   row/column slot parsing.
# - Rectangle packing by arbitrary translation is replaced by a smaller search:
#   once quadrant ownership is known, each rotated piece has exactly one corner
#   translation candidate in the final square.
# d) Placement legality:
# - Generator builds the output first, extracts four connected pieces, rotates
#   them, and places them in non-overlapping 2x2 slots with fixed gaps inside
#   the 20x20 input.
# - Tab depths are capped so every piece bbox stays compact enough for this
#   layout to fit the canvas up to N=7.
# e) Visual observability:
# - Across examples a human sees three simultaneous changes: no grey marker,
#   one piece is visibly larger than the others, and the input is no longer a
#   random scatter but a clean 2x2 arrangement with stable color roles.
# f) Visual cleanliness:
# - Objects never overlap or touch across slots, and only four foreground
#   colors are used, so the grid stays legible.
# g) Inferable change:
# - The color rule is fixed and repeated: 1=TL, 2=TR, 3=BL, 4=BR.
# - The spatial rule is also fixed and repeated: the input meta-layout is
#   [TR, TL] on the top row and [BR, BL] on the bottom row.

import itertools
import math
import random
from collections import deque

import numpy as np

from arc_utils import get_bg as _bg_color, is_contiguous, neighbors8


BG = 0
CANVAS_H = 20
CANVAS_W = 20
N_MIN = 2
N_MAX = 7
MAX_ATTEMPTS = 500
SLOT_GAP = 2

TL, TR, BL, BR = 0, 1, 2, 3
COLOR_TO_QUAD = {1: TL, 2: TR, 3: BL, 4: BR}
QUAD_TO_COLOR = {quad: color for color, quad in COLOR_TO_QUAD.items()}
INPUT_SLOT_ORDER = {
    "top_left": TR,
    "top_right": TL,
    "bottom_left": BR,
    "bottom_right": BL,
}


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
    out: list[list[tuple[int, int, int]]] = []
    for k in range(4):
        rot = _rotate_cells(cells, k)
        key = tuple(rot)
        if key not in seen:
            seen.add(key)
            out.append(rot)
    return out


def _bbox(abs_cells: list[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _, _ in abs_cells]
    cs = [c for _, c, _ in abs_cells]
    return min(rs), min(cs), max(rs), max(cs)


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
            for nr, nc in neighbors8(r, c):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] != bg:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        colors = {col for _, _, col in abs_cells}
        assert len(colors) == 1, "each piece must be single-color in this variant"
        b = _bbox(abs_cells)
        comps.append(
            {
                "cells": _normalize(abs_cells),
                "abs": abs_cells,
                "bbox": b,
                "area": len(abs_cells),
                "color": next(iter(colors)),
                "center": ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0),
            }
        )
    return comps, bg


def _unique_anchor_index(comps: list[dict]) -> int:
    areas = [comp["area"] for comp in comps]
    max_area = max(areas)
    assert areas.count(max_area) == 1, "largest-area anchor must be unique"
    return areas.index(max_area)


def _layout_slots(comps: list[dict]) -> dict[str, int]:
    assert len(comps) == 4, "input must contain exactly four separated pieces"
    by_row = sorted(range(4), key=lambda i: (comps[i]["center"][0], comps[i]["center"][1]))
    top = sorted(by_row[:2], key=lambda i: comps[i]["center"][1])
    bottom = sorted(by_row[2:], key=lambda i: comps[i]["center"][1])
    assert max(comps[i]["center"][0] for i in top) < min(comps[i]["center"][0] for i in bottom), (
        "input pieces must form a visible two-row arrangement"
    )
    assert comps[top[0]]["center"][1] < comps[top[1]]["center"][1], "top row must have distinct left/right order"
    assert comps[bottom[0]]["center"][1] < comps[bottom[1]]["center"][1], (
        "bottom row must have distinct left/right order"
    )
    return {
        "top_left": top[0],
        "top_right": top[1],
        "bottom_left": bottom[0],
        "bottom_right": bottom[1],
    }


def _quadrant_roles_from_layout(slots: dict[str, int], anchor_idx: int) -> dict[int, int]:
    assert slots["top_right"] == anchor_idx, "largest anchor must occupy the input top-right slot"
    return {
        anchor_idx: TL,
        slots["top_left"]: TR,
        slots["bottom_right"]: BL,
        slots["bottom_left"]: BR,
    }


def _piece_dims(cells: list[tuple[int, int, int]]) -> tuple[int, int]:
    return max(r for r, _, _ in cells) + 1, max(c for _, c, _ in cells) + 1


def _corner_offset(quad: int, h: int, w: int, side: int) -> tuple[int, int]:
    if quad == TL:
        return 0, 0
    if quad == TR:
        return 0, side - w
    if quad == BL:
        return side - h, 0
    return side - h, side - w


def _place_piece(board: np.ndarray, cells: list[tuple[int, int, int]], quad: int) -> bool:
    side = board.shape[0]
    h, w = _piece_dims(cells)
    tr, tc = _corner_offset(quad, h, w, side)
    for r, c, _ in cells:
        rr, cc = r + tr, c + tc
        if not (0 <= rr < side and 0 <= cc < side) or board[rr, cc] != -1:
            return False
    for r, c, col in cells:
        board[r + tr, c + tc] = col
    return True


def _solve_from_corner_roles(piece_by_quad: dict[int, list[tuple[int, int, int]]]) -> np.ndarray:
    total_area = sum(len(cells) for cells in piece_by_quad.values())
    side = math.isqrt(total_area)
    assert side * side == total_area, "piece areas must tile a square"
    assert len(piece_by_quad) == 4, "must recover all four quadrants"

    order = [TL, TR, BL, BR]
    variant_lists = [_variants(piece_by_quad[q]) for q in order]
    for choice in itertools.product(*variant_lists):
        board = np.full((side, side), -1, dtype=int)
        ok = True
        for quad, cells in zip(order, choice):
            if not _place_piece(board, cells, quad):
                ok = False
                break
        if ok and np.all(board != -1):
            return board.astype(int)
    raise AssertionError("no rotation assignment tiled the final square")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble four rotated tabbed pieces into one solid square by combining ordered layout, area anchoring, and quadrant colors.

    Input structure:
    - Background is the most common color. There are exactly four separated 8-connected single-color pieces using colors 1, 2, 3, and 4.
    - The pieces are not randomly scattered: they occupy a visible 2x2 input layout. The top row contains the two top-output pieces, the bottom row contains the two bottom-output pieces, and the unique largest piece sits in the input's top-right slot.

    Rule:
    - Split the input into four components and identify the unique largest-area component; that piece is the anchor and must become the output's top-left piece.
    - Read the structured input layout around that anchor: the piece left of it becomes the output's top-right part, the piece below it becomes bottom-left, and the diagonal piece becomes bottom-right.
    - Use the fixed color code 1=TL, 2=TR, 3=BL, 4=BR to confirm those inferred quadrant roles before rotating each piece into place.
    - Rotate each piece as needed and place it flush against its designated output corner; the correct rotation set is the one that tiles the square with no gaps or overlaps.

    Key insight:
    - The old grey corner marker is gone. The anchor is now recognized by being largest, the input arrangement advertises a stable spatial pattern around that anchor, and the colors verify the recovered quadrant labels.
    """
    comps, _bg = _components(np.asarray(grid, dtype=int))
    assert len(comps) == 4, "input must contain exactly four pieces"

    anchor_idx = _unique_anchor_index(comps)
    slots = _layout_slots(comps)
    layout_roles = _quadrant_roles_from_layout(slots, anchor_idx)

    piece_by_quad: dict[int, list[tuple[int, int, int]]] = {}
    for comp_idx, quad in layout_roles.items():
        comp = comps[comp_idx]
        color = int(comp["color"])
        assert color in COLOR_TO_QUAD, "colors must be the quadrant code 1/2/3/4"
        color_quad = COLOR_TO_QUAD[color]
        assert color_quad == quad, "input layout, anchor area, and color code must agree"
        assert quad not in piece_by_quad, "quadrant roles must be unique"
        piece_by_quad[quad] = comp["cells"]

    return _solve_from_corner_roles(piece_by_quad)


def _build_ownership(N: int, p: int, hl: int, hr: int, vt: int, vb: int) -> np.ndarray | None:
    side = 2 * N
    own = np.zeros((side, side), dtype=int)
    own[:N, :N] = TL
    own[:N, N:] = TR
    own[N:, :N] = BL
    own[N:, N:] = BR

    # TL grows downward; BR grows upward.
    own[N : N + hl, p] = TL
    own[N - hr : N, side - 1 - p] = BR

    # TL grows rightward; BL grows rightward.
    own[p, N : N + vt] = TL
    own[side - 1 - p, N : N + vb] = BL

    for quad in (TL, TR, BL, BR):
        mask = (own == quad).astype(int)
        if mask.sum() == 0 or not is_contiguous(mask, background=0, connectivity=8):
            return None
    return own


def _extract_piece(own: np.ndarray, quad: int) -> list[tuple[int, int, int]]:
    rows, cols = np.where(own == quad)
    min_r, min_c = int(rows.min()), int(cols.min())
    color = QUAD_TO_COLOR[quad]
    cells = [(int(r) - min_r, int(c) - min_c, color) for r, c in zip(rows, cols)]
    cells.sort()
    return cells


def _cells_to_patch(cells: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = _piece_dims(cells)
    patch = np.zeros((h, w), dtype=int)
    for r, c, col in cells:
        patch[r, c] = col
    return patch


def _ownership_areas(own: np.ndarray) -> dict[int, int]:
    return {quad: int(np.sum(own == quad)) for quad in (TL, TR, BL, BR)}


def _place_patches_in_slots(patches: dict[int, np.ndarray]) -> np.ndarray | None:
    tl = patches[TR]
    tr = patches[TL]
    bl = patches[BR]
    br = patches[BL]

    left_col_w = max(tl.shape[1], bl.shape[1])
    right_col_w = max(tr.shape[1], br.shape[1])
    top_row_h = max(tl.shape[0], tr.shape[0])
    bottom_row_h = max(bl.shape[0], br.shape[0])
    total_h = top_row_h + SLOT_GAP + bottom_row_h
    total_w = left_col_w + SLOT_GAP + right_col_w
    if total_h > CANVAS_H or total_w > CANVAS_W:
        return None

    r0 = random.randint(0, CANVAS_H - total_h)
    c0 = random.randint(0, CANVAS_W - total_w)
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=int)
    placements = {
        TR: (r0, c0),
        TL: (r0, c0 + left_col_w + SLOT_GAP),
        BR: (r0 + top_row_h + SLOT_GAP, c0),
        BL: (r0 + top_row_h + SLOT_GAP, c0 + left_col_w + SLOT_GAP),
    }

    for quad, patch in patches.items():
        pr, pc = placements[quad]
        region = canvas[pr : pr + patch.shape[0], pc : pc + patch.shape[1]]
        mask = patch != 0
        if np.any(region[mask] != 0):
            return None
        region[mask] = patch[mask]
    return canvas


def _random_ownership() -> tuple[np.ndarray, np.ndarray] | None:
    N = random.randint(N_MIN, N_MAX)
    p = random.randint(1, N - 1)
    max_depth = min(2, N - 1)

    for _ in range(80):
        hl = random.randint(1, max_depth)
        hr = random.randint(1, max_depth)
        vt = random.randint(1, max_depth)
        vb = random.randint(1, max_depth)
        own = _build_ownership(N, p, hl, hr, vt, vb)
        if own is None:
            continue
        areas = _ownership_areas(own)
        if areas[TL] != max(areas.values()) or list(areas.values()).count(areas[TL]) != 1:
            continue
        output = np.zeros((2 * N, 2 * N), dtype=int)
        for quad, color in QUAD_TO_COLOR.items():
            output[own == quad] = color
        return own, output
    return None


def _gen_attempt() -> np.ndarray | None:
    built = _random_ownership()
    if built is None:
        return None
    own, output = built

    pieces = {quad: _extract_piece(own, quad) for quad in (TL, TR, BL, BR)}
    rotated = {quad: random.choice(_variants(cells)) for quad, cells in pieces.items()}
    patches = {quad: _cells_to_patch(rotated[quad]) for quad in rotated}
    canvas = _place_patches_in_slots(patches)
    if canvas is None:
        return None

    try:
        solved = main(canvas)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
