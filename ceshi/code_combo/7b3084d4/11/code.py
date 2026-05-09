# Combo variant of 7b3084d4 (index 11)
# combo_spec: D1/3+D6/1
# Name: 面积排序两阶段分组合并
#
# Design analysis
# a) Concrete behavior trace:
#    Original 7b3084d4 uses 4 interlocking tabbed pieces. The solver rotates
#    them and backtracks over geometric placements until the union becomes one
#    tight rectangle with the grey marker at output (0,0).
#    This combo removes interlock reasoning. Example: if the scattered piece
#    areas are 12, 8, 6, 4, then the largest two pieces form the top subgroup
#    and the smallest two pieces form the bottom subgroup. Each subgroup is
#    merged left-to-right into a row-rectangle, then the two row-rectangles are
#    stacked vertically. The anchor is the largest piece, so its grey corner
#    pixel becomes the top-left of the whole output.
# b) Complexity calibration:
#    Pure area sorting alone would be too weak if ties or square pieces created
#    orientation ambiguity. The generator therefore enforces 4 distinct areas,
#    makes the anchor strictly largest, and avoids square rectangles. Within
#    each pair, only the intended orientation gives equal heights, so the
#    two-stage merge is visible and stable.
# c) Implicit assumptions reviewed:
#    The original solver assumed color-separable pieces, tabbed silhouettes,
#    and a search over all packings. Those assumptions no longer hold. The new
#    solver instead assumes:
#    - exactly 4 separated 8-connected components,
#    - exactly one singleton grey marker inside one component,
#    - every component is a solid rectangle up to rotation,
#    - the anchor component is the unique largest piece,
#    - the top pair and bottom pair can each be oriented to share a row height,
#      and both rows have the same total width.
#    The generator constructs inputs that satisfy each assumption directly.
# d) Placement legality:
#    The generator is output-first. It samples a valid two-row rectangle layout,
#    extracts 4 rectangular pieces, rotates them independently, and scatters
#    them on a 20x20 canvas with padding via random_free_location so pieces
#    never overlap, touch, or leave the grid.
# e) Visual observability:
#    Across examples, solvers can see 4 isolated rectangles of clearly different
#    sizes. The output preserves the pieces but arranges them by area order:
#    largest/second-largest on the first row, third/fourth on the second row.
#    No hidden value ordering such as "color id rank" is used.
# f) Visual cleanliness:
#    Each piece is a solid monochrome rectangle except for the grey anchor
#    corner. The output is a clean filled rectangle assembled from four obvious
#    rectangular blocks, so the two-stage grouping is easy to read.
# g) Change inferability:
#    Colors only identify pieces; they do not drive sorting or recoloring.
#    Because size order and row grouping are both visible in every sample, a
#    human can infer the rule from a few examples.

import random
from collections import deque

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import neighbors8, random_free_location


BG = 0
ANCHOR_COLOR = 7
MARKER_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 8, 9]
CANVAS_H = 20
CANVAS_W = 20
PLACEMENT_PADDING = 1
MAX_ATTEMPTS = 400


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


def _components(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = int(_bg_color(g))
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
        comps.append(
            {
                "cells": _normalize(abs_cells),
                "area": len(abs_cells),
            }
        )
    return comps, bg


def _anchor_index(pieces: list[dict]) -> int:
    idxs = [i for i, p in enumerate(pieces) if any(col == MARKER_COLOR for _, _, col in p["cells"])]
    assert len(idxs) == 1, "expected exactly one marker-bearing anchor piece"
    return idxs[0]


def _rect_orientations(
    piece: dict,
    *,
    marker_top_left: bool = False,
) -> list[dict]:
    outs: list[dict] = []
    seen_dims: set[tuple[int, int]] = set()
    for var in _variants(piece["cells"]):
        h = max(r for r, _, _ in var) + 1
        w = max(c for _, c, _ in var) + 1
        if len(var) != h * w:
            continue
        if marker_top_left:
            marker_pts = [(r, c) for r, c, col in var if col == MARKER_COLOR]
            if len(marker_pts) != 1 or marker_pts[0] != (0, 0):
                continue
        dims = (h, w)
        if dims in seen_dims:
            continue
        seen_dims.add(dims)
        outs.append({"cells": var, "h": h, "w": w})
    assert outs, "piece is not a rotatable solid rectangle"
    return outs


def _paint(
    board: np.ndarray,
    cells: list[tuple[int, int, int]],
    top: int,
    left: int,
) -> None:
    for r, c, col in cells:
        board[top + r, left + c] = col


def _assemble_rows(pieces: list[dict]) -> np.ndarray:
    assert len(pieces) == 4, "expected exactly four pieces"
    anchor_idx = _anchor_index(pieces)
    areas = [int(p["area"]) for p in pieces]
    assert len(set(areas)) == 4, "piece areas must be distinct"

    ordered = sorted(
        range(4),
        key=lambda i: (-int(pieces[i]["area"]), 0 if i == anchor_idx else 1),
    )
    assert ordered[0] == anchor_idx, "the anchor must be the largest piece in this combo"

    row0 = [pieces[ordered[0]], pieces[ordered[1]]]
    row1 = [pieces[ordered[2]], pieces[ordered[3]]]

    row0_a = _rect_orientations(row0[0], marker_top_left=True)
    row0_b = _rect_orientations(row0[1])
    row1_a = _rect_orientations(row1[0])
    row1_b = _rect_orientations(row1[1])

    for a0 in row0_a:
        for b0 in row0_b:
            if a0["h"] != b0["h"]:
                continue
            top_h = int(a0["h"])
            width = int(a0["w"]) + int(b0["w"])
            for a1 in row1_a:
                for b1 in row1_b:
                    if a1["h"] != b1["h"]:
                        continue
                    if int(a1["w"]) + int(b1["w"]) != width:
                        continue
                    bot_h = int(a1["h"])
                    out = np.zeros((top_h + bot_h, width), dtype=int)
                    _paint(out, a0["cells"], 0, 0)
                    _paint(out, b0["cells"], 0, int(a0["w"]))
                    _paint(out, a1["cells"], top_h, 0)
                    _paint(out, b1["cells"], top_h, int(a1["w"]))
                    return out

    raise AssertionError("could not orient the area-ranked groups into two equal-width row rectangles")


def _rect_cells(h: int, w: int, color: int, *, marker: bool = False) -> list[tuple[int, int, int]]:
    cells = []
    for r in range(h):
        for c in range(w):
            col = MARKER_COLOR if marker and (r, c) == (0, 0) else color
            cells.append((r, c, col))
    return cells


def _cells_to_patch(cells: list[tuple[int, int, int]]) -> np.ndarray:
    h = max(r for r, _, _ in cells) + 1
    w = max(c for _, c, _ in cells) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c, col in cells:
        patch[r, c] = col
    return patch


def _scatter_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=int)
    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        patch = patches[idx]
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
        region = canvas[r0:r0 + patch.shape[0], c0:c0 + patch.shape[1]]
        mask = patch != 0
        region[mask] = patch[mask]
    return canvas


def _sample_layout() -> tuple[dict, list[int]] | None:
    for _ in range(200):
        width = random.randint(4, 8)
        top_h = random.randint(2, 6)
        bot_h = random.randint(1, min(4, top_h))
        top_left_w = random.randint(width // 2 + 1, width - 1)
        bot_left_w = random.randint(width // 2 + 1, width - 1)

        dims = [
            (top_h, top_left_w),
            (top_h, width - top_left_w),
            (bot_h, bot_left_w),
            (bot_h, width - bot_left_w),
        ]
        areas = [h * w for h, w in dims]

        if top_h + bot_h == width:
            continue
        if areas != sorted(areas, reverse=True):
            continue
        if len(set(areas)) != 4:
            continue
        if any(h == w for h, w in dims):
            continue
        if len({dims[0][0], dims[0][1], dims[1][1]}) != 3:
            continue
        if len({dims[2][0], dims[2][1], dims[3][1]}) != 3:
            continue
        return {
            "width": width,
            "top_h": top_h,
            "bot_h": bot_h,
            "top_left_w": top_left_w,
            "bot_left_w": bot_left_w,
        }, areas
    return None


def _build_target(layout: dict) -> tuple[np.ndarray, list[list[tuple[int, int, int]]], dict]:
    pool = COLOR_POOL.copy()
    random.shuffle(pool)
    colors = [ANCHOR_COLOR, pool[0], pool[1], pool[2]]

    top_h = int(layout["top_h"])
    bot_h = int(layout["bot_h"])
    width = int(layout["width"])
    top_left_w = int(layout["top_left_w"])
    bot_left_w = int(layout["bot_left_w"])

    pieces = [
        _rect_cells(top_h, top_left_w, colors[0], marker=True),
        _rect_cells(top_h, width - top_left_w, colors[1]),
        _rect_cells(bot_h, bot_left_w, colors[2]),
        _rect_cells(bot_h, width - bot_left_w, colors[3]),
    ]

    out = np.zeros((top_h + bot_h, width), dtype=int)
    _paint(out, pieces[0], 0, 0)
    _paint(out, pieces[1], 0, top_left_w)
    _paint(out, pieces[2], top_h, 0)
    _paint(out, pieces[3], top_h, bot_left_w)

    meta = {
        "areas": [len(p) for p in pieces],
        "row_heights": (top_h, bot_h),
        "width": width,
        "splits": (top_left_w, bot_left_w),
    }
    return out, pieces, meta


def _fusion_signature(grid: np.ndarray) -> dict:
    pieces, _bg = _components(np.asarray(grid, dtype=int))
    ordered = sorted((int(p["area"]) for p in pieces), reverse=True)
    out = main(grid)
    h, w = out.shape
    marker_pos = tuple(map(int, np.argwhere(out == MARKER_COLOR)[0]))
    top_h = 0
    for r in range(h):
        if out[r, 0] == out[r, 1 if w > 1 else 0]:
            top_h += 1
        else:
            break
    return {
        "areas_desc": tuple(ordered),
        "output_shape": (int(h), int(w)),
        "marker": marker_pos,
        "top_h": int(top_h),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble four scattered rectangles by area rank, using a two-stage row merge.

    Input structure:
    - Background is the most common color on a 20x20 canvas.
    - Exactly four separated 8-connected pieces appear on the canvas.
    - One piece contains a unique grey corner pixel; that marked piece is the anchor.
    - Every piece is a solid rectangle up to rotation, and the four areas are all different.

    Rule:
    - Extract the four pieces and rank them by area from largest to smallest.
    - Split that order into two groups: the two largest pieces and the two smallest pieces.
    - For each group, orient the two rectangles so they share a common height and place them side by side to form a row-rectangle.
    - Put the larger-area row on top, the smaller-area row below it, and orient the anchor so its grey corner lands at output position (0,0).

    Key insight:
    - The final arrangement is determined by numeric size order, not by tab/slot interlock. The puzzle is solved by sorting, grouping, and performing the required two-stage rectangle merge.
    """
    pieces, _bg = _components(np.asarray(grid, dtype=int))
    assert pieces, "input must contain non-background pieces"
    return _assemble_rows(pieces)


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    sampled = _sample_layout()
    if sampled is None:
        return None
    layout, _areas = sampled
    expected, pieces, meta = _build_target(layout)
    rotated = [_rotate_cells(piece, random.randint(0, 3)) for piece in pieces]
    canvas = _scatter_patches([_cells_to_patch(piece) for piece in rotated])
    if canvas is None:
        return None
    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return canvas, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("generate_input: exhausted attempts")
