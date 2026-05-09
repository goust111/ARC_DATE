# Combo variant of 7b3084d4 (index 8)
# combo_spec: D2/1+D4/2+D9/2
# Name: 非矩形锚定块双标记定向
#
# Design analysis
# - Concrete fusion:
#   The assembled target is still one tight rectangle, but the anchor is now
#   the only non-rectangular polyomino.  It forms a full-width top band with a
#   downward tab, so its two marker pixels can genuinely occupy output (0,0)
#   and (0,W-1).  The other three pieces remain plain rectangles that fill the
#   complement below.
# - Small example:
#   Original logic: pick the piece adjacent to a singleton marker, rotate it so
#   that the marker reaches the top-left corner, then pack four color-rectangles.
#   Fused logic: first find the most irregular connected component (bbox area
#   minus cell count is uniquely maximal), treat that as the anchor, then use
#   the ordered pair of marker colors to orient it so marker 5 is at the output
#   top-left and marker 2 is at the output top-right.  Only after that do we
#   place the three remaining pieces.
# - Complexity calibration:
#   This keeps one visually salient irregular piece instead of making all four
#   shapes arbitrary.  That avoids ambiguous packings while still breaking the
#   original rectangle-by-color assumption.  The tab position and size vary, but
#   only within a small family, so multiple samples still support the same rule.
# - Broken original assumptions:
#   The original solver assumed one color equals one piece, a single marker
#   identifies the anchor, and anchor orientation is a pure rotation.  None of
#   those survive here: the anchor is multi-color, shape identifies it, and the
#   generator may mirror the anchor before scattering it.
# - Placement legality:
#   The generator builds output first, extracts four pieces, applies valid
#   rotations / reflections, and scatters the resulting patches on a 20x20
#   canvas with padding via random_free_location, so pieces do not overlap or
#   touch.
# - Observability / cleanliness:
#   Human-solvable clues are all visible: one piece is conspicuously non-
#   rectangular, its two unique marker colors sit on the same edge, and the
#   solved output places them at the two upper corners.  No hidden ordering by
#   color id is required.

import random
from collections import deque

import numpy as np

from arc_utils import get_bg as _bg_color, is_contiguous, neighbors8, random_free_location


BG = 0
ANCHOR_COLOR = 7
MARKER_TL = 5
MARKER_TR = 2
COLOR_POOL = [1, 3, 4, 6, 8, 9]
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


def _reflect_cells(cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    return _normalize([(r, -c, col) for r, c, col in cells])


def _transform_cells(
    cells: list[tuple[int, int, int]], rotation: int = 0, reflect: bool = False
) -> list[tuple[int, int, int]]:
    pts = _reflect_cells(cells) if reflect else list(cells)
    return _rotate_cells(pts, rotation)


def _variants(
    cells: list[tuple[int, int, int]], allow_flip: bool = False
) -> list[list[tuple[int, int, int]]]:
    bases = [cells]
    if allow_flip:
        bases.append(_reflect_cells(cells))
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    outs: list[list[tuple[int, int, int]]] = []
    for base in bases:
        for k in range(4):
            var = _rotate_cells(base, k)
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


def _dims(area: int) -> list[tuple[int, int]]:
    out = []
    for h in range(1, int(area**0.5) + 1):
        if area % h == 0:
            w = area // h
            out.append((h, w))
            if h != w:
                out.append((w, h))
    out.sort(key=lambda t: (abs(t[0] - t[1]), t[0], t[1]))
    return out


def _component_bbox_area(cells: list[tuple[int, int, int]]) -> int:
    max_r = max(r for r, _, _ in cells)
    max_c = max(c for _, c, _ in cells)
    return (max_r + 1) * (max_c + 1)


def _extract_components(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
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
            for nr, nc in neighbors8(r, c):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] != bg:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        cells = _normalize(abs_cells)
        bbox_area = _component_bbox_area(cells)
        comps.append(
            {
                "cells": cells,
                "area": len(cells),
                "bbox_area": bbox_area,
                "irregularity": bbox_area - len(cells),
            }
        )
    return comps, bg


def _find_anchor_index(comps: list[dict]) -> int | None:
    if not comps:
        return None
    gaps = [int(comp["irregularity"]) for comp in comps]
    best = max(gaps)
    if best <= 0:
        return None
    idxs = [i for i, gap in enumerate(gaps) if gap == best]
    if len(idxs) != 1:
        return None
    return idxs[0]


def _marker_positions(cells: list[tuple[int, int, int]]) -> dict[int, tuple[int, int]]:
    pos: dict[int, tuple[int, int]] = {}
    for r, c, col in cells:
        if col in (MARKER_TL, MARKER_TR):
            if col in pos:
                return {}
            pos[col] = (r, c)
    return pos


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


def _all_placements(
    variant: list[tuple[int, int, int]], H: int, W: int
) -> list[list[tuple[int, int, int]]]:
    ph = max(r for r, _, _ in variant) + 1
    pw = max(c for _, c, _ in variant) + 1
    out = []
    for tr in range(H - ph + 1):
        for tc in range(W - pw + 1):
            out.append([(r + tr, c + tc, col) for r, c, col in variant])
    return out


def _anchor_placements(
    cells: list[tuple[int, int, int]], H: int, W: int
) -> list[list[tuple[int, int, int]]]:
    outs = []
    for var in _variants(cells, allow_flip=True):
        pos = _marker_positions(var)
        if set(pos) != {MARKER_TL, MARKER_TR}:
            continue
        (r0, c0) = pos[MARKER_TL]
        (r1, c1) = pos[MARKER_TR]
        if r0 != r1 or c1 <= c0:
            continue
        if c1 - c0 != W - 1:
            continue
        placed = [(r - r0, c - c0, col) for r, c, col in var]
        if all(0 <= r < H and 0 <= c < W for r, c, _ in placed):
            outs.append(placed)
    return outs


def _solve_packing(comps: list[dict], H: int, W: int) -> np.ndarray | None:
    anchor_idx = _find_anchor_index(comps)
    if anchor_idx is None:
        return None

    board = np.full((H, W), -1, dtype=int)
    piece_vars = []
    for i, comp in enumerate(comps):
        piece_vars.append(_variants(comp["cells"], allow_flip=(i == anchor_idx)))

    remaining = [i for i in range(len(comps)) if i != anchor_idx]
    for anchor_place in _anchor_placements(comps[anchor_idx]["cells"], H, W):
        if not _placeable(board, anchor_place):
            continue
        _apply(board, anchor_place)

        def backtrack(rem: list[int]) -> bool:
            if not rem:
                return True

            best_i = None
            best_opts = None
            for i in rem:
                opts = []
                for var in piece_vars[i]:
                    for placement in _all_placements(var, H, W):
                        if _placeable(board, placement):
                            opts.append(placement)
                if not opts:
                    return False
                if best_opts is None or len(opts) < len(best_opts):
                    best_i = i
                    best_opts = opts
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
            return board
        _apply(board, anchor_place, -1)
    return None


def _solve_from_components(comps: list[dict]) -> np.ndarray:
    area = sum(comp["area"] for comp in comps)
    for H, W in _dims(area):
        out = _solve_packing(comps, H, W)
        if out is not None:
            return out.astype(int)
    raise AssertionError("packing failed for all candidate dimensions")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble four scattered puzzle pieces into one rectangle when the anchor is the only irregular shape.

    Input structure:
    - Background is the most common color on a 20x20 canvas.
    - There are exactly four separated 8-connected components.
    - Three components are plain solid rectangles.
    - One component is a non-rectangular anchor polyomino and contains two marker pixels of colors 5 and 2.

    Rule:
    - Identify the anchor by shape, not by its fill color: it is the unique component whose bounding box contains holes or missing cells.
    - Try rotations and reflections of that anchor until marker 5 lands at output (0,0) and marker 2 lands at output (0,W-1) on the same top edge.
    - Then place the remaining three pieces, allowing 90-degree rotations, so all four pieces tile a complete rectangle with no overlap or gaps.

    Key insight:
    - The anchor is the most irregular component, and the ordered pair of marker pixels fixes its left-right orientation as well as its top edge, so packing only succeeds in the orientation where the two markers become the two upper corners.
    """
    comps, _bg = _extract_components(np.asarray(grid, dtype=int))
    assert len(comps) == 4, "input must contain exactly four separated pieces"
    anchor_idx = _find_anchor_index(comps)
    assert anchor_idx is not None, "input must contain one uniquely irregular anchor piece"
    pos = _marker_positions(comps[anchor_idx]["cells"])
    assert set(pos) == {MARKER_TL, MARKER_TR}, "anchor must contain exactly the two marker colors"
    return _solve_from_components(comps)


def _rect_cells(
    r0: int, c0: int, h: int, w: int, color: int
) -> list[tuple[int, int, int]]:
    return [(r, c, color) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]


def _sample_layout() -> dict:
    S = random.randint(5, 8)
    band_h = random.randint(2, S - 2)
    tab_depth = random.randint(1, S - band_h - 1)
    for _ in range(100):
        tab_w = random.randint(1, S - 2)
        tab_c = random.randint(1, S - tab_w - 1)
        left_w = tab_c
        right_w = S - (tab_c + tab_w)
        mid_h = S - (band_h + tab_depth)
        if left_w <= 0 or right_w <= 0 or mid_h <= 0:
            continue
        if left_w == right_w:
            continue
        return {
            "S": S,
            "band_h": band_h,
            "tab_depth": tab_depth,
            "tab_w": tab_w,
            "tab_c": tab_c,
        }
    raise RuntimeError("failed to sample asymmetric anchor layout")


def _build_output_and_pieces() -> tuple[np.ndarray, list[list[tuple[int, int, int]]], dict]:
    spec = _sample_layout()
    S = spec["S"]
    band_h = spec["band_h"]
    tab_depth = spec["tab_depth"]
    tab_w = spec["tab_w"]
    tab_c = spec["tab_c"]

    colors = random.sample(COLOR_POOL, 3)
    left_color, mid_color, right_color = colors

    output = np.zeros((S, S), dtype=int)
    anchor_mask = np.zeros((S, S), dtype=bool)
    anchor_mask[:band_h, :] = True
    anchor_mask[band_h : band_h + tab_depth, tab_c : tab_c + tab_w] = True
    output[anchor_mask] = ANCHOR_COLOR
    output[band_h:, :tab_c] = left_color
    output[band_h + tab_depth :, tab_c : tab_c + tab_w] = mid_color
    output[band_h:, tab_c + tab_w :] = right_color
    output[0, 0] = MARKER_TL
    output[0, S - 1] = MARKER_TR

    assert is_contiguous(anchor_mask.astype(int), background=0, connectivity=8)

    anchor_cells = []
    left_cells = []
    mid_cells = []
    right_cells = []

    for r in range(S):
        for c in range(S):
            val = int(output[r, c])
            if anchor_mask[r, c]:
                anchor_cells.append((r, c, val))
            elif band_h <= r and c < tab_c:
                left_cells.append((r, c, val))
            elif r >= band_h + tab_depth and tab_c <= c < tab_c + tab_w:
                mid_cells.append((r, c, val))
            elif band_h <= r and c >= tab_c + tab_w:
                right_cells.append((r, c, val))

    pieces = [
        _normalize(anchor_cells),
        _normalize(left_cells),
        _normalize(mid_cells),
        _normalize(right_cells),
    ]
    meta = {
        **spec,
        "irregularity": _component_bbox_area(pieces[0]) - len(pieces[0]),
    }
    return output, pieces, meta


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
        region = canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        mask = patch != 0
        region[mask] = patch[mask]
    return canvas


def _rotation_only_anchor_possible(cells: list[tuple[int, int, int]], width: int) -> bool:
    for var in _variants(cells, allow_flip=False):
        pos = _marker_positions(var)
        if set(pos) != {MARKER_TL, MARKER_TR}:
            continue
        (r0, c0) = pos[MARKER_TL]
        (r1, c1) = pos[MARKER_TR]
        if r0 == r1 and c1 > c0 and c1 - c0 == width - 1:
            placed = [(r - r0, c - c0, col) for r, c, col in var]
            if min(r for r, _, _ in placed) >= 0 and min(c for _, c, _ in placed) >= 0:
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    expected, pieces, meta = _build_output_and_pieces()

    anchor_reflect = random.choice([False, True])
    anchor_rot = random.randint(0, 3)
    anchor_in = _transform_cells(pieces[0], rotation=anchor_rot, reflect=anchor_reflect)
    other_in = [_rotate_cells(piece, random.randint(0, 3)) for piece in pieces[1:]]

    patches = [_cells_to_patch(anchor_in)] + [_cells_to_patch(piece) for piece in other_in]
    canvas = _scatter_patches(patches)
    if canvas is None:
        return None

    try:
        out = main(canvas)
    except Exception:
        return None
    if not np.array_equal(out, expected):
        return None

    # Prevent silent degeneration: require the anchor to be visibly irregular and
    # keep a healthy mix of mirror-needed instances.
    if meta["irregularity"] <= 0:
        return None
    if anchor_reflect and _rotation_only_anchor_possible(anchor_in, meta["S"]):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
