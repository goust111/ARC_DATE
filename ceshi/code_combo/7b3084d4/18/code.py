# Combo variant of 7b3084d4 (index 18)
# combo_spec: D2/2+D6/2+D9/0
# Name: 可变数量块+干扰过滤+最大面积锚定
#
# Design analysis
# a) Concrete behavior trace:
#    Original samples contain exactly four useful pieces plus a grey singleton.
#    The singleton picks the anchor, and all four colored pieces are always used.
#    In this combo, a typical input may contain six separated colored components:
#    four real pieces that tile a rectangle, plus two smaller distractors.  There
#    is no grey marker at all.  The solver must first extract 8-connected
#    components, detect the unique largest one as the anchor, keep that anchor in
#    its observed orientation, and then search subsets that include it until one
#    subset tiles a full rectangle.  The anchor's own top-left corner becomes
#    output (0,0), so removing the marker does not create orientation ambiguity.
# b) Complexity calibration:
#    Valid piece count varies from 3 to 6, but distractors are limited to 1 or 2.
#    Real pieces all have area >= 4, distractors have area <= 3, and the anchor
#    is forced to be the unique largest piece.  That keeps the change visible
#    while avoiding an explosion of equally plausible subsets.
# c) Hidden-assumption audit:
#    The original solver assumes one color corresponds to one whole piece, there
#    is a singleton marker, the anchor color is identified by marker adjacency,
#    and every non-background object must be used.  None of those assumptions
#    survive here.  This solver therefore works on connected components, ignores
#    color values when choosing the anchor, and enumerates candidate subsets.
# d) Placement legality:
#    The generator builds the output rectangle first, converts each piece into a
#    tight patch, randomly rotates only non-anchor valid pieces, adds distractor
#    patches, and scatters everything on a 20x20 canvas with one-cell padding.
# e) Visual observability:
#    The new behavior is visible in the pixels: there is no grey corner marker,
#    one piece is visibly larger than the others, and some extra tiny shapes are
#    present but absent from the output.  No invisible ordering by color is used.
# f) Visual cleanliness:
#    Every input component has a distinct solid color, pieces are separated by at
#    least one background cell, and distractors are deliberately small so the
#    useful assembly remains easy to read.
# g) Inferability:
#    Across a few samples, a human can see that the output always begins with the
#    largest input piece in the upper-left, while only a subset of the remaining
#    pieces are rotated and packed around it into a clean rectangle.
#
# main() assumption audit:
# - `_extract_components()` assumes each scattered object is isolated by at least
#   one background cell; `generate_input()` enforces that with padded placement.
# - `_find_anchor_index()` assumes the largest piece is unique; generation rejects
#   any tie for maximum area.
# - subset search assumes the true solution includes the anchor and has 3-6 real
#   pieces; generation always builds samples in that range and adds 1-2 distractors.
# - packing assumes only non-anchor pieces may rotate; the generator preserves the
#   anchor's observed orientation so the output is uniquely defined.

import itertools
import random
from collections import deque

import numpy as np

from arc_utils import get_bg as _bg_color, neighbors8, random_free_location


BG = 0
CANVAS_H = 20
CANVAS_W = 20
MAX_ATTEMPTS = 500
MIN_VALID_PIECES = 3
MAX_VALID_PIECES = 6
MIN_VALID_AREA = 4
MAX_DISTRACTORS = 2
COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]


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
        var = _rotate_cells(cells, k)
        key = tuple(var)
        if key not in seen:
            seen.add(key)
            out.append(var)
    return out


def _piece_dims(cells: list[tuple[int, int, int]]) -> tuple[int, int]:
    return max(r for r, _, _ in cells) + 1, max(c for _, c, _ in cells) + 1


def _cells_to_patch(cells: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = _piece_dims(cells)
    patch = np.zeros((h, w), dtype=int)
    for r, c, col in cells:
        patch[r, c] = col
    return patch


def _extract_components(grid: np.ndarray) -> tuple[list[dict], int]:
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
        local = _normalize(abs_cells)
        colors = sorted({col for _, _, col in local})
        comps.append(
            {
                "cells": local,
                "area": len(local),
                "color": int(colors[0]),
                "bbox": (
                    min(r for r, _, _ in abs_cells),
                    min(c for _, c, _ in abs_cells),
                    max(r for r, _, _ in abs_cells),
                    max(c for _, c, _ in abs_cells),
                ),
            }
        )
    return comps, bg


def _find_anchor_index(comps: list[dict]) -> int:
    assert comps, "input must contain at least one non-background component"
    max_area = max(comp["area"] for comp in comps)
    idxs = [i for i, comp in enumerate(comps) if comp["area"] == max_area]
    assert len(idxs) == 1, "largest piece must be unique"
    return idxs[0]


def _dims(area: int) -> list[tuple[int, int]]:
    out = []
    for h in range(1, int(area**0.5) + 1):
        if area % h == 0:
            w = area // h
            out.append((h, w))
            if h != w:
                out.append((w, h))
    out.sort(key=lambda hw: (abs(hw[0] - hw[1]), hw[0], hw[1]))
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


def _all_placements(cells: list[tuple[int, int, int]], H: int, W: int) -> list[list[tuple[int, int, int]]]:
    ph, pw = _piece_dims(cells)
    out: list[list[tuple[int, int, int]]] = []
    for tr in range(H - ph + 1):
        for tc in range(W - pw + 1):
            out.append([(r + tr, c + tc, col) for r, c, col in cells])
    return out


def _solve_subset(comps: list[dict], subset_idxs: tuple[int, ...], anchor_idx: int, H: int, W: int) -> np.ndarray | None:
    board = np.full((H, W), -1, dtype=int)
    anchor_cells = comps[anchor_idx]["cells"]
    ah, aw = _piece_dims(anchor_cells)
    if ah > H or aw > W:
        return None
    _apply(board, anchor_cells)

    remaining = [i for i in subset_idxs if i != anchor_idx]
    options: dict[int, list[list[tuple[int, int, int]]]] = {}
    for idx in remaining:
        placements: list[list[tuple[int, int, int]]] = []
        for var in _variants(comps[idx]["cells"]):
            placements.extend(_all_placements(var, H, W))
        options[idx] = placements

    def backtrack(rem: list[int]) -> bool:
        if not rem:
            return True
        best_idx = None
        best_opts = None
        for idx in rem:
            fits = [placement for placement in options[idx] if _placeable(board, placement)]
            if not fits:
                return False
            if best_opts is None or len(fits) < len(best_opts):
                best_idx = idx
                best_opts = fits
                if len(best_opts) == 1:
                    break
        assert best_idx is not None and best_opts is not None
        next_rem = [idx for idx in rem if idx != best_idx]
        for placement in best_opts:
            _apply(board, placement)
            if backtrack(next_rem):
                return True
            _apply(board, placement, -1)
        return False

    if backtrack(remaining):
        return board
    return None


def _ordered_subset_candidates(comps: list[dict], anchor_idx: int) -> list[tuple[int, ...]]:
    n = len(comps)
    non_anchor = [i for i in range(n) if i != anchor_idx]
    out: list[tuple[int, ...]] = []
    for size in range(min(MAX_VALID_PIECES, n), MIN_VALID_PIECES - 1, -1):
        subsets = list(itertools.combinations(non_anchor, size - 1))
        subsets.sort(key=lambda idxs: -sum(comps[i]["area"] for i in idxs))
        out.extend((anchor_idx,) + idxs for idxs in subsets)
    return out


def _solve_components(comps: list[dict]) -> tuple[np.ndarray, dict]:
    anchor_idx = _find_anchor_index(comps)
    for subset_idxs in _ordered_subset_candidates(comps, anchor_idx):
        area = sum(comps[i]["area"] for i in subset_idxs)
        for H, W in _dims(area):
            solved = _solve_subset(comps, subset_idxs, anchor_idx, H, W)
            if solved is not None:
                return solved.astype(int), {
                    "anchor_idx": anchor_idx,
                    "subset_idxs": subset_idxs,
                    "subset_colors": {comps[i]["color"] for i in subset_idxs},
                    "subset_size": len(subset_idxs),
                }
    raise AssertionError("no anchor-containing subset can tile a rectangle")


def _subset_can_pack(comps: list[dict], subset_idxs: tuple[int, ...], anchor_idx: int) -> bool:
    area = sum(comps[i]["area"] for i in subset_idxs)
    for H, W in _dims(area):
        if _solve_subset(comps, subset_idxs, anchor_idx, H, W) is not None:
            return True
    return False


def _partition_rect(h: int, w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    if count == 0:
        return []
    if h <= 0 or w <= 0:
        return None
    if h * w < count * MIN_VALID_AREA:
        return None
    if count == 1:
        return [(0, 0, h, w)]

    for _ in range(120):
        first_count = random.randint(1, count - 1)
        second_count = count - first_count
        choices = []
        if h >= 2:
            min_top = max(1, (first_count * MIN_VALID_AREA + w - 1) // w)
            max_top = h - max(1, (second_count * MIN_VALID_AREA + w - 1) // w)
            if min_top <= max_top:
                choices.append(("h", min_top, max_top))
        if w >= 2:
            min_left = max(1, (first_count * MIN_VALID_AREA + h - 1) // h)
            max_left = w - max(1, (second_count * MIN_VALID_AREA + h - 1) // h)
            if min_left <= max_left:
                choices.append(("v", min_left, max_left))
        if not choices:
            return None
        orient, lo, hi = random.choice(choices)
        cut = random.randint(lo, hi)
        if orient == "h":
            first = _partition_rect(cut, w, first_count)
            second = _partition_rect(h - cut, w, second_count)
            if first is None or second is None:
                continue
            shifted = [(r + cut, c, hh, ww) for r, c, hh, ww in second]
            return first + shifted
        first = _partition_rect(h, cut, first_count)
        second = _partition_rect(h, w - cut, second_count)
        if first is None or second is None:
            continue
        shifted = [(r, c + cut, hh, ww) for r, c, hh, ww in second]
        return first + shifted
    return None


def _build_layout(valid_count: int) -> tuple[np.ndarray, list[list[tuple[int, int, int]]]]:
    for _ in range(300):
        lo = 7 if valid_count <= 4 else 9
        H = random.randint(lo, 12)
        W = random.randint(lo, 12)
        top_count = random.randint(1, valid_count - 2)
        bottom_count = valid_count - 1 - top_count
        candidates = []
        for ah in range(2, H - 1):
            for aw in range(2, W - 1):
                if ah * (W - aw) < top_count * MIN_VALID_AREA:
                    continue
                if (H - ah) * W < bottom_count * MIN_VALID_AREA:
                    continue
                candidates.append((ah, aw))
        random.shuffle(candidates)
        for ah, aw in candidates:
            top_rects = _partition_rect(ah, W - aw, top_count)
            bottom_rects = _partition_rect(H - ah, W, bottom_count)
            if top_rects is None or bottom_rects is None:
                continue
            rects = [(0, 0, ah, aw)]
            rects.extend((r, c + aw, hh, ww) for r, c, hh, ww in top_rects)
            rects.extend((r + ah, c, hh, ww) for r, c, hh, ww in bottom_rects)
            areas = [hh * ww for _, _, hh, ww in rects]
            if areas[0] <= max(areas[1:]):
                continue
            colors = random.sample(COLORS, valid_count)
            output = np.zeros((H, W), dtype=int)
            pieces: list[list[tuple[int, int, int]]] = []
            for color, (r0, c0, hh, ww) in zip(colors, rects):
                output[r0:r0 + hh, c0:c0 + ww] = color
                pieces.append([(r, c, color) for r in range(hh) for c in range(ww)])
            return output, pieces
    raise RuntimeError("failed to build a valid rectangular layout")


def _make_distractor(color: int) -> list[tuple[int, int, int]]:
    templates = [
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (0, 2)],
    ]
    base = [(r, c, color) for r, c in random.choice(templates)]
    return random.choice(_variants(base))


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
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        region = canvas[r0:r0 + patch.shape[0], c0:c0 + patch.shape[1]]
        mask = patch != 0
        region[mask] = patch[mask]
    return canvas


def _fusion_checks(canvas: np.ndarray, expected: np.ndarray, valid_colors: set[int], valid_count: int) -> bool:
    comps, _bg = _extract_components(canvas)
    if len(comps) <= valid_count:
        return False
    try:
        solved, meta = _solve_components(comps)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False
    if meta["subset_colors"] != valid_colors:
        return False
    anchor_idx = meta["anchor_idx"]
    if meta["subset_size"] != valid_count:
        return False
    if _subset_can_pack(comps, tuple(range(len(comps))), anchor_idx):
        return False
    distractor_idxs = [i for i, comp in enumerate(comps) if comp["color"] not in valid_colors]
    non_anchor = [i for i in range(len(comps)) if i != anchor_idx]
    for size in range(MIN_VALID_PIECES, min(MAX_VALID_PIECES, len(comps)) + 1):
        for rest in itertools.combinations(non_anchor, size - 1):
            subset = (anchor_idx,) + rest
            if not any(i in distractor_idxs for i in subset):
                continue
            if _subset_can_pack(comps, subset, anchor_idx):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    valid_count = random.choice([3, 4, 5, 5, 6, 6])
    distractor_count = 1 if valid_count == 6 else random.randint(1, MAX_DISTRACTORS)
    output, pieces = _build_layout(valid_count)
    used_colors = {cells[0][2] for cells in pieces}
    free_colors = [color for color in COLORS if color not in used_colors]
    if len(free_colors) < distractor_count:
        return None

    transformed = [pieces[0]]
    transformed.extend(random.choice(_variants(cells)) for cells in pieces[1:])
    valid_colors = {cells[0][2] for cells in transformed}
    for color in random.sample(free_colors, distractor_count):
        transformed.append(_make_distractor(color))

    canvas = _scatter_patches([_cells_to_patch(cells) for cells in transformed])
    if canvas is None:
        return None
    if not _fusion_checks(canvas, output, valid_colors, valid_count):
        return None
    return canvas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble a subset of scattered colored pieces into one rectangle, using the largest piece as the anchor.

    Input structure:
    - Background is the most common color. The non-background objects are separated 8-connected components.
    - There are 3-6 genuine puzzle pieces plus 1-2 extra distractor pieces, all single-color and scattered apart.
    - No grey marker is present. The unique largest component is the anchor piece.

    Rule:
    - Extract all disconnected pieces and identify the largest one by area.
    - Keep that anchor in its observed orientation and place its own top-left corner at output (0,0).
    - Among all anchor-containing subsets, find the subset whose pieces can tile a full rectangle when the non-anchor pieces are allowed to rotate by 90 degrees.

    Key insight:
    - The missing marker is replaced by an area cue: the biggest piece defines both which subset matters and how the final rectangle is oriented.
    """
    comps, _bg = _extract_components(np.asarray(grid, dtype=int))
    solved, _meta = _solve_components(comps)
    return solved


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
