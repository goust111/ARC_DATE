# Combo variant of 7b3084d4 (index 12)
# combo_spec: D8/2+D15/2
# Name: 十字分割单方向tab拼图
#
# Design analysis for the fused variant
# a) Concrete behavior:
#    The assembled target is now a genuine 5-piece cross partition:
#    a full-width top strip, a middle row split into left / center / right,
#    and a full-width bottom strip.  The center piece is the anchor.
#    Only the left-center and center-right seams carry tabs; the top-middle
#    and middle-bottom seams are straight.
# b) Complexity calibration:
#    The visible structure repeats cleanly across examples: two wide strips,
#    one center anchor, and exactly two horizontal-direction interlocks.
# c) Broken original assumptions:
#    - the puzzle no longer has 4 quadrant pieces
#    - the anchor is no longer an outer corner piece
#    - tabs no longer exist on both axes
# d) Placement legality:
#    The generator builds the solved rectangle first, extracts five
#    8-connected pieces, rotates them independently, and scatters them on a
#    20x20 canvas with a one-cell moat.
# e) Visual observability:
#    Humans can infer the rule from a few samples because the center piece is
#    always flanked by the only tabbed seams, while the top and bottom seams
#    stay perfectly straight.
# f) Clean visualization:
#    Each piece keeps one dominant color, only the center anchor has the grey
#    marker, and there are no distractor annotations.
# g) Inferability of changes:
#    There is no hidden color remapping.  The rule is purely geometric:
#    cross-partition topology plus single-axis tabs.

import itertools
import random
from collections import deque

import numpy as np

from arc_utils import get_bg as _bg_color, is_contiguous, neighbors8, random_free_location


BG = 0
ANCHOR_COLOR = 7
MARKER_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 8, 9]

CANVAS_H = 20
CANVAS_W = 20
PLACEMENT_PADDING = 1
MAX_ATTEMPTS = 500

CENTER = 0
TOP = 1
LEFT = 2
RIGHT = 3
BOTTOM = 4

STRIP_MIN = 2
STRIP_MAX = 4
MID_H_MIN = 3
MID_H_MAX = 5
MID_W_MIN = 2
MID_W_MAX = 4
SIDE_W_MIN = 2
SIDE_W_MAX = 4


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


def _shift_cells(
    cells: list[tuple[int, int, int]], dr: int, dc: int
) -> list[tuple[int, int, int]]:
    return [(r + dr, c + dc, col) for r, c, col in cells]


def _cells_to_patch(cells: list[tuple[int, int, int]]) -> np.ndarray:
    h = max(r for r, _, _ in cells) + 1
    w = max(c for _, c, _ in cells) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c, col in cells:
        patch[r, c] = col
    return patch


def _components(g: np.ndarray) -> tuple[list[dict], int]:
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
        comps.append({"cells": _normalize(abs_cells), "abs": abs_cells})
    return comps, bg


def _marker_pos(cells: list[tuple[int, int, int]]) -> tuple[int, int] | None:
    pts = [(r, c) for r, c, col in cells if col == MARKER_COLOR]
    if len(pts) == 1:
        return pts[0]
    return None


def _find_anchor_index(comps: list[dict]) -> int:
    idxs = [i for i, comp in enumerate(comps) if _marker_pos(comp["cells"]) is not None]
    assert len(idxs) == 1, "input must contain exactly one marker-bearing anchor piece"
    return idxs[0]


def _row_profile(cells: list[tuple[int, int, int]]) -> dict | None:
    h = max(r for r, _, _ in cells) + 1
    w = max(c for _, c, _ in cells) + 1
    rows: list[list[int]] = [[] for _ in range(h)]
    for r, c, _ in cells:
        rows[r].append(c)
    mins: list[int] = []
    maxs: list[int] = []
    for cols in rows:
        if not cols:
            return None
        cols = sorted(cols)
        if cols != list(range(cols[0], cols[-1] + 1)):
            return None
        mins.append(cols[0])
        maxs.append(cols[-1])
    return {"h": h, "w": w, "mins": mins, "maxs": maxs}


def _is_rectangle_profile(prof: dict | None) -> bool:
    return prof is not None and all(v == 0 for v in prof["mins"]) and all(
        v == prof["w"] - 1 for v in prof["maxs"]
    )


def _row_union_is_contiguous(
    mins_a: list[int], maxs_a: list[int], mins_b: list[int], maxs_b: list[int]
) -> bool:
    for a0, a1, b0, b1 in zip(mins_a, maxs_a, mins_b, maxs_b):
        merged_min = min(a0, b0)
        merged_max = max(a1, b1)
        count = (a1 - a0 + 1) + (b1 - b0 + 1)
        if count != merged_max - merged_min + 1:
            return False
    return True


def _match_left_center(
    left_cells: list[tuple[int, int, int]], center_cells: list[tuple[int, int, int]]
) -> dict | None:
    left = _row_profile(left_cells)
    center = _row_profile(center_cells)
    if left is None or center is None or left["h"] != center["h"]:
        return None
    if any(v != 0 for v in left["mins"]):
        return None

    shifts = {left["maxs"][r] + 1 - center["mins"][r] for r in range(left["h"])}
    if len(shifts) != 1:
        return None
    anchor_col = next(iter(shifts))
    if anchor_col <= 0:
        return None

    center_mins = [v + anchor_col for v in center["mins"]]
    center_maxs = [v + anchor_col for v in center["maxs"]]
    if not _row_union_is_contiguous(left["mins"], left["maxs"], center_mins, center_maxs):
        return None

    return {"anchor_col": anchor_col, "left": list(left_cells)}


def _match_center_right(
    center_cells: list[tuple[int, int, int]], right_cells: list[tuple[int, int, int]]
) -> dict | None:
    center = _row_profile(center_cells)
    right = _row_profile(right_cells)
    if center is None or right is None or center["h"] != right["h"]:
        return None
    if any(v != right["w"] - 1 for v in right["maxs"]):
        return None

    shifts = {center["maxs"][r] + 1 - right["mins"][r] for r in range(center["h"])}
    if len(shifts) != 1:
        return None
    right_col = next(iter(shifts))
    if right_col <= 0:
        return None

    right_mins = [v + right_col for v in right["mins"]]
    right_maxs = [v + right_col for v in right["maxs"]]
    if not _row_union_is_contiguous(center["mins"], center["maxs"], right_mins, right_maxs):
        return None

    mid_width = max(center["w"], right_col + right["w"])
    return {"right_col": right_col, "mid_width": mid_width, "right": _shift_cells(right_cells, 0, right_col)}


def _paint(
    board: np.ndarray, cells: list[tuple[int, int, int]], dr: int, dc: int
) -> bool:
    for r, c, _ in cells:
        rr, cc = r + dr, c + dc
        if not (0 <= rr < board.shape[0] and 0 <= cc < board.shape[1]):
            return False
        if board[rr, cc] != -1:
            return False
    for r, c, col in cells:
        board[r + dr, c + dc] = col
    return True


def _solve_cross_tabs(comps: list[dict]) -> np.ndarray:
    assert len(comps) == 5, "variant expects exactly five scattered pieces"
    anchor_idx = _find_anchor_index(comps)
    piece_vars = [_variants(comp["cells"]) for comp in comps]

    anchor_candidates: list[tuple[list[tuple[int, int, int]], dict]] = []
    for var in piece_vars[anchor_idx]:
        if _marker_pos(var) != (0, 0):
            continue
        prof = _row_profile(var)
        if prof is None:
            continue
        if prof["mins"][0] != 0 or prof["maxs"][0] != prof["w"] - 1:
            continue
        if prof["mins"][-1] != 0 or prof["maxs"][-1] != prof["w"] - 1:
            continue
        anchor_candidates.append((var, prof))
    assert anchor_candidates, "anchor must rotate so its marker is the center piece's top-left"

    others = [i for i in range(len(comps)) if i != anchor_idx]
    for anchor, anchor_prof in anchor_candidates:
        for left_idx, right_idx in itertools.permutations(others, 2):
            left_rest = [i for i in others if i not in (left_idx, right_idx)]
            for left_var in piece_vars[left_idx]:
                left_match = _match_left_center(left_var, anchor)
                if left_match is None:
                    continue
                for right_var in piece_vars[right_idx]:
                    right_match = _match_center_right(anchor, right_var)
                    if right_match is None:
                        continue
                    total_width = left_match["anchor_col"] + right_match["mid_width"]
                    if total_width <= anchor_prof["w"]:
                        continue
                    for top_idx, bottom_idx in itertools.permutations(left_rest, 2):
                        for top_var in piece_vars[top_idx]:
                            top_prof = _row_profile(top_var)
                            if not _is_rectangle_profile(top_prof) or top_prof["w"] != total_width:
                                continue
                            for bottom_var in piece_vars[bottom_idx]:
                                bottom_prof = _row_profile(bottom_var)
                                if not _is_rectangle_profile(bottom_prof) or bottom_prof["w"] != total_width:
                                    continue

                                out = np.full(
                                    (top_prof["h"] + anchor_prof["h"] + bottom_prof["h"], total_width),
                                    -1,
                                    dtype=int,
                                )
                                placements = [
                                    (top_var, 0, 0),
                                    (left_var, top_prof["h"], 0),
                                    (anchor, top_prof["h"], left_match["anchor_col"]),
                                    (
                                        right_var,
                                        top_prof["h"],
                                        left_match["anchor_col"] + right_match["right_col"],
                                    ),
                                    (bottom_var, top_prof["h"] + anchor_prof["h"], 0),
                                ]
                                ok = True
                                for cells, dr, dc in placements:
                                    if not _paint(out, cells, dr, dc):
                                        ok = False
                                        break
                                if ok and np.all(out != -1):
                                    return out.astype(int)

    raise AssertionError("could not recover the 5-piece cross layout from the scattered pieces")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble five scattered pieces into a cross-partitioned rectangle.

    Input structure:
    - Background is the most common color on a 20x20 canvas.
    - There are exactly five separated pieces; one center anchor carries a single grey marker.
    - The solved layout is top strip / middle row (left-center-right) / bottom strip.

    Rule:
    - Rotate the center anchor so the marker is the piece's local top-left.
    - Use the only tabbed seams to attach left and right neighbors to the center.
    - Fill the remaining full-width rectangles as the top and bottom strips.

    Key insight:
    - This is a true D8/2 cross partition with five pieces, but D15/2 removes all
      vertical-axis tabs, so only the horizontal-direction interlocks carry matching evidence.
    """
    g = np.asarray(grid, dtype=int)
    comps, _bg = _components(g)
    assert comps, "input must contain non-background pieces"
    return _solve_cross_tabs(comps)


def _build_cross_ownership(
    top_h: int,
    mid_h: int,
    bottom_h: int,
    left_w: int,
    center_w: int,
    right_w: int,
    left_tab: dict,
    right_tab: dict,
) -> np.ndarray | None:
    h = top_h + mid_h + bottom_h
    w = left_w + center_w + right_w
    own = np.zeros((h, w), dtype=int)
    own[:top_h, :] = TOP
    own[top_h:top_h + mid_h, :left_w] = LEFT
    own[top_h:top_h + mid_h, left_w:left_w + center_w] = CENTER
    own[top_h:top_h + mid_h, left_w + center_w:] = RIGHT
    own[top_h + mid_h:, :] = BOTTOM

    own[top_h + int(left_tab["row"]), left_w:left_w + int(left_tab["depth"])] = LEFT
    own[
        top_h + int(right_tab["row"]),
        left_w + center_w - int(right_tab["depth"]):left_w + center_w,
    ] = RIGHT

    for label in (TOP, LEFT, CENTER, RIGHT, BOTTOM):
        mask = (own == label).astype(int)
        if mask.sum() == 0:
            return None
        if not is_contiguous(mask, background=0, connectivity=8):
            return None
    return own


def _extract_piece(
    own: np.ndarray,
    label: int,
    color: int,
    *,
    marker_abs: tuple[int, int] | None = None,
) -> list[tuple[int, int, int]]:
    rows, cols = np.where(own == label)
    min_r, min_c = int(rows.min()), int(cols.min())
    cells: list[tuple[int, int, int]] = []
    for r, c in zip(rows, cols):
        rr, cc = int(r), int(c)
        col = MARKER_COLOR if marker_abs == (rr, cc) else color
        cells.append((rr - min_r, cc - min_c, col))
    return _normalize(cells)


def _shape_key(cells: list[tuple[int, int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r, c) for r, c, col in cells if col != MARKER_COLOR))


def _shape_variant_keys(cells: list[tuple[int, int, int]]) -> set[tuple[tuple[int, int], ...]]:
    mono = [(r, c, 1) for r, c, col in cells if col != MARKER_COLOR]
    return {_shape_key(var) for var in _variants(mono)}


def _sample_params() -> tuple[int, int, int, int, int, int, dict, dict] | None:
    top_h = random.randint(STRIP_MIN, STRIP_MAX)
    bottom_h = random.randint(STRIP_MIN, STRIP_MAX)
    if top_h == bottom_h:
        return None

    mid_h = random.randint(MID_H_MIN, MID_H_MAX)
    left_w = random.randint(SIDE_W_MIN, SIDE_W_MAX)
    center_w = random.randint(MID_W_MIN, MID_W_MAX)
    right_w = random.randint(SIDE_W_MIN, SIDE_W_MAX)

    left_tab = {
        "row": random.randint(1, mid_h - 2),
        "depth": random.randint(1, center_w - 1),
    }
    right_tab = {
        "row": random.randint(1, mid_h - 2),
        "depth": random.randint(1, center_w - 1),
    }

    if left_w == right_w and left_tab == right_tab:
        return None
    return top_h, mid_h, bottom_h, left_w, center_w, right_w, left_tab, right_tab


def _build_target() -> tuple[np.ndarray, list[list[tuple[int, int, int]]]] | None:
    sampled = _sample_params()
    if sampled is None:
        return None

    top_h, mid_h, bottom_h, left_w, center_w, right_w, left_tab, right_tab = sampled
    own = _build_cross_ownership(top_h, mid_h, bottom_h, left_w, center_w, right_w, left_tab, right_tab)
    if own is None:
        return None

    pool = COLOR_POOL.copy()
    random.shuffle(pool)
    colors = {
        CENTER: ANCHOR_COLOR,
        TOP: pool[0],
        LEFT: pool[1],
        RIGHT: pool[2],
        BOTTOM: pool[3],
    }
    output = np.zeros(own.shape, dtype=int)
    for label, color in colors.items():
        output[own == label] = color

    marker_abs = (top_h, left_w)
    output[marker_abs] = MARKER_COLOR
    pieces = [
        _extract_piece(own, CENTER, colors[CENTER], marker_abs=marker_abs),
        _extract_piece(own, TOP, colors[TOP]),
        _extract_piece(own, LEFT, colors[LEFT]),
        _extract_piece(own, RIGHT, colors[RIGHT]),
        _extract_piece(own, BOTTOM, colors[BOTTOM]),
    ]

    left_keys = _shape_variant_keys(pieces[2])
    right_keys = _shape_variant_keys(pieces[3])
    if left_keys & right_keys:
        return None
    if _shape_key(pieces[1]) == _shape_key(pieces[4]):
        return None

    return output, pieces


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


def _gen_attempt() -> np.ndarray | None:
    built = _build_target()
    if built is None:
        return None
    expected, pieces = built

    rotated = [_rotate_cells(piece, random.randint(0, 3)) for piece in pieces]
    canvas = _scatter_patches([_cells_to_patch(piece) for piece in rotated])
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
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
