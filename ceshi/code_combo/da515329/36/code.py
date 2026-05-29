# Combo variant of da515329 (index 36)
# combo_spec: D6/3+D17/1
# Name: 矩形墙壁 + 反弹路径

from __future__ import annotations

import random
import numpy as np


_BG = 0
_PATH = 8
_WALL = 2
_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# Design analysis for the fused variant:
# a) Concrete trace:
#    Original da515329 turns orthogonally with a spacing rule, so an upward arm tip
#    first heads east and then keeps trying right/straight/left to build a spiral.
#    Here that same upward tip instead launches a diagonal ray toward the north-east.
#    If the cell ahead in the row direction is blocked, its vertical velocity flips;
#    if the cell ahead in the column direction is blocked, its horizontal velocity
#    flips; if the diagonal target is a wall corner, both axes flip. With a solid
#    rectangle in the north-east quadrant, the ray zigzags between the rectangle face
#    and the outer border rather than curling inward.
# b) Complexity calibration:
#    The input keeps only three visible roles: background, teal source cross, and one
#    wall color. Geometry varies through a small number of rectangle sizes/gaps, but
#    every sample repeats the same ray rule four times, so the new behavior is visible
#    without becoming underconstrained.
# c) Broken assumptions from the original solver:
#    - "Only teal objects exist" no longer holds, so non-teal nonzero cells must be
#      treated as obstacles, not as path.
#    - The 2/1 stride, right-turn priority, parity cleanup, and border cap fixes are
#      all specific to spirals and are removed.
#    - Paths are no longer self-spacing walls; they may revisit parity classes, so the
#      solver tracks finite reflection states instead of local spacing constraints.
# d) Placement legality:
#    The generator chooses a center only after rectangle sizes/gaps are known, so all
#    arms and rectangles fit inside the grid, stay disjoint, and leave open corridors.
# e) Visual observability:
#    A human sees teal arm tips, several solid rectangular walls, and output zigzags
#    that bounce off both borders and wall faces. The wall effect is visible because
#    the generator rejects cases whose output matches the no-wall control.
# f) Visual cleanliness:
#    Walls use a single color and occupy quadrant-local blocks. This avoids ambiguous
#    overlays near the cross center and keeps each corridor leg readable.
# g) New implicit assumptions:
#    `main()` assumes the only teal cells in the input belong to one center-hole cross,
#    and every nonzero non-teal cell is part of a wall rectangle. `generate_input()`
#    guarantees both conditions and also guarantees that the four arm directions exist.


def _inb(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _right(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _diag_from_outward(d: tuple[int, int]) -> tuple[int, int]:
    rd = _right(d)
    return (d[0] + rd[0], d[1] + rd[1])


def _find_cross_center(grid: np.ndarray) -> tuple[int, int]:
    row_counts = np.sum(grid == _PATH, axis=1)
    col_counts = np.sum(grid == _PATH, axis=0)
    return int(np.argmax(row_counts)), int(np.argmax(col_counts))


def _collect_arm_tips(grid: np.ndarray, cr: int, cc: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = grid.shape
    arms: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells: list[tuple[int, int]] = []
        while _inb(h, w, r + dr, c + dc) and int(grid[r + dr, c + dc]) == _PATH:
            r += dr
            c += dc
            cells.append((int(r), int(c)))
        if cells:
            arms.append((d, cells[-1]))
    return arms


def _trace_paths(
    grid: np.ndarray,
    *,
    ignore_walls: bool = False,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    wall_mask = np.zeros((h, w), dtype=bool) if ignore_walls else ((g != _BG) & (g != _PATH))
    source_mask = g == _PATH

    cr, cc = _find_cross_center(g)
    arms = _collect_arm_tips(g, cr, cc)

    metrics: list[dict[str, int]] = []
    max_steps = h * w * 4

    for outward, start in arms:
        dr, dc = _diag_from_outward(outward)
        r, c = start
        seen: set[tuple[int, int, int, int]] = set()
        steps = 0
        wall_bounces = 0
        border_bounces = 0

        while steps < max_steps:
            state = (int(r), int(c), int(dr), int(dc))
            if state in seen:
                break
            seen.add(state)

            row_r, row_c = int(r + dr), int(c)
            col_r, col_c = int(r), int(c + dc)
            raw_row_block = not _inb(h, w, row_r, row_c)
            raw_col_block = not _inb(h, w, col_r, col_c)
            wall_row_block = (not raw_row_block) and bool(wall_mask[row_r, row_c])
            wall_col_block = (not raw_col_block) and bool(wall_mask[col_r, col_c])
            blocked_r = raw_row_block or wall_row_block
            blocked_c = raw_col_block or wall_col_block

            diag_r, diag_c = int(r + dr), int(c + dc)
            if (
                _inb(h, w, diag_r, diag_c)
                and bool(wall_mask[diag_r, diag_c])
                and not blocked_r
                and not blocked_c
            ):
                blocked_r = True
                blocked_c = True
                wall_row_block = True
                wall_col_block = True

            if blocked_r:
                dr = -dr
                if raw_row_block:
                    border_bounces += 1
                else:
                    wall_bounces += 1
            if blocked_c:
                dc = -dc
                if raw_col_block:
                    border_bounces += 1
                else:
                    wall_bounces += 1

            nr, nc = int(r + dr), int(c + dc)
            if not _inb(h, w, nr, nc):
                break
            if bool(wall_mask[nr, nc]):
                break
            if bool(source_mask[nr, nc]):
                break

            r, c = nr, nc
            if int(out[r, c]) == _BG:
                out[r, c] = _PATH
            steps += 1

        metrics.append(
            {
                "steps": int(steps),
                "wall_bounces": int(wall_bounces),
                "border_bounces": int(border_bounces),
            }
        )

    return out, metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire four teal diagonal rays from a center-hole cross and let rectangular walls reflect them.

    Input structure:
    - bg = 0 (black).
    - A single teal (8) plus-sign cross with a one-cell hole at its center.
    - One or more non-teal rectangular wall blocks that stay fixed and act as mirrors.

    Rule:
    - Each arm tip launches a ray in the diagonal direction obtained by turning its
      outward direction clockwise and combining the two cardinal components.
    - On every step, a blocked row move flips the vertical sign, and a blocked column
      move flips the horizontal sign. Borders and wall faces both count as blockers.
    - If the next diagonal cell is a wall corner, both axes flip at once.
    - Rays paint only empty cells, stop when they would enter the source cross, and
      keep bouncing until their reflection state repeats.

    Key insight: The cross still provides four synchronized launch points, but the
    geometry is now controlled by axis-wise reflections against solid rectangles, so
    the output becomes a set of corridor-limited zigzags instead of nested spirals.
    """

    out, _metrics = _trace_paths(grid)
    return out


# --------------- generator ---------------

_ARM_MIN, _ARM_MAX = 2, 4
_SIZE_MIN, _SIZE_MAX = 18, 24
_RECT_H_MIN, _RECT_H_MAX = 3, 5
_RECT_W_MIN, _RECT_W_MAX = 3, 5
_GAP_MIN, _GAP_MAX = 2, 4
_CENTER_PAD = 2
_MAX_OUTER_RETRIES = 400
_MIN_PATH_FILL = 0.12
_MAX_PATH_FILL = 0.36
_MIN_TOTAL_WALL_BOUNCES = 8
_MIN_TOTAL_BORDER_BOUNCES = 6
_MIN_DIFF_FROM_CONTROL = 10


def _build_cross(grid: np.ndarray, arm: int, cr: int, cc: int) -> None:
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = _PATH
    grid[cr + offsets, cc] = _PATH
    grid[cr, cc - offsets] = _PATH
    grid[cr, cc + offsets] = _PATH


def _paint_rect(grid: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    grid[top : top + height, left : left + width] = _WALL


def _make_rectangles(
    size: int,
    arm: int,
    cr: int,
    cc: int,
    params: dict[str, int],
) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []

    ne_h, ne_w = params["ne_h"], params["ne_w"]
    nw_h, nw_w = params["nw_h"], params["nw_w"]
    se_h, se_w = params["se_h"], params["se_w"]
    sw_h, sw_w = params["sw_h"], params["sw_w"]
    top_gap, bottom_gap = params["top_gap"], params["bottom_gap"]
    left_gap, right_gap = params["left_gap"], params["right_gap"]

    rects.append((cr - top_gap - ne_h + 1, cc + right_gap, ne_h, ne_w))
    rects.append((cr - top_gap - nw_h + 1, cc - left_gap - nw_w + 1, nw_h, nw_w))
    rects.append((cr + bottom_gap, cc + right_gap, se_h, se_w))
    rects.append((cr + bottom_gap, cc - left_gap - sw_w + 1, sw_h, sw_w))

    for top, left, height, width in rects:
        if top < 0 or left < 0 or top + height > size or left + width > size:
            return []

        bottom = top + height - 1
        right = left + width - 1

        # Keep a one-cell moat around the center cross for readability.
        if bottom >= cr - arm - 1 and top <= cr + arm + 1 and right >= cc - 1 and left <= cc + 1:
            return []
        if right >= cc - arm - 1 and left <= cc + arm + 1 and bottom >= cr - 1 and top <= cr + 1:
            return []

    return rects


def _sample_layout() -> tuple[np.ndarray, list[dict[str, int]]] | None:
    arm = random.randint(_ARM_MIN, _ARM_MAX)
    size = random.randint(_SIZE_MIN, _SIZE_MAX)

    params = {
        "ne_h": random.randint(_RECT_H_MIN, _RECT_H_MAX),
        "ne_w": random.randint(_RECT_W_MIN, _RECT_W_MAX),
        "nw_h": random.randint(_RECT_H_MIN, _RECT_H_MAX),
        "nw_w": random.randint(_RECT_W_MIN, _RECT_W_MAX),
        "se_h": random.randint(_RECT_H_MIN, _RECT_H_MAX),
        "se_w": random.randint(_RECT_W_MIN, _RECT_W_MAX),
        "sw_h": random.randint(_RECT_H_MIN, _RECT_H_MAX),
        "sw_w": random.randint(_RECT_W_MIN, _RECT_W_MAX),
        "top_gap": random.randint(_GAP_MIN, _GAP_MAX),
        "bottom_gap": random.randint(_GAP_MIN, _GAP_MAX),
        "left_gap": random.randint(_GAP_MIN, _GAP_MAX),
        "right_gap": random.randint(_GAP_MIN, _GAP_MAX),
    }

    top_need = max(arm, params["top_gap"] + max(params["ne_h"], params["nw_h"]) - 1) + _CENTER_PAD
    bottom_need = max(arm, params["bottom_gap"] + max(params["se_h"], params["sw_h"]) - 1) + _CENTER_PAD
    left_need = max(arm, params["left_gap"] + max(params["nw_w"], params["sw_w"]) - 1) + _CENTER_PAD
    right_need = max(arm, params["right_gap"] + max(params["ne_w"], params["se_w"]) - 1) + _CENTER_PAD

    cr_lo, cr_hi = top_need, size - 1 - bottom_need
    cc_lo, cc_hi = left_need, size - 1 - right_need
    if cr_lo > cr_hi or cc_lo > cc_hi:
        return None

    cr = random.randint(cr_lo, cr_hi)
    cc = random.randint(cc_lo, cc_hi)

    rects = _make_rectangles(size, arm, cr, cc, params)
    if len(rects) != 4:
        return None

    grid = np.zeros((size, size), dtype=int)
    _build_cross(grid, arm, cr, cc)
    for top, left, height, width in rects:
        if np.any(grid[top : top + height, left : left + width] != _BG):
            return None
        _paint_rect(grid, top, left, height, width)

    out, metrics = _trace_paths(grid)
    control, _control_metrics = _trace_paths(np.where(grid == _WALL, _BG, grid), ignore_walls=True)

    added = int(np.sum((out == _PATH) & (grid != _PATH)))
    fill = added / float(size * size)
    total_wall = sum(m["wall_bounces"] for m in metrics)
    total_border = sum(m["border_bounces"] for m in metrics)
    lengths = [m["steps"] for m in metrics]
    diff_control = int(np.sum(out != control))

    if len(metrics) != 4:
        return None
    if min(lengths) < 8:
        return None
    if sum(lengths) < size + 20:
        return None
    if total_wall < _MIN_TOTAL_WALL_BOUNCES:
        return None
    if total_border < _MIN_TOTAL_BORDER_BOUNCES:
        return None
    if sum(m["wall_bounces"] > 0 for m in metrics) < 4:
        return None
    if not (_MIN_PATH_FILL <= fill <= _MAX_PATH_FILL):
        return None
    if diff_control < _MIN_DIFF_FROM_CONTROL:
        return None
    if np.array_equal(out, grid):
        return None

    return grid, metrics


def generate_input() -> np.ndarray:
    for _ in range(_MAX_OUTER_RETRIES):
        sample = _sample_layout()
        if sample is None:
            continue
        grid, _metrics = sample
        out = main(grid)
        if np.array_equal(out, main(grid)):
            return grid
    raise RuntimeError("generate_input: failed after retries")
