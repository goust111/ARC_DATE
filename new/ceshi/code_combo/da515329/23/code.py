# Combo variant of da515329 (index 23)
# combo_spec: D6/0+D15/0
# Name: 45度转弯 + 直线射线

# Design analysis
# ===============
# a) Concrete behavior and trace:
#    The source task starts from one teal center-hole cross and grows four long
#    orthogonal spirals. This combo removes the spiral entirely (D6) and keeps
#    only the "launch from the four tips" part, while D15 changes the launch
#    heading from a 90-degree clockwise turn to a 45-degree clockwise turn.
#    I make that concrete as one straight diagonal ray per arm tip:
#      - top tip: NE
#      - right tip: SE
#      - bottom tip: SW
#      - left tip: NW
#    Small trace with arm_len=2:
#      input tips are at (cr-2,cc), (cr,cc+2), (cr+2,cc), (cr,cc-2)
#      original task would then walk east/south/west/north and keep bending
#      into a maze-like spiral
#      fused task instead paints:
#        (cr-3,cc+1), (cr-4,cc+2), ... from the top tip,
#        (cr+1,cc+3), (cr+2,cc+4), ... from the right tip,
#        and likewise for the other two tips until the border.
#    The visible result is a plus-sign seed with four corner-directed diagonal
#    rays, i.e. an X-like overlay built from the cross endpoints.
#
# b) Complexity calibration:
#    The fusion is intentionally simple but not trivial:
#      - D6 is real because no turning state, spacing rule, or spiral envelope
#        remains; every path is a single unbroken ray.
#      - D15 is real because the emitted direction is diagonal, not orthogonal.
#    To keep the change visible, the generator rejects samples where any ray is
#    too short or where the diagonal output would be too similar to a plain
#    cardinal straight-ray baseline.
#
# c) Hidden-assumption audit of the original main():
#    - `right(outward)` and all turn-priority logic assumed a 4-direction spiral.
#      That assumption is invalid: the combo needs exactly one 45-degree launch,
#      then no further turns.
#    - `spacing_ok`, `can_move(length=2/1)`, agent phase, and parity/border
#      post-processing all assumed self-avoiding orthogonal corridors that try to
#      fill about half the grid. None of those assumptions survive once the path
#      type becomes straight rays.
#    - Center detection via densest teal row/column still holds because the input
#      object family stays the same: one equal-arm plus-sign cross with a hole.
#    Therefore the new solver keeps only cross detection and replaces the rest of
#    the pipeline with deterministic endpoint-to-border diagonal painting.
#
# d) Placement legality:
#    The generator still constructs only one legal equal-arm cross on a square
#    black canvas. It samples centers only when every clockwise diagonal ray has
#    at least a small in-bounds run, so no emitted ray immediately exits.
#
# e) Visual observability:
#    Across 2-4 examples, a human can see the same input cross and the same
#    output principle: each tip extends toward one corner-adjacent diagonal
#    direction. Different center placements change the four ray lengths, which
#    exposes the border-conditioned nature of the rule without introducing extra
#    colors or hidden metadata.
#
# f) Visual cleanliness:
#    There are only two layers: the original teal cross and the added teal
#    diagonal rays. Rays do not meander or overlap into dense knots, so the
#    output remains easy to parse by eye.
#
# g) Change inferability:
#    No color remapping is used. The only transformation is geometric and fully
#    visible: cardinal input arms imply diagonal output rays. Because the same
#    one-color rule repeats four times in every sample, the fusion can be
#    inferred from a small number of ARC examples.
#
# h) New assumptions introduced by the fused solver and generator guarantees:
#    - Solver assumption: there is exactly one teal center-hole cross.
#      Generator guarantee: it creates only that object.
#    - Solver assumption: all four arms exist and share one arm length.
#      Generator guarantee: the input is built from one sampled arm length.
#    - Solver assumption: teal appears only on the cross in the input, so densest
#      row/column recovers the center hole.
#      Generator guarantee: no extra teal markers or obstacles are added.
#    - Solver assumption: the intended D15 effect is the clockwise 45-degree
#      heading. Generator guarantee: it measures the Hamming difference against a
#      cardinal straight-ray baseline and rejects weak/degenerate placements.

from __future__ import annotations

import math
import random

import numpy as np


CROSS_COLOR = 8
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
MAX_OUTER_RETRIES = 300
MIN_DIAG_LEN = 3
MIN_DIAGONAL_ADDED = 12
MIN_DIAG_EFFECT = 8

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_CW45_FROM_OUTWARD = {
    (-1, 0): (-1, 1),   # up -> northeast
    (0, 1): (1, 1),     # right -> southeast
    (1, 0): (1, -1),    # down -> southwest
    (0, -1): (-1, -1),  # left -> northwest
}


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    """Build one center-hole plus sign with equal arm length."""
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _find_cross(
    grid: np.ndarray,
) -> tuple[int, int, list[tuple[tuple[int, int], list[tuple[int, int]]]], int]:
    """Recover center-hole row/col, directional arm cells, and common arm length."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    row_counts = np.sum(g == CROSS_COLOR, axis=1)
    col_counts = np.sum(g == CROSS_COLOR, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    arms = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells = []
        while inb(r + dr, c + dc) and g[r + dr, c + dc] == CROSS_COLOR:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((d, cells))

    arm_len = max((len(cells) for _, cells in arms), default=0)
    return cr, cc, arms, arm_len


def _paint_ray(out: np.ndarray, start: tuple[int, int], d: tuple[int, int]) -> int:
    """Paint one diagonal ray starting one step beyond `start`; return its length."""
    h, w = out.shape
    r, c = start
    dr, dc = d
    length = 0

    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        out[r, c] = CROSS_COLOR
        length += 1
    return length


def _solve_variant(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return fused output plus the four emitted diagonal ray lengths."""
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _cr, _cc, arms, arm_len = _find_cross(g)
    if arm_len == 0:
        return out, []

    ray_lengths = []
    for outward, cells in arms:
        tip = cells[-1]
        ray_lengths.append(_paint_ray(out, tip, _CW45_FROM_OUTWARD[outward]))
    return out, ray_lengths


def _solve_cardinal_baseline(grid: np.ndarray) -> np.ndarray:
    """D6-only baseline: emit straight rays in the arm's outward cardinal direction."""
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _cr, _cc, arms, arm_len = _find_cross(g)
    if arm_len == 0:
        return out

    for outward, cells in arms:
        _paint_ray(out, cells[-1], outward)
    return out


def _fusion_metrics(grid: np.ndarray) -> dict[str, object]:
    """Quantify that the fused output really uses diagonal straight rays."""
    g = np.asarray(grid, dtype=int)
    out, ray_lengths = _solve_variant(g)
    baseline = _solve_cardinal_baseline(g)

    added_mask = (out == CROSS_COLOR) & (g != CROSS_COLOR)
    rr, cc = np.indices(g.shape)
    cr, cc0, _arms, _arm_len = _find_cross(g)
    diagonal_added = int(np.count_nonzero(added_mask & (rr != cr) & (cc != cc0)))
    diag_effect = int(np.count_nonzero(out != baseline))

    return {
        "ray_lengths": tuple(int(x) for x in ray_lengths),
        "min_ray_len": int(min(ray_lengths)) if ray_lengths else 0,
        "diagonal_added": int(diagonal_added),
        "diag_effect_vs_cardinal": int(diag_effect),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a teal center-hole cross by diagonal straight rays.

    Input structure:
    - bg = 0 (black). The grid contains one teal (8) plus-sign cross with an empty center cell.
    - The four arms are axis-aligned and have equal length.

    Rule:
    - Detect the cross center and its four arm tips.
    - From each tip, launch one straight ray whose heading is the arm's outward direction rotated 45 degrees clockwise.
    - Keep painting teal cells along that diagonal until the next step would leave the grid.

    Key insight:
    - The fusion keeps the original "start from the four tips" scaffold, but replaces the spiral with a single unturned ray and replaces the 90-degree launch with a 45-degree clockwise launch. That turns the output from a maze-like orthogonal fill into a clean X-like overlay around the cross.
    """

    out, _ray_lengths = _solve_variant(grid)
    return out


def _candidate_centers(size: int, arm: int) -> list[tuple[int, int]]:
    """Enumerate centers where every clockwise diagonal ray is visibly nontrivial."""
    centers = []
    lo = arm + MIN_DIAG_LEN
    hi = size - arm - MIN_DIAG_LEN - 1
    if lo > hi:
        return centers

    for cr in range(lo, hi + 1):
        for cc in range(lo, hi + 1):
            top_len = min(cr - arm, size - 1 - cc)
            right_len = min(size - 1 - cr, size - 1 - (cc + arm))
            bottom_len = min(size - 1 - (cr + arm), cc)
            left_len = min(cr, cc - arm)
            ray_lengths = (top_len, right_len, bottom_len, left_len)
            if min(ray_lengths) < MIN_DIAG_LEN:
                continue
            centers.append((cr, cc))
    return centers


def _gen_attempt() -> np.ndarray | None:
    """Sample one cross placement and keep only visibly fused diagonal-ray cases."""
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    size = 4 * arm + 2 * math.ceil(arm / 2) + GRID_SIZE_BASE

    candidates = _candidate_centers(size, arm)
    random.shuffle(candidates)

    for cr, cc in candidates:
        inp = _build_input(arm, size, cr, cc)
        out = main(inp)
        metrics = _fusion_metrics(inp)
        if metrics["min_ray_len"] < MIN_DIAG_LEN:
            continue
        if metrics["diagonal_added"] < MIN_DIAGONAL_ADDED:
            continue
        if metrics["diag_effect_vs_cardinal"] < MIN_DIAG_EFFECT:
            continue
        if not np.array_equal(out, _solve_variant(inp)[0]):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after retries")
