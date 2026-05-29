# Combo variant of 45a5af55 (index 2)
# combo_spec: D1/1+D2/2+D5/0
# Name: 竖直色条矩形嵌套

from __future__ import annotations

import math

import numpy as np

# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    Original task:
#    - Input is full-width horizontal stripes.
#    - Read bottom->top.
#    - Stripe height is both layer thickness and the center-square side.
#    - Output is always a square nesting.
#
#    Fused task:
#    - Input becomes full-height vertical bars.
#    - Read left->right.
#    - The leftmost bar is the center, but the output is now a similar-rectangle nesting.
#    - Bar width still controls the "scale step", while the reduced input aspect
#      `input_width : input_height` controls how that step expands along x/y.
#      After the source flips to vertical bars, total input width is already consumed by
#      the bar-width sequence, so the independently chosen input height becomes the cue
#      for the vertical stretch component of the rectangle ratio.
#
#    Small trace:
#    - Bars left->right = [(1, red), (1, blue), (2, green)] and input shape = 6x4.
#    - Reduced aspect is 2:3, so one thickness unit means pad_x=2, pad_y=3.
#    - Original solver would inspect rows and fail to recover runs because rows are multi-color.
#    - A D1-only vertical-bar solver would output squares: 1x1 -> 3x3 -> 7x7.
#    - This combo solver outputs rectangles:
#         center red = 3x2
#         wrap blue by (3 rows, 2 cols) -> 9x6
#         wrap green by (6 rows, 4 cols) -> 21x14
#      The differences are jointly caused by vertical parsing, left->right ordering, and
#      aspect-ratio-aware frame construction.
#
# b) Complexity calibration:
#    - Keep 2-6 bars and widths in 1..4, matching the original nesting depth/thickness regime.
#    - Restrict reduced aspect ratios to a small tall-rectangle set so the D2 change is obvious
#      without introducing too many arbitrary possibilities.
#    - The generator enforces at least two bars and a non-square aspect, making both D1/D5 and D2
#      visible in every sample.
#
# c) Implicit-assumption audit of the original main():
#    - `g[r, 0]` assumed row-based objects. Invalid once objects are vertical bars.
#    - `(g[rr] == col).all()` assumed each whole row is monochrome. Invalid here; rows are mixed.
#    - `np.full((h0, h0), ...)` assumed the center is square. Invalid once D2 makes rectangles.
#    - `new_n = cur.shape[0] + 2*h` assumed both axes grow equally. Invalid here; we need
#      axis-specific growth derived from the reduced input aspect.
#    - No conflict is introduced between D1 and D5 because both point to the same parse change:
#      detect full-height column runs and read them left->right.
#
# d) Placement legality:
#    - Generator constructs the input as a horizontal concatenation of full-height monochrome bars,
#      so bars are disjoint and in-bounds.
#    - It rejects samples whose derived output would exceed ARC-size bounds.
#    - Adjacent colors are forced distinct so intended run boundaries remain visible.
#
# e) Visual observability:
#    - Inputs visibly contain vertical segmentation, not horizontal stripes.
#    - Outputs are visibly rectangular and taller than wide, so the D2 change cannot collapse back
#      into the original square behavior.
#    - Because all layers share one reduced aspect ratio, a human can infer the rectangle rule from
#      a few examples instead of needing hidden arithmetic.
#
# f) Visual cleanliness:
#    - No background clutter or overlapping markers are added.
#    - Each cell belongs to exactly one input bar or one output layer.
#
# g) Color inferability:
#    - Colors are copied directly from bars to nested layers in reading order.
#    - No color remapping or color-based sorting is used.
#
# h) Degeneration check idea:
#    - A valid fused sample should satisfy:
#      * every input column is monochrome, but not every row is monochrome;
#      * output is non-square;
#      * output aspect equals the reduced input aspect.
#    - The post-edit validation script measures these properties over random samples.
#
# i) New assumptions introduced by this solver:
#    - Each bar is a contiguous run of identical full-height columns.
#    - Reading order is left->right.
#    - The overall input shape has a non-square reduced aspect chosen from a small allowed set.
#    - `generate_input()` guarantees all of these.

_ASPECT_OPTIONS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 3),
)


def _parse_vertical_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    """Parse full-height bars from left to right as (color, width)."""
    g = np.array(grid, dtype=int)
    _h, w = g.shape

    runs: list[tuple[int, int]] = []
    c = 0
    while c < w:
        col = int(g[0, c])
        assert bool((g[:, c] == col).all())

        width = 1
        cc = c + 1
        while cc < w and bool((g[:, cc] == col).all()):
            width += 1
            cc += 1

        runs.append((col, width))
        c = cc

    return runs


def _build_nested_rectangles(
    runs: list[tuple[int, int]],
    aspect: tuple[int, int],
) -> np.ndarray:
    """Build nested similar rectangles from ordered (color, width) runs."""
    assert runs
    rw, rh = aspect

    c0, w0 = runs[0]
    cur = np.full((int(w0 * rh), int(w0 * rw)), int(c0), dtype=int)

    for col, width in runs[1:]:
        pad_y = int(width * rh)
        pad_x = int(width * rw)
        nxt = np.full(
            (int(cur.shape[0] + 2 * pad_y), int(cur.shape[1] + 2 * pad_x)),
            int(col),
            dtype=int,
        )
        nxt[pad_y : pad_y + cur.shape[0], pad_x : pad_x + cur.shape[1]] = cur
        cur = nxt

    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested rectangular frames from full-height vertical color bars.

    Input structure:
    - Full-height vertical bars of distinct adjacent non-bg colors, placed left to right.
    - Each bar has uniform color and variable width.
    - The overall input shape encodes a reduced aspect ratio.

    Rule:
    - Read bars left to right.
    - The leftmost bar becomes the center solid rectangle.
    - If the reduced input aspect is `rw:rh`, then one bar-width unit becomes
      `rw` columns and `rh` rows in the output geometry.
    - Each later bar wraps the current shape with a same-color rectangular frame
      whose horizontal and vertical thicknesses are scaled by that aspect ratio.
    """
    g = np.array(grid, dtype=int)
    h_total, w_total = g.shape
    runs = _parse_vertical_runs(g)
    assert runs

    scale = math.gcd(int(h_total), int(w_total))
    aspect = (int(w_total // scale), int(h_total // scale))
    return _build_nested_rectangles(runs, aspect)


def _gen_attempt() -> np.ndarray | None:
    widths: list[int] | None = None
    input_h = 0
    rw = rh = 0
    for idx in np.random.permutation(len(_ASPECT_OPTIONS)):
        cand_rw, cand_rh = _ASPECT_OPTIONS[int(idx)]
        for _ in range(40):
            candidate_widths = [int(np.random.randint(1, 5)) for _ in range(int(np.random.randint(2, 7)))]
            total_w = int(sum(candidate_widths))
            output_scale = int(candidate_widths[0] + 2 * sum(candidate_widths[1:]))
            if total_w % cand_rw != 0:
                continue
            input_scale = total_w // cand_rw
            candidate_h = int(input_scale * cand_rh)
            if candidate_h < 6 or candidate_h > 20:
                continue
            out_h = int(output_scale * cand_rh)
            out_w = int(output_scale * cand_rw)
            if out_h > 30 or out_w > 30:
                continue
            widths = candidate_widths
            input_h = candidate_h
            rw, rh = cand_rw, cand_rh
            break
        if widths is not None:
            break

    if widths is None:
        return None

    cols: list[int] = []
    prev = None
    for _ in range(len(widths)):
        c = int(np.random.randint(1, 10))
        while prev is not None and c == prev:
            c = int(np.random.randint(1, 10))
        cols.append(c)
        prev = c

    bars = [np.full((input_h, width), col, dtype=int) for col, width in zip(cols, widths)]
    grid = np.hstack(bars)

    expected = _build_nested_rectangles(list(zip(cols, widths)), (rw, rh))
    if not np.array_equal(main(grid), expected):
        return None
    if expected.shape[0] == expected.shape[1]:
        return None
    if bool(np.all(np.all(grid == grid[:, :1], axis=1))):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
