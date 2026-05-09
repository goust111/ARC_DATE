# Combo variant of 45a5af55 (index 1)
# combo_spec: D1/0+D3/0
# Name: 顶读反向递减厚度嵌套

from __future__ import annotations

import numpy as np


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Input stays the same object class as the source task: a single vertical
#      stack of full-width monochrome horizontal stripes.
#    - Original task:
#        top->bottom runs = [(A, 2), (B, 4), (C, 1), (D, 3)]
#        parsed build order = bottom->top = [(D, 3), (C, 1), (B, 4), (A, 2)]
#        output = center 3x3 D, then wrap thickness 1 by C, thickness 4 by B,
#        thickness 2 by A. Final side = 3 + 2*(1+4+2) = 17.
#    - This combo variant:
#        same input runs = [(A, 2), (B, 4), (C, 1), (D, 3)]
#        build order = top->bottom = [(A, 2), (B, 4), (C, 1), (D, 3)]
#        geometric thicknesses ignore heights and become [4, 3, 2, 1].
#        output = center 4x4 A, then wrap thickness 3 by B, thickness 2 by C,
#        thickness 1 by D. Final side = 4 + 2*(3+2+1) = 16 = 4^2.
#    - The fusion is not a simple union of two independent tweaks: reversing the
#      read direction changes the color order, while decreasing thickness by
#      layer index changes the geometry. The visible nesting depends on both.
#
# b) Complexity calibration
#    - To keep the rule human-inferable from a few ARC examples, the generator
#      keeps the input simple: 3-5 stripes, one stack, no gaps, no extra marks.
#    - To make the D3 component visible, stripe heights are treated as distractor
#      data: the generator forces repeated heights and multiple mismatches between
#      input heights and the combo thickness sequence [n, n-1, ..., 1].
#    - To make the D1 component visible, stripe colors are sampled without
#      replacement, so reversing the read order changes the full nesting palette.
#
# c) Implicit assumptions review against the original main()
#    - The original row parser assumed each row is monochrome across the full
#      width. That assumption still stands and is enforced by generate_input().
#    - The original bottom-up scan encoded an order assumption that breaks under
#      D1/0. The combo parser must scan top-down instead.
#    - `c0, h0 = runs[0]` in the source code assumed the first parsed height is
#      the center side length. Under D3/0 this is false: the center size is now
#      `n_runs`, independent of the stripe's actual height.
#    - `for col, h in runs[1:]` in the source code assumed every frame thickness
#      equals the corresponding stripe height. Under the combo this is replaced by
#      `thickness = n_runs - idx`, where `idx` is the top-down layer index.
#    - The fill-and-inset construction itself remains valid because each new frame
#      still surrounds the previous square symmetrically on all four sides.
#
# d) Placement legality
#    - The generator only stacks `np.full((height, width), color)` blocks with
#      `np.vstack`, so stripes never overlap and never leave gaps.
#    - Output legality is automatic once layer sizes are positive: for n stripes
#      the side length is exactly n^2, so restricting n to 3-5 keeps outputs in
#      the ARC-friendly range 9x9 to 25x25.
#
# e) Visual observability
#    - A human can compare input stripe order with output nesting order and see
#      that the top stripe, not the bottom stripe, becomes the center color.
#    - The decreasing-thickness rule is also visible because outer rings get
#      thinner even when their input stripes are taller, and equal-height stripes
#      can produce different frame thicknesses depending on depth.
#    - The generator rejects samples where the combo output accidentally matches
#      the source rule, D1-only rule, or D3-only rule on the same input.
#
# f) Visual cleanliness
#    - There is still only one input object family and one output object family.
#      No overlaps, auxiliary markers, or mixed semantics share the same region.
#      The output remains a clean concentric-square image.
#
# g) Change inferability
#    - No recoloring is introduced. Each stripe color is copied directly into one
#      output layer, and the only inference burden is the structural one:
#      determine the top-down layer order and the index-based thickness schedule.
#
# h) New main() assumptions audited against generate_input()
#    - `main()` assumes there is at least one stripe; the generator always makes
#      3-5 stripes.
#    - `main()` assumes each stripe row is uniform and stripes are contiguous; the
#      generator guarantees both by construction.
#    - `main()` assumes thicknesses are derived from the number of parsed stripes,
#      so the parser must recover the intended run count. Unique adjacent colors
#      guarantee that each generated stripe remains a separate run.


def _parse_runs_top_down(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return full-width stripe runs as (color, height) in top-to-bottom order."""
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    runs: list[tuple[int, int]] = []
    r = 0
    while r < h:
        row = g[r]
        color = int(row[0])
        if not bool((row == color).all()):
            raise ValueError("Each input row must be monochrome across the full width.")

        height = 1
        rr = r + 1
        while rr < h and bool((g[rr] == color).all()):
            height += 1
            rr += 1

        runs.append((color, height))
        r = rr

    if not runs:
        raise ValueError("Input must contain at least one stripe.")
    return runs


def _decreasing_layer_sizes(n_runs: int) -> list[int]:
    """Top-down geometric sizes: center side n, then frames n-1 down to 1."""
    return [int(n_runs - idx) for idx in range(n_runs)]


def _render_nested(colors_inner_to_outer: list[int], layer_sizes_inner_to_outer: list[int]) -> np.ndarray:
    """Render a solid center followed by concentric square frames."""
    if not colors_inner_to_outer or len(colors_inner_to_outer) != len(layer_sizes_inner_to_outer):
        raise ValueError("Colors and layer sizes must be non-empty and aligned.")

    first_size = int(layer_sizes_inner_to_outer[0])
    cur = np.full((first_size, first_size), int(colors_inner_to_outer[0]), dtype=int)

    for color, thickness in zip(colors_inner_to_outer[1:], layer_sizes_inner_to_outer[1:]):
        thickness = int(thickness)
        new_n = int(cur.shape[0] + 2 * thickness)
        nxt = np.full((new_n, new_n), int(color), dtype=int)
        nxt[thickness : thickness + cur.shape[0], thickness : thickness + cur.shape[1]] = cur
        cur = nxt

    return cur


def _solve_combo_from_top_runs(runs_top_down: list[tuple[int, int]]) -> np.ndarray:
    """Active combo rule: top-down colors, decreasing index-based sizes."""
    colors = [color for color, _height in runs_top_down]
    return _render_nested(colors, _decreasing_layer_sizes(len(runs_top_down)))


def _solve_original_from_top_runs(runs_top_down: list[tuple[int, int]]) -> np.ndarray:
    """Source task behavior: bottom-up colors, height-based sizes."""
    bottom_to_top = list(reversed(runs_top_down))
    colors = [color for color, _height in bottom_to_top]
    sizes = [height for _color, height in bottom_to_top]
    return _render_nested(colors, sizes)


def _solve_d1_only_from_top_runs(runs_top_down: list[tuple[int, int]]) -> np.ndarray:
    """Top-down colors, but thickness still equals each stripe height."""
    colors = [color for color, _height in runs_top_down]
    sizes = [height for _color, height in runs_top_down]
    return _render_nested(colors, sizes)


def _solve_d3_only_from_top_runs(runs_top_down: list[tuple[int, int]]) -> np.ndarray:
    """Bottom-up colors with the decreasing index-based thickness schedule."""
    bottom_to_top = list(reversed(runs_top_down))
    colors = [color for color, _height in bottom_to_top]
    return _render_nested(colors, _decreasing_layer_sizes(len(bottom_to_top)))


def _behavior_metrics(runs_top_down: list[tuple[int, int]]) -> dict[str, int]:
    """
    Quantify whether the sample expresses the combo-specific behavior.

    Metrics:
    - height_mismatch_count: how many top-down stripes disagree with combo sizes.
    - repeated_height_pairs: repeated heights help show that equal input heights
      do not force equal output thicknesses.
    - differs_from_original / differs_from_d1 / differs_from_d3: whether the
      same input would yield a different output under the three nearby rules.
    """
    heights = [height for _color, height in runs_top_down]
    combo_sizes = _decreasing_layer_sizes(len(runs_top_down))
    combo_out = _solve_combo_from_top_runs(runs_top_down)
    return {
        "height_mismatch_count": int(sum(int(h != s) for h, s in zip(heights, combo_sizes))),
        "repeated_height_pairs": int(len(heights) - len(set(heights))),
        "differs_from_original": int(not np.array_equal(combo_out, _solve_original_from_top_runs(runs_top_down))),
        "differs_from_d1": int(not np.array_equal(combo_out, _solve_d1_only_from_top_runs(runs_top_down))),
        "differs_from_d3": int(not np.array_equal(combo_out, _solve_d3_only_from_top_runs(runs_top_down))),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from a stack of horizontal color stripes.

    Combo rule:
    - Parse full-width stripes from top to bottom.
    - If there are n stripes, the top stripe becomes an n x n solid center square.
    - The next stripes wrap outward in the same top-down color order.
    - Geometric thicknesses are [n-1, n-2, ..., 1] for the outer layers, so the
      full inner-to-outer size sequence is [n, n-1, ..., 1].
    """
    runs_top_down = _parse_runs_top_down(np.array(grid, dtype=int))
    return _solve_combo_from_top_runs(runs_top_down)


def _gen_attempt() -> np.ndarray | None:
    n_runs = int(np.random.randint(3, 6))
    combo_sizes = _decreasing_layer_sizes(n_runs)

    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    if len(set(heights)) == n_runs:
        return None
    if sum(int(h != s) for h, s in zip(heights, combo_sizes)) < 2:
        return None

    width = int(np.random.randint(6, 16))
    colors = [int(c) for c in np.random.choice(np.arange(1, 10), size=n_runs, replace=False)]

    stripes = [np.full((height, width), color, dtype=int) for color, height in zip(colors, heights)]
    grid = np.vstack(stripes)
    runs_top_down = _parse_runs_top_down(grid)
    metrics = _behavior_metrics(runs_top_down)

    if metrics["repeated_height_pairs"] < 1:
        return None
    if metrics["height_mismatch_count"] < 2:
        return None
    if metrics["differs_from_original"] != 1:
        return None
    if metrics["differs_from_d1"] != 1:
        return None
    if metrics["differs_from_d3"] != 1:
        return None

    combo_out = _solve_combo_from_top_runs(runs_top_down)
    if not np.array_equal(main(grid), combo_out):
        return None
    if combo_out.shape != (n_runs * n_runs, n_runs * n_runs):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
