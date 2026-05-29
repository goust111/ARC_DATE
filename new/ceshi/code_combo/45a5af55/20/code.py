# Combo variant of 45a5af55 (index 20)
# combo_spec: D1/0+D3/0+D6/2
# Name: 顶读递减混色嵌套

from __future__ import annotations

import numpy as np


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Input stays a single stack of full-width monochrome horizontal stripes.
#    - Let the top->bottom stripes be [(2, 3), (8, 1), (5, 4), (7, 2)] where
#      each pair is (color, input_height).
#    - Original task:
#        parse bottom->top -> [(7, 2), (5, 4), (8, 1), (2, 3)]
#        sizes from heights -> [2, 4, 1, 3]
#        colors copied directly -> [7, 5, 8, 2]
#        output = center 2x2 of 7, then +4 of 5, then +1 of 8, then +3 of 2
#    - Combo task:
#        parse top->bottom -> [(2, 3), (8, 1), (5, 4), (7, 2)]
#        geometric sizes ignore heights -> [4, 3, 2, 1]
#        center color stays top stripe color -> 2
#        outer layer colors use adjacent stripe max in read order:
#            layer 1 color = max(8, 2) = 8
#            layer 2 color = max(5, 8) = 8
#            layer 3 color = max(7, 5) = 7
#        output = center 4x4 of 2, then +3 of 8, then +2 of 8, then +1 of 7
#    - The three changes are coupled: reversing the read order changes adjacency
#      for the mixing rule, and that same order also decides the decreasing size
#      schedule [n, n-1, ..., 1].
#
# b) Complexity calibration
#    - The generator uses 4-5 stripes so the decreasing geometry is visible and
#      the mixed-color rule has multiple adjacent pairs to compare.
#    - Stripe heights are sampled independently from the geometric sizes and are
#      forced to disagree in at least two positions. Equal heights are preferred
#      so humans can see that equal stripe heights can still produce different
#      frame thicknesses purely from layer order.
#    - Colors are sampled from a small fixed palette {2, 5, 8}. This keeps the
#      dominance relation compact enough that repeated pairwise evidence appears
#      across examples instead of scattering over too many colors.
#
# c) Implicit assumptions review for main()
#    - The parser still assumes each row is monochrome across the full width.
#      `generate_input()` guarantees this by stacking `np.full` rectangles.
#    - The original `runs[0]` meant "bottom stripe"; that is no longer valid.
#      Here `runs[0]` must be the top stripe because the combo reads top-down.
#    - The original `h0` and later `h` values controlled geometry directly.
#      That assumption is broken by D3/0, so sizes are recomputed from `n_runs`.
#    - The original `col` controlled the rendered layer color directly. That is
#      broken by D6/2, so each outer layer color is derived from two adjacent
#      source stripes instead of copied one-to-one.
#    - The renderer still assumes every new layer surrounds the previous square
#      symmetrically on all four sides; the decreasing size schedule preserves
#      that, so the fill-and-insert construction remains legal.
#
# d) Placement legality
#    - The input generator creates one vertical stripe stack only, so there is no
#      overlap or clipping in the input.
#    - Output size for n stripes is n^2 because the inner size is n and the
#      outer thicknesses are n-1 down to 1. Restricting n to 4 or 5 keeps the
#      output at 16x16 or 25x25, well inside ARC limits.
#
# e) Visual observability
#    - Direction change is visible because the top stripe becomes the center.
#    - Geometry change is visible because frame thicknesses follow a strict
#      inner-thick / outer-thin staircase unrelated to input stripe heights.
#    - Color change is the least human-obvious component because it depends on a
#      fixed dominance relation over colors. To keep it inferable, the generator
#      uses only three colors and requires both "current stripe wins" and "inner
#      stripe wins" cases in the same sample, so the dominance pattern leaves
#      repeated local evidence instead of a one-off recoloring.
#
# f) Visual cleanliness
#    - The output remains a single concentric-square object with no annotations.
#      Mixed colors only replace frame colors; they do not add extra marks.
#
# g) Change inferability
#    - The generator rejects samples where the mixed-color sequence collapses to
#      a single color or where no frame color changes after mixing. That avoids
#      degenerate cases where D6/2 is present in code but invisible in practice.
#
# h) Core degeneracy checks used by the generator
#    - Reject if combo output equals the top-down/no-mix output. This verifies
#      that D6/2 actually changes at least one rendered layer.
#    - Reject if combo output equals a top-down/height-driven output. This
#      verifies that D3/0 changes geometry on the sampled input.
#    - Reject if combo output equals the bottom-up combo-style output. This
#      verifies that D1/0 changes which stripe becomes central and which pairs
#      are considered adjacent for mixing.
#
# i) New solver assumptions audited against generate_input()
#    - At least four runs exist: guaranteed by sampling `n_runs` in {4, 5}.
#    - Adjacent stripes have different colors: enforced during color sampling.
#    - The top-down color sequence contains at least one rise and one fall, so
#      the mixed-color rule exhibits both possible outcomes.
#    - The mixed-color sequence has at least two distinct colors, so the output
#      does not visually collapse into a single-color square.


def _parse_runs_top_down(grid: np.ndarray) -> list[tuple[int, int]]:
    """Parse full-width monochrome stripes in top-to-bottom order."""
    g = np.array(grid, dtype=int)
    height, _width = g.shape

    runs: list[tuple[int, int]] = []
    r = 0
    while r < height:
        row = g[r]
        col = int(row[0])
        if not bool((row == col).all()):
            raise ValueError("Each input row must be monochrome across the full width.")

        stripe_h = 1
        rr = r + 1
        while rr < height and bool((g[rr] == col).all()):
            stripe_h += 1
            rr += 1

        runs.append((col, stripe_h))
        r = rr

    if not runs:
        raise ValueError("Input must contain at least one stripe.")
    return runs


def _decreasing_layer_sizes(n_runs: int) -> list[int]:
    """Return center-to-outer geometry sizes [n, n-1, ..., 1]."""
    return [int(n_runs - idx) for idx in range(n_runs)]


def _mixed_layer_colors_top_down(stripe_colors_top_down: list[int]) -> list[int]:
    """
    Return rendered layer colors from inner to outer.

    The center keeps the top stripe color. Every outer layer color is the max of
    the current stripe color and the immediately inner stripe color in top-down
    read order.
    """
    if not stripe_colors_top_down:
        raise ValueError("At least one stripe color is required.")

    mixed = [int(stripe_colors_top_down[0])]
    for idx in range(1, len(stripe_colors_top_down)):
        mixed.append(int(max(stripe_colors_top_down[idx - 1], stripe_colors_top_down[idx])))
    return mixed


def _render_nested(layer_colors_inner_to_outer: list[int], layer_sizes_inner_to_outer: list[int]) -> np.ndarray:
    """Render concentric squares where layer_sizes[0] is the center side length."""
    if len(layer_colors_inner_to_outer) != len(layer_sizes_inner_to_outer):
        raise ValueError("Colors and sizes must have the same number of layers.")
    if not layer_colors_inner_to_outer:
        raise ValueError("At least one layer is required.")

    cur = np.full(
        (int(layer_sizes_inner_to_outer[0]), int(layer_sizes_inner_to_outer[0])),
        int(layer_colors_inner_to_outer[0]),
        dtype=int,
    )

    for col, thickness in zip(layer_colors_inner_to_outer[1:], layer_sizes_inner_to_outer[1:]):
        thickness = int(thickness)
        new_n = int(cur.shape[0] + 2 * thickness)
        nxt = np.full((new_n, new_n), int(col), dtype=int)
        nxt[thickness : thickness + cur.shape[0], thickness : thickness + cur.shape[1]] = cur
        cur = nxt

    return cur


def _solve_combo_from_runs_top_down(runs_top_down: list[tuple[int, int]]) -> np.ndarray:
    """Active combo behavior: top-down order, decreasing sizes, adjacent max colors."""
    stripe_colors = [int(col) for col, _height in runs_top_down]
    layer_sizes = _decreasing_layer_sizes(len(runs_top_down))
    layer_colors = _mixed_layer_colors_top_down(stripe_colors)
    return _render_nested(layer_colors, layer_sizes)


def _solve_bottom_up_combo_from_runs(runs_top_down: list[tuple[int, int]]) -> np.ndarray:
    """
    Direction-only comparison baseline.

    This keeps the combo geometry and adjacent-max rule, but reads the stripes
    bottom-up to verify that D1/0 is visible on generated samples.
    """
    runs_bottom_up = list(reversed(runs_top_down))
    stripe_colors = [int(col) for col, _height in runs_bottom_up]
    layer_sizes = _decreasing_layer_sizes(len(runs_bottom_up))
    layer_colors = _mixed_layer_colors_top_down(stripe_colors)
    return _render_nested(layer_colors, layer_sizes)


def _solve_top_down_height_mixed_from_runs(runs_top_down: list[tuple[int, int]]) -> np.ndarray:
    """Geometry comparison baseline: top-down, original heights, combo colors."""
    stripe_colors = [int(col) for col, _height in runs_top_down]
    heights = [int(height) for _col, height in runs_top_down]
    layer_colors = _mixed_layer_colors_top_down(stripe_colors)
    return _render_nested(layer_colors, heights)


def _solve_top_down_decreasing_direct_from_runs(runs_top_down: list[tuple[int, int]]) -> np.ndarray:
    """Color comparison baseline: top-down and decreasing sizes, but no mixing."""
    stripe_colors = [int(col) for col, _height in runs_top_down]
    layer_sizes = _decreasing_layer_sizes(len(runs_top_down))
    return _render_nested(stripe_colors, layer_sizes)


def _behavior_metrics(runs_top_down: list[tuple[int, int]]) -> dict[str, int]:
    """Quantify whether each combo control variable is active on a sample."""
    stripe_colors = [int(col) for col, _height in runs_top_down]
    heights = [int(height) for _col, height in runs_top_down]
    layer_sizes = _decreasing_layer_sizes(len(runs_top_down))
    mixed_colors = _mixed_layer_colors_top_down(stripe_colors)
    rise_count = sum(int(stripe_colors[idx] > stripe_colors[idx - 1]) for idx in range(1, len(stripe_colors)))
    fall_count = sum(int(stripe_colors[idx] < stripe_colors[idx - 1]) for idx in range(1, len(stripe_colors)))
    return {
        "height_mismatch_count": int(sum(int(h != s) for h, s in zip(heights, layer_sizes))),
        "mixed_layer_change_count": int(
            sum(int(mixed_colors[idx] != stripe_colors[idx]) for idx in range(1, len(stripe_colors)))
        ),
        "rise_count": int(rise_count),
        "fall_count": int(fall_count),
        "mixed_color_variety": int(len(set(mixed_colors))),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from a stack of horizontal color stripes.

    Combo rule:
    - Read stripes top to bottom.
    - If there are n stripes, the center-to-outer sizes are [n, n-1, ..., 1].
    - The top stripe becomes the center solid square.
    - Each outer frame color is max(current stripe color, immediately inner stripe color)
      using the top-down stripe order.
    """
    runs_top_down = _parse_runs_top_down(np.array(grid, dtype=int))
    return _solve_combo_from_runs_top_down(runs_top_down)


def _sample_heights(n_runs: int) -> list[int] | None:
    """Sample distractor heights that visibly disagree with the decreasing geometry."""
    target_sizes = _decreasing_layer_sizes(n_runs)
    for _ in range(80):
        heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
        if len(set(heights)) == n_runs:
            continue
        if sum(int(h != s) for h, s in zip(heights, target_sizes)) < 2:
            continue
        return heights
    return None


def _sample_colors(n_runs: int) -> list[int] | None:
    """Sample a compact color sequence that makes the adjacent-max rule visible."""
    palette = [2, 5, 8]
    for _ in range(120):
        colors = [int(palette[int(np.random.randint(0, len(palette)))])]
        while len(colors) < n_runs:
            c = int(palette[int(np.random.randint(0, len(palette)))])
            if c != colors[-1]:
                colors.append(c)

        if len(set(colors)) < 3:
            continue

        metrics = _behavior_metrics([(col, 1) for col in colors])
        if metrics["rise_count"] == 0 or metrics["fall_count"] == 0:
            continue
        if metrics["mixed_layer_change_count"] == 0:
            continue
        if metrics["mixed_color_variety"] < 2:
            continue
        return colors
    return None


def _gen_attempt() -> np.ndarray | None:
    n_runs = int(np.random.randint(4, 6))
    heights = _sample_heights(n_runs)
    if heights is None:
        return None

    colors = _sample_colors(n_runs)
    if colors is None:
        return None

    width = int(np.random.randint(6, 16))
    stripes = [np.full((int(h), width), int(col), dtype=int) for col, h in zip(colors, heights)]
    grid = np.vstack(stripes)

    runs_top_down = _parse_runs_top_down(grid)
    metrics = _behavior_metrics(runs_top_down)
    combo_out = _solve_combo_from_runs_top_down(runs_top_down)
    bottom_up_combo_out = _solve_bottom_up_combo_from_runs(runs_top_down)
    height_mixed_out = _solve_top_down_height_mixed_from_runs(runs_top_down)
    direct_color_out = _solve_top_down_decreasing_direct_from_runs(runs_top_down)

    if metrics["height_mismatch_count"] < 2:
        return None
    if metrics["mixed_layer_change_count"] < 1:
        return None
    if metrics["rise_count"] < 1 or metrics["fall_count"] < 1:
        return None
    if metrics["mixed_color_variety"] < 2:
        return None
    if np.array_equal(combo_out, bottom_up_combo_out):
        return None
    if np.array_equal(combo_out, height_mixed_out):
        return None
    if np.array_equal(combo_out, direct_color_out):
        return None
    if not np.array_equal(main(grid), combo_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
