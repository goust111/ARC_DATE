# Combo variant of 45a5af55 (index 6)
# combo_spec: D3/0+D6/2
# Name: 递减厚度混色嵌套

from __future__ import annotations

import numpy as np


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
# - Baseline task: parse full-width horizontal stripes from bottom to top as
#   (color, height). The bottom stripe becomes the center solid square with side
#   length = its height. Each later stripe wraps the current square with a frame
#   of thickness = its own height and color = its own stripe color.
# - Combo geometry change (D3/0): if there are n stripes total, geometric sizes
#   become [n, n-1, ..., 1] from inner to outer, regardless of input heights.
# - Combo color change (D6/2): layer 0 keeps the bottom stripe color because it
#   has no inner neighbor. Every outer layer i>=1 uses
#   max(stripe_color[i-1], stripe_color[i]), i.e. the larger value between the
#   current stripe and the immediately inner stripe.
# - Small trace with runs bottom->top:
#   colors  = [2, 7, 4, 9]
#   heights = [1, 4, 2, 3]
#   original:
#     center side 1 color 2
#     then +4 color 7, +2 color 4, +3 color 9
#   combo:
#     thicknesses = [4, 3, 2, 1]
#     layer colors = [2, max(2,7), max(7,4), max(4,9)] = [2, 7, 7, 9]
#   Differences:
#   - geometry no longer reads stripe heights at all;
#   - outer colors are not owned by their own stripe alone, so the third layer
#     above stays 7 instead of reverting to 4.
#
# b) Complexity calibration
# - The generator uses 4-5 stripes. Fewer layers weaken both clues; more layers
#   would push the deterministic n^2 output size above a tidy ARC scale.
# - Heights are sampled as distractors and must disagree with the variant
#   thicknesses on at least two layers. Repeated heights are encouraged so a
#   solver cannot explain geometry via a simple one-to-one height reading.
# - Colors are sampled without replacement, and we require at least two outer
#   layers whose mixed color differs from their own stripe color. This keeps the
#   color-fusion effect visible rather than collapsing back to raw colors.
#
# c) Hidden-assumption audit of the original main()
# - Parsing consecutive full-width uniform rows is still valid; input structure
#   is unchanged.
# - `c0, h0 = runs[0]` was no longer valid for geometry because the center size
#   now depends on layer count, not on bottom stripe height.
# - `for col, h in runs[1:]` also broke in two ways:
#   `h` can no longer be used as thickness, and `col` can no longer be used as
#   the final layer color directly.
# - The square-wrapping construction itself remains safe once it is driven by
#   recomputed thicknesses and transformed colors.
#
# d) Placement legality
# - The generator still creates one vertical stack of full-width stripes, so no
#   free placement, overlap, or boundary issues arise in the input.
# - With n in {4, 5}, the output side is exactly n^2, hence at most 25.
# - Adjacent stripe colors must differ or the parser would merge runs; generator
#   enforces sampling without replacement.
#
# e) Visual observability
# - Thickness is visually recoverable because every sample shows the same
#   inside-thick / outside-thin progression even when input stripe heights vary.
# - The color rule is less human-friendly because it depends on ARC color ids.
#   To reduce opacity, the generator forces several outer layers to inherit the
#   inner-adjacent color instead of their own stripe color, so the mixed-color
#   effect is clearly present in the training distribution.
#
# f) Visual cleanliness
# - Output remains a single nested-square object with one flat color per layer.
#   No extra markers or overlapping object families are introduced.
#
# g) Color-change inferability
# - The mapping is structured locally: each outer layer depends only on a stripe
#   pair (inner,current), never on a global lookup table.
# - Generator checks prevent color degeneracy by requiring multiple changed
#   layers and at least three distinct output colors across the nested layers.
#
# h) Anti-degeneracy policy
# - Samples are rejected if the combo output equals:
#   1. the original task output,
#   2. the D3-only output (changed geometry, raw colors), or
#   3. the color-only output (mixed colors, original heights).
#   This ensures both fused dimensions are active in each accepted sample.
#
# i) New assumptions introduced by the combo solver
# - `_mixed_layer_colors` assumes layer order matters and every outer layer has
#   exactly one inner neighbor. The generator therefore preserves a simple pure
#   stripe stack with no accidental merged runs.
# - The solver assumes every row of a stripe is uniform across width; generator
#   guarantees this via `np.full`.
# - Because the center layer has no inner neighbor, the implementation leaves
#   its color unchanged; the generator is consistent with that convention.


def _parse_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    """Parse full-width uniform color stripes from bottom to top."""
    g = np.array(grid, dtype=int)
    h, _w = g.shape

    runs: list[tuple[int, int]] = []
    r = int(h - 1)
    while r >= 0:
        row = g[r]
        col = int(row[0])
        if not bool((row == col).all()):
            raise ValueError("Each stripe row must be uniform across the width.")

        height = 1
        rr = int(r - 1)
        while rr >= 0 and bool((g[rr] == col).all()):
            height += 1
            rr -= 1
        runs.append((col, height))
        r = int(rr)

    return runs


def _variant_thicknesses(n_runs: int) -> list[int]:
    """Bottom->top geometric sizes for the combo variant."""
    return [int(n_runs - idx) for idx in range(n_runs)]


def _mixed_layer_colors(colors_bottom_to_top: list[int]) -> list[int]:
    """
    Center keeps its own color. Every outer layer takes the larger value between
    its own stripe color and the immediately inner stripe color.
    """
    if not colors_bottom_to_top:
        raise ValueError("At least one layer is required.")

    mixed = [int(colors_bottom_to_top[0])]
    for idx in range(1, len(colors_bottom_to_top)):
        mixed.append(int(max(colors_bottom_to_top[idx - 1], colors_bottom_to_top[idx])))
    return mixed


def _render_nested(colors_bottom_to_top: list[int], layer_sizes_bottom_to_top: list[int]) -> np.ndarray:
    """Render nested squares where layer_sizes[0] is the center side length."""
    if len(colors_bottom_to_top) != len(layer_sizes_bottom_to_top) or not colors_bottom_to_top:
        raise ValueError("Colors and layer sizes must be non-empty and aligned.")

    cur = np.full(
        (int(layer_sizes_bottom_to_top[0]), int(layer_sizes_bottom_to_top[0])),
        int(colors_bottom_to_top[0]),
        dtype=int,
    )

    for col, thickness in zip(colors_bottom_to_top[1:], layer_sizes_bottom_to_top[1:]):
        thickness = int(thickness)
        new_n = int(cur.shape[0] + 2 * thickness)
        nxt = np.full((new_n, new_n), int(col), dtype=int)
        nxt[thickness : thickness + cur.shape[0], thickness : thickness + cur.shape[1]] = cur
        cur = nxt

    return cur


def _original_output(runs: list[tuple[int, int]]) -> np.ndarray:
    colors = [int(col) for col, _height in runs]
    heights = [int(height) for _col, height in runs]
    return _render_nested(colors, heights)


def _d3_only_output(runs: list[tuple[int, int]]) -> np.ndarray:
    colors = [int(col) for col, _height in runs]
    return _render_nested(colors, _variant_thicknesses(len(runs)))


def _color_only_output(runs: list[tuple[int, int]]) -> np.ndarray:
    colors = _mixed_layer_colors([int(col) for col, _height in runs])
    heights = [int(height) for _col, height in runs]
    return _render_nested(colors, heights)


def _combo_output(runs: list[tuple[int, int]]) -> np.ndarray:
    colors = _mixed_layer_colors([int(col) for col, _height in runs])
    thicknesses = _variant_thicknesses(len(runs))
    return _render_nested(colors, thicknesses)


def _behavior_signature(runs: list[tuple[int, int]]) -> tuple[int, int, int]:
    """
    Return counts that quantify whether both fused controls are active:
    - height_mismatches: how many stripe heights disagree with combo thicknesses
    - mixed_color_changes: how many outer layers changed color vs raw stripes
    - distinct_combo_colors: number of distinct colors in combo layer sequence
    """
    heights = [int(height) for _col, height in runs]
    colors = [int(col) for col, _height in runs]
    thicknesses = _variant_thicknesses(len(runs))
    mixed_colors = _mixed_layer_colors(colors)

    height_mismatches = sum(int(h != t) for h, t in zip(heights, thicknesses))
    mixed_color_changes = sum(
        int(mixed_colors[idx] != colors[idx])
        for idx in range(1, len(colors))
    )
    distinct_combo_colors = len(set(mixed_colors))
    return height_mismatches, mixed_color_changes, distinct_combo_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square layers from a stack of full-width stripes.

    Combo rule:
    - Parse stripes from bottom to top.
    - If there are n stripes, geometric sizes are [n, n-1, ..., 1].
    - The center square keeps the bottom stripe color.
    - Each outer layer uses max(current stripe color, inner-adjacent stripe
      color) instead of using its own stripe color directly.
    """
    runs = _parse_runs(np.array(grid, dtype=int))
    if not runs:
        raise ValueError("Input must contain at least one stripe.")
    return _combo_output(runs)


def _gen_attempt() -> np.ndarray | None:
    n_runs = int(np.random.choice([4, 5]))
    thicknesses = _variant_thicknesses(n_runs)

    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    if len(set(heights)) == n_runs:
        return None
    if sum(int(h != t) for h, t in zip(heights, thicknesses)) < 2:
        return None

    colors = [int(x) for x in np.random.permutation(np.arange(1, 10))[:n_runs]]
    runs = list(zip(colors, heights))

    height_mismatches, mixed_color_changes, distinct_combo_colors = _behavior_signature(runs)
    if height_mismatches < 2:
        return None
    if mixed_color_changes < 2:
        return None
    if distinct_combo_colors < 3:
        return None

    combo = _combo_output(runs)
    if np.array_equal(combo, _original_output(runs)):
        return None
    if np.array_equal(combo, _d3_only_output(runs)):
        return None
    if np.array_equal(combo, _color_only_output(runs)):
        return None

    width = int(np.random.randint(6, 16))
    stripes = [
        np.full((int(height), width), int(col), dtype=int)
        for col, height in reversed(runs)
    ]
    grid = np.vstack(stripes)

    parsed_runs = _parse_runs(grid)
    if parsed_runs != runs:
        return None
    if not np.array_equal(main(grid), combo):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
