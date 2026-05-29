# Combo variant of 45a5af55 (index 21)
# combo_spec: D3/0+D6/1+D8/0
# Name: 高度排序递减互换色嵌套
#
# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    - Original task:
#      parse full-width color stripes from bottom to top, use that same order as
#      inner->outer layer order, and reuse each stripe height directly as the
#      center size / frame thickness.
#    - Fused task:
#      1. Parse the same stripes bottom->top, but do not trust that order as the
#         nesting order.
#      2. Reorder stripes by height descending to get the inner->outer layer
#         sequence. Stable tie-break would keep bottom->top order, but the
#         generator avoids ties so humans only need the visible height rule.
#      3. Replace the geometry with layer-index sizes [n, n-1, ..., 1], where n
#         is the number of stripes. These sizes are attached to the sorted layer
#         order, not to bottom->top order and not to the stripes' raw heights.
#      4. Swap adjacent layer colors on that sorted sequence: layers (1,2),
#         (3,4), ... exchange colors, while geometry stays in place.
#    - Small trace with bottom->top runs [(1,2), (4,4), (7,1)]:
#      original: order [(1,2),(4,4),(7,1)] -> colors [1,4,7], sizes [2,4,1]
#                => center color 1 size 2, then +4 color 4, then +1 color 7.
#      fused:    sort by height desc -> [(4,4),(1,2),(7,1)]
#                layer sizes become [3,2,1]
#                colors on sorted layers [4,1,7] swap to [1,4,7]
#                => center color 1 size 3, then +2 color 4, then +1 color 7.
#      Differences:
#      - the tallest stripe, even when not at the bottom, becomes the innermost
#        layer owner before the color-swap step;
#      - thickness no longer equals any stripe's height;
#      - color assignment is determined by pairwise swapping after sorting.
#    - Spec note:
#      the combo prose contains one conflicting sentence claiming the tallest
#      stripe becomes the outermost thinnest frame. That conflicts with both the
#      explicit D8/0 detail ("height descending determines nesting order" with
#      tallest as center) and the D3/0 detail ("[n..1] from inner to outer").
#      This implementation follows the explicit per-dimension details because
#      they are mutually consistent and operationally precise.
# b) Complexity calibration:
#    - The generator uses exactly 3 stripes. That is enough to show all three
#      fused effects while avoiding the 4-stripe degeneracy where unique heights
#      {1,2,3,4} would sort to [4,3,2,1], numerically matching the D3 geometry.
#    - Heights are unique and sampled from a 3-element subset of {1,2,3,4} that
#      must include 4 but cannot be {1,2,3}. This keeps D8 visually readable and
#      guarantees the sorted visible heights differ from the D3 layer sizes [3,2,1].
#    - Colors are distinct, so the D6 swap is visibly recoverable from a few
#      examples rather than collapsing under accidental equal-color pairs.
# c) Implicit-assumption audit against the original main():
#    - Parsing consecutive equal-color full rows is still valid.
#    - `runs[0]` is no longer the center layer; the original assumption fails
#      because layer order is now height-sorted.
#    - The old `h0 = runs[0][1]` center-size assumption fails; the center size is
#      now `len(runs)` regardless of the winning stripe's raw height.
#    - The old loop `for col, h in runs[1:]` assumed one stripe supplied both
#      color and thickness to one layer. In the fused task, thickness comes from
#      the layer index, while color may come from the adjacent sorted layer after
#      swapping. Those concerns must be split apart.
#    - The fill-and-insert construction remains valid because the new thicknesses
#      are all positive and still define clean nested squares.
# d) Placement legality:
#    - Input placement is unchanged: one vertical stack of full-width stripes, so
#      there is no overlap or out-of-bounds placement issue.
#    - With 3 layers and sizes [3,2,1], every output is 9x9 because
#      3 + 2*(2+1) = 9. This stays well within ARC-friendly bounds.
# e) Visual observability:
#    - Unique heights make the height-descending layer order visible.
#    - Rejecting samples where the tallest stripe is already at the bottom avoids
#      collapse back to the original spatial-order rule.
#    - Distinct colors plus a consistent swap of the first two sorted layers make
#      the color-exchange pattern observable across examples.
# f) Visual cleanliness:
#    - The task still shows one clean stripe stack in the input and one clean
#      nested-square object in the output. No extra markers or overlays are added.
# g) Color-rule inferability:
#    - Only adjacent sorted layers swap colors. This is structured, low-entropy,
#      and repeated across all generated examples. No hidden arithmetic on color
#      values is used.
# h) Anti-degeneracy policy:
#    - The generator rejects any sample where the fused output equals the original
#      task output or any single-dimension reference behavior (D3-only, D6-only,
#      D8-only). This ensures the sample actually exercises the fusion.
# i) New main() assumptions audited against generate_input():
#    - Solver assumption: every stripe is full-width and row-uniform.
#      Generator guarantee: each stripe is built with `np.full`.
#    - Solver assumption: adjacent stripes have different colors so parsed runs
#      match intended layer count. Generator guarantee: colors are sampled without
#      replacement.
#    - Solver assumption: height-based ordering is visually meaningful.
#      Generator guarantee: stripe heights are unique.
#    - Solver assumption: D3 sizes should not visually collapse to sorted heights.
#      Generator guarantee: sampled height set is never {1,2,3}.

from __future__ import annotations

import numpy as np


def _parse_runs_bottom_to_top(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return full-width stripe runs as (color, height) from bottom to top."""
    g = np.array(grid, dtype=int)
    h, _w = g.shape

    runs: list[tuple[int, int]] = []
    r = int(h - 1)
    while r >= 0:
        row = g[r]
        col = int(row[0])
        if not bool((row == col).all()):
            raise ValueError("Each stripe row must be uniform across the full width.")

        run_h = 1
        rr = int(r - 1)
        while rr >= 0 and bool((g[rr] == col).all()):
            run_h += 1
            rr -= 1
        runs.append((col, run_h))
        r = int(rr)

    return runs


def _order_runs_by_height_desc(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Stable height-descending order, interpreted as inner->outer layers."""
    indexed_runs = list(enumerate(runs))
    ordered = sorted(indexed_runs, key=lambda item: (-item[1][1], item[0]))
    return [run for _idx, run in ordered]


def _variant_layer_sizes(n_layers: int) -> list[int]:
    """Inner->outer geometric sizes: n, n-1, ..., 1."""
    return [int(n_layers - idx) for idx in range(n_layers)]


def _swap_adjacent_colors(colors: list[int]) -> list[int]:
    """Swap colors of layers (1,2), (3,4), ... while preserving layer geometry."""
    swapped = list(colors)
    for i in range(0, len(swapped) - 1, 2):
        swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
    return swapped


def _build_nested(colors_inner_to_outer: list[int], sizes_inner_to_outer: list[int]) -> np.ndarray:
    """Render the nested squares, where sizes[0] is the center side length."""
    if not colors_inner_to_outer or not sizes_inner_to_outer:
        raise ValueError("At least one layer is required.")
    if len(colors_inner_to_outer) != len(sizes_inner_to_outer):
        raise ValueError("Color and geometry sequences must have the same length.")

    cur = np.full(
        (int(sizes_inner_to_outer[0]), int(sizes_inner_to_outer[0])),
        int(colors_inner_to_outer[0]),
        dtype=int,
    )
    for col, thickness in zip(colors_inner_to_outer[1:], sizes_inner_to_outer[1:]):
        thickness = int(thickness)
        new_n = int(cur.shape[0] + 2 * thickness)
        nxt = np.full((new_n, new_n), int(col), dtype=int)
        nxt[thickness : thickness + cur.shape[0], thickness : thickness + cur.shape[1]] = cur
        cur = nxt
    return cur


def _original_output(runs_bottom_to_top: list[tuple[int, int]]) -> np.ndarray:
    colors = [col for col, _height in runs_bottom_to_top]
    heights = [height for _col, height in runs_bottom_to_top]
    return _build_nested(colors, heights)


def _d3_only_output(runs_bottom_to_top: list[tuple[int, int]]) -> np.ndarray:
    colors = [col for col, _height in runs_bottom_to_top]
    return _build_nested(colors, _variant_layer_sizes(len(runs_bottom_to_top)))


def _d6_only_output(runs_bottom_to_top: list[tuple[int, int]]) -> np.ndarray:
    colors = [col for col, _height in runs_bottom_to_top]
    heights = [height for _col, height in runs_bottom_to_top]
    return _build_nested(_swap_adjacent_colors(colors), heights)


def _d8_only_output(runs_bottom_to_top: list[tuple[int, int]]) -> np.ndarray:
    ordered_runs = _order_runs_by_height_desc(runs_bottom_to_top)
    colors = [col for col, _height in ordered_runs]
    heights = [height for _col, height in ordered_runs]
    return _build_nested(colors, heights)


def _fused_output(runs_bottom_to_top: list[tuple[int, int]]) -> np.ndarray:
    ordered_runs = _order_runs_by_height_desc(runs_bottom_to_top)
    colors = [col for col, _height in ordered_runs]
    swapped_colors = _swap_adjacent_colors(colors)
    sizes = _variant_layer_sizes(len(ordered_runs))
    return _build_nested(swapped_colors, sizes)


def _fusion_metrics(runs_bottom_to_top: list[tuple[int, int]]) -> dict[str, int]:
    """Quantify whether the three fused controls are all active on this sample."""
    ordered_runs = _order_runs_by_height_desc(runs_bottom_to_top)
    raw_heights = [height for _col, height in runs_bottom_to_top]
    ordered_heights = [height for _col, height in ordered_runs]
    ordered_colors = [col for col, _height in ordered_runs]
    swapped_colors = _swap_adjacent_colors(ordered_colors)
    layer_sizes = _variant_layer_sizes(len(ordered_runs))

    return {
        "sorted_order_changes": int(ordered_runs != runs_bottom_to_top),
        "d3_geometry_changes": int(ordered_heights != layer_sizes),
        "color_swap_changes": int(swapped_colors != ordered_colors),
        "bottom_max_not_center": int(int(np.argmax(raw_heights)) != 0),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from stacked horizontal stripes.

    Fused rule:
    - Parse full-width stripes from bottom to top.
    - Reorder stripes by height descending to determine inner->outer layer order.
    - Replace geometry with layer sizes [n, n-1, ..., 1].
    - Swap adjacent colors on that sorted layer sequence.
    - Render the center square plus outward frames from those fused attributes.
    """
    runs = _parse_runs_bottom_to_top(np.array(grid, dtype=int))
    if not runs:
        raise ValueError("Input must contain at least one stripe.")
    return _fused_output(runs)


def _gen_attempt() -> np.ndarray | None:
    n_runs = 3

    height_pool = [1, 2, 3, 4]
    heights = list(np.random.choice(height_pool, size=n_runs, replace=False))
    if 4 not in heights:
        return None
    if set(heights) == {1, 2, 3}:
        return None

    np.random.shuffle(heights)  # bottom -> top order
    if heights == sorted(heights, reverse=True):
        return None
    if int(np.argmax(heights)) == 0:
        return None

    width = int(np.random.randint(6, 16))
    colors = [int(x) for x in np.random.permutation(np.arange(1, 10))[:n_runs]]
    runs = list(zip(colors, heights))

    metrics = _fusion_metrics(runs)
    if min(metrics.values()) != 1:
        return None

    fused = _fused_output(runs)
    if np.array_equal(fused, _original_output(runs)):
        return None
    if np.array_equal(fused, _d3_only_output(runs)):
        return None
    if np.array_equal(fused, _d6_only_output(runs)):
        return None
    if np.array_equal(fused, _d8_only_output(runs)):
        return None

    stripes = [
        np.full((int(height), width), int(col), dtype=int)
        for col, height in reversed(runs)
    ]
    grid = np.vstack(stripes)

    parsed_runs = _parse_runs_bottom_to_top(grid)
    if parsed_runs != runs:
        return None
    if not np.array_equal(main(grid), fused):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
