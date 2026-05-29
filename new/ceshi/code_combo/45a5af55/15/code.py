# Combo variant of 45a5af55 (index 15)
# combo_spec: D10/2+D11/1
# Name: 数量编码多层单条嵌套

from __future__ import annotations

import numpy as np

"""
Design analysis for the fused variant.

a) Concrete behavior and trace
   Base task parses full-width horizontal stripes bottom->top as `(color, height)`.
   The bottom stripe becomes a solid `height x height` center; each later stripe wraps
   a same-color square frame of thickness `height`.

   This combo changes two linked stages at once:
   1. D11/1: a stripe of height N no longer means "one layer of thickness N". It means
      "N consecutive one-cell layers of the same color". So `(2, 3), (5, 2)` expands to
      the unit-layer sequence `[2, 2, 2, 5, 5]`.
   2. D10/2: the number of source stripes globally transforms that expanded sequence
      before rasterization. Directly rotating/flipping the final concentric square would
      be visually latent because perfect nested squares are D4-symmetric, so the count
      control is concretized on the radial instruction sequence itself:
      - 2 stripes: reverse the full unit-layer sequence (horizontal-flip analogue)
      - 3 stripes: rotate the full unit-layer sequence left by 1 (90-degree analogue)

   Small trace:
   - Input bottom->top runs: `(4, 2), (7, 1), (3, 2)`.
   - Original rule: build layers `[4 thickness 2] -> [7 thickness 1] -> [3 thickness 2]`.
   - Fused rule: expand to `[4, 4, 7, 3, 3]`; there are 3 stripes, so rotate left by 1
     to get `[4, 7, 3, 3, 4]`; then render a 1x1 center with successive 1-cell frames.
   Difference:
   - Height is now interpreted before geometry as repeated one-cell layers.
   - Stripe count no longer changes a hidden symmetry; it visibly rearranges the radial
     color sequence, often splitting one stripe's color into inner and outer rings.

b) Complexity calibration
   The generator only uses 2 or 3 stripes. That is deliberate:
   - 2 stripes supports the "reverse / flip analogue" case.
   - 3 stripes supports the "rotate by one layer / 90-degree analogue" case.
   More control states would be harder to infer from a few ARC examples and would not add
   useful visual evidence. Heights stay in 1..4 and at least one stripe has height > 1,
   so D11 is present in every sample.

c) Implicit assumptions review against the original main()
   - Original row parser assumed each stripe row is uniform across width. Still true.
   - Original constructor assumed one parsed stripe corresponds to one rendered layer whose
     thickness equals height. Broken by D11/1; we must expand each run into N unit layers.
   - Original logic assumed bottom->top run order is the same as inner->outer layer order.
     Broken by D10/2; after expansion, count-controlled sequence transform changes it.
   - Original output size formula depended on grouped thicknesses. New size depends only on
     the number of unit layers: side = `2 * len(expanded_layers) - 1`.

d) Placement legality
   Generator placement is simple and legal: it still creates only stacked full-width
   stripes with no gaps, no overlap, and no out-of-bounds writes. The output geometry is
   constructed analytically from the parsed runs, so there is no spatial search step.

e) Visual observability
   Literal flips/rotations of perfect concentric squares are invisible, so the count-coded
   transformation must act before rasterization to remain observable. The chosen sequence
   transforms are global, count-controlled, and easy to compare across examples:
   - 2 stripes: the two color groups swap inner/outer roles.
   - 3 stripes: one repeated color often appears both near the center and at the outside.

f) Visual tidiness
   Inputs remain clean stripe stacks. Outputs remain single compact nested-square objects.
   No extra markers or clutter are introduced; the only novelty is radial color ordering.

g) Inferrability of the fused change
   The generator enforces a visible non-base effect:
   - output from the fused rule must differ from the original task output;
   - 3-stripe cases force the bottom stripe height >= 2, so after the count transform that
     stripe's color becomes split across disjoint radial positions, which cannot happen in
     the original grouped-thickness interpretation.

h) Variant behavior metric
   For a generated sample we track:
   - `changed_positions`: how many unit-layer positions differ between the canonical
     expanded sequence and the count-transformed sequence;
   - `split_color_count`: how many colors occupy more than one disjoint radial segment in
     the transformed sequence.
   A 3-stripe sample is accepted only if `split_color_count >= 1`, guaranteeing genuine
   fusion rather than a simple run reorder.

i) New solver assumptions and generator guarantees
   - Solver assumes the grid is a non-empty stack of full-width uniform stripes. Generator
     guarantees exactly that.
   - Solver assumes at least one stripe exists. Generator uses 2 or 3.
   - Solver assumes adjacent stripes have different colors so run parsing is unambiguous.
     Generator samples distinct colors without adjacent repeats.
   - Solver assumes the transformed layer sequence is non-empty. Heights are positive.
"""


def _parse_runs_bottom_to_top(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.array(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        raise ValueError("grid must be a non-empty 2D array")

    h, _w = g.shape
    runs: list[tuple[int, int]] = []
    r = int(h - 1)
    while r >= 0:
        color = int(g[r, 0])
        if not bool((g[r] == color).all()):
            raise ValueError("each stripe row must be uniform across width")
        height = 1
        rr = int(r - 1)
        while rr >= 0:
            if not bool((g[rr] == int(g[rr, 0])).all()):
                raise ValueError("each stripe row must be uniform across width")
            if int(g[rr, 0]) != color:
                break
            height += 1
            rr -= 1
        runs.append((int(color), int(height)))
        r = int(rr)

    if not runs:
        raise ValueError("expected at least one stripe")
    return runs


def _expand_runs_to_unit_layers(runs_bottom_to_top: list[tuple[int, int]]) -> list[int]:
    layers: list[int] = []
    for color, height in runs_bottom_to_top:
        layers.extend([int(color)] * int(height))
    return layers


def _transform_layers_by_count(layers: list[int], stripe_count: int) -> list[int]:
    if not layers:
        raise ValueError("layer sequence must be non-empty")
    if stripe_count <= 1:
        return list(layers)
    if stripe_count == 2:
        return list(reversed(layers))

    shift = 1 if len(layers) > 1 else 0
    return list(layers[shift:] + layers[:shift])


def _render_unit_layers(layers_inner_to_outer: list[int]) -> np.ndarray:
    if not layers_inner_to_outer:
        raise ValueError("layer sequence must be non-empty")

    cur = np.full((1, 1), int(layers_inner_to_outer[0]), dtype=int)
    for color in layers_inner_to_outer[1:]:
        new_side = int(cur.shape[0] + 2)
        nxt = np.full((new_side, new_side), int(color), dtype=int)
        nxt[1:-1, 1:-1] = cur
        cur = nxt
    return cur


def _render_original_runs(runs_bottom_to_top: list[tuple[int, int]]) -> np.ndarray:
    color0, height0 = runs_bottom_to_top[0]
    cur = np.full((int(height0), int(height0)), int(color0), dtype=int)
    for color, height in runs_bottom_to_top[1:]:
        new_side = int(cur.shape[0] + 2 * int(height))
        nxt = np.full((new_side, new_side), int(color), dtype=int)
        nxt[height : height + cur.shape[0], height : height + cur.shape[1]] = cur
        cur = nxt
    return cur


def _split_color_count(layers_inner_to_outer: list[int]) -> int:
    segments: dict[int, int] = {}
    prev = None
    for color in layers_inner_to_outer:
        if color != prev:
            segments[int(color)] = int(segments.get(int(color), 0) + 1)
        prev = int(color)
    return sum(1 for count in segments.values() if int(count) > 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Parse bottom->top stripes, expand heights into repeated unit layers, apply the
    stripe-count control to that layer sequence, then rasterize nested 1-cell frames.
    """
    runs = _parse_runs_bottom_to_top(np.array(grid, dtype=int))
    base_layers = _expand_runs_to_unit_layers(runs)
    transformed_layers = _transform_layers_by_count(base_layers, len(runs))
    return _render_unit_layers(transformed_layers)


def _gen_attempt() -> np.ndarray | None:
    stripe_count = int(np.random.choice([2, 3]))
    width = int(np.random.randint(6, 15))

    colors = [int(c) for c in np.random.choice(np.arange(1, 10), size=stripe_count, replace=False)]
    heights = [int(np.random.randint(1, 5)) for _ in range(stripe_count)]

    if sum(heights) > 15:
        return None
    if all(int(h) == 1 for h in heights):
        return None

    if stripe_count == 3:
        # Force the bottom stripe to expand to multiple unit layers so the count-controlled
        # rotation can split that color across separated radial positions.
        heights[0] = max(2, int(heights[0]))
        if sum(heights) > 15:
            return None

    runs_bottom_to_top = [(int(color), int(height)) for color, height in zip(colors, heights)]
    stripes_top_to_bottom = [
        np.full((int(height), width), int(color), dtype=int)
        for color, height in reversed(runs_bottom_to_top)
    ]
    inp = np.vstack(stripes_top_to_bottom)

    base_layers = _expand_runs_to_unit_layers(runs_bottom_to_top)
    transformed_layers = _transform_layers_by_count(base_layers, stripe_count)
    fused_out = _render_unit_layers(transformed_layers)
    original_out = _render_original_runs(runs_bottom_to_top)

    changed_positions = int(sum(a != b for a, b in zip(base_layers, transformed_layers)))
    if changed_positions == 0:
        return None
    if stripe_count == 3 and _split_color_count(transformed_layers) < 1:
        return None
    if np.array_equal(fused_out, original_out):
        return None
    if not np.array_equal(main(inp), fused_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
