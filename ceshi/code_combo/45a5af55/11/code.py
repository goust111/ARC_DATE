# Combo variant of 45a5af55 (index 11)
# combo_spec: D6/1+D8/0
# Name: 高度排序互换色嵌套

from __future__ import annotations

import numpy as np

"""
Design analysis for the fused variant.

a) Concrete behavior and trace
   Example input parsed bottom->top as runs:
   [(1, 2), (4, 4), (6, 1), (8, 3)] where each tuple is (color, height).

   Original 45a5af55:
   - use bottom->top order directly
   - layer order inner->outer = [(1, 2), (4, 4), (6, 1), (8, 3)]
   - output colors by layer = [1, 4, 6, 8]
   - output thicknesses / center size = [2, 4, 1, 3]

   D8/0 component:
   - ignore vertical position for nesting order
   - stable-sort by height descending
   - ordered inner->outer = [(4, 4), (8, 3), (1, 2), (6, 1)]

   D6/1 component applied after that ordering:
   - swap adjacent layer colors in pairs while keeping each layer's own height
   - layer 1 gets layer 2 color, layer 2 gets layer 1 color
   - layer 3 gets layer 4 color, layer 4 gets layer 3 color
   - fused inner->outer layers become [(8, 4), (4, 3), (6, 2), (1, 1)]

   The key difference is joint, not independent:
   - heights decide which stripe owns each geometric layer
   - then the colors are swapped on that new layer order
   So the center is the tallest stripe's height, but usually not the tallest
   stripe's own color.

   Spec ambiguity note:
   - The combo summary text contains one line saying "highest -> outermost".
   - The detailed D8/0 item and the provided single-variant reference code both
     define height-descending order as "tallest -> center / innermost".
   - This implementation follows the detailed item plus the reference code,
     because that pair is internally consistent and executable.

b) Complexity calibration
   The generator keeps the object vocabulary minimal: only stacked full-width
   stripes in the input and one concentric-square object in the output.
   Difficulty comes from two coupled latent rules:
   - spatial order is not the nesting order
   - even after discovering the new order, colors are reassigned by adjacent
     swaps on that reordered list
   To keep this human-legible, the generator uses 4 layers most of the time,
   unique heights, unique colors, and rejects monotone height orders that would
   look too close to the original or to a simple reversal.

c) Implicit assumptions review of the original main()
   Original assumptions:
   - each row is uniform across the full width
   - consecutive equal-color rows belong to one stripe run
   - parsed bottom->top order is also the geometric layer order
   - each layer's color and thickness come from the same parsed run

   What breaks in the combo:
   - bottom->top order is no longer the layer order, so `runs[0]` cannot be
     assumed to be the center layer
   - after reordering by height, the layer color may come from the adjacent
     ordered layer, so color and thickness are no longer read from the same run

   What still holds:
   - run parsing itself is still valid because the input object family is
     unchanged
   - the nested-square construction algorithm is still valid once we provide
     the transformed inner->outer layer sequence

d) Placement legality
   The generator emits only vertical stacks of full-width stripes, so there is
   no overlap ambiguity in the input. Output legality is automatic because the
   nested squares are synthesized from layer heights. A hard cap of side <= 30
   is enforced before accepting a sample.

e) Visual observability
   Humans can see stripe heights directly, so sorting by height is visually
   inferable. Equal heights would hide the ordering rule, so the generator uses
   unique heights only. Distinct colors make the adjacent swaps visible. By
   preferring 4 layers, most samples show two swap pairs instead of just one.

f) Visual cleanliness
   There is exactly one structured object in the input and one in the output.
   No markers, overlays, or interacting object types are added, so the result
   stays visually clean despite the harder reasoning rule.

g) Color-change inferability
   The color remapping is not arbitrary. It is governed by adjacency after the
   height sort: (1<->2), (3<->4), ... in layer index space. Because the
   geometry still follows the reordered heights, the learner can align "which
   ring has which thickness" with "which ring received which neighboring color".

h) Generator anti-degeneracy checks
   A sample is rejected if:
   - the height-sorted order equals the original bottom->top order
   - the fused output matches the original output
   - the fused output matches the D8-only output
   - the fused output matches the D6-only output
   This ensures both control variables are active in accepted examples.

i) New assumptions introduced by the fused solver
   - Stable sorting among equal heights is defined using original bottom->top
     order. The generator avoids ties so humans do not need that tie-break, but
     main() still implements it for robustness.
   - Adjacent color swapping is defined on parsed stripe runs, not raw rows, so
     generate_input() must keep neighboring stripes different in color. It does
     this by sampling colors without replacement.
"""


def _parse_runs_bottom_to_top(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return (color, height) stripe runs ordered bottom -> top."""
    g = np.array(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")
    h, w = g.shape
    if h == 0 or w == 0:
        raise ValueError("grid must be non-empty")

    runs: list[tuple[int, int]] = []
    r = h - 1
    while r >= 0:
        color = int(g[r, 0])
        if not bool((g[r] == color).all()):
            raise ValueError("each stripe row must be uniform across the width")
        run_h = 1
        rr = r - 1
        while rr >= 0:
            row_color = int(g[rr, 0])
            if not bool((g[rr] == row_color).all()):
                raise ValueError("each stripe row must be uniform across the width")
            if row_color != color:
                break
            run_h += 1
            rr -= 1
        runs.append((color, run_h))
        r = rr

    if not runs:
        raise ValueError("expected at least one stripe")
    return runs


def _order_runs_by_height_desc(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Interpret tallest stripe as innermost; keep bottom->top order on ties."""
    indexed = list(enumerate(runs))
    return [run for _idx, run in sorted(indexed, key=lambda item: (-item[1][1], item[0]))]


def _swap_adjacent_layer_colors(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Keep each layer's height but swap colors of layers (1,2), (3,4), ..."""
    swapped = list(runs)
    for i in range(0, len(swapped) - 1, 2):
        c0, h0 = swapped[i]
        c1, h1 = swapped[i + 1]
        swapped[i] = (c1, h0)
        swapped[i + 1] = (c0, h1)
    return swapped


def _build_nested_square_layers(runs_by_layer: list[tuple[int, int]]) -> np.ndarray:
    """Build the output from inner -> outer (color, height) layer specs."""
    if not runs_by_layer:
        raise ValueError("expected at least one layer")

    center_color, center_size = runs_by_layer[0]
    cur = np.full((int(center_size), int(center_size)), int(center_color), dtype=int)
    for color, thickness in runs_by_layer[1:]:
        thickness = int(thickness)
        new_side = int(cur.shape[0] + 2 * thickness)
        nxt = np.full((new_side, new_side), int(color), dtype=int)
        nxt[thickness : thickness + cur.shape[0], thickness : thickness + cur.shape[1]] = cur
        cur = nxt
    return cur


def _original_output(runs_bottom_to_top: list[tuple[int, int]]) -> np.ndarray:
    return _build_nested_square_layers(runs_bottom_to_top)


def _d8_only_output(runs_bottom_to_top: list[tuple[int, int]]) -> np.ndarray:
    return _build_nested_square_layers(_order_runs_by_height_desc(runs_bottom_to_top))


def _d6_only_output(runs_bottom_to_top: list[tuple[int, int]]) -> np.ndarray:
    return _build_nested_square_layers(_swap_adjacent_layer_colors(runs_bottom_to_top))


def _combo_layers(runs_bottom_to_top: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = _order_runs_by_height_desc(runs_bottom_to_top)
    return _swap_adjacent_layer_colors(ordered)


def _combo_output(runs_bottom_to_top: list[tuple[int, int]]) -> np.ndarray:
    return _build_nested_square_layers(_combo_layers(runs_bottom_to_top))


def _behavior_metrics(runs_bottom_to_top: list[tuple[int, int]]) -> tuple[int, int]:
    """
    Return:
    - order_displacement: how many stripes move after height sorting
    - recolored_layers: how many layers change color after pair swapping
    """
    ordered = _order_runs_by_height_desc(runs_bottom_to_top)
    order_displacement = sum(1 for a, b in zip(runs_bottom_to_top, ordered) if a != b)
    recolored_layers = sum(
        1 for (orig_c, _), (new_c, _) in zip(ordered, _swap_adjacent_layer_colors(ordered)) if orig_c != new_c
    )
    return order_displacement, recolored_layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve the fused variant.

    Rule:
    - parse full-width horizontal stripes from bottom to top
    - sort those stripe runs by height descending, using bottom->top order as a
      stable tie-break if heights match
    - interpret that sorted list as inner -> outer layer order
    - swap adjacent layer colors on that sorted list
    - keep every layer's own height as its thickness / center size
    """
    runs_bottom_to_top = _parse_runs_bottom_to_top(np.array(grid, dtype=int))
    return _combo_output(runs_bottom_to_top)


def _gen_attempt() -> np.ndarray | None:
    n_runs = int(np.random.choice([3, 4, 4]))
    heights = [int(x) for x in np.random.choice(np.arange(1, 5), size=n_runs, replace=False)]
    np.random.shuffle(heights)

    if heights == sorted(heights, reverse=True):
        return None
    if heights == sorted(heights):
        return None

    max_idx = int(heights.index(max(heights)))
    min_idx = int(heights.index(min(heights)))
    if max_idx in (0, n_runs - 1) and min_idx in (0, n_runs - 1):
        return None

    colors = [int(x) for x in np.random.choice(np.arange(1, 10), size=n_runs, replace=False)]
    runs_bottom_to_top = list(zip(colors, heights))

    combo = _combo_output(runs_bottom_to_top)
    if combo.shape[0] > 30:
        return None

    original = _original_output(runs_bottom_to_top)
    d8_only = _d8_only_output(runs_bottom_to_top)
    d6_only = _d6_only_output(runs_bottom_to_top)
    if np.array_equal(combo, original):
        return None
    if np.array_equal(combo, d8_only):
        return None
    if np.array_equal(combo, d6_only):
        return None

    order_displacement, recolored_layers = _behavior_metrics(runs_bottom_to_top)
    if order_displacement < 2:
        return None
    if recolored_layers < 2:
        return None

    width = int(np.random.randint(6, 16))
    stripes_top_to_bottom = [
        np.full((int(height), width), int(color), dtype=int)
        for color, height in reversed(runs_bottom_to_top)
    ]
    grid = np.vstack(stripes_top_to_bottom)

    parsed = _parse_runs_bottom_to_top(grid)
    if parsed != runs_bottom_to_top:
        return None
    if not np.array_equal(main(grid), combo):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
