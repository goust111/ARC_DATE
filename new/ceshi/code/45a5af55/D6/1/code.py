# Variant of 45a5af55
# Dimension: D6 — 颜色到框架层的映射规则 (transform)
# Change: 每层框架直接使用对应色条的颜色（一对一映射）... → 相邻框架颜色互换
from __future__ import annotations

import numpy as np

# Design analysis for the D6 variant
# ==================================
# Reference-task baseline:
# - The requested source-task solver was read from
#   `/home/lixinyu/snap/new_code/solvers/arc2/a25697e4.py`.
# - The useful baseline idea is not its geometry, but its discipline:
#   first recover the structural placement, then apply a color-swap rule
#   controlled by a transformation property. Here we keep 45a5af55's geometry
#   exactly and only replace the per-layer color assignment.
#
# a) Concrete behavior and trace:
# - Input stays identical to the original task: full-width uniform stripes,
#   stacked vertically, read bottom -> top as layer 1, layer 2, ...
# - Geometry also stays identical: layer i keeps its own stripe height hi.
#   Layer 1 still defines the center square side length; each later layer still
#   wraps outward with thickness hi.
# - Only the color-owner of each layer changes. Adjacent layers swap colors in
#   pairs: layer 1 uses stripe-2 color, layer 2 uses stripe-1 color; layer 3
#   uses stripe-4 color, layer 4 uses stripe-3 color, etc. If the layer count
#   is odd, the final unpaired outermost layer keeps its own color.
# - Small trace:
#   runs bottom->top = [(1,2), (4,1), (6,3), (8,2)]
#   original output colors by layer = [1,4,6,8]
#   variant output colors by layer  = [4,1,8,6]
#   heights stay [2,1,3,2], so the output size is unchanged; only which color
#   appears on each nested square level changes.
#
# b) Complexity calibration:
# - If we allowed only 2 layers, the swap exists but is weakly supported.
# - The generator therefore prefers 4 or 6 layers so there are at least two
#   swapped pairs in most samples, making the pattern visible across examples.
# - Colors are sampled without replacement, which avoids accidental equality
#   between different swapped pairs and makes the reassignment easier to see.
#
# c) Hidden-assumption audit of the original main():
# - Parsing bottom->top run lengths remains valid because the input format is
#   unchanged.
# - The original `c0, h0 = runs[0]` assumed the center color comes from the
#   bottom stripe itself. That is no longer valid: the center still uses h0,
#   but its color now comes from the paired layer when one exists.
# - The original loop `for col, h in runs[1:]` assumed each frame's color and
#   thickness came from the same stripe. In the variant, thickness still comes
#   from that stripe, but color may come from its adjacent partner, so the
#   builder must consume a transformed color sequence instead of raw `runs`.
# - No overwrite hazards are introduced because the output is still built from
#   outside-in with solid square fills and central insertion exactly as before.
#
# d) Placement legality:
# - There is only one object family: the stripe stack. No collisions or free
#   placement are involved.
# - The only geometric validity condition is that the final output side length
#   h1 + 2 * sum(h2..) stays within ARC-friendly bounds. The generator enforces
#   a side length <= 30, matching the original task scale.
# - Adjacent stripe colors must differ; otherwise the parser would merge them
#   into one run. The generator guarantees pairwise-distinct consecutive rows by
#   sampling all colors without replacement.
#
# e) Visual observability:
# - Humans must notice that the thickness order is unchanged while colors jump
#   between neighboring layers. Distinct colors plus at least two swap pairs
#   make this legible from a few examples.
# - Keeping geometry fixed is useful here: when only the color assignment moves,
#   the learner can isolate the new rule instead of re-learning shape changes.
#
# f) Visual cleanliness:
# - The task remains visually clean because the output is still a single nested
#   square figure with one color per layer and no extra markers or overlaps.
#
# g) Color-change inferability:
# - The recoloring is structured by layer index only, not by arbitrary lookup.
# - Using a small number of unique colors and repeated pairing across layers
#   gives a stable clue: every neighboring pair exchanges color ownership.
#
# h) Generator anti-degeneracy check:
# - The generator rejects any sample whose swapped-color output would equal the
#   original one. With unique adjacent colors and >= 4 layers this should never
#   happen, but the explicit check prevents accidental regression.
#
# i) New assumptions introduced by the variant solver:
# - `_swap_adjacent_layer_colors` assumes layer order is meaningful, so
#   `generate_input()` must keep the input as a pure stripe stack.
# - Pairwise swapping is defined on parsed runs, not raw rows, so the generator
#   must avoid adjacent equal colors that would collapse the intended layer
#   count.


def _parse_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return (color, height) runs from bottom to top."""
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    runs: list[tuple[int, int]] = []
    r = int(H - 1)
    while r >= 0:
        col = int(g[r, 0])
        h = 1
        rr = int(r - 1)
        while rr >= 0 and bool((g[rr] == col).all()):
            h += 1
            rr -= 1
        runs.append((col, h))
        r = rr
    return runs


def _swap_adjacent_layer_colors(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Keep heights in place but swap colors of layers (1,2), (3,4), ..."""
    swapped = list(runs)
    for i in range(0, len(swapped) - 1, 2):
        c0, h0 = swapped[i]
        c1, h1 = swapped[i + 1]
        swapped[i] = (c1, h0)
        swapped[i + 1] = (c0, h1)
    return swapped


def _build_output(runs: list[tuple[int, int]]) -> np.ndarray:
    """Build nested square layers from bottom->top (color, height) pairs."""
    assert runs
    c0, h0 = runs[0]
    cur = np.full((h0, h0), c0, dtype=int)
    for col, h in runs[1:]:
        new_n = int(cur.shape[0] + 2 * h)
        nxt = np.full((new_n, new_n), int(col), dtype=int)
        nxt[h : h + cur.shape[0], h : h + cur.shape[1]] = cur
        cur = nxt
    return cur


def _original_output(runs: list[tuple[int, int]]) -> np.ndarray:
    return _build_output(runs)


def _variant_output(runs: list[tuple[int, int]]) -> np.ndarray:
    return _build_output(_swap_adjacent_layer_colors(runs))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from a stack of horizontal stripes, but swap the
    color ownership of adjacent layers.

    Input structure:
    - Full-width horizontal stripes of distinct non-bg colors, stacked vertically.
    - Each stripe has uniform color and variable height (1-4 rows).

    Variant rule:
    - Read stripes bottom to top.
    - Layer thicknesses are unchanged from the original task.
    - Adjacent layers swap colors in pairs: layer 1 uses stripe-2 color,
      layer 2 uses stripe-1 color, layer 3 uses stripe-4 color, etc.
    - If there is an unpaired outermost layer, it keeps its own color.
    """
    runs = _parse_runs(np.array(grid, dtype=int))
    assert runs
    return _variant_output(runs)


def _gen_attempt() -> np.ndarray | None:
    n_runs = int(np.random.choice([4, 6]))
    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    if heights[0] + 2 * sum(heights[1:]) > 30:
        return None

    palette = np.random.permutation(np.arange(1, 10))[:n_runs]
    cols = [int(x) for x in palette]
    runs = list(zip(cols, heights))

    original = _original_output(runs)
    variant = _variant_output(runs)
    if np.array_equal(original, variant):
        return None

    W = int(np.random.randint(6, 16))
    stripes = [
        np.full((int(h), W), int(col), dtype=int)
        for col, h in reversed(runs)
    ]
    inp = np.vstack(stripes)

    # Self-check: parser and solver must recover the intended swapped-color output.
    parsed_runs = _parse_runs(inp)
    if parsed_runs != runs:
        return None
    if not np.array_equal(main(inp), variant):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
