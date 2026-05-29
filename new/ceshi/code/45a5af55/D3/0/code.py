# Variant of 45a5af55
# Dimension: D3 — 色条高度到框架厚度的映射 (transform)
# Change: 框架厚度 = 色条高度（直接等值映射）... → 框架厚度 = 色条数量 - 色条序号（反向递减）

from __future__ import annotations

import numpy as np


# Design analysis for the D3 variant
# a) Concrete behavior and trace
#    Input stays as full-width horizontal stripes, read from bottom to top.
#    The original task used each stripe height directly as geometry:
#    bottom height -> center-square side, later heights -> frame thicknesses.
#    This variant removes that equality. If there are n stripes total, then the
#    geometric thickness sequence becomes [n, n-1, ..., 1] from inner to outer.
#    Example with bottom->top runs:
#    colors=[1, 8, 6, 2], input_heights=[2, 4, 1, 3], n=4.
#    Original geometry: [2, 4, 1, 3] -> center 2x2, then +4, +1, +3.
#    Variant geometry:  [4, 3, 2, 1] -> center 4x4, then +3, +2, +1.
#    The parsing stage is unchanged; only the output-construction stage changes.
#
# b) Complexity calibration
#    To make the new rule visible, generator samples 3-5 stripes and forces
#    several input heights to disagree with their layer-based thicknesses.
#    It also prefers repeated heights, so humans can see that equal stripe
#    heights can still produce different frame thicknesses depending on depth.
#
# c) Implicit assumptions in the original main()
#    - Parsing consecutive equal-color full rows remains valid and unchanged.
#    - `c0, h0 = runs[0]` originally assumed the first height is the center size.
#      That assumption is no longer valid; the center size is now `len(runs)`.
#    - `for col, h in runs[1:]` originally assumed every outer thickness equals
#      the corresponding stripe height. That also breaks here; thickness must be
#      recomputed from the stripe index: `len(runs) - idx`.
#    - The original code's unconditional fill of `nxt` remains safe because each
#      new frame still fully surrounds the previous square without overlap bugs.
#
# d) Placement legality
#    Generator still produces a single vertical stack of full-width stripes, so
#    there is no risk of overlap or out-of-bounds placement in the input grid.
#    Output size is deterministic for n stripes: n^2 x n^2, because
#    n + 2 * sum(n-1 ... 1) = n^2. Restricting n to 3-5 keeps outputs <= 25.
#
# e) Visual observability
#    Across samples, the key visible clue is that outer frames get thinner while
#    inner ones get thicker regardless of stripe heights. Repeated stripe heights
#    and multiple mismatches prevent accidental "height == thickness" readings.
#
# f) Visual cleanliness
#    The task still contains one simple input object class (stacked stripes) and
#    one clean output object class (nested squares). No extra markers are added.
#
# g) Color inferability
#    Colors are preserved by layer and only consecutive stripes are forced to be
#    different. Non-consecutive repeats are allowed, matching the original task's
#    simplicity while keeping the color rule easy to infer.
#
# h) No source-task adaptation was needed because the prompt specifies no source.
#
# i) New main() assumptions audited against generate_input()
#    - Every row in a stripe is uniform: guaranteed by generator's `np.full`.
#    - Consecutive stripes have different colors: enforced during sampling.
#    - At least one run exists: generator always creates 3-5 runs.
#    - Variant behavior is non-degenerate: generator rejects samples whose
#      outputs under the original and variant rules would be identical.


def _parse_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    """Parse full-width color stripes from bottom to top as (color, height)."""
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


def _render_nested(colors_bottom_to_top: list[int], layer_sizes_bottom_to_top: list[int]) -> np.ndarray:
    """Render nested squares where layer_sizes[0] is the center side length."""
    if not colors_bottom_to_top or not layer_sizes_bottom_to_top:
        raise ValueError("At least one layer is required.")

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


def _variant_layer_sizes(n_runs: int) -> list[int]:
    """Bottom->top geometric sizes: innermost thickest, outermost thinnest."""
    return [int(n_runs - idx) for idx in range(n_runs)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from a stack of horizontal color stripes.

    Variant rule:
    - Parse full-width color stripes from bottom to top.
    - If there are n stripes, the geometric sizes are [n, n-1, ..., 1]
      from inner to outer, regardless of the stripes' input heights.
    - The bottom stripe becomes an n x n solid center square.
    - Each later stripe wraps the current square with a frame whose thickness
      depends only on its layer index, not on the stripe height in the input.
    """
    runs = _parse_runs(np.array(grid, dtype=int))
    if not runs:
        raise ValueError("Input must contain at least one stripe.")

    colors = [col for col, _height in runs]
    layer_sizes = _variant_layer_sizes(len(runs))
    return _render_nested(colors, layer_sizes)


def _gen_attempt() -> np.ndarray | None:
    n_runs = int(np.random.randint(3, 6))
    variant_sizes = _variant_layer_sizes(n_runs)

    # Heights are now distractors. Force repeated heights and several mismatches
    # so the sample does not collapse back into the original rule.
    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    if len(set(heights)) == n_runs:
        return None
    if sum(int(h != v) for h, v in zip(heights, variant_sizes)) < 2:
        return None

    width = int(np.random.randint(6, 16))

    colors: list[int] = []
    prev = None
    for _ in range(n_runs):
        col = int(np.random.randint(1, 10))
        while prev is not None and col == prev:
            col = int(np.random.randint(1, 10))
        colors.append(col)
        prev = col

    stripes = [
        np.full((height, width), col, dtype=int)
        for col, height in reversed(list(zip(colors, heights)))
    ]
    grid = np.vstack(stripes)

    runs = _parse_runs(grid)
    variant_out = _render_nested([col for col, _ in runs], variant_sizes)
    legacy_out = _render_nested([col for col, _ in runs], [height for _col, height in runs])

    if np.array_equal(variant_out, legacy_out):
        return None
    if not np.array_equal(main(grid), variant_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
