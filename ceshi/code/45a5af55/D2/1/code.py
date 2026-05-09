# Variant of 45a5af55
# Dimension: D2 — 框架形状 (transform)
# Change: 构建正方形嵌套框架（每层为等厚方形环）... → 构建十字形嵌套框架

from __future__ import annotations

import numpy as np


# Design analysis for the D2 variant
# a) Concrete behavior and trace
#    Input stays identical to the seed task: full-width horizontal stripes,
#    each stripe having one color and a height in 1..4. We still read stripes
#    from bottom to top. The change is only in how one stripe expands the
#    partial output.
#    Seed behavior on runs [(2,2), (5,1)]:
#      - Start with a 2x2 solid block of color 2.
#      - Add color 5 as a square ring of thickness 1, giving a 4x4 square.
#    Variant behavior on the same runs:
#      - Start with the same 2x2 solid block of color 2.
#      - Add color 5 only as four orthogonal arm segments of thickness 1:
#        top, bottom, left, right. The 2x2 center remains color 2 and the
#        four corners stay background 0. Result:
#          0 5 5 0
#          5 2 2 5
#          5 2 2 5
#          0 5 5 0
#    So the changed step is the per-layer wrapping rule inside main(); input
#    construction can stay as stripe stacks because the same stripe heights now
#    control arm thickness instead of square-frame thickness.
#
# b) Complexity calibration
#    The task is not simplified to a single cross: multiple stripes still create
#    multiple nested layers. Heights vary in 1..4, so arm thickness visibly
#    changes between layers. Distinct adjacent stripe colors are preserved to
#    keep the nesting easy to read without introducing extra object types.
#
# c) Implicit-assumption audit of the seed solver
#    - Parsing bottom->top runs still assumes each row is monochrome and that
#      consecutive stripes differ in color. The generator preserves that.
#    - The seed line `nxt = np.full((new_n, new_n), col)` relied on the old
#      square-ring property: the new layer occupied every cell outside the
#      centered previous square. That is false in the variant, because corners
#      must remain empty background. We therefore start from zeros and paint
#      only the four arm rectangles.
#    - The centered insertion `nxt[h:h+n, h:h+n] = cur` remains valid: the
#      previous shape is still centered and every new layer grows by h on all
#      four sides.
#
# d) Placement legality
#    Output legality is trivial because there is only one centered composite
#    object. Each arm rectangle is written into a disjoint margin region:
#      top    rows [0:h),        cols [h:h+n)
#      bottom rows [h+n:h+n+h),  cols [h:h+n)
#      left   rows [h:h+n),      cols [0:h)
#      right  rows [h:h+n),      cols [h+n:h+n+h)
#    These regions do not overlap each other except at boundaries, do not
#    exceed the canvas, and do not overwrite the centered inner shape.
#
# e) Visual observability
#    The D2 change is directly visible because corners become background rather
#    than colored outer-frame cells. On the same input, the variant output is a
#    plus-like nested structure instead of a filled square. Even two-layer
#    samples already expose the rule; larger samples show repeated evidence that
#    every new stripe contributes four orthogonal arms, not a ring.
#
# f) Visual cleanliness
#    The canvas contains only one centered object plus background. No extra
#    markers, no overlap ambiguity, and no touching unrelated objects.
#
# g) Color inference
#    No new color mapping is introduced. Stripe colors transfer directly to the
#    corresponding center block / arm layer. Background is fixed to color 0, and
#    the generator excludes 0 from stripe colors so the empty corners stay
#    unambiguous.
#
# h) Generator fit
#    Because the input format is unchanged, generation only needs to keep output
#    size <= 30 and avoid adjacent equal colors. The solver itself is the
#    correctness oracle for generated inputs.
#
# i) New solver assumptions
#    The new solver assumes:
#    - at least one stripe exists;
#    - every row inside a stripe is constant across the width;
#    - adjacent stripes have different colors so run parsing is well defined.
#    generate_input() constructs exactly that form.


def _parse_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    """Parse full-width monochrome stripe runs from bottom to top."""
    g = np.array(grid, dtype=int)
    h, _w = g.shape

    runs: list[tuple[int, int]] = []
    r = int(h - 1)
    while r >= 0:
        col = int(g[r, 0])
        run_h = 1
        rr = int(r - 1)
        while rr >= 0 and bool((g[rr] == col).all()):
            run_h += 1
            rr -= 1
        runs.append((col, run_h))
        r = rr
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested cross-shaped frames from a stack of horizontal color stripes.

    Rule:
    - Read stripes bottom to top.
    - Bottom stripe becomes the center solid square with side length = stripe
      height.
    - Each later stripe adds four centered arm segments in that stripe's color.
      Each arm has thickness = stripe height and extends outward from the
      current shape; corners remain background 0.
    """
    runs = _parse_runs(np.array(grid, dtype=int))
    assert runs

    c0, h0 = runs[0]
    cur = np.full((h0, h0), c0, dtype=int)

    for col, h in runs[1:]:
        n = int(cur.shape[0])
        new_n = int(n + 2 * h)
        nxt = np.zeros((new_n, new_n), dtype=int)

        # Four orthogonal arm rectangles of thickness h around the current core.
        nxt[0:h, h : h + n] = col
        nxt[h + n : new_n, h : h + n] = col
        nxt[h : h + n, 0:h] = col
        nxt[h : h + n, h + n : new_n] = col

        nxt[h : h + n, h : h + n] = cur
        cur = nxt

    return cur


def _gen_attempt() -> np.ndarray | None:
    width = int(np.random.randint(6, 15))
    n_runs = int(np.random.randint(2, 7))
    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]

    # Output side length follows the same growth accounting as the seed task.
    if heights[0] + 2 * sum(heights[1:]) > 30:
        return None

    colors: list[int] = []
    prev = None
    for _ in range(n_runs):
        col = int(np.random.randint(1, 10))
        while prev is not None and col == prev:
            col = int(np.random.randint(1, 10))
        colors.append(col)
        prev = col

    stripes = [
        np.full((stripe_h, width), stripe_col, dtype=int)
        for stripe_col, stripe_h in reversed(list(zip(colors, heights)))
    ]
    inp = np.vstack(stripes)

    # Internal oracle check: generated inputs must satisfy the centered-cross
    # assumptions and visibly use the new behavior (background corners).
    out = main(inp)
    if out.shape[0] > 30:
        return None
    if len(colors) >= 2 and not all(int(out[r, c]) == 0 for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
