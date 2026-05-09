# Variant of 45a5af55
# Dimension: D3 — 色条高度到框架厚度的映射 (transform)
# Change: 框架厚度 = 色条高度（直接等值映射）... → 框架厚度 = 色条高度 × 2

from __future__ import annotations

import numpy as np


"""
Design analysis for this D3 variant

a) Concrete behavior and trace
   - Input stays the same as the original task: full-width horizontal stripes,
     each stripe is a solid color block with height 1..k, stacked vertically.
   - The solver still reads stripes from bottom to top.
   - The bottom stripe still becomes the center solid square with side length
     equal to its stripe height. This keeps the mutation isolated to D3:
     only the mapping from stripe height to frame thickness changes.
   - Every later stripe now contributes a square frame whose thickness is
     2 * stripe_height instead of stripe_height.
   - Small trace:
       bottom->top runs = [(2, 1), (5, 2), (7, 1)]
       original: center 1x1, then add thickness 2, then add thickness 1
                 sizes: 1 -> 5 -> 7
       variant:  center 1x1, then add thickness 4, then add thickness 2
                 sizes: 1 -> 9 -> 13
     The color nesting order is unchanged; only the amount each outer layer
     expands is different.

b) Complexity calibration
   - The rule should remain visually inferable from a few examples, so the
     generator keeps the clean stripe vocabulary and changes only the output
     growth law.
   - To make the doubled-thickness effect obvious, the generator requires at
     least one outer stripe of height >= 2. That yields a frame thickness of
     at least 4, which is visibly different from the original task.
   - Consecutive stripe colors remain distinct so run parsing is unambiguous.

c) Implicit-assumption review of main()
   - Converting to a numpy array still assumes a rectangular grid; unchanged.
   - The run parser assumes each stripe fills the full width and every row in a
     stripe is identical to the first row of that stripe; the generator keeps
     this invariant.
   - `c0, h0 = runs[0]` and the center square initialization still assume the
     bottom run exists; the generator always creates at least two runs.
   - The original `new_n = cur.shape[0] + 2 * h` relied on "frame thickness =
     h". Under the variant that line would silently produce the old behavior,
     so it must change to `cur.shape[0] + 4 * h`, because a frame of thickness
     `2h` adds `2h` on the left and `2h` on the right.
   - The insertion slice must shift by the actual new thickness (`2h`), not by
     `h`, or the inner square would be misplaced.

d) Placement legality
   - Input generation is trivial placement-wise: stripes are stacked with no
     overlap and no background gaps.
   - The solver's output size under the variant is:
       output_side = h0 + 4 * sum(outer_heights)
     The generator caps this at 30 so the produced ARC grid remains compact.
   - Input width remains arbitrary because the solver only needs full-width
     stripes, not a width/output-size relation.

e) Visual observability
   - Humans can directly count stripe heights in the input and compare them to
     frame thicknesses in the output.
   - Requiring at least one outer height >= 2 makes the multiplicative change
     visible: a 2-row stripe yields a 4-cell-thick frame instead of a 2-cell
     frame, which is hard to mistake for the original rule.

f) Visual cleanliness
   - The task remains a single clean stack in the input and a single centered
     nested-square construction in the output. No overlaps, markers, or extra
     decorations are introduced.

g) Color inference
   - No new color mapping is added. Colors are preserved layer-by-layer in the
     same bottom-to-top order as the original task, so the only new inference
     burden is the doubled thickness.
"""


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from a stack of horizontal color stripes.

    Variant rule:
    - Read stripes bottom to top.
    - Bottom stripe becomes the center solid square with side length equal to
      its stripe height.
    - Each later stripe wraps the current shape with a square frame whose
      thickness is 2 * stripe_height.
    """
    g = np.array(grid, dtype=int)
    h_grid, _w_grid = g.shape

    # Parse full-width stripe runs from bottom to top.
    runs: list[tuple[int, int]] = []
    r = h_grid - 1
    while r >= 0:
        color = int(g[r, 0])
        height = 1
        rr = r - 1
        while rr >= 0 and bool((g[rr] == color).all()):
            height += 1
            rr -= 1
        runs.append((color, height))
        r = rr

    assert runs

    center_color, center_height = runs[0]
    cur = np.full((center_height, center_height), center_color, dtype=int)

    for color, height in runs[1:]:
        thickness = 2 * int(height)
        new_n = int(cur.shape[0] + 2 * thickness)
        nxt = np.full((new_n, new_n), int(color), dtype=int)
        nxt[
            thickness : thickness + cur.shape[0],
            thickness : thickness + cur.shape[1],
        ] = cur
        cur = nxt

    return cur


def _gen_attempt() -> np.ndarray | None:
    width = int(np.random.randint(6, 15))
    n_runs = int(np.random.randint(3, 6))

    # Keep samples compact while still making the doubled-thickness effect
    # visible through at least one outer stripe of height >= 2.
    heights = [int(np.random.randint(1, 4)) for _ in range(n_runs)]
    if max(heights[1:]) < 2:
        return None

    output_side = int(heights[0] + 4 * sum(heights[1:]))
    if output_side > 30:
        return None

    colors: list[int] = []
    prev = None
    for _ in range(n_runs):
        color = int(np.random.randint(1, 10))
        while prev is not None and color == prev:
            color = int(np.random.randint(1, 10))
        colors.append(color)
        prev = color

    stripes = [
        np.full((height, width), color, dtype=int)
        for color, height in reversed(list(zip(colors, heights)))
    ]
    grid = np.vstack(stripes)

    # Generator-side guard against accidental solver/generator mismatch.
    out = main(grid)
    assert out.shape == (output_side, output_side)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
