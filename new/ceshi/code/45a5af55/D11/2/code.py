# Variant of 45a5af55
# Dimension: D11 — 源对象基数与嵌套关系 (meta)
# Change: N个色条 → 1个嵌套框架结构（many:1 基数关系）... → 色条两两配对：每对生成一个小框架，所有小框架再嵌套组合

from __future__ import annotations

import numpy as np

"""
Design analysis for the D11 variant

a) Concrete behavior and trace
   - Input stays close to the source task: a stack of full-width horizontal color stripes.
   - The important change is grouping. Read stripes bottom -> top as runs r0, r1, r2, r3, ...
   - Adjacent runs are paired: (r0, r1), (r2, r3), ...
   - Inside each pair, the upper stripe becomes the inner layer and the lower stripe becomes the
     outer layer. Pair order is then preserved globally: the first pair is the innermost mini-frame,
     the next pair wraps it, etc.
   - Example with bottom-up runs:
       r0 = (A,1), r1 = (B,2), r2 = (C,1), r3 = (D,3)
     Original task:
       center A(1) -> wrap B(2) -> wrap C(1) -> wrap D(3)
       final inner->outer layers: A, B, C, D
     Variant:
       pair1 makes mini-frame: center B(2), outer A(1)
       pair2 wraps pair1 with inner D(3), outer C(1)
       final inner->outer layers: B, A, D, C
     The difference is not a cosmetic rewording; every pair reverses local nesting.

b) Complexity calibration
   - If the pairing only changed bookkeeping while preserving stripe order, the task would collapse
     back to the original solver. Reversing the two members within each pair makes the grouping
     visible in every nontrivial example.
   - The generator uses 2-3 pairs, so the solver must infer both "pair locally" and "nest globally".
   - Colors are chosen distinct within a sample to make pairwise reversal easy to see.

c) Implicit-assumption review of the original main()
   - Original parser assumes every row of a run is identical across the full width. That still holds.
   - Original `c0, h0 = runs[0]` assumes the lowest stripe is the center block. This is no longer
     valid; the center now comes from runs[1], the upper member of the first pair.
   - Original loop `for col, h in runs[1:]` assumes every later stripe is a single wrap layer in the
     same order it appears bottom->top. In the variant, layers must be consumed in steps of 2 and
     applied as (inner-from-upper, outer-from-lower), so that loop would silently produce the wrong
     output if left unchanged.
   - Because the new rule depends on pairs, an odd number of runs would leave one stripe unmatched.
     To keep the puzzle unambiguous, the generator always emits an even run count and the solver
     asserts that precondition.

d) Placement legality
   - Input generator only stacks full-width stripes vertically, so there is no overlap or out-of-
     bounds placement issue in the input.
   - Output size is controlled analytically before accepting a sample:
       side = h1 + 2*h0 + 2*sum(h2..)
     This guarantees the produced nested square fits within the ARC 30x30 limit.

e) Visual observability
   - With 2-3 pairs, the center color is never the bottom stripe color. This alone distinguishes the
     variant from the source task.
   - Every pair contributes a visible two-layer chunk whose local color order is opposite the input's
     bottom->top order within that pair.
   - Distinct colors and at least two pairs make the pair-then-nest rule recoverable from 2-4 demos.

f) Visual neatness
   - The task still produces one clean centered square composition with no clutter.
   - Only concentric square layers are used; there are no extra markers or overlapping symbols.

g) Color inferability
   - No arbitrary color remapping is introduced. Colors are preserved exactly; only their structural
     positions change according to pair membership.

h) Generator anti-degeneration check
   - A sample is rejected if the variant output equals the source-task output on the same input. This
     ensures the produced training data actually exercises the D11 change.

i) New solver assumptions and generator support
   - `main()` assumes: full-width uniform stripes, even run count, at least one pair.
   - `generate_input()` guarantees all three and keeps total side length <= 30.
"""


def _parse_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    """Parse full-width uniform stripe runs from bottom to top."""
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


def _wrap(cur: np.ndarray, col: int, thickness: int) -> np.ndarray:
    """Center `cur` inside a square layer of color `col` and thickness `thickness`."""
    new_n = int(cur.shape[0] + 2 * thickness)
    nxt = np.full((new_n, new_n), int(col), dtype=int)
    nxt[thickness : thickness + cur.shape[0], thickness : thickness + cur.shape[1]] = cur
    return nxt


def _build_original_output(runs: list[tuple[int, int]]) -> np.ndarray:
    """Source-task construction, used only to reject degenerate generator samples."""
    c0, h0 = runs[0]
    cur = np.full((h0, h0), c0, dtype=int)
    for col, thickness in runs[1:]:
        cur = _wrap(cur, col, thickness)
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square layers from paired horizontal stripes.

    Variant rule:
    - Parse stripes bottom -> top.
    - Group adjacent stripes into pairs: (r0,r1), (r2,r3), ...
    - For each pair, the upper stripe is the inner layer and the lower stripe is the outer layer.
    - The first pair forms the innermost mini-frame; later pairs wrap the current result.
    """
    runs = _parse_runs(np.array(grid, dtype=int))
    assert runs and len(runs) % 2 == 0

    # First pair: the upper stripe becomes the center solid square, then the lower stripe wraps it.
    inner_col, inner_h = runs[1]
    cur = np.full((inner_h, inner_h), inner_col, dtype=int)
    outer_col, outer_h = runs[0]
    cur = _wrap(cur, outer_col, outer_h)

    # Each later pair contributes a two-layer module: inner layer from the upper stripe, then
    # outer layer from the lower stripe.
    for i in range(2, len(runs), 2):
        outer_col, outer_h = runs[i]
        inner_col, inner_h = runs[i + 1]
        cur = _wrap(cur, inner_col, inner_h)
        cur = _wrap(cur, outer_col, outer_h)

    return cur


def _gen_attempt() -> np.ndarray | None:
    width = int(np.random.randint(7, 16))
    n_pairs = int(np.random.randint(2, 4))
    n_runs = 2 * n_pairs
    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]

    # Variant output side length:
    # center = heights[1], all other runs contribute frame thickness on both sides.
    out_side = int(heights[1] + 2 * (sum(heights) - heights[1]))
    if out_side > 30:
        return None

    colors = [int(c) for c in np.random.permutation(np.arange(1, 10))[:n_runs]]
    stripes = [
        np.full((run_h, width), col, dtype=int)
        for col, run_h in reversed(list(zip(colors, heights)))
    ]
    grid = np.vstack(stripes)

    runs = _parse_runs(grid)
    if np.array_equal(main(grid), _build_original_output(runs)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            # Final self-check: parser assumptions hold and solver succeeds.
            _ = main(result)
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
