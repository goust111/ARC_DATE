# Variant of 45a5af55
# Dimension: D5 — 输入源对象类型 (src)
# Change: 全宽水平色条（每行颜色相同的连续行块）... → 全高竖直色条（每列颜色相同的连续列块）

from __future__ import annotations

import numpy as np

"""
Design analysis for the D5/src variant

2. Reference-task review
- `source_task` is absent in the prompt, so there is no extra solver to inspect.

3a. Concretization and behavior trace
- Original input: several full-width horizontal stripes, each stripe is a run of equal-color rows.
- Variant input: several full-height vertical stripes, each stripe is a run of equal-color columns.
- I keep the output rule unchanged: the first stripe in reading order becomes the center solid square, and every later stripe wraps a same-color square frame whose thickness equals that stripe's size along the reading axis.
- I concretize the under-specified reading direction to left-to-right. Supporting both left-to-right and right-to-left without any extra cue would make the task ambiguous: the same set of vertical bars could imply two different output color orders.

- Small trace:
  input widths/colors left->right = [(4, red), (1, blue), (2, yellow)]
  original horizontal logic would have used heights bottom->top = [4, 1, 2]
  variant logic instead reads column runs left->right = [4, 1, 2]
  center = 4x4 red square
  then add blue frame of thickness 1 -> 6x6
  then add yellow frame of thickness 2 -> 10x10
- The only changed stage is "how runs are parsed from the input". Frame construction itself stays identical.

3b. Complexity calibration
- The variant is not simplified to a mere transpose of the output. The output stays the same family of nested square frames; only the perceptual route for extracting run sizes changes from row grouping to column grouping.
- I keep widths in the same small range 1..4 and use 2..6 runs, matching the original task's complexity. This gives repeated evidence in each sample while keeping the nesting visually legible.

3c. Implicit-assumption review of the original `main()`
- `H, _W = g.shape`: original code only needed the number of rows because stripe thickness lived on the vertical axis. In the variant, stripe thickness is the run width, so the solver must depend on `W`.
- `col = g[r, 0]`: assumes each relevant object is identified from a row sample at column 0. This breaks for vertical stripes; we must sample `g[0, c]`.
- `(g[rr] == col).all()`: assumes a whole row is uniform. In the variant the analogous assumption is that a whole column is uniform, so this must become `(g[:, cc] == col).all()`.
- decrementing `r` from bottom to top hard-codes the old reading axis and direction. The variant must increment `c` from left to right.
- The frame-building loop does not rely on horizontal-vs-vertical stripes, so it remains valid unchanged.

3d. Placement legality
- Every generated stripe spans the full grid height and occupies a disjoint consecutive column interval.
- Total input width is bounded to keep the grid compact, and the derived output side length is also bounded (`<= 30`), so there is no overlap or out-of-bounds risk.
- Adjacent stripes always use different colors; otherwise two runs would merge and the solver would observe fewer stripes than intended.

3e. Visual observability
- With at least two stripes, the input rows are not uniform, so the picture no longer resembles the original horizontal-bar task.
- Multiple samples will show that only column widths matter. Height varies independently, which helps a human infer that the full-height vertical bars are the source objects.
- Fixing the reading direction to left->right makes the examples inferable from 2-4 pairs; no hidden tie-breaker is required.

3f. Visual cleanliness
- The input contains only one object family: clean, full-height bars touching edge-to-edge.
- The output remains the original nested-square image, which is already visually tidy and easy to parse.

3g. Color-change inferability
- Colors are copied directly from input stripes to output layers; there is no remapping.
- Only adjacent colors are forced distinct, just enough to preserve run boundaries.

8. Variant-behavior metric to validate non-degeneration
- Metric: count how many rows are monochrome versus how many columns are monochrome.
- For a true vertical-stripe sample with at least two runs:
  * every column is monochrome;
  * not every row is monochrome.
- The original horizontal-stripe generator has the opposite signature.
- After implementation, testing will sample generated inputs and confirm this signature holds broadly.

9. New implicit assumptions introduced by the new `main()`
- Each stripe is a full-height monochrome column run.
- Reading order is left-to-right.
- Adjacent runs have different colors so run boundaries are visible.
- `generate_input()` enforces all three by construction.
"""


def _build_nested_output(runs: list[tuple[int, int]]) -> np.ndarray:
    """Build the nested-square output from ordered (color, thickness) runs."""
    assert runs
    c0, w0 = runs[0]
    cur = np.full((int(w0), int(w0)), int(c0), dtype=int)
    for col, w in runs[1:]:
        w = int(w)
        new_n = int(cur.shape[0] + 2 * w)
        nxt = np.full((new_n, new_n), int(col), dtype=int)
        nxt[w : w + cur.shape[0], w : w + cur.shape[1]] = cur
        cur = nxt
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from a row-invariant stack of vertical color stripes.

    Input structure:
    - Full-height vertical stripes of distinct non-bg colors, placed left to right.
    - Each stripe has uniform color and variable width (1-4 columns).

    Rule:
    - Read stripes from left to right.
    - Leftmost stripe becomes the center solid square with side length = stripe width.
    - Each subsequent stripe wraps the current shape with a square frame of thickness = stripe width, in that stripe's color.
    """
    g = np.array(grid, dtype=int)
    _H, W = g.shape

    # Parse stripe runs from left to right.
    runs: list[tuple[int, int]] = []  # (color, width), left -> right
    c = 0
    while c < W:
        col = int(g[0, c])
        assert bool((g[:, c] == col).all())
        w = 1
        cc = c + 1
        while cc < W and bool((g[:, cc] == col).all()):
            w += 1
            cc += 1
        runs.append((int(col), int(w)))
        c = int(cc)

    return _build_nested_output(runs)


def _gen_attempt() -> np.ndarray | None:
    H = int(np.random.randint(6, 15))
    n_runs = int(np.random.randint(2, 7))
    widths = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    total_width = int(sum(widths))
    out_side = int(widths[0] + 2 * sum(widths[1:]))

    # Keep both input and output sizes in the same compact regime as the original.
    if total_width < 6 or total_width > 15 or out_side > 30:
        return None

    cols: list[int] = []
    prev = None
    for _ in range(n_runs):
        c = int(np.random.randint(1, 10))
        while prev is not None and c == prev:
            c = int(np.random.randint(1, 10))
        cols.append(c)
        prev = c

    stripes = [np.full((H, w), col, dtype=int) for col, w in zip(cols, widths)]
    inp = np.hstack(stripes)

    # Generator-side self-check: the solver must accept the constructed input.
    out = main(inp)
    if out.shape != (out_side, out_side):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
