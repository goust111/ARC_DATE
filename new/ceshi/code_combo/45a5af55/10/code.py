# Combo variant of 45a5af55 (index 10)
# combo_spec: D5/0+D13/0
# Name: 分隔竖直色条嵌套

from __future__ import annotations

import numpy as np

"""
Design analysis for the fused D5/0 + D13/0 variant

3a. Concretization and behavior trace
- Original task input: full-width horizontal color stripes touching each other vertically.
  The solver reads stripe heights from bottom to top.
- Fused input: full-height vertical color stripes, read left to right, and adjacent stripes are
  separated by 1-2 full-height background columns of color 0.
- Concrete trace:
  * Suppose the semantic bars are [(2, width 3), (5, width 1), (7, width 2)] left->right.
  * The fused input may look like: 222 | 0 | 5 | 00 | 77, where every symbol denotes a full
    column repeated down the entire grid height.
  * Original solver would have misread rows and included every separator as data.
  * Fused solver instead skips the 0-columns, extracts semantic runs [(2, 3), (5, 1), (7, 2)],
    builds a 3x3 solid 2-square, wraps a thickness-1 frame of color 5, then wraps a thickness-2
    frame of color 7, producing an 9x9 nested square image.
- The two changed stages must cooperate: the solver has to detect that the meaningful objects are
  vertical runs rather than horizontal ones, and also that some full-height columns are only
  separators and contribute no layer.

3b. Complexity calibration
- The visual change is strong: rows are no longer monochrome, columns are. This avoids accidental
  similarity to the original task.
- Separator widths are restricted to 1 or 2 columns. This keeps the formatting signal visible in
  every sample without making the input noisy or introducing too many regimes.
- Semantic stripe widths remain in 1..4 and the number of stripes remains in 2..6, matching the
  original nesting depth and keeping the output legible.

3c. Implicit-assumption review of the original main()
- `H, _W = g.shape` originally focused on row count because thickness lived on the vertical axis.
  In the fused variant thickness is column width, so parsing must depend on `W`.
- `col = g[r, 0]` assumed every relevant object could be sampled from a row. This fails after the
  D5 rotation and after D13 separators, because row 0 now crosses multiple colors.
- `(g[rr] == col).all()` assumed a full row was monochrome. In the fused variant the analogous
  invariant is that each relevant full-height column is monochrome.
- The bottom-to-top loop over rows no longer matches the intended reading order. It must become a
  left-to-right scan over columns.
- The original code assumed every encountered run is semantic. D13 breaks that: color 0 columns
  are formatting only and must be skipped.
- The frame-construction loop itself remains valid once the ordered `(color, width)` runs are
  recovered.

3d. Placement legality
- Each semantic bar occupies a distinct consecutive block of full-height columns.
- Each separator is a distinct consecutive block of full-height background columns inserted only
  between neighboring semantic bars.
- The generator bounds input width and output side length to stay within ARC-like sizes and avoid
  overlap or out-of-bounds conditions.

3e. Visual observability
- Across 2-4 examples a human can see that the input is made of tall vertical bars, not horizontal
  stripes, because every column is monochrome while rows alternate colors.
- The repeated presence of 0-colored full-height gaps between bars makes it inferable that those
  columns are delimiters rather than semantic layers.
- Reading direction is fixed to left->right to avoid ambiguity. The same bar set would otherwise
  admit two incompatible nesting orders.

3f. Visual cleanliness
- Only two visual element types exist in the input: semantic full-height non-zero bars and clean
  full-height 0 separators. They never overlap.
- The output remains the original tidy nested-square family, so the transformation stays readable.

3g. Color-change inferability
- No color remapping is introduced. Semantic bar colors are copied directly into the corresponding
  output square layers.
- Color 0 is reserved for separators and never used as a semantic bar color, so humans can infer
  the distinction structurally rather than from hidden numeric tricks.

8. Metric to validate non-degeneration
- Metric 1: every generated input should contain at least one full-height background column, so the
  separator mechanism is active in all samples.
- Metric 2: every column should be monochrome, but not every row should be monochrome. This
  distinguishes the fused generator from the original horizontal-stripe generator.
- Metric 3: semantic bar count should be at least 2, ensuring actual nesting rather than a trivial
  one-layer case.

9. New implicit assumptions introduced by the fused main()
- Every semantic bar is a full-height monochrome non-zero column run.
- Every separator column is color 0 for its full height.
- Reading order is left to right.
- Adjacent semantic bars have different colors so boundaries are visible.
- `generate_input()` enforces all of these conditions and verifies the exact expected output.
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


def _parse_vertical_runs_with_separators(
    grid: np.ndarray, bg: int = 0
) -> list[tuple[int, int]]:
    """
    Parse semantic full-height vertical bars while ignoring full-height bg separators.

    Returns ordered `(color, width)` runs from left to right.
    """
    g = np.array(grid, dtype=int)
    _H, W = g.shape
    runs: list[tuple[int, int]] = []

    c = 0
    while c < W:
        col = int(g[0, c])
        assert bool((g[:, c] == col).all())
        if col == bg:
            c += 1
            continue

        w = 1
        cc = c + 1
        while cc < W and int(g[0, cc]) == col:
            assert bool((g[:, cc] == col).all())
            w += 1
            cc += 1
        runs.append((col, w))
        c = cc

    assert runs
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from separated full-height vertical color bars.

    Fused rule:
    - Non-zero full-height vertical bars are the semantic objects.
    - Full-height 0-columns between bars are separators and must be ignored.
    - Read semantic bars from left to right.
    - The leftmost semantic bar becomes the center solid square whose side equals the bar width.
    - Each later semantic bar wraps one outer square frame of matching color and thickness equal
      to that bar's width.
    """
    runs = _parse_vertical_runs_with_separators(np.array(grid, dtype=int), bg=0)
    return _build_nested_output(runs)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    H = int(np.random.randint(6, 15))
    n_runs = int(np.random.randint(2, 7))
    widths = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    gaps = [int(np.random.randint(1, 3)) for _ in range(n_runs - 1)]

    input_width = int(sum(widths) + sum(gaps))
    output_side = int(widths[0] + 2 * sum(widths[1:]))
    if input_width < 6 or input_width > 18 or output_side > 30:
        return None

    cols: list[int] = []
    prev = None
    for _ in range(n_runs):
        c = int(np.random.randint(1, 10))
        while c == bg or (prev is not None and c == prev):
            c = int(np.random.randint(1, 10))
        cols.append(c)
        prev = c

    pieces: list[np.ndarray] = []
    for idx, (col, w) in enumerate(zip(cols, widths)):
        pieces.append(np.full((H, w), col, dtype=int))
        if idx < n_runs - 1:
            pieces.append(np.full((H, gaps[idx]), bg, dtype=int))
    grid = np.hstack(pieces)

    runs = list(zip(cols, widths))
    expected = _build_nested_output(runs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
