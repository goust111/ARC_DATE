# Variant of 45a5af55
# Dimension: D1 — 嵌套构建方向 (transform)
# Change: 自底向上读取色条，最底层色条作为中心实心方块，逐层向外包裹框架... → 自左向右读取竖直色条构建嵌套框架

from __future__ import annotations

import numpy as np

# Design analysis for this variant
# - Reference-task check: source_task is None, so there is no extra solver to inspect.
# - a) Concrete behavior:
#   The original input is a stack of full-width horizontal stripes. Here the input becomes a
#   striping of full-height vertical bars. Each bar is uniform top-to-bottom and has width 1-4.
#   Reading order changes from bottom->top to left->right. The leftmost bar becomes a solid
#   center square whose side length equals that bar's width. Each later bar contributes an
#   outer square frame of the same color and thickness equal to that bar's width.
#   Small trace:
#   widths/colors left->right = [(2, red), (1, blue), (3, green)].
#   Original solver would inspect rows and only see repeated mixed-color rows, so it would not
#   recover the three stripe widths. The variant solver instead groups columns into runs of
#   widths 2, 1, 3, then builds: 2x2 red square -> 4x4 with blue frame -> 10x10 with green frame.
# - b) Complexity calibration:
#   We keep the same run-width range (1-4), same distinct-adjacent color rule, and 2-6 bars.
#   This preserves the original reasoning load while making the orientation switch visible.
#   Using multiple bars avoids degenerate one-color cases; the generator biases to at least 2.
# - c) Implicit-assumption audit of the original main():
#   `col = g[r, 0]` assumed a row can be identified by its first cell.
#   `while ... (g[rr] == col).all()` assumed every stripe is a full-width uniform row block.
#   Both fail once stripes are vertical: every row becomes multi-color, while every column is
#   uniform. The frame-construction code itself is orientation-agnostic because it only consumes
#   `(color, thickness)` runs, so only the parsing stage must change.
# - d) Placement legality:
#   The generator constructs the input by horizontally concatenating full-height constant-color
#   rectangles, so bars never overlap, never go out of bounds, and every column remains solver-valid.
# - e) Visual observability:
#   Across 2-4 examples, the new rule is visible because the input now shows vertical segmentation
#   while the output remains concentric square nesting. The solver must infer that bar width, not
#   bar height, controls frame thickness.
# - f) Visual cleanliness:
#   There is no background clutter or overlapping markup. Each grid cell belongs to exactly one bar.
# - g) Color inferability:
#   Colors are copied directly from input bars to output layers. No hidden remapping is introduced.
# - h) Generator/sovler consistency:
#   The generator derives an expected output from the sampled runs and only accepts an input when
#   `main(input)` matches that expectation.
# - i) New assumptions introduced by this main():
#   Every input column must be uniform in color from top to bottom, and adjacent bars are encoded
#   by contiguous same-color columns. The generator guarantees both properties.


def _build_nested_from_runs(runs: list[tuple[int, int]]) -> np.ndarray:
    assert runs

    c0, w0 = runs[0]
    cur = np.full((int(w0), int(w0)), int(c0), dtype=int)

    for col, width in runs[1:]:
        width = int(width)
        new_n = int(cur.shape[0] + 2 * width)
        nxt = np.full((new_n, new_n), int(col), dtype=int)
        nxt[width : width + cur.shape[0], width : width + cur.shape[1]] = cur
        cur = nxt

    return cur


def _parse_vertical_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.array(grid, dtype=int)
    _h, w = g.shape

    runs: list[tuple[int, int]] = []
    c = 0
    while c < w:
        col = int(g[0, c])
        assert bool((g[:, c] == col).all())

        width = 1
        cc = c + 1
        while cc < w and bool((g[:, cc] == col).all()):
            width += 1
            cc += 1

        runs.append((col, width))
        c = cc

    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from a stack of vertical color bars.

    Input structure:
    - Full-height vertical bars of distinct adjacent non-bg colors, placed left to right.
    - Each bar has uniform color and variable width (1-4 columns).

    Rule:
    - Read bars left to right.
    - The leftmost bar becomes the center solid square with side length = bar width.
    - Each subsequent bar wraps the current shape with a square frame whose thickness equals that bar width.
    """
    runs = _parse_vertical_runs(np.array(grid, dtype=int))
    return _build_nested_from_runs(runs)


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(6, 16))
    n_runs = int(np.random.randint(2, 7))
    widths = [int(np.random.randint(1, 5)) for _ in range(n_runs)]

    output_side = int(widths[0] + 2 * sum(widths[1:]))
    if output_side > 30:
        return None

    cols: list[int] = []
    prev = None
    for _ in range(n_runs):
        c = int(np.random.randint(1, 10))
        while prev is not None and c == prev:
            c = int(np.random.randint(1, 10))
        cols.append(c)
        prev = c

    bars = [np.full((h, width), col, dtype=int) for col, width in zip(cols, widths)]
    inp = np.hstack(bars)
    expected = _build_nested_from_runs(list(zip(cols, widths)))
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
