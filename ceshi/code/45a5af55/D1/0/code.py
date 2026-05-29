# Variant of 45a5af55
# Dimension: D1 — 嵌套构建方向 (transform)
# Change: 自底向上读取色条，最底层色条作为中心实心方块，逐层向外包裹框架...
#         → 自顶向下读取色条构建嵌套框架

from __future__ import annotations

import numpy as np


# Design analysis for the D1 variant
# ==================================
# 1) Concrete behavior and trace
#    - Input remains a stack of full-width horizontal stripes, each stripe having
#      a uniform color and an integer height.
#    - Original solver logic:
#        top->bottom stripes = [(A, 2), (B, 1), (C, 3)]
#        original parse order = bottom->top = [(C, 3), (B, 1), (A, 2)]
#        output = center 3x3 C, then wrap 1-cell B frame, then wrap 2-cell A frame.
#    - Variant logic:
#        same input stripes = [(A, 2), (B, 1), (C, 3)]
#        new parse order = top->bottom = [(A, 2), (B, 1), (C, 3)]
#        output = center 2x2 A, then wrap 1-cell B frame, then wrap 3-cell C frame.
#    - The whole input format stays fixed; only the stripe reading direction flips.
#      This is the intended D1 change and it fully changes the color nesting.
#
# 2) Complexity calibration
#    - The task should still be inferable from a few ARC examples, so the objects
#      remain simple full-width stripes and square nestings.
#    - To keep the reversal visually obvious, the generator rejects degenerate
#      inputs where the new top-down output equals the old bottom-up output.
#      This avoids palindromic color/height sequences from masking the variant.
#
# 3) Implicit assumptions review for main()
#    - Each row belongs to exactly one full-width stripe. The parser assumes every
#      row is monochrome. The generator guarantees this via np.full stripes.
#    - Stripes are contiguous vertical runs; no background gaps appear between
#      them. The generator stacks stripes with np.vstack, so this holds.
#    - Output is always square because every wrapping step adds the same thickness
#      on all four sides. This remains valid under reversed read order.
#    - The old code's only variant-sensitive assumption was that the first parsed
#      run should come from the bottom. That is no longer true, so parsing must be
#      performed top-down instead of bottom-up.
#
# 4) Placement legality
#    - The generator only creates legal stripe stacks: no overlap, no cropping,
#      and every stripe spans the full width.
#    - Output size for a run list [(c0, h0), (c1, h1), ...] is
#      h0 + 2 * sum(h1:), so the generator bounds this by 30 to stay ARC-sized.
#
# 5) Visual observability
#    - Reversing the read order changes which color becomes the center and which
#      becomes the outer frame. Humans can detect this directly if examples are
#      not degenerate, hence the explicit generator-side rejection check.
#
# 6) Visual cleanliness
#    - No extra markers or clutter are added. The output stays a clean concentric
#      square pattern, matching the source task's visual language.
#
# 7) Color change inferability
#    - There is no arbitrary recoloring. Output colors are copied directly from
#      the input stripes; only their nesting roles change with position/order.


def _parse_runs_top_down(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return (color, height) stripe runs in top-to-bottom order."""
    g = np.array(grid, dtype=int)
    H, _W = g.shape

    runs: list[tuple[int, int]] = []
    r = 0
    while r < H:
        row = g[r]
        col = int(row[0])
        if not bool((row == col).all()):
            raise ValueError("Each input row must be monochrome across the full width")

        h = 1
        rr = r + 1
        while rr < H and bool((g[rr] == col).all()):
            h += 1
            rr += 1

        runs.append((col, h))
        r = rr

    if not runs:
        raise ValueError("Input must contain at least one stripe")
    return runs


def _build_nested_from_runs(runs: list[tuple[int, int]]) -> np.ndarray:
    """Build concentric squares where the first run becomes the center."""
    c0, h0 = runs[0]
    cur = np.full((h0, h0), c0, dtype=int)

    for col, h in runs[1:]:
        new_n = int(cur.shape[0] + 2 * h)
        nxt = np.full((new_n, new_n), col, dtype=int)
        nxt[h : h + cur.shape[0], h : h + cur.shape[1]] = cur
        cur = nxt

    return cur


def _solve_bottom_up_for_rejection(grid: np.ndarray) -> np.ndarray:
    """
    Old task behavior, kept only so the generator can reject samples where
    top-down and bottom-up happen to produce the same output.
    """
    runs = _parse_runs_top_down(grid)
    return _build_nested_from_runs(list(reversed(runs)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from a stack of horizontal color stripes.

    Variant rule:
    - Read stripes top to bottom.
    - Top stripe becomes the center solid square with side length = stripe height.
    - Each subsequent lower stripe wraps the current shape with a square frame of
      thickness = that stripe's height.
    """
    runs = _parse_runs_top_down(np.array(grid, dtype=int))
    return _build_nested_from_runs(runs)


def _gen_attempt() -> np.ndarray | None:
    W = int(np.random.randint(6, 15))
    n_runs = int(np.random.randint(3, 7))
    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    if heights[0] + 2 * sum(heights[1:]) > 30:
        return None

    cols: list[int] = []
    prev = None
    for _ in range(n_runs):
        c = int(np.random.randint(1, 10))
        while prev is not None and c == prev:
            c = int(np.random.randint(1, 10))
        cols.append(c)
        prev = c

    stripes = [np.full((h, W), col, dtype=int) for col, h in zip(cols, heights)]
    grid = np.vstack(stripes)

    # Enforce visible D1 behavior: the same input should usually change if one
    # applies the original bottom-up construction instead of the variant.
    if np.array_equal(main(grid), _solve_bottom_up_for_rejection(grid)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is None:
            continue

        # Sanity-check the generated instance against the active solver.
        out = main(result)
        if out.ndim == 2 and out.shape[0] == out.shape[1]:
            return result

    raise RuntimeError("generate_input failed after 300 attempts")
