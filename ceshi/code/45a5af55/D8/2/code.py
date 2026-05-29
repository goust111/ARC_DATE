# Variant of 45a5af55
# Dimension: D8 — 色条读取与处理顺序 (notes)
# Change: 严格自底向上顺序读取色条，顺序即为嵌套层级（底层=中心，顶层=最外层）... → 奇偶位置分组：奇数位色条从底向上构建内核，偶数位色条从顶向下构建外壳

from __future__ import annotations

import numpy as np


# Design analysis for the D8 variant
# a) Concrete behavior:
#    - Input stays as full-width horizontal color stripes.
#    - Number stripes by spatial position from bottom to top: 1, 2, 3, ...
#    - Odd-position stripes (1st, 3rd, 5th, ...) build the inner kernel in normal
#      bottom->top order. The 1st stripe is the center solid square.
#    - Even-position stripes (2nd, 4th, 6th, ...) are not interleaved with the odd
#      ones. Instead, after the odd kernel is fixed, the even stripes become the
#      outer shell. Their placement order is top->down, so the topmost even stripe
#      is the outermost frame, the next even stripe below it is the next frame in,
#      and so on.
#    - Small trace with bottom->top runs [(1,1), (2,2), (3,1), (4,2)]:
#      original center->outside order = 1,2,3,4
#      variant center kernel from odds = 1,3
#      variant outer shell from evens top->down = 4,2
#      so variant center->outside order becomes 1,3,2,4.
# b) Complexity calibration:
#    - If there are only 2 stripes, the variant collapses to the original rule.
#    - If there are only 3 stripes, the odd/even grouping is visible, but the
#      "even top->down" part has only one even stripe and is underconstrained.
#    - The generator therefore requires at least 4 stripes, usually 4-6, so both
#      phases are present and the even-shell ordering matters.
# c) Old main() assumptions that break under the variant:
#    - The bottom->top parser still works unchanged because input geometry is the same.
#    - The old "for col, h in runs[1:]" assumes every later stripe immediately wraps
#      the current square. That is the key invalid assumption: even stripes are now
#      delayed into phase 2 instead of wrapping at first sight.
#    - The old code also relied on a single monotone growth process. The variant needs
#      a two-stage build, so the solver must first isolate odd/even runs, then place
#      them with different orders.
# d) Placement legality:
#    - Input still has only stacked stripes, so there is no overlap issue in the input.
#    - Output placement is always legal because all frame thicknesses are positive and
#      the final side length is exactly kernel_size + 2 * sum(even_thicknesses).
# e) Visual observability:
#    - Unique colors across all stripes make layer order visually legible.
#    - Requiring at least two even-position stripes ensures the top->down shell rule
#      changes which color becomes outermost/next-outermost.
# f) Visual cleanliness:
#    - The task remains a single centered nested-square object, so the output stays
#      tidy and easy to inspect.
# g) Color inference:
#    - No color remapping is introduced. Colors are preserved; only grouping/order
#      changes. Unique stripe colors prevent ambiguous "same color, different layer"
#      cases that would hide the new ordering.
# h) Generator anti-degeneration policy:
#    - Enforce n_runs >= 4.
#    - Enforce all stripe colors distinct.
#    - Reject samples where the variant output matches the original output, because
#      that would fail to express the D8 change.
# i) New implicit assumptions in main():
#    - Every row is monochrome and the whole grid is a vertical stack of stripes.
#      generate_input() guarantees this directly.
#    - There is at least one odd stripe; true because n_runs >= 4.
#    - All stripe heights are positive integers; guaranteed by construction.
#    - The final square side must remain <= 30; generator filters by this bound.


def _parse_runs_bottom_to_top(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.array(grid, dtype=int)
    h_total, _w = g.shape
    runs: list[tuple[int, int]] = []
    r = h_total - 1
    while r >= 0:
        col = int(g[r, 0])
        height = 1
        rr = r - 1
        while rr >= 0 and bool((g[rr] == col).all()):
            height += 1
            rr -= 1
        runs.append((col, height))
        r = rr
    return runs


def _wrap_sequence_center_out(runs: list[tuple[int, int]]) -> np.ndarray:
    assert runs
    c0, h0 = runs[0]
    cur = np.full((h0, h0), c0, dtype=int)
    for col, height in runs[1:]:
        new_n = cur.shape[0] + 2 * height
        nxt = np.full((new_n, new_n), col, dtype=int)
        nxt[height : height + cur.shape[0], height : height + cur.shape[1]] = cur
        cur = nxt
    return cur


def _build_original_output(runs: list[tuple[int, int]]) -> np.ndarray:
    return _wrap_sequence_center_out(runs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from full-width color stripes using odd/even grouping.

    Variant rule:
    - Parse stripes bottom to top.
    - Odd-position stripes (1st, 3rd, 5th, ...) build the inner kernel in bottom->top order.
    - Even-position stripes (2nd, 4th, 6th, ...) build the shell in top->down order.
      Equivalently: after the odd kernel is complete, the topmost even stripe becomes
      the outermost frame, then the next topmost even stripe becomes the next frame in.
    """
    runs = _parse_runs_bottom_to_top(np.array(grid, dtype=int))
    assert runs

    odd_runs = runs[::2]
    even_runs = runs[1::2]

    kernel = _wrap_sequence_center_out(odd_runs)
    inner_size = int(kernel.shape[0])
    shell_thickness = sum(height for _col, height in even_runs)
    final_n = inner_size + 2 * shell_thickness

    if not even_runs:
        return kernel

    cur = np.empty((final_n, final_n), dtype=int)
    offset = 0
    cur_size = final_n

    # Phase 2 is explicit outer->inner placement of even-position stripes.
    for col, height in reversed(even_runs):
        cur[offset : offset + cur_size, offset : offset + cur_size] = col
        offset += height
        cur_size -= 2 * height

    cur[offset : offset + inner_size, offset : offset + inner_size] = kernel
    return cur


def _gen_attempt() -> np.ndarray | None:
    width = int(np.random.randint(7, 16))
    n_runs = int(np.random.randint(4, 7))
    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]

    input_h = sum(heights)
    odd_total = sum(heights[::2])
    even_total = sum(heights[1::2])
    final_side = heights[0] + 2 * (odd_total - heights[0]) + 2 * even_total
    if input_h > 30 or final_side > 30:
        return None

    cols = list(np.random.permutation(np.arange(1, 10))[:n_runs])
    runs = list(zip(cols, heights))

    # Build bottom stripe first in "runs"; input rows are stacked from top to bottom.
    stripes = [
        np.full((height, width), col, dtype=int)
        for col, height in reversed(runs)
    ]
    grid = np.vstack(stripes)

    # Reject degenerate samples that accidentally behave like the original task.
    parsed = _parse_runs_bottom_to_top(grid)
    if len(parsed) != n_runs:
        return None
    if np.array_equal(main(grid), _build_original_output(parsed)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
