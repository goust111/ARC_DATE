# Variant of 45a5af55
# Dimension: D8 — 色条读取与处理顺序 (notes)
# Change: 严格自底向上顺序读取色条，顺序即为嵌套层级（底层=中心，顶层=最外层）... → 按色条高度降序排列确定嵌套层级

from __future__ import annotations

import numpy as np

# Design analysis for this D8 variant
# - Source-task review: no source_task was provided for this variant, so there is no
#   extra reference solver to inspect beyond the original 45a5af55 solver.
# - a) Concrete behavior and trace:
#   The input is still a vertical stack of full-width horizontal stripes. Each stripe
#   is a solid rectangle whose height is the only geometric signal. The output is
#   still a set of nested square layers where layer thickness equals stripe height.
#   Original task trace on bottom->top runs [(7, 1), (2, 4), (5, 2)]:
#   center = 1x1 color 7, then wrap thickness 4 in color 2, then wrap thickness 2
#   in color 5. Variant trace on the same input:
#   sort by height descending -> [(2, 4), (5, 2), (7, 1)], so center = 4x4 color 2,
#   then color 5 wraps with thickness 2, then color 7 wraps with thickness 1.
#   The visible difference is that the tallest stripe, regardless of location, now
#   determines the innermost solid square.
# - b) Complexity calibration:
#   Equal heights would make the new ordering ambiguous to a human, so the generator
#   uses unique heights only. To keep the rule visually legible from a few examples,
#   it also rejects monotone bottom->top height orders; otherwise the variant could
#   collapse back to the original rule or look like a simple reversal.
# - c) Hidden-assumption audit against the original main():
#   1. Parsing contiguous full-width runs from bottom to top is still valid because
#      the input object type did not change.
#   2. The old `runs[0]` center assumption is no longer valid: the bottom stripe is
#      not necessarily the center. We must reorder runs after parsing.
#   3. The old wrapping loop relied on spatial order. In the variant it must iterate
#      in height-sorted order instead.
#   4. A deterministic tie-break is still useful for robustness, so the solver keeps
#      bottom->top order among equal heights. The generator avoids ties, so humans
#      are not asked to infer that secondary rule.
# - d) Placement legality:
#   The generator still places only full-width stripes stacked flush vertically, so
#   there is no overlap or out-of-bounds risk. Output legality is automatic because
#   nested squares are synthesized directly from stripe heights.
# - e) Visual observability:
#   Samples are forced to have 3-4 stripes with unique, non-monotone heights. This
#   makes it visible that the tallest stripe can appear in the middle or near the top
#   while still becoming the center in the output.
# - f) Visual cleanliness:
#   There is only one object family in the input and one in the output; no extra
#   markers or overlays are introduced.
# - g) Color inference:
#   Colors are not remapped. Each stripe keeps its own color when it becomes its
#   corresponding square layer, so the only inference burden is the new ordering rule.
# - h) Variant-specific generator metric:
#   We treat a sample as non-degenerate only if the variant output differs from the
#   original bottom->top output on the same input. The generator checks this directly.
# - i) New solver assumptions and generator guarantees:
#   The solver assumes every stripe row is uniform and stripes are contiguous runs.
#   The generator guarantees that by construction. The solver also assumes at least
#   one stripe exists; the generator always creates 3 or 4 of them.


def _parse_runs_bottom_to_top(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return (color, height) runs in bottom->top order."""
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


def _build_nested_square_layers(runs: list[tuple[int, int]]) -> np.ndarray:
    """Interpret `runs` as inner->outer layer order."""
    assert runs
    center_col, center_h = runs[0]
    cur = np.full((int(center_h), int(center_h)), int(center_col), dtype=int)
    for col, thickness in runs[1:]:
        thickness = int(thickness)
        new_n = int(cur.shape[0] + 2 * thickness)
        nxt = np.full((new_n, new_n), int(col), dtype=int)
        nxt[thickness : thickness + cur.shape[0], thickness : thickness + cur.shape[1]] = cur
        cur = nxt
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from stacked horizontal stripes.

    Variant rule:
    - Parse the stripes as before.
    - Ignore their vertical positions when deciding nesting order.
    - Sort stripes by height descending.
    - Tallest stripe becomes the center solid square.
    - Each shorter stripe wraps outward with frame thickness equal to its height.

    Determinism note:
    - If two stripes share a height, keep their original bottom->top order as a
      stable tie-break. The generator avoids equal heights, so visible examples
      emphasize the primary height-sorting rule.
    """
    runs = _parse_runs_bottom_to_top(np.array(grid, dtype=int))
    assert runs

    indexed_runs = list(enumerate(runs))
    ordered_runs = [
        run for _idx, run in sorted(indexed_runs, key=lambda item: (-item[1][1], item[0]))
    ]
    return _build_nested_square_layers(ordered_runs)


def _gen_attempt() -> np.ndarray | None:
    n_runs = int(np.random.randint(3, 5))  # 3 or 4 runs keeps unique heights easy.
    heights = list(np.random.choice(np.arange(1, 5), size=n_runs, replace=False))
    np.random.shuffle(heights)  # heights are in bottom->top order after shuffling.

    if heights == sorted(heights, reverse=True):
        return None
    if heights == sorted(heights):
        return None

    max_idx = int(heights.index(max(heights)))
    min_idx = int(heights.index(min(heights)))
    if max_idx in (0, n_runs - 1) and min_idx in (0, n_runs - 1):
        return None

    width = int(np.random.randint(6, 15))
    cols = list(np.random.choice(np.arange(1, 10), size=n_runs, replace=False))
    runs = list(zip(cols, heights))  # bottom -> top

    stripes = [np.full((h, width), col, dtype=int) for col, h in reversed(runs)]
    grid = np.vstack(stripes)

    variant_out = main(grid)
    original_out = _build_nested_square_layers(runs)
    if np.array_equal(variant_out, original_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
