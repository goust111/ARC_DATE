# Variant of 45a5af55
# Dimension: D1 — 嵌套构建方向 (transform)
# Change: 自底向上读取色条，最底层色条作为中心实心方块，逐层向外包裹框架... → 按色条高度排序决定嵌套层级

from __future__ import annotations

import numpy as np

# Design analysis for the D1 variant
# a) Concrete behavior:
#    The input remains a stack of full-width horizontal color stripes. The output
#    is still a nested square built from the stripe heights, but the controlling
#    order changes: first parse the stripes as objects, then sort those objects by
#    height from small to large. Example with bottom->top runs [(7,3), (2,1), (4,2)]:
#    original logic builds 7-center(3), then wraps 2(thickness 1), then wraps 4(thickness 2);
#    variant logic builds 2-center(1), then wraps 4(thickness 2), then wraps 7(thickness 3).
#    The same input objects therefore produce a different color nesting even though
#    the set of stripe heights is unchanged.
# b) Complexity calibration:
#    Equal heights would make "sort by height" visually ambiguous, so the generator
#    uses unique heights only. At least 3 stripes are generated so solvers must infer
#    an ordering rule rather than a trivial center/outer swap.
# c) Implicit assumptions in the original main():
#    - The bottom run is the center (`runs[0]`) -> no longer valid.
#    - The remaining runs are consumed in parsed order -> no longer valid.
#    - Adjacent parsed colors only need to differ -> insufficient now, because sorting
#      can create new adjacency. The generator therefore uses globally unique colors.
#    - The parser assumes each row is a uniform full-width stripe -> still valid and
#      preserved by the generator.
# d) Placement legality:
#    The generator only stacks non-overlapping full-width stripes, so parsing is exact.
#    Heights are chosen from 1..5 with no repeats; this keeps total output size within
#    ARC-scale bounds while preserving the nesting rule.
# e) Visual observability:
#    The generator rejects samples where the bottom stripe is already the shortest or
#    where the variant output accidentally matches the original bottom-to-top output.
#    That forces training examples to visibly demonstrate that spatial order is not the
#    governing rule.
# f) Visual cleanliness:
#    Inputs contain only one object family (uniform stripes) with no overlap, and outputs
#    contain clean square frames. Unique colors prevent two adjacent sorted layers from
#    visually merging into one thicker band.
# g) Color-rule inferability:
#    Colors are not remapped; they stay attached to their source stripes. The only changed
#    signal is which stripe becomes which nesting layer, and that signal is recoverable from
#    stripe height alone.
# 9) New solver assumptions and generator guarantees:
#    - `main()` now assumes the intended tasks have a unique shortest stripe for an
#      unambiguous center; `generate_input()` enforces unique heights.
#    - Stable sorting is used, so even tied heights would be processed deterministically,
#      but generated tasks avoid that branch.
#    - Every row must be a solid color stripe; the generator constructs exactly that.


def _parse_runs_bottom_to_top(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.array(grid, dtype=int)
    h, _w = g.shape

    runs: list[tuple[int, int]] = []
    r = int(h - 1)
    while r >= 0:
        col = int(g[r, 0])
        assert bool((g[r] == col).all())

        run_h = 1
        rr = int(r - 1)
        while rr >= 0 and bool((g[rr] == col).all()):
            run_h += 1
            rr -= 1

        runs.append((col, run_h))
        r = int(rr)

    assert runs
    return runs


def _build_nested_square(runs: list[tuple[int, int]]) -> np.ndarray:
    c0, h0 = runs[0]
    cur = np.full((int(h0), int(h0)), int(c0), dtype=int)

    for col, height in runs[1:]:
        t = int(height)
        new_n = int(cur.shape[0] + 2 * t)
        nxt = np.full((new_n, new_n), int(col), dtype=int)
        nxt[t : t + cur.shape[0], t : t + cur.shape[1]] = cur
        cur = nxt

    return cur


def _main_bottom_to_top(grid: np.ndarray) -> np.ndarray:
    return _build_nested_square(_parse_runs_bottom_to_top(grid))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from full-width horizontal stripes, but choose the
    nesting order by stripe height rather than by stripe position.

    Variant rule:
    - Parse the input into stripe runs.
    - Sort runs by height from shortest to tallest.
    - The shortest stripe becomes the center solid square.
    - Each taller stripe wraps the current square with a same-color frame whose
      thickness equals that stripe's height.

    Stable sorting keeps behavior deterministic if equal heights ever appear, but
    generated tasks avoid ties so the "height decides layer" pattern is explicit.
    """
    runs = _parse_runs_bottom_to_top(grid)
    ordered = sorted(runs, key=lambda item: item[1])
    return _build_nested_square(ordered)


def _gen_attempt() -> np.ndarray | None:
    width = int(np.random.randint(6, 15))
    n_runs = int(np.random.randint(3, 6))

    heights = [int(x) for x in np.random.choice(np.arange(1, 6), size=n_runs, replace=False)]
    bottom_to_top_heights = [int(x) for x in np.random.permutation(heights)]

    # Force a visible departure from the original task: the bottom stripe cannot
    # already be the unique shortest stripe.
    if bottom_to_top_heights[0] == min(bottom_to_top_heights):
        return None

    colors = [int(x) for x in np.random.choice(np.arange(1, 10), size=n_runs, replace=False)]
    bottom_to_top_runs = list(zip(colors, bottom_to_top_heights))
    stripes = [
        np.full((int(height), width), int(color), dtype=int)
        for color, height in reversed(bottom_to_top_runs)
    ]
    candidate = np.vstack(stripes)

    # Reject degenerate cases where the new rule and the original rule still agree.
    if np.array_equal(main(candidate), _main_bottom_to_top(candidate)):
        return None

    return candidate


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
