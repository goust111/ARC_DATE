# Variant of 45a5af55
# Dimension: D1 — 嵌套构建方向 (transform)
# Change: 自底向上读取色条，最底层色条作为中心实心方块，逐层向外包裹框架... → 交替内外层构建：奇数层从底部取、偶数层从顶部取

from __future__ import annotations

import numpy as np

# Design analysis
# a) Concrete behavior and trace:
#    Input stays a stack of full-width horizontal stripes. Output stays a nested
#    square made of one center solid block plus surrounding square frames. The
#    only changed dimension is which stripe supplies each successive layer.
#    Example with top->bottom stripes [(1,2), (2,1), (3,4), (4,2)]:
#    - Original center-out order was bottom->top:      [(4,2), (3,4), (2,1), (1,2)]
#    - Variant center-out order is bottom, top, ...:   [(4,2), (1,2), (3,4), (2,1)]
#    The second layer now comes from the top stripe instead of the stripe just
#    above the bottom. So the visible ring colors no longer follow the vertical
#    order in the input; they alternate between the two ends of the stripe stack.
# b) Complexity calibration:
#    If there are only 2 stripes, bottom-then-top is identical to the original
#    rule. Therefore the generator must use at least 3 stripes. To make the
#    alternation visible, it also uses globally distinct colors, so swapping two
#    layers cannot look accidentally unchanged.
# c) Implicit-assumption audit of the original main():
#    - Parsing bottom->top rows assumed the construction order equals the scan
#      order. That is no longer true, so parsing and build order must be split.
#    - `c0, h0 = runs[0]` assumed the first parsed run is always the center.
#      In the variant, only the first selected run is the center; later runs are
#      chosen alternately from bottom and top.
#    - `for col, h in runs[1:]` assumed the remaining runs are consumed in fixed
#      bottom->top order. This is the exact line that would silently produce the
#      old task if left unchanged.
# d) Placement legality:
#    The generator still makes a simple vertical stripe stack spanning the full
#    width, so stripes never overlap except by intended row adjacency, and there
#    is no spatial placement ambiguity. Output size remains
#    center_height + 2 * sum(other_heights), so bounding it by 30 is sufficient.
# e) Visual observability:
#    Human solvers only see a few examples, so the generator must avoid samples
#    where the new order collapses to the old one. It enforces n_runs >= 3 and
#    rejects any sampled input whose variant output equals the original output.
# f) Visual cleanliness:
#    No extra markers or background clutter are added. The task remains a clean
#    "stripe stack -> nested square" mapping, with only the layer sourcing order
#    changed. This keeps the image legible while still making the rule harder.
# g) Color inference:
#    Colors are not remapped. A stripe's color is copied directly onto its
#    assigned square layer. Using unique colors makes the alternating source
#    pattern easier to infer from few examples.
# 9) New main() assumptions and generator guarantees:
#    - Every stripe row is uniform across the full width.
#    - The input is a complete vertical partition into stripe runs.
#    - There are at least 3 runs, so both bottom and top selection paths matter.
#    - Distinct colors across runs ensure that if build order changes, the
#      rendered output changes too. generate_input() constructs inputs to satisfy
#      all of these assumptions and validates them with main().


def _parse_stripes_top_to_bottom(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    runs: list[tuple[int, int]] = []
    r = 0
    while r < H:
        col = int(g[r, 0])
        assert bool((g[r] == col).all())
        h = 1
        rr = r + 1
        while rr < H and bool((g[rr] == col).all()):
            h += 1
            rr += 1
        runs.append((col, h))
        r = rr
    assert runs
    return runs


def _alternate_center_out_order(runs_top_to_bottom: list[tuple[int, int]]) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    top = 0
    bottom = len(runs_top_to_bottom) - 1
    take_bottom = True
    while top <= bottom:
        if take_bottom:
            order.append(runs_top_to_bottom[bottom])
            bottom -= 1
        else:
            order.append(runs_top_to_bottom[top])
            top += 1
        take_bottom = not take_bottom
    return order


def _original_center_out_order(runs_top_to_bottom: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return list(reversed(runs_top_to_bottom))


def _build_nested_square(runs_center_out: list[tuple[int, int]]) -> np.ndarray:
    c0, h0 = runs_center_out[0]
    cur = np.full((h0, h0), c0, dtype=int)
    for col, h in runs_center_out[1:]:
        new_n = int(cur.shape[0] + 2 * h)
        nxt = np.full((new_n, new_n), int(col), dtype=int)
        nxt[h : h + cur.shape[0], h : h + cur.shape[1]] = cur
        cur = nxt
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from a stack of horizontal color stripes.

    Variant rule:
    - Parse the stripes in their geometric top->bottom order.
    - Select center-out layers alternately from the bottom end, then the top end,
      then the new bottom end, then the new top end, and so on.
    - The first selected stripe becomes the center solid square with side length
      equal to its height; each later selected stripe wraps the current square as
      a frame of matching thickness.
    """
    runs_top_to_bottom = _parse_stripes_top_to_bottom(np.array(grid, dtype=int))
    runs_center_out = _alternate_center_out_order(runs_top_to_bottom)
    return _build_nested_square(runs_center_out)


def _gen_attempt() -> np.ndarray | None:
    W = int(np.random.randint(7, 15))
    n_runs = int(np.random.randint(3, 7))
    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    if heights[-1] + 2 * sum(heights[:-1]) > 30:
        return None

    cols = [int(c) for c in np.random.choice(np.arange(1, 10), size=n_runs, replace=False)]
    runs_top_to_bottom = list(zip(cols, heights))
    stripes = [np.full((h, W), col, dtype=int) for col, h in runs_top_to_bottom]
    grid = np.vstack(stripes)

    variant_out = _build_nested_square(_alternate_center_out_order(runs_top_to_bottom))
    original_out = _build_nested_square(_original_center_out_order(runs_top_to_bottom))

    if np.array_equal(variant_out, original_out):
        return None
    if not np.array_equal(main(grid), variant_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
