# Variant of 31f7f899
# Dimension: D6 — 基线结构 (src)
# Change: 单条水平基线跨越网格全宽，位于网格中间行（H//2），颜色固定为6... → 垂直基线

from __future__ import annotations

import numpy as np

from arc_utils import bbox_w, find_components, get_bg, paste


# Design analysis for the D6 variant
# a) Concrete behavior and trace
#    Original task: a full-width horizontal baseline sits on row H//2, and each object is a
#    vertical monochrome bar centered on that row. The solver extracts each bar shape relative
#    to the baseline row, sorts shapes by height, then reassigns those shapes to the left-to-right
#    color slots.
#    Variant: rotate the whole spatial arrangement by 90 degrees. Now the baseline is a full-height
#    vertical line on column W//2, and each object is a horizontal monochrome bar centered on that
#    column. Slots are no longer ordered left-to-right; they are ordered top-to-bottom by the bars'
#    anchor rows. Shapes are ranked by width (shortest to longest), then pasted into those fixed
#    row slots while keeping each slot's original color.
#    Small trace: suppose rows 1, 4, 7 contain red/green/yellow horizontal bars of widths 7/3/5.
#    Original logic would try to use row statistics and bar heights, which is wrong after rotation.
#    Variant logic detects the baseline column, records slot rows [1,4,7], sorts widths [3,5,7],
#    and outputs green-width3 at top slot's color, yellow-width5 at middle slot's color, red-width7
#    at bottom slot's color. The difference is exactly the axis swap: row slots + width ranking.
# b) Complexity calibration
#    We keep the same task family and only rotate its geometry. Bars still use a small set of odd
#    extents and a thickness of 1 or 3, so objects are easy to see but not degenerate. We require
#    at least 3 bars with distinct widths and distinct colors so the top-to-bottom redistribution is
#    visible in a few examples.
# c) Hidden assumptions in the original main()
#    - Baseline search iterated over rows and assumed the dominant non-bg run lived on one row.
#      That fails here; it must iterate over columns.
#    - Shape size used bbox_h(), which measured vertical extent. After rotation, the discriminative
#      measure is bbox_w().
#    - Slot anchor used columns touching the base row. After rotation, anchors are rows touching the
#      base column.
#    - Relative coordinates were stored around (base_row, anchor_col). Now they must be stored around
#      (anchor_row, base_col).
#    - Output reconstruction drew a horizontal baseline and sorted by x. Now it must draw a vertical
#      baseline and sort by y.
#    These are harmless in the original orientation because every bar intersects the base row and
#    varies only in height; under the variant they would either detect the wrong baseline or sort on
#    the wrong axis.
# d) Placement legality
#    The generator uses output-first construction. It places k horizontal rectangles with a shared
#    center column W//2, separated vertically by at least one empty row. Thickness is 1 or 3, chosen
#    only when all bars fit and enough baseline-color cells remain in the center column so baseline
#    detection stays reliable. Every patch is pasted fully in-bounds, and bars never overlap because
#    their row intervals are disjoint.
# e) Visual observability
#    Humans can see the rotation immediately: the constant structural element is a vertical pink line,
#    while colored bars extend left/right at different rows. Across examples, colors stay attached to
#    row slots but widths migrate in sorted order from top to bottom. Distinct odd widths and at least
#    3 bars make the permutation visible and prevent the task from collapsing back into the original
#    horizontal-baseline reading.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort horizontal bars by width and redistribute them across fixed colored row-slots.

    Input structure:
    - bg background; a vertical base line of a single color spans the full grid height.
    - Several monochrome horizontal bars of distinct colors extend left/right from that line.

    Rule:
    - Rank bar shapes by width (shortest to longest).
    - Rank original y-positions (slots) top to bottom.
    - Place the i-th shortest shape into the i-th topmost slot, recoloring it to that slot's original color.
    - The base line is preserved, with bars overwriting it at their intersections just as in the input.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    best = None  # (count, col, color)
    for c in range(W):
        vals, counts = np.unique(g[:, c], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == bg:
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(c), int(v))
    assert best is not None
    _cnt, base_c, base_col = best

    objects: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {bg, int(base_col)}:
            continue

        pts = comp["coords"]
        b = comp["bbox"]
        width = int(bbox_w(b))

        base_rows = sorted(int(r) for r, c in pts if int(c) == int(base_c))
        if base_rows:
            anchor = int(base_rows[len(base_rows) // 2])
        else:
            rows_sorted = sorted(int(r) for r, _c in pts)
            anchor = int(rows_sorted[len(rows_sorted) // 2])

        rel = [(int(r) - int(anchor), int(c) - int(base_c)) for r, c in pts]
        objects.append(
            {
                "color": col,
                "anchor": anchor,
                "width": width,
                "size": int(len(rel)),
                "rel": rel,
            }
        )

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(objects, key=lambda o: (int(o["width"]), int(o["size"]), int(o["anchor"]), int(o["color"])))

    out = np.full_like(g, bg)
    out[:, int(base_c)] = int(base_col)

    for slot, shape in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in shape["rel"]:
            rr = int(slot_anchor) + int(dr)
            cc = int(base_c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = slot_col

    return out


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_tops(n: int, k: int, thickness: int) -> list[int]:
    used = int(k * thickness + (k - 1))
    extra = int(n - used)
    if extra < 0:
        raise ValueError("bars do not fit")
    gaps = _rand_composition(extra, k + 1)
    tops: list[int] = []
    r = int(gaps[0])
    for i in range(k):
        tops.append(int(r))
        r += int(thickness)
        if i < k - 1:
            r += 1 + int(gaps[i + 1])
    return tops


def _draw_scene(
    n: int,
    base_c: int,
    base_color: int,
    bg: int,
    tops: list[int],
    thickness: int,
    colors: list[int],
    widths: list[int],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[:, int(base_c)] = int(base_color)
    for top, color, width in zip(tops, colors, widths):
        patch = np.full((int(thickness), int(width)), int(color), dtype=int)
        paste(grid, patch, int(top), int(base_c - width // 2))
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(7, 18, 2)))
    base_c = n // 2
    bg = 8
    base_color = 6

    width_choices = list(range(1, min(n, 11) + 1, 2))
    max_count = min(5, len(width_choices), (n + 1) // 2)
    if max_count < 3:
        return None
    k = int(np.random.randint(3, max_count + 1))

    feasible_thicknesses = [1]
    if n - 3 * k > 3 and n >= 4 * k - 1:
        feasible_thicknesses.append(3)
    thickness = int(np.random.choice(feasible_thicknesses))

    tops = _slot_tops(n, k, thickness)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]

    sorted_widths = sorted(int(w) for w in np.random.choice(width_choices, size=k, replace=False))
    out = _draw_scene(n, base_c, base_color, bg, tops, thickness, colors, sorted_widths)

    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    in_widths = [int(sorted_widths[i]) for i in perm]
    inp = _draw_scene(n, base_c, base_color, bg, tops, thickness, colors, in_widths)

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
