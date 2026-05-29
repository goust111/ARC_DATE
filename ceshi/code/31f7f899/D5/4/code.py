# Variant of 31f7f899
# Dimension: D5 — 变换：重新着色 (transform)
# Change: recolor — 条形形状放入新槽位后，颜色变为该槽位的原始颜色... → 条形高度归一化后放入

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, find_components, get_bg, paste

# Design analysis
# - Source-task review: none. The variant is marked "creative", so there is no
#   external reference solver to inspect before editing this task.
# - a) Concrete behavior:
#   The input keeps the original geometry: one long horizontal baseline, plus
#   several same-width monochrome vertical bars that cross the baseline and have
#   distinct odd heights. The bar x-positions define fixed slots.
#   Original solver:
#     1. measure each bar height;
#     2. sort shapes by height;
#     3. move the i-th shortest bar shape to the i-th leftmost slot;
#     4. recolor that moved shape to the destination slot color.
#   Variant solver:
#     1. still rank bars by height;
#     2. still use left-to-right slots as destinations;
#     3. keep the SOURCE color of each ranked bar instead of recoloring to the
#        destination slot color;
#     4. normalize every moved bar to one common maximum odd height
#        `2 * min(base_r, H - 1 - base_r) + 1`, so it fills the full vertical
#        range available around the baseline.
#   Small trace:
#     Suppose three slots from left to right have colors [2, 4, 7], and their
#     input heights are [5, 1, 3]. Height order is color 4, then 7, then 2.
#     Original output would keep slot colors [2, 4, 7] and place heights
#     [1, 3, 5] there. Variant output places colors [4, 7, 2] into the three
#     slots, and all three bars are stretched to the same full height. Height is
#     no longer visible in the output; only the color-to-position permutation
#     reveals the ranking.
# - b) Complexity calibration:
#   If the output normalized both height and color-to-slot identity, the task
#   would collapse because every output bar would look identical. To keep the
#   sorting signal observable, the moved bar must retain its original color.
#   Heights remain distinct and odd in the input, and bar colors stay unique, so
#   the permutation is unambiguous in 2-4 ARC examples.
# - c) Original-code assumptions affected by the variant:
#   * Base-row detection by the most frequent non-background color still holds,
#     because the input construction is unchanged.
#   * Component extraction still assumes bars are monochrome 4-connected objects
#     distinct from the baseline color; generator preserves that.
#   * The original `rel` pixel list encoded the full bar silhouette and was safe
#     because output preserved shape height. That assumption breaks here because
#     the output must ignore source height. We therefore keep only horizontal
#     column offsets and rebuild a canonical full-height bar at the destination.
#   * The original paint loop used `slot["color"]`; that relied on the old
#     recolor rule. The variant must use `shape["color"]` instead.
# - d) Placement legality:
#   Generator keeps all bars disjoint by giving every bar the same width and at
#   least one empty-column gap between neighboring slots. The normalized output
#   uses the same slot anchors and horizontal widths, so bars still do not
#   overlap or leave the grid.
# - e) Visual observability:
#   Across examples, the input visibly shows different bar heights and shuffled
#   colors. The output shows equal-height bars whose left-to-right color order is
#   sorted by those hidden input heights. Because non-identity permutations are
#   enforced during generation, the new rule is visible and differs from the
#   original recolor behavior.
# - New implicit assumptions after the edit:
#   * Every object is a vertical bar whose occupied columns are contiguous and
#     identical across rows. The generator guarantees that by drawing solid
#     rectangles.
#   * Height ties would make color ordering ambiguous. The generator avoids ties
#     by sampling distinct heights without replacement.
#   * Slots are identified by the bar anchor columns found in the input. The
#     generator preserves one bar per slot, so sorting anchors remains valid.


def _detect_base(g: np.ndarray, bg: int) -> tuple[int, int]:
    best = None  # (count, row, color)
    for r in range(g.shape[0]):
        vals, counts = np.unique(g[r, :], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == int(bg):
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(r), int(v))
    assert best is not None
    _count, base_r, base_col = best
    return int(base_r), int(base_col)


def _extract_objects(g: np.ndarray, bg: int, base_r: int, base_col: int) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {int(bg), int(base_col)}:
            continue

        pts = [(int(r), int(c)) for r, c in comp["coords"]]
        height = int(bbox_h(comp["bbox"]))

        base_cols = sorted(c for r, c in pts if r == int(base_r))
        if base_cols:
            anchor = int(base_cols[len(base_cols) // 2])
        else:
            cols_sorted = sorted(c for _r, c in pts)
            anchor = int(cols_sorted[len(cols_sorted) // 2])

        col_offsets = sorted({int(c) - int(anchor) for _r, c in pts})
        objects.append(
            {
                "color": int(col),
                "anchor": int(anchor),
                "height": int(height),
                "size": int(len(pts)),
                "col_offsets": col_offsets,
            }
        )
    return objects


def _normalized_height(H: int, base_r: int) -> int:
    half = min(int(base_r), int(H - 1 - base_r))
    return int(2 * half + 1)


def _paint_normalized_bar(
    out: np.ndarray,
    *,
    base_r: int,
    anchor: int,
    color: int,
    col_offsets: list[int],
    height: int,
) -> None:
    half = int(height // 2)
    top = int(base_r - half)
    bottom = int(base_r + half)
    H, W = out.shape
    for dc in col_offsets:
        cc = int(anchor + dc)
        if 0 <= cc < W and 0 <= top <= bottom < H:
            out[top : bottom + 1, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort vertical bars by height, move their colors into left-to-right slots,
    and normalize every placed bar to one common maximum height.
    """
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    bg = int(get_bg(g))
    base_r, base_col = _detect_base(g, bg)

    objects = _extract_objects(g, bg, base_r, base_col)

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(objects, key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])))

    out = np.full_like(g, bg)
    out[int(base_r), :] = int(base_col)

    full_h = _normalized_height(H, base_r)
    for slot, shape in zip(slots, shapes):
        _paint_normalized_bar(
            out,
            base_r=int(base_r),
            anchor=int(slot["anchor"]),
            color=int(shape["color"]),
            col_offsets=list(shape["col_offsets"]),
            height=int(full_h),
        )

    return out


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_lefts(n: int, k: int, width: int) -> list[int]:
    used = int(k * width + (k - 1))
    extra = int(n - used)
    if extra < 0:
        raise ValueError("bars do not fit")
    gaps = _rand_composition(extra, k + 1)
    lefts: list[int] = []
    c = int(gaps[0])
    for i in range(k):
        lefts.append(int(c))
        c += int(width)
        if i < k - 1:
            c += 1 + int(gaps[i + 1])
    return lefts


def _draw_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    lefts: list[int],
    width: int,
    colors: list[int],
    heights: list[int],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)
    for left, color, height in zip(lefts, colors, heights):
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - height // 2), int(left))
    return grid


def _draw_normalized_output(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    lefts: list[int],
    width: int,
    sorted_colors: list[int],
) -> np.ndarray:
    full_h = _normalized_height(int(n), int(base_r))
    heights = [int(full_h)] * len(lefts)
    return _draw_scene(n, base_r, base_color, bg, lefts, width, sorted_colors, heights)


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(7, 18, 2)))
    base_r = n // 2
    bg = 8
    base_color = 6

    max_input_height = min(n, 11)
    if max_input_height == _normalized_height(n, base_r):
        max_input_height -= 2
    height_choices = list(range(1, max_input_height + 1, 2))
    max_count = min(5, len(height_choices), (n + 1) // 2)
    if max_count < 3:
        return None
    k = int(np.random.randint(3, max_count + 1))

    feasible_widths = [1]
    if n - 3 * k > 3 and n >= 4 * k - 1:
        feasible_widths.append(3)
    width = int(np.random.choice(feasible_widths))

    lefts = _slot_lefts(n, k, width)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]

    sorted_heights = sorted(int(h) for h in np.random.choice(height_choices, size=k, replace=False))
    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]

    in_heights = [int(sorted_heights[i]) for i in perm]
    inp = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, in_heights)

    order = np.argsort(in_heights)
    sorted_colors = [int(colors[i]) for i in order]
    expected = _draw_normalized_output(n, base_r, base_color, bg, lefts, width, sorted_colors)

    if np.array_equal(inp, expected):
        return None
    if sorted_colors == colors:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
