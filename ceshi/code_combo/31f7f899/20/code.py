# Combo variant of 31f7f899 (index 20)
# combo_spec: D5/4+D13/0
# Name: 高度归一化+非对称条形

from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste

# Design analysis
# - a) Concrete behavior and trace:
#   The original task shows several vertical monochrome bars crossing one shared
#   baseline row. Each bar is symmetric around the baseline, so `bbox_h(...)`
#   already equals the intended height. The solver sorts bars by that height,
#   moves the i-th shortest silhouette into the i-th leftmost slot, and recolors
#   the moved pixels to the destination slot color.
#   This combo changes two linked parts at once:
#   1. Input bars are asymmetric rectangles described by `(top_extent,
#      bottom_extent)`, where `top_extent != bottom_extent`.
#   2. The sorting key is still the TOTAL height `top_extent + bottom_extent + 1`,
#      but after sorting the output no longer preserves those source heights:
#      every moved bar is normalized to one common full height while keeping the
#      SOURCE color of the ranked bar.
#   Small trace on a 9x9 grid with baseline row 4:
#     slot colors left->right = [2, 4, 7]
#     input bars left->right = [(top,bottom) = (1,3), (3,1), (2,1)]
#     their total heights are [5, 5, 4]
#   That trace is ambiguous because of the tie, so the generator forbids it.
#   A valid trace is:
#     input bars left->right = [(1,3), (3,2), (2,1)]
#     totals               = [5, 6, 4]
#     total-height order   = color 7, then 2, then 4
#   Original solver on symmetric tasks would output slot colors [2,4,7] with the
#   ranked source silhouettes. Combo output instead paints three equal-height
#   bars whose left->right colors become [7,2,4]. Height differences disappear,
#   so the ranking is only visible through the color permutation.
# - b) Complexity calibration:
#   If output bars were normalized while also keeping destination-slot colors,
#   the task would collapse because every output bar would become visually
#   identical apart from fixed slot location. To keep the sort observable, the
#   solver must move source colors with the ranked bars. The generator therefore
#   enforces unique bar colors, distinct total heights, non-identity permutations,
#   and asymmetric bars for every object. It also rejects samples where sorting by
#   total height would accidentally match sorting by top extent or bottom extent,
#   because that would weaken the need to infer the correct height metric.
# - c) Hidden-assumption audit of the original main():
#   * Base-line detection by the longest non-background row still holds because
#     the input retains one full-width baseline of one color.
#   * Component extraction still assumes every bar is one monochrome 4-connected
#     object distinct from the baseline color. The generator preserves that.
#   * The original `bbox_h(...)` height measure remains valid here only because
#     the combo explicitly sorts by TOTAL height. What breaks is the assumption
#     that the extracted relative pixel set should be replayed unchanged in the
#     output; normalization requires rebuilding a canonical full-height bar.
#   * The original paint step used `slot["color"]`, which depended on the old
#     recolor rule. The combo must instead use `shape["color"]`.
#   * New assumption: each source object is a solid vertical rectangle, so its
#     horizontal footprint can be summarized by column offsets and then stretched
#     vertically. The generator guarantees this by drawing filled rectangles.
#   * New assumption: total-height ties would make the color permutation less
#     clear. The generator guarantees pairwise distinct totals.
# - d) Placement legality:
#   Odd grid sizes with `base_r = n // 2` give equal capacity above and below the
#   baseline. Each source bar uses `1 <= top,bottom <= base_r` and `top != bottom`,
#   so it always crosses the baseline, stays within bounds, and is never already
#   full-height. Horizontal slots all share one width and have at least one empty
#   column between neighbors, so neither the input bars nor the normalized output
#   bars overlap.
# - e) Visual observability:
#   Across examples, a human can see that input bars are asymmetric and have
#   different total heights, but output bars all become the same height. The only
#   consistent surviving cue is which color lands in which left-to-right slot.
#   Because the generator rejects cases where top-only or bottom-only sorting
#   agrees with total-height sorting, the examples visibly support the fused rule:
#   compute total height from an asymmetric bar, then use that hidden ranking to
#   decide the output color permutation.


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
        rows = [r for r, _c in pts]
        cols = [c for _r, c in pts]
        top_extent = int(base_r - min(rows))
        bottom_extent = int(max(rows) - base_r)
        total_height = int(top_extent + bottom_extent + 1)

        base_cols = sorted(c for r, c in pts if r == int(base_r))
        if base_cols:
            anchor = int(base_cols[len(base_cols) // 2])
        else:
            anchor = int(sorted(cols)[len(cols) // 2])

        col_offsets = sorted({int(c) - int(anchor) for _r, c in pts})
        objects.append(
            {
                "color": int(col),
                "anchor": int(anchor),
                "top": int(top_extent),
                "bottom": int(bottom_extent),
                "height": int(total_height),
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
    if not (0 <= top <= bottom < H):
        return
    for dc in col_offsets:
        cc = int(anchor + dc)
        if 0 <= cc < W:
            out[top : bottom + 1, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - detect the shared baseline;
    - treat each non-base component as an asymmetric vertical bar;
    - rank bars by total height `top + bottom + 1`;
    - assign the ranked bars to slots from left to right;
    - keep each ranked bar's source color, but normalize every placed bar to one
      common full height.
    """
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    bg = int(get_bg(g))
    base_r, base_col = _detect_base(g, bg)

    objects = _extract_objects(g, bg, base_r, base_col)
    assert objects

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(
        objects,
        key=lambda o: (
            int(o["height"]),
            int(o["top"]),
            int(o["bottom"]),
            int(o["size"]),
            int(o["anchor"]),
            int(o["color"]),
        ),
    )

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
    bars: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)
    for left, color, (top, bottom) in zip(lefts, colors, bars):
        patch = np.full((int(top + bottom + 1), int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - top), int(left))
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
    bars = [(int(full_h // 2), int(full_h // 2))] * len(lefts)
    return _draw_scene(n, base_r, base_color, bg, lefts, width, sorted_colors, bars)


def _rank_order(values: list[int]) -> tuple[int, ...]:
    return tuple(i for i, _v in sorted(enumerate(values), key=lambda iv: (int(iv[1]), int(iv[0]))))


def _sample_asymmetric_bars(k: int, base_r: int) -> list[tuple[int, int]] | None:
    candidates = [(int(t), int(b)) for t in range(1, base_r + 1) for b in range(1, base_r + 1) if t != b]
    if len(candidates) < k:
        return None

    for _ in range(400):
        pick = np.random.choice(len(candidates), size=k, replace=False)
        bars = [candidates[int(i)] for i in pick]
        tops = [int(t) for t, _b in bars]
        bottoms = [int(b) for _t, b in bars]
        totals = [int(t + b + 1) for t, b in bars]
        if len(set(totals)) != k:
            continue
        if any(t == b for t, b in bars):
            continue
        if _rank_order(totals) == _rank_order(tops):
            continue
        if _rank_order(totals) == _rank_order(bottoms):
            continue
        return sorted(bars, key=lambda bar: (int(bar[0] + bar[1] + 1), int(bar[0]), int(bar[1])))
    return None


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(9, 18, 2)))
    base_r = n // 2
    bg = 8
    base_color = 6

    max_count = min(5, base_r)
    if max_count < 3:
        return None
    k = int(np.random.randint(3, max_count + 1))

    feasible_widths = [1]
    if n - 3 * k > 3 and n >= 4 * k - 1:
        feasible_widths.append(3)
    width = int(np.random.choice(feasible_widths))

    lefts = _slot_lefts(n, k, width)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]

    bars_sorted = _sample_asymmetric_bars(k, base_r)
    if bars_sorted is None:
        return None

    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    bars_input = [bars_sorted[int(i)] for i in perm]
    inp = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, bars_input)

    tops_input = [int(t) for t, _b in bars_input]
    bottoms_input = [int(b) for _t, b in bars_input]
    totals_input = [int(t + b + 1) for t, b in bars_input]
    order = [int(i) for i in np.argsort(totals_input)]
    if tuple(order) == _rank_order(tops_input):
        return None
    if tuple(order) == _rank_order(bottoms_input):
        return None
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
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
