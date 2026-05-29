# Combo variant of 31f7f899 (index 11)
# combo_spec: D2/0+D9/0+D13/0
# Name: 颜色消歧+共享颜色+非对称条

from __future__ import annotations

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    The scene still has one full-width baseline and 4-5 vertical rectangular bars
#    crossing that baseline. Unlike the seed task, bars may share colors and are
#    vertically asymmetric: a bar is described by `(top, bottom)` where `top !=
#    bottom`, so its total height is `top + bottom + 1`.
#    Small example on a 13x13 grid with base row 6:
#      input slots left->right colors = [5, 2, 5, 7]
#      input bars                = [(1,4), (3,1), (2,2), (4,1)]
#      totals                    = [6, 5, 5, 6]
#      color ranks               = [5, 2, 5, 7] under palette [1,2,3,4,5,7,9]
#    Original logic would have treated colors as unique identities and symmetric
#    height as `bbox_h`, so the bars with total 5 would be ambiguous or would be
#    ordered by incidental x-position. In the combo rule, shape order is:
#      1. total height = top + bottom + 1,
#      2. palette order of the *input* bar color when totals tie,
#      3. anchor only as a deterministic fallback for unresolved same-color ties.
#    Output then pastes the i-th ranked shape into the i-th leftmost slot, while
#    keeping each destination slot's original color. Shared colors therefore force
#    the solver to distinguish "left blue" from "right blue" spatially, and
#    asymmetry forces it to infer total height rather than only the upper or lower
#    reach.
#
# b) Complexity calibration
#    The generator keeps the geometry simple: all objects are solid rectangles of
#    width 1 or 3, and there is exactly one duplicated non-baseline color. This is
#    enough to make shared-color slot identity visible without flooding the task
#    with many indistinguishable objects. Each generated sample also enforces:
#      - every bar is asymmetric,
#      - at least one pair of bars has the same total height but different colors,
#      - the combo order differs from sorting by top only, bottom only, and total
#        height alone with stable x-order.
#    That makes the fused rule observable in a few examples but still keeps the
#    hypothesis space human-sized.
#
# c) Hidden-assumption audit of the original solver
#    - Original base detection chose the most frequent non-background `(row,color)`.
#      That can fail once a duplicated bar color covers many cells, so the combo
#      solver instead finds the row with maximal non-background coverage and reads
#      the baseline color from the preserved row edges.
#    - Original component extraction relied on color uniqueness. Shared colors break
#      that interpretation, so we recover slots by scanning contiguous non-baseline
#      runs on the baseline row and then read each bar from its slot columns.
#    - Original `bbox_h` was sufficient because bars were symmetric. Here the
#      correct primary key is still total height, but it must be reconstructed from
#      `top` and `bottom` relative to the baseline rather than assumed from the
#      generative symmetry.
#    - D2/0 changes the tie-break key from incidental geometry/position to palette
#      order. With D9/0, color ties can still remain unresolved, so `anchor` is
#      retained only as a last deterministic fallback. The generator avoids samples
#      where that tertiary fallback becomes the visible deciding factor.
#
# d) Placement legality
#    The generator reserves one baseline-colored cell on each side of the row and
#    at least one baseline-colored gap between neighboring slots. Bars are pasted
#    fully in-bounds with `top <= base_r` and `bottom <= n-1-base_r`, so they never
#    overlap or spill outside the grid. Because each slot owns a disjoint column
#    interval, extracting a bar from its slot columns is exact.
#
# e) Visual observability
#    A human looking at 2-4 examples can see three stable cues:
#      - some slots share the same color, so color is not a unique identity;
#      - bars with equal total height swap according to color order rather than
#        staying in their original left-right order;
#      - the upper and lower tails move together as one asymmetric shape, so the
#        ordering cannot be based on only the top reach or only the bottom reach.
#    That combination exposes the intended fused rule instead of collapsing back to
#    the seed task.


PALETTE_ORDER = [1, 2, 3, 4, 5, 7, 9]
COLOR_RANK = {int(c): i for i, c in enumerate(PALETTE_ORDER)}


def _find_base_row_and_color(grid: np.ndarray, bg: int) -> tuple[int, int]:
    non_bg_counts = np.sum(grid != int(bg), axis=1)
    base_r = int(np.argmax(non_bg_counts))
    row = grid[base_r]
    base_col = int(row[0])
    assert int(non_bg_counts[base_r]) == int(grid.shape[1])
    assert int(row[-1]) == int(base_col)
    return int(base_r), int(base_col)


def _extract_objects(grid: np.ndarray, base_r: int, base_col: int, bg: int) -> list[dict[str, object]]:
    row = grid[base_r]
    H, W = grid.shape
    objects: list[dict[str, object]] = []
    c = 0
    while c < W:
        if int(row[c]) == int(base_col):
            c += 1
            continue

        c0 = int(c)
        slot_col = int(row[c0])
        while c + 1 < W and int(row[c + 1]) != int(base_col):
            c += 1
        c1 = int(c)

        cols = grid[:, c0 : c1 + 1]
        local_coords = np.argwhere(cols != int(bg))
        assert local_coords.size > 0

        coords = [(int(r), int(c0 + dc)) for r, dc in local_coords]
        rows = [int(r) for r, _c in coords]
        anchor = (int(c0) + int(c1)) // 2
        top = int(base_r - min(rows))
        bottom = int(max(rows) - base_r)
        total = int(top + bottom + 1)
        rel = [(int(r) - int(base_r), int(col) - int(anchor)) for r, col in coords]

        objects.append(
            {
                "color": int(slot_col),
                "anchor": int(anchor),
                "top": int(top),
                "bottom": int(bottom),
                "height": int(total),
                "size": int(len(rel)),
                "rel": rel,
            }
        )
        c += 1

    assert objects
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Recover slots from contiguous non-baseline runs on the baseline row.
    - Each slot keeps only its x-position and destination color identity.
    - Read each source bar's asymmetric `(top, bottom)` extent from its slot.
    - Sort source shapes by `(total_height, palette_color_order, anchor)`.
    - Paste the ranked shapes into left-to-right slots, recoloring them to the
      destination slot color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    base_r, base_col = _find_base_row_and_color(g, bg)

    objects = _extract_objects(g, base_r, base_col, bg)
    slots = sorted(objects, key=lambda o: int(o["anchor"]))
    shapes = sorted(
        objects,
        key=lambda o: (
            int(o["height"]),
            int(COLOR_RANK[int(o["color"])]),
            int(o["anchor"]),
        ),
    )

    out = np.full((H, W), int(bg), dtype=int)
    out[int(base_r), :] = int(base_col)

    for slot, shape in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(slot_col)

    return out


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_lefts(n: int, k: int, width: int) -> list[int]:
    min_used = int(k * width + (k - 1) + 2)
    extra = int(n - min_used)
    if extra < 0:
        raise ValueError("bars do not fit with edge margins")
    gaps = _rand_composition(extra, k + 1)
    lefts: list[int] = []
    c = 1 + int(gaps[0])
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


def _render_from_order(
    shape: tuple[int, int],
    base_r: int,
    base_col: int,
    bg: int,
    objects: list[dict[str, object]],
    shape_order: list[int],
) -> np.ndarray:
    H, W = shape
    out = np.full((H, W), int(bg), dtype=int)
    out[int(base_r), :] = int(base_col)
    slot_order = [int(i) for i in sorted(range(len(objects)), key=lambda idx: int(objects[idx]["anchor"]))]
    for slot_idx, shape_idx in zip(slot_order, shape_order):
        slot_col = int(objects[slot_idx]["color"])
        slot_anchor = int(objects[slot_idx]["anchor"])
        for dr, dc in objects[shape_idx]["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(slot_col)
    return out


def _sample_colors(k: int) -> list[int]:
    dup = int(np.random.choice(PALETTE_ORDER))
    others = [int(c) for c in np.random.choice([c for c in PALETTE_ORDER if c != dup], size=k - 2, replace=False)]
    colors = [dup, dup, *others]
    for _ in range(80):
        np.random.shuffle(colors)
        dup_idx = [i for i, c in enumerate(colors) if int(c) == int(dup)]
        if abs(int(dup_idx[0]) - int(dup_idx[1])) >= 1:
            return [int(c) for c in colors]
    return [int(c) for c in colors]


def _sample_bars(colors: list[int], max_extent: int) -> list[tuple[int, int]] | None:
    k = len(colors)
    for _ in range(2000):
        bars: list[tuple[int, int]] = []
        for _idx in range(k):
            top = int(np.random.randint(1, max_extent + 1))
            bottom = int(np.random.randint(1, max_extent + 1))
            if top == bottom:
                break
            bars.append((int(top), int(bottom)))
        if len(bars) != k:
            continue

        tops = [int(t) for t, _b in bars]
        bottoms = [int(b) for _t, b in bars]
        totals = [int(t + b + 1) for t, b in bars]

        if all(len({totals[i], totals[j]}) == 2 for i in range(k) for j in range(i + 1, k)):
            continue

        if any(
            int(totals[i]) == int(totals[j]) and int(colors[i]) == int(colors[j])
            for i in range(k)
            for j in range(i + 1, k)
        ):
            continue

        tie_pairs = [
            (i, j)
            for i in range(k)
            for j in range(i + 1, k)
            if int(totals[i]) == int(totals[j]) and int(colors[i]) != int(colors[j])
        ]
        if not tie_pairs:
            continue
        if not any(tuple(bars[i]) != tuple(bars[j]) for i, j in tie_pairs):
            continue

        order_combo = tuple(
            i
            for i in sorted(
                range(k),
                key=lambda idx: (int(totals[idx]), int(COLOR_RANK[int(colors[idx])]), int(idx)),
            )
        )
        order_total_only = tuple(i for i in sorted(range(k), key=lambda idx: (int(totals[idx]), int(idx))))
        order_top = tuple(i for i in sorted(range(k), key=lambda idx: (int(tops[idx]), int(idx))))
        order_bottom = tuple(i for i in sorted(range(k), key=lambda idx: (int(bottoms[idx]), int(idx))))

        if order_combo == tuple(range(k)):
            continue
        if order_combo == order_total_only:
            continue
        if order_combo == order_top:
            continue
        if order_combo == order_bottom:
            continue

        dup_colors = [c for c in set(int(x) for x in colors) if colors.count(c) > 1]
        assert len(dup_colors) == 1
        dup = int(dup_colors[0])
        dup_totals = [int(totals[i]) for i, c in enumerate(colors) if int(c) == int(dup)]
        if len(set(dup_totals)) != len(dup_totals):
            continue

        if len(set(order_combo)) != k:
            continue

        return [(int(t), int(b)) for t, b in bars]
    return None


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(11, 18, 2)))
    base_r = n // 2
    bg = 8
    base_color = 6

    k_max = min(5, base_r)
    if k_max < 4:
        return None
    k = int(np.random.randint(4, k_max + 1))

    feasible_widths = [1]
    if int(k * 3 + (k - 1) + 2) <= int(n):
        feasible_widths.append(3)
    width = int(np.random.choice(feasible_widths))

    lefts = _slot_lefts(n, k, width)
    colors = _sample_colors(k)
    bars = _sample_bars(colors, base_r)
    if bars is None:
        return None

    inp = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, bars)
    out = main(inp)
    if np.array_equal(inp, out):
        return None

    objects = _extract_objects(inp, base_r, base_color, bg)
    totals = [int(o["height"]) for o in objects]
    tops = [int(o["top"]) for o in objects]
    bottoms = [int(o["bottom"]) for o in objects]
    cols = [int(o["color"]) for o in objects]
    combo_order = [
        i
        for i in sorted(
            range(k),
            key=lambda idx: (int(totals[idx]), int(COLOR_RANK[int(cols[idx])]), int(idx)),
        )
    ]
    total_only_order = [i for i in sorted(range(k), key=lambda idx: (int(totals[idx]), int(idx)))]
    top_only_order = [i for i in sorted(range(k), key=lambda idx: (int(tops[idx]), int(idx)))]
    bottom_only_order = [i for i in sorted(range(k), key=lambda idx: (int(bottoms[idx]), int(idx)))]
    original_like_order = [
        i
        for i in sorted(
            range(k),
            key=lambda idx: (int(totals[idx]), int(objects[idx]["size"]), int(objects[idx]["anchor"]), int(cols[idx])),
        )
    ]

    if tuple(combo_order) == tuple(total_only_order):
        return None
    if tuple(combo_order) == tuple(top_only_order):
        return None
    if tuple(combo_order) == tuple(bottom_only_order):
        return None
    if not any(int(totals[i]) == int(totals[j]) and int(cols[i]) != int(cols[j]) for i in range(k) for j in range(i + 1, k)):
        return None
    if len(set(cols)) != k - 1:
        return None
    if not any(int(totals[i]) == int(totals[j]) and int(cols[i]) != int(cols[j]) and tuple(bars[i]) != tuple(bars[j]) for i in range(k) for j in range(i + 1, k)):
        return None
    combo_render = _render_from_order(inp.shape, base_r, base_color, bg, objects, combo_order)
    if not np.array_equal(combo_render, out):
        return None
    if np.array_equal(out, _render_from_order(inp.shape, base_r, base_color, bg, objects, total_only_order)):
        return None
    if np.array_equal(out, _render_from_order(inp.shape, base_r, base_color, bg, objects, top_only_order)):
        return None
    if np.array_equal(out, _render_from_order(inp.shape, base_r, base_color, bg, objects, bottom_only_order)):
        return None
    if np.array_equal(out, _render_from_order(inp.shape, base_r, base_color, bg, objects, original_like_order)):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 600 attempts")


# Review of new implicit assumptions
# ==================================
# - `_find_base_row_and_color()` assumes the baseline is the unique row with
#   maximal non-background coverage and that its left/right edge cells remain
#   baseline-colored. `_slot_lefts()` reserves edge margins and `_draw_scene()`
#   never paints outside slot columns, so the generator guarantees this.
# - `_extract_objects()` assumes each slot corresponds to one contiguous
#   non-baseline run on the baseline row and that every bar stays within that run's
#   column interval. The generator enforces at least one baseline gap between slots
#   and only draws axis-aligned rectangles inside each slot width.
# - `main()` assumes every source bar crosses the baseline, so `top` and `bottom`
#   are measurable relative to the same base row. `_draw_scene()` always pastes
#   rectangles spanning rows `base_r-top ... base_r+bottom`, so this holds.
# - The solver's third tie-break key is `anchor`, but the generator rejects cases
#   where equal-total equal-color bars would make that fallback the visible rule.
#   This keeps the training signal on the intended combo behavior while still
#   making `main()` deterministic on edge cases.
