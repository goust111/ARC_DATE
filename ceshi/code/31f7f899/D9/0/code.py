# Variant of 31f7f899
# Dimension: D9 — 条形数量与颜色约束 (src)
# Change: 3-5个条形，每个使用不同的非背景非基线色，从 {1,2,3,4,5,7,9} 中无放回选取... → 允许条形共享颜色

from __future__ import annotations

import numpy as np

from arc_utils import get_bg, paste

# Design analysis for the D9 variant
# ==================================
# a) Concrete behavior and trace
#    The scene still consists of one horizontal baseline plus 3-5 vertical bars
#    centered on that baseline. The output rule is still "sort bar shapes by
#    height and place them back into the left-to-right slots", but slot identity
#    is now purely spatial. Example:
#      slots x=[2, 6, 10], colors=[2, 2, 5], input heights=[7, 3, 5]
#      sorted heights=[3, 5, 7]
#      output keeps slot colors at x=[2, 6, 10], so the left red slot gets
#      height 3, the middle red slot gets height 5, and the right green slot
#      gets height 7.
#    In the original task, color uniqueness made "red slot" unambiguous. In the
#    variant, the two red bars must be distinguished by x-position.
#
# b) Complexity calibration
#    Repeated colors are made explicit in every generated sample by forcing
#    exactly one duplicated non-baseline color. The duplicate pair is kept in
#    two different slots, usually non-adjacent, so the repeated-color behavior
#    is visible without creating a combinatorial mess of many identical bars.
#    Heights remain unique, preserving a single clear sorting order.
#
# c) Hidden-assumption audit against the original solver
#    - Original base detection used the (row, color) with the largest count.
#      That is unsafe once a bar color can repeat, because a duplicated bar
#      color may outnumber the baseline color on the baseline row.
#    - Original object extraction labeled monochromatic components and attached
#      meaning to slot colors. Repeated colors do not merge disconnected bars,
#      but the intended identity source is now x-position, not color.
#    - Original slot sorting used (anchor, color) and shape sorting used color
#      as a late tie-break. That is harmless with unique colors but conceptually
#      wrong for the variant.
#    The new solver therefore:
#      1. finds the baseline row as the unique row with the most non-background
#         cells,
#      2. gets the baseline color from the row edges, which the generator keeps
#         untouched by bars,
#      3. scans contiguous non-baseline runs on that row to recover slots, and
#      4. extracts each bar directly from its slot columns, independent of color.
#
# d) Placement legality
#    The generator reserves at least one baseline-colored cell at both left and
#    right edges, and at least one baseline-colored gap between neighboring bars.
#    Bars are pasted fully in-bounds and never overlap because each slot owns a
#    disjoint column interval. Output-first construction is retained: create the
#    sorted output, permute heights to form the input, then verify main(input).
#
# e) Visual observability
#    Every generated sample contains a duplicate color pair, so at least two
#    slots share the same color. This makes the change visible across only a few
#    examples: the user can no longer infer object identity from color labels and
#    must track "left red" versus "middle red" by location.


def _find_base_row_and_color(grid: np.ndarray, bg: int) -> tuple[int, int]:
    non_bg_counts = np.sum(grid != int(bg), axis=1)
    base_r = int(np.argmax(non_bg_counts))
    row = grid[base_r]
    base_col = int(row[0])
    assert int(non_bg_counts[base_r]) == int(grid.shape[1])
    assert int(row[-1]) == int(base_col)
    return base_r, base_col


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

        anchor = (int(c0) + int(c1)) // 2
        min_r = min(int(r) for r, _c in coords)
        max_r = max(int(r) for r, _c in coords)
        rel = [(int(r) - int(base_r), int(col) - int(anchor)) for r, col in coords]

        objects.append(
            {
                "color": int(slot_col),
                "anchor": int(anchor),
                "height": int(max_r - min_r + 1),
                "size": int(len(rel)),
                "rel": rel,
            }
        )
        c += 1

    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort vertical bars by height and redistribute them across fixed x-slots.

    Variant rule:
    - Slots are the contiguous non-baseline runs on the baseline row.
    - Slots keep their x-position and their original color, even if multiple
      slots share that color.
    - Bar shapes are sorted by height (shortest -> tallest) and pasted into the
      left-to-right slots.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    base_r, base_col = _find_base_row_and_color(g, bg)

    objects = _extract_objects(g, base_r, base_col, bg)
    slots = sorted(objects, key=lambda o: int(o["anchor"]))
    shapes = sorted(objects, key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"])))

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


def _sample_colors(k: int) -> list[int]:
    palette = [1, 2, 3, 4, 5, 7, 9]
    dup = int(np.random.choice(palette))
    others = [int(c) for c in np.random.choice([c for c in palette if c != dup], size=k - 2, replace=False)]
    colors = [dup, dup, *others]
    for _ in range(50):
        np.random.shuffle(colors)
        dup_idx = [i for i, c in enumerate(colors) if int(c) == int(dup)]
        if abs(int(dup_idx[0]) - int(dup_idx[1])) >= 2 or k <= 3:
            return [int(c) for c in colors]
    return [int(c) for c in colors]


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


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(7, 18, 2)))
    base_r = n // 2
    bg = 8
    base_color = 6

    height_choices = list(range(1, min(n, 11) + 1, 2))
    max_count = min(5, len(height_choices), (n - 1) // 2)
    if max_count < 3:
        return None
    k = int(np.random.randint(3, max_count + 1))

    feasible_widths = [1]
    if int(k * 3 + (k - 1) + 2) <= int(n):
        feasible_widths.append(3)
    width = int(np.random.choice(feasible_widths))

    lefts = _slot_lefts(n, k, width)
    colors = _sample_colors(k)

    sorted_heights = sorted(int(h) for h in np.random.choice(height_choices, size=k, replace=False))
    out = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, sorted_heights)

    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    in_heights = [int(sorted_heights[i]) for i in perm]
    inp = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, in_heights)

    if np.array_equal(inp, out):
        return None
    if len(set(int(c) for c in colors)) != k - 1:
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


# Review of new implicit assumptions
# ==================================
# - main() assumes the baseline row is the only row with maximal non-background
#   coverage. generate_input() guarantees this by keeping a full-width baseline.
# - main() assumes the baseline color is still visible at both row edges.
#   generate_input() guarantees at least one untouched edge cell on both sides.
# - main() assumes different slots are separated by at least one baseline cell,
#   so scanning non-baseline runs yields exact slot intervals. _slot_lefts()
#   enforces that gap.
# - main() assumes each slot's full bar lies inside its own column interval.
#   generate_input() only uses axis-aligned rectangles pasted entirely in-bounds.
# - main() assumes height order is strict. generate_input() samples heights
#   without replacement, so sorting is unambiguous.
