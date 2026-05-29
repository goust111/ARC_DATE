# Combo variant of 45a5af55 (index 25)
# combo_spec: D6/0+D9/1+D10/2
# Name: 固定调色板旋转数量编码嵌套

from __future__ import annotations

import numpy as np

"""
Design analysis for the fused variant
====================================

a) Concrete behavior and trace
------------------------------
Base task:
- Parse full-width horizontal stripes bottom -> top.
- Bottom stripe becomes the solid center square, later stripes wrap outward as
  same-color square frames whose thickness equals stripe height.

Naive fusion problem:
- If we only sort stripes by color and then rotate the finished nested square,
  the rotation is invisible: concentric square frames are 90-degree symmetric.
- So D9/1 would formally run but would not change any pixels. That is a
  degenerate fusion for ARC because one control variable would have no visible
  effect.

Implemented fused rule:
- Input stripe colors are used only to rank stripes by a fixed visible color
  order, not by vertical position. The generator uses a clean rainbow-like
  subset so this order is inferable by eye.
- Heights from the sorted stripes still define the nested-square geometry.
- Output colors no longer inherit the input stripe colors. Instead, every layer
  is painted from a fixed 4-color palette in clockwise side order:
    top, right, bottom, left
  and layer i starts at palette index (count_mod4 + i) mod 4.
- After building the full multicolor nest, rotate the whole result by
  90 degrees * (number_of_stripes mod 4).

Small trace:
- Suppose bottom->top parsed runs are:
    [(blue, 2), (red, 1), (yellow, 2)]
  using visible order red < yellow < blue.
- Original task:
    center = blue 2x2
    wrap red thickness 1
    wrap yellow thickness 2
- Fused task:
    sort by visible color rank -> [(red, 1), (yellow, 2), (blue, 2)]
    n = 3, so count_mod4 = 3
    center uses palette[(3 + 0) mod 4]
    yellow frame uses palette cycle starting at index 0 around its four sides
    blue frame uses palette cycle starting at index 1 around its four sides
    rotate final picture by 270 degrees
- Difference:
    1. input vertical order no longer decides nesting
    2. output colors come from a fixed palette, not the stripe colors
    3. stripe count changes both side-color phase and final orientation

b) Complexity calibration
-------------------------
- The scene stays simple: one stack of full-width stripes, no extra markers or
  scattered objects.
- The new reasoning load is limited to three coupled rules:
    1. color rank -> layer order
    2. fixed palette cycle -> output coloring
    3. stripe count mod 4 -> both palette phase and rotation
- We keep only 3 or 5 stripes in generation. That is enough to show repeated
  nested structure and a non-zero count code, without introducing too many
  cases.

c) Implicit-assumption review of the original solver
----------------------------------------------------
- Original `g[r, 0]` and row-equality parsing assumed each stripe row is
  perfectly uniform. This still holds; generator keeps full-width solid rows.
- Original logic assumed bottom->top stripe order equals inner->outer layer
  order. This is intentionally broken. We must parse spatial order first, then
  re-order runs by color rank before constructing layers.
- Original frame fill `np.full((new_n, new_n), color)` assumed each frame is
  monochrome. That assumption breaks because the combo uses a fixed palette
  cycle on each frame's sides. We therefore paint each layer into an existing
  canvas with explicit side regions.
- Original post-processing ended immediately after nesting. The combo adds a
  mandatory second stage: rotate by the stripe-count code.

d) Placement legality
---------------------
- Input placement remains trivial and legal: stacked full-width stripes with no
  gaps or overlap.
- Output size is still:
    center_height + 2 * sum(outer_heights)
  and generator rejects sizes above 30.
- Side-color painting is done on exact annulus regions, so different layers do
  not overlap incorrectly and every write stays in bounds.

e) Visual observability
-----------------------
- Sorting by hidden numeric color ids would be a bad ARC clue. To avoid that,
  the generator restricts input colors to a fixed visible progression:
    red -> yellow -> green -> cyan -> blue
  The solver uses a broader rank table, but generation sticks to that clearer
  subset.
- The fixed output palette is distinct from the input palette, so it is obvious
  that input colors are no longer copied into output colors.
- Making each frame side-colored rather than monochrome is the minimal change
  that keeps "fixed palette cycle" while making the final rotation visible.

f) Visual cleanliness
---------------------
- Output still contains only one compact nested object.
- Each frame annulus is partitioned cleanly into four side bands; corners belong
  deterministically to top/bottom bands, so there is no ambiguous overlap.
- No background holes or extra decorations are introduced.

g) Inferrability of the color change
------------------------------------
- Input colors carry exactly one role: ordering stripes by the fixed visible
  progression.
- Output colors carry exactly one role: they follow the same fixed side-cycle
  palette on every sample, with a stripe-count-dependent phase shift.
- Because 5-stripe samples repeat the 4-color cycle across layers, humans get
  repeated evidence that the palette rule is cyclic rather than arbitrary.

h) Quantitative anti-degeneration check
---------------------------------------
- The generator rejects any sample where the color-sort order accidentally
  matches bottom->top order, because then D6 would be latent.
- It also rejects any sample where the final output equals a baseline that uses
  the same sorted geometry but count_mod4 = 0 and no rotation. That ensures the
  stripe-count code is visibly active on every generated instance.

i) New solver assumptions and generator guarantees
--------------------------------------------------
- Solver assumes all stripe colors are distinct and belong to a known visible
  rank table. Generator samples without replacement from the chosen input chain.
- Solver assumes stripe count is at least 2 and uses count_mod4. Generator uses
  only 3 or 5 stripes, so count_mod4 is always non-zero and visually meaningful.
- Solver assumes a square output canvas exactly determined by the stripe heights.
  Generator enforces the side-length bound <= 30.
"""


# Generator uses a clean visible subset of this rank table:
# red -> yellow -> green -> cyan -> blue
# The full table provides stable fallback ordering for any non-zero ARC color.
INPUT_COLOR_ORDER: tuple[int, ...] = (2, 4, 3, 8, 1, 6, 7, 9, 5)
INPUT_RANK = {color: idx for idx, color in enumerate(INPUT_COLOR_ORDER)}

# Fixed output palette, distinct from the generator's input palette.
OUTPUT_PALETTE: tuple[int, ...] = (7, 6, 9, 5)

# Clean visible input chain used by the generator.
GEN_INPUT_CHAIN: tuple[int, ...] = (2, 4, 3, 8, 1)


def _parse_runs_bottom_to_top(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return [(color, height), ...] in bottom->top spatial order."""
    g = np.array(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        raise ValueError("grid must be a non-empty 2D array")

    h, _w = g.shape
    runs: list[tuple[int, int]] = []
    r = h - 1
    while r >= 0:
        color = int(g[r, 0])
        if color == 0:
            raise ValueError("input stripes must be non-zero colors")
        if not bool((g[r] == color).all()):
            raise ValueError("each stripe row must be uniform across the full width")
        top = r
        while top - 1 >= 0:
            if not bool((g[top - 1] == int(g[top - 1, 0])).all()):
                raise ValueError("each stripe row must be uniform across the full width")
            if int(g[top - 1, 0]) != color:
                break
            top -= 1
        runs.append((color, int(r - top + 1)))
        r = top - 1

    if not runs:
        raise ValueError("expected at least one stripe")
    return runs


def _sort_runs_by_visible_color(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Reorder parsed runs by the fixed visible color ranking."""
    return sorted(runs, key=lambda item: (INPUT_RANK.get(int(item[0]), 999), int(item[0])))


def _frame_specs(sorted_runs: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    """
    Return per-layer geometry in final coordinates.

    Each tuple is:
    - layer_index from inner to outer
    - top/left offset
    - outer side length of this layer
    - thickness for this layer
    """
    heights = [int(height) for _color, height in sorted_runs]
    sides = [heights[0]]
    for thick in heights[1:]:
        sides.append(int(sides[-1] + 2 * thick))

    offsets = [0] * len(sorted_runs)
    acc = 0
    for idx in range(len(sorted_runs) - 1, -1, -1):
        offsets[idx] = int(acc)
        if idx > 0:
            acc += int(heights[idx])

    return [
        (idx, int(offsets[idx]), int(sides[idx]), int(heights[idx]))
        for idx in range(len(sorted_runs))
    ]


def _palette_color(start_idx: int, step: int) -> int:
    return int(OUTPUT_PALETTE[(start_idx + step) % len(OUTPUT_PALETTE)])


def _paint_layer(out: np.ndarray, top: int, side: int, thick: int, start_idx: int, is_center: bool) -> None:
    """Paint one layer using a clockwise palette cycle on its sides."""
    if is_center:
        out[top : top + side, top : top + side] = _palette_color(start_idx, 0)
        return

    bottom = int(top + side)
    inner_top = int(top + thick)
    inner_bottom = int(bottom - thick)

    # top, right, bottom, left in clockwise order
    out[top:inner_top, top:bottom] = _palette_color(start_idx, 0)
    out[inner_top:inner_bottom, inner_bottom:bottom] = _palette_color(start_idx, 1)
    out[inner_bottom:bottom, top:bottom] = _palette_color(start_idx, 2)
    out[inner_top:inner_bottom, top:inner_top] = _palette_color(start_idx, 3)


def _render_sorted_runs(sorted_runs: list[tuple[int, int]], count_code: int) -> np.ndarray:
    """Build the nested output before the final rotation."""
    specs = _frame_specs(sorted_runs)
    final_side = int(specs[-1][2])
    out = np.zeros((final_side, final_side), dtype=int)

    for layer_idx, top, side, thick in specs:
        start_idx = int((count_code + layer_idx) % len(OUTPUT_PALETTE))
        _paint_layer(out, int(top), int(side), int(thick), int(start_idx), is_center=(layer_idx == 0))

    return out


def _baseline_without_count_code(sorted_runs: list[tuple[int, int]]) -> np.ndarray:
    """Same geometry and palette rule, but with count_mod4 disabled."""
    return _render_sorted_runs(sorted_runs, count_code=0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    1. Parse full-width stripes bottom -> top.
    2. Reorder them by a fixed visible color ranking.
    3. Keep stripe heights as the nested-square geometry code.
    4. Recolor every layer from a fixed 4-color clockwise palette cycle.
       Layer i starts at palette index (n_stripes mod 4 + i) mod 4.
    5. Rotate the finished output by 90 degrees * (n_stripes mod 4).
    """
    runs = _parse_runs_bottom_to_top(np.array(grid, dtype=int))
    sorted_runs = _sort_runs_by_visible_color(runs)
    count_code = int(len(sorted_runs) % 4)
    out = _render_sorted_runs(sorted_runs, count_code=count_code)
    return np.rot90(out, k=count_code)


def _inversion_count(seq: list[int]) -> int:
    total = 0
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            total += int(seq[i] > seq[j])
    return int(total)


def _gen_attempt() -> np.ndarray | None:
    width = int(np.random.randint(7, 16))
    n_runs = int(np.random.choice([3, 5]))

    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    if int(2 * sum(heights) - 1) > 30:
        return None

    sorted_colors = [int(c) for c in np.random.choice(GEN_INPUT_CHAIN, size=n_runs, replace=False)]
    sorted_colors.sort(key=lambda c: INPUT_RANK[c])

    perm = [int(x) for x in np.random.permutation(n_runs)]
    if perm == list(range(n_runs)):
        return None

    colors_bottom_to_top = [int(sorted_colors[idx]) for idx in perm]
    ranks_bottom_to_top = [int(INPUT_RANK[c]) for c in colors_bottom_to_top]
    if _inversion_count(ranks_bottom_to_top) == 0:
        return None

    stripes_top_to_bottom = [
        np.full((int(height), width), int(color), dtype=int)
        for color, height in reversed(list(zip(colors_bottom_to_top, heights)))
    ]
    inp = np.vstack(stripes_top_to_bottom)

    runs = _parse_runs_bottom_to_top(inp)
    sorted_runs = _sort_runs_by_visible_color(runs)
    actual_final_side = int(_frame_specs(sorted_runs)[-1][2])
    if actual_final_side > 30:
        return None
    actual = main(inp)
    if np.array_equal(actual, _baseline_without_count_code(sorted_runs)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
