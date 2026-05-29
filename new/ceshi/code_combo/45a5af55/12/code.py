# Combo variant of 45a5af55 (index 12)
# combo_spec: D7/1+D9/2
# Name: 带边距对称修剪嵌套

from __future__ import annotations

import numpy as np

"""
Design analysis for the fused variant
=====================================

a) Concrete behavior and traced example
---------------------------------------
Take bottom->top runs [(2, 2), (5, 1), (7, 2)].

Original task:
- color 2, height 2 -> 2x2 solid center
- wrap a 1-thick full square frame of color 5 -> 4x4
- wrap a 2-thick full square frame of color 7 -> 8x8
- return that full 8x8 nested square directly

Fused task:
1) Stage-1 construction is still exactly the original 8x8 full nested square.
2) Stage-2 symmetric pruning acts on each true outer frame separately.
   - For a frame of thickness t, each of its four outer corners loses the cells
     whose local corner-distance satisfies r+c < t, i.e. a 45-degree cut.
   - With t=1 this removes only the single corner pixel.
   - With t=2 this removes a 3-cell right triangle from each corner.
   - The center square is not a frame, so it stays solid.
3) Stage-3 places the pruned nested figure inside a 1-cell background border.
   Therefore the final output here is 10x10: the inner 8x8 octagonal nest plus
   one background row/column on each side.

Difference from the base task:
- D7 changes the output size by adding a fixed visible background margin.
- D9 changes the geometry of every outer layer from square-ring corners to
  symmetrically clipped corners, producing octagonal-looking rings.
- These two changes interact: the background border makes the corner clipping
  visible all the way to the outer silhouette instead of being flush to the edge.

b) Complexity calibration
-------------------------
- The generator uses 3-6 stripes so there are at least two pruned outer layers.
- Outer heights stay in 1-4 as in the base task, but at least one outer height is
  forced to be >= 2 so the diagonal cuts are visually obvious in most samples.
- The border width is fixed to 1 for the whole task. That is a small discrete rule
  humans can infer from few examples without introducing many free parameters.

c) Implicit-assumption review of the original main()
----------------------------------------------------
- Parsing by `g[r, 0]` and row-uniformity still works because the input remains a
  pure stack of full-width monochrome stripes with no input background.
- The original `return cur` encoded the assumption that the stripe-derived square
  itself was the final canvas. D7 breaks that: we now need an outer background
  canvas of side `nested_side + 2 * margin`.
- The original wrapping step `nxt = full(color)` assumed each outer layer remains
  a complete square frame in the final answer. D9 breaks that: frames must be
  built first, then selectively clipped at all four corners.
- A naive "clip the final filled square in-place" is risky because it mixes layer
  geometry. The fused solver instead reconstructs each layer in final coordinates,
  so clipping only touches the intended annulus of that layer.

d) Placement legality
---------------------
- Input legality is unchanged from the base task: one contiguous stripe stack,
  full-width rows, no overlap, no background holes.
- Output legality follows from exact geometry:
  - inner nested side = h0 + 2 * sum(other heights)
  - output side = inner side + 2
  - every layer is painted inside its own bounding square, offset by the outer
    layers plus the fixed border, so nothing goes out of bounds or overlaps the
    wrong layer.
- The generator rejects cases where the final output side would exceed 30.

e) Visual observability
-----------------------
- A human can still see that stripe height controls layer thickness, because the
  offsets between layers are unchanged from the base task.
- The new border is obvious: output is always larger than the unclipped nested
  figure by exactly one background cell on every side.
- The new pruning is also obvious: all four corners of every outer ring are cut in
  the same symmetric 45-degree way, so the silhouette is no longer a perfect
  square. This is much easier to infer than an asymmetric corner deletion.

f) Visual cleanliness
---------------------
- The final grid has only three region types:
  1) a solid center square
  2) pruned octagonal ring layers
  3) uniform background border and corner-cut holes
- Because each ring is painted only on its annulus, clipped corners never intrude
  into inner colors, and there is no ambiguous overlap.

g) Color-change inferability
----------------------------
- No color remapping is introduced.
- Each stripe color still becomes the corresponding nested layer color.
- Background 0 appears only as output padding / removed corner cells, so its role
  is structural and easy to infer from the geometry alone.

Post-change assumptions introduced by the new solver
----------------------------------------------------
- Input rows must remain monochrome across the full width.
- The input must contain at least one stripe, and the generator guarantees 3-6.
- Colors are sampled from 1..9 so the output background 0 stays distinct.
- Corner clipping assumes positive outer thicknesses; the generator uses heights
  in 1..4 and only clips true outer frames, never the center layer.
"""


BG = 0
MARGIN = 1


def _parse_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return bottom->top (color, height) runs from a full-width stripe stack."""
    g = np.array(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")
    h, w = g.shape
    if h <= 0 or w <= 0:
        raise ValueError("grid must be non-empty")

    runs: list[tuple[int, int]] = []
    r = h - 1
    while r >= 0:
        color = int(g[r, 0])
        if not bool((g[r] == color).all()):
            raise ValueError("each input row must be monochrome")
        run_h = 1
        rr = r - 1
        while rr >= 0:
            if not bool((g[rr] == int(g[rr, 0])).all()):
                raise ValueError("each input row must be monochrome")
            if int(g[rr, 0]) != color:
                break
            run_h += 1
            rr -= 1
        runs.append((color, run_h))
        r = rr

    if not runs:
        raise ValueError("expected at least one stripe")
    return runs


def _build_full_nested(runs: list[tuple[int, int]]) -> np.ndarray:
    """Base-task stage: build the full nested square with no border and no pruning."""
    center_color, center_side = runs[0]
    cur = np.full((center_side, center_side), center_color, dtype=int)
    for color, thick in runs[1:]:
        new_side = int(cur.shape[0] + 2 * thick)
        nxt = np.full((new_side, new_side), color, dtype=int)
        nxt[thick : thick + cur.shape[0], thick : thick + cur.shape[1]] = cur
        cur = nxt
    return cur


def _layer_specs(runs: list[tuple[int, int]]) -> list[tuple[int, int, int, int, bool]]:
    """
    Per-layer geometry in coordinates of the unclipped nested square.

    Returns tuples:
    - color
    - top/left offset of the layer's outer square
    - outer side length of that layer
    - thickness (for center, this equals the center side but is not used for clipping)
    - whether this is the center layer
    """
    heights = [height for _color, height in runs]
    sides = [heights[0]]
    for thick in heights[1:]:
        sides.append(sides[-1] + 2 * thick)

    offsets = [0] * len(runs)
    acc = 0
    for idx in range(len(runs) - 1, -1, -1):
        offsets[idx] = acc
        if idx > 0:
            acc += heights[idx]

    return [
        (runs[idx][0], offsets[idx], sides[idx], heights[idx], idx == 0)
        for idx in range(len(runs))
    ]


def _corner_keep_mask(side: int, thick: int) -> np.ndarray:
    """Mask for a square after removing a 45-degree corner triangle of depth `thick` at each outer corner."""
    rr, cc = np.indices((side, side))
    cut_tl = rr + cc < thick
    cut_tr = rr + (side - 1 - cc) < thick
    cut_bl = (side - 1 - rr) + cc < thick
    cut_br = (side - 1 - rr) + (side - 1 - cc) < thick
    return ~(cut_tl | cut_tr | cut_bl | cut_br)


def _build_pruned_with_margin(runs: list[tuple[int, int]]) -> np.ndarray:
    """
    Fused output:
    - build the original nested-square layers
    - prune each true outer frame symmetrically at the four corners
    - place the result inside a fixed 1-cell background margin
    """
    specs = _layer_specs(runs)
    inner_side = int(specs[-1][2])
    out_side = int(inner_side + 2 * MARGIN)
    out = np.full((out_side, out_side), BG, dtype=int)

    for color, offset, side, thick, is_center in specs:
        top = int(MARGIN + offset)
        left = int(MARGIN + offset)
        block = out[top : top + side, left : left + side]

        if is_center:
            block[:, :] = int(color)
            continue

        annulus = np.ones((side, side), dtype=bool)
        annulus[thick : side - thick, thick : side - thick] = False
        keep = annulus & _corner_keep_mask(side, thick)
        block[keep] = int(color)

    return out


def _expected_output_from_runs(runs: list[tuple[int, int]]) -> np.ndarray:
    """Generator-side expectation helper."""
    return _build_pruned_with_margin(runs)


def _full_with_margin(runs: list[tuple[int, int]]) -> np.ndarray:
    """D7-only reference: nested square with border but without symmetric pruning."""
    full = _build_full_nested(runs)
    out = np.full((full.shape[0] + 2 * MARGIN, full.shape[1] + 2 * MARGIN), BG, dtype=int)
    out[MARGIN : MARGIN + full.shape[0], MARGIN : MARGIN + full.shape[1]] = full
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve the fused variant:
    1) parse bottom->top full-width stripes
    2) build the same nested-square layer geometry as the original task
    3) clip the four corners of each true frame by 45-degree cuts
    4) embed the result in a 1-cell background border
    """
    runs = _parse_runs(np.array(grid, dtype=int))
    return _build_pruned_with_margin(runs)


def _gen_attempt() -> np.ndarray | None:
    width = int(np.random.randint(6, 16))
    n_runs = int(np.random.randint(3, 7))
    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]

    # Make the octagonal pruning visible: at least one outer layer must have a
    # diagonal cut deeper than one pixel.
    if max(heights[1:]) < 2:
        return None

    inner_side = int(heights[0] + 2 * sum(heights[1:]))
    if inner_side + 2 * MARGIN > 30:
        return None

    colors: list[int] = []
    used = {BG}
    for _ in range(n_runs):
        color = int(np.random.randint(1, 10))
        while color in used:
            color = int(np.random.randint(1, 10))
        colors.append(color)
        used.add(color)

    runs = list(zip(colors, heights))
    inp = np.vstack(
        [
            np.full((height, width), color, dtype=int)
            for color, height in reversed(runs)
        ]
    )

    actual = main(inp)
    expected = _expected_output_from_runs(runs)
    if not np.array_equal(actual, expected):
        return None

    full_with_margin = _full_with_margin(runs)
    pruned_cells = int(np.count_nonzero((full_with_margin != BG) & (actual == BG)))
    if pruned_cells <= 0:
        return None

    # Avoid accidental collapse to alternate hypotheses such as "output side =
    # input width" or "just add a margin without pruning".
    if width == actual.shape[1]:
        return None
    if np.array_equal(actual, full_with_margin):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
