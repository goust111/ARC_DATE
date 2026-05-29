# Variant of 45a5af55
# Dimension: D10 — 符号编码方式 (symbol_instructions)
# Change: 色条高度直接编码框架厚度，色条颜色直接编码框架颜色（隐式编码，无显式图例）... → 色条中嵌入标记像素编码额外参数
from __future__ import annotations

"""
Design analysis for the D10 control-pixel variant.

a) Concrete behavior and trace
   - Original input: each horizontal stripe is perfectly uniform, so height alone says
     how thick the corresponding output square layer is.
   - Variant input: each stripe is still full-width and still keeps its own color/height,
     but now each stripe contains exactly one control pixel of a dedicated marker color.
     The control pixel's column encodes one extra instruction for that stripe:
       * left third   -> plain layer (original behavior)
       * middle third -> after wrapping, paint a centered vertical bar of that layer color
       * right third  -> after wrapping, paint a centered horizontal bar of that layer color
   - Small trace:
       bottom stripe = color 2, height 2, marker in left third
       upper stripe  = color 7, height 1, marker in middle third
     Original logic:
       1. make 2x2 solid block of color 2
       2. wrap one-cell color-7 frame around it => 4x4 square frame with 2x2 center
     Variant logic:
       1. same 2x2 solid block of color 2
       2. same wrap to 4x4, then marker says "vertical bar", so paint the centered
          1-column bar of color 7 through the full 4x4 square
     Difference:
       the outer layer no longer depends only on height/color; the marker column changes
       the internal pattern of that layer while preserving the nested-square scaffold.

b) Complexity calibration
   - The new control has 3 modes, not an unconstrained position->shape mapping.
   - Three bins are enough to be visually obvious, but still simple enough to infer from
     2-4 ARC examples.
   - We keep all other structure stable: full-width stripes, bottom-to-top reading order,
     square growth, and stripe height = layer thickness.

c) Implicit assumptions review against the original main()
   - Original `col = g[r, 0]` assumed every row in a stripe is uniform. That breaks once a
     marker can appear anywhere in the row. Fix: use the majority color of each row.
   - Original `while (g[rr] == col).all()` assumed exact row equality across the stripe.
     That also breaks on the marker row. Fix: group rows by majority color instead.
   - Original runs stored only `(color, height)`. Variant needs marker position, so runs
     become `(color, height, mode)`.
   - Original wrapping logic is still valid as the base layer constructor; the variant only
     adds a post-wrap overlay step for the extra instruction.

d) Placement legality
   - Generator still stacks stripes flush with no gaps and no overlap.
   - Each stripe gets exactly one marker pixel, chosen inside that stripe's rectangle.
   - Width is constrained to at least 9 so all three control bins are non-empty.
   - Total output side length remains bounded exactly as before:
     `inner_height + 2 * sum(outer_heights) <= 30`.

e) Visual observability
   - The marker pixels all share one dedicated control color, so humans can identify them as
     a symbolic layer rather than part of the stripe colors.
   - The resulting vertical/horizontal bars are large, axis-aligned, and easy to compare
     across examples.
   - The generator forces non-plain outer layers, and usually includes both vertical and
     horizontal cases, so the position semantics are repeatedly supported.

f) Visual tidiness
   - The only overlaid structure is a centered bar within an already square layer.
   - No extra objects are scattered around the grid; the input remains a clean stripe stack.
   - Output stays compact and readable because bars align with the square centerlines.

g) Color-change inferability
   - Marker color itself carries no per-layer meaning beyond "this is a control pixel".
   - Layer color still comes from the stripe color, exactly as in the original task.
   - The only new thing to infer is how marker column bucket changes the same-colored layer.

h) Variant behavior metric
   - A generated example is considered meaningfully variant if at least one outer layer uses
     mode 1 or mode 2, because then the output differs from the original solver on the same
     input. The generator enforces that condition.

i) New solver assumptions and generator guarantees
   - Solver assumes each stripe has a unique majority color and exactly one off-color pixel.
     Generator enforces unique stripe colors and inserts one marker per stripe.
   - Solver assumes all rows belonging to one stripe share the same majority color.
     Generator never places more than one marker inside a stripe, so this remains true.
   - Solver assumes width >= 3 for a stable row majority; generator uses width >= 9.
"""

import numpy as np


def _mode_from_col(col: int, width: int) -> int:
    left_cut = width // 3
    right_cut = (2 * width) // 3
    if col < left_cut:
        return 0
    if col < right_cut:
        return 1
    return 2


def _majority_color(row: np.ndarray) -> int:
    vals, counts = np.unique(row, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _parse_runs(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    row_major = [_majority_color(g[r]) for r in range(H)]

    runs: list[tuple[int, int, int]] = []
    r = H - 1
    while r >= 0:
        stripe_color = int(row_major[r])
        top = r
        while top - 1 >= 0 and int(row_major[top - 1]) == int(stripe_color):
            top -= 1

        stripe = g[top : r + 1]
        marker_pos = np.argwhere(stripe != int(stripe_color))
        assert len(marker_pos) == 1
        marker_c = int(marker_pos[0, 1])
        mode = _mode_from_col(marker_c, g.shape[1])

        runs.append((int(stripe_color), int(r - top + 1), int(mode)))
        r = top - 1

    assert runs
    return runs


def _apply_mode(square: np.ndarray, color: int, thickness: int, mode: int) -> None:
    n = int(square.shape[0])
    start = int((n - thickness) // 2)
    stop = int(start + thickness)

    if mode == 1:
        square[:, start:stop] = int(color)
    elif mode == 2:
        square[start:stop, :] = int(color)


def _render_runs(runs: list[tuple[int, int, int]]) -> np.ndarray:
    color0, height0, mode0 = runs[0]
    cur = np.full((int(height0), int(height0)), int(color0), dtype=int)
    _apply_mode(cur, int(color0), int(height0), int(mode0))

    for color, height, mode in runs[1:]:
        height = int(height)
        new_n = int(cur.shape[0] + 2 * height)
        nxt = np.full((new_n, new_n), int(color), dtype=int)
        nxt[height : height + cur.shape[0], height : height + cur.shape[1]] = cur
        _apply_mode(nxt, int(color), int(height), int(mode))
        cur = nxt

    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square layers from horizontal stripes with per-stripe control pixels.

    Base rule from 45a5af55:
    - Read horizontal stripes bottom to top.
    - Stripe height is the square/frame thickness.
    - Stripe color is the layer color.

    D10 change:
    - Every stripe contains one marker pixel of a dedicated control color.
    - The marker column chooses an extra layer instruction:
      left third = plain layer, middle third = centered vertical bar,
      right third = centered horizontal bar.
    """
    runs = _parse_runs(np.array(grid, dtype=int))
    return _render_runs(runs)


def _bucket_bounds(width: int, mode: int) -> tuple[int, int]:
    left_cut = width // 3
    right_cut = (2 * width) // 3
    if mode == 0:
        return 0, left_cut
    if mode == 1:
        return left_cut, right_cut
    return right_cut, width


def _gen_attempt() -> np.ndarray | None:
    width = int(np.random.randint(9, 16))
    n_runs = int(np.random.randint(3, 7))

    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    if int(heights[0] + 2 * sum(heights[1:])) > 30:
        return None

    palette = list(range(10))
    marker_color = int(np.random.choice(palette))
    stripe_palette = [c for c in palette if c != marker_color]
    stripe_colors = [int(c) for c in np.random.choice(stripe_palette, size=n_runs, replace=False)]

    outer_modes = [1, 2]
    if n_runs - 1 >= 3:
        outer_modes.append(0)
    while len(outer_modes) < n_runs - 1:
        outer_modes.append(int(np.random.randint(0, 3)))
    np.random.shuffle(outer_modes)
    modes = [int(np.random.randint(0, 3))] + [int(m) for m in outer_modes]

    runs: list[tuple[int, int, int]] = [
        (int(color), int(height), int(mode))
        for color, height, mode in zip(stripe_colors, heights, modes)
    ]

    stripe_blocks: list[np.ndarray] = []
    for color, height, mode in reversed(runs):
        block = np.full((int(height), width), int(color), dtype=int)
        row = int(np.random.randint(0, int(height)))
        lo, hi = _bucket_bounds(width, int(mode))
        assert lo < hi
        col = int(np.random.randint(lo, hi))
        block[row, col] = int(marker_color)
        stripe_blocks.append(block)

    inp = np.vstack(stripe_blocks)
    expected = _render_runs(runs)
    assert np.array_equal(main(inp), expected)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
