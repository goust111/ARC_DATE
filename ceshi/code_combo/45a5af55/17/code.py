# Combo variant of 45a5af55 (index 17)
# combo_spec: D1/1+D5/0+D13/0
# Name: 竖直分隔色条左读嵌套

from __future__ import annotations

import numpy as np

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    Original 45a5af55 reads full-width horizontal stripes from bottom to top. Here the source
#    objects are full-height vertical bars, and neighboring bars are separated by 1-2 columns of
#    background color 0. The solver must ignore those separator columns and read only the
#    non-zero bars from left to right.
#    Example widths/colors/gaps left->right:
#      [(2, red), gap 1, (1, blue), gap 2, (3, green)]
#    Original logic would inspect rows and fail, because each row is now multi-color and the
#    separators live on the wrong axis. The combo logic instead parses ordered non-zero column
#    runs [(red, 2), (blue, 1), (green, 3)] and builds:
#      2x2 red center -> wrap 1-thick blue frame -> wrap 3-thick green frame.
#    The three dimensions act together: D5 changes the source object family to vertical bars,
#    D13 inserts ignorable background separators between those bars, and D1 changes the nesting
#    readout direction to left->right over the surviving bars.
# b) Complexity calibration:
#    Widths remain in the original 1..4 range and the number of semantic bars remains 2..6,
#    preserving the core nesting difficulty. Gap widths are limited to 1 or 2 so the separator
#    pattern is always visible but not noisy. Every generated sample has at least one separator,
#    so the combo does not collapse into the simpler pure-vertical variant.
# c) Implicit-assumption audit of the original main():
#    - `col = g[r, 0]` assumed a stripe can be identified from a row sample. False now; rows are
#      multi-color, so we must sample columns with `g[0, c]`.
#    - `(g[rr] == col).all()` assumed whole rows are monochrome semantic objects. False now; the
#      relevant monochrome units are columns, so the axis changes to `(g[:, cc] == col).all()`.
#    - The bottom-to-top loop over `r` assumed adjacency along rows and no separator rows. False
#      now on both counts; the parser must sweep columns left->right and skip full-height 0-columns.
#    - The frame-building loop only consumes ordered `(color, thickness)` runs, so it remains
#      valid once parsing is repaired.
# d) Placement legality:
#    The generator concatenates disjoint full-height rectangles horizontally, alternating semantic
#    non-zero bars with 0-colored separator bars. This guarantees no overlap, no out-of-bounds
#    writes, and perfect compatibility with the parser's assumptions.
# e) Visual observability:
#    Across 2-4 examples, a human can see vertical segmentation, regular black separator columns,
#    and outputs that preserve the left-to-right color order as center-to-outside nesting order.
#    Because input height varies independently of semantic widths, it is visible that bar width,
#    not height, controls frame thickness.
# f) Visual cleanliness:
#    The input contains only two object roles: full-height semantic bars and full-height separator
#    bars. There is no clutter, overlap, or partial marking inside bars, so the scene stays tidy.
# g) Change inferability:
#    No color remapping is introduced. Bar colors transfer directly to output layers. Color 0 is
#    reserved for separators and never used as a semantic bar color, so the "ignore these columns"
#    rule is structurally recoverable from examples.
# h) Degeneration check target:
#    A useful metric is `(separator_columns, semantic_bars, monochrome_columns, monochrome_rows)`.
#    For this combo, separator_columns > 0, every column is monochrome, and not every row is
#    monochrome once there are at least two semantic bars. This distinguishes the combo from the
#    original horizontal task and from a no-separator vertical simplification.
# i) New assumptions introduced by this main() and generator guarantees:
#    - Each column is uniform top-to-bottom.
#    - Semantic bars use non-zero colors and separator columns use exactly 0.
#    - Adjacent semantic bars have different colors so bar boundaries are observable.
#    - At least one semantic bar exists after skipping separators.
#    `generate_input()` enforces all of these and self-checks `main(input)` against the intended
#    nested output.


def _build_nested_from_runs(runs: list[tuple[int, int]]) -> np.ndarray:
    assert runs

    center_color, center_width = runs[0]
    cur = np.full((int(center_width), int(center_width)), int(center_color), dtype=int)

    for color, width in runs[1:]:
        width = int(width)
        new_side = int(cur.shape[0] + 2 * width)
        nxt = np.full((new_side, new_side), int(color), dtype=int)
        nxt[width : width + cur.shape[0], width : width + cur.shape[1]] = cur
        cur = nxt

    return cur


def _parse_vertical_runs_with_separators(
    grid: np.ndarray, bg: int = 0
) -> list[tuple[int, int]]:
    g = np.array(grid, dtype=int)
    _h, w = g.shape

    runs: list[tuple[int, int]] = []
    c = 0
    while c < w:
        col = int(g[0, c])
        assert bool((g[:, c] == col).all())

        width = 1
        cc = c + 1
        while cc < w and int(g[0, cc]) == col:
            assert bool((g[:, cc] == col).all())
            width += 1
            cc += 1

        if col != bg:
            runs.append((int(col), int(width)))
        c = int(cc)

    assert runs
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from separated vertical color bars.

    Combo rule:
    - Input consists of full-height vertical bars.
    - Semantic bars are non-zero and may be separated by 1-2 full-height columns of color 0.
    - Read semantic bars from left to right, ignoring separator columns.
    - The leftmost semantic bar becomes the center solid square with side length = bar width.
    - Each later semantic bar wraps the current shape with a same-color square frame whose
      thickness equals that bar's width.
    """
    runs = _parse_vertical_runs_with_separators(np.array(grid, dtype=int), bg=0)
    return _build_nested_from_runs(runs)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    height = int(np.random.randint(6, 16))
    n_runs = int(np.random.randint(2, 7))
    widths = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    gaps = [int(np.random.randint(1, 3)) for _ in range(n_runs - 1)]

    input_width = int(sum(widths) + sum(gaps))
    output_side = int(widths[0] + 2 * sum(widths[1:]))
    if input_width < 6 or input_width > 15 or output_side > 30:
        return None

    colors: list[int] = []
    prev = None
    for _ in range(n_runs):
        color = int(np.random.randint(1, 10))
        while color == bg or (prev is not None and color == prev):
            color = int(np.random.randint(1, 10))
        colors.append(color)
        prev = color

    pieces: list[np.ndarray] = []
    for idx, (color, width) in enumerate(zip(colors, widths)):
        pieces.append(np.full((height, width), int(color), dtype=int))
        if idx < n_runs - 1:
            pieces.append(np.full((height, gaps[idx]), bg, dtype=int))
    grid = np.hstack(pieces)

    expected = _build_nested_from_runs(list(zip(colors, widths)))
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
