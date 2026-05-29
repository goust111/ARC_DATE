# Combo variant of 45a5af55 (index 13)
# combo_spec: D8/1+D9/1
# Name: 色值排序旋转嵌套

from __future__ import annotations

import numpy as np

"""
Design analysis for the fused variant.

a) Concrete behavior trace
   Example bottom->top runs: [(5, 2), (1, 1), (3, 2)].
   Original task:
   - read bottom->top literally
   - center = color 5, size 2
   - wrap 1-thick frame of color 1
   - wrap 2-thick frame of color 3
   Fused task:
   - same runs are first sorted by color value -> [(1, 1), (3, 2), (5, 2)]
   - center = color 1, size 1
   - wrap 2-thick frame of color 3
   - wrap 2-thick frame of color 5
   - rotate by 90 * (3 mod 4) = 270 degrees
   Difference: the nesting order no longer comes from vertical position; it comes
   from the visible palette ordering chosen by the generator.

b) Complexity calibration
   The generator keeps the source scene simple: only full-width uniform stripes.
   The extra reasoning load comes from two coupled latent variables:
   - ordering key = color rank, not spatial order
   - rotation parameter = number of stripes mod 4
   Heights still control frame thickness exactly as in the base task, so the task
   is harder but not structurally noisy.

c) Implicit assumptions review
   The base solver assumed:
   - every row is uniform across the full width
   - consecutive stripes have different colors
   - stripe order bottom->top equals nesting order
   In the combo, the first two assumptions still hold; the third is intentionally
   broken. main() therefore parses runs exactly as before, but sorts the parsed
   runs by color before constructing the nested square.

d) Placement legality
   The generator emits only stacked full-width stripes, so there is no overlap or
   out-of-bounds placement issue inside the input. Output size is determined from
   stripe heights and is capped at <= 30 cells per side.

e) Visual observability
   Sorting by raw ARC ids would be human-invisible if arbitrary colors were used.
   To avoid that, the generator restricts colors to monotone-luminance subsets of
   the ARC palette (notably subsets of [1, 2, 3, 5, 8]), so "smaller color id"
   also looks roughly darker/earlier in a visible brightness progression.

f) Visual cleanliness
   The input contains only contiguous stripes and the output only concentric square
   frames, so objects never overlap ambiguously.

g) Inferrability caveat
   A mathematical limitation remains: a concentric square nest is invariant under
   90-degree rotations, so D9 is a latent rule here. The implementation still
   applies the specified rotation exactly, and the generator spans multiple stripe
   counts, but the visible difference from the base task comes primarily from D8.
"""


# Strictly increasing by ARC id and also by approximate luminance, so the solver's
# color sort corresponds to a visible dark->light progression instead of a hidden
# numeric trick.
VISIBLE_COLOR_CHAIN: tuple[int, ...] = (1, 2, 3, 5, 8)


def _parse_runs_bottom_to_top(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return [(color, height), ...] ordered bottom -> top."""
    g = np.array(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")

    h, w = g.shape
    if h == 0 or w == 0:
        raise ValueError("grid must be non-empty")

    runs: list[tuple[int, int]] = []
    r = h - 1
    while r >= 0:
        color = int(g[r, 0])
        if not bool((g[r] == color).all()):
            raise ValueError("each stripe row must be uniform across the full width")
        height = 1
        rr = r - 1
        while rr >= 0:
            if not bool((g[rr] == int(g[rr, 0])).all()):
                raise ValueError("each stripe row must be uniform across the full width")
            if int(g[rr, 0]) != color:
                break
            height += 1
            rr -= 1
        runs.append((color, height))
        r = rr

    if not runs:
        raise ValueError("expected at least one stripe")
    return runs


def _build_nested_square(runs_by_layer: list[tuple[int, int]]) -> np.ndarray:
    """Build the standard nested-square output from inner -> outer layers."""
    center_color, center_size = runs_by_layer[0]
    cur = np.full((center_size, center_size), center_color, dtype=int)

    for color, thickness in runs_by_layer[1:]:
        new_side = int(cur.shape[0] + 2 * thickness)
        nxt = np.full((new_side, new_side), color, dtype=int)
        nxt[thickness : thickness + cur.shape[0], thickness : thickness + cur.shape[1]] = cur
        cur = nxt

    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    1. Parse full-width horizontal stripes from bottom to top.
    2. Ignore that spatial order for nesting; instead sort stripes by color value.
       Smallest color becomes the center, largest becomes the outermost frame.
    3. Use stripe heights exactly as in the base task: center side = inner height,
       later stripe heights = frame thicknesses.
    4. Rotate the completed result by 90 degrees * (number_of_stripes mod 4).
       np.rot90 uses counterclockwise steps; for concentric squares this is
       geometrically equivalent to any 90-degree convention.
    """
    runs_bottom_to_top = _parse_runs_bottom_to_top(np.array(grid, dtype=int))
    runs_by_color = sorted(runs_bottom_to_top, key=lambda item: int(item[0]))
    out = _build_nested_square(runs_by_color)
    return np.rot90(out, k=len(runs_bottom_to_top) % 4)


def _gen_attempt() -> np.ndarray | None:
    n_runs = int(np.random.randint(2, 6))
    color_idx = np.sort(np.random.choice(len(VISIBLE_COLOR_CHAIN), size=n_runs, replace=False))
    colors_sorted = [int(VISIBLE_COLOR_CHAIN[i]) for i in color_idx]

    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    center_height = heights[0]
    if center_height + 2 * sum(heights[1:]) > 30:
        return None

    # Shuffle bottom->top order so the combo does not collapse back to the base task.
    perm = np.random.permutation(n_runs)
    if list(perm) == list(range(n_runs)):
        return None
    colors_bottom_to_top = [colors_sorted[int(i)] for i in perm]
    heights_bottom_to_top = [heights[int(i)] for i in perm]

    width = int(np.random.randint(6, 15))
    stripes_top_to_bottom = [
        np.full((height, width), color, dtype=int)
        for color, height in reversed(list(zip(colors_bottom_to_top, heights_bottom_to_top)))
    ]
    return np.vstack(stripes_top_to_bottom)


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
