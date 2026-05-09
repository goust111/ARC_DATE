# Variant of 45a5af55
# Dimension: D3 — 色条高度到框架厚度的映射 (transform)
# Change: 框架厚度 = 色条高度（直接等值映射）... → 框架厚度统一为1，色条高度编码重复次数

from __future__ import annotations

import numpy as np

# Design analysis
# a) Concrete behavior / trace:
#    Input is still a stack of full-width uniform horizontal stripes read from bottom to top.
#    In the original task, an outer stripe of height h became one solid square frame of thickness h.
#    That makes height and visible thickness identical.
#    In this variant, every outer stripe with height h means "repeat this color h times as 1-pixel frames".
#    A naive implementation would be visually identical to the original, because h adjacent frames of the
#    same color merge into one thick band. To keep the D3 change observable, repeated frames of the same
#    stripe are separated by a 1-pixel background (0) gap:
#      h=1 -> one color ring
#      h=2 -> color ring, 0-gap, color ring
#      h=3 -> color ring, 0-gap, color ring, 0-gap, color ring
#    The bottom stripe still provides the center solid square with side length = its height; the D3 change
#    is only about how outer stripes expand around the current shape.
#    Example: bottom (2,2), then outer (5,3)
#      original: 2x2 center -> one 8x8 frame of color 5 with thickness 3
#      variant : 2x2 center -> 1px color-5 frame, 1px 0-gap, 1px color-5 frame, 1px 0-gap,
#                1px color-5 frame, so the total shell is 5 cells per side instead of 3, and the
#                repeated count is visible.
#
# b) Complexity calibration:
#    Heights remain small (1..4), run counts remain small (2..5), and colors are distinct within a sample.
#    This keeps the puzzle readable while ensuring repeated rings appear often enough to reveal the new rule.
#
# c) Implicit assumptions in the original main():
#    - `col = g[r, 0]`: assumes each row is a uniform full-width stripe. Still required.
#    - `while (g[rr] == col).all()`: assumes contiguous same-color rows encode stripe height. Still valid.
#    - `cur = full((h0, h0), c0)`: assumes bottom stripe creates the center square. Kept unchanged.
#    - `new_n = cur.shape[0] + 2*h` plus one fill/write: assumes height directly equals frame thickness.
#      This is the line that breaks under the variant; it must be replaced by repeated 1-pixel wraps with
#      explicit 0-gaps between wraps of the same stripe.
#
# d) Placement legality:
#    The generator only creates vertical stacks of full-width stripes, so there is no overlap issue in input.
#    Output construction is purely nested and centered, so it cannot self-overlap incorrectly.
#
# e) Visual observability:
#    The generator forces at least one outer stripe to have height > 1, so at least one sample feature cannot
#    collapse back to the original thick-band behavior.
#
# f) Visual cleanliness:
#    Only three element types appear in output: solid center square, 1-pixel colored rings, and 0-color gaps.
#    The gaps are regular and centered, so the images stay orderly rather than noisy.
#
# g) Color inference:
#    Colors are preserved directly from input stripes; no color remapping is introduced.
#
# 9) New assumptions after modification:
#    - Background color 0 is reserved for the separator gaps in output; generator therefore uses input colors 1..9.
#    - Solver expects the input to contain only uniform full-width stripes with no background rows; generator
#      guarantees that structure.


def _parse_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    """Parse full-width stripe runs from bottom to top as (color, height)."""
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    runs: list[tuple[int, int]] = []
    r = h - 1
    while r >= 0:
        row = g[r]
        col = int(row[0])
        assert col != 0
        assert bool((row == col).all())
        run_h = 1
        rr = r - 1
        while rr >= 0 and bool((g[rr] == col).all()):
            run_h += 1
            rr -= 1
        runs.append((col, run_h))
        r = rr
    assert runs
    return runs


def _wrap_repeated_unit_frames(cur: np.ndarray, color: int, repeats: int) -> np.ndarray:
    """
    Add `repeats` copies of a 1-pixel frame in `color`.

    To make repeated same-color frames visible, consecutive copies are separated by a 1-pixel 0-color gap.
    """
    out = np.array(cur, dtype=int)
    for i in range(int(repeats)):
        colored = np.full((out.shape[0] + 2, out.shape[1] + 2), int(color), dtype=int)
        colored[1:-1, 1:-1] = out
        out = colored
        if i != repeats - 1:
            gapped = np.zeros((out.shape[0] + 2, out.shape[1] + 2), dtype=int)
            gapped[1:-1, 1:-1] = out
            out = gapped
    return out


def _build_output_from_runs(runs: list[tuple[int, int]]) -> np.ndarray:
    center_color, center_size = runs[0]
    cur = np.full((int(center_size), int(center_size)), int(center_color), dtype=int)
    for color, repeats in runs[1:]:
        cur = _wrap_repeated_unit_frames(cur, int(color), int(repeats))
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square structures from bottom-to-top horizontal stripes.

    Variant rule:
    - Bottom stripe still becomes the center solid square of side = stripe height.
    - Every later stripe of height h contributes h separate 1-pixel frames in that color.
    - Repeated frames from the same stripe are separated by 1-pixel 0-color gaps so the repetition count is visible.
    """
    runs = _parse_runs(np.array(grid, dtype=int))
    return _build_output_from_runs(runs)


def _gen_attempt() -> np.ndarray | None:
    width = int(np.random.randint(6, 15))
    n_runs = int(np.random.randint(2, 6))

    center_h = int(np.random.randint(1, 5))
    outer_heights = [int(np.random.randint(1, 5)) for _ in range(n_runs - 1)]
    if all(h == 1 for h in outer_heights):
        outer_heights[int(np.random.randint(0, len(outer_heights)))] = int(np.random.randint(2, 5))
    heights = [center_h, *outer_heights]

    output_size = int(center_h + sum(4 * h - 2 for h in outer_heights))
    if output_size > 30:
        return None

    colors = np.random.choice(np.arange(1, 10), size=n_runs, replace=False).astype(int).tolist()
    runs = list(zip(colors, heights))
    stripes = [
        np.full((int(run_h), width), int(color), dtype=int)
        for color, run_h in reversed(runs)
    ]
    inp = np.vstack(stripes)

    expected = _build_output_from_runs(runs)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
