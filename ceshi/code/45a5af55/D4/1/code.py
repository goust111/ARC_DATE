# Variant of 45a5af55
# Dimension: D4 — 中心填充方式 (transform)
# Change: 中心为实心正方形（用最底层色条颜色完全填充）... → 中心为棋盘格图案
#
# Design analysis
# - Source-task review: no separate source_task was provided, so the baseline is the
#   original 45a5af55 solver/generator only.
# - a) Concrete behavior trace:
#   Input stays the same: full-width horizontal stripes, read from bottom to top.
#   Original output:
#     bottom stripe (color 7, height 3) -> 3x3 solid center of 7
#     next stripe (color 4, height 1) -> wrap that center in a 1-cell frame of 4
#   Variant output:
#     bottom stripe (color 7, height 3) -> 3x3 checkerboard using 7 and bg color 0:
#       7 0 7
#       0 7 0
#       7 0 7
#     next stripe (color 4, height 1) still wraps the whole current square unchanged.
#   The only pipeline stage that changes is center initialization; stripe parsing and
#   outer frame growth stay identical.
# - b) Complexity calibration:
#   The change is intentionally narrow: one new binary pattern at the center. To keep
#   the pattern visible, the generator forbids bottom-stripe height 1; otherwise a
#   1x1 "checkerboard" would collapse back to the original solid-center behavior.
# - c) Implicit assumption audit for main():
#   `runs` parsing assumes each stripe row is uniform across the full width. This still
#   holds because the input format is unchanged.
#   `cur = np.full((h0, h0), c0, ...)` relied on "center is solid". That assumption no
#   longer holds, so this line must be replaced by explicit checkerboard construction.
#   The frame-wrap loop only assumes `cur` is already a square array, not that it is
#   monochrome, so it remains valid.
# - d) Placement legality:
#   The generator still stacks stripes vertically with no gaps or overlaps. Output size
#   is checked with the same bound as before, so frames never exceed 30x30.
# - e) Visual observability:
#   Every generated sample has center size at least 2, so the solver/user can actually
#   see alternating 0/non-0 cells. Because outer frames remain solid, the changed rule
#   is localized and easy to compare across examples.
# - f) Visual cleanliness:
#   Only the innermost square contains two colors; all outer rings remain solid. This
#   keeps the output legible and avoids clutter.
# - g) Color inference:
#   The new color relationship is structural, not arbitrary: center cells alternate
#   between the bottom stripe color and fixed background color 0 by parity.
# - New assumptions introduced by the variant:
#   main() now assumes the intended background color for the checkerboard is 0.
#   generate_input() guarantees all stripe colors are in 1..9, so 0 is reserved and the
#   checkerboard is unambiguous.

from __future__ import annotations

import numpy as np

BG = 0


def _make_center_checkerboard(color: int, size: int) -> np.ndarray:
    yy, xx = np.indices((size, size))
    return np.where((yy + xx) % 2 == 0, int(color), BG).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from a stack of horizontal color stripes.

    Variant rule:
    - Read stripes bottom to top.
    - Bottom stripe becomes the center square with side length = stripe height,
      filled as a checkerboard alternating between the stripe color and bg color 0.
    - Each subsequent stripe wraps the current shape with a square frame of
      thickness = stripe height, in that stripe's color.
    """
    g = np.array(grid, dtype=int)
    H, _W = g.shape

    # Parse stripe runs from bottom to top.
    runs: list[tuple[int, int]] = []  # (color, height), bottom -> top
    r = int(H - 1)
    while r >= 0:
        col = int(g[r, 0])
        h = 1
        rr = int(r - 1)
        while rr >= 0 and bool((g[rr] == col).all()):
            h += 1
            rr -= 1
        runs.append((col, h))
        r = rr

    assert runs

    c0, h0 = runs[0]
    cur = _make_center_checkerboard(c0, int(h0))

    for col, h in runs[1:]:
        h = int(h)
        new_n = int(cur.shape[0] + 2 * h)
        nxt = np.full((new_n, new_n), int(col), dtype=int)
        nxt[h : h + cur.shape[0], h : h + cur.shape[1]] = cur
        cur = nxt

    return cur


def _gen_attempt() -> np.ndarray | None:
    W = int(np.random.randint(6, 15))
    n_runs = int(np.random.randint(2, 7))
    heights = [int(np.random.randint(2, 5))]
    heights.extend(int(np.random.randint(1, 5)) for _ in range(n_runs - 1))
    if heights[0] + 2 * sum(heights[1:]) > 30:
        return None

    cols: list[int] = []
    prev = None
    for _ in range(n_runs):
        c = int(np.random.randint(1, 10))
        while prev is not None and c == prev:
            c = int(np.random.randint(1, 10))
        cols.append(c)
        prev = c

    stripes = [
        np.full((h, W), col, dtype=int)
        for col, h in reversed(list(zip(cols, heights)))
    ]
    candidate = np.vstack(stripes)

    # Self-check that the variant is actually expressed: the center checkerboard must
    # introduce background cells, which the original solid-center task never would.
    if not np.any(main(candidate) == BG):
        return None
    return candidate


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
