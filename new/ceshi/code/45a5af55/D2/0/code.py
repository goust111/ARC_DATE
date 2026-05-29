# Variant of 45a5af55
# Dimension: D2 — 框架形状 (transform)
# Change: 构建正方形嵌套框架（每层为等厚方形环）... → 构建同心圆形/菱形嵌套框架

from __future__ import annotations

import numpy as np

# Design analysis for the D2 variant
# ==================================
# a) Concrete behavior and trace
# - Input stays the same as the source task: full-width horizontal stripes, read
#   from bottom to top. Each stripe contributes one color and one integer height.
# - The changed part is the output geometry. Instead of nesting axis-aligned
#   squares, we nest Manhattan diamonds (a square rotated by 45 degrees).
# - We interpret stripe height as radial thickness in L1 distance:
#   * bottom stripe of height h0 -> solid diamond with radius h0 - 1
#   * next stripe of height h -> all cells whose Manhattan distance from the
#     center lies in (old_radius, old_radius + h]
# - Small trace:
#   runs bottom->top = [(4, 2), (7, 1), (3, 2)]
#   source task:
#     4 -> 2x2 solid square
#     7 -> wrap 1-cell square frame => 4x4
#     3 -> wrap 2-cell square frame => 8x8
#   this variant:
#     4 -> radius-1 solid diamond (a 3x3 plus shape)
#     7 -> add the radius-2 shell
#     3 -> add radii 3 and 4 shell
#     final bounding box is 9x9 with zero-valued corners outside the outer
#     diamond, so the visual silhouette is unmistakably diamond-shaped.
#
# b) Complexity calibration
# - Keeping stripe heights in 1..4 preserves the original difficulty: the solver
#   still has to parse runs and reuse heights as geometric thickness.
# - To avoid a too-trivial regime where every layer has thickness 1, the
#   generator requires at least one stripe with height >= 2. This makes the
#   "thickness as radial extent" rule observable from just a few examples.
# - The number of runs stays in 2..6, which is enough repetition to reveal the
#   nesting rule without creating clutter.
#
# c) Implicit assumptions in the original main()
# - Parsing loop assumes each stripe row is uniform across the entire width.
#   This remains true because the input format is unchanged.
# - `cur = full((h0, h0), c0)` assumed the innermost object is an axis-aligned
#   square. That is no longer valid: the center must now be a solid diamond.
# - `new_n = cur.shape[0] + 2 * h` still remains valid if `cur.shape[0]` is the
#   current diamond's bounding-box size. Expanding the Manhattan radius by h
#   grows the bounding box by exactly 2*h.
# - `nxt[...] = cur` was safe for solid square interiors. It is unsafe for
#   diamonds because `cur` contains zero corners inside its bounding box; a full
#   overwrite would punch holes into the new ring. The variant therefore only
#   overlays non-zero cells from the previous stage.
#
# d) Placement legality
# - Output is built around a fixed center. Each new ring is defined by an exact
#   Manhattan-distance interval, so rings never overlap incorrectly.
# - Generator caps `sum(heights)` so the final bounding box
#   `2 * sum(heights) - 1` stays within 29, keeping all outputs comfortably
#   inside normal ARC sizes.
#
# e) Visual observability
# - The change is visible in every non-degenerate sample because the output now
#   has zero-valued corner regions and diagonal edges instead of square edges.
# - Using the same stripe input but a different geometric interpretation makes
#   the D2 change easy to isolate for a human observer.
#
# f) Visual cleanliness
# - There is only one object family in the output: concentric diamonds. Empty
#   corners stay background 0, so no region carries competing annotations.
#
# g) Color inferability
# - Colors are unchanged from the source task: ring colors still follow stripe
#   order bottom->top. No extra color remapping is introduced.
#
# New solver assumptions and generator guarantees
# ----------------------------------------------
# - main() now assumes the intended output background is 0. generate_input()
#   guarantees stripe colors are chosen from 1..9, so 0 remains reserved.
# - main() assumes there is at least one stripe. generate_input() always creates
#   2..6 stripes.
# - main() assumes output size is odd because a diamond has a single center
#   point. generate_input() enforces this indirectly via the radius formula.


def _parse_runs_bottom_to_top(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    runs: list[tuple[int, int]] = []
    r = int(h - 1)
    while r >= 0:
        col = int(g[r, 0])
        assert bool((g[r] == col).all())
        run_h = 1
        rr = int(r - 1)
        while rr >= 0 and bool((g[rr] == col).all()):
            run_h += 1
            rr -= 1
        runs.append((col, run_h))
        r = rr
    return runs


def _distance_grid(radius: int) -> np.ndarray:
    size = int(2 * radius + 1)
    center = int(radius)
    ys, xs = np.indices((size, size))
    return np.abs(ys - center) + np.abs(xs - center)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested diamond frames from a stack of horizontal color stripes.

    The input format is unchanged from the source task; only the output shape is
    rotated into Manhattan diamonds. Background 0 is used outside the outermost
    diamond so the silhouette remains visible.
    """
    runs = _parse_runs_bottom_to_top(np.array(grid, dtype=int))
    assert not (not runs)

    c0, h0 = runs[0]
    radius = int(h0 - 1)
    cur = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=int)
    cur[_distance_grid(radius) <= radius] = int(c0)

    for col, h in runs[1:]:
        h = int(h)
        old_radius = int(radius)
        radius = int(radius + h)
        new_n = int(2 * radius + 1)
        nxt = np.zeros((new_n, new_n), dtype=int)

        dist = _distance_grid(radius)
        ring_mask = (dist <= radius) & (dist > old_radius)
        nxt[ring_mask] = int(col)

        offset = h
        sub = nxt[offset : offset + cur.shape[0], offset : offset + cur.shape[1]]
        sub[cur != 0] = cur[cur != 0]
        cur = nxt

    return cur


def _gen_attempt() -> np.ndarray | None:
    n_runs = int(np.random.randint(2, 7))
    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    if sum(heights) > 15:
        return None
    if max(heights) == 1:
        return None

    width = int(np.random.randint(6, 15))

    cols: list[int] = []
    prev = None
    for _ in range(n_runs):
        c = int(np.random.randint(1, 10))
        while prev is not None and c == prev:
            c = int(np.random.randint(1, 10))
        cols.append(c)
        prev = c

    stripes = [
        np.full((run_h, width), col, dtype=int)
        for col, run_h in reversed(list(zip(cols, heights)))
    ]
    inp = np.vstack(stripes)

    out = main(inp)
    if out.shape[0] > 29 or out.shape[1] > 29:
        return None
    if not np.any(out == 0):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
