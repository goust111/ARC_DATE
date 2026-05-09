# Variant of 45a5af55
# Dimension: D12 — 框架层的装饰模式 (transform)
# Change: 每层框架为纯色填充（单一颜色的均匀方形环）... → 框架角落使用特殊颜色标记

from __future__ import annotations

import numpy as np


# Design analysis for the D12 variant
# a) Concrete behavior:
#    Input stays the same: full-width horizontal stripes, read from bottom to top.
#    The bottom stripe still becomes the central solid square. Each higher stripe still
#    wraps the current shape with a square frame whose thickness equals that stripe's
#    height. The only change is on each newly created frame: its four extreme corner
#    pixels are recolored to the immediately inner stripe color. Example with runs
#    bottom->top [(2, 2), (5, 1), (7, 2)]:
#    - Original: center 2x2 of color 2, wrapped by a thickness-1 frame of color 5,
#      then a thickness-2 frame of color 7.
#    - Variant: same construction, but the four corners of the color-5 frame become 2,
#      and the four outer corners of the color-7 frame become 5.
#    Difference: only four pixels per frame change, but they encode the local
#    inside/outside color relationship instead of leaving the ring uniform.
# b) Complexity calibration:
#    The variant must be visible from a few examples, so the generator favors 3-6
#    stripes rather than 2. That produces at least two decorated frames, which shows
#    the repeated rule "corner = next inner stripe color" instead of a one-off anomaly.
#    Output size is capped so the single-pixel corners are not visually lost inside a
#    huge frame.
# c) Implicit-assumption review:
#    - Parsing runs from bottom to top is still valid.
#    - The original `np.full(..., col)` for a new frame is no longer sufficient:
#      leaving it untouched would erase the new corner decoration entirely.
#    - The inner insertion `nxt[h:h+cur.shape[0], h:h+cur.shape[1]] = cur` is still
#      valid, but the corner color must come from the previous stripe color, not from
#      `cur[0, 0]`, because `cur` may already have decorated corners from an even
#      deeper layer. The variant therefore tracks stripe colors explicitly.
# d) Placement legality:
#    No new objects are placed in the input; it remains a simple stripe stack.
#    In the output, each decoration touches only the four corner cells of the newly
#    added outer square, so there is no overlap ambiguity. The corners are guaranteed
#    to belong to the current frame because every frame adds at least one cell of margin.
# e) Visual observability:
#    Across 2-4 examples, solvers can compare several frames and notice that only the
#    four corners differ from the frame body, and that each corner color matches the
#    stripe immediately below it in the input order. Using existing stripe colors avoids
#    introducing unexplained new colors.
# f) Visual cleanliness:
#    The decoration is sparse and regular: four symmetric corner markers per frame.
#    Nothing overlaps the center square, and frames remain easy to read as nested rings.
# g) Color inferability:
#    Adjacent stripes are always generated with different colors, so corner markers are
#    always visible and consistently attributable to the next inner stripe.
# h) Generator strategy:
#    The generator samples a stripe sequence, renders the expected decorated output from
#    that sequence, then builds the stripe input and verifies `main(input)` reproduces
#    the same output. It also checks that the decorated output differs from the plain
#    original-rendering baseline, preventing silent regression to the old task.
# i) New solver assumptions:
#    `main()` now assumes each row is uniform and each decorated frame's corner color is
#    the previous stripe color. `generate_input()` guarantees this by constructing only
#    full-width stripes and enforcing adjacent color differences.


def _parse_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.array(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        raise ValueError("grid must be a non-empty 2D array")

    h_total, _w = g.shape
    runs: list[tuple[int, int]] = []
    r = int(h_total - 1)
    while r >= 0:
        row = g[int(r)]
        col = int(row[0])
        if not bool((row == col).all()):
            raise ValueError("each input row must be a uniform full-width stripe")

        h = 1
        rr = int(r - 1)
        while rr >= 0 and bool((g[int(rr)] == col).all()):
            h += 1
            rr -= 1
        runs.append((col, h))
        r = int(rr)

    if not runs:
        raise ValueError("no stripes found")
    return runs


def _render_plain_frames(runs: list[tuple[int, int]]) -> np.ndarray:
    c0, h0 = runs[0]
    cur = np.full((h0, h0), c0, dtype=int)
    for col, h in runs[1:]:
        new_n = int(cur.shape[0] + 2 * h)
        nxt = np.full((new_n, new_n), col, dtype=int)
        nxt[h : h + cur.shape[0], h : h + cur.shape[1]] = cur
        cur = nxt
    return cur


def _render_variant_frames(runs: list[tuple[int, int]]) -> np.ndarray:
    c0, h0 = runs[0]
    cur = np.full((h0, h0), c0, dtype=int)
    inner_base_color = c0

    for col, h in runs[1:]:
        new_n = int(cur.shape[0] + 2 * h)
        nxt = np.full((new_n, new_n), col, dtype=int)
        nxt[h : h + cur.shape[0], h : h + cur.shape[1]] = cur

        # D12 decoration: the outermost four corner cells of this frame show the
        # immediately inner stripe color instead of the frame's own color.
        nxt[0, 0] = inner_base_color
        nxt[0, -1] = inner_base_color
        nxt[-1, 0] = inner_base_color
        nxt[-1, -1] = inner_base_color

        cur = nxt
        inner_base_color = col

    return cur


def _build_input_from_runs(runs: list[tuple[int, int]], width: int) -> np.ndarray:
    stripes = [
        np.full((height, width), color, dtype=int)
        for color, height in reversed(runs)
    ]
    return np.vstack(stripes)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from a stack of horizontal color stripes.

    D12 variant: each non-central frame keeps its original thickness/color, but the
    four outer corner pixels of that frame are recolored to the immediately inner
    stripe color.
    """
    runs = _parse_runs(grid)
    return _render_variant_frames(runs)


def _gen_attempt() -> np.ndarray | None:
    n_runs = int(np.random.randint(3, 7))
    widths = int(np.random.randint(7, 15))
    heights = [int(np.random.randint(1, 4)) for _ in range(n_runs)]
    out_size = heights[0] + 2 * sum(heights[1:])
    if out_size > 24 or out_size < 7:
        return None

    colors: list[int] = []
    prev = None
    for _ in range(n_runs):
        c = int(np.random.randint(1, 10))
        while prev is not None and c == prev:
            c = int(np.random.randint(1, 10))
        colors.append(c)
        prev = c

    runs = list(zip(colors, heights))
    expected = _render_variant_frames(runs)
    plain = _render_plain_frames(runs)

    # Prevent regression to the original task: the variant must visibly change output.
    if int((expected != plain).sum()) != 4 * (n_runs - 1):
        return None

    inp = _build_input_from_runs(runs, widths)
    if not np.array_equal(main(inp), expected):
        raise RuntimeError("generator produced an inconsistent sample")
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
