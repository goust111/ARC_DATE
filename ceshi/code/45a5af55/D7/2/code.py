# Variant of 45a5af55
# Dimension: D7 — 输出网格形状确定方式 (grid_shape)
# Change: 正方形：N = 底层色条高度 + 2 × 其余色条高度之和... → 输出尺寸由输入网格宽度决定，框架居中放置

from __future__ import annotations

import numpy as np


# Design analysis for the D7 variant
# a) Concrete behavior and trace
#    Original behavior: if bottom->top stripe heights are [2(red), 1(blue), 3(green)],
#    the solver builds a 2x2 red center, wraps 1 blue ring -> 4x4, then wraps 3 green
#    ring -> 10x10, and returns that 10x10 square directly.
#    Variant behavior: the same stripe stack still defines the inner 10x10 nested shape,
#    but the final output side is no longer 10. Instead it is the full input-grid width W.
#    If W=14, the 10x10 nested shape is pasted into the center of a 14x14 background grid,
#    leaving a 2-cell background border on every side.
#    Therefore the changed step is the last stage of the pipeline: "return cur" becomes
#    "place cur into a W x W canvas".
#
# b) Complexity calibration
#    The variant needs a visible background region; otherwise W-based sizing can collapse
#    back into the original rule. To make the new behavior observable but not noisy, the
#    generator keeps 2-5 stripes, stripe heights 1-3, and forces W > inner square size.
#    This preserves the original nested-frame logic while making the new outer padding
#    obvious in nearly every sample.
#
# c) Implicit-assumption audit of the original main()
#    - "H, _W = g.shape": only used to know the input width; still valid, but width now
#      becomes the output side length.
#    - Row parsing via g[r, 0] and (g[rr] == col).all(): this relied on every row of the
#      entire input being a uniform stripe. That no longer holds once explicit background
#      margins are added. The new solver first crops the non-background rectangle, then
#      performs the same uniform-row run parsing inside that crop.
#    - "return cur": this relied on output size being fully determined by stripe heights.
#      This is exactly the D7 assumption that changes, so the solver now pastes cur into
#      a background square of side input_width.
#
# d) Placement legality
#    The generator creates one rectangular non-background stripe block inside a background
#    field. The block never overlaps anything else, is fully in-bounds, and each stripe
#    occupies whole rows of that block. The derived inner nested square is guaranteed to
#    fit inside the final W x W output canvas.
#
# e) Visual observability
#    The user must infer that output side = input width, not stripe-derived size. To make
#    this visible, the generator always enforces a positive border thickness
#    (input_width - inner_nested_side) >= 2. The stripe block in the input also has
#    visible horizontal background margins, so the full input width is an obvious cue.
#
# f) Visual cleanliness
#    Only two regions exist in the input: one clean rectangular stripe stack and one clean
#    background. The output similarly contains a centered square motif plus a uniform
#    background border. No overlapping markers or irregular masks are introduced.
#
# g) Color inferability
#    No new color mapping is introduced. Colors keep their original roles: each stripe
#    color becomes the matching ring color, and the input background color becomes the
#    padding color in the output. The background is shown explicitly in the input, so the
#    padding color is inferable from structure rather than guessed.
#
# h) Variant-specific non-degeneration target
#    If input width happened to equal the stripe-derived nested size, the sample would
#    collapse back to the original task. The generator forbids that by construction.
#
# i) New solver assumptions and generator guarantees
#    The solver now assumes:
#    - one background color occupies all four corners;
#    - all non-background cells form one solid rectangle;
#    - each row inside that rectangle is monochrome;
#    - adjacent stripe runs have different colors;
#    - the stripe-derived nested square fits within the input width.
#    The generator enforces each of these conditions directly.


def _build_nested_square(runs: list[tuple[int, int]]) -> np.ndarray:
    """Build the original nested-square object from bottom->top (color, height) runs."""
    c0, h0 = runs[0]
    cur = np.full((int(h0), int(h0)), int(c0), dtype=int)

    for col, h in runs[1:]:
        h = int(h)
        new_n = int(cur.shape[0] + 2 * h)
        nxt = np.full((new_n, new_n), int(col), dtype=int)
        nxt[h : h + cur.shape[0], h : h + cur.shape[1]] = cur
        cur = nxt

    return cur


def _infer_bg(g: np.ndarray) -> int:
    corners = [
        int(g[0, 0]),
        int(g[0, -1]),
        int(g[-1, 0]),
        int(g[-1, -1]),
    ]
    return max(set(corners), key=corners.count)


def _parse_runs(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    non_bg = g != int(bg)
    assert bool(non_bg.any())

    rows, cols = np.where(non_bg)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    block = g[r0 : r1 + 1, c0 : c1 + 1]

    # The variant generator produces a solid rectangular stripe block.
    assert not bool((block == int(bg)).any())

    runs: list[tuple[int, int]] = []
    r = int(block.shape[0] - 1)
    while r >= 0:
        row = block[int(r)]
        col = int(row[0])
        assert col != int(bg)
        assert bool((row == col).all())
        h = 1
        rr = int(r - 1)
        while rr >= 0 and bool((block[int(rr)] == col).all()):
            h += 1
            rr -= 1
        runs.append((int(col), int(h)))
        r = int(rr)

    assert not (not runs)
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from a stripe stack, but place the result inside a
    background square whose side length equals the input-grid width.
    """
    g = np.array(grid, dtype=int)
    _H, W = g.shape

    bg = _infer_bg(g)
    runs = _parse_runs(g, bg)
    cur = _build_nested_square(runs)

    n = int(cur.shape[0])
    assert n <= int(W)

    out = np.full((int(W), int(W)), int(bg), dtype=int)
    top = int((W - n) // 2)
    left = int((W - n) // 2)
    out[top : top + n, left : left + n] = cur
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.randint(0, 10))

    n_runs = int(np.random.randint(2, 6))
    heights = [int(np.random.randint(1, 4)) for _ in range(n_runs)]
    inner_n = int(heights[0] + 2 * sum(heights[1:]))

    # Force the D7 behavior to be visible: the final canvas must be strictly larger
    # than the stripe-derived nested square.
    outer_pad = int(np.random.randint(1, 4))
    W = int(inner_n + 2 * outer_pad)
    if W > 30:
        return None

    block_h = int(sum(heights))
    vertical_pad = int(np.random.randint(1, 4))
    H = int(block_h + 2 * vertical_pad)
    if H > 30:
        return None

    side_margin = int(np.random.randint(1, min(4, W // 2)))
    stripe_w = int(W - 2 * side_margin)
    if stripe_w <= 0:
        return None

    colors: list[int] = []
    used = {int(bg)}
    for _ in range(n_runs):
        c = int(np.random.randint(0, 10))
        while c in used or (colors and c == colors[-1]):
            c = int(np.random.randint(0, 10))
        colors.append(c)
        used.add(c)

    inp = np.full((H, W), int(bg), dtype=int)
    r = int(vertical_pad)
    for col, h in reversed(list(zip(colors, heights))):
        inp[r : r + h, side_margin : side_margin + stripe_w] = int(col)
        r += h

    # Independent expected-output assembly for generator self-check.
    runs = list(zip(colors, heights))
    inner = _build_nested_square(runs)
    expected = np.full((W, W), int(bg), dtype=int)
    top = int((W - inner.shape[0]) // 2)
    expected[top : top + inner.shape[0], top : top + inner.shape[1]] = inner

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
