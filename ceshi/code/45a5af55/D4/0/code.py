# Variant of 45a5af55
# Dimension: D4 — 中心填充方式 (transform)
# Change: 中心为实心正方形（用最底层色条颜色完全填充）... → 中心为空心（背景色）正方形

from __future__ import annotations

import numpy as np


"""
Design analysis for the D4 variant
==================================

a) Concrete behavior and trace
Input still represents one stack of full-width horizontal color stripes, read from
bottom to top. To make the new "leave the center as background" rule visible, the
generator places the stripe stack inside a black (0) background canvas. The solver
extracts the non-zero bounding box, so the stripe object itself still behaves like
the original task.

Original flow on runs [(c0, h0), (c1, h1), ...] from bottom upward:
- Start with a solid h0 x h0 square of color c0.
- For each later stripe, add a square frame of thickness hi in color ci.

Variant flow:
- Start with a background square of side h0.
- Wrap that square with a frame of thickness h0 in color c0.
  This makes the innermost object a hollow square ring, not a solid center.
- For each later stripe, keep the original wrapping rule: add a square frame of
  thickness hi in color ci around the current shape.

Small example with bottom-to-top runs [(2, 1), (5, 2)] and background 0:
- Original:
  bottom stripe -> [[2]]
  wrap by color 5 thickness 2 -> a 5-frame around a solid 2 center
- Variant:
  start with [[0]]
  wrap by color 2 thickness 1 -> a 3x3 ring with a 0 hole
  wrap by color 5 thickness 2 -> a larger 5-frame around that ring
Difference: the bottom stripe no longer fills the center pixel(s); instead it
creates the innermost ring and preserves a background hole.

b) Complexity calibration
The change is intentionally narrow: only the innermost construction changes.
Using hole side = h0 keeps the stripe height meaningful and produces a clearly
visible hollow center for every h0 >= 1. The generator keeps 2-6 runs and
heights 1-4, so multiple layers still exist and the new center behavior remains
visually obvious instead of degenerating into a trivial one-layer task.

c) Implicit-assumption audit against the original main()
- The original line `cur = np.full((h0, h0), c0, dtype=int)` assumed the bottom
  stripe must become a solid block. This is exactly the old property that changes.
- The later wrapping loop only assumes "current shape is a square" and "stripe
  height is frame thickness". Those assumptions remain valid, so that loop can be
  preserved unchanged after changing how `cur` is initialized.
- The original parser assumed the whole grid was occupied by stripes. That is no
  longer true once background is shown explicitly, so the new solver first crops
  the non-background bounding box before parsing runs.

d) Placement legality
The generator builds one contiguous stripe block, then embeds it in a 0 canvas
with at least one-cell margins on every side. There is only one object, so no
overlap can occur. The non-zero bounding box always contains a clean stack of
uniform rows, which matches the solver's expectations.

e) Visual observability
Across 2-4 examples, a human can now see:
- input: a stripe stack on black background
- output: the same nested-frame idea as the base task, but the exact center is
  black rather than filled with the bottom stripe color
Because black already appears as background in the input, the center hole is a
natural continuation of the input vocabulary rather than an unexplained new color.

f) Visual cleanliness
The output remains a single centered square construction with no overlapping
objects. The only extra visual element is the black input margin, which improves
readability instead of cluttering the scene.

g) Color inference
No new color mapping is introduced. Stripe colors are copied directly into frame
colors, and 0 stays the background color in both input and the new hollow center.

Post-change solver assumptions to satisfy in the generator
----------------------------------------------------------
1. Background is 0, and at least one non-zero stripe exists.
2. All non-zero cells belong to one rectangular stripe block.
3. Inside that block, every row is monochrome and rows of the same color form one
   contiguous run.
4. The derived output size `3*h0 + 2*sum(other_heights)` must stay <= 30.
The generator explicitly enforces each of these conditions.
"""


BG = 0


def _wrap_with_frame(inner: np.ndarray, color: int, thickness: int) -> np.ndarray:
    new_n = int(inner.shape[0] + 2 * thickness)
    nxt = np.full((new_n, new_n), int(color), dtype=int)
    nxt[thickness : thickness + inner.shape[0], thickness : thickness + inner.shape[1]] = inner
    return nxt


def _extract_stripe_block(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    mask = g != BG
    if not bool(mask.any()):
        raise ValueError("input contains no stripe block")

    rows, cols = np.where(mask)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    block = g[r0 : r1 + 1, c0 : c1 + 1]

    if bool((block == BG).any()):
        raise ValueError("stripe block contains background gaps")
    return block


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from a stack of horizontal color stripes.

    Variant D4:
    - Read stripes bottom to top.
    - Bottom stripe becomes the innermost hollow square frame:
      thickness = stripe height, center remains background (0), and the hole side
      length is also stripe height.
    - Each subsequent stripe wraps the current square with a frame whose thickness
      equals that stripe's height.
    """
    block = _extract_stripe_block(np.array(grid, dtype=int))
    H, _W = block.shape

    runs: list[tuple[int, int]] = []  # (color, height), bottom -> top
    r = int(H - 1)
    while r >= 0:
        col = int(block[r, 0])
        if not bool((block[r] == col).all()):
            raise ValueError("non-uniform row inside stripe block")
        h = 1
        rr = int(r - 1)
        while rr >= 0 and bool((block[rr] == col).all()):
            h += 1
            rr -= 1
        runs.append((col, h))
        r = rr

    if not runs:
        raise ValueError("no stripe runs found")

    c0, h0 = runs[0]
    cur = np.full((h0, h0), BG, dtype=int)
    cur = _wrap_with_frame(cur, c0, h0)

    for col, h in runs[1:]:
        cur = _wrap_with_frame(cur, int(col), int(h))

    return cur


def _gen_attempt() -> np.ndarray | None:
    n_runs = int(np.random.randint(2, 7))
    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]

    out_size = int(3 * heights[0] + 2 * sum(heights[1:]))
    if out_size > 30:
        return None

    stripe_h = int(sum(heights))
    stripe_w = int(np.random.randint(6, 16))

    top = int(np.random.randint(1, 4))
    bottom = int(np.random.randint(1, 4))
    left = int(np.random.randint(1, 4))
    right = int(np.random.randint(1, 4))

    canvas_h = stripe_h + top + bottom
    canvas_w = stripe_w + left + right
    if canvas_h > 30 or canvas_w > 30:
        return None

    cols: list[int] = []
    prev = None
    for _ in range(n_runs):
        c = int(np.random.randint(1, 10))
        while prev is not None and c == prev:
            c = int(np.random.randint(1, 10))
        cols.append(c)
        prev = c

    stripes = [np.full((h, stripe_w), col, dtype=int) for col, h in reversed(list(zip(cols, heights)))]
    block = np.vstack(stripes)

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    canvas[top : top + stripe_h, left : left + stripe_w] = block

    # Output-first validation by direct solver check.
    out = main(canvas)
    if out.shape != (out_size, out_size):
        raise ValueError("generated sample produced unexpected output size")
    if int(out[out_size // 2, out_size // 2]) != BG:
        raise ValueError("generated sample did not produce a hollow center")
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
