# Combo variant of 45a5af55 (index 24)
# combo_spec: D2/0+D4/0+D9/3
# Name: 菱形空心洪水填充嵌套

from __future__ import annotations

import numpy as np


"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
The input is still a single stack of full-width horizontal color stripes, but the
generator embeds that stack inside a visible 0-background canvas so the solver can
reliably crop the stripe block and the task vocabulary explicitly includes the
background color used by the hollow center.

Read runs from bottom to top: [(c0, h0), (c1, h1), ...].

Base task:
- bottom stripe -> solid h0 x h0 square
- each later stripe -> wrap a square frame of thickness hi

This combo:
1. Geometry switch (D2): build concentric Manhattan diamonds instead of squares.
2. Hollow-center inversion (D4): the bottom stripe does not immediately become a
   solid center. Instead it becomes a hollow diamond ring whose thickness is h0
   and whose hole is itself a diamond of radius h0 - 1.
3. Flood-fill stage (D9): after all diamond rings are built, start from the exact
   center cell and 4-neighbor flood fill the hollow region with c0. The diagonal
   diamond walls block the fill from leaking to the background corners.

Small trace with runs bottom->top = [(4, 2), (7, 1), (3, 2)]:
- original:
  4 -> 2x2 solid square
  7 -> add square frame thickness 1
  3 -> add square frame thickness 2
- combo pre-fill:
  4 -> hollow diamond ring:
       outer radius 3, inner hole radius 1
  7 -> add diamond shell at radius 4
  3 -> add diamond shell at radii 5..6
- combo final:
  flood fill the center-connected hole (radius 1) with color 4
  result: a diamond output with outer shells 7 and 3, and an enlarged inner
  color-4 diamond spanning both the former ring and the filled hole.

The crucial difference is that the first stripe now controls two things at once:
the thickness of the innermost ring and the size of the later flood-filled hole.
That makes the final inner diamond much larger than in plain D2.

b) Complexity calibration
- Heights stay in 1..4 and run count stays in 2..6, preserving the original
  "stripe height becomes geometric thickness" reasoning load.
- To keep the flood-fill step visibly meaningful, the generator forces h0 >= 2,
  so the filled hole always contains at least 5 cells instead of a trivial single
  pixel.
- Colors are sampled without repetition, which removes ambiguity about whether a
  repeated color should be merged across layers.

c) Implicit-assumption audit against the original main()
- The original parser assumed the whole grid was stripe data. That breaks once a
  visible 0 background is introduced, so the solver must first crop the non-zero
  stripe block.
- The original `cur = full((h0, h0), c0)` assumed the innermost object is a
  solid axis-aligned square. That is invalid now; the first object is a hollow
  diamond ring.
- The original square-size recurrence `cur.shape[0] + 2*h` is no longer the right
  semantic object, but the same numeric growth still holds for diamond bounding
  boxes when Manhattan radius increases by h.
- The original full overwrite of the centered subarray would be unsafe because a
  hollow diamond and the outer background both contain legitimate zeros. The new
  code therefore overlays only non-zero cells from the previous stage.
- A new assumption appears in the flood-fill step: the center cell must be inside
  a closed 4-connected background pocket. The generator guarantees this by making
  the innermost diamond ring thickness equal to h0 around a hole of radius h0-1.

d) Placement legality
- The generator places exactly one rectangular stripe block on a 0 canvas, with
  at least one-cell margins on all sides.
- Stripe rows are monochrome and contiguous, so run parsing is unambiguous.
- Output size is bounded by `4*h0 - 1 + 2*sum(other_heights) <= 29`, which keeps
  every generated output within normal ARC limits.

e) Visual observability
- Across examples, the output silhouette is visibly diamond-shaped rather than
  square, so D2 is observable.
- The inner color region is much larger than the bottom stripe height alone would
  suggest under the plain diamond rule, which makes the hollow-then-fill behavior
  recoverable from examples.
- Input uses explicit background 0, and output keeps 0 in the outside corners, so
  the notion of "fillable background region bounded by walls" is visually grounded
  rather than relying on invisible numeric conventions.

f) Visual cleanliness
- There is only one object family in the output: concentric diamonds.
- Background 0 appears only as canvas margin, outer corners, and the temporary
  fillable hole, so regions stay easy to distinguish.
- No overlapping annotations or secondary markers are introduced.

g) Change inferability
- No color remapping is used; stripe order still maps directly to layer colors.
- The only new color behavior is that c0 appears both on the innermost ring and
  in the flood-filled center, which is structurally cued by position.

Post-change solver assumptions that the generator satisfies
-----------------------------------------------------------
1. Background is 0 and all non-zero cells belong to one rectangular stripe block.
2. Every row inside that block is monochrome, and equal-color rows form one
   contiguous run.
3. The bottom run height is at least 2, so the center hole is visibly larger than
   one cell before flood fill.
4. Adjacent rings use different colors, and in practice all run colors are unique.
5. The pre-fill center region is closed under 4-neighbor connectivity, so flood
   fill cannot escape into the outer background corners.
"""


BG = 0


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


def _parse_runs_bottom_to_top(block: np.ndarray) -> list[tuple[int, int]]:
    g = np.array(block, dtype=int)
    h, _w = g.shape
    runs: list[tuple[int, int]] = []
    r = int(h - 1)
    while r >= 0:
        col = int(g[r, 0])
        if col == BG:
            raise ValueError("background row inside stripe block")
        if not bool((g[r] == col).all()):
            raise ValueError("non-uniform row inside stripe block")
        run_h = 1
        rr = int(r - 1)
        while rr >= 0 and bool((g[rr] == col).all()):
            run_h += 1
            rr -= 1
        runs.append((col, run_h))
        r = rr
    if not runs:
        raise ValueError("no stripe runs found")
    return runs


def _distance_grid(radius: int) -> np.ndarray:
    size = int(2 * radius + 1)
    center = int(radius)
    ys, xs = np.indices((size, size))
    return np.abs(ys - center) + np.abs(xs - center)


def _build_hollow_diamond_ring(color: int, thickness: int) -> tuple[np.ndarray, int]:
    hole_radius = int(thickness - 1)
    outer_radius = int(hole_radius + thickness)
    dist = _distance_grid(outer_radius)
    grid = np.zeros((2 * outer_radius + 1, 2 * outer_radius + 1), dtype=int)
    ring_mask = (dist <= outer_radius) & (dist > hole_radius)
    grid[ring_mask] = int(color)
    return grid, outer_radius


def _wrap_with_diamond_ring(
    inner: np.ndarray, old_radius: int, color: int, thickness: int
) -> tuple[np.ndarray, int]:
    new_radius = int(old_radius + thickness)
    dist = _distance_grid(new_radius)
    nxt = np.zeros((2 * new_radius + 1, 2 * new_radius + 1), dtype=int)
    ring_mask = (dist <= new_radius) & (dist > old_radius)
    nxt[ring_mask] = int(color)

    offset = int(thickness)
    sub = nxt[offset : offset + inner.shape[0], offset : offset + inner.shape[1]]
    sub[inner != BG] = inner[inner != BG]
    return nxt, new_radius


def _build_prefill_output(runs: list[tuple[int, int]]) -> tuple[np.ndarray, int]:
    c0, h0 = runs[0]
    cur, radius = _build_hollow_diamond_ring(int(c0), int(h0))
    for col, h in runs[1:]:
        cur, radius = _wrap_with_diamond_ring(cur, radius, int(col), int(h))
    return cur, int(c0)


def _flood_fill_center(grid: np.ndarray, color: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    n, m = out.shape
    start_r, start_c = n // 2, m // 2
    if int(out[start_r, start_c]) != BG:
        return out

    stack = [(start_r, start_c)]
    out[start_r, start_c] = int(color)
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < n and 0 <= nc < m and int(out[nr, nc]) == BG:
                out[nr, nc] = int(color)
                stack.append((nr, nc))
    return out


def _center_background_area(grid: np.ndarray) -> int:
    g = np.array(grid, dtype=int)
    n, m = g.shape
    start = (n // 2, m // 2)
    if int(g[start]) != BG:
        return 0

    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < n and 0 <= nc < m and int(g[nr, nc]) == BG and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested diamonds from a stack of horizontal color stripes.

    Combo rule:
    - crop the single non-zero stripe block from the 0 background canvas
    - read stripe runs bottom to top
    - bottom stripe creates a hollow diamond ring:
      hole radius = h0 - 1, ring thickness = h0
    - later stripes add outer diamond shells of matching thickness
    - finally flood fill the center-connected hollow region with the bottom color
    """
    block = _extract_stripe_block(np.array(grid, dtype=int))
    runs = _parse_runs_bottom_to_top(block)
    prefill, fill_color = _build_prefill_output(runs)
    return _flood_fill_center(prefill, fill_color)


def _gen_attempt() -> np.ndarray | None:
    n_runs = int(np.random.randint(2, 7))
    heights = [int(np.random.randint(2, 5))]
    heights.extend(int(np.random.randint(1, 5)) for _ in range(n_runs - 1))

    out_size = int(4 * heights[0] - 1 + 2 * sum(heights[1:]))
    if out_size > 29:
        return None

    stripe_h = int(sum(heights))
    stripe_w = int(np.random.randint(6, 16))

    top = int(np.random.randint(1, 4))
    bottom = int(np.random.randint(1, 4))
    left = int(np.random.randint(1, 4))
    right = int(np.random.randint(1, 4))

    canvas_h = int(stripe_h + top + bottom)
    canvas_w = int(stripe_w + left + right)
    if canvas_h > 30 or canvas_w > 30:
        return None

    cols = np.random.choice(np.arange(1, 10), size=n_runs, replace=False).tolist()
    stripes = [
        np.full((run_h, stripe_w), col, dtype=int)
        for col, run_h in reversed(list(zip(cols, heights)))
    ]
    block = np.vstack(stripes)

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    canvas[top : top + stripe_h, left : left + stripe_w] = block

    runs = _parse_runs_bottom_to_top(block)
    prefill, fill_color = _build_prefill_output(runs)
    fill_area = _center_background_area(prefill)
    out = _flood_fill_center(prefill, fill_color)

    if out.shape != (out_size, out_size):
        raise ValueError("generated sample produced unexpected output size")
    if fill_area < 5:
        raise ValueError("generated sample produced a trivial fill region")
    if int(prefill[out_size // 2, out_size // 2]) != BG:
        raise ValueError("prefill center was not hollow")
    if int(out[out_size // 2, out_size // 2]) != int(cols[0]):
        raise ValueError("final center was not flood-filled with the bottom color")
    if not np.any(out == BG):
        raise ValueError("output lost the visible diamond-background corners")
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
