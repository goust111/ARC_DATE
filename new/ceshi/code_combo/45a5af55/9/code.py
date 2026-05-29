# Combo variant of 45a5af55 (index 9)
# combo_spec: D4/0+D9/3
# Name: 空心框架洪水填充

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parents[4]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import flood_fill


"""
Design analysis for the fused variant
=====================================

a) Concrete behavior trace
Input is one stack of full-width horizontal color stripes embedded in a 0
background canvas. The visible stripe object is still read bottom -> top, and a
stripe's height still controls square thickness exactly as in the base task.

Example bottom->top runs: [(2, 1), (5, 2), (7, 1)] with background 0.

Original task:
1. Start from a solid 1x1 center block of color 2.
2. Wrap a 2-thick color-5 frame around it -> side 5.
3. Wrap a 1-thick color-7 frame around that -> side 7.

Fused task stage 1 (D4):
1. Start from a 1x1 background square.
2. Wrap a 1-thick color-2 frame around it, so the innermost object is a 3x3
   hollow ring with a 1x1 0 hole.
3. Wrap the color-5 and color-7 layers exactly like the base task.

Fused task stage 2 (D9):
4. Start a flood fill from the geometric center of the output, using the bottom
   stripe color 2. Because stage 1 left a closed central hole, the fill is
   trapped inside that hole and recolors exactly the central cavity.

Net difference:
- Compared with the base task, the innermost layer does not begin as a solid h0
  square. It first becomes a hollow 3*h0 square ring and only then receives a
  fill step. After the fill, the central color-2 area has side 3*h0 rather than
  h0, so the final output side grows from `h0 + 2*sum(other)` to
  `3*h0 + 2*sum(other)`.
- Compared with the pure D4 hollow-center variant, the center is no longer left
  black; it is specifically recolored by a second flood-fill phase.

b) Complexity calibration
The combo keeps the scene simple: one stripe block, one output object, 2-6
stripes, heights in 1-4. The visible novelty comes from one coupled two-step
mechanism:
- stage 1 must be understood as "make the bottom stripe into a hollow ring"
- stage 2 must be understood as "fill only that trapped hole with the same inner
  stripe color"
This is stronger than either isolated change: if the generator produced only one
stripe or allowed the center fill to touch the outside background, the task would
collapse into a trivial recolor. The generator therefore always creates at least
two stripes and embeds them in a background canvas only at input time, never
inside the stripe block itself.

c) Implicit-assumption audit against the original main()
- Original assumption: the whole input grid is the stripe block. This breaks once
  0 background margins are introduced to make the hollow/fill behavior legible.
  The fused solver must first crop the non-zero bounding box.
- Original line `cur = np.full((h0, h0), c0, dtype=int)` assumed the bottom run
  immediately becomes a solid center. This no longer holds. The new solver must
  instead start from a background hole of side h0 and wrap it with a c0 frame of
  thickness h0.
- Original wrapping logic assumed every later stripe simply adds an outer frame.
  That remains valid after the new initialization, so the later loop can stay
  structurally the same.
- New stage-2 assumption: after stage 1, the geometric center cell lies inside a
  background cavity. The generator guarantees this because the innermost ring is
  always constructed with hole side h0 >= 1 and all later operations are symmetric.
- New stage-2 assumption: the center hole is sealed off from the outer 0 canvas.
  That holds because the input background is not part of the output geometry; the
  output is built from the cropped stripe block alone, so the only 0 region in
  stage 1 is the enclosed center hole.

d) Placement legality
The generator builds one rectangular stripe block with no internal 0 gaps, then
places it inside a larger 0 canvas with at least one-cell margins. Since there is
only one object, no overlap is possible. Output legality is straightforward
because the solver constructs only concentric squares whose final side length is
checked to remain <= 30.

e) Visual observability
Across 2-4 examples, a human can see:
- input: a colored stripe stack sitting in black background
- output: the familiar nested-square construction, but the bottom stripe now
  controls an enlarged central region produced by a hole-then-fill sequence
The background 0 already exists visibly in the input, so the temporary hole in
stage 1 uses an established color. The flood-filled center is also inferable
because it exactly matches the innermost stripe color and stays bounded by that
ring.

f) Visual cleanliness
The result is still a single centered square construction. No layer overlaps
ambiguously: every stripe contributes one clean annulus, and the only 0 region in
the intermediate state is the centered cavity. The final output contains no noisy
markers or disconnected fragments.

g) Color inference
There is no arbitrary color remapping. Each stripe keeps its own color as its
frame color. The only color choice introduced by the combo is structurally
determined: the flood fill uses the bottom stripe's color because that stripe is
also the ring enclosing the cavity. This keeps the color reasoning local and
observable instead of depending on hidden numeric ids.

Post-change solver assumptions the generator must satisfy
---------------------------------------------------------
1. Background is 0 and all stripe colors are in 1..9.
2. All non-zero cells belong to one rectangular stripe block.
3. Inside that block, each row is monochrome and contiguous rows of the same
   color form one stripe run.
4. There are at least two stripe runs so the nesting structure stays visible.
5. The combo output side `3*h0 + 2*sum(other_heights)` stays <= 30.
6. The cropped stripe block contains no 0 cells, ensuring the only output cavity
   before flood fill is the intended central hole.
The generator below enforces each of these constraints directly.
"""


BG = 0


def _wrap_with_frame(inner: np.ndarray, color: int, thickness: int) -> np.ndarray:
    new_side = int(inner.shape[0] + 2 * thickness)
    out = np.full((new_side, new_side), int(color), dtype=int)
    out[thickness : thickness + inner.shape[0], thickness : thickness + inner.shape[1]] = inner
    return out


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
        color = int(g[r, 0])
        if color == BG:
            raise ValueError("background row inside stripe block")
        if not bool((g[r] == color).all()):
            raise ValueError("non-uniform row inside stripe block")

        run_h = 1
        rr = int(r - 1)
        while rr >= 0:
            if not bool((g[rr] == int(g[rr, 0])).all()):
                raise ValueError("non-uniform row inside stripe block")
            if int(g[rr, 0]) != color:
                break
            run_h += 1
            rr -= 1

        runs.append((color, run_h))
        r = rr

    if not runs:
        raise ValueError("no stripe runs found")
    return runs


def _build_hollow_nested(runs: list[tuple[int, int]]) -> np.ndarray:
    inner_color, inner_height = runs[0]
    cur = np.full((inner_height, inner_height), BG, dtype=int)
    cur = _wrap_with_frame(cur, inner_color, inner_height)

    for color, thickness in runs[1:]:
        cur = _wrap_with_frame(cur, int(color), int(thickness))

    return cur


def _build_original_nested(runs: list[tuple[int, int]]) -> np.ndarray:
    inner_color, inner_height = runs[0]
    cur = np.full((inner_height, inner_height), inner_color, dtype=int)

    for color, thickness in runs[1:]:
        cur = _wrap_with_frame(cur, int(color), int(thickness))

    return cur


def _fill_center_hole(hollow: np.ndarray, fill_color: int) -> np.ndarray:
    out = np.array(hollow, copy=True)
    center_r = int(out.shape[0] // 2)
    center_c = int(out.shape[1] // 2)
    if int(out[center_r, center_c]) != BG:
        raise ValueError("expected a hollow center before flood fill")
    flood_fill(out, center_r, center_c, int(fill_color), connectivity=4)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    1. Crop the non-zero stripe block from the 0 background canvas.
    2. Read stripe runs bottom -> top.
    3. Build nested square frames, but make the bottom stripe the innermost hollow
       ring instead of a solid center.
    4. Flood-fill the central cavity from the geometric center using the bottom
       stripe color.
    """
    block = _extract_stripe_block(np.array(grid, dtype=int))
    runs = _parse_runs_bottom_to_top(block)

    hollow = _build_hollow_nested(runs)
    return _fill_center_hole(hollow, runs[0][0])


def _gen_attempt() -> np.ndarray | None:
    n_runs = int(np.random.randint(2, 7))
    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]

    output_side = int(3 * heights[0] + 2 * sum(heights[1:]))
    if output_side > 30:
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

    colors = list(np.random.choice(np.arange(1, 10), size=n_runs, replace=False))
    runs = [(int(colors[i]), int(heights[i])) for i in range(n_runs)]

    stripes_top_to_bottom = [
        np.full((height, stripe_w), color, dtype=int)
        for color, height in reversed(runs)
    ]
    block = np.vstack(stripes_top_to_bottom)

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    canvas[top : top + stripe_h, left : left + stripe_w] = block

    parsed_runs = _parse_runs_bottom_to_top(_extract_stripe_block(canvas))
    if parsed_runs != runs:
        raise ValueError("generated input did not preserve stripe runs")

    hollow = _build_hollow_nested(runs)
    original = _build_original_nested(runs)
    out = main(canvas)

    if out.shape != (output_side, output_side):
        raise ValueError("generated sample produced unexpected output size")
    if int(hollow[hollow.shape[0] // 2, hollow.shape[1] // 2]) != BG:
        raise ValueError("stage-1 hollow center was not preserved")
    if int(out[out.shape[0] // 2, out.shape[1] // 2]) != runs[0][0]:
        raise ValueError("stage-2 flood fill did not recolor the center")
    if np.array_equal(out, hollow):
        raise ValueError("combo output collapsed to the hollow-only variant")
    if np.array_equal(out, original):
        raise ValueError("combo output collapsed to the original task")
    if int(out.shape[0] - original.shape[0]) != 2 * heights[0]:
        raise ValueError("center expansion did not match the intended combo geometry")

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
