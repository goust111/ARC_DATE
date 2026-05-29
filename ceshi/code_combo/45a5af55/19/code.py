# Combo variant of 45a5af55 (index 19)
# combo_spec: D3/1+D4/0+D12/0
# Name: 薄层空心条纹框架嵌套

from __future__ import annotations

import numpy as np


"""
Design analysis for combo D3/1 + D4/0 + D12/0
==============================================

a) Concrete behavior and trace
Input is one vertical stack of full-width uniform stripes. For readability and to
make background 0 part of the visual vocabulary, the stack is embedded in a 0
canvas; the solver crops the non-zero bounding box before parsing.

Original task on bottom-to-top runs [(c0, h0), (c1, h1), ...]:
- Start with a solid h0 x h0 square of color c0.
- Each later stripe adds one solid square frame of thickness hi in color ci.

Combo behavior:
- Treat every stripe height as a repeat count, not as a geometric thickness.
- Start from a canonical 1x1 background seed. This resolves the D3/D4 collision:
  once height no longer means thickness, the bottom stripe should not secretly keep
  a second geometric role such as "hole side = h0".
- Read runs bottom to top. For each stripe (color, h), add h separate 1-pixel
  frames of that color.
- Frame decoration alternates by unit-layer depth from the center outward:
  layer 1 horizontal stripes, layer 2 vertical stripes, layer 3 horizontal, etc.
  In each frame, colored cells alternate with background 0 according to that
  orientation, so repeated same-color layers remain visually separable.

Small trace with runs [(2, 2), (5, 1), (7, 2)]:
- Original:
  1. color 2, h=2 -> solid 2x2 center
  2. color 5, h=1 -> one solid 1-thick frame
  3. color 7, h=2 -> one solid 2-thick outer frame
- Combo:
  1. seed = [[0]]
  2. color 2, repeat 2 -> 1px horizontal striped ring, then 1px vertical striped ring
  3. color 5, repeat 1 -> next horizontal striped ring
  4. color 7, repeat 2 -> next vertical striped ring, then next horizontal striped ring
Difference:
- center is hollow instead of solid
- every visible shell is exactly 1 pixel thick
- height controls how many rings a stripe contributes
- rings are striped rather than solid

b) Complexity calibration
- Heights stay in 1..4 and run count in 3..6, so the task is still compact.
- The generator forces at least one height > 1, otherwise D3 could collapse into
  "one stripe -> one ring" and become visually ambiguous.
- The total number of unit layers is kept at least 4, so both stripe orientations
  appear and the nested repetition is observable in 2-4 examples.
- Using a fixed 1-cell hole avoids adding a hidden second parameter that would make
  the bottom stripe behave unlike all others.

c) Hidden-assumption audit of the original main()
- `g[r, 0]` and rowwise equality checks assume each row in the stripe object is
  monochrome and full-width. This still holds inside the cropped stripe block.
- The original code assumed the whole input grid was stripes; that breaks once a 0
  margin is added, so we must crop the non-zero bounding box first.
- `cur = full((h0, h0), c0)` assumed the bottom stripe creates a solid square whose
  side equals its height. This fails under both D3 and D4, and is replaced by a
  fixed 1x1 background seed plus repeated unit-frame wrapping.
- `new_n = cur.shape[0] + 2 * h` assumed one stripe adds one frame whose thickness
  equals stripe height. Under the combo, height means repeat count, so this becomes
  h applications of `+2` growth with alternating striped decoration.
- Filling a whole shell with one color no longer holds because D12 requires some
  ring cells to stay background 0.

d) Placement legality
- The generator places exactly one rectangular stripe block onto a 0 canvas with
  positive margins, so there is no overlap and the crop is unambiguous.
- Output construction is purely centered nested wrapping, so no self-overlap or
  out-of-bounds placement is possible.
- Output size is `1 + 2 * sum(heights)` and the generator enforces that it stays
  within the 30x30 ARC limit.

e) Visual observability
- A human can see the same stripe stack input object across examples.
- In the output, the center is always 0, immediately signaling that the original
  solid-center rule no longer applies.
- Heights > 1 now create repeated same-color rings rather than one thick band.
- Alternating horizontal/vertical striping leaves regular 0 slits in the rings, so
  repeated 1-pixel layers are visible without introducing arbitrary separators.

f) Visual cleanliness
- There is still only one output object: a single concentric square construction.
- Background 0 has one coherent role across input and output: canvas background,
  hollow center, and stripe alternation partner.
- The stripe pattern is deterministic and regular, so the image stays structured
  rather than noisy.

g) Change inferability
- No color remapping is introduced: every stripe color stays attached to its own
  rings.
- The only special color is background 0, but its role is spatially explicit:
  outside the object, at the center hole, and alternating within ring decoration.
- Because orientation alternates by depth, repeated examples reveal that ring order
  matters geometrically rather than numerically by color value.

Post-change solver assumptions that the generator must satisfy
--------------------------------------------------------------
1. Background is 0 and there is exactly one non-zero stripe block.
2. Inside that block, each row is uniform and adjacent runs of equal color have
   already been merged into a single stripe height.
3. Stripe colors are non-zero and adjacent stripes use different colors.
4. At least one stripe height is > 1, so the "height -> repeat count" rule is
   visually exercised.
5. The total number of repeated unit layers is at least 4, so both stripe
   orientations appear.
The generator below enforces all five conditions and also rejects samples whose
output accidentally matches simpler non-combo constructions.
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


def _parse_runs(block: np.ndarray) -> list[tuple[int, int]]:
    """Parse the cropped stripe block bottom-to-top as (color, height)."""
    g = np.array(block, dtype=int)
    h, _w = g.shape
    runs: list[tuple[int, int]] = []
    r = int(h - 1)
    while r >= 0:
        row = g[r]
        color = int(row[0])
        if color == BG:
            raise ValueError("background row inside stripe block")
        if not bool((row == color).all()):
            raise ValueError("non-uniform row inside stripe block")

        run_h = 1
        rr = int(r - 1)
        while rr >= 0 and bool((g[rr] == color).all()):
            run_h += 1
            rr -= 1
        runs.append((color, run_h))
        r = rr

    if not runs:
        raise ValueError("no stripe runs found")
    return runs


def _wrap_striped_unit_frame(cur: np.ndarray, color: int, horizontal: bool) -> np.ndarray:
    """Add one 1-pixel striped frame around `cur` using `color` and background 0."""
    cur_n = int(cur.shape[0])
    new_n = int(cur_n + 2)
    nxt = np.full((new_n, new_n), BG, dtype=int)

    rows = np.arange(new_n)[:, None]
    cols = np.arange(new_n)[None, :]
    ring_mask = np.ones((new_n, new_n), dtype=bool)
    ring_mask[1 : 1 + cur_n, 1 : 1 + cur_n] = False

    if horizontal:
        stripe_mask = (np.minimum(rows, new_n - 1 - rows) % 2) == 0
    else:
        stripe_mask = (np.minimum(cols, new_n - 1 - cols) % 2) == 0

    nxt[ring_mask & stripe_mask] = int(color)
    nxt[1 : 1 + cur_n, 1 : 1 + cur_n] = cur
    return nxt


def _build_combo_output(runs: list[tuple[int, int]]) -> np.ndarray:
    """Build the combo output from bottom-to-top stripe runs."""
    cur = np.full((1, 1), BG, dtype=int)
    layer_idx = 0
    for color, repeats in runs:
        for _ in range(int(repeats)):
            layer_idx += 1
            cur = _wrap_striped_unit_frame(
                cur,
                color=int(color),
                horizontal=bool(layer_idx % 2 == 1),
            )
    return cur


def _build_original_output(runs: list[tuple[int, int]]) -> np.ndarray:
    """Reference for degeneration checks: the source task's solid nested frames."""
    center_color, center_size = runs[0]
    cur = np.full((int(center_size), int(center_size)), int(center_color), dtype=int)
    for color, thickness in runs[1:]:
        t = int(thickness)
        nxt = np.full((cur.shape[0] + 2 * t, cur.shape[1] + 2 * t), int(color), dtype=int)
        nxt[t : t + cur.shape[0], t : t + cur.shape[1]] = cur
        cur = nxt
    return cur


def _build_hollow_solid_unit_output(runs: list[tuple[int, int]]) -> np.ndarray:
    """Reference for degeneration checks: hollow center + repeated unit frames, but no stripes."""
    cur = np.full((1, 1), BG, dtype=int)
    for color, repeats in runs:
        for _ in range(int(repeats)):
            nxt = np.full((cur.shape[0] + 2, cur.shape[1] + 2), int(color), dtype=int)
            nxt[1:-1, 1:-1] = cur
            cur = nxt
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build dense hollow striped nested frames from a stack of horizontal stripes.

    Combo rule:
    - Crop the unique non-zero stripe block from the background canvas.
    - Read stripe runs from bottom to top.
    - Start from a 1x1 background center.
    - Each stripe of height h contributes h separate 1-pixel frames in that color.
    - Unit-frame orientations alternate by depth from the center outward:
      horizontal, vertical, horizontal, vertical, ...
    """
    block = _extract_stripe_block(np.array(grid, dtype=int))
    runs = _parse_runs(block)
    return _build_combo_output(runs)


def _combo_metrics(out: np.ndarray) -> dict[str, int]:
    """
    Quantify observable combo behavior for generator validation.

    - `hole_cells`: number of 0 cells in the central connected hollow.
    - `interior_bg_cells`: 0 cells away from the border; this should exceed the
      central hole count once striped rings are present.
    """
    n = int(out.shape[0])
    hole_cells = 1 if int(out[n // 2, n // 2]) == BG else 0
    interior_bg_cells = int(np.count_nonzero(out[1:-1, 1:-1] == BG)) if n > 2 else 0
    return {
        "hole_cells": hole_cells,
        "interior_bg_cells": interior_bg_cells,
    }


def _gen_attempt() -> np.ndarray | None:
    n_runs = int(np.random.randint(3, 7))
    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    total_layers = int(sum(heights))
    out_size = int(1 + 2 * total_layers)
    if out_size > 30 or total_layers < 4:
        return None
    if all(h == 1 for h in heights):
        return None

    stripe_h = int(sum(heights))
    stripe_w = int(np.random.randint(7, 16))
    top = int(np.random.randint(1, 4))
    bottom = int(np.random.randint(1, 4))
    left = int(np.random.randint(1, 4))
    right = int(np.random.randint(1, 4))
    canvas_h = int(stripe_h + top + bottom)
    canvas_w = int(stripe_w + left + right)
    if canvas_h > 30 or canvas_w > 30:
        return None

    colors = np.random.choice(np.arange(1, 10), size=n_runs, replace=False).astype(int).tolist()
    runs = list(zip(colors, heights))
    stripes = [
        np.full((int(run_h), stripe_w), int(color), dtype=int)
        for color, run_h in reversed(runs)
    ]
    block = np.vstack(stripes)

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    canvas[top : top + stripe_h, left : left + stripe_w] = block

    expected = _build_combo_output(runs)
    actual = main(canvas)
    if not np.array_equal(actual, expected):
        return None
    if actual.shape != (out_size, out_size):
        return None

    metrics = _combo_metrics(actual)
    if metrics["hole_cells"] != 1:
        return None
    if metrics["interior_bg_cells"] <= metrics["hole_cells"]:
        return None
    if np.array_equal(actual, _build_original_output(runs)):
        return None
    if np.array_equal(actual, _build_hollow_solid_unit_output(runs)):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
