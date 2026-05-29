# Combo variant of 45a5af55 (index 7)
# combo_spec: D3/2+D7/2
# Name: 双倍厚度定宽嵌套

from __future__ import annotations

import numpy as np

"""
Design analysis for the fused variant.

a) Concrete behavior and trace
   Use bottom->top runs [(2, 1), (5, 2), (7, 1)] and input width W = 9.
   Original task:
   - center = 1x1 color 2
   - add color-5 frame of thickness 2 -> 5x5
   - add color-7 frame of thickness 1 -> 7x7
   - return the 7x7 square directly
   Fused task:
   - parse the same stripe runs from the non-background rectangle
   - center still = 1x1 color 2
   - add color-5 frame of thickness 4 because thickness = 2 * stripe_height
     -> 9x9
   - add color-7 frame of thickness 2 -> 13x13
   - output side is not 13; it is fixed to the input width W = 9
   - therefore place the 13x13 nest at the center of a 9x9 output and crop the
     outer overflow, yielding the centered 9x9 window of that larger nest
   Difference from the base task: both the growth law and the final canvas-size
   law change. The larger doubled-thickness construction is then filtered through
   the fixed-width canvas, so padding appears when it is smaller than W and
   cropping appears when it is larger than W.

b) Complexity calibration
   The generator keeps a single rectangular stripe stack inside a uniform
   background so the scene stays clean. To make the fusion observable instead of
   collapsing to a near-original sample, it enforces:
   - at least one outer stripe has height >= 2, so doubled thickness is visible;
   - |W - doubled_nested_side| >= 2, so the width-controlled canvas matters;
   - the fused output differs from the D7-only baseline (same fixed canvas but
     original thickness), so D3 does not get hidden by cropping;
   - at least three non-background colors remain visible in the output.

c) Implicit-assumption review of main()
   The original main() assumed:
   - every input row is a full-width monochrome stripe;
   - output size is determined entirely by stripe heights;
   - every constructed layer fits because the output grows with the layers.
   In the fused variant these assumptions fail:
   - background margins mean the stripe block is only a sub-rectangle, so we must
     crop to the non-background bounding box before parsing runs;
   - output side is now the input width, so `return cur` is invalid;
   - doubled thickness can make the constructed square larger than the canvas, so
     `assert n <= W` is no longer legal and must be replaced by centered crop/pad.

d) Placement legality
   The generator places one solid stripe rectangle inside a background canvas with
   positive side margins and vertical margins. The stripe block never overlaps any
   other object, never touches the outer border, and always fits within the input.
   The output is assembled by a pure centered-overlap computation, so it cannot go
   out of bounds regardless of whether padding or cropping occurs.

e) Visual observability
   A human can see the full input width because the background is visible to the
   left and right of the stripe block. The output is always exactly W x W, and
   either shows a background border (padding case) or clipped outer rings
   (cropping case). The doubled-thickness law is supported by requiring an outer
   stripe of height >= 2 and by rejecting any sample where the fused output would
   match the D7-only baseline.

f) Visual cleanliness
   The input contains only two clean regions: a uniform background and a single
   stripe rectangle. The output contains only a centered square nest plus optional
   uniform background padding. No markers, overlaps, or irregular masks are added.

g) Inferrability of colors and structure
   Colors are preserved structurally: each stripe color keeps its layer identity,
   and the corner background color becomes the padding color whenever padding is
   present. No hidden color-id arithmetic is introduced; only visible stripe
   height, nesting order, and canvas width matter.

h) New solver assumptions and generator guarantees
   The fused solver now assumes:
   - one background color occupies all four corners;
   - all non-background cells form a single solid rectangle;
   - every row inside that rectangle is monochrome;
   - adjacent stripe runs use different colors;
   - the intended output is the centered overlap of the doubled-thickness nest
     with a W x W canvas.
   The generator enforces each assumption directly.
"""


def _build_nested_square(
    runs: list[tuple[int, int]],
    *,
    doubled_outer_thickness: bool,
) -> np.ndarray:
    """Build the nested square from bottom->top (color, height) runs."""
    center_color, center_height = runs[0]
    cur = np.full((int(center_height), int(center_height)), int(center_color), dtype=int)

    for color, height in runs[1:]:
        thickness = 2 * int(height) if doubled_outer_thickness else int(height)
        new_side = int(cur.shape[0] + 2 * thickness)
        nxt = np.full((new_side, new_side), int(color), dtype=int)
        nxt[
            thickness : thickness + cur.shape[0],
            thickness : thickness + cur.shape[1],
        ] = cur
        cur = nxt

    return cur


def _infer_bg(grid: np.ndarray) -> int:
    corners = [
        int(grid[0, 0]),
        int(grid[0, -1]),
        int(grid[-1, 0]),
        int(grid[-1, -1]),
    ]
    return max(set(corners), key=corners.count)


def _parse_runs(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    """Parse bottom->top stripe runs from the non-background rectangle."""
    non_bg = grid != int(bg)
    if not bool(non_bg.any()):
        raise ValueError("expected a non-background stripe block")

    rows, cols = np.where(non_bg)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    block = grid[r0 : r1 + 1, c0 : c1 + 1]

    if bool((block == int(bg)).any()):
        raise ValueError("non-background cells must form a solid rectangle")

    runs: list[tuple[int, int]] = []
    r = int(block.shape[0] - 1)
    while r >= 0:
        row = block[int(r)]
        color = int(row[0])
        if color == int(bg):
            raise ValueError("stripe rows cannot use the background color")
        if not bool((row == color).all()):
            raise ValueError("each stripe row must be monochrome")

        height = 1
        rr = int(r - 1)
        while rr >= 0:
            prev_row = block[int(rr)]
            if not bool((prev_row == int(prev_row[0])).all()):
                raise ValueError("each stripe row must be monochrome")
            if int(prev_row[0]) != color:
                break
            height += 1
            rr -= 1
        runs.append((color, height))
        r = int(rr)

    if not runs:
        raise ValueError("expected at least one stripe run")
    return runs


def _place_centered(src: np.ndarray, side: int, bg: int) -> np.ndarray:
    """
    Place a square source into a side x side square canvas by centered overlap.

    If src is smaller, background padding remains around it.
    If src is larger, the centered side x side window of src is returned.
    """
    src_n = int(src.shape[0])
    out = np.full((int(side), int(side)), int(bg), dtype=int)

    if src_n <= int(side):
        dst0 = int((side - src_n) // 2)
        out[dst0 : dst0 + src_n, dst0 : dst0 + src_n] = src
        return out

    src0 = int((src_n - side) // 2)
    return np.array(src[src0 : src0 + side, src0 : src0 + side], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - infer the background from the corners and crop to the stripe rectangle;
    - read stripe runs from bottom to top;
    - center square side = bottom stripe height;
    - every outer frame thickness = 2 * stripe height;
    - final output side = input width W;
    - place the full doubled-thickness nest at the center of a W x W output,
      leaving background padding if it is smaller or cropping outer layers if it
      is larger.
    """
    g = np.array(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        raise ValueError("grid must be a non-empty 2D array")

    _h, w = g.shape
    bg = _infer_bg(g)
    runs = _parse_runs(g, bg)
    full_nest = _build_nested_square(runs, doubled_outer_thickness=True)
    return _place_centered(full_nest, int(w), int(bg))


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.randint(0, 10))

    n_runs = int(np.random.randint(3, 6))
    heights = [int(np.random.randint(1, 4)) for _ in range(n_runs)]
    if max(heights[1:]) < 2:
        return None

    doubled_side = int(heights[0] + 4 * sum(heights[1:]))
    base_side = int(heights[0] + 2 * sum(heights[1:]))
    if doubled_side > 30:
        return None

    mode = "pad" if np.random.rand() < 0.5 else "crop"
    if mode == "pad":
        pad = int(np.random.randint(1, 4))
        w = int(doubled_side + 2 * pad)
    else:
        crop = int(np.random.randint(1, 4))
        w = int(doubled_side - 2 * crop)
    if w < 7 or w > 30:
        return None
    if abs(w - doubled_side) < 2:
        return None

    block_h = int(sum(heights))
    vpad = int(np.random.randint(1, 4))
    h = int(block_h + 2 * vpad)
    if h > 30:
        return None

    max_side_margin = min(4, (w - 1) // 2)
    if max_side_margin < 1:
        return None
    side_margin = int(np.random.randint(1, max_side_margin + 1))
    stripe_w = int(w - 2 * side_margin)
    if stripe_w <= 0:
        return None

    colors: list[int] = []
    used = {int(bg)}
    for _ in range(n_runs):
        color = int(np.random.randint(0, 10))
        while color in used or (colors and color == colors[-1]):
            color = int(np.random.randint(0, 10))
        colors.append(color)
        used.add(color)

    inp = np.full((h, w), int(bg), dtype=int)
    r = int(vpad)
    for color, height in reversed(list(zip(colors, heights))):
        inp[r : r + height, side_margin : side_margin + stripe_w] = int(color)
        r += int(height)

    runs = list(zip(colors, heights))
    fused_inner = _build_nested_square(runs, doubled_outer_thickness=True)
    fused_out = _place_centered(fused_inner, int(w), int(bg))

    # Reject degeneration where D3 disappears under the same width-controlled D7 canvas.
    d7_only_inner = _build_nested_square(runs, doubled_outer_thickness=False)
    d7_only_out = _place_centered(d7_only_inner, int(w), int(bg))
    if np.array_equal(fused_out, d7_only_out):
        return None

    visible_non_bg = {int(x) for x in np.unique(fused_out) if int(x) != int(bg)}
    if len(visible_non_bg) < 3:
        return None

    # Also avoid collapsing too close to the original growth law.
    if w == int(base_side):
        return None

    if not np.array_equal(main(inp), fused_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 600 attempts")
