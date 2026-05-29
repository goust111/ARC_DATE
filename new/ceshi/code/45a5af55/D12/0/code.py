# Variant of 45a5af55
# Dimension: D12 — 框架层的装饰模式 (transform)
# Change: 每层框架为纯色填充（单一颜色的均匀方形环）... → 每层框架为交替色条纹（该层颜色与背景色交替）

from __future__ import annotations

import numpy as np


# Design analysis for this D12 variant
# ------------------------------------
# 1) Concrete behavior
# - Input stays the same as the source task: a full grid of full-width horizontal
#   color stripes, each stripe being a single non-zero color and having height 1-4.
# - Parsing is still bottom -> top. The bottom stripe becomes the center solid
#   square with side length equal to its stripe height.
# - Each later stripe still wraps the current square with a frame of thickness
#   equal to that stripe's height, but the frame is no longer solid:
#   odd-numbered frame layers from the center use horizontal 1-cell stripes,
#   even-numbered frame layers use vertical 1-cell stripes.
#   In both cases the frame alternates between that layer's color and background 0.
# - Small trace:
#   source logic with runs [(2, 2), (5, 1)] builds a 2x2 center of color 2, then a
#   solid 1-thick frame of color 5 around it.
#   variant logic builds the same 2x2 center, then a 4x4 frame where row 0 and row 3
#   are color 5 while the side cells on row 1/2 are background 0, because layer 1 is
#   horizontal-striped. The geometric nesting is unchanged; only the ring fill changes.
#
# 2) Complexity calibration
# - The new rule adds only one extra degree of freedom: frame decoration orientation
#   alternates by layer parity. This is visually obvious and repeated across layers.
# - The stripe period is fixed at 1 cell, avoiding extra hidden parameters.
# - Generator uses at least 3 runs so outputs usually contain both horizontal and
#   vertical striped rings, making the pattern inferable from few examples.
#
# 3) Hidden-assumption audit of the original main()
# - Parsing rows as uniform stripes still holds because input construction is unchanged.
# - The original `nxt = full(color)` assumed every cell of a new frame ring should be
#   overwritten by one solid color. That assumption breaks here because some ring cells
#   must remain background 0. We therefore initialize new layers with 0 and paint only
#   the colored stripes of the ring before pasting the previous square in the center.
# - The center insertion `nxt[h:h+cur_n, h:h+cur_n] = cur` remains valid because the
#   inner square is still copied verbatim and must overwrite any temporary background.
#
# 4) Placement legality
# - Input stripes still cover the full width and stack with no gaps or overlap.
# - Output size remains `h0 + 2 * sum(other heights)`, exactly as in the source task,
#   so no frame extends out of bounds.
# - Because the generator keeps adjacent stripe colors distinct and all stripe colors
#   non-zero, the parser can recover runs unambiguously.
#
# 5) Visual observability and cleanliness
# - Alternating with 0 makes the new decoration immediately different from the source
#   task's solid rings.
# - Requiring at least two frame layers ensures both orientations can appear within one
#   sample. The stripe pattern is regular and does not overlap unrelated objects, so the
#   image stays clean rather than cluttered.
#
# 6) Color inference
# - Only one new color role is introduced: background 0 inside frame stripes.
#   The original stripe color is still the "active" layer color, so the mapping remains
#   structured and easy to infer from position and repetition.
#
# 7) New assumptions introduced here
# - `main()` now assumes the intended background for decorative stripes is color 0.
#   `generate_input()` enforces non-zero stripe colors so this role stays distinct.
# - `main()` alternates orientation by frame index counted from the center outward.
#   `generate_input()` ensures multiple frames are common enough for that rule to be
#   visible and not collapse into a one-layer edge case.


BG = 0


def _parse_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    """Parse full-width stripe runs from bottom to top as (color, height)."""
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    runs: list[tuple[int, int]] = []
    r = h - 1
    while r >= 0:
        col = int(g[r, 0])
        run_h = 1
        rr = r - 1
        while rr >= 0 and bool((g[rr] == col).all()):
            run_h += 1
            rr -= 1
        runs.append((col, run_h))
        r = rr
    return runs


def _wrap_with_striped_frame(
    cur: np.ndarray, color: int, thickness: int, horizontal: bool
) -> np.ndarray:
    """Wrap `cur` with a striped square ring of `color`/BG."""
    cur_n = int(cur.shape[0])
    new_n = int(cur_n + 2 * thickness)
    nxt = np.full((new_n, new_n), BG, dtype=int)

    rows = np.arange(new_n)[:, None]
    cols = np.arange(new_n)[None, :]
    ring_mask = np.ones((new_n, new_n), dtype=bool)
    ring_mask[thickness : thickness + cur_n, thickness : thickness + cur_n] = False

    if horizontal:
        stripe_mask = (np.minimum(rows, new_n - 1 - rows) % 2) == 0
    else:
        stripe_mask = (np.minimum(cols, new_n - 1 - cols) % 2) == 0

    nxt[ring_mask & stripe_mask] = int(color)
    nxt[thickness : thickness + cur_n, thickness : thickness + cur_n] = cur
    return nxt


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested square frames from a stack of horizontal color stripes.

    D12 variant:
    - The center square is still solid.
    - Frame layers alternate their decoration by depth:
      layer 1 (nearest the center) = horizontal stripes of (layer color, 0),
      layer 2 = vertical stripes, then alternate outward.
    """
    runs = _parse_runs(np.array(grid, dtype=int))
    assert runs

    center_color, center_size = runs[0]
    cur = np.full((center_size, center_size), center_color, dtype=int)

    for layer_idx, (color, thickness) in enumerate(runs[1:], start=1):
        cur = _wrap_with_striped_frame(
            cur=cur,
            color=int(color),
            thickness=int(thickness),
            horizontal=bool(layer_idx % 2 == 1),
        )

    return cur


def _gen_attempt() -> np.ndarray | None:
    width = int(np.random.randint(8, 16))
    n_runs = int(np.random.randint(3, 7))
    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    out_size = int(heights[0] + 2 * sum(heights[1:]))
    if out_size > 30 or out_size < 9:
        return None

    colors: list[int] = []
    prev = None
    for _ in range(n_runs):
        color = int(np.random.randint(1, 10))
        while prev is not None and color == prev:
            color = int(np.random.randint(1, 10))
        colors.append(color)
        prev = color

    stripes = [
        np.full((height, width), color, dtype=int)
        for color, height in reversed(list(zip(colors, heights)))
    ]
    inp = np.vstack(stripes)

    # Self-check: generated examples must actually exhibit the D12 behavior.
    out = main(inp)
    if out.shape != (out_size, out_size):
        return None
    if not np.any(out == BG):
        return None

    # Require both frame orientations to be present in the output-generating logic.
    if len(colors) < 3:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
