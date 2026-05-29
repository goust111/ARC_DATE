# Combo variant of 45a5af55 (index 8)
# combo_spec: D3/1+D12/0
# Name: 薄层条纹框架嵌套

from __future__ import annotations

import numpy as np


# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior / trace
# - Input stays a stack of full-width horizontal stripes, parsed bottom -> top as
#   `(color, height)` runs.
# - Original task:
#   bottom `(c0, h0)` -> solid `h0 x h0` center square.
#   each later `(c, h)` -> one solid square frame of thickness `h`.
# - Fused rule:
#   bottom `(c0, h0)` still becomes the solid center square.
#   each later `(c, h)` no longer means "one thick frame of width h". Instead it
#   means "repeat color c for h consecutive 1-pixel frames".
#   Every such 1-pixel frame is decorated with alternating color/background stripes:
#     odd layer from the center -> horizontal stripe ring,
#     even layer from the center -> vertical stripe ring.
# - Small trace with runs `[(2, 2), (5, 3), (7, 1)]`:
#   original:
#     `2x2` center -> add one solid color-5 frame of thickness 3 -> add one solid
#     color-7 frame of thickness 1.
#   fused:
#     `2x2` center
#     -> layer 1: color-5 horizontal striped 1px ring
#     -> layer 2: color-5 vertical striped 1px ring
#     -> layer 3: color-5 horizontal striped 1px ring
#     -> layer 4: color-7 vertical striped 1px ring
#   Difference:
#     stripe height is now a repetition count, not a visible band thickness, and
#     the decoration makes those repeated same-color layers visually separable
#     without inserting artificial gaps.
#
# b) Complexity calibration
# - Heights remain in `1..4` and run counts remain small, so the puzzle is still
#   readable.
# - The only new free choices are:
#   1. height reinterpreted as repeat count;
#   2. striped orientation alternating by actual layer depth.
#   Both are low-entropy, repeated across the image, and visible in 2-4 examples.
# - The generator forces at least one outer height > 1 and at least two outer
#   layers total, so the fused behavior cannot collapse into the original task.
#
# c) Hidden-assumption audit of the original `main()`
# - Parsing rows by reading `g[r, 0]` and checking whole-row equality still works;
#   input construction is unchanged and every row remains a uniform full-width stripe.
# - `cur = full((h0, h0), c0)` still holds: the bottom stripe still defines the
#   center solid square.
# - The original `new_n = cur.shape[0] + 2 * h` followed by filling the whole ring
#   with `color` relied on the assumption that stripe height equals one solid frame
#   thickness. That breaks here in two ways:
#   1. height now means "how many 1px wraps to perform";
#   2. each wrap is only partially colored because its ring alternates with BG=0.
# - Therefore we replace the single thick wrap with `h` repeated unit wraps, and we
#   count stripe orientation by the global frame-layer index rather than by input run.
#
# d) Placement legality
# - Input stripes are full-width and stacked vertically, so there is no overlap or
#   ambiguity in the input.
# - Each output step expands the current square by exactly 1 pixel on every side, so
#   repeated nesting is centered and cannot self-overlap incorrectly.
# - Output size is `center_h + 2 * sum(outer_heights)`, which stays within bounds by
#   generator filtering.
#
# e) Visual observability
# - Human solvers can see two coupled signals:
#   1. a taller stripe produces more nested copies of the same color;
#   2. the copies alternate between horizontal and vertical stripe rings.
# - The generator enforces at least one height > 1, ensuring the height-to-repeat
#   reinterpretation is visible instead of degenerating into a single ring.
# - Background 0 appears inside the striped frames, so the decoration is obvious and
#   not confusable with the original solid-band output.
#
# f) Visual cleanliness
# - Only three visual roles appear: solid center square, colored stripe segments, and
#   background 0 inside the ring decoration.
# - Because frames remain perfectly nested and centered, the output stays structured
#   rather than noisy even when there are many layers.
#
# g) Change inferability
# - No arbitrary color remapping is introduced: each stripe keeps its own color.
# - The new behavior depends on position in the nesting sequence, not on hidden color
#   values, so it is inferable from geometry alone.
#
# New assumptions introduced by the fused solver
# - Background 0 is reserved for the stripe alternation inside decorative rings.
#   The generator therefore uses only input colors 1..9.
# - The parity rule is defined over actual 1px frame layers from the center outward.
#   The generator guarantees there are enough layers for both parities to appear.


BG = 0


def _parse_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    """Parse full-width uniform stripe runs from bottom to top as (color, height)."""
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    runs: list[tuple[int, int]] = []
    r = h - 1
    while r >= 0:
        row = g[r]
        color = int(row[0])
        assert color != BG
        assert bool((row == color).all())
        run_h = 1
        rr = r - 1
        while rr >= 0 and bool((g[rr] == color).all()):
            run_h += 1
            rr -= 1
        runs.append((color, run_h))
        r = rr
    assert runs
    return runs


def _wrap_with_striped_unit_frame(
    cur: np.ndarray, color: int, horizontal: bool
) -> np.ndarray:
    """Wrap `cur` with one 1-pixel striped ring in `color`/BG."""
    cur_n = int(cur.shape[0])
    new_n = cur_n + 2
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


def _build_output_from_runs(runs: list[tuple[int, int]]) -> np.ndarray:
    """Apply the fused nesting rule to parsed `(color, height)` runs."""
    center_color, center_size = runs[0]
    cur = np.full((int(center_size), int(center_size)), int(center_color), dtype=int)

    layer_idx = 1
    for color, repeats in runs[1:]:
        for _ in range(int(repeats)):
            cur = _wrap_with_striped_unit_frame(
                cur=cur,
                color=int(color),
                horizontal=bool(layer_idx % 2 == 1),
            )
            layer_idx += 1

    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build dense nested striped frames from horizontal input stripes.

    Fused rule:
    - Bottom stripe -> center solid square with side = stripe height.
    - Every later stripe of height h contributes h consecutive 1-pixel square rings.
    - Ring decoration alternates by actual layer depth:
      layer 1 from the center = horizontal stripes of (color, 0),
      layer 2 = vertical stripes, and so on outward.
    """
    runs = _parse_runs(np.array(grid, dtype=int))
    return _build_output_from_runs(runs)


def _original_solid_output(runs: list[tuple[int, int]]) -> np.ndarray:
    """Reference construction for the source task, used only by generator checks."""
    center_color, center_size = runs[0]
    cur = np.full((int(center_size), int(center_size)), int(center_color), dtype=int)
    for color, thickness in runs[1:]:
        new_n = int(cur.shape[0] + 2 * thickness)
        nxt = np.full((new_n, new_n), int(color), dtype=int)
        nxt[thickness : thickness + cur.shape[0], thickness : thickness + cur.shape[1]] = cur
        cur = nxt
    return cur


def _gen_attempt() -> np.ndarray | None:
    width = int(np.random.randint(7, 16))
    n_runs = int(np.random.randint(3, 7))

    center_h = int(np.random.randint(1, 5))
    outer_heights = [int(np.random.randint(1, 5)) for _ in range(n_runs - 1)]
    if all(h == 1 for h in outer_heights):
        outer_heights[int(np.random.randint(0, len(outer_heights)))] = int(np.random.randint(2, 5))
    heights = [center_h, *outer_heights]

    total_outer_layers = int(sum(outer_heights))
    output_size = int(center_h + 2 * total_outer_layers)
    if output_size > 30 or output_size < 8:
        return None

    colors = np.random.choice(np.arange(1, 10), size=n_runs, replace=False).astype(int).tolist()
    runs = list(zip(colors, heights))

    stripes = [
        np.full((int(run_h), width), int(color), dtype=int)
        for color, run_h in reversed(runs)
    ]
    inp = np.vstack(stripes)

    out = main(inp)
    if out.shape != (output_size, output_size):
        return None

    # Fusion-specific observability guards:
    # - striped decoration must introduce background cells;
    # - at least one stripe must use height as repetition count;
    # - there must be enough outer layers for both stripe orientations to appear;
    # - the fused output must not collapse to the original solid construction.
    if not np.any(out == BG):
        return None
    if max(outer_heights) <= 1:
        return None
    if total_outer_layers < 2:
        return None
    if np.array_equal(out, _original_solid_output(runs)):
        return None

    # Solver/generator consistency check.
    if not np.array_equal(out, _build_output_from_runs(runs)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
