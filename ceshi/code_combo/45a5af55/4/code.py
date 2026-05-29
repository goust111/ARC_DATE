# Combo variant of 45a5af55 (index 4)
# combo_spec: D2/1+D9/2
# Name: 十字框架去角嵌套

from __future__ import annotations

import numpy as np

"""
Design analysis for the fused variant.

a) Concrete behavior and trace
   Bottom->top runs still encode layer colors and thicknesses. The fusion has
   two stages, and both must be visible.

   Example runs: [(2, 2), (5, 1), (7, 2)].
   Seed task:
   - center = 2x2 solid color 2
   - wrap a 1-thick square frame of color 5
   - wrap a 2-thick square frame of color 7
   - result is a fully nested square pattern.

   D2-only stage-1 cross construction:
   - center stays the same 2x2 solid block
   - color 5 adds four orthogonal 1-thick arms
   - color 7 adds four orthogonal 2-thick arms
   - full stage-1 cross becomes
       0 0 7 7 7 7 0 0
       0 0 7 7 7 7 0 0
       7 7 0 5 5 0 7 7
       7 7 5 2 2 5 7 7
       7 7 5 2 2 5 7 7
       7 7 0 5 5 0 7 7
       0 0 7 7 7 7 0 0
       0 0 7 7 7 7 0 0

   Fused stage-2 symmetric pruning:
   - locate the centered band occupied by the innermost solid square
   - keep only cells lying in that horizontal band or that vertical band
   - delete all colored cells in the four off-axis corner regions
   - final output becomes
       0 0 0 7 7 0 0 0
       0 0 0 7 7 0 0 0
       0 0 0 5 5 0 0 0
       7 7 5 2 2 5 7 7
       7 7 5 2 2 5 7 7
       0 0 0 5 5 0 0 0
       0 0 0 7 7 0 0 0
       0 0 0 7 7 0 0 0

   Difference:
   - compared with the seed, stage 1 already changes square rings into crosses
   - compared with D2-only, stage 2 removes the diagonal side blocks and keeps
     only a tighter plus sign aligned to the center square's row/column span.

b) Complexity calibration
   The task is still a single centered object with few free variables:
   stripe order, stripe heights, and stripe colors. The generator uses 3-6
   stripes so the nested growth and the later pruning both have repeated visual
   evidence. Heights remain 1..4 to preserve the seed task's scale.

c) Implicit-assumption audit of the seed solver
   The original parsing logic assumed:
   - each row is monochrome across the full width
   - equal-color rows form contiguous stripes
   - every new layer fills the entire outer square annulus
   The first two assumptions still hold and are preserved by generate_input().
   The third is false twice over in the combo:
   - D2 breaks it because outer layers are crosses, not full square rings
   - D9 breaks it again because even those cross arms get pruned afterward
   Therefore main() must:
   1) parse runs exactly as before
   2) build the full nested cross explicitly on a zero background
   3) rebuild the final answer through a keep-mask aligned to the center square
   A naive "fill outer square with the layer color" or "just reuse D2 output"
   would miss the fused second step.

d) Placement legality
   Input legality is unchanged: the generator emits full-width stacked stripes,
   so there is no overlap and no ambiguity. Output legality is guaranteed by the
   same size equation as the seed task:
     final_side = center_height + 2 * sum(other_heights)
   Cross arms are painted into disjoint margin rectangles around the previous
   centered shape, then pruning only deletes cells from that finished canvas.

e) Visual observability
   A human can infer the rule from a few examples because:
   - stripe heights still control layer thickness
   - stage 1 leaves visible cross-like intermediate geometry
   - stage 2 consistently removes four symmetric off-axis regions
   Background 0 is never used as an input stripe color, so the deletions are
   explicit and visually attributable to the pruning step.

f) Visual cleanliness
   The output contains one centered plus-shaped composite with no overlaps among
   unrelated objects. Colors remain attached to their layer distances, and the
   removed corner regions are empty rather than overwritten by new symbols.

g) Change inferability
   No color remapping is introduced. A stripe color remains the layer color.
   The only new semantics are geometric: cross growth followed by symmetric
   deletion outside the center-aligned horizontal/vertical bands.

h) Generator anti-degeneracy checks
   Passing main(generate_input()) is not enough, because some inputs might make
   the fused pruning invisible. The generator therefore rejects instances unless:
   - at least one colored cell is removed in each of the four symmetric corner
     regions relative to the center bands
   - the fused output differs from both the stage-1 cross and the original seed
     nested-square output

i) New solver assumptions to audit
   main() newly assumes:
   - the innermost stripe height defines a centered square that can be located in
     the final canvas by offset = sum(outer stripe heights)
   - keeping rows/cols through that square leaves a non-empty plus sign
   generate_input() enforces positive stripe heights, at least three stripes, and
   final size <= 30, so these assumptions always hold on generated inputs.
"""


def _parse_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return (color, height) runs ordered from bottom to top."""
    g = np.array(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")

    h, w = g.shape
    if h == 0 or w == 0:
        raise ValueError("grid must be non-empty")

    runs: list[tuple[int, int]] = []
    r = h - 1
    while r >= 0:
        color = int(g[r, 0])
        if not bool((g[r] == color).all()):
            raise ValueError("each stripe row must be uniform across the full width")

        run_h = 1
        rr = r - 1
        while rr >= 0:
            if not bool((g[rr] == int(g[rr, 0])).all()):
                raise ValueError("each stripe row must be uniform across the full width")
            if int(g[rr, 0]) != color:
                break
            run_h += 1
            rr -= 1
        runs.append((color, run_h))
        r = rr

    if not runs:
        raise ValueError("expected at least one stripe")
    return runs


def _build_nested_cross(runs: list[tuple[int, int]]) -> np.ndarray:
    """Stage 1: build the full nested cross before pruning."""
    center_color, center_size = runs[0]
    cur = np.full((center_size, center_size), center_color, dtype=int)

    for color, thick in runs[1:]:
        inner_side = int(cur.shape[0])
        new_side = int(inner_side + 2 * thick)
        nxt = np.zeros((new_side, new_side), dtype=int)

        nxt[0:thick, thick : thick + inner_side] = color
        nxt[thick + inner_side : new_side, thick : thick + inner_side] = color
        nxt[thick : thick + inner_side, 0:thick] = color
        nxt[thick : thick + inner_side, thick + inner_side : new_side] = color
        nxt[thick : thick + inner_side, thick : thick + inner_side] = cur
        cur = nxt

    return cur


def _build_nested_square(runs: list[tuple[int, int]]) -> np.ndarray:
    """Seed-task geometry, used only for anti-degeneracy checks in generation."""
    center_color, center_size = runs[0]
    cur = np.full((center_size, center_size), center_color, dtype=int)

    for color, thick in runs[1:]:
        new_side = int(cur.shape[0] + 2 * thick)
        nxt = np.full((new_side, new_side), color, dtype=int)
        nxt[thick : thick + cur.shape[0], thick : thick + cur.shape[1]] = cur
        cur = nxt

    return cur


def _center_band_geometry(runs: list[tuple[int, int]]) -> tuple[int, int]:
    """Return (offset, size) of the center square in final-canvas coordinates."""
    center_size = int(runs[0][1])
    offset = int(sum(height for _color, height in runs[1:]))
    return offset, center_size


def _prune_to_centered_plus(full_cross: np.ndarray, runs: list[tuple[int, int]]) -> np.ndarray:
    """
    Stage 2: keep only the horizontal and vertical bands passing through the
    centered inner square; delete the four off-axis corner regions.
    """
    offset, center_size = _center_band_geometry(runs)
    out = np.zeros_like(full_cross)
    keep = np.zeros_like(full_cross, dtype=bool)
    keep[offset : offset + center_size, :] = True
    keep[:, offset : offset + center_size] = True
    out[keep] = full_cross[keep]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    1) parse bottom->top full-width stripes
    2) build the D2 nested cross
    3) prune it to the centered plus by deleting the four symmetric off-axis
       corner regions
    """
    runs = _parse_runs(np.array(grid, dtype=int))
    full_cross = _build_nested_cross(runs)
    return _prune_to_centered_plus(full_cross, runs)


def _corner_removal_counts(full_cross: np.ndarray, pruned: np.ndarray, runs: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Count removed colored cells in the four symmetric corner regions."""
    offset, center_size = _center_band_geometry(runs)
    end = int(offset + center_size)
    removed = (full_cross != 0) & (pruned == 0)
    return (
        int(np.count_nonzero(removed[:offset, :offset])),
        int(np.count_nonzero(removed[:offset, end:])),
        int(np.count_nonzero(removed[end:, :offset])),
        int(np.count_nonzero(removed[end:, end:])),
    )


def _expected_output_from_runs(runs: list[tuple[int, int]]) -> np.ndarray:
    return _prune_to_centered_plus(_build_nested_cross(runs), runs)


def _gen_attempt() -> np.ndarray | None:
    width = int(np.random.randint(7, 16))
    n_runs = int(np.random.randint(3, 7))
    heights = [int(np.random.randint(1, 5)) for _ in range(n_runs)]
    final_side = int(heights[0] + 2 * sum(heights[1:]))
    if final_side > 30:
        return None

    colors: list[int] = []
    prev = None
    for _ in range(n_runs):
        color = int(np.random.randint(1, 10))
        while prev is not None and color == prev:
            color = int(np.random.randint(1, 10))
        colors.append(color)
        prev = color

    runs = list(zip(colors, heights))
    stripes = [
        np.full((height, width), color, dtype=int)
        for color, height in reversed(runs)
    ]
    inp = np.vstack(stripes)

    actual = main(inp)
    expected = _expected_output_from_runs(runs)
    if not np.array_equal(actual, expected):
        return None
    if actual.shape[0] > 30:
        return None

    full_cross = _build_nested_cross(runs)
    seed_square = _build_nested_square(runs)
    if np.array_equal(actual, full_cross):
        return None
    if np.array_equal(actual, seed_square):
        return None

    corner_removed = _corner_removal_counts(full_cross, actual, runs)
    if min(corner_removed) <= 0:
        return None

    removed_total = int(np.count_nonzero((full_cross != 0) & (actual == 0)))
    if removed_total < 4:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
