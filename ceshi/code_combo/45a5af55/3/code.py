# Combo variant of 45a5af55 (index 3)
# combo_spec: D2/0+D7/0
# Name: 菱形紧凑裁剪嵌套

from __future__ import annotations

import numpy as np

# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
# - The input format stays the same as the source task: full-width horizontal
#   stripes of distinct non-zero colors, stacked vertically.
# - We still read stripes bottom -> top, and each stripe height is still the
#   controlling quantity. What changes is the geometry of each wrapping step and
#   the output sizing rule.
# - Source task:
#   runs = [(4, 2), (7, 1), (3, 2)]  # bottom -> top
#   4 with h=2 -> 2x2 solid square
#   7 with h=1 -> wrap a 1-thick square frame => 4x4
#   3 with h=2 -> wrap a 2-thick square frame => 8x8
# - Combo task:
#   4 with h=2 -> solid diamond of Manhattan radius 1
#   7 with h=1 -> add the radius-2 shell
#   3 with h=2 -> add shells at radii 3 and 4
#   This gives a diamond-shaped nesting in a 9x9 bounding box. We then apply a
#   tight crop to non-zero cells. For centered diamonds the crop equals the
#   outer diamond's enclosing rectangle, so the final result keeps diagonal
#   edges and zero-valued corners but has no removable all-zero border rows or
#   columns.
#
# b) Complexity calibration
# - Heights remain in 1..4 and colors remain unchanged, so the main inference
#   burden still comes from understanding that stripe height controls geometry.
# - To keep the diamond rule visually obvious, the generator rejects the fully
#   trivial regime where every stripe has height 1; at least one layer has
#   thickness >= 2.
# - The generator also caps sum(heights) so the final cropped bounding box
#   2 * sum(heights) - 1 stays within normal ARC sizes.
#
# c) Implicit assumption review against the original main()
# - Parsing from the bottom still assumes each row is uniform across the whole
#   width. That remains valid because the input format is unchanged.
# - The original `cur = full((h0, h0), c0)` assumed the center object is an
#   axis-aligned square. That fails here; the center is instead a solid diamond
#   of radius h0 - 1 inside its own bounding box.
# - The original `nxt = full((new_n, new_n), col)` assumed every cell in the
#   expanded bounding box belongs to the new frame color. For diamonds this is
#   false: corner cells outside the current outer diamond must remain 0.
# - The original rectangular overwrite `nxt[h:h+..., h:h+...] = cur` was safe
#   for solid square interiors. It is unsafe for diamonds because `cur` contains
#   zero corners inside its bbox; a blind overwrite could punch holes into a new
#   ring. The combo solver therefore overlays only non-zero cells.
# - D7 adds a new output-shape assumption: the returned grid must already be
#   cropped to the non-zero bbox. main() now enforces that explicitly with a
#   crop helper, and generate_input() validates that the crop does not collapse
#   the intended diamond.
#
# d) Placement legality
# - There is only one output object family: concentric Manhattan diamonds around
#   a single center. Rings are defined by disjoint L1-distance intervals, so
#   they never overlap illegally.
# - The generator only emits non-zero stripe colors, so output background 0 is
#   unambiguous and the crop target is well-defined.
#
# e) Visual observability
# - Compared with the source task, diagonal boundaries and zero-valued corners
#   make the geometric change immediately visible.
# - Compared with an uncropped padded construction, the result has no fully
#   blank outer border rows or columns. This is the observable effect of D7 in
#   this combination.
#
# f) Visual cleanliness
# - The output contains only one nested structure, with one color per radial
#   band. There are no overlapping annotations or competing marks.
#
# g) Change inferability
# - No color remapping is introduced. Colors appear in the output in the same
#   bottom->top stripe order as the source task, so the only new inference is
#   the geometric reinterpretation plus the tight-crop invariant.
#
# New solver assumptions and generator guarantees
# ----------------------------------------------
# - main() assumes at least one stripe run exists; generate_input() always
#   creates 2..6 runs.
# - main() assumes all stripe rows are uniform and consecutive equal-color rows
#   form one run; generate_input() builds exactly that structure.
# - main() assumes 0 is reserved for output background; generate_input() samples
#   colors from 1..9 only.
# - main() assumes the intended object is centered and uses Manhattan shells;
#   generate_input() only needs to supply stripe heights because the geometry is
#   reconstructed deterministically by the solver.


def _parse_runs_bottom_to_top(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    runs: list[tuple[int, int]] = []
    r = int(h - 1)
    while r >= 0:
        col = int(g[r, 0])
        assert bool((g[r] == col).all())
        run_h = 1
        rr = int(r - 1)
        while rr >= 0 and bool((g[rr] == col).all()):
            run_h += 1
            rr -= 1
        runs.append((col, run_h))
        r = rr
    return runs


def _distance_grid(radius: int) -> np.ndarray:
    size = int(2 * radius + 1)
    center = int(radius)
    ys, xs = np.indices((size, size))
    return np.abs(ys - center) + np.abs(xs - center)


def _tight_crop_nonzero(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    ys, xs = np.nonzero(g)
    if len(ys) == 0:
        return np.zeros((0, 0), dtype=int)
    return g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested Manhattan-diamond frames from bottom-up stripe runs, then
    return the non-zero tight crop.
    """
    runs = _parse_runs_bottom_to_top(np.array(grid, dtype=int))
    assert not (not runs)

    c0, h0 = runs[0]
    radius = int(h0 - 1)
    cur = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=int)
    cur[_distance_grid(radius) <= radius] = int(c0)

    for col, h in runs[1:]:
        h = int(h)
        old_radius = int(radius)
        radius = int(radius + h)
        dist = _distance_grid(radius)
        nxt = np.zeros(dist.shape, dtype=int)
        ring_mask = (dist <= radius) & (dist > old_radius)
        nxt[ring_mask] = int(col)

        offset = h
        sub = nxt[offset : offset + cur.shape[0], offset : offset + cur.shape[1]]
        sub[cur != 0] = cur[cur != 0]
        cur = nxt

    return _tight_crop_nonzero(cur)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_runs = int(rng.integers(2, 7))
    heights = [int(rng.integers(1, 5)) for _ in range(n_runs)]
    if sum(heights) > 15:
        return None
    if max(heights) == 1:
        return None

    width = int(rng.integers(6, 15))

    cols: list[int] = []
    prev = None
    for _ in range(n_runs):
        c = int(rng.integers(1, 10))
        while prev is not None and c == prev:
            c = int(rng.integers(1, 10))
        cols.append(c)
        prev = c

    stripes = [
        np.full((run_h, width), col, dtype=int)
        for col, run_h in reversed(list(zip(cols, heights)))
    ]
    inp = np.vstack(stripes)

    out = main(inp)
    if out.shape[0] > 29 or out.shape[1] > 29:
        return None

    # Keep the fusion observable: we want an actual diamond silhouette with
    # corner background, but no removable all-zero outer border after cropping.
    if not np.any(out == 0):
        return None
    if int(out[0, 0]) != 0 or int(out[0, -1]) != 0 or int(out[-1, 0]) != 0 or int(out[-1, -1]) != 0:
        return None
    if np.all(out[0] == 0) or np.all(out[-1] == 0):
        return None
    if np.all(out[:, 0] == 0) or np.all(out[:, -1] == 0):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
