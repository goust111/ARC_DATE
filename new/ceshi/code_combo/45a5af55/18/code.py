# Combo variant of 45a5af55 (index 18)
# combo_spec: D2/0+D4/1+D7/0
# Name: 菱形棋盘中心紧凑嵌套

from __future__ import annotations

import numpy as np

BG = 0

"""
Design analysis for the fused variant.

a) Concrete behavior trace
   Example bottom->top runs: [(4, 3), (7, 1), (2, 2)].
   Original task:
   - 4 with height 3 -> 3x3 solid square
   - wrap 1-thick square frame of 7 -> 5x5
   - wrap 2-thick square frame of 2 -> 9x9
   Fused task:
   - 4 with height 3 -> center becomes a radius-2 diamond inside a 5x5 bbox
     and only diamond cells participate; within that diamond, cells alternate
     between color 4 and background 0 as a checkerboard.
   - 7 with height 1 -> add the radius-3 diamond shell of color 7.
   - 2 with height 2 -> add radius-4 and radius-5 shells of color 2.
   - finally crop to the non-background bbox of the outer diamond.
   Differences from the source are coupled rather than independent:
   square geometry becomes Manhattan-diamond geometry, the center is no longer
   solid, and the output extent is determined from actual painted cells instead
   of the source task's square-side formula.

b) Complexity calibration
   The input remains a clean stack of full-width stripes, so the only reasoning
   burden comes from reinterpreting stripe heights in a new geometry. To keep the
   checkerboard visible, the generator forces the bottom stripe height >= 3. To
   keep the diamond thickness rule observable, the generator allows 2..6 runs and
   caps total height so outputs stay <= 29x29.

c) Implicit assumptions review
   Base main() assumed:
   - every row is uniform across the full width
   - the center can be created by `full((h0, h0), color)`, i.e. as a solid square
   - copying the previous stage by raw array overwrite is always safe
   In the combo:
   - the first assumption still holds and the generator preserves it
   - the second is broken twice: the center shape is a diamond, and its fill uses
     background holes as part of the intended pattern
   - the third becomes subtle: copying only non-zero cells would destroy the
     checkerboard holes, but copying the whole bounding box would carve unwanted
     holes into the new outer ring. The correct fix is to preserve the previous
     diamond support mask and copy exactly that mask.

d) Placement legality
   The solver grows all layers around a single fixed center using Manhattan
   distance intervals, so shells are disjoint by construction. The generator only
   emits vertically stacked stripes with no gaps or overlaps, and it enforces
   `sum(heights) <= 15`, which implies outer bbox side `2*sum(heights)-1 <= 29`.

e) Visual observability
   A human sees three stable cues across examples:
   - diagonal diamond silhouette with empty corners
   - center region containing alternating color / background cells
   - output tightly bounded by the outer diamond rather than by a larger padded
     canvas
   None of these rely on hidden color ordering or other non-visual metadata.

f) Visual cleanliness
   There is still only one object family in the output: concentric diamonds.
   Checkerboard holes are confined to the center diamond, while outer shells stay
   solid, so the pattern remains readable rather than cluttered.

g) Inferrability of the change
   Colors are not remapped. Each stripe keeps its own color and only changes role:
   bottom stripe controls the center checkerboard, upper stripes control outward
   shell thicknesses. Because the generator always produces visible center holes
   and at least one outer shell, the fused behavior does not collapse back to the
   original nested-square task.

New solver assumptions introduced here
   - background is 0; generate_input() reserves 0 by choosing stripe colors in 1..9
   - stripes are contiguous runs of full-width uniform rows
   - stripe heights are positive integers
   - the output should be interpreted via Manhattan distance from a single center
   generate_input() guarantees each of these assumptions.
"""


def _parse_runs_bottom_to_top(grid: np.ndarray) -> list[tuple[int, int]]:
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


def _distance_grid(radius: int) -> np.ndarray:
    size = int(2 * radius + 1)
    center = int(radius)
    ys, xs = np.indices((size, size))
    return np.abs(ys - center) + np.abs(xs - center)


def _diamond_mask(radius: int) -> np.ndarray:
    return _distance_grid(radius) <= int(radius)


def _make_center_checkerboard_diamond(color: int, height: int) -> np.ndarray:
    radius = int(height - 1)
    size = int(2 * radius + 1)
    ys, xs = np.indices((size, size))
    diamond = _diamond_mask(radius)
    out = np.full((size, size), BG, dtype=int)
    out[diamond & (((ys + xs) % 2) == 0)] = int(color)
    return out


def _tight_crop_non_bg(grid: np.ndarray, bg: int = BG) -> np.ndarray:
    coords = np.argwhere(np.array(grid, dtype=int) != int(bg))
    if coords.size == 0:
        return np.zeros((0, 0), dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.array(grid, dtype=int)[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _build_original_square(runs: list[tuple[int, int]]) -> np.ndarray:
    color0, h0 = runs[0]
    cur = np.full((int(h0), int(h0)), int(color0), dtype=int)
    for color, thickness in runs[1:]:
        thickness = int(thickness)
        side = int(cur.shape[0] + 2 * thickness)
        nxt = np.full((side, side), int(color), dtype=int)
        nxt[thickness : thickness + cur.shape[0], thickness : thickness + cur.shape[1]] = cur
        cur = nxt
    return cur


def _count_center_checkerboard_holes(out: np.ndarray, runs: list[tuple[int, int]]) -> int:
    center_height = int(runs[0][1])
    center_radius = int(center_height - 1)
    inner_side = int(2 * center_radius + 1)
    offset = int(sum(height for _color, height in runs[1:]))
    patch = np.array(out, dtype=int)[offset : offset + inner_side, offset : offset + inner_side]
    mask = _diamond_mask(center_radius)
    return int(np.count_nonzero((patch == BG) & mask))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build concentric diamond shells from horizontal color stripes.

    Fused rule:
    - parse full-width horizontal stripes from bottom to top
    - bottom stripe of height h0 becomes a checkerboard-filled center diamond
      with Manhattan radius h0 - 1
    - each later stripe of height h adds a same-color diamond shell covering the
      distance interval (old_radius, old_radius + h]
    - copy the previous stage only on its diamond support mask so background holes
      inside the center survive but background corners outside the inner diamond do
      not erase the new shell
    - return the non-background tight crop
    """
    runs = _parse_runs_bottom_to_top(np.array(grid, dtype=int))

    center_color, center_height = runs[0]
    radius = int(center_height - 1)
    cur = _make_center_checkerboard_diamond(int(center_color), int(center_height))

    for color, thickness in runs[1:]:
        thickness = int(thickness)
        old_radius = int(radius)
        radius = int(radius + thickness)

        dist = _distance_grid(radius)
        nxt = np.full((2 * radius + 1, 2 * radius + 1), BG, dtype=int)
        ring_mask = (dist <= radius) & (dist > old_radius)
        nxt[ring_mask] = int(color)

        offset = thickness
        prev_mask = _diamond_mask(old_radius)
        sub = nxt[offset : offset + cur.shape[0], offset : offset + cur.shape[1]]
        sub[prev_mask] = cur[prev_mask]
        cur = nxt

    return _tight_crop_non_bg(cur, bg=BG)


def _gen_attempt() -> np.ndarray | None:
    width = int(np.random.randint(6, 15))
    n_runs = int(np.random.randint(2, 7))

    heights = [int(np.random.randint(3, 5))]
    heights.extend(int(np.random.randint(1, 5)) for _ in range(n_runs - 1))
    if sum(heights) > 15:
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

    runs = _parse_runs_bottom_to_top(inp)
    out = main(inp)

    expected_side = int(2 * sum(height for _color, height in runs) - 1)
    if out.shape != (expected_side, expected_side):
        return None

    outer_radius = int(sum(height for _color, height in runs) - 1)
    if np.count_nonzero((out == BG) & (~_diamond_mask(outer_radius))) == 0:
        return None

    if _count_center_checkerboard_holes(out, runs) == 0:
        return None

    source_out = _build_original_square(runs)
    if out.shape == source_out.shape and np.array_equal(out, source_out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
