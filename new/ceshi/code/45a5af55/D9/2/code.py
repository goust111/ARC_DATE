# Variant of 45a5af55
# Dimension: D9 — 多步处理与迭代结构 (multi_step)
# Change: 单阶段迭代：逐层读取色条并包裹框架，无额外处理步骤
#   → 先构建嵌套框架，再对每层框架进行对称性修剪

from __future__ import annotations

import numpy as np


"""
Design analysis for the D9 variant
==================================

a) Concrete behavior and traced example
---------------------------------------
Original behavior:
- Parse full-width horizontal stripes from bottom to top.
- The bottom stripe height h0 becomes a solid h0 x h0 center square.
- Each higher stripe of height h wraps the current square with a full square frame
  of thickness h, producing a fully filled nested-square output.

Variant behavior:
- Stage 1 stays the same: build the complete nested square construction first.
- Stage 2 prunes every outer frame into a canonical bottom-left "L" piece.
  For a frame with outer side s and thickness h, only these parts survive:
  1) the full left arm: columns [0, h)
  2) the full bottom arm: rows [s-h, s)
- The innermost center square is not pruned; only true frames are pruned.
- Deleted cells become background 0. The input never uses 0, so the pruning is
  visually explicit.

Small trace:
- Runs bottom->top: [(2, 2), (5, 1), (7, 2)]
- Stage 1:
  1) center = 2x2 solid color 2
  2) wrap with thickness 1 color 5 -> 4x4 full frame around the center
  3) wrap with thickness 2 color 7 -> 8x8 full frame around that
- Stage 2:
  1) keep the 2x2 center square unchanged
  2) prune the color-5 frame to its left arm and bottom arm
  3) prune the color-7 frame to its left arm and bottom arm
- Difference from original:
  the original output is a full set of concentric square frames;
  the variant output contains the same nested construction logic, but every outer
  layer has a missing top/right counterpart, so the second stage is necessary.

b) Complexity calibration
-------------------------
- The variant is visibly different even with only two outer layers, but the
  generator uses 3-6 stripes so the two-stage pattern is obvious.
- The pruning rule has one fixed orientation for the whole task. That keeps the
  branch count low enough for ARC while still making the second phase observable.
- Heights remain in 1-4, preserving the original task's scale and keeping output
  sizes <= 30.

c) Implicit-assumption review of the original main()
----------------------------------------------------
- `col = g[r, 0]` assumes each row is uniform. Still valid because the generator
  continues to build only full-width stripes.
- `while (g[rr] == col).all()` assumes stripes are contiguous runs of identical
  rows. Still valid.
- The original `nxt = full(color)` plus center overwrite assumed every new layer
  remained a complete square ring. That assumption is no longer valid for the
  final output, because outer layers now need holes.
- Therefore the variant must separate:
  1) stage-1 full nested construction
  2) stage-2 pruning on isolated layer geometry
- A naive "paint zeros into the top-right area of the final array" would be wrong,
  because it could cut through inner colors. The variant instead computes each
  layer's final coordinates and rebuilds the pruned result layer by layer.

d) Placement legality
---------------------
- Input placement is unchanged: vertically stacked full-width stripes, no overlap,
  no empty rows, no out-of-bounds risk.
- Output legality is guaranteed by computing the same final square size as the
  original task, then placing each layer inside that square using exact offsets
  implied by the stripe heights.
- The generator rejects oversize instances where the original final side would
  exceed 30.

e) Visual observability
-----------------------
- With 2-4 ARC examples, the solver can still infer "stripe height becomes layer
  thickness", because the nested offsets are unchanged from the base task.
- The new stage is visible because each outer layer turns into a bottom-left L,
  and the missing area is background 0 rather than a hidden overwrite.
- Using at least 3 runs in generation makes the "build then prune repeatedly"
  pattern much easier to spot than a single-frame case.

f) Visual cleanliness
---------------------
- The result contains only three clean region types:
  1) center solid square
  2) left/bottom arms for each frame
  3) background 0 in removed areas
- There is no ambiguous overlap because each kept arm lies entirely in the frame's
  annulus and never covers the inner square.

g) Color-change inferability
----------------------------
- Colors are not remapped. A stripe's color still becomes the corresponding layer
  color.
- The only new semantic change is geometric pruning, so the task does not burden
  the solver with extra color logic.

Post-change assumptions to audit
--------------------------------
- Parsing assumes no background rows in the input. The generator guarantees the
  input is fully tiled by stripes.
- Pruning assumes the innermost layer is a solid square and every later layer has
  positive thickness. The generator guarantees 3-6 runs with heights in 1-4.
- Background 0 must remain visually distinct, so the generator samples colors only
  from 1..9.
"""


def _parse_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return (color, height) runs from bottom to top."""
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


def _build_full_nested(runs: list[tuple[int, int]]) -> np.ndarray:
    """Original stage-1 construction: full nested squares with no pruning."""
    c0, h0 = runs[0]
    cur = np.full((h0, h0), c0, dtype=int)
    for col, thick in runs[1:]:
        new_n = int(cur.shape[0] + 2 * thick)
        nxt = np.full((new_n, new_n), col, dtype=int)
        nxt[thick : thick + cur.shape[0], thick : thick + cur.shape[1]] = cur
        cur = nxt
    return cur


def _layer_specs(runs: list[tuple[int, int]]) -> list[tuple[int, int, int, int, bool]]:
    """
    Geometry for each layer in final coordinates.

    Returns tuples:
    - color
    - top/left offset in the final square
    - side length of this layer's outer bounding square
    - thickness (center layer stores its own side length here but is not pruned)
    - is_center
    """
    heights = [height for _color, height in runs]
    sides = [heights[0]]
    for thick in heights[1:]:
        sides.append(sides[-1] + 2 * thick)

    offsets = [0] * len(runs)
    acc = 0
    for idx in range(len(runs) - 1, -1, -1):
        offsets[idx] = acc
        if idx > 0:
            acc += heights[idx]

    return [
        (runs[idx][0], offsets[idx], sides[idx], heights[idx], idx == 0)
        for idx in range(len(runs))
    ]


def _prune_full_nested(full: np.ndarray, runs: list[tuple[int, int]]) -> np.ndarray:
    """Stage 2: keep only the left and bottom arms of each outer frame."""
    out = np.zeros_like(full)
    for color, top, side, thick, is_center in _layer_specs(runs):
        if is_center:
            out[top : top + side, top : top + side] = full[top : top + side, top : top + side]
            continue

        out[top : top + side, top : top + thick] = full[top : top + side, top : top + thick]
        out[top + side - thick : top + side, top : top + side] = full[
            top + side - thick : top + side, top : top + side
        ]
    return out


def _expected_output_from_runs(runs: list[tuple[int, int]]) -> np.ndarray:
    full = _build_full_nested(runs)
    return _prune_full_nested(full, runs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    1) build the original nested square construction
    2) prune every outer frame into a bottom-left L-shape
    """
    runs = _parse_runs(np.array(grid, dtype=int))
    assert runs
    full = _build_full_nested(runs)
    return _prune_full_nested(full, runs)


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
        col = int(np.random.randint(1, 10))
        while prev is not None and col == prev:
            col = int(np.random.randint(1, 10))
        colors.append(col)
        prev = col

    runs = list(zip(colors, heights))
    stripes = [
        np.full((height, width), color, dtype=int)
        for color, height in reversed(runs)
    ]
    inp = np.vstack(stripes)

    expected = _expected_output_from_runs(runs)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    full = _build_full_nested(runs)
    removed = int(np.count_nonzero((full != 0) & (actual == 0)))
    if removed < 3:
        return None
    if np.array_equal(full, actual):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
