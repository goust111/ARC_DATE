# Variant of 45a5af55
# Dimension: D2 — 框架形状 (transform)
# Change: 构建正方形嵌套框架（每层为等厚方形环）... → 构建矩形嵌套框架（宽高比由输入宽度编码）

from __future__ import annotations

import math

import numpy as np


# Design analysis for the D2 variant
# a) Concrete behavior:
#    The input is still a stack of full-width horizontal stripes, so the solver can
#    keep reading stripe runs from bottom to top. The change happens only when a run
#    is turned into geometry. In the original task, a stripe of height h contributes
#    a square side/thickness h. Here the input width/height is first reduced to an
#    aspect ratio rw:rh. The bottom stripe becomes a solid rectangle of size
#    (h * rh) x (h * rw), and every higher stripe wraps the current shape with a
#    similar rectangular frame: vertical thickness h * rh, horizontal thickness
#    h * rw. Example with ratio 2:1 and bottom-to-top heights [2, 1]:
#    original => center 2x2, then wrap by 1 to get 4x4; variant => center 2x4,
#    then wrap by (1 row, 2 cols) to get 4x8.
# b) Complexity calibration:
#    The aspect ratio is restricted to a small set of non-square reduced ratios so
#    the rule is visible but not chaotic. Using only a few ratios keeps examples
#    learnable while still forcing ratio reasoning beyond the square baseline.
# c) Implicit-assumption audit:
#    - Parsing consecutive uniform rows is still valid because the input remains a
#      stripe stack.
#    - The original `np.full((h0, h0), ...)` assumed the center is square; this is
#      no longer valid and must use (h0 * rh, h0 * rw).
#    - The original `new_n = cur.shape[0] + 2 * h` assumed both axes expand equally;
#      now the two axes expand by different amounts derived from the reduced aspect.
#    - Using a reduced ratio from the whole input grid adds a new assumption: the
#      generator must choose widths that encode the intended ratio exactly.
# d) Placement legality:
#    The generator still creates only full-width stripes, so there is no risk of
#    overlap inside the input. It additionally guarantees that the chosen height sum
#    is compatible with a reduced aspect ratio and that the resulting output stays
#    within ARC-sized bounds.
# e) Visual observability:
#    The output is always non-square, and the chosen aspect varies across samples.
#    Because every nested layer keeps the same reduced ratio, the rectangular rule is
#    obvious from just a few examples rather than being hidden in only the outer box.
# f) Visual cleanliness:
#    The image remains a single centered nesting of solid frames, with no extra
#    markers or overlays. Scaling thickness consistently by aspect ratio preserves
#    clean, easily segmented borders.
# g) Color inference:
#    Colors are unchanged from the original task: stripe colors map directly to the
#    corresponding nested frames. No additional color remapping is introduced.
# 9) New solver assumptions checked against generator:
#    - `g.shape` must encode a non-square aspect ratio exactly after reduction.
#      The generator enforces this by choosing widths from a fixed reduced-ratio set.
#    - `rw * output_scale` and `rh * output_scale` must fit the output canvas size.
#      The generator rejects attempts that would exceed 30 cells on either axis.
#    - All rows inside one stripe must be identical. The generator still builds each
#      stripe via `np.full`, so the parser sees exactly the intended runs.

_ASPECT_OPTIONS: tuple[tuple[int, int], ...] = ((2, 1), (3, 2), (4, 3))


def _parse_runs(g: np.ndarray) -> list[tuple[int, int]]:
    """Parse full-width color stripes from bottom to top as (color, height)."""
    h_total = int(g.shape[0])
    runs: list[tuple[int, int]] = []
    r = h_total - 1
    while r >= 0:
        col = int(g[r, 0])
        h = 1
        rr = r - 1
        while rr >= 0 and bool((g[rr] == col).all()):
            h += 1
            rr -= 1
        runs.append((col, h))
        r = rr
    return runs


def _build_output(runs: list[tuple[int, int]], aspect: tuple[int, int]) -> np.ndarray:
    """Build similar nested rectangles using the reduced input aspect ratio."""
    rw, rh = aspect
    c0, h0 = runs[0]
    cur = np.full((h0 * rh, h0 * rw), c0, dtype=int)

    for col, h in runs[1:]:
        pad_y = h * rh
        pad_x = h * rw
        nxt = np.full((cur.shape[0] + 2 * pad_y, cur.shape[1] + 2 * pad_x), col, dtype=int)
        nxt[pad_y : pad_y + cur.shape[0], pad_x : pad_x + cur.shape[1]] = cur
        cur = nxt

    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested rectangular frames from a stack of horizontal color stripes.

    The stripe-reading logic is unchanged from the original task. The only new
    ingredient is that the output geometry uses the reduced input aspect ratio
    `input_width : input_height`.
    """
    g = np.array(grid, dtype=int)
    h_total, w_total = g.shape
    runs = _parse_runs(g)
    assert runs

    scale = math.gcd(h_total, w_total)
    aspect = (w_total // scale, h_total // scale)
    return _build_output(runs, aspect)


def _gen_attempt() -> np.ndarray | None:
    n_runs = int(np.random.randint(2, 6))
    heights = [int(np.random.randint(1, 4)) for _ in range(n_runs)]
    total_h = sum(heights)
    output_scale = heights[0] + 2 * sum(heights[1:])

    candidates: list[tuple[int, int, int]] = []
    for rw, rh in _ASPECT_OPTIONS:
        if total_h % rh != 0:
            continue
        input_scale = total_h // rh
        width = input_scale * rw
        if width < 6 or width > 18:
            continue
        if rw * output_scale > 30 or rh * output_scale > 30:
            continue
        candidates.append((rw, rh, width))

    if not candidates:
        return None

    _rw, _rh, width = candidates[int(np.random.randint(len(candidates)))]

    cols: list[int] = []
    prev = None
    for _ in range(n_runs):
        c = int(np.random.randint(1, 10))
        while prev is not None and c == prev:
            c = int(np.random.randint(1, 10))
        cols.append(c)
        prev = c

    stripes = [
        np.full((h, width), col, dtype=int)
        for col, h in reversed(list(zip(cols, heights)))
    ]
    grid = np.vstack(stripes)

    # Validate the encoded aspect and ensure the variant did not collapse back to a square.
    out = main(grid)
    if out.shape[0] == out.shape[1]:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
