# Variant of 31f7f899
# Dimension: D1 — 匹配条件：按高度排序映射槽位 (match_cond)
# Change: size_ordering — 将条形按高度升序排序，槽位按列位置从左到右排序，第i短的条形放入第i个槽位... → 按形状匹配凹槽

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import find_components, get_bg


"""
Design analysis for the D1 variant
==================================

a) Concrete visual behavior and trace
-------------------------------------
Original behavior:
- One full-width base row stays fixed.
- Several vertical monochrome bars stand above it.
- Solver sorts bar silhouettes by height and reassigns them to left-to-right slots.

Variant behavior:
- The fixed base row is preserved, but each slot now also has a colored pedestal below it.
- Each pedestal is a 3x4 colored block with a bg "groove" cut from its top side.
- Above the base row sits one irregular 3x3 monochrome piece per slot.
- Each upper piece is the vertical mirror of exactly one pedestal groove.
- Input permutes the upper shapes across slots; output moves each shape to the slot whose
  groove has the same normalized geometry, recoloring it to that slot's pedestal color.

Small trace:
- Suppose slot A (left, blue) has groove shape `cross`, slot B (right, red) has groove
  shape `bent`, but the upper pieces are swapped in the input.
- Original code would only look at heights, so with equal-height shapes it has no useful
  signal and would not recover the swap.
- Variant code extracts upper shape signatures, extracts pedestal-hole signatures, matches
  `cross -> cross` and `bent -> bent`, then redraws the upper pieces above the matching slots.

b) Complexity calibration
-------------------------
- Shapes are not rectangles; each is a connected 5-cell polyomino in a 3x3 box.
- All shapes share the same bbox size (3x3) and the same area (5 cells), so height/size
  ordering is intentionally uninformative.
- A sample uses 3-4 slots chosen from a 7-shape catalog; all chosen shapes are unique.
- This is enough repetition to make the geometric rule visible while keeping the search
  space small and unambiguous.

c) Implicit-assumption audit against the original `main()`
----------------------------------------------------------
- Original base-row detection still works because the base row remains the longest non-bg
  monochrome run in the grid.
- Original object extraction assumed every movable object was a bar whose anchor could be
  inferred from cells on the base row. That fails here because upper pieces do not touch
  the base row and lower pedestals are no longer movable.
- Original ranking by `height` is invalid because all shapes now share the same height.
- Original `rel` coordinates were measured from a bar anchor on the base row. Here we
  instead normalize upper pieces and pedestal grooves against fixed slot-left coordinates:
  upper pieces use `(base_r - 1 - r, c - left)`, grooves use `(r - (base_r + 1), c - left)`.
- Original redraw loop recolored shapes by slot color after sorting. That recoloring
  remains valid, but the match key changes from numeric ordering to exact shape signature.

d) Placement legality
---------------------
- Every pedestal occupies a non-overlapping 3x4 rectangle below the base row.
- Every upper piece occupies a non-overlapping mirrored 3x3 region above the base row.
- Slot spacing is computed with the same left-to-right packing idea as the original task,
  using width 3 and at least one bg column between slots.
- The generator is output-first: build the solved scene, permute only upper shapes to form
  the input, then reject the sample unless `main(inp) == out`.

e) Visual observability
-----------------------
- In each input, the lower colored pedestals expose different bg grooves, and the upper
  pieces visibly reuse those silhouettes but in the wrong columns.
- Because shapes have the same height and area, a human cannot explain the examples by
  "sort by size"; the only stable cue is geometric matching.
- Keeping pedestal colors fixed makes the slot identity obvious across input/output.

New assumptions introduced by the new `main()`
----------------------------------------------
- Each upper piece spans the full 3-column slot width and full 3-row shape height.
  `generate_input()` guarantees this by choosing from a catalog where every shape uses
  columns {0,2} and rows {0,2}.
- Each pedestal bbox is exactly 3x4 and starts immediately below the base row.
  The generator always constructs pedestals in that layout.
- Each pedestal contains exactly one groove signature and all signatures are unique.
  The generator samples unique shapes without replacement.
- Upper pieces and pedestal grooves use the same signature vocabulary.
  The generator builds both from the same catalog and only permutes the upper assignment.
"""


SHAPE_CATALOG: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),  # cross
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),  # falling zig
    ((0, 1), (0, 2), (1, 0), (1, 1), (2, 0)),  # rising zig
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),  # bent right
    ((0, 2), (1, 0), (1, 1), (1, 2), (2, 0)),  # bent left
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),  # top bar + stem
    ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2)),  # bottom bar + stem
)


def _signature(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    best = None  # (count, row, color)
    for r in range(H):
        vals, counts = np.unique(g[r, :], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == bg:
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(r), int(v))
    assert best is not None
    _cnt, base_r, base_col = best

    upper_pieces: dict[tuple[tuple[int, int], ...], dict[str, int]] = {}
    slots: dict[tuple[tuple[int, int], ...], dict[str, int]] = {}

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    for comp in comps:
        col = int(comp["color"])
        if col in {bg, int(base_col)}:
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        occ = {(int(r), int(c)) for r, c in comp["coords"]}

        if r1 < base_r:
            rel = [(int(base_r - 1 - r), int(c - c0)) for r, c in occ]
            sig = _signature(rel)
            upper_pieces[sig] = {"left": int(c0)}
    below = g[base_r + 1 :, :]
    below_mask = (below != bg) & (below != int(base_col))
    active_cols = [bool(x) for x in below_mask.any(axis=0)]
    c = 0
    while c < W:
        if not active_cols[c]:
            c += 1
            continue
        left = int(c)
        while c + 1 < W and active_cols[c + 1]:
            c += 1
        right = int(c)

        box = g[base_r + 1 : base_r + 5, left : right + 1]
        vals = sorted({int(v) for v in box.flatten() if int(v) not in {bg, int(base_col)}})
        assert len(vals) == 1
        hole = []
        for dr in range(box.shape[0]):
            for dc in range(box.shape[1]):
                if int(box[dr, dc]) == bg:
                    hole.append((int(dr), int(dc)))
        sig = _signature(hole)
        slots[sig] = {"left": int(left), "color": int(vals[0])}
        c += 1

    assert upper_pieces
    assert len(upper_pieces) == len(slots)
    assert set(upper_pieces) == set(slots)

    out = np.array(g, copy=True)
    out[:base_r, :] = int(bg)
    out[base_r, :] = int(base_col)

    for sig, slot in slots.items():
        left = int(slot["left"])
        color = int(slot["color"])
        for dr, dc in sig:
            rr = int(base_r - 1 - dr)
            cc = int(left + dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = color

    return out


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_lefts(n: int, k: int, width: int) -> list[int]:
    used = int(k * width + (k - 1))
    extra = int(n - used)
    if extra < 0:
        raise ValueError("slots do not fit")
    gaps = _rand_composition(extra, k + 1)
    lefts: list[int] = []
    c = int(gaps[0])
    for i in range(k):
        lefts.append(int(c))
        c += int(width)
        if i < k - 1:
            c += 1 + int(gaps[i + 1])
    return lefts


def _draw_piece(grid: np.ndarray, base_r: int, left: int, color: int, shape: tuple[tuple[int, int], ...]) -> None:
    for dr, dc in shape:
        grid[int(base_r - 1 - dr), int(left + dc)] = int(color)


def _draw_pedestal(
    grid: np.ndarray,
    base_r: int,
    left: int,
    color: int,
    bg: int,
    shape: tuple[tuple[int, int], ...],
) -> None:
    top = int(base_r + 1)
    grid[top : top + 4, left : left + 3] = int(color)
    for dr, dc in shape:
        grid[int(top + dr), int(left + dc)] = int(bg)


def _draw_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    lefts: list[int],
    colors: list[int],
    slot_shapes: list[tuple[tuple[int, int], ...]],
    piece_shapes: list[tuple[tuple[int, int], ...]],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)
    for left, color, slot_shape, piece_shape in zip(lefts, colors, slot_shapes, piece_shapes):
        _draw_piece(grid, int(base_r), int(left), int(color), piece_shape)
        _draw_pedestal(grid, int(base_r), int(left), int(color), int(bg), slot_shape)
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice([11, 13, 15, 17]))
    base_r = int(n // 2 - 1)
    bg = 8
    base_color = 6

    max_count = min(4, len(SHAPE_CATALOG), (n + 1) // 4)
    if max_count < 3:
        return None
    k = int(np.random.randint(3, max_count + 1))

    lefts = _slot_lefts(n, k, 3)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]
    shape_ids = [int(i) for i in np.random.choice(len(SHAPE_CATALOG), size=k, replace=False)]
    slot_shapes = [SHAPE_CATALOG[i] for i in shape_ids]

    out = _draw_scene(n, base_r, base_color, bg, lefts, colors, slot_shapes, slot_shapes)

    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    piece_shapes = [slot_shapes[int(i)] for i in perm]
    inp = _draw_scene(n, base_r, base_color, bg, lefts, colors, slot_shapes, piece_shapes)

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
