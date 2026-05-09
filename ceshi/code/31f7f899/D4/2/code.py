# Variant of 31f7f899
# Dimension: D4 — 目标槽位：原地替换 (dst)
# Change: 条形的原始列位置即为槽位，保持位置不变，只替换条形形状... → 输出为紧凑裁剪

from __future__ import annotations

import numpy as np
from arc_utils import bbox_coords, bbox_h, find_components, get_bg, paste

"""
Design analysis for the D4 variant

a) Concrete behavior and trace
   - Input keeps the same core scene: several monochrome vertical rectangles intersect
     a horizontal baseline. Each color defines a slot; each slot keeps its original x
     position.
   - The original task rendered the sorted result back onto the full input canvas. This
     variant still sorts shapes by height and reassigns them to the left-to-right slots,
     but then returns only the minimal bounding box around all non-background pixels.
   - To make tight-crop observable, the generator narrows the baseline from “full row”
     to “only the span covering the slots”, while also leaving outer background margins.
   - Small trace:
       input heights at slots x=[2,6,10] might be [7,3,5]
       original logic -> sorted heights [3,5,7] are drawn at x=[2,6,10] on the same
       full canvas, preserving all empty margins
       variant logic -> draw that same sorted full-canvas result first, then crop to the
       bbox of all non-bg cells; margins disappear and output shape shrinks
     Difference: sorting rule is unchanged; only destination canvas selection changes.

b) Complexity calibration
   - Heights are distinct odd numbers, so the shortest-to-tallest order is unambiguous.
   - Bar width is uniform within a sample (1 or 3), matching the source task’s visual
     simplicity and preventing extra matching ambiguity.
   - There are 3..5 bars, enough to make sorting visible but not so many that the scene
     becomes cluttered.
   - Tight-crop is made visually obvious by forcing positive outer margins in both axes,
     so output dimensions are strictly smaller than input dimensions.

c) Hidden assumptions in the original main()
   - `out = np.full_like(g, bg); out[base_r, :] = base_col` assumed the baseline spans
     the whole row. That is invalid here because full-row painting would destroy the
     crop signal. The fix is to preserve only the original baseline cells.
   - Reusing the original base-row detector purely by maximum count becomes brittle once
     the baseline is no longer full-width. Here the base row/color is detected by
     preferring colors that appear in multiple separated runs on one row; the segmented
     baseline satisfies that, normal bars do not.
   - The anchor computation assumes each bar intersects the base row. The generator keeps
     odd heights and centered placement so this remains true.
   - Rendering into the full canvas is still safe: slot anchors are unchanged, widths are
     unchanged, and sorted shapes fit exactly where input shapes fit. Cropping happens
     only after rendering.

d) Placement legality
   - Bars are placed with at least one empty/baseline column between neighboring bars, so
     different colors never touch and monochrome component extraction remains clean.
   - Heights are chosen to fit strictly inside the canvas with positive top/bottom
     margins, so no paste goes out of bounds and the crop removes real background.
   - The baseline covers the entire slot span, so every bar intersects the same baseline
     row and the solver can recover anchors consistently.

e) Visual observability
   - From 2-4 examples, a solver can see two coupled changes:
       1. bar heights are sorted into left-to-right slots while keeping slot colors
       2. the output is not the original canvas size; it is the tight bbox of the sorted
          arrangement
   - This is observable because the cropped output removes external margins on both axes
     in every generated sample, while the internal spacing between slots remains intact.

Post-change solver assumptions audited against the generator
   - Base color only appears on the baseline row: generator guarantees this.
   - Baseline has at least two separated visible runs: guaranteed by k>=3 and a minimum
     one-column gap between adjacent bars.
   - Every non-base component is a single monochrome bar intersecting the baseline row:
     guaranteed by centered odd-height rectangles.
   - Sorting by `(height, size, anchor, color)` is deterministic even if future examples
     accidentally tie on height; current generator uses distinct heights, so ties do not
     affect correctness.
"""


def _count_runs(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0
    starts = mask & np.concatenate(([True], ~mask[:-1]))
    return int(starts.sum())


def _find_base_row_color(g: np.ndarray, bg: int) -> tuple[int, int]:
    H, _W = g.shape
    best_multi = None  # (runs, count, row, color)
    best_any = None  # (count, row, color)
    for r in range(H):
        row = g[r, :]
        for v in sorted(set(map(int, row)) - {int(bg)}):
            mask = row == int(v)
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            cand_any = (cnt, int(r), int(v))
            if best_any is None or cand_any > best_any:
                best_any = cand_any
            runs = _count_runs(mask)
            if runs >= 2:
                cand_multi = (runs, cnt, int(r), int(v))
                if best_multi is None or cand_multi > best_multi:
                    best_multi = cand_multi
    if best_multi is not None:
        _runs, _cnt, base_r, base_col = best_multi
        return int(base_r), int(base_col)
    assert best_any is not None
    _cnt, base_r, base_col = best_any
    return int(base_r), int(base_col)


def _tight_crop(g: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(g != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort vertical bars by height, redraw them in the original slots, then tight-crop.

    Input structure:
    - bg background.
    - A horizontal segmented baseline of one color spans exactly the slot region.
    - Several monochrome vertical bars of distinct colors intersect that baseline.

    Rule:
    - Rank bar shapes by height (shortest to tallest).
    - Rank slot x-positions left to right.
    - Place the i-th shortest bar shape into the i-th leftmost slot, recoloring it to
      that slot's original color.
    - Preserve the baseline cells.
    - Return the minimal bounding box around the resulting non-background pixels.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    base_r, base_col = _find_base_row_color(g, bg)
    base_mask = np.zeros_like(g, dtype=bool)
    base_mask[int(base_r), :] = g[int(base_r), :] == int(base_col)

    objects: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {int(bg), int(base_col)}:
            continue

        pts = comp["coords"]
        height = int(bbox_h(comp["bbox"]))

        base_cols = sorted(int(c) for r, c in pts if int(r) == int(base_r))
        if base_cols:
            anchor = int(base_cols[len(base_cols) // 2])
        else:
            cols_sorted = sorted(int(c) for _r, c in pts)
            anchor = int(cols_sorted[len(cols_sorted) // 2])

        rel = [(int(r) - int(base_r), int(c) - int(anchor)) for r, c in pts]
        objects.append(
            {
                "color": int(col),
                "anchor": int(anchor),
                "height": int(height),
                "size": int(len(rel)),
                "rel": rel,
            }
        )

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(objects, key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])))

    out = np.full((H, W), int(bg), dtype=int)
    out[base_mask] = int(base_col)

    for slot, shape in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(slot_col)

    return _tight_crop(out, bg)


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_lefts(span_w: int, k: int, width: int) -> list[int]:
    used = int(k * width + (k - 1))
    extra = int(span_w - used)
    if extra < 0:
        raise ValueError("bars do not fit")
    gaps = _rand_composition(extra, k + 1)
    lefts: list[int] = []
    c = int(gaps[0])
    for i in range(k):
        lefts.append(int(c))
        c += int(width)
        if i < k - 1:
            c += 1 + int(gaps[i + 1])
    return lefts


def _draw_scene(
    H: int,
    W: int,
    base_r: int,
    base_left: int,
    base_right: int,
    base_color: int,
    bg: int,
    lefts: list[int],
    width: int,
    colors: list[int],
    heights: list[int],
) -> np.ndarray:
    grid = np.full((int(H), int(W)), int(bg), dtype=int)
    grid[int(base_r), int(base_left) : int(base_right) + 1] = int(base_color)
    for left, color, height in zip(lefts, colors, heights):
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - height // 2), int(left))
    return grid


def _gen_attempt() -> np.ndarray | None:
    bg = 8
    base_color = 6
    palette = [1, 2, 3, 4, 5, 7, 9]

    k = int(np.random.randint(3, 6))
    height_pool = np.array([1, 3, 5, 7, 9, 11], dtype=int)
    sorted_heights = sorted(int(h) for h in np.random.choice(height_pool, size=k, replace=False))
    tallest = int(sorted_heights[-1])

    top_margin = int(np.random.randint(1, 4))
    bottom_margin = int(np.random.randint(1, 4))
    H = int(tallest + top_margin + bottom_margin)
    base_r = int(top_margin + tallest // 2)

    width = int(np.random.choice([1, 3]))
    used_w = int(k * width + (k - 1))
    span_w = int(used_w + np.random.randint(0, 7))
    local_lefts = _slot_lefts(span_w, k, width)

    outer_left = int(np.random.randint(1, 5))
    outer_right = int(np.random.randint(1, 5))
    W = int(span_w + outer_left + outer_right)
    lefts = [int(outer_left + x) for x in local_lefts]
    base_left = int(outer_left)
    base_right = int(outer_left + span_w - 1)

    colors = [int(c) for c in np.random.choice(palette, size=k, replace=False)]

    out_full = _draw_scene(
        H=H,
        W=W,
        base_r=base_r,
        base_left=base_left,
        base_right=base_right,
        base_color=base_color,
        bg=bg,
        lefts=lefts,
        width=width,
        colors=colors,
        heights=sorted_heights,
    )
    out = _tight_crop(out_full, bg)

    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    in_heights = [int(sorted_heights[i]) for i in perm]
    inp = _draw_scene(
        H=H,
        W=W,
        base_r=base_r,
        base_left=base_left,
        base_right=base_right,
        base_color=base_color,
        bg=bg,
        lefts=lefts,
        width=width,
        colors=colors,
        heights=in_heights,
    )

    if np.array_equal(inp, out_full):
        return None
    if out.shape[0] >= inp.shape[0] or out.shape[1] >= inp.shape[1]:
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
