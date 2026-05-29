# Variant of 31f7f899
# Dimension: D10 — 多步骤序列化处理 (meta)
# Change: 单步全局排序——一次性将所有条形按高度排序分配到所有槽位... → 交替排序方向

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, find_components, get_bg, paste

"""
Design analysis for the D10 variant
==================================

1. Concrete behavior and trace
The scene keeps the original task's visual grammar:
- background color fills the grid,
- one horizontal baseline spans the whole width,
- several monochrome vertical bars sit on that baseline,
- each x-position is a fixed "slot" whose color identity is given by the bar
  currently occupying it in the input.

Original task:
- collect all bar shapes,
- sort them globally by height ascending,
- place the shortest shape in the leftmost slot, next shortest in the next slot,
  and so on.

This variant changes only the redistribution order:
- sort the bar shapes by height ascending once,
- then consume that sorted list from alternating ends while scanning slots
  left-to-right,
- slot 1 gets the shortest remaining shape,
- slot 2 gets the tallest remaining shape,
- slot 3 gets the next-shortest remaining shape,
- slot 4 gets the next-tallest remaining shape, etc.

Small trace with slot colors [A,B,C,D] and heights [1,3,5,7]:
- original output heights by slot: [1,3,5,7]
- variant output heights by slot:  [1,7,3,5]
The slot colors stay [A,B,C,D]; only the shape heights move. This makes the
output visibly zigzag rather than monotonically increasing.

2. Complexity calibration
To keep the alternating rule human-visible:
- generator uses at least 4 slots, so the output cannot look like a trivial
  near-sort,
- heights are distinct odd numbers, which preserves the original centered-bar
  geometry and removes ambiguity in the sorted order,
- slot colors remain distinct, so each slot is still visually trackable.

3. Implicit-assumption audit of the original solver
The original main() relies on several properties that must remain true:
- background = most common color. Generator keeps a large uniform background.
- baseline detection picks the (row,color) pair with maximal count excluding bg.
  This works only if one long baseline row remains present; generator preserves
  the full-width baseline.
- bars are found as monochrome connected components distinct from bg and the
  baseline color. Generator therefore never reuses the baseline color for bars.
- anchor is inferred from the median column of each component, which is stable
  only because every bar width is odd (1 or 3). Generator keeps odd widths.
- redistributing a shape to a slot assumes shapes do not collide after moving.
  Because slots are the original non-overlapping bar positions and all bars
  share the same width in one sample, reassigning heights cannot create lateral
  overlap.

The line that materially changes is the old global zip(slots, shapes). That zip
assumed a single monotone ordering. In the variant, slots stay left-to-right,
but the shape stream becomes serialized low/high/low/high.

4. Placement legality
The generator stays output-first:
- choose grid size, odd bar heights, a single odd bar width, and non-overlapping
  slot positions,
- build the expected output using the alternating-end order,
- derive an input by permuting the same heights,
- reject the sample unless main(input) reproduces the constructed output.

Legal placement is guaranteed because:
- heights never exceed the grid height,
- bars are pasted at fixed slot left edges,
- slots are spaced so equal-width bars never overlap,
- all bars are vertical rectangles centered on the baseline row, matching the
  solver's extracted relative coordinates.

5. Visual observability
The intended clue is that outputs are not merely sorted. Across examples, the
left-to-right heights alternate between low-end and high-end picks, creating a
consistent zigzag profile. Requiring >=4 bars avoids accidental similarity to
the original task. The generator also rejects degenerate cases where the variant
output would equal the original global-ascending output.

6. New assumptions introduced by the variant
The new main() additionally assumes:
- the ascending height order is well-defined enough that "next shortest" and
  "next tallest" are meaningful; generator enforces distinct heights,
- there are enough bars for alternation to matter visually; generator uses at
  least 4 bars,
- slot order is still determined solely by x-position; generator preserves the
  original left-to-right separated layout.
"""


def _alternating_end_order(items: list[object]) -> list[object]:
    ordered: list[object] = []
    lo = 0
    hi = len(items) - 1
    for idx in range(len(items)):
        if idx % 2 == 0:
            ordered.append(items[lo])
            lo += 1
        else:
            ordered.append(items[hi])
            hi -= 1
    return ordered


def _extract_objects(g: np.ndarray) -> tuple[int, int, int, int, int, list[dict[str, object]]]:
    H, W = g.shape
    bg = int(get_bg(g))

    # Baseline = the non-bg color occurrence with the largest row count.
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

    objects: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {bg, int(base_col)}:
            continue

        pts = comp["coords"]
        height = int(bbox_h(comp["bbox"]))

        # With odd widths, the median occupied column is the bar anchor/slot.
        base_cols = sorted(int(c) for r, c in pts if int(r) == int(base_r))
        if base_cols:
            anchor = int(base_cols[len(base_cols) // 2])
        else:
            cols_sorted = sorted(int(c) for _r, c in pts)
            anchor = int(cols_sorted[len(cols_sorted) // 2])

        rel = [(int(r) - int(base_r), int(c) - int(anchor)) for r, c in pts]
        objects.append(
            {
                "color": col,
                "anchor": anchor,
                "height": height,
                "size": int(len(rel)),
                "rel": rel,
            }
        )

    return H, W, bg, int(base_r), int(base_col), objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Preserve each slot's x-position/color, but serialize the sorted shapes with
    alternating direction: shortest, tallest, next-shortest, next-tallest, ...
    """
    g = np.array(grid, dtype=int)
    H, W, bg, base_r, base_col, objects = _extract_objects(g)

    # Slot identity is still purely left-to-right.
    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    # Height ranking is the same as the original task.
    shapes = sorted(objects, key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])))
    shape_stream = _alternating_end_order(shapes)

    out = np.full_like(g, bg)
    out[int(base_r), :] = int(base_col)

    for slot, shape in zip(slots, shape_stream):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(slot_col)

    return out


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_lefts(n: int, k: int, width: int) -> list[int]:
    used = int(k * width + (k - 1))
    extra = int(n - used)
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
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    lefts: list[int],
    width: int,
    colors: list[int],
    heights: list[int],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)
    for left, color, height in zip(lefts, colors, heights):
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - height // 2), int(left))
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(7, 18, 2)))
    base_r = n // 2
    bg = 8
    base_color = 6

    height_choices = list(range(1, min(n, 11) + 1, 2))
    max_count = min(5, len(height_choices), (n + 1) // 2)
    if max_count < 4:
        return None
    k = int(np.random.randint(4, max_count + 1))

    feasible_widths = [1]
    if n - 3 * k > 3 and n >= 4 * k - 1:
        feasible_widths.append(3)
    width = int(np.random.choice(feasible_widths))

    lefts = _slot_lefts(n, k, width)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]

    sorted_heights = sorted(int(h) for h in np.random.choice(height_choices, size=k, replace=False))
    out_heights = [int(h) for h in _alternating_end_order(sorted_heights)]
    if out_heights == sorted_heights:
        return None
    out = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, out_heights)

    input_heights: list[int] | None = None
    for _ in range(20):
        perm = np.random.permutation(k)
        candidate = [int(sorted_heights[i]) for i in perm]
        if candidate != out_heights:
            input_heights = candidate
            break
    if input_heights is None:
        return None

    inp = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, input_heights)
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
