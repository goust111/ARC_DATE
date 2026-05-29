# Combo variant of 31f7f899 (index 16)
# combo_spec: D1/2+D10/3
# Name: 交替排序方向+降序排列

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, find_components, get_bg, paste

"""
Fusion design analysis
======================

This keeps the original scene grammar:
- one background color,
- one full-width baseline,
- several monochrome vertical bars whose x-positions define fixed slots,
- each slot keeps its original color identity in the output.

The fused rule combines two changes on the original "sort by height ascending"
mapping.

1. Concrete behavior and trace
Original task:
- sort bar shapes by height ascending,
- scan slots from left to right,
- place the i-th shortest shape into the i-th slot.

Fusion rule here:
- sort bar shapes by height descending first,
- while scanning slots left to right, alternate which end of that descending
  list is consumed,
- odd-numbered slots take the next tallest remaining shape,
- even-numbered slots take the next shortest remaining shape.

With slot colors [A,B,C,D] and bar heights [1,3,5,7]:
- original output heights: [1,3,5,7]
- pure descending only:    [7,5,3,1]
- fused output heights:    [7,1,5,3]

So the leftmost slot does reveal the D1 inversion (the tallest shape moves to
the first slot), but the full output is not a simple monotone descending run;
it alternates high, low, high, low and therefore forms a zigzag profile.

2. Complexity calibration
To keep the fusion legible rather than degenerate:
- generator uses at least 4 bars, so the alternation is visually observable,
- heights are distinct odd numbers, so "next tallest" and "next shortest" are
  unambiguous and the bars remain centered on the baseline,
- slot colors are distinct, so slot identity is easy to track,
- a single odd width is used within one sample, which keeps anchors stable and
  prevents lateral overlap after redistribution.

3. Implicit-assumption audit of the original solver
Relevant original assumptions and whether they still hold:
- get_bg() assumes background is the dominant color. Generator keeps a large
  uniform background area.
- baseline detection assumes one non-bg color dominates some row. Generator
  preserves a full-width baseline.
- component extraction assumes each bar is a separate monochrome component and
  does not reuse the baseline color. Generator enforces distinct non-baseline
  bar colors.
- anchor extraction assumes odd bar widths so the median occupied column is the
  true slot center. Generator keeps widths in {1, 3}.
- redistribution assumes moving a shape to another slot will not create overlap.
  Generator uses one common width per sample and non-overlapping slot spacing,
  so remapping heights between existing slots remains legal.

The key changed assumption is the old `zip(slots, shapes)` monotone mapping.
That line no longer works because slot order stays left-to-right but the shape
stream must now be serialized from alternating ends of a descending ranking.

4. Placement legality
Generator is output-first:
- choose grid size, slot positions, colors, width, and distinct odd heights,
- construct the intended output using the fused height order,
- derive an input by permuting the same heights,
- reject unless `main(input)` reproduces that output.

This guarantees:
- no out-of-bounds placement because chosen heights fit the grid,
- no overlap because slots are spaced for the chosen width,
- bar geometry matches what `main()` extracts because bars are solid vertical
  rectangles centered on the baseline.

5. Visual observability
Across a few examples, a human should notice two coupled facts:
- the leftmost slot tends to receive the tallest bar, ruling out ascending sort,
- neighboring slots alternate between very tall and very short bars, ruling out
  simple descending sort.

Requiring at least 4 bars and rejecting outputs equal to simpler baselines keeps
the fusion visibly distinct from the original task and from the single-dimension
variants.

6. New assumptions introduced by the fused solver
New assumptions beyond the original task:
- the height order should be distinct enough that alternating high/low picks are
  readable, so the generator uses distinct heights,
- there must be enough slots for alternation to matter, so the generator uses
  at least 4 bars,
- slot order remains purely geometric (x-position), so generator preserves a
  clean left-to-right arrangement.

7. Audit of the new solver logic
- `_extract_objects()` still assumes the baseline is detectable and every bar is
  separable; generator preserves that.
- sorting `slots` by anchor assumes slot identity is position-based only; this
  remains true.
- sorting `shapes` by descending height assumes height is the primary feature;
  distinct heights make this robust and tie-breakers cover rare equal-shape
  metadata if needed.
- `_alternating_descending_order()` assumes there are enough shapes that taking
  from opposite ends is meaningful; generator enforces `k >= 4`.
- the placement loop assumes each relative shape can be recolored independently;
  monochrome rectangles satisfy that directly.
"""


def _alternating_descending_order(items: list[object]) -> list[object]:
    """Consume a descending-sorted list as high, low, high, low, ..."""
    ordered: list[object] = []
    hi_end = 0
    lo_end = len(items) - 1
    for idx in range(len(items)):
        if idx % 2 == 0:
            ordered.append(items[hi_end])
            hi_end += 1
        else:
            ordered.append(items[lo_end])
            lo_end -= 1
    return ordered


def _extract_objects(g: np.ndarray) -> tuple[int, int, int, int, int, list[dict[str, object]]]:
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

    objects: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {bg, int(base_col)}:
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
    Preserve slot x-positions/colors, but reorder shapes by descending height
    and assign them to slots with alternating direction:
    tallest, shortest, next-tallest, next-shortest, ...
    """
    g = np.array(grid, dtype=int)
    H, W, bg, base_r, base_col, objects = _extract_objects(g)

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes_desc = sorted(
        objects,
        key=lambda o: (-int(o["height"]), -int(o["size"]), int(o["anchor"]), int(o["color"])),
    )
    shape_stream = _alternating_descending_order(shapes_desc)

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
    probs = np.full(int(parts), 1.0 / int(parts))
    return [int(x) for x in np.random.multinomial(int(total), probs)]


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


def _ascending_alternating_order(items: list[int]) -> list[int]:
    ordered: list[int] = []
    lo = 0
    hi = len(items) - 1
    for idx in range(len(items)):
        if idx % 2 == 0:
            ordered.append(int(items[lo]))
            lo += 1
        else:
            ordered.append(int(items[hi]))
            hi -= 1
    return ordered


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(9, 18, 2)))
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

    heights_asc = sorted(int(h) for h in np.random.choice(height_choices, size=k, replace=False))
    heights_desc = list(reversed(heights_asc))
    out_heights = [int(h) for h in _alternating_descending_order(heights_desc)]

    asc_baseline = list(heights_asc)
    desc_baseline = list(heights_desc)
    d10_baseline = _ascending_alternating_order(heights_asc)
    baseline_orders = (asc_baseline, desc_baseline, d10_baseline)
    if any(out_heights == baseline for baseline in baseline_orders):
        return None

    out = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, out_heights)

    input_heights: list[int] | None = None
    for _ in range(30):
        perm = np.random.permutation(k)
        candidate = [int(heights_asc[i]) for i in perm]
        if candidate == out_heights:
            continue
        if candidate == desc_baseline:
            continue
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
