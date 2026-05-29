# Combo variant of 31f7f899 (index 2)
# combo_spec: D1/4+D14/0
# Name: 按面积排序+宽度各异

from __future__ import annotations

"""
Design analysis for the combo variant "按面积排序+宽度各异".

a) Concrete behavior and trace
   - Scene skeleton stays close to the source task: one full-width horizontal base line
     plus several monochrome bars that cross that line.
   - Source task:
       * all bars inside one sample shared the same width
       * sorting key was height only
       * the i-th shortest bar moved into the i-th leftmost slot
   - Combo task:
       * each bar has its own odd width from {1, 3, 5}
       * heights also vary
       * sorting key becomes area = width * height
       * the i-th smallest-area bar moves into the i-th leftmost slot and is recolored
         to that slot's original color
   - Example with three slots whose colors left->right are [red, green, yellow]:
       input shapes = [(w=1,h=9), (w=5,h=3), (w=3,h=5)]
       heights ascending = [(5,3), (3,5), (1,9)]
       areas ascending   = [15, 15, 9] would be ambiguous, so such a sample is banned
       Instead use [(w=1,h=9), (w=5,h=3), (w=3,h=7)]:
       heights ascending = [(5,3), (3,7), (1,9)]
       areas ascending   = [(1,9)=9, (5,3)=15, (3,7)=21]
       Original rule would place widths [5,3,1] left->right.
       Combo rule places widths [1,5,3] left->right.
     The difference is visually driven by jointly considering width and height.

b) Complexity calibration
   - Widths are limited to {1, 3, 5}. That is enough to create nontrivial area order
     while keeping the hypothesis space ARC-like.
   - Each sample uses exactly three bars, one per width. This avoids width ties and
     makes the width vocabulary visible in every example.
   - Heights are distinct odd values, but the generator rejects samples where:
       * area order has any tie
       * area order equals height order
       * area order equals width order
     This prevents collapse back to a simpler variant.

c) Hidden-assumption audit of the original solver
   - Base-line detection:
       The original code picks the most frequent non-background color count on a row.
       This still works only if the base row keeps more cells of the base color than
       any bar color gets on any row. The generator guarantees this by using exactly
       widths {1,3,5} on 15x15 or 17x17 grids, so the base row keeps 6 or 8 base-color
       cells, strictly more than the widest bar's 5 cells.
   - Component extraction:
       The original solver assumes each bar is a single monochrome 4-connected object.
       Still valid because every generated bar is one solid rectangle and colors are
       distinct across bars.
   - Anchor logic:
       The original anchor uses the median column on the base row, which implicitly
       assumes a stable center. This is preserved by restricting widths/heights to odd
       numbers, so every bar has a unique center column crossing the base row.
   - Sorting key:
       Original code stored only height because shared width made that sufficient.
       This breaks here. The solver must measure per-object width and area.
   - Placement safety:
       Original generator reserved space based on one global width. That would fail
       once a width-5 shape can be moved into a slot that originally held width 1.
       The combo generator instead reserves each slot as a width-5 footprint and
       centers narrower bars inside it, so any shape can legally move to any slot.

d) Placement legality
   - Grid size is chosen from {15, 17}; base row is the middle row.
   - Slots are represented by center anchors whose width-5 footprints are disjoint.
   - Bars are odd-height rectangles centered on the base row, so every bar crosses the
     base line and stays within bounds.
   - Because every slot can host width 5, redistributing shapes cannot create overlap
     or clipping.

e) Visual observability
   - Across examples, the output is not monotone in height and not monotone in width.
     It is monotone in area, which requires combining both dimensions.
   - The generator explicitly rejects samples where the combo output equals the
     height-sorted output, so the fusion behavior is required rather than accidental.

New implicit assumptions introduced by this solver
   - Bars are solid odd-by-odd rectangles crossing the detected base row.
   - Colors are unique per bar, so monochromatic connected components identify bars.
   - Areas are unique in generated samples, avoiding tie ambiguity in the mapping.
   - The generator enforces these assumptions before accepting a sample.
"""

import numpy as np
from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste


def _extract_scene(grid: np.ndarray) -> tuple[int, int, int, list[dict[str, object]]]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    best = None  # (count, row, color)
    for r in range(h):
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
        box = comp["bbox"]
        height = int(bbox_h(box))
        width = int(bbox_w(box))
        area = int(comp["area"])

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
                "width": int(width),
                "area": int(area),
                "size": int(len(rel)),
                "rel": rel,
            }
        )

    return bg, int(base_r), int(base_col), objects


def _redistribute(grid: np.ndarray, key_name: str) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg, base_r, base_col, objects = _extract_scene(g)
    assert objects

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(
        objects,
        key=lambda o: (
            int(o[key_name]),
            int(o["height"]),
            int(o["width"]),
            int(o["size"]),
            int(o["anchor"]),
            int(o["color"]),
        ),
    )

    out = np.full((h, w), bg, dtype=int)
    out[base_r, :] = int(base_col)

    for slot, shape in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(slot_col)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort bars by pixel area and redistribute them across fixed colored slots.

    Slots are the original x-positions ordered left->right. Each slot keeps its color
    identity; only bar silhouettes are permuted, now using area instead of height.
    """
    return _redistribute(grid, "area")


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_anchors(n: int, k: int, max_width: int) -> list[int]:
    used = int(k * max_width)
    extra = int(n - used)
    if extra < 0:
        raise ValueError("width-5 slot footprints do not fit")

    gaps = _rand_composition(extra, k + 1)
    anchors: list[int] = []
    left = int(gaps[0])
    for i in range(k):
        anchors.append(int(left + max_width // 2))
        left += int(max_width)
        if i < k - 1:
            left += int(gaps[i + 1])
    return anchors


def _draw_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    anchors: list[int],
    colors: list[int],
    shapes: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)
    for anchor, color, (width, height) in zip(anchors, colors, shapes):
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        top = int(base_r - height // 2)
        left = int(anchor - width // 2)
        paste(grid, patch, top, left)
    return grid


def _order(values: list[int]) -> tuple[int, ...]:
    return tuple(i for i, _v in sorted(enumerate(values), key=lambda iv: (int(iv[1]), int(iv[0]))))


def _sample_target_shapes(n: int) -> list[tuple[int, int]] | None:
    widths = [1, 3, 5]
    height_choices = list(range(1, min(n, 11) + 1, 2))

    for _ in range(120):
        heights = [int(h) for h in np.random.choice(height_choices, size=3, replace=False)]
        shapes_by_width = [(int(widths[i]), int(heights[i])) for i in range(3)]

        areas = [int(w * h) for w, h in shapes_by_width]
        only_heights = [int(h) for _w, h in shapes_by_width]
        only_widths = [int(w) for w, _h in shapes_by_width]

        if len(set(areas)) != 3:
            continue

        area_order = _order(areas)
        height_order = _order(only_heights)
        width_order = _order(only_widths)

        if area_order == height_order:
            continue
        if area_order == width_order:
            continue

        return [shapes_by_width[int(i)] for i in area_order]
    return None


def _height_reference(grid: np.ndarray) -> np.ndarray:
    return _redistribute(grid, "height")


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice([15, 17]))
    base_r = n // 2
    bg = 8
    base_color = 6
    k = 3

    anchors = _slot_anchors(n, k, 5)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]

    target_shapes = _sample_target_shapes(n)
    if target_shapes is None:
        return None

    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        perm = np.array([1, 2, 0])
    input_shapes = [target_shapes[int(i)] for i in perm]

    out = _draw_scene(n, base_r, base_color, bg, anchors, colors, target_shapes)
    inp = _draw_scene(n, base_r, base_color, bg, anchors, colors, input_shapes)

    if np.array_equal(inp, out):
        return None
    if np.array_equal(main(inp), _height_reference(inp)):
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
