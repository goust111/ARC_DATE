# Variant of 31f7f899
# Dimension: D1 — 匹配条件：按高度排序映射槽位 (match_cond)
# Change: size_ordering — 将条形按高度升序排序，槽位按列位置从左到右排序，第i短的条形放入第i个槽位... → 按宽度排序映射

from __future__ import annotations

"""
Design analysis for the D1 width-sorting variant.

a) Concrete behavior and trace
   - Visual structure stays close to the source task: a full-width horizontal base line,
     plus several monochrome rectangular bars centered on that base row.
   - The original task used equal-width bars and redistributed shapes by HEIGHT.
     That made the vertical extent the only relevant size cue.
   - This variant makes the bars have visibly different odd widths {1, 3, 5} while still
     allowing their heights to vary independently. The slots are still the original x
     positions, and the slot colors are still preserved.
   - Example trace with three bars:
       slot colors left->right = [red, green, yellow]
       input shapes at those slots = [(w=5,h=3), (w=1,h=9), (w=3,h=5)]
       original logic would sort by h: [(5,3), (3,5), (1,9)] placed left->right
       variant logic sorts by w:   [(1,9), (3,5), (5,3)] placed left->right
     The moved silhouettes differ horizontally, so the rule change is directly visible.

b) Complexity calibration
   - Widths are limited to three discrete odd values {1, 3, 5}. This is enough to make the
     horizontal key obvious, but small enough to stay ARC-like and avoid a noisy search space.
   - Each sample uses exactly three bars, one per width. This removes tie ambiguity in the
     match step and guarantees that width ordering is visually supported inside every example.
   - Heights remain distinct odd values, but their order is intentionally de-correlated from
     the width order so the samples do not collapse back to the original height-sorting task.

c) Audit of original assumptions that changed
   - The source solver stored only `height` and implicitly assumed width was irrelevant.
     Here width must be extracted from the bbox and used as the primary sort key.
   - The source generator placed every slot with one shared width. That is no longer safe:
     once widths vary, placing a wide shape into a narrow slot can overlap neighbors unless
     the slot geometry is designed for the MAX width, not the original width.
   - The anchor computation via the median column on the base row still works because all
     widths are odd and the rectangles are horizontally centered on their slot anchors.
   - Repainting the full base row before drawing bars remains safe because the bars are pasted
     afterwards, exactly like the source task.

d) Placement legality
   - Slot anchors are chosen by reserving disjoint width-5 footprints. Any width in {1,3,5}
     can then be centered on any slot without overlap or clipping.
   - Heights are odd and at most the grid size, so vertical centering on the base row stays
     in bounds.
   - The generator is output-first: build the width-sorted target, permute shapes to make the
     input, and verify `main(input) == output` before accepting the sample.

e) Visual observability
   - A human can see that bars differ by both width and height, but only the widths line up
     systematically from thin to thick in the outputs.
   - To prevent accidental degeneration, the generator rejects samples where width order and
     height order induce the same redistributed output. So across examples, the horizontal cue
     is necessary rather than incidental.

New implicit assumptions after the change
   - `main()` assumes bars are monochrome connected components distinct from the base color.
   - It assumes slot anchors are meaningful centers for odd-width rectangles.
   - The generator guarantees those assumptions by using distinct bar colors, odd widths,
     centered placement, and non-overlapping width-5 slot reservations.
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
                "width": width,
                "size": int(len(rel)),
                "rel": rel,
            }
        )

    return bg, int(base_r), int(base_col), objects


def _redistribute(grid: np.ndarray, key_name: str) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg, base_r, base_col, objects = _extract_scene(g)

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
    out[base_r, :] = base_col

    for slot, shape in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in shape["rel"]:
            rr = base_r + int(dr)
            cc = slot_anchor + int(dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = slot_col

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort bars by width and redistribute them across fixed colored slots.

    Slots are still the original x-positions ordered left->right, and each slot preserves
    its original color identity. Only the ranking key changes: width, not height.
    """
    return _redistribute(grid, "width")


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


def _height_reference(grid: np.ndarray) -> np.ndarray:
    return _redistribute(grid, "height")


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice([15, 17]))
    base_r = n // 2
    bg = 8
    base_color = 6
    k = 3

    widths = [1, 3, 5]
    anchors = _slot_anchors(n, k, max(widths))
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]

    height_choices = list(range(1, min(n, 11) + 1, 2))
    heights_sorted = sorted(int(h) for h in np.random.choice(height_choices, size=k, replace=False))

    # Assign heights to the width-ranked shapes in a non-monotone way so width order != height order.
    width_to_height_perm = np.random.permutation(k)
    if np.array_equal(width_to_height_perm, np.arange(k)):
        width_to_height_perm = np.array([1, 2, 0])
    target_shapes = [(int(widths[i]), int(heights_sorted[width_to_height_perm[i]])) for i in range(k)]

    # Then scramble which shape appears in each colored slot in the input.
    input_perm = np.random.permutation(k)
    if np.array_equal(input_perm, np.arange(k)):
        input_perm = np.array([1, 2, 0])
    input_shapes = [target_shapes[int(i)] for i in input_perm]

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
