# Variant of 31f7f899
# Dimension: D1 — 匹配条件：按高度排序映射槽位 (match_cond)
# Change: size_ordering — 将条形按高度升序排序，槽位按列位置从左到右排序，第i短的条形放入第i个槽位... → 按条与基线交叉宽度排序

from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg


# Design analysis for the D1 variant
# ----------------------------------
# 1) Concrete behavior:
#    The original task uses plain vertical bars whose only informative attribute is height.
#    This variant keeps the same global scaffold:
#    - one full-width horizontal baseline,
#    - several colored objects attached to / crossing that baseline,
#    - slot identity still comes from the original left-to-right anchor positions and colors.
#    The changed part is the matching key. Each object now has the same bbox height and the
#    same total pixel count, but a different number of cells touching the baseline. The solver
#    must sort objects by that "footprint" width on the baseline: 1, 3, 5.
#
#    Small trace:
#    - old logic: if three bars had heights 3, 5, 7 at anchors x=2,7,12, the shortest bar
#      moved to the left slot, the middle bar to the middle slot, and the tallest to the right.
#    - new logic here: the three objects all have height 5 and size 9, but their baseline
#      footprints are 5, 1, 3 in the input. The solver must rank them as 1 < 3 < 5 and place
#      them into the left, middle, right slots respectively.
#    - difference: height is now a useless cue; only the contact width on the baseline matters.
#
# 2) Complexity calibration:
#    If footprint width were allowed to correlate with height or total size, the task could
#    silently collapse back into the original one because sorting by height/size would still
#    work. To prevent that, all generated objects share:
#    - identical bbox height: 5
#    - identical area: 9 pixels
#    - distinct baseline footprint widths: 1, 3, 5
#    This keeps the new cue visually obvious while removing old accidental cues.
#
# 3) Hidden-assumption audit of the original main():
#    - `height = bbox_h(b)`: the original solver assumes height is the discriminative attribute.
#      In this variant that assumption is false, so height can no longer drive matching.
#    - `base_cols` and `anchor`: this still works, but it now becomes more important because
#      wide footprints have several baseline cells. The generator therefore uses odd widths and
#      centers every shape on a single anchor so the median baseline column is stable.
#    - `rel = ... (r-base_r, c-anchor)`: still valid, but only if every object actually touches
#      the baseline. The generator guarantees that by construction.
#    - `shapes = sorted(... height, size, ...)`: this is the key line that must change. In the
#      new task, height and size are intentionally equalized, so sorting must use footprint width.
#
# 4) Placement legality:
#    The widest object spans 5 columns around its anchor. Anchors are therefore sampled with a
#    minimum gap of 5 columns and with a 2-column border margin. That guarantees:
#    - no object overlaps another in the input or output,
#    - no object crosses the grid boundary after reassignment,
#    - every generated scene remains solvable by the same anchor-based relocation logic.
#
# 5) Visual observability:
#    Across 2-4 examples, a human can notice that all objects are equally tall, yet the output
#    orders them from narrowest baseline contact to widest. Because colors stay tied to slots
#    rather than shapes, the "shape moved, slot color stayed" pattern from the original task
#    remains intact, while the discriminating feature is visibly different.
#
# 6) New assumptions introduced by this solver:
#    - every non-baseline object has at least one pixel on the baseline,
#    - baseline footprint widths are distinct enough to define an ordering,
#    - anchors are far enough apart that relocating a wider footprint cannot create overlaps.
#    `generate_input()` enforces all three conditions.


_SHAPE_BY_FOOTPRINT: dict[int, list[tuple[int, int]]] = {
    # Same height (5) and same area (9), different width where the object crosses the baseline.
    1: [(-2, -1), (-2, 0), (-2, 1), (-1, 0), (0, 0), (1, 0), (2, -1), (2, 0), (2, 1)],
    3: [(-2, 0), (-1, 0), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1), (2, 0)],
    5: [(-2, 0), (-1, 0), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

_FOOTPRINTS = [1, 3, 5]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort baseline-crossing objects by footprint width on the baseline, then redistribute them
    across the fixed left-to-right colored slots.
    """
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

    objects: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {bg, int(base_col)}:
            continue

        pts = [(int(r), int(c)) for r, c in comp["coords"]]
        base_cols = sorted(c for r, c in pts if r == int(base_r))
        if base_cols:
            anchor = int(base_cols[len(base_cols) // 2])
        else:
            cols_sorted = sorted(c for _r, c in pts)
            anchor = int(cols_sorted[len(cols_sorted) // 2])

        rel = [(r - int(base_r), c - int(anchor)) for r, c in pts]
        objects.append(
            {
                "color": int(col),
                "anchor": int(anchor),
                "footprint": int(len(base_cols)),
                "size": int(len(rel)),
                "rel": rel,
            }
        )

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(objects, key=lambda o: (int(o["footprint"]), int(o["size"]), int(o["anchor"]), int(o["color"])))

    out = np.full_like(g, bg)
    out[int(base_r), :] = int(base_col)

    for slot, shape in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(slot_col)

    return out


def _draw_shape(grid: np.ndarray, base_r: int, anchor: int, color: int, footprint: int) -> None:
    for dr, dc in _SHAPE_BY_FOOTPRINT[int(footprint)]:
        rr = int(base_r) + int(dr)
        cc = int(anchor) + int(dc)
        grid[rr, cc] = int(color)


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _sample_anchors(n: int) -> list[int]:
    # Border margin = 2 on both sides. Minimum anchor gap = 5 so width-5 footprints never overlap.
    extra = int(n - 15)
    if extra < 0:
        raise ValueError("grid too small for width-ordered variant")
    g0, g1, g2, _g3 = _rand_composition(extra, 4)
    a0 = 2 + g0
    a1 = a0 + 5 + g1
    a2 = a1 + 5 + g2
    return [int(a0), int(a1), int(a2)]


def _draw_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    anchors: list[int],
    colors: list[int],
    footprints: list[int],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)
    for anchor, color, footprint in zip(anchors, colors, footprints):
        _draw_shape(grid, int(base_r), int(anchor), int(color), int(footprint))
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice([15, 17, 19]))
    base_r = n // 2
    bg = 8
    base_color = 6
    k = len(_FOOTPRINTS)

    anchors = _sample_anchors(n)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]

    out = _draw_scene(n, base_r, base_color, bg, anchors, colors, list(_FOOTPRINTS))

    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    in_footprints = [int(_FOOTPRINTS[i]) for i in perm]
    inp = _draw_scene(n, base_r, base_color, bg, anchors, colors, in_footprints)

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
