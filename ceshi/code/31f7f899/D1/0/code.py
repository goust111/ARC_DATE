# Variant of 31f7f899
# Dimension: D1 — 匹配条件：按高度排序映射槽位 (match_cond)
# Change: size_ordering — 将条形按高度升序排序，槽位按列位置从左到右排序，第i短的条形放入第i个槽位... → 按颜色匹配对应槽位

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_h, find_components, get_bg, paste

# Design analysis
# - Concrete behavior: the scene keeps the original "horizontal baseline + several
#   monochrome vertical bars" layout, but each slot now has a one-cell color marker
#   on the top row. Input bars are permuted across the existing x-slots; output moves
#   each bar to the slot whose marker has the same color. Bar shapes/heights stay with
#   the source bar, and markers/base are preserved.
# - Trace on one slot assignment: original code would collect all bar shapes, sort them
#   by height, then place the shortest shape at the leftmost slot regardless of color.
#   In the variant, a blue bar of height 7 ignores its height rank and instead moves to
#   the slot whose top marker is blue; a red marker can sit left of a taller green bar,
#   so the destination order is no longer derivable from size ordering.
# - Complexity calibration: markers are singletons and bars always have area >= 3, so
#   there is a clean visual and algorithmic split between "slot descriptor" and "bar".
#   Colors remain unique per bar/slot, keeping matching one-to-one and avoiding
#   ambiguity. Heights are still distinct so the old and new rules can diverge clearly.
# - Hidden assumptions checked from the original solver:
#   * `slots = sorted(objects, key=anchor)` encoded the old left-to-right slot identity;
#     this no longer holds, so slots must be read from dedicated marker components.
#   * `slot_col = slot["color"]` used slot color as the recolor target; in the variant
#     the moving bar keeps its own color, so repainting must use the source bar color.
#   * The old component loop assumed every non-base component was a bar. With markers
#     present, singleton components must be filtered out explicitly.
# - Placement legality: the generator keeps one empty row between the top markers and
#   the tallest bar, and reuses the original spacing logic for slot footprints, so bars
#   neither overlap nor touch markers under 4-connectivity.
# - Visual observability: across 2-4 examples a human can track that every color moves
#   to the same-colored marker, even when the resulting left-to-right height order is
#   not sorted. The generator also rejects samples where the old height-sorting rule
#   would accidentally produce the same output.


def _find_base(g: np.ndarray, bg: int) -> tuple[int, int]:
    best = None  # (count, row, color)
    for r in range(int(g.shape[0])):
        vals, counts = np.unique(g[int(r), :], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == int(bg):
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(r), int(v))
    assert best is not None
    _cnt, base_r, base_col = best
    return int(base_r), int(base_col)


def _extract_scene(
    g: np.ndarray, bg: int, base_r: int, base_col: int
) -> tuple[dict[int, tuple[int, int]], list[dict[str, object]]]:
    markers: dict[int, tuple[int, int]] = {}
    objects: list[dict[str, object]] = []

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {int(bg), int(base_col)}:
            continue

        pts = comp["coords"]
        area = int(comp["area"])
        if area == 1:
            r, c = (int(x) for x in pts[0])
            if r != int(base_r):
                markers[int(col)] = (int(r), int(c))
                continue

        b = comp["bbox"]
        height = int(bbox_h(b))
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

    return markers, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Redistribute vertical bars to the slots whose markers share the same color.

    Input structure:
    - A horizontal base line spans the grid.
    - Several monochrome vertical bars cross the base line.
    - A one-cell marker near the top row marks each destination slot with a color.

    Rule:
    - Each bar keeps its own shape and color.
    - Move the bar to the x-slot whose marker has the same color.
    - Preserve the base line and the markers.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    base_r, base_col = _find_base(g, bg)
    markers, objects = _extract_scene(g, bg, base_r, base_col)

    assert len(markers) == len(objects) and len(objects) > 0

    out = np.full_like(g, int(bg))
    out[int(base_r), :] = int(base_col)
    for color, (mr, mc) in markers.items():
        out[int(mr), int(mc)] = int(color)

    for obj in objects:
        color = int(obj["color"])
        assert color in markers
        _mr, slot_anchor = markers[color]
        for dr, dc in obj["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[int(rr), int(cc)] = int(color)

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
    for i in range(int(k)):
        lefts.append(int(c))
        c += int(width)
        if i < int(k) - 1:
            c += 1 + int(gaps[i + 1])
    return lefts


def _draw_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    lefts: list[int],
    width: int,
    marker_colors: list[int],
    bar_colors: list[int],
    heights: list[int],
) -> np.ndarray:
    grid = np.full((int(n), int(n)), int(bg), dtype=int)
    grid[int(base_r), :] = int(base_color)
    for left, marker_color in zip(lefts, marker_colors):
        anchor = int(left) + int(width // 2)
        grid[0, int(anchor)] = int(marker_color)
    for left, color, height in zip(lefts, bar_colors, heights):
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - height // 2), int(left))
    return grid


def _old_rule_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_r, base_col = _find_base(g, bg)
    markers, objects = _extract_scene(g, bg, base_r, base_col)

    out = np.full_like(g, int(bg))
    out[int(base_r), :] = int(base_col)
    for color, (mr, mc) in markers.items():
        out[int(mr), int(mc)] = int(color)

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(objects, key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])))
    for slot, shape in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                out[int(rr), int(cc)] = int(slot_col)
    return out


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(11, 18, 2)))
    base_r = int(n // 2)
    bg = 8
    base_color = 6

    max_height = min(9, int(2 * (base_r - 2) + 1))
    height_choices = list(range(3, int(max_height) + 1, 2))
    max_count = min(5, len(height_choices), (n + 1) // 2)
    if max_count < 3:
        return None
    k = int(np.random.randint(3, int(max_count) + 1))

    feasible_widths = [1]
    if n - 3 * k > 3 and n >= 4 * k - 1:
        feasible_widths.append(3)
    width = int(np.random.choice(feasible_widths))

    lefts = _slot_lefts(n, k, width)
    palette = [1, 2, 3, 4, 5, 7, 9]
    colors = [int(c) for c in np.random.choice(palette, size=k, replace=False)]

    chosen_heights = [int(h) for h in np.random.choice(height_choices, size=k, replace=False)]
    perm_heights = np.random.permutation(k)
    height_by_color = {int(colors[i]): int(chosen_heights[int(perm_heights[i])]) for i in range(k)}

    marker_order = [int(colors[i]) for i in np.random.permutation(k)]
    out_heights = [int(height_by_color[c]) for c in marker_order]
    out = _draw_scene(n, base_r, base_color, bg, lefts, width, marker_order, marker_order, out_heights)

    input_order = [int(colors[i]) for i in np.random.permutation(k)]
    fixed = sum(int(a == b) for a, b in zip(input_order, marker_order))
    if input_order == marker_order or fixed > k - 2:
        return None
    in_heights = [int(height_by_color[c]) for c in input_order]
    inp = _draw_scene(n, base_r, base_color, bg, lefts, width, marker_order, input_order, in_heights)

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_old_rule_output(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
