# Variant of 31f7f899
# Dimension: D4 — 目标槽位：原地替换 (dst)
# Change: 条形的原始列位置即为槽位，保持位置不变，只替换条形形状... → 槽位由标记像素指定顺序
#
# Design analysis for this D4 variant
# -----------------------------------
# Reference-task note:
# - The prompt cites 581f7754 as the inspiration for "marker_alignment".
# - In the provided workspace, only `/ARC-data/.../581f7754.json` exists; no solver file for 581f7754 is present.
# - So the implementation baseline here uses the source-task idea at the mechanism level only:
#   slot order is instructed by small marker objects rather than by spatial order.
#
# a) Concrete behavior trace
# - Original task:
#   1. Find all bar objects crossing the base row.
#   2. Sort bars by height.
#   3. Sort destination slots by x-position.
#   4. Put the shortest shape into the leftmost slot, recolored to that slot's color.
# - Variant:
#   1. Each slot/bar column has a small marker floating above it.
#   2. The marker is a same-color vertical stack whose pixel count encodes the slot's receiving rank:
#      1 pixel = first destination, 2 pixels = second destination, etc.
#   3. Shapes are still sorted by height, but destination slots are sorted by marker size, not by x.
# - Small example:
#   columns [2, 5, 8] with marker sizes [3, 1, 2] and bar heights [7, 3, 5].
#   Original output would assign heights [3, 5, 7] to x-order [2, 5, 8].
#   Variant output assigns height 3 to column 5 (marker=1), height 5 to column 8 (marker=2),
#   and height 7 to column 2 (marker=3).
#   The bars stay in place as slots; only the rule used to rank slots changes.
#
# b) Complexity calibration
# - If markers were too tiny or random-looking, the new control layer would be hard to infer.
# - If markers touched the bars, component extraction would become ambiguous.
# - To keep the rule readable from 2-4 ARC examples:
#   * bar widths are fixed at 1 column,
#   * markers all share one dedicated color,
#   * marker sizes are exactly 1..k with k in {3,4},
#   * the marker order is forced to be non-left-to-right so the variant is visually different
#     from the original task.
#
# c) Implicit-assumption audit of the original solver
# - Original `slots = sorted(objects, key=lambda o: anchor)` assumes x-order is the destination rule.
#   This is the main broken assumption under D4 and must be replaced.
# - Original component loop treats every non-base object as a bar. With added markers, that would wrongly
#   include instruction symbols as movable shapes. The new solver therefore splits components into:
#   * bars: components touching the base row,
#   * markers: components not touching the base row.
# - Original rendering preserved only the base line. In this variant, markers are part of the scene semantics,
#   so the solver must copy them unchanged into the output.
#
# d) Placement legality
# - Markers sit in a reserved top band: rows `0..order-1`.
# - The generator bounds bar heights so even the tallest output bar leaves at least one blank row below
#   the deepest marker. This guarantees no overlap after reassignment.
# - Bar anchors are sampled with spacing so columns remain visually distinct.
# - Because markers are vertical stacks on the same column as their slot, slot↔marker matching is exact.
#
# e) Visual observability
# - A human sees that bars are still being sorted by height, but the receiving order no longer follows
#   left-to-right position. The small stacks above the bars explain why.
# - The generator rejects identity marker order, so examples do not silently collapse back to the
#   original spatial-order behavior.

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, find_components, get_bg, paste, randomly_spaced_indices


def _find_base(grid: np.ndarray, bg: int) -> tuple[int, int]:
    best = None  # (count, row, color)
    for r in range(grid.shape[0]):
        vals, counts = np.unique(grid[r, :], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == bg:
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(r), int(v))
    assert best is not None
    _cnt, base_r, base_col = best
    return int(base_r), int(base_col)


def _anchor_from_pts(pts: np.ndarray, base_r: int) -> int:
    base_cols = sorted(int(c) for r, c in pts if int(r) == int(base_r))
    if base_cols:
        return int(base_cols[len(base_cols) // 2])
    cols_sorted = sorted(int(c) for _r, c in pts)
    return int(cols_sorted[len(cols_sorted) // 2])


def _parse_scene(grid: np.ndarray) -> tuple[int, int, int, list[dict[str, object]], list[dict[str, object]]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_r, base_col = _find_base(g, bg)

    bars: list[dict[str, object]] = []
    markers: list[dict[str, object]] = []

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {bg, base_col}:
            continue

        pts = comp["coords"]
        anchor = _anchor_from_pts(pts, base_r)
        touches_base = any(int(r) == int(base_r) for r, _c in pts)

        if touches_base:
            rel = [(int(r) - int(base_r), int(c) - int(anchor)) for r, c in pts]
            bars.append(
                {
                    "color": col,
                    "anchor": anchor,
                    "height": int(bbox_h(comp["bbox"])),
                    "size": int(len(rel)),
                    "rel": rel,
                }
            )
        else:
            markers.append(
                {
                    "color": col,
                    "anchor": anchor,
                    "order": int(comp["area"]),
                    "coords": [(int(r), int(c)) for r, c in pts],
                }
            )

    marker_by_anchor = {int(m["anchor"]): m for m in markers}
    for bar in bars:
        anchor = int(bar["anchor"])
        assert anchor in marker_by_anchor
        bar["order"] = int(marker_by_anchor[anchor]["order"])

    assert len(bars) >= 3
    assert len(markers) == len(bars)
    return bg, base_r, base_col, bars, markers


def _render_output(
    g: np.ndarray,
    bg: int,
    base_r: int,
    base_col: int,
    bars: list[dict[str, object]],
    markers: list[dict[str, object]],
    *,
    spatial_slots: bool,
) -> np.ndarray:
    H, W = g.shape
    out = np.full_like(g, bg)
    out[base_r, :] = int(base_col)

    for marker in markers:
        col = int(marker["color"])
        for r, c in marker["coords"]:
            out[int(r), int(c)] = col

    if spatial_slots:
        slots = sorted(bars, key=lambda o: (int(o["anchor"]), int(o["color"])))
    else:
        slots = sorted(bars, key=lambda o: (int(o["order"]), int(o["anchor"]), int(o["color"])))
    shapes = sorted(bars, key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])))

    for slot, shape in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = slot_col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Sort bar shapes by height (shortest -> tallest), same as the original task.
    - Destination slots are no longer ranked left-to-right.
    - Each slot has a floating marker; the marker's pixel count gives the slot's receiving rank.
    - The i-th shortest shape goes to the slot whose marker has size i, recolored to that slot's color.
    - Base line and markers are preserved.
    """
    g = np.array(grid, dtype=int)
    bg, base_r, base_col, bars, markers = _parse_scene(g)
    return _render_output(g, bg, base_r, base_col, bars, markers, spatial_slots=False)


def _legacy_spatial_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, base_r, base_col, bars, markers = _parse_scene(g)
    return _render_output(g, bg, base_r, base_col, bars, markers, spatial_slots=True)


def _height_choices(base_r: int, k: int) -> list[int]:
    # Tallest bar must stay below the reserved marker band (rows 0..k-1) with a blank separator row.
    max_height = min(9, 2 * (int(base_r) - (int(k) + 1)) + 1)
    return [int(h) for h in range(1, max_height + 1, 2)]


def _draw_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    anchors: list[int],
    colors: list[int],
    heights: list[int],
    orders: list[int],
    marker_color: int,
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)

    for anchor, order in zip(anchors, orders):
        patch = np.full((int(order), 1), int(marker_color), dtype=int)
        paste(grid, patch, 0, int(anchor))

    for anchor, color, height in zip(anchors, colors, heights):
        patch = np.full((int(height), 1), int(color), dtype=int)
        paste(grid, patch, int(base_r - height // 2), int(anchor))

    return grid


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice([15, 17, 19]))
    base_r = n // 2
    bg = 8
    base_color = 6

    feasible_ks = [k for k in (3, 4) if len(_height_choices(base_r, k)) >= k]
    if not feasible_ks:
        return None
    k = int(np.random.choice(feasible_ks))

    anchors = [int(x) for x in sorted(randomly_spaced_indices(n, k, border_size=1, padding=1))]

    palette = [1, 2, 3, 4, 5, 7, 9]
    marker_color = int(np.random.choice(palette))
    bar_palette = [c for c in palette if int(c) != marker_color]
    colors = [int(c) for c in np.random.choice(bar_palette, size=k, replace=False)]

    heights_sorted = sorted(int(h) for h in np.random.choice(_height_choices(base_r, k), size=k, replace=False))

    orders = [int(x) for x in np.random.permutation(np.arange(1, k + 1))]
    if orders == list(range(1, k + 1)):
        a, b = np.random.choice(k, size=2, replace=False)
        orders[a], orders[b] = orders[b], orders[a]

    out_heights = [int(heights_sorted[order - 1]) for order in orders]
    out = _draw_scene(n, base_r, base_color, bg, anchors, colors, out_heights, orders, marker_color)

    perm = [int(x) for x in np.random.permutation(k)]
    in_heights = [int(heights_sorted[i]) for i in perm]
    if in_heights == out_heights:
        a, b = np.random.choice(k, size=2, replace=False)
        in_heights[a], in_heights[b] = in_heights[b], in_heights[a]

    inp = _draw_scene(n, base_r, base_color, bg, anchors, colors, in_heights, orders, marker_color)

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_legacy_spatial_output(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
