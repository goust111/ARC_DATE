# Combo variant of 31f7f899 (index 17)
# combo_spec: D4/4+D11/1
# Name: 双向标记系统
#
# Design analysis for this combo variant
# -------------------------------------
# a) Concrete behavior trace
# - Original task:
#   1. Detect the bars crossing the long base row.
#   2. Sort bar shapes by height.
#   3. Sort destination slots by left-to-right position.
#   4. Put the i-th shortest shape into the i-th leftmost slot, recolored to that slot's color.
# - This combo changes both control axes at once.
#   * Each slot has a floating vertical marker above it. The marker area encodes that slot's
#     receiving rank: 1 pixel = first receiving slot, 2 pixels = second, etc.
#   * Each source bar has a side marker in a separate right-hand column. The number of isolated
#     dots in that column encodes which receiving rank that bar's shape should go to.
# - Small concrete trace with three bars at anchors A,B,C:
#   * slot-marker orders above [A,B,C] are [2,3,1], so slot C is the first receiver,
#     slot A is the second receiver, slot B is the third receiver.
#   * source-side targets next to [A,B,C] are [3,1,2], so the shape at A asks for receiver 3,
#     the shape at B asks for receiver 1, the shape at C asks for receiver 2.
#   * Therefore:
#       shape(A) -> slot B,
#       shape(B) -> slot C,
#       shape(C) -> slot A.
#   * The final colors are still determined by the destination slots, not by the moved shapes.
# - Difference from the original:
#   the original used one implicit chain "shortest -> leftmost". Here both halves are explicit:
#   "which source shape moves" is read from the bar-side dots, and "which physical slot is rank i"
#   is read from the slot-side stacks.
#
# b) Complexity calibration
# - To keep the fused rule observable but not noisy:
#   * bar width is fixed to 1,
#   * the number alphabet is restricted to counts 1..k with k in {3,4},
#   * slot markers and bar-side markers use different dedicated colors,
#   * every sample uses each count exactly once in both systems.
# - This avoids ambiguous component parsing and makes the two instruction layers visually distinct:
#   top stacks identify slots, right-side dot columns identify source bars.
# - The generator rejects cases where either control layer becomes irrelevant, so the dataset does
#   not silently collapse back to the original task or to either single-variant behavior.
#
# c) Implicit-assumption audit of the original solver
# - Original component loop:
#     "every non-base component is a movable bar"
#   Broken here, because we add two families of non-base instruction components. The new parser
#   must separate: bars touching the base, slot markers above anchors, and source markers in the
#   side columns.
# - Original slot ordering:
#     `sorted(objects, key=lambda o: anchor)`
#   Broken by D4/4, because destination rank is no longer spatial; it comes from the slot markers.
# - Original shape ordering:
#     `sorted(objects, key=lambda o: height, ...)`
#   Broken by D11/1, because the moved shape is no longer determined by height; it comes from the
#   explicit bar-side target marker.
# - Original rendering kept only the base line.
#   Broken for this combo because slot markers are part of the persistent slot structure and must
#   survive into the output. The bar-side markers are input-only instructions and are removed.
# - New assumptions introduced by the combo and enforced by the generator:
#   * each slot marker column is exactly the bar anchor column,
#   * each source-marker column is exactly two columns to the right of its bar,
#   * slot-marker areas are a permutation of 1..k,
#   * source-marker dot counts are a permutation of 1..k,
#   * bars are short enough that output bars never overwrite the top slot-marker band.
#
# d) Placement legality
# - Horizontal layout per slot is:
#     [bar][blank][source-marker], plus at least one blank before the next slot.
#   This guarantees no overlap between bars and side markers.
# - Slot markers sit in the top rows of the bar column itself, so bar heights are bounded by
#   `base_r - height//2 >= k + 1`; this leaves at least one empty separator row below the deepest
#   slot marker.
# - Source markers are isolated single pixels placed in one reserved side column, with one empty
#   row between consecutive dots under 4-connectivity.
# - The generator validates `main(input) == output` and also validates that weaker interpretations
#   (original / D4-only / D11-only) do not reproduce the same output.
#
# e) Visual observability
# - Across several examples a solver can see that:
#   * the top stacks scramble which physical slot is "1st / 2nd / 3rd receiver",
#   * the side dot columns scramble which source bar goes to receiver 1/2/3,
#   * neither height order alone nor left-to-right order alone explains the output.
# - Because the generator rejects monotone left-to-right outputs and single-variant degeneracies,
#   the two instruction systems stay behaviorally visible rather than decorative.

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste


def _find_base(grid: np.ndarray, bg: int) -> tuple[int, int]:
    best: tuple[int, int, int] | None = None  # (count, row, color)
    H, _W = grid.shape
    for r in range(H):
        vals, counts = np.unique(grid[r, :], return_counts=True)
        for v, cnt in zip(vals, counts):
            col = int(v)
            area = int(cnt)
            if col == bg:
                continue
            if best is None or area > best[0]:
                best = (area, int(r), col)
    assert best is not None
    _count, base_r, base_col = best
    return int(base_r), int(base_col)


def _anchor_from_pts(pts: np.ndarray, base_r: int) -> int:
    base_cols = sorted(int(c) for r, c in pts if int(r) == int(base_r))
    if base_cols:
        return int(base_cols[len(base_cols) // 2])
    cols_sorted = sorted(int(c) for _r, c in pts)
    return int(cols_sorted[len(cols_sorted) // 2])


def _parse_scene(
    grid: np.ndarray,
) -> tuple[int, int, int, list[dict[str, object]], list[dict[str, object]]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_r, base_col = _find_base(g, bg)

    bars: list[dict[str, object]] = []
    floating_by_color: dict[int, list[dict[str, object]]] = {}

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {bg, base_col}:
            continue

        pts = comp["coords"]
        if any(int(r) == int(base_r) for r, _c in pts):
            anchor = _anchor_from_pts(pts, base_r)
            rel = [(int(r) - int(base_r), int(c) - int(anchor)) for r, c in pts]
            bars.append(
                {
                    "color": col,
                    "anchor": int(anchor),
                    "right": int(comp["bbox"][3]),
                    "height": int(bbox_h(comp["bbox"])),
                    "size": int(len(rel)),
                    "rel": rel,
                }
            )
        else:
            floating_by_color.setdefault(col, []).append(comp)

    k = len(bars)
    assert k >= 3

    anchors = {int(bar["anchor"]) for bar in bars}
    side_cols = {int(bar["right"]) + 2 for bar in bars}

    slot_marker_color: int | None = None
    for col, comps in floating_by_color.items():
        areas = sorted(int(comp["area"]) for comp in comps)
        cols = sorted(int(comp["bbox"][1]) for comp in comps)
        if len(comps) != k:
            continue
        if areas != list(range(1, k + 1)):
            continue
        if cols != sorted(anchors):
            continue
        if not all(int(comp["bbox"][0]) == 0 and int(bbox_w(comp["bbox"])) == 1 for comp in comps):
            continue
        slot_marker_color = int(col)
        break
    assert slot_marker_color is not None

    source_marker_color: int | None = None
    source_count_by_col: Counter[int] | None = None
    for col, comps in floating_by_color.items():
        if int(col) == int(slot_marker_color):
            continue
        if not all(int(comp["area"]) == 1 for comp in comps):
            continue
        counts = Counter(int(comp["bbox"][1]) for comp in comps)
        if sorted(counts.keys()) != sorted(side_cols):
            continue
        if sorted(int(v) for v in counts.values()) != list(range(1, k + 1)):
            continue
        source_marker_color = int(col)
        source_count_by_col = counts
        break
    assert source_marker_color is not None
    assert source_count_by_col is not None

    slot_markers: list[dict[str, object]] = []
    slot_order_by_anchor: dict[int, int] = {}
    for comp in floating_by_color[int(slot_marker_color)]:
        anchor = int(comp["bbox"][1])
        order = int(comp["area"])
        slot_order_by_anchor[anchor] = order
        slot_markers.append(
            {
                "color": int(slot_marker_color),
                "coords": [(int(r), int(c)) for r, c in comp["coords"]],
            }
        )

    for bar in bars:
        anchor = int(bar["anchor"])
        side_col = int(bar["right"]) + 2
        bar["slot_order"] = int(slot_order_by_anchor[anchor])
        bar["target"] = int(source_count_by_col[side_col])

    assert sorted(int(bar["slot_order"]) for bar in bars) == list(range(1, k + 1))
    assert sorted(int(bar["target"]) for bar in bars) == list(range(1, k + 1))
    return bg, base_r, base_col, bars, slot_markers


def _render_policy(
    shape: tuple[int, int],
    bg: int,
    base_r: int,
    base_col: int,
    bars: list[dict[str, object]],
    slot_markers: list[dict[str, object]],
    *,
    slot_mode: str,
    shape_mode: str,
) -> np.ndarray:
    H, W = map(int, shape)
    out = np.full((H, W), int(bg), dtype=int)
    out[int(base_r), :] = int(base_col)

    for marker in slot_markers:
        col = int(marker["color"])
        for r, c in marker["coords"]:
            out[int(r), int(c)] = col

    if slot_mode == "spatial":
        slots = sorted(bars, key=lambda o: (int(o["anchor"]), int(o["color"])))
    else:
        slots = sorted(bars, key=lambda o: (int(o["slot_order"]), int(o["anchor"]), int(o["color"])))

    if shape_mode == "height":
        shapes = sorted(
            bars,
            key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])),
        )
        pairs = list(zip(slots, shapes))
    else:
        shapes_by_target = {int(bar["target"]): bar for bar in bars}
        pairs = [(slot, shapes_by_target[idx]) for idx, slot in enumerate(slots, start=1)]

    for slot, shape_obj in pairs:
        slot_anchor = int(slot["anchor"])
        slot_color = int(slot["color"])
        for dr, dc in shape_obj["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(slot_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the two marker systems jointly:
    - slot-side stack size says which receiving rank a slot represents,
    - bar-side dot count says which receiving rank that source shape wants.
    Move each source shape to the slot with the matching rank, recoloring to the slot color.
    """
    g = np.array(grid, dtype=int)
    bg, base_r, base_col, bars, slot_markers = _parse_scene(g)
    return _render_policy(
        g.shape,
        bg,
        base_r,
        base_col,
        bars,
        slot_markers,
        slot_mode="order",
        shape_mode="target",
    )


def _render_original(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, base_r, base_col, bars, slot_markers = _parse_scene(g)
    return _render_policy(
        g.shape,
        bg,
        base_r,
        base_col,
        bars,
        slot_markers,
        slot_mode="spatial",
        shape_mode="height",
    )


def _render_d4_only(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, base_r, base_col, bars, slot_markers = _parse_scene(g)
    return _render_policy(
        g.shape,
        bg,
        base_r,
        base_col,
        bars,
        slot_markers,
        slot_mode="order",
        shape_mode="height",
    )


def _render_d11_only(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, base_r, base_col, bars, slot_markers = _parse_scene(g)
    return _render_policy(
        g.shape,
        bg,
        base_r,
        base_col,
        bars,
        slot_markers,
        slot_mode="spatial",
        shape_mode="target",
    )


def _rand_composition(total: int, parts: int) -> list[int]:
    probs = np.full(int(parts), 1.0 / int(parts))
    return [int(x) for x in np.random.multinomial(int(total), probs)]


def _slot_anchors(n: int, k: int) -> list[int]:
    used = int(3 * k + (k - 1))
    extra = int(n - used)
    if extra < 0:
        raise ValueError("bars and source markers do not fit")
    gaps = _rand_composition(extra, k + 1)
    anchors: list[int] = []
    c = int(gaps[0])
    for i in range(k):
        anchors.append(int(c))
        c += 3
        if i < k - 1:
            c += 1 + int(gaps[i + 1])
    return anchors


def _height_choices(base_r: int, k: int) -> list[int]:
    max_height = min(9, 2 * (int(base_r) - (int(k) + 1)) + 1)
    return [int(h) for h in range(1, max_height + 1, 2)]


def _source_marker_rows(n: int, base_r: int) -> list[int]:
    return [int(r) for r in range(0, int(n), 2) if int(r) != int(base_r)]


def _draw_output_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    anchors: list[int],
    colors: list[int],
    heights: list[int],
    slot_orders: list[int],
    slot_marker_color: int,
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[int(base_r), :] = int(base_color)

    for anchor, order in zip(anchors, slot_orders):
        patch = np.full((int(order), 1), int(slot_marker_color), dtype=int)
        paste(grid, patch, 0, int(anchor))

    for anchor, color, height in zip(anchors, colors, heights):
        patch = np.full((int(height), 1), int(color), dtype=int)
        paste(grid, patch, int(base_r - height // 2), int(anchor))

    return grid


def _draw_input_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    anchors: list[int],
    colors: list[int],
    heights: list[int],
    slot_orders: list[int],
    slot_marker_color: int,
    source_targets: list[int],
    source_marker_color: int,
) -> np.ndarray:
    grid = _draw_output_scene(
        n,
        base_r,
        base_color,
        bg,
        anchors,
        colors,
        heights,
        slot_orders,
        slot_marker_color,
    )
    rows = _source_marker_rows(n, base_r)
    for anchor, target in zip(anchors, source_targets):
        marker_c = int(anchor) + 2
        for r in rows[: int(target)]:
            grid[int(r), int(marker_c)] = int(source_marker_color)
    return grid


def _permutation_displacement(values: list[int]) -> int:
    return int(sum(1 for i, v in enumerate(values, start=1) if int(v) != int(i)))


def _height_rank_targets(heights: list[int], targets: list[int]) -> list[int]:
    ranked = sorted(zip(heights, targets), key=lambda x: (int(x[0]), int(x[1])))
    return [int(target) for _height, target in ranked]


def _output_heights_by_anchor(input_heights: list[int], slot_orders: list[int], targets: list[int]) -> list[int]:
    shape_by_target = {int(target): int(height) for height, target in zip(input_heights, targets)}
    return [int(shape_by_target[int(order)]) for order in slot_orders]


def _gen_attempt() -> np.ndarray | None:
    bg = 8
    base_color = 6
    slot_marker_color = 5
    source_marker_color = 7

    feasible: list[tuple[int, int]] = []
    for n in (15, 17, 19):
        base_r = n // 2
        for k in (3, 4):
            if len(_height_choices(base_r, k)) < k:
                continue
            if len(_source_marker_rows(n, base_r)) < k:
                continue
            if n - (3 * k + (k - 1)) < 0:
                continue
            feasible.append((n, k))
    if not feasible:
        return None

    n, k = feasible[int(np.random.randint(len(feasible)))]
    base_r = n // 2

    anchors = _slot_anchors(n, k)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 9], size=k, replace=False)]
    input_heights = [int(h) for h in np.random.choice(_height_choices(base_r, k), size=k, replace=False)]

    slot_orders = [int(x) for x in np.random.permutation(np.arange(1, k + 1))]
    if _permutation_displacement(slot_orders) < 2:
        return None

    source_targets = [int(x) for x in np.random.permutation(np.arange(1, k + 1))]
    if _permutation_displacement(source_targets) < 2:
        return None

    if _height_rank_targets(input_heights, source_targets) == list(range(1, k + 1)):
        return None

    out_heights = _output_heights_by_anchor(input_heights, slot_orders, source_targets)
    if out_heights == input_heights:
        return None
    if out_heights == sorted(out_heights):
        return None

    inp = _draw_input_scene(
        n,
        base_r,
        base_color,
        bg,
        anchors,
        colors,
        input_heights,
        slot_orders,
        slot_marker_color,
        source_targets,
        source_marker_color,
    )
    out = _draw_output_scene(
        n,
        base_r,
        base_color,
        bg,
        anchors,
        colors,
        out_heights,
        slot_orders,
        slot_marker_color,
    )

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_render_original(inp), out):
        return None
    if np.array_equal(_render_d4_only(inp), out):
        return None
    if np.array_equal(_render_d11_only(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
