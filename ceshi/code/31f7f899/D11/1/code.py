# Variant of 31f7f899
# Dimension: D11 — 符号编码参数 (meta)
# Change: 无符号编码——排序规则（升序、左到右）是隐含的固定规则... → 标记像素指定目标槽位映射

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import bbox_h, find_components, get_bg, paste

#
# Reference solver reviewed for regeneration:
# - /home/lixinyu/snap/new_code/solvers/arc2/9385bd28.py
#   That solver reads explicit symbolic markers near the controlled structures rather than
#   inferring behavior from geometry alone. This variant keeps the same spirit for 31f7f899:
#   each bar carries a local instruction token, and the solver decodes that token before
#   reusing the original slot-color-preserving redistribution logic.
#

# Design analysis for this D11 variant
# a) Concretization and behavior trace
# - The original task has k colored vertical bars crossing one long base row. The output
#   keeps the slot x-positions and slot colors, but the bar silhouettes are redistributed.
# - In this variant, each input bar has a small grey marker column two cells to its right.
#   The marker is made of 1..k isolated single-pixel dots in that column. The dot count
#   explicitly says which left-to-right slot this bar must move to.
# - Concrete trace with 3 bars:
#   original logic: input heights [7,3,5] at slots [A,B,C] -> shortest/middle/tallest go
#   to slots [A,B,C], so output heights become [3,5,7].
#   new logic: keep the same input bars, but attach dot counts [3,1,2]. Now the bar at A
#   goes to slot 3, the bar at B goes to slot 1, and the bar at C goes to slot 2, so the
#   output heights become [3,5,7] only if the markers happen to encode that order. In the
#   generator we deliberately choose a non-sorted target arrangement so the output is driven
#   by the symbols rather than by height ranking.
# - The main change therefore happens at the "choose which shape goes to which slot" stage:
#   implicit `sort by height` is replaced by explicit `read target slot from nearby dots`.
#
# b) Complexity calibration
# - We keep the scene archetype intact: same base line, same recoloring-to-slot behavior,
#   same fixed slots, same vertical bars.
# - To make the symbol observable without adding too much freedom, the marker alphabet is
#   only the counts 1..k, with k in {3,4}. Each sample uses every count exactly once.
# - Bar heights remain distinct and nontrivial (odd heights >= 3), so moving a shape is
#   visible. Output heights are forced to be non-ascending, preventing accidental recovery
#   of the old "sort by height" story from the examples.
#
# c) Audit of original solver assumptions
# - Base-row detection still works: markers never touch the base row, so the row with the
#   largest non-background count is still the base line.
# - The original code assumed every non-base, non-bg component was a bar. That is no longer
#   true; the grey dots are extra components and would be misread as tiny bars.
# - This follows the same kind of parser split seen in `9385bd28.py`: symbolic marker pieces
#   must be separated from the task objects before the transformation logic runs.
# - The original `shapes = sorted(objects, key=height,...)` depends on the old invariant
#   that height alone encodes the destination. This invariant is intentionally broken here.
# - The original slot detection by anchor still holds because each real bar still occupies
#   its original slot and crosses the base row.
#
# d) Placement legality
# - Each slot reserves `bar width + 1 blank + 1 marker column`, plus at least one blank
#   column before the next slot. This prevents bar/marker overlap and keeps markers uniquely
#   associated with the bar immediately to their left.
# - Marker dots are placed in a fixed parity row set and are separated by one empty row, so
#   each dot remains a 1-pixel component under 4-connectivity.
# - The generator validates `main(input) == output` before accepting a sample.
#
# e) Visual observability
# - Across 2-4 examples a human can see that bars with equal-looking geometry sometimes move
#   to non-height-sorted destinations, while the grey dot counts next to them match the
#   destination slot index. Because the final left-to-right heights are forced away from
#   monotone ascending order, the old explanation is no longer sufficient.
#
# New solver assumptions reviewed against the generator
# - There is exactly one marker color, and all its components are single pixels away from the
#   base row. The generator enforces this by fixing a dedicated grey marker color.
# - Each bar's marker column is exactly `bbox right + 2`. The generator constructs input
#   layout with one blank column between each bar and its marker.
# - Target indices are a permutation of 1..k. The generator assigns every target exactly once.


def _detect_base(grid: np.ndarray, bg: int) -> tuple[int, int]:
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


def _parse_scene(grid: np.ndarray) -> tuple[int, int, int, list[dict[str, object]]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_r, base_col = _detect_base(g, bg)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    by_color: dict[int, list[dict[str, object]]] = {}
    for comp in comps:
        col = int(comp["color"])
        if col in {bg, base_col}:
            continue
        by_color.setdefault(col, []).append(comp)

    marker_candidates: list[int] = []
    for col, color_comps in by_color.items():
        if all(int(comp["area"]) == 1 for comp in color_comps) and all(
            int(comp["coords"][0][0]) != int(base_r) for comp in color_comps
        ):
            marker_candidates.append(int(col))
    assert marker_candidates
    marker_color = max(marker_candidates, key=lambda col: len(by_color[col]))

    marker_count_by_col = Counter(
        int(comp["bbox"][1]) for comp in by_color[marker_color]
    )

    objects: list[dict[str, object]] = []
    for comp in comps:
        col = int(comp["color"])
        if col in {bg, base_col, marker_color}:
            continue

        pts = comp["coords"]
        b = comp["bbox"]
        height = int(bbox_h(b))

        base_cols = sorted(int(c) for r, c in pts if int(r) == int(base_r))
        if base_cols:
            anchor = int(base_cols[len(base_cols) // 2])
        else:
            cols_sorted = sorted(int(c) for _r, c in pts)
            anchor = int(cols_sorted[len(cols_sorted) // 2])

        rel = [(int(r) - int(base_r), int(c) - int(anchor)) for r, c in pts]
        target = int(marker_count_by_col.get(int(b[3]) + 2, 0))
        assert target >= 1
        objects.append(
            {
                "color": col,
                "anchor": anchor,
                "height": height,
                "size": int(len(rel)),
                "rel": rel,
                "target": target,
            }
        )

    targets = sorted(int(obj["target"]) for obj in objects)
    assert targets == list(range(1, len(objects) + 1))
    return bg, base_r, base_col, objects


def _render_by_target(
    shape: tuple[int, int],
    bg: int,
    base_r: int,
    base_col: int,
    objects: list[dict[str, object]],
) -> np.ndarray:
    H, W = map(int, shape)
    out = np.full((H, W), int(bg), dtype=int)
    out[int(base_r), :] = int(base_col)

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes_by_target = {int(obj["target"]): obj for obj in objects}

    for slot_index, slot in enumerate(slots, start=1):
        shape_obj = shapes_by_target[int(slot_index)]
        slot_anchor = int(slot["anchor"])
        slot_color = int(slot["color"])
        for dr, dc in shape_obj["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = slot_color
    return out


def _render_by_height(
    shape: tuple[int, int],
    bg: int,
    base_r: int,
    base_col: int,
    objects: list[dict[str, object]],
) -> np.ndarray:
    H, W = map(int, shape)
    out = np.full((H, W), int(bg), dtype=int)
    out[int(base_r), :] = int(base_col)

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(
        objects,
        key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])),
    )

    for slot, shape_obj in zip(slots, shapes):
        slot_anchor = int(slot["anchor"])
        slot_color = int(slot["color"])
        for dr, dc in shape_obj["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = slot_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read explicit dot-count instructions attached to each bar and move each bar to the
    requested slot while preserving the slot's original color identity.
    """
    g = np.array(grid, dtype=int)
    bg, base_r, base_col, objects = _parse_scene(g)
    return _render_by_target(g.shape, bg, base_r, base_col, objects)


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _marker_rows(n: int, base_r: int) -> list[int]:
    return [int(r) for r in range(0, int(n), 2) if int(r) != int(base_r)]


def _slot_lefts(n: int, k: int, width: int) -> list[int]:
    used = int(k * width + 2 * k + (k - 1))
    extra = int(n - used)
    if extra < 0:
        raise ValueError("bars and markers do not fit")
    gaps = _rand_composition(extra, k + 1)
    lefts: list[int] = []
    c = int(gaps[0])
    for i in range(k):
        lefts.append(int(c))
        c += int(width) + 2
        if i < k - 1:
            c += 1 + int(gaps[i + 1])
    return lefts


def _draw_bars(
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


def _draw_input_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    lefts: list[int],
    width: int,
    colors: list[int],
    heights: list[int],
    marker_color: int,
    targets: list[int],
) -> np.ndarray:
    grid = _draw_bars(n, base_r, base_color, bg, lefts, width, colors, heights)
    rows = _marker_rows(n, base_r)
    for left, target in zip(lefts, targets):
        marker_c = int(left) + int(width) + 1
        for r in rows[: int(target)]:
            grid[int(r), int(marker_c)] = int(marker_color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(7, 18, 2)))
    base_r = n // 2
    bg = 8
    base_color = 6
    marker_color = 5

    marker_rows = _marker_rows(n, base_r)
    height_choices = list(range(3, min(n, 11) + 1, 2))

    feasible: list[tuple[int, int]] = []
    max_k = min(4, len(height_choices), len(marker_rows))
    for k in range(3, max_k + 1):
        for width in (1, 3):
            if n - (k * width + 2 * k + (k - 1)) >= 0:
                feasible.append((k, width))
    if not feasible:
        return None

    k, width = feasible[int(np.random.randint(len(feasible)))]
    lefts = _slot_lefts(n, k, width)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 7, 9], size=k, replace=False)]

    chosen_heights = [int(h) for h in np.random.choice(height_choices, size=k, replace=False)]
    final_heights = chosen_heights.copy()
    for _ in range(20):
        np.random.shuffle(final_heights)
        if final_heights != sorted(final_heights):
            break
    if final_heights == sorted(final_heights):
        return None

    out = _draw_bars(n, base_r, base_color, bg, lefts, width, colors, final_heights)

    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        a, b = np.random.choice(k, size=2, replace=False)
        perm[a], perm[b] = perm[b], perm[a]
    in_heights = [int(final_heights[i]) for i in perm]
    targets = [int(i) + 1 for i in perm]
    inp = _draw_input_scene(
        n, base_r, base_color, bg, lefts, width, colors, in_heights, marker_color, targets
    )

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None

    bg2, base_r2, base_col2, objects2 = _parse_scene(inp)
    if np.array_equal(_render_by_height(inp.shape, bg2, base_r2, base_col2, objects2), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
