# Variant of 28a6681f
# Dimension: D9 — 溢流机制 (path)
# Change: 水填满一个盆地后从较低壁顶溢出流入相邻盆地，约50%样本展示可见溢流... → 连通器原理（水位趋于平衡）

from __future__ import annotations

import random

import numpy as np

# Design analysis
# a) Concrete geometry and behavior:
#    - The original task relied on open basins plus local falling / sliding. That cannot express
#      true communicating vessels because a water pixel would need to move down through a connector
#      and then rise inside a neighboring chamber.
#    - The variant therefore keeps the "bottom-anchored colored container" vocabulary but turns the
#      basin group into two enclosed vertical chambers linked by a one-row-low connector corridor.
#      Input water is piled inside only one wider source chamber. Output redistributes the same
#      total water across the whole enclosed cavity so that both chambers end at the same height.
#    - Small trace:
#      original logic on a left-full / right-empty pair would leave most water in the left cup
#      until a wall-top overflow exists; variant logic counts all water in the connected cavity and
#      refills the cavity from the bottom upward, so the right chamber gains water even though this
#      requires "rising" after passing through the bottom connector.
# b) Complexity calibration:
#    - Two chambers are enough to make the new rule visible. The source chamber is width 3 and the
#      other chamber width 2, so input is clearly lopsided while output is clearly balanced.
#    - Water volumes are sampled only from exact cumulative row capacities. This avoids ambiguous
#      partial top rows and makes the flat shared waterline human-visible.
# c) Original-code assumptions that break under D9:
#    - `settle()` assumes every move is local and monotone downward/sideways. That blocks the
#      upward leg required by a communicating-vessel connector, so it must be replaced.
#    - The old output check forbids background on left/right/below every water pixel. That matches
#      wedged basin water, but a flat shared level inside a wider chamber naturally has water with
#      air on one side, so the stability criterion must change.
#    - Guide-specific diagonal-connectivity assertions are irrelevant here because this variant no
#      longer uses edge-entering ramps as the core mechanism.
# d) Placement legality:
#    - The generator fixes a 10x10 grid, chooses a container span of width 8, adds one internal
#      partition that stops one row above the floor, and keeps the open cavity away from grid edges.
#      This guarantees no overlap, no leaks, and a single enclosed connected vessel component.
# e) Visual observability:
#    - Input: one chamber is tall and full while the neighbor is empty.
#    - Output: both chambers and the connector show one common horizontal waterline.
#      This is visibly different from wall-top overflow because the distant chamber fills without
#      needing the source chamber to crest over a higher local rim.
# f) Visual cleanliness:
#    - Only water and 2-3 solid colors are used, with a single large container and one partition.
#      No overlapping markers or hidden color encodings are needed.
# g) Color inference:
#    - Colors do not encode ordering or behavior. They only distinguish static walls/roof/partition.
# New solver assumptions audited against the generator:
#    - Every water-bearing open component is an enclosed cavity that does not touch the grid edge.
#    - The sampled water amount matches a whole communicating level: full lower rows, no partial
#      shared surface row. `generate_input()` enforces this exactly.
#    - Components are simple monotone chambers over a connector; no overhang-dependent edge cases
#      are generated.


def _collect_component(
    open_mask: np.ndarray, start: tuple[int, int], seen: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    H, W = open_mask.shape
    stack = [start]
    seen.add(start)
    comp: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        comp.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            nxt = (rr, cc)
            if 0 <= rr < H and 0 <= cc < W and open_mask[rr, cc] and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return comp


def _group_rows(component: list[tuple[int, int]]) -> dict[int, list[tuple[int, int]]]:
    by_row: dict[int, list[tuple[int, int]]] = {}
    for r, c in component:
        by_row.setdefault(r, []).append((r, c))
    for cells in by_row.values():
        cells.sort()
    return by_row


def _fill_lowest_rows(
    component: list[tuple[int, int]], water_count: int, require_exact_level: bool
) -> set[tuple[int, int]]:
    by_row = _group_rows(component)
    filled: set[tuple[int, int]] = set()
    remaining = int(water_count)
    for r in sorted(by_row, reverse=True):
        row_cells = by_row[r]
        if remaining >= len(row_cells):
            filled.update(row_cells)
            remaining -= len(row_cells)
            continue
        if remaining == 0:
            break
        if require_exact_level:
            raise AssertionError("water volume does not match a complete shared waterline")
        filled.update(row_cells[:remaining])
        remaining = 0
        break
    if remaining != 0:
        raise AssertionError("water volume exceeds vessel capacity")
    return filled


def _assert_flat_level(component: list[tuple[int, int]], filled: set[tuple[int, int]]) -> None:
    by_row = _group_rows(component)
    seen_air = False
    for r in sorted(by_row, reverse=True):
        row_cells = by_row[r]
        row_has_water = [cell in filled for cell in row_cells]
        if any(row_has_water) and not all(row_has_water):
            raise AssertionError("communicating-vessel row cannot be partially filled")
        if all(row_has_water):
            assert not seen_air, "water must occupy the lowest rows only"
        else:
            seen_air = True


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Rebalance water across connected vessel chambers until one common level is reached.

    Input structure:
    - bg = 0 and water = 1.
    - Other colors form bottom-anchored walls, a roof, and an internal partition whose bottom gap
      links neighboring chambers into one enclosed vessel network.
    - Water begins as a supported pile inside one source chamber while other connected chambers are
      partially empty.

    Rule:
    - For each enclosed open cavity that already contains water, count the total number of water
      pixels in that connected cavity.
    - Ignore the initial lopsided pile and refill that cavity from the lowest rows upward.
    - Because the chambers share a low connector, all connected chambers finish with the same
      horizontal waterline.

    Key insight: The answer depends on total water volume versus the full cross-section of the
    connected cavity at each height, not on local overflow paths over the partition top.
    """
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    bg = 0
    water = 1
    initial_water = int(np.count_nonzero(g == int(water)))

    if check_input:
        for r, c in np.argwhere(g == int(water)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(bg), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(bg)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(bg)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    open_mask = np.logical_or(g == int(bg), g == int(water))
    seen: set[tuple[int, int]] = set()
    for r, c in np.argwhere(g == int(water)):
        start = (int(r), int(c))
        if start in seen:
            continue
        component = _collect_component(open_mask, start, seen)
        touches_edge = any(rr in (0, H - 1) or cc in (0, W - 1) for rr, cc in component)
        if check_input:
            assert not touches_edge, "water must lie inside an enclosed communicating-vessel cavity"
        if touches_edge:
            continue
        water_count = sum(1 for rr, cc in component if int(g[rr, cc]) == int(water))
        filled = _fill_lowest_rows(component, water_count, require_exact_level=check_input)
        for rr, cc in component:
            g[rr, cc] = int(bg)
        for rr, cc in filled:
            g[rr, cc] = int(water)

    if check_output:
        assert int(np.count_nonzero(g == int(water))) == initial_water, "water count must be conserved"
        open_mask = np.logical_or(g == int(bg), g == int(water))
        seen = set()
        for r, c in np.argwhere(g == int(water)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(bg), "output water must be supported"
            start = (rr, cc)
            if start in seen:
                continue
            component = _collect_component(open_mask, start, seen)
            touches_edge = any(r0 in (0, H - 1) or c0 in (0, W - 1) for r0, c0 in component)
            assert not touches_edge, "output water leaked out of the vessel"
            filled = {(r0, c0) for r0, c0 in component if int(g[r0, c0]) == int(water)}
            _assert_flat_level(component, filled)

    return g


def _draw_case(
    top: int,
    left_wall: int,
    partition_col: int,
    source_on_left: bool,
    colors: tuple[int, int, int],
) -> tuple[np.ndarray, list[tuple[int, int]], list[int], list[int]]:
    g = np.zeros((10, 10), dtype=int)
    right_wall = left_wall + 7
    wall_color, roof_color, partition_color = colors

    g[9, left_wall : right_wall + 1] = wall_color
    g[top, left_wall : right_wall + 1] = roof_color
    g[top:10, left_wall] = wall_color
    g[top:10, right_wall] = wall_color
    g[top + 1 : 8, partition_col] = partition_color

    open_mask = np.logical_or(g == 0, g == 1)
    component = _collect_component(open_mask, (8, left_wall + 1), set())

    if source_on_left:
        source_cols = list(range(left_wall + 1, partition_col))
        other_cols = list(range(partition_col + 1, right_wall))
        source_order = source_cols[:]
    else:
        source_cols = list(range(partition_col + 1, right_wall))
        other_cols = list(range(left_wall + 1, partition_col))
        source_order = source_cols[::-1]

    assert len(source_cols) == 3
    assert len(other_cols) == 2
    return g, component, source_cols, source_order


def _exact_water_counts(top: int) -> list[int]:
    counts: list[int] = []
    source_capacity = 3 * (8 - top)
    for levels_above_connector in range(1, 7):
        water_count = 6 + 5 * levels_above_connector
        if water_count <= source_capacity:
            counts.append(water_count)
    return counts


def _make_output(base: np.ndarray, component: list[tuple[int, int]], water_count: int) -> np.ndarray:
    out = np.array(base, dtype=int).copy()
    for r, c in _fill_lowest_rows(component, water_count, require_exact_level=True):
        out[r, c] = 1
    return out


def _make_input(
    base: np.ndarray, top: int, source_order: list[int], water_count: int
) -> np.ndarray:
    inp = np.array(base, dtype=int).copy()
    remaining = int(water_count)
    for r in range(8, top, -1):
        for c in source_order:
            if remaining == 0:
                break
            inp[r, c] = 1
            remaining -= 1
        if remaining == 0:
            break
    if remaining != 0:
        raise AssertionError("source chamber was too small for the sampled water volume")
    return inp


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    top = random.choice([1, 2])
    left_wall = random.choice([0, 1, 2])
    source_on_left = bool(random.randrange(2))
    partition_col = left_wall + (4 if source_on_left else 3)
    colors = tuple(random.sample([2, 3, 4, 5, 6, 7, 8, 9], 3))

    base, component, _source_cols, source_order = _draw_case(
        top=top,
        left_wall=left_wall,
        partition_col=partition_col,
        source_on_left=source_on_left,
        colors=colors,
    )
    water_count = random.choice(_exact_water_counts(top))
    inp = _make_input(base, top, source_order, water_count)
    out = _make_output(base, component, water_count)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(100):
        inp, out = _build_case()
        solved = main(inp)
        if np.array_equal(solved, out):
            return inp
    raise AssertionError("failed to generate a valid communicating-vessel sample")
