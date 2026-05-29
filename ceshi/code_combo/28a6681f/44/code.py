# Combo variant of 28a6681f (index 44)
# combo_spec: D7/1+D16/1
# Name: 连通器原理+更大网格

from __future__ import annotations

import random
from typing import Any

import numpy as np


# Design analysis
# a) Concrete realization and behavior trace
#    - Keep the original visual language of blue water and colored bottom-anchored basin
#      walls, but enlarge the canvas to 15x15 or 20x20 and expand the lower structure into
#      3-5 adjacent cups.
#    - Each cup is visibly sealed below its rim by tall separating walls, so ordinary local
#      falling / sliding cannot move water from one cup into the next.
#    - Small trace: suppose three cups have floor support rows 12, 11, 10 and the tallest
#      visible separating wall reaches row 8. If 18 water cells are poured from a left shelf:
#      original logic would trap most water in the first reachable cup unless water rose to a
#      visible spill edge.
#      fused logic treats every cup in the lower structure as hiddenly connected at the base,
#      so all candidate cup cells are pooled and refilled from the lowest rows upward until the
#      common top water row becomes 10. Water appears in every cup even though 10 is still
#      below row 8's wall tops, so the spread cannot be explained by visible overflow.
#    - The two changes reinforce each other: the larger grid makes room for more cups, and the
#      communicating-vessel rule turns that extra room into a genuinely global equalization task.
# b) Complexity calibration
#    - Use 3-4 cups on 15x15 and 4-5 cups on 20x20 so the equal-height rule is repeated across
#      several basins rather than being inferred from only a pair.
#    - Cup widths are limited to 2-3 columns and floor depths to a small set of 3 bands. This
#      keeps the family readable while still forcing a global level computation.
#    - Water counts are chosen from exact full shared levels, avoiding noisy partial top rows
#      that would blur the intended "same surface in many cups" cue.
# c) Implicit assumption audit against the original solver
#    - Original `settle()` relied on visible empty cells encoding every legal move. Hidden base
#      channels violate that assumption, so trajectory simulation is replaced by geometry-based
#      equalization over parsed basin cells.
#    - Original code accepted any grid size. This combo intentionally narrows the family to
#      15x15 and 20x20 because the larger layout is part of the rule.
#    - Original output validation only required locally stable water. That still holds, but now
#      stability comes from filling whole cup columns up to a shared level, not from iterative
#      motion.
#    - Original overflow reasoning was local and sequential. Here the total water count alone
#      determines the final distribution once the communicating lower structure is identified.
# d) Placement legality
#    - The generator builds one chain of bottom-anchored wall columns and fills every interior
#      basin column with its own bottom floor stack, so all cups are in-bounds and non-overlapping.
#    - The source shelf sits high in the grid and is not bottom-connected, preventing the solver
#      from mistaking it for part of the basin structure.
#    - Width sampling is constrained so the full multi-cup structure always fits with at least a
#      one-column outer margin.
# e) Visual observability
#    - Input water is concentrated on one upper edge shelf; output water is redistributed across
#      several lower cups.
#    - The generator only accepts samples where at least three cups contain water and the shared
#      surface row stays strictly below every separating wall top, making the hidden-channel rule
#      visually necessary.
# f) Visual cleanliness
#    - Only background, solid colors, and blue water appear.
#    - The upper shelf and lower basin chain occupy separate vertical bands, so source and target
#      regions do not overlap visually even on the larger grids.
# g) Color-change inferability
#    - No color remapping is introduced. Blue always denotes water and all other colors are static
#      solids, so the rule must be inferred from spatial redistribution rather than from palette tricks.
# h) Variant metric used after implementation
#    - Measure `(active_basins, shared_surface_row, sealed_equalization)` where `sealed_equalization`
#      means at least three basins contain water, all non-empty basins share one top water row, and
#      that row lies below every visible rim. Generated samples target `sealed_equalization = True`.
# i) New solver assumptions to re-check
#    - `_extract_basins()` assumes the lower structure is a chain of bottom-connected wall columns
#      separating cup intervals whose interior columns are also bottom-connected floor stacks.
#    - `_fill_equalized()` assumes the intended family uses whole shared levels rather than partial
#      surface rows; `generate_input()` enforces this by sampling only exact level counts.
#    - `main()` assumes the hidden connection spans the entire parsed basin chain, so the generator
#      must always create exactly one communicating lower structure.


BG = 0
WATER = 1
ALLOWED_SHAPES = {(15, 15), (20, 20)}


def _bottom_stack_top(structure: np.ndarray, c: int) -> int | None:
    """Top row of a solid stack contiguous to the bottom, else None."""
    h = int(structure.shape[0])
    if int(structure[h - 1, c]) == BG:
        return None
    r = h - 1
    while r - 1 >= 0 and int(structure[r - 1, c]) != BG:
        r -= 1
    return int(r)


def _extract_basins(structure: np.ndarray) -> list[dict[str, Any]]:
    """
    Parse the controlled larger-grid family.

    Bottom-anchored tall stacks are cup walls. The columns between two consecutive walls are
    one basin when every interior column is also a bottom-anchored stack with a shorter height.
    """
    h, w = structure.shape
    wall_threshold = h - 6
    stack_top = [_bottom_stack_top(structure, c) for c in range(w)]
    wall_cols = [c for c, top in enumerate(stack_top) if top is not None and int(top) <= wall_threshold]
    basins: list[dict[str, Any]] = []
    for left, right in zip(wall_cols, wall_cols[1:]):
        if right - left <= 2:
            continue
        inner = list(range(left + 1, right))
        floor_tops = [stack_top[c] for c in inner]
        if any(top is None or int(top) <= wall_threshold for top in floor_tops):
            continue
        rim = max(int(stack_top[left]), int(stack_top[right]))
        supports = {c: int(stack_top[c]) - 1 for c in inner}
        if any(int(support) < rim for support in supports.values()):
            continue
        basins.append(
            {
                "left": int(left),
                "right": int(right),
                "rim": int(rim),
                "supports": supports,
            }
        )
    return basins


def _candidate_cells(structure: np.ndarray) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    basins = _extract_basins(structure)
    cells: list[tuple[int, int]] = []
    for basin in basins:
        rim = int(basin["rim"])
        for c, support in basin["supports"].items():
            for r in range(int(support), rim - 1, -1):
                cells.append((int(r), int(c)))
    cells.sort(key=lambda rc: (-int(rc[0]), int(rc[1])))
    return cells, basins


def _fill_equalized(structure: np.ndarray, water_count: int) -> np.ndarray:
    out = np.array(structure, dtype=int).copy()
    cells, basins = _candidate_cells(out)
    if len(basins) < 3:
        raise AssertionError("expected at least three communicating basins")
    if int(water_count) > len(cells):
        raise AssertionError("too much water for communicating-vessel capacity")
    for r, c in cells[: int(water_count)]:
        out[int(r), int(c)] = WATER
    return out


def _surface_rows(grid: np.ndarray, basins: list[dict[str, Any]]) -> list[int]:
    rows: list[int] = []
    for basin in basins:
        basin_rows: list[int] = []
        for c in basin["supports"]:
            water_rc = np.argwhere(grid[:, int(c)] == WATER)
            if len(water_rc) > 0:
                basin_rows.append(int(water_rc[0, 0]))
        if basin_rows:
            rows.append(min(basin_rows))
    return rows


def _fusion_signature(grid: np.ndarray) -> dict[str, Any]:
    structure = np.where(grid == WATER, BG, grid)
    basins = _extract_basins(structure)
    surface_rows = _surface_rows(grid, basins)
    active_basins = len(surface_rows)
    shared_surface = active_basins > 0 and len(set(surface_rows)) == 1
    max_rim = max((int(basin["rim"]) for basin in basins), default=-1)
    sealed = bool(shared_surface and surface_rows[0] > max_rim and active_basins >= 3)
    return {
        "basin_count": int(len(basins)),
        "active_basins": int(active_basins),
        "surface_rows": surface_rows,
        "sealed_equalization": sealed,
    }


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Equalize blue water across a larger multi-basin vessel using a hidden communicating-channel rule.

    Input structure:
    - Background is 0 and water is color 1. Other colors are fixed solids.
    - The grid is 15x15 or 20x20. The lower region contains one chain of 3-5 bottom-anchored cups
      separated by tall wall columns, while the upper left or upper right contains a supported source shelf.
    - Water begins packed on that shelf instead of already occupying the lower cups.

    Rule:
    - Count all water pixels, erase them from the source, and treat every parsed lower cup as if its base were
      connected to the others by hidden pipes.
    - Refill the union of cup cells from the lowest available rows upward, so the total water amount determines
      one global level across all reachable cups.
    - The visible walls, floors, and shelf do not move.

    Key insight: The result depends on the total volume and the shared lower vessel geometry, not on visible local
    drop paths; several sealed-looking cups end with the same water surface because they communicate below the floor.
    """
    g = np.array(grid, dtype=int).copy()
    if g.shape not in ALLOWED_SHAPES:
        raise AssertionError("grid must be 15x15 or 20x20")

    if check_input:
        for r, c in np.argwhere(g == WATER):
            rr = int(r)
            cc = int(c)
            if rr < g.shape[0] - 1:
                assert int(g[rr + 1, cc]) != BG, "input water must be supported"

    water_count = int(np.sum(g == WATER))
    structure = np.where(g == WATER, BG, g)
    basins = _extract_basins(structure)
    if len(basins) < 3:
        raise AssertionError("could not parse a communicating multi-basin structure")
    out = _fill_equalized(structure, water_count)

    if check_output:
        for r, c in np.argwhere(out == WATER):
            rr = int(r)
            cc = int(c)
            assert rr == g.shape[0] - 1 or int(out[rr + 1, cc]) != BG
            assert cc == 0 or int(out[rr, cc - 1]) != BG
            assert cc == g.shape[1] - 1 or int(out[rr, cc + 1]) != BG
    return out


def _choose_widths(rng: random.Random, w: int) -> list[int]:
    count_choices = [3, 4] if w == 15 else [4, 5]
    for _ in range(200):
        basin_count = int(rng.choice(count_choices))
        widths = [int(rng.choice([2, 3])) for _ in range(basin_count)]
        span = sum(widths) + basin_count + 1
        if span <= w - 2:
            return widths
    raise AssertionError("failed to sample basin widths")


def _build_structure(rng: random.Random) -> tuple[np.ndarray, list[dict[str, Any]]]:
    h = int(rng.choice([15, 20]))
    w = h
    grid = np.zeros((h, w), dtype=int)
    palette = rng.sample([2, 3, 4, 5, 6, 7, 8, 9], k=4)

    widths = _choose_widths(rng, w)
    span = sum(widths) + len(widths) + 1
    left_wall = int(rng.randint(1, w - span - 1))
    wall_cols = [int(left_wall)]
    cur = int(left_wall)
    for width in widths:
        cur += int(width) + 1
        wall_cols.append(int(cur))

    wall_choices = [h // 2 - 1, h // 2, h // 2 + 1]
    floor_choices = [h - 4, h - 3, h - 2]
    wall_tops = [int(rng.choice(wall_choices)) for _ in wall_cols]
    if len(set(wall_tops)) == 1:
        wall_tops[-1] = int(wall_choices[(wall_choices.index(wall_tops[-1]) + 1) % len(wall_choices)])

    floor_tops = [int(rng.choice(floor_choices)) for _ in widths]
    if len(set(floor_tops)) == 1:
        floor_tops[0] = int(floor_choices[(floor_choices.index(floor_tops[0]) + 1) % len(floor_choices)])

    for idx, c in enumerate(wall_cols):
        grid[int(wall_tops[idx]) :, int(c)] = int(palette[idx % len(palette)])

    for basin_idx, width in enumerate(widths):
        color = int(palette[(basin_idx + 1) % len(palette)])
        floor_top = int(floor_tops[basin_idx])
        for c in range(wall_cols[basin_idx] + 1, wall_cols[basin_idx + 1]):
            grid[int(floor_top) :, int(c)] = color

    basins = _extract_basins(grid)
    if len(basins) != len(widths):
        raise AssertionError("generated structure did not parse cleanly")
    return grid, basins


def _shared_level_count(basins: list[dict[str, Any]], level: int) -> int:
    return int(
        sum(int(support) - int(level) + 1 for basin in basins for support in basin["supports"].values())
    )


def _source_layout(shape: tuple[int, int], side: str) -> tuple[int, list[int]]:
    h, w = shape
    shelf_row = 4 if h == 15 else 5
    shelf_width = 6 if w == 15 else 7
    cols = list(range(shelf_width)) if side == "left" else list(range(w - shelf_width, w))
    return int(shelf_row), cols


def _source_capacity(shape: tuple[int, int], side: str) -> int:
    shelf_row, cols = _source_layout(shape, side)
    return int(shelf_row * len(cols))


def _valid_levels(basins: list[dict[str, Any]], capacity: int) -> list[tuple[int, int]]:
    max_rim = max(int(basin["rim"]) for basin in basins)
    min_support = min(int(support) for basin in basins for support in basin["supports"].values())
    levels: list[tuple[int, int]] = []
    for level in range(max_rim + 1, min_support + 1):
        water_count = _shared_level_count(basins, int(level))
        if 12 <= water_count <= int(capacity):
            levels.append((int(level), int(water_count)))
    return levels


def _add_source_shelf(grid: np.ndarray, side: str, color: int) -> None:
    shelf_row, cols = _source_layout(grid.shape, side)
    grid[int(shelf_row), cols] = int(color)


def _place_source_water(grid: np.ndarray, water_count: int, side: str) -> None:
    shelf_row, cols = _source_layout(grid.shape, side)
    col_order = cols if side == "left" else list(reversed(cols))
    order: list[tuple[int, int]] = []
    for c in col_order:
        for r in range(shelf_row - 1, -1, -1):
            order.append((int(r), int(c)))
    if int(water_count) > len(order):
        raise AssertionError("source shelf cannot support this much water")
    for r, c in order[: int(water_count)]:
        if int(grid[r, c]) != BG:
            raise AssertionError("source placement overlapped solids")
        grid[int(r), int(c)] = WATER


def generate_input() -> np.ndarray:
    for _ in range(500):
        rng = random.Random(random.randrange(1 << 30))
        structure, basins = _build_structure(rng)
        source_side = str(rng.choice(["left", "right"]))
        capacity = _source_capacity(structure.shape, source_side)
        levels = _valid_levels(basins, capacity)
        if not levels:
            continue

        level, water_count = rng.choice(levels)
        shelf_color = int(rng.choice([2, 3, 4, 5, 6, 7, 8, 9]))
        base = np.array(structure, dtype=int).copy()
        _add_source_shelf(base, source_side, shelf_color)

        output = _fill_equalized(base, int(water_count))
        signature = _fusion_signature(output)
        if signature["basin_count"] != len(basins):
            continue
        if signature["active_basins"] != len(basins):
            continue
        if not signature["sealed_equalization"]:
            continue

        inp = np.array(base, dtype=int).copy()
        _place_source_water(inp, int(water_count), source_side)
        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, output):
            return inp
    raise RuntimeError("failed to generate a valid larger-grid communicating-vessel sample")
