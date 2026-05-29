# Combo variant of 28a6681f (index 8)
# combo_spec: D8/1+D9/2
# Name: 多盆地+单向溢流

from __future__ import annotations

import random

import numpy as np


# Design analysis for combo D8/1 + D9/2
# --------------------------------------
# a) Concretization and behavior trace
#    - Keep the source-task visual family: 10x10 grid, blue water (1), black background (0),
#      bottom-anchored solid walls, and an upper side scaffold that holds the source water.
#    - The fused change fixes the lower area to a long chain of 4 or 5 narrow basins. The top
#      scaffold touches exactly one grid edge, so its side is the global direction clue:
#          left scaffold  -> overflow may progress only left-to-right
#          right scaffold -> overflow may progress only right-to-left
#    - Small example:
#         widths = [1, 1, 2, 1, 1], divider tops = [7, 6, 7, 7], direction = left->right.
#         Capacities before each one-way spill are [2, 3, 4, 2, ...].
#         Original 28a6681f-style bilateral reasoning would inspect local geometry and could let
#         water redistribute to whichever neighboring side is lower.
#         Fused rule ignores reverse opportunities: with 11 water pixels, basin 1 gets 2, basin 2
#         gets 3, basin 3 gets 4, and the remaining 2 occupy basin 4; basin 5 stays dry.
#    - Difference from the source task:
#         the output is no longer determined by local nearest-drop motion between only 2-3 cups.
#         Instead the solver must decode one global direction bit and then track a long, directed
#         capacity chain across 4-5 cups.
#
# b) Complexity calibration
#    - The basin count is always 4 or 5, never fewer, so the long cascade is visually unavoidable.
#    - Basin widths are only 1 or 2 cells. That keeps the chain readable on a 10x10 canvas while
#      still allowing several distinct capacities and long spill sequences.
#    - Divider heights vary in only two or three rows (mostly tops 6 or 7). This is enough to make
#      capacity patterns differ across examples, but not so much that each sample becomes a new rule.
#    - Water amount is sampled from whole-row counts only, so every occupied basin ends with neat
#      bottom-aligned rows rather than noisy partial puddles.
#
# c) Hidden-assumption audit against the original main()
#    - The original `settle()` assumed a symmetric local movement rule: if a water pixel can reach a
#      drop to either side, geometry decides. That breaks under D9/2 because reverse spill must stay
#      forbidden even when a lower reverse lip exists.
#    - The original solver also assumed only 2-3 neighboring basins. D8/1 breaks that by extending
#      the chain to 4-5 basins, so local simulation state is not the easiest or clearest abstraction.
#    - The fused solver therefore switches from per-pixel falling to a basin-capacity interpretation:
#      detect the basin chain, count total water, then allocate it one basin at a time along the
#      scaffold-indicated direction. Reverse allocation is never considered.
#    - New assumptions introduced here:
#         1. bottom-anchored solids define the basin chain and are separable from the upper scaffold;
#         2. the upper scaffold touches exactly one side edge and thereby encodes direction;
#         3. generated water amounts align with full basin rows.
#      `generate_input()` explicitly enforces all three.
#
# d) Placement legality
#    - The lower chain always uses single-column internal dividers and occupies the full width without
#      any overlap. Basin columns and divider columns are constructed from a width composition whose
#      total exactly equals 10 after inserting the dividers.
#    - The upper scaffold sits in rows 0..4 only, while basin water is constrained to rows 6..8 in
#      accepted samples. This keeps the clue layer and the basin layer visually and spatially disjoint.
#    - Source water is painted only in scaffold-supported cells, packed bottom-up and toward the
#      scaffold side, so every input water cell is supported and has at least one non-background
#      horizontal neighbor or an edge.
#
# e) Visual observability
#    - Across examples, humans see the same bottom motif: many small cups in a row, not 2-3 large cups.
#    - The only high-level variable that flips is the side of the upper scaffold. When that side flips,
#      the occupied basin prefix flips too. That makes the direction cue and the long cascade jointly
#      recoverable from 2-4 examples.
#    - Because water counts are large enough to reach at least the fourth basin in accepted samples,
#      the long-chain effect is not subtle or accidental.
#
# f) Visual cleanliness
#    - Only one special cue color is introduced: the scaffold is fixed to color 8 and is absent from
#      the bottom structure palette. Basin walls keep 2-3 ordinary solid colors.
#    - No extra markers are overlaid inside the basins. The lower half contains only cups, walls, and
#      final water, so the one-way cascade remains easy to read.
#
# g) Change inferability / anti-degeneration target
#    - The key non-degeneration metric is `diff_vs_reverse`: compare the generated output with the
#      output obtained from the exact same input under the opposite overflow direction. Accepted
#      samples require a positive and non-trivial difference, proving that the global clue matters.
#    - A second metric is `chain_span`: the farthest occupied basin index measured from the source side.
#      Accepted samples require `chain_span >= 3`, which forces a genuinely long 4-5-basin cascade.
#
# h) New main() assumptions and generator contract
#    - Assumption: row 8 contains exactly the open basin intervals separated by single divider columns.
#      Guaranteed by the width-composition renderer.
#    - Assumption: divider heights are read from bottom-connected solids only; the upper scaffold may
#      pass above divider columns without changing basin geometry. Guaranteed by separating detection
#      into bottom-anchored structure vs upper scaffold.
#    - Assumption: source water count fits inside the scaffold reservoir and basin fill stays below the
#      scaffold. Guaranteed by `SOURCE_CAPACITY` and by sampling whole-row counts with at most 3 rows
#      in the last occupied basin.


BG = 0
WATER = 1
GUIDE = 8
H = 10
W = 10
SHELF_ROW = 4
SOURCE_CAPACITY = 16
MAX_ATTEMPTS = 600
STRUCTURE_PALETTE = (2, 3, 4, 5, 6, 7, 9)
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _anchored_mask(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    solid = (g != int(BG)) & (g != int(WATER))
    anchored = np.zeros_like(solid, dtype=bool)
    stack = [(H - 1, int(c)) for c in range(W) if bool(solid[H - 1, c])]
    while stack:
        r, c = stack.pop()
        if anchored[r, c]:
            continue
        anchored[r, c] = True
        for dr, dc in ORTHO_DIRS:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W and bool(solid[rr, cc]) and not anchored[rr, cc]:
                stack.append((rr, cc))
    return anchored


def _detect_direction(grid: np.ndarray) -> int:
    g = np.array(grid, dtype=int)
    left = any(int(g[r, 0]) not in (int(BG), int(WATER)) for r in range(0, 7))
    right = any(int(g[r, W - 1]) not in (int(BG), int(WATER)) for r in range(0, 7))
    if left == right:
        raise AssertionError("expected exactly one side scaffold")
    return 1 if left else -1


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    anchored = _anchored_mask(g)

    # Basin columns are defined by the absence of bottom-anchored structure on row 8.
    # Output water may already occupy those cells, so water itself must not hide a basin.
    open_on_row8 = [bool(not anchored[8, c]) for c in range(W)]
    basins: list[tuple[int, int]] = []
    c = 0
    while c < W:
        if open_on_row8[c]:
            start = int(c)
            while c + 1 < W and open_on_row8[c + 1]:
                c += 1
            basins.append((int(start), int(c)))
        c += 1

    if not (4 <= len(basins) <= 5):
        raise AssertionError(f"expected 4-5 basins, found {len(basins)}")

    gaps = [int(basins[i][1] + 1) for i in range(len(basins) - 1)]
    partition_tops: list[int] = []
    for gap in gaps:
        if not anchored[8, gap]:
            raise AssertionError("divider column must be bottom-anchored at row 8")
        row = 8
        while row - 1 >= 0 and anchored[row - 1, gap]:
            row -= 1
        partition_tops.append(int(row))

    widths = [int(c1 - c0 + 1) for c0, c1 in basins]
    return {
        "anchored": anchored,
        "basins": basins,
        "partition_tops": partition_tops,
        "widths": widths,
    }


def _ordered_chain(scene: dict[str, object], direction: int) -> tuple[list[tuple[int, int]], list[int]]:
    basins = list(scene["basins"])  # type: ignore[arg-type]
    tops = list(scene["partition_tops"])  # type: ignore[arg-type]
    if direction == 1:
        return basins, tops
    return list(reversed(basins)), list(reversed(tops))


def _source_cells(direction: int) -> list[tuple[int, int]]:
    if direction == 1:
        return [(int(r), int(c)) for r in range(SHELF_ROW - 1, -1, -1) for c in range(1, 5)]
    return [(int(r), int(c)) for r in range(SHELF_ROW - 1, -1, -1) for c in range(8, 4, -1)]


def _fill_lowest_rows(
    out: np.ndarray,
    basin: tuple[int, int],
    amount: int,
    direction: int,
) -> None:
    if amount <= 0:
        return
    c0, c1 = basin
    width = int(c1 - c0 + 1)
    full_rows, partial = divmod(int(amount), int(width))
    for row in range(8, 8 - int(full_rows), -1):
        out[int(row), int(c0) : int(c1) + 1] = int(WATER)
    if partial:
        row = int(8 - full_rows)
        if direction == 1:
            out[row, int(c0) : int(c0) + int(partial)] = int(WATER)
        else:
            out[row, int(c1) - int(partial) + 1 : int(c1) + 1] = int(WATER)


def _solve_directed(grid: np.ndarray, direction: int | None = None) -> tuple[np.ndarray, dict[str, object], int]:
    g = np.array(grid, dtype=int).copy()
    scene = _parse_scene(g)
    flow_dir = _detect_direction(g) if direction is None else int(direction)
    water_count = int(np.count_nonzero(g == int(WATER)))
    g[g == int(WATER)] = int(BG)

    ordered_basins, ordered_tops = _ordered_chain(scene, flow_dir)
    remaining = int(water_count)
    for idx, basin in enumerate(ordered_basins):
        width = int(basin[1] - basin[0] + 1)
        if idx < len(ordered_basins) - 1:
            lip_top = int(ordered_tops[idx])
            capacity = int(width * (9 - lip_top))
        else:
            capacity = int(width * 9)
        take = min(int(remaining), int(capacity))
        _fill_lowest_rows(g, basin, take, flow_dir)
        remaining -= int(take)
        if remaining == 0:
            break

    if remaining != 0:
        raise AssertionError("water exceeds directed basin-chain capacity")
    return g, scene, flow_dir


def _validate_input(grid: np.ndarray) -> None:
    g = np.array(grid, dtype=int)
    assert g.shape == (H, W), "grid must be 10x10"
    assert not np.any(g == int(GUIDE)) or np.any(g[:7] == int(GUIDE)), "guide cue must be visible in upper rows"
    _ = _parse_scene(g)
    direction = _detect_direction(g)
    source_zone = set(_source_cells(direction))
    assert np.count_nonzero(g == int(WATER)) > 0, "input needs source water"
    for r, c in np.argwhere(g == int(WATER)):
        rr = int(r)
        cc = int(c)
        assert (rr, cc) in source_zone, "generated inputs keep all water in the upper source reservoir"
        assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
        left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
        right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(BG)
        assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"


def _validate_output(
    inp: np.ndarray,
    out: np.ndarray,
    scene: dict[str, object],
    direction: int,
) -> None:
    assert int(np.count_nonzero(inp == int(WATER))) == int(np.count_nonzero(out == int(WATER))), "water count changed"

    source_zone = set(_source_cells(direction))
    for r, c in source_zone:
        assert int(out[r, c]) != int(WATER), "output water must leave the source reservoir"
    assert not np.any(out[:6] == int(WATER)), "accepted samples keep final water below the upper scaffold band"

    basin_union = set()
    for c0, c1 in scene["basins"]:  # type: ignore[assignment]
        for r in range(0, 9):
            for c in range(int(c0), int(c1) + 1):
                basin_union.add((int(r), int(c)))
    for r, c in np.argwhere(out == int(WATER)):
        assert (int(r), int(c)) in basin_union, "final water must stay inside basin columns above the floor"

    replay, _scene2, _dir2 = _solve_directed(out, direction)
    assert np.array_equal(replay, out), "output is not a fixed point of the directed fill rule"


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Redistribute water through a long one-way basin cascade driven by a side scaffold cue.

    Input structure:
    - Background is 0 and water is 1.
    - Bottom-anchored solids form a chain of 4 or 5 narrow basins separated by single divider walls.
    - A non-bottom scaffold touches exactly one side edge in the upper half and holds the source water;
      that scaffold side is the global clue for the allowed overflow direction.

    Rule:
    - Count the total number of water cells, clear them from the source reservoir, and identify the
      basin chain from the bottom-anchored divider columns.
    - Starting from the scaffold-side basin, fill each basin from the bottom upward only up to that
      basin's outgoing lip; only the excess may continue to the next basin in the same direction.
    - Reverse spill is forbidden: even if a lower wall exists on the opposite side, water never
      allocates to basins against the scaffold-indicated direction.

    Key insight:
    - This is not local symmetric overflow anymore. The solver must decode one global direction cue
      and then follow a long 4-5-step one-way capacity chain across the basin sequence.
    """
    g = np.array(grid, dtype=int).copy()
    if check_input:
        _validate_input(g)
    out, scene, direction = _solve_directed(g)
    if check_output:
        _validate_output(g, out, scene, direction)
    return out


def _ordered_caps(widths: list[int], tops: list[int], direction: int) -> tuple[list[int], list[int], list[int]]:
    ordered_widths = list(widths) if direction == 1 else list(reversed(widths))
    ordered_tops = list(tops) if direction == 1 else list(reversed(tops))
    caps: list[int] = []
    basin_count = len(ordered_widths)
    for idx, width in enumerate(ordered_widths):
        if idx < basin_count - 1:
            caps.append(int(width * (9 - ordered_tops[idx])))
        else:
            caps.append(int(width * 9))
    return ordered_widths, ordered_tops, caps


def _reachable_water_options(widths: list[int], tops: list[int], direction: int) -> list[tuple[int, int, int]]:
    ordered_widths, ordered_tops, caps = _ordered_caps(widths, tops, direction)
    basin_count = len(ordered_widths)
    options: list[tuple[int, int, int]] = []
    for reach in range(4, basin_count + 1):
        prefix = int(sum(caps[: reach - 1]))
        last_width = int(ordered_widths[reach - 1])
        max_rows = int(9 - ordered_tops[reach - 1]) if reach < basin_count else 3
        for rows in range(1, int(max_rows) + 1):
            count = int(prefix + last_width * rows)
            if 9 <= count <= SOURCE_CAPACITY:
                options.append((int(reach), int(rows), int(count)))
    return options


def _render_scene(
    widths: list[int],
    tops: list[int],
    direction: int,
    water_count: int,
    colors: tuple[int, int, int],
) -> np.ndarray:
    g = np.zeros((H, W), dtype=int)
    c0, c1, c2 = map(int, colors)

    if direction == 1:
        g[: SHELF_ROW + 1, 0] = int(GUIDE)
        g[SHELF_ROW, 0:5] = int(GUIDE)
    else:
        g[: SHELF_ROW + 1, W - 1] = int(GUIDE)
        g[SHELF_ROW, 5:W] = int(GUIDE)

    source_order = _source_cells(direction)
    assert int(water_count) <= len(source_order), "water does not fit in the source reservoir"
    for cell in source_order[: int(water_count)]:
        g[cell] = int(WATER)

    basins: list[tuple[int, int]] = []
    col = 0
    for idx, width in enumerate(widths):
        basins.append((int(col), int(col + width - 1)))
        col += int(width)
        if idx != len(widths) - 1:
            col += 1
    assert col == W, "basin composition must span the full width"

    for x in range(W):
        g[H - 1, x] = (c0, c1, c2)[x % 3]

    for idx in range(len(basins) - 1):
        gap = int(basins[idx][1] + 1)
        g[int(tops[idx]) : H - 1, gap] = (c0, c1, c2)[idx % 3]

    return g


def _occupied_basin_prefix(out: np.ndarray, scene: dict[str, object], direction: int) -> list[int]:
    ordered_basins, _ordered_tops = _ordered_chain(scene, direction)
    occupied: list[int] = []
    for idx, (c0, c1) in enumerate(ordered_basins):
        if np.any(out[:9, int(c0) : int(c1) + 1] == int(WATER)):
            occupied.append(int(idx))
    return occupied


def _variant_metrics(inp: np.ndarray) -> dict[str, int]:
    out, scene, direction = _solve_directed(inp)
    reverse_out, _scene2, _rev_dir = _solve_directed(inp, -int(direction))
    occupied = _occupied_basin_prefix(out, scene, direction)
    chain_span = int(max(occupied)) if occupied else -1
    ordered_basins, _ = _ordered_chain(scene, direction)
    far_cells = 0
    if occupied:
        c0, c1 = ordered_basins[max(occupied)]
        far_cells = int(np.count_nonzero(out[:9, int(c0) : int(c1) + 1] == int(WATER)))
    return {
        "basin_count": int(len(scene["basins"])),  # type: ignore[arg-type]
        "occupied_basins": int(len(occupied)),
        "chain_span": int(chain_span),
        "diff_vs_reverse": int(np.count_nonzero(out != reverse_out)),
        "water_count": int(np.count_nonzero(inp == int(WATER))),
        "far_basin_cells": int(far_cells),
    }


def generate_input() -> np.ndarray:
    rng = random.Random()
    width_templates = {
        4: [2, 2, 2, 1],
        5: [2, 1, 1, 1, 1],
    }

    for _ in range(MAX_ATTEMPTS):
        basin_count = int(rng.choice([4, 5]))
        widths = list(width_templates[basin_count])
        rng.shuffle(widths)
        direction = int(rng.choice([1, -1]))
        tops = [int(rng.choice([6, 7])) for _ in range(basin_count - 1)]

        options = _reachable_water_options(widths, tops, direction)
        if not options:
            continue
        reach, rows_last, water_count = rng.choice(options)
        colors = tuple(rng.sample(STRUCTURE_PALETTE, 3))
        inp = _render_scene(widths, tops, direction, water_count, colors)

        try:
            out = main(inp, check_input=True, check_output=True)
            metrics = _variant_metrics(inp)
        except AssertionError:
            continue

        if np.array_equal(inp, out):
            continue
        if metrics["occupied_basins"] < 4:
            continue
        if metrics["chain_span"] < 3:
            continue
        if metrics["diff_vs_reverse"] < 4:
            continue
        if metrics["far_basin_cells"] <= 0:
            continue
        if metrics["water_count"] != water_count:
            continue
        if reach != metrics["occupied_basins"]:
            continue

        return np.array(inp, dtype=int)

    raise RuntimeError("failed to generate a non-degenerate D8/1+D9/2 sample")
