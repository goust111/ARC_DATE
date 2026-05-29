# Combo variant of 28a6681f (index 48)
# combo_spec: D4/0+D19/2
# Name: 多色水源+预填水为不同颜色

from __future__ import annotations

import random

import numpy as np


# Design analysis for combo D4/0 + D19/2
# ---------------------------------------
# a) Concretization + behavior trace:
#    - The grid stays 10x10 and keeps the original basin family: bottom-anchored walls and
#      a pair of narrow lower shafts that can catch falling liquid. The fused palette is
#      deliberately fixed and repeated across samples:
#        * fresh moving water: blue (1) and teal (8),
#        * old already-stable water: yellow (6),
#        * all other non-zero colors: rigid solids.
#    - Both fresh colors start in separate supported edge reservoirs, while yellow old water
#      already occupies bottom-contiguous cells inside the two lower shafts.
#    - Concrete trace: suppose the left shaft already contains two yellow cells at rows 8-7
#      and the left source contains two blue cells. In the original task, only blue would be
#      liquid, teal would be treated as a wall, and no old-water distinction would exist.
#      In this combo, blue is simulated on a copy where teal is erased to background but the
#      yellow cells stay occupied. Blue therefore settles into rows 6-5 above yellow. Teal
#      is then simulated independently in the mirrored right shaft and also stacks above the
#      yellow support there. The difference is twofold at once: more than one color moves,
#      yet the prefilled yellow water does not.
#
# b) Complexity calibration:
#    - The color freedom is intentionally narrow. Using exactly two fresh colors plus one old
#      color is enough to make the fusion visible without turning the task into arbitrary
#      color remapping.
#    - The geometry is also narrow: both fresh sources feed into two one-cell-wide shafts.
#      This keeps the old-water effect legible because every sample visibly shows fresh water
#      landing above prefilled old water instead of in a large noisy basin.
#    - Template choice, source-color swap, and wall recoloring still give enough variation
#      across examples for ARC-style induction without making the liquid bookkeeping noisy.
#
# c) Hidden-assumption audit of the original main():
#    - `water = 1` is no longer valid. The solver must iterate over both fresh colors.
#    - "Every water-colored cell moves" is false now: the old-water color must remain fixed.
#    - "Any non-background non-water cell is a solid" is also false now: old water is not a
#      wall semantically, but it must behave as an occupied support cell for fresh water.
#    - Reassigning settled cells directly into a single shared grid is unsafe for multi-fresh
#      simulation, because one fresh color would block the other. Each fresh color therefore
#      settles on its own copy with the other fresh color erased to background.
#    - The original output checker only cared about one moving color. The fused checker must
#      separately verify fresh-water stability and unchanged old-water placement.
#
# d) Placement legality:
#    - Every template is hand-checked 10x10 geometry copied from the valid D4 single variant.
#    - The old-water stacks are inserted only into known background shaft cells `(r, 3)` and
#      `(r, 6)` for rows 8..5, so they never overlap walls or sources and always stay in
#      bounds.
#    - Source modules occupy disjoint edge shelves and are supported from below exactly as in
#      the base task, so the original movement assertions still apply.
#
# e) Visual observability:
#    - Both fresh colors are always present and both move. This makes the D4 part obvious.
#    - Yellow old water appears in multiple repeated basin cells rather than as a single tag.
#      Since it stays fixed from input to output while blue/teal move around it, the D19/2
#      distinction is recoverable from just a few examples.
#    - Samples are rejected unless removing old water changes the output, so the old-water
#      color is not decorative.
#
# f) Visual cleanliness:
#    - The scene has three clean semantic regions: upper source shelves, lower walls, and
#      narrow liquid shafts. No legends or overlapping annotations are added.
#    - Each shaft is only one column wide, so the final layering "yellow old water below,
#      blue/teal fresh water above" is visually crisp rather than blended.
#
# g) Change inferability / color discipline:
#    - No hidden ordering on color IDs is used. The same three liquid-role colors repeat in
#      every sample with consistent meaning.
#    - The key inferable rule is structural: blue and teal both behave like the old blue did,
#      but yellow water is already settled and does not move.
#
# h) Quantitative degeneration targets:
#    - Reject any candidate where either fresh color fails to move at least one cell.
#    - Reject any candidate where erasing old water yields the same output, because then the
#      D19/2 half would be inert.
#    - Later validation measures both fresh-color movement and old-water influence directly.
#
# 9) New solver assumptions guaranteed by generate_input():
#    - Only colors 1 and 8 are fresh moving water; color 6 is old stable water.
#    - Old-water cells form bottom-contiguous stacks in the two shaft columns and are already
#      stable before simulation starts.
#    - The two fresh colors settle into disjoint left/right shafts, so their independent
#      simulations never claim the same final cell.
#    - Source reservoirs are the standard supported edge modules with a unique outlet.


BG = 0
FRESH_WATER_COLORS = (1, 8)
OLD_WATER = 6
SOLID_COLORS = (2, 3, 4, 5, 7, 9)
GRID_SHAPE = (10, 10)
MAX_GENERATION_ATTEMPTS = 400

LEFT_SHAFT_COL = 3
RIGHT_SHAFT_COL = 6
SHAFT_ROWS_BOTTOM_UP = (8, 7, 6, 5)


BASE_TEMPLATES: tuple[np.ndarray, ...] = (
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
            [0, 2, 1, 1, 1, 1, 1, 1, 2, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
            [0, 2, 1, 1, 1, 1, 1, 1, 2, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
            [0, 2, 1, 1, 1, 1, 1, 1, 2, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
            [0, 2, 1, 1, 1, 1, 1, 1, 2, 0],
        ],
        dtype=int,
    ),
)


def _component_count(points: set[tuple[int, int]], use_diagonal: bool) -> int:
    remaining = set(points)
    if use_diagonal:
        neighbors = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                nxt = (int(r + dr), int(c + dc))
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
    return count


def _assert_old_water_stable(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    for r, c in np.argwhere(g == int(OLD_WATER)):
        rr = int(r)
        cc = int(c)
        assert cc == 0 or int(g[rr, cc - 1]) != BG, "old water must be left-supported"
        assert rr == h - 1 or int(g[rr + 1, cc]) != BG, "old water must be floor-supported"
        assert cc == w - 1 or int(g[rr, cc + 1]) != BG, "old water must be right-supported"


def _simulate_single_color(
    grid: np.ndarray,
    water_color: int,
    check_input: bool,
    check_output: bool,
) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape

    for other in FRESH_WATER_COLORS:
        if other != water_color:
            g[g == int(other)] = BG

    if check_input:
        for r, c in np.argwhere(g == int(water_color)):
            rr = int(r)
            cc = int(c)
            assert rr == h - 1 or int(g[rr + 1, cc]) != BG, "fresh input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == BG
            right_bg = cc + 1 < w and int(g[rr, cc + 1]) == BG
            assert not (left_bg and right_bg), "fresh input water cannot have bg on both horizontal sides"

    bottom_colors = set(map(int, g[h - 1]))
    upper_colors = sorted(set(map(int, g[:7].flatten())) - {BG, water_color, OLD_WATER} - bottom_colors)
    for color in upper_colors:
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, w - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(h * w):
            if rr + 1 < h and int(g[rr + 1, cc]) == BG:
                rr += 1
                continue

            options: list[tuple[int, int]] = []
            if rr + 1 < h and cc - 1 >= 0 and int(g[rr, cc - 1]) == BG and int(g[rr + 1, cc - 1]) == BG:
                options.append((rr + 1, cc - 1))
            if rr + 1 < h and cc + 1 < w and int(g[rr, cc + 1]) == BG and int(g[rr + 1, cc + 1]) == BG:
                options.append((rr + 1, cc + 1))
            if options:
                assert len(options) == 1, f"ambiguous diagonal moves: {options}"
                rr, cc = options[0]
                continue

            if rr + 1 < h and int(g[rr + 1, cc]) != BG:
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == BG:
                    left -= 1
                right = int(cc)
                while right + 1 < w and int(g[rr, right + 1]) == BG:
                    right += 1

                drops = [int(x) for x in range(left, right + 1) if int(g[rr + 1, int(x)]) == BG]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                target = int(nearest[0])
                assert target != int(cc)
                step = -1 if target < cc else 1
                nx = int(cc + step)
                assert int(g[rr, nx]) == BG
                cc = nx
                continue

            break
        else:
            raise AssertionError("fresh water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(h * w):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(water_color))], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(water_color):
                continue
            g[r, c] = BG
            rr, cc = settle(r, c)
            moved = bool(moved or ((rr, cc) != (r, c)))
            g[rr, cc] = int(water_color)
        if not moved:
            break
    else:
        raise AssertionError("fresh water simulation did not converge")

    if check_output:
        for r, c in np.argwhere(g == int(water_color)):
            rr = int(r)
            cc = int(c)
            assert cc == 0 or int(g[rr, cc - 1]) != BG, "fresh output water must be left-supported"
            assert rr == h - 1 or int(g[rr + 1, cc]) != BG, "fresh output water must be floor-supported"
            assert cc == w - 1 or int(g[rr, cc + 1]) != BG, "fresh output water must be right-supported"

    return g


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Settle multiple fresh water colors into basins that may already contain distinct old water.

    Input structure:
    - Background is 0 on a 10x10 grid.
    - Fresh moving water can be blue (1) or teal (8); old already-stable water is yellow (6).
    - All remaining non-zero colors are static guides, walls, shelves, or basin floors.
    - The old water already occupies stable shaft cells, while the fresh colors begin in supported edge reservoirs.

    Rule:
    - Keep the old-water cells fixed for the whole transformation and treat them as occupied support cells.
    - For each fresh color separately, erase the other fresh color to background and run the original local physics:
      fall straight down, use the unique open diagonal when blocked, and slide horizontally toward the nearest drop.
    - After every fresh color settles independently, overlay those settled masks back onto the unchanged structure plus old water.

    Key insight: The solver has to separate two semantic axes at once: multiple colors can be active moving liquid,
    but a third liquid-like color is not active at all and only marks prefilled stable water that supports the new flow.
    """
    g = np.array(grid, dtype=int)
    assert g.shape == GRID_SHAPE, f"expected a {GRID_SHAPE[0]}x{GRID_SHAPE[1]} grid"

    if check_input:
        _assert_old_water_stable(g)

    base = np.where(np.isin(g, FRESH_WATER_COLORS), BG, g)
    result = base.copy()
    occupied = np.zeros_like(g, dtype=bool)

    present_fresh = [color for color in FRESH_WATER_COLORS if np.any(g == int(color))]
    assert present_fresh, "input must contain at least one fresh-water color"

    for water_color in present_fresh:
        settled = _simulate_single_color(g, water_color, check_input, check_output)
        mask = settled == int(water_color)
        assert not np.any((base != BG) & mask), "fresh water may not overlap solids or old water"
        assert not np.any(occupied & mask), "independent fresh-water colors cannot share a final cell"
        occupied |= mask
        result[mask] = int(water_color)

    if check_output:
        _assert_old_water_stable(result)
        assert np.array_equal(result == int(OLD_WATER), g == int(OLD_WATER)), "old water must stay fixed"

    return result


def _colorize_template(template: np.ndarray, rng: random.Random) -> np.ndarray:
    solids = list(SOLID_COLORS)
    rng.shuffle(solids)
    core_color = solids[0]
    foot_color = solids[1]
    accent_color = solids[2]

    g = np.zeros_like(template, dtype=int)
    g[template == 1] = int(core_color)
    g[template == 2] = int(foot_color)

    center_mask = np.zeros_like(template, dtype=bool)
    center_mask[5:9, 4:6] = template[5:9, 4:6] == 1
    g[center_mask] = int(accent_color)
    return g


def _module_colors(structure: np.ndarray, rng: random.Random) -> tuple[int, int]:
    used = set(map(int, structure.flatten())) | set(FRESH_WATER_COLORS) | {OLD_WATER, BG}
    candidates = [color for color in SOLID_COLORS if color not in used]
    if len(candidates) < 2:
        candidates = [color for color in SOLID_COLORS if color not in set(FRESH_WATER_COLORS) | {OLD_WATER}]
        rng.shuffle(candidates)
    return int(candidates[0]), int(candidates[1])


def _add_left_source(g: np.ndarray, water_color: int, solid_color: int) -> None:
    g[0, 2] = int(solid_color)
    g[1, 0] = int(solid_color)
    g[2, 0:2] = int(solid_color)
    g[0, 0:2] = int(water_color)
    g[1, 1] = int(water_color)


def _add_right_source(g: np.ndarray, water_color: int, solid_color: int) -> None:
    g[0, 7] = int(solid_color)
    g[1, 9] = int(solid_color)
    g[2, 8:10] = int(solid_color)
    g[0, 8:10] = int(water_color)
    g[1, 8] = int(water_color)


def _add_old_stack(g: np.ndarray, col: int, count: int) -> None:
    assert 1 <= int(count) <= 2, f"unsupported old-water stack height {count}"
    for row in SHAFT_ROWS_BOTTOM_UP[:count]:
        assert int(g[row, col]) == BG, "old water must be placed into a free shaft cell"
        g[row, col] = int(OLD_WATER)


def _moved_cells(inp: np.ndarray, out: np.ndarray, color: int) -> int:
    return int(np.count_nonzero((inp == int(color)) ^ (out == int(color))) // 2)


def _fresh_above_old_count(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    total = 0
    for color in FRESH_WATER_COLORS:
        for r, c in np.argwhere(g == int(color)):
            rr = int(r)
            cc = int(c)
            if rr + 1 < g.shape[0] and int(g[rr + 1, cc]) == int(OLD_WATER):
                total += 1
    return int(total)


def _erase_old_water(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    g[g == int(OLD_WATER)] = BG
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(MAX_GENERATION_ATTEMPTS):
        template = np.array(rng.choice(BASE_TEMPLATES), dtype=int)
        g = _colorize_template(template, rng)
        left_support, right_support = _module_colors(g, rng)

        left_color, right_color = FRESH_WATER_COLORS
        if rng.random() < 0.5:
            left_color, right_color = right_color, left_color

        _add_old_stack(g, LEFT_SHAFT_COL, 1)
        _add_old_stack(g, RIGHT_SHAFT_COL, 1)
        _add_left_source(g, left_color, left_support)
        _add_right_source(g, right_color, right_support)

        try:
            out = main(g)
            out_without_old = main(_erase_old_water(g))
        except AssertionError:
            continue

        moved = {color: _moved_cells(g, out, color) for color in FRESH_WATER_COLORS}
        if min(moved.values()) < 2:
            continue
        if int(np.count_nonzero(g == int(OLD_WATER))) < 2:
            continue
        if _fresh_above_old_count(out) < 2:
            continue
        if np.array_equal(out, out_without_old):
            continue

        return g

    raise AssertionError("failed to generate a valid multicolor-old-water input")
