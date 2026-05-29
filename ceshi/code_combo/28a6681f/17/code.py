# Combo variant of 28a6681f (index 17)
# combo_spec: D9/1+D19/0
# Name: 阶梯递增壁高+连通器原理

from __future__ import annotations

import random

import numpy as np

BASIN_COLS = [(1, 2), (4, 5), (7, 8)]
INTERNAL_WALL_COLS = (3, 6)
WIDTHS = [2, 2, 2]

# Design analysis
# a) Concrete geometry and behavior:
#    - The original task used local gravity, diagonal slips, and row-wise sliding. That works for
#      ordinary overflow but it does not directly encode "compute one shared level from total
#      volume" as the core rule.
#    - The fused variant keeps the visual vocabulary of adjacent bottom basins, but standardizes it
#      into three side-by-side chambers separated by two bottom-anchored dividers whose heights are
#      strictly increasing from left to right.
#    - Water always starts as one supported block inside only the leftmost basin. The output is not
#      the result of pixel-by-pixel falling. Instead, total water is redistributed over the longest
#      left prefix of basins whose separating walls lie below the final water depth.
#    - Small trace:
#      Example widths = [2, 2, 2], divider heights = [1, 3], water count = 12.
#      Original local logic would first overfill basin 1, spill over the height-1 wall, then keep
#      packing reachable low cells near the spill path. The communicating-vessel rule instead asks:
#      if basins 1-2 communicate, what common depth matches 12 cells? Answer: depth 3 across width
#      4, while the height-3 divider to basin 3 is not yet submerged. So output is a flat 3-row
#      level across only the first two basins.
# b) Complexity calibration:
#    - Three basins are enough to show the staircase gating effect without making parsing brittle.
#    - Basin widths are fixed and repeated across samples, while divider heights and total water
#      vary over a small discrete set. That keeps the latent rule inferable from 2-4 examples.
#    - Samples are restricted to exact whole-row levels so the shared waterline is visually crisp
#      and no noisy partial top row needs extra conventions.
# c) Original-code assumptions that break under the combo:
#    - The old `settle()` routine assumes every move is local and monotone downward/sideways. Here
#      the decisive computation is global: total volume divided by the width of the currently
#      connected basin prefix.
#    - The old solver treats overflow as an emergent side effect of local motion. In this combo,
#      divider height is a threshold variable that explicitly decides connectivity.
#    - The old output invariant demanded each water cell be wedged against non-background on left,
#      right, and below. Flat communicating levels across open basin interiors violate that, so the
#      output check must instead verify exact occupancy of the predicted basin rows.
# d) Placement legality:
#    - The generator uses a fixed 10x10 scaffold: outer walls at columns 0 and 9, internal walls
#      at columns 3 and 6, and a solid floor on row 9.
#    - Divider heights are sampled from a small hand-picked set of strictly increasing pairs.
#    - Water counts are sampled only from cases whose final depth is compatible with those divider
#      heights and whose entire source block fits inside basin 1 without unsupported pixels.
# e) Visual observability:
#    - Input always shows water piled only in the left basin.
#    - Some outputs equalize across basins 1-2 while basin 3 stays empty; other outputs equalize
#      across all three basins. Across examples, the only changing cue that explains this is the
#      relation between total water depth and the staircase divider heights.
#    - This makes the fusion visible: the lower wall opens communication first, while the higher
#      wall delays access to the rightmost basin.
# f) Visual cleanliness:
#    - Only water plus three solid colors are used.
#    - No guides, labels, or overlapping markers are added. The grid stays readable as "three cups
#      separated by taller and taller walls".
# g) Color inference:
#    - Colors do not encode the rule. They only distinguish the floor/outer frame from the two
#      internal dividers so the height ordering is easy to read.
# New solver assumptions audited against the generator:
#    - Every sample contains exactly three basin gaps and exactly two internal dividers.
#    - Internal divider columns are contiguous bottom-anchored solids and their heights are strictly
#      increasing left-to-right.
#    - Total water count always corresponds to exactly one integer shared depth over a left prefix
#      of basins. `generate_input()` constructs only those counts.
#    - Water may begin in any supported arrangement, but the generator uses a rectangular left
#      source block because it is simple, legal, and visually lopsided.


def _bottom_anchored_height(solid: np.ndarray, c: int) -> int:
    H = solid.shape[0]
    h = 0
    r = H - 2
    while r >= 0 and bool(solid[r, c]):
        h += 1
        r -= 1
    if r >= 0 and bool(np.any(solid[: r + 1, c])):
        raise AssertionError("wall columns must be contiguous from the floor upward")
    return int(h)


def _parse_structure(
    g: np.ndarray, bg: int, water: int, check: bool
) -> tuple[list[int], list[tuple[int, ...]], list[int], list[int], int]:
    H, W = g.shape
    solid = np.logical_and(g != int(bg), g != int(water))
    if check:
        assert H == 10 and W == 10, "this combo generator uses a fixed 10x10 scaffold"
        assert bool(np.all(solid[H - 1])), "bottom row must be a continuous solid floor"

    wall_cols = [int(c) for c in range(W) if bool(np.any(solid[: H - 1, c]))]
    if check:
        assert len(wall_cols) == 4, "expected outer walls plus two internal divider columns"
        assert wall_cols == sorted(wall_cols), "wall columns must be ordered"

    basin_cols: list[tuple[int, ...]] = []
    for left, right in zip(wall_cols, wall_cols[1:]):
        cols = tuple(int(c) for c in range(left + 1, right))
        if check:
            assert cols, "every basin gap must have positive width"
            assert not bool(np.any(solid[: H - 1, left + 1 : right])), "basin interiors must be open"
        basin_cols.append(cols)

    wall_heights = [_bottom_anchored_height(solid, c) for c in wall_cols]
    thresholds = [int(h) for h in wall_heights[1:-1]]
    widths = [len(cols) for cols in basin_cols]
    max_level = int(min(wall_heights[0], wall_heights[-1]))

    if check:
        assert len(widths) == 3, "expected three basins"
        assert all(w > 0 for w in widths), "basins must be non-empty"
        assert thresholds[0] < thresholds[1], "internal divider heights must increase strictly"
        assert all(h > 0 for h in thresholds), "internal dividers must rise above the floor"
        assert max_level > 0, "outer walls must contain at least one water row"

    return wall_cols, basin_cols, thresholds, widths, max_level


def _find_shared_profile(
    widths: list[int], thresholds: list[int], water_count: int, max_level: int, check: bool
) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    basin_count = len(widths)
    for prefix in range(1, basin_count + 1):
        active_width = sum(widths[:prefix])
        if water_count % active_width != 0:
            continue
        level = water_count // active_width
        if level <= 0 or level > max_level:
            continue
        if not all(th < level for th in thresholds[: prefix - 1]):
            continue
        if not all(th >= level for th in thresholds[prefix - 1 :]):
            continue
        candidates.append((int(prefix), int(level)))

    if check:
        assert len(candidates) == 1, "water count must imply exactly one communicating prefix"
    if not candidates:
        raise AssertionError("could not derive a valid shared water profile")
    return candidates[0]


def _target_cells(
    H: int, basin_cols: list[tuple[int, ...]], prefix: int, level: int
) -> set[tuple[int, int]]:
    filled: set[tuple[int, int]] = set()
    top_row = H - 1 - level
    for cols in basin_cols[:prefix]:
        for r in range(top_row, H - 1):
            for c in cols:
                filled.add((int(r), int(c)))
    return filled


def _paint_profile(g: np.ndarray, basin_cols: list[tuple[int, ...]], prefix: int, level: int) -> np.ndarray:
    out = np.array(g, dtype=int).copy()
    out[out == 1] = 0
    for r, c in _target_cells(out.shape[0], basin_cols, prefix, level):
        out[r, c] = 1
    return out


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Refill stair-stepped communicating basins to one shared level over the reachable prefix.

    Input structure:
    - Background is 0 and water is 1.
    - Other colors form a continuous floor, two outer side walls, and two internal divider columns.
    - The internal divider heights increase strictly from left to right, so the three neighboring
      basins have progressively harder overflow thresholds.

    Rule:
    - Count the total amount of water without simulating local falling paths.
    - A divider lets water communicate across it only if the final water depth is strictly higher
      than that divider's height.
    - Find the longest left prefix of basins opened by that rule, then refill exactly those basins
      from the bottom upward to one common integer depth.

    Key insight: The answer is determined by divider-height thresholds plus total water volume over
    basin widths, not by replaying the original gravity-and-sliding microphysics.
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

    _wall_cols, basin_cols, thresholds, widths, max_level = _parse_structure(g, bg, water, check_input)
    prefix, level = _find_shared_profile(widths, thresholds, initial_water, max_level, check_input)
    out = _paint_profile(g, basin_cols, prefix, level)

    if check_output:
        assert int(np.count_nonzero(out == int(water))) == initial_water, "water count must be conserved"
        actual = {(int(r), int(c)) for r, c in np.argwhere(out == int(water))}
        expected = _target_cells(H, basin_cols, prefix, level)
        assert actual == expected, "output water must match the predicted shared level exactly"
        for r, c in actual:
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(out[rr + 1, cc]) != int(bg), "output water must be supported"

    return out


def _draw_base(top: int, thresholds: tuple[int, int], colors: tuple[int, int, int]) -> np.ndarray:
    g = np.zeros((10, 10), dtype=int)
    outer_color, wall1_color, wall2_color = colors

    g[9, :] = outer_color
    g[top:, 0] = outer_color
    g[top:, 9] = outer_color
    g[9 - thresholds[0] : 10, INTERNAL_WALL_COLS[0]] = wall1_color
    g[9 - thresholds[1] : 10, INTERNAL_WALL_COLS[1]] = wall2_color
    return g


def _make_output(base: np.ndarray, level: int) -> np.ndarray:
    solid = np.logical_and(base != 0, base != 1)
    thresholds = [_bottom_anchored_height(solid, c) for c in INTERNAL_WALL_COLS]
    prefix = 1 + sum(int(th < level) for th in thresholds)
    return _paint_profile(base, BASIN_COLS, prefix, level)


def _make_input(base: np.ndarray, top: int, water_count: int) -> np.ndarray:
    inp = np.array(base, dtype=int).copy()
    remaining = int(water_count)
    for r in range(8, top - 1, -1):
        for c in BASIN_COLS[0]:
            if remaining == 0:
                break
            inp[r, c] = 1
            remaining -= 1
        if remaining == 0:
            break
    if remaining != 0:
        raise AssertionError("left source basin was too small for the sampled water volume")
    return inp


def _candidate_specs() -> list[tuple[int, tuple[int, int], int]]:
    specs: list[tuple[int, tuple[int, int], int]] = []
    for top in (0, 1):
        source_capacity = WIDTHS[0] * (9 - top)
        for thresholds in ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4)):
            for level in (2, 3, 4):
                prefix = 1 + sum(int(th < level) for th in thresholds)
                if prefix < 2:
                    continue
                water_count = level * sum(WIDTHS[:prefix])
                if water_count > source_capacity:
                    continue
                specs.append((int(top), thresholds, int(level)))
    return specs


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    all_specs = _candidate_specs()
    three_basin_specs = [spec for spec in all_specs if spec[1] == (1, 2) and spec[2] == 3 and spec[0] == 0]
    two_basin_specs = [spec for spec in all_specs if spec not in three_basin_specs]

    if random.random() < 0.35:
        top, thresholds, level = random.choice(three_basin_specs)
    else:
        top, thresholds, level = random.choice(two_basin_specs)

    colors = tuple(random.sample([2, 3, 4, 5, 6, 7, 8, 9], 3))
    base = _draw_base(top, thresholds, colors)
    prefix = 1 + sum(int(th < level) for th in thresholds)
    water_count = level * sum(WIDTHS[:prefix])
    inp = _make_input(base, top, water_count)
    out = _make_output(base, level)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(100):
        inp, out = _build_case()
        solved = main(inp)
        if np.array_equal(solved, out):
            return inp
    raise AssertionError("failed to generate a valid stair-communicating sample")
