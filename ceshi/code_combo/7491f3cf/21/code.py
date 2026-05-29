# Combo variant of 7491f3cf (index 21)
# combo_spec: D11/1+D12/0
# Name: cue与面板同色+双色编码
#
# Design analysis:
# a) Concrete fused behavior and trace
#    - The 7x25 layout stays the same: four 5x5 tiles in one row, tile3 blank in
#      the input and filled in the output.
#    - The output composition rule also stays the same as the base task:
#      flood-fill from the indicator, blocked by the wall; indicator-side cells
#      take tile1, the opposite side takes tile2, and wall cells keep the
#      original tile1-first / tile2-fallback-on-4-connected-wall behavior.
#    - The fusion changes how tile0 must be read:
#      1. D12/0: wall and indicator no longer share one cue color. The wall is
#         one color, the indicator is a different color.
#      2. cue与面板同色: the wall color is not an arbitrary cue color anymore. It
#         must equal tile1's foreground color or tile2's foreground color.
#    - Small trace:
#      suppose tile1 uses red, tile2 uses green, and the wall itself is also red.
#      The indicator is blue at (0,2), and the wall is the horizontal 5-pixel
#      line through row 2. A human must understand:
#        * the continuous 5-cell red line in tile0 is still the wall, even
#          though red also appears as tile1's source color in tile1;
#        * the lone blue cell is the indicator because it is the singleton cue
#          color, not because blue matches any source tile;
#        * after the flood-fill, the top half still takes tile1 and the bottom
#          half still takes tile2. The shared wall/source color is a decoding
#          ambiguity in the input, not a source-selection rule.
#
# b) Complexity calibration
#    - The fused cue uses exactly two cue colors, not more: one 5-cell wall
#      color and one singleton indicator color. This keeps the visual rule small.
#    - The only extra branch is binary and visible: the wall color matches either
#      tile1 or tile2. The generator alternates these two branches across calls
#      so a short sample set does not collapse into a single case.
#    - The indicator color is kept distinct from both source colors. That avoids
#      accidentally introducing a second color-matching rule that the spec never
#      asked for.
#
# c) Implicit-assumption audit
#    - Original assumption: cue_color is separate from source colors. Broken.
#      The new solver must explicitly allow the 5-cell wall color to equal one
#      of the source foreground colors.
#    - Original assumption: "all non-bg cue pixels can just be unioned, then the
#      unique 8-isolated one is the indicator" still works mechanically, but it
#      no longer captures the intended fused semantics. The new solver therefore
#      parses tile0 by color roles: exactly one singleton indicator color and
#      exactly one 5-cell wall color.
#    - Original assumption: the cue tile has only one active color. Broken by
#      D12/0; the new solver asserts two cue colors with counts 5 and 1.
#    - New assumption introduced here: the 5-cell cue color must match exactly
#      one of {tile1_fg, tile2_fg}, while the singleton indicator color must
#      match neither. The generator enforces both.
#
# d) Placement legality
#    - Wall shapes come from the original prevalidated pool, so they remain
#      inside bounds, pass through (2,2), keep the indicator 8-isolated, and
#      partition the 5x5 tile under 4-connected flood fill.
#    - Reusing tile1/tile2's foreground color for the wall changes only color,
#      not coordinates, so it cannot create overlap or out-of-bounds failures.
#
# e) Visual observability
#    - A human can compare tile0's wall color against tiles1-2 and directly see
#      that the wall reuses one source color.
#    - The singleton indicator still has a dedicated different color, so the cue
#      remains parseable from just 2-4 examples without any hidden numeric order.
#    - Across generated samples, the reused wall color sometimes matches tile1
#      and sometimes tile2, which makes the fusion pattern visible rather than
#      accidental.
#
# f) Visual cleanliness
#    - Tile0 contains only 6 non-background cells total: one contiguous 5-cell
#      wall and one detached indicator. Even with color reuse, the scene stays
#      sparse and readable.
#    - Border color is sampled separately from all active tile colors, so the
#      frame never interferes with the cue/source comparison.
#
# g) Change inferability and validation target
#    - This combo changes cue encoding and cue/source color coupling, not the
#      region-to-source composite rule. Therefore the right non-degeneracy check
#      is not "output must differ from the base task", but "generated inputs must
#      actually realize the fused cue condition".
#    - The validation metric used after implementation is:
#        wall_match_count = how often wall_color == tile1_fg vs tile2_fg
#        indicator_distinct_rate = fraction with indicator_color not in {t1_fg,t2_fg}
#        source_visible_rate = fraction where both tile1 and tile2 contribute at
#        least one visible pixel to tile3.
#      A sample is only accepted if all three properties hold.
#
# h) New solver assumptions checked against generator
#    - tile1/tile2 share one background and each has exactly one foreground color.
#    - tile0 has exactly two cue colors with counts 5 and 1.
#    - the 5-cell cue color equals exactly one source foreground color.
#    - the singleton cue color differs from both source foreground colors.
#    - both the inside region and the outside region receive at least one visible
#      source pixel in the composite, so the output does not collapse to one side.

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path("/home/lixinyu/snap/new")
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from arc_utils import flood_fill, label_mask, neighbors8


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
GEN_RETRIES = 300
LBEND_PROB = 0.2

_NEXT_WALL_MATCHES_T1 = False

VALID_INDICATOR_POS = {
    (0, 0),
    (0, 2),
    (0, 4),
    (2, 0),
    (2, 4),
    (4, 0),
    (4, 2),
    (4, 4),
}

_ORBITS = [
    [(0, 0), (4, 4)],
    [(0, 4), (4, 0)],
    [(2, 2)],
    [(1, 1), (3, 3)],
    [(1, 3), (3, 1)],
    [(0, 1), (1, 0), (3, 4), (4, 3)],
    [(0, 2), (2, 0), (2, 4), (4, 2)],
    [(0, 3), (3, 0), (1, 4), (4, 1)],
    [(1, 2), (2, 1), (2, 3), (3, 2)],
]

_STRAIGHT_CONFIGS = [
    ((0, 2), frozenset({(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)})),
    ((4, 2), frozenset({(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)})),
    ((2, 0), frozenset({(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)})),
    ((2, 4), frozenset({(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)})),
    ((0, 0), frozenset({(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)})),
    ((4, 4), frozenset({(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)})),
    ((0, 4), frozenset({(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)})),
    ((4, 0), frozenset({(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)})),
]

_LBEND_CONFIGS = [
    ((0, 0), frozenset({(2, 2), (1, 2), (0, 2), (2, 1), (2, 0)})),
    ((0, 0), frozenset({(2, 2), (3, 2), (4, 2), (2, 3), (2, 4)})),
    ((4, 4), frozenset({(2, 2), (1, 2), (0, 2), (2, 1), (2, 0)})),
    ((4, 4), frozenset({(2, 2), (3, 2), (4, 2), (2, 3), (2, 4)})),
    ((0, 4), frozenset({(2, 2), (1, 2), (0, 2), (2, 3), (2, 4)})),
    ((0, 4), frozenset({(2, 2), (3, 2), (4, 2), (2, 1), (2, 0)})),
    ((4, 0), frozenset({(2, 2), (1, 2), (0, 2), (2, 3), (2, 4)})),
    ((4, 0), frozenset({(2, 2), (3, 2), (4, 2), (2, 1), (2, 0)})),
    ((4, 2), frozenset({(2, 2), (1, 1), (0, 0), (1, 3), (0, 4)})),
    ((0, 2), frozenset({(2, 2), (3, 1), (4, 0), (3, 3), (4, 4)})),
    ((2, 4), frozenset({(2, 2), (1, 1), (0, 0), (3, 1), (4, 0)})),
    ((2, 0), frozenset({(2, 2), (1, 3), (0, 4), (3, 3), (4, 4)})),
]


def _majority_color(tile: np.ndarray) -> int:
    return int(Counter(map(int, tile.flatten())).most_common(1)[0][0])


def _single_fg_color(tile: np.ndarray, bg: int, *, name: str) -> int:
    fg = {int(v) for v in tile.flatten() if int(v) != int(bg)}
    assert len(fg) == 1, f"{name} must use exactly one foreground color"
    return next(iter(fg))


def _sample_symmetric_pattern(
    bg: int,
    fg: int,
    rng: random.Random,
    *,
    min_px: int,
    max_px: int,
) -> np.ndarray | None:
    for _ in range(PATTERN_RETRIES):
        order = list(range(len(_ORBITS)))
        rng.shuffle(order)
        tile = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
        count = 0
        for idx in order:
            orbit = _ORBITS[idx]
            if count + len(orbit) > max_px:
                continue
            if rng.random() < ORBIT_INCLUDE_PROB:
                for r, c in orbit:
                    tile[r, c] = fg
                count += len(orbit)
        if min_px <= count <= max_px:
            return tile
    return None


def _reflection_from_indicator(indicator: tuple[int, int]):
    ir, ic = indicator
    dr, dc = 2 - ir, 2 - ic
    if dr == 0:
        return lambda r, c: (4 - r, c)
    if dc == 0:
        return lambda r, c: (r, 4 - c)
    if dr * dc > 0:
        return lambda r, c: (c, r)
    return lambda r, c: (4 - c, 4 - r)


def _wall_is_4_connected(wall: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    return bool(n_comp == 1)


def _inside_from_wall(
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
    indicator: tuple[int, int],
) -> set[tuple[int, int]]:
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    return {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}


def _compose_output_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    inside: set[tuple[int, int]],
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
    wall_4conn: bool,
) -> np.ndarray:
    out3 = np.full((TILE_SZ, TILE_SZ), int(bg), dtype=int)
    wall = set(wall)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1, v2 = int(t1[r, c]), int(t2[r, c])
            if (r, c) in inside:
                if v1 != int(bg):
                    out3[r, c] = v1
            elif (r, c) in wall:
                if v1 != int(bg):
                    out3[r, c] = v1
                elif wall_4conn and v2 != int(bg):
                    out3[r, c] = v2
            else:
                if v2 != int(bg):
                    out3[r, c] = v2
    return out3


def _parse_tile0(
    t0: np.ndarray,
    *,
    t1_fg: int,
    t2_fg: int,
    shared_bg: int,
) -> tuple[tuple[int, int], int, set[tuple[int, int]], int]:
    t0_bg = _majority_color(t0)
    assert t0_bg == shared_bg, "tile0 must use the shared tile background"

    coords_by_color: dict[int, set[tuple[int, int]]] = {}
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            color = int(t0[r, c])
            if color == t0_bg:
                continue
            coords_by_color.setdefault(color, set()).add((int(r), int(c)))

    assert len(coords_by_color) == 2, "tile0 must contain exactly two cue colors"

    singleton_groups = [(color, coords) for color, coords in coords_by_color.items() if len(coords) == 1]
    wall_groups = [(color, coords) for color, coords in coords_by_color.items() if len(coords) == 5]
    assert len(singleton_groups) == 1, "indicator color must occupy exactly one cell"
    assert len(wall_groups) == 1, "wall color must occupy exactly five cells"

    indicator_color, indicator_coords = singleton_groups[0]
    wall_color, wall_coords = wall_groups[0]
    indicator = next(iter(indicator_coords))
    wall = set(wall_coords)
    cue_pixels = wall | {indicator}

    assert indicator_color != wall_color, "indicator and wall must use different colors"
    assert wall_color in {t1_fg, t2_fg}, "wall color must match tile1 or tile2 foreground"
    assert indicator_color not in {t1_fg, t2_fg}, "indicator color must stay distinct from source foregrounds"
    assert indicator in VALID_INDICATOR_POS, f"indicator {indicator} not at corner/edge-midpoint"
    assert not any((nr, nc) in cue_pixels for nr, nc in neighbors8(*indicator)), "indicator must be 8-isolated"
    assert (2, 2) in wall, "wall must pass through the grid center (2,2)"

    reflect = _reflection_from_indicator(indicator)
    assert wall == {reflect(r, c) for r, c in wall}, "wall must be symmetric about indicator→center axis"

    sides_touched = set()
    for r, c in wall:
        if r == 0:
            sides_touched.add("top")
        if r == 4:
            sides_touched.add("bottom")
        if c == 0:
            sides_touched.add("left")
        if c == 4:
            sides_touched.add("right")
    assert len(sides_touched) >= 2, f"wall must span edge-to-edge, only touches {sides_touched}"

    return indicator, indicator_color, wall, wall_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two source tiles into the blank fourth tile using a fused cue tile.

    Input structure:
    - Four 5x5 tiles appear in one row inside a fixed 7x25 border-separated grid.
    - Tile0 contains a 5-pixel center-crossing wall plus one detached indicator.
      The wall and indicator use different colors.
    - Tile1 and tile2 are monochrome symmetric source patterns with a shared
      background; tile3 is blank in the input.

    Rule:
    - Parse tile0 by cue roles: the singleton cue color is the indicator and the
      5-cell cue color is the wall.
    - The wall cue color must equal tile1's or tile2's foreground color, but
      this color reuse does not change the original composition logic.
    - Flood-fill from the indicator, blocked by the wall. Indicator-side cells
      copy tile1, opposite-side cells copy tile2, and wall cells prefer tile1
      with tile2 fallback only when the wall is 4-connected.

    Key insight:
    - The fused ambiguity lives in the input encoding, not in the output rule:
      the wall is identified by being the continuous 5-cell structure, even when
      its color is reused by one source panel.
    """
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, _t3 = tiles

    bg = _majority_color(t1)
    assert border != bg, "border color must differ from tile background"
    assert _majority_color(t2) == bg, "tile1 and tile2 must share the same background"

    t1_fg = _single_fg_color(t1, bg, name="tile1")
    t2_fg = _single_fg_color(t2, bg, name="tile2")
    assert t1_fg != t2_fg, "tile1 and tile2 foreground colors must differ"

    indicator, _indicator_color, wall, _wall_color = _parse_tile0(
        t0,
        t1_fg=t1_fg,
        t2_fg=t2_fg,
        shared_bg=bg,
    )

    inside = _inside_from_wall(wall, indicator)
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - wall
    assert outside, "wall must partition the grid — outside region is empty"

    out3 = _compose_output_tile(t1, t2, bg, inside, wall, _wall_is_4_connected(wall))
    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    global _NEXT_WALL_MATCHES_T1

    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]

    t1_color, t2_color, indicator_color, border = rng.sample(other, 4)
    wall_matches_t1 = _NEXT_WALL_MATCHES_T1
    _NEXT_WALL_MATCHES_T1 = not _NEXT_WALL_MATCHES_T1
    wall_color = t1_color if wall_matches_t1 else t2_color

    pool = _LBEND_CONFIGS if rng.random() < LBEND_PROB else _STRAIGHT_CONFIGS
    indicator, wall = rng.choice(pool)
    inside = _inside_from_wall(wall, indicator)
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - set(wall)
    assert outside, "prevalidated wall pool must partition the tile"

    t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    if not any(int(t1[r, c]) != tile_bg for r, c in inside):
        return None
    if not any(int(t2[r, c]) != tile_bg for r, c in outside):
        return None

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = wall_color
    t0[indicator[0], indicator[1]] = indicator_color

    out3 = _compose_output_tile(t1, t2, tile_bg, inside, wall, _wall_is_4_connected(wall))

    t3_in = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
    for j, tile in enumerate([t0, t1, t2, t3_in]):
        g[1:6, 1 + 6 * j : 1 + 6 * j + 5] = tile

    expected = g.copy()
    expected[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    got = main(g)
    if not np.array_equal(got, expected):
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
