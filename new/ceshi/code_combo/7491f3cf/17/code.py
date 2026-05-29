# Combo variant of 7491f3cf (index 17)
# combo_spec: D2/0+D12/0
# Name: 颜色匹配对应+双色编码
#
# Design analysis:
# a) Concrete fused behavior and trace
#    - Keep the original 7x25 frame: four 5x5 tiles in one row, with tile3 blank
#      in the input and filled in the output.
#    - Tile0 still contains a 5-pixel wall passing through (2,2) plus one detached
#      indicator pixel at a corner or edge midpoint. The geometry of the wall and
#      the flood-fill partition are unchanged from the original task.
#    - The fusion changes two things at once:
#      1. D12/0: wall and indicator no longer share one cue color. Wall uses a
#         dedicated wall_color A. Indicator uses a separate color B.
#      2. D2/0: color B is not arbitrary. It equals either tile1's foreground
#         color or tile2's foreground color, and that color match decides which
#         source tile fills the indicator-side region.
#    - Small trace:
#      original:
#        wall splits tile, indicator marks the flood-fill seed, and solver always
#        copies tile1 into the indicator-side region and tile2 into the other side.
#      fused variant:
#        the same flood-fill recovers the indicator-side region, but if the lone
#        indicator pixel is colored like tile2, then inside must use tile2 and
#        outside must use tile1. Wall pixels also inherit the inside-source-first
#        priority, otherwise tile1 would still have hidden fixed privilege.
#    - Pixel-level consequence:
#      tile0 visibly shows two semantic colors: one for "barrier", one for
#      "choose the matching source". The output depends on both the region split
#      and the color-match branch.
#
# b) Complexity calibration
#    - The branch space stays binary: indicator color matches tile1 or tile2.
#      That is enough to force reasoning while remaining learnable from 2-4 ARC
#      examples.
#    - Two-color cueing reduces perceptual ambiguity in tile0, which compensates
#      for the added mapping rule. This keeps the fusion from becoming too opaque.
#    - Generator alternates the two branch values across calls so a sample set does
#      not silently collapse into the original always-tile1-inside behavior.
#
# c) Implicit-assumption audit of the original main()
#    - "All cue pixels in tile0 share one color": broken by D12/0. New solver must
#      expect exactly two non-background cue colors in tile0.
#    - "Indicator is detected as the unique 8-isolated cue pixel": no longer the
#      intended decoding rule. New solver identifies the indicator by its unique
#      singleton color and the wall by its 5-pixel color class.
#    - "Inside always uses tile1": broken by D2/0. New solver must read tile1/tile2
#      foreground colors, compare them to the indicator color, and route every
#      inside/outside/wall write through chosen inside/outside source variables.
#    - "Wall can reuse tile1 semantics on boundary pixels": only valid if "tile1 is
#      inside" remains fixed. The fused solver instead gives the wall the same
#      inside-source-first behavior regardless of which source the color match picks.
#
# d) Placement legality
#    - Generator reuses the original verified wall+indicator configuration pool, so
#      no new overlap or out-of-bounds cases are introduced.
#    - Wall uses exactly 5 cells; indicator uses exactly 1 distinct cell.
#    - Indicator remains detached from the wall in the same valid positions as the
#      original task, so flood-fill seeding and visual parsing stay clean.
#
# e) Visual observability
#    - A human can now see the clue directly: compare the indicator color in tile0
#      to tile1/tile2 foreground colors.
#    - Because tile1 and tile2 still use distinct monochrome foregrounds, the color
#      match can be inferred without any hidden numeric ordering.
#    - The training distribution explicitly includes both match branches, so the
#      correspondence rule is recoverable rather than accidental.
#
# f) Visual cleanliness
#    - The wall is monochrome and contiguous as one object; the indicator is a lone
#      dot in another color. The two semantics no longer compete within one color.
#    - Wall color is kept distinct from border, tile background, tile1 foreground,
#      and tile2 foreground to avoid accidental echoes.
#
# g) Change inferability and generator safeguards
#    - Color usage is still sparse: border, shared tile background, wall color, and
#      two source colors. The only new color rule is a single visible equality:
#      indicator color == one source tile's foreground color.
#    - Generator rejects tile2-inside samples whose fused output would equal the
#      original fixed-mapping output. That makes the D2 branch materially active.
#    - Generator also requires both chosen sources to contribute visible pixels so
#      the composite does not degenerate into a one-source picture.
#
# h) New solver assumptions checked against generator
#    - tile1 and tile2 each have exactly one foreground color and share one
#      background color: guaranteed by the pattern sampler and asserted in main().
#    - tile0 has exactly two cue colors, with counts 5 and 1 respectively:
#      guaranteed by construction and asserted in main().
#    - the singleton cue color equals exactly one of {tile1_fg, tile2_fg}:
#      guaranteed by construction and asserted in main().

from __future__ import annotations

import random
from collections import Counter

import numpy as np

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

_NEXT_INSIDE_IS_T1 = False

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

_VALID_INDICATOR_POS = {
    (0, 0), (0, 4), (4, 0), (4, 4),
    (0, 2), (2, 0), (2, 4), (4, 2),
}


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
        p = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
        count = 0
        for i in order:
            orbit = _ORBITS[i]
            if count + len(orbit) > max_px:
                continue
            if rng.random() < ORBIT_INCLUDE_PROB:
                for r, c in orbit:
                    p[r, c] = fg
                count += len(orbit)
        if min_px <= count <= max_px:
            return p
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


def _wall_is_4_connected(wall: set[tuple[int, int]]) -> bool:
    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    return bool(n_comp == 1)


def _inside_from_wall(wall: set[tuple[int, int]], indicator: tuple[int, int]) -> set[tuple[int, int]]:
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    return {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}


def _composite_tile(
    inside_tile: np.ndarray,
    outside_tile: np.ndarray,
    bg: int,
    inside: set[tuple[int, int]],
    wall: set[tuple[int, int]],
    wall_4conn: bool,
) -> np.ndarray:
    out3 = np.full((TILE_SZ, TILE_SZ), int(bg), dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            vin = int(inside_tile[r, c])
            vout = int(outside_tile[r, c])
            if (r, c) in inside:
                if vin != int(bg):
                    out3[r, c] = vin
            elif (r, c) in wall:
                if vin != int(bg):
                    out3[r, c] = vin
                elif wall_4conn and vout != int(bg):
                    out3[r, c] = vout
            else:
                if vout != int(bg):
                    out3[r, c] = vout
    return out3


def _parse_tile0(
    t0: np.ndarray,
    *,
    t1_fg: int,
    t2_fg: int,
) -> tuple[tuple[int, int], int, set[tuple[int, int]], int]:
    t0_bg = _majority_color(t0)
    coords_by_color: dict[int, set[tuple[int, int]]] = {}
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            color = int(t0[r, c])
            if color == t0_bg:
                continue
            coords_by_color.setdefault(color, set()).add((int(r), int(c)))

    assert len(coords_by_color) == 2, "tile0 must contain exactly two cue colors"

    singleton = [(color, coords) for color, coords in coords_by_color.items() if len(coords) == 1]
    wall_groups = [(color, coords) for color, coords in coords_by_color.items() if len(coords) == 5]
    assert len(singleton) == 1, "indicator color must occupy exactly one cell"
    assert len(wall_groups) == 1, "wall color must occupy exactly five cells"

    indicator_color, indicator_coords = singleton[0]
    wall_color, wall_coords = wall_groups[0]
    indicator = next(iter(indicator_coords))
    wall = set(wall_coords)

    assert indicator_color != wall_color, "wall and indicator must use different colors"
    assert indicator_color in {t1_fg, t2_fg}, "indicator color must match tile1 or tile2 foreground"
    assert wall_color not in {t1_fg, t2_fg}, "wall color must stay distinct from source foregrounds"
    assert indicator in _VALID_INDICATOR_POS, f"indicator {indicator} not at corner/edge-midpoint"
    assert (2, 2) in wall, "wall must pass through the grid center (2,2)"
    assert not any((nr, nc) in wall for nr, nc in neighbors8(*indicator)), "indicator must be detached from wall"

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
    Composite two source tiles into the blank fourth tile using a two-color cue tile.

    Input structure:
    - Four 5x5 tiles in one row inside a fixed 7x25 border-separated frame.
    - Tile0 contains a 5-pixel wall in one color and one detached indicator pixel
      in a second color.
    - Tile1 and tile2 are single-foreground-color symmetric source patterns.
    - Tile3 is blank in the input.

    Rule:
    - Recover the inside region by flood-filling from the indicator, blocked by
      the wall exactly as in the original task.
    - Determine which source belongs to the indicator-side region by color match:
      if the indicator color matches tile1's foreground, inside uses tile1;
      otherwise inside uses tile2.
    - Outside uses the other tile. Wall pixels prefer the inside-source pixel and
      fall back to the outside-source pixel only when the wall is 4-connected.

    Key insight:
    - Tile0 now uses color to separate cue roles: one color means "wall" and the
      other means "pick the source tile with this same foreground color".
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

    indicator, indicator_color, wall, _wall_color = _parse_tile0(t0, t1_fg=t1_fg, t2_fg=t2_fg)
    inside = _inside_from_wall(wall, indicator)
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - wall
    assert outside, "wall must partition the grid — outside region is empty"

    if indicator_color == t1_fg:
        inside_tile, outside_tile = t1, t2
    else:
        inside_tile, outside_tile = t2, t1

    wall_4conn = _wall_is_4_connected(wall)
    out3 = _composite_tile(inside_tile, outside_tile, bg, inside, wall, wall_4conn)

    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _flood_inside(wall: frozenset[tuple[int, int]], indicator: tuple[int, int]) -> set[tuple[int, int]]:
    return _inside_from_wall(set(wall), indicator)


def _count_fg_hits(tile: np.ndarray, bg: int, coords: set[tuple[int, int]]) -> int:
    return sum(int(tile[r, c] != bg) for r, c in coords)


def _gen_attempt(rng: random.Random, *, inside_is_t1: bool) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    wall_color, t1_color, t2_color, border = rng.sample(other, 4)

    pool = _LBEND_CONFIGS if rng.random() < LBEND_PROB else _STRAIGHT_CONFIGS
    indicator, wall = rng.choice(pool)
    inside = _flood_inside(wall, indicator)
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - set(wall)

    t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    indicator_color = t1_color if inside_is_t1 else t2_color
    inside_tile, outside_tile = (t1, t2) if inside_is_t1 else (t2, t1)

    if _count_fg_hits(inside_tile, tile_bg, inside) == 0:
        return None
    if _count_fg_hits(outside_tile, tile_bg, outside) == 0:
        return None

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = wall_color
    t0[indicator[0], indicator[1]] = indicator_color

    wall_4conn = _wall_is_4_connected(set(wall))
    out3 = _composite_tile(inside_tile, outside_tile, tile_bg, inside, set(wall), wall_4conn)

    if not np.any(out3 == t1_color) or not np.any(out3 == t2_color):
        return None

    if not inside_is_t1:
        original_out3 = _composite_tile(t1, t2, tile_bg, inside, set(wall), wall_4conn)
        if np.array_equal(out3, original_out3):
            return None

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
    global _NEXT_INSIDE_IS_T1

    desired_inside_is_t1 = bool(_NEXT_INSIDE_IS_T1)
    _NEXT_INSIDE_IS_T1 = not _NEXT_INSIDE_IS_T1

    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng, inside_is_t1=desired_inside_is_t1)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
