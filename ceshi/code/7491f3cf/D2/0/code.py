# Variant of 7491f3cf
# Dimension: D2 — 内外区域与图案的对应关系 (match_cond)
# Change: 指示像素侧对应面板1（inside→tile1），另一侧对应面板2（outside→tile2），对应关系固定... → 颜色匹配决定对应
#
# Design analysis:
#   variant_goal:
#     keep:
#       - 7x25 overall layout, four 5x5 panels, same wall geometry and flood-fill logic
#       - tile1/tile2 remain single-foreground-color symmetric source patterns
#       - tile3 is still produced by compositing one source on the indicator side and the other on the opposite side
#     change:
#       - original task hard-coded "inside uses tile1"
#       - new variant makes the mapping conditional on the indicator color:
#         indicator color == tile1 foreground => inside uses tile1
#         indicator color == tile2 foreground => inside uses tile2
#
#   concrete_pixel_level_behavior:
#     tile0:
#       - background stays tile_bg
#       - wall stays a single cue color distinct from tile backgrounds and source colors
#       - indicator is still the unique 8-isolated cue pixel by geometry, but its color is no longer the wall color
#       - indicator color now equals either tile1's foreground color or tile2's foreground color
#     tile1/tile2:
#       - each tile has exactly one non-bg foreground color, distinct from each other
#     tile3 output:
#       - find inside region exactly as before by flood-filling from the indicator and blocking on the wall
#       - choose which source belongs to inside by color match, then composite relative to that choice
#
#   behavior_trace_example:
#     original:
#       - indicator at top edge, wall splits tile, inside region found
#       - solver always copies tile1 into inside and tile2 into outside
#     variant:
#       - same indicator position and same inside region
#       - if indicator pixel is colored like tile2, inside must now use tile2 and outside must use tile1
#       - wall priority must follow the inside-source as well, otherwise tile1 would keep a hidden fixed privilege and D2 would not be the true control variable
#
#   complexity_calibration:
#     - only two branch values exist: indicator matches tile1 or indicator matches tile2
#     - that keeps the conditional small enough for ARC, but still forces explicit color reasoning
#     - generator alternates the branch across calls so a training set built from repeated calls is unlikely to collapse into the original fixed mapping
#
#   implicit_assumption_audit_for_main:
#     - old code assumed tile1 is always the inside source in all three write paths:
#       inside branch, wall primary branch, and outside default branch
#     - those assumptions break under the variant because the correct source may now be tile2
#     - old code also assumed all cue pixels in tile0 can share one color; that is no longer valid because the indicator now carries semantic color information
#     - solver therefore must:
#       - recover the isolated indicator by geometry, not by color
#       - read tile1/tile2 foreground colors
#       - assert indicator color matches exactly one of those foreground colors
#       - route all compositing through inside-source/outside-source variables instead of hard-coded tile indices
#
#   placement_legality_and_generator_guarantees:
#     - wall and indicator positions are still chosen from the original verified configuration pool, so no overlap/out-of-bounds risk is introduced
#     - indicator remains 8-isolated because only its color changed, not its position
#     - tile1/tile2 keep shared tile_bg and distinct single foreground colors
#     - generator validates every sample by running main(input)
#
#   visual_observability:
#     - the clue is directly visible: the isolated indicator has the same color as one source tile
#     - because tile1 and tile2 have different foreground colors, a human can compare tile0 with tile1/tile2 and infer which source fills the indicator side
#     - branch alternation across generate_input calls ensures both match cases appear regularly instead of silently reverting to the original always-tile1 behavior
#
#   visual_cleanliness:
#     - wall color is kept separate from source colors so the indicator remains the only color-match clue in tile0
#     - border is sampled distinct from cue/tile colors in the generator to avoid accidental color echoes around panels
#
#   new_main_assumptions_checked_against_generator:
#     - each source tile has exactly one non-bg color: guaranteed by the pattern sampler
#     - indicator color differs from wall color and equals one source foreground color: guaranteed by generator construction
#     - tile1 and tile2 share the same background: guaranteed by generator construction and asserted in main

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import flood_fill, label_mask, neighbors8


def _majority_color(tile: np.ndarray) -> int:
    return int(Counter(map(int, tile.flatten())).most_common(1)[0][0])


def _single_fg_color(tile: np.ndarray, bg: int, *, name: str) -> int:
    fg = {int(v) for v in tile.flatten() if int(v) != int(bg)}
    assert len(fg) == 1, f"{name} must use exactly one foreground color"
    return next(iter(fg))


def _wall_is_4_connected(wall: set[tuple[int, int]]) -> bool:
    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    return bool(n_comp == 1)


def _composite_tile(
    inside_tile: np.ndarray,
    outside_tile: np.ndarray,
    bg: int,
    inside: set[tuple[int, int]],
    wall: set[tuple[int, int]],
    wall_4conn: bool,
) -> np.ndarray:
    """
    Build tile3 relative to the chosen inside/outside sources.

    This preserves the original wall behavior but re-anchors it to the inside
    source: wall uses the inside-source pixel first, and only falls back to the
    outside-source pixel when the wall is 4-connected.
    """
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant solver for 7491f3cf.

    Same geometric task as the original, except the isolated indicator pixel now
    decides the inside/outside source mapping by color match with tile1/tile2.
    """
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, _t3 = tiles

    b = _majority_color(t1)
    assert border != b, "border color must differ from tile background"
    assert _majority_color(t2) == b, "tile1 and tile2 must share the same background"

    t1_fg = _single_fg_color(t1, b, name="tile1")
    t2_fg = _single_fg_color(t2, b, name="tile2")
    assert t1_fg != t2_fg, "tile1 and tile2 foreground colors must differ"

    t0_bg = _majority_color(t0)
    cue_set = {(int(r), int(c)) for r in range(TILE_SZ) for c in range(TILE_SZ) if int(t0[r, c]) != t0_bg}
    assert cue_set, "no cue pixels found"

    # Indicator = the unique 8-isolated cue pixel.
    indicator = None
    for r, c in cue_set:
        if not any((nr, nc) in cue_set for nr, nc in neighbors8(r, c)):
            assert indicator is None, "multiple 8-isolated cue pixels"
            indicator = (r, c)
    assert indicator is not None, "no 8-isolated cue pixel found"

    valid_pos = {(0, 0), (0, 4), (4, 0), (4, 4), (0, 2), (2, 0), (2, 4), (4, 2)}
    assert indicator in valid_pos, f"indicator {indicator} not at corner/edge-midpoint"

    wall = cue_set - {indicator}
    assert (2, 2) in wall, "wall must pass through the grid center (2,2)"

    wall_colors = {int(t0[r, c]) for r, c in wall}
    assert len(wall_colors) == 1, "wall must be monochrome"
    wall_color = next(iter(wall_colors))
    indicator_color = int(t0[indicator[0], indicator[1]])
    assert indicator_color != wall_color, "indicator color must differ from wall color"
    assert indicator_color in {t1_fg, t2_fg}, "indicator color must match tile1 or tile2 foreground"

    # Wall must be symmetric about the line from indicator to center (2,2).
    ir, ic = indicator
    dr, dc = 2 - ir, 2 - ic
    if dr == 0:
        _refl = lambda r, c: (4 - r, c)
    elif dc == 0:
        _refl = lambda r, c: (r, 4 - c)
    elif dr * dc > 0:
        _refl = lambda r, c: (c, r)
    else:
        _refl = lambda r, c: (4 - c, 4 - r)
    assert wall == {_refl(r, c) for r, c in wall}, "wall must be symmetric about indicator→center axis"

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

    # Flood-fill from the indicator to recover the inside region.
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    inside = {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - wall
    assert outside, "wall must partition the grid — outside region is empty"

    if indicator_color == t1_fg:
        inside_tile, outside_tile = t1, t2
    else:
        inside_tile, outside_tile = t2, t1

    wall_4conn = _wall_is_4_connected(wall)
    out3 = _composite_tile(inside_tile, outside_tile, b, inside, wall, wall_4conn)

    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


# --------------- generator ---------------

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


def _flood_inside(wall: frozenset[tuple[int, int]], indicator: tuple[int, int]) -> set[tuple[int, int]]:
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    return {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}


def _gen_attempt(rng: random.Random, *, inside_is_t1: bool) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color, border = rng.sample(other, 4)

    pool = _LBEND_CONFIGS if rng.random() < LBEND_PROB else _STRAIGHT_CONFIGS
    indicator, wall = rng.choice(pool)
    inside = _flood_inside(wall, indicator)

    t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    indicator_color = t1_color if inside_is_t1 else t2_color
    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = cue_color
    t0[indicator[0], indicator[1]] = indicator_color

    wall_4conn = _wall_is_4_connected(set(wall))
    inside_tile, outside_tile = (t1, t2) if inside_is_t1 else (t2, t1)
    out3 = _composite_tile(inside_tile, outside_tile, tile_bg, inside, set(wall), wall_4conn)

    # For the tile2-inside branch, explicitly reject accidental degeneration
    # back to the original fixed mapping if it ever occurred.
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
