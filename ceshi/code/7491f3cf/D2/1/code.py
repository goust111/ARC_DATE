# Variant of 7491f3cf
# Dimension: D2 — 内外区域与图案的对应关系 (match_cond)
# Change: 指示像素侧对应面板1（inside→tile1），另一侧对应面板2（outside→tile2），对应关系固定
#   → 改为按“区域面积 vs 图案密度”决定：较大区域取较密图案，较小区域取较疏图案。

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import flood_fill, label_mask, neighbors8


# -----------------------------------------------------------------------------
# Design analysis for this D2 variant
#
# a) Concrete behavior:
#    - Layout stays unchanged: four 5x5 tiles in one row, tile0 is cue, tiles1-2
#      are source patterns, tile3 is blank input and becomes the composite output.
#    - Tile0 still contains a 5-pixel wall plus one isolated indicator pixel.
#    - The variant changes only the mapping rule between the two flood-fill
#      regions and the two source tiles.
#    - Tile1 is generated as the sparser pattern; tile2 is generated as the
#      denser pattern. Solver does not assume this by index and instead measures
#      non-background pixel counts directly.
#    - The larger flood-fill region receives the denser pattern; the smaller
#      region receives the sparser pattern. The wall continues to prefer the
#      pattern assigned to the indicator side, with the original tile2-style
#      fallback behavior preserved abstractly as “other pattern fallback when
#      the wall is 4-connected”.
#    - Example trace:
#      indicator=(0,0), wall={(0,2),(1,2),(2,0),(2,1),(2,2)} gives inside area
#      4 and outside area 16. Original rule: inside<-tile1, outside<-tile2.
#      Variant rule: small inside<-sparse tile, large outside<-dense tile.
#      If the reflected L-bend is used instead, inside area becomes 16, so the
#      indicator side switches to the dense tile. That is the key visible change.
#
# b) Complexity calibration:
#    - Straight walls split 10 vs 10, which makes “larger region” undefined.
#      They are removed from generation so the new rule is always active.
#    - Only two density bands are used, with a clear gap:
#        sparse: 6-8 pixels, dense: 11-13 pixels.
#      This keeps the task readable by eye while still requiring an actual
#      comparison rather than a brittle exact-count lookup.
#    - The generator alternates whether the indicator side is the larger or
#      smaller region across calls, so small training sets still expose both
#      branches of the new rule.
#
# c) Old implicit assumptions that break under the variant:
#    - `if (r,c) in inside: use v1` assumed inside always corresponds to tile1.
#      This is no longer valid; the solver must compare region areas and tile
#      densities first.
#    - Wall handling `prefer v1, fallback v2` also depended on that fixed
#      correspondence. It is rewritten to prefer the tile assigned to the
#      indicator side, regardless of whether it is tile1 or tile2.
#    - Generator choice `straight or L-bend` assumed both were valid under the
#      old fixed mapping. Under size-ordering, straight walls are ambiguous.
#    - Overlapping tile1/tile2 density ranges were harmless before, but now the
#      solver needs a strict sparse/dense ordering. Generation enforces it.
#
# d) Placement legality:
#    - All wall configurations remain precomputed valid 5-pixel cue shapes that
#      pass through the center, keep the indicator 8-isolated, and partition the
#      5x5 tile under 4-connected flood fill.
#    - Tiles remain fixed 5x5 and assembled into the original 7x25 frame, so no
#      new overlap or out-of-bounds risk is introduced.
#
# e) Visual observability:
#    - The rule is inferable from a few examples because both signals are visible:
#      the cue tile visibly creates a 4-vs-16 split, and tiles1-2 visibly differ
#      in density. Which source color fills the indicator side changes depending
#      on whether that side is larger or smaller.
#    - No hidden use of color ids, ordering by numeric value, or non-visual
#      metadata is introduced.
#
# f) Visual cleanliness:
#    - The scene still uses one cue color plus one monochrome color per source
#      tile. There is no extra overlay or auxiliary marker beyond the original
#      wall/indicator mechanism, so the output remains easy to parse.
#
# g) Color inferability:
#    - Colors are still arbitrary distinct ARC palette ids, but color identity is
#      not the control variable. Density and region area drive the mapping, which
#      are both directly visible from shape structure alone.
#
# New solver assumptions introduced by this variant:
#    - The two flood-fill regions must have unequal area.
#    - The two source tiles must have unequal non-background pixel counts.
#    - The background color is shared across tiles1-3.
#   `generate_input()` guarantees all three conditions.
# -----------------------------------------------------------------------------


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

SPARSE_MIN_PX, SPARSE_MAX_PX = 6, 8
DENSE_MIN_PX, DENSE_MAX_PX = 11, 13
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
GEN_RETRIES = 300


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


_NEXT_INSIDE_LARGE = False


def _majority_color(tile: np.ndarray) -> int:
    return int(Counter(map(int, tile.flatten())).most_common(1)[0][0])


def _non_bg_count(tile: np.ndarray, bg: int) -> int:
    return int(np.count_nonzero(tile != bg))


def _flood_inside(wall: frozenset[tuple[int, int]], indicator: tuple[int, int]) -> set[tuple[int, int]]:
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    return {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}


def _wall_is_4conn(wall: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    return bool(n_comp == 1)


def _choose_region_tiles(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    inside_area: int,
    outside_area: int,
) -> tuple[np.ndarray, np.ndarray]:
    t1_count = _non_bg_count(t1, bg)
    t2_count = _non_bg_count(t2, bg)
    assert t1_count != t2_count, "source tiles must differ in density"
    assert inside_area != outside_area, "regions must have unequal area"

    if t1_count < t2_count:
        sparse_tile, dense_tile = t1, t2
    else:
        sparse_tile, dense_tile = t2, t1

    if inside_area > outside_area:
        return dense_tile, sparse_tile
    return sparse_tile, dense_tile


def _compose_output_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    inside: set[tuple[int, int]],
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
) -> np.ndarray:
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - set(wall)
    assert outside, "wall must partition the grid — outside region is empty"

    inside_tile, outside_tile = _choose_region_tiles(t1, t2, bg, len(inside), len(outside))
    wall_4conn = _wall_is_4conn(wall)

    out3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            vin = int(inside_tile[r, c])
            vout = int(outside_tile[r, c])
            if (r, c) in inside:
                if vin != bg:
                    out3[r, c] = vin
            elif (r, c) in wall:
                if vin != bg:
                    out3[r, c] = vin
                elif wall_4conn and vout != bg:
                    out3[r, c] = vout
            else:
                if vout != bg:
                    out3[r, c] = vout
    return out3


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


_INSIDE_SMALL_CONFIGS = []
_INSIDE_LARGE_CONFIGS = []
for _indicator, _wall in _LBEND_CONFIGS:
    _inside = _flood_inside(_wall, _indicator)
    _outside_area = TILE_SZ * TILE_SZ - len(_inside) - len(_wall)
    if len(_inside) < _outside_area:
        _INSIDE_SMALL_CONFIGS.append((_indicator, _wall))
    elif len(_inside) > _outside_area:
        _INSIDE_LARGE_CONFIGS.append((_indicator, _wall))
    else:
        raise AssertionError("LBend config unexpectedly produced equal regions")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite tiles1-2 into tile3 by comparing region area against pattern density.

    Variant rule:
    - Flood-fill from the indicator, blocked by the wall.
    - Larger region gets the denser source pattern.
    - Smaller region gets the sparser source pattern.
    - Wall prefers the pattern assigned to the indicator side, with the original
      4-connected fallback behavior preserved for the other pattern.
    """
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, t3_in = tiles
    bg = _majority_color(t3_in)
    assert border != bg, "border color must differ from tile background"
    assert np.all(t3_in == bg), "tile 3 input must be blank"

    t0_bg = _majority_color(t0)
    cue_set = {(int(r), int(c)) for r in range(TILE_SZ) for c in range(TILE_SZ) if int(t0[r, c]) != t0_bg}
    assert cue_set, "no cue pixels found"

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

    ir, ic = indicator
    dr, dc = 2 - ir, 2 - ic
    if dr == 0:
        reflect = lambda r, c: (4 - r, c)
    elif dc == 0:
        reflect = lambda r, c: (r, 4 - c)
    elif dr * dc > 0:
        reflect = lambda r, c: (c, r)
    else:
        reflect = lambda r, c: (4 - c, 4 - r)
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

    inside = _flood_inside(frozenset(wall), indicator)

    out3 = _compose_output_tile(t1, t2, bg, inside, wall)

    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _gen_attempt(rng: random.Random, *, inside_large: bool) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color = rng.sample(other, 3)
    border = rng.choice(other)

    pool = _INSIDE_LARGE_CONFIGS if inside_large else _INSIDE_SMALL_CONFIGS
    indicator, wall = rng.choice(pool)
    inside = _flood_inside(wall, indicator)
    outside_area = TILE_SZ * TILE_SZ - len(inside) - len(wall)
    if len(inside) == outside_area:
        return None

    t1 = _sample_symmetric_pattern(
        tile_bg,
        t1_color,
        rng,
        min_px=SPARSE_MIN_PX,
        max_px=SPARSE_MAX_PX,
    )
    t2 = _sample_symmetric_pattern(
        tile_bg,
        t2_color,
        rng,
        min_px=DENSE_MIN_PX,
        max_px=DENSE_MAX_PX,
    )
    if t1 is None or t2 is None:
        return None
    if _non_bg_count(t1, tile_bg) >= _non_bg_count(t2, tile_bg):
        return None

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = cue_color
    t0[indicator[0], indicator[1]] = cue_color

    out3 = _compose_output_tile(t1, t2, tile_bg, inside, wall)

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
    global _NEXT_INSIDE_LARGE

    preferred_inside_large = _NEXT_INSIDE_LARGE
    _NEXT_INSIDE_LARGE = not _NEXT_INSIDE_LARGE

    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng, inside_large=preferred_inside_large)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
