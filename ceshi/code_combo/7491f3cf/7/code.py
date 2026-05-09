# Combo variant of 7491f3cf (index 7)
# combo_spec: D2/1+D11/2
# Name: 面积决定对应+面板同色

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import flood_fill, label_mask, neighbors8


# -----------------------------------------------------------------------------
# Design analysis for combo D2/1 + D11/2
#
# a) Concrete behavior:
#    - The layout stays identical to the original task: four 5x5 panels in one
#      row inside a 7x25 border frame. Tile0 is the cue tile, tiles1-2 are the
#      two source patterns, tile3 is blank input and becomes the output.
#    - The cue still consists of a 5-pixel wall plus one 8-isolated indicator.
#      Flood-filling from the indicator splits the tile into the indicator-side
#      region ("inside") and the opposite region ("outside").
#    - The mapping rule changes from fixed panel index to area-vs-density:
#      larger region gets the denser source pattern, smaller region gets the
#      sparser source pattern. This variant fixes the orientation to
#      large->dense and small->sparse for all samples.
#    - The color rule also changes: tiles1 and 2 use the same foreground color.
#      They differ only by shape and density, so the solver cannot rely on color
#      to tell which source contributed each output pixel.
#    - Example trace:
#      indicator=(0,0), wall={(2,2),(1,2),(0,2),(2,1),(2,0)} yields inside area
#      4 and outside area 16. If tile1 has 12 fg pixels and tile2 has 7 fg
#      pixels, then the fused rule assigns outside<-tile1 and inside<-tile2
#      even though tile1 is panel1 and both sources have the same color.
#      Under the original rule the same input would give inside<-tile1 and
#      outside<-tile2, so the output differs despite the shared color.
#
# b) Complexity calibration:
#    - Straight walls are removed from generation because they create a 10-vs-10
#      split, which makes "larger region" undefined. Only valid L-bend walls
#      with unequal region areas are used.
#    - Density bands are separated with a visible gap:
#        sparse: 6-8 pixels, dense: 11-13 pixels.
#      This makes the dense/sparse distinction visually robust even when both
#      panels share the same color.
#    - The generator alternates whether the indicator side is the larger or the
#      smaller region, so a small training set still exposes both branches.
#      It also deliberately anti-aligns tile1's density with the indicator-side
#      size so the fused rule does not collapse back to the original fixed
#      inside->tile1 / outside->tile2 behavior.
#
# c) Implicit-assumption audit:
#    - Original assumption: inside always uses tile1. Broken by D2/1. The solver
#      must compare region areas and source densities before compositing.
#    - Original assumption: tile1 and tile2 colors differ. Broken by D11/2. The
#      solver must not use source color identity as a proxy for source identity.
#    - Original wall rule "prefer tile1 on the wall, fallback tile2 if the wall
#      is 4-connected" depended on fixed inside->tile1. It is rewritten as
#      "prefer the pattern assigned to the indicator side, fallback the other".
#    - Original generator allowed overlapping density ranges and straight walls.
#      Those are no longer safe because the fused solver needs strict density
#      ordering and unequal region sizes. Generation now guarantees both.
#    - New solver assumptions:
#        * tile3 input is blank and defines the shared tile background.
#        * tiles1 and 2 are monochrome and use the same single non-bg color.
#        * tiles1 and 2 have different non-bg counts.
#        * inside and outside areas are different.
#      `generate_input()` explicitly constructs all four conditions.
#
# d) Placement legality:
#    - Wall shapes are drawn from a prevalidated set of 5-pixel center-crossing
#      L-bends. They remain inside bounds, keep the indicator 8-isolated, and
#      partition the 5x5 tile under 4-connected flood fill.
#    - Source tiles are still independent 5x5 monochrome patterns; no object
#      overlap is introduced beyond the intended overlay in the composite tile.
#
# e) Visual observability:
#    - A human can inspect the cue tile and see which region is larger.
#    - A human can inspect tiles1-2 and see which one is denser, even though
#      their colors match.
#    - Across examples, the indicator side sometimes receives the shape from
#      panel1 and sometimes from panel2; the stable explanation is region size
#      versus source density, not color or fixed panel index.
#    - No non-visual ordering such as numeric color ids is used anywhere.
#
# f) Visual cleanliness:
#    - The scene still contains only three semantic colors inside the tiles:
#      tile background, one cue color, and one shared source color. This keeps
#      the same-color trick readable instead of noisy.
#    - Because the output only paints a single foreground color onto a blank
#      tile, there is no multi-layer ambiguity inside tile3.
#
# g) Inferability of the change:
#    - The color change is intentionally minimal: tiles1 and 2 share exactly one
#      foreground color. The structure that disambiguates them is their density
#      and silhouette, both visible from the input alone.
#    - The area->density rule is fixed rather than switching per sample, which
#      avoids asking the solver to infer multiple opposing correspondences from
#      just a few demonstrations.
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


def _single_fg_color(tile: np.ndarray, bg: int) -> int:
    colors = {int(v) for v in tile.flatten() if int(v) != bg}
    assert colors, "source tile must contain foreground pixels"
    assert len(colors) == 1, "source tile must be monochrome"
    return next(iter(colors))


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


def _flood_inside(
    wall: frozenset[tuple[int, int]],
    indicator: tuple[int, int],
) -> set[tuple[int, int]]:
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    return {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}


def _wall_is_4conn(wall: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        mask[r, c] = True
    _, n_comp = label_mask(mask, connectivity=4)
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

    if t1_count > t2_count:
        dense_tile, sparse_tile = t1, t2
    else:
        dense_tile, sparse_tile = t2, t1

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


_INSIDE_SMALL_CONFIGS: list[tuple[tuple[int, int], frozenset[tuple[int, int]]]] = []
_INSIDE_LARGE_CONFIGS: list[tuple[tuple[int, int], frozenset[tuple[int, int]]]] = []
for _indicator, _wall in _LBEND_CONFIGS:
    _inside = _flood_inside(_wall, _indicator)
    _outside_area = TILE_SZ * TILE_SZ - len(_inside) - len(_wall)
    if len(_inside) < _outside_area:
        _INSIDE_SMALL_CONFIGS.append((_indicator, _wall))
    elif len(_inside) > _outside_area:
        _INSIDE_LARGE_CONFIGS.append((_indicator, _wall))
    else:
        raise AssertionError("L-bend config unexpectedly produced equal regions")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two same-colored source panels into tile3 using area-vs-density matching.

    Input structure:
    - Four 5x5 panels are arranged in one row inside a 7x25 border frame.
    - Panel 0 contains a 5-pixel wall plus one 8-isolated indicator pixel.
    - Panels 1 and 2 are monochrome source patterns that share the same
      foreground color but have different shapes and densities.
    - Panel 3 is blank input and must be filled.

    Rule:
    - Flood-fill from the indicator using 4-connectivity, blocked by the wall.
    - Compare the two region areas and the two source densities.
    - Larger region receives the denser pattern; smaller region receives the
      sparser pattern, regardless of whether that pattern came from panel 1 or 2.
    - On wall cells, prefer the pattern assigned to the indicator side and use
      the other pattern only as fallback when the wall is 4-connected.

    Key insight:
    - Panels 1 and 2 share the same foreground color, so source identity is not
      color-coded. The only stable cue is the relationship between region area
      and source density.
    """
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, t3_in = tiles

    bg = _majority_color(t3_in)
    assert border != bg, "border color must differ from tile background"
    assert np.all(t3_in == bg), "tile 3 input must be blank"

    fg1 = _single_fg_color(t1, bg)
    fg2 = _single_fg_color(t2, bg)
    assert fg1 == fg2, "tiles 1 and 2 must share the same foreground color"

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
    cue_color, pattern_color, border = rng.sample(other, 3)

    pool = _INSIDE_LARGE_CONFIGS if inside_large else _INSIDE_SMALL_CONFIGS
    indicator, wall = rng.choice(pool)
    inside = _flood_inside(wall, indicator)
    outside_area = TILE_SZ * TILE_SZ - len(inside) - len(wall)
    if len(inside) == outside_area:
        return None

    # Anti-align tile1 with the indicator-side size so the fused rule cannot
    # silently degenerate back into the original fixed inside->tile1 mapping.
    t1_dense = not inside_large

    t1 = _sample_symmetric_pattern(
        tile_bg,
        pattern_color,
        rng,
        min_px=DENSE_MIN_PX if t1_dense else SPARSE_MIN_PX,
        max_px=DENSE_MAX_PX if t1_dense else SPARSE_MAX_PX,
    )
    t2 = _sample_symmetric_pattern(
        tile_bg,
        pattern_color,
        rng,
        min_px=SPARSE_MIN_PX if t1_dense else DENSE_MIN_PX,
        max_px=SPARSE_MAX_PX if t1_dense else DENSE_MAX_PX,
    )
    if t1 is None or t2 is None:
        return None

    t1_count = _non_bg_count(t1, tile_bg)
    t2_count = _non_bg_count(t2, tile_bg)
    if t1_count == t2_count:
        return None
    if t1_dense and t1_count <= t2_count:
        return None
    if (not t1_dense) and t1_count >= t2_count:
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
