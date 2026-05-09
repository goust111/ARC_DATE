# Variant of 7491f3cf
# Dimension: D9 — Flood-fill连通性 (meta)
# Change: 4-连通flood-fill（上下左右），从指示像素出发被墙阻挡，确定inside区域... → 8-连通flood-fill

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import flood_fill, label_mask, neighbors8


# Design analysis for the D9 variant
# ----------------------------------
# variant_spec:
#   unchanged_layout:
#     grid: 7x25
#     tiles: four 5x5 panels in one row with 1-cell border separators
#     tile0: cue panel = wall + one isolated indicator pixel
#     tile1/tile2: source patterns
#     tile3: blank input, composed output
#   core_change:
#     original: flood-fill from indicator used 4-connectivity
#     variant: flood-fill from indicator uses 8-connectivity
#   concrete_behavior:
#     blocking_walls:
#       - horizontal row through center
#       - vertical column through center
#       - orthogonal L-bends through center
#       These still split the board under both 4-conn and 8-conn flood-fill.
#     leaky_walls:
#       - thin main diagonal
#       - thin anti-diagonal
#       - diagonal V-like 5-pixel walls from the original task
#       These split under 4-conn but not under 8-conn because diagonal gaps are traversable.
#       In this variant they are still legal cue shapes, but they no longer create a non-empty
#       outside region: the indicator flood reaches every non-wall cell.
#   example_trace:
#     cue:
#       indicator: (0,4)
#       wall: {(0,0),(1,1),(2,2),(3,3),(4,4)}   # thin main diagonal
#     original_4_conn:
#       flood from (0,4) cannot cross the diagonal cells, so inside is the upper-right half
#       and outside is the lower-left half
#     variant_8_conn:
#       flood from (0,4) can step around the line through diagonal corners, e.g.
#       (0,4)->(1,4)->(2,3)->(3,2)->(4,1), so every non-wall cell becomes inside
#       and outside becomes empty
#     observable_difference:
#       tile3 changes from "t1 on one side, t2 on the other" to "almost all t1"
#       for the same visual diagonal cue
#   complexity_calibration:
#     - Keep the original cue grammar and panel layout unchanged so the task remains locally readable.
#     - Use only two wall families at generation time:
#       1. blocking family: output still mixes t1/t2 spatially
#       2. leaky family: output collapses to t1 on all non-wall cells
#     - Alternate families across generator calls so a small training set is likely to include both.
#   implicit_assumptions_audit:
#     - Original `assert outside` depended on every legal wall partitioning under 4-conn.
#       This is false after the D9 change, so the assertion must be removed.
#     - The indicator remains the unique 8-isolated cue pixel. This still holds.
#     - Wall symmetry about the indicator->center axis still holds for all reused wall shapes.
#     - Wall fallback on wall cells depends on wall 4-connectivity, not flood connectivity.
#       That behavior is left unchanged because D9 only changes the inside-region connectivity.
#   placement_legality:
#     - All reused walls stay inside 5x5, pass through center, and touch >=2 sides.
#     - Indicator stays at a corner or edge midpoint and is 8-isolated from the wall.
#     - No overlap issues change: tile0 encodes cue only, tile1/tile2 are independent patterns,
#       tile3 input stays blank.
#   visual_observability:
#     - Leaky diagonal cases are the critical evidence for 8-connectivity; they are forced to
#       produce a different tile3 than the old 4-conn rule.
#     - Blocking row/column/L cases remain present as contrast, so the cue is not reduced to
#       a trivial "always copy tile1" rule.
#   new_solver_assumptions_and_generator_guarantees:
#     - Solver now allows `outside == empty`.
#     - Generator guarantees that leaky samples have `output_8 != output_4`.
#     - Generator also guarantees visible support for the selected regions:
#       t1 contributes inside pixels, and whenever outside is non-empty, t2 contributes outside pixels.


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
GEN_RETRIES = 300

# Alternate families so short training sets are likely to show both 8-conn behaviors.
_GEN_COUNTER = 0

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

_BLOCKING_CONFIGS = (
    _STRAIGHT_CONFIGS[:4]
    + _LBEND_CONFIGS[:8]
)

_LEAKY_CONFIGS = (
    _STRAIGHT_CONFIGS[4:]
    + _LBEND_CONFIGS[8:]
)


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


def _flood_inside(
    wall: frozenset[tuple[int, int]],
    indicator: tuple[int, int],
    *,
    connectivity: int,
) -> set[tuple[int, int]]:
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=connectivity)
    return {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}


def _compose_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    wall: frozenset[tuple[int, int]],
    inside: set[tuple[int, int]],
) -> np.ndarray:
    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    wall_4conn = (n_comp == 1)

    out3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1, v2 = int(t1[r, c]), int(t2[r, c])
            if (r, c) in inside:
                if v1 != bg:
                    out3[r, c] = v1
            elif (r, c) in wall:
                if v1 != bg:
                    out3[r, c] = v1
                elif wall_4conn and v2 != bg:
                    out3[r, c] = v2
            else:
                if v2 != bg:
                    out3[r, c] = v2
    return out3


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite the two source tiles into the fourth panel.

    D9 variant:
    - The indicator flood-fill uses 8-connectivity.
    - Therefore some thin diagonal walls no longer partition the tile, and `outside`
      may legitimately be empty.
    """
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, _t3 = tiles
    bg = int(Counter(t1.flatten()).most_common(1)[0][0])
    assert border != bg, "border color must differ from tile background"

    t0_bg = int(Counter(t0.flatten()).most_common(1)[0][0])
    cue_set = {(int(r), int(c)) for r in range(5) for c in range(5) if int(t0[r, c]) != t0_bg}
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

    inside = _flood_inside(wall, indicator, connectivity=8)
    assert indicator in inside, "indicator must lie in the filled region"

    out3 = _compose_tile(t1, t2, bg, wall, inside)

    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _choose_family() -> str:
    global _GEN_COUNTER
    family = "leaky" if (_GEN_COUNTER % 2 == 0) else "blocking"
    _GEN_COUNTER += 1
    return family


def _region_has_fg(tile: np.ndarray, bg: int, coords: set[tuple[int, int]]) -> bool:
    return any(int(tile[r, c]) != bg for r, c in coords)


def _gen_attempt(rng: random.Random, family: str) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color = rng.sample(other, 3)
    border = rng.choice(other)

    pool = _LEAKY_CONFIGS if family == "leaky" else _BLOCKING_CONFIGS
    indicator, wall = rng.choice(pool)

    inside8 = _flood_inside(wall, indicator, connectivity=8)
    outside8 = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside8 - set(wall)
    inside4 = _flood_inside(wall, indicator, connectivity=4)

    if family == "leaky":
        if outside8:
            return None
    else:
        if not outside8:
            return None

    t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    if not _region_has_fg(t1, tile_bg, inside8):
        return None
    if outside8 and not _region_has_fg(t2, tile_bg, outside8):
        return None

    out3_8 = _compose_tile(t1, t2, tile_bg, wall, inside8)
    if family == "leaky":
        out3_4 = _compose_tile(t1, t2, tile_bg, wall, inside4)
        if np.array_equal(out3_8, out3_4):
            return None

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = cue_color
    t0[indicator[0], indicator[1]] = cue_color

    t3_in = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
    for j, tile in enumerate([t0, t1, t2, t3_in]):
        g[1:6, 1 + 6 * j : 1 + 6 * j + 5] = tile

    expected = g.copy()
    expected[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3_8
    got = main(g)
    if not np.array_equal(got, expected):
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    family = _choose_family()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng, family)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts for family={family}")
