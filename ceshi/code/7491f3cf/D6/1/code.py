# Variant of 7491f3cf
# Dimension: D6 — 合成变换规则 (transform)
# Change: 区域像素合成——inside区域取tile1非背景像素，outside区域取tile2非背景像素，墙上取tile1优先、tile2回退（仅4-连通墙时） → 异或合成

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import flood_fill, label_mask, neighbors8


"""
Design analysis for the D6 variant
==================================

spec:
  unchanged_structure:
    - Grid stays 7x25 with four 5x5 tiles in one row.
    - Tile 0 still contains one 8-isolated indicator and one 5-pixel wall through (2,2).
    - Tile 1 and tile 2 stay single-foreground-color symmetric source patterns.
    - Tile 3 input stays blank and must be inferred.
  changed_transform:
    - The original rule chose "which tile wins" by inside/outside/wall position.
    - The variant instead compares tile1 and tile2 occupancy at every aligned cell.
    - Output cell is non-background iff exactly one of {tile1, tile2} is non-background.
    - The output color is the lone surviving source color at that cell.
    - If both are foreground or both are background, the output is background.

  behavior_trace:
    - Example cell: inside region, t1=red, t2=bg.
      original -> red, variant -> red.
    - Example cell: inside region, t1=red, t2=green.
      original -> red, variant -> bg because the overlap cancels.
    - Example cell: outside region, t1=red, t2=bg.
      original -> bg, variant -> red because XOR keeps the lone foreground.
    - So the variant is not "pick one side's winner"; it is "remove overlaps, keep exclusives".

  complexity_calibration:
    - A naive XOR generator could accidentally produce examples close to the original rule,
      e.g. almost no overlaps or all exclusive pixels on one side.
    - To keep the D6 change visible, generation enforces:
      1. some overlapped cells, so cancellation is visible,
      2. some tile1-only and tile2-only cells, so both source colors survive,
      3. XOR-visible pixels on both inside and outside, so the wall still aligns with a
         meaningful spatial split even though it no longer resolves conflicts,
      4. a minimum number of cells where XOR output differs from the original compositor.

  implicit_assumption_audit_of_original_main:
    - Parsing border, tile extraction, indicator detection, wall symmetry, and flood-fill
      all remain valid because the input format is unchanged.
    - The old output loop assumed region membership alone determined which source could paint
      a cell. That assumption is invalid now because a tile1-only pixel outside the region or
      a tile2-only pixel inside the region must still survive under XOR.
    - The old wall-specific fallback based on 4-connectivity is also invalid: in the variant,
      a wall cell with both sources present must cancel to background, not prefer tile1/tile2.
    - Therefore only the final composition stage changes; geometric validation remains useful.

  placement_legality:
    - No new objects are added, so non-overlap and bounds conditions are unchanged.
    - Source patterns remain fully inside their 5x5 tiles.
    - The generator still verifies its constructed input by calling main() and matching the
      intended tile3 output.

  visual_observability:
    - Humans can infer the rule because output tile3 looks like the symmetric difference of
      tile1 and tile2: shared parts disappear, unique parts remain in their own colors.
    - The generator avoids degenerate samples where XOR and the original rule look too similar.

  visual_cleanliness:
    - The output uses only tile_bg plus the two source colors; no extra color mapping is added.
    - Cancellation simplifies crowded overlaps instead of adding more layers, so the grid stays
      readable.

  color_inference:
    - The variant does not remap colors numerically.
    - A surviving pixel keeps the color of the unique contributing source, which is directly
      observable from tile1/tile2.
"""


# ── Tunables ──────────────────────────────────────────────────────────────────
TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
GEN_RETRIES = 400
LBEND_PROB = 0.2

MIN_OVERLAP = 3
MIN_T1_ONLY = 2
MIN_T2_ONLY = 2
MIN_DIFF_FROM_ORIGINAL = 4


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


def _flood_inside(wall: frozenset[tuple[int, int]], indicator: tuple[int, int]) -> set[tuple[int, int]]:
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    return {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}


def _wall_is_4_connected(wall: frozenset[tuple[int, int]]) -> bool:
    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    return n_comp == 1


def _compose_original(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    inside: set[tuple[int, int]],
    wall: set[tuple[int, int]],
    wall_4conn: bool,
) -> np.ndarray:
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


def _compose_xor(t1: np.ndarray, t2: np.ndarray, bg: int) -> np.ndarray:
    out3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1, v2 = int(t1[r, c]), int(t2[r, c])
            has1 = v1 != bg
            has2 = v2 != bg
            if has1 ^ has2:
                out3[r, c] = v1 if has1 else v2
    return out3


def _mask_from_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, _t3 = tiles
    b = int(Counter(t1.flatten()).most_common(1)[0][0])
    assert border != b, "border color must differ from tile background"

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

    tmp = np.zeros((5, 5), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    inside = {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}
    outside = {(r, c) for r in range(5) for c in range(5)} - inside - wall
    assert outside, "wall must partition the grid — outside region is empty"

    # The region split is still parsed and validated so the cue tile keeps the same
    # geometric meaning as the original task, but D6 changes the compositor itself:
    # tile3 becomes the occupancy XOR of tile1 and tile2.
    out3 = _compose_xor(t1, t2, b)

    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color = rng.sample(other, 3)
    border = rng.choice(other)

    pool = _LBEND_CONFIGS if rng.random() < LBEND_PROB else _STRAIGHT_CONFIGS
    indicator, wall = rng.choice(pool)
    inside = _flood_inside(wall, indicator)
    wall_4conn = _wall_is_4_connected(wall)

    t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    occ1 = t1 != tile_bg
    occ2 = t2 != tile_bg
    overlap = occ1 & occ2
    t1_only = occ1 & ~occ2
    t2_only = occ2 & ~occ1

    inside_mask = _mask_from_coords(inside)
    wall_mask = _mask_from_coords(set(wall))
    outside_mask = ~(inside_mask | wall_mask)
    xor_mask = t1_only | t2_only

    if int(overlap.sum()) < MIN_OVERLAP:
        return None
    if int(t1_only.sum()) < MIN_T1_ONLY or int(t2_only.sum()) < MIN_T2_ONLY:
        return None
    if int((xor_mask & inside_mask).sum()) == 0:
        return None
    if int((xor_mask & outside_mask).sum()) == 0:
        return None
    if int((overlap & (inside_mask | outside_mask)).sum()) == 0:
        return None

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = cue_color
    t0[indicator[0], indicator[1]] = cue_color

    out3 = _compose_xor(t1, t2, tile_bg)
    original_out3 = _compose_original(t1, t2, tile_bg, inside, set(wall), wall_4conn)
    if int(np.sum(out3 != original_out3)) < MIN_DIFF_FROM_ORIGINAL:
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
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
