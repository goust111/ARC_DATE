# Combo variant of 7491f3cf (index 19)
# combo_spec: D6/1+D9/0
# Name: 异或合成+8-连通flood-fill

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import flood_fill, label_mask, neighbors8


"""
Design analysis for combo D6/1 + D9/0
=====================================

spec:
  unchanged_structure:
    - The grid is still 7x25 with four 5x5 tiles laid out in one row.
    - Tile 0 is still a cue tile containing a 5-pixel wall through (2,2) plus one
      8-isolated indicator pixel.
    - Tile 1 and tile 2 remain single-foreground-color symmetric source patterns.
    - Tile 3 input remains blank and must be inferred.

  fused_rule_interpretation:
    - D9 changes the flood-fill connectivity from 4-neighbor to 8-neighbor, so the
      inside region may expand through diagonal gaps.
    - D6 changes the compositor from overwrite-by-region to XOR-filtered routing:
      a cell can paint only when tile1 and tile2 have exclusive occupancy there.
    - To keep both dimensions active in the combo, region routing is preserved:
      inside accepts tile1-only pixels, outside accepts tile2-only pixels, wall keeps
      the original tile1-first routing but still requires exclusivity.
    - This is stricter than the original compositor and sparser than overwrite logic,
      while still making the cue wall matter under 8-connectivity.

  behavior_trace:
    - Use a leaky diagonal cue:
      indicator=(0,4), wall={(0,0),(1,1),(2,2),(3,3),(4,4)}.
    - Original 4-conn overwrite:
      cells below-left of the diagonal are outside, so tile2 can appear there even if
      tile1 is absent.
    - Fused 8-conn XOR-routing:
      the same below-left cells become inside because the flood can slip diagonally
      around the thin wall; now only tile1-only pixels may survive there, tile2-only
      pixels disappear, and overlaps cancel everywhere.
    - Concrete cell trace at a leaked cell:
      t1=bg, t2=green.
      original 4-conn overwrite -> green because the cell is outside.
      fused combo -> bg because 8-conn relabels it as inside and inside only accepts
      exclusive tile1 pixels.
    - Another cell:
      t1=red, t2=green on an inside cell.
      original overwrite -> red.
      fused combo -> bg because overlap cancels under XOR.

  complexity_calibration:
    - If generation used only blocking row/column walls, D9 would be hard to observe.
      Therefore the generator alternates between blocking and leaky wall families.
    - If generation allowed almost no overlap, D6 would be visually weak.
      Therefore the generator enforces overlap plus both tile1-only and tile2-only mass.
    - If generation allowed only "correct-side" exclusive pixels, the task could look
      like a simpler masked copy. Therefore the generator also enforces wrong-side
      exclusive pixels so region routing is visibly necessary.

  implicit_assumption_audit:
    - Border parsing, tile extraction, indicator detection, symmetry checks, and wall
      side-touch checks all remain valid because the cue grammar is unchanged.
    - The original `assert outside` is no longer valid: under 8-connectivity, thin
      diagonal walls can leak and `outside` may be empty.
    - The original overwrite compositor assumed overlap could be resolved by priority.
      That is invalid now: overlap must cancel to background.
    - A naive global XOR would make D9 inert, because changing connectivity would not
      affect output. The fused implementation therefore uses XOR as an occupancy filter
      inside the original spatial routing framework.

  placement_legality:
    - All reused wall shapes stay in-bounds, pass through center, and touch >=2 sides.
    - Indicator remains a legal corner or edge-midpoint and is 8-isolated from the wall.
    - Pattern pixels never leave their 5x5 tiles.
    - The generator validates each constructed input against `main()`.

  visual_observability:
    - Leaky examples show the 8-connectivity change because thin diagonal walls stop
      separating the tile into two painted halves.
    - Overlaps disappearing creates visible holes compared with the dense original
      overwrite behavior.
    - Wrong-side exclusive pixels show that output is not just global XOR: the wall still
      matters because exclusives on the wrong side are suppressed.

  visual_cleanliness:
    - Output uses only tile_bg plus the two source colors.
    - Cancellation reduces clutter rather than adding new symbols.
    - The cue tile stays simple: one wall and one isolated marker.

  color_inference:
    - No numeric color remapping is used.
    - A surviving pixel keeps its original source color, but only if it is exclusive
      and allowed by the 8-connected region routing.
"""


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
GEN_RETRIES = 500

MIN_OVERLAP = 3
MIN_T1_ONLY = 2
MIN_T2_ONLY = 2
MIN_DIFF_FROM_OVERWRITE8 = 4
MIN_DIFF_FROM_COMBO4 = 2
MIN_DIFF_FROM_GLOBAL_XOR = 2

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

_BLOCKING_CONFIGS = _STRAIGHT_CONFIGS[:4] + _LBEND_CONFIGS[:8]
_LEAKY_CONFIGS = _STRAIGHT_CONFIGS[4:] + _LBEND_CONFIGS[8:]


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


def _compose_global_xor(t1: np.ndarray, t2: np.ndarray, bg: int) -> np.ndarray:
    out3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1, v2 = int(t1[r, c]), int(t2[r, c])
            has1 = v1 != bg
            has2 = v2 != bg
            if has1 ^ has2:
                out3[r, c] = v1 if has1 else v2
    return out3


def _compose_combo(
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
            has1 = v1 != bg
            has2 = v2 != bg
            if not (has1 ^ has2):
                continue
            if (r, c) in inside:
                if has1:
                    out3[r, c] = v1
            elif (r, c) in wall:
                if has1:
                    out3[r, c] = v1
                elif wall_4conn and has2:
                    out3[r, c] = v2
            else:
                if has2:
                    out3[r, c] = v2
    return out3


def _mask_from_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _choose_family() -> str:
    global _GEN_COUNTER
    family = "leaky" if (_GEN_COUNTER % 2 == 0) else "blocking"
    _GEN_COUNTER += 1
    return family


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite the two source tiles into the fourth panel under the combo rule.

    Input structure:
    - Four 5x5 panels appear in a single row with 1-cell border separators.
    - Panel 0 contains a 5-pixel wall through the center plus one 8-isolated indicator.
    - Panels 1 and 2 are the aligned source patterns; panel 3 is blank input.

    Rule:
    - Flood-fill from the indicator using 8-connectivity, blocked by the wall, to obtain
      the inside region. Under thin diagonal walls, outside may be empty.
    - A cell can paint only if exactly one of tile1 and tile2 is non-background there.
    - Inside accepts tile1-only pixels; outside accepts tile2-only pixels.
    - Wall cells keep original routing: tile1-only always survives, and tile2-only survives
      there only when the wall is 4-connected.

    Key insight:
    - The cue wall still determines which source is allowed to contribute, but overlap is
      canceled first by XOR. Using 8-connected flood-fill makes diagonal walls leak, so the
      same cue can reclassify would-be outside cells as inside and suppress tile2 there.
    """
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, _t3 = tiles
    bg = int(Counter(t1.flatten()).most_common(1)[0][0])
    assert border != bg, "border color must differ from tile background"

    t0_bg = int(Counter(t0.flatten()).most_common(1)[0][0])
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

    inside = _flood_inside(wall, indicator, connectivity=8)
    assert indicator in inside, "indicator must lie in the filled region"

    wall_4conn = _wall_is_4_connected(frozenset(wall))
    out3 = _compose_combo(t1, t2, bg, inside, wall, wall_4conn)

    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _gen_attempt(rng: random.Random, family: str) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color = rng.sample(other, 3)
    border = rng.choice(other)

    pool = _LEAKY_CONFIGS if family == "leaky" else _BLOCKING_CONFIGS
    indicator, wall = rng.choice(pool)
    wall_4conn = _wall_is_4_connected(wall)

    inside8 = _flood_inside(wall, indicator, connectivity=8)
    inside4 = _flood_inside(wall, indicator, connectivity=4)
    wall_set = set(wall)
    outside8 = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside8 - wall_set
    outside4 = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside4 - wall_set

    if family == "leaky":
        if outside8 or not outside4:
            return None
    else:
        if not outside8:
            return None

    t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    occ1 = t1 != tile_bg
    occ2 = t2 != tile_bg
    overlap = occ1 & occ2
    t1_only = occ1 & ~occ2
    t2_only = occ2 & ~occ1

    if int(overlap.sum()) < MIN_OVERLAP:
        return None
    if int(t1_only.sum()) < MIN_T1_ONLY or int(t2_only.sum()) < MIN_T2_ONLY:
        return None

    inside8_mask = _mask_from_coords(inside8)
    outside8_mask = _mask_from_coords(outside8)
    inside4_mask = _mask_from_coords(inside4)
    wall_mask = _mask_from_coords(wall_set)
    leak_band_mask = inside8_mask & ~inside4_mask & ~wall_mask
    wrong_side_mask = (t1_only & outside8_mask) | (t2_only & inside8_mask)

    out_combo8 = _compose_combo(t1, t2, tile_bg, inside8, wall_set, wall_4conn)
    out_combo4 = _compose_combo(t1, t2, tile_bg, inside4, wall_set, wall_4conn)
    out_overwrite8 = _compose_original(t1, t2, tile_bg, inside8, wall_set, wall_4conn)
    out_global_xor = _compose_global_xor(t1, t2, tile_bg)

    if int(np.sum(out_combo8 != out_overwrite8)) < MIN_DIFF_FROM_OVERWRITE8:
        return None
    if int(np.sum(out_combo8 != out_global_xor)) < MIN_DIFF_FROM_GLOBAL_XOR:
        return None

    if family == "leaky":
        if int(np.sum(out_combo8 != out_combo4)) < MIN_DIFF_FROM_COMBO4:
            return None
        if int((leak_band_mask & (t1_only | t2_only)).sum()) == 0:
            return None
    else:
        if int((t1_only & inside8_mask).sum()) == 0:
            return None
        if int((t2_only & outside8_mask).sum()) == 0:
            return None

    if int(wrong_side_mask.sum()) == 0:
        return None
    if int(np.sum(out_combo8 != tile_bg)) == 0:
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
    expected[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out_combo8
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
