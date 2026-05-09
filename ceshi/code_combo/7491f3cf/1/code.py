# Combo variant of 7491f3cf (index 1)
# combo_spec: D1/0+D9/0
# Name: 十字交叉分割+8-连通flood-fill

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import flood_fill, neighbors8


# Design analysis for the combo variant
# -------------------------------------
# combo_variant_spec:
#   overview:
#     - Keep the original 7x25 layout with four 5x5 tiles in one row.
#     - Replace the original single 5-pixel wall by a centered orthogonal cross:
#       full row 2 plus full col 2 (9 cue pixels total), plus one isolated indicator.
#     - Replace the original binary inside/outside composition by a four-way rule:
#       primary quadrant from tile1, diagonal quadrant from tile2, the other two
#       quadrants from a mixed rule, and the cross itself stays blank.
#     - Use 8-connected flood-fill from the indicator to identify the primary
#       quadrant, so the solver must still reason via reachability instead of
#       hard-coding the indicator corner.
#   concrete_trace:
#     - Example cue: indicator=(0,0), wall=row2 ∪ col2.
#     - Original task logic no longer applies because it assumes one 5-pixel
#       wall, one filled region, and one complementary outside region.
#     - In this combo:
#         8-conn flood from (0,0) reaches exactly {(0,0),(0,1),(1,0),(1,1)}.
#         Reflecting that 2x2 block through the center gives the diagonal
#         quadrant {(3,3),(3,4),(4,3),(4,4)}.
#         The remaining non-wall cells are the two mixed quadrants.
#       Output rule:
#         primary quadrant  -> copy tile1 non-bg
#         diagonal quadrant -> copy tile2 non-bg
#         mixed quadrants   -> union(tile1,tile2), with tile1 priority only as a
#                              malformed-input fallback
#         cross wall        -> keep tile background
#     - Difference from the original:
#         The non-primary area is no longer uniform. Two side quadrants mix both
#         sources, while the opposite corner is reserved for tile2 only.
#   complexity_calibration:
#     - Keep the cross shape fixed and centered. The only cue category that
#       changes across examples is which corner hosts the indicator. This gives
#       four repeated support cases instead of an explosion of wall grammars.
#     - Fix the mixed rule to a single option: union of tile1 and tile2 in the
#       two adjacent quadrants. This is richer than a binary split but still
#       easy to compare across examples.
#     - Generator guarantees the mixed area visibly uses both source colors and
#       differs from both degenerate baselines:
#         1. "mixed quadrants are tile1 only"
#         2. "mixed quadrants are tile2 only"
#       That prevents accidental collapse back to a simpler two-region task.
#   implicit_assumptions_audit:
#     - Original `wall` assumptions that break:
#         * size 5
#         * one symmetric 8-connected path
#         * edge-midpoint indicators being legal
#         * existence of a single complementary `outside` region
#         * wall cells participating in source blending
#     - New solver assumptions:
#         * exactly one 8-isolated indicator
#         * indicator is at a corner
#         * wall is exactly the centered row+column cross
#         * 8-conn flood from the indicator yields one 2x2 corner quadrant
#         * reflecting that quadrant through the center yields the diagonal one
#     - Generator enforces all of those conditions directly.
#   placement_legality_and_visual_observability:
#     - The cross uses 9 cells, the indicator is corner-isolated, so cue objects
#       never overlap and always stay in bounds.
#     - Tile3 input remains blank.
#     - Keeping the cross blank in the output preserves clean separators between
#       quadrants; the human can visually parse four regions instead of reading
#       through overdraw on the boundary.
#     - Source colors remain single-color and distinct, so the mixed quadrants
#       visibly display a two-color union. No rule depends on numeric color ids.
#   new_solver_assumptions_and_generator_guarantees:
#     - The generator retries until:
#         * tile1 contributes in the primary quadrant
#         * tile2 contributes in the diagonal quadrant
#         * both tile1-only and tile2-only cells appear in the mixed quadrants
#         * tile1/tile2 do not overlap inside the mixed quadrants, so the union
#           is directly visible and not hidden behind precedence
#     - Therefore the combo behavior is observable in 2-4 examples and remains
#       distinct from the original binary flood-fill task.


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 300
GEN_RETRIES = 500

_GEN_COUNTER = 0

_MID = TILE_SZ // 2
_CROSS_WALL = frozenset(
    {(r, _MID) for r in range(TILE_SZ)} | {(_MID, c) for c in range(TILE_SZ)}
)
_CORNER_INDICATORS = ((0, 0), (0, 4), (4, 0), (4, 4))

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

_QUADRANTS = {
    "NW": frozenset((r, c) for r in range(0, _MID) for c in range(0, _MID)),
    "NE": frozenset((r, c) for r in range(0, _MID) for c in range(_MID + 1, TILE_SZ)),
    "SW": frozenset((r, c) for r in range(_MID + 1, TILE_SZ) for c in range(0, _MID)),
    "SE": frozenset((r, c) for r in range(_MID + 1, TILE_SZ) for c in range(_MID + 1, TILE_SZ)),
}
_INDICATOR_TO_LABEL = {
    (0, 0): "NW",
    (0, 4): "NE",
    (4, 0): "SW",
    (4, 4): "SE",
}
_OPPOSITE = {"NW": "SE", "NE": "SW", "SW": "NE", "SE": "NW"}


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


def _flood_region(
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


def _extract_regions(
    wall: frozenset[tuple[int, int]],
    indicator: tuple[int, int],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    primary = _flood_region(wall, indicator, connectivity=8)
    assert indicator in primary, "indicator must lie in the filled region"
    assert len(primary) == 4, f"primary quadrant must be 2x2, got size={len(primary)}"

    diagonal = {(TILE_SZ - 1 - r, TILE_SZ - 1 - c) for r, c in primary}
    assert len(diagonal) == 4, "diagonal quadrant must also be 2x2"

    non_wall = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - set(wall)
    mixed = non_wall - primary - diagonal
    assert len(mixed) == 8, f"mixed region must contain the other two quadrants, got size={len(mixed)}"
    return primary, diagonal, mixed


def _compose_combo_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    primary: set[tuple[int, int]],
    diagonal: set[tuple[int, int]],
    mixed: set[tuple[int, int]],
) -> np.ndarray:
    out3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r, c in primary:
        v1 = int(t1[r, c])
        if v1 != bg:
            out3[r, c] = v1
    for r, c in diagonal:
        v2 = int(t2[r, c])
        if v2 != bg:
            out3[r, c] = v2
    for r, c in mixed:
        v1, v2 = int(t1[r, c]), int(t2[r, c])
        if v1 != bg:
            out3[r, c] = v1
        elif v2 != bg:
            out3[r, c] = v2
    return out3


def _compose_t1_only_mixed(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    primary: set[tuple[int, int]],
    diagonal: set[tuple[int, int]],
    mixed: set[tuple[int, int]],
) -> np.ndarray:
    out3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for region in (primary, mixed):
        for r, c in region:
            v1 = int(t1[r, c])
            if v1 != bg:
                out3[r, c] = v1
    for r, c in diagonal:
        v2 = int(t2[r, c])
        if v2 != bg:
            out3[r, c] = v2
    return out3


def _compose_t2_only_mixed(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    primary: set[tuple[int, int]],
    diagonal: set[tuple[int, int]],
    mixed: set[tuple[int, int]],
) -> np.ndarray:
    out3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r, c in primary:
        v1 = int(t1[r, c])
        if v1 != bg:
            out3[r, c] = v1
    for region in (diagonal, mixed):
        for r, c in region:
            v2 = int(t2[r, c])
            if v2 != bg:
                out3[r, c] = v2
    return out3


def _assemble_grid(border: int, tiles: list[np.ndarray]) -> np.ndarray:
    g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
    for j, tile in enumerate(tiles):
        g[1:6, 1 + 6 * j : 1 + 6 * j + 5] = tile
    return g


def _count_non_bg(tile: np.ndarray, bg: int, coords: set[tuple[int, int]]) -> int:
    return sum(int(tile[r, c]) != bg for r, c in coords)


def _has_mixed_visibility(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    mixed_labels: tuple[str, str],
) -> bool:
    t1_only_total = 0
    t2_only_total = 0
    for label in mixed_labels:
        coords = _QUADRANTS[label]
        t1_only = 0
        t2_only = 0
        overlap = 0
        for r, c in coords:
            a = int(t1[r, c]) != bg
            b = int(t2[r, c]) != bg
            if a and b:
                overlap += 1
            elif a:
                t1_only += 1
            elif b:
                t2_only += 1
        if overlap:
            return False
        if t1_only == 0 or t2_only == 0:
            return False
        t1_only_total += t1_only
        t2_only_total += t2_only
    return t1_only_total > 0 and t2_only_total > 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two pattern tiles into a fourth panel using a cross-shaped partition.

    Input structure:
    - Four 5×5 panels in one row, border-separated.
    - Panel 0: cue pixels = one centered cross wall (full row + full column)
               plus one 8-isolated corner indicator.
    - Panels 1–2: source patterns. Panel 3: blank (to fill).

    Rule:
    - Run 8-connected flood-fill from the indicator, blocked by the cross wall.
    - The reached quadrant is the primary quadrant and copies panel 1 non-bg.
    - The diagonally opposite quadrant copies panel 2 non-bg.
    - The other two quadrants use the union of panels 1 and 2.
    - The cross wall itself stays background in the output.

    Key insight:
    - With 8-connectivity, thin diagonal walls would leak, so the cue must be an
      orthogonal cross. The indicator marks one of four quadrants, not one side
      of a binary partition, and the two adjacent quadrants share a mixed rule.
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
    assert indicator in _CORNER_INDICATORS, f"indicator {indicator} must be at a corner"

    wall = cue_set - {indicator}
    assert len(wall) == 9, f"cross wall must contain 9 pixels, got {len(wall)}"
    assert wall == _CROSS_WALL, "wall must be the centered orthogonal cross"
    assert (_MID, _MID) in wall, "cross must pass through the center"

    primary, diagonal, mixed = _extract_regions(wall, indicator)
    out3 = _compose_combo_tile(t1, t2, bg, primary, diagonal, mixed)

    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _choose_indicator() -> tuple[int, int]:
    global _GEN_COUNTER
    indicator = _CORNER_INDICATORS[_GEN_COUNTER % len(_CORNER_INDICATORS)]
    _GEN_COUNTER += 1
    return indicator


def _gen_attempt(rng: random.Random, indicator: tuple[int, int]) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color = rng.sample(other, 3)
    border = rng.choice(other)

    wall = _CROSS_WALL
    primary, diagonal, mixed = _extract_regions(wall, indicator)

    label = _INDICATOR_TO_LABEL[indicator]
    mixed_labels = tuple(sorted(set(_QUADRANTS) - {label, _OPPOSITE[label]}))

    t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    if _count_non_bg(t1, tile_bg, primary) == 0:
        return None
    if _count_non_bg(t2, tile_bg, diagonal) == 0:
        return None
    if not _has_mixed_visibility(t1, t2, tile_bg, mixed_labels):
        return None

    out3 = _compose_combo_tile(t1, t2, tile_bg, primary, diagonal, mixed)
    out_t1_mixed = _compose_t1_only_mixed(t1, t2, tile_bg, primary, diagonal, mixed)
    out_t2_mixed = _compose_t2_only_mixed(t1, t2, tile_bg, primary, diagonal, mixed)
    if np.array_equal(out3, out_t1_mixed) or np.array_equal(out3, out_t2_mixed):
        return None

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = cue_color
    t0[indicator[0], indicator[1]] = cue_color

    t3_in = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    g = _assemble_grid(border, [t0, t1, t2, t3_in])

    expected = g.copy()
    expected[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    got = main(g)
    if not np.array_equal(got, expected):
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    indicator = _choose_indicator()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng, indicator)
        if g is not None:
            return g
    raise RuntimeError(
        f"generate_input failed after {GEN_RETRIES} attempts for indicator={indicator}"
    )
