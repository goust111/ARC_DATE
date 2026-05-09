# Combo variant of 7491f3cf (index 9)
# combo_spec: D3/2+D9/0
# Name: 不对称墙+8-连通flood-fill

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import flood_fill, in_bounds, label_mask, neighbors8

"""
Design analysis for the combo variant
=====================================

source_task:
  original_task: 7491f3cf
  combo_spec: D3/2 + D9/0

concretized_rule:
  unchanged_backbone:
    - The grid is still 7x25: four 5x5 tiles in one row, separated by 1-cell borders.
    - Tile 0 still uses one cue color for both the wall and the indicator.
    - Tile 1 and Tile 2 remain single-color symmetric source patterns.
    - Tile 3 input is blank and output is composed from Tile 1 / Tile 2.
  fused_change:
    - D3/2 removes wall symmetry and allows any asymmetric 8-connected wall through (2,2),
      touching at least two sides, with size restricted to 6-8 pixels so the difference from
      the original task is visible.
    - D9/0 changes the indicator flood-fill from 4-connectivity to 8-connectivity.
      Therefore some thin asymmetric walls that would split the board under 4-connectivity
      become leaky under 8-connectivity and may leave no outside region at all.

flow_trace:
  original_like_case:
    - indicator = (0,4)
    - symmetric wall = {(0,0),(1,1),(2,2),(3,3),(4,4)}
    - Under the old rule, 4-connected flood from the indicator stays on the upper-right side,
      so Tile 3 mixes Tile 1 and Tile 2 across the diagonal.
  combo_blocking_case:
    - indicator = (4,4)
    - asymmetric wall = {(0,2),(1,2),(2,2),(3,1),(4,0),(1,4)}
    - The wall is not mirror-symmetric. Under 8-connectivity it still blocks enough motion
      that inside and outside are both non-empty, but the split is jagged and cannot be inferred
      by a simple axis argument.
  combo_leaky_case:
    - indicator = (0,4)
    - asymmetric wall = {(0,0),(1,1),(2,2),(3,1),(4,0),(1,3)}
    - Under 4-connectivity, the diagonal-ish chain appears to split the tile.
    - Under 8-connectivity, flood can slip through corner gaps around the thin wall, so the
      reachable region expands and may cover every non-wall cell. Then Tile 3 becomes almost
      entirely Tile 1 outside the wall itself.
  observable_difference:
    - D3/2 forces the solver to trace geometry instead of exploiting symmetry.
    - D9/0 forces the solver to use 8-connected reachability instead of assuming every thin
      edge-to-edge wall creates a true partition.

complexity_calibration:
  wall_family_control:
    - Walls are generated from 2-3 short center-to-edge arms plus optional attached cells.
      This keeps the space varied enough to look different from the base task without becoming
      visually noisy or combinatorially unstructured.
    - Wall size is kept in 6-8 cells. At 5 cells many samples still look too close to the
      original straight/L families; above 8 cells Tile 0 becomes cluttered on a 5x5 board.
  behavior_control:
    - Generator alternates between two behavior families:
      * blocking: outside under 8-connectivity is non-empty
      * leaky: outside under 8-connectivity is empty but outside under 4-connectivity was non-empty
    - This gives repeated support for both aspects of the combo in a short training set.

implicit_assumption_audit:
  main_line_by_line:
    - Indicator detection via "unique 8-isolated cue pixel" still works, but only if the wall
      stays a single 8-connected component and is kept away from all 8-neighbors of the indicator.
      The generator enforces both.
    - The original symmetry assertion is invalid and is removed.
    - The original `assert outside` is invalid because 8-connected flood may consume all non-wall
      cells for thin leaky walls. Main now allows empty outside.
    - The wall-cell blending rule still depends on wall 4-connectivity rather than fill
      connectivity. This assumption survives the combo and is preserved.
    - Because the wall is now arbitrary, main adds an explicit 8-connectivity assertion on wall.
  new_combo_conflicts_checked:
    - Asymmetry plus 8-connectivity could accidentally create walls that neither partition under 4
      nor show any leak signal. Such walls are rejected in the pool builder.
    - A wall could be asymmetric but visually too close to the original if it is symmetric by
      accident. We explicitly reject symmetric walls.

placement_legality:
  guarantees:
    - Indicator is always one of the original eight legal border positions.
    - Wall cells are in-bounds, include the center, touch at least two sides, and never occupy
      the indicator or any of its 8-neighbors.
    - Tile 0 uses one cue color only; Tile 1 / Tile 2 patterns are independent and confined to
      their own 5x5 tiles; Tile 3 input stays blank.
    - Output-first validation is retained: every generated input is run through main and compared
      against the hand-built expected output.

visual_observability:
  thought_experiment:
    - Across 2-4 examples, a human sees that Tile 0 no longer contains neat mirrored barriers.
      Some samples still split Tile 3 into two source regions, while other samples with thin
      diagonal-ish asymmetric walls suddenly let Tile 1 flood almost everywhere.
    - The cue color only marks geometry; no hidden ordering of color ids is used.
    - The structural difference is visible from shape and reachability, not from numeric color values.

cleanliness_and_inferability:
  - Single-color cues keep wall and indicator legible.
  - Source patterns remain sparse symmetric blobs so the composed partition is readable.
  - The generator additionally enforces visible support:
    * Tile 1 must contribute somewhere inside the 8-connected region.
    * If outside exists under 8-connectivity, Tile 2 must contribute there.
    * For leaky samples, the 8-connected output must differ from the old 4-connected output.
"""


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
GEN_RETRIES = 300

WALL_MIN_PX, WALL_MAX_PX = 6, 8
WALL_POOL_ATTEMPTS = 12000
WALL_POOL_SEED = 0
BLOCKING_POOL_TARGET = 160
LEAKY_POOL_TARGET = 96

VALID_INDICATORS = {
    (0, 0),
    (0, 2),
    (0, 4),
    (2, 0),
    (2, 4),
    (4, 0),
    (4, 2),
    (4, 4),
}
SIDES = ("top", "bottom", "left", "right")

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


def _indicator_reflection(indicator: tuple[int, int]):
    ir, ic = indicator
    dr, dc = 2 - ir, 2 - ic
    if dr == 0:
        return lambda r, c: (4 - r, c)
    if dc == 0:
        return lambda r, c: (r, 4 - c)
    if dr * dc > 0:
        return lambda r, c: (c, r)
    return lambda r, c: (4 - c, 4 - r)


def _is_symmetric_wall(
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
    indicator: tuple[int, int],
) -> bool:
    reflect = _indicator_reflection(indicator)
    return set(wall) == {reflect(r, c) for r, c in wall}


def _wall_sides(wall: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> set[str]:
    sides: set[str] = set()
    for r, c in wall:
        if r == 0:
            sides.add("top")
        if r == TILE_SZ - 1:
            sides.add("bottom")
        if c == 0:
            sides.add("left")
        if c == TILE_SZ - 1:
            sides.add("right")
    return sides


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
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
    indicator: tuple[int, int],
    *,
    connectivity: int,
) -> set[tuple[int, int]]:
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=connectivity)
    return {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}


def _wall_4_connected(wall: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        mask[r, c] = True
    _, n_comp = label_mask(mask, connectivity=4)
    return n_comp == 1


def _wall_8_connected(wall: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        mask[r, c] = True
    _, n_comp = label_mask(mask, connectivity=8)
    return n_comp == 1


def _edge_cells_for_side(side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(TILE_SZ)]
    if side == "bottom":
        return [(TILE_SZ - 1, c) for c in range(TILE_SZ)]
    if side == "left":
        return [(r, 0) for r in range(TILE_SZ)]
    if side == "right":
        return [(r, TILE_SZ - 1) for r in range(TILE_SZ)]
    raise ValueError(f"unknown side: {side}")


def _random_shortest_path(
    start: tuple[int, int],
    target: tuple[int, int],
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    path = [start]
    cur = start
    while cur != target:
        cr, cc = cur
        dist = max(abs(target[0] - cr), abs(target[1] - cc))
        options: list[tuple[int, int]] = []
        for nr, nc in neighbors8(cr, cc):
            if not in_bounds((TILE_SZ, TILE_SZ), nr, nc):
                continue
            if (nr, nc) in forbidden and (nr, nc) != target:
                continue
            if max(abs(target[0] - nr), abs(target[1] - nc)) == dist - 1:
                options.append((nr, nc))
        if not options:
            return None
        cur = rng.choice(options)
        path.append(cur)
    return path


def _wall_frontier(wall: set[tuple[int, int]], forbidden: set[tuple[int, int]]) -> list[tuple[int, int]]:
    frontier: set[tuple[int, int]] = set()
    for r, c in wall:
        for nr, nc in neighbors8(r, c):
            if not in_bounds((TILE_SZ, TILE_SZ), nr, nc):
                continue
            if (nr, nc) in wall or (nr, nc) in forbidden:
                continue
            frontier.add((nr, nc))
    return sorted(frontier)


def _sample_wall_candidate(rng: random.Random) -> tuple[tuple[int, int], frozenset[tuple[int, int]]] | None:
    indicator = rng.choice(sorted(VALID_INDICATORS))
    forbidden = {indicator}
    for nr, nc in neighbors8(*indicator):
        if in_bounds((TILE_SZ, TILE_SZ), nr, nc):
            forbidden.add((nr, nc))

    target_size = rng.randint(WALL_MIN_PX, WALL_MAX_PX)
    available_by_side = {
        side: [cell for cell in _edge_cells_for_side(side) if cell not in forbidden]
        for side in SIDES
    }
    usable_sides = [side for side in SIDES if available_by_side[side]]
    if len(usable_sides) < 2:
        return None

    arm_count = 3 if target_size >= 7 and rng.random() < 0.7 else 2
    arm_sides = rng.sample(usable_sides, k=min(arm_count, len(usable_sides)))

    wall: set[tuple[int, int]] = {(2, 2)}
    for side in arm_sides:
        anchor = rng.choice(available_by_side[side])
        path = _random_shortest_path((2, 2), anchor, forbidden, rng)
        if path is None:
            return None
        wall.update(path)

    while len(wall) < target_size:
        frontier = _wall_frontier(wall, forbidden)
        if not frontier:
            return None
        edge_frontier = [cell for cell in frontier if 0 in cell or TILE_SZ - 1 in cell]
        pool = edge_frontier if edge_frontier and rng.random() < 0.55 else frontier
        wall.add(rng.choice(pool))

    return indicator, frozenset(wall)


def _classify_wall(
    indicator: tuple[int, int],
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
) -> str | None:
    if len(wall) < WALL_MIN_PX or len(wall) > WALL_MAX_PX:
        return None
    if indicator in wall or (2, 2) not in wall:
        return None
    if _is_symmetric_wall(wall, indicator):
        return None
    if len(_wall_sides(wall)) < 2:
        return None
    if not _wall_8_connected(wall):
        return None
    for nr, nc in neighbors8(*indicator):
        if (nr, nc) in wall:
            return None

    inside4 = _flood_inside(wall, indicator, connectivity=4)
    outside4 = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside4 - set(wall)
    if not inside4 or not outside4:
        return None

    inside8 = _flood_inside(wall, indicator, connectivity=8)
    outside8 = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside8 - set(wall)
    if not inside8:
        return None
    if not outside8:
        return "leaky" if inside8 != inside4 else None
    return "blocking"


def _build_wall_pools() -> tuple[
    list[tuple[tuple[int, int], frozenset[tuple[int, int]]]],
    list[tuple[tuple[int, int], frozenset[tuple[int, int]]]],
]:
    rng = random.Random(WALL_POOL_SEED)
    blocking_seen: set[tuple[tuple[int, int], frozenset[tuple[int, int]]]] = set()
    leaky_seen: set[tuple[tuple[int, int], frozenset[tuple[int, int]]]] = set()

    for _ in range(WALL_POOL_ATTEMPTS):
        candidate = _sample_wall_candidate(rng)
        if candidate is None:
            continue
        indicator, wall = candidate
        family = _classify_wall(indicator, wall)
        if family == "blocking" and len(blocking_seen) < BLOCKING_POOL_TARGET:
            blocking_seen.add((indicator, wall))
        elif family == "leaky" and len(leaky_seen) < LEAKY_POOL_TARGET:
            leaky_seen.add((indicator, wall))
        if len(blocking_seen) >= BLOCKING_POOL_TARGET and len(leaky_seen) >= LEAKY_POOL_TARGET:
            break

    min_blocking = max(48, BLOCKING_POOL_TARGET // 2)
    min_leaky = max(24, LEAKY_POOL_TARGET // 2)
    if len(blocking_seen) < min_blocking or len(leaky_seen) < min_leaky:
        raise RuntimeError(
            f"wall pools too small: blocking={len(blocking_seen)}, leaky={len(leaky_seen)}"
        )

    blocking = sorted(blocking_seen, key=lambda item: (item[0], sorted(item[1])))
    leaky = sorted(leaky_seen, key=lambda item: (item[0], sorted(item[1])))
    return blocking, leaky


BLOCKING_WALLS, LEAKY_WALLS = _build_wall_pools()


def _compose_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
    inside: set[tuple[int, int]],
) -> np.ndarray:
    wall_4conn = _wall_4_connected(wall)
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
    Composite two pattern tiles into the fourth panel using an asymmetric wall cue.

    Input structure:
    - Four 5x5 panels in one row, separated by 1-cell borders inside a 7x25 grid.
    - Panel 0 contains one cue color only: an asymmetric 8-connected wall plus one
      8-isolated indicator pixel on a corner or edge midpoint.
    - Panels 1 and 2 are source patterns; panel 3 is blank in the input.

    Rule:
    - Recover the wall as all cue pixels except the unique 8-isolated indicator.
    - Flood-fill from the indicator using 8-connectivity, blocked by the wall, to obtain
      the inside region.
    - Inside cells copy non-background pixels from panel 1; non-wall cells outside inside
      copy non-background pixels from panel 2.
    - Wall cells prefer panel 1 and use panel 2 only as fallback when the wall itself is
      4-connected.

    Key insight:
    - The wall is no longer symmetric, so the split cannot be inferred from mirror geometry,
      and 8-connected flood-fill can cross thin diagonal gaps that would block 4-connected fill.
      The correct partition comes only from tracing 8-connected reachability around the irregular wall.
    """
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, _t3 = tiles
    bg = int(Counter(t1.flatten()).most_common(1)[0][0])
    assert border != bg, "border color must differ from tile background"

    t0_bg = int(Counter(t0.flatten()).most_common(1)[0][0])
    cue_set = {
        (int(r), int(c))
        for r in range(TILE_SZ)
        for c in range(TILE_SZ)
        if int(t0[r, c]) != t0_bg
    }
    assert cue_set, "no cue pixels found"

    indicator = None
    for r, c in cue_set:
        if not any((nr, nc) in cue_set for nr, nc in neighbors8(r, c)):
            assert indicator is None, "multiple 8-isolated cue pixels"
            indicator = (r, c)
    assert indicator is not None, "no 8-isolated cue pixel found"
    assert indicator in VALID_INDICATORS, f"indicator {indicator} not at corner/edge-midpoint"

    wall = cue_set - {indicator}
    assert wall, "wall missing"
    assert (2, 2) in wall, "wall must pass through the grid center (2,2)"
    assert _wall_8_connected(wall), "wall must be 8-connected"

    sides_touched = _wall_sides(wall)
    assert len(sides_touched) >= 2, f"wall must span at least two sides, only touches {sides_touched}"

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

    pool = LEAKY_WALLS if family == "leaky" else BLOCKING_WALLS
    indicator, wall = rng.choice(pool)

    inside8 = _flood_inside(wall, indicator, connectivity=8)
    outside8 = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside8 - set(wall)
    inside4 = _flood_inside(wall, indicator, connectivity=4)
    outside4 = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside4 - set(wall)

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
