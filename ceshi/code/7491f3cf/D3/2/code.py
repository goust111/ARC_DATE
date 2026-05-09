# Variant of 7491f3cf
# Dimension: D3 — 墙线形状类型 (match_cond)
# Change: 直线（行/列/对角线/反对角线）或L弯，经过中心(2,2)，关于指示像素到中心连线对称... → 不对称墙

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import flood_fill, in_bounds, label_mask, neighbors8

"""
Design analysis for the D3 variant
==================================

source_task:
  original_task: 7491f3cf
  source_task: None

variant_spec:
  keep_fixed:
    - Grid is still 7x25 with four 5x5 tiles separated by 1-cell border bars.
    - Tile 0 still contains one cue color only: wall pixels + one isolated indicator pixel.
    - Tile 1 / Tile 2 stay as single-color symmetric source patterns.
    - Tile 3 input is blank and output is still formed by flood-fill partitioning.
  core_change:
    - Old D3: the wall had to be one of a few symmetric 5-pixel shapes.
    - New D3: the wall may be any 8-connected asymmetric shape, as long as it:
      * passes through center (2,2)
      * touches at least two sides of the 5x5 tile
      * stays 8-separated from the indicator pixel
      * splits the tile into non-empty inside/outside regions under 4-connected flood-fill
  concrete_visual_form:
    - The wall is now usually 6-8 pixels rather than exactly 5.
    - It is built from 2-3 short arms leaving the center plus optional attached bump cells.
    - This creates hooks, off-center forks, and uneven barriers; the inside region is often
      a jagged blob instead of a clean half-plane or L-corner.

flow_trace:
  old_case:
    - Example indicator = (4,4), wall = anti-diagonal {(0,4),(1,3),(2,2),(3,1),(4,0)}.
    - Because the wall is symmetric, a solver could mentally infer the opposite side by axis
      reasoning without tracing every reachable pixel.
  new_case:
    - Example indicator = (4,4), wall = {(0,2),(1,2),(2,2),(3,2),(4,1),(1,4)}.
    - Flood-fill from (4,4) is blocked by both the vertical stem and the extra top-right tooth.
    - The reachable "inside" cells are no longer describable by a single straight split; the
      solver must actually run the 4-connected reachability process.
  difference:
    - The composition rule itself is unchanged, but the cue geometry now carries less symmetry
      and more local detail, so flood-fill is the only robust way to recover the partition.

complexity_calibration:
  why_not_keep_only_5_pixels:
    - Many 5-pixel walls still look close to the original straight/L families.
    - Allowing 6-8 pixels makes the D3 change visually obvious while staying readable on 5x5.
  why_not_allow_unbounded_shapes:
    - Too many wall pixels would clutter tile 0 and obscure the partition.
    - Limiting the wall to 6-8 cells keeps the cue legible and leaves both regions visible.
  branch_control:
    - Candidate walls are generated from a small family: 2-3 center-to-edge arms plus 0-2
      attached cells. This is more varied than the original but still structured enough that
      repeated examples support the rule.

implicit_assumption_audit:
  main_related_lines:
    - Indicator detection via "unique 8-isolated cue pixel" still holds because the wall is
      generated as a single 8-connected component, so only the indicator is isolated.
    - The old symmetry assertion is invalid in this variant and must be removed.
    - Flood-fill from the indicator remains correct because the rule change affects only wall
      geometry, not the reachability/compositing semantics.
    - The wall/tile blending rule depending on 4-connectivity is still valid and is preserved.
  new_assertions:
    - Since symmetry is gone, main now explicitly checks that the wall is 8-connected.
    - Generator guarantees this by only adding cells adjacent to the current wall.

placement_legality:
  wall_generation_guarantees:
    - All wall cells lie inside the 5x5 tile.
    - The indicator is chosen from the original eight legal border positions.
    - No wall cell may occupy the indicator or any of its 8-neighbors.
    - The wall must touch at least two tile sides and include the center.
    - After construction, a flood-fill check enforces non-empty inside and outside regions.

visual_observability:
  human-facing_signal:
    - Tile 0 visibly departs from the old "perfect mirror" cue shapes.
    - Tile 3 inherits irregular cutouts from that asymmetric wall, so the new behavior is
      observable both in the cue tile and in the composed output.
  color_reasoning:
    - Colors still function only as role markers: cue, tile1, tile2, background, border.
    - No ordering or hidden numeric meaning is introduced.

cleanliness_check:
  - Only one cue color is used in tile 0, so wall and indicator stay legible.
  - Tile 1 and Tile 2 remain sparse enough that the partitioned output is readable.
  - The wall stays small enough that it separates regions without filling most of the tile.
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
WALL_POOL_TARGET = 320
WALL_POOL_ATTEMPTS = 4000
WALL_POOL_SEED = 0

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


def _is_symmetric_wall(wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
                       indicator: tuple[int, int]) -> bool:
    refl = _indicator_reflection(indicator)
    return set(wall) == {refl(r, c) for r, c in wall}


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


def _flood_inside(wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
                  indicator: tuple[int, int]) -> set[tuple[int, int]]:
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
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


def _random_shortest_path(start: tuple[int, int], target: tuple[int, int],
                          forbidden: set[tuple[int, int]], rng: random.Random) -> list[tuple[int, int]] | None:
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
    arm_count = 3 if target_size >= 7 and rng.random() < 0.7 else 2
    if len(usable_sides) < 2:
        return None
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


def _is_valid_variant_wall(indicator: tuple[int, int],
                           wall: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    if len(wall) < WALL_MIN_PX or len(wall) > WALL_MAX_PX:
        return False
    if (2, 2) not in wall:
        return False
    if indicator in wall:
        return False
    if _is_symmetric_wall(wall, indicator):
        return False
    if len(_wall_sides(wall)) < 2:
        return False
    if not _wall_8_connected(wall):
        return False
    for nr, nc in neighbors8(*indicator):
        if (nr, nc) in wall:
            return False
    inside = _flood_inside(wall, indicator)
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - set(wall)
    return bool(inside) and bool(outside)


def _build_wall_pool() -> list[tuple[tuple[int, int], frozenset[tuple[int, int]]]]:
    rng = random.Random(WALL_POOL_SEED)
    seen: set[tuple[tuple[int, int], frozenset[tuple[int, int]]]] = set()
    for _ in range(WALL_POOL_ATTEMPTS):
        candidate = _sample_wall_candidate(rng)
        if candidate is None:
            continue
        indicator, wall = candidate
        if _is_valid_variant_wall(indicator, wall):
            seen.add((indicator, wall))
        if len(seen) >= WALL_POOL_TARGET:
            break
    if len(seen) < max(64, WALL_POOL_TARGET // 2):
        raise RuntimeError(f"wall pool too small: {len(seen)}")
    return sorted(seen, key=lambda item: (item[0], sorted(item[1])))


WALL_CONFIGS = _build_wall_pool()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two pattern tiles into a fourth panel using flood-fill from an indicator pixel.

    Variant D3 change:
    - The wall is no longer required to be symmetric.
    - It may be any 8-connected asymmetric barrier through the center touching >=2 sides.
    """
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j: 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, _t3 = tiles
    b = int(Counter(t1.flatten()).most_common(1)[0][0])
    assert border != b, "border color must differ from tile background"

    t0_bg = int(Counter(t0.flatten()).most_common(1)[0][0])
    cue_set = {(int(r), int(c)) for r in range(TILE_SZ) for c in range(TILE_SZ) if int(t0[r, c]) != t0_bg}
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

    inside = _flood_inside(wall, indicator)
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - wall
    assert inside, "inside region is empty"
    assert outside, "wall must partition the grid — outside region is empty"

    wall_4conn = _wall_4_connected(wall)

    out3 = np.full((TILE_SZ, TILE_SZ), b, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1, v2 = int(t1[r, c]), int(t2[r, c])
            if (r, c) in inside:
                if v1 != b:
                    out3[r, c] = v1
            elif (r, c) in wall:
                if v1 != b:
                    out3[r, c] = v1
                elif wall_4conn and v2 != b:
                    out3[r, c] = v2
            else:
                if v2 != b:
                    out3[r, c] = v2

    out = g.copy()
    out[1:6, 1 + 6 * 3: 1 + 6 * 3 + 5] = out3
    return out


def _sample_symmetric_pattern(bg: int, fg: int, rng: random.Random, *,
                              min_px: int, max_px: int) -> np.ndarray | None:
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


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color = rng.sample(other, 3)
    border = rng.choice(other)

    indicator, wall = rng.choice(WALL_CONFIGS)
    inside = _flood_inside(wall, indicator)

    t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = cue_color
    t0[indicator[0], indicator[1]] = cue_color

    wall_4conn = _wall_4_connected(wall)
    out3 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1, v2 = int(t1[r, c]), int(t2[r, c])
            if (r, c) in inside:
                if v1 != tile_bg:
                    out3[r, c] = v1
            elif (r, c) in wall:
                if v1 != tile_bg:
                    out3[r, c] = v1
                elif wall_4conn and v2 != tile_bg:
                    out3[r, c] = v2
            else:
                if v2 != tile_bg:
                    out3[r, c] = v2

    t3_in = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
    for j, tile in enumerate([t0, t1, t2, t3_in]):
        g[1:6, 1 + 6 * j: 1 + 6 * j + 5] = tile

    expected = g.copy()
    expected[1:6, 1 + 6 * 3: 1 + 6 * 3 + 5] = out3
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
