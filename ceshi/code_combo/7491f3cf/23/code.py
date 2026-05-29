# Combo variant of 7491f3cf (index 23)
# combo_spec: D3/2+D9/0+D10/2
# Name: 不对称墙+8-连通+墙上叠合

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import flood_fill, in_bounds, label_mask, neighbors8

"""
Design analysis
===============

a) Concrete behavior and flow trace
-----------------------------------
- Original task:
  1. Tile 0 contains a symmetric 5-pixel wall plus one isolated indicator.
  2. Flood-fill from the indicator uses 4-connectivity, so thin diagonals split the board.
  3. Tile 3 copies tile1 on the filled side, tile2 on the other side.
  4. Wall cells prefer tile1; tile2 is allowed on the wall only when the wall is 4-connected.
- Combo task:
  1. Tile 0 still contains one cue color only, but the wall is now an asymmetric 8-connected
     path through (2,2), usually 6-8 pixels and touching at least two sides.
  2. Flood-fill from the indicator now uses 8-connectivity, so some thin diagonal-ish walls
     leak completely while wider hooked walls still seal.
  3. Wall cells use OR-style blending: tile1 if present, else tile2 if present, else background.
- Example:
  indicator=(4,4), wall={(0,2),(1,2),(2,2),(3,2),(4,1),(1,4)}.
  Under the old rule, 4-neighbor flood leaves a separate outside region.
  Under the combo rule, 8-neighbor flood may still leave outside cells if the hook is wide
  enough, but a thinner diagonal hook would leak and make every non-wall cell "inside".
  On the wall itself, any tile2-only wall pixel is now copied instead of being suppressed
  merely because the wall is not 4-connected.

b) Complexity calibration
-------------------------
- The wall is limited to 6-8 cells. This is enough to look clearly different from the original
  straight/L grammar without cluttering a 5x5 panel.
- The generator alternates two families:
  - diag_sealed: asymmetric, not 4-connected, still partitions under 8-connectivity.
  - leaky: asymmetric walls that partition under 4-connectivity but not under 8-connectivity.
- This keeps the hypothesis space small enough for ARC while still forcing all three changes:
  D3 via asymmetry, D9 via leaky vs sealed 8-fill, D10 via visible two-color wall bands.

c) Implicit assumption audit for main()
--------------------------------------
- Unique 8-isolated cue pixel:
  still valid because the generator keeps the entire wall in one 8-connected component and
  keeps every wall cell out of the indicator's 8-neighborhood.
- Symmetry assertion:
  invalid after D3/2 and removed. The solver must not reflect the wall anymore.
- `assert outside`:
  invalid after D9/0 because 8-connected flood can legitimately consume every non-wall cell.
  The solver therefore allows `outside == empty`.
- Wall fallback based on 4-connectivity:
  invalid after D10/2. Wall composition now depends only on tile occupancy at that wall cell:
  tile1 first, then tile2, else background.
- New solver-side structural checks:
  the wall must still pass through the center, touch at least two sides, and be 8-connected.

d) Placement legality
---------------------
- Indicator positions stay in the original eight legal border/corner spots.
- Wall generation forbids the indicator and all eight adjacent cells, so the indicator remains
  uniquely isolated.
- Every wall stays inside the 5x5 tile, includes (2,2), touches at least two tile sides,
  and is one 8-connected component.
- Tiles 1 and 2 are still independent 5x5 single-color symmetric patterns, so no overlap or
  out-of-bounds issue is introduced by the fusion itself.

e) Visual observability
-----------------------
- D3 is visible directly in tile 0 because the wall is no longer mirror-structured.
- D9 is visible because the generator includes both leaky and sealed 8-connectivity cases.
  In leaky samples, tile3 becomes almost all tile1 away from the wall; in sealed samples,
  tile2 still occupies a genuine outside region.
- D10 is visible because the generator requires at least one wall pixel to output tile1's
  color and at least one different wall pixel to output tile2's color, producing a clear
  mixed "fusion band" on the wall.

f) Cleanliness and g) inferability
----------------------------------
- Border, cue, tile1, and tile2 colors are all distinct to avoid accidental visual confusion.
- Only two semantic regimes are shown repeatedly, so the rule is inferable from a few examples:
  "follow 8-neighbor reachability from the marker, then merge the two source tiles with the
  wall acting as a two-source blend band."
- No hidden numeric property of colors is used; all distinctions are geometric or occupancy-based.

Post-change solver assumptions and generator guarantees
-------------------------------------------------------
- The solver assumes exactly one isolated indicator and an 8-connected wall; the generator
  constructs inputs that satisfy both.
- The generator rejects samples unless the produced tile3 differs from the old 4-connected /
  old-wall-rule counterfactual. This prevents silent regression back to original behavior.
"""


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
PAIR_RETRIES = 80
GEN_RETRIES = 320

WALL_MIN_PX, WALL_MAX_PX = 6, 8
WALL_POOL_ATTEMPTS = 5000
WALL_POOL_SEED = 23
MIN_DIAG_SEALED = 24
MIN_LEAKY = 96

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
GEN_FAMILIES = ("diag_sealed", "leaky")
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


def _wall_frontier(
    wall: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
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
    wall: frozenset[tuple[int, int]],
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
    if outside8:
        return "orth_sealed" if _wall_4_connected(wall) else "diag_sealed"
    return "leaky"


def _build_wall_families() -> dict[str, list[tuple[tuple[int, int], frozenset[tuple[int, int]]]]]:
    rng = random.Random(WALL_POOL_SEED)
    families: dict[str, set[tuple[tuple[int, int], frozenset[tuple[int, int]]]]] = {
        "orth_sealed": set(),
        "diag_sealed": set(),
        "leaky": set(),
    }
    for _ in range(WALL_POOL_ATTEMPTS):
        candidate = _sample_wall_candidate(rng)
        if candidate is None:
            continue
        indicator, wall = candidate
        family = _classify_wall(indicator, wall)
        if family is None:
            continue
        families[family].add((indicator, wall))
        if len(families["diag_sealed"]) >= MIN_DIAG_SEALED and len(families["leaky"]) >= MIN_LEAKY:
            break

    if len(families["diag_sealed"]) < MIN_DIAG_SEALED or len(families["leaky"]) < MIN_LEAKY:
        raise RuntimeError(
            "wall pools too small: "
            f"diag_sealed={len(families['diag_sealed'])}, leaky={len(families['leaky'])}"
        )

    return {
        name: sorted(items, key=lambda item: (item[0], sorted(item[1])))
        for name, items in families.items()
    }


WALL_FAMILIES = _build_wall_families()


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
        for idx in order:
            orbit = _ORBITS[idx]
            if count + len(orbit) > max_px:
                continue
            if rng.random() < ORBIT_INCLUDE_PROB:
                for r, c in orbit:
                    p[r, c] = fg
                count += len(orbit)
        if min_px <= count <= max_px:
            return p
    return None


def _compose_combo_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    wall: frozenset[tuple[int, int]],
    inside: set[tuple[int, int]],
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
                elif v2 != bg:
                    out3[r, c] = v2
            else:
                if v2 != bg:
                    out3[r, c] = v2
    return out3


def _compose_original_counterfactual(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    wall: frozenset[tuple[int, int]],
    inside4: set[tuple[int, int]],
) -> np.ndarray:
    wall_4conn = _wall_4_connected(wall)
    out3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1, v2 = int(t1[r, c]), int(t2[r, c])
            if (r, c) in inside4:
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


def _region_has_fg(tile: np.ndarray, bg: int, coords: set[tuple[int, int]]) -> bool:
    return any(int(tile[r, c]) != bg for r, c in coords)


def _wall_outputs_both_colors(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    wall: frozenset[tuple[int, int]],
) -> bool:
    has_t1 = any(int(t1[r, c]) != bg for r, c in wall)
    has_t2_only = any(int(t1[r, c]) == bg and int(t2[r, c]) != bg for r, c in wall)
    return has_t1 and has_t2_only


def _wall_differs_from_original(
    combo: np.ndarray,
    counterfactual: np.ndarray,
    wall: frozenset[tuple[int, int]],
) -> bool:
    return any(int(combo[r, c]) != int(counterfactual[r, c]) for r, c in wall)


def _nonwall_differs_between_regions(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    wall: frozenset[tuple[int, int]],
    inside_a: set[tuple[int, int]],
    inside_b: set[tuple[int, int]],
) -> bool:
    out_a = _compose_combo_tile(t1, t2, bg, wall, inside_a)
    out_b = _compose_combo_tile(t1, t2, bg, wall, inside_b)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            if (r, c) in wall:
                continue
            if int(out_a[r, c]) != int(out_b[r, c]):
                return True
    return False


def _choose_family() -> str:
    global _GEN_COUNTER
    family = GEN_FAMILIES[_GEN_COUNTER % len(GEN_FAMILIES)]
    _GEN_COUNTER += 1
    return family


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two pattern tiles into the fourth panel by reading an asymmetric cue wall.

    Input structure:
    - Four 5x5 panels in one row, separated by a 1-cell border frame.
    - Panel 0 contains a single cue color encoding one 8-connected wall plus one isolated
      indicator pixel at a corner or edge midpoint.
    - Panels 1 and 2 are source patterns; panel 3 is blank in the input.

    Rule:
    - Treat the cue pixels except the isolated indicator as the wall.
    - Flood-fill from the indicator with 8-connectivity, blocked by the wall, to get `inside`.
    - Copy non-background pixels from panel 1 on `inside`, from panel 2 on `outside`,
      and on wall cells use OR-style overlay: panel 1 if present, else panel 2 if present.

    Key insight:
    - Symmetry is no longer reliable. The solver must explicitly recover an asymmetric
      8-connected barrier, run 8-neighbor reachability, and treat the wall itself as a
      mixed-source fusion band rather than a special-case fallback.
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
    assert indicator in VALID_INDICATORS, f"indicator {indicator} not at corner/edge-midpoint"

    wall = cue_set - {indicator}
    assert wall, "wall missing"
    assert (2, 2) in wall, "wall must pass through the grid center (2,2)"
    assert _wall_8_connected(wall), "wall must be 8-connected"

    sides_touched = _wall_sides(wall)
    assert len(sides_touched) >= 2, f"wall must touch at least two sides, only touches {sides_touched}"

    inside = _flood_inside(wall, indicator, connectivity=8)
    assert indicator in inside, "indicator must lie in the filled region"

    out3 = _compose_combo_tile(t1, t2, bg, wall, inside)
    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _gen_attempt(rng: random.Random, family: str) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color, border = rng.sample(other, 4)

    indicator, wall = rng.choice(WALL_FAMILIES[family])
    inside8 = _flood_inside(wall, indicator, connectivity=8)
    outside8 = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside8 - set(wall)
    inside4 = _flood_inside(wall, indicator, connectivity=4)
    outside4 = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside4 - set(wall)

    if family == "diag_sealed":
        if not outside8 or _wall_4_connected(wall):
            return None
    elif family == "leaky":
        if outside8 or not outside4:
            return None
    else:
        return None

    for _ in range(PAIR_RETRIES):
        t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
        t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
        if t1 is None or t2 is None:
            continue
        if not _region_has_fg(t1, tile_bg, inside8):
            continue
        if outside8 and not _region_has_fg(t2, tile_bg, outside8):
            continue
        if not _wall_outputs_both_colors(t1, t2, tile_bg, wall):
            continue

        combo_out = _compose_combo_tile(t1, t2, tile_bg, wall, inside8)
        counterfactual_out = _compose_original_counterfactual(t1, t2, tile_bg, wall, inside4)
        if np.array_equal(combo_out, counterfactual_out):
            continue

        if family == "diag_sealed":
            if not _wall_differs_from_original(combo_out, counterfactual_out, wall):
                continue
        else:
            if not _nonwall_differs_between_regions(t1, t2, tile_bg, wall, inside8, inside4):
                continue

        t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
        for r, c in wall:
            t0[r, c] = cue_color
        t0[indicator[0], indicator[1]] = cue_color

        t3_in = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
        g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
        for j, tile in enumerate([t0, t1, t2, t3_in]):
            g[1:6, 1 + 6 * j : 1 + 6 * j + 5] = tile

        expected = g.copy()
        expected[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = combo_out
        got = main(g)
        if np.array_equal(got, expected):
            return g
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    family = _choose_family()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng, family)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts for family={family}")
