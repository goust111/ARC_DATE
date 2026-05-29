# Combo variant of 7491f3cf (index 16)
# combo_spec: D8/0+D16/2
# Name: 2x2矩阵布局+输出面板在中间
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior:
#    Original: 4 panels are laid out in a single row and roles are hard-coded as
#    cue, tile1, tile2, output. Fused variant: panels form a 2x2 matrix inside a
#    13x13 grid. The cue panel can occupy any corner slot. The blank output panel
#    is not diagonal from the cue; it is one of the cue's two orthogonal
#    neighbors, so it behaves like a "middle" panel in 2D adjacency terms. The
#    remaining orthogonal neighbor of the cue is tile1 and the diagonal panel is
#    tile2. Example: if cue is at bottom-left and blank output is at top-left,
#    then bottom-right is tile1 and top-right is tile2.
# b) Complexity calibration:
#    The layout change is visually obvious because the whole grid becomes 13x13
#    with a cross-shaped border. The role change is also visible because the
#    blank panel can appear in different slots, but source-role recovery still
#    uses only one small repeated rule: relative slot adjacency to the cue.
# c) Broken implicit assumptions from the original solver:
#    - Shape is no longer 7x25.
#    - Panels are no longer extracted by a single horizontal stride.
#    - Panel index no longer determines role.
#    - The output target is no longer "last panel".
#    The new solver replaces those assumptions with content-based cue/output
#    identification and slot-geometry reasoning.
# d) Placement legality:
#    All four panels still occupy disjoint 5x5 windows separated by 1-cell border
#    lines at rows/cols 0, 6, 12, so there is no overlap and no out-of-bounds
#    placement.
# e) Visual observability:
#    Humans can see the blank tile directly, can spot the cue tile by its 5-pixel
#    wall plus isolated indicator, and can infer tile1/tile2 from their 2D
#    relation to the cue. No rule depends on raw color ids.
# f) Cleanliness:
#    Each panel remains isolated inside its own 5x5 area. The matrix border makes
#    slot structure explicit, avoiding clutter despite the role permutation.
# g) New solver assumptions and generator guarantees:
#    - Exactly one panel is uniform blank.
#    - Exactly one non-blank panel is a valid cue tile.
#    - The blank slot is orthogonally adjacent to the cue slot.
#    - Of the two remaining slots, one is orthogonally adjacent to the cue
#      (tile1) and the other is diagonal (tile2).
#    The generator enforces all four conditions.

from __future__ import annotations

import random

import numpy as np

from arc_utils import flood_fill, get_bg, label_mask, neighbors8


TILE_SZ = 5
GRID_ROWS = GRID_COLS = 13
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
GEN_RETRIES = 300
LBEND_PROB = 0.2

SLOTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
VALID_INDICATOR_POS = {
    (0, 0), (0, 4), (4, 0), (4, 4),
    (0, 2), (2, 0), (2, 4), (4, 2),
}


def _slot_slice(sr: int, sc: int) -> tuple[slice, slice]:
    r0 = 1 + 6 * sr
    c0 = 1 + 6 * sc
    return slice(r0, r0 + TILE_SZ), slice(c0, c0 + TILE_SZ)


def _extract_tiles(grid: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    return {
        slot: grid[_slot_slice(*slot)].copy()
        for slot in SLOTS
    }


def _slot_dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _orth_neighbors(slot: tuple[int, int]) -> list[tuple[int, int]]:
    return [other for other in SLOTS if _slot_dist(slot, other) == 1]


def _analyze_cue_tile(
    tile: np.ndarray,
) -> tuple[tuple[int, int], set[tuple[int, int]], set[tuple[int, int]], bool] | None:
    bg = int(get_bg(tile))
    cue_set = {(int(r), int(c)) for r in range(TILE_SZ) for c in range(TILE_SZ) if int(tile[r, c]) != bg}
    if len(cue_set) != 6:
        return None

    indicator = None
    for r, c in cue_set:
        if not any((nr, nc) in cue_set for nr, nc in neighbors8(r, c)):
            if indicator is not None:
                return None
            indicator = (r, c)
    if indicator is None or indicator not in VALID_INDICATOR_POS:
        return None

    wall = cue_set - {indicator}
    if (2, 2) not in wall:
        return None

    ir, ic = indicator
    dr, dc = 2 - ir, 2 - ic
    if dr == 0:
        refl = lambda r, c: (4 - r, c)
    elif dc == 0:
        refl = lambda r, c: (r, 4 - c)
    elif dr * dc > 0:
        refl = lambda r, c: (c, r)
    else:
        refl = lambda r, c: (4 - c, 4 - r)
    if wall != {refl(r, c) for r, c in wall}:
        return None

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
    if len(sides_touched) < 2:
        return None

    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    inside = {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - wall
    if not outside:
        return None

    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    wall_4conn = (n_comp == 1)
    return indicator, wall, inside, wall_4conn


def _composite_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    wall: set[tuple[int, int]],
    inside: set[tuple[int, int]],
    wall_4conn: bool,
) -> np.ndarray:
    out = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1 = int(t1[r, c])
            v2 = int(t2[r, c])
            if (r, c) in inside:
                if v1 != bg:
                    out[r, c] = v1
            elif (r, c) in wall:
                if v1 != bg:
                    out[r, c] = v1
                elif wall_4conn and v2 != bg:
                    out[r, c] = v2
            else:
                if v2 != bg:
                    out[r, c] = v2
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two pattern tiles into a blank panel using a cue panel in a 2x2 matrix.

    Input structure:
    - Four 5x5 panels arranged as a 2x2 matrix inside a 13x13 border-separated grid.
    - One panel is a cue: wall boundary (8-connected, through center) + one isolated indicator.
    - Two panels are source patterns. One panel is uniformly blank and is the output target.

    Rule:
    - Identify the blank target panel by content, not by fixed position.
    - Identify the cue panel by its wall+indicator structure.
    - Among the two remaining panels, the one orthogonally adjacent to the cue is tile1
      and the diagonal panel is tile2. Flood-fill from the indicator, blocked by the wall:
      inside gets tile1, outside gets tile2, and wall uses tile1 with tile2 fallback only
      when the wall is 4-connected.

    Key insight:
    - The matrix layout removes any reliable left-to-right panel order. Roles must be
      recovered from panel content plus 2D slot adjacency.
    """
    g = np.array(grid, dtype=int)
    assert g.shape == (GRID_ROWS, GRID_COLS), f"expected {GRID_ROWS}x{GRID_COLS}, got {g.shape}"

    border = int(g[0, 0])
    assert np.all(g[0, :] == border) and np.all(g[6, :] == border) and np.all(g[12, :] == border)
    assert np.all(g[:, 0] == border) and np.all(g[:, 6] == border) and np.all(g[:, 12] == border)

    tiles = _extract_tiles(g)
    blank_slots = [slot for slot, tile in tiles.items() if np.all(tile == int(tile[0, 0]))]
    assert len(blank_slots) == 1, f"expected one blank output panel, found {len(blank_slots)}"
    out_slot = blank_slots[0]
    bg = int(tiles[out_slot][0, 0])
    assert border != bg, "border color must differ from tile background"

    cue_candidates = []
    for slot, tile in tiles.items():
        if slot == out_slot:
            continue
        if int(get_bg(tile)) != bg:
            continue
        info = _analyze_cue_tile(tile)
        if info is not None:
            cue_candidates.append((slot, info))
    assert len(cue_candidates) == 1, f"expected one cue panel, found {len(cue_candidates)}"
    cue_slot, cue_info = cue_candidates[0]
    _indicator, wall, inside, wall_4conn = cue_info

    assert _slot_dist(cue_slot, out_slot) == 1, "blank output must be orthogonally adjacent to cue"
    remaining = [slot for slot in SLOTS if slot not in {cue_slot, out_slot}]
    t1_candidates = [slot for slot in remaining if _slot_dist(slot, cue_slot) == 1]
    t2_candidates = [slot for slot in remaining if _slot_dist(slot, cue_slot) == 2]
    assert len(t1_candidates) == 1 and len(t2_candidates) == 1, "unable to recover source roles from slot geometry"
    t1_slot = t1_candidates[0]
    t2_slot = t2_candidates[0]

    t1 = tiles[t1_slot]
    t2 = tiles[t2_slot]
    assert int(get_bg(t1)) == bg and int(get_bg(t2)) == bg, "all panels must share the same tile background"

    out_tile = _composite_tile(t1, t2, bg, wall, inside, wall_4conn)
    out = g.copy()
    out[_slot_slice(*out_slot)] = out_tile
    return out


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
        pat = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
        count = 0
        for i in order:
            orbit = _ORBITS[i]
            if count + len(orbit) > max_px:
                continue
            if rng.random() < ORBIT_INCLUDE_PROB:
                for r, c in orbit:
                    pat[r, c] = fg
                count += len(orbit)
        if min_px <= count <= max_px:
            return pat
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


def _assemble_grid(
    border: int,
    cue_tile: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    cue_slot: tuple[int, int],
    out_slot: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    remaining = [slot for slot in SLOTS if slot not in {cue_slot, out_slot}]
    t1_slot = [slot for slot in remaining if _slot_dist(slot, cue_slot) == 1][0]
    t2_slot = [slot for slot in remaining if _slot_dist(slot, cue_slot) == 2][0]

    blank = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    role_tiles = {
        cue_slot: cue_tile,
        t1_slot: t1,
        t2_slot: t2,
        out_slot: blank,
    }
    grid = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
    for slot, tile in role_tiles.items():
        grid[_slot_slice(*slot)] = tile
    return grid, t1_slot, t2_slot


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    bg = rng.choice(palette)
    other = [c for c in palette if c != bg]
    cue_color, t1_color, t2_color = rng.sample(other, 3)
    border = rng.choice(other)

    pool = _LBEND_CONFIGS if rng.random() < LBEND_PROB else _STRAIGHT_CONFIGS
    indicator, wall = rng.choice(pool)
    inside = _flood_inside(wall, indicator)

    t1 = _sample_symmetric_pattern(bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    cue_tile = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r, c in wall:
        cue_tile[r, c] = cue_color
    cue_tile[indicator[0], indicator[1]] = cue_color

    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    wall_4conn = (n_comp == 1)
    out_tile = _composite_tile(t1, t2, bg, set(wall), inside, wall_4conn)

    cue_slot = rng.choice(SLOTS)
    out_slot = rng.choice(_orth_neighbors(cue_slot))
    grid, _t1_slot, _t2_slot = _assemble_grid(border, cue_tile, t1, t2, bg, cue_slot, out_slot)

    expected = grid.copy()
    expected[_slot_slice(*out_slot)] = out_tile
    got = main(grid)
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
