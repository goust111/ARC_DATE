from __future__ import annotations
import random
from collections import Counter
import numpy as np
from arc_utils import get_bg, flood_fill, neighbors8, in_bounds, label_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two pattern tiles into a fourth panel using flood-fill from an indicator pixel.

    Input structure:
    - Four 5×5 panels in one row, border-separated.
    - Panel 0: cue pixels = wall boundary (8-connected, edge-to-edge, through center)
               + one 8-isolated indicator pixel.
    - Panels 1–2: source patterns. Panel 3: blank (to fill).

    Rule:
    - Flood-fill (4-conn) from indicator, blocked by wall → inside region.
    - Inside → non-bg from panel 1; outside → non-bg from panel 2.
    - Wall → panel 1 if non-bg; panel 2 fallback when wall is 4-connected.

    Key insight: The cue panel's isolated indicator and wall partition the tile into inside/outside masks that select between the two source panels.
    """
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, _t3 = tiles
    b = int(Counter(t1.flatten()).most_common(1)[0][0])
    assert border != b, "border color must differ from tile background"

    t0_bg = int(Counter(t0.flatten()).most_common(1)[0][0])
    cue_set = {(int(r), int(c)) for r in range(5) for c in range(5) if int(t0[r, c]) != t0_bg}
    assert cue_set, "no cue pixels found"

    # --- Indicator = the unique 8-isolated cue pixel ---
    indicator = None
    for r, c in cue_set:
        if not any((nr, nc) in cue_set for nr, nc in neighbors8(r, c)):
            assert indicator is None, "multiple 8-isolated cue pixels"
            indicator = (r, c)
    assert indicator is not None, "no 8-isolated cue pixel found"

    # Indicator must sit at a tile corner or edge midpoint
    valid_pos = {(0,0),(0,4),(4,0),(4,4),(0,2),(2,0),(2,4),(4,2)}
    assert indicator in valid_pos, f"indicator {indicator} not at corner/edge-midpoint"

    wall = cue_set - {indicator}

    # --- Structural assertions (feedback for generate_input agents) ---
    assert (2, 2) in wall, "wall must pass through the grid center (2,2)"

    # Wall must be symmetric about the line from indicator to center (2,2)
    ir, ic = indicator
    dr, dc = 2 - ir, 2 - ic
    if dr == 0:
        _refl = lambda r, c: (4 - r, c)
    elif dc == 0:
        _refl = lambda r, c: (r, 4 - c)
    elif dr * dc > 0:
        _refl = lambda r, c: (c, r)
    else:
        _refl = lambda r, c: (4 - c, 4 - r)
    assert wall == {_refl(r, c) for r, c in wall}, "wall must be symmetric about indicator→center axis"

    sides_touched = set()
    for r, c in wall:
        if r == 0: sides_touched.add("top")
        if r == 4: sides_touched.add("bottom")
        if c == 0: sides_touched.add("left")
        if c == 4: sides_touched.add("right")
    assert len(sides_touched) >= 2, f"wall must span edge-to-edge, only touches {sides_touched}"

    # --- Flood fill inside region ---
    EMPTY, WALL, INSIDE = 0, 1, 2
    tmp = np.zeros((5, 5), dtype=int)
    for r, c in wall:
        tmp[r, c] = WALL
    flood_fill(tmp, indicator[0], indicator[1], INSIDE, connectivity=4)
    inside = {(int(r), int(c)) for r, c in np.argwhere(tmp == INSIDE)}

    outside = {(r, c) for r in range(5) for c in range(5)} - inside - wall
    assert outside, "wall must partition the grid — outside region is empty"

    # Wall 4-connectivity determines boundary blending behavior:
    # axis-aligned walls (4-connected) allow t2 fallback; diagonal walls don't.
    wall_mask = np.zeros((5, 5), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    wall_4conn = (n_comp == 1)

    # --- Build output tile ---
    out3 = np.full((5, 5), b, dtype=int)
    for r in range(5):
        for c in range(5):
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
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out

# --------------- generator (subagent) ---------------

# ── Tunables ──────────────────────────────────────────────────────────────────
TILE_SZ = 5                    # each panel is 5x5
GRID_ROWS, GRID_COLS = 7, 25  # fixed output grid shape
NUM_COLORS = 10                # ARC color palette size

T1_MIN_PX, T1_MAX_PX = 8, 12  # tile-1 non-bg pixel range
T2_MIN_PX, T2_MAX_PX = 9, 12  # tile-2 non-bg pixel range
ORBIT_INCLUDE_PROB = 0.55      # per-orbit inclusion probability for pattern sampling
PATTERN_RETRIES = 200          # max retries per _sample_symmetric_pattern call
GEN_RETRIES = 300              # max retries in generate_input outer loop
LBEND_PROB = 0.2               # L-bend wall probability (eval ratio ~4:1 straight:L-bend)

# ── Symmetric pattern orbits (D1+D2 symmetry on 5x5) ─────────────────────────
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


def _sample_symmetric_pattern(bg: int, fg: int, rng: random.Random, *,
                              min_px: int, max_px: int) -> np.ndarray | None:
    """Sample a 5x5 grid symmetric under both diagonals, with fg pixel count in
    [min_px, max_px].  Randomizes orbit order, includes each greedily within
    the max budget, retries until min is met."""
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


# ── Precomputed (indicator, wall) configurations ─────────────────────────────
# 8 straight + 12 L-bend combos, all verified to partition the 5x5 grid.
_STRAIGHT_CONFIGS = [
    ((0, 2), frozenset({(2,0),(2,1),(2,2),(2,3),(2,4)})),          # row
    ((4, 2), frozenset({(2,0),(2,1),(2,2),(2,3),(2,4)})),          # row
    ((2, 0), frozenset({(0,2),(1,2),(2,2),(3,2),(4,2)})),          # col
    ((2, 4), frozenset({(0,2),(1,2),(2,2),(3,2),(4,2)})),          # col
    ((0, 0), frozenset({(0,4),(1,3),(2,2),(3,1),(4,0)})),          # anti-diag
    ((4, 4), frozenset({(0,4),(1,3),(2,2),(3,1),(4,0)})),          # anti-diag
    ((0, 4), frozenset({(0,0),(1,1),(2,2),(3,3),(4,4)})),          # diag
    ((4, 0), frozenset({(0,0),(1,1),(2,2),(3,3),(4,4)})),          # diag
]
_LBEND_CONFIGS = [
    ((0, 0), frozenset({(2,2),(1,2),(0,2),(2,1),(2,0)})),          # U+L
    ((0, 0), frozenset({(2,2),(3,2),(4,2),(2,3),(2,4)})),          # D+R
    ((4, 4), frozenset({(2,2),(1,2),(0,2),(2,1),(2,0)})),          # U+L
    ((4, 4), frozenset({(2,2),(3,2),(4,2),(2,3),(2,4)})),          # D+R
    ((0, 4), frozenset({(2,2),(1,2),(0,2),(2,3),(2,4)})),          # U+R
    ((0, 4), frozenset({(2,2),(3,2),(4,2),(2,1),(2,0)})),          # D+L
    ((4, 0), frozenset({(2,2),(1,2),(0,2),(2,3),(2,4)})),          # U+R
    ((4, 0), frozenset({(2,2),(3,2),(4,2),(2,1),(2,0)})),          # D+L
    ((4, 2), frozenset({(2,2),(1,1),(0,0),(1,3),(0,4)})),          # NW+NE
    ((0, 2), frozenset({(2,2),(3,1),(4,0),(3,3),(4,4)})),          # SW+SE
    ((2, 4), frozenset({(2,2),(1,1),(0,0),(3,1),(4,0)})),          # NW+SW
    ((2, 0), frozenset({(2,2),(1,3),(0,4),(3,3),(4,4)})),          # NE+SE
]


def _flood_inside(wall: frozenset, indicator: tuple[int, int]) -> set[tuple[int, int]]:
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    return {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}


# ── Main generation loop ─────────────────────────────────────────────────────

def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    # 1. Colors: border != tile_bg; cue/t1/t2 mutually distinct and != tile_bg.
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color = rng.sample(other, 3)
    border = rng.choice(other)

    # 2. Wall + indicator (4:1 straight:L-bend ratio).
    pool = _LBEND_CONFIGS if rng.random() < LBEND_PROB else _STRAIGHT_CONFIGS
    indicator, wall = rng.choice(pool)
    inside = _flood_inside(wall, indicator)

    # 3. Tile-1 / Tile-2 symmetric patterns.
    t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng,
                                   min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng,
                                   min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    # 4. Tile-0 (cue tile): bg + wall + indicator drawn in cue_color.
    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = cue_color
    t0[indicator[0], indicator[1]] = cue_color

    # 5. Composite output tile-3.
    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    wall_4conn = (n_comp == 1)
    outside = set((r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)) - inside - wall
    out3 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1, v2 = int(t1[r, c]), int(t2[r, c])
            if (r, c) in inside:
                if v1 != tile_bg: out3[r, c] = v1
            elif (r, c) in wall:
                if v1 != tile_bg: out3[r, c] = v1
                elif wall_4conn and v2 != tile_bg: out3[r, c] = v2
            else:
                if v2 != tile_bg: out3[r, c] = v2

    # 6. Assemble 7x25 input grid (tile-3 input is blank).
    t3_in = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
    for j, tile in enumerate([t0, t1, t2, t3_in]):
        g[1:6, 1 + 6 * j: 1 + 6 * j + 5] = tile

    # 7. Oracle: solver output must match hand-built composite.
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

