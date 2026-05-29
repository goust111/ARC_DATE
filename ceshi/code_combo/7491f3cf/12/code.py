from __future__ import annotations

# Combo variant of 7491f3cf (index 12)
# combo_spec: D4/0+D6/0
# Name: 多色图案源+颜色叠加混合
#
# Design analysis for the fused variant:
# - Concrete behavior:
#   Original task copies tile1 on the indicator side of the wall and tile2 on
#   the far side. Here that spatial rule stays in place as the "base" rule, but
#   the two sources are now multicolor and any coordinate where both tiles are
#   non-background becomes a paint-mixed third color instead of a simple copy.
#   Example: if a cell is inside the wall and tile1 has red while tile2 also has
#   blue, the old task would still show red there; this combo shows pink.
# - Complexity calibration:
#   To keep the new behavior visible rather than noisy, source colors are
#   restricted to the primary set {blue, red, yellow}; overlaps deterministically
#   map to the secondary set {green, pink, orange}. Each sample is forced to
#   contain repeated overlaps and at least two distinct mixed pair types.
# - Implicit assumptions audit:
#   The original solver assumes a shared tile background, a single cue color in
#   tile0, one 8-isolated indicator, a wall through the center, and symmetric
#   5x5 source patterns. Those assumptions still hold. The new assumption is
#   that generated multicolor overlaps live inside the fixed paint-mix family,
#   which the generator enforces explicitly.
# - Placement legality:
#   All source pixels still come from precomputed diagonal-symmetry orbits inside
#   the 5x5 tile, so there is no overlap/out-of-bounds placement issue beyond
#   normal orbit coloring. Wall shapes are reused from the original task and
#   flood-fill legality is checked before accepting a sample.
# - Visual observability:
#   Accepted samples must show both original-style evidence (exclusive tile1
#   pixels on the inside and exclusive tile2 pixels on the outside) and new
#   fusion evidence (multiple mixed-color overlap cells, spanning more than one
#   region). This prevents silent regression back to the original single-source
#   copy behavior.

import random
import numpy as np
from arc_utils import flood_fill, get_bg, label_mask, neighbors8


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.58
PATTERN_RETRIES = 240
PAIR_RETRIES = 240
GEN_RETRIES = 360
LBEND_PROB = 0.2

PRIMARY_COLORS = (1, 2, 4)  # blue, red, yellow
SECONDARY_COLORS = {3, 6, 7}  # green, pink, orange
PAINT_MIX = {
    frozenset((1, 2)): 6,  # blue + red -> pink
    frozenset((1, 4)): 3,  # blue + yellow -> green
    frozenset((2, 4)): 7,  # red + yellow -> orange
}
SAFE_NEUTRALS = (0, 5, 8, 9)
VALID_INDICATOR_POS = {
    (0, 0), (0, 4), (4, 0), (4, 4),
    (0, 2), (2, 0), (2, 4), (4, 2),
}


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


def _sample_support_orbits(
    rng: random.Random, *, min_px: int, max_px: int
) -> list[int] | None:
    for _ in range(PATTERN_RETRIES):
        order = list(range(len(_ORBITS)))
        rng.shuffle(order)
        chosen: list[int] = []
        count = 0
        for idx in order:
            orbit = _ORBITS[idx]
            if count + len(orbit) > max_px:
                continue
            if rng.random() < ORBIT_INCLUDE_PROB:
                chosen.append(idx)
                count += len(orbit)
        if min_px <= count <= max_px and len(chosen) >= 3:
            return chosen
    return None


def _assign_orbit_colors(orbits: list[int], rng: random.Random) -> dict[int, int]:
    max_k = min(3, len(orbits))
    k = 2 if max_k == 2 else (3 if rng.random() < 0.45 else 2)
    colors = rng.sample(list(PRIMARY_COLORS), k)
    order = list(orbits)
    rng.shuffle(order)
    out: dict[int, int] = {}
    for i, orbit_id in enumerate(order):
        if i < k:
            out[orbit_id] = colors[i]
        else:
            out[orbit_id] = rng.choice(colors)
    return out


def _render_pattern(bg: int, orbit_colors: dict[int, int]) -> np.ndarray:
    tile = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for orbit_id, color in orbit_colors.items():
        for r, c in _ORBITS[orbit_id]:
            tile[r, c] = int(color)
    return tile


def _cue_parts(t0: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    t0_bg = get_bg(t0)
    cue_set = {(int(r), int(c)) for r in range(TILE_SZ) for c in range(TILE_SZ) if int(t0[r, c]) != t0_bg}
    assert cue_set, "no cue pixels found"

    indicator = None
    for r, c in cue_set:
        if not any((nr, nc) in cue_set for nr, nc in neighbors8(r, c)):
            assert indicator is None, "multiple 8-isolated cue pixels"
            indicator = (r, c)
    assert indicator is not None, "no 8-isolated cue pixel found"
    assert indicator in VALID_INDICATOR_POS, f"indicator {indicator} not at corner/edge-midpoint"
    wall = cue_set - {indicator}
    return indicator, wall


def _assert_wall_shape(indicator: tuple[int, int], wall: set[tuple[int, int]]) -> None:
    assert (2, 2) in wall, "wall must pass through the grid center (2,2)"
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
    assert wall == {refl(r, c) for r, c in wall}, "wall must be symmetric about indicator->center axis"

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


def _flood_regions(
    wall: set[tuple[int, int]], indicator: tuple[int, int]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], bool]:
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    inside = {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - wall
    assert outside, "wall must partition the grid - outside region is empty"

    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    return inside, outside, (n_comp == 1)


def _mix_colors(a: int, b: int, bg: int) -> int:
    a = int(a)
    b = int(b)
    if a == b:
        return a
    key = frozenset((a, b))
    if key in PAINT_MIX:
        return int(PAINT_MIX[key])
    for color in (3, 6, 7, 8, 9, 5, 0, 1, 2, 4):
        if color not in {a, b, bg}:
            return color
    return a


def _base_color(
    pos: tuple[int, int],
    v1: int,
    v2: int,
    bg: int,
    inside: set[tuple[int, int]],
    wall: set[tuple[int, int]],
    wall_4conn: bool,
) -> int:
    if pos in inside:
        return int(v1) if int(v1) != bg else bg
    if pos in wall:
        if int(v1) != bg:
            return int(v1)
        if wall_4conn and int(v2) != bg:
            return int(v2)
        return bg
    return int(v2) if int(v2) != bg else bg


def _composite_tile(
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
            if v1 != bg and v2 != bg:
                out3[r, c] = _mix_colors(v1, v2, bg)
            else:
                out3[r, c] = _base_color((r, c), v1, v2, bg, inside, wall, wall_4conn)
    return out3


def _behavior_signature_from_parts(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    inside: set[tuple[int, int]],
    outside: set[tuple[int, int]],
    wall: set[tuple[int, int]],
    wall_4conn: bool,
) -> dict[str, int]:
    mixed_total = 0
    mixed_inside = 0
    mixed_outside = 0
    mixed_wall = 0
    mixed_colors: set[int] = set()
    exclusive_t1_inside = 0
    exclusive_t2_outside = 0
    overlap_total = 0

    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            pos = (r, c)
            v1, v2 = int(t1[r, c]), int(t2[r, c])
            if pos in inside and v1 != bg and v2 == bg:
                exclusive_t1_inside += 1
            if pos in outside and v2 != bg and v1 == bg:
                exclusive_t2_outside += 1
            if v1 == bg or v2 == bg:
                continue

            overlap_total += 1
            base = _base_color(pos, v1, v2, bg, inside, wall, wall_4conn)
            mixed = _mix_colors(v1, v2, bg)
            if mixed != base:
                mixed_total += 1
                mixed_colors.add(mixed)
                if pos in inside:
                    mixed_inside += 1
                elif pos in wall:
                    mixed_wall += 1
                else:
                    mixed_outside += 1

    return {
        "overlap_total": overlap_total,
        "mixed_total": mixed_total,
        "mixed_inside": mixed_inside,
        "mixed_outside": mixed_outside,
        "mixed_wall": mixed_wall,
        "distinct_mix_colors": len(mixed_colors),
        "exclusive_t1_inside": exclusive_t1_inside,
        "exclusive_t2_outside": exclusive_t2_outside,
    }


def behavior_signature(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    tiles = [g[1:6, 1 + 6 * j: 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, _t3 = tiles
    bg = get_bg(t1)
    indicator, wall = _cue_parts(t0)
    _assert_wall_shape(indicator, wall)
    inside, outside, wall_4conn = _flood_regions(wall, indicator)
    return _behavior_signature_from_parts(t1, t2, bg, inside, outside, wall, wall_4conn)


def _sample_multicolor_pair(
    bg: int,
    rng: random.Random,
    inside: set[tuple[int, int]],
    outside: set[tuple[int, int]],
    wall: set[tuple[int, int]],
    wall_4conn: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PAIR_RETRIES):
        s1 = _sample_support_orbits(rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
        s2 = _sample_support_orbits(rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
        if s1 is None or s2 is None:
            continue

        shared = set(s1) & set(s2)
        if len(shared) < 2:
            continue
        if sum(len(_ORBITS[i]) for i in shared) < 4:
            continue
        if not (set(s1) - set(s2)) or not (set(s2) - set(s1)):
            continue

        oc1 = _assign_orbit_colors(s1, rng)
        oc2 = _assign_orbit_colors(s2, rng)
        t1 = _render_pattern(bg, oc1)
        t2 = _render_pattern(bg, oc2)

        colors1 = {int(v) for v in t1.flatten() if int(v) != bg}
        colors2 = {int(v) for v in t2.flatten() if int(v) != bg}
        if not (2 <= len(colors1) <= 3 and 2 <= len(colors2) <= 3):
            continue

        overlap_pairs = set()
        for orbit_id in shared:
            c1 = oc1[orbit_id]
            c2 = oc2[orbit_id]
            if c1 != c2:
                overlap_pairs.add(frozenset((c1, c2)))
        if len(overlap_pairs) < 2:
            continue

        stats = _behavior_signature_from_parts(t1, t2, bg, inside, outside, wall, wall_4conn)
        if stats["mixed_total"] < 4:
            continue
        if stats["distinct_mix_colors"] < 2:
            continue
        if stats["mixed_inside"] == 0 or stats["mixed_outside"] == 0:
            continue
        if stats["exclusive_t1_inside"] < 2 or stats["exclusive_t2_outside"] < 2:
            continue

        return t1, t2
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two multicolor pattern tiles into a fourth panel using flood-fill
    from an indicator pixel and color mixing on overlaps.

    Input structure:
    - Four 5x5 panels in one row, border-separated.
    - Panel 0: cue pixels = wall boundary (8-connected, edge-to-edge, through
      center) + one 8-isolated indicator pixel.
    - Panels 1-2: multicolor symmetric source patterns using 2-3 foreground
      colors each. Panel 3: blank (to fill).

    Rule:
    - Flood-fill (4-conn) from the indicator, blocked by the wall -> inside.
    - Inside keeps tile1 as the base source; outside keeps tile2 as the base
      source; wall keeps the original tile1-first behavior with tile2 fallback
      only for 4-connected walls.
    - New fusion rule: whenever tile1 and tile2 both have non-background pixels
      at the same coordinate, output a mixed third color instead of either base
      source color.

    Key insight:
    - The wall still decides which source owns each region, but overlaps are no
      longer simple priority conflicts: they become a visible color-composition
      event. Solving the task requires combining spatial partition reasoning and
      repeated color-mix correspondences.
    """
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j: 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, _t3 = tiles
    bg = get_bg(t1)
    assert border != bg, "border color must differ from tile background"

    indicator, wall = _cue_parts(t0)
    _assert_wall_shape(indicator, wall)
    inside, _outside, wall_4conn = _flood_regions(wall, indicator)

    out = g.copy()
    out3 = _composite_tile(t1, t2, bg, inside, wall, wall_4conn)
    out[1:6, 1 + 6 * 3: 1 + 6 * 3 + 5] = out3
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    tile_bg, cue_color, border = rng.sample(list(SAFE_NEUTRALS), 3)

    pool = _LBEND_CONFIGS if rng.random() < LBEND_PROB else _STRAIGHT_CONFIGS
    indicator, wall = rng.choice(pool)
    inside, outside, wall_4conn = _flood_regions(set(wall), indicator)

    sampled = _sample_multicolor_pair(tile_bg, rng, inside, outside, set(wall), wall_4conn)
    if sampled is None:
        return None
    t1, t2 = sampled

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = cue_color
    t0[indicator[0], indicator[1]] = cue_color

    out3 = _composite_tile(t1, t2, tile_bg, inside, set(wall), wall_4conn)
    t3_in = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)

    g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
    for j, tile in enumerate([t0, t1, t2, t3_in]):
        g[1:6, 1 + 6 * j: 1 + 6 * j + 5] = tile

    expected = g.copy()
    expected[1:6, 1 + 6 * 3: 1 + 6 * 3 + 5] = out3
    got = main(g)
    if not np.array_equal(got, expected):
        return None

    stats = behavior_signature(g)
    if stats["mixed_total"] < 4 or stats["distinct_mix_colors"] < 2:
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
