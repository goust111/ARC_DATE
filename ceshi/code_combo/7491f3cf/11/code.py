# Combo variant of 7491f3cf (index 11)
# combo_spec: D1/1+D7/1
# Name: 曼哈顿距离分割+内部指示像素

from __future__ import annotations

from collections import Counter
from collections import deque
import random

import numpy as np


# Design analysis for this combo variant
# a) Concrete behavior and trace:
#    Original task: panel 0 had 6 cue pixels, namely a 5-pixel wall plus one edge/corner indicator.
#    The solver flood-filled from that indicator, treated the reached side as panel-1 territory, and
#    used panel 2 on the other side. In a concrete old example, a horizontal wall through row 2 plus
#    an indicator at (0,2) made the top half come from panel 1 and the bottom half from panel 2.
#    This combo removes the wall completely. Panel 0 now has exactly two cue pixels: a center marker
#    fixed at (2,2) and one internal indicator at a non-edge, non-center location such as (1,2) or
#    (1,1). The Manhattan distance from the indicator to the center is the threshold k. If the
#    indicator is at (1,2), then k=1 and panel-1 territory is the 5-cell cross
#    {(2,2),(1,2),(2,1),(2,3),(3,2)}; if it is at (1,1), then k=2 and panel-1 territory expands to
#    the 13-cell diamond containing all cells with |r-2|+|c-2|<=2. Every other cell comes from
#    panel 2. The visible difference from the old task is that the split is no longer a straight or
#    bent barrier; it is a radius-based center-out partition whose size is encoded by an interior
#    cue location.
# b) Complexity calibration:
#    The 5x5 geometry naturally limits the threshold to two visually distinct cases, k in {1,2},
#    because the indicator must be internal but not the center. That is enough variation to make the
#    rule visible without turning the task into an over-parameterized search problem. The generator
#    balances both k values and rejects samples unless the distance-2 ring actually changes the
#    output, so the threshold cue is always behaviorally relevant.
# c) Implicit-assumption audit of the original main():
#    - "There is a unique 8-isolated indicator" no longer applies. In the combo there are exactly
#      two cue pixels, and neither needs to be 8-isolated from a wall because there is no wall.
#    - "Indicator must be at a corner or edge midpoint" is broken by D7/1. The new indicator is
#      required to be internal: 0<r<4, 0<c<4, and (r,c)!=(2,2).
#    - "Cue pixels other than the indicator form a wall through the center" is removed. The center
#      cue is now a point marker, not part of a blocking structure.
#    - Flood fill, wall symmetry, side-touch checks, and wall 4-connectivity fallback all depended on
#      the wall abstraction. None of those premises survive the Manhattan split, so they are deleted
#      rather than patched.
# d) Placement legality:
#    The cue tile is always legal because it places exactly two pixels inside a fixed 5x5 panel:
#    one at the center and one chosen from the 8 internal non-center cells. They never overlap and
#    never leave bounds. Pattern tiles remain 5x5 single-color patterns on a shared background.
# e) Visual observability:
#    Humans can infer the combo from a few examples because the cue tile explicitly shows both the
#    partition center and a radius sample. Across examples, the only changing cue is the internal
#    indicator, and its Manhattan distance to the center visibly matches whether the output uses
#    panel 1 only on the small cross (k=1) or on the larger diamond (k=2).
# f) Visual cleanliness:
#    Panel 0 is simpler than the original: only two cue pixels and no overlapping wall geometry. The
#    output remains a clean two-source composite, and the symmetric source patterns keep the result
#    legible even though the split is radial instead of linear.
# g) Color-change inferability:
#    No arbitrary recoloring is introduced. Panel 1 and panel 2 keep their own single foreground
#    colors, so the only thing the solver must infer is spatial ownership, not a hidden color map.
# h) Quantitative anti-degeneracy metric:
#    `variant_metrics()` reports the threshold `k`, the size of the Manhattan-inside region, the
#    number of output cells changed by switching between k=1 and k=2, and how many legacy wall-based
#    composites would accidentally match the same source tiles. The generator accepts a sample only
#    when the threshold changes the output and the produced composite matches none of the legacy
#    wall-based outputs.
# i) New assumptions introduced by the new main():
#    - Panel 0 contains exactly two non-background cue pixels.
#    - One of them is the center marker at (2,2); the other is the internal indicator.
#    - The indicator therefore determines k in {1,2}.
#    - All four tiles share the same background color, and panel 3 is blank on input.
#    `generate_input()` constructs inputs that satisfy each of these assumptions explicitly.


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
GEN_RETRIES = 400

CENTER = (2, 2)
INNER_RING = frozenset({(1, 2), (2, 1), (2, 3), (3, 2)})
MID_RING = frozenset({(0, 2), (1, 1), (1, 3), (2, 0), (2, 4), (3, 1), (3, 3), (4, 2)})
OUTER_RING = frozenset(
    {
        (0, 0),
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 0),
        (1, 4),
        (3, 0),
        (3, 4),
        (4, 0),
        (4, 1),
        (4, 3),
        (4, 4),
    }
)
INTERNAL_INDICATORS = (
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 3),
    (3, 1),
    (3, 2),
    (3, 3),
)
K1_INDICATORS = tuple(p for p in INTERNAL_INDICATORS if abs(p[0] - 2) + abs(p[1] - 2) == 1)
K2_INDICATORS = tuple(p for p in INTERNAL_INDICATORS if abs(p[0] - 2) + abs(p[1] - 2) == 2)

# Symmetric pattern orbits under both diagonals.
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

# Legacy configs kept only for anti-degeneracy checking.
_LEGACY_STRAIGHT_CONFIGS = [
    ((0, 2), frozenset({(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)})),
    ((4, 2), frozenset({(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)})),
    ((2, 0), frozenset({(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)})),
    ((2, 4), frozenset({(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)})),
    ((0, 0), frozenset({(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)})),
    ((4, 4), frozenset({(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)})),
    ((0, 4), frozenset({(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)})),
    ((4, 0), frozenset({(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)})),
]
_LEGACY_LBEND_CONFIGS = [
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
LEGACY_CONFIGS = tuple(_LEGACY_STRAIGHT_CONFIGS + _LEGACY_LBEND_CONFIGS)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _parse_tiles(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return tuple(grid[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4))  # type: ignore[return-value]


def _threshold_region(k: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ) if _manhattan((r, c), CENTER) <= k}


def _compose_manhattan(t1: np.ndarray, t2: np.ndarray, bg: int, k: int) -> np.ndarray:
    out = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            if _manhattan((r, c), CENTER) <= k:
                if int(t1[r, c]) != bg:
                    out[r, c] = int(t1[r, c])
            else:
                if int(t2[r, c]) != bg:
                    out[r, c] = int(t2[r, c])
    return out


def _legacy_inside(wall: frozenset[tuple[int, int]], indicator: tuple[int, int]) -> set[tuple[int, int]]:
    q = deque([indicator])
    seen = {indicator}
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < TILE_SZ and 0 <= nc < TILE_SZ):
                continue
            if (nr, nc) in wall or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return seen


def _wall_is_4_connected(wall: frozenset[tuple[int, int]]) -> bool:
    start = next(iter(wall))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if nxt in wall and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(wall)


def _compose_legacy(t1: np.ndarray, t2: np.ndarray, bg: int, indicator: tuple[int, int], wall: frozenset[tuple[int, int]]) -> np.ndarray:
    inside = _legacy_inside(wall, indicator)
    wall_4conn = _wall_is_4_connected(wall)
    out = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1, v2 = int(t1[r, c]), int(t2[r, c])
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


def _visible_threshold_delta(t1: np.ndarray, t2: np.ndarray, bg: int) -> int:
    out1 = _compose_manhattan(t1, t2, bg, 1)
    out2 = _compose_manhattan(t1, t2, bg, 2)
    return int(np.count_nonzero(out1 != out2))


def _legacy_match_count(t1: np.ndarray, t2: np.ndarray, bg: int, out3: np.ndarray) -> int:
    return sum(int(np.array_equal(out3, _compose_legacy(t1, t2, bg, indicator, wall))) for indicator, wall in LEGACY_CONFIGS)


def _ring_fg_count(tile: np.ndarray, bg: int, cells: frozenset[tuple[int, int]]) -> int:
    return sum(int(tile[r, c] != bg) for r, c in cells)


def _sample_symmetric_pattern(bg: int, fg: int, rng: random.Random, *, min_px: int, max_px: int) -> np.ndarray | None:
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


def _sample_patterns(bg: int, t1_color: int, t2_color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PATTERN_RETRIES):
        t1 = _sample_symmetric_pattern(bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
        t2 = _sample_symmetric_pattern(bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
        if t1 is None or t2 is None:
            continue
        if _ring_fg_count(t1, bg, INNER_RING | frozenset({CENTER})) == 0:
            continue
        if _ring_fg_count(t1, bg, MID_RING) == 0:
            continue
        if _ring_fg_count(t2, bg, MID_RING) == 0:
            continue
        if _ring_fg_count(t2, bg, OUTER_RING) == 0:
            continue
        if _visible_threshold_delta(t1, t2, bg) == 0:
            continue
        return t1, t2
    return None


def _choose_indicator(rng: random.Random) -> tuple[int, int]:
    pool = K1_INDICATORS if rng.random() < 0.5 else K2_INDICATORS
    return rng.choice(pool)


def _build_cue_tile(bg: int, cue_color: int, indicator: tuple[int, int]) -> np.ndarray:
    t0 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    t0[CENTER[0], CENTER[1]] = cue_color
    t0[indicator[0], indicator[1]] = cue_color
    return t0


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    t0, t1, t2, _t3 = _parse_tiles(g)
    bg = _get_bg(t1)
    cue = [(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ) if int(t0[r, c]) != bg]
    if len(cue) != 2 or CENTER not in cue:
        raise ValueError("cue tile must contain exactly two cue pixels including the center marker")
    indicator = next(p for p in cue if p != CENTER)
    k = _manhattan(indicator, CENTER)
    if k not in (1, 2):
        raise ValueError("indicator must imply threshold k in {1,2}")
    out3 = _compose_manhattan(t1, t2, bg, k)
    return {
        "k": int(k),
        "inside_size": int(len(_threshold_region(k))),
        "threshold_delta_cells": int(_visible_threshold_delta(t1, t2, bg)),
        "mid_ring_activity": int(_ring_fg_count(t1, bg, MID_RING) + _ring_fg_count(t2, bg, MID_RING)),
        "legacy_match_count": int(_legacy_match_count(t1, t2, bg, out3)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two pattern tiles into a fourth panel using a Manhattan-distance threshold cue.

    Input structure:
    - Four 5×5 panels in one row, border-separated.
    - Panel 0: exactly two cue pixels in one color: a center marker at (2,2) and one internal
      indicator pixel at a non-edge, non-center location.
    - Panels 1–2: source patterns. Panel 3: blank (to fill).

    Rule:
    - Let k be the Manhattan distance from the indicator to the center marker.
    - For each cell in panel 3, if its Manhattan distance to the center is <= k, copy panel 1's
      non-background pixel there; otherwise copy panel 2's non-background pixel there.

    Key insight:
    - Panel 0 no longer encodes a blocking wall. It encodes a radius: the center marker defines the
      origin and the internal indicator defines the threshold k. The output is a center-out radial
      composite rather than a wall-separated flood-fill composite.
    """
    g = np.array(grid, dtype=int)
    assert g.shape == (GRID_ROWS, GRID_COLS), f"expected {(GRID_ROWS, GRID_COLS)}, got {g.shape}"

    border = int(g[0, 0])
    t0, t1, t2, t3 = _parse_tiles(g)
    bg = _get_bg(t1)

    assert border != bg, "border color must differ from tile background"
    assert _get_bg(t0) == bg and _get_bg(t2) == bg and _get_bg(t3) == bg, "all tiles must share the same background"
    assert np.all(t3 == bg), "panel 3 input must be blank"

    cue = [(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ) if int(t0[r, c]) != bg]
    assert len(cue) == 2, f"expected exactly two cue pixels, found {len(cue)}"
    assert CENTER in cue, "panel 0 must include the center marker at (2,2)"
    cue_colors = {int(t0[r, c]) for r, c in cue}
    assert len(cue_colors) == 1, "both cue pixels must share one cue color"

    indicator = next(p for p in cue if p != CENTER)
    ir, ic = indicator
    assert 0 < ir < TILE_SZ - 1 and 0 < ic < TILE_SZ - 1, "indicator must be internal, not on the border"
    assert indicator != CENTER, "indicator must differ from the center marker"

    k = _manhattan(indicator, CENTER)
    assert k in (1, 2), f"indicator {indicator} must imply threshold 1 or 2, got {k}"

    out3 = _compose_manhattan(t1, t2, bg, k)
    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    bg = rng.choice(palette)
    other = [c for c in palette if c != bg]
    cue_color, t1_color, t2_color, border = rng.sample(other, 4)

    indicator = _choose_indicator(rng)
    k = _manhattan(indicator, CENTER)

    patterns = _sample_patterns(bg, t1_color, t2_color, rng)
    if patterns is None:
        return None
    t1, t2 = patterns

    out3 = _compose_manhattan(t1, t2, bg, k)
    if _legacy_match_count(t1, t2, bg, out3) != 0:
        return None

    t0 = _build_cue_tile(bg, cue_color, indicator)
    t3_in = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)

    g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
    for j, tile in enumerate((t0, t1, t2, t3_in)):
        g[1:6, 1 + 6 * j : 1 + 6 * j + 5] = tile

    expected = g.copy()
    expected[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    got = main(g)
    if not np.array_equal(got, expected):
        return None

    metrics = variant_metrics(g)
    if metrics["threshold_delta_cells"] <= 0 or metrics["legacy_match_count"] != 0:
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
