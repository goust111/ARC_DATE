# Combo variant of 7491f3cf (index 18)
# combo_spec: D5/2+D8/0
# Name: 裁剪输出+2x2矩阵布局

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import flood_fill, label_mask, neighbors8

# Design analysis for D5/2 + D8/0:
# a) Concretization:
#    Original: a 7x25 strip with tiles [cue | t1 | t2 | blank], and the solver writes the
#    composite back into tile 3. Fusion: the same four 5x5 tiles are placed in a 2x2 matrix
#    inside a 13x13 border grid: cue at top-left, t1 at top-right, t2 at bottom-left, blank
#    at bottom-right. The reasoning inside each 5x5 tile is unchanged, but the output is now
#    only the 5x5 composite tile rather than the full framed grid.
# b) Complexity calibration:
#    The task still requires detecting the wall, the isolated indicator, flood-filling the
#    indicator side, and blending two distinct source tiles. To keep the fusion visually
#    observable, generation rejects trivial cases where one source contributes almost nothing
#    or both source masks are identical.
# c) Broken implicit assumptions from the original:
#    Any code assuming a 1x4 strip, width 25, or that the answer is embedded back into the
#    large canvas is now invalid. The new solver therefore extracts panels by 2D quadrant
#    offsets and returns only the synthesized 5x5 tile.
# d) Placement legality:
#    The 13x13 matrix uses border rows/cols {0,6,12}; each tile occupies a disjoint 5x5 box
#    anchored at (1,1), (1,7), (7,1), (7,7), so nothing overlaps or leaves the canvas.
# e) Visual observability:
#    Across examples, humans can see three informative quadrants in the input and a single
#    cropped 5x5 answer. That makes the format change explicit instead of hidden in metadata.
# f) Visual cleanliness:
#    The border grid cleanly separates the four panels. The output crop removes irrelevant
#    border/context, so the answer remains a simple 5x5 composite rather than a busy 13x13.
# g) Inferability of changes:
#    No invisible color ordering is used. The only new cues are spatial: top-left is the cue,
#    top-right and bottom-left are the two sources, and the returned 5x5 corresponds to the
#    missing bottom-right panel.

TILE_SZ = 5
GRID_ROWS = 13
GRID_COLS = 13
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
GEN_RETRIES = 300
LBEND_PROB = 0.2
MIN_INSIDE_HITS = 2
MIN_OUTSIDE_HITS = 2

_PANEL_TOPLEFTS = {
    "cue": (1, 1),
    "t1": (1, 7),
    "t2": (7, 1),
    "dst": (7, 7),
}
_VALID_INDICATORS = {
    (0, 0),
    (0, 2),
    (0, 4),
    (2, 0),
    (2, 4),
    (4, 0),
    (4, 2),
    (4, 4),
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


def _extract_tile(grid: np.ndarray, key: str) -> np.ndarray:
    r0, c0 = _PANEL_TOPLEFTS[key]
    return grid[r0 : r0 + TILE_SZ, c0 : c0 + TILE_SZ].copy()


def _majority_color(tile: np.ndarray) -> int:
    return int(Counter(map(int, tile.flatten())).most_common(1)[0][0])


def _reflector(indicator: tuple[int, int]):
    ir, ic = indicator
    dr, dc = 2 - ir, 2 - ic
    if dr == 0:
        return lambda r, c: (4 - r, c)
    if dc == 0:
        return lambda r, c: (r, 4 - c)
    if dr * dc > 0:
        return lambda r, c: (c, r)
    return lambda r, c: (4 - c, 4 - r)


def _inside_region(wall: set[tuple[int, int]], indicator: tuple[int, int]) -> set[tuple[int, int]]:
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    return {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}


def _wall_is_4_connected(wall: set[tuple[int, int]]) -> bool:
    mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        mask[r, c] = True
    _, n_comp = label_mask(mask, connectivity=4)
    return bool(n_comp == 1)


def _composite_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    tile_bg: int,
    inside: set[tuple[int, int]],
    wall: set[tuple[int, int]],
) -> np.ndarray:
    out = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    wall_4conn = _wall_is_4_connected(wall)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1 = int(t1[r, c])
            v2 = int(t2[r, c])
            if (r, c) in inside:
                if v1 != tile_bg:
                    out[r, c] = v1
            elif (r, c) in wall:
                if v1 != tile_bg:
                    out[r, c] = v1
                elif wall_4conn and v2 != tile_bg:
                    out[r, c] = v2
            elif v2 != tile_bg:
                out[r, c] = v2
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two source panels into the missing fourth panel, then return only that panel.

    Input structure:
    - A 13x13 grid containing four 5x5 panels in a 2x2 matrix with 1-cell borders.
    - Top-left panel: cue pixels = one 8-connected wall through the center plus one isolated
      indicator pixel. Top-right panel: source tile 1. Bottom-left panel: source tile 2.
    - Bottom-right panel is blank in the input and represents the missing destination panel.

    Rule:
    - Flood-fill (4-connected) from the indicator inside the cue panel, blocked by the wall.
    - Use non-background pixels from tile 1 on the indicator side, and from tile 2 on the
      opposite side. On wall cells, prefer tile 1; if the wall is 4-connected and tile 1 is
      blank there, tile 2 may fill that wall pixel.
    - Return only the synthesized 5x5 destination panel, not the full 13x13 canvas.

    Key insight:
    - The bottom-right panel is conceptually the same missing composite as in the original
      task, but the evidence is arranged in quadrants and the output is a tight crop of just
      the destination tile.
    """
    g = np.array(grid, dtype=int)
    assert g.shape == (GRID_ROWS, GRID_COLS), f"expected 13x13 input, got {g.shape}"

    border = int(g[0, 0])
    assert np.all(g[0, :] == border) and np.all(g[6, :] == border) and np.all(g[12, :] == border)
    assert np.all(g[:, 0] == border) and np.all(g[:, 6] == border) and np.all(g[:, 12] == border)

    t0 = _extract_tile(g, "cue")
    t1 = _extract_tile(g, "t1")
    t2 = _extract_tile(g, "t2")
    _t3 = _extract_tile(g, "dst")

    tile_bg = _majority_color(t1)
    assert border != tile_bg, "border color must differ from tile background"
    assert _majority_color(t2) == tile_bg, "source tiles must share a background color"
    assert _majority_color(_t3) == tile_bg, "destination input tile must be blank"

    t0_bg = _majority_color(t0)
    cue_set = {(int(r), int(c)) for r in range(TILE_SZ) for c in range(TILE_SZ) if int(t0[r, c]) != t0_bg}
    assert cue_set, "no cue pixels found"

    indicator = None
    for r, c in cue_set:
        if not any((nr, nc) in cue_set for nr, nc in neighbors8(r, c)):
            assert indicator is None, "multiple 8-isolated cue pixels"
            indicator = (r, c)
    assert indicator is not None, "no 8-isolated cue pixel found"
    assert indicator in _VALID_INDICATORS, f"indicator {indicator} not at corner/edge-midpoint"

    wall = cue_set - {indicator}
    assert (2, 2) in wall, "wall must pass through center"

    refl = _reflector(indicator)
    assert wall == {refl(r, c) for r, c in wall}, "wall must be symmetric about indicator-center axis"

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

    inside = _inside_region(wall, indicator)
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - wall
    assert outside, "wall must partition the tile"

    return _composite_tile(t1, t2, tile_bg, inside, wall)


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


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color = rng.sample(other, 3)
    border = rng.choice(other)

    pool = _LBEND_CONFIGS if rng.random() < LBEND_PROB else _STRAIGHT_CONFIGS
    indicator, wall = rng.choice(pool)
    inside = _inside_region(set(wall), indicator)
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - set(wall)

    t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    if np.array_equal(t1 != tile_bg, t2 != tile_bg):
        return None

    inside_hits = sum(int(t1[r, c] != tile_bg) for r, c in inside)
    outside_hits = sum(int(t2[r, c] != tile_bg) for r, c in outside)
    if inside_hits < MIN_INSIDE_HITS or outside_hits < MIN_OUTSIDE_HITS:
        return None

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = cue_color
    t0[indicator[0], indicator[1]] = cue_color

    out3 = _composite_tile(t1, t2, tile_bg, inside, set(wall))
    if not (np.any(out3 == t1_color) and np.any(out3 == t2_color)):
        return None

    t3_in = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
    for key, tile in (("cue", t0), ("t1", t1), ("t2", t2), ("dst", t3_in)):
        r0, c0 = _PANEL_TOPLEFTS[key]
        g[r0 : r0 + TILE_SZ, c0 : c0 + TILE_SZ] = tile

    got = main(g)
    if not np.array_equal(got, out3):
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
