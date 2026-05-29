# Variant of 7491f3cf
# Dimension: D12 — 面板0中cue的编码方式 (symbol_instructions)
# Change: 面板0用同一种cue_color绘制墙线和指示像素，指示像素通过8-孤立性（无8-邻居是cue）与墙区分... → 无显式指示像素——由墙形状暗示方向

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import flood_fill, label_mask


# Design analysis for the D12 variant:
# a) Concrete behavior
#    - Tile 0 no longer contains a detached indicator pixel. It contains only a
#      5-pixel wall passing through the center.
#    - To keep the direction visually recoverable, only bent 2-ray walls are
#      allowed: 4 orthogonal elbows and 4 diagonal wedges. Straight row/col/diag
#      lines are removed because without an explicit dot they do not say which
#      side is "inside".
#    - The inside region is the 4-cell pocket geometrically between the two rays:
#      top-left / top-right / bottom-left / bottom-right for elbows, and top /
#      bottom / left / right for wedges.
#    - Small trace example:
#      old rule on wall {(2,2),(3,2),(4,2),(2,3),(2,4)} with indicator (0,0)
#      flood-filled the large top-left region. New rule uses the same wall
#      without any dot, reads the bend opening toward bottom-right, and flood-
#      fills from seed (4,4), producing the 2x2 bottom-right pocket instead.
#      The composition rule itself is unchanged; only cue decoding changes.
# b) Complexity calibration
#    - Supported cue shapes are limited to 8 repeated categories, enough for
#      variety but still small enough for humans to infer from a few examples.
#    - Removing ambiguous straight walls prevents collisions between multiple
#      plausible "inside" interpretations.
# c) Original-code dependency audit
#    - `indicator = unique 8-isolated cue pixel`: invalid now, removed.
#    - `indicator in valid_pos`: invalid now, removed.
#    - symmetry check relative to indicator->center axis: invalid now, replaced
#      by exact membership in the supported bent-wall lookup table.
#    - flood-fill seed was the indicator itself: invalid now, replaced by an
#      implicit seed inferred from wall geometry.
#    - wall 4-connectivity blending logic is still valid and preserved.
# d) Placement legality
#    - Each tile remains 5x5 in the original 7x25 frame.
#    - Tile 0 has exactly 5 cue pixels, all part of the wall, so there is no
#      overlap or out-of-bounds risk. Every supported wall passes through (2,2).
# e) Visual observability
#    - The absence of an isolated dot is itself visible.
#    - The bend opening is also visible, so the new rule can be inferred from
#      2-4 examples without relying on hidden numeric properties.
# f) Visual neatness
#    - Tile 0 now shows a single clean cue object instead of a wall plus a
#      detached pixel, reducing clutter.
# g) Color reasoning
#    - No color-ordering or numeric color logic is introduced. Colors only mark
#      border, tile background, cue, tile1 pattern, and tile2 pattern.
# h) Generator guarantees
#    - It only emits supported bent walls.
#    - It rejects samples where tile1 contributes nothing inside or tile2
#      contributes nothing outside, so the implicit-direction rule stays visible.
#    - It still uses an oracle check: `main(input)` must match the hand-built
#      expected output.
# i) New solver assumptions
#    - Tile 0's cue must be one of the 8 supported bent-wall shapes.
#    - `generate_input()` enforces exactly that, so `main()` is not left with an
#      unconstrained geometry-inference problem.


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
GEN_RETRIES = 300
ORTHOGONAL_BEND_PROB = 0.65

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


_ORTHOGONAL_BENDS = [
    (
        "corner_tl",
        frozenset({(2, 2), (1, 2), (0, 2), (2, 1), (2, 0)}),
        (0, 0),
    ),
    (
        "corner_tr",
        frozenset({(2, 2), (1, 2), (0, 2), (2, 3), (2, 4)}),
        (0, 4),
    ),
    (
        "corner_bl",
        frozenset({(2, 2), (3, 2), (4, 2), (2, 1), (2, 0)}),
        (4, 0),
    ),
    (
        "corner_br",
        frozenset({(2, 2), (3, 2), (4, 2), (2, 3), (2, 4)}),
        (4, 4),
    ),
]

_DIAGONAL_BENDS = [
    (
        "wedge_top",
        frozenset({(2, 2), (1, 1), (0, 0), (1, 3), (0, 4)}),
        (0, 2),
    ),
    (
        "wedge_bottom",
        frozenset({(2, 2), (3, 1), (4, 0), (3, 3), (4, 4)}),
        (4, 2),
    ),
    (
        "wedge_left",
        frozenset({(2, 2), (1, 1), (0, 0), (3, 1), (4, 0)}),
        (2, 0),
    ),
    (
        "wedge_right",
        frozenset({(2, 2), (1, 3), (0, 4), (3, 3), (4, 4)}),
        (2, 4),
    ),
]

_ALL_BENDS = _ORTHOGONAL_BENDS + _DIAGONAL_BENDS
_WALL_TO_SEED = {wall: seed for _name, wall, seed in _ALL_BENDS}
_WALL_TO_NAME = {wall: name for name, wall, _seed in _ALL_BENDS}


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


def _inside_from_wall(wall: frozenset[tuple[int, int]]) -> set[tuple[int, int]]:
    assert wall in _WALL_TO_SEED, "unsupported wall geometry"
    seed = _WALL_TO_SEED[wall]
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, seed[0], seed[1], 2, connectivity=4)
    inside = {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}
    assert inside, "implicit inside region is empty"
    return inside


def _composite_output_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    wall: frozenset[tuple[int, int]],
    inside: set[tuple[int, int]],
) -> np.ndarray:
    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    wall_4conn = n_comp == 1

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
    Composite two pattern tiles into the fourth panel using the wall shape in
    tile 0 as an implicit direction cue.

    Tile 0 now contains only a 5-pixel bent wall. The wall geometry itself
    chooses which side is "inside"; there is no detached indicator pixel.
    """
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, _t3 = tiles
    bg = int(Counter(t1.flatten()).most_common(1)[0][0])
    assert border != bg, "border color must differ from tile background"

    t0_bg = int(Counter(t0.flatten()).most_common(1)[0][0])
    wall = frozenset(
        (int(r), int(c))
        for r in range(TILE_SZ)
        for c in range(TILE_SZ)
        if int(t0[r, c]) != t0_bg
    )
    assert wall, "no cue pixels found"
    assert len(wall) == 5, f"cue wall must have 5 pixels, got {len(wall)}"
    assert (2, 2) in wall, "wall must pass through the grid center (2,2)"
    assert wall in _WALL_TO_SEED, f"unsupported implicit cue geometry: {sorted(wall)}"

    inside = _inside_from_wall(wall)
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - wall
    assert outside, "wall must leave a non-empty outside region"

    out3 = _composite_output_tile(t1, t2, bg, wall, inside)
    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color = rng.sample(other, 3)
    border = rng.choice(other)

    pool = _ORTHOGONAL_BENDS if rng.random() < ORTHOGONAL_BEND_PROB else _DIAGONAL_BENDS
    _name, wall, _seed = rng.choice(pool)
    inside = _inside_from_wall(wall)
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - wall

    t1 = _sample_symmetric_pattern(
        tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX
    )
    t2 = _sample_symmetric_pattern(
        tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX
    )
    if t1 is None or t2 is None:
        return None

    t1_inside_hits = sum(int(t1[r, c] != tile_bg) for r, c in inside)
    t2_outside_hits = sum(int(t2[r, c] != tile_bg) for r, c in outside)
    if t1_inside_hits == 0 or t2_outside_hits == 0:
        return None

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = cue_color

    out3 = _composite_output_tile(t1, t2, tile_bg, wall, inside)
    if np.array_equal(out3, t2):
        return None

    t3_in = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
    for j, tile in enumerate([t0, t1, t2, t3_in]):
        g[1:6, 1 + 6 * j : 1 + 6 * j + 5] = tile

    expected = g.copy()
    expected[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
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
