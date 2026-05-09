# Combo variant of 7491f3cf (index 13)
# combo_spec: D12/1+D16/1
# Name: 无显式指示像素+cue与源合并3面板

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import flood_fill, get_bg, label_mask


# -----------------------------------------------------------------------------
# Design analysis for the fused D12/1 + D16/1 variant
#
# a) Concrete fused behavior and trace
#    - Layout shrinks from 4 panels (cue, t1, t2, output) to 3 panels
#      (merged[cue+t1], t2, output), so the full grid becomes 7x19.
#    - The merged left panel now contains two foreground roles:
#      one 5-pixel cue wall in `cue_color`, plus the tile1 source pattern in
#      `t1_color`. There is no detached indicator pixel anymore.
#    - Because the explicit dot is gone, straight row/col/diag walls would be
#      ambiguous. The cue family is therefore restricted to the 8 bent shapes
#      from the D12 single-variant: 4 orthogonal elbows and 4 diagonal wedges.
#    - Example trace:
#      original merged-layout logic could read 6 cue pixels (wall + isolated
#      dot) from the left panel, flood-fill from the dot, then blend t1/t2.
#      Here the same panel instead shows only the 5-pixel bend. The solver
#      must first separate colors, recognize which color forms a legal bent
#      wall, infer the implicit seed from that wall's opening, then use the
#      remaining foreground color as tile1. The compositing stage itself stays
#      the same.
# b) Complexity calibration
#    - The panel-count change is visually obvious in every example.
#    - The no-dot change is also visually obvious.
#    - Wall geometry is limited to 8 reusable categories so humans can infer
#      the cue semantics from 2-4 examples instead of facing an open-ended
#      shape-to-region mapping problem.
#    - Tile1 is symmetric and overlaid with the cue, but its cells are blocked
#      from the cue wall positions so the cue remains readable.
# c) Implicit-assumption audit against the original solver
#    - `tiles = ... for j in range(4)`: invalid; there are now 3 panels.
#    - `t0, t1, t2, _t3 = tiles`: invalid; cue and tile1 share panel 0.
#    - `bg = mode(t1)`: unsafe now because the merged panel can have more
#      non-background than background. Background must be read from the pure
#      tile2 panel instead.
#    - `cue_set = non-bg(t0)`: invalid because panel 0 contains both cue and
#      tile1. The solver must test each foreground color as a cue candidate.
#    - `indicator = unique 8-isolated cue pixel`: invalid because the explicit
#      indicator is removed.
#    - Flood-fill from the indicator: replaced by a lookup from wall geometry
#      to an implicit seed position.
# d) Placement legality
#    - All panels remain 5x5 and stay within the fixed 7x19 border frame.
#    - The cue wall always has exactly 5 in-bounds cells and passes through
#      (2,2). Tile1 is sampled from symmetry orbits that avoid those 5 cells,
#      so cue and source never overlap in the merged panel.
# e) Visual observability
#    - A human can see three persistent clues at once: only 3 panels exist,
#      the left panel mixes two colors, and the cue color has no isolated dot.
#    - The bend opening is visibly correlated with which side receives tile1 in
#      the output, so the implicit-direction rule is grounded in geometry, not
#      in hidden color ids or numeric metadata.
# f) Visual neatness
#    - Only one overlay is allowed in panel 0: cue over tile1, with disjoint
#      cells. This keeps the cue crisp instead of burying it under collisions.
#    - Exactly two non-bg colors appear in the merged panel, which makes the
#      split between structure and content legible.
# g) Generator anti-degeneracy guarantees
#    - Every sample uses the 3-panel merged layout.
#    - Every cue has exactly 5 cells and zero isolated cue pixels.
#    - Every cue is one of the supported implicit-direction bends.
#    - Generator rejects cases where tile1 contributes nothing inside or tile2
#      contributes nothing outside, so the inferred direction actually matters.
# h) New solver assumptions introduced here
#    - Exactly one foreground color in the merged panel forms a supported
#      5-cell bent wall.
#    - The remaining foreground color in the merged panel is tile1's color.
#    - Tile2 remains a single-color symmetric source tile whose background is
#      the majority color.
# i) How the generator enforces those assumptions
#    - It samples cue and tile1 colors distinctly, blocks tile1 from wall cells,
#      validates that the merged panel exposes exactly one cue candidate, and
#      oracle-checks that `main(input)` matches the hand-built output.
# -----------------------------------------------------------------------------


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 19
PANEL_COUNT = 3
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
    blocked: set[tuple[int, int]] | frozenset[tuple[int, int]] = frozenset(),
) -> np.ndarray | None:
    blocked = set(blocked)
    allowed_orbits = [orbit for orbit in _ORBITS if not (set(orbit) & blocked)]
    if not allowed_orbits:
        return None

    for _ in range(PATTERN_RETRIES):
        order = list(range(len(allowed_orbits)))
        rng.shuffle(order)
        p = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
        count = 0
        for i in order:
            orbit = allowed_orbits[i]
            if count + len(orbit) > max_px:
                continue
            if rng.random() < ORBIT_INCLUDE_PROB:
                for r, c in orbit:
                    p[r, c] = fg
                count += len(orbit)
        if min_px <= count <= max_px:
            return p
    return None


def _inside_from_wall(
    wall: frozenset[tuple[int, int]],
) -> tuple[set[tuple[int, int]], bool]:
    assert wall in _WALL_TO_SEED, f"unsupported implicit cue geometry: {sorted(wall)}"
    seed = _WALL_TO_SEED[wall]

    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        tmp[r, c] = 1
        wall_mask[r, c] = True

    flood_fill(tmp, seed[0], seed[1], 2, connectivity=4)
    inside = {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}
    assert inside, "implicit inside region is empty"

    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - set(wall)
    assert outside, "implicit wall must leave a non-empty outside region"

    wall_4conn = label_mask(wall_mask, connectivity=4)[1] == 1
    return inside, wall_4conn


def _extract_cue_from_merged(
    tile: np.ndarray, bg: int
) -> tuple[int, frozenset[tuple[int, int]], set[tuple[int, int]], bool]:
    candidates = []
    for color in sorted(set(map(int, tile.flatten())) - {int(bg)}):
        coords = frozenset(
            (int(r), int(c))
            for r in range(TILE_SZ)
            for c in range(TILE_SZ)
            if int(tile[r, c]) == int(color)
        )
        if len(coords) != 5 or (2, 2) not in coords or coords not in _WALL_TO_SEED:
            continue
        inside, wall_4conn = _inside_from_wall(coords)
        candidates.append((int(color), coords, inside, wall_4conn))

    assert len(candidates) == 1, f"expected exactly one implicit cue color, found {len(candidates)}"
    return candidates[0]


def _build_output_tile(
    merged: np.ndarray,
    t2: np.ndarray,
    *,
    bg: int,
    cue_color: int,
    wall: frozenset[tuple[int, int]],
    inside: set[tuple[int, int]],
    wall_4conn: bool,
) -> np.ndarray:
    out = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v_merged = int(merged[r, c])
            v2 = int(t2[r, c])
            t1_here = v_merged if v_merged not in {bg, cue_color} else bg
            if (r, c) in inside:
                if t1_here != bg:
                    out[r, c] = t1_here
            elif (r, c) in wall:
                if t1_here != bg:
                    out[r, c] = t1_here
                elif wall_4conn and v2 != bg:
                    out[r, c] = v2
            else:
                if v2 != bg:
                    out[r, c] = v2
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two source tiles into the output panel using an implicit wall cue
    that has been merged into the tile1 panel.

    Input structure:
    - Three 5x5 panels in one row, border-separated, for a 7x19 grid.
    - Panel 0: merged panel containing a 5-pixel cue wall in one color and the
      tile1 source pattern in a second non-background color.
    - Panel 1: tile2 source pattern.
    - Panel 2: blank output panel.

    Rule:
    - Infer the background from panel 1.
    - In panel 0, identify the unique foreground color whose 5 cells form one
      of the supported bent walls through the center; that color is the cue.
    - Use the wall geometry itself to infer the inside region, with no explicit
      indicator pixel.
    - Inside pixels copy non-background from tile1; outside pixels copy
      non-background from tile2; wall pixels prefer tile1 and use tile2 only as
      4-connected-wall fallback.

    Key insight:
    - The left panel must be parsed in two stages: split cue color from tile1
      color, then decode the wall opening as an implicit direction marker.
    """
    g = np.array(grid, dtype=int)
    assert g.shape == (GRID_ROWS, GRID_COLS), f"expected {(GRID_ROWS, GRID_COLS)}, got {g.shape}"

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(PANEL_COUNT)]
    merged, t2, _out_in = tiles

    bg = int(get_bg(t2))
    assert border != bg, "border color must differ from tile background"

    cue_color, wall, inside, wall_4conn = _extract_cue_from_merged(merged, bg)

    merged_non_bg = set(map(int, merged.flatten())) - {bg}
    assert cue_color in merged_non_bg, "cue color missing from merged panel"
    t1_colors = merged_non_bg - {cue_color}
    assert len(t1_colors) == 1, f"expected exactly one tile1 color, found {t1_colors}"

    t2_colors = set(map(int, t2.flatten())) - {bg}
    assert len(t2_colors) == 1, f"expected exactly one tile2 color, found {t2_colors}"

    out_tile = _build_output_tile(
        merged,
        t2,
        bg=bg,
        cue_color=cue_color,
        wall=wall,
        inside=inside,
        wall_4conn=wall_4conn,
    )

    out = g.copy()
    out[1:6, 1 + 6 * 2 : 1 + 6 * 2 + 5] = out_tile
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color = rng.sample(other, 3)
    border = rng.choice(other)

    pool = _ORTHOGONAL_BENDS if rng.random() < ORTHOGONAL_BEND_PROB else _DIAGONAL_BENDS
    _wall_name, wall, _seed = rng.choice(pool)
    inside, wall_4conn = _inside_from_wall(wall)
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - set(wall)

    t1 = _sample_symmetric_pattern(
        tile_bg,
        t1_color,
        rng,
        min_px=T1_MIN_PX,
        max_px=T1_MAX_PX,
        blocked=wall,
    )
    t2 = _sample_symmetric_pattern(
        tile_bg,
        t2_color,
        rng,
        min_px=T2_MIN_PX,
        max_px=T2_MAX_PX,
    )
    if t1 is None or t2 is None:
        return None

    t1_inside_hits = sum(int(t1[r, c] != tile_bg) for r, c in inside)
    t2_outside_hits = sum(int(t2[r, c] != tile_bg) for r, c in outside)
    if t1_inside_hits == 0 or t2_outside_hits == 0:
        return None

    merged = t1.copy()
    for r, c in wall:
        merged[r, c] = cue_color

    try:
        found_cue_color, found_wall, found_inside, found_wall_4conn = _extract_cue_from_merged(
            merged, tile_bg
        )
    except AssertionError:
        return None
    if found_cue_color != cue_color or found_wall != wall:
        return None
    if found_inside != inside or found_wall_4conn != wall_4conn:
        return None

    out_tile = _build_output_tile(
        merged,
        t2,
        bg=tile_bg,
        cue_color=cue_color,
        wall=wall,
        inside=inside,
        wall_4conn=wall_4conn,
    )
    if np.array_equal(out_tile, t2):
        return None

    out_in = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
    for j, tile in enumerate([merged, t2, out_in]):
        g[1:6, 1 + 6 * j : 1 + 6 * j + 5] = tile

    expected = g.copy()
    expected[1:6, 1 + 6 * 2 : 1 + 6 * 2 + 5] = out_tile
    try:
        got = main(g)
    except AssertionError:
        return None
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
