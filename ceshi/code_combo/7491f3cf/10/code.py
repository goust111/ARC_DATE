# Combo variant of 7491f3cf (index 10)
# combo_spec: D5/0+D16/0
# Name: 预填充目标+面板顺序可变

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import flood_fill, get_bg, label_mask, neighbors8


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace:
#    The canvas is still four 5x5 panels in one row. One panel is the cue:
#    5 wall pixels plus 1 isolated indicator. Two panels are still symmetric
#    source patterns. The remaining panel is now a partially prefilled output,
#    not a blank block. A concrete valid layout is:
#      index 0 = cue, index 1 = tile1, index 2 = prefilled dst, index 3 = tile2.
#    Original task: panel 3 is all background, so the solver can hardcode roles.
#    Combo task: first find the cue structurally; then find the unique non-cue
#    panel that is not D1+D2-symmetric and treat it as the partially solved dst;
#    finally assign the two symmetric source panels by distance to the cue
#    (nearer = tile1/inside, farther = tile2/outside) and fill only the dst
#    background cells after checking its visible pixels already match the true
#    composite.
# b) Complexity calibration:
#    The generator keeps only one new degree of freedom for role assignment:
#    among the three non-cue positions, exactly one is reserved for dst and the
#    other two must have distinct distances to the cue. This avoids hidden
#    tie-breaking while still allowing the cue to appear at any of the four
#    indices. Prefill is limited to 3..6 cells so the dst is visibly incomplete.
# c) Implicit-assumption audit against the base solver:
#    - Fixed tuple unpacking `t0,t1,t2,t3` is invalid once cue/dst move.
#    - "The output panel is the unique blank panel" is invalid because D5 adds
#      correct visible pixels to the dst.
#    - Background can no longer be read from a hardcoded source index; it must be
#      recovered from the tiles themselves and checked to be shared.
#    - Overwriting the whole dst panel without checking visible content is no
#      longer safe; non-bg dst pixels must already equal the computed answer.
#    The flood-fill and wall blending logic itself remains unchanged after role
#    recovery.
# d) Placement legality:
#    The generator still uses the original 7x25 border-separated geometry, so no
#    object can overlap or leave bounds. The dst prefill is copied from a fully
#    computed legal output tile, so every prefilled cell is in-bounds and correct.
# e) Visual observability:
#    Across examples a human can spot:
#    - one panel with the special wall+indicator cue structure;
#    - two panels with clear diagonal symmetry;
#    - one remaining panel with only a few already-correct colored pixels.
#    Because the cue moves across all four slots and source roles depend on
#    cue-relative distance, the meta-change is visible from input/output pairs.
# f) Visual cleanliness:
#    No extra marker colors are introduced. The dst prefill uses the same two
#    source colors as the final answer and stays sparse, so the grid does not
#    become cluttered.
# g) Color inferability:
#    Color ids never determine order. The only color facts are structural:
#    border differs from tile background, cue uses one single non-bg color, and
#    source/dst colors are inherited from the two source tiles.
# h) Variant metric:
#    Generated samples should satisfy:
#    - cue_idx is not fixed;
#    - `1 < prefilled_nonbg < final_nonbg`;
#    - the dst prefill is not D1+D2-symmetric;
#    - the two source panels are D1+D2-symmetric;
#    - source distances to cue are distinct.
# i) New solver assumptions and generator guarantees:
#    - Assumption: exactly one tile parses as a cue under its local majority bg.
#      Guarantee: cue has exactly 6 one-color pixels; dst prefill always uses both
#      source colors and sources have 8..12 pixels, so no other tile can match.
#    - Assumption: among non-cue panels, exactly one is not D1+D2-symmetric.
#      Guarantee: both sources are sampled from the original symmetric orbit
#      family and the dst prefill is rejected unless it breaks that symmetry.
#    - Assumption: the two source panels have distinct distances to the cue.
#      Guarantee: dst index is sampled only from layouts that leave distinct
#      distances for the remaining two source positions.


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_TILES = 4
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
GEN_RETRIES = 300
LBEND_PROB = 0.2
PREFILL_MIN = 3
PREFILL_MAX = 6
VALID_INDICATORS = {
    (0, 0),
    (0, 4),
    (4, 0),
    (4, 4),
    (0, 2),
    (2, 0),
    (2, 4),
    (4, 2),
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


def _panel_slice(j: int) -> tuple[slice, slice]:
    return slice(1, 6), slice(1 + 6 * j, 1 + 6 * j + TILE_SZ)


def _extract_tiles(grid: np.ndarray) -> list[np.ndarray]:
    return [grid[_panel_slice(j)].copy() for j in range(NUM_TILES)]


def _wall_reflection(indicator: tuple[int, int]):
    ir, ic = indicator
    dr, dc = 2 - ir, 2 - ic
    if dr == 0:
        return lambda r, c: (4 - r, c)
    if dc == 0:
        return lambda r, c: (r, 4 - c)
    if dr * dc > 0:
        return lambda r, c: (c, r)
    return lambda r, c: (4 - c, 4 - r)


def _is_d1d2_symmetric(tile: np.ndarray) -> bool:
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v = int(tile[r, c])
            if int(tile[c, r]) != v:
                return False
            if int(tile[4 - c, 4 - r]) != v:
                return False
    return True


def _parse_cue(tile: np.ndarray, bg: int) -> tuple[tuple[int, int], frozenset[tuple[int, int]]] | None:
    cue_pixels = {(int(r), int(c)) for r in range(TILE_SZ) for c in range(TILE_SZ) if int(tile[r, c]) != bg}
    if len(cue_pixels) != 6:
        return None
    cue_colors = {int(tile[r, c]) for r, c in cue_pixels}
    if len(cue_colors) != 1:
        return None

    indicator = None
    for r, c in cue_pixels:
        if not any((nr, nc) in cue_pixels for nr, nc in neighbors8(r, c)):
            if indicator is not None:
                return None
            indicator = (r, c)
    if indicator is None or indicator not in VALID_INDICATORS:
        return None

    wall = frozenset(cue_pixels - {indicator})
    if (2, 2) not in wall:
        return None

    refl = _wall_reflection(indicator)
    if set(wall) != {refl(r, c) for r, c in wall}:
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
    return indicator, wall


def _find_cue_tile(tiles: list[np.ndarray]) -> tuple[int, int, tuple[int, int], frozenset[tuple[int, int]]]:
    matches: list[tuple[int, int, tuple[int, int], frozenset[tuple[int, int]]]] = []
    for idx, tile in enumerate(tiles):
        bg = get_bg(tile)
        parsed = _parse_cue(tile, bg)
        if parsed is not None:
            indicator, wall = parsed
            matches.append((idx, bg, indicator, wall))
    assert len(matches) == 1, "expected exactly one cue panel"
    return matches[0]


def _find_dst_and_sources(tiles: list[np.ndarray], cue_idx: int) -> tuple[int, list[int]]:
    remaining = [idx for idx in range(NUM_TILES) if idx != cue_idx]
    dst_candidates = [idx for idx in remaining if not _is_d1d2_symmetric(tiles[idx])]
    assert len(dst_candidates) == 1, "expected exactly one non-symmetric dst panel"
    dst_idx = dst_candidates[0]
    source_indices = [idx for idx in remaining if idx != dst_idx]
    assert len(source_indices) == 2, "expected exactly two source panels"
    assert all(_is_d1d2_symmetric(tiles[idx]) for idx in source_indices), "source panels must be symmetric"
    return dst_idx, source_indices


def _source_role_indices(cue_idx: int, dst_idx: int) -> tuple[int, int]:
    source_indices = [idx for idx in range(NUM_TILES) if idx not in (cue_idx, dst_idx)]
    ranked = sorted((abs(idx - cue_idx), idx) for idx in source_indices)
    assert len(ranked) == 2, "expected exactly two source panels"
    assert ranked[0][0] != ranked[1][0], "ambiguous source-role distances"
    return ranked[0][1], ranked[1][1]


def _flood_inside(wall: frozenset[tuple[int, int]], indicator: tuple[int, int]) -> set[tuple[int, int]]:
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    return {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}


def _wall_is_4_connected(wall: frozenset[tuple[int, int]]) -> bool:
    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    return n_comp == 1


def _compose_tile(
    tile_bg: int,
    tile1: np.ndarray,
    tile2: np.ndarray,
    inside: set[tuple[int, int]],
    wall: frozenset[tuple[int, int]],
) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    wall_4conn = _wall_is_4_connected(wall)
    out = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    from_t1: set[tuple[int, int]] = set()
    from_t2: set[tuple[int, int]] = set()
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            pos = (r, c)
            v1, v2 = int(tile1[r, c]), int(tile2[r, c])
            if pos in inside:
                if v1 != tile_bg:
                    out[r, c] = v1
                    from_t1.add(pos)
            elif pos in wall:
                if v1 != tile_bg:
                    out[r, c] = v1
                    from_t1.add(pos)
                elif wall_4conn and v2 != tile_bg:
                    out[r, c] = v2
                    from_t2.add(pos)
            else:
                if v2 != tile_bg:
                    out[r, c] = v2
                    from_t2.add(pos)
    return out, from_t1, from_t2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two symmetric source tiles into a partially prefilled destination tile,
    with all panel roles inferred from structure instead of fixed positions.

    Input structure:
    - Four 5×5 panels in one row, border-separated inside a fixed 7×25 frame.
    - Exactly one panel is the cue: a 5-pixel wall through the center plus one
      8-isolated indicator pixel, all in a single color.
    - Exactly two non-cue panels are D1+D2-symmetric source patterns.
    - The remaining panel is the destination, already prefilled with some correct
      non-background output pixels.

    Rule:
    - Find the cue structurally, then identify the unique non-symmetric non-cue
      panel as the destination.
    - Among the two symmetric panels, the one nearer to the cue index is tile1
      (inside/wall-preferred) and the farther one is tile2 (outside/wall-fallback).
    - Flood-fill from the indicator, blocked by the wall, to obtain the inside
      region. Compose the full destination exactly as in the base task, and fill
      only the destination background cells after validating existing prefilled cells.

    Key insight:
    - The output panel is no longer the blank one. Role recovery now combines
      cue geometry, symmetry class, and cue-relative distance before composition.
    """
    g = np.array(grid, dtype=int)
    assert g.shape == (GRID_ROWS, GRID_COLS), f"unexpected grid shape {g.shape}"

    border = int(g[0, 0])
    tiles = _extract_tiles(g)

    cue_idx, tile_bg, indicator, wall = _find_cue_tile(tiles)
    assert border != tile_bg, "border color must differ from tile background"
    for tile in tiles:
        assert get_bg(tile) == tile_bg, "all tiles must share the same background color"

    dst_idx, _source_candidates = _find_dst_and_sources(tiles, cue_idx)
    tile1_idx, tile2_idx = _source_role_indices(cue_idx, dst_idx)
    tile1, tile2, dst = tiles[tile1_idx], tiles[tile2_idx], tiles[dst_idx]

    inside = _flood_inside(wall, indicator)
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - set(wall)
    assert outside, "wall must partition the grid — outside region is empty"

    composite, _from_t1, _from_t2 = _compose_tile(tile_bg, tile1, tile2, inside, wall)

    out_tile = dst.copy()
    seen_prefill = 0
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            if int(out_tile[r, c]) != tile_bg:
                seen_prefill += 1
                assert int(out_tile[r, c]) == int(composite[r, c]), "prefilled dst pixel is inconsistent"
                assert int(composite[r, c]) != tile_bg, "prefilled dst pixel cannot sit on a bg output cell"
            else:
                out_tile[r, c] = composite[r, c]
    assert seen_prefill >= 1, "combo variant expects at least one prefilled dst pixel"

    out = g.copy()
    out[_panel_slice(dst_idx)] = out_tile
    return out


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
        pattern = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
        count = 0
        for orbit_idx in order:
            orbit = _ORBITS[orbit_idx]
            if count + len(orbit) > max_px:
                continue
            if rng.random() < ORBIT_INCLUDE_PROB:
                for r, c in orbit:
                    pattern[r, c] = fg
                count += len(orbit)
        if min_px <= count <= max_px:
            return pattern
    return None


def _valid_dst_positions(cue_idx: int) -> list[int]:
    valid = []
    for dst_idx in range(NUM_TILES):
        if dst_idx == cue_idx:
            continue
        if cue_idx == 0 and dst_idx == 3:
            continue
        try:
            _source_role_indices(cue_idx, dst_idx)
        except AssertionError:
            continue
        valid.append(dst_idx)
    return valid


def _choose_layout(rng: random.Random) -> tuple[int, int, int, int]:
    cue_idx = rng.randrange(NUM_TILES)
    dst_idx = rng.choice(_valid_dst_positions(cue_idx))
    tile1_idx, tile2_idx = _source_role_indices(cue_idx, dst_idx)
    return cue_idx, dst_idx, tile1_idx, tile2_idx


def _choose_prefill(
    out_tile: np.ndarray,
    tile_bg: int,
    from_t1: set[tuple[int, int]],
    from_t2: set[tuple[int, int]],
    rng: random.Random,
) -> np.ndarray | None:
    all_fg = [(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ) if int(out_tile[r, c]) != tile_bg]
    if len(all_fg) < PREFILL_MIN + 1:
        return None
    if not from_t1 or not from_t2:
        return None

    limit = min(PREFILL_MAX, len(all_fg) - 1)
    if limit < PREFILL_MIN:
        return None

    for _ in range(40):
        need = rng.randint(PREFILL_MIN, limit)
        chosen = {
            rng.choice(sorted(from_t1)),
            rng.choice(sorted(from_t2)),
        }
        remaining = [pos for pos in all_fg if pos not in chosen]
        rng.shuffle(remaining)
        for pos in remaining:
            if len(chosen) >= need:
                break
            chosen.add(pos)

        if len(chosen) != need:
            continue

        dst_in = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
        for r, c in chosen:
            dst_in[r, c] = int(out_tile[r, c])

        if _is_d1d2_symmetric(dst_in):
            continue
        return dst_in
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color = rng.sample(other, 3)
    border = rng.choice(other)

    pool = _LBEND_CONFIGS if rng.random() < LBEND_PROB else _STRAIGHT_CONFIGS
    indicator, wall = rng.choice(pool)
    inside = _flood_inside(wall, indicator)

    tile1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    tile2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if tile1 is None or tile2 is None:
        return None

    cue_tile = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in wall:
        cue_tile[r, c] = cue_color
    cue_tile[indicator[0], indicator[1]] = cue_color

    out_tile, from_t1, from_t2 = _compose_tile(tile_bg, tile1, tile2, inside, wall)
    dst_in = _choose_prefill(out_tile, tile_bg, from_t1, from_t2, rng)
    if dst_in is None:
        return None

    cue_idx, dst_idx, tile1_idx, tile2_idx = _choose_layout(rng)
    tiles: list[np.ndarray | None] = [None] * NUM_TILES
    tiles[cue_idx] = cue_tile
    tiles[dst_idx] = dst_in
    tiles[tile1_idx] = tile1
    tiles[tile2_idx] = tile2

    g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
    for idx, tile in enumerate(tiles):
        assert tile is not None
        g[_panel_slice(idx)] = tile

    expected = g.copy()
    expected[_panel_slice(dst_idx)] = out_tile
    got = main(g)
    if not np.array_equal(got, expected):
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
