# Variant of 7491f3cf
# Dimension: D16 — 面板顺序与角色分配 (meta)
# Change: 固定顺序：面板0=cue，面板1=tile1(inside源)，面板2=tile2(outside源)，面板3=输出... → 面板顺序可变——由cue面板的位置决定

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import flood_fill, get_bg, label_mask, neighbors8


# Design analysis for D16
# ----------------------
# a) Concrete behavior and trace
#    - The 7x25 canvas still contains four 5x5 panels in one row with 1-cell borders.
#    - Exactly one panel is still the cue panel: 5-pixel wall + 1 isolated indicator.
#    - Exactly one panel is still blank: all 25 cells are the shared tile background.
#    - The other two panels are still the source patterns.
#    - New visible change: the cue panel may appear at panel index 0/1/2/3 instead of
#      always index 0. The blank output panel may also appear at any other index.
#    - Role assignment is no longer "panel 1 is inside source, panel 2 is outside source".
#      Instead, after locating the cue panel, compare the two source panels by linear
#      distance to the cue index:
#        nearer source -> tile1 (inside / wall-preferred)
#        farther source -> tile2 (outside / wall-fallback)
#    - Example trace:
#      layout indices = [blank, cue, far-source, near-source] is invalid for the new
#      generator because the two sources would both be distance 1 from the cue only in
#      some placements; the variant avoids such ambiguous layouts.
#      A valid example is [source-near, cue, blank, source-far]:
#        original code would wrongly read panel 0 as cue and panel 3 as output;
#        new code first finds blank at index 2, cue at index 1, then treats index 0
#        as tile1 because |0-1|=1 and index 3 as tile2 because |3-1|=2.
# b) Complexity calibration
#    - The variant changes a meta-structure that is easy to see across examples: cue
#      moves, blank moves, and source roles follow cue-relative distance.
#    - To avoid hidden tie-breaking, the generator only emits layouts where the two
#      source distances to the cue are distinct.
#    - Cue index is sampled from all four positions, so the change does not collapse
#      back to the original left-to-right convention.
# c) Implicit-assumption audit against original main()
#    - `t0, t1, t2, _t3 = tiles` assumed fixed roles by index; this is invalid now.
#    - `b = most_common(t1)` assumed panel 1 is a source tile with the shared bg; now
#      the background must be read from the blank panel, which is role-identifiable.
#    - Writing output into fixed panel 3 is invalid; output must overwrite whichever
#      panel is the blank tile.
#    - All flood-fill, wall checks, and composite logic stay valid once the cue/source
#      roles are recovered correctly.
# d) Placement legality
#    - Tiles remain 5x5, border layout remains unchanged, so no overlap risk is added.
#    - The generator enforces exactly one blank tile, exactly one cue tile, and two
#      source tiles with distinct distances to the cue.
# e) Visual observability
#    - Humans can detect the blank tile immediately because it is monochrome.
#    - Humans can detect the cue panel by its wall+indicator structure.
#    - With 2-4 examples where the cue appears in multiple positions, the new rule
#      "source role depends on distance to cue" is visible from the input/output pairs.
# f) Visual cleanliness
#    - No extra marks are added. The grid stays as clean as the original task:
#      one cue panel, two pattern panels, one blank panel.
# g) Color inferability
#    - Colors still have the same function classes as the original task:
#      border, shared background, cue color, source color 1, source color 2.
#      No ordering depends on numeric color ids.
# New main() assumptions and generator guarantees
#    - Assumption: exactly one tile is fully blank. Generator guarantee: yes.
#    - Assumption: exactly one non-blank tile matches the cue geometry with 6 cue pixels.
#      Generator guarantee: yes.
#    - Assumption: among the two remaining source panels, distances to cue are distinct.
#      Generator guarantee: yes; ambiguous layouts are never sampled.


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
VALID_INDICATORS = {(0, 0), (0, 4), (4, 0), (4, 4), (0, 2), (2, 0), (2, 4), (4, 2)}


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


def _source_role_indices(cue_idx: int, blank_idx: int) -> tuple[int, int]:
    source_indices = [idx for idx in range(NUM_TILES) if idx not in (cue_idx, blank_idx)]
    ranked = sorted((abs(idx - cue_idx), idx) for idx in source_indices)
    assert len(ranked) == 2, "expected exactly two source panels"
    assert ranked[0][0] != ranked[1][0], "ambiguous source-role distances"
    return ranked[0][1], ranked[1][1]


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


def _parse_cue(tile: np.ndarray, bg: int) -> tuple[tuple[int, int], frozenset[tuple[int, int]]] | None:
    if get_bg(tile) != bg:
        return None
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


def _find_blank_tile(tiles: list[np.ndarray]) -> tuple[int, int]:
    candidates = [idx for idx, tile in enumerate(tiles) if np.unique(tile).size == 1]
    assert len(candidates) == 1, "expected exactly one blank panel"
    blank_idx = candidates[0]
    return blank_idx, int(tiles[blank_idx][0, 0])


def _find_cue_tile(tiles: list[np.ndarray], bg: int, blank_idx: int) -> tuple[int, tuple[int, int], frozenset[tuple[int, int]]]:
    matches: list[tuple[int, tuple[int, int], frozenset[tuple[int, int]]]] = []
    for idx, tile in enumerate(tiles):
        if idx == blank_idx:
            continue
        parsed = _parse_cue(tile, bg)
        if parsed is not None:
            indicator, wall = parsed
            matches.append((idx, indicator, wall))
    assert len(matches) == 1, "expected exactly one cue panel"
    return matches[0]


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


def _composite_tile(
    tile_bg: int,
    tile1: np.ndarray,
    tile2: np.ndarray,
    inside: set[tuple[int, int]],
    wall: frozenset[tuple[int, int]],
) -> np.ndarray:
    wall_4conn = _wall_is_4_connected(wall)
    out = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1, v2 = int(tile1[r, c]), int(tile2[r, c])
            if (r, c) in inside:
                if v1 != tile_bg:
                    out[r, c] = v1
            elif (r, c) in wall:
                if v1 != tile_bg:
                    out[r, c] = v1
                elif wall_4conn and v2 != tile_bg:
                    out[r, c] = v2
            else:
                if v2 != tile_bg:
                    out[r, c] = v2
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    assert g.shape == (GRID_ROWS, GRID_COLS), f"unexpected grid shape {g.shape}"

    border = int(g[0, 0])
    tiles = _extract_tiles(g)

    blank_idx, tile_bg = _find_blank_tile(tiles)
    assert border != tile_bg, "border color must differ from tile background"
    for tile in tiles:
        assert get_bg(tile) == tile_bg, "all tiles must share the same background color"

    cue_idx, indicator, wall = _find_cue_tile(tiles, tile_bg, blank_idx)
    tile1_idx, tile2_idx = _source_role_indices(cue_idx, blank_idx)
    tile1, tile2 = tiles[tile1_idx], tiles[tile2_idx]

    inside = _flood_inside(wall, indicator)
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - set(wall)
    assert outside, "wall must partition the grid — outside region is empty"

    out_tile = _composite_tile(tile_bg, tile1, tile2, inside, wall)
    out = g.copy()
    out[_panel_slice(blank_idx)] = out_tile
    return out


def _sample_symmetric_pattern(bg: int, fg: int, rng: random.Random, *, min_px: int, max_px: int) -> np.ndarray | None:
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


def _valid_blank_positions(cue_idx: int) -> list[int]:
    valid = []
    for blank_idx in range(NUM_TILES):
        if blank_idx == cue_idx:
            continue
        try:
            _source_role_indices(cue_idx, blank_idx)
        except AssertionError:
            continue
        valid.append(blank_idx)
    return valid


def _choose_layout(rng: random.Random) -> tuple[int, int, int, int]:
    cue_idx = rng.randrange(NUM_TILES)
    blank_idx = rng.choice(_valid_blank_positions(cue_idx))
    tile1_idx, tile2_idx = _source_role_indices(cue_idx, blank_idx)
    return cue_idx, blank_idx, tile1_idx, tile2_idx


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

    out_tile = _composite_tile(tile_bg, tile1, tile2, inside, wall)
    blank_tile = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)

    cue_idx, blank_idx, tile1_idx, tile2_idx = _choose_layout(rng)
    tiles = [None] * NUM_TILES
    tiles[cue_idx] = cue_tile
    tiles[blank_idx] = blank_tile
    tiles[tile1_idx] = tile1
    tiles[tile2_idx] = tile2

    g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
    for idx, tile in enumerate(tiles):
        assert tile is not None
        g[_panel_slice(idx)] = tile

    expected = g.copy()
    expected[_panel_slice(blank_idx)] = out_tile
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
