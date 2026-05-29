# Variant of 7491f3cf
# Dimension: D16 — 面板顺序与角色分配 (meta)
# Change: 固定顺序：面板0=cue，面板1=tile1(inside源)，面板2=tile2(outside源)，面板3=输出... → cue与源图案合并——3面板布局

from __future__ import annotations

import random

import numpy as np

from arc_utils import flood_fill, get_bg, label_mask, neighbors8


# -----------------------------------------------------------------------------
# Design analysis for the D16 variant
#
# variant_spec:
#   layout:
#     original: 4 panels, 7x25 grid, panel0=cue, panel1=t1, panel2=t2, panel3=output
#     variant:  3 panels, 7x19 grid, panel0=merged(cue+t1), panel1=t2, panel2=output
#   visible effect:
#     - The leftmost panel now contains two foreground roles at once:
#       cue_color = wall + isolated indicator, and t1_color = the inside source pattern.
#     - Humans must visually split the merged panel into "structural cue" vs "drawable pattern".
#   behavior trace:
#     - Original: solver reads cue from panel0 and pattern from panel1 independently.
#     - Variant: solver must first identify which color in panel0 is the 6-pixel cue
#       (5-pixel wall through center + 1 isolated indicator), then treat the remaining
#       non-bg color in the same panel as tile1's source pattern.
#     - After that separation, the inside/outside compositing rule is unchanged.
#   complexity calibration:
#     - Keep the original small family of walls (straight or L-bend).
#     - Keep tile2 unchanged.
#     - Keep tile1 symmetric, but forbid it from occupying the cue's 6 cells so the merged
#       panel is readable and the t1 source is not destroyed by the overlay.
#     - Tile1 stays in the original 8-12 pixel range; this remains feasible because the
#       cue removes only a subset of symmetry orbits.
#   implicit-assumption audit against original main():
#     - `tiles = ... for j in range(4)` no longer holds; the variant has 3 panels.
#     - `t0, t1, t2, _t3 = tiles` no longer holds; cue and t1 are co-located.
#     - `b = mode(t1)` was safe before because panel1 was a pure source tile. In the merged
#       panel the background may not be the majority color, so background must be inferred
#       from a pure tile (`t2`).
#     - `cue_set = non-bg pixels of t0` no longer holds; non-bg in panel0 contains both cue
#       and pattern, so the solver must search color-by-color for the unique valid cue.
#   legality / generator guarantees:
#     - merged panel uses exactly three colors: bg, cue_color, t1_color.
#     - cue_color occupies exactly 6 cells and is disjoint from t1 cells.
#     - cue still satisfies the original structural constraints.
#     - panel1 (`t2`) and panel2 (blank output slot) keep the same bg as the merged panel.
#   visual observability:
#     - The new rule is visible in every sample because panel count drops from 4 to 3, and
#       the left panel always contains both a 6-cell cue and 8-12 pattern cells.
#     - The key metric is the number of non-cue pattern pixels visible in the merged panel;
#       generation keeps this in [8, 12], so the merged-role behavior cannot disappear.
#   new solver assumptions introduced here:
#     - Exactly one foreground color in the merged panel satisfies the cue structure.
#     - The remaining non-bg color in the merged panel is tile1's source color.
#     - Generator enforces both assumptions explicitly.
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
LBEND_PROB = 0.2

VALID_INDICATORS = {
    (0, 0), (0, 2), (0, 4),
    (2, 0),         (2, 4),
    (4, 0), (4, 2), (4, 4),
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


def _reflect_across_indicator_axis(indicator: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    ir, ic = indicator
    dr, dc = 2 - ir, 2 - ic
    if dr == 0:
        return 4 - r, c
    if dc == 0:
        return r, 4 - c
    if dr * dc > 0:
        return c, r
    return 4 - c, 4 - r


def _parse_cue(cue_set: set[tuple[int, int]]) -> tuple[tuple[int, int], set[tuple[int, int]], set[tuple[int, int]], bool] | None:
    if len(cue_set) != 6:
        return None

    isolated = []
    for r, c in cue_set:
        if not any((nr, nc) in cue_set for nr, nc in neighbors8(r, c)):
            isolated.append((r, c))
    if len(isolated) != 1:
        return None

    indicator = isolated[0]
    if indicator not in VALID_INDICATORS:
        return None

    wall = cue_set - {indicator}
    if len(wall) != 5 or (2, 2) not in wall:
        return None

    if wall != {_reflect_across_indicator_axis(indicator, r, c) for r, c in wall}:
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

    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    if label_mask(wall_mask, connectivity=8)[1] != 1:
        return None

    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    inside = {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - wall
    if not inside or not outside:
        return None

    wall_4conn = (label_mask(wall_mask, connectivity=4)[1] == 1)
    return indicator, wall, inside, wall_4conn


def _extract_cue_from_merged(tile: np.ndarray, bg: int) -> tuple[int, tuple[int, int], set[tuple[int, int]], set[tuple[int, int]], bool]:
    candidates = []
    for color in sorted(set(map(int, tile.flatten())) - {int(bg)}):
        coords = {
            (int(r), int(c))
            for r in range(TILE_SZ)
            for c in range(TILE_SZ)
            if int(tile[r, c]) == int(color)
        }
        parsed = _parse_cue(coords)
        if parsed is not None:
            indicator, wall, inside, wall_4conn = parsed
            candidates.append((int(color), indicator, wall, inside, wall_4conn))
    assert len(candidates) == 1, f"expected exactly one cue color, found {len(candidates)}"
    return candidates[0]


def _sample_symmetric_pattern(
    bg: int,
    fg: int,
    rng: random.Random,
    *,
    min_px: int,
    max_px: int,
    blocked: set[tuple[int, int]] | frozenset[tuple[int, int]] = frozenset(),
) -> np.ndarray | None:
    allowed_orbits = [orbit for orbit in _ORBITS if not (set(orbit) & set(blocked))]
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


def _build_output_tile(
    merged: np.ndarray,
    t2: np.ndarray,
    *,
    bg: int,
    cue_color: int,
    wall: set[tuple[int, int]],
    inside: set[tuple[int, int]],
    wall_4conn: bool,
) -> np.ndarray:
    out = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1 = int(merged[r, c])
            v2 = int(t2[r, c])
            t1_here = v1 if v1 not in {bg, cue_color} else bg
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
    Variant solver for the 3-panel merged-cue layout.

    Input structure:
    - Three 5x5 panels in one row, border-separated.
    - Panel 0: merged tile = cue wall + indicator + tile1 source pattern.
    - Panel 1: tile2 source pattern.
    - Panel 2: blank output panel.

    Rule:
    - Identify the unique cue color in panel 0 by structure.
    - Treat the remaining non-bg color in panel 0 as tile1.
    - Composite inside/wall/outside exactly as in the original task.
    """
    g = np.array(grid, dtype=int)
    assert g.shape == (GRID_ROWS, GRID_COLS), f"expected {(GRID_ROWS, GRID_COLS)}, got {g.shape}"

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j: 1 + 6 * j + 5].copy() for j in range(PANEL_COUNT)]
    merged, t2, _out_in = tiles

    bg = int(get_bg(t2))
    assert border != bg, "border color must differ from tile background"

    cue_color, _indicator, wall, inside, wall_4conn = _extract_cue_from_merged(merged, bg)

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
    out[1:6, 1 + 6 * 2: 1 + 6 * 2 + 5] = out_tile
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color = rng.sample(other, 3)
    border = rng.choice(other)

    pool = _LBEND_CONFIGS if rng.random() < LBEND_PROB else _STRAIGHT_CONFIGS
    indicator, wall = rng.choice(pool)
    cue_cells = set(wall) | {indicator}

    t1 = _sample_symmetric_pattern(
        tile_bg,
        t1_color,
        rng,
        min_px=T1_MIN_PX,
        max_px=T1_MAX_PX,
        blocked=cue_cells,
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

    merged = t1.copy()
    for r, c in wall:
        merged[r, c] = cue_color
    merged[indicator[0], indicator[1]] = cue_color

    try:
        found_cue_color, found_indicator, found_wall, inside, wall_4conn = _extract_cue_from_merged(merged, tile_bg)
    except AssertionError:
        return None
    if found_cue_color != cue_color or found_indicator != indicator or found_wall != set(wall):
        return None

    out_tile = _build_output_tile(
        merged,
        t2,
        bg=tile_bg,
        cue_color=cue_color,
        wall=set(wall),
        inside=inside,
        wall_4conn=wall_4conn,
    )

    out_in = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
    for j, tile in enumerate([merged, t2, out_in]):
        g[1:6, 1 + 6 * j: 1 + 6 * j + 5] = tile

    expected = g.copy()
    expected[1:6, 1 + 6 * 2: 1 + 6 * 2 + 5] = out_tile
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
