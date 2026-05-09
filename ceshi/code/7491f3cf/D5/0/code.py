# Variant of 7491f3cf
# Dimension: D5 — 输出目标面板 (dst)
# Change: 面板3为全背景色的空白面板，需要填充合成结果... → 预填充部分内容的目标面板

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import flood_fill, label_mask, neighbors8


# Design analysis for the D5 variant
# ==================================
# a) Concrete behavior:
#    The 7x25 layout and cue logic stay unchanged. The only visual change is tile3:
#    it is no longer a pure 5x5 background block. Instead, tile3 already contains a
#    sparse subset of the final composite's non-background pixels. These are not new
#    hints or distractors; they are correct output pixels placed in their final
#    positions. The solver must therefore treat tile3 as a partially solved
#    destination, verify the visible pixels are consistent with the composition rule,
#    and fill the remaining background cells.
#    Small trace:
#    - Original: tile3 input = all bg, compose out3 from t1/t2, write whole out3.
#    - Variant: tile3 input = bg everywhere except, say, 4 correct pixels already
#      copied from out3. The solver computes the same full composite, checks that the
#      4 visible pixels agree with it, then fills the rest. The output is still the
#      full composite, but the input now visibly shows partial completion.
# b) Complexity calibration:
#    If only one pixel were prefilled, the change would be easy to miss. If nearly all
#    pixels were prefilled, the task would collapse into trivial clean-up. The generator
#    therefore pre-fills a small visible subset (3..6 cells, capped below the full
#    answer) and requires contributions from both source sides whenever possible.
# c) Original assumptions reviewed:
#    - `t3` was ignored entirely, so "blank destination" was an implicit assumption.
#    - Overwriting every destination cell was harmless only because all cells started
#      as background.
#    In this variant those assumptions are no longer neutral. The new solver computes
#    the expected composite, then checks every non-background pixel already present in
#    tile3. Any visible prefill must exactly match the expected composite at that
#    coordinate. Only background cells are treated as missing and filled in.
# d) Placement legality:
#    Tile geometry, wall geometry, and source-pattern generation remain identical to
#    the base task, so no new overlap risk is introduced. Prefilled tile3 pixels are
#    copied from the already validated final composite, so they are always in-bounds
#    and semantically legal.
# e) Visual observability:
#    Across examples, tile3 visibly contains a few colored cells before solving. They
#    sit in the same positions and colors they will occupy in the output, making the
#    "fill the partially completed target" rule visible in 2-3 examples.
# f) Visual cleanliness:
#    No extra marker color is introduced. Prefill uses the same two source colors as
#    the true answer, avoiding clutter. Because tile3 stays mostly background, the
#    partial-answer state remains legible.
# g) Color inferability:
#    There is no new color mapping. Prefilled cells inherit the ordinary composition
#    colors, so the human only needs to infer "some result pixels are already shown".
# h) Variant metric:
#    For generated inputs, `prefilled_nonbg = count(tile3_input != bg)` must satisfy
#    `1 < prefilled_nonbg < count(tile3_output != bg)`, and both source colors should
#    appear in the prefill when the output contains both.
# i) New solver assumptions and generator guarantees:
#    - Assumption: any non-bg pixel already in tile3 is a correct final pixel.
#      Guarantee: generator copies prefilled pixels directly from `out3`.
#    - Assumption: tile3 is still background-majority so bg remains visually obvious.
#      Guarantee: generator limits prefill to at most 6 cells in a 5x5 tile.


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
GEN_RETRIES = 300
LBEND_PROB = 0.2
PREFILL_MIN = 3
PREFILL_MAX = 6

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


def _flood_inside(wall: frozenset[tuple[int, int]], indicator: tuple[int, int]) -> set[tuple[int, int]]:
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    return {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}


def _compose_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    tile_bg: int,
    inside: set[tuple[int, int]],
    wall: set[tuple[int, int]],
) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    wall_4conn = n_comp == 1

    out3 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    from_t1: set[tuple[int, int]] = set()
    from_t2: set[tuple[int, int]] = set()
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1, v2 = int(t1[r, c]), int(t2[r, c])
            pos = (r, c)
            if pos in inside:
                if v1 != tile_bg:
                    out3[r, c] = v1
                    from_t1.add(pos)
            elif pos in wall:
                if v1 != tile_bg:
                    out3[r, c] = v1
                    from_t1.add(pos)
                elif wall_4conn and v2 != tile_bg:
                    out3[r, c] = v2
                    from_t2.add(pos)
            else:
                if v2 != tile_bg:
                    out3[r, c] = v2
                    from_t2.add(pos)
    return out3, from_t1, from_t2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Same composition rule as 7491f3cf, but tile3 is a partially prefilled dst tile.

    Visible non-background pixels already present in tile3 must be correct output
    pixels; the solver fills the remaining background cells.
    """
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, t3 = tiles
    tile_bg = int(Counter(t1.flatten()).most_common(1)[0][0])
    assert border != tile_bg, "border color must differ from tile background"

    t0_bg = int(Counter(t0.flatten()).most_common(1)[0][0])
    cue_set = {(int(r), int(c)) for r in range(5) for c in range(5) if int(t0[r, c]) != t0_bg}
    assert cue_set, "no cue pixels found"

    indicator = None
    for r, c in cue_set:
        if not any((nr, nc) in cue_set for nr, nc in neighbors8(r, c)):
            assert indicator is None, "multiple 8-isolated cue pixels"
            indicator = (r, c)
    assert indicator is not None, "no 8-isolated cue pixel found"

    valid_pos = {(0, 0), (0, 4), (4, 0), (4, 4), (0, 2), (2, 0), (2, 4), (4, 2)}
    assert indicator in valid_pos, f"indicator {indicator} not at corner/edge-midpoint"

    wall = cue_set - {indicator}
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
    assert wall == {refl(r, c) for r, c in wall}, "wall must be symmetric about indicator→center axis"

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

    inside = _flood_inside(frozenset(wall), indicator)
    outside = {(r, c) for r in range(5) for c in range(5)} - inside - wall
    assert outside, "wall must partition the grid — outside region is empty"

    composite, _from_t1, _from_t2 = _compose_tile(t1, t2, tile_bg, inside, wall)

    out3 = t3.copy()
    seen_prefill = 0
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            if int(out3[r, c]) != tile_bg:
                seen_prefill += 1
                assert int(out3[r, c]) == int(composite[r, c]), "prefilled dst pixel is inconsistent"
                assert int(composite[r, c]) != tile_bg, "prefilled dst pixel cannot sit on a bg output cell"
            else:
                out3[r, c] = composite[r, c]
    assert seen_prefill >= 1, "D5 variant expects at least one prefilled dst pixel"

    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _choose_prefill(
    out3: np.ndarray,
    tile_bg: int,
    from_t1: set[tuple[int, int]],
    from_t2: set[tuple[int, int]],
    rng: random.Random,
) -> np.ndarray | None:
    all_fg = [(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ) if int(out3[r, c]) != tile_bg]
    if len(all_fg) < PREFILL_MIN + 1:
        return None
    if not from_t1 or not from_t2:
        return None

    limit = min(PREFILL_MAX, len(all_fg) - 1)
    if limit < PREFILL_MIN:
        return None

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
        return None

    t3_in = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in chosen:
        t3_in[r, c] = int(out3[r, c])
    return t3_in


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color = rng.sample(other, 3)
    border = rng.choice(other)

    pool = _LBEND_CONFIGS if rng.random() < LBEND_PROB else _STRAIGHT_CONFIGS
    indicator, wall = rng.choice(pool)
    inside = _flood_inside(wall, indicator)

    t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = cue_color
    t0[indicator[0], indicator[1]] = cue_color

    out3, from_t1, from_t2 = _compose_tile(t1, t2, tile_bg, inside, set(wall))
    t3_in = _choose_prefill(out3, tile_bg, from_t1, from_t2, rng)
    if t3_in is None:
        return None

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
