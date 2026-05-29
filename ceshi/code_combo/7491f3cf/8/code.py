# Combo variant of 7491f3cf (index 8)
# combo_spec: D6/2+D11/2
# Name: 分区颜色互换+面板同色

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import flood_fill, label_mask, neighbors8


"""
design_analysis:
  a_specific_behavior:
    - Layout stays unchanged: four 5x5 tiles in one row inside the fixed 7x25 border frame.
    - Tile0 still contains only the cue wall and one isolated indicator. Tile3 input is still
      blank. The fusion acts only on how tile1 and tile2 are routed into tile3.
    - D6/2 says "swap shape donor and color donor by region":
      inside  -> tile2 shape in tile1 color
      outside -> tile1 shape in tile2 color
      wall    -> tile2 shape in tile1 color, with tile1 fallback on 4-connected walls.
    - D11/2 forces tile1 and tile2 to use the same foreground color. That collapses the color
      swap visually, so the observable rule becomes:
      inside  -> tile2 shape
      outside -> tile1 shape
      wall    -> tile2 shape, with tile1 fallback only on 4-connected walls
      and every kept pixel is painted in the single shared source color.
    - Concrete trace:
      suppose an inside cell is foreground only in tile2. Original task with same-color sources
      would leave that cell blank because it only consults tile1 inside. The fused variant fills
      it, because inside now consults tile2's shape. Likewise, if an outside cell is foreground
      only in tile1, the fused variant fills it while the original routing would leave it blank.
  b_complexity_calibration:
    - The task should not degenerate into "copy whichever source already matches everywhere".
      The generator therefore rejects samples unless tile2 contributes visible inside-only
      evidence and tile1 contributes visible outside-only evidence.
    - The source tiles stay single-color and symmetric like the original task, so the only new
      reasoning burden is donor-shape reassignment, not extra object parsing.
    - Thresholds stay modest to preserve generator yield while still forcing visible swap
      evidence in both regions and a nontrivial Hamming gap from same-color original routing.
  c_implicit_assumptions_review:
    - Original main() copied raw tile pixels; the combo can no longer infer provenance from color,
      so it must treat source tiles as binary shape masks plus one shared foreground color.
    - The solver therefore newly assumes tile1 and tile2 each have exactly one non-background
      color and that those colors are equal. The generator enforces both.
    - Majority-color background inference still works because each 5x5 source tile has at most
      12 foreground pixels, so background remains the majority.
    - The old wall branch relied on source colors only incidentally. In the combo it still matters
      structurally, but only as a shape-source priority rule.
  d_placement_legality:
    - No new objects are added. All reused wall configurations already fit in 5x5, pass through
      the center, and partition the tile under 4-connected flood fill.
    - Tile assembly remains fixed: four 5x5 panels separated by 1-cell border columns.
    - Output writes one final color per cell, so there is no overlap ambiguity.
  e_visual_observability:
    - Because tile1 and tile2 share a color, the only visible clue is which source shape appears
      on each side of the wall. The generator therefore enforces region-local exclusive evidence
      rather than relying on numeric color ids.
    - Across examples, a human can compare tile3's inside fragment against tile2 and tile3's
      outside fragment against tile1. That is enough to infer the fused rule in 2-4 samples.
  f_visual_cleanliness:
    - Tile0 remains visually separate from the source panels because cue color differs from the
      shared source color.
    - Tile3 uses only tile background plus one shared foreground color, so the panel stays clean
      instead of mixing multiple recoloring schemes.
  g_color_inference:
    - The combo intentionally creates an apparent paradox: a "color swap" rule with no visible
      recoloring because both sources already have the same color. This is still inferable because
      the partition cue explains the shape swap, and the absence of recoloring is consistent
      across all samples.
  new_main_assumptions_audit:
    - _single_fg_color() assumes one foreground color per source tile; generate_input() builds
      binary single-color patterns.
    - _parse_problem() assumes both source tiles share background and foreground colors; the
      generator enforces that directly.
    - _compose_combo_tile() assumes only foreground membership matters for source selection;
      pattern generation stays binary so no finer color structure is lost.
    - The wall fallback branch assumes the original distinction between 4-connected and diagonal
      walls still exists; generation still samples both straight and L-bend families.
"""


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
PATTERN_PAIR_RETRIES = 100
GEN_RETRIES = 300
LBEND_PROB = 0.2

MIN_INSIDE_PAYLOAD = 2
MIN_OUTSIDE_PAYLOAD = 2
MIN_INSIDE_EXCLUSIVE = 1
MIN_OUTSIDE_EXCLUSIVE = 1
MIN_OUTPUT_DELTA = 4
MIN_SHAPE_DELTA = 4


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
    bg: int, fg: int, rng: random.Random, *, min_px: int, max_px: int
) -> np.ndarray | None:
    for _ in range(PATTERN_RETRIES):
        order = list(range(len(_ORBITS)))
        rng.shuffle(order)
        tile = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
        count = 0
        for idx in order:
            orbit = _ORBITS[idx]
            if count + len(orbit) > max_px:
                continue
            if rng.random() < ORBIT_INCLUDE_PROB:
                for r, c in orbit:
                    tile[r, c] = fg
                count += len(orbit)
        if min_px <= count <= max_px:
            return tile
    return None


def _single_fg_color(tile: np.ndarray, bg: int, name: str) -> int:
    colors = sorted({int(v) for v in tile.flatten() if int(v) != bg})
    assert len(colors) == 1, f"{name} must contain exactly one non-background color"
    return int(colors[0])


def _flood_inside(
    wall: frozenset[tuple[int, int]], indicator: tuple[int, int]
) -> set[tuple[int, int]]:
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    return {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}


def _wall_is_4conn(wall: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    return bool(n_comp == 1)


def _compose_same_color_baseline_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    inside: set[tuple[int, int]],
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
    wall_4conn: bool,
    shared_color: int,
) -> np.ndarray:
    out3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            has1 = int(t1[r, c]) != bg
            has2 = int(t2[r, c]) != bg
            if (r, c) in inside:
                if has1:
                    out3[r, c] = shared_color
            elif (r, c) in wall:
                if has1:
                    out3[r, c] = shared_color
                elif wall_4conn and has2:
                    out3[r, c] = shared_color
            else:
                if has2:
                    out3[r, c] = shared_color
    return out3


def _compose_combo_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    inside: set[tuple[int, int]],
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
    wall_4conn: bool,
    shared_color: int,
) -> np.ndarray:
    out3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            has1 = int(t1[r, c]) != bg
            has2 = int(t2[r, c]) != bg
            if (r, c) in inside:
                if has2:
                    out3[r, c] = shared_color
            elif (r, c) in wall:
                if has2:
                    out3[r, c] = shared_color
                elif wall_4conn and has1:
                    out3[r, c] = shared_color
            else:
                if has1:
                    out3[r, c] = shared_color
    return out3


def _variant_metrics(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    inside: set[tuple[int, int]],
    outside: set[tuple[int, int]],
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
    wall_4conn: bool,
    shared_color: int,
) -> dict[str, int]:
    m1 = t1 != bg
    m2 = t2 != bg
    baseline = _compose_same_color_baseline_tile(
        t1=t1,
        t2=t2,
        bg=bg,
        inside=inside,
        wall=wall,
        wall_4conn=wall_4conn,
        shared_color=shared_color,
    )
    variant = _compose_combo_tile(
        t1=t1,
        t2=t2,
        bg=bg,
        inside=inside,
        wall=wall,
        wall_4conn=wall_4conn,
        shared_color=shared_color,
    )
    return {
        "inside_payload": int(sum(bool(m2[r, c]) for r, c in inside)),
        "outside_payload": int(sum(bool(m1[r, c]) for r, c in outside)),
        "inside_exclusive": int(sum(bool(m2[r, c] and not m1[r, c]) for r, c in inside)),
        "outside_exclusive": int(sum(bool(m1[r, c] and not m2[r, c]) for r, c in outside)),
        "shape_delta": int(np.sum(m1 != m2)),
        "delta_vs_baseline": int(np.sum(baseline != variant)),
    }


def _parse_problem(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, _t3 = tiles

    bg = int(Counter(t1.flatten()).most_common(1)[0][0])
    assert border != bg, "border color must differ from tile background"
    assert int(Counter(t2.flatten()).most_common(1)[0][0]) == bg, "tile1/tile2 must share background"

    t0_bg = int(Counter(t0.flatten()).most_common(1)[0][0])
    assert t0_bg == bg, "cue tile must share the same background"
    cue_set = {
        (int(r), int(c))
        for r in range(TILE_SZ)
        for c in range(TILE_SZ)
        if int(t0[r, c]) != t0_bg
    }
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
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - wall
    assert outside, "wall must partition the grid — outside region is empty"

    wall_4conn = _wall_is_4conn(wall)
    t1_color = _single_fg_color(t1, bg, "tile1")
    t2_color = _single_fg_color(t2, bg, "tile2")
    assert t1_color == t2_color, "tile1 and tile2 must share the same foreground color"

    return {
        "g": g,
        "bg": bg,
        "tiles": tiles,
        "inside": inside,
        "outside": outside,
        "wall": wall,
        "wall_4conn": wall_4conn,
        "shared_color": t1_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two same-colored pattern tiles into a fourth panel using flood-fill from an indicator pixel.

    Input structure:
    - Four 5×5 panels in one row, border-separated.
    - Panel 0: cue pixels = wall boundary (8-connected, edge-to-edge, through center)
               + one 8-isolated indicator pixel.
    - Panels 1–2: source patterns with the same single foreground color but different shapes.
      Panel 3 is blank and must be filled.

    Rule:
    - Flood-fill (4-conn) from indicator, blocked by wall → inside region.
    - Inside → tile2's non-bg shape, painted in the shared source color.
    - Outside → tile1's non-bg shape, painted in the shared source color.
    - Wall → tile2 shape first; if the wall is 4-connected and tile2 is blank there, tile1
      can fill as fallback.

    Key insight:
    - The combo starts from a shape/color swap, but tile1 and tile2 already use the same
      foreground color. So the visible effect is not recoloring; it is a partitioned shape
      swap: inside matches tile2, outside matches tile1.
    """
    parsed = _parse_problem(grid)
    g = parsed["g"]
    _t0, t1, t2, _t3 = parsed["tiles"]

    out3 = _compose_combo_tile(
        t1=t1,
        t2=t2,
        bg=parsed["bg"],
        inside=parsed["inside"],
        wall=parsed["wall"],
        wall_4conn=parsed["wall_4conn"],
        shared_color=parsed["shared_color"],
    )
    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, shared_color = rng.sample(other, 2)
    border = rng.choice(other)

    pool = _LBEND_CONFIGS if rng.random() < LBEND_PROB else _STRAIGHT_CONFIGS
    indicator, wall = rng.choice(pool)
    inside = _flood_inside(wall, indicator)
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - wall
    wall_4conn = _wall_is_4conn(wall)

    for _ in range(PATTERN_PAIR_RETRIES):
        t1 = _sample_symmetric_pattern(
            tile_bg, shared_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX
        )
        t2 = _sample_symmetric_pattern(
            tile_bg, shared_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX
        )
        if t1 is None or t2 is None:
            continue

        metrics = _variant_metrics(
            t1=t1,
            t2=t2,
            bg=tile_bg,
            inside=inside,
            outside=outside,
            wall=wall,
            wall_4conn=wall_4conn,
            shared_color=shared_color,
        )
        if metrics["shape_delta"] < MIN_SHAPE_DELTA:
            continue
        if metrics["inside_payload"] < MIN_INSIDE_PAYLOAD:
            continue
        if metrics["outside_payload"] < MIN_OUTSIDE_PAYLOAD:
            continue
        if metrics["inside_exclusive"] < MIN_INSIDE_EXCLUSIVE:
            continue
        if metrics["outside_exclusive"] < MIN_OUTSIDE_EXCLUSIVE:
            continue
        if metrics["delta_vs_baseline"] < MIN_OUTPUT_DELTA:
            continue

        t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
        for r, c in wall:
            t0[r, c] = cue_color
        t0[indicator[0], indicator[1]] = cue_color

        out3 = _compose_combo_tile(
            t1=t1,
            t2=t2,
            bg=tile_bg,
            inside=inside,
            wall=wall,
            wall_4conn=wall_4conn,
            shared_color=shared_color,
        )

        t3_in = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
        g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
        for j, tile in enumerate([t0, t1, t2, t3_in]):
            g[1:6, 1 + 6 * j : 1 + 6 * j + 5] = tile

        expected = g.copy()
        expected[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
        got = main(g)
        if np.array_equal(got, expected):
            return g
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
