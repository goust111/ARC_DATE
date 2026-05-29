# Variant of 7491f3cf
# Dimension: D6 — 合成变换规则 (transform)
# Change: 区域像素合成——inside区域取tile1非背景像素，outside区域取tile2非背景像素，墙上取tile1优先、... → 分区颜色互换

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import flood_fill, label_mask, neighbors8


"""
design_analysis:
  a_specific_behavior:
    - Layout stays unchanged: four 5x5 tiles in a row inside a fixed 7x25 border frame.
    - Tile0 still contains only the cue wall and the isolated indicator; tile1 and tile2 are
      still single-color source patterns; tile3 input stays blank.
    - The changed behavior is only in tile3 output:
      inside region   -> use tile2's shape mask, but paint those pixels with tile1's color
      outside region  -> use tile1's shape mask, but paint those pixels with tile2's color
      wall pixels     -> keep the original wall special case structurally, but swap shape/color
                         roles in the same spirit: tile2 shape in tile1 color, tile1 fallback
                         in tile2 color only when the wall is 4-connected.
    - Concrete trace:
      if a cell is inside and tile2 has a foreground pixel while tile1 does not, the original
      task leaves it blank but the variant paints it in tile1's color. If a cell is outside
      and tile1 has a foreground pixel while tile2 does not, the original task leaves it blank
      but the variant paints it in tile2's color. This is the visible signature of the swap.
  b_complexity_calibration:
    - Keep the original 5x5 objects and the original wall families so the task remains local
      and legible.
    - Keep one foreground color per source tile; otherwise the solver would need extra matching
      logic unrelated to D6.
    - To avoid degeneracy, the generator rejects cases where the new output is too close to the
      old output or where one swapped region receives no visible contribution.
  c_implicit_assumptions_review:
    - Original main() could copy raw source pixels directly; the variant cannot, because shape
      donor and color donor are separated.
    - The new solver therefore depends on tile1 and tile2 each having exactly one non-background
      color. The generator already uses that scheme, and main() now asserts it explicitly.
    - Using t1's majority color as the shared tile background was harmless before and remains
      valid because both source patterns still have 8-12 foreground cells on a 5x5 grid.
    - The old wall fallback only mattered because raw pixel colors came from the sources; now it
      still matters, but only as a shape-source fallback. The code must not copy v1/v2 directly.
  d_placement_legality:
    - No new objects are introduced. All source pixels remain confined to 5x5 tiles.
    - The wall configurations are reused from the original task; they already guarantee the wall
      passes through the center, touches the border, and yields non-empty inside/outside regions.
    - Output construction writes exactly one color per cell, so there is no overlap ambiguity.
  e_visual_observability:
    - Humans can compare tile3 against both tile1 and tile2: the shape fragment in one region
      matches one source tile, while the color matches the other.
    - The generator enforces a minimum Hamming distance between variant output and original-task
      output, plus exclusive evidence on both sides of the partition, so the new rule is not
      hidden behind accidentally similar source masks.
  f_visual_cleanliness:
    - Tile0 remains a pure cue tile, and tile3 uses only tile_bg plus the two source colors.
    - There is no stacking of cue pixels with source pixels inside the same panel.
  g_color_inference:
    - Only two non-background colors appear in the sources and output, and they stay tied to
      whole-region behavior rather than numeric color ids.
    - The wall/indicator gives the spatial partition cue, making the color swap inferable from
      2-4 examples instead of looking like arbitrary recoloring.
  new_main_assumptions_audit:
    - _single_fg_color() assumes exactly one foreground color in each source tile; generate_input()
      guarantees this by construction.
    - _compose_variant_tile() assumes only foreground membership matters for shape transfer; the
      generator guarantees binary single-color patterns.
    - The wall fallback branch assumes the original distinction between 4-connected and diagonal/
      disconnected walls is preserved; generation still samples both straight and L-bend walls.
"""


# ── Tunables ──────────────────────────────────────────────────────────────────
TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
PATTERN_PAIR_RETRIES = 80
GEN_RETRIES = 300
LBEND_PROB = 0.2

MIN_INSIDE_PAYLOAD = 2
MIN_OUTSIDE_PAYLOAD = 2
MIN_INSIDE_EXCLUSIVE = 1
MIN_OUTSIDE_EXCLUSIVE = 1
MIN_OUTPUT_DELTA = 4


# ── Symmetric pattern orbits (D1+D2 symmetry on 5x5) ─────────────────────────
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


# ── Precomputed (indicator, wall) configurations ─────────────────────────────
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


def _single_fg_color(tile: np.ndarray, bg: int, name: str) -> int:
    colors = sorted({int(v) for v in tile.flatten() if int(v) != bg})
    assert len(colors) == 1, f"{name} must contain exactly one non-background color"
    return int(colors[0])


def _flood_inside(wall: frozenset[tuple[int, int]], indicator: tuple[int, int]) -> set[tuple[int, int]]:
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


def _compose_original_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    inside: set[tuple[int, int]],
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
    wall_4conn: bool,
    t1_color: int,
    t2_color: int,
) -> np.ndarray:
    out3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            has1 = int(t1[r, c]) != bg
            has2 = int(t2[r, c]) != bg
            if (r, c) in inside:
                if has1:
                    out3[r, c] = t1_color
            elif (r, c) in wall:
                if has1:
                    out3[r, c] = t1_color
                elif wall_4conn and has2:
                    out3[r, c] = t2_color
            else:
                if has2:
                    out3[r, c] = t2_color
    return out3


def _compose_variant_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    inside: set[tuple[int, int]],
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
    wall_4conn: bool,
    t1_color: int,
    t2_color: int,
) -> np.ndarray:
    out3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            has1 = int(t1[r, c]) != bg
            has2 = int(t2[r, c]) != bg
            if (r, c) in inside:
                if has2:
                    out3[r, c] = t1_color
            elif (r, c) in wall:
                if has2:
                    out3[r, c] = t1_color
                elif wall_4conn and has1:
                    out3[r, c] = t2_color
            else:
                if has1:
                    out3[r, c] = t2_color
    return out3


def _variant_metrics(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    inside: set[tuple[int, int]],
    outside: set[tuple[int, int]],
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
    wall_4conn: bool,
    t1_color: int,
    t2_color: int,
) -> dict[str, int]:
    inside_payload = sum(int(t2[r, c]) != bg for r, c in inside)
    outside_payload = sum(int(t1[r, c]) != bg for r, c in outside)
    inside_exclusive = sum(
        (int(t2[r, c]) != bg) and (int(t1[r, c]) == bg) for r, c in inside
    )
    outside_exclusive = sum(
        (int(t1[r, c]) != bg) and (int(t2[r, c]) == bg) for r, c in outside
    )
    wall_primary = sum(int(t2[r, c]) != bg for r, c in wall)
    old_out = _compose_original_tile(t1, t2, bg, inside, wall, wall_4conn, t1_color, t2_color)
    new_out = _compose_variant_tile(t1, t2, bg, inside, wall, wall_4conn, t1_color, t2_color)
    return {
        "inside_payload": int(inside_payload),
        "outside_payload": int(outside_payload),
        "inside_exclusive": int(inside_exclusive),
        "outside_exclusive": int(outside_exclusive),
        "wall_primary": int(wall_primary),
        "delta_vs_original": int(np.sum(old_out != new_out)),
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
    cue_set = {(int(r), int(c)) for r in range(TILE_SZ) for c in range(TILE_SZ) if int(t0[r, c]) != t0_bg}
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
    assert t1_color != t2_color, "tile1 and tile2 must use different foreground colors"

    return {
        "g": g,
        "bg": bg,
        "tiles": tiles,
        "inside": inside,
        "outside": outside,
        "wall": wall,
        "wall_4conn": wall_4conn,
        "t1_color": t1_color,
        "t2_color": t2_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    parsed = _parse_problem(grid)
    g = parsed["g"]
    t0, t1, t2, _t3 = parsed["tiles"]
    _ = t0
    out3 = _compose_variant_tile(
        t1=t1,
        t2=t2,
        bg=parsed["bg"],
        inside=parsed["inside"],
        wall=parsed["wall"],
        wall_4conn=parsed["wall_4conn"],
        t1_color=parsed["t1_color"],
        t2_color=parsed["t2_color"],
    )
    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color = rng.sample(other, 3)
    border = rng.choice(other)

    pool = _LBEND_CONFIGS if rng.random() < LBEND_PROB else _STRAIGHT_CONFIGS
    indicator, wall = rng.choice(pool)
    inside = _flood_inside(wall, indicator)
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - wall
    wall_4conn = _wall_is_4conn(wall)

    for _ in range(PATTERN_PAIR_RETRIES):
        t1 = _sample_symmetric_pattern(
            tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX
        )
        t2 = _sample_symmetric_pattern(
            tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX
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
            t1_color=t1_color,
            t2_color=t2_color,
        )
        if metrics["inside_payload"] < MIN_INSIDE_PAYLOAD:
            continue
        if metrics["outside_payload"] < MIN_OUTSIDE_PAYLOAD:
            continue
        if metrics["inside_exclusive"] < MIN_INSIDE_EXCLUSIVE:
            continue
        if metrics["outside_exclusive"] < MIN_OUTSIDE_EXCLUSIVE:
            continue
        if metrics["delta_vs_original"] < MIN_OUTPUT_DELTA:
            continue

        t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
        for r, c in wall:
            t0[r, c] = cue_color
        t0[indicator[0], indicator[1]] = cue_color

        out3 = _compose_variant_tile(
            t1=t1,
            t2=t2,
            bg=tile_bg,
            inside=inside,
            wall=wall,
            wall_4conn=wall_4conn,
            t1_color=t1_color,
            t2_color=t2_color,
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
