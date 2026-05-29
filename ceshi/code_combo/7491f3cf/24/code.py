# Combo variant of 7491f3cf (index 24)
# combo_spec: D5/0+D6/2+D16/0
# Name: 预填充目标+颜色互换+面板顺序可变

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import flood_fill, get_bg, label_mask, neighbors8


"""
design_analysis:
  a_specific_behavior:
    - The canvas remains a fixed 7x25 strip with four 5x5 panels.
    - Exactly one panel is the cue: 5 wall pixels plus 1 isolated indicator.
    - Exactly two panels are still single-foreground symmetric source patterns.
    - The remaining panel is the destination, but unlike the base task it is not
      blank: it already shows a sparse subset of the final answer.
    - Panel order is variable. The solver must first recover roles from content:
        destination = the only panel already mixing both source colors
        cue         = the only non-destination panel matching wall+indicator geometry
        tile1/tile2 = the two remaining panels, ranked by distance to the cue
      The source tile nearer to the cue is tile1, the farther one is tile2.
    - The fused transform is the D6 color/shape swap:
        inside region  -> tile2 shape painted with tile1 color
        outside region -> tile1 shape painted with tile2 color
        wall pixels    -> tile2 shape in tile1 color, with tile1-shape fallback in
                          tile2 color only when the wall is 4-connected
    - Concrete trace:
      suppose the cue is at panel index 1, destination at index 0, and the source
      panels at indices 2 and 3. Then tile1 is panel 2 and tile2 is panel 3 because
      their distances to the cue are 1 and 2. If an inside cell is occupied only in
      tile2, the base task would not necessarily color-match it this way; the combo
      paints it using tile1's color, and the destination may already show that exact
      pixel as a hint. The solver verifies the hint and fills the rest.
  b_complexity_calibration:
    - The generator always pre-fills destination pixels from both output colors. This
      makes the destination visibly distinct from cue/source panels, which each have
      only one non-background color.
    - Pattern pairs are rejected when the swapped output is too close to the base
      output or when one swapped side contributes too little. That keeps the D6
      behavior observable instead of accidentally degenerating.
    - Layouts with equal source-to-cue distances are rejected, so D16 role recovery
      is crisp and does not require hidden tie-breaking.
  c_implicit_assumptions_review:
    - Base `t0,t1,t2,t3 = tiles` indexing is invalid because panel order now varies.
      The fused solver must parse roles before composing anything.
    - D16's "blank tile identifies destination" assumption is invalid because D5
      turns the destination into a partially solved panel. The combo instead uses the
      stronger invariant that only the destination mixes both source colors.
    - Base direct pixel copying from tile1/tile2 is invalid because D6 separates
      shape donor and color donor. Composition must operate on foreground masks plus
      each source tile's unique foreground color.
    - The new destination-identification rule assumes both source colors appear in
      the prefill. The generator guarantees this explicitly.
  d_placement_legality:
    - Every object stays inside its original 5x5 panel, so no new geometric overlap
      mode is introduced.
    - Cue walls are sampled only from the original verified configuration set, so the
      wall still passes through the center and still partitions inside/outside.
    - Prefilled pixels are copied directly from a validated final composite, so they
      are always in-bounds and semantically correct.
  e_visual_observability:
    - Humans can find the cue by its wall+indicator geometry.
    - Humans can find the destination because it is the only panel already showing
      both source colors together.
    - Across examples, cue position changes, destination position changes, and source
      roles follow cue-relative distance, so the D16 meta-rule is visible.
    - The output makes the D6 rule legible because region membership controls whose
      shape is used and whose color is used, and the destination prefill already
      confirms that interpretation locally.
  f_visual_cleanliness:
    - No extra marker colors are introduced. The destination uses only background plus
      the two source colors.
    - Prefill remains sparse, so it reads as an incomplete answer rather than noise.
  g_color_inference:
    - Color change is not arbitrary recoloring. One source donates shape and the
      other donates color in region-dependent fashion, with the wall indicating the
      partition. The prefill exposes both colors in their final roles.
  new_main_assumptions_audit:
    - Assumption: exactly one panel has two distinct non-background colors. Generator
      guarantee: destination prefill always contains both output colors, while cue and
      source panels remain single-foreground.
    - Assumption: source panels each have exactly one non-background color. Generator
      guarantee: symmetric patterns are binary single-color masks by construction.
    - Assumption: the two source slots have distinct distances to the cue. Generator
      guarantee: ambiguous cue/destination layouts are never sampled.
    - Assumption: every visible destination hint matches the fused composite. Generator
      guarantee: all prefill pixels are copied from the final output tile.
"""


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_TILES = 4
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
PATTERN_PAIR_RETRIES = 80
GEN_RETRIES = 300
LBEND_PROB = 0.2
PREFILL_MIN = 3
PREFILL_MAX = 6

MIN_INSIDE_PAYLOAD = 2
MIN_OUTSIDE_PAYLOAD = 2
MIN_INSIDE_EXCLUSIVE = 1
MIN_OUTSIDE_EXCLUSIVE = 1
MIN_OUTPUT_DELTA = 4

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
    cue_pixels = {
        (int(r), int(c))
        for r in range(TILE_SZ)
        for c in range(TILE_SZ)
        if int(tile[r, c]) != bg
    }
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


def _find_destination_tile(tiles: list[np.ndarray], bg: int) -> int:
    candidates = []
    for idx, tile in enumerate(tiles):
        if get_bg(tile) != bg:
            continue
        fg_colors = {int(v) for v in tile.flatten() if int(v) != bg}
        if len(fg_colors) == 2:
            candidates.append(idx)
    assert len(candidates) == 1, "expected exactly one prefilled destination panel"
    return candidates[0]


def _find_cue_tile(
    tiles: list[np.ndarray],
    bg: int,
    dst_idx: int,
) -> tuple[int, tuple[int, int], frozenset[tuple[int, int]]]:
    matches: list[tuple[int, tuple[int, int], frozenset[tuple[int, int]]]] = []
    for idx, tile in enumerate(tiles):
        if idx == dst_idx:
            continue
        parsed = _parse_cue(tile, bg)
        if parsed is not None:
            indicator, wall = parsed
            matches.append((idx, indicator, wall))
    assert len(matches) == 1, "expected exactly one cue panel"
    return matches[0]


def _source_role_indices(cue_idx: int, dst_idx: int) -> tuple[int, int]:
    source_indices = [idx for idx in range(NUM_TILES) if idx not in (cue_idx, dst_idx)]
    ranked = sorted((abs(idx - cue_idx), idx) for idx in source_indices)
    assert len(ranked) == 2, "expected exactly two source panels"
    assert ranked[0][0] != ranked[1][0], "ambiguous source-role distances"
    return ranked[0][1], ranked[1][1]


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
    out = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            has1 = int(t1[r, c]) != bg
            has2 = int(t2[r, c]) != bg
            pos = (r, c)
            if pos in inside:
                if has1:
                    out[r, c] = t1_color
            elif pos in wall:
                if has1:
                    out[r, c] = t1_color
                elif wall_4conn and has2:
                    out[r, c] = t2_color
            else:
                if has2:
                    out[r, c] = t2_color
    return out


def _compose_combo_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    inside: set[tuple[int, int]],
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
    wall_4conn: bool,
    t1_color: int,
    t2_color: int,
) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    out = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    t1_color_positions: set[tuple[int, int]] = set()
    t2_color_positions: set[tuple[int, int]] = set()
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            has1 = int(t1[r, c]) != bg
            has2 = int(t2[r, c]) != bg
            pos = (r, c)
            if pos in inside:
                if has2:
                    out[r, c] = t1_color
                    t1_color_positions.add(pos)
            elif pos in wall:
                if has2:
                    out[r, c] = t1_color
                    t1_color_positions.add(pos)
                elif wall_4conn and has1:
                    out[r, c] = t2_color
                    t2_color_positions.add(pos)
            else:
                if has1:
                    out[r, c] = t2_color
                    t2_color_positions.add(pos)
    return out, t1_color_positions, t2_color_positions


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
        (int(t2[r, c]) != bg) and (int(t1[r, c]) == bg)
        for r, c in inside
    )
    outside_exclusive = sum(
        (int(t1[r, c]) != bg) and (int(t2[r, c]) == bg)
        for r, c in outside
    )
    old_out = _compose_original_tile(
        t1,
        t2,
        bg,
        inside,
        wall,
        wall_4conn,
        t1_color,
        t2_color,
    )
    new_out, t1_color_positions, t2_color_positions = _compose_combo_tile(
        t1,
        t2,
        bg,
        inside,
        wall,
        wall_4conn,
        t1_color,
        t2_color,
    )
    return {
        "inside_payload": int(inside_payload),
        "outside_payload": int(outside_payload),
        "inside_exclusive": int(inside_exclusive),
        "outside_exclusive": int(outside_exclusive),
        "delta_vs_original": int(np.sum(old_out != new_out)),
        "t1_color_pixels": int(len(t1_color_positions)),
        "t2_color_pixels": int(len(t2_color_positions)),
    }


def _parse_problem(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    assert g.shape == (GRID_ROWS, GRID_COLS), f"unexpected grid shape {g.shape}"

    border = int(g[0, 0])
    tiles = _extract_tiles(g)

    backgrounds = [get_bg(tile) for tile in tiles]
    assert len(set(backgrounds)) == 1, "all panels must share the same background"
    bg = int(backgrounds[0])
    assert border != bg, "border color must differ from tile background"

    dst_idx = _find_destination_tile(tiles, bg)
    cue_idx, indicator, wall = _find_cue_tile(tiles, bg, dst_idx)
    t1_idx, t2_idx = _source_role_indices(cue_idx, dst_idx)

    t1 = tiles[t1_idx]
    t2 = tiles[t2_idx]
    dst = tiles[dst_idx]
    t1_color = _single_fg_color(t1, bg, "tile1")
    t2_color = _single_fg_color(t2, bg, "tile2")
    assert t1_color != t2_color, "tile1 and tile2 must use different foreground colors"

    dst_fg_colors = {int(v) for v in dst.flatten() if int(v) != bg}
    assert dst_fg_colors == {t1_color, t2_color}, "destination prefill must use both source colors"

    inside = _flood_inside(wall, indicator)
    outside = {
        (r, c)
        for r in range(TILE_SZ)
        for c in range(TILE_SZ)
    } - inside - set(wall)
    assert outside, "wall must partition the grid — outside region is empty"

    return {
        "g": g,
        "bg": bg,
        "tiles": tiles,
        "dst_idx": dst_idx,
        "cue_idx": cue_idx,
        "t1_idx": t1_idx,
        "t2_idx": t2_idx,
        "indicator": indicator,
        "wall": wall,
        "inside": inside,
        "outside": outside,
        "wall_4conn": _wall_is_4conn(wall),
        "t1_color": t1_color,
        "t2_color": t2_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a partially prefilled destination by locating panel roles and composing
    a cue-guided color-swapped fusion of the two source patterns.

    Input structure:
    - Four 5x5 panels appear in one row inside a fixed 7x25 border frame.
    - One panel is a cue: a 5-pixel wall through the center plus one isolated
      indicator pixel.
    - Two panels are single-color symmetric source patterns sharing one background.
    - One panel is the destination: it already contains a sparse subset of the final
      answer and therefore mixes both source colors.

    Rule:
    - Identify the destination as the only panel with both source colors present.
    - Identify the cue by wall+indicator geometry.
    - Among the two remaining panels, the source nearer to the cue is tile1 and the
      farther source is tile2.
    - Flood-fill from the indicator, blocked by the wall, to obtain the inside region.
    - Inside and wall use tile2's shape painted with tile1's color; outside uses
      tile1's shape painted with tile2's color; wall falls back to tile1-shape in
      tile2 color only when the wall is 4-connected.
    - Any non-background pixel already present in the destination must agree with the
      computed final answer; background cells are filled in.

    Key insight:
    - The destination is not blank, so panel roles cannot be read from fixed indices
      or from a monochrome output slot. The solver must use visible structure twice:
      cue geometry to recover the partition, and the destination's two-color prefill
      to recover where the completed answer belongs.
    """
    parsed = _parse_problem(grid)
    g = parsed["g"]
    tiles = parsed["tiles"]
    dst_idx = parsed["dst_idx"]
    t1 = tiles[parsed["t1_idx"]]
    t2 = tiles[parsed["t2_idx"]]
    dst = tiles[dst_idx]

    composite, _t1_positions, _t2_positions = _compose_combo_tile(
        t1=t1,
        t2=t2,
        bg=parsed["bg"],
        inside=parsed["inside"],
        wall=parsed["wall"],
        wall_4conn=parsed["wall_4conn"],
        t1_color=parsed["t1_color"],
        t2_color=parsed["t2_color"],
    )

    out_tile = dst.copy()
    seen_prefill = 0
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            if int(out_tile[r, c]) != parsed["bg"]:
                seen_prefill += 1
                assert int(out_tile[r, c]) == int(composite[r, c]), "prefilled dst pixel is inconsistent"
                assert int(composite[r, c]) != parsed["bg"], "prefilled dst pixel cannot sit on a bg output cell"
            else:
                out_tile[r, c] = composite[r, c]
    assert seen_prefill >= 1, "combo variant expects at least one prefilled dst pixel"

    out = g.copy()
    out[_panel_slice(dst_idx)] = out_tile
    return out


def _choose_prefill(
    out_tile: np.ndarray,
    bg: int,
    t1_color_positions: set[tuple[int, int]],
    t2_color_positions: set[tuple[int, int]],
    rng: random.Random,
) -> np.ndarray | None:
    all_fg = [
        (r, c)
        for r in range(TILE_SZ)
        for c in range(TILE_SZ)
        if int(out_tile[r, c]) != bg
    ]
    if len(all_fg) < PREFILL_MIN + 1:
        return None
    if not t1_color_positions or not t2_color_positions:
        return None

    limit = min(PREFILL_MAX, len(all_fg) - 1)
    if limit < PREFILL_MIN:
        return None

    need = rng.randint(PREFILL_MIN, limit)
    chosen = {
        rng.choice(sorted(t1_color_positions)),
        rng.choice(sorted(t2_color_positions)),
    }

    remaining = [pos for pos in all_fg if pos not in chosen]
    rng.shuffle(remaining)
    for pos in remaining:
        if len(chosen) >= need:
            break
        chosen.add(pos)

    if len(chosen) != need:
        return None

    t3_in = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r, c in chosen:
        t3_in[r, c] = int(out_tile[r, c])

    fg_colors = {int(v) for v in t3_in.flatten() if int(v) != bg}
    if len(fg_colors) != 2:
        return None
    return t3_in


def _valid_destination_positions(cue_idx: int) -> list[int]:
    valid = []
    for dst_idx in range(NUM_TILES):
        if dst_idx == cue_idx:
            continue
        try:
            _source_role_indices(cue_idx, dst_idx)
        except AssertionError:
            continue
        valid.append(dst_idx)
    return valid


def _choose_layout(rng: random.Random) -> tuple[int, int, int, int]:
    cue_idx = rng.randrange(NUM_TILES)
    dst_idx = rng.choice(_valid_destination_positions(cue_idx))
    t1_idx, t2_idx = _source_role_indices(cue_idx, dst_idx)
    return cue_idx, dst_idx, t1_idx, t2_idx


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color, border = rng.sample(other, 4)

    pool = _LBEND_CONFIGS if rng.random() < LBEND_PROB else _STRAIGHT_CONFIGS
    indicator, wall = rng.choice(pool)
    inside = _flood_inside(wall, indicator)
    outside = {
        (r, c)
        for r in range(TILE_SZ)
        for c in range(TILE_SZ)
    } - inside - set(wall)
    wall_4conn = _wall_is_4conn(wall)

    for _ in range(PATTERN_PAIR_RETRIES):
        t1 = _sample_symmetric_pattern(
            tile_bg,
            t1_color,
            rng,
            min_px=T1_MIN_PX,
            max_px=T1_MAX_PX,
        )
        t2 = _sample_symmetric_pattern(
            tile_bg,
            t2_color,
            rng,
            min_px=T2_MIN_PX,
            max_px=T2_MAX_PX,
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
        if metrics["t1_color_pixels"] < 2 or metrics["t2_color_pixels"] < 2:
            continue

        out_tile, t1_color_positions, t2_color_positions = _compose_combo_tile(
            t1=t1,
            t2=t2,
            bg=tile_bg,
            inside=inside,
            wall=wall,
            wall_4conn=wall_4conn,
            t1_color=t1_color,
            t2_color=t2_color,
        )
        dst_in = _choose_prefill(
            out_tile,
            tile_bg,
            t1_color_positions,
            t2_color_positions,
            rng,
        )
        if dst_in is None:
            continue

        cue_tile = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
        for r, c in wall:
            cue_tile[r, c] = cue_color
        cue_tile[indicator[0], indicator[1]] = cue_color

        cue_idx, dst_idx, t1_idx, t2_idx = _choose_layout(rng)
        tiles: list[np.ndarray | None] = [None] * NUM_TILES
        tiles[cue_idx] = cue_tile
        tiles[dst_idx] = dst_in
        tiles[t1_idx] = t1
        tiles[t2_idx] = t2

        g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
        for idx, tile in enumerate(tiles):
            assert tile is not None
            g[_panel_slice(idx)] = tile

        expected = g.copy()
        expected[_panel_slice(dst_idx)] = out_tile
        got = main(g)
        if np.array_equal(got, expected):
            return g
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
