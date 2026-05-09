# Combo variant of 7491f3cf (index 4)
# combo_spec: D5/1+D6/1
# Name: 干扰像素目标+异或合成

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import flood_fill, label_mask, neighbors8


# Design analysis for the combo variant
# -------------------------------------
# fused_spec:
#   unchanged_geometry:
#     - The whole grid stays 7x25 with four 5x5 panels separated by border cells.
#     - Panel 0 still contains exactly one 8-isolated indicator plus a 5-pixel wall
#       through (2,2), and the wall still partitions the tile under 4-connected flood fill.
#     - Panels 1 and 2 remain single-foreground-color symmetric source patterns.
#   fused_changes:
#     - D6/1 replaces the original region-based compositor with occupancy XOR:
#       output is foreground only where exactly one of tile1/tile2 is foreground.
#     - D5/1 makes panel 3 unreliable: it contains cue-colored distractor pixels and
#       blanks, so the solver must overwrite the entire destination tile.
#   concrete_trace:
#     - Example aligned cell A: tile1=red, tile2=green.
#       Original task: region picks one source by inside/outside/wall.
#       Combo task: both are foreground, so XOR cancels -> background.
#       If panel3 already shows a cue-colored speck there, solver must erase it too.
#     - Example aligned cell B: tile1=red, tile2=bg.
#       Original task: red survived only if the region selected tile1 there.
#       Combo task: red survives everywhere because it is the unique foreground.
#       If panel3 already has a wrong cue-colored pixel there, solver must recolor it red.
#     - Example aligned cell C: tile1=bg, tile2=bg.
#       Original task: background. Combo task: still background.
#       A prefilled cue-colored distractor on such a cell must be removed.
#   complexity_calibration:
#     - A weak generator could collapse back toward the original rule if overlaps were rare.
#     - It could also make D5 visually trivial if every noisy pixel landed on XOR-foreground.
#     - Therefore generation enforces:
#       1. visible overlap cells so XOR cancellation is observable,
#       2. visible tile1-only and tile2-only cells so both source colors survive,
#       3. XOR-visible cells on both inside and outside so the preserved cue geometry still
#          coexists with the new rule and examples do not look one-sided,
#       4. a minimum Hamming distance between XOR output and original output,
#       5. at least one distractor that must be erased to background,
#       6. at least one distractor that must be recolored to a source color,
#       7. at least two XOR-foreground cells left blank so panel 3 is not "just cleanup".
#   implicit_assumption_audit_of_original_main:
#     - Parsing border, tile extraction, indicator detection, wall symmetry checks, and
#       flood-fill partitioning remain valid because the input layout is unchanged.
#     - The original composition loop assumed region membership determined which source could
#       paint each cell. D6/1 breaks that assumption, so the final compositor must be replaced.
#     - The original wall-specific tile2 fallback for 4-connected walls is invalid under XOR:
#       overlap now cancels regardless of region or wall connectivity.
#     - Ignoring panel 3 was optional in the blank-target task but becomes mandatory here
#       because non-background pixels in panel 3 are guaranteed untrustworthy distractors.
#   placement_legality:
#     - The generator only adds distractors inside panel 3, within the 5x5 bounds.
#     - Source patterns stay confined to their own panels and use a shared background color.
#     - Distinct colors are sampled for border, tile_bg, cue_color, t1_color, and t2_color,
#       so the distractor color cannot be mistaken for a valid XOR output color.
#   visual_observability_and_cleanliness:
#     - Humans can read XOR from the disappearance of overlaps and the survival of exclusives.
#     - Humans can read D5 from cue-colored specks in panel 3 that either vanish or change
#       into source colors in the output.
#     - Using only one distractor color avoids clutter; the output still uses only tile_bg
#       plus the two source colors, so the final panel remains cleaner than the noisy input.
#   new_main_assumptions_and_generator_support:
#     - main() now assumes tile1 and tile2 each have a majority background color; generation
#       preserves the original 8-12 / 9-12 foreground-count ranges, so background stays major.
#     - main() assumes the cue tile still forms a valid partition even though XOR no longer
#       uses inside/outside to choose a winner; generation retains the original validated wall
#       catalogue and asserts non-empty outside.
#     - main() assumes panel 3 may be overwritten wholesale; generation explicitly constructs
#       panel 3 as a mixture of wrong cue-colored pixels and background blanks.


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
GEN_RETRIES = 600
LBEND_PROB = 0.2

MIN_OVERLAP = 3
MIN_T1_ONLY = 2
MIN_T2_ONLY = 2
MIN_XOR_INSIDE = 1
MIN_XOR_OUTSIDE = 1
MIN_DIFF_FROM_ORIGINAL = 4
MIN_NOISY_BG = 1
MIN_NOISY_FG = 1
MIN_BLANK_XOR_FG = 2


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


def _extract_tiles(grid: np.ndarray) -> list[np.ndarray]:
    return [grid[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]


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
        for idx in order:
            orbit = _ORBITS[idx]
            if count + len(orbit) > max_px:
                continue
            if rng.random() < ORBIT_INCLUDE_PROB:
                for r, c in orbit:
                    pattern[r, c] = fg
                count += len(orbit)
        if min_px <= count <= max_px:
            return pattern
    return None


def _flood_inside(
    wall: frozenset[tuple[int, int]], indicator: tuple[int, int]
) -> set[tuple[int, int]]:
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    return {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}


def _mask_from_coords(coords: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _wall_is_4_connected(wall: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    wall_mask = _mask_from_coords(wall)
    _, n_comp = label_mask(wall_mask, connectivity=4)
    return n_comp == 1


def _parse_cue(
    t0: np.ndarray,
) -> tuple[tuple[int, int], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    t0_bg = int(Counter(t0.flatten()).most_common(1)[0][0])
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
        reflect = lambda r, c: (4 - r, c)
    elif dc == 0:
        reflect = lambda r, c: (r, 4 - c)
    elif dr * dc > 0:
        reflect = lambda r, c: (c, r)
    else:
        reflect = lambda r, c: (4 - c, 4 - r)
    assert wall == {reflect(r, c) for r, c in wall}, "wall must be symmetric about indicator→center axis"

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
    return indicator, wall, inside, outside


def _compose_original(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    inside: set[tuple[int, int]],
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
) -> np.ndarray:
    wall_4conn = _wall_is_4_connected(wall)
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


def _compose_xor(t1: np.ndarray, t2: np.ndarray, bg: int) -> np.ndarray:
    out3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1, v2 = int(t1[r, c]), int(t2[r, c])
            has1 = v1 != bg
            has2 = v2 != bg
            if has1 ^ has2:
                out3[r, c] = v1 if has1 else v2
    return out3


def _make_noisy_target(
    out3: np.ndarray,
    bg: int,
    noise_color: int,
    rng: random.Random,
) -> np.ndarray | None:
    fg_positions = [(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ) if int(out3[r, c]) != bg]
    bg_positions = [(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ) if int(out3[r, c]) == bg]

    max_noisy_fg = min(3, len(fg_positions) - MIN_BLANK_XOR_FG)
    max_noisy_bg = min(3, len(bg_positions))
    if max_noisy_fg < MIN_NOISY_FG or max_noisy_bg < MIN_NOISY_BG:
        return None

    num_noisy_fg = rng.randint(MIN_NOISY_FG, max_noisy_fg)
    num_noisy_bg = rng.randint(MIN_NOISY_BG, max_noisy_bg)
    noisy_fg = set(rng.sample(fg_positions, num_noisy_fg))
    noisy_bg = set(rng.sample(bg_positions, num_noisy_bg))

    t3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r, c in noisy_fg | noisy_bg:
        t3[r, c] = noise_color
    return t3


def _compute_combo_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    t0, t1, t2, t3 = _extract_tiles(g)
    bg = int(Counter(t1.flatten()).most_common(1)[0][0])
    _indicator, wall, inside, outside = _parse_cue(t0)

    occ1 = t1 != bg
    occ2 = t2 != bg
    overlap = occ1 & occ2
    t1_only = occ1 & ~occ2
    t2_only = occ2 & ~occ1
    xor_out = _compose_xor(t1, t2, bg)
    original_out = _compose_original(t1, t2, bg, inside, wall)
    inside_mask = _mask_from_coords(inside)
    outside_mask = _mask_from_coords(outside)
    noisy_mask = t3 != bg
    xor_mask = xor_out != bg

    return {
        "overlap": int(overlap.sum()),
        "t1_only": int(t1_only.sum()),
        "t2_only": int(t2_only.sum()),
        "xor_inside": int((xor_mask & inside_mask).sum()),
        "xor_outside": int((xor_mask & outside_mask).sum()),
        "diff_from_original": int(np.sum(xor_out != original_out)),
        "noisy_bg": int((noisy_mask & ~xor_mask).sum()),
        "noisy_fg": int((noisy_mask & xor_mask).sum()),
        "blank_xor_fg": int((~noisy_mask & xor_mask).sum()),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct panel 3 for the combo variant by combining cue parsing with XOR composition.

    Input structure:
    - Four 5x5 panels appear in one row, separated by a 1-cell-wide border.
    - Panel 0 contains a 5-pixel wall plus one 8-isolated indicator pixel.
    - Panels 1 and 2 contain the two source patterns.
    - Panel 3 may already contain cue-colored distractor pixels mixed with blanks.

    Rule:
    - Parse the cue tile exactly as in the original task: indicator, symmetric wall, and
      4-connected flood-fill partition are validated from panel 0.
    - Ignore the existing content of panel 3 completely.
    - For each aligned cell, output the unique non-background source color iff exactly one of
      tile1 or tile2 is non-background; otherwise output the shared background color.

    Key insight:
    - The cue geometry still tells you the input is the same task family, but the destination
      is now the occupancy XOR of tiles 1 and 2, and panel 3 must be overwritten wholesale
      because its prefilled pixels are deliberate mistakes rather than partial truth.
    """
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    t0, t1, t2, t3 = _extract_tiles(g)
    _ = t3

    bg = int(Counter(t1.flatten()).most_common(1)[0][0])
    assert border != bg, "border color must differ from tile background"

    _indicator, _wall, _inside, _outside = _parse_cue(t0)
    out3 = _compose_xor(t1, t2, bg)

    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg, border, cue_color, t1_color, t2_color = rng.sample(palette, 5)

    pool = _LBEND_CONFIGS if rng.random() < LBEND_PROB else _STRAIGHT_CONFIGS
    indicator, wall = rng.choice(pool)
    inside = _flood_inside(wall, indicator)
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - wall

    t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    occ1 = t1 != tile_bg
    occ2 = t2 != tile_bg
    overlap = occ1 & occ2
    t1_only = occ1 & ~occ2
    t2_only = occ2 & ~occ1
    xor_mask = t1_only | t2_only
    inside_mask = _mask_from_coords(inside)
    outside_mask = _mask_from_coords(outside)

    if int(overlap.sum()) < MIN_OVERLAP:
        return None
    if int(t1_only.sum()) < MIN_T1_ONLY or int(t2_only.sum()) < MIN_T2_ONLY:
        return None
    if int((xor_mask & inside_mask).sum()) < MIN_XOR_INSIDE:
        return None
    if int((xor_mask & outside_mask).sum()) < MIN_XOR_OUTSIDE:
        return None

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = cue_color
    t0[indicator[0], indicator[1]] = cue_color

    out3 = _compose_xor(t1, t2, tile_bg)
    original_out3 = _compose_original(t1, t2, tile_bg, inside, wall)
    if int(np.sum(out3 != original_out3)) < MIN_DIFF_FROM_ORIGINAL:
        return None

    t3_in = _make_noisy_target(out3, tile_bg, cue_color, rng)
    if t3_in is None:
        return None

    g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
    for j, tile in enumerate([t0, t1, t2, t3_in]):
        g[1:6, 1 + 6 * j : 1 + 6 * j + 5] = tile

    metrics = _compute_combo_metrics(g)
    if metrics["noisy_bg"] < MIN_NOISY_BG:
        return None
    if metrics["noisy_fg"] < MIN_NOISY_FG:
        return None
    if metrics["blank_xor_fg"] < MIN_BLANK_XOR_FG:
        return None
    if metrics["diff_from_original"] < MIN_DIFF_FROM_ORIGINAL:
        return None

    expected = g.copy()
    expected[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
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
