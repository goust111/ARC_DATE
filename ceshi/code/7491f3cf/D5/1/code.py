from __future__ import annotations

# Variant of 7491f3cf
# Dimension: D5 — 输出目标面板 (dst)
# Change: 面板3为全背景色的空白面板，需要填充合成结果 → 目标面板带干扰像素
#
# Design analysis for the D5 variant
# ----------------------------------
# variant_spec:
#   unchanged_layout:
#     - Whole grid stays 7x25 with four 5x5 panels separated by 1-cell borders.
#     - Panel 0 still contains a cue wall plus one isolated indicator.
#     - Panels 1 and 2 still provide the two source patterns.
#   changed_panel3_input:
#     - Panel 3 is no longer fully blank.
#     - It still contains many background cells, so the solver must fill missing output pixels.
#     - It also contains several prefilled non-background distractor pixels in cue_color.
#     - These distractors are guaranteed to be wrong: cue_color is distinct from tile_bg,
#       t1_color, and t2_color, so every cue-colored pixel in panel 3 must be replaced.
#   changed_solver_view:
#     - Panel 3 is now explicitly unreliable. The authoritative structure still comes only
#       from panel 0 (partition) and panels 1-2 (content). The solver should reconstruct the
#       full destination tile and overwrite panel 3, thereby both correcting wrong pixels and
#       filling blank ones.
#   forced_visibility:
#     - Every generated sample contains:
#       1. at least one distractor placed on a cell whose correct output is background
#          (so the solver must erase a wrong pixel),
#       2. at least one distractor placed on a cell whose correct output is non-background
#          (so the solver must recolor a wrong pixel),
#       3. at least two correct non-background output cells left blank in the input
#          (so the solver must still perform ordinary filling, not just cleanup).
#   small_trace:
#     - Original behavior at panel-3 construction:
#       t3_in = all background.
#       main() computes out3 from t0/t1/t2 and writes it into panel 3.
#     - Variant behavior at panel-3 construction:
#       t3_in starts as background, then a few cells are changed to cue_color noise.
#       Example:
#         correct out3[r,c] = bg, but t3_in[r,c] = cue_color  -> must be erased to bg.
#         correct out3[r,c] = t2_color, but t3_in[r,c] = cue_color -> must be recolored.
#         correct out3[r,c] = t1_color, but t3_in[r,c] = bg -> must be filled as before.
#     - Difference:
#       The merge rule itself is unchanged; the destination tile now mixes blanks and wrong
#       pixels, so the solver must treat panel 3 as editable rather than trustworthy.
#   old-assumption-audit:
#     - `t0, t1, t2, _t3 = tiles` in the original code silently ignored panel 3.
#       That was harmless when panel 3 was blank; it remains correct here only if we make the
#       intended contract explicit: panel 3 is unreliable and must be reconstructed.
#     - `out3 = np.full(..., bg)` followed by per-cell overwrite is still valid.
#       In the variant this full reconstruction is essential because some existing panel-3
#       pixels are wrong and must be replaced, including replacement back to background.
#     - No step may copy non-background pixels from the input panel 3 into the result.
#       Doing so would preserve distractors and break D5.
#   placement_legality:
#     - Distractors are inserted only inside panel 3, so they never overlap wall/source tiles.
#     - They are always within the 5x5 bounds.
#     - Generator keeps the original wall validity checks and still verifies samples by calling
#       `main(input)` and comparing against a hand-built expected output.
#   observability:
#     - Colors are kept structurally distinct: border, tile_bg, cue_color, t1_color, t2_color
#       are all different in generated samples.
#     - Panel 3 visibly contains isolated cue-colored specks that disappear or change color in
#       the output, making the D5 change readable from a few examples.

import random
from collections import Counter

import numpy as np

from arc_utils import flood_fill, label_mask, neighbors8


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
GEN_RETRIES = 300
LBEND_PROB = 0.2


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


def _wall_is_4conn(wall: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    return n_comp == 1


def _compose_tile3(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    inside: set[tuple[int, int]],
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
) -> np.ndarray:
    wall_4conn = _wall_is_4conn(wall)
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
    Composite two pattern tiles into panel 3 using the cue wall and indicator.

    Variant D5 change:
    - Panel 3 may already contain non-background distractor pixels.
    - Those pixels are not trustworthy; the solver reconstructs panel 3 completely.
    """
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, t3 = tiles
    _ = t3  # panel 3 is intentionally ignored as a source of truth

    bg = int(Counter(t1.flatten()).most_common(1)[0][0])
    assert border != bg, "border color must differ from tile background"

    t0_bg = int(Counter(t0.flatten()).most_common(1)[0][0])
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

    out3 = _compose_tile3(t1, t2, bg, inside, wall)

    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _make_noisy_target(out3: np.ndarray, bg: int, noise_color: int, rng: random.Random) -> np.ndarray | None:
    fg_positions = [(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ) if int(out3[r, c]) != bg]
    bg_positions = [(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ) if int(out3[r, c]) == bg]
    if len(fg_positions) < 3 or not bg_positions:
        return None

    recolor_pos = rng.choice(fg_positions)
    erase_pos = rng.choice(bg_positions)
    noisy_fg = {recolor_pos}
    noisy_bg = {erase_pos}

    remaining_fg = [pos for pos in fg_positions if pos != recolor_pos]
    remaining_bg = [pos for pos in bg_positions if pos != erase_pos]
    extra_steps = rng.randint(1, 3)
    for _ in range(extra_steps):
        choices: list[str] = []
        if len(remaining_fg) > 2:
            choices.append("fg")
        if remaining_bg:
            choices.append("bg")
        if not choices:
            break
        bucket = rng.choice(choices)
        if bucket == "fg":
            pos = rng.choice(remaining_fg)
            noisy_fg.add(pos)
            remaining_fg.remove(pos)
        else:
            pos = rng.choice(remaining_bg)
            noisy_bg.add(pos)
            remaining_bg.remove(pos)

    fill_count = len(fg_positions) - len(noisy_fg)
    if fill_count < 2:
        return None

    t3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r, c in noisy_fg | noisy_bg:
        t3[r, c] = noise_color

    erase_count = len(noisy_bg)
    recolor_count = len(noisy_fg)
    assert erase_count >= 1 and recolor_count >= 1 and fill_count >= 2
    return t3


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg, border, cue_color, t1_color, t2_color = rng.sample(palette, 5)

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

    out3 = _compose_tile3(t1, t2, tile_bg, inside, wall)
    t3_in = _make_noisy_target(out3, tile_bg, cue_color, rng)
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
