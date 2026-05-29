# Combo variant of 7491f3cf (index 20)
# combo_spec: D4/2+D13/0
# Name: 三源面板+链式合成

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import flood_fill, label_mask, neighbors8

# Design analysis for D4/2 + D13/0:
# a) Concretization:
#    Original: [cue | src1 | src2 | blank] with one wall and one synthesis. Fusion:
#    [cue | src1 | src2 | src3 | blank_mid | blank_out]. The cue tile now contains two
#    straight walls that share only the center pixel. The visually upper indicator/wall pair
#    defines step 1, which composes src1+src2 into panel 4; the lower/right pair defines
#    step 2, which composes panel 4+src3 into panel 5.
#    Example: if step 1 is a horizontal wall with indicator at (0,2), panel 4 takes src1 on
#    the top side and src2 on the bottom side. If step 2 is a diagonal wall with indicator at
#    (4,0), panel 5 then takes panel 4 on that indicator side and src3 elsewhere.
# b) Complexity calibration:
#    The fusion really uses three sources and two partitions. Generation rejects degenerate
#    cases where panel 4 is effectively one source, where panel 5 ignores one of the three
#    sources, or where the chained final panel matches a direct src1+src3 composition.
# c) Broken implicit assumptions from the original:
#    `main()` can no longer assume width 25, 4 tiles, one cue color, one indicator, or
#    `wall = cue_set - {indicator}`. It must parse three cue colors: step-1 color, step-2
#    color, and a shared-center color. It also cannot stop after one composite; panel 4 is an
#    explicit intermediate result and panel 5 is the final result.
# d) Placement legality:
#    The layout is a 7x37 strip: six disjoint 5x5 panels separated by 1-cell borders. The two
#    chosen walls are distinct straight center-crossing lines whose overlap is exactly {(2,2)};
#    therefore the cue tile never needs multi-color overlap beyond that shared center pixel.
# e) Visual observability:
#    Humans can see two blank targets in the input and two filled targets in the output. That
#    makes the chain explicit: panel 4 demonstrates which cue structure acts first, panel 5
#    demonstrates the second reuse of the same compositing rule with src3.
# f) Visual cleanliness:
#    Each source remains a single-color symmetric 5x5 pattern, while the cue tile uses only
#    three cue colors and one shared overlap pixel. This keeps panel roles separable instead of
#    mixing many layered marks inside the same 5x5 area.
# g) Inferability of changes:
#    No rule depends on raw color ids. Step order is structural: the lexicographically earlier
#    indicator (guaranteed to be on the top edge in generated examples) is always the first
#    synthesis, which is visibly confirmed by panel 4 in every training pair.

TILE_SZ = 5
NUM_PANELS = 6
GRID_ROWS = 7
GRID_COLS = 37
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
T3_MIN_PX, T3_MAX_PX = 8, 11
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
GEN_RETRIES = 400

VALID_INDICATORS = {
    (0, 0), (0, 2), (0, 4),
    (2, 0), (2, 4),
    (4, 0), (4, 2), (4, 4),
}
ALL_CELLS = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)}

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
    ((0, 2), frozenset({(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)})),  # row
    ((4, 2), frozenset({(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)})),  # row
    ((2, 0), frozenset({(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)})),  # col
    ((2, 4), frozenset({(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)})),  # col
    ((0, 0), frozenset({(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)})),  # anti
    ((4, 4), frozenset({(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)})),  # anti
    ((0, 4), frozenset({(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)})),  # diag
    ((4, 0), frozenset({(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)})),  # diag
]
_STRAIGHT_LOOKUP = {(indicator, wall) for indicator, wall in _STRAIGHT_CONFIGS}


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


def _flood_inside(wall: frozenset[tuple[int, int]], indicator: tuple[int, int]) -> set[tuple[int, int]]:
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    return {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}


def _wall_is_4conn(wall: frozenset[tuple[int, int]]) -> bool:
    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    return n_comp == 1


def _compose_pair(
    left: np.ndarray,
    right: np.ndarray,
    left_owner: np.ndarray,
    right_owner: np.ndarray,
    bg: int,
    wall: frozenset[tuple[int, int]],
    indicator: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, set[tuple[int, int]], bool]:
    inside = _flood_inside(wall, indicator)
    outside = ALL_CELLS - inside - set(wall)
    assert inside, "inside region is empty"
    assert outside, "outside region is empty"
    wall_4conn = _wall_is_4conn(wall)

    out = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    owner = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            lv = int(left[r, c])
            rv = int(right[r, c])
            if (r, c) in inside:
                if lv != bg:
                    out[r, c] = lv
                    owner[r, c] = int(left_owner[r, c])
            elif (r, c) in wall:
                if lv != bg:
                    out[r, c] = lv
                    owner[r, c] = int(left_owner[r, c])
                elif wall_4conn and rv != bg:
                    out[r, c] = rv
                    owner[r, c] = int(right_owner[r, c])
            else:
                if rv != bg:
                    out[r, c] = rv
                    owner[r, c] = int(right_owner[r, c])
    return out, owner, inside, wall_4conn


def _tile_slices() -> list[tuple[slice, slice]]:
    return [(slice(1, 6), slice(1 + 6 * j, 1 + 6 * j + 5)) for j in range(NUM_PANELS)]


def _extract_tiles(g: np.ndarray) -> list[np.ndarray]:
    return [g[rs, cs].copy() for rs, cs in _tile_slices()]


def _parse_cue_step(
    tile: np.ndarray,
    *,
    step_color: int,
    shared_color: int,
) -> tuple[tuple[int, int], frozenset[tuple[int, int]]]:
    own = {(int(r), int(c)) for r in range(TILE_SZ) for c in range(TILE_SZ) if int(tile[r, c]) == step_color}
    shared = {(int(r), int(c)) for r in range(TILE_SZ) for c in range(TILE_SZ) if int(tile[r, c]) == shared_color}
    assert shared == {(2, 2)}, f"shared cue should be only center, got {sorted(shared)}"
    assert len(own) == 5, f"step color {step_color} should have 5 pixels, got {len(own)}"

    cue_union = own | shared
    indicators = []
    for pos in own:
        if not any((nr, nc) in cue_union for nr, nc in neighbors8(*pos)):
            indicators.append(pos)
    assert len(indicators) == 1, f"step color {step_color} should have one isolated indicator"
    indicator = indicators[0]
    assert indicator in VALID_INDICATORS, f"indicator {indicator} not at valid border position"
    wall = frozenset(cue_union - {indicator})
    assert (indicator, wall) in _STRAIGHT_LOOKUP, "cue step does not match a legal straight wall config"
    return indicator, wall


def _parse_task(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_ROWS, GRID_COLS), f"expected {(GRID_ROWS, GRID_COLS)}, got {g.shape}"
    border = int(g[0, 0])
    tiles = _extract_tiles(g)
    t0, t1, t2, t3, t4, t5 = tiles
    bg = int(Counter(map(int, t1.flatten())).most_common(1)[0][0])
    assert border != bg, "border color must differ from tile background"
    assert np.all(t4 == bg), "panel 4 input must be blank"
    assert np.all(t5 == bg), "panel 5 input must be blank"

    t0_bg = int(Counter(map(int, t0.flatten())).most_common(1)[0][0])
    cue_colors = sorted({int(v) for v in t0.flatten() if int(v) != t0_bg})
    assert len(cue_colors) == 3, f"expected 3 cue colors, got {cue_colors}"
    shared_candidates = [color for color in cue_colors if int(np.sum(t0 == color)) == 1 and int(t0[2, 2]) == color]
    assert len(shared_candidates) == 1, "could not identify shared-center cue color"
    shared_color = shared_candidates[0]
    step_colors = [color for color in cue_colors if color != shared_color]

    parsed_steps = []
    for color in step_colors:
        indicator, wall = _parse_cue_step(t0, step_color=color, shared_color=shared_color)
        parsed_steps.append({"color": color, "indicator": indicator, "wall": wall})
    parsed_steps.sort(key=lambda item: tuple(item["indicator"]))
    assert parsed_steps[0]["indicator"][0] == 0, "first step must be the upper cue structure"
    assert parsed_steps[1]["indicator"][0] > 0, "second step must be below the first cue structure"

    return {
        "grid": g,
        "bg": bg,
        "tiles": tiles,
        "steps": parsed_steps,
    }


def _solve_with_trace(grid: np.ndarray) -> dict[str, object]:
    parsed = _parse_task(grid)
    g = parsed["grid"]
    bg = int(parsed["bg"])
    t0, t1, t2, t3, _blank4, _blank5 = parsed["tiles"]
    step1, step2 = parsed["steps"]

    owner1 = np.where(t1 != bg, 1, 0)
    owner2 = np.where(t2 != bg, 2, 0)
    owner3 = np.where(t3 != bg, 3, 0)

    mid, mid_owner, inside1, wall1_4conn = _compose_pair(
        t1,
        t2,
        owner1,
        owner2,
        bg,
        step1["wall"],
        step1["indicator"],
    )
    final, final_owner, inside2, wall2_4conn = _compose_pair(
        mid,
        t3,
        mid_owner,
        owner3,
        bg,
        step2["wall"],
        step2["indicator"],
    )

    out = g.copy()
    slices = _tile_slices()
    out[slices[4][0], slices[4][1]] = mid
    out[slices[5][0], slices[5][1]] = final
    return {
        "output": out,
        "mid": mid,
        "final": final,
        "mid_owner": mid_owner,
        "final_owner": final_owner,
        "inside1": inside1,
        "inside2": inside2,
        "wall1_4conn": wall1_4conn,
        "wall2_4conn": wall2_4conn,
        "t0": t0,
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "step1": step1,
        "step2": step2,
        "bg": bg,
    }


def analyze_chain_behavior(grid: np.ndarray) -> dict[str, int]:
    trace = _solve_with_trace(grid)
    bg = int(trace["bg"])
    t1 = trace["t1"]
    t3 = trace["t3"]
    owner1 = np.where(t1 != bg, 1, 0)
    owner3 = np.where(t3 != bg, 3, 0)
    direct13, _owner13, _inside13, _wall13 = _compose_pair(
        t1,
        t3,
        owner1,
        owner3,
        bg,
        trace["step2"]["wall"],
        trace["step2"]["indicator"],
    )
    final_owner = trace["final_owner"]
    return {
        "mid_from_1": int(np.sum(trace["mid_owner"] == 1)),
        "mid_from_2": int(np.sum(trace["mid_owner"] == 2)),
        "final_from_1": int(np.sum(final_owner == 1)),
        "final_from_2": int(np.sum(final_owner == 2)),
        "final_from_3": int(np.sum(final_owner == 3)),
        "chain_delta_vs_direct13": int(np.sum(trace["final"] != direct13)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Chain two wall-guided composites across three source panels to fill two blank panels.

    Input structure:
    - Six 5x5 panels in one row, border-separated.
    - Panel 0: two cue structures sharing the center pixel. Each structure has one straight wall
      and one 8-isolated indicator. The upper indicator/wall pair is step 1; the lower/right
      pair is step 2.
    - Panels 1-3: three source patterns. Panels 4-5: blanks for the intermediate and final
      results.

    Rule:
    - Parse the two cue structures from panel 0.
    - Step 1: composite panel 1 and panel 2 using the first wall/indicator, producing the
      intermediate panel 4.
    - Step 2: composite panel 4 and panel 3 using the second wall/indicator, producing the
      final panel 5.
    - Within each step, the indicator side takes non-background pixels from the left source;
      the other side takes non-background pixels from the right source; wall pixels use the left
      source first and fall back to the right source only for 4-connected walls.

    Key insight:
    Panel 0 does not describe one partition anymore. It encodes an ordered pair of partitions,
    and the presence of two blank targets shows that the task is a visible chain: first build an
    intermediate composite, then reuse that intermediate as a source in the second composite.
    """
    return _solve_with_trace(grid)["output"]


def _build_chain_configs() -> list[tuple[tuple[int, int], frozenset[tuple[int, int]], tuple[int, int], frozenset[tuple[int, int]]]]:
    chain = []
    for indicator1, wall1 in _STRAIGHT_CONFIGS:
        if indicator1[0] != 0:
            continue
        for indicator2, wall2 in _STRAIGHT_CONFIGS:
            if indicator2[0] == 0:
                continue
            if wall1 == wall2:
                continue
            if wall1 & wall2 != {(2, 2)}:
                continue
            if indicator1 in wall2 or indicator2 in wall1 or indicator1 == indicator2:
                continue
            chain.append((indicator1, wall1, indicator2, wall2))
    assert chain, "no chain configs built"
    return chain


CHAIN_CONFIGS = _build_chain_configs()


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    border, shared_color, step1_color, step2_color, t1_color, t2_color, t3_color = rng.sample(other, 7)

    indicator1, wall1, indicator2, wall2 = rng.choice(CHAIN_CONFIGS)

    t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    t3 = _sample_symmetric_pattern(tile_bg, t3_color, rng, min_px=T3_MIN_PX, max_px=T3_MAX_PX)
    if t1 is None or t2 is None or t3 is None:
        return None

    owner1 = np.where(t1 != tile_bg, 1, 0)
    owner2 = np.where(t2 != tile_bg, 2, 0)
    owner3 = np.where(t3 != tile_bg, 3, 0)

    mid, mid_owner, _inside1, _wall1_4conn = _compose_pair(t1, t2, owner1, owner2, tile_bg, wall1, indicator1)
    final, final_owner, _inside2, _wall2_4conn = _compose_pair(mid, t3, mid_owner, owner3, tile_bg, wall2, indicator2)

    if int(np.sum(mid_owner == 1)) == 0 or int(np.sum(mid_owner == 2)) == 0:
        return None
    if any(int(np.sum(final_owner == src)) == 0 for src in (1, 2, 3)):
        return None
    if np.array_equal(final, mid):
        return None

    direct13, _owner13, _inside13, _wall13 = _compose_pair(t1, t3, owner1, owner3, tile_bg, wall2, indicator2)
    if np.array_equal(final, direct13):
        return None

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    overlap = wall1 & wall2
    assert overlap == {(2, 2)}, f"expected center-only overlap, got {sorted(overlap)}"
    for r, c in wall1 - overlap:
        t0[r, c] = step1_color
    for r, c in wall2 - overlap:
        t0[r, c] = step2_color
    t0[2, 2] = shared_color
    t0[indicator1[0], indicator1[1]] = step1_color
    t0[indicator2[0], indicator2[1]] = step2_color

    blank = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
    for (rs, cs), tile in zip(_tile_slices(), [t0, t1, t2, t3, blank, blank]):
        g[rs, cs] = tile

    expected = g.copy()
    slices = _tile_slices()
    expected[slices[4][0], slices[4][1]] = mid
    expected[slices[5][0], slices[5][1]] = final
    got = main(g)
    if not np.array_equal(got, expected):
        return None
    metrics = analyze_chain_behavior(g)
    if metrics["chain_delta_vs_direct13"] <= 0:
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
