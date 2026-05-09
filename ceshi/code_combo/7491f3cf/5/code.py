# Combo variant of 7491f3cf (index 5)
# combo_spec: D8/1+D15/2
# Name: 可变面板尺寸+可变宽度边框

from __future__ import annotations

import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT / "solvers"))

from arc_utils import flood_fill, label_mask, neighbors8


# -----------------------------------------------------------------------------
# Design analysis for combo D8/1 + D15/2
#
# 1. Concrete behavior trace
#    - Original task hard-coded a 7x25 grid: four 5x5 tiles, one-cell border.
#      The solver simply sliced `[1:6, 1+6*j : 1+6*j+5]`.
#    - This combo keeps the same visual grammar ("4 panels in one row, cue tile
#      then two source tiles then blank target tile"), but two layout degrees of
#      freedom now vary together:
#      * tile size n is any supported odd size
#      * border/separator width k is 1..3
#    - Example:
#      * Original: n=5, k=1 gives shape 7x25 and tile-3 slice `[1:6, 19:24]`.
#      * Combo:    n=7, k=3 gives shape 13x43 and tile-3 slice `[3:10, 33:40]`.
#      * The cue logic inside tile0 is unchanged: find the unique 8-isolated
#        indicator, treat the rest as the wall, flood-fill from the indicator,
#        then copy tile1 pixels to the indicator side and tile2 pixels to the
#        other side.
#    - Difference: the solver must first recover k from the actual border bands,
#      then recover n from the remaining interior, and only then can it apply
#      the original compositing rule.
#
# 2. Complexity calibration
#    - Allowing every odd n would over-spread the training distribution and make
#      2-4 ARC examples too weak. The generator therefore uses n in {5, 7, 9}.
#    - Although the prompt mentions 3x3 as a possible family member, the
#      original cue constraints make 3x3 impossible here: any legal corner or
#      edge-midpoint indicator would be 8-adjacent to the center pixel, so it
#      cannot be isolated from a wall that must pass through the center.
#    - Border width uses k in {1, 2, 3}. This is visually obvious but still a
#      small discrete set, so humans can infer the rule without extra metadata.
#    - We keep the original limited wall family (straight lines and L/V bends)
#      and the original diagonal-symmetric pattern family to avoid adding new
#      reasoning dimensions beyond layout parsing.
#
# 3. Implicit-assumption audit of the original main()
#    - `g[1:6, 1+6*j:...]` assumed both n=5 and k=1. The combo replaces this
#      with `_infer_layout()` + `_tile_slice(n, k, j)`.
#    - `(2, 2)` as the fixed center becomes `(n//2, n//2)`.
#    - The hard-coded valid indicator set becomes the general set of 4 corners
#      plus 4 edge midpoints for the inferred n.
#    - The flood-fill, wall mask, inside/outside enumeration, and tile-3 write
#      back all now use n instead of literal 5.
#    - The original parser implicitly trusted `g[0,0]` as a 1-cell border. The
#      combo parser now verifies that the border color forms:
#      * exactly two full-width horizontal runs of thickness k
#      * exactly five full-height vertical runs of thickness k
#      * total shape `(n + 2k, 4n + 5k)`
#    - New assumption introduced by the combo solver: all separator bands share
#      one uniform color and thickness. The generator enforces exactly that.
#
# 4. Placement legality and cleanliness
#    - Tiles are placed at rows `[k : k+n)` and columns
#      `[k + j*(n+k) : k + j*(n+k) + n)`, so they never overlap separators.
#    - The wall library is generated in tile-local coordinates and filtered by
#      legality checks: passes through center, indicator stays 8-isolated, wall
#      touches at least two sides, and flood-fill yields non-empty inside and
#      outside regions.
#    - Colors are chosen pairwise distinct for border, tile background, cue,
#      tile1, and tile2. This makes the wider separators easy to perceive and
#      avoids accidental visual ambiguity.
#
# 5. Visual observability / anti-degeneracy
#    - Humans can see both varying total grid size and varying border thickness
#      directly from the raw pixels, without depending on hidden color ids.
#    - The generator intentionally samples both `n>5` and `k>1`, so the combo
#      does not collapse back to the original `(n=5, k=1)` layout.
#    - The test script at the end of this turn measures layout-signature
#      diversity `(n, k)` across generated samples to confirm the fused behavior
#      is actually present in practice.
# -----------------------------------------------------------------------------


NUM_COLORS = 10
SIZE_CHOICES = (5, 7, 9)
BORDER_WIDTH_CHOICES = (1, 2, 3)
PATTERN_RETRIES = 300
GEN_RETRIES = 300
ORBIT_INCLUDE_PROB = 0.55
LBEND_PROB = 0.2

PATTERN_BOUNDS = {
    5: {"t1": (8, 12), "t2": (9, 12)},
    7: {"t1": (14, 20), "t2": (16, 22)},
    9: {"t1": (22, 32), "t2": (24, 36)},
}


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    out: list[tuple[int, int]] = []
    start = prev = indices[0]
    for x in indices[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append((start, prev))
        start = prev = x
    out.append((start, prev))
    return out


def _tile_slice(n: int, k: int, j: int) -> tuple[slice, slice]:
    return slice(k, k + n), slice(k + j * (n + k), k + j * (n + k) + n)


def _infer_layout(g: np.ndarray) -> tuple[int, int, list[np.ndarray]]:
    h, w = map(int, g.shape)
    border = int(g[0, 0])

    row_runs = _runs([r for r in range(h) if np.all(g[r, :] == border)])
    col_runs = _runs([c for c in range(w) if np.all(g[:, c] == border)])
    assert len(row_runs) == 2, f"expected 2 horizontal border bands, got {row_runs}"
    assert len(col_runs) == 5, f"expected 5 vertical border bands, got {col_runs}"

    top0, top1 = row_runs[0]
    bot0, bot1 = row_runs[1]
    assert top0 == 0, "top border must start at row 0"
    k = top1 - top0 + 1
    assert 1 <= k <= 3, f"border width must be 1..3, got k={k}"
    assert bot1 == h - 1 and bot0 == h - k, "bottom border thickness mismatch"
    assert bot1 - bot0 + 1 == k, "top/bottom border widths differ"
    assert all(end - start + 1 == k for start, end in col_runs), "separator widths differ"

    n = h - 2 * k
    assert n >= 5 and n % 2 == 1, f"tile size must be odd >= 5, got n={n}"
    assert w == 4 * n + 5 * k, f"grid width {w} incompatible with n={n}, k={k}"

    expected_col_runs = [(j * (n + k), j * (n + k) + k - 1) for j in range(5)]
    assert col_runs == expected_col_runs, f"separator columns mismatch: {col_runs} vs {expected_col_runs}"

    tiles = []
    for j in range(4):
        rs, cs = _tile_slice(n, k, j)
        tiles.append(g[rs, cs].copy())
    return n, k, tiles


def _valid_indicator_positions(n: int) -> set[tuple[int, int]]:
    last = n - 1
    mid = n // 2
    return {
        (0, 0),
        (0, last),
        (last, 0),
        (last, last),
        (0, mid),
        (mid, 0),
        (mid, last),
        (last, mid),
    }


def _reflection_for_indicator(n: int, indicator: tuple[int, int]):
    last = n - 1
    mid = n // 2
    ir, ic = indicator
    dr, dc = mid - ir, mid - ic
    if dr == 0:
        return lambda r, c: (last - r, c)
    if dc == 0:
        return lambda r, c: (r, last - c)
    if dr * dc > 0:
        return lambda r, c: (c, r)
    return lambda r, c: (last - c, last - r)


def _flood_inside(n: int, wall: frozenset[tuple[int, int]], indicator: tuple[int, int]) -> set[tuple[int, int]]:
    tmp = np.zeros((n, n), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    return {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}


def _wall_is_valid(n: int, indicator: tuple[int, int], wall: frozenset[tuple[int, int]]) -> bool:
    mid = n // 2
    last = n - 1
    if indicator in wall or (mid, mid) not in wall:
        return False
    if indicator not in _valid_indicator_positions(n):
        return False
    if any((nr, nc) in wall for nr, nc in neighbors8(*indicator)):
        return False
    refl = _reflection_for_indicator(n, indicator)
    if wall != {refl(r, c) for r, c in wall}:
        return False

    touched = set()
    for r, c in wall:
        if r == 0:
            touched.add("top")
        if r == last:
            touched.add("bottom")
        if c == 0:
            touched.add("left")
        if c == last:
            touched.add("right")
    if len(touched) < 2:
        return False

    inside = _flood_inside(n, wall, indicator)
    outside = {(r, c) for r in range(n) for c in range(n)} - inside - wall
    return bool(inside) and bool(outside)


def _ray(n: int, start: tuple[int, int], dr: int, dc: int) -> list[tuple[int, int]]:
    out = []
    r, c = start
    while 0 <= r < n and 0 <= c < n:
        out.append((r, c))
        r += dr
        c += dc
    return out


@lru_cache(maxsize=None)
def _wall_configs(n: int) -> tuple[tuple[tuple[int, int], frozenset[tuple[int, int]]], ...]:
    mid = n // 2
    last = n - 1

    row = frozenset((mid, c) for c in range(n))
    col = frozenset((r, mid) for r in range(n))
    diag = frozenset((i, i) for i in range(n))
    anti = frozenset((i, last - i) for i in range(n))

    recipes = [
        ((0, mid), row),
        ((last, mid), row),
        ((mid, 0), col),
        ((mid, last), col),
        ((0, 0), anti),
        ((last, last), anti),
        ((0, last), diag),
        ((last, 0), diag),
        ((0, 0), frozenset(_ray(n, (mid, mid), -1, 0) + _ray(n, (mid, mid), 0, -1))),
        ((0, 0), frozenset(_ray(n, (mid, mid), 1, 0) + _ray(n, (mid, mid), 0, 1))),
        ((last, last), frozenset(_ray(n, (mid, mid), -1, 0) + _ray(n, (mid, mid), 0, -1))),
        ((last, last), frozenset(_ray(n, (mid, mid), 1, 0) + _ray(n, (mid, mid), 0, 1))),
        ((0, last), frozenset(_ray(n, (mid, mid), -1, 0) + _ray(n, (mid, mid), 0, 1))),
        ((0, last), frozenset(_ray(n, (mid, mid), 1, 0) + _ray(n, (mid, mid), 0, -1))),
        ((last, 0), frozenset(_ray(n, (mid, mid), -1, 0) + _ray(n, (mid, mid), 0, 1))),
        ((last, 0), frozenset(_ray(n, (mid, mid), 1, 0) + _ray(n, (mid, mid), 0, -1))),
        ((last, mid), frozenset(_ray(n, (mid, mid), -1, -1) + _ray(n, (mid, mid), -1, 1))),
        ((0, mid), frozenset(_ray(n, (mid, mid), 1, -1) + _ray(n, (mid, mid), 1, 1))),
        ((mid, last), frozenset(_ray(n, (mid, mid), -1, -1) + _ray(n, (mid, mid), 1, -1))),
        ((mid, 0), frozenset(_ray(n, (mid, mid), -1, 1) + _ray(n, (mid, mid), 1, 1))),
    ]

    unique = []
    seen = set()
    for indicator, wall in recipes:
        key = (indicator, tuple(sorted(wall)))
        if key in seen or not _wall_is_valid(n, indicator, wall):
            continue
        seen.add(key)
        unique.append((indicator, wall))
    assert unique, f"no valid wall configs for n={n}"
    return tuple(unique)


def _is_straight_wall(n: int, wall: frozenset[tuple[int, int]]) -> bool:
    last = n - 1
    return (
        all(r == n // 2 for r, _ in wall)
        or all(c == n // 2 for _, c in wall)
        or all(r == c for r, c in wall)
        or all(r + c == last for r, c in wall)
    )


@lru_cache(maxsize=None)
def _orbits(n: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    last = n - 1
    seen: set[tuple[int, int]] = set()
    out = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orbit = sorted({(r, c), (c, r), (last - c, last - r), (last - r, last - c)})
            orbit_t = tuple(orbit)
            seen.update(orbit_t)
            out.append(orbit_t)
    return tuple(out)


def _sample_symmetric_pattern(
    n: int,
    bg: int,
    fg: int,
    rng: random.Random,
    *,
    min_px: int,
    max_px: int,
) -> np.ndarray | None:
    area = n * n
    assert max_px < area - max_px, "background must stay majority"
    orbits = list(_orbits(n))
    for _ in range(PATTERN_RETRIES):
        order = list(range(len(orbits)))
        rng.shuffle(order)
        p = np.full((n, n), bg, dtype=int)
        count = 0
        for idx in order:
            orbit = orbits[idx]
            if count + len(orbit) > max_px:
                continue
            if rng.random() < ORBIT_INCLUDE_PROB:
                for r, c in orbit:
                    p[r, c] = fg
                count += len(orbit)
        if min_px <= count <= max_px:
            return p
    return None


def _compose_output_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    wall: frozenset[tuple[int, int]],
    inside: set[tuple[int, int]],
) -> np.ndarray:
    n = t1.shape[0]
    wall_mask = np.zeros((n, n), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    wall_4conn = n_comp == 1

    out = np.full((n, n), bg, dtype=int)
    for r in range(n):
        for c in range(n):
            v1, v2 = int(t1[r, c]), int(t2[r, c])
            if (r, c) in inside:
                if v1 != bg:
                    out[r, c] = v1
            elif (r, c) in wall:
                if v1 != bg:
                    out[r, c] = v1
                elif wall_4conn and v2 != bg:
                    out[r, c] = v2
            elif v2 != bg:
                out[r, c] = v2
    return out


def _layout_signature(grid: np.ndarray) -> tuple[int, int]:
    n, k, _tiles = _infer_layout(np.array(grid, dtype=int))
    return n, k


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two source panels into the fourth panel after parsing a variable layout.

    Input structure:
    - Four square panels appear in one row.
    - All outer borders and all inter-panel separators share one uniform color and
      a uniform width k, where k may be 1, 2, or 3.
    - Each panel has odd side length n. Panel 0 contains cue pixels (wall +
      indicator), panels 1 and 2 contain source patterns, and panel 3 is blank.

    Rule:
    - Infer border width k and panel size n from the border bands, then extract
      the four panels.
    - In panel 0, the unique 8-isolated cue pixel is the indicator and the other
      cue pixels are the wall. Flood-fill from the indicator using 4-connectivity,
      blocked by the wall.
    - Inside the flood-filled region, copy non-background pixels from panel 1.
      Outside the wall, copy non-background pixels from panel 2.
      On wall cells, prefer panel 1; if the wall is 4-connected, panel 2 may
      fill remaining wall cells as a fallback.

    Key insight:
    - The semantic rule is the original one; only the preprocessing changes.
      Solving the combo means recovering `(n, k)` from the border geometry before
      reusing the original indicator/wall compositing logic inside the panels.
    """
    g = np.array(grid, dtype=int)
    n, k, tiles = _infer_layout(g)
    t0, t1, t2, _t3 = tiles

    border = int(g[0, 0])
    bg = int(Counter(map(int, t1.flatten())).most_common(1)[0][0])
    assert border != bg, "border color must differ from tile background"

    t0_bg = int(Counter(map(int, t0.flatten())).most_common(1)[0][0])
    cue_set = {(int(r), int(c)) for r in range(n) for c in range(n) if int(t0[r, c]) != t0_bg}
    assert cue_set, "no cue pixels found"

    indicator = None
    for r, c in cue_set:
        if not any((nr, nc) in cue_set for nr, nc in neighbors8(r, c)):
            assert indicator is None, "multiple 8-isolated cue pixels"
            indicator = (r, c)
    assert indicator is not None, "no 8-isolated cue pixel found"
    assert indicator in _valid_indicator_positions(n), f"indicator {indicator} not at corner/edge-midpoint"

    wall = frozenset(cue_set - {indicator})
    assert _wall_is_valid(n, indicator, wall), "invalid wall geometry"

    inside = _flood_inside(n, wall, indicator)
    out3 = _compose_output_tile(t1, t2, bg, wall, inside)

    out = g.copy()
    rs, cs = _tile_slice(n, k, 3)
    out[rs, cs] = out3
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.choice(SIZE_CHOICES)
    k = rng.choice(BORDER_WIDTH_CHOICES)

    tile_bg, border, cue_color, t1_color, t2_color = rng.sample(range(NUM_COLORS), 5)

    configs = _wall_configs(n)
    straight = [(indicator, wall) for indicator, wall in configs if _is_straight_wall(n, wall)]
    bends = [cfg for cfg in configs if cfg not in straight]
    pool = bends if bends and rng.random() < LBEND_PROB else straight or configs
    indicator, wall = rng.choice(pool)
    inside = _flood_inside(n, wall, indicator)
    outside = {(r, c) for r in range(n) for c in range(n)} - inside - wall

    bounds = PATTERN_BOUNDS[n]
    t1 = _sample_symmetric_pattern(n, tile_bg, t1_color, rng, min_px=bounds["t1"][0], max_px=bounds["t1"][1])
    t2 = _sample_symmetric_pattern(n, tile_bg, t2_color, rng, min_px=bounds["t2"][0], max_px=bounds["t2"][1])
    if t1 is None or t2 is None:
        return None

    if not any(int(t1[r, c]) != tile_bg for r, c in inside):
        return None
    if not any(int(t2[r, c]) != tile_bg for r, c in outside):
        return None

    t0 = np.full((n, n), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = cue_color
    t0[indicator[0], indicator[1]] = cue_color

    out3 = _compose_output_tile(t1, t2, tile_bg, wall, inside)
    if np.array_equal(out3, np.full((n, n), tile_bg, dtype=int)):
        return None

    g = np.full((n + 2 * k, 4 * n + 5 * k), border, dtype=int)
    tiles = [t0, t1, t2, np.full((n, n), tile_bg, dtype=int)]
    for j, tile in enumerate(tiles):
        rs, cs = _tile_slice(n, k, j)
        g[rs, cs] = tile

    expected = g.copy()
    rs, cs = _tile_slice(n, k, 3)
    expected[rs, cs] = out3
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
