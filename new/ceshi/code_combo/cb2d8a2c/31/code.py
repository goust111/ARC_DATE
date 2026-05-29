# Combo variant of cb2d8a2c (index 31)
# combo_spec: D3/1+D8/0+D19/1
# Name: 间距编码距离+位置编码方向+形状标记

# Design analysis for the combo variant D3/1 + D8/0 + D19/1
# a) Concrete appearance + behavior trace:
#    The original task used one-cell-thick wall-attached bars whose alternating blue cells directly counted the
#    keep-out radius. In this fusion, each bar still attaches to a wall and still blocks a green top-to-bottom
#    route, but its payload is now split into three visible layers:
#      - a 3x3 red cap at the free end contains one blue symbol: dot / plus / X;
#      - two extra blue guide pixels sit on the red stem;
#      - the two guide pixels are placed either near the wall end or near the free-end cap.
#    Example, left-attached horizontal bar:
#      old: `1 2 1 2 1 2 2 ...` -> radius = number of blue cells.
#      new: `[blue-pair on stem] ... [3x3 cap with X]`.
#      The X says "large band", the gap inside the blue pair says whether that large band means 5 or 6, and the
#      pair's overall location says whether the route should use the outer wall lane or the lane just outside the
#      obstacle halo. The output path therefore differs in both detour width and chosen crossing column.
# b) Complexity calibration:
#    The exact radius has only six values (1..6), grouped into three shape bands:
#      - dot  -> small (1 or 2)
#      - plus -> medium (3 or 4)
#      - X    -> large (5 or 6)
#    That keeps the mapping learnable from a few examples while making D19 visible. D3 stays active because the
#    same symbol can produce two radii depending on the guide-pixel spacing. D8 stays active because the same bar
#    width and radius can still yield two different routes depending on whether the guide pair hugs the wall end
#    or the cap end.
# c) Hidden-assumption audit against the original main():
#    - `blue_count = sum(...)` is invalid: total blue count no longer equals radius, and plus/X share the same
#      cap blue-count before adding the guide pair.
#    - Generic shortest-path search is no longer enough to express the fused rule, because D8 introduces a lane
#      preference even when several legal columns exist. The solver must decode a preferred crossing column per
#      bar and honor it deliberately.
#    - Original bars were 1-pixel thick everywhere. Here each component has a 3x3 end-cap, so spacing along the
#      main axis must leave room for a 3-row obstacle body plus its Manhattan halo.
#    - The old alternating blue/red stem pattern is gone. Any solver logic that assumes all blue cells lie on one
#      straight alternating strip would misread both the cap symbol and the two guide pixels.
# d) Placement legality:
#    The generator guarantees that every obstacle component:
#      - is a single 4-connected wall-attached horizontal bar after canonical rotation,
#      - has exactly one 3x3 cap fully inside the grid,
#      - has exactly two stem guide pixels outside the cap,
#      - leaves a legal open corridor wider than the decoded radius on the non-wall side,
#      - is separated from neighboring bars by a blank band between their forbidden halos.
# e) Visual observability:
#    A human can compare two bars with the same symbol but different guide-pair spacing and see that the output
#    gives them different clearances. They can also compare two bars with the same symbol and spacing but with the
#    guide pair near different ends and see that the path chooses different lanes. Because dot/plus/X visibly
#    partition the radii into small/medium/large groups, the three layers reinforce each other instead of hiding.
# f) Visual cleanliness:
#    The bar body remains a simple red stem plus one 3x3 cap, so the added coding stays localized. The blue pair
#    is confined to the stem centerline, the symbol stays inside the cap, and the green path runs in empty space.
#    There is no ambiguous overlap between legend pixels and traversed pixels in the input.
# g) Change inferability:
#    The palette remains the same four colors as the source task: azure background, red obstacles, blue code
#    pixels, green route. Blue still turns red in the output, but the structure of that blue now explains both
#    "how far away" and "which lane", so the recolor is still inferable from geometry rather than a raw palette
#    permutation.
# 9) New main() assumptions and generator guarantees:
#    - After rotating the grid so the green source is on the top edge, every obstacle component is horizontal and
#      touches either the left wall or the right wall.
#    - Every component bbox is exactly 3 rows tall and at least 7 columns wide.
#    - The cap occupies the free end of the bbox; the two guide pixels lie on the middle stem row outside the cap.
#    - The guide-pair spacing is either 0 or 1 red cells, yielding exact radii 1..6 by shape band + spacing.
#    - The guide pair is placed clearly in the wall half or cap half of the stem, so the direction bit is
#      decodable by comparing the pair midpoint with the stem midpoint.

import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "arc_utils.py").exists():
        sys.path.insert(0, str(_parent))
        break

from arc_utils import components_mask, neighbors4

BG = 8
BLUE = 1
RED = 2
GREEN = 3

ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 2, 4
RADIUS_LO, RADIUS_HI = 1, 6
BAR_LEN_MIN, BAR_LEN_MAX = 7, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 500

_DOT = frozenset({(1, 1)})
_PLUS = frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)})
_X = frozenset({(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)})

_BASE_TO_SHAPE = {
    1: _DOT,
    3: _PLUS,
    5: _X,
}
_SHAPE_TO_BASE = {shape: base for base, shape in _BASE_TO_SHAPE.items()}
_BASE_TO_MASK = {
    base: np.array(
        [[(r, c) in shape for c in range(3)] for r in range(3)],
        dtype=bool,
    )
    for base, shape in _BASE_TO_SHAPE.items()
}


def _radius_to_base_spacing(radius: int) -> tuple[int, int]:
    assert RADIUS_LO <= radius <= RADIUS_HI
    band = (radius - 1) // 2
    base = 1 + 2 * band
    spacing = radius - base
    return base, spacing


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _min_main_len(radii: tuple[int, ...]) -> int:
    last_center = radii[0] + 2
    for prev_radius, radius in zip(radii, radii[1:]):
        last_center += prev_radius + radius + 4
    return last_center + radii[-1] + 3


def _length_bounds(cross_size: int, radius: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, cross_size - radius - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _to_canonical(grid: np.ndarray) -> tuple[np.ndarray, str]:
    g = np.array(grid, copy=True)
    starts = np.argwhere(g == GREEN)
    assert starts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, starts[0])
    h, w = g.shape
    if sr == 0:
        return g, "top"
    if sr == h - 1:
        return np.flipud(g).copy(), "bottom"
    if sc == 0:
        return g.T.copy(), "left"
    if sc == w - 1:
        return np.fliplr(g).T.copy(), "right"
    raise AssertionError("green start must be on a border edge")


def _from_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _cap_shape(cap: np.ndarray) -> int:
    blue_shape = frozenset((r, c) for r in range(3) for c in range(3) if int(cap[r, c]) == BLUE)
    assert blue_shape in _SHAPE_TO_BASE, f"unsupported cap shape: {sorted(blue_shape)}"
    return _SHAPE_TO_BASE[blue_shape]


def _decode_bars(g: np.ndarray, *, use_spacing: bool, use_direction: bool) -> list[dict]:
    h, w = g.shape
    bar_mask = (g == BLUE) | (g == RED)
    comps = components_mask(bar_mask, connectivity=4)
    bars: list[dict] = []
    for comp in comps:
        rows = comp[:, 0]
        cols = comp[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        assert r1 - r0 + 1 == 3, "canonical bars must be 3 rows tall"
        assert c0 == 0 or c1 == w - 1, "each bar must touch a side wall"
        side = 0 if c0 == 0 else 1
        if side == 0:
            cap = g[r0 : r1 + 1, c1 - 2 : c1 + 1]
            stem_lo, stem_hi = c0, c1 - 3
        else:
            cap = g[r0 : r1 + 1, c0 : c0 + 3]
            stem_lo, stem_hi = c0 + 3, c1
        assert stem_hi >= stem_lo, "bar stem must leave room for guide pixels"

        base = _cap_shape(cap)
        guide_cols = [c for c in range(stem_lo, stem_hi + 1) if int(g[r0 + 1, c]) == BLUE]
        assert len(guide_cols) == 2, "every bar must have exactly two stem guide pixels"
        guide_cols.sort()
        spacing = guide_cols[1] - guide_cols[0] - 1
        assert spacing in (0, 1), "guide-pixel spacing must be 0 or 1"
        radius = base + (spacing if use_spacing else 0)

        stem_mid = (stem_lo + stem_hi) / 2.0
        pair_mid = (guide_cols[0] + guide_cols[1]) / 2.0
        wall_biased = pair_mid < stem_mid if side == 0 else pair_mid > stem_mid
        prefer_outer = wall_biased and use_direction

        near_col = c1 + radius + 1 if side == 0 else c0 - radius - 1
        outer_col = w - 1 if side == 0 else 0
        assert 0 <= near_col < w, "decoded radius leaves no legal near lane"
        target_col = outer_col if prefer_outer else near_col

        bars.append(
            {
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "side": side,
                "base": base,
                "spacing": spacing,
                "radius": radius,
                "prefer_outer": bool(prefer_outer),
                "target_col": int(target_col),
                "comp": comp,
            }
        )
    bars.sort(key=lambda bar: (bar["r0"] + bar["r1"], bar["c0"]))
    return bars


def _build_blocked(h: int, w: int, bars: list[dict]) -> np.ndarray:
    forbidden = np.zeros((h, w), dtype=bool)
    for bar in bars:
        radius = int(bar["radius"])
        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in bar["comp"]:
            rr, cc = int(r), int(c)
            dist[rr, cc] = 0
            q.append((rr, cc))
        while q:
            r, c = q.popleft()
            d = int(dist[r, c])
            forbidden[r, c] = True
            if d == radius:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] == -1:
                    dist[rr, cc] = d + 1
                    q.append((rr, cc))
    return forbidden


def _append_segment(path: list[tuple[int, int]], end_r: int, end_c: int) -> None:
    cur_r, cur_c = path[-1]
    if cur_r == end_r and cur_c == end_c:
        return
    if cur_r == end_r:
        step = 1 if end_c > cur_c else -1
        for c in range(cur_c + step, end_c + step, step):
            path.append((cur_r, c))
        return
    assert cur_c == end_c, "segments must be axis-aligned"
    step = 1 if end_r > cur_r else -1
    for r in range(cur_r + step, end_r + step, step):
        path.append((r, cur_c))


def _solve_canonical(g: np.ndarray, *, use_spacing: bool = True, use_direction: bool = True) -> np.ndarray:
    h, w = g.shape
    starts = np.argwhere(g == GREEN)
    assert starts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, starts[0])
    assert sr == 0, "canonical grid must start from the top edge"

    bars = _decode_bars(g, use_spacing=use_spacing, use_direction=use_direction)
    blocked = _build_blocked(h, w, bars)
    blocked[sr, sc] = False

    path = [(sr, sc)]
    for bar in bars:
        approach_r = bar["r0"] - bar["radius"] - 1
        exit_r = bar["r1"] + bar["radius"] + 1
        assert approach_r >= path[-1][0], "bars must be ordered with non-overlapping halos"
        _append_segment(path, approach_r, path[-1][1])
        _append_segment(path, approach_r, bar["target_col"])
        _append_segment(path, exit_r, bar["target_col"])
    _append_segment(path, h - 1, path[-1][1])

    for r, c in path:
        assert not blocked[r, c], "decoded route must stay outside all forbidden zones"

    out = np.array(g, copy=True)
    for r, c in path:
        if out[r, c] == BG:
            out[r, c] = GREEN
    out[out == BLUE] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route a green border-to-border path through wall-attached bars whose blue code is split across a symbol cap, a spacing pair, and a lane-bias position.

    Input structure:
    - bg = azure (8). Each obstacle is a red wall-attached bar (2) with a 3x3 red cap at its free end and exactly two extra blue guide pixels (1) on the stem.
    - The 3x3 cap holds one blue symbol (dot / plus / X), and a single green (3) start pixel lies on one border edge.

    Rule:
    - Rotate mentally so the green source is on the top edge. For each bar, the cap symbol gives the radius band, the gap between the two stem blue pixels gives the exact radius within that band, and the pair's position along the stem chooses the crossing lane.
    - Draw the green route straight toward the opposite edge, turning at right angles above each forbidden halo to reach that bar's preferred legal lane, then continue below the halo.
    - After the route is drawn, all blue instruction pixels become red.

    Key insight: Blue no longer carries one scalar by counting; the cap shape gives a coarse size class, the stem-pair spacing refines it, and the stem-pair location selects which legal corridor the path should occupy.
    """
    canonical, edge = _to_canonical(grid)
    solved = _solve_canonical(canonical, use_spacing=True, use_direction=True)
    return _from_canonical(solved, edge)


def _choose_radii(count: int, main_hi: int, cross_size: int) -> tuple[int, ...] | None:
    choices = [
        combo
        for combo in itertools.product(range(RADIUS_LO, RADIUS_HI + 1), repeat=count)
        if _min_main_len(combo) <= main_hi and all(_length_bounds(cross_size, radius) is not None for radius in combo)
    ]
    if not choices:
        return None
    diverse = [
        combo
        for combo in choices
        if len(set(combo)) >= 2
        and len({(radius - 1) // 2 for radius in combo}) >= 2
    ]
    pool = diverse if diverse else choices
    return random.choice(pool)


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, radius: int, prefer_outer: bool) -> None:
    _, w = grid.shape
    base, spacing = _radius_to_base_spacing(radius)
    stem_len = length - 3
    pair_span = spacing + 2
    assert stem_len >= pair_span

    if side == 0:
        grid[row, :stem_len] = RED
        cap = grid[row - 1 : row + 2, stem_len:length]
        cap[:, :] = RED
        cap[_BASE_TO_MASK[base]] = BLUE
        pair_start = 0 if prefer_outer else stem_len - pair_span
        guide_cols = [pair_start, pair_start + spacing + 1]
        for c in guide_cols:
            grid[row, c] = BLUE
    else:
        left = w - length
        cap = grid[row - 1 : row + 2, left : left + 3]
        cap[:, :] = RED
        cap[_BASE_TO_MASK[base]] = BLUE
        stem_lo = left + 3
        grid[row, stem_lo:w] = RED
        pair_start = stem_len - pair_span if prefer_outer else 0
        guide_cols = [stem_lo + pair_start, stem_lo + pair_start + spacing + 1]
        for c in guide_cols:
            grid[row, c] = BLUE


def _canonical_grid(
    main_size: int,
    cross_size: int,
    radii: tuple[int, ...],
    first_side: int,
    preferences: tuple[bool, ...],
) -> np.ndarray | None:
    count = len(radii)
    slack = main_size - _min_main_len(radii)
    if slack < 0:
        return None

    extras = _random_composition(slack, count + 1)
    row = radii[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, count):
        row += radii[i - 1] + radii[i] + 4 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    lengths: list[int] = []
    for row, radius, side, prefer_outer in zip(
        rows,
        radii,
        ((first_side + i) % 2 for i in range(count)),
        preferences,
    ):
        bounds = _length_bounds(cross_size, radius)
        if bounds is None:
            return None
        length = random.randint(*bounds)
        lengths.append(length)
        _paint_bar(grid, row, side, length, radius, prefer_outer)

    first_len = lengths[0]
    if first_side == 0:
        start_col = random.randint(1, first_len - 2)
    else:
        start_col = random.randint(cross_size - first_len + 1, cross_size - 2)
    grid[0, start_col] = GREEN
    return grid


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    return bool(np.any(out[:, 0] == GREEN))


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    cross_size = random.randint(cross_lo, cross_hi)
    feasible_counts = [
        count for count in range(BAR_COUNT_LO, BAR_COUNT_HI + 1) if _choose_radii(count, main_hi, cross_size) is not None
    ]
    if not feasible_counts:
        return None
    count = random.choice(feasible_counts)
    radii = _choose_radii(count, main_hi, cross_size)
    if radii is None:
        return None

    main_size = random.randint(max(main_lo, _min_main_len(radii)), main_hi)
    first_side = random.randrange(2)
    if count == 1:
        preferences = (bool(random.getrandbits(1)),)
    else:
        preferences = tuple(random.choice((False, True)) for _ in range(count))
        if all(preferences) or not any(preferences):
            flip_idx = random.randrange(count)
            preferences = tuple((not pref) if i == flip_idx else pref for i, pref in enumerate(preferences))

    canonical = _canonical_grid(main_size, cross_size, radii, first_side, preferences)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    try:
        out = main(inp)
        if not _reaches_opposite_edge(out, edge):
            return None
        if not np.array_equal(main(inp), out):
            return None
        # Reject degenerate samples where removing either spacing or direction leaves the same output.
        canonical_inp, canonical_edge = _to_canonical(inp)
        fused = _solve_canonical(canonical_inp, use_spacing=True, use_direction=True)
        shape_only = _solve_canonical(canonical_inp, use_spacing=False, use_direction=False)
        direction_only = _solve_canonical(canonical_inp, use_spacing=True, use_direction=False)
        if np.array_equal(fused, shape_only):
            return None
        if np.array_equal(fused, direction_only):
            return None
        if canonical_edge != edge:
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c combo input")
