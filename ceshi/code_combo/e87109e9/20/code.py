# Combo variant of e87109e9 (index 20)
# combo_spec: D6/1+D13/1
# Name: 图例编码碰撞次数加转弯限制

"""
design_analysis:
  concrete_variant:
    keep_from_source:
      - input still has a 6x24 legend band on top and a 24-column main area below
      - the main area still contains one solid teal square and four edge-touching obstacle rectangles
      - obstacle color still matches legend color directly
    fused_change:
      - each legend bar continues to use left/right placement for turn direction
      - bar height now encodes that color's total number of turning collisions before later hits pass straight through
      - in addition, every beam has a global limit of 2 turns; when a beam reaches a third turn opportunity it stops instead of turning
    trace_example:
      - let the north obstacle turn left into west, west turn left into south, and the south color have quota 1
      - the north beam hits north, then west, then arrives at south as its third obstacle
      - under the original task it would turn again at south and then continue east across open background
      - under D13 alone it would also turn at south because south quota is still unused
      - under the fused rule it stops before turning because the beam has already spent 2 turns
      - then the east beam reaches south second and spends the only south collision
      - later south- or west-origin beams hitting south no longer turn and instead pass through that obstacle
    visible_difference:
      - one branch ends flush against an obstacle because of the per-beam turn cap
      - another branch penetrates a previously used obstacle color because that color's quota has been exhausted
  complexity_calibration:
    choices:
      - keep only four obstacle colors and four legend entries so the matching remains human-trackable
      - use a fixed global turn cap of 2, which is small enough to infer from a few examples
      - force every generated sample to contain both a stop event and at least one pass-through event
    why_not_more_complex:
      - adding more than one global cap value would burden inference without adding useful structure
      - allowing arbitrary obstacle layouts would make multi-counter interactions visually noisy
  implicit_assumptions_review:
    original_assumptions_that_break:
      - legend parsing can no longer treat each colored bar as a pure left/right token because height now matters
      - hitting an obstacle no longer always implies a turn
      - a beam reaching an obstacle after two previous turns can no longer keep following the old recurrence
    new_solver_assumptions:
      - each legend cell contains exactly one contiguous bottom-aligned vertical bar
      - the beam footprint never overlaps two obstacle colors at once
      - if a beam is already inside an obstacle due to pass-through, the next step cannot switch obstacle colors mid-block
      - the main area is generated so that the intended paths do not cycle
    generator_guards:
      - all four obstacle colors appear in the legend
      - the east obstacle span explicitly avoids the south lane, while the west span also covers the north lane to realize the 3-hit north path without introducing loops
      - every candidate grid is validated against the fused solver and against alternate simulations
  placement_legality:
    - obstacle rectangles are chosen from non-overlapping edge-touching spans derived from the square position and obstacle thicknesses
    - required span intervals guarantee each planned collision exists
    - excluded span intervals prevent unintended repeat collisions that would create cycles or ambiguous traces
    - the teal square is placed with a margin from all obstacles and boundaries, preserving clear beam corridors
  visual_observability:
    - the color quota is visible from bar height alone, with values restricted to 1..4 because the legend interior has four rows
    - the turn direction is still readable from bar side
    - the global turn cap is visible because one beam reliably dies on its third obstacle while another beam still turns on its second obstacle
    - the generator rejects any sample where the fused output matches the source logic, the quota-only logic, or the turn-cap-only logic
  neatness_and_inferability:
    - the layout stays close to the original sparse edge-bar style, so the new behavior is carried by path differences rather than clutter
    - color changes are not used as hidden indices; only geometry and visible legend tokens matter
    - each accepted sample contains repeated evidence for the south obstacle's quota rule, making the multi-counter interaction inferable from few examples
"""

import random

import numpy as np

from arc_utils import bbox_coords, components_mask

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)

# Color palettes
LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
LEGEND_BG = 5
TEAL = 8

# Canvas dimensions
GRID_H_RANGE = (19, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]

# Teal square
TEAL_SZ2_PROB = 0.75
SQUARE_EDGE_GAP = 5

# Obstacles
OBS_THICKNESS_RANGE = (2, 4)
OBS_BOTH_GAP = 2

# Span selection
SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.4

# Fused rule
MAX_TURNS_PER_BEAM = 2

# Retry budget
MAX_RETRIES = 500


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, tuple[str, int]]:
    """
    Legend is a top region with a background color (typically 5) and 4 symbol cells.
    Each cell contains one bottom-aligned vertical bar:
      - bar on left  -> 'L' (turn counterclockwise)
      - bar on right -> 'R' (turn clockwise)
      - bar height   -> max number of turning collisions for that color
    """
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    rules: dict[int, tuple[str, int]] = {}
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        interior = legend[1 : legend_h - 1, a + 1 : b]
        colors = set(map(int, np.unique(interior)))
        colors.discard(bg)
        colors.discard(0)
        assert len(colors) == 1, "each legend cell must encode exactly one color"
        color = int(next(iter(colors)))
        ys, xs = np.where(interior == color)
        assert ys.size >= 1, "legend bar must have positive height"
        x = int(xs[0])
        assert np.all(xs == x), "legend bar must be a single vertical strip"
        quota = int(ys.size)
        expect = np.arange(interior.shape[0] - quota, interior.shape[0])
        assert np.array_equal(np.sort(ys), expect), "legend bar must be bottom-aligned and contiguous"
        turn = "L" if x <= (interior.shape[1] - 1) // 2 else "R"
        assert color not in rules, "legend colors must be distinct"
        rules[color] = (turn, quota)
    assert len(rules) == 4, "legend must encode four colors"
    return rules


def _find_teal_square(g: np.ndarray, teal: int = TEAL) -> tuple[int, int, int]:
    """Return (r0,c0,size) for a solid square teal component."""
    candidates: list[tuple[int, int, int]] = []
    for cells in components_mask(g == teal, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh != ww:
            continue
        if int(np.sum(g[r0 : r1 + 1, c0 : c1 + 1] == teal)) != hh * ww:
            continue
        candidates.append((int(r0), int(c0), int(hh)))
    assert len(candidates) == 1, "main grid must contain exactly one solid teal square"
    return candidates[0]


def _rot(d: int, turn: str) -> int:
    return (d - 1) % 4 if turn == "L" else (d + 1) % 4


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four teal beams from a square, with obstacle colors carrying finite collision quotas and beams carrying a finite turn budget.

    Input structure:
    - The top band is a grey legend split into four black cells, each holding one colored vertical bar.
    - Bar side still tells whether that obstacle color turns beams left or right.
    - Bar height now tells how many turning collisions that color permits before later hits become straight pass-throughs.
    - Below the legend is the main area with one solid teal square, one uniform background color, and colored obstacle rectangles touching the borders.

    Rule:
    - Remove the legend from the output.
    - Launch one square-shaped teal beam in each cardinal direction from the teal source.
    - A beam paints teal only on background cells as it moves.
    - When a beam reaches a fresh obstacle color, first check that color's remaining collision quota. If the quota is exhausted, the beam enters that obstacle and continues straight through it.
    - If the quota is still available, the beam may turn only if it has used fewer than 2 turns already. A successful turn spends both one color collision and one beam turn. If the beam has already used 2 turns, it stops at that obstacle instead of turning.

    Key insight: the solver must maintain two independent counters at once: a global per-beam turn count and a per-color turning-collision count decoded from legend bar height.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    legend_rules = _parse_legend(g, legend_h)

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    sr, sc, sz = _find_teal_square(base, teal=TEAL)

    out = base.copy()
    turn_used_by_color = {color: 0 for color in legend_rules}

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < out.shape[0] and c + sz - 1 < out.shape[1]

    def paint_block(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = TEAL
        out[r : r + sz, c : c + sz] = blk

    def obstacle_color_at(r: int, c: int) -> int | None:
        blk = out[r : r + sz, c : c + sz]
        cols = set(map(int, np.unique(blk)))
        cols.discard(bg)
        cols.discard(TEAL)
        if not cols:
            return None
        assert len(cols) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(cols)))

    for d0 in range(4):
        r, c, d = sr, sc, d0
        turns_used = 0
        seen: set[tuple[int, int, int, int]] = set()
        while True:
            state = (r, c, d, turns_used)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            current_obstacle = obstacle_color_at(r, c)
            next_obstacle = obstacle_color_at(nr, nc)
            if next_obstacle is None:
                r, c = nr, nc
                paint_block(r, c)
                continue

            if current_obstacle is not None:
                assert current_obstacle == next_obstacle, "penetration cannot switch obstacle colors mid-crossing"
                r, c = nr, nc
                paint_block(r, c)
                continue

            assert next_obstacle in legend_rules, "obstacle color must be encoded by legend"
            turn, quota = legend_rules[next_obstacle]
            if turn_used_by_color[next_obstacle] >= quota:
                r, c = nr, nc
                paint_block(r, c)
                continue
            if turns_used >= MAX_TURNS_PER_BEAM:
                break

            turn_used_by_color[next_obstacle] += 1
            turns_used += 1
            d = _rot(d, turn)

    return out


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a (start, end) span covering all required intervals and avoiding excluded ones."""
    req0 = min(a for a, _ in required)
    req1 = max(b for _, b in required)
    lo = max(min_len, req1 - req0 + 1)
    hi = min(max_len, axis_len)
    choices: list[tuple[int, int, int]] = []
    for length in range(lo, hi + 1):
        start0 = max(0, req1 - length + 1)
        start1 = min(req0, axis_len - length)
        for start in range(start0, start1 + 1):
            end = start + length - 1
            if all(end < a or start > b for a, b in excluded):
                choices.append((length, start, end))
    if not choices:
        return None

    lengths = sorted({length for length, _, _ in choices})
    if random.random() < SPAN_COMPACT_PROB:
        cutoff = lengths[min(len(lengths) - 1, max(0, len(lengths) // 2))]
        pool = [(start, end) for length, start, end in choices if length <= cutoff]
    else:
        pool = [(start, end) for _, start, end in choices]
    return random.choice(pool)


def _rand_thickness(lo: int, hi: int) -> int | None:
    return random.randint(lo, hi) if hi >= lo else None


def _thickness_pair(axis: int, sz: int) -> tuple[int, int] | None:
    """Compute obstacle thicknesses for both ends of an axis (N/S or W/E)."""
    lo, hi_cap = OBS_THICKNESS_RANGE
    room = axis - sz - OBS_BOTH_GAP
    if room < 2 * lo:
        return None
    a = random.randint(lo, min(hi_cap, room - lo))
    b = _rand_thickness(lo, min(hi_cap, room - a))
    return (a, b) if b is not None else None


def _obstacle_rect(side: int, thickness: int, span: tuple[int, int], h: int, w: int) -> tuple[int, int, int, int]:
    """Return (r0, r1, c0, c1) for an edge-touching obstacle rectangle."""
    s0, s1 = span
    if side == _N:
        return 0, thickness - 1, s0, s1
    if side == _S:
        return h - thickness, h - 1, s0, s1
    if side == _W:
        return s0, s1, 0, thickness - 1
    return s0, s1, w - thickness, w - 1


def _make_legend(legend_colors: list[int], rules: dict[int, tuple[str, int]]) -> np.ndarray:
    """Build the 6x24 legend band from 4 colors and their (turn, quota) rules."""
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        turn, quota = rules[color]
        bar_col = start if turn == "L" else start + 3
        legend[5 - quota : 5, bar_col] = color
    return legend


def _simulate_paths(
    base: np.ndarray,
    rules: dict[int, tuple[str, int]],
    bg: int,
    *,
    use_quotas: bool,
    max_turns: int | None,
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int, str]]], dict[int, int], int, int]:
    """Simulate beams under configurable quota and turn-cap rules."""
    vals, counts = np.unique(base, return_counts=True)
    assert int(vals[counts.argmax()]) == bg

    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    out = base.copy()
    h, w = out.shape
    beam_events: list[list[tuple[int, int, int, int, str]]] = []
    turn_used_by_color = {color: 0 for color in rules}
    pass_events = 0
    stop_events = 0

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = TEAL
        out[r : r + sz, c : c + sz] = blk

    def obstacle_at(r: int, c: int) -> int | None:
        colors = set(map(int, np.unique(out[r : r + sz, c : c + sz])))
        colors.discard(bg)
        colors.discard(TEAL)
        if not colors:
            return None
        assert len(colors) == 1
        return int(next(iter(colors)))

    for d0 in range(4):
        r, c, d = int(sr), int(sc), d0
        turns_used = 0
        events: list[tuple[int, int, int, int, str]] = []
        seen: set[tuple[int, int, int, int]] = set()
        while True:
            state = (r, c, d, turns_used)
            assert state not in seen
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            current_obs = obstacle_at(r, c)
            next_obs = obstacle_at(nr, nc)
            if next_obs is None:
                r, c = nr, nc
                paint(r, c)
                continue

            if current_obs is not None:
                assert current_obs == next_obs
                r, c = nr, nc
                paint(r, c)
                continue

            assert next_obs in rules
            turn, quota = rules[next_obs]
            if use_quotas and turn_used_by_color[next_obs] >= quota:
                events.append((d, next_obs, nr, nc, "pass"))
                pass_events += 1
                r, c = nr, nc
                paint(r, c)
                continue
            if max_turns is not None and turns_used >= max_turns:
                events.append((d, next_obs, nr, nc, "stop"))
                stop_events += 1
                break

            turn_used_by_color[next_obs] += 1
            turns_used += 1
            events.append((d, next_obs, nr, nc, "turn"))
            d = _rot(d, turn)
        beam_events.append(events)

    return out, beam_events, turn_used_by_color, pass_events, stop_events


def _gen_attempt() -> tuple[np.ndarray, dict[str, int]] | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3

    ns = _thickness_pair(h, sz)
    ew = _thickness_pair(w, sz)
    if ns is None or ew is None:
        return None
    thickness = {_N: ns[0], _S: ns[1], _W: ew[0], _E: ew[1]}

    min_r = max(SQUARE_EDGE_GAP, thickness[_N] + 2)
    max_r = min(h - sz - SQUARE_EDGE_GAP, h - thickness[_S] - sz - 2)
    min_c = max(SQUARE_EDGE_GAP, thickness[_W] + 2)
    max_c = min(w - sz - SQUARE_EDGE_GAP, w - thickness[_E] - sz - 2)
    if min_r > max_r or min_c > max_c:
        return None
    sr = random.randint(min_r, max_r)
    sc = random.randint(min_c, max_c)

    lane = {
        _N: thickness[_N] + 1,
        _S: h - thickness[_S] - sz,
        _W: thickness[_W] + 1,
        _E: w - thickness[_E] - sz,
    }

    interval = lambda start: (start, start + sz - 1)
    min_span_len = SPAN_MIN_MULT * sz
    max_span_len = min(SPAN_MAX_MULT * sz, max(h, w))

    span_n = _choose_span(
        w,
        [interval(sc)],
        [],
        min_span_len,
        min(max_span_len, w),
    )
    span_e = _choose_span(
        h,
        [interval(sr), interval(lane[_N])],
        [interval(lane[_S])],
        min_span_len,
        min(max_span_len, h),
    )
    span_s = _choose_span(
        w,
        [interval(sc), interval(lane[_W]), interval(lane[_E])],
        [],
        min_span_len,
        min(max_span_len, w),
    )
    span_w = _choose_span(
        h,
        [interval(sr), interval(lane[_N])],
        [],
        min_span_len,
        min(max_span_len, h),
    )
    if span_n is None or span_e is None or span_s is None or span_w is None:
        return None

    spans = {_N: span_n, _E: span_e, _S: span_s, _W: span_w}

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    random.shuffle(legend_colors)
    side_color = {_N: legend_colors[0], _E: legend_colors[1], _S: legend_colors[2], _W: legend_colors[3]}
    bg = random.choice([c for c in BG_COLOR_POOL if c not in legend_colors])

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL
    occupied = np.zeros((h, w), dtype=bool)

    for side in (_N, _E, _S, _W):
        r0, r1, c0, c1 = _obstacle_rect(side, thickness[side], spans[side], h, w)
        region = occupied[r0 : r1 + 1, c0 : c1 + 1]
        if region.any():
            return None
        if not (r1 < sr - 1 or r0 > sr + sz or c1 < sc - 1 or c0 > sc + sz):
            return None
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        base[r0 : r1 + 1, c0 : c1 + 1] = side_color[side]

    vals, cnts = np.unique(base, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    side_turn = {_N: "L", _E: "R", _S: "L", _W: "L"}
    quotas = {
        side_color[_N]: random.randint(2, 4),
        side_color[_E]: random.randint(2, 4),
        side_color[_S]: 1,
        side_color[_W]: random.randint(2, 4),
    }
    rules = {side_color[side]: (side_turn[side], quotas[side_color[side]]) for side in (_N, _E, _S, _W)}

    try:
        combo_out, combo_events, combo_turn_used, pass_events, stop_events = _simulate_paths(
            base,
            rules,
            bg,
            use_quotas=True,
            max_turns=MAX_TURNS_PER_BEAM,
        )
        d13_out, d13_events, _, d13_passes, d13_stops = _simulate_paths(
            base,
            rules,
            bg,
            use_quotas=True,
            max_turns=None,
        )
        d6_out, d6_events, _, d6_passes, d6_stops = _simulate_paths(
            base,
            rules,
            bg,
            use_quotas=False,
            max_turns=MAX_TURNS_PER_BEAM,
        )
        src_out, src_events, _, src_passes, src_stops = _simulate_paths(
            base,
            rules,
            bg,
            use_quotas=False,
            max_turns=None,
        )
    except AssertionError:
        return None

    south_color = side_color[_S]
    north_events = combo_events[_N]
    if pass_events < 1 or stop_events < 1:
        return None
    if d13_stops != 0 or d6_passes != 0 or src_passes != 0 or src_stops != 0:
        return None
    if combo_turn_used[south_color] != 1:
        return None
    if len(north_events) < 3 or north_events[-1][4] != "stop" or north_events[-1][1] != south_color:
        return None
    if not any(event[4] == "pass" and event[1] == south_color for beam in combo_events for event in beam):
        return None
    if np.array_equal(combo_out, d13_out) or np.array_equal(combo_out, d6_out) or np.array_equal(combo_out, src_out):
        return None
    if len(d13_events[_N]) < 3 or d13_events[_N][-1][4] != "turn":
        return None
    if not any(event[4] == "pass" for beam in d13_events for event in beam):
        return None
    if any(event[4] == "pass" for beam in d6_events for event in beam):
        return None
    if any(event[4] == "stop" for beam in d13_events for event in beam):
        return None

    legend = _make_legend(legend_colors, rules)
    grid = np.vstack([legend, base])
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, combo_out):
        return None

    metrics = {
        "pass_events": pass_events,
        "stop_events": stop_events,
        "south_turns_used": combo_turn_used[south_color],
        "combo_minus_source": int(np.count_nonzero(combo_out != src_out)),
        "combo_minus_d13": int(np.count_nonzero(combo_out != d13_out)),
        "combo_minus_d6": int(np.count_nonzero(combo_out != d6_out)),
    }
    return grid, metrics


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            grid, _metrics = result
            return grid
    raise RuntimeError("failed to generate fused e87109e9 combo input")
