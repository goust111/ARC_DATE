# Combo variant of e87109e9 (index 36)
# combo_spec: D2/1+D3/0+D13/1
# Name: 箭头图例双参数编码可变角度

"""
design_analysis:
  concrete_behavior:
    input_appearance:
      - keep the original 6x24 legend band and 24-column main grid
      - replace each legend bar with a bottom-aligned colored hook / arrow token inside a 4x4 black cell
      - the token has three visible parameters:
        - facing left vs right encodes quarter-turn handedness
        - a 2-pixel top arm means 90 degrees, a 3-pixel top arm means 180 degrees
        - the token height encodes how many turning collisions that color may spend
    trace_example:
      - suppose the north obstacle uses a tall 3-pixel right-facing hook and the south obstacle uses a short left-facing hook
      - in the original task a "right" entry always meant a clockwise quarter-turn
      - here the north beam instead reverses 180 degrees at its first hit, travels back down, and reaches the south obstacle
      - if the south color has quota 1 and is encountered again later, that later beam no longer turns there and passes straight through the south block to the boundary
      - the visible delta is therefore not just a different legend glyph: the output shows both a bounce-back corridor and a missing corner after quota exhaustion
  complexity_calibration:
    - keep exactly four legend cells and four edge obstacles so all legend colors are grounded in the main area
    - force one actually-used 180-degree obstacle and at least one used left-turn and one used right-turn obstacle, so direction and angle are both inferable from a few examples
    - force at least two repeated obstacle encounters in the uncapped simulation, then lower at least one quota, so the height parameter always affects the output
  hidden_assumptions_review:
    original_breakages:
      - legend parsing can no longer assume "one colored column whose side alone is meaningful"
      - hit handling can no longer assume "every collision turns" or "every turn is 90 degrees"
      - span planning can no longer assume every second-hit lane is the post-turn side lane, because 180-degree reversals reuse the incoming corridor
    repaired_assumptions:
      - solver parses each symbol as a hook with an exact top-row pattern plus a bottom-aligned stem
      - rotation logic stores handedness and angle separately; half-turns formally keep handedness even though both handed half-turns end in the reversed direction
      - generator computes the outgoing interval per action: 90-degree turns use the side lane, 180-degree turns reuse the source corridor
  placement_legality:
    - obstacles remain solid edge-touching rectangles with the original thickness and span bounds
    - the teal square still has a safety margin from every edge and obstacle
    - required/excluded span intervals are used to guarantee the planned first and second hits while blocking accidental third hits or mixed-color footprints
  visual_observability:
    - the three encoded parameters are all pixel-visible: arm side, arm length, stem height
    - accepted samples are filtered so the output differs both from an uncapped version and from an "all angles become 90 degrees" baseline
    - this prevents the fusion from collapsing back to either the D13-only or the original-turn behavior
  neatness:
    - the legend still looks like four clean, isolated cells rather than a dense codebook
    - the main area keeps the source task's sparse geometry: one teal square and one obstacle per edge
    - pass-through only happens inside edge-touching rectangles, so the resulting painted traces remain easy to read
  inference_support:
    - colors are still mapped only by direct legend-color matching; there is no hidden ordering or numeric use of color ids
    - generator validation requires the 180-degree rule and the quota rule to change outputs on the same input, making the fused behavior observable in examples

  solver_side_assumptions_guaranteed_by_generator:
    - color 5 is confined to the top legend band
    - each legend cell contains exactly one monochrome hook token with one of four legal top-row patterns
    - the main area contains exactly one solid teal square
    - each beam footprint meets at most one obstacle color at a time
    - no simulated beam revisits the same (row, col, direction) state
"""

import random
import sys
from pathlib import Path

import numpy as np

for _parent in Path(__file__).resolve().parents:
    _solvers = _parent / "solvers"
    if (_solvers / "arc_utils.py").exists():
        sys.path.append(str(_solvers))
        break

from arc_utils import bbox_coords, components_mask


_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, tuple[str, int, int]]:
    """
    Legend is a top region with a background color (typically 5) and 4 symbol cells.
    Each cell contains one bottom-aligned hook token:
      - left-facing hook  -> handedness 'L'
      - right-facing hook -> handedness 'R'
      - top arm length 2  -> 90-degree turn
      - top arm length 3  -> 180-degree turn
      - stem height 1..4  -> max number of turning collisions for that color
    """
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    rules: dict[int, tuple[str, int, int]] = {}
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        interior = legend[1 : legend_h - 1, a + 1 : b]
        assert interior.shape == (4, 4), "legend cell interior must be 4x4"
        colors = set(map(int, np.unique(interior)))
        colors.discard(bg)
        colors.discard(0)
        assert len(colors) == 1, "each legend cell must encode exactly one color"
        color = int(next(iter(colors)))

        ys, xs = np.where(interior == color)
        assert ys.size >= 2, "each hook token must have at least two colored pixels"
        min_y = int(ys.min())
        max_y = int(ys.max())
        assert max_y == interior.shape[0] - 1, "hook must be bottom-aligned"
        quota = interior.shape[0] - min_y

        top_cols = sorted(map(int, xs[ys == min_y]))
        if top_cols == [0, 1]:
            hand, angle, stem_x = "L", 90, 1
        elif top_cols == [0, 1, 2]:
            hand, angle, stem_x = "L", 180, 1
        elif top_cols == [2, 3]:
            hand, angle, stem_x = "R", 90, 2
        elif top_cols == [1, 2, 3]:
            hand, angle, stem_x = "R", 180, 2
        else:
            raise AssertionError("unknown hook pattern in legend cell")

        expected_rows = np.arange(min_y, interior.shape[0])
        stem_rows = sorted(map(int, ys[xs == stem_x]))
        assert stem_rows == list(expected_rows), "hook stem must be contiguous and bottom-aligned"
        for row in range(min_y + 1, interior.shape[0]):
            row_cols = sorted(map(int, xs[ys == row]))
            assert row_cols == [stem_x], "only the stem may continue below the top arm"

        assert color not in rules, "legend colors must be distinct"
        rules[color] = (hand, angle, quota)

    assert len(rules) == 4, "legend must encode four colors"
    return rules


def _find_teal_square(g: np.ndarray, teal: int = 8) -> tuple[int, int, int]:
    """Return (r0, c0, size) for the unique solid teal square component."""
    candidates: list[tuple[int, int, int]] = []
    for cells in components_mask(g == teal, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh != ww:
            continue
        size = hh
        if int(np.sum(g[r0 : r1 + 1, c0 : c1 + 1] == teal)) != size * size:
            continue
        candidates.append((int(r0), int(c0), int(size)))
    assert len(candidates) == 1, "main grid must contain exactly one solid teal square"
    return candidates[0]


def _ccw(d: int) -> int:
    return (d - 1) % 4


def _cw(d: int) -> int:
    return (d + 1) % 4


def _opp(d: int) -> int:
    return (d + 2) % 4


def _rot(d: int, hand: str, angle: int) -> int:
    if angle == 90:
        return _ccw(d) if hand == "L" else _cw(d)
    assert angle == 180, f"unsupported angle: {angle}"
    return _opp(d)


def _next_side(side: int, hand: str, angle: int) -> int:
    return _rot(side, hand, angle)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch four teal beams from a square, with each obstacle color carrying a hook-shaped three-parameter legend rule.

    Input structure:
    - Top strip: a grey legend split into four black 4x4 cells, each containing one colored hook token.
    - The hook's facing gives left-vs-right handedness, its top-arm size distinguishes 90-degree from 180-degree turns, and its height gives that color's total turning quota.
    - Below the legend is the main area with one solid teal square, a dominant background, and edge-touching colored obstacle rectangles.

    Rule:
    - Remove the legend from the output.
    - Fire one sz-by-sz teal beam in each cardinal direction from the source square.
    - A beam paints teal only onto background cells while sliding.
    - When a beam first tries to enter obstacle color C and C still has quota left, spend one collision from C and rotate according to that hook: short hooks mean 90-degree left/right turns, long hooks mean 180-degree reversals.
    - Once a color has exhausted its quota, later beams no longer turn there and instead pass straight through that obstacle until they leave the grid.

    Key insight: each legend token fuses three visible controls at once, so the solver must read symbol geometry rather than a single left/right bar: facing chooses handedness, arm size chooses turn angle, and height chooses how many times that turn rule may still fire.
    """
    g = np.asarray(grid, dtype=int)

    legend_bg = 5
    rows_with_legend = np.where(np.any(g == legend_bg, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    legend_rules = _parse_legend(g, legend_h)

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    teal = 8
    sr, sc, sz = _find_teal_square(base, teal=teal)

    out = base.copy()
    turn_used = {color: 0 for color in legend_rules}

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < out.shape[0] and c + sz - 1 < out.shape[1]

    def paint_block(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = teal
        out[r : r + sz, c : c + sz] = blk

    def obstacle_color_at(r: int, c: int) -> int | None:
        blk = out[r : r + sz, c : c + sz]
        cols = set(map(int, np.unique(blk)))
        cols.discard(bg)
        cols.discard(teal)
        if not cols:
            return None
        assert len(cols) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(cols)))

    for d0 in range(4):
        r, c, d = sr, sc, d0
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
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
            hand, angle, quota = legend_rules[next_obstacle]
            if turn_used[next_obstacle] < quota:
                turn_used[next_obstacle] += 1
                d = _rot(d, hand, angle)
            else:
                r, c = nr, nc
                paint_block(r, c)

    return out


# --------------- generator (o5x) ---------------

LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
LEGEND_BG = 5
TEAL = 8

GRID_H_RANGE = (19, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]

TEAL_SZ2_PROB = 0.75
SQUARE_EDGE_GAP = 5

OBS_THICKNESS_RANGE = (2, 4)
OBS_BOTH_GAP = 2

SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.4

MAX_RETRIES = 500


def _native_interval(side: int, sr: int, sc: int, sz: int) -> tuple[int, int]:
    return (sc, sc + sz - 1) if side in (_N, _S) else (sr, sr + sz - 1)


def _lane_interval(side: int, lane: dict[int, int], sz: int) -> tuple[int, int]:
    return (lane[side], lane[side] + sz - 1)


def _outgoing_interval(
    side: int,
    hand: str,
    angle: int,
    sr: int,
    sc: int,
    sz: int,
    lane: dict[int, int],
) -> tuple[int, int]:
    return _native_interval(side, sr, sc, sz) if angle == 180 else _lane_interval(side, lane, sz)


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a span that covers every required interval and avoids every excluded interval."""
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


def _simulate_paths(
    base: np.ndarray,
    rules: dict[int, tuple[str, int, int]],
    bg: int,
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int, str]]], dict[int, int], int]:
    """Simulate beams with handedness, variable turn angle, and per-color collision quotas."""
    vals, counts = np.unique(base, return_counts=True)
    assert int(vals[counts.argmax()]) == bg

    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    out = base.copy()
    h, w = out.shape
    beam_events: list[list[tuple[int, int, int, int, str]]] = []
    turn_used = {color: 0 for color in rules}
    pass_events = 0

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = TEAL
        out[r : r + sz, c : c + sz] = blk

    def obstacle_at(r: int, c: int) -> int | None:
        cols = set(map(int, np.unique(out[r : r + sz, c : c + sz])))
        cols.discard(bg)
        cols.discard(TEAL)
        if not cols:
            return None
        assert len(cols) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(cols)))

    for d0 in range(4):
        r, c, d = int(sr), int(sc), d0
        events: list[tuple[int, int, int, int, str]] = []
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
            assert state not in seen, "beam path must not cycle"
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
                assert current_obs == next_obs, "penetration cannot switch obstacle colors mid-crossing"
                r, c = nr, nc
                paint(r, c)
                continue

            assert next_obs in rules
            hand, angle, quota = rules[next_obs]
            if turn_used[next_obs] < quota:
                turn_used[next_obs] += 1
                events.append((d, next_obs, nr, nc, "turn"))
                d = _rot(d, hand, angle)
            else:
                events.append((d, next_obs, nr, nc, "pass"))
                pass_events += 1
                r, c = nr, nc
                paint(r, c)
        beam_events.append(events)

    return out, beam_events, turn_used, pass_events


def _make_legend(legend_colors: list[int], rules: dict[int, tuple[str, int, int]]) -> np.ndarray:
    """Build the 6x24 legend band from 4 colors and their (hand, angle, quota) rules."""
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        hand, angle, quota = rules[color]
        assert 1 <= quota <= 4, "quota must fit inside the legend cell"
        top_row = 5 - quota
        if hand == "L":
            cols = [0, 1] if angle == 90 else [0, 1, 2]
        else:
            cols = [2, 3] if angle == 90 else [1, 2, 3]
        for col in cols:
            legend[top_row, start + col] = color
        stem_col = 1 if hand == "L" else 2
        legend[top_row:5, start + stem_col] = color
    return legend


def _rand_thickness(lo: int, hi: int) -> int | None:
    return random.randint(lo, hi) if hi >= lo else None


def _thickness_pair(axis: int, sz: int) -> tuple[int, int] | None:
    """Compute obstacle thicknesses for both ends of an axis with both sides present."""
    lo, hi_cap = OBS_THICKNESS_RANGE
    room = axis - sz - OBS_BOTH_GAP
    if room < 2 * lo:
        return None
    a = random.randint(lo, min(hi_cap, room - lo))
    b = _rand_thickness(lo, min(hi_cap, room - a))
    return (a, b) if b is not None else None


def _obstacle_rect(
    side: int,
    thickness: int,
    span: tuple[int, int],
    h: int,
    w: int,
) -> tuple[int, int, int, int]:
    s0, s1 = span
    if side == _N:
        return 0, thickness - 1, s0, s1
    if side == _S:
        return h - thickness, h - 1, s0, s1
    if side == _W:
        return s0, s1, 0, thickness - 1
    return s0, s1, w - thickness, w - 1


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    present = {_N, _E, _S, _W}

    ns = _thickness_pair(h, sz)
    ew = _thickness_pair(w, sz)
    if ns is None or ew is None:
        return None
    thickness = {_N: ns[0], _S: ns[1], _W: ew[0], _E: ew[1]}

    min_r = max(SQUARE_EDGE_GAP, thickness[_N] + 1)
    max_r = min(h - sz - SQUARE_EDGE_GAP, h - thickness[_S] - sz - 1)
    min_c = max(SQUARE_EDGE_GAP, thickness[_W] + 1)
    max_c = min(w - sz - SQUARE_EDGE_GAP, w - thickness[_E] - sz - 1)
    if min_r > max_r or min_c > max_c:
        return None
    sr, sc = random.randint(min_r, max_r), random.randint(min_c, max_c)

    lane = {
        _N: thickness[_N] + 1,
        _S: h - thickness[_S] - sz,
        _W: thickness[_W] + 1,
        _E: w - thickness[_E] - sz,
    }

    u_side = random.randrange(4)
    stop_side = _opp(u_side)
    remaining = [s for s in (_N, _E, _S, _W) if s not in (u_side, stop_side)]
    chain_side = random.choice(remaining)
    other_side = next(s for s in remaining if s != chain_side)

    side_rule: dict[int, tuple[str, int]] = {}
    side_rule[u_side] = (random.choice(["L", "R"]), 180)
    side_rule[chain_side] = ("L" if _ccw(chain_side) == u_side else "R", 90)

    stop_hand = random.choice(["L", "R"])
    side_rule[stop_side] = (stop_hand, 90)
    side_rule[other_side] = ("R" if stop_hand == "L" else "L", 90)

    chains = {
        u_side: stop_side,
        chain_side: u_side,
    }

    required: dict[int, list[tuple[int, int]]] = {s: [] for s in present}
    excluded: dict[int, list[tuple[int, int]]] = {s: [] for s in present}
    for side in present:
        required[side].append(_native_interval(side, sr, sc, sz))

    for side in present:
        hand, angle = side_rule[side]
        nxt = _next_side(side, hand, angle)
        interval = _outgoing_interval(side, hand, angle, sr, sc, sz, lane)
        if chains.get(side) == nxt:
            required[nxt].append(interval)
            nxt_hand, nxt_angle = side_rule[nxt]
            after = _next_side(nxt, nxt_hand, nxt_angle)
            if after in present:
                followup = interval if nxt_angle == 180 else _lane_interval(nxt, lane, sz)
                excluded[after].append(followup)
        elif nxt in present:
            excluded[nxt].append(interval)

    spans: dict[int, tuple[int, int]] = {}
    for side in present:
        axis_len = w if side in (_N, _S) else h
        span = _choose_span(
            axis_len,
            required[side],
            excluded[side],
            SPAN_MIN_MULT * sz,
            min(SPAN_MAX_MULT * sz, axis_len),
        )
        if span is None:
            return None
        spans[side] = span

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    side_color = dict(zip((_N, _E, _S, _W), legend_colors))
    bg = random.choice([c for c in BG_COLOR_POOL if c not in legend_colors])

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL
    occupied = np.zeros((h, w), dtype=bool)
    for side in present:
        r0, r1, c0, c1 = _obstacle_rect(side, thickness[side], spans[side], h, w)
        region = occupied[r0 : r1 + 1, c0 : c1 + 1]
        if region.any():
            return None
        if not (r1 < sr - 1 or r0 > sr + sz or c1 < sc - 1 or c0 > sc + sz):
            return None
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        base[r0 : r1 + 1, c0 : c1 + 1] = side_color[side]

    vals, counts = np.unique(base, return_counts=True)
    if int(vals[counts.argmax()]) != bg:
        return None

    uncapped_rules = {
        side_color[side]: (*side_rule[side], 4)
        for side in present
    }
    try:
        uncapped_out, uncapped_events, uncapped_turns, _ = _simulate_paths(base, uncapped_rules, bg)
    except AssertionError:
        return None

    total_uncapped = sum(len(beam) for beam in uncapped_events)
    if total_uncapped != 6 or any(len(beam) > 2 for beam in uncapped_events):
        return None

    hit_colors = {color for beam in uncapped_events for _, color, _, _, kind in beam if kind == "turn"}
    if set(side_color.values()) != hit_colors:
        return None

    repeated_colors = [color for color, cnt in uncapped_turns.items() if cnt >= 2]
    if len(repeated_colors) < 2:
        return None

    used_small_hands = {
        side_rule[side][0]
        for side in present
        if side_rule[side][1] == 90
    }
    if used_small_hands != {"L", "R"}:
        return None

    saturated = set(random.sample(repeated_colors, random.randint(1, 2)))
    rules: dict[int, tuple[str, int, int]] = {}
    for side in present:
        color = side_color[side]
        hand, angle = side_rule[side]
        turns_here = uncapped_turns[color]
        if color in saturated:
            quota = 1
        else:
            quota = random.randint(turns_here, 4)
        rules[color] = (hand, angle, quota)

    try:
        expected, events, turn_used, pass_events = _simulate_paths(base, rules, bg)
    except AssertionError:
        return None
    if pass_events < 1:
        return None
    if np.array_equal(expected, uncapped_out):
        return None

    for color in saturated:
        if turn_used[color] != 1:
            return None

    angle_baseline = {
        color: (hand, 90, quota)
        for color, (hand, _angle, quota) in rules.items()
    }
    try:
        angle_out, _angle_events, _angle_turns, _angle_passes = _simulate_paths(base, angle_baseline, bg)
    except AssertionError:
        return None
    if np.array_equal(expected, angle_out):
        return None

    used_big_hits = {
        color
        for beam in events
        for _, color, _, _, kind in beam
        if kind == "turn" and rules[color][1] == 180
    }
    if not used_big_hits:
        return None

    legend = _make_legend(legend_colors, rules)
    inp = np.vstack([legend, base])
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e87109e9 combo variant input")
