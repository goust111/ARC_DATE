# Combo variant of e87109e9 (index 31)
# combo_spec: D2/1+D3/1
# Name: 条长编码角度加可变角度转弯

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


# Design analysis for the D2/1 + D3/1 combo variant
# a) Concrete behavior and trace
#    The board keeps the original split layout: a 6x24 grey legend on top and the
#    cropped playfield below. The change is inside each 4x4 legend cell:
#    - horizontal slot still matters: left-slot bar means "turn left / CCW", right-slot
#      bar means "turn right / CW";
#    - vertical length now also matters: a short 2-cell bar means 90 degrees, while a
#      long 4-cell bar means 180 degrees.
#    Example trace: suppose the north obstacle is red and its legend bar is short-right.
#    Original task: a right bar always meant a 90-degree clockwise bend, so the north
#    beam turns east. In this combo, that remains true because the bar is short. Now
#    suppose the south obstacle is blue and its legend bar is long-left. Original task
#    would have turned the south beam east because only left/right existed. The combo
#    instead reads both attributes, sees "left" + "long", and performs a 180-degree
#    reversal: the south beam bounces back north along the same corridor. The visible
#    difference is therefore not just new legend art; it changes whether a collision
#    creates a sideways branch or a fold-back path.
# b) Complexity calibration
#    A naive fusion would be too weak if long bars were rare or if all hit short bars
#    used the same side, because a solver could ignore one dimension and still succeed.
#    To keep the rule legible and necessary, generation always uses all four edge
#    obstacles and forces three visibly distinct hit categories in every sample:
#    - one actually-hit long bar, which creates the new 180-degree behavior;
#    - at least one actually-hit short-left bar;
#    - at least one actually-hit short-right bar.
#    This keeps the legend binary in each dimension and repeats the categories enough
#    for a human to infer them from a few examples.
# c) Hidden-assumption audit
#    The original solver assumed:
#    - every legend bar had full height, so x-position alone was sufficient;
#    - every non-left bar meant a 90-degree clockwise turn.
#    Both assumptions break here. The parser must now recover one color, one x-position,
#    and one bar length from each legend cell. The movement step can no longer map
#    directly from side to {CCW,CW}; it must map to {(CCW,90), (CW,90), (CCW,180),
#    (CW,180)} and then reduce 180-degree cases to an about-face. On the generator side,
#    once 180-degree turns exist, a beam can revisit the source corridor, so span
#    constraints must explicitly forbid unintended third hits and cycles.
# d) Placement legality
#    Obstacles remain solid edge-touching rectangles with the original thickness/span
#    ranges and at least one-cell clearance from the teal square. The generator is
#    still path-first: it picks the square, edge obstacles, and desired hit structure,
#    translates those intended paths into required/excluded intervals on obstacle spans,
#    and rejects any placement where rectangles overlap, touch the square, or make a
#    beam collide with a mixed-color footprint.
# e) Visual observability
#    Humans can read the combo because both encoded dimensions are directly visible in
#    the legend. Short and long bars differ by a 2-vs-4 height contrast; left and right
#    differ by the existing slot contrast. To avoid samples where one dimension is only
#    decorative, the generator rejects boards whose output matches a position-only
#    interpretation (all bars treated as 90-degree) or whose output matches a
#    position-collapsed interpretation (all short bars forced to the same side).
# f) Visual cleanliness
#    The legend keeps one monochrome bar per cell inside the familiar black 4x4 window,
#    so the added information does not create symbol clutter. The main board remains
#    sparse: one teal square, four solid edge bars, and up to one second collision on
#    one beam. Constraints cap the routing at five total hits and at most two hits per
#    beam, which keeps the painted output readable.
# g) Change inferability
#    Colors themselves are still just lookup keys; no numeric color property is used.
#    Each obstacle color appears in the legend exactly once and on the board exactly once.
#    The fusion signal comes from structured visual cues attached to that color: x-slot
#    and bar height. Because all four legend colors are used every time, solvers cannot
#    ignore the legend or infer behavior from obstacle order alone.
#
# New solver-side assumptions enforced by generation:
# - grey 5 is confined to the top legend band;
# - the main grid contains exactly one solid teal square component;
# - each legend cell contains exactly one vertical monochrome bar with height 2 or 4;
# - every obstacle footprint seen by a beam is monochrome;
# - every generated sample has one used long action and both used short directions;
# - no beam revisits the same (top-left position, direction) state.


Action = tuple[str, int]  # (hand, quarter_turns) where hand in {"L","R"} and turns in {1,2}

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)

LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
LEGEND_BG = 5
TEAL = 8

GRID_H_RANGE = (19, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]
LEGEND_SHORT_ROWS = slice(3, 5)  # inside the full legend, bottom-aligned 2-cell bar

TEAL_SZ2_PROB = 0.75
SQUARE_EDGE_GAP = 6

OBS_THICKNESS_RANGE = (2, 4)
OBS_SINGLE_MARGIN = 6
OBS_BOTH_GAP = 2
SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.4

OBS_COUNT = 4
TARGET_HITS = 5
MAX_RETRIES = 600


def _find_teal_square(g: np.ndarray, teal: int = TEAL) -> tuple[int, int, int]:
    """Return (r0, c0, size) for the unique solid teal square component."""
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


def _rot_dir(d: int, hand: str, turns: int) -> int:
    if turns == 2:
        return (d + 2) % 4
    assert turns == 1, f"unsupported turn count: {turns}"
    return (d - 1) % 4 if hand == "L" else (d + 1) % 4


def _next_side(side: int, action: Action) -> int:
    hand, turns = action
    return _rot_dir(side, hand, turns)


def _native_interval(side: int, sr: int, sc: int, sz: int) -> tuple[int, int]:
    return (sc, sc + sz - 1) if side in (_N, _S) else (sr, sr + sz - 1)


def _lane_interval(side: int, lane: dict[int, int], sz: int) -> tuple[int, int]:
    return lane[side], lane[side] + sz - 1


def _turn_target(
    side: int,
    action: Action,
    incoming_interval: tuple[int, int],
    lane: dict[int, int],
    sz: int,
) -> tuple[int, tuple[int, int]]:
    """Return (next_side, interval_on_next_side) after a hit at *side*."""
    hand, turns = action
    nxt = _rot_dir(side, hand, turns)
    interval = _lane_interval(side, lane, sz) if turns == 1 else incoming_interval
    return nxt, interval


def _add_followup_exclusions(
    target: int,
    target_action: Action,
    present: set[int],
    lane: dict[int, int],
    sz: int,
    incoming_interval: tuple[int, int],
    excluded: dict[int, list[tuple[int, int]]],
) -> None:
    """Prevent a chained second collision from immediately causing a third one."""
    after, after_interval = _turn_target(target, target_action, incoming_interval, lane, sz)
    if after in present:
        excluded[after].append(after_interval)


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, Action]:
    """
    Legend is a top region with a background color (typically 5) and 4 cells.
    In each cell, a vertical bar encodes:
      - horizontal slot: left -> hand='L', right -> hand='R'
      - bar height: short (2) -> 90 degrees, long (4) -> 180 degrees
    """
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    actions: dict[int, Action] = {}
    full_h = legend_h - 2
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
        assert xs.size in (full_h // 2, full_h), "bar height must encode short or long"
        assert len(set(map(int, xs))) == 1, "legend bar must be vertical"
        x = int(xs[0])
        hand = "L" if x <= (interior.shape[1] - 1) // 2 else "R"
        turns = 1 if xs.size == full_h // 2 else 2
        actions[color] = (hand, turns)

    assert len(actions) == 4, "legend must encode four turn colors"
    return actions


def _simulate_paths(
    base: np.ndarray,
    actions: dict[int, Action],
    bg: int,
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]]]:
    """Simulate 4 beams, returning the painted grid and per-beam hit records."""
    vals, counts = np.unique(base, return_counts=True)
    assert int(vals[counts.argmax()]) == bg

    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    out = base.copy()
    h, w = out.shape
    hits: list[list[tuple[int, int, int, int]]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint_block(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = TEAL
        out[r : r + sz, c : c + sz] = blk

    def obstacle_color_at_next(r: int, c: int) -> int | None:
        blk = out[r : r + sz, c : c + sz]
        colors = set(map(int, np.unique(blk)))
        colors.discard(bg)
        colors.discard(TEAL)
        if not colors:
            return None
        assert len(colors) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(colors)))

    for d0 in range(4):
        r, c, d = sr, sc, d0
        beam_hits: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            oc = obstacle_color_at_next(nr, nc)
            if oc is None:
                r, c = nr, nc
                paint_block(r, c)
                continue

            assert oc in actions, "obstacle color must be encoded by legend"
            beam_hits.append((d, oc, nr, nc))
            hand, turns = actions[oc]
            d = _rot_dir(d, hand, turns)
            assert len(beam_hits) <= 2, "generator only accepts up to two hits per beam"

        hits.append(beam_hits)

    return out, hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four teal square-beams whose obstacle responses depend on both legend side and legend length.

    Input structure:
    - Top band: a grey legend split into four black 4x4 windows, each containing one colored vertical bar.
    - Inside each legend window, left vs right slot encodes turn handedness, while short vs long bar height encodes turn angle.
    - Below the legend: a main grid with one solid teal square, a dominant background color, and colored edge obstacles whose colors appear in the legend.

    Rule:
    - Remove the legend from the output.
    - Launch one sz-by-sz teal beam from the square in each cardinal direction.
    - A beam paints teal only onto background cells.
    - When the next beam footprint meets obstacle color C, read C's legend bar: left means counterclockwise, right means clockwise; short means turn 90 degrees, long means reverse 180 degrees. Continue from the same position in that new direction.
    - Stop a beam when its next sz-by-sz footprint would leave the grid.

    Key insight:
    - The legend is now two-dimensional code: horizontal placement still chooses left/right turning sense, but bar height independently chooses whether that turn is a quarter-turn or a full reversal, so solving requires decoding both attributes for each obstacle color.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    actions = _parse_legend(g, legend_h)

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"
    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])

    out, _hits = _simulate_paths(base, actions, bg)
    return out


def _rand_thickness(lo: int, hi: int) -> int | None:
    return random.randint(lo, hi) if hi >= lo else None


def _thickness_pair(axis: int, sz: int, a_present: bool, b_present: bool) -> tuple[int | None, int | None] | None:
    lo, hi_cap = OBS_THICKNESS_RANGE
    if a_present and b_present:
        room = axis - sz - OBS_BOTH_GAP
        if room < 2 * lo:
            return None
        a = random.randint(lo, min(hi_cap, room - lo))
        b = _rand_thickness(lo, min(hi_cap, room - a))
        return (a, b) if b is not None else None
    if a_present:
        t = _rand_thickness(lo, min(hi_cap, axis - sz - OBS_SINGLE_MARGIN))
        return (t, None) if t is not None else None
    if b_present:
        t = _rand_thickness(lo, min(hi_cap, axis - sz - OBS_SINGLE_MARGIN))
        return (None, t) if t is not None else None
    return None, None


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a span that covers all required intervals and avoids all excluded ones."""
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


def _make_legend(legend_colors: list[int], actions: dict[int, Action]) -> np.ndarray:
    """Build the 6x24 legend band from 4 colors and their (hand, angle) assignments."""
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        hand, turns = actions[color]
        bar_col = start if hand == "L" else start + 3
        if turns == 1:
            legend[LEGEND_SHORT_ROWS, bar_col] = color
        else:
            legend[1:5, bar_col] = color
    return legend


def _all_short(actions: dict[int, Action]) -> dict[int, Action]:
    return {color: (hand, 1) for color, (hand, _turns) in actions.items()}


def _force_short_hand(actions: dict[int, Action], hand: str) -> dict[int, Action]:
    forced: dict[int, Action] = {}
    for color, (old_hand, turns) in actions.items():
        forced[color] = (hand if turns == 1 else old_hand, turns)
    return forced


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    present = {_N, _E, _S, _W}

    ns = _thickness_pair(h, sz, True, True)
    ew = _thickness_pair(w, sz, True, True)
    if ns is None or ew is None:
        return None
    thickness = {s: t for s, t in zip((_N, _S, _W, _E), (*ns, *ew)) if t is not None}

    min_r, max_r = SQUARE_EDGE_GAP, h - sz - SQUARE_EDGE_GAP
    min_c, max_c = SQUARE_EDGE_GAP, w - sz - SQUARE_EDGE_GAP
    min_r = max(min_r, thickness[_N] + 1)
    max_r = min(max_r, h - thickness[_S] - sz - 1)
    min_c = max(min_c, thickness[_W] + 1)
    max_c = min(max_c, w - thickness[_E] - sz - 1)
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
    stop_side = (u_side + 2) % 4
    laterals = [side for side in (_N, _E, _S, _W) if side not in (u_side, stop_side)]
    random.shuffle(laterals)

    side_action: dict[int, Action] = {u_side: (random.choice(["L", "R"]), 2)}
    side_action[laterals[0]] = ("L", 1)
    side_action[laterals[1]] = ("R", 1)
    side_action[stop_side] = (random.choice(["L", "R"]), 1)

    required: dict[int, list[tuple[int, int]]] = {s: [] for s in present}
    excluded: dict[int, list[tuple[int, int]]] = {s: [] for s in present}
    for side in present:
        required[side].append(_native_interval(side, sr, sc, sz))

    for side in present:
        native = _native_interval(side, sr, sc, sz)
        nxt, interval = _turn_target(side, side_action[side], native, lane, sz)
        if side == u_side:
            assert nxt == stop_side, "the long source must target the opposite side"
            required[nxt].append(interval)
            _add_followup_exclusions(nxt, side_action[nxt], present, lane, sz, interval, excluded)
        else:
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
    obstacle_colors = random.sample(legend_colors, OBS_COUNT)
    side_color = dict(zip(sorted(present), obstacle_colors))
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

    actions = {side_color[s]: side_action[s] for s in present}

    try:
        expected, hits = _simulate_paths(base, actions, bg)
    except AssertionError:
        return None

    hit_counts = [len(beam) for beam in hits]
    if sum(hit_counts) != TARGET_HITS or max(hit_counts) > 2 or hit_counts.count(2) != 1:
        return None

    hit_colors = {color for beam in hits for _, color, _, _ in beam}
    if set(side_color.values()) != hit_colors:
        return None

    u_color = side_color[u_side]
    if len(hits[u_side]) != 2 or hits[u_side][0][1] != u_color:
        return None

    short_left_hit = any(side_action[s] == ("L", 1) for s in present)
    short_right_hit = any(side_action[s] == ("R", 1) for s in present)
    if not (short_left_hit and short_right_hit):
        return None

    try:
        pos_only_out, _ = _simulate_paths(base, _all_short(actions), bg)
        all_left_out, _ = _simulate_paths(base, _force_short_hand(actions, "L"), bg)
        all_right_out, _ = _simulate_paths(base, _force_short_hand(actions, "R"), bg)
    except AssertionError:
        return None

    delta_len = int(np.sum(expected != pos_only_out))
    delta_left = int(np.sum(expected != all_left_out))
    delta_right = int(np.sum(expected != all_right_out))
    if delta_len < 2 * sz or delta_left < sz or delta_right < sz:
        return None

    legend = _make_legend(legend_colors, actions)
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
    raise RuntimeError("failed to generate e87109e9 combo input")
