# Variant of e87109e9
# Dimension: D14 — 多步处理与射线顺序 (multi_step)
# Change: 所有射线从源方块同时向4个方向发射，各自独立追踪路径直到离开网格，无射线间顺序依赖... → 链式反应传播

import random

import numpy as np

from arc_utils import bbox_coords, components_mask


# Design analysis for the D14 variant
# -----------------------------------
# variant_spec:
#   source_change:
#     original: one isolated teal square emits 4 independent beams
#     new: the input contains one solid teal square plus one detached teal starter strip
#     starter_strip:
#       shape: 1xsz or szx1, where sz is the source square size
#       placement: aligned with exactly one side of the square, with a 1-cell bg gap
#       role: visibly marks the single initial direction of the chain ray
#   behavior_trace:
#     original_flow:
#       - start 4 beams from the square
#       - each beam turns independently when the next sz x sz footprint meets an obstacle
#       - the final output is the union of 4 unrelated traces
#     new_flow:
#       - detect the single starter strip and launch only that one beam
#       - when the next footprint meets a colored obstacle, rotate in place using the legend
#       - keep following the same ray through every later turn until it exits the grid
#     concrete_example:
#       - if the starter strip is above the square, the beam first travels north
#       - on hitting a top obstacle whose legend bar is on the right, it turns east
#       - if it then hits a right obstacle with the same right-bar encoding, it turns south
#       - the visible output is one long bent corridor rather than a cross
#   complexity_calibration:
#     - the source needs an asymmetric cue, otherwise a human cannot know which of 4 directions to launch
#     - keeping the beam width equal to the square size preserves the original task's visual language
#     - the generator uses 3 or 4 obstacle hits so the chain effect is obvious and not reducible to a single bend
#   implicit_assumption_audit:
#     - original _find_teal_square assumed all teal cells belonged to one square; now solver must separate square vs starter strip
#     - original main loop iterated over 4 start directions; now that would be wrong because it would repaint the old cross output
#     - obstacle lookup may safely ignore pre-existing teal because the starter strip sits directly on the intended path
#     - cycle detection remains necessary because a single sequential ray could otherwise loop around the perimeter
#   placement_legality:
#     - the starter strip is always outside the square and outside all obstacles
#     - the square keeps at least one bg cell from every obstacle, and at least two cells from the start-side obstacle so the strip fits
#     - obstacle spans are chosen to force the intended next hit, and for 4-side chains the first obstacle explicitly misses the wraparound lane
#   observability:
#     - in input/output pairs the difference from the original task is directly visible: only one side of the square is pre-indicated and only one polyline is drawn
#     - turn direction is still encoded by left/right legend bars, so the original clue remains interpretable
#   cleanliness:
#     - teal is confined to the source square, one detached strip, and the final ray
#     - obstacle rectangles still hug the boundary, avoiding clutter in the interior


_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    """Read the top legend and map obstacle color -> turn direction."""
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    turn: dict[int, str] = {}
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        interior = legend[1 : legend_h - 1, a + 1 : b]
        colors = set(map(int, np.unique(interior)))
        colors.discard(bg)
        colors.discard(0)
        assert len(colors) == 1, "each legend cell must encode exactly one color"
        col = int(next(iter(colors)))
        xs = np.where(interior == col)[1]
        x = int(xs[0])
        turn[col] = "L" if x <= (interior.shape[1] - 1) // 2 else "R"
    assert len(turn) == 4, "legend must encode four turn colors"
    return turn


def _find_source(g: np.ndarray, teal: int = 8) -> tuple[int, int, int, int]:
    """
    Return (r0, c0, size, start_dir) for the teal square plus detached starter strip.
    The starter strip is separated by exactly one bg cell and points to the only launch direction.
    """
    comps = list(components_mask(g == teal, connectivity=4))
    assert len(comps) == 2, "main grid must contain one teal square and one teal starter strip"

    square_candidates: list[tuple[int, int, int]] = []
    for cells in comps:
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh != ww or hh < 2:
            continue
        if int(np.sum(g[r0 : r1 + 1, c0 : c1 + 1] == teal)) != hh * ww:
            continue
        square_candidates.append((int(r0), int(c0), int(hh)))
    assert len(square_candidates) == 1, "there must be exactly one solid teal source square"

    sr, sc, sz = square_candidates[0]
    for cells in comps:
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = int(r1 - r0 + 1)
        ww = int(c1 - c0 + 1)
        area = int(cells.shape[0])
        if (r0, c0, hh) == square_candidates[0]:
            continue
        assert area == sz, "starter strip must have the same length as the square side"
        if hh == 1 and ww == sz and int(r0) == sr - 2 and int(c0) == sc:
            return sr, sc, sz, _N
        if hh == 1 and ww == sz and int(r0) == sr + sz + 1 and int(c0) == sc:
            return sr, sc, sz, _S
        if hh == sz and ww == 1 and int(c0) == sc - 2 and int(r0) == sr:
            return sr, sc, sz, _W
        if hh == sz and ww == 1 and int(c0) == sc + sz + 1 and int(r0) == sr:
            return sr, sc, sz, _E
    raise AssertionError("starter strip must align with exactly one side of the source square")


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace one continuous teal ray from a square, following legend-controlled turns.

    Input structure:
    - Top band: a grey legend with four colored vertical bars; each bar sits on the left or right side of its cell.
    - Below: the main grid with a solid teal source square, one detached teal starter strip showing the launch side, and colored obstacle blocks.

    Rule:
    - Remove the legend from the output.
    - Launch a single sz-wide ray from the square in the direction indicated by the starter strip.
    - As the ray moves, it paints teal on background cells only.
    - When the next footprint would hit obstacle color C, read C in the legend and rotate the ray left or right, then continue tracing the same ray.
    - Stop only when that one continuous ray leaves the grid.

    Key insight: The detached teal strip selects the only starting direction, while the legend still decides each later turn, so the output is a chained polyline instead of four independent beams.
    """
    g = np.asarray(grid, dtype=int)

    legend_bg = 5
    rows_with_legend = np.where(np.any(g == legend_bg, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    legend_turn = _parse_legend(g, legend_h)

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    teal = 8
    sr, sc, sz, d = _find_source(base, teal=teal)

    out = base.copy()

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < out.shape[0] and c + sz - 1 < out.shape[1]

    def paint_block(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = teal
        out[r : r + sz, c : c + sz] = blk

    def obstacle_color_at_next(nr: int, nc: int) -> int | None:
        blk = out[nr : nr + sz, nc : nc + sz]
        cols = set(map(int, np.unique(blk)))
        cols.discard(bg)
        cols.discard(teal)
        if not cols:
            return None
        assert len(cols) == 1, "ray footprint must not hit mixed obstacle colors"
        return int(next(iter(cols)))

    r, c = sr, sc
    seen: set[tuple[int, int, int]] = set()
    while True:
        state = (r, c, d)
        assert state not in seen, "ray path must not cycle"
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

        assert oc in legend_turn, "obstacle color must be encoded by legend"
        d = _rot(d, legend_turn[oc])

    return out


# --------------- generator (single_chain) ---------------

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

# Teal source
TEAL_SZ2_PROB = 0.75
SQUARE_EDGE_GAP = 5
STARTER_GAP = 1

# Obstacles
OBS_COUNT_4_PROB = 0.6
OBS_THICKNESS_RANGE = (2, 4)
OBS_SINGLE_MARGIN = 6
OBS_BOTH_GAP = 2

# Span selection
SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.4

# Retry budget
MAX_RETRIES = 400


def _sequence(start_side: int, turn: str, count: int) -> list[int]:
    delta = -1 if turn == "L" else 1
    return [int((start_side + delta * i) % 4) for i in range(count)]


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a span covering all required intervals and none of the excluded ones."""
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


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


def _rand_thickness(lo: int, hi: int) -> int | None:
    return random.randint(lo, hi) if hi >= lo else None


def _thickness_pair(
    axis: int, sz: int, a_present: bool, b_present: bool
) -> tuple[int | None, int | None] | None:
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


def _obstacle_rect(
    side: int, thickness: int, span: tuple[int, int], h: int, w: int
) -> tuple[int, int, int, int]:
    s0, s1 = span
    if side == _N:
        return 0, thickness - 1, s0, s1
    if side == _S:
        return h - thickness, h - 1, s0, s1
    if side == _W:
        return s0, s1, 0, thickness - 1
    return s0, s1, w - thickness, w - 1


def _lane_after_hit(side: int, thickness: dict[int, int], h: int, w: int, sz: int) -> int:
    if side == _N:
        return thickness[_N]
    if side == _S:
        return h - thickness[_S] - sz
    if side == _W:
        return thickness[_W]
    return w - thickness[_E] - sz


def _starter_cells(sr: int, sc: int, sz: int, start_dir: int) -> list[tuple[int, int]]:
    if start_dir == _N:
        return [(sr - 2, c) for c in range(sc, sc + sz)]
    if start_dir == _S:
        return [(sr + sz + 1, c) for c in range(sc, sc + sz)]
    if start_dir == _W:
        return [(r, sc - 2) for r in range(sr, sr + sz)]
    return [(r, sc + sz + 1) for r in range(sr, sr + sz)]


def _simulate_chain(
    base: np.ndarray, turns: dict[int, str], bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], int]:
    """Simulate the single chained ray on the cropped main grid."""
    sr, sc, sz, start_dir = _find_source(base, teal=TEAL)
    out = base.copy()
    h, w = out.shape
    hits: list[tuple[int, int, int, int]] = []

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
        assert len(cols) == 1
        return int(next(iter(cols)))

    r, c, d = int(sr), int(sc), int(start_dir)
    seen: set[tuple[int, int, int]] = set()
    while True:
        state = (r, c, d)
        assert state not in seen
        seen.add(state)
        dr, dc = _DIRS[d]
        nr, nc = r + dr, c + dc
        if not in_bounds(nr, nc):
            break
        obs = obstacle_at(nr, nc)
        if obs is None:
            r, c = nr, nc
            paint(r, c)
            continue
        assert obs in turns
        hits.append((d, obs, nr, nc))
        d = _rot(d, turns[obs])

    return out, hits, start_dir


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    obstacle_count = 4 if random.random() < OBS_COUNT_4_PROB else 3
    turn_dir = random.choice(["L", "R"])
    start_side = random.choice([_N, _E, _S, _W])
    sequence = _sequence(start_side, turn_dir, obstacle_count)
    present = set(sequence)

    ns = _thickness_pair(h, sz, _N in present, _S in present)
    ew = _thickness_pair(w, sz, _W in present, _E in present)
    if ns is None or ew is None:
        return None
    thickness = {s: t for s, t in zip((_N, _S, _W, _E), (*ns, *ew)) if t is not None}

    min_r, max_r = SQUARE_EDGE_GAP, h - sz - SQUARE_EDGE_GAP
    min_c, max_c = SQUARE_EDGE_GAP, w - sz - SQUARE_EDGE_GAP
    if _N in present:
        min_r = max(min_r, thickness[_N] + 1)
    if _S in present:
        max_r = min(max_r, h - thickness[_S] - sz - 1)
    if _W in present:
        min_c = max(min_c, thickness[_W] + 1)
    if _E in present:
        max_c = min(max_c, w - thickness[_E] - sz - 1)
    if start_side == _N:
        min_r = max(min_r, thickness[_N] + 2 + STARTER_GAP)
    if start_side == _S:
        max_r = min(max_r, h - thickness[_S] - sz - 2 - STARTER_GAP)
    if start_side == _W:
        min_c = max(min_c, thickness[_W] + 2 + STARTER_GAP)
    if start_side == _E:
        max_c = min(max_c, w - thickness[_E] - sz - 2 - STARTER_GAP)
    if min_r > max_r or min_c > max_c:
        return None

    sr = random.randint(min_r, max_r)
    sc = random.randint(min_c, max_c)

    lane = {side: _lane_after_hit(side, thickness, h, w, sz) for side in present}
    required: dict[int, list[tuple[int, int]]] = {side: [] for side in present}
    excluded: dict[int, list[tuple[int, int]]] = {side: [] for side in present}

    for i, side in enumerate(sequence):
        if i == 0:
            base_coord = sc if side in (_N, _S) else sr
            required[side].append((base_coord, base_coord + sz - 1))
        else:
            prev = sequence[i - 1]
            required[side].append((lane[prev], lane[prev] + sz - 1))

    if obstacle_count == 4:
        first = sequence[0]
        last = sequence[-1]
        excluded[first].append((lane[last], lane[last] + sz - 1))

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
    obstacle_colors = random.sample(legend_colors, obstacle_count)
    side_color = dict(zip(sequence, obstacle_colors))
    bg = random.choice([c for c in BG_COLOR_POOL if c not in legend_colors])

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL
    for r, c in _starter_cells(sr, sc, sz, start_side):
        base[r, c] = TEAL

    for side in present:
        r0, r1, c0, c1 = _obstacle_rect(side, thickness[side], spans[side], h, w)
        if np.any(base[r0 : r1 + 1, c0 : c1 + 1] != bg):
            return None
        base[r0 : r1 + 1, c0 : c1 + 1] = side_color[side]

    vals, cnts = np.unique(base, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    turns = {side_color[side]: turn_dir for side in present}
    for color in legend_colors:
        turns.setdefault(color, random.choice(["L", "R"]))

    try:
        expected, hits, parsed_start = _simulate_chain(base, turns, bg)
    except AssertionError:
        return None

    if parsed_start != start_side:
        return None
    if len(hits) != obstacle_count:
        return None
    if [d for d, _, _, _ in hits] != sequence:
        return None
    if len({color for _, color, _, _ in hits}) != obstacle_count:
        return None
    if set(side_color.values()) != {color for _, color, _, _ in hits}:
        return None

    legend = _make_legend(legend_colors, turns)
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
    raise RuntimeError("failed to generate e87109e9 D14 input")
