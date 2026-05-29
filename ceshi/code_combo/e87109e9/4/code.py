from __future__ import annotations

# Combo variant of e87109e9 (index 4)
# combo_spec: D6/1+D12/0
# Name: 限次转弯加障碍物消失

import random

import numpy as np

from arc_utils import bbox_coords, components_mask

# ---------------------------------------------------------------------------
# Design analysis for the fused variant
#
# a) Concrete behavior:
#    Keep the original visual vocabulary: a 6x24 legend band on top, then one
#    teal square and edge-touching obstacle rectangles below. The fused rule is
#    stateful. Example:
#      - N beam hits the top obstacle, turns east, then hits the right obstacle.
#      - On that second hit the right obstacle disappears and the N beam stops.
#      - Later the E beam is processed; because the right obstacle is already
#        gone, it travels straight through that location to the border.
#    Compared with the original task, two things change together:
#      1. the second obstacle hit is terminal;
#      2. the hit obstacle is removed for later beams.
#
# b) Complexity calibration:
#    The generator always creates two shared-obstacle chains, N->E and S->W.
#    This is strong enough to make ordering visible, but still uses only four
#    obstacles and four legend colors, so the scene stays legible.
#
# c) Hidden-assumption audit against the original solver:
#      - Original code assumed obstacle hits do not mutate the board.
#        No longer true: the simulation must run on one shared mutable grid.
#      - Original code assumed beams can be simulated independently.
#        No longer true: later beams depend on earlier erasures.
#      - Original code assumed "2 hits in generator" was only a data property.
#        No longer true: the solver must enforce a hard cap of 2 turns.
#      - Original code turned on every hit and continued forever until an edge.
#        No longer true: after the second hit the beam terminates immediately.
#
# d) Placement legality:
#    Obstacles are still solid single-color rectangles touching one edge. Their
#    spans are chosen to cover the required beam lanes, avoid corner overlap,
#    stay off the teal square, and keep at least one-cell clearance.
#
# e) Visual observability:
#    The rule is human-visible because the same obstacle is visibly present in
#    the input, absent in the output, and then crossed by a later straight beam.
#    Turn order is not encoded by color values; it is the fixed cardinal order
#    N, E, S, W, repeated across samples.
#
# f) Visual cleanliness:
#    Interactions happen on clean orthogonal lanes. Obstacles remain rectangles,
#    the legend remains unchanged, and beam overlap is only teal-on-teal.
#
# g) Non-degeneracy target:
#    The generator rejects any sample whose fused output matches either:
#      - a "turn-cap only" baseline (no obstacle erasure), or
#      - an "erasure only" baseline (no stop after the second hit).
#    This forces both fused dimensions to matter in every kept instance.
# ---------------------------------------------------------------------------

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

TEAL_SZ2_PROB = 0.75
SQUARE_EDGE_GAP = 5
OBS_THICKNESS_RANGE = (2, 4)
OBS_BOTH_GAP = 2
SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.45
TURN_CAP = 2
MAX_RETRIES = 400


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    legend = grid[:legend_h, :]
    assert legend.size
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2

    turn: dict[int, str] = {}
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        interior = legend[1 : legend_h - 1, a + 1 : b]
        colors = set(map(int, np.unique(interior)))
        colors.discard(bg)
        colors.discard(0)
        assert len(colors) == 1
        color = int(next(iter(colors)))
        x = int(np.where(interior == color)[1][0])
        turn[color] = "L" if x <= (interior.shape[1] - 1) // 2 else "R"
    assert len(turn) == 4
    return turn


def _find_teal_square(g: np.ndarray, teal: int = TEAL) -> tuple[int, int, int]:
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
    assert len(candidates) == 1
    return candidates[0]


def _rot(d: int, turn: str) -> int:
    return (d - 1) % 4 if turn == "L" else (d + 1) % 4


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
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
        cutoff = lengths[min(len(lengths) - 1, len(lengths) // 2)]
        pool = [(start, end) for length, start, end in choices if length <= cutoff]
    else:
        pool = [(start, end) for _, start, end in choices]
    return random.choice(pool)


def _thickness_pair(axis: int, sz: int) -> tuple[int, int] | None:
    lo, hi_cap = OBS_THICKNESS_RANGE
    room = axis - sz - OBS_BOTH_GAP
    if room < 2 * lo:
        return None
    a = random.randint(lo, min(hi_cap, room - lo))
    b = random.randint(lo, min(hi_cap, room - a))
    return a, b


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


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors, strict=True):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


def _simulate_variant(
    base: np.ndarray,
    turns: dict[int, str],
    bg: int,
    *,
    turn_cap: int | None,
    erase_on_hit: bool,
    stop_after_cap: bool,
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]]]:
    out = base.copy()
    H, W = out.shape
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    hits: list[list[tuple[int, int, int, int]]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < H and c + sz - 1 < W

    def paint(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = TEAL
        out[r : r + sz, c : c + sz] = blk

    def obstacle_at(r: int, c: int) -> int | None:
        blk = out[r : r + sz, c : c + sz]
        colors = set(map(int, np.unique(blk)))
        colors.discard(bg)
        colors.discard(TEAL)
        if not colors:
            return None
        assert len(colors) == 1
        return int(next(iter(colors)))

    for d0 in range(4):
        r, c, d = sr, sc, d0
        turns_used = 0
        beam_hits: list[tuple[int, int, int, int]] = []
        steps_left = H * W * 8
        while steps_left > 0:
            steps_left -= 1
            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            oc = obstacle_at(nr, nc)
            if oc is None:
                r, c = nr, nc
                paint(r, c)
                continue

            assert oc in turns
            beam_hits.append((d, oc, nr, nc))
            if erase_on_hit:
                out[out == oc] = bg
            turns_used += 1
            if stop_after_cap and turn_cap is not None and turns_used >= turn_cap:
                break
            d = _rot(d, turns[oc])
        else:
            raise AssertionError("beam simulation exceeded step budget")
        hits.append(beam_hits)

    return out, hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four teal beams from a square, but make hits stateful across beams.

    Input structure:
    - A 6-row legend band sits on top. Inside each legend cell, a colored bar on
      the left means counterclockwise; on the right means clockwise.
    - Below the legend is the main board: one solid teal square, one background
      color, and several edge-touching obstacle rectangles whose colors appear in
      the legend.

    Rule:
    - Remove the legend in the output.
    - Process the four beams in fixed order: north, east, south, west.
    - Each beam is a sliding teal copy of the source square that paints only over
      background cells.
    - When a beam meets an obstacle color, that whole obstacle disappears to the
      background color. The beam counts that as a turn event.
    - After its first hit, the beam turns according to the legend and continues.
      After its second hit, it stops immediately.
    - Later beams use the already-mutated board, so they may pass through a place
      where an earlier beam removed an obstacle.

    Key insight:
    The legend still tells you left-vs-right turns, but now the hidden structure
    is temporal: beams are not independent. You must use both the N/E/S/W beam
    order and the 2-hit turn cap to explain the output.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h))

    turns = _parse_legend(g, legend_h)
    base = g[legend_h:, :].copy()
    assert base.size

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    out, _ = _simulate_variant(
        base,
        turns,
        bg,
        turn_cap=TURN_CAP,
        erase_on_hit=True,
        stop_after_cap=True,
    )
    return out


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3

    ns = _thickness_pair(h, sz)
    ew = _thickness_pair(w, sz)
    if ns is None or ew is None:
        return None
    tN, tS = ns
    tW, tE = ew

    lane_N = tN + 1
    lane_S = h - tS - sz
    lane_W = tW + 1
    lane_E = w - tE - sz

    min_r = max(SQUARE_EDGE_GAP, lane_N + 2)
    max_r = min(h - sz - SQUARE_EDGE_GAP, lane_S - 2)
    min_c = max(SQUARE_EDGE_GAP, lane_W + 2)
    max_c = min(w - sz - SQUARE_EDGE_GAP, lane_E - 2)
    if min_r > max_r or min_c > max_c:
        return None
    sr = random.randint(min_r, max_r)
    sc = random.randint(min_c, max_c)

    req_cols = [(sc, sc + sz - 1)]
    req_rows_E = [(sr, sr + sz - 1), (lane_N, lane_N + sz - 1)]
    req_rows_W = [(sr, sr + sz - 1), (lane_S, lane_S + sz - 1)]

    spans: dict[int, tuple[int, int]] = {}
    spans[_N] = _choose_span(
        w,
        req_cols,
        [(0, tW - 1), (w - tE, w - 1)],
        SPAN_MIN_MULT * sz,
        min(SPAN_MAX_MULT * sz, w),
    )
    spans[_S] = _choose_span(
        w,
        req_cols,
        [(0, tW - 1), (w - tE, w - 1)],
        SPAN_MIN_MULT * sz,
        min(SPAN_MAX_MULT * sz, w),
    )
    spans[_E] = _choose_span(
        h,
        req_rows_E,
        [(0, tN - 1), (h - tS, h - 1)],
        SPAN_MIN_MULT * sz,
        min(SPAN_MAX_MULT * sz, h),
    )
    spans[_W] = _choose_span(
        h,
        req_rows_W,
        [(0, tN - 1), (h - tS, h - 1)],
        SPAN_MIN_MULT * sz,
        min(SPAN_MAX_MULT * sz, h),
    )
    if any(span is None for span in spans.values()):
        return None

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    side_color = {
        _N: legend_colors[0],
        _E: legend_colors[1],
        _S: legend_colors[2],
        _W: legend_colors[3],
    }
    bg = random.choice([c for c in BG_COLOR_POOL if c not in legend_colors])

    turns = {
        side_color[_N]: "R",
        side_color[_E]: random.choice(["L", "R"]),
        side_color[_S]: "R",
        side_color[_W]: random.choice(["L", "R"]),
    }

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL
    occupied = np.zeros((h, w), dtype=bool)
    thickness = {_N: tN, _E: tE, _S: tS, _W: tW}

    for side in (_N, _E, _S, _W):
        r0, r1, c0, c1 = _obstacle_rect(side, thickness[side], spans[side], h, w)
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
            return None
        if not (r1 < sr - 1 or r0 > sr + sz or c1 < sc - 1 or c0 > sc + sz):
            return None
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        base[r0 : r1 + 1, c0 : c1 + 1] = side_color[side]

    vals, cnts = np.unique(base, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        fused, hits = _simulate_variant(
            base,
            turns,
            bg,
            turn_cap=TURN_CAP,
            erase_on_hit=True,
            stop_after_cap=True,
        )
        cap_only, _ = _simulate_variant(
            base,
            turns,
            bg,
            turn_cap=TURN_CAP,
            erase_on_hit=False,
            stop_after_cap=True,
        )
        erase_only, _ = _simulate_variant(
            base,
            turns,
            bg,
            turn_cap=None,
            erase_on_hit=True,
            stop_after_cap=False,
        )
        original_like, _ = _simulate_variant(
            base,
            turns,
            bg,
            turn_cap=None,
            erase_on_hit=False,
            stop_after_cap=False,
        )
    except AssertionError:
        return None

    hit_colors = [[color for _d, color, _r, _c in beam] for beam in hits]
    expected_hits = [
        [side_color[_N], side_color[_E]],
        [],
        [side_color[_S], side_color[_W]],
        [],
    ]
    if hit_colors != expected_hits:
        return None
    if np.array_equal(fused, cap_only):
        return None
    if np.array_equal(fused, erase_only):
        return None
    if np.array_equal(fused, original_like):
        return None

    legend = _make_legend(legend_colors, turns)
    inp = np.vstack([legend, base])
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, fused):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e87109e9 combo input")
