# Variant of e87109e9
# Dimension: D14 — 多步处理与射线顺序 (multi_step)
# Change: 所有射线从源方块同时向4个方向发射，各自独立追踪路径直到离开网格，无射线间顺序依赖... → 按方向顺序依次发射

import random

import numpy as np

from arc_utils import bbox_coords, components_mask

# Design analysis
# ----------------
# spec_variant:
#   preserved_visual_language:
#     legend: keep the original 6x24 grey legend strip with four 4-column cells and left/right bars
#     main_grid: keep one solid teal source square and 3 edge-touching solid obstacle rectangles
#     colors: keep obstacle colors as a subset of the 4 legend colors; keep one legend color possibly unused
#   core_change:
#     original_behavior:
#       - each of the four sz x sz beams is traced independently
#       - previously painted teal is ignored during obstacle checks, so beam order is irrelevant
#     new_behavior:
#       - beams are emitted in the visible order up -> right -> down -> left
#       - after one beam finishes, every newly painted trail cell (excluding the source square) becomes a new teal obstacle
#       - because these new obstacles have no legend entry, the variant uses one fixed convention:
#         meeting an earlier teal trail causes a clockwise turn
#       - colored obstacles still cause left/right turns exactly as before
#   concrete_small_example:
#     - suppose the up beam hits a north obstacle, turns right at an east obstacle, then travels downward
#     - under the old logic the later right beam would keep moving east and eventually reach the east obstacle
#     - under the new logic that same right beam is emitted after the up beam, so it first meets the earlier
#       downward teal trail, turns clockwise around it, and reaches a different continuation
#     - the visual difference is a bent later arm even though the input is unchanged
#   complexity_calibration:
#     - if generation were left fully random, most samples would still match the old task because beam paths would
#       rarely cross after turning
#     - to make the sequential dependency visible, the generator uses a small family of "hook" templates:
#         * up -> right -> down
#         * up -> left -> down
#         * right -> down -> left
#       These produce one early hooked trail that crosses one or two later straight lanes, so order-driven
#       trail turns are common and humans can infer the direction order from only a few examples
#     - a pure "stop on earlier trail" rule was rejected because many blocked later beams merely retrace cells
#       that are already teal, which collapses back to the original output despite different internal execution
#   main_line_assumption_audit:
#     - legend parsing still assumes color 5 is confined to the top legend band; generator preserves that
#     - teal obstacle checks in the old solver discarded color 8 unconditionally; that is no longer safe because
#       earlier teal trails now act as later obstacles. The new solver therefore tracks earlier trails with a boolean
#       mask instead of inferring from colors in the painted output
#     - the source square must not count as a blocker, otherwise every beam would fail on its first step because
#       the sliding sz x sz footprint overlaps the source. The generator still produces exactly one solid teal square
#     - the current beam must not react to itself; only completed earlier beams are added to the earlier-trail mask
#   placement_legality:
#     - all obstacles are edge-touching rectangles with thickness 2-4 and spans chosen to cover the required beam lanes
#     - spans are also clamped to avoid corner overlap between adjacent edge obstacles
#     - the source square is placed so the hooked early beam has room to reach all three planned obstacles and so the
#       later beams have at least one visible segment before their first trail-induced turn
#   observability_and_tidiness:
#     - no ordering depends on color ids; only geometric direction order matters
#     - each sample contains at least one later beam that turns on an earlier trail, and usually two
#     - obstacles stay on the borders while trails occupy the interior, so the output remains visually separable

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    """Decode obstacle-turn direction from the legend strip."""
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, counts = np.unique(legend, return_counts=True)
    bg = int(vals[counts.argmax()])

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
        color = int(next(iter(colors)))
        xs = np.where(interior == color)[1]
        x = int(xs[0])
        turn[color] = "L" if x <= (interior.shape[1] - 1) // 2 else "R"
    assert len(turn) == 4, "legend must encode four turn colors"
    return turn


def _find_teal_square(g: np.ndarray, teal: int = 8) -> tuple[int, int, int]:
    """Return (r0, c0, size) for the unique solid teal square."""
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


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def _simulate_sequential(
    base: np.ndarray, turns: dict[int, str], bg: int
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Emit beams in fixed order, letting later beams turn clockwise on earlier trails."""
    out = base.copy()
    sr, sc, sz = _find_teal_square(base, teal=8)
    h, w = out.shape

    source_mask = np.zeros_like(base, dtype=bool)
    source_mask[sr : sr + sz, sc : sc + sz] = True
    earlier_trail_mask = np.zeros_like(base, dtype=bool)
    beam_stats: list[dict[str, object]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def obstacle_color_at_next(nr: int, nc: int) -> int | None:
        blk = base[nr : nr + sz, nc : nc + sz]
        cols = set(map(int, np.unique(blk)))
        cols.discard(bg)
        cols.discard(8)
        if not cols:
            return None
        assert len(cols) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(cols)))

    for d0 in range(4):
        r, c, d = sr, sc, d0
        seen: set[tuple[int, int, int]] = set()
        beam_mask = np.zeros_like(base, dtype=bool)
        beam_hits: list[tuple[int, int, int, int]] = []
        trail_turns = 0

        while True:
            state = (r, c, d)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            if earlier_trail_mask[nr : nr + sz, nc : nc + sz].any():
                d = _rot_cw(d)
                trail_turns += 1
                continue

            oc = obstacle_color_at_next(nr, nc)
            if oc is None:
                r, c = nr, nc
                blk = out[r : r + sz, c : c + sz]
                blk[blk == bg] = 8
                out[r : r + sz, c : c + sz] = blk
                beam_mask[r : r + sz, c : c + sz] = True
                beam_mask[source_mask] = False
                continue

            assert oc in turns, "obstacle color must be encoded by legend"
            beam_hits.append((d, oc, nr, nc))
            d = _rot(d, turns[oc])

        earlier_trail_mask |= beam_mask
        beam_stats.append(
            {
                "dir": d0,
                "trail_turns": trail_turns,
                "hits": beam_hits,
                "trail_cells": int(beam_mask.sum()),
            }
        )

    return out, beam_stats


def _simulate_independent(base: np.ndarray, turns: dict[int, str], bg: int) -> np.ndarray:
    """Original behavior: each beam ignores other beams' trails."""
    out = base.copy()
    sr, sc, sz = _find_teal_square(base, teal=8)
    h, w = out.shape

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def obstacle_color_at_next(nr: int, nc: int) -> int | None:
        blk = base[nr : nr + sz, nc : nc + sz]
        cols = set(map(int, np.unique(blk)))
        cols.discard(bg)
        cols.discard(8)
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

            oc = obstacle_color_at_next(nr, nc)
            if oc is None:
                r, c = nr, nc
                blk = out[r : r + sz, c : c + sz]
                blk[blk == bg] = 8
                out[r : r + sz, c : c + sz] = blk
                continue

            assert oc in turns, "obstacle color must be encoded by legend"
            d = _rot(d, turns[oc])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint teal beams in a fixed emission order so earlier trails affect later ones.

    Input structure:
    - Top strip: a grey legend band with four colored vertical bars; each bar sits on the left or right side of its cell.
    - Below it: one uniform-background playfield containing a solid teal source square and several solid obstacle rectangles.

    Rule:
    - Remove the legend strip in the output.
    - Emit four sz x sz beams from the teal square in the order up, then right, then down, then left.
    - A beam moves one cell at a time, painting teal onto background cells; when its next footprint meets a colored obstacle, it turns 90 degrees according to the legend (left bar = counterclockwise, right bar = clockwise).
    - After a beam finishes, its newly painted trail becomes a new teal obstacle for the remaining beams; when a later beam meets such a trail, it turns clockwise.
    - A beam stops only when its next step would leave the grid.

    Key insight: The legend still explains how obstacle colors turn a beam, but the decisive new cue is temporal: earlier teal trails are re-used as turn-causing obstacles, so identical geometry yields different outputs depending on the fixed firing order.
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

    vals, counts = np.unique(base, return_counts=True)
    bg = int(vals[counts.argmax()])

    out, _beam_stats = _simulate_sequential(base, legend_turn, bg)
    return out


# --------------- generator ---------------

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
SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.45

MAX_RETRIES = 300

_TEMPLATES = [
    {"name": "up_clockwise", "sides": (_N, _E, _S), "turn": "R"},
    {"name": "up_counterclockwise", "sides": (_N, _W, _S), "turn": "L"},
    {"name": "right_clockwise", "sides": (_E, _S, _W), "turn": "R"},
]


def _lane_for_side(side: int, thickness: int, h: int, w: int, sz: int) -> int:
    if side == _N:
        return thickness + 1
    if side == _S:
        return h - thickness - sz
    if side == _W:
        return thickness + 1
    return w - thickness - sz


def _pick_span(
    axis_len: int,
    required: list[tuple[int, int]],
    min_len: int,
    max_len: int,
    low: int = 0,
    high: int | None = None,
) -> tuple[int, int] | None:
    """Pick a span covering every required interval and staying inside [low, high]."""
    if high is None:
        high = axis_len - 1
    if low > high:
        return None
    req0 = min(a for a, _ in required)
    req1 = max(b for _, b in required)
    if req0 < low or req1 > high:
        return None

    lo = max(min_len, req1 - req0 + 1)
    hi = min(max_len, high - low + 1)
    choices: list[tuple[int, int, int]] = []
    for length in range(lo, hi + 1):
        start0 = max(low, req1 - length + 1)
        start1 = min(req0, high - length + 1)
        for start in range(start0, start1 + 1):
            end = start + length - 1
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


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


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


def _choose_source(
    template: dict[str, object],
    h: int,
    w: int,
    sz: int,
    lane: dict[int, int],
) -> tuple[int, int] | None:
    sides = template["sides"]

    if sides == (_N, _E, _S):
        min_r = max(SQUARE_EDGE_GAP, lane[_N] + 2)
        max_r = min(h - sz - SQUARE_EDGE_GAP, lane[_S] - 2)
        min_c = SQUARE_EDGE_GAP
        max_c = min(w - sz - SQUARE_EDGE_GAP, lane[_E] - 2)
    elif sides == (_N, _W, _S):
        min_r = max(SQUARE_EDGE_GAP, lane[_N] + 2)
        max_r = min(h - sz - SQUARE_EDGE_GAP, lane[_S] - 2)
        min_c = max(SQUARE_EDGE_GAP, lane[_W] + 2)
        max_c = w - sz - SQUARE_EDGE_GAP
    else:
        min_r = SQUARE_EDGE_GAP
        max_r = min(h - sz - SQUARE_EDGE_GAP, lane[_S] - 2)
        min_c = max(SQUARE_EDGE_GAP, lane[_W] + 2)
        max_c = min(w - sz - SQUARE_EDGE_GAP, lane[_E] - 2)

    if min_r > max_r or min_c > max_c:
        return None

    # Bias toward central placements to keep the hook readable.
    r_choices = list(range(min_r, max_r + 1))
    c_choices = list(range(min_c, max_c + 1))
    r_mid = (min_r + max_r) / 2
    c_mid = (min_c + max_c) / 2
    r_choices.sort(key=lambda x: (abs(x - r_mid), random.random()))
    c_choices.sort(key=lambda x: (abs(x - c_mid), random.random()))
    sr = r_choices[0]
    sc = c_choices[0]
    return int(sr), int(sc)


def _required_spans(
    template: dict[str, object],
    sr: int,
    sc: int,
    sz: int,
    lane: dict[int, int],
) -> dict[int, list[tuple[int, int]]]:
    sides = template["sides"]
    required = {side: [] for side in sides}

    if sides == (_N, _E, _S):
        required[_N].append((sc, sc + sz - 1))
        required[_E].append((lane[_N], lane[_N] + sz - 1))
        required[_E].append((sr, sr + sz - 1))
        required[_S].append((lane[_E], lane[_E] + sz - 1))
        required[_S].append((sc, sc + sz - 1))
    elif sides == (_N, _W, _S):
        required[_N].append((sc, sc + sz - 1))
        required[_W].append((lane[_N], lane[_N] + sz - 1))
        required[_W].append((sr, sr + sz - 1))
        required[_S].append((lane[_W], lane[_W] + sz - 1))
        required[_S].append((sc, sc + sz - 1))
    else:
        required[_E].append((sr, sr + sz - 1))
        required[_S].append((lane[_E], lane[_E] + sz - 1))
        required[_S].append((sc, sc + sz - 1))
        required[_W].append((lane[_S], lane[_S] + sz - 1))
        required[_W].append((sr, sr + sz - 1))

    return required


def _build_spans(
    template: dict[str, object],
    h: int,
    w: int,
    sz: int,
    thickness: dict[int, int],
    sr: int,
    sc: int,
    lane: dict[int, int],
) -> dict[int, tuple[int, int]] | None:
    sides = template["sides"]
    required = _required_spans(template, sr, sc, sz, lane)
    min_len = SPAN_MIN_MULT * sz
    max_len = min(SPAN_MAX_MULT * sz, max(h, w))
    spans: dict[int, tuple[int, int]] = {}

    if sides == (_N, _E, _S):
        spans[_N] = _pick_span(w, required[_N], min_len, min(max_len, w), high=w - thickness[_E] - 1)
        spans[_E] = _pick_span(h, required[_E], min_len, min(max_len, h), low=thickness[_N])
        spans[_S] = _pick_span(w, required[_S], min_len, min(max_len, w), high=w - thickness[_E] - 1)
    elif sides == (_N, _W, _S):
        spans[_N] = _pick_span(w, required[_N], min_len, min(max_len, w), low=thickness[_W])
        spans[_W] = _pick_span(h, required[_W], min_len, min(max_len, h), low=thickness[_N])
        spans[_S] = _pick_span(w, required[_S], min_len, min(max_len, w), low=thickness[_W])
    else:
        spans[_E] = _pick_span(h, required[_E], min_len, min(max_len, h), high=h - thickness[_S] - 1)
        spans[_S] = _pick_span(w, required[_S], min_len, min(max_len, w), low=thickness[_W])
        spans[_W] = _pick_span(h, required[_W], min_len, min(max_len, h), high=h - thickness[_S] - 1)

    if any(span is None for span in spans.values()):
        return None
    return spans


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    template = random.choice(_TEMPLATES)
    sides = template["sides"]

    thickness = {side: random.randint(*OBS_THICKNESS_RANGE) for side in sides}
    lane = {side: _lane_for_side(side, thickness[side], h, w, sz) for side in sides}

    src = _choose_source(template, h, w, sz, lane)
    if src is None:
        return None
    sr, sc = src

    spans = _build_spans(template, h, w, sz, thickness, sr, sc, lane)
    if spans is None:
        return None

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    side_colors = dict(zip(sides, random.sample(legend_colors, len(sides))))
    turns = {side_colors[side]: template["turn"] for side in sides}
    for color in legend_colors:
        turns.setdefault(color, random.choice(["L", "R"]))

    bg_choices = [c for c in BG_COLOR_POOL if c not in legend_colors]
    bg = random.choice(bg_choices)
    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL

    occupied = np.zeros((h, w), dtype=bool)
    occupied[sr : sr + sz, sc : sc + sz] = True
    for side in sides:
        r0, r1, c0, c1 = _obstacle_rect(side, thickness[side], spans[side], h, w)
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
            return None
        if not (r1 < sr - 1 or r0 > sr + sz or c1 < sc - 1 or c0 > sc + sz):
            return None
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        base[r0 : r1 + 1, c0 : c1 + 1] = side_colors[side]

    vals, counts = np.unique(base, return_counts=True)
    if int(vals[counts.argmax()]) != bg:
        return None

    try:
        expected, stats = _simulate_sequential(base, turns, bg)
        old = _simulate_independent(base, turns, bg)
    except AssertionError:
        return None

    trail_turn_dirs = [int(info["dir"]) for info in stats if int(info["trail_turns"]) > 0]
    total_hits = sum(len(info["hits"]) for info in stats)
    if len(trail_turn_dirs) < 1:
        return None
    if total_hits < 3:
        return None
    if np.array_equal(expected, old):
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
