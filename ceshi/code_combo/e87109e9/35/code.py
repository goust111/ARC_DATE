# Combo variant of e87109e9 (index 35)
# combo_spec: D6/0+D12/0+D14/0
# Name: 顺序发射轨迹碰撞障碍消失

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, components_mask


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    The visible language is still the original one: a 6x24 legend strip on top,
#    one solid teal source square below, and four edge-touching colored
#    rectangles. The fusion changes only the time dynamics.
#
#    Stable template used by the generator:
#    - N obstacle turns clockwise, so the up beam bends right and erases the
#      north obstacle.
#    - E obstacle turns clockwise, so the right beam bends down and erases the
#      east obstacle.
#    - S obstacle turns counterclockwise, so the down beam bends right and then
#      stops on the already-painted downward E trail.
#    - W obstacle turns clockwise, so the left beam bends up. Because the north
#      obstacle was already erased, this beam can continue through the top lane;
#      without erasure it would hit the north obstacle and bend again.
#
#    Small trace:
#    - Original task: all 4 beams are independent, obstacles stay fixed, and all
#      beams run until the border.
#    - Fused task:
#      1. Beam N hits N, N disappears, beam paints a top horizontal trail.
#      2. Beam E hits E, E disappears, beam paints a right-side vertical trail.
#      3. Beam S hits S, S disappears, turns right, then terminates when its next
#         footprint would enter E's trail.
#      4. Beam W hits W, W disappears, turns up, and reaches the top because N
#         is no longer there.
#    The output difference is therefore not one isolated tweak: obstacle removal
#    changes later geometry, and trail collisions truncate later paths.
#
# b) Complexity calibration
#    A fully random generator would often degenerate back toward the original
#    task, or would hide the erase effect behind an earlier trail collision. To
#    keep the combo legible, the generator uses one constrained four-obstacle
#    template with fixed turn roles {N:R, E:R, S:L, W:R}. Randomness remains in
#    board size, source size, source placement, obstacle thickness, obstacle
#    spans, and color assignment. This keeps visual variety while preserving one
#    readable causal story per sample.
#
# c) Main() assumption audit
#    - The original solver discarded every teal pixel while reading the next
#      footprint. That is no longer safe because painted trails are now active
#      blockers. The combo solver therefore tracks trail occupancy explicitly.
#    - A naive teal-collision test would fail immediately because a sliding
#      sz-by-sz footprint overlaps its own current location on every normal step.
#      The collision check must ignore trail cells inside the current footprint.
#    - The source square is teal but is not a "trail" obstacle. The combo solver
#      excludes the source mask from accumulated trail masks.
#    - Obstacle lookup in the original code assumed obstacles persist forever.
#      That becomes false after D12; the combo solver mutates the working grid
#      and erases the whole obstacle component as soon as it is hit.
#    - Cycle checks on (r, c, d) remain valid here because trails only grow and
#      obstacles only disappear, so revisiting the same motion state would imply a
#      self-intersection that the collision rule already forbids. The generator
#      further constrains the layout to one-hit early beams and a single planned
#      trail stop.
#
# d) Placement legality
#    Obstacles remain solid edge-touching rectangles with distinct colors. The
#    generator chooses source bounds before spans, then chooses each span from a
#    required/excluded interval system so rectangles stay in-bounds, avoid corner
#    overlap, do not touch the source square, and force the intended lanes.
#
# e) Visual observability
#    The order cue is geometric, not numeric: the first two long trails exist in
#    the output before the shorter third trail is truncated against one of them.
#    The erase cue is also geometric: the final left beam reaches a corridor that
#    would still be blocked by the top obstacle under a no-erase rule. No color
#    ordering or hidden metadata is used.
#
# f) Visual cleanliness
#    All obstacles stay on the border and all interactions happen in open
#    interior corridors. Only one later beam is truncated by trail collision, so
#    the board does not become a dense knot of overlapping teal blocks.
#
# g) Change inferability
#    Colors keep their original role as legend keys only. There is no new color
#    remapping burden; every meaningful change is encoded by path geometry,
#    object persistence, and the fixed emission order.
#
# h) Generator guarantees for new assumptions
#    Accepted samples must satisfy all of the following:
#    - combo stop reasons are exactly ['edge', 'edge', 'trail', 'edge']
#    - combo hit counts are exactly [1, 1, 1, 1]
#    - in the no-erase ablation, the W beam gains an extra obstacle hit
#    - combo output differs from both the no-erase sequential ablation and the
#      original independent no-collision/no-erase ablation by a visible margin
#    These checks prevent degeneration into the original task or into a partial
#    variant where one of the fused controls becomes irrelevant.

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
SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.45

MAX_RETRIES = 500


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
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
        x = int(np.where(interior == color)[1][0])
        turn[color] = "L" if x <= (interior.shape[1] - 1) // 2 else "R"

    assert len(turn) == 4, "legend must encode four turn colors"
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
    assert len(candidates) == 1, "main grid must contain exactly one solid teal square"
    return candidates[0]


def _rot(d: int, turn: str) -> int:
    return (d - 1) % 4 if turn == "L" else (d + 1) % 4


def _simulate_dynamic(
    base: np.ndarray,
    turns: dict[int, str],
    bg: int,
    *,
    collision_stop: bool,
    erase_obstacles: bool,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    out = base.copy()
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    h, w = out.shape

    source_mask = np.zeros_like(base, dtype=bool)
    source_mask[sr : sr + sz, sc : sc + sz] = True
    earlier_trail_mask = np.zeros_like(base, dtype=bool)

    colors = [int(c) for c in np.unique(base) if int(c) not in (bg, TEAL)]
    obstacle_masks = {color: (base == color) for color in colors}

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def has_trail_collision(
        r: int, c: int, nr: int, nc: int, beam_mask: np.ndarray
    ) -> bool:
        trail_mask = earlier_trail_mask | beam_mask
        for rr in range(nr, nr + sz):
            for cc in range(nc, nc + sz):
                if not trail_mask[rr, cc]:
                    continue
                if r <= rr < r + sz and c <= cc < c + sz:
                    continue
                return True
        return False

    def paint_block(r: int, c: int, beam_mask: np.ndarray) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = TEAL
        out[r : r + sz, c : c + sz] = blk
        beam_mask[r : r + sz, c : c + sz] = True
        beam_mask[source_mask] = False

    def obstacle_color_at_next(nr: int, nc: int) -> int | None:
        blk = out[nr : nr + sz, nc : nc + sz]
        cols = set(map(int, np.unique(blk)))
        cols.discard(bg)
        cols.discard(TEAL)
        if not cols:
            return None
        assert len(cols) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(cols)))

    infos: list[dict[str, object]] = []
    for d0 in range(4):
        r, c, d = sr, sc, d0
        beam_mask = np.zeros_like(base, dtype=bool)
        hits: list[tuple[int, int, int, int]] = []
        stop_reason = "edge"
        seen: set[tuple[int, int, int]] = set()
        steps = 0

        while True:
            state = (r, c, d)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                stop_reason = "edge"
                break

            if collision_stop and has_trail_collision(r, c, nr, nc, beam_mask):
                stop_reason = "trail"
                break

            oc = obstacle_color_at_next(nr, nc)
            if oc is None:
                r, c = nr, nc
                paint_block(r, c, beam_mask)
                steps += 1
                continue

            assert oc in turns, "obstacle color must be encoded by legend"
            hits.append((d, oc, nr, nc))
            if erase_obstacles:
                mask = obstacle_masks[oc]
                out[mask] = bg
                obstacle_masks[oc] = np.zeros_like(mask)
            d = _rot(d, turns[oc])

        earlier_trail_mask |= beam_mask
        infos.append(
            {
                "dir": d0,
                "hits": hits,
                "stop": stop_reason,
                "steps": steps,
                "trail_cells": int(beam_mask.sum()),
            }
        )

    return out, infos


def _simulate_original_like(base: np.ndarray, turns: dict[int, str], bg: int) -> np.ndarray:
    out, _infos = _simulate_dynamic(
        base,
        turns,
        bg,
        collision_stop=False,
        erase_obstacles=False,
    )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit teal beams in a fixed order while trail collisions stop later beams and hit obstacles vanish.

    Input structure:
    - Top strip: a grey legend band with four colored vertical bars; the bar's left/right placement inside each cell encodes counterclockwise or clockwise turning.
    - Below it: one uniform-background board containing a solid teal square and several solid edge-touching obstacle rectangles whose colors appear in the legend.

    Rule:
    - Remove the legend strip and work on the lower board only.
    - Fire four sz-by-sz beams from the teal square in the order up, then right, then down, then left.
    - A beam slides one cell at a time, painting teal on background cells. If the next footprint meets a colored obstacle, the beam turns according to the legend and that entire obstacle disappears from the board.
    - If the next footprint would enter any already-painted trail cell outside the beam's own current footprint, the beam terminates immediately instead of turning. Trails laid by earlier beams therefore become new blockers for later beams.
    - A beam also stops when its next step would leave the board.

    Key insight: The legend still tells you how colors turn a beam, but solving the task now requires time-order simulation because the board state mutates twice: hit obstacles are erased, and completed trails become blocking geometry for the remaining beams.
    """
    g = np.asarray(grid, dtype=int)

    legend_bg = LEGEND_BG
    rows_with_legend = np.where(np.any(g == legend_bg, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    legend_turn = _parse_legend(g, legend_h)
    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, counts = np.unique(base, return_counts=True)
    bg = int(vals[counts.argmax()])
    out, _infos = _simulate_dynamic(
        base,
        legend_turn,
        bg,
        collision_stop=True,
        erase_obstacles=True,
    )
    return out


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


def _rand_thickness(lo: int, hi: int) -> int | None:
    return random.randint(lo, hi) if hi >= lo else None


def _thickness_pair(axis: int, sz: int) -> tuple[int, int] | None:
    lo, hi_cap = OBS_THICKNESS_RANGE
    room = axis - sz - 6
    if room < 2 * lo:
        return None
    a = random.randint(lo, min(hi_cap, room - lo))
    b = _rand_thickness(lo, min(hi_cap, room - a))
    if b is None:
        return None
    return a, b


def _pick_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
    *,
    low: int = 0,
    high: int | None = None,
) -> tuple[int, int] | None:
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


def _choose_source(
    h: int,
    w: int,
    sz: int,
    thickness: dict[int, int],
) -> tuple[int, int] | None:
    lane_n = thickness[_N] + 1
    lane_s = h - thickness[_S] - sz
    lane_w = thickness[_W] + 1
    lane_e = w - thickness[_E] - sz

    min_r = max(SQUARE_EDGE_GAP, lane_n + 2)
    max_r = min(h - sz - SQUARE_EDGE_GAP, lane_s - 2)
    min_c = max(SQUARE_EDGE_GAP, lane_w + sz + 1)
    max_c = min(w - sz - SQUARE_EDGE_GAP, lane_e - 2)
    if min_r > max_r or min_c > max_c:
        return None

    rows = list(range(min_r, max_r + 1))
    cols = list(range(min_c, max_c + 1))
    r_mid = (min_r + max_r) / 2
    c_mid = (min_c + max_c) / 2
    rows.sort(key=lambda x: (abs(x - r_mid), random.random()))
    cols.sort(key=lambda x: (abs(x - c_mid), random.random()))
    return int(rows[0]), int(cols[0])


def _build_spans(
    h: int,
    w: int,
    sz: int,
    sr: int,
    sc: int,
    thickness: dict[int, int],
) -> dict[int, tuple[int, int]] | None:
    lane_n = thickness[_N] + 1
    lane_s = h - thickness[_S] - sz
    lane_w = thickness[_W] + 1
    lane_e = w - thickness[_E] - sz

    min_len = SPAN_MIN_MULT * sz
    max_len_w = min(SPAN_MAX_MULT * sz, w)
    max_len_h = min(SPAN_MAX_MULT * sz, h)

    spans: dict[int, tuple[int, int]] = {}
    spans[_N] = _pick_span(
        w,
        [(lane_w, lane_w + sz - 1), (sc, sc + sz - 1)],
        [],
        min_len,
        max_len_w,
        low=thickness[_W],
        high=w - thickness[_E] - 1,
    )
    spans[_E] = _pick_span(
        h,
        [(sr, sr + sz - 1)],
        [(lane_n, lane_n + sz - 1)],
        min_len,
        max_len_h,
        low=thickness[_N],
        high=h - thickness[_S] - 1,
    )
    spans[_S] = _pick_span(
        w,
        [(sc, sc + sz - 1)],
        [(lane_e, lane_e + sz - 1)],
        min_len,
        max_len_w,
        low=thickness[_W],
        high=w - thickness[_E] - 1,
    )
    spans[_W] = _pick_span(
        h,
        [(sr, sr + sz - 1)],
        [],
        min_len,
        max_len_h,
        low=thickness[_N],
        high=h - thickness[_S] - 1,
    )
    if any(span is None for span in spans.values()):
        return None
    return spans


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3

    ns = _thickness_pair(h, sz)
    ew = _thickness_pair(w, sz)
    if ns is None or ew is None:
        return None
    thickness = {_N: ns[0], _S: ns[1], _W: ew[0], _E: ew[1]}

    src = _choose_source(h, w, sz, thickness)
    if src is None:
        return None
    sr, sc = src

    spans = _build_spans(h, w, sz, sr, sc, thickness)
    if spans is None:
        return None

    turns_by_side = {_N: "R", _E: "R", _S: "L", _W: "R"}
    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    side_colors = dict(zip((_N, _E, _S, _W), random.sample(legend_colors, 4)))
    turns = {side_colors[side]: turns_by_side[side] for side in (_N, _E, _S, _W)}

    bg = random.choice([c for c in BG_COLOR_POOL if c not in legend_colors])
    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL

    occupied = np.zeros((h, w), dtype=bool)
    occupied[sr : sr + sz, sc : sc + sz] = True
    for side in (_N, _E, _S, _W):
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
        combo, combo_infos = _simulate_dynamic(
            base,
            turns,
            bg,
            collision_stop=True,
            erase_obstacles=True,
        )
        no_erase, no_erase_infos = _simulate_dynamic(
            base,
            turns,
            bg,
            collision_stop=True,
            erase_obstacles=False,
        )
        original = _simulate_original_like(base, turns, bg)
    except AssertionError:
        return None

    combo_stops = [str(info["stop"]) for info in combo_infos]
    combo_hits = [len(info["hits"]) for info in combo_infos]
    no_erase_hits = [len(info["hits"]) for info in no_erase_infos]
    if combo_stops != ["edge", "edge", "trail", "edge"]:
        return None
    if combo_hits != [1, 1, 1, 1]:
        return None
    if no_erase_hits[3] < 2:
        return None
    if np.array_equal(combo, no_erase):
        return None
    if np.array_equal(combo, original):
        return None
    if int(np.sum(combo != no_erase)) < 2 * sz:
        return None
    if int(np.sum(combo != original)) < 4 * sz:
        return None

    legend = _make_legend(legend_colors, turns)
    inp = np.vstack([legend, base])
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, combo):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e87109e9 combo input")
