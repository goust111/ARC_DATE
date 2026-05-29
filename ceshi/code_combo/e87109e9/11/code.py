# Combo variant of e87109e9 (index 11)
# combo_spec: D6/0+D16/1
# Name: 边缘反弹加射线碰撞终止

import random
import sys

import numpy as np

_SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if _SOLVER_DIR not in sys.path:
    sys.path.append(_SOLVER_DIR)

from arc_utils import bbox_coords, components_mask


# Design analysis for the fused D6/0 + D16/1 variant
# ===================================================
# a) Concrete behavior and trace
#    The visible ingredients stay close to the base task: a 6x24 legend strip, one
#    solid teal source square, and several edge-touching obstacle rectangles. The
#    fused rule changes only the beam dynamics after launch.
#
#    A representative trace in the generator template is:
#    - Beam N goes straight to the top wall, bounces 180 degrees, returns through
#      the center lane, hits the south obstacle, turns east, and paints a long
#      horizontal run near the bottom.
#    - Beam E goes to the right wall, bounces back, reaches the west obstacle, turns
#      south, and then stops exactly when its next footprint would enter Beam N's
#      already-painted bottom run.
#    - Beams S and W start later and are cut off almost immediately by the old N/E
#      trails.
#    Relative to the original task, the crucial new geometry is that a wall hit no
#    longer ends the beam; it can come back and trigger obstacle turns on the return
#    trip, creating branches that do not exist in the edge-stop version.
#
# b) Complexity calibration
#    A naive composition of "180-degree bounce" and "self-collision stop" would
#    collapse: after any bounce, the next step would immediately re-enter the beam's
#    own previous footprint, making D16 visually indistinguishable from the original
#    edge-stop task. To keep both dimensions observable, the solver treats finished
#    earlier beams as hard blockers, while self-interaction inside the current beam is
#    handled by state-cycle termination. The generator then forces:
#    - at least one beam whose extra branch exists only because of bouncing, and
#    - at least two later beams that stop because of old teal trails.
#
# c) Main() assumption audit
#    Relevant assumptions from the base solver and how they change:
#    - `if not in_bounds(...): break` is no longer valid. A border touch now flips
#      direction by 180 degrees and keeps the beam at the same footprint.
#    - "Any teal in the next footprint is harmless" is false once older beams leave
#      trails. Collision checks must ignore the current footprint overlap but stop on
#      older-beam teal elsewhere in the next footprint.
#    - With bouncing enabled, a beam can revisit its own states. Using the old
#      `assert state not in seen` would reject legitimate fused samples, so repeated
#      `(r, c, d)` is treated as a finished self-loop with no further new paint.
#    - Obstacle lookup still assumes the next footprint contains at most one obstacle
#      color after discarding teal and background. The generator keeps rectangles
#      disjoint, so this remains safe.
#
# d) Placement legality
#    The generator keeps four solid edge-touching rectangles, one per side, and uses
#    span constraints to avoid corner overlaps. North/east rectangles are decoys and
#    are explicitly kept off the active lanes, while south/west rectangles are placed
#    so the intended bounced return paths hit them.
#
# e) Visual observability
#    Across 2-3 samples, a human can see two repeated effects:
#    - long extra branches appear only after a wall turnaround, and
#    - later beams terminate when they approach previously painted teal corridors.
#    No hidden color ordering is used. Colors still serve only as legend keys for
#    left/right obstacle turns.
#
# f) Visual cleanliness
#    Active paths are confined to one center-return beam and one side-return beam, so
#    the output reads as a large L-shape plus a blocking vertical lane instead of a
#    dense maze. Decoy rectangles stay away from the active lanes to avoid ambiguous
#    overlaps.
#
# g) Color reasoning
#    The fused variant introduces no new color semantics. Teal remains the unique beam
#    color, the legend still only encodes left/right turn direction, and all obstacle
#    colors are drawn from the same small pool as in the base task.
#
# h) Generator guarantees for new assumptions
#    Every accepted sample must satisfy all of the following:
#    - fused output equals `main(input)`;
#    - total bounce count is at least 3, with Beam N bouncing multiple times;
#    - at least 2 beams stop because of old trails;
#    - fused output differs from both the collision-only and bounce-only counterfactuals.
#    This prevents degeneration back to either single-dimension behavior.

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
OBS_THICKNESS_RANGE = (2, 4)
SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.45
MAX_RETRIES = 500


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
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
        color = int(next(iter(colors)))
        xs = np.where(interior == color)[1]
        x = int(xs[0])
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


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


def _trace_beams(
    base: np.ndarray,
    turns: dict[int, str],
    bg: int,
    *,
    bounce_at_edge: bool,
    collision_stop: bool,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    h, w = base.shape
    out = base.copy()
    blocked = np.zeros((h, w), dtype=bool)

    source_mask = np.zeros((h, w), dtype=bool)
    source_mask[sr : sr + sz, sc : sc + sz] = True

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint(r: int, c: int, beam_mask: np.ndarray) -> None:
        footprint = out[r : r + sz, c : c + sz]
        bg_mask = base[r : r + sz, c : c + sz] == bg
        footprint[bg_mask] = TEAL
        out[r : r + sz, c : c + sz] = footprint
        beam_mask[r : r + sz, c : c + sz] |= bg_mask

    def obstacle_at(r: int, c: int) -> int | None:
        blk = out[r : r + sz, c : c + sz]
        colors = set(map(int, np.unique(blk)))
        colors.discard(bg)
        colors.discard(TEAL)
        if not colors:
            return None
        assert len(colors) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(colors)))

    def collides_with_blocked(r: int, c: int, nr: int, nc: int) -> bool:
        footprint = blocked[nr : nr + sz, nc : nc + sz]
        if not footprint.any():
            return False
        for rr in range(nr, nr + sz):
            for cc in range(nc, nc + sz):
                if not blocked[rr, cc]:
                    continue
                if r <= rr < r + sz and c <= cc < c + sz:
                    continue
                return True
        return False

    infos: list[dict[str, object]] = []
    for d0 in range(4):
        r, c, d = sr, sc, d0
        seen: set[tuple[int, int, int]] = set()
        beam_mask = np.zeros((h, w), dtype=bool)
        info: dict[str, object] = {"hits": [], "bounces": 0, "stop": "cycle", "steps": 0}

        while True:
            state = (r, c, d)
            if state in seen:
                info["stop"] = "cycle"
                break
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                if not bounce_at_edge:
                    info["stop"] = "edge"
                    break
                d = (d + 2) % 4
                info["bounces"] = int(info["bounces"]) + 1
                continue

            if collision_stop and collides_with_blocked(r, c, nr, nc):
                info["stop"] = "trail"
                break

            obs = obstacle_at(nr, nc)
            if obs is None:
                r, c = nr, nc
                paint(r, c, beam_mask)
                info["steps"] = int(info["steps"]) + 1
                continue

            assert obs in turns, "obstacle color must be encoded by legend"
            cast_hits = info["hits"]
            assert isinstance(cast_hits, list)
            cast_hits.append((d, obs, nr, nc))
            d = _rot(d, turns[obs])

        blocked |= beam_mask & ~source_mask
        infos.append(info)

    return out, infos


def _counterfactual_delta(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def _build_active_spans(
    h: int,
    w: int,
    sz: int,
    sr: int,
    sc: int,
    thickness: dict[int, int],
) -> dict[int, tuple[int, int]] | None:
    span_min = SPAN_MIN_MULT * sz
    span_max = min(SPAN_MAX_MULT * sz, max(h, w))

    south_span = _choose_span(
        w,
        [(sc, sc + sz - 1)],
        [(0, thickness[_W]), (w - thickness[_E], w - 1)],
        span_min,
        min(span_max, w),
    )
    west_span = _choose_span(
        h,
        [(sr, sr + sz - 1)],
        [(0, thickness[_N]), (h - thickness[_S], h - 1)],
        span_min,
        min(span_max, h),
    )
    if south_span is None or west_span is None:
        return None

    north_anchor_lo = max(sc + sz + 3, w // 2)
    north_anchor_hi = w - thickness[_E] - sz - 2
    if north_anchor_lo > north_anchor_hi:
        return None
    north_anchor = random.randint(north_anchor_lo, north_anchor_hi)
    north_span = _choose_span(
        w,
        [(north_anchor, north_anchor + sz - 1)],
        [
            (0, thickness[_W]),
            (w - thickness[_E], w - 1),
            (sc - 1, sc + sz),
        ],
        span_min,
        min(span_max, w),
    )
    if north_span is None:
        return None

    east_anchor_lo = thickness[_N] + 1
    east_anchor_hi = sr - 3
    if east_anchor_lo > east_anchor_hi:
        return None
    east_anchor = random.randint(east_anchor_lo, east_anchor_hi)
    lane_s = h - thickness[_S] - sz
    east_span = _choose_span(
        h,
        [(east_anchor, east_anchor + sz - 1)],
        [
            (0, thickness[_N]),
            (h - thickness[_S], h - 1),
            (sr - 1, sr + sz),
            (lane_s, lane_s + sz - 1),
        ],
        span_min,
        min(span_max, h),
    )
    if east_span is None:
        return None

    return {_N: north_span, _E: east_span, _S: south_span, _W: west_span}


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    thickness = {side: random.randint(*OBS_THICKNESS_RANGE) for side in (_N, _E, _S, _W)}

    min_r = max(4, thickness[_N] + 3)
    max_r = h - thickness[_S] - sz - 4
    min_c = max(5, thickness[_W] + 4)
    max_c = w - sz - 4
    if min_r > max_r or min_c > max_c:
        return None

    sr = random.randint(min_r, max_r)
    sc = random.randint(min_c, max_c)

    spans = _build_active_spans(h, w, sz, sr, sc, thickness)
    if spans is None:
        return None

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    side_color = dict(zip((_N, _E, _S, _W), random.sample(legend_colors, 4)))
    bg = random.choice([c for c in BG_COLOR_POOL if c not in legend_colors])

    turns = {
        side_color[_N]: random.choice(["L", "R"]),
        side_color[_E]: random.choice(["L", "R"]),
        side_color[_S]: "L",
        side_color[_W]: "L",
    }

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL

    occupied = np.zeros((h, w), dtype=bool)
    for side in (_N, _E, _S, _W):
        r0, r1, c0, c1 = _obstacle_rect(side, thickness[side], spans[side], h, w)
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
            return None
        if not (r1 < sr - 1 or r0 > sr + sz or c1 < sc - 1 or c0 > sc + sz):
            return None
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        base[r0 : r1 + 1, c0 : c1 + 1] = side_color[side]

    legend = _make_legend(legend_colors, turns)
    inp = np.vstack([legend, base])

    try:
        fused_out, infos = _trace_beams(base, turns, bg, bounce_at_edge=True, collision_stop=True)
        bounce_only_out, bounce_only_infos = _trace_beams(base, turns, bg, bounce_at_edge=True, collision_stop=False)
        collision_only_out, collision_only_infos = _trace_beams(base, turns, bg, bounce_at_edge=False, collision_stop=True)
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, fused_out):
        return None

    stops = [str(info["stop"]) for info in infos]
    bounces = [int(info["bounces"]) for info in infos]
    trail_stops = sum(stop == "trail" for stop in stops)
    if stops[0] != "cycle":
        return None
    if trail_stops < 2:
        return None
    if bounces[0] < 3 or sum(bounces) < 4:
        return None

    fused_vs_collision = _counterfactual_delta(fused_out, collision_only_out)
    fused_vs_bounce = _counterfactual_delta(fused_out, bounce_only_out)
    if fused_vs_collision < 6 * sz:
        return None
    if fused_vs_bounce < 2 * sz:
        return None

    if [str(info["stop"]) for info in collision_only_infos].count("trail") >= trail_stops:
        return None
    if sum(int(info["bounces"]) for info in bounce_only_infos) < sum(bounces):
        return None

    return inp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch four teal beams from the square, using obstacle turns plus edge rebounds.

    Input structure:
    - The top band is a dedicated legend: each of the four colored vertical bars sits on the
      left or right side of its cell to encode a counterclockwise or clockwise obstacle turn.
    - The lower board contains one solid teal square on a uniform background and several
      edge-touching obstacle rectangles whose colors appear in the legend.

    Rule:
    - Remove the legend and simulate the four beams on the lower board in fixed order N, E, S, W.
    - Each beam is an sz-by-sz sliding copy of the source square and paints teal only onto
      background cells. On meeting an obstacle color, it turns according to that color's legend bar.
    - When a beam would step out of the grid, it does not stop; it reverses direction in place and
      continues. When a later beam would enter an older completed teal trail outside its own current
      footprint, that beam terminates immediately.
    - If a beam revisits the same `(top-left row, top-left col, direction)` state, it is finished
      because any further motion would only retrace an already-seen loop.

    Key insight: The legend still controls obstacle turns, but the final geometry now depends on two
    coupled effects: wall rebounds create return trips that open new branches, and those earlier
    branches then act as blockers for later beams.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    turns = _parse_legend(g, legend_h)
    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    out, _ = _trace_beams(base, turns, bg, bounce_at_edge=True, collision_stop=True)
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e87109e9 combo input")
