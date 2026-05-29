# Variant of e87109e9
# Dimension: D6 — 射线终止条件 (dst)
# Change: 射线遇到障碍物时转弯继续，最终在到达网格边缘时终止
#   → 遇到另一条射线轨迹时终止

import random
import sys

import numpy as np

_SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if _SOLVER_DIR not in sys.path:
    sys.path.append(_SOLVER_DIR)

from arc_utils import bbox_coords, components_mask


# Design analysis for the D6 variant
# ==================================
# a) Concrete behavior and trace
#    The visual language stays the same: a 6x24 legend on top, one solid teal
#    square in the lower board, and 4 edge-touching obstacle rectangles whose
#    colors appear in the legend. The change is only in how the moving teal
#    square stops. The square still slides one cell at a time and still turns
#    on obstacle contact using the legend. The new part is that each later beam
#    is blocked by already-painted teal pixels from earlier beams.
#
#    Small trace for blueprint C:
#    - Beam N hits the north obstacle, turns left, and reaches the left edge.
#    - Beam E hits the east obstacle, turns down, and reaches the bottom edge.
#    - Beam S hits the south obstacle, turns right, but stops when its next
#      footprint would enter the already-painted vertical E trail.
#    - Beam W hits the west obstacle, turns up, but stops when its next
#      footprint would enter the already-painted horizontal N trail.
#    Under the old rule, beams S and W would continue all the way to the grid
#    edge. Under the new rule, they truncate early at the first contact with
#    earlier teal trails.
#
# b) Complexity calibration
#    If we let the generator be unconstrained, many samples would accidentally
#    behave exactly like the original edge-stop task. To keep D6 observable, the
#    generator only emits layouts where the first two beams survive and the last
#    two are cut short by prior trails. Two stable blueprints are used:
#    - C: hit counts [1, 1, 1, 1]
#    - D: hit counts [2, 1, 1, 0]
#    This preserves obstacle-turn reasoning while making trail-collision
#    dependence obvious in 2-3 examples.
#
# c) Main() assumption audit
#    The original solver had a hidden assumption: any teal value in the next
#    footprint was harmless because teal only came from the moving beam itself.
#    That becomes false here. A naive "if teal in next block: stop" would break
#    immediately because a sliding sz x sz footprint overlaps its current
#    position on every normal step. The new solver therefore ignores teal cells
#    that lie inside the current footprint and only treats teal outside the
#    current footprint as a collision.
#
#    The original obstacle query also assumed the next footprint never mixed a
#    trail pixel with an obstacle. In the variant that can happen because the
#    overlap with the current footprint is teal. The solver keeps discarding teal
#    before reading the obstacle color, so turn detection still works.
#
# d) Placement legality
#    Obstacles remain solid edge-touching rectangles, never overlap each other,
#    and never touch the teal source square. The generator selects spans with
#    explicit required/excluded intervals so the intended beams hit the intended
#    obstacles while avoiding accidental corner overlaps or unintended second
#    hits.
#
# e) Visual observability
#    The rule is visible from geometry alone: two earlier long trails exist, and
#    two later beams terminate exactly when they would enter those trails. No
#    color-id ordering is used; the legend still only encodes left/right turns.
#
# f) Visual cleanliness
#    The generator keeps all non-source objects as large rectangles on the outer
#    border. Trail interactions happen in open background corridors, so the
#    stopping points are readable and not buried inside obstacle clutter.
#
# g) Color reasoning
#    Colors do not change. Obstacle colors are only keys for legend lookup, same
#    as the original task. This avoids introducing any new color mapping burden.
#
# h) Generator guarantees for new assumptions
#    The solver now assumes beam order matters and that later beams may be
#    clipped by earlier teal trails. The generator guarantees that by accepting
#    only samples with stop reasons ['edge', 'edge', 'trail', 'trail'] in the
#    fixed processing order N, E, S, W, plus a large output difference from the
#    old edge-stop simulation.

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)


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


def _find_teal_square(g: np.ndarray, teal: int = 8) -> tuple[int, int, int]:
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


def _trace_beams(
    base: np.ndarray,
    turns: dict[int, str],
    bg: int,
    *,
    collision_stop: bool,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    teal = 8
    sr, sc, sz = _find_teal_square(base, teal=teal)
    out = base.copy()
    H, W = out.shape
    infos: list[dict[str, object]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < H and c + sz - 1 < W

    def paint_block(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = teal
        out[r : r + sz, c : c + sz] = blk

    def has_trail_collision(r: int, c: int, nr: int, nc: int) -> bool:
        for rr in range(nr, nr + sz):
            for cc in range(nc, nc + sz):
                if int(out[rr, cc]) != teal:
                    continue
                if r <= rr < r + sz and c <= cc < c + sz:
                    continue
                return True
        return False

    def obstacle_color_at_next(nr: int, nc: int) -> int | None:
        blk = out[nr : nr + sz, nc : nc + sz]
        cols = set(map(int, np.unique(blk)))
        cols.discard(bg)
        cols.discard(teal)
        if not cols:
            return None
        assert len(cols) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(cols)))

    for d0 in range(4):
        r, c, d = sr, sc, d0
        hits: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        stop_reason = "edge"
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
            if collision_stop and has_trail_collision(r, c, nr, nc):
                stop_reason = "trail"
                break

            oc = obstacle_color_at_next(nr, nc)
            if oc is None:
                r, c = nr, nc
                paint_block(r, c)
                steps += 1
                continue

            assert oc in turns, "obstacle color must be encoded by legend"
            hits.append((d, oc, nr, nc))
            d = _rot(d, turns[oc])

        infos.append(
            {
                "hits": hits,
                "stop": stop_reason,
                "steps": steps,
            }
        )

    return out, infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch four teal beams from a square, turning at obstacle colors from a legend.

    Input structure:
    - Top band: a grey legend that contains four colored vertical bars; left/right placement inside each cell encodes turn direction.
    - Below: a main board with one solid teal square, a uniform background, and several edge-touching obstacle rectangles whose colors appear in the legend.

    Rule:
    - Crop away the legend and work only on the lower board.
    - Emit one sz x sz teal beam in each cardinal direction from the teal square, processing them in fixed order N, E, S, W.
    - A beam paints teal over background cells as it slides. On meeting an obstacle color, it turns left or right according to the legend and keeps going.
    - A beam now terminates as soon as its next footprint would enter any already-painted teal pixel outside its own current footprint. Grid boundaries still act as the physical fallback stop.

    Key insight: The legend still explains every turn, but the final shape depends on beam order because earlier teal trails block later beams.
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
    out, _infos = _trace_beams(base, legend_turn, bg, collision_stop=True)
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
MAX_RETRIES = 400


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


def _build_constraints(
    pattern: str,
    h: int,
    w: int,
    sz: int,
    sr: int,
    sc: int,
    thickness: dict[int, int],
) -> tuple[dict[int, str], dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    turns_by_side = (
        {_N: "L", _E: "R", _S: "L", _W: "R"}
        if pattern == "C"
        else {_N: "L", _E: "R", _S: "R", _W: "L"}
    )

    required: dict[int, list[tuple[int, int]]] = {
        _N: [(sc, sc + sz - 1)],
        _S: [(sc, sc + sz - 1)],
        _W: [(sr, sr + sz - 1)],
        _E: [(sr, sr + sz - 1)],
    }
    excluded: dict[int, list[tuple[int, int]]] = {side: [] for side in (_N, _E, _S, _W)}

    hit_pos = {
        _N: thickness[_N],
        _S: h - thickness[_S] - sz,
        _W: thickness[_W],
        _E: w - thickness[_E] - sz,
    }

    # Keep edge-touching rectangles from overlapping at the corners.
    excluded[_N].extend([(0, thickness[_W] - 1), (w - thickness[_E], w - 1)])
    excluded[_S].extend([(0, thickness[_W] - 1), (w - thickness[_E], w - 1)])
    excluded[_W].extend([(0, thickness[_N] - 1), (h - thickness[_S], h - 1)])
    excluded[_E].extend([(0, thickness[_N] - 1), (h - thickness[_S], h - 1)])

    if pattern == "C":
        # N must escape left rather than hit W; E must escape down rather than hit S.
        excluded[_W].append((hit_pos[_N], hit_pos[_N] + sz - 1))
        excluded[_S].append((hit_pos[_E], hit_pos[_E] + sz - 1))
        # W needs to reach N's horizontal trail before touching the top obstacle.
        excluded[_N].append((hit_pos[_W], hit_pos[_W] + sz - 1))
    else:
        # N must hit W, then descend on the left side; both later lower beams stop on that trail.
        required[_W].append((hit_pos[_N], hit_pos[_N] + sz - 1))
        excluded[_S].append((hit_pos[_W], hit_pos[_W] + sz - 1))
        excluded[_S].append((hit_pos[_E], hit_pos[_E] + sz - 1))

    return turns_by_side, required, excluded


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    pattern = random.choice(["C", "D"])

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
    sr, sc = random.randint(min_r, max_r), random.randint(min_c, max_c)

    turns_by_side, required, excluded = _build_constraints(
        pattern, h, w, sz, sr, sc, thickness
    )

    spans: dict[int, tuple[int, int]] = {}
    for side in (_N, _E, _S, _W):
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
    side_color = dict(zip((_N, _E, _S, _W), random.sample(legend_colors, 4)))
    bg = random.choice([c for c in BG_COLOR_POOL if c not in legend_colors])

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

    turns = {side_color[side]: turns_by_side[side] for side in (_N, _E, _S, _W)}
    legend = _make_legend(legend_colors, turns)
    inp = np.vstack([legend, base])

    try:
        new_out, infos = _trace_beams(base, turns, bg, collision_stop=True)
        old_out, old_infos = _trace_beams(base, turns, bg, collision_stop=False)
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, new_out):
        return None

    stops = [str(info["stop"]) for info in infos]
    old_stops = [str(info["stop"]) for info in old_infos]
    hit_counts = [len(info["hits"]) for info in infos]
    if stops != ["edge", "edge", "trail", "trail"]:
        return None
    if old_stops == stops:
        return None
    if pattern == "C" and hit_counts != [1, 1, 1, 1]:
        return None
    if pattern == "D" and hit_counts != [2, 1, 1, 0]:
        return None
    if int(np.sum(new_out != old_out)) < 4 * sz:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e87109e9 D6 input")
