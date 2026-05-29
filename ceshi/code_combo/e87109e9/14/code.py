# Combo variant of e87109e9 (index 14)
# combo_spec: D4/1+D15/0
# Name: 背景色编码周期加虚线轨迹
#
# Design analysis for the combo:
# - Concrete behavior: the original task paints every beam step as teal. Here the
#   beam geometry is unchanged, but the main-grid background color becomes a
#   global dash parameter. If bg=3, each beam paints 3 translated square-steps,
#   leaves 1 translated square-step as background, then repeats. Turns still come
#   only from the legend. Example: a north-going beam with steps s1..s8 was
#   solid in the base task; now with bg=3 it paints s1,s2,s3, skips s4, paints
#   s5,s6,s7, skips s8. If it turns after s5, the count continues across the
#   corner so the dashed trail follows the whole path rather than each segment
#   resetting independently.
# - Complexity calibration: only the paint schedule changes; obstacle geometry,
#   square width, and legend parsing stay stable. To keep the variation visible
#   and learnable, the generator uses only bg in {2,3,4} and cycles them across
#   calls, so train/test sets are likely to show repeated support instead of a
#   one-off arbitrary color-id mapping.
# - Assumption audit against the original main():
#   1. "Every empty move paints teal" is no longer valid; painting must depend on
#      a step counter and the bg-derived period.
#   2. Obstacle detection must not depend on already-painted trail cells, because
#      skipped dash cells remain bg; the solver therefore reads obstacle colors
#      from the immutable cropped base grid.
#   3. The original output could match the solid-path baseline when all paths are
#      short. The generator now rejects such cases by comparing dashed and solid
#      outputs and requiring visible missing teal cells.
# - Placement legality: the generator keeps the original edge-touching rectangle
#   construction and path-first span constraints, so obstacles remain disjoint,
#   in-bounds, and at least one cell away from the teal square.
# - Visual observability / neatness: only one extra visual layer is added, namely
#   regular one-step gaps inside the existing teal tracks. The generator requires
#   multiple skipped dash steps and a measurable teal-count drop versus the solid
#   baseline so the combo is plainly visible instead of degenerating back to the
#   original task.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask


_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    """Decode obstacle-color -> turn direction from the top legend band."""
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


def _find_teal_square(grid: np.ndarray, teal: int = 8) -> tuple[int, int, int]:
    """Return (r0, c0, size) for the unique solid teal square."""
    candidates: list[tuple[int, int, int]] = []
    for cells in components_mask(grid == teal, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w:
            continue
        if int(np.sum(grid[r0 : r1 + 1, c0 : c1 + 1] == teal)) != h * h:
            continue
        candidates.append((int(r0), int(c0), int(h)))
    assert len(candidates) == 1, "main grid must contain exactly one solid teal square"
    return candidates[0]


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def _dash_period_from_bg(bg: int) -> int:
    """The combo rule: the main background color code is the dash paint length."""
    assert bg not in (0, 5, 8), "main-grid background must be distinct from black/legend/teal"
    assert 1 <= bg <= 9, "ARC colors are single digits"
    return int(bg)


def _simulate_beams(
    base: np.ndarray,
    legend_turn: dict[int, str],
    bg: int,
    dash_period: int | None,
) -> tuple[np.ndarray, list[dict[str, int | tuple[int, ...]]]]:
    """
    Simulate the four beams on the cropped main grid.

    If dash_period is None, paint every step (solid baseline).
    Otherwise paint `dash_period` translated square-steps, then skip 1 step.
    """
    teal = 8
    sr, sc, sz = _find_teal_square(base, teal=teal)
    out = base.copy()
    H, W = out.shape
    stats: list[dict[str, int | tuple[int, ...]]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < H and c + sz - 1 < W

    def paint_block(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = teal
        out[r : r + sz, c : c + sz] = blk

    def obstacle_color_at_next(nr: int, nc: int) -> int | None:
        blk = base[nr : nr + sz, nc : nc + sz]
        colors = set(map(int, np.unique(blk)))
        colors.discard(bg)
        colors.discard(teal)
        if not colors:
            return None
        assert len(colors) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(colors)))

    cycle = None if dash_period is None else dash_period + 1

    for d0 in range(4):
        r, c, d = sr, sc, d0
        seen: set[tuple[int, int, int]] = set()
        steps = 0
        hits = 0
        painted = 0
        skipped = 0
        hit_colors: list[int] = []

        while True:
            state = (r, c, d)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            obstacle = obstacle_color_at_next(nr, nc)
            if obstacle is None:
                r, c = nr, nc
                do_paint = cycle is None or (steps % cycle) < dash_period
                if do_paint:
                    paint_block(r, c)
                    painted += 1
                else:
                    skipped += 1
                steps += 1
                continue

            assert obstacle in legend_turn, "obstacle color must be encoded by legend"
            d = _rot(d, legend_turn[obstacle])
            hits += 1
            hit_colors.append(int(obstacle))

        stats.append(
            {
                "steps": steps,
                "hits": hits,
                "painted_steps": painted,
                "skipped_steps": skipped,
                "hit_colors": tuple(hit_colors),
            }
        )

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four teal beams from a square, but paint them as background-controlled dashed paths.

    Input structure:
    - The top band is still a 4-cell legend on color 5; each colored vertical bar says whether that obstacle color causes a left or right turn.
    - The cropped main area contains one solid teal source square, several colored obstacle blocks, and one uniform non-black background color.

    Rule:
    - Remove the legend from the output.
    - Launch one beam in each cardinal direction, using the same moving square footprint as the original task and the same turn-on-obstacle logic.
    - The difference is the trail style: if the main background color is k, then along each beam path you paint k moved square-positions, leave the next moved square-position as background, and repeat this dash cycle even across turns.
    - Beams still stop only at the grid boundary.

    Key insight: the legend explains *where* beams turn, but the background color explains *how densely* the trajectory is painted; the background is no longer decorative metadata.
    """
    g = np.asarray(grid, dtype=int)

    legend_bg = 5
    rows_with_legend = np.where(np.any(g == legend_bg, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must stay in a top band"

    legend_turn = _parse_legend(g, legend_h)
    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    dash_period = _dash_period_from_bg(bg)
    out, _stats = _simulate_beams(base, legend_turn, bg, dash_period=dash_period)
    return out


# --------------- generator ---------------

# Color palettes
LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_PERIOD_OPTIONS = [2, 3, 4]
LEGEND_BG = 5
TEAL = 8

# Canvas dimensions
GRID_H_RANGE = (19, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]

# Teal square
TEAL_SZ2_PROB = 0.75
SQUARE_EDGE_GAP = 6

# Obstacles
OBS_COUNT_4_PROB = 0.65
OBS_THICKNESS_RANGE = (2, 4)
OBS_SINGLE_MARGIN = 6
OBS_BOTH_GAP = 2

# Span selection
SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.4

# Retry budget
MAX_RETRIES = 400

# Background choices are cycled to provide repeated support across consecutive calls.
_GEN_BG_INDEX = 0


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
        cutoff = lengths[min(len(lengths) - 1, max(0, len(lengths) // 2))]
        pool = [(start, end) for length, start, end in choices if length <= cutoff]
    else:
        pool = [(start, end) for _, start, end in choices]
    return random.choice(pool)


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors, strict=True):
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


def _select_chains(present: set[int], second_hits: int) -> dict[int, int] | None:
    candidates = [
        (side, nxt)
        for side in present
        for nxt in ((side - 1) % 4, (side + 1) % 4)
        if nxt in present
    ]
    random.shuffle(candidates)
    selected: list[tuple[int, int]] = []
    used: set[int] = set()
    for side, nxt in candidates:
        if len(selected) == second_hits:
            break
        if side in used or nxt in used:
            continue
        selected.append((side, nxt))
        used.update((side, nxt))
    return dict(selected) if len(selected) == second_hits else None


def _assign_turns(present: set[int], chains: dict[int, int]) -> dict[int, str]:
    side_turn: dict[int, str] = {}
    for side in present:
        if side in chains:
            nxt = chains[side]
            side_turn[side] = "L" if nxt == (side - 1) % 4 else "R"
        else:
            outward = [t for t in ("L", "R") if _rot(side, t) not in present]
            side_turn[side] = random.choice(outward if outward else ["L", "R"])
    return side_turn


def _compute_span_constraints(
    present: set[int],
    side_turn: dict[int, str],
    chains: dict[int, int],
    sr: int,
    sc: int,
    sz: int,
    lane: dict[int, int],
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    required: dict[int, list[tuple[int, int]]] = {s: [] for s in present}
    excluded: dict[int, list[tuple[int, int]]] = {s: [] for s in present}

    for side in present:
        base_coord = sc if side in (_N, _S) else sr
        required[side].append((base_coord, base_coord + sz - 1))

    def lane_interval(side: int) -> tuple[int, int]:
        return lane[side], lane[side] + sz - 1

    for side in present:
        nxt = _rot(side, side_turn[side])
        if nxt not in present:
            continue
        if chains.get(side) == nxt:
            required[nxt].append(lane_interval(side))
            after = _rot(nxt, side_turn[nxt])
            if after in present:
                excluded[after].append(lane_interval(nxt))
        else:
            excluded[nxt].append(lane_interval(side))

    return required, excluded


def _gen_attempt(bg: int) -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    obstacle_count = 4 if random.random() < OBS_COUNT_4_PROB else 3
    present = set(random.sample([_N, _E, _S, _W], obstacle_count))
    target_hits = random.choice([4, 5])
    second_hits = target_hits - obstacle_count
    if second_hits < 0 or second_hits > 2:
        return None

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
    if min_r > max_r or min_c > max_c:
        return None
    sr, sc = random.randint(min_r, max_r), random.randint(min_c, max_c)

    lane: dict[int, int] = {}
    if _N in present:
        lane[_N] = thickness[_N] + 1
    if _S in present:
        lane[_S] = h - thickness[_S] - sz
    if _W in present:
        lane[_W] = thickness[_W] + 1
    if _E in present:
        lane[_E] = w - thickness[_E] - sz

    chains = _select_chains(present, second_hits)
    if chains is None:
        return None
    side_turn = _assign_turns(present, chains)
    required, excluded = _compute_span_constraints(present, side_turn, chains, sr, sc, sz, lane)

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

    legend_colors = [c for c in LEGEND_COLOR_POOL if c != bg]
    assert len(legend_colors) == 4
    obstacle_colors = random.sample(legend_colors, obstacle_count)
    side_color = dict(zip(sorted(present), obstacle_colors, strict=False))

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

    vals, cnts = np.unique(base, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    turns = {side_color[s]: side_turn[s] for s in present}
    for color in legend_colors:
        turns.setdefault(color, random.choice(["L", "R"]))

    try:
        dashed_expected, stats = _simulate_beams(base, turns, bg, dash_period=_dash_period_from_bg(bg))
        solid_expected, _solid_stats = _simulate_beams(base, turns, bg, dash_period=None)
    except AssertionError:
        return None

    hit_counts = [int(beam["hits"]) for beam in stats]
    skipped_total = sum(int(beam["skipped_steps"]) for beam in stats)
    long_beams = sum(int(beam["skipped_steps"]) > 0 for beam in stats)
    visible_drop = int(np.count_nonzero((solid_expected == TEAL) & (dashed_expected != TEAL)))
    hit_colors = {
        int(color)
        for beam in stats
        for color in beam["hit_colors"]
    }
    if sum(hit_counts) != target_hits or any(c > 2 for c in hit_counts):
        return None
    if set(side_color.values()) - hit_colors:
        return None
    if skipped_total < 2 or long_beams < 1 or visible_drop < sz:
        return None

    legend = _make_legend(legend_colors, turns)
    inp = np.vstack([legend, base])
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, dashed_expected):
        return None
    if np.array_equal(dashed_expected, solid_expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_BG_INDEX

    preferred = BG_PERIOD_OPTIONS[_GEN_BG_INDEX % len(BG_PERIOD_OPTIONS)]
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(preferred)
        if grid is not None:
            _GEN_BG_INDEX += 1
            return grid

    for bg in BG_PERIOD_OPTIONS:
        for _ in range(MAX_RETRIES // 4):
            grid = _gen_attempt(bg)
            if grid is not None:
                _GEN_BG_INDEX += 1
                return grid
    raise RuntimeError("failed to generate e87109e9 combo input")
