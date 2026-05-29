# Combo variant of e87109e9 (index 10)
# combo_spec: D11/1+D13/2
# Name: 三态障碍物加干扰图例

import random

import numpy as np

from arc_utils import bbox_coords, components_mask

# Design analysis for the combo variant
# a) Concrete behavior:
#    - Keep the original crop rule: the top legend band is removed in the output.
#    - The teal square still emits four sz x sz beams.
#    - The legend now has 5 entries, not 4. Each entry is a 3-column cell:
#        left bar   -> 'L' (counterclockwise turn),
#        middle bar -> 'P' (penetrate straight through),
#        right bar  -> 'R' (clockwise turn).
#    - Only 3 legend colors actually appear in the main grid. The other 2 legend
#      colors are distractors and never occur below the legend.
#    - Small trace:
#        old task: a north-facing beam hits a top obstacle and must turn there.
#        combo task: that same beam can first cross an interior obstacle whose
#        legend marker is centered, then later hit an edge obstacle and turn.
#        Meanwhile two extra legend entries are visible but irrelevant because
#        their colors are absent from the main region.
# b) Complexity calibration:
#    - The main region always shows exactly 3 obstacle colors, and those 3 realize
#      all three behaviors {L, P, R}. This makes the fused rule visible in one
#      sample instead of requiring cross-sample accumulation.
#    - The 5-entry legend keeps distractor pressure high enough that a solver must
#      filter by color presence, but behavior decoding still has only 3 states.
#    - Two turning edge rectangles share one of the two turning colors, so the
#      variant does not collapse back to the original "one color per rectangle".
# c) Implicit assumptions broken from the original solver:
#    - Parsing can no longer assume only left/right markers; the center position
#      must be recognized explicitly.
#    - Parsing can no longer assume every legend entry is relevant; only colors
#      present in the post-crop main grid are actionable.
#    - Validation code can no longer treat repeated (color, behavior) contacts as
#      one obstacle hit, because two distinct rectangles may share the same turn
#      color. The simulator therefore resets contact compression after bg steps.
# d) Placement legality:
#    - The generator keeps the D11 path-first skeleton: 3 edge obstacles plus
#      1 interior pass-through rectangle, all axis-aligned and non-overlapping.
#    - The pass-through rectangle sits strictly between the square and one edge
#      obstacle, with bg gaps that keep objects visually separate.
#    - Span constraints still force deterministic beam lanes and avoid accidental
#      secondary collisions.
# e) Visual observability:
#    - Human-visible cues are purely geometric: 5 legend cells above, only 3 of
#      those colors below, and the middle legend position visibly means "no turn".
#    - Every generated sample contains at least one beam that exits the far side
#      of an obstacle, so the penetrate state is directly observable.
# f) Visual cleanliness:
#    - The extra complexity lives in the legend, not by stacking more objects into
#      the main grid. The main field still has one teal source and four rectangles.
#    - Shared-color turning rectangles are spatially separated, so they do not
#      merge into ambiguous composite shapes.
# g) Inference of the fused change:
#    - Distractors are identified by absence from the main grid, not by numeric id.
#    - Actual behaviors are inferred from repeated support: two rectangles often
#      share the same turn color, while the pass-through color appears on a single
#      interior obstacle that a beam crosses without bending.
# h) New solver assumptions enforced by the generator:
#    - The main field contains exactly one solid teal square and exactly 3 obstacle
#      colors, each present in the legend.
#    - Each next beam footprint intersects at most one obstacle color.
#    - The active legend colors realize the three distinct behaviors L/P/R, and
#      beam paths are acyclic under the generated layout.

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)

# Color palettes
LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [7, 9]
LEGEND_BG = 5
TEAL = 8

# Canvas dimensions
GRID_H_RANGE = (19, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 5, 9, 13, 17]
LEGEND_CELL_W = 3

# Teal square
TEAL_SZ2_PROB = 0.75
SQUARE_EDGE_GAP = 5
PASS_SIDE_EXTRA_GAP = 6

# Obstacles
OBS_THICKNESS_RANGE = (2, 4)
OBS_SINGLE_MARGIN = 6
OBS_BOTH_GAP = 2
PASS_GAP = 1

# Span selection
SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.4
PASS_SPAN_MAX_MULT = 9

MAX_RETRIES = 500


def _rot(d: int, turn: str) -> int:
    if turn == "L":
        return (d - 1) % 4
    if turn == "R":
        return (d + 1) % 4
    assert turn == "P"
    return d


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    """
    Parse the 5-cell ternary legend.

    Each non-empty cell contains one vertical colored bar:
      x=0 -> 'L'
      x=1 -> 'P'
      x=2 -> 'R'
    """
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 6, "legend separator columns must exist"

    behavior: dict[int, str] = {}
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 1:
            continue
        interior = legend[1 : legend_h - 1, a + 1 : b]
        colors = set(map(int, np.unique(interior)))
        colors.discard(bg)
        colors.discard(0)
        if not colors:
            continue
        assert len(colors) == 1, "each legend cell must encode exactly one color"
        col = int(next(iter(colors)))
        xs = sorted(set(map(int, np.where(interior == col)[1])))
        assert len(xs) == 1, "legend markers must be 1-column bars"
        x = xs[0]
        assert 0 <= x < interior.shape[1], "legend marker must lie inside its cell"
        if x == 0:
            behavior[col] = "L"
        elif x == interior.shape[1] - 1:
            behavior[col] = "R"
        else:
            behavior[col] = "P"
    assert len(behavior) == len(LEGEND_CELL_STARTS), "legend must encode five colors"
    return behavior


def _find_teal_square(g: np.ndarray, teal: int = TEAL) -> tuple[int, int, int]:
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


def _active_obstacle_colors(base: np.ndarray, bg: int) -> set[int]:
    colors = set(map(int, np.unique(base)))
    colors.discard(bg)
    colors.discard(TEAL)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four teal beams from a square, using a 5-entry ternary legend with distractors.

    Input structure:
    - Top band: grey (5) legend with five cells; each colored vertical marker sits at left, middle, or right to encode counterclockwise turn, pass-through, or clockwise turn.
    - Below: main grid with one bg color, one solid teal (8) square, and obstacle rectangles drawn in only three of the legend colors. The other two legend colors are decoys that never appear below.

    Rule:
    - Remove the legend from the output.
    - Launch one sz-by-sz teal beam in each cardinal direction from the square.
    - Paint teal only onto bg cells visited by a beam footprint.
    - When the next footprint intersects an obstacle of color C, first decide whether C is one of the colors actually present in the main grid, then consult its legend entry:
      left -> turn counterclockwise in place; middle -> move straight through; right -> turn clockwise in place.
    - Beams stop when the next sz-by-sz footprint would leave the grid.

    Key insight: The legend contains more color instructions than the main field uses, so the solver must first discard the two absent colors and then read the remaining three colors as a ternary left/middle/right behavior code.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must stay in the top band"

    legend_behavior = _parse_legend(g, legend_h)
    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    active_colors = _active_obstacle_colors(base, bg)
    assert len(active_colors) == 3, "main grid must contain exactly three obstacle colors"

    active_behavior = {c: legend_behavior[c] for c in active_colors}
    assert set(active_behavior.values()) == {"L", "P", "R"}, "active colors must realize the three ternary states"

    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    out = base.copy()

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < out.shape[0] and c + sz - 1 < out.shape[1]

    def paint_block(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = TEAL
        out[r : r + sz, c : c + sz] = blk

    def obstacle_color_at_next(nr: int, nc: int) -> int | None:
        blk = out[nr : nr + sz, nc : nc + sz]
        cols = set(map(int, np.unique(blk)))
        cols.discard(bg)
        cols.discard(TEAL)
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
                paint_block(r, c)
                continue

            assert oc in active_behavior, "every obstacle color in the main grid must be encoded by an active legend entry"
            step_behavior = active_behavior[oc]
            if step_behavior == "P":
                r, c = nr, nc
                paint_block(r, c)
            else:
                d = _rot(d, step_behavior)

    return out


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a span covering required intervals while avoiding excluded ones."""
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
    base: np.ndarray, behavior: dict[int, str], bg: int
) -> tuple[np.ndarray, list[list[tuple[int, str, int, int, int]]]]:
    """Simulate the four beams and return painted output plus compressed hit traces."""
    vals, counts = np.unique(base, return_counts=True)
    assert int(vals[counts.argmax()]) == bg

    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    out = base.copy()
    h, w = out.shape
    hits: list[list[tuple[int, str, int, int, int]]] = []

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
        beam_hits: list[tuple[int, str, int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        last_contact: tuple[int, str] | None = None
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
                last_contact = None
                r, c = nr, nc
                paint(r, c)
                continue

            step_behavior = behavior[obs]
            contact = (obs, step_behavior)
            if contact != last_contact:
                beam_hits.append((d, step_behavior, obs, nr, nc))
            last_contact = contact

            if step_behavior == "P":
                r, c = nr, nc
                paint(r, c)
            else:
                d = _rot(d, step_behavior)
                last_contact = None
        hits.append(beam_hits)

    return out, hits


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

    def lane_interval(s: int) -> tuple[int, int]:
        return lane[s], lane[s] + sz - 1

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


def _make_legend(legend_colors: list[int], behavior: dict[int, str]) -> np.ndarray:
    """Build the 6x24 legend band from 5 colors and their ternary markers."""
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors, strict=False):
        legend[1:5, start : start + LEGEND_CELL_W] = 0
        mark = behavior[color]
        if mark == "L":
            legend[1:5, start] = color
        elif mark == "R":
            legend[1:5, start + LEGEND_CELL_W - 1] = color
        else:
            legend[1:5, start + 1] = color
    return legend


def _pass_rect(
    pass_side: int,
    span: tuple[int, int],
    band_start: int,
    band_thickness: int,
) -> tuple[int, int, int, int]:
    s0, s1 = span
    if pass_side in (_N, _S):
        return band_start, band_start + band_thickness - 1, s0, s1
    return s0, s1, band_start, band_start + band_thickness - 1


def _choose_pass_band(
    pass_side: int,
    square_r: int,
    square_c: int,
    sz: int,
    thickness: int,
    edge_thickness: int,
    h: int,
    w: int,
) -> int | None:
    if pass_side == _N:
        lo = edge_thickness + PASS_GAP + 1
        hi = square_r - thickness - PASS_GAP - 1
    elif pass_side == _S:
        lo = square_r + sz + PASS_GAP
        hi = h - edge_thickness - thickness - PASS_GAP - 1
    elif pass_side == _W:
        lo = edge_thickness + PASS_GAP + 1
        hi = square_c - thickness - PASS_GAP - 1
    else:
        lo = square_c + sz + PASS_GAP
        hi = w - edge_thickness - thickness - PASS_GAP - 1
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3

    pass_side = random.randrange(4)
    hand = random.choice([-1, 1])
    chain_side = (pass_side + hand) % 4
    solo_side = (pass_side + 2) % 4
    missing_side = (pass_side - hand) % 4
    present = {pass_side, chain_side, solo_side}
    chains = {chain_side: pass_side}

    side_turn = {
        pass_side: "L" if _rot(pass_side, "L") == missing_side else "R",
        chain_side: "L" if _rot(chain_side, "L") == pass_side else "R",
        solo_side: "L" if _rot(solo_side, "L") == missing_side else "R",
    }
    assert set(side_turn.values()) == {"L", "R"}

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

    if pass_side == _N:
        min_r = max(min_r, thickness[_N] + PASS_SIDE_EXTRA_GAP)
    elif pass_side == _S:
        max_r = min(max_r, h - thickness[_S] - sz - PASS_SIDE_EXTRA_GAP)
    elif pass_side == _W:
        min_c = max(min_c, thickness[_W] + PASS_SIDE_EXTRA_GAP)
    else:
        max_c = min(max_c, w - thickness[_E] - sz - PASS_SIDE_EXTRA_GAP)

    if min_r > max_r or min_c > max_c:
        return None
    sr = random.randint(min_r, max_r)
    sc = random.randint(min_c, max_c)

    lane: dict[int, int] = {}
    if _N in present:
        lane[_N] = thickness[_N] + 1
    if _S in present:
        lane[_S] = h - thickness[_S] - sz
    if _W in present:
        lane[_W] = thickness[_W] + 1
    if _E in present:
        lane[_E] = w - thickness[_E] - sz

    required, excluded = _compute_span_constraints(
        present, side_turn, chains, sr, sc, sz, lane
    )

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

    pass_thickness = random.randint(*OBS_THICKNESS_RANGE)
    band_start = _choose_pass_band(
        pass_side, sr, sc, sz, pass_thickness, thickness[pass_side], h, w
    )
    if band_start is None:
        return None

    if pass_side in (_N, _S):
        axis_len = w
        req = [(sc, sc + sz - 1)]
        excl = [(lane[chain_side], lane[chain_side] + sz - 1)]
    else:
        axis_len = h
        req = [(sr, sr + sz - 1)]
        excl = [(lane[chain_side], lane[chain_side] + sz - 1)]
    pass_span = _choose_span(
        axis_len,
        req,
        excl,
        SPAN_MIN_MULT * sz,
        min(PASS_SPAN_MAX_MULT * sz, axis_len - 2),
    )
    if pass_span is None:
        return None

    legend_colors = LEGEND_COLOR_POOL.copy()
    random.shuffle(legend_colors)
    pass_color, left_color, right_color = random.sample(legend_colors, 3)
    distractor_colors = [c for c in legend_colors if c not in {pass_color, left_color, right_color}]
    turn_color = {"L": left_color, "R": right_color}

    behavior = {
        pass_color: "P",
        left_color: "L",
        right_color: "R",
        distractor_colors[0]: random.choice(["L", "P", "R"]),
        distractor_colors[1]: random.choice(["L", "P", "R"]),
    }
    side_color = {side: turn_color[side_turn[side]] for side in present}
    bg = random.choice(BG_COLOR_POOL)

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

    pr0, pr1, pc0, pc1 = _pass_rect(
        pass_side, pass_span, band_start, pass_thickness
    )
    region = occupied[pr0 : pr1 + 1, pc0 : pc1 + 1]
    if region.any():
        return None
    if not (pr1 < sr - 1 or pr0 > sr + sz or pc1 < sc - 1 or pc0 > sc + sz):
        return None
    occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
    base[pr0 : pr1 + 1, pc0 : pc1 + 1] = pass_color

    active_colors = _active_obstacle_colors(base, bg)
    if active_colors != {pass_color, left_color, right_color}:
        return None

    try:
        expected, hits = _simulate_paths(base, behavior, bg)
    except AssertionError:
        return None

    hit_by_beam = [tuple((mark, color) for _, mark, color, _, _ in beam) for beam in hits]
    want = {
        pass_side: (("P", pass_color), (side_turn[pass_side], turn_color[side_turn[pass_side]])),
        chain_side: (
            (side_turn[chain_side], turn_color[side_turn[chain_side]]),
            (side_turn[pass_side], turn_color[side_turn[pass_side]]),
        ),
        solo_side: ((side_turn[solo_side], turn_color[side_turn[solo_side]]),),
        missing_side: (),
    }
    for d in range(4):
        if hit_by_beam[d] != want[d]:
            return None

    legend = _make_legend(legend_colors, behavior)
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
