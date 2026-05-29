# Combo variant of e87109e9 (index 7)
# combo_spec: D3/2+D10/0
# Name: 底部图例加保留输出

import random

import numpy as np

from arc_utils import bbox_coords, components_mask

# Design analysis for combo D3/2 + D10/0
# a) Concretization and behavior trace
#    Original board:
#    - Rows 0..5 are the legend. `main()` detects them by searching for color 5,
#      parses four left/right vertical bars, crops those rows away, solves only
#      the lower playfield, and returns the cropped playfield.
#    Fused board:
#    - Rows H-6..H-1 are the legend instead. The playable region is now above it.
#      The output keeps the entire input extent: only the upper playfield is
#      modified by teal beam traces, while the bottom 6 legend rows remain
#      unchanged and visible.
#    Small trace:
#    - Original: a north-bound beam hits a red obstacle; the solver reads red's
#      bar from the top legend, turns, and the final output has no legend rows.
#    - Fused: the same hit is resolved from the bottom legend; the teal trace is
#      drawn in the upper playfield, and the bottom legend stays in the answer.
#    - Difference: both the lookup location and the returned frame changed.
# b) Complexity calibration
#    The routing problem stays intact: one teal source square, 3-4 edge
#    obstacles, 4 beams, and 4-5 total turns remain unchanged. This avoids
#    oversimplification. The only freedoms introduced are (1) legend placement
#    at the bottom and (2) keeping the solved board full-size. Both are binary
#    and visually obvious across a few examples.
# c) Implicit-assumption audit for the original main()
#    - `rows_with_legend == np.arange(legend_h)` assumed color 5 only appeared in
#      a top band. Broken by D3/2; the new code must instead require a bottom
#      band and reject any color-5 rows above it.
#    - `base = g[legend_h:, :]` assumed the playable region is below the legend.
#      Broken by D3/2; it becomes `base = g[:legend_start, :]`.
#    - `return out` on the cropped playfield assumed D10 original behavior.
#      Broken by D10/0; the solver must splice the solved playfield back into the
#      full grid and keep the legend rows intact.
#    - The obstacle scan still relies on the fact that teal trail cells can be
#      ignored as blockers. This remains valid because trajectories still paint
#      only color 8.
# d) Placement legality
#    The generator still constructs the playfield first, using the original
#    thickness/span constraints so obstacles stay disjoint, edge-attached, and at
#    least one cell away from the source square. The legend is appended as a
#    separate 6x24 band, so it cannot overlap any playfield object or beam.
# e) Visual observability
#    Humans can see the fusion from 2-4 examples because:
#    - the only grey-5 band is visibly at the bottom rather than the top;
#    - the answer preserves that band instead of deleting it;
#    - the teal traces appear only in the upper playfield, so the bottom band is
#      clearly a read-only control strip.
#    The generator validates all three properties.
# f) Visual cleanliness
#    The legend occupies its own dedicated band, so no beam can pass through it.
#    This is cleaner than overlaying instructions inside the playfield and keeps
#    the output easy to parse: upper area changes, bottom area does not.
# g) Color / inference discipline
#    No rule depends on numeric color ordering. Colors only identify obstacles and
#    their legend entries. The variant changes placement and output framing, not
#    the meaning of the colors themselves.
# h) New solver assumptions introduced here
#    - Color 5 is reserved for the bottom legend band and does not appear in the
#      playfield.
#    - The legend occupies the final contiguous 6 rows of the grid.
#    - The playable region above the legend contains exactly one solid teal
#      square and 3-4 solid obstacle rectangles whose colors are encoded by the
#      legend.
#    The generator explicitly enforces each assumption and oracle-checks them by
#    comparing `main(input)` with the simulated full-grid target.

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)

# Color palettes
LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
LEGEND_BG = 5
TEAL = 8

# Canvas dimensions
GRID_H_RANGE = (19, 24)  # playfield height only
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]

# Teal square
TEAL_SZ2_PROB = 0.75
SQUARE_EDGE_GAP = 5

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


def _parse_legend(legend: np.ndarray) -> dict[int, str]:
    """
    Legend is a dedicated 6-row band with background 5 and four vertical bars.
    Each bar's left/right position encodes turn direction:
      - left  -> 'L' (turn counterclockwise)
      - right -> 'R' (turn clockwise)
    """
    assert legend.size, "legend must be non-empty"
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    turn: dict[int, str] = {}
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        interior = legend[1 : legend.shape[0] - 1, a + 1 : b]
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


def _find_teal_square(g: np.ndarray, teal: int = TEAL) -> tuple[int, int, int]:
    """Return (r0, c0, size) for the unique solid teal square."""
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
    assert len(candidates) == 1, "playfield must contain exactly one solid teal square"
    return candidates[0]


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def _legend_start(grid: np.ndarray) -> int:
    rows_with_legend = np.where(np.any(grid == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    start = int(rows_with_legend.min())
    assert np.array_equal(
        rows_with_legend, np.arange(start, grid.shape[0])
    ), "legend color must be confined to one bottom band"
    assert grid.shape[0] - start == LEGEND_H, "legend must use the final 6 rows"
    return start


def _simulate_paths(
    base: np.ndarray, turns: dict[int, str], bg: int
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]]]:
    """Simulate four square beams on the playfield only."""
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

            obs = obstacle_color_at_next(nr, nc)
            if obs is None:
                r, c = nr, nc
                paint_block(r, c)
                continue

            assert obs in turns, "obstacle color must be encoded by legend"
            beam_hits.append((d, obs, nr, nc))
            d = _rot(d, turns[obs])
            assert len(beam_hits) <= 2, "generator only supports up to two hits per beam"
        hits.append(beam_hits)

    return out, hits


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    """Build the 6x24 legend band from 4 colors and their turn assignments."""
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a span covering all required intervals while avoiding all excluded intervals."""
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


def _rand_thickness(lo: int, hi: int) -> int | None:
    return random.randint(lo, hi) if hi >= lo else None


def _thickness_pair(
    axis: int, sz: int, a_present: bool, b_present: bool
) -> tuple[int | None, int | None] | None:
    """Compute obstacle thicknesses for the two ends of an axis."""
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
    """Return (r0, r1, c0, c1) for an edge-touching obstacle rectangle."""
    s0, s1 = span
    if side == _N:
        return 0, thickness - 1, s0, s1
    if side == _S:
        return h - thickness, h - 1, s0, s1
    if side == _W:
        return s0, s1, 0, thickness - 1
    return s0, s1, w - thickness, w - 1


def _select_chains(present: set[int], second_hits: int) -> dict[int, int] | None:
    """Pick non-overlapping (source -> adjacent target) pairs for multi-bounce beams."""
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
    """Assign L/R turn direction for each obstacle side."""
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
    """Build required/excluded intervals for each obstacle's span selection."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four teal beam-squares from the source, using a bottom legend and keeping the full grid.

    Input structure:
    - Upper region: the playable field with one solid teal square, a uniform background, and 3-4 solid edge-attached obstacle rectangles.
    - Bottom 6 rows: a grey legend band with four colored vertical bars; each bar's left/right placement inside its cell encodes counterclockwise vs clockwise turning for that color.

    Rule:
    - Read the turn mapping from the bottom legend, but do not erase that legend.
    - Launch one sz-by-sz teal beam from the square in each cardinal direction across the upper playfield only.
    - When the next beam footprint is background, paint that footprint teal and keep moving.
    - When the next footprint hits a single obstacle color, rotate according to the legend entry for that color and continue from the same position.
    - Beams stop at the playfield border; the final answer keeps the original full canvas, with the solved playfield above and the legend unchanged below.

    Key insight: The motion rule is still encoded by left-vs-right bars, but now the clue lives at the bottom and remains visible in the output, so the solver must separate "reference rows" from "paintable rows".
    """
    g = np.asarray(grid, dtype=int)
    legend_start = _legend_start(g)
    legend = g[legend_start:, :]
    base = g[:legend_start, :].copy()
    assert base.size, "playfield must remain above the legend"

    turns = _parse_legend(legend)
    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    solved_base, _hits = _simulate_paths(base, turns, bg)

    out = g.copy()
    out[:legend_start, :] = solved_base
    return out


def _gen_attempt() -> np.ndarray | None:
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

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    obstacle_colors = random.sample(legend_colors, obstacle_count)
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

    turns = {side_color[s]: side_turn[s] for s in present}
    for color in legend_colors:
        turns.setdefault(color, random.choice(["L", "R"]))

    try:
        expected_base, hits = _simulate_paths(base, turns, bg)
    except AssertionError:
        return None

    hit_counts = [len(beam) for beam in hits]
    hit_colors = {color for beam in hits for _, color, _, _ in beam}
    if sum(hit_counts) != target_hits or any(c > 2 for c in hit_counts):
        return None
    if set(side_color.values()) - hit_colors:
        return None

    legend = _make_legend(legend_colors, turns)
    inp = np.vstack([base, legend])
    expected = np.vstack([expected_base, legend])

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    # Non-degeneracy checks for the fused behavior:
    # - legend must truly be bottom-only;
    # - output must preserve the legend rows exactly;
    # - output must keep full input size.
    if np.any(inp[:h, :] == LEGEND_BG):
        return None
    if actual.shape != inp.shape:
        return None
    if not np.array_equal(actual[-LEGEND_H:, :], legend):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e87109e9 combo D3/2+D10/0 input")
