# Variant of e87109e9
# Dimension: D3 — 图例编码方式 (symbol_instructions)
# Change: 顶部灰色区域包含4个图例条目，每个为4行高的单色竖条在4格宽空间的左侧或右侧——左侧=逆时针，右侧=顺时针... → 无集中图例——规则由障碍物邻近像素编码

import random

import numpy as np

from arc_utils import bbox_coords, components_mask


# Design analysis for the D3 variant
# a) Concretization and behavior trace
#    The input is now a single board with no cropped legend band. Each colored
#    obstacle rectangle keeps its original role, but gains one grey marker pixel
#    on its inward-facing side. For a horizontal obstacle (top/bottom edge), the
#    marker is placed at the left or right endpoint of that inward side. For a
#    vertical obstacle (left/right edge), the marker is placed at the top or
#    bottom endpoint of that inward side. Endpoint-start means CCW ('L'),
#    endpoint-end means CW ('R'). Small trace:
#    - Original: beam hits a red obstacle, solver looks up red in the top legend,
#      finds the bar on the left, then turns CCW.
#    - Variant: beam hits the same red obstacle, solver inspects the grey pixel
#      adjacent to that obstacle, sees it on the start endpoint of the inner
#      edge, then turns CCW.
#    - Difference: the control information moved from one global symbol bank to
#      per-obstacle local markers.
# b) Complexity calibration
#    The variant keeps 3-4 obstacles, distinct obstacle colors, 4-way beams, and
#    4-5 total obstacle hits. This preserves the original routing complexity.
#    The only new freedom is the local marker endpoint choice, which is binary
#    per obstacle and visually repeated enough to infer.
# c) Implicit-assumption audit for main()
#    - The original code assumed a dedicated top grey legend band existed and
#      could be cropped by searching for grey rows. That is now false; grey is a
#      sparse local marker color inside the main board.
#    - The original obstacle-turn lookup used a color->turn table built from the
#      legend. That table is now built by scanning obstacle components and their
#      neighboring marker pixels.
#    - The original output started from the cropped main board. The new output
#      must instead start from the full input board after erasing marker pixels
#      back to background.
# d) Placement legality
#    The generator still lays out solid edge obstacles first, then chooses one
#    marker cell per obstacle on the obstacle's inward side. It rejects any case
#    where a marker would overlap another obstacle, the teal square, another
#    marker, or leave the board. This guarantees the solver can find exactly one
#    marker for each obstacle.
# e) Visual observability
#    Humans can now compare input/output and notice two things immediately:
#    there is no top legend, and each obstacle carries a single local grey clue.
#    To avoid degenerate samples where the clue type looks constant, the
#    generator requires every sample to contain both CCW and CW markers.
# f) Visual cleanliness
#    Markers are single isolated pixels on obstacle inner edges, so they do not
#    stack with other elements. The output removes them, avoiding mixed symbol
#    clutter in the final painted board.
# g) Color-inference discipline
#    No rule depends on numeric color ordering. Obstacle colors stay distinct but
#    semantically neutral; only the marker position relative to its obstacle
#    determines the turn.
# h) New solver assumptions introduced by the variant
#    - Marker color 5 is reserved for local instruction pixels and never used as
#      background or as an obstacle color.
#    - Every obstacle is a solid non-square rectangle whose long axis reveals
#      whether it is a top/bottom bar or a left/right bar.
#    - Each obstacle has exactly one marker on its inward side and that marker
#      sits at one of the two endpoints of that side.
#    The generator explicitly enforces all three assumptions.

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)

OBSTACLE_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
MARKER = 5
TEAL = 8

GRID_H_RANGE = (19, 24)
GRID_W = 24

TEAL_SZ2_PROB = 0.75
SQUARE_EDGE_GAP = 5

OBS_COUNT_4_PROB = 0.65
OBS_THICKNESS_RANGE = (2, 4)
OBS_SINGLE_MARGIN = 6
OBS_BOTH_GAP = 2

SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.4

MAX_RETRIES = 300


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
    assert len(candidates) == 1, "main grid must contain exactly one solid teal square"
    return candidates[0]


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def _obstacle_side_from_bbox(
    r0: int, c0: int, r1: int, c1: int, shape: tuple[int, int]
) -> int:
    """Infer which edge owns the rectangle from its aspect ratio and touched edge."""
    h, w = shape
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    if ww > hh:
        if r0 == 0:
            return _N
        if r1 == h - 1:
            return _S
    else:
        if c0 == 0:
            return _W
        if c1 == w - 1:
            return _E
    raise AssertionError("obstacle must be an edge-attached non-square rectangle")


def _marker_cell_from_rect(side: int, rect: tuple[int, int, int, int], turn: str) -> tuple[int, int]:
    """Return the unique marker cell for a rectangle and turn."""
    r0, r1, c0, c1 = rect
    if side == _N:
        return r1 + 1, c0 if turn == "L" else c1
    if side == _S:
        return r0 - 1, c0 if turn == "L" else c1
    if side == _W:
        return r0 if turn == "L" else r1, c1 + 1
    return r0 if turn == "L" else r1, c0 - 1


def _parse_local_turns(grid: np.ndarray, bg: int) -> dict[int, str]:
    """Read obstacle turns from adjacent marker pixels."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    obstacle_mask = (g != bg) & (g != TEAL) & (g != MARKER)
    turns: dict[int, str] = {}

    for cells in components_mask(obstacle_mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        color = int(g[cells[0, 0], cells[0, 1]])
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == color), "obstacle must be a solid rectangle"

        side = _obstacle_side_from_bbox(r0, c0, r1, c1, (h, w))
        if side == _N:
            assert r1 + 1 < h
            line = g[r1 + 1, c0 : c1 + 1]
            marker_positions = np.where(line == MARKER)[0]
            assert marker_positions.size == 1, "north obstacle must have one local marker"
            x = int(marker_positions[0])
            assert x in (0, c1 - c0), "north marker must sit at one endpoint"
            turn = "L" if x == 0 else "R"
        elif side == _S:
            assert r0 - 1 >= 0
            line = g[r0 - 1, c0 : c1 + 1]
            marker_positions = np.where(line == MARKER)[0]
            assert marker_positions.size == 1, "south obstacle must have one local marker"
            x = int(marker_positions[0])
            assert x in (0, c1 - c0), "south marker must sit at one endpoint"
            turn = "L" if x == 0 else "R"
        elif side == _W:
            assert c1 + 1 < w
            line = g[r0 : r1 + 1, c1 + 1]
            marker_positions = np.where(line == MARKER)[0]
            assert marker_positions.size == 1, "west obstacle must have one local marker"
            x = int(marker_positions[0])
            assert x in (0, r1 - r0), "west marker must sit at one endpoint"
            turn = "L" if x == 0 else "R"
        else:
            assert c0 - 1 >= 0
            line = g[r0 : r1 + 1, c0 - 1]
            marker_positions = np.where(line == MARKER)[0]
            assert marker_positions.size == 1, "east obstacle must have one local marker"
            x = int(marker_positions[0])
            assert x in (0, r1 - r0), "east marker must sit at one endpoint"
            turn = "L" if x == 0 else "R"

        turns[color] = turn

    assert 3 <= len(turns) <= 4, "input must contain three or four obstacles"
    return turns


def _clean_base(grid: np.ndarray, bg: int) -> np.ndarray:
    base = np.asarray(grid, dtype=int).copy()
    base[base == MARKER] = bg
    return base


def _simulate_paths(
    base: np.ndarray, turns: dict[int, str], bg: int
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]]]:
    """Simulate the four beams on a clean base grid."""
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

            assert oc in turns, "every obstacle color must have a local turn marker"
            beam_hits.append((d, oc, nr, nc))
            d = _rot(d, turns[oc])

        hits.append(beam_hits)

    return out, hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Send four teal square-beams through a board whose turn instructions live next to the obstacles.

    Input structure:
    - A single main grid with one solid teal source square, 3-4 solid edge obstacles, and one grey marker pixel next to each obstacle.
    - Each marker sits on the obstacle's inward-facing side at one of two endpoint slots, locally encoding the obstacle's turn direction.

    Rule:
    - The grey marker pixels are instructional only and disappear in the output.
    - Launch one sz-by-sz teal beam from the source square in each cardinal direction.
    - A beam paints teal only onto background cells.
    - When the next beam footprint meets an obstacle, inspect that obstacle's adjacent marker: start-endpoint means turn counterclockwise, far-endpoint means turn clockwise. Then continue in the new direction.
    - Beams stop when the next footprint would leave the board.

    Key insight:
    - There is no detached legend anymore; each obstacle carries its own local one-pixel clue, so the solver must decode turns obstacle-by-obstacle before tracing the beams.
    """
    g = np.asarray(grid, dtype=int)
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[counts.argmax()])
    assert bg != MARKER, "marker color cannot be the board background"

    turns = _parse_local_turns(g, bg)
    base = _clean_base(g, bg)
    out, _hits = _simulate_paths(base, turns, bg)
    return out


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a span covering all required intervals and avoiding all excluded ones."""
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
    """Compute obstacle thicknesses for the two ends of one axis."""
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
    """Assign CCW/CW turn direction for each obstacle side."""
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
    """Build required and forbidden intervals for each obstacle span."""
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
    if len(set(side_turn.values())) < 2:
        return None

    required, excluded = _compute_span_constraints(present, side_turn, chains, sr, sc, sz, lane)

    spans: dict[int, tuple[int, int]] = {}
    rects: dict[int, tuple[int, int, int, int]] = {}
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
        rects[side] = _obstacle_rect(side, thickness[side], span, h, w)

    obstacle_colors = random.sample(OBSTACLE_COLOR_POOL, obstacle_count)
    side_color = dict(zip(sorted(present), obstacle_colors))
    bg = random.choice([c for c in BG_COLOR_POOL if c not in obstacle_colors and c != MARKER])

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL
    occupied = np.zeros((h, w), dtype=bool)
    for side in present:
        r0, r1, c0, c1 = rects[side]
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

    inp = base.copy()
    for side in present:
        mr, mc = _marker_cell_from_rect(side, rects[side], side_turn[side])
        if not (0 <= mr < h and 0 <= mc < w):
            return None
        if inp[mr, mc] != bg:
            return None
        inp[mr, mc] = MARKER

    turns = {side_color[s]: side_turn[s] for s in present}

    try:
        expected, hits = _simulate_paths(base, turns, bg)
    except AssertionError:
        return None

    hit_counts = [len(beam) for beam in hits]
    hit_colors = {color for beam in hits for _, color, _, _ in beam}
    if sum(hit_counts) != target_hits or any(c > 2 for c in hit_counts):
        return None
    if set(side_color.values()) - hit_colors:
        return None

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
    raise RuntimeError("failed to generate e87109e9 D3 variant input")
