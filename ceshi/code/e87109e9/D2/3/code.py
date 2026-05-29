# Variant of e87109e9
# Dimension: D2 — 转弯规则 (path)
# Change: 遇到障碍物时根据图例查找该颜色——图例中颜色条在左侧表示逆时针转弯，在右侧表示顺时针转弯... → 交替转弯（忽略图例颜色）

import random

import numpy as np

from arc_utils import bbox_coords, components_mask

# Design analysis for the D2 variant
# source_task: none
#
# a) Concrete behavior / trace
#    - The input still has the same two regions as the original task:
#      a 6x24 grey legend band on top and a 19-24 x 24 main grid below.
#    - The main grid still contains one solid teal square and 3-4 edge-touching
#      obstacle rectangles in colors drawn from the legend palette.
#    - The output still removes the legend and paints four sz-by-sz teal beams.
#    - The changed step is only the turning rule at collisions. Example:
#      north beam hits the north obstacle first, so it must turn clockwise to
#      the east. If it then hits the east obstacle, that second hit must turn
#      counterclockwise back toward the north. In the original task the two
#      turns would both be controlled by obstacle colors; here they depend only
#      on hit index 1, 2, 3, ...
#
# b) Complexity calibration
#    - If every beam had at most one collision, the variant would collapse into
#      "always turn clockwise once", which is visible but weak.
#    - To make the alternation observable, generation forces at least one
#      second-hit chain. Therefore some obstacle is hit twice: once as another
#      beam's second collision and once as its own beam's first collision.
#      That obstacle must behave differently in the two contexts, which a
#      color-fixed legend cannot express.
#
# c) Hidden assumptions from the original main()
#    - Parsing the legend into color -> turn is no longer valid; the legend is
#      kept only as a removable top band.
#    - State tracking can no longer use only (r, c, d). Under alternating
#      turns, revisiting the same location/direction with opposite parity is a
#      different future state, so cycle detection must include hit-count parity.
#    - Assertions that every obstacle color appears in the legend or maps to a
#      single turn direction are no longer solver requirements.
#
# d) Placement legality
#    - Obstacles remain axis-aligned edge-touching rectangles with no overlap.
#    - Each obstacle must cover the source beam lane for its own side.
#    - A selected chain src -> cw(src) requires the target obstacle to cover the
#      redirected lane coming from src.
#    - Non-chain adjacent sides must exclude that redirected lane so accidental
#      second hits do not appear.
#    - After a second hit, the beam turns back outward; the source obstacle must
#      exclude the target lane to prevent an unintended third hit.
#
# e) Visual observability
#    - The legend bars are still present, but their left/right positions are
#      randomized independently of the true path behavior.
#    - Because at least one obstacle participates in both a first-hit and a
#      second-hit event, there is no consistent color -> turn explanation.
#      A human must infer that turn order, not obstacle identity, matters.
#
# f) Visual cleanliness
#    - The variant keeps the original clean layout: one legend strip, one main
#      board, large edge-touching obstacle bars, and wide teal paths.
#    - Generation uses the same spacing style as the source task so beams and
#      obstacles remain easy to separate visually.
#
# g) Color-rule audit
#    - No color ids are used for ordering, ranking, or turn parity.
#    - Colors only mark obstacle regions and legend decoration. Path logic is
#      purely geometric plus per-beam collision-count parity.

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)


def _legend_colors(grid: np.ndarray, legend_h: int) -> list[int]:
    """Read legend cell colors for structure validation only; bar position is ignored."""
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])
    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    colors: list[int] = []
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        interior = legend[1 : legend_h - 1, a + 1 : b]
        cell_colors = set(map(int, np.unique(interior)))
        cell_colors.discard(bg)
        cell_colors.discard(0)
        assert len(cell_colors) == 1, "each legend cell must contain exactly one colored bar"
        colors.append(int(next(iter(cell_colors))))
    assert len(colors) == 4, "legend must encode four colors"
    return colors


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace four teal beams from a square, alternating turn direction at collisions.

    Input structure:
    - The top band is still a grey legend strip with four colored bars, but its left/right bar placement is now decorative rather than operational.
    - The rows below contain a uniform-background board with one solid teal square and several colored obstacle rectangles.

    Rule:
    - Remove the entire legend band from the output.
    - Launch one moving teal square in each cardinal direction from the source square.
    - As each beam advances, it paints teal onto background cells only.
    - When the next step would overlap an obstacle, the beam turns clockwise on its 1st, 3rd, 5th, ... collisions and counterclockwise on its 2nd, 4th, 6th, ... collisions, regardless of obstacle color.
    - A beam stops once its next step would leave the main grid.

    Key insight: Turn direction is no longer encoded by legend cells or obstacle colors; it is determined solely by each beam's collision-count parity.
    """
    g = np.asarray(grid, dtype=int)

    legend_bg = 5
    rows_with_legend = np.where(np.any(g == legend_bg, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"
    _legend_colors(g, legend_h)

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    teal = 8
    sr, sc, sz = _find_teal_square(base, teal=teal)

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
        assert len(cols) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(cols)))

    for d0 in range(4):
        r, c, d = sr, sc, d0
        hit_parity = 0
        seen: set[tuple[int, int, int, int]] = set()
        while True:
            state = (r, c, d, hit_parity)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            if obstacle_color_at_next(nr, nc) is None:
                r, c = nr, nc
                paint_block(r, c)
                continue

            d = _rot_cw(d) if hit_parity == 0 else _rot_ccw(d)
            hit_parity ^= 1

    return out


# --------------- generator (o5x) ---------------

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


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a span covering all required intervals while missing all excluded intervals."""
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
    base: np.ndarray, bg: int
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int, int]]]]:
    """Simulate the alternating-turn beams and return the painted grid plus hit logs."""
    vals, counts = np.unique(base, return_counts=True)
    assert int(vals[counts.argmax()]) == bg

    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    out = base.copy()
    h, w = out.shape
    hits: list[list[tuple[int, int, int, int, int]]] = []

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
        assert len(colors) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(colors)))

    for d0 in range(4):
        r, c, d = sr, sc, d0
        hit_parity = 0
        beam_hits: list[tuple[int, int, int, int, int]] = []
        seen: set[tuple[int, int, int, int]] = set()
        while True:
            state = (r, c, d, hit_parity)
            assert state not in seen, "beam path must not cycle"
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
            beam_hits.append((hit_parity + 1, d, obs, nr, nc))
            d = _rot_cw(d) if hit_parity == 0 else _rot_ccw(d)
            hit_parity ^= 1
            assert len(beam_hits) <= 2, "generator should create at most two hits per beam"
        hits.append(beam_hits)

    return out, hits


def _random_legend_dirs(colors: list[int]) -> dict[int, str]:
    """Assign decorative L/R legend bars without encoding the real rule."""
    while True:
        dirs = {color: random.choice(["L", "R"]) for color in colors}
        if len(set(dirs.values())) == 2:
            return dirs


def _make_legend(legend_colors: list[int], legend_dirs: dict[int, str]) -> np.ndarray:
    """Build the unchanged-looking 6x24 legend band."""
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if legend_dirs[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


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
    """Pick non-overlapping clockwise second-hit chains src -> cw(src)."""
    candidates = [side for side in present if _rot_cw(side) in present]
    random.shuffle(candidates)
    chains: dict[int, int] = {}
    used: set[int] = set()
    for side in candidates:
        if len(chains) == second_hits:
            break
        tgt = _rot_cw(side)
        if side in used or tgt in used:
            continue
        chains[side] = tgt
        used.update((side, tgt))
    return chains if len(chains) == second_hits else None


def _compute_span_constraints(
    present: set[int],
    chains: dict[int, int],
    sr: int,
    sc: int,
    sz: int,
    lane: dict[int, int],
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    """Build required/excluded intervals for the alternating-turn geometry."""
    required: dict[int, list[tuple[int, int]]] = {s: [] for s in present}
    excluded: dict[int, list[tuple[int, int]]] = {s: [] for s in present}

    for side in present:
        base_coord = sc if side in (_N, _S) else sr
        required[side].append((base_coord, base_coord + sz - 1))

    def lane_interval(side: int) -> tuple[int, int]:
        return lane[side], lane[side] + sz - 1

    for src, tgt in chains.items():
        required[tgt].append(lane_interval(src))
        excluded[src].append(lane_interval(tgt))

    for side in present:
        tgt = _rot_cw(side)
        if tgt in present and chains.get(side) != tgt:
            excluded[tgt].append(lane_interval(side))

    return required, excluded


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    obstacle_count = 4 if random.random() < OBS_COUNT_4_PROB else 3
    present = set(random.sample([_N, _E, _S, _W], obstacle_count))
    second_hits = random.choice([1, 2]) if obstacle_count == 4 else 1

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

    required, excluded = _compute_span_constraints(present, chains, sr, sc, sz, lane)

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

    try:
        expected, hits = _simulate_paths(base, bg)
    except AssertionError:
        return None

    hit_counts = [len(beam) for beam in hits]
    total_hits = sum(hit_counts)
    if total_hits != obstacle_count + second_hits:
        return None
    if any(c > 2 for c in hit_counts):
        return None
    if sum(c == 2 for c in hit_counts) != second_hits:
        return None

    dual_role_obstacles = {
        color
        for beam in hits
        for hit_idx, _d, color, _r, _c in beam
        if hit_idx == 2
    }
    if not dual_role_obstacles:
        return None

    legend_dirs = _random_legend_dirs(legend_colors)
    legend = _make_legend(legend_colors, legend_dirs)
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
    raise RuntimeError("failed to generate e87109e9 D2 variant input")
