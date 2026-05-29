# Combo variant of e87109e9 (index 32)
# combo_spec: D2/3+D18/1
# Name: 源色匹配障碍加交替转弯

import random

import numpy as np

from arc_utils import bbox_coords, components_mask

# Design analysis for the combo variant
# source_task: e87109e9
#
# a) Concrete behavior / trace
#    - The input still has the original two-zone layout: a 6x24 grey legend strip
#      on top and a 19-24 x 24 main board below.
#    - The crucial visual change is that the source is no longer fixed teal. It is
#      a 2x2 or 3x3 interior square whose color exactly matches one edge obstacle.
#    - Example: suppose the source square is color 4 and the only color-4 obstacle
#      is the north bar. The solver must launch a single beam northward first.
#      At the first collision with that north obstacle, the beam turns clockwise to
#      the east. At the second collision, with the east obstacle, it turns
#      counterclockwise back north and exits. In the original task, all four beams
#      launched regardless of color and each collision used the legend; here the
#      color match selects the unique starting side, then collision order controls
#      later turns.
#
# b) Complexity calibration
#    - If the generator allowed only one collision, the D2 change would collapse
#      into "always turn clockwise once", which is too weak.
#    - Therefore every generated active beam is forced to make exactly two hits:
#      first on the source-matched obstacle, then on its clockwise-adjacent
#      obstacle. This makes both the color match and the alternating CW/CCW rule
#      visible in a single trace.
#    - The board still contains all four edge obstacles so the scene keeps the
#      original richness and wrong starting sides remain plausible distractors.
#
# c) Hidden assumptions from the original main()
#    - Original code assumes the source/beam color (8) never equals any obstacle
#      color, so `obstacle_at_next()` can safely ignore painted beam cells by
#      discarding that color. That assumption breaks here because the first obstacle
#      has the same color as the beam. The fix is to detect collisions against the
#      immutable base grid, not against the incrementally painted output grid.
#    - Original code also hard-codes four launches. That is no longer valid; the
#      solver must infer one start side from the same-color obstacle.
#    - Original cycle state `(r, c, d)` is no longer sufficient once turn behavior
#      depends on collision parity. The new state must include hit-count parity.
#    - Parsing legend left/right into operational turns is no longer valid. The
#      legend remains a removable color panel only.
#
# d) Placement legality
#    - Obstacles remain solid edge-touching rectangles, one per side, with no
#      overlap and with at least one-cell clearance from the source square.
#    - All four sides are present so there is exactly one obstacle whose color
#      matches the source and exactly one clockwise-adjacent second-hit target.
#    - Span constraints enforce:
#        1. the matched obstacle covers the direct lane from the source;
#        2. the clockwise-adjacent obstacle covers the redirected lane after hit 1;
#        3. the matched obstacle excludes the return lane after hit 2 so the path
#           exits instead of bouncing forever.
#
# e) Visual observability
#    - Humans can see the source square and exactly one edge obstacle sharing a
#      color. That repeated color is the visible clue for the starting side.
#    - The legend bars still show left/right positions, but generation explicitly
#      rejects samples where those bar positions accidentally explain the active
#      two-hit path. So the alternating-turn rule is operational rather than
#      decorative.
#    - Generation also rejects samples where starting from a wrong side would
#      produce the same final output, so the color-match cue genuinely matters.
#
# f) Visual cleanliness
#    - The variant keeps the same sparse board grammar as the source task:
#      one source square, four large edge bars, and one beam path.
#    - Only one beam is painted, which actually makes the output easier to parse
#      than the original four-beam union while preserving the task's geometric
#      character.
#
# g) Change inferability audit
#    - No hidden ordering over color ids is used. The only color semantics is an
#      explicit visible equality relation: source color = one obstacle color.
#    - There are only four obstacle/legend colors per board, and exactly one of
#      them repeats at the source. That is a small enough cue set for ARC-style
#      inference from 2-4 examples.

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)

# Color palettes
LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
LEGEND_BG = 5

# Canvas dimensions
GRID_H_RANGE = (19, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]

# Source square
SRC_SZ2_PROB = 0.75
SQUARE_EDGE_GAP = 5

# Obstacles
OBS_THICKNESS_RANGE = (2, 4)
OBS_SINGLE_MARGIN = 6
OBS_BOTH_GAP = 2

# Span selection
SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.4

# Retry budget
MAX_RETRIES = 400


def _legend_colors(grid: np.ndarray, legend_h: int) -> list[int]:
    """Read legend colors for structure validation only; bar position is decorative."""
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


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def _find_source_square(g: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Return (r0, c0, size, color) for the unique interior solid square source."""
    candidates: list[tuple[int, int, int, int]] = []
    h, w = g.shape
    for cells in components_mask(g != bg, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh != ww:
            continue
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        block = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(map(int, np.unique(block)))
        colors.discard(bg)
        if len(colors) != 1:
            continue
        color = int(next(iter(colors)))
        size = hh
        if int(np.sum(block == color)) != size * size:
            continue
        candidates.append((int(r0), int(c0), int(size), color))
    assert len(candidates) == 1, "main grid must contain exactly one solid interior source square"
    return candidates[0]


def _infer_start_side(
    base: np.ndarray, src: tuple[int, int, int, int]
) -> int:
    """Find the unique edge obstacle sharing the source color and return its side."""
    sr, sc, sz, src_color = src
    src_bbox = (sr, sc, sr + sz - 1, sc + sz - 1)
    matches: list[int] = []
    h, w = base.shape
    for cells in components_mask(base == src_color, connectivity=4):
        r0, c0, r1, c1 = map(int, bbox_coords(cells))
        if (r0, c0, r1, c1) == src_bbox:
            continue
        sides: list[int] = []
        if r0 == 0:
            sides.append(_N)
        if c1 == w - 1:
            sides.append(_E)
        if r1 == h - 1:
            sides.append(_S)
        if c0 == 0:
            sides.append(_W)
        assert len(sides) == 1, "matched obstacle must touch exactly one edge"
        matches.append(sides[0])
    assert len(matches) == 1, "source color must match exactly one obstacle"
    return matches[0]


def _simulate_single_beam(
    base: np.ndarray,
    bg: int,
    src: tuple[int, int, int, int],
    start_side: int,
    turn_mode: str = "alternate",
    legend_turns: dict[int, str] | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]:
    """Simulate the single active beam from the inferred or supplied start side."""
    sr, sc, sz, beam_color = src
    out = base.copy()
    h, w = out.shape
    src_mask = np.zeros_like(base, dtype=bool)
    src_mask[sr : sr + sz, sc : sc + sz] = True

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint_block(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = beam_color
        out[r : r + sz, c : c + sz] = blk

    def obstacle_color_at_next(nr: int, nc: int) -> int | None:
        blk = base[nr : nr + sz, nc : nc + sz]
        src_blk = src_mask[nr : nr + sz, nc : nc + sz]
        mask = (blk != bg) & (~src_blk)
        if not np.any(mask):
            return None
        colors = set(map(int, np.unique(blk[mask])))
        assert len(colors) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(colors)))

    r, c, d = sr, sc, start_side
    hit_parity = 0
    hits: list[tuple[int, int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    while True:
        state = (r, c, d, hit_parity)
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

        hits.append((hit_parity + 1, d, obs, nr, nc))
        if turn_mode == "alternate":
            d = _rot_cw(d) if hit_parity == 0 else _rot_ccw(d)
            hit_parity ^= 1
        else:
            assert legend_turns is not None and obs in legend_turns
            d = _rot(d, legend_turns[obs])

    return out, hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace one source-colored beam whose launch side is chosen by a color match, then alternate turns at obstacles.

    Input structure:
    - The top band is still a grey legend strip with four colored bars, but its left/right placement is now decorative rather than operational.
    - The main board below has one solid interior square source and four large edge-touching obstacle bars.
    - Exactly one obstacle shares the source square's color.

    Rule:
    - Remove the legend band from the output.
    - Find the unique interior square source and identify the edge obstacle with the same color.
    - Launch a single moving copy of the source square toward that matched obstacle.
    - As the beam moves, it paints only background cells in the source color.
    - On the 1st, 3rd, 5th, ... obstacle collision it turns clockwise; on the 2nd, 4th, 6th, ... collision it turns counterclockwise. Obstacle color and legend bar position do not affect turning.
    - Stop when the next step would leave the main grid.

    Key insight: The repeated source/obstacle color selects the unique initial side, while later turns depend only on collision order parity.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"
    _legend_colors(g, legend_h)

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    src = _find_source_square(base, bg)
    start_side = _infer_start_side(base, src)
    out, _hits = _simulate_single_beam(base, bg, src, start_side, turn_mode="alternate")
    return out


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


def _random_legend_dirs(colors: list[int], active_colors: tuple[int, int]) -> dict[int, str]:
    """Assign decorative legend bars while forbidding the true active turn sequence R then L."""
    first_color, second_color = active_colors
    while True:
        dirs = {color: random.choice(["L", "R"]) for color in colors}
        if len(set(dirs.values())) < 2:
            continue
        if dirs[first_color] == "R" and dirs[second_color] == "L":
            continue
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


def _compute_span_constraints(
    sr: int,
    sc: int,
    sz: int,
    start_side: int,
    target_side: int,
    lane: dict[int, int],
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    """Build required/excluded intervals for the active two-hit beam and the distractor bars."""
    present = {_N, _E, _S, _W}
    required: dict[int, list[tuple[int, int]]] = {s: [] for s in present}
    excluded: dict[int, list[tuple[int, int]]] = {s: [] for s in present}

    for side in present:
        base_coord = sc if side in (_N, _S) else sr
        required[side].append((base_coord, base_coord + sz - 1))

    def lane_interval(side: int) -> tuple[int, int]:
        return lane[side], lane[side] + sz - 1

    required[target_side].append(lane_interval(start_side))
    excluded[start_side].append(lane_interval(target_side))
    return required, excluded


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < SRC_SZ2_PROB else 3
    present = {_N, _E, _S, _W}

    ns = _thickness_pair(h, sz, True, True)
    ew = _thickness_pair(w, sz, True, True)
    if ns is None or ew is None:
        return None
    thickness = {s: t for s, t in zip((_N, _S, _W, _E), (*ns, *ew)) if t is not None}

    min_r, max_r = SQUARE_EDGE_GAP, h - sz - SQUARE_EDGE_GAP
    min_c, max_c = SQUARE_EDGE_GAP, w - sz - SQUARE_EDGE_GAP
    min_r = max(min_r, thickness[_N] + 1)
    max_r = min(max_r, h - thickness[_S] - sz - 1)
    min_c = max(min_c, thickness[_W] + 1)
    max_c = min(max_c, w - thickness[_E] - sz - 1)
    if min_r > max_r or min_c > max_c:
        return None
    sr, sc = random.randint(min_r, max_r), random.randint(min_c, max_c)

    lane = {
        _N: thickness[_N] + 1,
        _S: h - thickness[_S] - sz,
        _W: thickness[_W] + 1,
        _E: w - thickness[_E] - sz,
    }

    start_side = random.choice([_N, _E, _S, _W])
    target_side = _rot_cw(start_side)
    required, excluded = _compute_span_constraints(sr, sc, sz, start_side, target_side, lane)

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
    side_color = dict(zip([_N, _E, _S, _W], random.sample(legend_colors, 4)))
    src_color = side_color[start_side]
    bg_choices = [c for c in BG_COLOR_POOL if c not in legend_colors]
    if not bg_choices:
        return None
    bg = random.choice(bg_choices)

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = src_color
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

    src = (sr, sc, sz, src_color)
    try:
        expected, hits = _simulate_single_beam(base, bg, src, start_side, turn_mode="alternate")
    except AssertionError:
        return None

    if len(hits) != 2:
        return None
    if hits[0][1] != start_side:
        return None
    if hits[0][2] != src_color or hits[1][2] != side_color[target_side]:
        return None

    legend_dirs = _random_legend_dirs(legend_colors, (src_color, side_color[target_side]))
    try:
        legend_based, _ = _simulate_single_beam(
            base, bg, src, start_side, turn_mode="legend", legend_turns=legend_dirs
        )
    except AssertionError:
        return None
    if np.array_equal(legend_based, expected):
        return None

    for wrong_side in [d for d in (_N, _E, _S, _W) if d != start_side]:
        try:
            wrong_out, _ = _simulate_single_beam(base, bg, src, wrong_side, turn_mode="alternate")
        except AssertionError:
            continue
        if np.array_equal(wrong_out, expected):
            return None

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
    raise RuntimeError("failed to generate e87109e9 combo input")
