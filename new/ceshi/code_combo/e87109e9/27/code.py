# Combo variant of e87109e9 (index 27)
# combo_spec: D4/2+D13/1
# Name: 图例颜色循环涂色加碰撞次数编码

"""
design_analysis:
  a_specific_behavior:
    fused_rule:
      - keep the original 6x24 legend strip, the square teal source, and the
        edge-touching rectangular obstacles
      - legend cell order now matters: the four visible legend colors define a
        repeating trail cycle A -> B -> C -> D -> A
      - legend bar side still encodes turn direction for that obstacle color
      - legend bar height additionally encodes that color's total turn budget
      - every beam starts on segment 1 with color A
      - every new obstacle encounter advances the segment color by one step,
        regardless of whether that encounter turns or passes through
      - if the hit color still has quota left, spend one quota unit and turn
      - if the quota is exhausted, enter the obstacle and continue straight
    trace_example:
      setup:
        - legend order is [red, yellow, blue, green]
        - blue has height 1 and turns clockwise
        - a beam first reaches a red obstacle, later a blue obstacle, and a
          different beam reaches blue again after blue has been exhausted
      original_task:
        - all visible trail cells stay teal
        - both blue contacts turn
      fused_task:
        - source -> first obstacle paints red
        - after first encounter, next visible stretch paints yellow
        - first blue contact spends blue's only turn and the next stretch
          paints blue
        - later blue contact no longer turns; the beam enters blue, exits on
          the far side, and the post-obstacle stretch paints green
      visible_difference:
        - the legend must be read in two ways at once: left-to-right order for
          trail colors and per-color height for how many turns are allowed
  b_complexity_calibration:
    - keep the obstacle geometry sparse and edge-touching as in the source task
    - limit budgets to 1..4 because the legend already has exactly four usable
      interior rows
    - require at least one repeated obstacle encounter so the quota dimension
      changes the path
    - require at least the first three cycle colors to appear on trails so the
      color-order dimension is not a cosmetic one-color recolor
  c_implicit_assumptions_review:
    original_assumptions_that_break:
      - parsing only color -> turn is no longer enough; solver needs legend
        order and per-color quota too
      - reading collisions from the mutable output is unsafe because trail
        colors now reuse legend colors; collision checks must read immutable base
      - every obstacle contact no longer implies a turn because exhausted colors
        are penetrated
      - a cycle guard on only (row, col, dir) is too weak once segment index
        and global quota usage can alter the future from the same location
    fused_resolution:
      - legend parsing validates one bottom-aligned vertical bar per cell and
        returns both ordered colors and (turn, quota) rules
      - simulation state includes current segment index and the quota-usage
        vector, preventing false cycle assertions
      - entering an exhausted obstacle advances the segment color once, then
        preserves that color while the beam is inside the obstacle
  d_placement_legality:
    - reuse the original span/thickness construction so obstacles stay
      non-overlapping, on-grid, and separated from the teal square
    - all penetrated obstacles still touch an outer edge, so once a beam starts
      moving through an exhausted block it deterministically exits or reaches the
      boundary without ambiguous interior branching
    - every sampled grid is verified by running the fused solver against the
      oracle simulation
  e_visual_observability:
    - the color cycle is encoded only by visible left-to-right legend order
    - the quota is encoded only by visible bar height; no hidden color-id sort
      or invisible metadata is used
    - accepted samples must contain at least one pass-through event and at least
      three distinct cycle colors on painted trails
  f_visual_cleanliness:
    - beams still paint only former background cells, so obstacles remain legible
      and penetration is shown as a gap through the obstacle rather than noisy
      overwriting
    - first-arrival painting prevents later beams from recoloring earlier trails,
      keeping shared corridors stable instead of visually flickering
  g_color_inference:
    - the trail palette is capped at the four legend colors, reusing a small
      visible set that the solver can directly read from the top strip
    - segment advancement is tied to obstacle encounters, giving a structural
      cue for when the color changes rather than an arbitrary recolor
  main_new_assumptions_checked_against_generator:
    - each legend cell contains exactly one distinct color bar and those colors
      are all distinct
    - obstacle footprints expose at most one non-background, non-source color
      at a time
    - repeated encounters of a saturated color remain solvable because the
      generator only uses edge-touching rectangles, so straight penetration
      cannot wander indefinitely in the interior
"""

import random

import numpy as np

from arc_utils import bbox_coords, components_mask

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)

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
MAX_RETRIES = 700


def _parse_legend(grid: np.ndarray, legend_h: int) -> tuple[list[int], dict[int, tuple[str, int]]]:
    """
    Read legend order plus per-color (turn, quota) rules.

    Each cell contains one bottom-aligned vertical bar:
      - left bar  -> 'L'
      - right bar -> 'R'
      - bar height -> allowed number of turning collisions for that color
    """
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    order: list[int] = []
    rules: dict[int, tuple[str, int]] = {}
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        interior = legend[1 : legend_h - 1, a + 1 : b]
        colors = set(map(int, np.unique(interior)))
        colors.discard(bg)
        colors.discard(0)
        assert len(colors) == 1, "each legend cell must encode exactly one color"
        color = int(next(iter(colors)))
        ys, xs = np.where(interior == color)
        assert ys.size >= 1, "legend bar must have positive height"
        x = int(xs[0])
        assert np.all(xs == x), "legend bar must be a single vertical strip"
        quota = int(ys.size)
        expected_rows = np.arange(interior.shape[0] - quota, interior.shape[0])
        assert np.array_equal(np.sort(ys), expected_rows), "legend bar must be bottom-aligned and contiguous"
        turn = "L" if x <= (interior.shape[1] - 1) // 2 else "R"
        assert color not in rules, "legend colors must be distinct"
        order.append(color)
        rules[color] = (turn, quota)
    assert len(order) == 4, "legend must encode four ordered colors"
    return order, rules


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


def _rot(d: int, turn: str) -> int:
    return (d - 1) % 4 if turn == "L" else (d + 1) % 4


def _simulate_fused(
    base: np.ndarray,
    cycle_colors: list[int],
    rules: dict[int, tuple[str, int]],
    bg: int,
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int, str]]], dict[int, int], int, set[int]]:
    """Simulate the fused rule and return output plus event statistics."""
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    out = base.copy()
    h, w = out.shape
    turn_used = {color: 0 for color in cycle_colors}
    beam_events: list[list[tuple[int, int, int, int, str]]] = []
    pass_events = 0
    used_path_colors: set[int] = set()

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def obstacle_at(r: int, c: int) -> int | None:
        colors = set(map(int, np.unique(base[r : r + sz, c : c + sz])))
        colors.discard(bg)
        colors.discard(TEAL)
        if not colors:
            return None
        assert len(colors) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(colors)))

    def paint(r: int, c: int, color: int) -> None:
        base_blk = base[r : r + sz, c : c + sz]
        out_blk = out[r : r + sz, c : c + sz]
        mask = (base_blk == bg) & (out_blk == bg)
        out_blk[mask] = color
        out[r : r + sz, c : c + sz] = out_blk

    for d0 in range(4):
        r, c, d = int(sr), int(sc), d0
        seg_idx = 0
        events: list[tuple[int, int, int, int, str]] = []
        seen: set[tuple[int, int, int, int, tuple[int, ...]]] = set()
        while True:
            usage = tuple(turn_used[color] for color in cycle_colors)
            state = (r, c, d, seg_idx, usage)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            current_obs = obstacle_at(r, c)
            next_obs = obstacle_at(nr, nc)
            if next_obs is None:
                r, c = nr, nc
                color = int(cycle_colors[seg_idx])
                paint(r, c, color)
                used_path_colors.add(color)
                continue

            if current_obs is not None:
                assert current_obs == next_obs, "penetration cannot switch obstacle colors mid-crossing"
                r, c = nr, nc
                continue

            assert next_obs in rules, "obstacle color must be encoded by legend"
            turn, quota = rules[next_obs]
            seg_idx = (seg_idx + 1) % len(cycle_colors)
            if turn_used[next_obs] < quota:
                turn_used[next_obs] += 1
                events.append((d, next_obs, nr, nc, "turn"))
                d = _rot(d, turn)
            else:
                events.append((d, next_obs, nr, nc, "pass"))
                pass_events += 1
                r, c = nr, nc
        beam_events.append(events)

    return out, beam_events, turn_used, pass_events, used_path_colors


def _simulate_uncapped_turns(
    base: np.ndarray,
    cycle_colors: list[int],
    turns: dict[int, str],
    bg: int,
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int, str]]], dict[int, int], int, set[int]]:
    """Convenience wrapper with maximal quotas, used to detect quota effects."""
    rules = {color: (turns[color], 4) for color in cycle_colors}
    return _simulate_fused(base, cycle_colors, rules, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch four square-width beams whose segment colors cycle through the legend while each obstacle color has a limited turn budget.

    Input structure:
    - The top band is a grey legend split into four black cells, each containing one colored vertical bar.
    - A bar's horizontal side still tells whether that obstacle color turns beams counterclockwise or clockwise.
    - The same bar's height, read bottom-aligned over 1..4 pixels, gives how many times that color is still allowed to cause a turn.
    - The left-to-right order of the four legend colors is a separate signal: it is the repeating trail-color cycle used by every beam.
    - The main grid below contains one solid teal source square, a uniform background, and 3-4 solid obstacle rectangles colored from the legend.

    Rule:
    - Remove the legend from the output.
    - Fire one beam in each cardinal direction using the source square as a sliding sz-by-sz footprint.
    - Every beam starts with the first legend color; each time it encounters a new obstacle, it advances to the next legend color for the next visible segment.
    - If the encountered obstacle color still has remaining quota, spend one quota unit and turn according to that color's legend side without entering the obstacle.
    - If that color's quota has already been exhausted, the beam keeps going straight through the obstacle instead; obstacle cells themselves are not repainted, and only background cells on either side receive trail color.
    - Trail painting only fills still-background cells, so earlier beams keep their colors.

    Key insight: the legend carries two coupled instructions at once: the visible order of its four colors controls the segment-color cycle, while each color's own bar side and bar height govern how that obstacle redirects beams and how long it can keep doing so.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    cycle_colors, rules = _parse_legend(g, legend_h)

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    out, _, _, _, _ = _simulate_fused(base, cycle_colors, rules, bg)
    return out


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a span covering all required intervals while avoiding all excluded ones."""
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


def _make_legend(legend_colors: list[int], rules: dict[int, tuple[str, int]]) -> np.ndarray:
    """Build the fused 6x24 legend strip from ordered colors and (turn, quota) rules."""
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        turn, quota = rules[color]
        bar_col = start if turn == "L" else start + 3
        legend[5 - quota : 5, bar_col] = color
    return legend


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
    """Pick non-overlapping source->adjacent-target pairs for beams with two hits."""
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
    """Assign L/R turns per side, preferring outward exits for non-chain sources."""
    side_turn: dict[int, str] = {}
    for side in present:
        if side in chains:
            nxt = chains[side]
            side_turn[side] = "L" if nxt == (side - 1) % 4 else "R"
            continue
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

    target_hits = 5 if obstacle_count == 4 else random.choice([4, 5])
    second_hits = target_hits - obstacle_count
    if second_hits < 1 or second_hits > 2:
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

    vals, cnts = np.unique(base, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    turns = {side_color[s]: side_turn[s] for s in present}
    for color in legend_colors:
        turns.setdefault(color, random.choice(["L", "R"]))

    try:
        uncapped_out, uncapped_events, uncapped_turns, _, _ = _simulate_uncapped_turns(
            base, legend_colors, turns, bg
        )
    except AssertionError:
        return None

    uncapped_total = sum(len(beam) for beam in uncapped_events)
    used_colors = set(side_color.values())
    hit_colors = {color for beam in uncapped_events for _, color, _, _, _ in beam}
    repeated_colors = [color for color in used_colors if uncapped_turns[color] >= 2]
    if uncapped_total != target_hits:
        return None
    if used_colors - hit_colors:
        return None
    if not repeated_colors:
        return None

    saturated = set(random.sample(repeated_colors, random.randint(1, min(2, len(repeated_colors)))))
    limits: dict[int, int] = {}
    for color in legend_colors:
        if color in saturated:
            limits[color] = random.randint(1, uncapped_turns[color] - 1)
        elif color in used_colors:
            limits[color] = random.randint(uncapped_turns[color], 4)
        else:
            limits[color] = random.randint(1, 4)
    rules = {color: (turns[color], limits[color]) for color in legend_colors}

    try:
        expected, events, turn_used, pass_events, used_path_colors = _simulate_fused(
            base, legend_colors, rules, bg
        )
    except AssertionError:
        return None

    if pass_events < 1:
        return None
    if np.array_equal(expected, uncapped_out):
        return None
    if not set(legend_colors[:3]).issubset(used_path_colors):
        return None
    if len(used_path_colors) < 3:
        return None
    if not any(any(kind == "pass" for _, _, _, _, kind in beam) for beam in events):
        return None
    for color in saturated:
        if turn_used[color] != limits[color]:
            return None

    legend = _make_legend(legend_colors, rules)
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
