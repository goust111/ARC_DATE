# Variant of e87109e9
# Dimension: D6 — 射线终止条件 (dst)
# Change: 射线遇到障碍物时转弯继续，最终在到达网格边缘时终止... → 遇到特定颜色障碍物时终止（不转弯）

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, components_mask

# Design analysis for the D6 variant
# 1. Concrete legend/object change:
#    The input still has the same 6x24 legend band and the same main grid layout.
#    Three legend cells keep the original single vertical bar, placed on the left or
#    right edge of the 4-wide cell to encode CCW/CW turning. One legend cell is now
#    visibly different: it shows the same color as two vertical bars, one on each
#    edge of the cell. That color is the stop color. Obstacles keep the same edge-
#    attached rectangle geometry as the source task, but when a beam's next square
#    footprint meets the stop-colored obstacle it halts immediately instead of
#    turning. Example trace: in the original task, a north-going beam that meets the
#    north obstacle would turn west/east and continue painting. In this variant, if
#    that north obstacle uses the double-bar legend color, the beam ends at the last
#    free square before collision and paints nothing beyond that point.
# 2. Complexity calibration:
#    The change stays focused on D6 by introducing exactly one stop color and keeping
#    the other three legend colors as turn colors. Every generated puzzle uses the
#    stop color on a real obstacle that is actually hit by at least one beam, and the
#    grid still contains multiple turning obstacles so the solver must distinguish the
#    two obstacle functions inside the same example.
# 3. Old implicit assumptions that break:
#    The original _parse_legend assumed every legend color mapped to L/R, so a cell
#    with multiple x-positions for the same color would be misread as an L-turn.
#    The original beam loop also assumed every obstacle color must be in legend_turn
#    and always changes direction on contact. Both assumptions are false now: one
#    legend color must be parsed separately as stop_color, and obstacle handling must
#    branch on "stop vs turn" before rotating.
# 4. Placement legality and generator guarantees:
#    Obstacles are still non-overlapping edge rectangles with at least one-cell
#    clearance from the teal square. The generator chooses one present side as the
#    stop side, assigns its obstacle the stop color, and builds the same span
#    constraints as the source generator for all turning interactions. Because the
#    stop obstacle still covers the direct lane from the square to its edge, at least
#    one beam is guaranteed to terminate by color before reaching the boundary.
# 5. Visual observability:
#    The legend change is local and highly visible: one cell has double side bars
#    while the others have a single bar. In the output, some beams bend around
#    ordinary colors while the beam aimed at the stop color ends flush against that
#    obstacle, so the D6 change is visible from a few examples without relying on
#    hidden color-id ordering.
# 6. Visual cleanliness:
#    The variant does not add new object types in the main grid, only a special role
#    for one existing obstacle color. The legend remains sparse, obstacles remain
#    solid rectangles, and beam paint still only fills background cells, so the output
#    stays easy to parse.
# 7. New solver assumptions and matching generator support:
#    The solver now assumes exactly one legend cell uses the double-bar encoding and
#    exactly three colors remain as turn colors. The generator enforces that pattern,
#    keeps obstacle colors unique, and validates every sample by running main() on the
#    constructed input and comparing against the oracle simulation.

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)


def _parse_legend(grid: np.ndarray, legend_h: int) -> tuple[dict[int, str], int]:
    """
    Legend is a top region with a background color (typically 5) and 4 vertical-bar cells.
    Single bars still encode turn direction:
      - bar on left  -> 'L' (turn counterclockwise)
      - bar on right -> 'R' (turn clockwise)
    One special cell has the same color on both outer columns; that color is the
    terminal obstacle color and does not encode a turn.
    """
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, counts = np.unique(legend, return_counts=True)
    bg = int(vals[counts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    turn: dict[int, str] = {}
    stop_color: int | None = None
    seen_colors: set[int] = set()
    cell_count = 0
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        cell_count += 1
        interior = legend[1 : legend_h - 1, a + 1 : b]
        colors = set(map(int, np.unique(interior)))
        colors.discard(bg)
        colors.discard(0)
        assert len(colors) == 1, "each legend cell must encode exactly one color"
        col = int(next(iter(colors)))
        assert col not in seen_colors, "legend colors must be distinct"
        seen_colors.add(col)

        xs = sorted(set(map(int, np.where(interior == col)[1])))
        if len(xs) == 1:
            x = xs[0]
            turn[col] = "L" if x <= (interior.shape[1] - 1) // 2 else "R"
        else:
            assert stop_color is None, "legend must encode exactly one stop color"
            assert xs == [0, interior.shape[1] - 1], "stop marker must occupy both sides"
            stop_color = col

    assert cell_count == 4, "legend must have four cells"
    assert stop_color is not None, "legend must encode one stop color"
    assert len(turn) == 3, "legend must encode three turn colors"
    return turn, int(stop_color)


def _find_teal_square(g: np.ndarray, teal: int = 8) -> tuple[int, int, int]:
    """Return (r0,c0,size) for a solid square teal component."""
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


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four teal beams from a square, where one legend color ends the beam and the
    others still turn it.

    Input structure:
    - Top band: grey (5) legend with four 4-column cells. A single left/right bar
      encodes CCW/CW turning for that color, while the one cell with matching bars on
      both outer edges marks the stop color.
    - Below: main grid with one background color, a solid teal (8) square, and colored
      obstacle rectangles touching the outer edges.

    Rule:
    - Remove the legend from the output.
    - Shoot a sz-by-sz teal beam from the square in each cardinal direction.
    - Moving beams paint teal only onto background cells.
    - If the next footprint meets an ordinary legend color, rotate according to its
      left/right bar and continue.
    - If the next footprint meets the stop color, terminate that beam immediately
      without turning.

    Key insight: The special legend cell is not another left/right case; its double
    bar means "this obstacle color ends a beam", so the solver must split obstacle
    colors into turn colors versus one terminate color before tracing paths.
    """
    g = np.asarray(grid, dtype=int)

    legend_bg = 5
    rows_with_legend = np.where(np.any(g == legend_bg, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    legend_turn, stop_color = _parse_legend(g, legend_h)

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, counts = np.unique(base, return_counts=True)
    bg = int(vals[counts.argmax()])
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

            if oc == stop_color:
                break

            assert oc in legend_turn, "obstacle color must be encoded by legend"
            d = _rot(d, legend_turn[oc])

    return out


# --------------- generator ---------------

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
MAX_RETRIES = 500


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


def _simulate_paths(
    base: np.ndarray, turns: dict[int, str], stop_color: int, bg: int
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int, bool]]]]:
    """Simulate the four beams, recording whether each obstacle hit is terminal."""
    vals, counts = np.unique(base, return_counts=True)
    assert int(vals[counts.argmax()]) == bg

    teal_locs = np.argwhere(base == TEAL)
    assert teal_locs.size
    sr, sc = teal_locs.min(axis=0)
    r1, c1 = teal_locs.max(axis=0)
    sz = int(r1 - sr + 1)
    assert sz == int(c1 - sc + 1)

    out = base.copy()
    h, w = out.shape
    hits: list[list[tuple[int, int, int, int, bool]]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = TEAL

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
        beam_hits: list[tuple[int, int, int, int, bool]] = []
        seen: set[tuple[int, int, int]] = set()
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
                r, c = nr, nc
                paint(r, c)
                continue
            is_stop = obs == stop_color
            beam_hits.append((d, obs, nr, nc, is_stop))
            if is_stop:
                break
            assert obs in turns
            d = _rot(d, turns[obs])
            assert len(beam_hits) <= 2
        hits.append(beam_hits)

    return out, hits


def _make_legend(legend_colors: list[int], turns: dict[int, str], stop_color: int) -> np.ndarray:
    """Build the 6x24 legend; the stop color is drawn as two outer bars."""
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        if color == stop_color:
            legend[1:5, start] = color
            legend[1:5, start + 3] = color
        else:
            bar_col = start if turns[color] == "L" else start + 3
            legend[1:5, bar_col] = color
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


def _select_chains(
    present: set[int], stop_side: int, second_hits: int
) -> dict[int, int] | None:
    """Pick non-overlapping source->target pairs for beams that hit a second obstacle."""
    turn_sides = present - {stop_side}
    candidates = [
        (side, nxt)
        for side in turn_sides
        for nxt in ((side - 1) % 4, (side + 1) % 4)
        if nxt in present
    ]
    random.shuffle(candidates)
    selected: list[tuple[int, int]] = []
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    for side, nxt in candidates:
        if len(selected) == second_hits:
            break
        if side in used_sources or nxt in used_targets:
            continue
        selected.append((side, nxt))
        used_sources.add(side)
        used_targets.add(nxt)
    return dict(selected) if len(selected) == second_hits else None


def _assign_turns(
    present: set[int], stop_side: int, chains: dict[int, int]
) -> dict[int, str]:
    """Assign L/R for the non-terminal obstacle sides."""
    side_turn: dict[int, str] = {}
    for side in present:
        if side == stop_side:
            continue
        if side in chains:
            nxt = chains[side]
            side_turn[side] = "L" if nxt == (side - 1) % 4 else "R"
            continue
        outward = [t for t in ("L", "R") if _rot(side, t) not in present]
        side_turn[side] = random.choice(outward if outward else ["L", "R"])
    return side_turn


def _compute_span_constraints(
    present: set[int],
    stop_side: int,
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
        if side == stop_side:
            continue
        nxt = _rot(side, side_turn[side])
        if nxt not in present:
            continue
        if chains.get(side) == nxt:
            required[nxt].append(lane_interval(side))
            if nxt != stop_side:
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
    stop_side = random.choice(tuple(present))

    target_hits = random.choice([4, 5])
    second_hits = target_hits - obstacle_count
    if second_hits < 0 or second_hits > 2 or second_hits > obstacle_count - 1:
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

    chains = _select_chains(present, stop_side, second_hits)
    if chains is None:
        return None

    side_turn = _assign_turns(present, stop_side, chains)
    required, excluded = _compute_span_constraints(
        present, stop_side, side_turn, chains, sr, sc, sz, lane
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
    stop_color = side_color[stop_side]
    bg_choices = [c for c in BG_COLOR_POOL if c not in legend_colors]
    bg = random.choice(bg_choices)

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

    turns = {side_color[s]: side_turn[s] for s in present if s != stop_side}
    for color in legend_colors:
        if color != stop_color:
            turns.setdefault(color, random.choice(["L", "R"]))

    try:
        expected, hits = _simulate_paths(base, turns, stop_color, bg)
    except AssertionError:
        return None

    hit_counts = [len(beam) for beam in hits]
    all_hits = [hit for beam in hits for hit in beam]
    hit_colors = {color for _, color, _, _, _ in all_hits}
    stop_hits = sum(1 for *_, is_stop in all_hits if is_stop)
    if sum(hit_counts) != target_hits or any(c > 2 for c in hit_counts):
        return None
    if stop_hits < 1:
        return None
    if set(side_color.values()) - hit_colors:
        return None

    legend = _make_legend(legend_colors, turns, stop_color)
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
    raise RuntimeError("failed to generate e87109e9 D6 variant input")
