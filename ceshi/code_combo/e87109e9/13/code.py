# Combo variant of e87109e9 (index 13)
# combo_spec: D6/2+D17/0
# Name: 转弯障碍与终止障碍双重编码

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

# Design analysis for combo D6/2 + D17/0
# 1. Concrete fused behavior on the pixel grid:
#    The source task had four legend cells and every obstacle color belonged to one
#    of those cells, so every obstacle always meant "turn left/right". The fused
#    task keeps those four turning cells but also adds a separate stop marker in the
#    legend: a solid 2-column pillar in the middle gutter. Obstacles with one of the
#    four cell colors still turn beams; obstacles with the extra stop-marker color do
#    not have a turn entry and instead terminate the beam immediately. Example trace:
#    a north-moving beam that meets a red top obstacle now checks whether red is one
#    of the four cell colors or the standalone pillar color. In the first case it
#    turns in place and continues; in the second case it stops flush before contact.
# 2. Complexity calibration:
#    The variant must be visibly different from both the original and the D6-only
#    case. Using four ordinary turn entries plus one separate stop color keeps the
#    legend readable while making the split explicit. Each sample contains exactly
#    one stop obstacle that is actually hit and three turning obstacles that are also
#    hit, so the solver cannot ignore either mechanism.
# 3. Broken implicit assumptions from the source solver:
#    The original parser assumed every relevant legend color lived inside the four
#    left/right cells and mapped directly to L/R. That is no longer true because one
#    obstacle color is encoded by a separate marker, not by a cell entry. The beam
#    tracer also assumed every obstacle hit must rotate. Now obstacle handling must
#    first distinguish stop color versus turn colors, then only rotate for the turn
#    subset.
# 4. Placement legality and generator guarantees:
#    Obstacles remain edge-touching solid rectangles with at least one cell of
#    clearance from the teal square and no overlap with one another. The generator
#    places all four side obstacles, chooses one side as the stop obstacle, and uses
#    span constraints so each side is hit by the beam launched toward it. Planned
#    second hits are optional but limited, which keeps paths non-cyclic and easy to
#    validate.
# 5. Visual observability:
#    A human can spot two legend object types: four black 4x4 cells with a single
#    colored side bar, plus one centered colored pillar. In outputs, the pillar color
#    always produces a beam that ends abruptly, while the cell colors produce bends.
#    That contrast is visible across a few examples without relying on color-number
#    ordering or hidden metadata.
# 6. Visual cleanliness:
#    The fused rule adds only one compact legend marker and does not stack multiple
#    marks on the same obstacle. The main grid still contains just the square, a
#    uniform background, and solid edge rectangles, so the resulting input/output
#    pairs stay sparse and parseable.
# 7. Inference support for color mapping:
#    There are only five non-background colors involved in the legend: four turn
#    colors and one stop color. The structural cue is not the numeric color id but
#    whether the color appears in a turning cell or in the standalone pillar. This
#    keeps the mapping learnable from 2-4 examples.
# 8. New solver assumptions and matching generator support:
#    The solver now assumes the four turn cells sit at fixed column starts and the
#    stop marker occupies the fixed middle 2-column gutter. The generator enforces
#    that exact layout, ensures the stop color is distinct from all turn colors, and
#    validates every sample by comparing main() against an oracle simulation.
# 9. Review of new implicit assumptions in main():
#    - Legend cropping assumes color 5 appears only in the top band; generator keeps
#      main-grid backgrounds away from 5.
#    - Turn-cell parsing assumes each 4x4 cell contains exactly one nonzero color in
#      exactly one border column; generator draws cells that way.
#    - Stop parsing assumes the middle pillar is a single solid color and distinct
#      from the turn-cell colors; generator enforces both.
#    - Beam simulation assumes every obstacle footprint is monochrome and paths do
#      not cycle; generator uses non-overlapping rectangles and pre-validates traces.

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)


def _parse_legend(grid: np.ndarray, legend_h: int) -> tuple[dict[int, str], int]:
    """
    Parse four turning entries plus one separate stop marker from the legend band.

    Turning entries occupy fixed 4-column cells. A single colored bar on the left
    edge means CCW, and a single bar on the right edge means CW. The stop color is
    not one of those entries; it appears as a centered solid pillar.
    """
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, counts = np.unique(legend, return_counts=True)
    bg = int(vals[counts.argmax()])

    turn: dict[int, str] = {}
    seen_colors: set[int] = set()
    for start in LEGEND_CELL_STARTS:
        cell = legend[1:5, start : start + 4]
        colors = set(map(int, np.unique(cell)))
        colors.discard(bg)
        colors.discard(0)
        assert len(colors) == 1, "each turn cell must encode exactly one color"
        color = int(next(iter(colors)))
        assert color not in seen_colors, "turn colors must be distinct"
        seen_colors.add(color)

        xs = sorted(set(map(int, np.where(cell == color)[1])))
        assert xs in ([0], [3]), "turn entry must be a single edge bar"
        turn[color] = "L" if xs == [0] else "R"

    marker = legend[1:5, STOP_MARKER_COLS[0] : STOP_MARKER_COLS[1] + 1]
    marker_colors = set(map(int, np.unique(marker)))
    marker_colors.discard(bg)
    marker_colors.discard(0)
    assert len(marker_colors) == 1, "stop marker must encode exactly one color"
    stop_color = int(next(iter(marker_colors)))
    assert stop_color not in seen_colors, "stop color must differ from turn colors"
    assert np.all(marker == stop_color), "stop marker must be a solid pillar"
    return turn, stop_color


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


def _rot(d: int, turn: str) -> int:
    return (d - 1) % 4 if turn == "L" else (d + 1) % 4


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four teal beams from a square, using two legend-coded obstacle functions.

    Input structure:
    - Top band: a grey (5) legend with four ordinary 4-column cells plus one extra
      centered marker. In each ordinary cell, a single colored side bar encodes a
      left-turn or right-turn rule for that color. The centered marker is a separate
      stop color and is not a turn entry.
    - Below: a main grid with one background color, one solid teal (8) square, and
      solid obstacle rectangles touching the outer edges.

    Rule:
    - Remove the legend from the output.
    - Launch a sz-by-sz teal beam from the square in each cardinal direction.
    - As a beam advances, it paints teal onto background cells only.
    - If the next beam footprint meets a color listed in a turn cell, rotate in
      place according to that cell and continue.
    - If the next footprint meets the standalone stop color, terminate the beam
      immediately without rotating.

    Key insight: Obstacle behavior is split by legend role, not by obstacle shape.
    Colors shown inside the four left/right cells are turning colors, while the extra
    centered marker names the one color that ends a beam outright.
    """
    g = np.asarray(grid, dtype=int)

    legend_bg = 5
    rows_with_legend = np.where(np.any(g == legend_bg, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend must be the top band"

    legend_turn, stop_color = _parse_legend(g, legend_h)

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, counts = np.unique(base, return_counts=True)
    bg = int(vals[counts.argmax()])
    teal = TEAL
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
        colors = set(map(int, np.unique(blk)))
        colors.discard(bg)
        colors.discard(teal)
        if not colors:
            return None
        assert len(colors) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(colors)))

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

            assert oc in legend_turn, "obstacle color must be a turn color or the stop color"
            d = _rot(d, legend_turn[oc])

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
STOP_MARKER_COLS = (11, 12)

TEAL_SZ2_PROB = 0.75
SQUARE_EDGE_GAP = 5

OBS_THICKNESS_RANGE = (2, 4)
OBS_SINGLE_MARGIN = 6
OBS_BOTH_GAP = 2

SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.4

MAX_RETRIES = 600


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
    base: np.ndarray,
    turns: dict[int, str],
    stop_color: int,
    bg: int,
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int, bool]]]]:
    """Simulate the four beams and record every obstacle hit plus stop-vs-turn."""
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


def _simulate_counterfactual_stop_turn(
    base: np.ndarray,
    turns: dict[int, str],
    stop_color: int,
    bg: int,
    stop_turn: str,
) -> np.ndarray | None:
    """
    Counterfactual check: force the stop color to behave like a turn color.

    Returns None if that counterfactual would cycle.
    """
    turn_map = dict(turns)
    turn_map[stop_color] = stop_turn

    teal_locs = np.argwhere(base == TEAL)
    assert teal_locs.size
    sr, sc = teal_locs.min(axis=0)
    r1, c1 = teal_locs.max(axis=0)
    sz = int(r1 - sr + 1)
    assert sz == int(c1 - sc + 1)

    out = base.copy()
    h, w = out.shape

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
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
            if state in seen:
                return None
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
            d = _rot(d, turn_map[obs])

    return out


def _make_legend(legend_colors: list[int], turns: dict[int, str], stop_color: int) -> np.ndarray:
    """Build the 6x24 legend with four turn cells plus a centered stop marker."""
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    legend[1:5, STOP_MARKER_COLS[0] : STOP_MARKER_COLS[1] + 1] = stop_color
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


def _select_chains(
    present: set[int], stop_side: int, second_hits: int
) -> dict[int, int] | None:
    """Pick source->target pairs for planned second hits."""
    if second_hits == 0:
        return {}
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
    """Assign L/R rules for the turning obstacle sides."""
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
    """Build required/excluded intervals for each obstacle span."""
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
    present = {_N, _E, _S, _W}
    stop_side = random.choice(tuple(present))

    target_hits = random.choice([4, 5])
    second_hits = target_hits - len(present)
    if second_hits < 0 or second_hits > 1:
        return None

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
    stop_color = next(color for color in LEGEND_COLOR_POOL if color not in legend_colors)
    turn_sides = sorted(present - {stop_side})
    turn_obstacle_colors = random.sample(legend_colors, len(turn_sides))
    side_color = {side: color for side, color in zip(turn_sides, turn_obstacle_colors)}
    side_color[stop_side] = stop_color
    bg = random.choice([c for c in BG_COLOR_POOL if c not in legend_colors and c != stop_color])

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

    turns = {side_color[s]: side_turn[s] for s in turn_sides}
    for color in legend_colors:
        turns.setdefault(color, random.choice(["L", "R"]))

    try:
        expected, hits = _simulate_paths(base, turns, stop_color, bg)
    except AssertionError:
        return None

    hit_counts = [len(beam) for beam in hits]
    all_hits = [hit for beam in hits for hit in beam]
    hit_colors = {color for _, color, _, _, _ in all_hits}
    stop_hits = sum(1 for *_, is_stop in all_hits if is_stop)
    turn_hits = len(all_hits) - stop_hits
    if sum(hit_counts) != target_hits or any(c > 2 for c in hit_counts):
        return None
    if stop_hits < 1 or turn_hits < 3:
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
    for stop_turn in ("L", "R"):
        counterfactual = _simulate_counterfactual_stop_turn(base, turns, stop_color, bg, stop_turn)
        if counterfactual is not None and np.array_equal(counterfactual, expected):
            return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e87109e9 combo variant input")
