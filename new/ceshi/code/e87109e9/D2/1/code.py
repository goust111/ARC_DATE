# Variant of e87109e9
# Dimension: D2 — 转弯规则 (path)
# Change: 遇到障碍物时根据图例查找该颜色——图例中颜色条在左侧表示逆时针转弯，在右侧表示顺时针转弯... → 转弯角度可变（90度/180度）

import random
import sys
from pathlib import Path

import numpy as np

for _parent in Path(__file__).resolve().parents:
    _solvers = _parent / "solvers"
    if (_solvers / "arc_utils.py").exists():
        sys.path.append(str(_solvers))
        break

from arc_utils import bbox_coords, components_mask


# Design analysis for the D2 variant
# a) Concrete behavior:
#    The layout stays the same: a 6x24 legend band on top and a cropped main grid below.
#    The four legend cells still contain one colored vertical bar each, but the meaning changes:
#    a bar on the left means that obstacle color causes a 90-degree counterclockwise turn,
#    while a bar on the right means that obstacle color causes a 180-degree reversal.
#    In the old task a "right" bar always produced a perpendicular branch; here it can instead
#    send the beam back along the same corridor. Example: a north-going beam that hits a right-bar
#    obstacle used to bend east; now it flips south, retraces its own lane, crosses the teal source,
#    and may then hit the south obstacle. That is the core behavioral difference this variant exposes.
# b) Complexity calibration:
#    Only two visible action types are used in a sample: left-bars repeat the original 90-degree bend,
#    right-bars consistently mean a bounce-back. To keep the difference readable, generation forces
#    at least one used obstacle to be a 180-degree obstacle whose opposite side is also present, so
#    the reversal creates a genuine two-hit path rather than a trivial dead-end.
# c) Hidden assumptions review:
#    The original solver assumed every non-left legend entry meant a clockwise turn. That is no longer
#    valid, so both legend parsing and direction updates must be rewritten. The original generator also
#    relied on the fact that left/right always move into adjacent sides; 180-degree turns can create
#    back-and-forth cycles between opposite edges, so generation now explicitly forbids opposite pairs
#    that both reverse and adds span exclusions to stop third hits.
# d) Placement legality:
#    Obstacles are still edge-touching rectangles with the original thickness and span ranges.
#    Every placed obstacle must cover the source beam cross-section on its own side, must not overlap
#    the teal square or another obstacle, and must satisfy extra required/excluded span intervals so
#    the simulated paths match the intended two-hit structure.
# e) Visual observability:
#    The generator guarantees that at least one beam first hits a 180-degree obstacle and then reaches
#    a second obstacle after reversing. Samples are also filtered so the new output differs from the
#    old clockwise interpretation on the exact same input by multiple pixels.
# f) Visual cleanliness:
#    The variant keeps the original sparse geometry: one teal square, 3-4 solid edge bars, one legend.
#    Constraints reject cases with mixed-color collision footprints, cycles, or more than two hits per
#    beam, so the painted output remains legible instead of turning into a dense maze.
# g) Color inference:
#    Colors are never ordered or transformed numerically. A color's role is visible only through its
#    matching legend bar position, and the number of active obstacle colors remains small (3-4 used,
#    4 shown in the legend), which is enough for a human to infer the mapping from examples.
#
# New solver-side assumptions that generation must satisfy:
# - color 5 is still confined to the top legend band;
# - the main area still contains exactly one solid teal square component;
# - every obstacle footprint touched by a beam is monochrome;
# - at least one actually-hit obstacle uses the right-bar / 180-degree rule;
# - no generated beam path revisits the same (position, direction) state.


_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    """
    Legend is a top region with a background color (typically 5) and 4 vertical bars.
    Each bar's horizontal position inside its cell encodes the path change:
      - bar on left  -> 'L' (turn 90 degrees counterclockwise)
      - bar on right -> 'U' (turn 180 degrees / reverse)
    """
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    action: dict[int, str] = {}
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        interior = legend[1 : legend_h - 1, a + 1 : b]
        colors = set(map(int, np.unique(interior)))
        colors.discard(bg)
        colors.discard(0)
        assert len(colors) == 1, "each legend cell must encode exactly one color"
        col = int(next(iter(colors)))
        xs = np.where(interior == col)[1]
        x = int(xs[0])
        action[col] = "L" if x <= (interior.shape[1] - 1) // 2 else "U"
    assert len(action) == 4, "legend must encode four turn colors"
    return action


def _find_teal_square(g: np.ndarray, teal: int = 8) -> tuple[int, int, int]:
    """Return (r0, c0, size) for the unique solid teal square component."""
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


def _ccw(d: int) -> int:
    return (d - 1) % 4


def _opp(d: int) -> int:
    return (d + 2) % 4


def _rot(d: int, action: str) -> int:
    if action == "L":
        return _ccw(d)
    if action == "R":
        return (d + 1) % 4
    assert action == "U", f"unknown action: {action}"
    return _opp(d)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four teal beams from a square, with obstacle colors controlling left turns or reversals.

    Input structure:
    - Top band: a grey legend with four colored vertical bars; each color's bar position encodes which path change that obstacle color causes.
    - Below: the main grid with one solid teal square, a single dominant background, and 3-4 colored edge obstacles.

    Rule:
    - Remove the legend from the output.
    - Launch one sz-by-sz teal beam from the square in each cardinal direction.
    - When a beam moves over background, it paints those background cells teal.
    - When a beam meets obstacle color C, consult the legend: left bar means turn 90 degrees counterclockwise, right bar means reverse 180 degrees. Continue from the same position using the new direction.
    - Stop when the next sz-by-sz footprint would leave the grid.

    Key insight: the legend no longer distinguishes two sideways bends; it distinguishes a bend versus a bounce-back, so right-bar colors make beams retrace and potentially reach the opposite obstacle.
    """
    g = np.asarray(grid, dtype=int)

    legend_bg = 5
    rows_with_legend = np.where(np.any(g == legend_bg, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    legend_action = _parse_legend(g, legend_h)

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

            assert oc in legend_action, "obstacle color must be encoded by legend"
            d = _rot(d, legend_action[oc])

    return out


# --------------- generator (o5x) ---------------

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

OBS_COUNT_4_PROB = 0.65
OBS_THICKNESS_RANGE = (2, 4)
OBS_SINGLE_MARGIN = 6
OBS_BOTH_GAP = 2

SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.4

MAX_RETRIES = 400


def _native_interval(side: int, sr: int, sc: int, sz: int) -> tuple[int, int]:
    return (sc, sc + sz - 1) if side in (_N, _S) else (sr, sr + sz - 1)


def _lane_interval(side: int, lane: dict[int, int], sz: int) -> tuple[int, int]:
    return lane[side], lane[side] + sz - 1


def _simulate_paths(
    base: np.ndarray,
    actions: dict[int, str],
    bg: int,
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]]]:
    """Simulate 4 beams, returning the painted grid and per-beam hit records."""
    vals, counts = np.unique(base, return_counts=True)
    assert int(vals[counts.argmax()]) == bg

    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    out = base.copy()
    H, W = out.shape
    hits: list[list[tuple[int, int, int, int]]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < H and c + sz - 1 < W

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
        r, c, d = int(sr), int(sc), d0
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

            obs = obstacle_at(nr, nc)
            if obs is None:
                r, c = nr, nc
                paint(r, c)
                continue

            assert obs in actions
            beam_hits.append((d, obs, nr, nc))
            d = _rot(d, actions[obs])
            assert len(beam_hits) <= 2, "generator only accepts up to two hits per beam"
        hits.append(beam_hits)

    return out, hits


def _make_legend(legend_colors: list[int], actions: dict[int, str]) -> np.ndarray:
    """Build the 6x24 legend band from 4 colors and their action assignments."""
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if actions[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


def _rand_thickness(lo: int, hi: int) -> int | None:
    return random.randint(lo, hi) if hi >= lo else None


def _thickness_pair(axis: int, sz: int, a_present: bool, b_present: bool) -> tuple[int | None, int | None] | None:
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


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a span that covers every required interval and avoids every excluded interval."""
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


def _add_followup_exclusions(
    target: int,
    target_action: str,
    present: set[int],
    lane: dict[int, int],
    sz: int,
    incoming_interval: tuple[int, int],
    excluded: dict[int, list[tuple[int, int]]],
) -> None:
    """Prevent a beam from taking a third hit after its second collision at target."""
    if target_action == "L":
        after = _ccw(target)
        if after in present:
            excluded[after].append(_lane_interval(target, lane, sz))
    else:
        assert target_action == "U"
        after = _opp(target)
        if after in present:
            excluded[after].append(incoming_interval)


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    obstacle_count = 4 if random.random() < OBS_COUNT_4_PROB else 3

    u_side = random.randrange(4)
    present = {_opp(u_side), u_side}
    others = [s for s in (_N, _E, _S, _W) if s not in present]
    need_more = obstacle_count - 2
    if need_more > 0:
        present.update(random.sample(others, need_more))

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

    side_action = {side: "L" for side in present}
    side_action[u_side] = "U"
    stop_side = _opp(u_side)

    chain_candidates = [s for s in present if side_action[s] == "L" and s != stop_side and _ccw(s) in present]
    chain_sources: set[int] = set()
    if chain_candidates and random.random() < 0.65:
        chain_sources.add(random.choice(chain_candidates))

    required: dict[int, list[tuple[int, int]]] = {s: [] for s in present}
    excluded: dict[int, list[tuple[int, int]]] = {s: [] for s in present}
    for side in present:
        required[side].append(_native_interval(side, sr, sc, sz))

    _add_followup_exclusions(
        stop_side,
        side_action[stop_side],
        present,
        lane,
        sz,
        _native_interval(stop_side, sr, sc, sz),
        excluded,
    )

    for side in present:
        if side_action[side] != "L":
            continue
        nxt = _ccw(side)
        if nxt not in present:
            continue
        interval = _lane_interval(side, lane, sz)
        if side in chain_sources:
            required[nxt].append(interval)
            _add_followup_exclusions(nxt, side_action[nxt], present, lane, sz, interval, excluded)
        else:
            excluded[nxt].append(interval)

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

    actions = {side_color[s]: side_action[s] for s in present}
    for color in legend_colors:
        actions.setdefault(color, random.choice(["L", "U"]))

    try:
        expected, hits = _simulate_paths(base, actions, bg)
    except AssertionError:
        return None

    target_hits = obstacle_count + 1 + len(chain_sources)
    hit_counts = [len(beam) for beam in hits]
    hit_colors = {color for beam in hits for _, color, _, _ in beam}
    if sum(hit_counts) != target_hits or any(c > 2 for c in hit_counts):
        return None
    if set(side_color.values()) - hit_colors:
        return None

    u_color = side_color[u_side]
    if len(hits[u_side]) != 2 or hits[u_side][0][1] != u_color:
        return None

    legacy_actions = {color: ("L" if action == "L" else "R") for color, action in actions.items()}
    try:
        legacy_out, _legacy_hits = _simulate_paths(base, legacy_actions, bg)
    except AssertionError:
        return None
    variant_delta = int(np.sum(expected != legacy_out))
    if variant_delta < 2 * sz:
        return None

    legend = _make_legend(legend_colors, actions)
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
