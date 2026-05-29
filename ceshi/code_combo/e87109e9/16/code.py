# Combo variant of e87109e9 (index 16)
# combo_spec: D2/1+D19/0
# Name: 可变转弯角度含45度

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


# Design analysis for the combo variant D2/1 + D19/0
# a) Concrete behavior:
#    The visible layout stays close to the original task: a fixed 6x24 legend band
#    above a cropped main grid containing one solid teal square and a few colored
#    obstacle rectangles. The semantic change is in what the legend means. A left bar
#    still means "turn 90 degrees counterclockwise". A right bar no longer means a
#    90-degree clockwise turn; it now means a shallow 45-degree bend that toggles the
#    beam between cardinal and diagonal motion. Concretely:
#    - a north-going beam hitting a right-bar obstacle becomes a north-east diagonal;
#    - when that diagonal later hits the east obstacle, the same right-bar rule turns
#      it back into a north-going beam so it leaves the obstacle instead of re-hitting
#      it forever. The mixed path therefore contains both orthogonal square slides and
#      diagonal square slides, which is the new observable behavior.
# b) Complexity calibration:
#    The generator keeps only two action types, L and D, so the hypothesis space remains
#    small for humans. Every sample forces at least one actually-used D obstacle and at
#    least one actually-used L obstacle. It also forces at least one D-driven path to
#    spend multiple steps in a diagonal segment, so the fusion is visible rather than a
#    degenerate one-pixel kink.
# c) Hidden assumptions review:
#    The original solver assumed directions lived in 4 states and every hit preserved
#    cardinal motion. That breaks immediately once 45-degree motion exists, so main()
#    must move to 8 compass directions and the "right" action can no longer be a simple
#    clockwise 90-degree rotation. Another hidden assumption was that every turn points
#    away from the obstacle automatically; with diagonal motion that is false unless the
#    45-degree action is defined as a parity toggle (cardinal -> clockwise diagonal,
#    diagonal -> counterclockwise cardinal). The generator mirrors that exact rule so it
#    never creates self-colliding immediate re-hits.
# d) Placement legality:
#    To make 45-degree motion physically realizable with sz-by-sz square footprints, the
#    generator uses three floating solid rectangles instead of the original edge bars.
#    Their positions are chosen so that a north beam hits a D obstacle, leaves it cleanly,
#    reaches a second D obstacle along a diagonal, and a separate west beam hits an L
#    obstacle. All rectangles are checked for non-overlap and for clearance from the
#    teal source block.
# e) Visual observability:
#    A solver watching 2-4 examples can compare left-bar colors with right-bar colors:
#    left-bars produce the familiar orthogonal bends, while right-bars create visible
#    diagonal teal corridors before snapping back to a cardinal lane. Samples are also
#    filtered against the legacy right=90-degree interpretation so the output differs by
#    multiple pixels on the same input.
# f) Visual cleanliness:
#    The grid remains sparse: one source block, three obstacle rectangles, one legend.
#    Beams are still sz-by-sz square footprints, so diagonal motion becomes a clean
#    staircase rather than a noisy single-pixel trace. Constraints reject cycles, mixed
#    obstacle footprints, and paths with more than two hits per beam.
# g) Change inferability:
#    No numeric color ordering is used. A color's behavior is inferable only from its own
#    legend bar position, and the number of legend colors stays at four with repeated use.
#    Because both L and D appear in every accepted sample, a human can infer that bar
#    position controls not just "which side" but "which angle family" the beam follows.
#
# New solver-side assumptions that generation explicitly guarantees:
# - color 5 is confined to the legend band at the top;
# - the main area contains exactly one solid teal square component;
# - every obstacle footprint a beam tests is monochrome;
# - every generated obstacle is hit by at least one beam;
# - at least one actually-used obstacle has action D and yields a diagonal segment;
# - at least one actually-used obstacle has action L;
# - no beam revisits the same (position, direction) state.


_N, _NE, _E, _SE, _S, _SW, _W, _NW = range(8)
_DIRS8 = {
    _N: (-1, 0),
    _NE: (-1, 1),
    _E: (0, 1),
    _SE: (1, 1),
    _S: (1, 0),
    _SW: (1, -1),
    _W: (0, -1),
    _NW: (-1, -1),
}
_SIDES = [0, 2, 4, 6]
_N4, _E4, _S4, _W4 = range(4)

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

MAX_RETRIES = 1200


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    """
    Parse the legend band.

    A left-positioned bar means 'L': rotate 90 degrees counterclockwise.
    A right-positioned bar means 'D': take the 45-degree diagonal-toggle bend.
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
        action[col] = "L" if x <= (interior.shape[1] - 1) // 2 else "D"
    assert len(action) == 4, "legend must encode four turn colors"
    return action


def _find_teal_square(g: np.ndarray, teal: int = TEAL) -> tuple[int, int, int]:
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


def _rot_l(d: int) -> int:
    return (d - 2) % 8


def _rot_d(d: int) -> int:
    return (d + 1) % 8 if d % 2 == 0 else (d - 1) % 8


def _turn(d: int, action: str) -> int:
    if action == "L":
        return _rot_l(d)
    assert action == "D", f"unknown action: {action}"
    return _rot_d(d)


def _legacy_rot_cw(d4: int) -> int:
    return (d4 + 1) % 4


def _ccw_side(side4: int) -> int:
    return (side4 - 1) % 4


def _cw_side(side4: int) -> int:
    return (side4 + 1) % 4


def _side_to_dir(side4: int) -> int:
    return [_N, _E, _S, _W][side4]


def _dir_to_diag(side4: int) -> int:
    return [_NE, _SE, _SW, _NW][side4]


def _parse_base(grid: np.ndarray) -> tuple[np.ndarray, int, tuple[int, int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    legend_bg = LEGEND_BG
    rows_with_legend = np.where(np.any(g == legend_bg, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"
    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"
    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    src = _find_teal_square(base, teal=TEAL)
    return g, bg, src, base


def _simulate_paths(
    base: np.ndarray,
    actions: dict[int, str],
    bg: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Simulate the 8-direction combo rule and return painted output plus beam metadata."""
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    out = base.copy()
    h, w = out.shape
    beams: list[dict[str, object]] = []

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

    for side4 in range(4):
        r, c, d = int(sr), int(sc), _side_to_dir(side4)
        seen: set[tuple[int, int, int]] = set()
        hit_records: list[tuple[int, int, int, int, str]] = []
        diagonal_steps = 0
        while True:
            state = (r, c, d)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)
            dr, dc = _DIRS8[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break
            obs = obstacle_at(nr, nc)
            if obs is None:
                r, c = nr, nc
                if d % 2 == 1:
                    diagonal_steps += 1
                paint(r, c)
                continue
            assert obs in actions, "obstacle color must be encoded by legend"
            action = actions[obs]
            hit_records.append((d, obs, nr, nc, action))
            d = _turn(d, action)
            assert len(hit_records) <= 2, "generator only accepts up to two hits per beam"
        beams.append({"hits": hit_records, "diagonal_steps": diagonal_steps})
    return out, beams


def _simulate_legacy(base: np.ndarray, actions: dict[int, str], bg: int) -> np.ndarray:
    """Original-task semantics: left=CCW 90, right=clockwise 90."""
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    out = base.copy()
    h, w = out.shape
    dirs4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]

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
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
            assert state not in seen
            seen.add(state)
            dr, dc = dirs4[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break
            obs = obstacle_at(nr, nc)
            if obs is None:
                r, c = nr, nc
                paint(r, c)
                continue
            turn = actions[obs]
            d = (d - 1) % 4 if turn == "L" else _legacy_rot_cw(d)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four teal beams from a square, with legend-driven 90-degree or 45-degree turns.

    Input structure:
    - Top band: a grey legend with four colored vertical bars; a color's bar position now encodes which angle family that obstacle color uses.
    - Below: the cropped main grid with one solid teal square, one dominant background, and several colored obstacle rectangles.

    Rule:
    - Remove the legend from the output.
    - Launch one sz-by-sz teal beam in each cardinal direction from the square.
    - When a beam moves across background, it paints those background cells teal.
    - When a beam meets obstacle color C, consult the legend:
      left bar -> rotate 90 degrees counterclockwise;
      right bar -> take the 45-degree diagonal-toggle bend, meaning a cardinal beam turns into a diagonal beam and a diagonal beam turns back into a cardinal beam.
    - Stop when the next sz-by-sz footprint would leave the grid.

    Key insight: the legend no longer distinguishes two orthogonal bends; it distinguishes a sharp 90-degree left turn from a shallow 45-degree bend that creates visible diagonal staircase segments before returning to straight motion.
    """
    g = np.asarray(grid, dtype=int)

    legend_bg = LEGEND_BG
    rows_with_legend = np.where(np.any(g == legend_bg, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    legend_action = _parse_legend(g, legend_h)

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"
    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])

    out, _beams = _simulate_paths(base, legend_action, bg)
    return out


def _make_legend(legend_colors: list[int], actions: dict[int, str]) -> np.ndarray:
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
    side4: int,
    thickness: int,
    span: tuple[int, int],
    h: int,
    w: int,
) -> tuple[int, int, int, int]:
    s0, s1 = span
    if side4 == _N4:
        return 0, thickness - 1, s0, s1
    if side4 == _S4:
        return h - thickness, h - 1, s0, s1
    if side4 == _W4:
        return s0, s1, 0, thickness - 1
    return s0, s1, w - thickness, w - 1


def _native_interval(side4: int, sr: int, sc: int, sz: int) -> tuple[int, int]:
    return (sc, sc + sz - 1) if side4 in (_N4, _S4) else (sr, sr + sz - 1)


def _lane_positions(h: int, w: int, sz: int, thickness: dict[int, int]) -> dict[int, int]:
    lane: dict[int, int] = {}
    if _N4 in thickness:
        lane[_N4] = thickness[_N4]
    if _S4 in thickness:
        lane[_S4] = h - thickness[_S4] - sz
    if _W4 in thickness:
        lane[_W4] = thickness[_W4]
    if _E4 in thickness:
        lane[_E4] = w - thickness[_E4] - sz
    return lane


def _turn_anchor(side4: int, lane: dict[int, int], sr: int, sc: int) -> tuple[int, int]:
    if side4 == _N4:
        return lane[_N4], sc
    if side4 == _E4:
        return sr, lane[_E4]
    if side4 == _S4:
        return lane[_S4], sc
    return sr, lane[_W4]


def _arrival_interval(
    side4: int,
    action: str,
    lane: dict[int, int],
    sr: int,
    sc: int,
    sz: int,
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    if action == "L":
        target = _ccw_side(side4)
        coord = lane[side4]
        interval = (coord, coord + sz - 1)
        axis_len = w if target in (_N4, _S4) else h
        if interval[0] < 0 or interval[1] >= axis_len:
            return None
        return target, interval

    assert action == "D"
    target = _cw_side(side4)
    if target not in lane:
        return None
    r0, c0 = _turn_anchor(side4, lane, sr, sc)
    dr, dc = _DIRS8[_dir_to_diag(side4)]
    goal = lane[target]
    if target in (_W4, _E4):
        if dc == 0:
            return None
        k = goal - c0 if dc > 0 else c0 - goal
    else:
        if dr == 0:
            return None
        k = goal - r0 if dr > 0 else r0 - goal
    if k < 0:
        return None
    r = r0 + k * dr
    c = c0 + k * dc
    interval = (r, r + sz - 1) if target in (_W4, _E4) else (c, c + sz - 1)
    axis_len = h if target in (_W4, _E4) else w
    if interval[0] < 0 or interval[1] >= axis_len:
        return None
    return target, interval


def _return_interval_after_d_chain(source: int, target: int, lane: dict[int, int], sz: int) -> tuple[int, tuple[int, int]]:
    return source, (lane[target], lane[target] + sz - 1)


def _choose_bounded_span(
    lo_bound: int,
    hi_bound: int,
    required: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Choose a contiguous span inside [lo_bound, hi_bound] covering all required intervals."""
    req0 = min(a for a, _ in required)
    req1 = max(b for _, b in required)
    lo = max(min_len, req1 - req0 + 1)
    hi = min(max_len, hi_bound - lo_bound + 1)
    choices: list[tuple[int, int, int]] = []
    for length in range(lo, hi + 1):
        start0 = max(lo_bound, req1 - length + 1)
        start1 = min(req0, hi_bound - length + 1)
        for start in range(start0, start1 + 1):
            end = start + length - 1
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


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    obstacle_count = 3

    # Canonical floating layout:
    # - obstacle A sits above the source and uses D;
    # - obstacle B sits to the right and also uses D;
    # - obstacle C sits to the left and uses L.
    # This yields a stable N -> NE -> N chain plus a separate left-turn beam.
    t_a = random.randint(2, 4)
    t_b = random.randint(2, 4)
    t_c = random.randint(2, 4)
    len_a = random.randint(2, 4)
    len_b = random.randint(2, 4)
    len_c = random.randint(max(5, 2 * sz), min(10, h - 6))
    gap_n = random.randint(3, 6)
    gap_w = random.randint(max(3, sz), 6)
    k = random.randint(3, 5)

    sc_lo = max(len_a + 2, gap_w + t_c + 2)
    sc_hi = w - (k + sz + t_b) - 3
    if sc_lo > sc_hi:
        return None
    sc = random.randint(sc_lo, sc_hi)

    a_lane_min = max(t_a + 2, k + 2)
    a_lane_max = min(11, h - gap_n - sz - 3)
    if a_lane_min > a_lane_max:
        return None
    a_lane = random.randint(a_lane_min, a_lane_max)
    sr = a_lane + gap_n
    if sr + sz >= h - 2:
        return None

    ar0 = a_lane - t_a
    ac1 = sc
    ac0 = max(sc - len_a + 1, 1)
    if ac0 <= sc - gap_w - 1:
        return None

    br_hit_r = a_lane - k
    if br_hit_r < 1:
        return None
    bc0 = sc + k + sz
    br0 = br_hit_r
    br1 = br0 + len_b - 1

    cc_lane = sc - gap_w
    cc1 = cc_lane - 1
    cc0 = cc1 - t_c + 1
    if ar0 < 1 or cc0 < 1 or br1 >= h - 1 or bc0 + t_b - 1 >= w - 1:
        return None

    cr0_lo = max(a_lane + 1, 1)
    cr0_hi = min(sr, h - len_c - 1)
    if cr0_lo > cr0_hi:
        return None
    cr0 = random.randint(cr0_lo, cr0_hi)
    cr1 = cr0 + len_c - 1
    if not (cr0 <= sr <= cr1 - sz + 1):
        return None

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    obstacle_colors = random.sample(legend_colors, obstacle_count)
    bg = random.choice([c for c in BG_COLOR_POOL if c not in legend_colors])

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL
    occupied = np.zeros((h, w), dtype=bool)
    rect_specs = [
        ((_N4, obstacle_colors[0], "D"), (ar0, ar0 + t_a - 1, ac0, ac1)),
        ((_E4, obstacle_colors[1], "D"), (br0, br1, bc0, bc0 + t_b - 1)),
        ((_W4, obstacle_colors[2], "L"), (cr0, cr1, cc0, cc1)),
    ]

    side_action: dict[int, str] = {}
    side_color: dict[int, int] = {}
    for (side, color, action), (r0, r1, c0, c1) in rect_specs:
        region = occupied[r0 : r1 + 1, c0 : c1 + 1]
        if region.any():
            return None
        if not (r1 < sr - 1 or r0 > sr + sz or c1 < sc - 1 or c0 > sc + sz):
            return None
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        base[r0 : r1 + 1, c0 : c1 + 1] = color
        side_action[side] = action
        side_color[side] = color

    vals, cnts = np.unique(base, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    present = set(side_action)
    actions = {side_color[s]: side_action[s] for s in present}
    for color in legend_colors:
        actions.setdefault(color, random.choice(["L", "D"]))

    try:
        expected, beams = _simulate_paths(base, actions, bg)
    except AssertionError:
        return None

    hit_counts = [len(beam["hits"]) for beam in beams]
    hit_colors = {hit[1] for beam in beams for hit in beam["hits"]}
    diagonal_steps = [int(beam["diagonal_steps"]) for beam in beams]
    if any(c > 2 for c in hit_counts):
        return None
    if set(side_color.values()) - hit_colors:
        return None
    if sum(hit_counts) != obstacle_count:
        return None

    used_actions = {actions[hit[1]] for beam in beams for hit in beam["hits"]}
    if "D" not in used_actions or "L" not in used_actions:
        return None
    if max(diagonal_steps) < 2:
        return None
    if len(beams[_N4]["hits"]) < 2 or int(beams[_N4]["diagonal_steps"]) < 2:
        return None

    legacy_actions = {color: ("L" if action == "L" else "R") for color, action in actions.items()}
    try:
        legacy_out = _simulate_legacy(base, legacy_actions, bg)
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
    raise RuntimeError("failed to generate e87109e9 combo input")
