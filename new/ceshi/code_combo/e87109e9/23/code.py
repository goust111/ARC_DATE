# Combo variant of e87109e9 (index 23)
# combo_spec: D14/1+D16/0
# Name: 环绕网格加链式射线

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask


# Design analysis for the combo variant
# -------------------------------------
# fused_rule:
#   - Keep the D14 starter-strip cue so the input visibly selects one launch side.
#   - Replace the original 4 independent beams with one chained ray.
#   - Replace edge termination with torus-style wrap on the beam's top-left state grid.
# behavior_trace:
#   - Original e87109e9: from one teal square, launch N/E/S/W independently; each path ends
#     when its sz x sz footprint would leave the board.
#   - D14/1 alone: only one starter-marked launch direction survives; later turns are still
#     legend-controlled, but the path remains planar and stops at the border.
#   - This combo: still only one starter-marked launch direction, but after a free step crosses
#     a border the footprint reappears on the opposite side and continues until the next obstacle.
#     The path is therefore one polyline on a torus, not four planar rays.
#   - Concrete example: with a north starter, the ray can climb to row 0, wrap to the bottom,
#     travel upward there for several more steps, then meet an interior obstacle and turn east.
# complexity_calibration:
#   - The starter strip is necessary; without it, a human cannot distinguish "single chain"
#     from the original 4-way emission.
#   - Obstacles are kept as longer rectangles rather than tiny point markers so they remain
#     visually salient, but they are moved off the border so wrap-around produces visible motion
#     on both sides of the seam instead of an immediate hidden collision.
#   - The generator forces a 4-turn loop with at least one wrapped free move, so samples cannot
#     silently collapse back to the no-wrap D14 behavior.
# implicit_assumption_audit:
#   - Original _find_teal_square assumed all teal cells belonged to one square; the combo keeps
#     D14's split source (square + starter strip), so source parsing must still separate them.
#   - Original beam logic used in-bounds termination. That is invalid here because legal states
#     live on a torus of top-left positions of size (H-sz+1) x (W-sz+1).
#   - Original cycle detection asserted that cycles are impossible. That is false on a torus;
#     the solver now treats the first repeated (r,c,d) state as the natural stopping point.
#   - Obstacle lookup still ignores teal because the starter strip lies on the intended path.
# placement_legality:
#   - The generator plans a rectangular 4-corner cycle in torus state space, then places one
#     obstacle just beyond each corner in the incoming direction.
#   - Each obstacle rectangle is oriented perpendicular to the incoming motion, so the current
#     footprint stays clear while the next footprint collides.
#   - Spans explicitly avoid the other parallel lane, preventing accidental early collisions.
# observability_and_cleanliness:
#   - Input/output pairs show both combo dimensions directly: only one launch side is marked,
#     and the output ray visibly appears on opposite sides of the board before later turns.
#   - Obstacles stay monochrome and non-overlapping; teal is confined to the square, starter,
#     and the eventual wrapped path.
# new_solver_assumptions_checked_by_generator:
#   - exactly one solid teal square and exactly one aligned starter strip exist
#   - every obstacle footprint hit by the ray contains a single non-bg non-teal color
#   - the first repeated (r,c,d) state happens only after the intended 4-turn loop is traced
#   - at least one wrapped free move occurs, and the wrap-aware output differs from no-wrap


_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)


# Palettes
LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
LEGEND_BG = 5
TEAL = 8

# Geometry
GRID_H_RANGE = (19, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]
TEAL_SZ2_PROB = 0.75
STARTER_GAP = 1
OBSTACLE_THICKNESS_RANGE = (2, 4)
OBSTACLE_LEN_RANGE = (5, 11)
SEGMENT_MIN = 4
MAX_RETRIES = 900


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    """Read the top legend and map obstacle color -> turn direction."""
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    turn: dict[int, str] = {}
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        interior = legend[1 : legend_h - 1, a + 1 : b]
        colors = set(map(int, np.unique(interior)))
        colors.discard(bg)
        colors.discard(0)
        assert len(colors) == 1, "each legend cell must encode exactly one color"
        col = int(next(iter(colors)))
        x = int(np.where(interior == col)[1][0])
        turn[col] = "L" if x <= (interior.shape[1] - 1) // 2 else "R"
    assert len(turn) == 4, "legend must encode four turn colors"
    return turn


def _find_source(g: np.ndarray, teal: int = TEAL) -> tuple[int, int, int, int]:
    """
    Return (r0, c0, size, start_dir) for the teal square plus detached starter strip.
    The starter strip is separated by exactly one bg cell and points to the only launch direction.
    """
    comps = list(components_mask(g == teal, connectivity=4))
    assert len(comps) == 2, "main grid must contain one teal square and one teal starter strip"

    square_candidates: list[tuple[int, int, int]] = []
    for cells in comps:
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = int(r1 - r0 + 1)
        ww = int(c1 - c0 + 1)
        if hh != ww or hh < 2:
            continue
        if int(np.sum(g[r0 : r1 + 1, c0 : c1 + 1] == teal)) != hh * ww:
            continue
        square_candidates.append((int(r0), int(c0), hh))
    assert len(square_candidates) == 1, "there must be exactly one solid teal source square"

    sr, sc, sz = square_candidates[0]
    for cells in comps:
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = int(r1 - r0 + 1)
        ww = int(c1 - c0 + 1)
        area = int(cells.shape[0])
        if (int(r0), int(c0), hh) == square_candidates[0]:
            continue
        assert area == sz, "starter strip must have the same length as the square side"
        if hh == 1 and ww == sz and int(r0) == sr - 2 and int(c0) == sc:
            return sr, sc, sz, _N
        if hh == 1 and ww == sz and int(r0) == sr + sz + 1 and int(c0) == sc:
            return sr, sc, sz, _S
        if hh == sz and ww == 1 and int(c0) == sc - 2 and int(r0) == sr:
            return sr, sc, sz, _W
        if hh == sz and ww == 1 and int(c0) == sc + sz + 1 and int(r0) == sr:
            return sr, sc, sz, _E
    raise AssertionError("starter strip must align with exactly one side of the source square")


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def _state_dims(shape: tuple[int, int], sz: int) -> tuple[int, int]:
    h, w = map(int, shape)
    rows = h - sz + 1
    cols = w - sz + 1
    assert rows > 0 and cols > 0, "beam footprint must fit inside the main grid"
    return rows, cols


def _wrapped_step(r: int, c: int, d: int, shape: tuple[int, int], sz: int) -> tuple[int, int, bool]:
    rows, cols = _state_dims(shape, sz)
    dr, dc = _DIRS[d]
    nr = (r + dr) % rows
    nc = (c + dc) % cols
    wrapped = (
        (d == _N and r == 0)
        or (d == _S and r == rows - 1)
        or (d == _W and c == 0)
        or (d == _E and c == cols - 1)
    )
    return nr, nc, wrapped


def _starter_cells(sr: int, sc: int, sz: int, start_dir: int) -> list[tuple[int, int]]:
    if start_dir == _N:
        return [(sr - 2, c) for c in range(sc, sc + sz)]
    if start_dir == _S:
        return [(sr + sz + 1, c) for c in range(sc, sc + sz)]
    if start_dir == _W:
        return [(r, sc - 2) for r in range(sr, sr + sz)]
    return [(r, sc + sz + 1) for r in range(sr, sr + sz)]


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
    cutoff = lengths[min(len(lengths) - 1, max(0, len(lengths) // 2))]
    if random.random() < 0.5:
        pool = [(start, end) for length, start, end in choices if length <= cutoff]
    else:
        pool = [(start, end) for _, start, end in choices]
    return random.choice(pool)


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


def _obstacle_color(grid: np.ndarray, r: int, c: int, sz: int, bg: int) -> int | None:
    cols = set(map(int, np.unique(grid[r : r + sz, c : c + sz])))
    cols.discard(bg)
    cols.discard(TEAL)
    if not cols:
        return None
    assert len(cols) == 1, "beam footprint must not hit mixed obstacle colors"
    return int(next(iter(cols)))


def _paint_block(grid: np.ndarray, r: int, c: int, sz: int, bg: int) -> None:
    blk = grid[r : r + sz, c : c + sz]
    blk[blk == bg] = TEAL
    grid[r : r + sz, c : c + sz] = blk


def _simulate_chain(
    base: np.ndarray,
    turns: dict[int, str],
    bg: int,
    *,
    wrap: bool,
) -> tuple[np.ndarray, list[tuple[int, int, int, int, bool]], int]:
    sr, sc, sz, start_dir = _find_source(base, teal=TEAL)
    out = base.copy()
    h, w = out.shape
    hits: list[tuple[int, int, int, int, bool]] = []
    visible_wraps = 0

    def step(r: int, c: int, d: int) -> tuple[int, int, bool] | None:
        if wrap:
            return _wrapped_step(r, c, d, out.shape, sz)
        dr, dc = _DIRS[d]
        nr, nc = r + dr, c + dc
        if not (0 <= nr and 0 <= nc and nr + sz - 1 < h and nc + sz - 1 < w):
            return None
        return nr, nc, False

    r, c, d = int(sr), int(sc), int(start_dir)
    seen: set[tuple[int, int, int]] = set()
    while True:
        state = (r, c, d)
        if state in seen:
            break
        seen.add(state)

        nxt = step(r, c, d)
        if nxt is None:
            break
        nr, nc, wrapped = nxt
        obs = _obstacle_color(out, nr, nc, sz, bg)
        if obs is None:
            r, c = nr, nc
            _paint_block(out, r, c, sz, bg)
            if wrapped:
                visible_wraps += 1
            continue
        assert obs in turns, "obstacle color must be encoded by the legend"
        hits.append((d, obs, nr, nc, wrapped))
        d = _rot(d, turns[obs])

    return out, hits, visible_wraps


def _rect_from_hit(
    incoming_dir: int,
    hit_r: int,
    hit_c: int,
    sz: int,
    thickness: int,
    span: tuple[int, int],
) -> tuple[int, int, int, int]:
    s0, s1 = span
    if incoming_dir == _N:
        return hit_r - thickness + 1, hit_r, s0, s1
    if incoming_dir == _S:
        return hit_r + sz - 1, hit_r + sz + thickness - 2, s0, s1
    if incoming_dir == _W:
        return s0, s1, hit_c - thickness + 1, hit_c
    return s0, s1, hit_c + sz - 1, hit_c + sz + thickness - 2


def _starter_fits(sr: int, sc: int, sz: int, start_dir: int, h: int, w: int) -> bool:
    cells = _starter_cells(sr, sc, sz, start_dir)
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _advance_many(
    r: int,
    c: int,
    d: int,
    steps: int,
    shape: tuple[int, int],
    sz: int,
) -> tuple[int, int, bool]:
    wrapped_any = False
    for _ in range(steps):
        r, c, wrapped = _wrapped_step(r, c, d, shape, sz)
        wrapped_any |= wrapped
    return r, c, wrapped_any


def _segment_wraps(
    r: int,
    c: int,
    d: int,
    steps: int,
    shape: tuple[int, int],
    sz: int,
) -> bool:
    _, _, wrapped = _advance_many(r, c, d, steps, shape, sz)
    return wrapped


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    if h < sz + 5 or w < sz + 5:
        return None

    start_dir = random.choice([_N, _E, _S, _W])
    turn_dir = random.choice(["L", "R"])
    dirs = [start_dir]
    for _ in range(3):
        dirs.append(_rot(dirs[-1], turn_dir))

    rows, cols = _state_dims((h, w), sz)
    if rows <= SEGMENT_MIN or cols <= SEGMENT_MIN:
        return None

    sr = random.randint(0, rows - 1)
    sc = random.randint(0, cols - 1)
    if not _starter_fits(sr, sc, sz, start_dir, h, w):
        return None

    len_a_hi = max(SEGMENT_MIN, rows - 1)
    len_b_hi = max(SEGMENT_MIN, cols - 1)
    len_a = random.randint(SEGMENT_MIN, len_a_hi)
    len_b = random.randint(SEGMENT_MIN, len_b_hi)
    if len_a % rows == 0 or len_b % cols == 0:
        return None

    starts: list[tuple[int, int]] = [(sr, sc)]
    corners: list[tuple[int, int]] = []
    hit_positions: list[tuple[int, int]] = []
    segment_lengths = [len_a, len_b, len_a, len_b]
    any_segment_wraps = False
    cur_r, cur_c = sr, sc
    for d, seg_len in zip(dirs, segment_lengths, strict=True):
        any_segment_wraps |= _segment_wraps(cur_r, cur_c, d, seg_len, (h, w), sz)
        cur_r, cur_c, _ = _advance_many(cur_r, cur_c, d, seg_len, (h, w), sz)
        corners.append((cur_r, cur_c))
        hit_r, hit_c, _ = _wrapped_step(cur_r, cur_c, d, (h, w), sz)
        hit_positions.append((hit_r, hit_c))
        starts.append((cur_r, cur_c))
    starts = starts[:-1]
    if not any_segment_wraps or corners[-1] != (sr, sc):
        return None

    thicknesses = [random.randint(*OBSTACLE_THICKNESS_RANGE) for _ in range(4)]

    rects: list[tuple[int, int, int, int]] = []
    for i, (d, (hr, hc), thick) in enumerate(zip(dirs, hit_positions, thicknesses, strict=True)):
        if d in (_N, _S):
            required = [(hc, hc + sz - 1)]
            excluded = []
            for j, d_other in enumerate(dirs):
                if j == i or d_other not in (_N, _S):
                    continue
                lane = starts[j][1]
                excluded.append((lane, lane + sz - 1))
            span = _choose_span(
                w,
                required,
                excluded,
                max(OBSTACLE_LEN_RANGE[0], sz + 3),
                min(OBSTACLE_LEN_RANGE[1], w),
            )
        else:
            required = [(hr, hr + sz - 1)]
            excluded = []
            for j, d_other in enumerate(dirs):
                if j == i or d_other not in (_E, _W):
                    continue
                lane = starts[j][0]
                excluded.append((lane, lane + sz - 1))
            span = _choose_span(
                h,
                required,
                excluded,
                max(OBSTACLE_LEN_RANGE[0], sz + 3),
                min(OBSTACLE_LEN_RANGE[1], h),
            )
        if span is None:
            return None
        r0, r1, c0, c1 = _rect_from_hit(d, hr, hc, sz, thick, span)
        if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
            return None
        rects.append((r0, r1, c0, c1))

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    obstacle_colors = random.sample(legend_colors, 4)
    turns = {color: turn_dir for color in obstacle_colors}
    for color in legend_colors:
        turns.setdefault(color, random.choice(["L", "R"]))
    bg = random.choice([c for c in BG_COLOR_POOL if c not in legend_colors])

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL
    for r, c in _starter_cells(sr, sc, sz, start_dir):
        base[r, c] = TEAL

    occupied = base != bg
    for color, (r0, r1, c0, c1) in zip(obstacle_colors, rects, strict=True):
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
            return None
        base[r0 : r1 + 1, c0 : c1 + 1] = color
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True

    vals, cnts = np.unique(base, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        expected, hits, visible_wraps = _simulate_chain(base, turns, bg, wrap=True)
        nowrap, nowrap_hits, _ = _simulate_chain(base, turns, bg, wrap=False)
    except AssertionError:
        return None

    if len(hits) != 4 or len(nowrap_hits) >= len(hits):
        return None
    if visible_wraps < 1:
        return None
    if not np.any(expected != nowrap):
        return None
    if [d for d, _, _, _, _ in hits] != dirs:
        return None
    if [color for _, color, _, _, _ in hits] != obstacle_colors:
        return None

    legend = _make_legend(legend_colors, turns)
    inp = np.vstack([legend, base])
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace one starter-marked teal ray on a wrap-around grid, turning by legend colors.

    Input structure:
    - The top band is a grey legend split into four cells; each colored vertical bar sits on the
      left or right side of its cell to encode a counterclockwise or clockwise turn.
    - Below it is the main grid with one solid teal square, one detached teal starter strip that
      selects the launch side, and several colored rectangular obstacles.

    Rule:
    - Remove the legend in the output.
    - Launch exactly one sz-wide ray from the square in the direction marked by the starter strip.
    - Each free step paints teal only onto background cells.
    - If the next sz x sz footprint would meet obstacle color C, rotate left or right according to
      C's legend entry and keep tracing the same ray.
    - If the next free step would leave the board, wrap that footprint to the opposite side and
      continue on the torus; stop only when the full (position, direction) state repeats.

    Key insight: The starter strip tells you which one of the four cardinal directions is active,
    and the border is no longer terminal: you must follow a single legend-driven polyline through
    wrap-around space until it cycles.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must stay in the top band"

    legend_turn = _parse_legend(g, legend_h)

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    sr, sc, sz, d = _find_source(base, teal=TEAL)

    out = base.copy()
    r, c = int(sr), int(sc)
    seen: set[tuple[int, int, int]] = set()
    while True:
        state = (r, c, d)
        if state in seen:
            break
        seen.add(state)

        nr, nc, _ = _wrapped_step(r, c, d, out.shape, sz)
        obs = _obstacle_color(out, nr, nc, sz, bg)
        if obs is None:
            r, c = nr, nc
            _paint_block(out, r, c, sz, bg)
            continue

        assert obs in legend_turn, "obstacle color must be encoded by the legend"
        d = _rot(d, legend_turn[obs])

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e87109e9 combo input")
