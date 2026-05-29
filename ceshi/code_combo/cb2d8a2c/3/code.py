# Combo variant of cb2d8a2c (index 3)
# combo_spec: D1/0+D14/0
# Name: 对角线路径+切比雪夫距离

import heapq
import random

import numpy as np

from arc_utils import components_mask

BG = 8
BLUE = 1
RED = 2
GREEN = 3

# Diagonal directions only: NW, NE, SE, SW.
_DIRS_DIAG = [(-1, -1), (-1, 1), (1, 1), (1, -1)]

# --- Tunable parameters (kept close to the original task bounds) ---
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 1, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 400
MAX_LAYOUT_RETRIES = 120


def _segment_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    dr = r1 - r0
    dc = c1 - c0
    assert abs(dr) == abs(dc), "segment must be a 45-degree diagonal"
    sr = 1 if dr > 0 else -1
    sc = 1 if dc > 0 else -1
    return [(r0 + k * sr, c0 + k * sc) for k in range(abs(dr) + 1)]


def _paint_path(out: np.ndarray, path: list[tuple[int, int]]) -> None:
    for r, c in path:
        if out[r, c] == BG:
            out[r, c] = GREEN


def _extract_path(
    start: tuple[int, int],
    goal_state: tuple[int, int, int],
    prev: dict[tuple[int, int, int], tuple[int, int, int]],
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    st = goal_state
    while True:
        r, c, _ = st
        path.append((r, c))
        if (r, c) == start:
            break
        st = prev[st]
    path.reverse()
    return path


def _chebyshev_blocked(grid: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    bar_mask = (grid == BLUE) | (grid == RED)
    blocked = bar_mask.copy()
    for comp in components_mask(bar_mask, connectivity=4):
        blue_count = int(sum(1 for r, c in comp if grid[r, c] == BLUE))
        for r, c in comp:
            r0 = max(0, int(r) - blue_count)
            r1 = min(h - 1, int(r) + blue_count)
            c0 = max(0, int(c) - blue_count)
            c1 = min(w - 1, int(c) + blue_count)
            blocked[r0 : r1 + 1, c0 : c1 + 1] = True
    blocked[start] = False
    return blocked


def _start_goal_setup(h: int, w: int, sr: int, sc: int) -> tuple[list[int], str, int]:
    if sr == 0:
        return [2, 3], "row", h - 1
    if sr == h - 1:
        return [0, 1], "row", 0
    if sc == 0:
        return [1, 2], "col", w - 1
    if sc == w - 1:
        return [0, 3], "col", 0
    raise AssertionError("green start must be on a border edge")


# Design analysis for the fusion:
# 1. Concrete behavior:
#    In the original task, a top-start example with a left-wall bar would route by
#    going straight down, then making a horizontal detour outside a Manhattan
#    diamond. Here the same bar produces a square Chebyshev keepout, so the clean
#    route is a diagonal bounce: the path approaches the square's exposed top
#    corner, turns once, then continues on the opposite diagonal. The visible
#    output becomes a zigzag of 45-degree segments rather than orthogonal elbows.
# 2. Complexity calibration:
#    The path still has a small discrete vocabulary: 1-4 bars, blue counts 1-4,
#    alternating wall attachments, and one diagonal turn per bar. This keeps the
#    fusion visibly different from the source task without creating too many
#    unconstrained cases for a human to infer from 2-4 samples.
# 3. Hidden assumptions broken by D1/0 + D14/0:
#    - `neighbors4` Manhattan expansion no longer matches the geometry; it is
#      replaced by square dilation from each bar cell.
#    - A single orthogonal `init_dir` is no longer valid; the search must seed the
#      two inward diagonals for the chosen border.
#    - The original start placement inside the first bar's coverage is not stable
#      under corner-bounce geometry. The generator therefore switches to
#      path-first placement around intended turn points and then validates that the
#      solver reproduces the exact expected path.
# 4. Generator legality:
#    Bars are derived from diagonal turn points, so every bar row/length is chosen
#    to keep the intended turn exactly one cell outside the square keepout corner.
#    The generator checks row ordering, component separation, exact solver/output
#    equality, and opposite-edge reachability before accepting a sample.
# 5. Visual observability:
#    The fused behavior is directly visible: green segments are diagonal-only, the
#    path alternates between left/right turns, and blue-count changes visibly widen
#    square keepout zones because turns happen farther above the bars.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the border green marker to the opposite edge using only 45-degree diagonal motion, while respecting square keepout zones around the obstacle bars.

    Input structure:
    - Background is azure (8). Obstacle bars are made from red/blue cells (2/1).
    - Exactly one green (3) pixel sits on a border edge and marks the path start.

    Rule:
    - Each red/blue bar is still an obstacle, and its blue-pixel count sets the keepout radius.
    - The radius is measured with Chebyshev distance, so each bar grows into a square-edged forbidden region rather than a Manhattan diamond.
    - The green route travels with diagonal steps only, choosing the shortest valid corner-bounce path to the opposite edge.
    - After routing, all blue pixels recolor to red.

    Key insight:
    Blue pixels now encode the size of a square danger zone. Because diagonal neighbors also count as distance 1, the green path naturally skims the exposed corners of those squares instead of making orthogonal detours.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])

    init_dirs, goal_kind, goal_val = _start_goal_setup(h, w, sr, sc)

    def is_goal(r: int, c: int) -> bool:
        return (r == goal_val) if goal_kind == "row" else (c == goal_val)

    blocked = _chebyshev_blocked(g, (sr, sc))

    inf = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = []

    for d in init_dirs:
        state = (sr, sc, d)
        best_cost[state] = (0, 0)
        heapq.heappush(heap, ((0, 0), sr, sc, d))

    goal_state: tuple[int, int, int] | None = None
    while heap:
        (turns, steps), r, c, dprev = heapq.heappop(heap)
        if (turns, steps) != best_cost.get((r, c, dprev), inf):
            continue
        if is_goal(r, c) and (r, c) != (sr, sc):
            goal_state = (r, c, dprev)
            break
        for nd, (dr, dc) in enumerate(_DIRS_DIAG):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if blocked[rr, cc]:
                continue
            ncost = (turns + (0 if nd == dprev else 1), steps + 1)
            key = (rr, cc, nd)
            if ncost < best_cost.get(key, inf):
                best_cost[key] = ncost
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost, rr, cc, nd))

    assert goal_state is not None, "a valid diagonal path to the opposite edge must exist"

    out = g.copy()
    path = _extract_path((sr, sc), goal_state, prev)
    _paint_path(out, path)
    out[out == BLUE] = RED
    return out


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, 2 * blue_count - 1, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, cross_size - blue_count - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int) -> None:
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, grid.shape[1] - length :] = colors[::-1]


def _choose_blues(count: int) -> tuple[int, ...]:
    return tuple(random.randint(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI) for _ in range(count))


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    return bool(np.any(out[:, 0] == GREEN))


def _expected_output(inp: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    _paint_path(out, path)
    out[out == BLUE] = RED
    return out


def _path_turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for (r0, c0), (r1, c1), (r2, c2) in zip(path, path[1:], path[2:]):
        d1 = (r1 - r0, c1 - c0)
        d2 = (r2 - r1, c2 - c1)
        if d1 != d2:
            turns += 1
    return turns


def _bar_turn_column(width: int, side: int, blue_count: int, length: int) -> int:
    if side == 0:
        return length + blue_count
    return width - length - blue_count - 1


def _build_canonical_case(
    main_lo: int,
    main_hi: int,
    width: int,
    blues: tuple[int, ...],
    first_side: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    count = len(blues)
    sides = [(first_side + i) % 2 for i in range(count)]
    bounds = [_length_bounds(width, b) for b in blues]
    if any(b is None for b in bounds):
        return None

    for _ in range(MAX_LAYOUT_RETRIES):
        lengths = [random.randint(lo, hi) for lo, hi in bounds if lo is not None and hi is not None]
        turn_cols = [_bar_turn_column(width, side, blue_count, length) for side, blue_count, length in zip(sides, blues, lengths)]

        ok = True
        for i in range(count - 1):
            diff = abs(turn_cols[i + 1] - turn_cols[i])
            min_gap = 2
            if diff < min_gap:
                ok = False
                break
            if sides[i] == 0 and turn_cols[i + 1] <= turn_cols[i]:
                ok = False
                break
            if sides[i] == 1 and turn_cols[i + 1] >= turn_cols[i]:
                ok = False
                break
        if not ok:
            continue

        first_col = turn_cols[0]
        last_col = turn_cols[-1]
        if sides[0] == 0:
            start_min, start_max = first_col + 1, width - 1
        else:
            start_min, start_max = 0, first_col - 1
        if sides[-1] == 0:
            end_min, end_max = last_col + 1, width - 1
            end_diff_min = blues[-1] + 2
        else:
            end_min, end_max = 0, last_col - 1
            end_diff_min = blues[-1] + 2

        if start_min > start_max or end_min > end_max:
            continue

        inter_diffs = [abs(b - a) for a, b in zip(turn_cols, turn_cols[1:])]
        base_height = 1 + sum(inter_diffs)

        start_candidates = list(range(start_min, start_max + 1))
        end_candidates = list(range(end_min, end_max + 1))
        random.shuffle(start_candidates)
        random.shuffle(end_candidates)

        chosen: tuple[int, int, int, int] | None = None
        for start_col in start_candidates[: min(len(start_candidates), 10)]:
            d_start = abs(first_col - start_col)
            if d_start < 1:
                continue
            for end_col in end_candidates[: min(len(end_candidates), 10)]:
                d_end = abs(end_col - last_col)
                if d_end < end_diff_min:
                    continue
                height = base_height + d_start + d_end
                if main_lo <= height <= main_hi:
                    chosen = (start_col, end_col, d_start, d_end)
                    break
            if chosen is not None:
                break
        if chosen is None:
            continue

        start_col, end_col, d_start, _ = chosen
        turn_rows = [d_start]
        for diff in inter_diffs:
            turn_rows.append(turn_rows[-1] + diff)
        height = turn_rows[-1] + abs(end_col - last_col) + 1

        bar_rows = [turn_r + blue_count + 1 for turn_r, blue_count in zip(turn_rows, blues)]
        if any(not (1 <= row < height - 1) for row in bar_rows):
            continue
        if any(bar_rows[i + 1] - bar_rows[i] < 1 for i in range(count - 1)):
            continue

        grid = np.full((height, width), BG, dtype=int)
        for row, side, length, blue_count in zip(bar_rows, sides, lengths, blues):
            _paint_bar(grid, row, side, length, blue_count)
        if len(components_mask((grid == BLUE) | (grid == RED), connectivity=4)) != count:
            continue

        start = (0, start_col)
        turns = list(zip(turn_rows, turn_cols))
        end = (height - 1, end_col)
        points = [start, *turns, end]

        try:
            path: list[tuple[int, int]] = []
            for (r0, c0), (r1, c1) in zip(points, points[1:]):
                seg = _segment_cells(r0, c0, r1, c1)
                if path:
                    seg = seg[1:]
                path.extend(seg)
        except AssertionError:
            continue

        if _path_turn_count(path) != count:
            continue

        grid[start] = GREEN
        expected = _expected_output(grid, path)
        solved = main(grid)
        if not np.array_equal(solved, expected):
            continue
        if not np.array_equal(main(grid), solved):
            continue
        return grid, expected
    return None


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    count = random.choices([1, 2, 3, 4], weights=[1, 4, 3, 1], k=1)[0]
    blues = _choose_blues(count)
    first_side = random.randrange(2)

    width_choices = list(range(cross_lo, cross_hi + 1))
    random.shuffle(width_choices)
    for width in width_choices:
        built = _build_canonical_case(main_lo, main_hi, width, blues, first_side)
        if built is None:
            continue
        canonical_inp, canonical_out = built
        inp = _orient_grid(canonical_inp, edge)
        out = _orient_grid(canonical_out, edge)
        try:
            solved = main(inp)
        except Exception:
            continue
        if not np.array_equal(solved, out):
            continue
        if not _reaches_opposite_edge(solved, edge):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c diagonal/Chebyshev input")
