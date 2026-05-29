# Combo variant of cb2d8a2c (index 16)
# combo_spec: D9/0+D15/0
# Name: 蓝色擦除+路径削弱障碍

import heapq
import itertools
import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

BG = 8
BLUE = 1
RED = 2
GREEN = 3

# Design analysis for the fused rule:
# - Concrete behavior: keep the original pathfinding rule, then apply two erasures.
#   Example in canonical top-start orientation: suppose a left-attached horizontal bar
#   has blue_count=2 and length=8. In the original task, the green path rises to the
#   keep-out boundary, slides horizontally above the bar, and the output recolors the
#   bar's blue cells to red. In this combo, the same path is found first, but then
#   every blue cell is erased to background and every bar cell whose Manhattan distance
#   to the path is <= blue_count+1 is also erased. Because the path can never get
#   closer than blue_count+1, this removes exactly the bar segment that the detour
#   visibly "skims", shortening the bar from its open side.
# - Complexity calibration: the erosion threshold is fixed at blue_count+1 rather than
#   something looser. A smaller threshold would erase nothing; a much larger threshold
#   would wipe whole bars and blur the correspondence between path and damage. This
#   choice makes the blue-count radius still matter while producing a clear extra
#   visual effect beyond D9/0.
# - Hidden assumptions checked against the original solver: bars remain 1-cell-thick
#   4-connected components; their blue count is still read before any erasure; the
#   path is still solved against the intact bars/keep-out zones, so the obstacle
#   encoding is not lost. The new assumption is that each bar should lose at least one
#   cell in generated samples; the generator enforces that by comparing against the
#   original output and rejecting degenerate cases.
# - Placement legality and observability: the generator preserves the original spacing
#   constraints so bars never overlap and the path always exists. It also validates
#   that the combo output differs from the original behavior, that blue disappears, and
#   that each bar loses pixels, keeping the fused effect visible in a few examples.


def _start_and_goal(grid: np.ndarray) -> tuple[tuple[int, int], int, str, int]:
    start_pts = np.argwhere(grid == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])

    if sr == 0:
        return (sr, sc), 2, "row", grid.shape[0] - 1
    if sr == grid.shape[0] - 1:
        return (sr, sc), 0, "row", 0
    if sc == 0:
        return (sr, sc), 1, "col", grid.shape[1] - 1
    if sc == grid.shape[1] - 1:
        return (sr, sc), 3, "col", 0
    raise AssertionError("green start must be on a border edge")


def _is_goal(goal_kind: str, goal_val: int, r: int, c: int) -> bool:
    return (r == goal_val) if goal_kind == "row" else (c == goal_val)


def _extract_bars(grid: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray]:
    bar_mask = (grid == BLUE) | (grid == RED)
    bars: list[dict[str, object]] = []
    for comp in components_mask(bar_mask, connectivity=4):
        cells = [(int(r), int(c)) for r, c in comp]
        blue_count = sum(1 for r, c in cells if grid[r, c] == BLUE)
        bars.append({"cells": cells, "blue_count": blue_count})
    return bars, bar_mask


def _build_forbidden(shape: tuple[int, int], bars: list[dict[str, object]]) -> np.ndarray:
    h, w = shape
    forbidden = np.zeros((h, w), dtype=bool)
    for bar in bars:
        radius = int(bar["blue_count"])
        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in bar["cells"]:
            dist[r, c] = 0
            q.append((r, c))

        while q:
            r, c = q.popleft()
            d = int(dist[r, c])
            forbidden[r, c] = True
            if d == radius:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] == -1:
                    dist[rr, cc] = d + 1
                    q.append((rr, cc))
    return forbidden


def _find_path(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[dict[str, object]]]:
    g = np.array(grid, copy=False)
    (sr, sc), init_dir, goal_kind, goal_val = _start_and_goal(g)
    bars, bar_mask = _extract_bars(g)

    blocked = _build_forbidden(g.shape, bars) | bar_mask
    blocked[sr, sc] = False

    inf = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]

    goal_state: tuple[int, int, int] | None = None
    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), inf):
            continue
        if _is_goal(goal_kind, goal_val, r, c) and (r, c) != (sr, sc):
            goal_state = (r, c, dprev)
            break
        for nd, (dr, dc) in enumerate(_DIRS_URDL):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                continue
            if blocked[rr, cc]:
                continue
            ncost = (steps + 1, turns + (0 if nd == dprev else 1))
            key = (rr, cc, nd)
            if ncost < best_cost.get(key, inf):
                best_cost[key] = ncost
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost, rr, cc, nd))

    assert goal_state is not None, "a valid path to the opposite edge must exist"
    path: list[tuple[int, int]] = []
    st = goal_state
    while True:
        r, c, _ = st
        path.append((r, c))
        if (r, c) == (sr, sc):
            break
        st = prev[st]
    path.reverse()
    return path, bars


def _paint_path(base: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(base, copy=True)
    for r, c in path:
        if out[r, c] == BG:
            out[r, c] = GREEN
    return out


def _bar_path_distance(cells: list[tuple[int, int]], path: list[tuple[int, int]]) -> np.ndarray:
    dists = np.empty(len(cells), dtype=int)
    for i, (r, c) in enumerate(cells):
        dists[i] = min(abs(r - pr) + abs(c - pc) for pr, pc in path)
    return dists


def _baseline_output(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    path, bars = _find_path(grid)
    out = _paint_path(grid, path)
    out[out == BLUE] = RED
    return out, bars


def _combo_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, object]:
    baseline, bars = _baseline_output(inp)
    per_bar_erased: list[int] = []
    for bar in bars:
        erased = sum(1 for r, c in bar["cells"] if out[r, c] == BG)
        per_bar_erased.append(erased)
    return {
        "changed_vs_original": not np.array_equal(baseline, out),
        "blue_remaining": int(np.sum(out == BLUE)),
        "blue_erased": int(np.sum(inp == BLUE)),
        "extra_red_erased": int(np.sum((baseline == RED) & (out == BG))),
        "per_bar_erased": per_bar_erased,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route the green path as before, but let the detour scrub away both blue markers and the bar cells it skims.

    Input structure:
    - bg = azure (8). Each obstacle is a 1-cell-thick red/blue bar attached to one side wall.
    - A single green (3) border pixel marks the path's entry edge.
    - Blue (1) cells embedded in a bar still encode that bar's keep-out radius.

    Rule:
    - First solve the same shortest orthogonal path from the green start to the opposite edge while staying farther than each bar's blue-count radius from that bar.
    - Paint the resulting path green on background cells only.
    - Then erase all blue cells to background, and also erase any bar cell whose Manhattan distance to the path is at most blue_count + 1 for its own bar.

    Key insight: The blue cells are used only as hidden routing instructions. After the path is inferred, they vanish, and the path's closest pass trims away the obstacle segment it had to skirt.
    """
    g = np.array(grid, copy=True)
    path, bars = _find_path(g)
    out = _paint_path(g, path)

    out[out == BLUE] = BG
    for bar in bars:
        cells = bar["cells"]
        reach = int(bar["blue_count"]) + 1
        dists = _bar_path_distance(cells, path)
        for (r, c), dist in zip(cells, dists):
            if dist <= reach:
                out[r, c] = BG
    return out


# --------------- generator ---------------

# Grid size bounds (rows x cols from spec: 12-29 x 13-26)
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
# Bar counts and blue-per-bar range
BAR_COUNT_LO, BAR_COUNT_HI = 1, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
# Bar length bounds: absolute min/max and cross-axis percentage bounds
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
# Generation retry limit
MAX_GEN_RETRIES = 400


def _min_main_len(blues: tuple[int, ...]) -> int:
    last_bar = blues[0] + 2
    for a, b in zip(blues, blues[1:]):
        last_bar += a + b + 2
    return last_bar + 2


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_blues(count: int, main_hi: int) -> tuple[int, ...]:
    choices = [
        combo
        for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count)
        if _min_main_len(combo) <= main_hi
    ]
    return random.choice(choices)


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, 2 * blue_count - 1, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, cross_size - blue_count - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int) -> None:
    w = grid.shape[1]
    colors = np.full(length, RED, dtype=int)
    colors[:2 * blue_count - 1:2] = BLUE
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, w - length:] = colors[::-1]


def _canonical_grid(
    main_size: int,
    cross_size: int,
    blues: tuple[int, ...],
    first_side: int,
) -> np.ndarray | None:
    count = len(blues)
    slack = main_size - _min_main_len(blues)
    if slack < 0:
        return None

    extras = _random_composition(slack, count + 1)
    row = blues[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, count):
        row += blues[i - 1] + blues[i] + 2 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    lengths = []
    for row, blue_count, side in zip(rows, blues, ((first_side + i) % 2 for i in range(count))):
        bounds = _length_bounds(cross_size, blue_count)
        if bounds is None:
            return None
        length = random.randint(*bounds)
        lengths.append(length)
        _paint_bar(grid, row, side, length, blue_count)

    first_len = lengths[0]
    if first_side == 0:
        start_col = random.randint(1, first_len - 2)
    else:
        start_col = random.randint(cross_size - first_len + 1, cross_size - 2)
    grid[0, start_col] = GREEN
    return grid


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


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
    blues = _choose_blues(count, main_hi)
    main_size = random.randint(max(main_lo, _min_main_len(blues)), main_hi)
    cross_size = random.randint(cross_lo, cross_hi)
    first_side = random.randrange(2)

    canonical = _canonical_grid(main_size, cross_size, blues, first_side)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    try:
        out = main(inp)
        if not _reaches_opposite_edge(out, edge):
            return None
        if not np.array_equal(main(inp), out):
            return None
        metrics = _combo_metrics(inp, out)
        if metrics["blue_remaining"] != 0:
            return None
        if not metrics["changed_vs_original"]:
            return None
        if int(metrics["extra_red_erased"]) < count:
            return None
        if any(erased <= 0 for erased in metrics["per_bar_erased"]):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c combo input")
