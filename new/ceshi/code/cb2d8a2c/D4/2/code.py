# Variant of cb2d8a2c
# Dimension: D4 — 轨迹涂色模式 (trajectory)
# Change: 连续涂色——绿色路径经过的每个背景格子都被涂为绿色，形成连续的绿色线条... → 颜色随障碍条变化

# Design analysis for the D4 variant
# ==================================
# variant_summary:
#   keep_geometry: true
#   changed_stage: output painting only
#   source_task_baseline:
#     reference_solver: 195c6913
#     takeaway: use a small fixed visible palette with repeated category->color mapping;
#       do not randomize colors per instance and do not make the color logic more complex
#       than the underlying spatial rule.
#
# concrete_behavior:
#   original:
#     The shortest low-turn path is computed from the border green seed to the opposite edge.
#     Every visited background cell on that path is painted green.
#   variant:
#     The path geometry stays identical, but each obstacle bar now owns a colored "routing slab".
#     For a vertical trip, the slab is the row interval [bar_row-blue_count-1, bar_row+blue_count+1].
#     For a horizontal trip, the slab is the analogous column interval.
#     Background path cells inside that slab are painted with a fixed palette color determined
#     by the bar's blue_count:
#       1 blue -> yellow(4)
#       2 blue -> gray(5)
#       3 blue -> magenta(6)
#       4 blue -> orange(7)
#     Path cells outside all slabs remain green(3). Blue obstacle pixels still turn red(2).
#
# behavior_trace:
#   small_example:
#     Suppose the path starts at the top and there is a left-attached bar on row 7 with
#     blue_count=2. In the original task, the path segment that turns before row 7, moves
#     sideways around the keep-out region, and straightens after row 7 is all green.
#     In the variant, the very same cells are still the chosen route, but every background
#     path cell whose row lies in [4, 10] is painted gray(5). If the next bar has
#     blue_count=4, the next detour slab switches to orange(7). The difference is visible
#     exactly where the path bends around each obstacle, not in obstacle placement.
#
# complexity_calibration:
#   why_not_per_cell_distance_coloring:
#     Coloring by exact distance-to-bar would create noisy mixed stripes and would be harder
#     to infer from 2-4 ARC examples.
#   why_not_random_palette:
#     A per-instance palette would make the count->color rule impossible to learn across
#     examples. The mapping must be global and fixed.
#   why_generator_changes:
#     Single-bar or single-blue-count cases would often show only one detour color, which is
#     too weak for human induction. The generator therefore uses 2-4 bars and requires at
#     least two distinct blue_count values per sample.
#
# hidden_assumptions_audit:
#   original_main_dependencies:
#     - Dijkstra only depends on blocked cells, not on the final paint color. Safe to reuse.
#     - The path reconstruction loop writes `GREEN` unconditionally on background cells.
#       This is no longer valid because different path cells may need different colors.
#     - Blue->red conversion is still valid because path never enters obstacle cells.
#     - Bar components are assumed to be separated 4-connected lines. The generator already
#       guarantees that, and the variant keeps the same obstacle structure.
#   new_assumptions:
#     - Each bar has a well-defined single main-axis coordinate (one row for horizontal bars,
#       one column for vertical bars). This matches the original generator's 1-cell-thick bars.
#     - Routing slabs from consecutive bars do not overlap. The original spacing constraints
#       imply at least a zero-width or positive gap between slabs:
#       next_bar_main - prev_bar_main >= prev_blue + next_blue + 2.
#       Therefore [main-blue-1, main+blue+1] intervals are disjoint.
#
# placement_legality:
#   - Bars keep the original length, wall-attachment, alternation, and spacing constraints.
#   - Because slab ownership is derived from those same bar coordinates, no extra spatial
#     object must be placed in the input, so there is no new overlap risk.
#   - The green seed remains outside the first slab, and the opposite edge remains outside the
#     last slab, preserving visible green approach/exit segments.
#
# visual_observability:
#   - Humans can pair each detour color with the nearby bar whose blue markers count is visible.
#   - The same blue_count value always yields the same path color.
#   - Different bars in one sample usually show at least two colors, making the variant
#     distinguishable from the original all-green trajectory.
#
# cleanliness_and_inference:
#   - Only the path changes color; obstacles keep their original red/blue encoding and then
#     resolve to red in the output.
#   - The palette uses colors absent from the input obstacles/background, so the detour colors
#     stand out cleanly instead of blending with bar colors.

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
PATH_COLOR_BY_BLUE = {1: 4, 2: 5, 3: 6, 4: 7}


def _collect_bars(
    g: np.ndarray,
    *,
    path_is_vertical: bool,
) -> list[dict[str, object]]:
    """Extract bar components with their blue counts and main-axis coordinate."""
    bar_mask = (g == BLUE) | (g == RED)
    bar_comps = components_mask(bar_mask, connectivity=4)
    bars: list[dict[str, object]] = []
    for comp in bar_comps:
        blue_count = sum(1 for r, c in comp if g[r, c] == BLUE)
        assert blue_count in PATH_COLOR_BY_BLUE, "blue_count must stay within palette range"
        main_values = sorted({int(r if path_is_vertical else c) for r, c in comp})
        assert len(main_values) == 1, "each bar must occupy exactly one main-axis line"
        bars.append(
            {
                "coords": [(int(r), int(c)) for r, c in comp],
                "blue_count": int(blue_count),
                "main": int(main_values[0]),
            }
        )
    return bars


def _build_blocked(
    h: int,
    w: int,
    bars: list[dict[str, object]],
) -> np.ndarray:
    """Build the Manhattan keep-out mask from the extracted bars."""
    forbidden = np.zeros((h, w), dtype=bool)
    bar_mask = np.zeros((h, w), dtype=bool)
    for bar in bars:
        radius = int(bar["blue_count"])
        coords = list(bar["coords"])

        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in coords:
            bar_mask[r, c] = True
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
    return bar_mask | forbidden


def _routing_slabs(
    bars: list[dict[str, object]],
    *,
    axis_limit: int,
) -> list[tuple[int, int, int]]:
    """Main-axis intervals that inherit each bar's path color."""
    slabs: list[tuple[int, int, int]] = []
    for bar in bars:
        blue_count = int(bar["blue_count"])
        main = int(bar["main"])
        lo = max(0, main - blue_count - 1)
        hi = min(axis_limit - 1, main + blue_count + 1)
        slabs.append((lo, hi, PATH_COLOR_BY_BLUE[blue_count]))
    slabs.sort()
    for (_, prev_hi, _), (next_lo, _, _) in zip(slabs, slabs[1:]):
        assert prev_hi < next_lo or prev_hi + 1 == next_lo, "routing slabs must stay disjoint"
    return slabs


def _paint_path(
    out: np.ndarray,
    path: list[tuple[int, int]],
    *,
    path_is_vertical: bool,
    slabs: list[tuple[int, int, int]],
) -> None:
    """Paint background path cells with the obstacle-controlled palette."""
    for r, c in path:
        if out[r, c] != BG:
            continue
        main_value = int(r if path_is_vertical else c)
        color = GREEN
        for lo, hi, slab_color in slabs:
            if lo <= main_value <= hi:
                color = slab_color
                break
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a border-started path to the opposite edge, but color each obstacle bypass using a bar-specific palette color.

    Input structure:
    - bg = azure (8). Red/blue bars (2/1) are 1-cell-thick obstacles attached to alternating side walls.
    - A single green (3) seed sits on one border and marks the path start.

    Rule:
    - A bar's blue-pixel count still sets its Manhattan keep-out radius, so the route geometry is unchanged from the base task.
    - After the shortest low-turn route is found, background path cells near each bar are recolored by that bar's blue-count palette entry instead of all staying green.
    - Path cells outside those bar-local routing slabs stay green, and every blue obstacle pixel becomes red.

    Key insight: The blue markers now carry two linked meanings at once — they set how far the path must detour and also determine the color used while that detour is being traced.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

    dirs = _DIRS_URDL  # U, R, D, L

    # --- find the unique green start pixel on the border ---
    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])

    # Inward initial direction + opposite-edge target.
    if sr == 0:
        init_dir = 2  # down
        goal_kind, goal_val = "row", h - 1
        path_is_vertical = True
    elif sr == h - 1:
        init_dir = 0  # up
        goal_kind, goal_val = "row", 0
        path_is_vertical = True
    elif sc == 0:
        init_dir = 1  # right
        goal_kind, goal_val = "col", w - 1
        path_is_vertical = False
    elif sc == w - 1:
        init_dir = 3  # left
        goal_kind, goal_val = "col", 0
        path_is_vertical = False
    else:
        raise AssertionError("green start must be on a border edge")

    def is_goal(r: int, c: int) -> bool:
        return (r == goal_val) if goal_kind == "row" else (c == goal_val)

    # --- find bars, then build the keep-out region exactly as in the base task ---
    bars = _collect_bars(g, path_is_vertical=path_is_vertical)
    blocked = _build_blocked(h, w, bars)
    blocked[sr, sc] = False

    # --- Dijkstra: minimize (steps, turns) ---
    INF = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]

    goal_state: tuple[int, int, int] | None = None
    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), INF):
            continue
        if is_goal(r, c) and (r, c) != (sr, sc):
            goal_state = (r, c, dprev)
            break
        for nd, (dr, dc) in enumerate(dirs):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if blocked[rr, cc]:
                continue
            ncost = (steps + 1, turns + (0 if nd == dprev else 1))
            key = (rr, cc, nd)
            if ncost < best_cost.get(key, INF):
                best_cost[key] = ncost
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost, rr, cc, nd))

    # --- reconstruct and paint the multicolor path, then convert blue -> red ---
    out = g.copy()
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

    slabs = _routing_slabs(bars, axis_limit=(h if path_is_vertical else w))
    _paint_path(out, path, path_is_vertical=path_is_vertical, slabs=slabs)

    out[out == BLUE] = RED
    return out


# --------------- generator ---------------

# --- Tunable parameters (CFG) ---
# Grid size bounds (rows x cols from spec: 12-29 x 13-26)
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
# Bar counts and blue-per-bar range
BAR_COUNT_LO, BAR_COUNT_HI = 2, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
# Bar length bounds: absolute min/max and cross-axis percentage bounds
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
# Generation retry limit
MAX_GEN_RETRIES = 300


def _min_main_len(blues: tuple[int, ...]) -> int:
    """Minimum main-axis size to fit all bars with required spacing."""
    last_bar = blues[0] + 2
    for a, b in zip(blues, blues[1:]):
        last_bar += a + b + 2
    return last_bar + 2


def _random_composition(total: int, parts: int) -> list[int]:
    """Distribute `total` into `parts` non-negative integers uniformly (stars-and-bars)."""
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
    """Pick blue counts that fit and visibly exercise the color mapping."""
    choices = [
        combo
        for combo in itertools.product(range(BLUE_PER_BAR_LO, BLUE_PER_BAR_HI + 1), repeat=count)
        if _min_main_len(combo) <= main_hi and len(set(combo)) >= 2
    ]
    assert choices, "generator needs at least one visible multi-count configuration"
    return random.choice(choices)


def _length_bounds(cross_size: int, blue_count: int) -> tuple[int, int] | None:
    """Valid bar length range given cross-axis size and blue count."""
    lo = max(BAR_LEN_MIN, 2 * blue_count - 1, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, cross_size - blue_count - 1, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int) -> None:
    """Paint a single bar on `row` starting from the given wall side."""
    w = grid.shape[1]
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, w - length :] = colors[::-1]


def _canonical_grid(
    main_size: int,
    cross_size: int,
    blues: tuple[int, ...],
    first_side: int,
) -> np.ndarray | None:
    """Build the grid in canonical orientation (start edge = top, path goes down)."""
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
        start_col = random.randint(1, first_len - 1)
    else:
        start_col = random.randint(cross_size - first_len, cross_size - 2)
    grid[0, start_col] = GREEN
    return grid


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    """Rotate canonical (top-start) grid so the start edge matches `edge`."""
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    """Check whether the solved output reaches the edge opposite to `edge`."""
    path_colors = {GREEN, *PATH_COLOR_BY_BLUE.values()}
    if edge == "top":
        return bool(np.any(np.isin(out[-1, :], list(path_colors))))
    if edge == "bottom":
        return bool(np.any(np.isin(out[0, :], list(path_colors))))
    if edge == "left":
        return bool(np.any(np.isin(out[:, -1], list(path_colors))))
    return bool(np.any(np.isin(out[:, 0], list(path_colors))))


def _has_visible_variant_signal(out: np.ndarray) -> bool:
    """Require at least two distinct obstacle-driven detour colors in the solved output."""
    used = {int(v) for v in np.unique(out) if int(v) in PATH_COLOR_BY_BLUE.values()}
    return len(used) >= 2


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: pick parameters, build grid, validate via solver."""
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
        if not _has_visible_variant_signal(out):
            return None
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c input")
