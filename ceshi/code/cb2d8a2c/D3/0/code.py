# Variant of cb2d8a2c
# Dimension: D3 — 距离约束的编码方式 (trajectory)
# Change: distance_constraint——路径与每条障碍条的最小距离 = 该条上蓝色像素的数量，路径必须在该距离之外通过... → 距离约束取反（距离 = 总像素数 - 蓝色数量）
#
# Design analysis
# source_task: None, so no extra reference solver was read.
# variant_spec:
#   core_change:
#     old: each bar's keep-out radius equals its blue-cell count
#     new: each bar's keep-out radius equals its red-cell count = bar_length - blue_count
#   concrete_behavior:
#     input still shows 1-cell-thick side-attached bars made of blue/red cells and one green border start.
#     output still draws one green orthogonal path to the opposite edge and recolors blue to red.
#     the visible change is that the path now hugs blue-heavy bars more closely and gives red-heavy bars more room.
#   small_trace:
#     original: a length-7 bar with blue_count=3 had radius 3, so the path stayed Manhattan distance > 3 away.
#     variant: the same bar has red_count=4, so the forbidden halo expands to distance > 4 instead.
#     conversely, a length-7 bar with blue_count=4 now has red_count=3, so more blue means a tighter pass.
#   complexity_calibration:
#     to make the inversion visually inferable, all bars within one generated puzzle share the same total length.
#     then radius = shared_length - blue_count, so "more blue -> smaller detour" is directly comparable across bars.
#     the generator also requires at least two bars with different blue counts; otherwise the inversion is hard to see.
#   implicit_assumption_audit:
#     main() previously assumed "blue_count is the only encoded distance". The only solver line that truly depends on that
#     is the radius extraction inside the component loop; pathfinding, goal detection, and blue->red recoloring remain valid.
#     generator code depended on the old radius in three places:
#       1. start-to-first-bar spacing
#       2. spacing between consecutive bars
#       3. bar length / open-side gap feasibility
#     all three must be rewritten in terms of red_count, or generation would silently produce unsolved or misleading cases.
#   placement_legality:
#     bars still alternate attached walls and remain 1 cell thick.
#     cross-axis open gap is enforced as at least red_count + 1, guaranteeing a legal corridor around each bar.
#     main-axis gaps are enforced as red_a + red_b + 1, matching the solver's Manhattan halo rule.
#   visual_observability:
#     using 2-3 bars (instead of allowing singleton cases) keeps the new coding relation visible in each sample.
#     total length is shared per grid, but blue counts vary, so red counts vary inversely without introducing a new symbol system.
#   cleanliness:
#     no new colors, overlays, or markers are added; the puzzle stays as bars + one path.
#   color_change_inference:
#     color dynamics stay simple: only blue cells become red after solving, exactly as in the original task.

import heapq
import itertools
import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a green route from one border to the opposite border while skirting bar-shaped obstacles by a distance encoded in their red mass.

    Input structure:
    - The background is azure (8), with 1-cell-thick obstacle bars made from blue (1) and red (2) cells.
    - Exactly one green (3) cell lies on a border and marks the path's starting point.

    Rule:
    - For each bar, the forbidden Manhattan halo radius is the number of red cells on that bar, not the number of blue cells.
    - The green route advances to the opposite edge using axis-aligned moves and detours around every bar's halo.
    - Once the route is drawn, all blue cells on the bars are recolored red.

    Key insight: The blue cells no longer directly tell you how far to stay away; the true distance is encoded by the red remainder of each bar, so bars with more blue permit a closer pass.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

    BG = 8
    GREEN = 3
    BLUE = 1
    RED = 2

    dirs = _DIRS_URDL

    start_pts = np.argwhere(g == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])

    if sr == 0:
        init_dir = 2
        goal_kind, goal_val = "row", h - 1
    elif sr == h - 1:
        init_dir = 0
        goal_kind, goal_val = "row", 0
    elif sc == 0:
        init_dir = 1
        goal_kind, goal_val = "col", w - 1
    elif sc == w - 1:
        init_dir = 3
        goal_kind, goal_val = "col", 0
    else:
        raise AssertionError("green start must be on a border edge")

    def is_goal(r: int, c: int) -> bool:
        return (r == goal_val) if goal_kind == "row" else (c == goal_val)

    bar_mask = (g == BLUE) | (g == RED)
    bar_comps = components_mask(bar_mask, connectivity=4)

    forbidden = np.zeros((h, w), dtype=bool)
    for comp in bar_comps:
        red_count = sum(1 for r, c in comp if g[r, c] == RED)
        radius = int(red_count)

        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in comp:
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

    blocked = bar_mask | forbidden
    blocked[sr, sc] = False

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
    for r, c in path:
        if out[r, c] == BG:
            out[r, c] = GREEN

    out[out == BLUE] = RED
    return out


BG = 8
BLUE = 1
RED = 2
GREEN = 3

ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 2, 3
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 300


def _min_main_len(reds: tuple[int, ...]) -> int:
    """Minimum main-axis size needed for the red-count keep-out radii."""
    last_bar = reds[0] + 2
    for a, b in zip(reds, reds[1:]):
        last_bar += a + b + 2
    return last_bar + 2


def _random_composition(total: int, parts: int) -> list[int]:
    """Uniform stars-and-bars split of `total` into `parts` non-negative pieces."""
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _length_bounds(cross_size: int) -> tuple[int, int] | None:
    """Length range allowed by the original bar-span percentages."""
    lo = max(BAR_LEN_MIN, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int) -> None:
    """Paint one alternating-prefix bar attached to the left/top or right/bottom wall."""
    w = grid.shape[1]
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, w - length :] = colors[::-1]


def _feasible_bar_specs(count: int, cross_size: int, main_hi: int) -> list[tuple[int, tuple[int, ...], tuple[int, ...]]]:
    """
    Enumerate shared-length bar layouts that visibly exercise the inverted encoding.

    All bars in one puzzle share `length`, while individual blue counts vary.
    This makes red_count = length - blue_count easy to compare by eye.
    """
    bounds = _length_bounds(cross_size)
    if bounds is None:
        return []

    choices: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    for length in range(bounds[0], bounds[1] + 1):
        blue_hi = min(BLUE_PER_BAR_HI, (length + 1) // 2)
        if blue_hi < BLUE_PER_BAR_LO:
            continue
        for blues in itertools.product(range(BLUE_PER_BAR_LO, blue_hi + 1), repeat=count):
            if len(set(blues)) < 2:
                continue
            reds = tuple(length - blue for blue in blues)
            if cross_size - length < max(reds) + 1:
                continue
            if _min_main_len(reds) > main_hi:
                continue
            choices.append((length, blues, reds))
    return choices


def _canonical_grid(
    main_size: int,
    cross_size: int,
    length: int,
    blues: tuple[int, ...],
    reds: tuple[int, ...],
    first_side: int,
) -> np.ndarray | None:
    """Build the canonical top-start version of the puzzle."""
    slack = main_size - _min_main_len(reds)
    if slack < 0:
        return None

    extras = _random_composition(slack, len(reds) + 1)
    row = reds[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, len(reds)):
        row += reds[i - 1] + reds[i] + 2 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    for row, blue_count, side in zip(rows, blues, ((first_side + i) % 2 for i in range(len(blues)))):
        _paint_bar(grid, row, side, length, blue_count)

    if first_side == 0:
        start_col = random.randint(1, length - 1)
    else:
        start_col = random.randint(cross_size - length, cross_size - 2)
    grid[0, start_col] = GREEN
    return grid


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    """Rotate canonical top-start layout to the requested starting edge."""
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _reaches_opposite_edge(out: np.ndarray, edge: str) -> bool:
    """Check that the solved path reaches the side opposite the starting edge."""
    if edge == "top":
        return bool(np.any(out[-1, :] == GREEN))
    if edge == "bottom":
        return bool(np.any(out[0, :] == GREEN))
    if edge == "left":
        return bool(np.any(out[:, -1] == GREEN))
    return bool(np.any(out[:, 0] == GREEN))


def _gen_attempt() -> np.ndarray | None:
    """Generate one candidate input and keep it only if the solver validates it."""
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    param_choices: list[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = []
    for cross_size in range(cross_lo, cross_hi + 1):
        for count in range(BAR_COUNT_LO, BAR_COUNT_HI + 1):
            for length, blues, reds in _feasible_bar_specs(count, cross_size, main_hi):
                param_choices.append((cross_size, length, blues, reds))
    if not param_choices:
        return None

    cross_size, length, blues, reds = random.choice(param_choices)
    main_size = random.randint(max(main_lo, _min_main_len(reds)), main_hi)
    first_side = random.randrange(2)

    canonical = _canonical_grid(main_size, cross_size, length, blues, reds, first_side)
    if canonical is None:
        return None
    inp = _orient_grid(canonical, edge)

    try:
        out = main(inp)
        if not _reaches_opposite_edge(out, edge):
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
    raise RuntimeError("failed to generate a valid cb2d8a2c D3 variant input")
