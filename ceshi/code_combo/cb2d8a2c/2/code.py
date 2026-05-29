# Combo variant of cb2d8a2c (index 2)
# combo_spec: D1/1+D3/0
# Name: BFS最短路径+反向距离编码
#
# Design analysis
# a) Concrete behavior and small trace:
#    - The visible input vocabulary stays the same: azure background, side-attached 1-cell-thick blue/red bars,
#      and one green border seed. The output still paints a green route and recolors blue to red.
#    - The first change is the keep-out encoding. A bar of total length 8 with blue_count 3 used to impose radius 3.
#      In this combo it imposes radius 5 instead, because radius = 8 - 3. A blue-heavy bar therefore becomes safer,
#      not more dangerous.
#    - The second change is the routing policy. The original code used direction-state Dijkstra with a turn tiebreak,
#      which naturally favors long straight detours. Here we build the forbidden mask with the inverted radius and then
#      run plain BFS on cells, so the chosen route is any shortest legal path under that mask. On tight alternating bars,
#      the shortest valid route can step sideways earlier and produce a visibly more jagged snake.
#    - Example trace in canonical top-start orientation:
#      first bar length 8, blue_count 4 -> radius 4, second bar length 8, blue_count 2 -> radius 6.
#      Under the original task the radii would have been 4 and 2, so the lower bar would look easier than it now is.
#      Under this combo the second bar pushes the corridor much farther away, and BFS must find the shortest path through
#      the resulting slalom instead of following the old "one rectangular detour per bar" intuition.
# b) Complexity calibration:
#    - To make the inversion legible from only a few ARC examples, every bar in one generated puzzle shares the same total
#      length. Then comparing bars by eye is enough: more blue on equal-length bars means fewer red cells and thus a
#      smaller keep-out halo.
#    - The generator enforces at least two distinct blue counts, so the inverse coding can be compared within one sample.
#    - To make the BFS change visible, the generator only accepts samples whose solved path has more turns than the old
#      per-bar heuristic would suggest, i.e. more than two turns per bar.
# c) Implicit assumption audit:
#    - main() still assumes exactly one green start pixel on the border; the generator guarantees this.
#    - Bar detection still relies on 4-connected {blue, red} components; spacing guarantees different bars do not touch.
#    - The radius extraction line is no longer "count blue"; it is "component size - blue count". If generator spacing
#      stayed in terms of blue_count, valid paths would silently disappear, so spacing is rewritten in terms of inverted
#      radii everywhere.
#    - The new BFS tie-order uses the visible opening side of the first bar to break shortest-path ties. The generator
#      guarantees at least one bar and places the start within that first bar's span, so this bias is stable and visible.
# d) Placement legality:
#    - Bars still alternate attached walls, remain 1 cell thick, and obey the original length/span bounds.
#    - For each bar, cross-axis uncovered space is at least radius + 1, which is the exact feasibility condition for a
#      legal Manhattan pass around a side-attached bar.
#    - Main-axis gaps are computed from the inverted radii: start gap >= r_1 + 1, inter-bar gap >= r_i + r_{i+1} + 1,
#      and trailing gap >= 1. This prevents halo overlap from sealing the corridor unless the solver check rejects it.
# e) Visual observability:
#    - Shared bar length plus varied blue counts makes the inverse encoding directly comparable.
#    - The accepted outputs exhibit extra bends beyond the original "two turns per bar" silhouette, so the BFS change is
#      not hidden behind a path that still looks like the source task.
# f) Visual cleanliness:
#    - No extra markers or colors are introduced. The path only occupies background cells, bars remain readable, and blue
#      recoloring happens after routing just like the source task.
# g) Change inferability:
#    - Color behavior remains simple: blue always turns red after solving.
#    - The only new inference is structural: equal-length bars with more blue allow a closer pass, and the green route is
#      the shortest legal one through the inverted halos.

from collections import deque
import heapq
import itertools
import random

import numpy as np

from arc_utils import components_mask, neighbors4

BG = 8
BLUE = 1
RED = 2
GREEN = 3

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# --- Generator CFG ---
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 3, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_MAIN_SLACK = 4
MAX_GEN_RETRIES = 400


def _start_info(grid: np.ndarray) -> tuple[int, int, str, int, tuple[int, int]]:
    start_pts = np.argwhere(grid == GREEN)
    assert start_pts.shape[0] == 1, "grid must contain exactly one green start pixel"
    sr, sc = map(int, start_pts[0])

    h, w = grid.shape
    if sr == 0:
        return sr, sc, "top", 2, (1, 0)
    if sr == h - 1:
        return sr, sc, "bottom", 0, (-1, 0)
    if sc == 0:
        return sr, sc, "left", 1, (0, 1)
    if sc == w - 1:
        return sr, sc, "right", 3, (0, -1)
    raise AssertionError("green start must be on a border edge")


def _is_goal_fn(shape: tuple[int, int], start_edge: str):
    h, w = shape
    if start_edge == "top":
        return lambda r, c: r == h - 1
    if start_edge == "bottom":
        return lambda r, c: r == 0
    if start_edge == "left":
        return lambda r, c: c == w - 1
    return lambda r, c: c == 0


def _bar_components(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    bar_mask = (grid == BLUE) | (grid == RED)
    bar_comps = components_mask(bar_mask, connectivity=4)
    return bar_mask, list(bar_comps)


def _radius_for_component(grid: np.ndarray, comp: np.ndarray, radius_mode: str) -> int:
    blue_count = int(sum(1 for r, c in comp if grid[r, c] == BLUE))
    if radius_mode == "blue":
        return blue_count
    if radius_mode == "inverted":
        return int(len(comp) - blue_count)
    raise ValueError(f"unknown radius_mode: {radius_mode}")


def _blocked_mask(grid: np.ndarray, radius_mode: str) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = grid.shape
    bar_mask, bar_comps = _bar_components(grid)
    forbidden = np.zeros((h, w), dtype=bool)

    for comp in bar_comps:
        radius = _radius_for_component(grid, comp, radius_mode)
        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in comp:
            rr = int(r)
            cc = int(c)
            dist[rr, cc] = 0
            q.append((rr, cc))

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
    sr, sc, _, _, _ = _start_info(grid)
    blocked[sr, sc] = False
    return blocked, bar_comps


def _first_bar_side(bar_comps: list[np.ndarray], start_edge: str, shape: tuple[int, int]) -> int | None:
    if not bar_comps:
        return None

    if start_edge == "top":
        first = min(bar_comps, key=lambda comp: int(comp[:, 0].min()))
        return 0 if int(first[:, 1].min()) == 0 else 1
    if start_edge == "bottom":
        first = max(bar_comps, key=lambda comp: int(comp[:, 0].max()))
        return 0 if int(first[:, 1].min()) == 0 else 1
    if start_edge == "left":
        first = min(bar_comps, key=lambda comp: int(comp[:, 1].min()))
        return 0 if int(first[:, 0].min()) == 0 else 1
    first = max(bar_comps, key=lambda comp: int(comp[:, 1].max()))
    return 0 if int(first[:, 0].min()) == 0 else 1


def _bfs_dirs(start_edge: str, first_side: int | None) -> list[tuple[int, int]]:
    if start_edge == "top":
        return [(0, 1), (1, 0), (0, -1), (-1, 0)] if first_side == 0 else [(0, -1), (1, 0), (0, 1), (-1, 0)]
    if start_edge == "bottom":
        return [(0, 1), (-1, 0), (0, -1), (1, 0)] if first_side == 0 else [(0, -1), (-1, 0), (0, 1), (1, 0)]
    if start_edge == "left":
        return [(1, 0), (0, 1), (-1, 0), (0, -1)] if first_side == 0 else [(-1, 0), (0, 1), (1, 0), (0, -1)]
    return [(1, 0), (0, -1), (-1, 0), (0, 1)] if first_side == 0 else [(-1, 0), (0, -1), (1, 0), (0, 1)]


def _reconstruct_path(prev: dict[tuple[int, int], tuple[int, int] | None], goal: tuple[int, int]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _shortest_path(grid: np.ndarray, radius_mode: str) -> list[tuple[int, int]]:
    blocked, bar_comps = _blocked_mask(grid, radius_mode)
    h, w = grid.shape
    sr, sc, start_edge, _, _ = _start_info(grid)
    is_goal = _is_goal_fn((h, w), start_edge)
    dirs = _bfs_dirs(start_edge, _first_bar_side(bar_comps, start_edge, (h, w)))

    q = deque([(sr, sc)])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {(sr, sc): None}
    goal: tuple[int, int] | None = None

    while q:
        r, c = q.popleft()
        if (r, c) != (sr, sc) and is_goal(r, c):
            goal = (r, c)
            break
        for dr, dc in dirs:
            rr = r + dr
            cc = c + dc
            nxt = (rr, cc)
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if blocked[rr, cc] or nxt in prev:
                continue
            prev[nxt] = (r, c)
            q.append(nxt)

    assert goal is not None, "a valid shortest path to the opposite edge must exist"
    return _reconstruct_path(prev, goal)


def _turn_aware_path(grid: np.ndarray, radius_mode: str) -> list[tuple[int, int]]:
    blocked, _ = _blocked_mask(grid, radius_mode)
    h, w = grid.shape
    sr, sc, start_edge, init_dir, _ = _start_info(grid)
    is_goal = _is_goal_fn((h, w), start_edge)

    inf = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]
    goal_state: tuple[int, int, int] | None = None

    while heap:
        cost, r, c, dprev = heapq.heappop(heap)
        if cost != best_cost.get((r, c, dprev), inf):
            continue
        if (r, c) != (sr, sc) and is_goal(r, c):
            goal_state = (r, c, dprev)
            break
        for nd, (dr, dc) in enumerate(_DIRS_URDL):
            rr = r + dr
            cc = c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if blocked[rr, cc]:
                continue
            ncost = (cost[0] + 1, cost[1] + (0 if nd == dprev else 1))
            key = (rr, cc, nd)
            if ncost < best_cost.get(key, inf):
                best_cost[key] = ncost
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost, rr, cc, nd))

    assert goal_state is not None, "a valid turn-aware path to the opposite edge must exist"
    path: list[tuple[int, int]] = []
    cur = goal_state
    while True:
        r, c, _ = cur
        path.append((r, c))
        if (r, c) == (sr, sc):
            break
        cur = prev[cur]
    path.reverse()
    return path


def _paint_output(grid: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r, c in path:
        if out[r, c] == BG:
            out[r, c] = GREEN
    out[out == BLUE] = RED
    return out


def _count_turns(path: list[tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0
    steps = [(r1 - r0, c1 - c0) for (r0, c0), (r1, c1) in zip(path, path[1:])]
    return sum(1 for a, b in zip(steps, steps[1:]) if a != b)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a green route from one border to the opposite border by taking the shortest legal BFS path around bar halos whose distances are encoded backwards.

    Input structure:
    - The background is azure (8), with 1-cell-thick side-attached bars built from blue (1) and red (2) cells.
    - Exactly one green (3) cell lies on a border and marks the route's start.

    Rule:
    - Each bar forbids all cells within Manhattan distance <= (bar_length - blue_count), i.e. its red-cell count.
    - Among all cells that avoid every bar halo, draw the shortest axis-aligned path from the green start to the opposite edge using BFS.
    - After the path is painted, recolor every blue bar cell to red.

    Key insight: More blue on an equal-length bar now means less red and therefore a smaller keep-out halo, so the green path must be read as a shortest route through an inverted danger code rather than the old fixed-turn detours.
    """
    return _paint_output(grid, _shortest_path(np.array(grid, copy=True), radius_mode="inverted"))


def _min_main_len(radii: tuple[int, ...]) -> int:
    last_bar = radii[0] + 2
    for a, b in zip(radii, radii[1:]):
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


def _length_bounds(cross_size: int) -> tuple[int, int] | None:
    lo = max(BAR_LEN_MIN, (BAR_CROSS_PCT_LO * cross_size + 99) // 100)
    hi = min(BAR_LEN_MAX, (BAR_CROSS_PCT_HI * cross_size) // 100)
    if lo > hi:
        return None
    return lo, hi


def _paint_bar(grid: np.ndarray, row: int, side: int, length: int, blue_count: int) -> None:
    w = grid.shape[1]
    colors = np.full(length, RED, dtype=int)
    colors[: 2 * blue_count - 1 : 2] = BLUE
    if side == 0:
        grid[row, :length] = colors
    else:
        grid[row, w - length :] = colors[::-1]


def _feasible_bar_specs(count: int, cross_size: int, main_hi: int) -> list[tuple[int, tuple[int, ...], tuple[int, ...]]]:
    bounds = _length_bounds(cross_size)
    if bounds is None:
        return []

    choices: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    for length in range(bounds[0], bounds[1] + 1):
        blue_hi = min(BLUE_PER_BAR_HI, (length + 1) // 2)
        if blue_hi < BLUE_PER_BAR_LO:
            continue
        for blues in itertools.product(range(BLUE_PER_BAR_LO, blue_hi + 1), repeat=count):
            if len(set(blues)) < 2 or max(blues) - min(blues) < 1:
                continue
            radii = tuple(length - blue for blue in blues)
            open_gap = cross_size - length
            if open_gap < max(radii) + 1:
                continue
            if open_gap > max(radii) + 2:
                continue
            if _min_main_len(radii) > main_hi:
                continue
            choices.append((length, blues, radii))
    return choices


def _choose_start_col(cross_size: int, length: int, first_side: int) -> int:
    half = max(1, length // 2)
    if first_side == 0:
        return random.randint(1, min(length - 1, half))
    lo = max(cross_size - length, cross_size - half - 1)
    return random.randint(lo, cross_size - 2)


def _canonical_grid(
    main_size: int,
    cross_size: int,
    length: int,
    blues: tuple[int, ...],
    radii: tuple[int, ...],
    first_side: int,
) -> np.ndarray | None:
    slack = main_size - _min_main_len(radii)
    if slack < 0:
        return None

    extras = _random_composition(slack, len(radii) + 1)
    row = radii[0] + 2 + extras[0]
    rows = [row]
    for i in range(1, len(radii)):
        row += radii[i - 1] + radii[i] + 2 + extras[i]
        rows.append(row)

    grid = np.full((main_size, cross_size), BG, dtype=int)
    for row, blue_count, side in zip(rows, blues, ((first_side + i) % 2 for i in range(len(blues)))):
        _paint_bar(grid, row, side, length, blue_count)

    grid[0, _choose_start_col(cross_size, length, first_side)] = GREEN
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


def _bar_count(grid: np.ndarray) -> int:
    _, comps = _bar_components(grid)
    return len(comps)


def _candidate_metrics(inp: np.ndarray) -> dict[str, int | bool]:
    bfs_path = _shortest_path(inp, radius_mode="inverted")
    bfs_out = _paint_output(inp, bfs_path)
    original_out = _paint_output(inp, _turn_aware_path(inp, radius_mode="blue"))
    turny = _count_turns(bfs_path)
    bars = _bar_count(inp)
    return {
        "bars": bars,
        "turns": turny,
        "extra_turns": int(turny > 2 * bars),
        "diff_from_original": int(not np.array_equal(bfs_out, original_out)),
        "reaches_goal": int(_reaches_opposite_edge(bfs_out, _start_info(inp)[2])),
    }


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    for _ in range(60):
        count = random.randint(BAR_COUNT_LO, BAR_COUNT_HI)
        cross_size = random.randint(cross_lo, cross_hi)
        choices = _feasible_bar_specs(count, cross_size, main_hi)
        if not choices:
            continue

        length, blues, radii = random.choice(choices)
        min_main = max(main_lo, _min_main_len(radii))
        if min_main > main_hi:
            continue
        main_size = random.randint(min_main, min(main_hi, min_main + MAX_MAIN_SLACK))
        first_side = random.randrange(2)

        canonical = _canonical_grid(main_size, cross_size, length, blues, radii, first_side)
        if canonical is None:
            continue
        inp = _orient_grid(canonical, edge)

        try:
            out = main(inp)
            if not _reaches_opposite_edge(out, edge):
                continue
            if not np.array_equal(main(inp), out):
                continue
            metrics = _candidate_metrics(inp)
            if not metrics["diff_from_original"]:
                continue
            if not metrics["extra_turns"]:
                continue
            return inp
        except Exception:
            continue
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c combo input")
