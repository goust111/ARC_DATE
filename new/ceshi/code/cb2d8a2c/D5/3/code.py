# Variant of cb2d8a2c
# Dimension: D5 — 源对象类型 (src)
# Change: 网格边缘的单个绿色像素作为路径起点，向网格内部延伸... → 一对绿色像素定义起点和终点

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

# --- Tunable parameters (CFG) ---
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

# Design analysis for the D5 endpoint variant
# a) 具体化与行为追踪:
#    - 画面仍然是 azure 背景、若干贴墙红蓝障碍条、绿色折线路径。
#    - 可见变化是边缘现在有两个绿色单像素: 起点位于“第一根障碍条投影覆盖”的边上，终点位于对侧边缘的可通行走廊里。
#    - 原题在 solver 中把目标写成“到达对侧任意边界格即停止”; 现在必须继续走到指定终点格。
#    - 例子: 旧逻辑若先到达底边第 9 列就结束; 新逻辑在底边第 12 列再放一个绿点，路径必须先到达底边，再沿底边继续走到第 12 列。
# b) 复杂度校准:
#    - 障碍条数量、半径编码、交替贴墙等机制不变，仍有 1-4 根条和 1-4 档 keep-out 半径，视觉复杂度与原题同级。
#    - 终点不随机退化到旧题自然落点，生成器优先选与“旧题自然出口列/行”相差至少 2 的终点，让变体效果肉眼可见。
# c) 隐式假设审查:
#    - 原 `start_pts.shape[0] == 1` 已失效，必须改为恰有两个边界绿点。
#    - 原 `goal_kind/goal_val` 把终点硬编码成对侧整条边，这在变体中会提前停止，必须改成精确坐标目标。
#    - 原 `blocked[sr, sc] = False` 只给起点开口，变体还要保证终点格可进入。
#    - 原 Dijkstra 的初始朝向只对真正起点成立，因此需要先从两个同色绿点中识别哪一个是起点。
# d) 放置合法性:
#    - 生成器仍先放障碍条，再放起点，再用路径搜索挑选一个可达且合法的终点。
#    - 终点必须在对侧边缘、不能落在障碍条覆盖区、不能等于旧题自然落点，并且必须有精确路径可达。
# e) 视觉可观察性:
#    - 起点与终点同色会带来歧义，所以结构上强制“只有起点与最近的一根条在横向/纵向投影上对齐”，终点不满足这个特征。
#    - 因而人类可从样例中看到: 其中一个绿点像原题起点，另一个绿点像被连接的目标。
# f) 可视化整洁度:
#    - 终点只占一个边缘格，不新增额外标记层；路径仍只涂在背景格上，不覆盖红条，画面保持原题的稀疏整洁。
# g) 颜色变化可推断性:
#    - 颜色规则不扩张: 仍只有蓝像素变红、绿色表示路径，避免引入新的颜色映射负担。
# h) 变体量化指标:
#    - 统计“显式终点与旧题自然出口的边缘坐标差值” `endpoint_shift`。若该值长期为 0，则变体退化。
# i) main() 的新假设与 generator 保证:
#    - main() 现在假设两个绿点位于相对的两条边上，且恰有一个绿点落在其最近障碍条的投影覆盖内。
#    - generate_input() 通过起点放在首条覆盖区内、终点放在末条开放走廊内来保证这一点。


def _edge_of(r: int, c: int, h: int, w: int) -> str | None:
    if r == 0:
        return "top"
    if r == h - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == w - 1:
        return "right"
    return None


def _init_dir_for_edge(edge: str) -> int:
    return {"top": 2, "bottom": 0, "left": 1, "right": 3}[edge]


def _coverage_interval(length: int, side: int, cross_size: int) -> tuple[int, int]:
    if side == 0:
        return 0, length - 1
    return cross_size - length, cross_size - 1


def _bar_nearest_edge_covers_point(
    point: tuple[int, int],
    bar_comps: list[np.ndarray],
    shape: tuple[int, int],
) -> bool:
    r, c = point
    h, w = shape
    edge = _edge_of(r, c, h, w)
    assert edge is not None, "green terminals must lie on the border"

    if edge == "top":
        comp = min(bar_comps, key=lambda pts: int(np.min(pts[:, 0])))
        return int(np.min(comp[:, 1])) <= c <= int(np.max(comp[:, 1]))
    if edge == "bottom":
        comp = max(bar_comps, key=lambda pts: int(np.max(pts[:, 0])))
        return int(np.min(comp[:, 1])) <= c <= int(np.max(comp[:, 1]))
    if edge == "left":
        comp = min(bar_comps, key=lambda pts: int(np.min(pts[:, 1])))
        return int(np.min(comp[:, 0])) <= r <= int(np.max(comp[:, 0]))
    comp = max(bar_comps, key=lambda pts: int(np.max(pts[:, 1])))
    return int(np.min(comp[:, 0])) <= r <= int(np.max(comp[:, 0]))


def _build_blocked(
    g: np.ndarray,
    free_points: tuple[tuple[int, int], ...] = (),
) -> tuple[list[np.ndarray], np.ndarray]:
    h, w = g.shape
    bar_mask = (g == BLUE) | (g == RED)
    bar_comps = components_mask(bar_mask, connectivity=4)

    forbidden = np.zeros((h, w), dtype=bool)
    for comp in bar_comps:
        blue_count = sum(1 for r, c in comp if g[r, c] == BLUE)
        radius = int(blue_count)

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
    for r, c in free_points:
        blocked[r, c] = False
    return bar_comps, blocked


def _run_search(
    blocked: np.ndarray,
    start: tuple[int, int],
    init_dir: int,
) -> tuple[
    dict[tuple[int, int, int], tuple[int, int]],
    dict[tuple[int, int, int], tuple[int, int, int]],
]:
    dirs = _DIRS_URDL
    h, w = blocked.shape
    sr, sc = start
    inf = (10**9, 10**9)

    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]

    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), inf):
            continue
        for nd, (dr, dc) in enumerate(dirs):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if blocked[rr, cc]:
                continue
            ncost = (steps + 1, turns + (0 if nd == dprev else 1))
            key = (rr, cc, nd)
            if ncost < best_cost.get(key, inf):
                best_cost[key] = ncost
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost, rr, cc, nd))
    return best_cost, prev


def _best_state_for_cell(
    best_cost: dict[tuple[int, int, int], tuple[int, int]],
    goal: tuple[int, int],
) -> tuple[int, int, int] | None:
    gr, gc = goal
    state = None
    score = None
    for cand, cost in best_cost.items():
        r, c, _ = cand
        if (r, c) != (gr, gc):
            continue
        if score is None or cost < score:
            state = cand
            score = cost
    return state


def _best_state_for_predicate(
    best_cost: dict[tuple[int, int, int], tuple[int, int]],
    predicate,
    *,
    exclude: tuple[int, int] | None = None,
) -> tuple[int, int, int] | None:
    state = None
    score = None
    for cand, cost in best_cost.items():
        r, c, _ = cand
        if exclude is not None and (r, c) == exclude:
            continue
        if not predicate(r, c):
            continue
        if score is None or cost < score:
            state = cand
            score = cost
    return state


def _reconstruct_path(
    goal_state: tuple[int, int, int],
    prev: dict[tuple[int, int, int], tuple[int, int, int]],
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    st = goal_state
    while True:
        r, c, _ = st
        path.append((r, c))
        if st not in prev:
            break
        st = prev[st]
    path.reverse()
    return path


def _infer_terminals(
    g: np.ndarray,
    bar_comps: list[np.ndarray],
) -> tuple[tuple[int, int], tuple[int, int], int]:
    h, w = g.shape
    green_pts = [tuple(map(int, pt)) for pt in np.argwhere(g == GREEN)]
    assert len(green_pts) == 2, "grid must contain exactly two green border pixels"
    assert bar_comps, "grid must contain at least one obstacle bar"

    edges = [_edge_of(r, c, h, w) for r, c in green_pts]
    assert all(edge is not None for edge in edges), "green terminals must be on border edges"
    edge_set = set(edges)
    assert edge_set in ({"top", "bottom"}, {"left", "right"}), "green terminals must lie on opposite edges"

    covered = [_bar_nearest_edge_covers_point(pt, bar_comps, (h, w)) for pt in green_pts]
    assert sum(covered) == 1, "exactly one green terminal must align with the nearest bar coverage"

    start = green_pts[covered.index(True)]
    goal = green_pts[covered.index(False)]
    init_dir = _init_dir_for_edge(_edge_of(start[0], start[1], h, w))
    return start, goal, init_dir


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two edge green pixels with a shortest-turn path that obeys the red bars' blue-coded keep-out radii.

    Input structure:
    - bg = azure (8). Red/blue bars (2/1) are the same wall-attached obstacles as in the original task.
    - Exactly two green (3) border pixels are present: one is the true path start and the other is the required destination.

    Rule:
    - Each obstacle bar forbids all cells within Manhattan distance <= its blue-pixel count.
    - The start green pixel is the border one aligned with the nearest bar's span; the path begins by moving inward from that side and must finish on the other green pixel exactly.
    - After the route is painted, every blue pixel turns red.

    Key insight:
    - The bar geometry still determines the legal corridor, but now the second green border pixel removes the old freedom to stop at any opposite-edge cell, so the route must continue until it reaches the explicit endpoint.
    """
    g = np.array(grid, copy=True)

    bar_comps, blocked = _build_blocked(g)
    start, goal, init_dir = _infer_terminals(g, bar_comps)
    blocked[start] = False
    blocked[goal] = False

    best_cost, prev = _run_search(blocked, start, init_dir)
    goal_state = _best_state_for_cell(best_cost, goal)
    assert goal_state is not None, "a valid path to the explicit green endpoint must exist"

    out = g.copy()
    for r, c in _reconstruct_path(goal_state, prev):
        if out[r, c] == BG:
            out[r, c] = GREEN

    out[out == BLUE] = RED
    return out


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
) -> tuple[np.ndarray, list[int], list[int]] | None:
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
    lengths: list[int] = []
    sides = [(first_side + i) % 2 for i in range(count)]
    for row, blue_count, side in zip(rows, blues, sides):
        bounds = _length_bounds(cross_size, blue_count)
        if bounds is None:
            return None
        length = random.randint(*bounds)
        lengths.append(length)
        _paint_bar(grid, row, side, length, blue_count)

    first_lo, first_hi = _coverage_interval(lengths[0], sides[0], cross_size)
    if first_hi - first_lo < 1:
        return None
    start_col = random.randint(first_lo + (0 if sides[0] == 1 else 1), first_hi - (1 if sides[0] == 1 else 0))
    grid[0, start_col] = GREEN
    return grid, lengths, sides


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _choose_canonical_end_col(
    grid: np.ndarray,
    last_length: int,
    last_side: int,
) -> tuple[int, int] | None:
    h, w = grid.shape
    start = tuple(map(int, np.argwhere(grid == GREEN)[0]))

    _, blocked = _build_blocked(grid, free_points=(start,))
    best_cost, prev = _run_search(blocked, start, init_dir=2)

    legacy_goal = _best_state_for_predicate(
        best_cost,
        lambda r, c: r == h - 1,
        exclude=start,
    )
    if legacy_goal is None:
        return None
    legacy_col = int(legacy_goal[1])

    cov_lo, cov_hi = _coverage_interval(last_length, last_side, w)
    candidates: list[tuple[int, int]] = []
    for col in range(w):
        if cov_lo <= col <= cov_hi:
            continue
        state = _best_state_for_cell(best_cost, (h - 1, col))
        if state is None or col == legacy_col:
            continue
        candidates.append((abs(col - legacy_col), col))

    if not candidates:
        return None

    far_pool = [item for item in candidates if item[0] >= 2]
    shift, end_col = random.choice(far_pool or candidates)
    _ = _reconstruct_path(_best_state_for_cell(best_cost, (h - 1, end_col)), prev)
    return end_col, shift


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

    built = _canonical_grid(main_size, cross_size, blues, first_side)
    if built is None:
        return None
    canonical, lengths, sides = built

    end_pick = _choose_canonical_end_col(canonical, lengths[-1], sides[-1])
    if end_pick is None:
        return None
    end_col, _shift = end_pick
    canonical[-1, end_col] = GREEN

    inp = _orient_grid(canonical, edge)
    try:
        out = main(inp)
        if not np.array_equal(main(inp), out):
            return None
        if np.count_nonzero(inp == GREEN) != 2:
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cb2d8a2c D5 input")
