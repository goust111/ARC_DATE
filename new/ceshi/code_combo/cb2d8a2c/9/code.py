# Combo variant of cb2d8a2c (index 9)
# combo_spec: D5/3+D6/0
# Name: 起终点对+目标导向路径

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
ROWS_LO, ROWS_HI = 12, 29
COLS_LO, COLS_HI = 13, 26
BAR_COUNT_LO, BAR_COUNT_HI = 1, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MIN_ENDPOINT_SHIFT = 2
MAX_GEN_RETRIES = 600

# Design analysis for combo D5/3 + D6/0
# a) 具体化与行为追踪:
#    - 画面仍是 azure 背景、贴墙红蓝障碍条、绿色路径；但输入里现在有两个边缘绿点。
#    - 原题: 单绿点给出起点，路径避开各条 keep-out 区后，到达对侧任意边界格就停。
#    - 融合后: 两个绿点中，靠近“首根障碍条投影”的那个仍是起点；另一个绿点不再只是装饰，
#      它就是精确目标坐标。路径必须绕过全部障碍并恰好连到那一格。
#    - 小例子: 若 canonical top-start 布局下原题自然会在底边第 8 列出边，而第二个绿点在底边
#      第 11 列，则新题必须继续把路线导向第 11 列；这体现了 D5 的双端点约束和 D6 的精确终止。
# b) 复杂度校准:
#    - 不新增第三种显式目标物体，避免“两个绿点 + 一个黄点”造成端点语义冲突。
#    - 视觉差异通过“显式终点偏离原题自然出口”来保证；生成器要求偏移至少 2 格，避免退化。
# c) 隐式假设审查:
#    - 原 `start_pts.shape[0] == 1` 失效，必须改为恰有两个边缘绿点。
#    - 原 `goal_kind/goal_val` 的整条对侧边界目标失效，必须改成精确坐标目标。
#    - 起点方向不能从“任意一个绿点”推断，必须先判别哪一个与最近障碍条覆盖对齐。
#    - `blocked` 不能只给起点开口，显式终点也必须允许进入。
# d) 放置合法性:
#    - 生成器仍沿原 spec 先放条、再放起点；之后先求出原题自然出口，再挑选另一个合法边缘终点。
#    - 第二个绿点必须位于对侧边缘、在可达走廊中、且不在末条障碍覆盖投影内，以保证“恰有一个”
#      绿点符合起点判别条件。
# e) 视觉可观察性:
#    - 人类能从样例看到: 两个绿点分居相对边，但只有一侧那个绿点像原题入口，另一侧那个则像被连接的目标。
#    - 输出中路径终点精确落在第二个绿点处，而不是任意对侧出口。
# f) 可视化整洁度:
#    - 仍只用原题的颜色集合；路径只涂背景格，终点格本身已是绿色，不引入额外叠层。
# g) 变化的可推断性:
#    - 绿色仍表示“路径/端点”，蓝仍只表示半径编码并在输出变红；没有新增难以归纳的颜色映射。
# h) 量化指标:
#    - 用 `endpoint_shift` 衡量“显式终点”和“原题自然出口”在对侧边上的坐标差。若长期为 0，
#      则组合退化成原题；本生成器强制 `endpoint_shift >= 2`。
# i) main() 的新假设与 generator 保证:
#    - `main()` 假设两个绿点位于相对边，且仅有一个绿点落在其最近障碍条的投影覆盖内。
#    - `generate_input()` 通过起点落在首条覆盖区内、终点落在对侧开放走廊内来保证该判别唯一。


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


def _opposite_edge(edge: str) -> str:
    return {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[edge]


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
        for nd, (dr, dc) in enumerate(_DIRS_URDL):
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


def _legacy_goal_state(
    best_cost: dict[tuple[int, int, int], tuple[int, int]],
    start: tuple[int, int],
    start_edge: str,
    shape: tuple[int, int],
) -> tuple[int, int, int] | None:
    h, w = shape
    opposite = _opposite_edge(start_edge)
    if opposite == "top":
        return _best_state_for_predicate(best_cost, lambda r, c: r == 0, exclude=start)
    if opposite == "bottom":
        return _best_state_for_predicate(best_cost, lambda r, c: r == h - 1, exclude=start)
    if opposite == "left":
        return _best_state_for_predicate(best_cost, lambda r, c: c == 0, exclude=start)
    return _best_state_for_predicate(best_cost, lambda r, c: c == w - 1, exclude=start)


def _endpoint_shift_from_legacy(grid: np.ndarray) -> int:
    g = np.array(grid, copy=True)
    bar_comps, blocked = _build_blocked(g)
    start, goal, init_dir = _infer_terminals(g, bar_comps)
    blocked[start] = False
    blocked[goal] = False
    best_cost, _prev = _run_search(blocked, start, init_dir)

    h, w = g.shape
    start_edge = _edge_of(start[0], start[1], h, w)
    assert start_edge is not None
    legacy_goal = _legacy_goal_state(best_cost, start, start_edge, (h, w))
    assert legacy_goal is not None, "legacy opposite-edge exit must exist"

    if start_edge in ("top", "bottom"):
        return abs(int(goal[1]) - int(legacy_goal[1]))
    return abs(int(goal[0]) - int(legacy_goal[0]))


def _solve_explicit_path(grid: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    g = np.array(grid, copy=True)
    bar_comps, blocked = _build_blocked(g)
    start, goal, init_dir = _infer_terminals(g, bar_comps)
    blocked[start] = False
    blocked[goal] = False

    best_cost, prev = _run_search(blocked, start, init_dir)
    goal_state = _best_state_for_cell(best_cost, goal)
    assert goal_state is not None, "a valid path to the explicit endpoint must exist"
    return _reconstruct_path(goal_state, prev), start, goal


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect one edge green pixel to the other, but stop only when the route reaches that exact border target.

    Input structure:
    - bg = azure (8). Red/blue bars (2/1) are wall-attached obstacles exactly as in the base task.
    - Exactly two green (3) border pixels are present on opposite sides: one is the true start and the other is the required destination.

    Rule:
    - Each obstacle bar forbids every cell within Manhattan distance <= its blue-pixel count.
    - The green pixel aligned with the nearest bar span is the start; the path moves inward from that side, turns orthogonally as needed, and must finish on the other green pixel exactly instead of stopping at an arbitrary opposite-edge cell.
    - After the route is drawn, all blue pixels become red.

    Key insight:
    - The bars still encode the keep-out geometry, but the second green border pixel now acts as a precise goal coordinate, so the old open-ended escape problem becomes a fixed-endpoint connection problem.
    """
    out = np.array(grid, copy=True)
    path, _start, _goal = _solve_explicit_path(out)

    for r, c in path:
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


def _has_bottom_edge_suffix(path: list[tuple[int, int]], bottom_row: int) -> bool:
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if r1 == bottom_row and r2 == bottom_row and c1 != c2:
            return True
    return False


def _choose_canonical_end_col(
    grid: np.ndarray,
    last_length: int,
    last_side: int,
) -> tuple[int, int] | None:
    h, w = grid.shape
    start = tuple(map(int, np.argwhere(grid == GREEN)[0]))

    _bar_comps, blocked = _build_blocked(grid, free_points=(start,))
    best_cost, prev = _run_search(blocked, start, init_dir=2)

    legacy_goal = _legacy_goal_state(best_cost, start, "top", (h, w))
    if legacy_goal is None:
        return None
    legacy_col = int(legacy_goal[1])

    cov_lo, cov_hi = _coverage_interval(last_length, last_side, w)
    preferred: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []
    for col in range(w):
        if cov_lo <= col <= cov_hi:
            continue
        state = _best_state_for_cell(best_cost, (h - 1, col))
        if state is None:
            continue
        shift = abs(col - legacy_col)
        if shift < MIN_ENDPOINT_SHIFT:
            continue
        path = _reconstruct_path(state, prev)
        target_pool = preferred if _has_bottom_edge_suffix(path, h - 1) else fallback
        target_pool.append((shift, col))

    pool = preferred or fallback
    if not pool:
        return None
    shift, end_col = random.choice(pool)
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
        if _endpoint_shift_from_legacy(inp) < MIN_ENDPOINT_SHIFT:
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
