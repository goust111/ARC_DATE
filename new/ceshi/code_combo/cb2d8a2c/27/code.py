# Combo variant of cb2d8a2c (index 27)
# combo_spec: D5/3+D6/0+D12/0
# Name: 起终点+目标到达+逐条处理

# Design analysis for this combo variant
# -------------------------------------
# a) 具体化与行为追踪:
#    - 画面仍是 azure(8) 背景、若干红蓝障碍条、绿色折线路径，但边界上现在有两个绿点。
#    - 其中一个绿点像原题一样贴着“第一根障碍条的投影覆盖区”，它是真正起点；另一个绿点在对侧边界，是必须抵达的终点。
#    - 原题做法: 从单个起点一次性搜索到对侧任意边缘格，然后统一把所有蓝点变红。
#    - 融合后做法: 先按当前障碍状态搜索一条朝终点的全局路线，但只执行到“刚越过当前最近障碍条”的前缀；随后把这根条上的蓝点改成红点，重新计算到同一终点的路线，再处理下一根条。
#    - 小例子: 顶边起点、底边终点、左/右交替贴墙的三根横条。旧逻辑会在初始所有 keep-out 同时存在时一次性绕完整条路。新逻辑会先绕过第 1 根横条，接着第 1 根的扩张禁区消失，第二次规划可能贴着第 1 根旧禁区边缘向终点回切，因此最终路径会与一次性规划 visibly 不同。
# b) 复杂度校准:
#    - 为了让“逐条处理”在 2-4 个样例里肉眼可见，生成器把障碍条数量收紧为 2-4 根，避免退化成只有一次处理的单条情形。
#    - 终点仍只用一个绿像素，不再引入新颜色；路径差异主要体现在终点精确约束和后续段对已处理障碍旧禁区的再利用。
#    - 终点候选会优先筛掉退化样本: 若顺序规划与静态一次性规划完全相同，则该样本丢弃。
# c) 隐式假设审查:
#    - 原 `start_pts.shape[0] == 1` 不再成立，必须改成恰有两个边界绿点。
#    - 原 `goal_kind/goal_val` 把目标写死为“对侧整条边”，融合后必须识别精确终点坐标。
#    - 原 solver 默认所有障碍的蓝半径一直有效；D12 破坏了这一前提，所以 blocked mask 不能只构造一次，必须每处理一条后重建。
#    - 原 Dijkstra 只需要一个初始朝向；融合后每一段重规划还需要沿用上一段结束时的朝向，否则 turn 最优性会被破坏。
#    - 原绘制阶段只关心最终整条路径；现在虽然输出仍只有最终态，但内部求解必须保留分段边界，才能定义“哪一根障碍已被处理”。
# d) 放置合法性:
#    - 生成器仍沿用原题几何约束: 障碍条贴墙、交替左右、长度与蓝点数满足原 spec。
#    - 起点放在首条覆盖区内，终点放在末条覆盖区外，从而保证两个同色边界绿点中只有一个能被识别为起点。
#    - 每个候选输入都经过同一个顺序 solver 验证；不可达、终点识别歧义、或顺序行为退化的样本全部丢弃。
# e) 视觉可观察性:
#    - 输入里两个绿点给出“从哪里来、到哪里去”的强提示；输出里绿色折线恰好连到指定终点，而不是停在任意对侧出口。
#    - 多条障碍交替贴墙，本身就形成自然的“逐条通过”阅读顺序；顺序求解带来的后续贴边回切能让人看到路径不是一次定死的。
# f) 可视化整洁度:
#    - 仍只使用原题已有颜色，避免额外标记层。
#    - 路径只画在背景格，障碍条只发生蓝->红的局部颜色简化；对象层次与原题一样清晰。
# g) 变化的可推断性:
#    - 颜色变化规则仍然单一: 蓝像素最终都变红，绿色始终表示起点/终点/路径。
#    - 新信息完全由结构提供: 第二个绿点给出终点，分段规划则体现在路径几何而非隐藏标签。
# h) 变体量化指标:
#    - 使用 `stage_choice_gap_count`: 顺序分段规划在每根障碍“刚越过前沿线”的落点，与同一输入上一体化静态规划的对应落点相比，有多少根条发生了不同选择。
#    - 若该指标为 0，则逐条处理没有改变任何局部过杆决策，D12 退化；生成器会拒绝该样本。
# i) main() 的新假设与 generator 保证:
#    - `main()` 假设恰有两枚边界绿点，且仅有一枚位于离自己最近的障碍条投影覆盖内。
#    - `main()` 假设障碍条沿主轴可排序，且路径需要按这个顺序依次清除。
#    - `generate_input()` 通过“首条覆盖起点、末条避开终点、2-4 根交替贴墙、样本经顺序/静态双重检查”来保证这些假设成立。

import heapq
import itertools
import random
from collections import deque
from functools import lru_cache

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
BAR_COUNT_LO, BAR_COUNT_HI = 4, 4
BLUE_PER_BAR_LO, BLUE_PER_BAR_HI = 1, 4
BAR_LEN_MIN, BAR_LEN_MAX = 6, 15
BAR_CROSS_PCT_LO, BAR_CROSS_PCT_HI = 32, 79
MAX_GEN_RETRIES = 500


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


def _axis_sign_for_edge(edge: str) -> tuple[str, int]:
    if edge == "top":
        return "row", 1
    if edge == "bottom":
        return "row", -1
    if edge == "left":
        return "col", 1
    if edge == "right":
        return "col", -1
    raise AssertionError("unknown edge")


def _main_coord(cell: tuple[int, int], axis: str) -> int:
    return int(cell[0] if axis == "row" else cell[1])


def _step_dir(a: tuple[int, int], b: tuple[int, int]) -> int:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    return _DIRS_URDL.index((dr, dc))


def _path_turns(path: list[tuple[int, int]] | tuple[tuple[int, int], ...], init_dir: int) -> int:
    turns = 0
    prev_dir = init_dir
    for a, b in zip(path, path[1:]):
        nd = _step_dir(a, b)
        if nd != prev_dir:
            turns += 1
        prev_dir = nd
    return turns


def _coverage_interval(length: int, side: int, cross_size: int) -> tuple[int, int]:
    if side == 0:
        return 0, length - 1
    return cross_size - length, cross_size - 1


def _build_bar_infos(g: np.ndarray) -> list[dict]:
    infos: list[dict] = []
    for comp in components_mask((g == BLUE) | (g == RED), connectivity=4):
        coords = [tuple(map(int, pt)) for pt in comp]
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        blue_coords = [(r, c) for r, c in coords if int(g[r, c]) == BLUE]
        infos.append(
            {
                "coords": coords,
                "coord_set": set(coords),
                "blue_coords": blue_coords,
                "blue_count": len(blue_coords),
                "min_row": min(rows),
                "max_row": max(rows),
                "min_col": min(cols),
                "max_col": max(cols),
                "array": np.array(coords, dtype=int),
            }
        )
    return infos


def _bar_nearest_edge_covers_point(
    point: tuple[int, int],
    bar_infos: list[dict],
    shape: tuple[int, int],
) -> bool:
    r, c = point
    h, w = shape
    edge = _edge_of(r, c, h, w)
    assert edge is not None, "green terminals must lie on the border"

    if edge == "top":
        bar = min(bar_infos, key=lambda info: int(info["min_row"]))
        return int(bar["min_col"]) <= c <= int(bar["max_col"])
    if edge == "bottom":
        bar = max(bar_infos, key=lambda info: int(info["max_row"]))
        return int(bar["min_col"]) <= c <= int(bar["max_col"])
    if edge == "left":
        bar = min(bar_infos, key=lambda info: int(info["min_col"]))
        return int(bar["min_row"]) <= r <= int(bar["max_row"])
    bar = max(bar_infos, key=lambda info: int(info["max_col"]))
    return int(bar["min_row"]) <= r <= int(bar["max_row"])


def _infer_terminals(g: np.ndarray, bar_infos: list[dict]) -> tuple[tuple[int, int], tuple[int, int], int, str, int]:
    h, w = g.shape
    green_pts = [tuple(map(int, pt)) for pt in np.argwhere(g == GREEN)]
    assert len(green_pts) == 2, "grid must contain exactly two green border pixels"
    assert bar_infos, "grid must contain at least one obstacle bar"

    edges = [_edge_of(r, c, h, w) for r, c in green_pts]
    assert all(edge is not None for edge in edges), "green terminals must be on border edges"
    edge_set = set(edges)
    assert edge_set in ({"top", "bottom"}, {"left", "right"}), "green terminals must lie on opposite edges"

    covered = [_bar_nearest_edge_covers_point(pt, bar_infos, (h, w)) for pt in green_pts]
    assert sum(covered) == 1, "exactly one green terminal must align with the nearest bar coverage"

    start = green_pts[covered.index(True)]
    goal = green_pts[covered.index(False)]
    start_edge = _edge_of(start[0], start[1], h, w)
    assert start_edge is not None
    axis, sign = _axis_sign_for_edge(start_edge)
    return start, goal, _init_dir_for_edge(start_edge), axis, sign


def _sorted_bars(bar_infos: list[dict], axis: str, sign: int) -> list[dict]:
    key_name = "min_row" if axis == "row" else "min_col"
    return sorted(bar_infos, key=lambda info: sign * int(info[key_name]))


def _build_blocked(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    bar_mask = (g == BLUE) | (g == RED)
    forbidden = np.zeros((h, w), dtype=bool)

    for comp in components_mask(bar_mask, connectivity=4):
        blue_count = sum(1 for r, c in comp if int(g[r, c]) == BLUE)
        radius = int(blue_count)

        dist = np.full((h, w), -1, dtype=np.int16)
        q = deque()
        for r, c in comp:
            rr, cc = int(r), int(c)
            dist[rr, cc] = 0
            q.append((rr, cc))

        while q:
            r, c = q.popleft()
            d = int(dist[r, c])
            forbidden[r, c] = True
            if d == radius:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and int(dist[rr, cc]) == -1:
                    dist[rr, cc] = d + 1
                    q.append((rr, cc))

    return bar_mask | forbidden


def _search_path(
    blocked: np.ndarray,
    start: tuple[int, int],
    init_dir: int,
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    h, w = blocked.shape
    sr, sc = start
    gr, gc = goal
    blocked = np.array(blocked, copy=True)
    blocked[sr, sc] = False
    blocked[gr, gc] = False

    inf = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]
    goal_state: tuple[int, int, int] | None = None

    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), inf):
            continue
        if (r, c) == (gr, gc):
            goal_state = (r, c, dprev)
            break
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

    if goal_state is None:
        return None

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


def _search_to_opposite_edge(
    blocked: np.ndarray,
    start: tuple[int, int],
    init_dir: int,
    edge: str,
) -> list[tuple[int, int]] | None:
    h, w = blocked.shape
    sr, sc = start
    blocked = np.array(blocked, copy=True)
    blocked[sr, sc] = False

    def is_goal(r: int, c: int) -> bool:
        if edge == "top":
            return r == h - 1
        if edge == "bottom":
            return r == 0
        if edge == "left":
            return c == w - 1
        return c == 0

    inf = (10**9, 10**9)
    best_cost: dict[tuple[int, int, int], tuple[int, int]] = {(sr, sc, init_dir): (0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]
    goal_state: tuple[int, int, int] | None = None

    while heap:
        (steps, turns), r, c, dprev = heapq.heappop(heap)
        if (steps, turns) != best_cost.get((r, c, dprev), inf):
            continue
        if is_goal(r, c) and (r, c) != (sr, sc):
            goal_state = (r, c, dprev)
            break
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

    if goal_state is None:
        return None

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


def _first_clear_index(path: list[tuple[int, int]], bar: dict, axis: str, sign: int) -> int | None:
    if axis == "row":
        lo = int(bar["min_row"]) - int(bar["blue_count"])
        hi = int(bar["max_row"]) + int(bar["blue_count"])
    else:
        lo = int(bar["min_col"]) - int(bar["blue_count"])
        hi = int(bar["max_col"]) + int(bar["blue_count"])

    for idx, cell in enumerate(path):
        coord = _main_coord(cell, axis)
        if sign == 1 and coord > hi:
            return idx
        if sign == -1 and coord < lo:
            return idx
    return None


def _frontier_cells(
    blocked: np.ndarray,
    bar: dict,
    axis: str,
    sign: int,
) -> list[tuple[int, int]]:
    h, w = blocked.shape
    if axis == "row":
        coord = int(bar["max_row"]) + int(bar["blue_count"]) + 1 if sign == 1 else int(bar["min_row"]) - int(bar["blue_count"]) - 1
        if not (0 <= coord < h):
            return []
        return [(coord, c) for c in range(w) if not bool(blocked[coord, c])]

    coord = int(bar["max_col"]) + int(bar["blue_count"]) + 1 if sign == 1 else int(bar["min_col"]) - int(bar["blue_count"]) - 1
    if not (0 <= coord < w):
        return []
    return [(r, coord) for r in range(h) if not bool(blocked[r, coord])]


def _solve_static_path(grid: np.ndarray) -> list[tuple[int, int]] | None:
    g = np.array(grid, copy=True)
    bar_infos = _build_bar_infos(g)
    start, goal, init_dir, _axis, _sign = _infer_terminals(g, bar_infos)
    return _search_path(_build_blocked(g), start, init_dir, goal)


def _solve_sequential_path(
    grid: np.ndarray,
    *,
    with_meta: bool = False,
) -> list[tuple[int, int]] | tuple[list[tuple[int, int]], dict]:
    g = np.array(grid, copy=True)
    original_bar_infos = _build_bar_infos(g)
    start, goal, init_dir, axis, sign = _infer_terminals(g, original_bar_infos)
    ordered_bars = _sorted_bars(original_bar_infos, axis, sign)

    states = [g.copy()]
    for bar in ordered_bars:
        nxt = states[-1].copy()
        for r, c in bar["blue_coords"]:
            nxt[r, c] = RED
        states.append(nxt)
    blocked_stages = [_build_blocked(state) for state in states]

    @lru_cache(maxsize=None)
    def _solve_from(stage_idx: int, r: int, c: int, dprev: int):
        current = (int(r), int(c))
        if stage_idx == len(ordered_bars):
            tail = _search_path(blocked_stages[stage_idx], current, dprev, goal)
            if tail is None:
                return None
            tail_t = tuple(tail)
            return tail_t, ()

        bar = ordered_bars[stage_idx]
        frontier = _frontier_cells(blocked_stages[stage_idx], bar, axis, sign)
        assert frontier, "each obstacle stage must expose at least one frontier cell"
        ranked: list[tuple[tuple[int, int, int, int, int], list[tuple[int, int]], int, tuple[int, int]]] = []
        for cand in frontier:
            seg = _search_path(blocked_stages[stage_idx], current, dprev, cand)
            if seg is None:
                continue
            clear_idx = _first_clear_index(seg, bar, axis, sign)
            if clear_idx is None or clear_idx != len(seg) - 1:
                continue
            seg_cost = (len(seg) - 1, _path_turns(seg, dprev))
            cross_gap = abs((cand[1] - goal[1]) if axis == "row" else (cand[0] - goal[0]))
            rank = (
                abs(cand[0] - goal[0]) + abs(cand[1] - goal[1]),
                cross_gap,
                seg_cost[0],
                seg_cost[1],
                cand[0] * 100 + cand[1],
            )
            next_dir = dprev if len(seg) == 1 else _step_dir(seg[-2], seg[-1])
            ranked.append((rank, seg, next_dir, cand))

        ranked.sort(key=lambda item: item[0])
        for _rank, seg, next_dir, cand in ranked:
            rest = _solve_from(stage_idx + 1, cand[0], cand[1], next_dir)
            if rest is None:
                continue
            joined = tuple(seg[:-1] + list(rest[0]))
            return joined, (cand,) + rest[1]
        return None

    solved = _solve_from(0, start[0], start[1], init_dir)
    assert solved is not None, "a valid sequential route to the explicit endpoint must exist"
    full_path = list(solved[0])

    if not with_meta:
        return full_path
    return full_path, {"start": start, "goal": goal, "frontier_choices": list(solved[1])}


def _relaxed_reuse_count(grid: np.ndarray, path: list[tuple[int, int]]) -> int:
    g = np.array(grid, copy=True)
    bar_mask = (g == BLUE) | (g == RED)
    initial_blocked = _build_blocked(g)
    count = 0
    terminals = {tuple(map(int, pt)) for pt in np.argwhere(g == GREEN)}
    for cell in path:
        if cell in terminals:
            continue
        r, c = cell
        if bool(initial_blocked[r, c]) and not bool(bar_mask[r, c]):
            count += 1
    return count


def _path_diff_size(path_a: list[tuple[int, int]], path_b: list[tuple[int, int]] | None) -> int:
    if path_b is None:
        return len(set(path_a))
    return len(set(path_a) ^ set(path_b))


def _stage_clear_cells(
    path: list[tuple[int, int]],
    bar_infos: list[dict],
    axis: str,
    sign: int,
) -> list[tuple[int, int]]:
    clears: list[tuple[int, int]] = []
    ordered = _sorted_bars(bar_infos, axis, sign)
    start_idx = 0
    for bar in ordered:
        subpath = path[start_idx:]
        clear_rel = _first_clear_index(subpath, bar, axis, sign)
        assert clear_rel is not None, "path must clear every ordered bar"
        clear_abs = start_idx + clear_rel
        clears.append(path[clear_abs])
        start_idx = clear_abs
    return clears


def _stage_choice_gap_count(grid: np.ndarray, seq_path: list[tuple[int, int]], static_path: list[tuple[int, int]] | None) -> int:
    if static_path is None:
        return len(_build_bar_infos(grid))
    bar_infos = _build_bar_infos(grid)
    start, _goal, _init_dir, axis, sign = _infer_terminals(np.array(grid, copy=True), bar_infos)
    _ = start
    seq_clears = _stage_clear_cells(seq_path, bar_infos, axis, sign)
    static_clears = _stage_clear_cells(static_path, bar_infos, axis, sign)
    return sum(1 for a, b in zip(seq_clears, static_clears) if a != b)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a green route between two edge terminals by clearing obstacle bars one at a time and repeatedly replanning toward the endpoint.

    Input structure:
    - bg = azure (8). Red/blue bars (2/1) are wall-attached obstacles of the same type as in the base task.
    - Exactly two green (3) border pixels appear on opposite sides: one is the true start and the other is the exact destination.

    Rule:
    - Each bar blocks its own cells plus every cell within Manhattan distance <= its current blue-pixel count.
    - The start terminal is the green border pixel aligned with the nearest bar's span. From there, the path repeatedly plans toward the endpoint, but only advances until it has just cleared the next obstacle bar in order; that bar's blue pixels then turn red before the next planning round.
    - After all bars are processed, the remaining route segment reaches the other green terminal exactly, and every blue pixel in the grid has become red.

    Key insight:
    - Blue pixels are not just static danger widths: in this fused variant they are consumed bar by bar, so later route segments may reuse corridor cells that were forbidden earlier, while still being steered by the explicit second green endpoint.
    """
    g = np.array(grid, copy=True)
    path = _solve_sequential_path(g)

    out = g.copy()
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


def _choose_combo_blues(main_hi: int) -> tuple[int, ...]:
    choices = [
        combo
        for combo in itertools.product((1, 2), repeat=4)
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
) -> tuple[np.ndarray, list[int], list[int], list[int]] | None:
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
    if sides[0] == 0:
        start_col = random.randint(first_lo + 1, first_hi)
    else:
        start_col = random.randint(first_lo, first_hi - 1)
    grid[0, start_col] = GREEN
    return grid, rows, lengths, sides


def _orient_grid(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _goal_col_order(w: int, last_side: int, legacy_exit_col: int) -> list[int]:
    if last_side == 0:
        primary = list(range(w - 1, -1, -1))
    else:
        primary = list(range(w))
    return sorted(primary, key=lambda col: (-abs(col - legacy_exit_col), primary.index(col)))


def _choose_goal_col(
    canonical: np.ndarray,
    rows: list[int],
    lengths: list[int],
    sides: list[int],
) -> tuple[int, dict] | None:
    h, w = canonical.shape
    start = tuple(map(int, np.argwhere(canonical == GREEN)[0]))
    blocked = _build_blocked(canonical)
    blocked[start] = False
    legacy_path = _search_to_opposite_edge(blocked, start, 2, "top")
    if legacy_path is None:
        return None
    legacy_exit_col = int(legacy_path[-1][1])

    last_lo, last_hi = _coverage_interval(lengths[-1], sides[-1], w)
    candidates = _goal_col_order(w, sides[-1], legacy_exit_col)

    fallback: tuple[int, dict] | None = None
    for col in candidates:
        if last_lo <= col <= last_hi:
            continue
        if bool(blocked[h - 1, col]):
            continue

        trial = canonical.copy()
        trial[h - 1, col] = GREEN
        try:
            seq_path = _solve_sequential_path(trial)
        except Exception:
            continue
        static_path = _solve_static_path(trial)
        stage_gap = _stage_choice_gap_count(trial, seq_path, static_path)
        if stage_gap <= 0:
            continue
        diff = _path_diff_size(seq_path, static_path)
        if diff <= 0:
            continue

        endpoint_shift = abs(col - legacy_exit_col)
        metrics = {
            "stage_choice_gap_count": stage_gap,
            "path_diff_size": diff,
            "endpoint_shift": endpoint_shift,
            "legacy_exit_col": legacy_exit_col,
            "goal_col": col,
            "bar_count": len(rows),
        }
        if endpoint_shift >= 2:
            return col, metrics
        if fallback is None:
            fallback = (col, metrics)

    return fallback


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in ("top", "bottom"):
        main_lo, main_hi = ROWS_LO, ROWS_HI
        cross_lo, cross_hi = COLS_LO, COLS_HI
    else:
        main_lo, main_hi = COLS_LO, COLS_HI
        cross_lo, cross_hi = ROWS_LO, ROWS_HI

    count = 4
    blues = _choose_combo_blues(main_hi)
    min_main = max(main_lo, _min_main_len(blues))
    main_size = random.randint(min_main, min(main_hi, min_main + 3))
    cross_pref_lo = max(cross_lo, 22)
    cross_pref_hi = min(cross_hi, 24)
    cross_size = random.randint(cross_pref_lo, cross_pref_hi)
    first_side = random.randrange(2)

    built = _canonical_grid(main_size, cross_size, blues, first_side)
    if built is None:
        return None
    canonical, rows, lengths, sides = built

    goal_pick = _choose_goal_col(canonical, rows, lengths, sides)
    if goal_pick is None:
        return None
    goal_col, _metrics = goal_pick
    canonical[-1, goal_col] = GREEN

    inp = _orient_grid(canonical, edge)
    try:
        out = main(inp)
        seq_path = _solve_sequential_path(inp)
        static_path = _solve_static_path(inp)
        if _stage_choice_gap_count(inp, seq_path, static_path) <= 0:
            return None
        if _path_diff_size(seq_path, static_path) <= 0:
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
    raise RuntimeError("failed to generate a valid cb2d8a2c combo input")
