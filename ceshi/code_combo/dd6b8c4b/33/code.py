# Combo variant of dd6b8c4b (index 33)
# combo_spec: D2/3+D3/0+D15/0
# Name: 非背景均为障碍+逐步收集+反向距离排序
#
# Design analysis
# a) 具体化与行为追踪:
#    - 画面仍是固定 11x11，中心 (5,5) 的红点与外围 3x3 绿色目标块保留；橙色是唯一可走背景，墙和栗色都不再可穿越。
#    - 原题会把 `floor+maroon` 都当作 BFS 通道，所以一条走廊里的整串栗色会一次性全部变成候选，再按距离从近到远选前 k 个。
#    - 本融合版改成：BFS 只能走橙色；当前“可收集”的栗色只是不穿越它就能从已达地板触碰到的那些边界栗色。每轮只拿走 1 个，并立即把该位置变回地板后重算。
#    - 具体小例子：若北/西/东三条单格走廊的首个栗色距离分别是 1、3、2，且每条后面还各自连着一串连续栗色，则原题会把三条走廊里所有栗色都当作候选并偏向近端；本题第一轮只会在 {1,2,3} 中选 3，移除后西侧下一颗才会在第二轮以距离 4 露出来，因此顺序会变成 3,4,5...，而不是 1,2,3...。
# b) 复杂度校准:
#    - 若总可收集数 <= 3，或所有首层距离相同，则“最远优先”和“逐步解锁”都不够显眼。
#    - 生成器因此固定构造 3 条独立窄走廊：初始只暴露 3 个候选，但总可收集数至少 10，从而保证 `D15/0` 必然解锁后续栗色，且 `D3/0` 的远优先会改变哪几条走廊被清空。
#    - 走廊首个栗色距离被约束成互不相同，避免首轮排序退化为纯行列 tie-break。
# c) 隐式假设审查:
#    - 原 solver 的 `passable={floor,9}` 在本题已失效；现在栗色必须从通道集合里移除，否则不会产生“前一个栗色挡住后一个”的现象。
#    - 原 solver 一次性 `sort()+切前 k` 也失效；必须把“找候选 / 选 1 个 / 擦除 / 重算”写成循环，否则 `D15/0` 不会真正参与。
#    - 原先把 maroon 自己的 BFS 距离直接读出 `dist[r,c]` 不再成立，因为 maroon 不在 BFS 图里；现在距离改为“相邻已达地板的最短距离 + 1”，并额外兼容直接贴住目标块的栗色。
#    - 新 solver 仍假设存在唯一红中心、目标块固定为其周围 3x3；generator 会继续严格满足。
# d) 放置合法性:
#    - 生成器从全墙底板出发，只雕刻不相交的 floor 走廊和孤立口袋；目标块、走廊、栗色、墙之间不会重叠，也不会越界。
#    - 每个栗色源点都先对应到底层 floor，再覆盖成 9，保证被收集后恢复成橙色是合法的。
# e) 视觉可观察性:
#    - 输入里能直接看见三条从目标块通向外侧的单格走廊，每条走廊上有连续栗色链；输出里目标块被填满，同时只有某些走廊被大幅清空、另一些近端栗色被保留。
#    - 由于初始只暴露 3 个栗色，但最终会搬走 9 个，人类能从样例看出“不是一次性全判定”，而是边开路边搬运。
# f) 可视化整洁度:
#    - 元素只有背景、墙、目标块和栗色链，且每条链各占一条窄走廊，没有叠色或重叠对象，视觉关系清楚。
#    - 不可达干扰项只放在四角孤立口袋里，不会和主走廊混淆。
# g) 变化的可推断性:
#    - 颜色语义没有新增映射；变化完全由空间关系决定：橙色可走，其他颜色阻断，最远的暴露栗色先消失。
#    - 生成器显式拒绝“融合版输出 == 原题式单步近优先输出”的布局，确保样例真正体现组合规则。
# 9) main() 的新隐式假设:
#    - main() 允许处理“直接贴着目标块的栗色”，但 generator 默认仍保留至少一个 floor 开口，避免把所有入口都堵死导致样例过于晦涩。
#    - main() 依赖“被收集的栗色底下应当是 floor”；generator 通过先雕 floor 再覆盖 maroon 来保证这一点。
#    - main() 按 `(-distance, row, col)` 稳定打破平局；generator 让首层距离互异，所以核心行为不依赖不可见 tie-break。

import random

import numpy as np

from arc_utils import neighbors4


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9

DEST_CAPACITY = 9
GEN_RETRIES = 200
ACTIVE_ARMS = 3
REACHABLE_TOTAL_RANGE = (10, 13)
DISTRACTOR_RANGE = (0, 2)

PATH_LIBRARY: dict[str, list[tuple[int, int]]] = {
    "north": [(3, 5), (2, 5), (1, 5), (1, 4), (1, 3), (1, 2), (1, 1)],
    "west": [(5, 3), (5, 2), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1)],
    "east": [(5, 7), (5, 8), (5, 9), (4, 9), (3, 9), (2, 9), (1, 9)],
    "south": [(7, 5), (8, 5), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9)],
}
DISTRACTOR_POCKETS = [(0, 0), (0, 10), (10, 0), (10, 10)]
INF = 10**9


def _destination_cells(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _locate_destination(grid: np.ndarray) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red (2) center pixel"
    cr, cc = map(int, centers[0])
    dest_cells = _destination_cells(cr, cc, grid.shape[0], grid.shape[1])
    return dest_cells, set(dest_cells)


def _infer_floor(grid: np.ndarray, dest_cells: list[tuple[int, int]], dest_set: set[tuple[int, int]]) -> int:
    h, w = grid.shape
    ring_colors: list[int] = []
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set:
                ring_colors.append(int(grid[rr, cc2]))

    non_special = [x for x in ring_colors if x not in (RED_CENTER, DEST_COLOR, MAROON)]
    assert non_special, "ring around destination must contain floor/wall cells"
    vals, cnts = np.unique(non_special, return_counts=True)
    majority = int(vals[cnts.argmax()])
    others = [int(x) for x in np.unique(grid) if int(x) not in (RED_CENTER, DEST_COLOR, MAROON, majority)]
    if len(others) == 1:
        return others[0]
    if others:
        return max(others, key=lambda color: int((grid == color).sum()))
    return majority


def _floor_distance_map(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
    dest_set: set[tuple[int, int]],
    floor: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    h, w = grid.shape
    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set and int(grid[rr, cc2]) == floor:
                if (rr, cc2) not in seen:
                    seen.add((rr, cc2))
                    starts.append((rr, cc2))

    dist = np.full((h, w), INF, dtype=int)
    q = list(starts)
    for r, c in starts:
        dist[r, c] = 0

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        nd = int(dist[r, c]) + 1
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set:
                if int(grid[rr, cc2]) == floor and int(dist[rr, cc2]) == INF:
                    dist[rr, cc2] = nd
                    q.append((rr, cc2))
    return dist, starts


def _reachable_frontier_maroons(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
    dest_set: set[tuple[int, int]],
    floor: int,
) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    dist, _starts = _floor_distance_map(grid, dest_cells, dest_set, floor)
    best: dict[tuple[int, int], int] = {}

    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and int(grid[rr, cc2]) == MAROON:
                pos = (rr, cc2)
                prev = best.get(pos)
                if prev is None or 0 < prev:
                    best[pos] = 0

    for r in range(h):
        for c in range(w):
            if int(dist[r, c]) == INF:
                continue
            step = int(dist[r, c]) + 1
            for rr, cc2 in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc2 < w and int(grid[rr, cc2]) == MAROON:
                    pos = (rr, cc2)
                    prev = best.get(pos)
                    if prev is None or step < prev:
                        best[pos] = step

    maroons = [(d, r, c) for (r, c), d in best.items()]
    maroons.sort(key=lambda t: (-t[0], t[1], t[2]))
    return maroons


def _simulate_combo(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int]], list[int], list[tuple[int, int, int]], int, list[tuple[int, int]]]:
    work = np.asarray(grid, dtype=int).copy()
    dest_cells, dest_set = _locate_destination(work)
    floor = _infer_floor(work, dest_cells, dest_set)

    chosen: list[tuple[int, int]] = []
    chosen_distances: list[int] = []
    initial_frontier = _reachable_frontier_maroons(work, dest_cells, dest_set, floor)

    while len(chosen) < len(dest_cells):
        frontier = _reachable_frontier_maroons(work, dest_cells, dest_set, floor)
        if not frontier:
            break
        d, r, c = frontier[0]
        chosen.append((r, c))
        chosen_distances.append(int(d))
        work[r, c] = floor

    return chosen, chosen_distances, initial_frontier, floor, dest_cells


def _original_like_output(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    out = grid.copy()
    dest_cells, dest_set = _locate_destination(grid)
    floor = _infer_floor(grid, dest_cells, dest_set)
    passable = {floor, MAROON}
    h, w = grid.shape

    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set and int(grid[rr, cc2]) in passable:
                if (rr, cc2) not in seen:
                    seen.add((rr, cc2))
                    starts.append((rr, cc2))

    dist = np.full((h, w), INF, dtype=int)
    q = list(starts)
    for r, c in starts:
        dist[r, c] = 0

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        nd = int(dist[r, c]) + 1
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set:
                if int(grid[rr, cc2]) in passable and int(dist[rr, cc2]) == INF:
                    dist[rr, cc2] = nd
                    q.append((rr, cc2))

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    maroons = [t for t in maroons if t[0] < INF]
    maroons.sort()

    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}
    for r, c in chosen:
        out[r, c] = floor
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Iteratively move the farthest currently exposed maroon pixels into the 3x3 block around the lone red center.

    Input structure:
    - Orange cells are the only traversable background; green/red destination cells, magenta walls, and maroon source pixels all behave as solid obstacles.
    - A unique red cell marks the middle of a green 3x3 destination block, while maroon pixels sit on narrow floor corridors or in isolated pockets outside it.

    Rule:
    - Repeatedly flood-fill only through orange floor cells starting from floor openings next to the destination.
    - At each step, any maroon touching the reached floor region is collectible; choose the farthest such maroon (ties: row, then column), erase it back to floor, and recompute reachability.
    - Stop when the 3x3 block has taken 9 maroons or when no maroon is currently reachable, then write the collected maroons into the destination in row-major order.

    Key insight:
    - Because maroon pixels now block the flood-fill instead of extending it, a removed maroon can reveal new maroons behind it; the final survivors therefore depend on both the iterative unlocking process and the reversed distance preference.
    """

    grid = np.asarray(grid, dtype=int)
    out = grid.copy()
    chosen, _distances, _initial_frontier, floor, dest_cells = _simulate_combo(grid)

    for r, c in chosen:
        out[r, c] = floor
    for r, c in dest_cells[: len(chosen)]:
        out[r, c] = MAROON
    return out


def _blank_grid() -> tuple[np.ndarray, list[tuple[int, int]], set[tuple[int, int]]]:
    grid = np.full((GRID_H, GRID_W), WALL_COLOR, dtype=int)
    dest_cells = _destination_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)
    dest_set = set(dest_cells)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER
    return grid, dest_cells, dest_set


def _paint_floor(grid: np.ndarray, cells: list[tuple[int, int]], dest_set: set[tuple[int, int]]) -> None:
    for r, c in cells:
        assert 0 <= r < GRID_H and 0 <= c < GRID_W
        assert (r, c) not in dest_set
        grid[r, c] = FLOOR_COLOR


def _choose_arm_lengths(leads: list[int], path_lengths: list[int]) -> list[int] | None:
    for _ in range(80):
        lengths = [random.randint(3, plen - lead) for lead, plen in zip(leads, path_lengths)]
        total = sum(lengths)
        if REACHABLE_TOTAL_RANGE[0] <= total <= REACHABLE_TOTAL_RANGE[1]:
            return lengths
    return None


def _combo_metrics(grid: np.ndarray) -> dict[str, object]:
    chosen, distances, initial_frontier, _floor, _dest = _simulate_combo(grid)
    combo_out = main(grid)
    original_out = _original_like_output(grid)
    return {
        "chosen_count": len(chosen),
        "distances": distances,
        "initial_frontier_count": len(initial_frontier),
        "unlocked_count": max(0, len(chosen) - len(initial_frontier)),
        "differs_from_original": not np.array_equal(combo_out, original_out),
    }


def _gen_attempt() -> np.ndarray | None:
    grid, dest_cells, dest_set = _blank_grid()

    active_names = random.sample(sorted(PATH_LIBRARY), ACTIVE_ARMS)
    lead_values = random.sample([1, 2, 3], ACTIVE_ARMS)
    path_lengths = [len(PATH_LIBRARY[name]) for name in active_names]
    chain_lengths = _choose_arm_lengths(lead_values, path_lengths)
    if chain_lengths is None:
        return None

    source_cells: list[tuple[int, int]] = []
    for name, lead, chain_len in zip(active_names, lead_values, chain_lengths):
        path = PATH_LIBRARY[name]
        used = path[: lead + chain_len]
        _paint_floor(grid, used, dest_set)
        source_cells.extend(path[lead : lead + chain_len])

    distractor_count = random.randint(*DISTRACTOR_RANGE)
    distractor_cells = random.sample(DISTRACTOR_POCKETS, distractor_count)
    _paint_floor(grid, distractor_cells, dest_set)

    inp = grid.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    metrics = _combo_metrics(inp)
    if metrics["chosen_count"] != DEST_CAPACITY:
        return None
    if metrics["initial_frontier_count"] >= metrics["chosen_count"]:
        return None
    if not metrics["differs_from_original"]:
        return None

    distances = metrics["distances"]
    if len(distances) < DEST_CAPACITY:
        return None
    if len(set(distances[:3])) < 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
