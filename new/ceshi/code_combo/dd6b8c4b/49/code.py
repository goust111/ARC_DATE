# Combo variant of dd6b8c4b (index 49)
# combo_spec: D9/2+D19/1
# Name: 路径像素数距离+迷宫式墙壁

from __future__ import annotations

from heapq import heappop, heappush
from pathlib import Path
import random
import sys

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import neighbors4


"""
design_analysis:
  a_concretization_and_trace:
    - 原题从目标 3x3 块外的可通行格做普通 BFS，把所有可达 maroon 按步数远近搬回目标块。
      例如某个 maroon 离入口 4 步、另一个 6 步，那么 4 步的先进入目标，即使那条路一直贴着墙走。
    - 融合后仍然只允许沿 orange/maroon 的 4 连通路径到达，但“距离”不再是步数，而是：
      路径上每个经过的可通行像素，都要累计它 4 邻域里有多少个 magenta 墙像素。总代价越小越近。
    - 具体小例子：一枚 maroon 在入口旁边的 2 格死胡同里，路径两格都被三面墙夹住，总代价可能是
      3 + 3 = 6；另一枚更远的 maroon 在较宽主通道深处，沿路 5 格每格只贴 1 面墙，总代价只有 5。
      原题会选 2 步的近 token；融合后会优先选 5 步但更开阔的 token。
    - D9/2 的作用不是单独换个外形，而是提供“窄支路 / 宽主路 / 拐角 / 死胡同”这些局部结构，让
      墙邻居计数真正改变排序；如果还是原来的松散框架，D19/1 就很容易退化成“几乎等价于步数”。
  b_complexity_calibration:
    - 距离规则只引入一个新量：墙邻居总代价；没有再叠加颜色映射或多阶段条件，避免自由度过高。
    - 迷宫保留单一目标块、单一墙色、单一 token 色；复杂度来自通道几何而不是对象种类膨胀。
    - generator 强制同时出现较宽主通道和狭窄支路，并筛掉“新旧前 k 个来源集合完全相同”的样本，
      避免视觉上看不出与原题差别。
  c_implicit_assumption_audit:
    - 原 main() 的核心假设是“最短路 = 最少步数”。融合后这行假设失效，必须用 Dijkstra 替代 BFS。
    - 原 main() 默认距离与局部墙形状无关；融合后每次扩展都依赖候选格的墙邻居数，因此生成器必须
      保证 reachable 区里确实有不同宽度、不同贴墙程度的通道。
    - 原 main() 只需要至少一个入口；融合后若把所有可达通道都做成等宽直廊，新规则会退化。生成器
      因此显式做“主通道加宽 + 支路保持狭窄 + 近处死胡同”。
    - 原 main() 依赖“可达 maroon 数 = 要填入目标的数量”。这一点仍保留，所以生成器只放 k 个可达
      maroon，其余 maroon 全都放在隔离口袋里做 distractor。
  d_placement_legality:
    - 目标块固定在 11x11 的中心 3x3；除目标块外，其余单元只会是 floor、wall 或 maroon。
    - 迷宫从单个入口接到外部主通道，支路和加宽单元都只在非目标区域内生成，不越界、不覆盖目标块。
    - distractor 只放在与主迷宫 4 不连通的孤立 floor 口袋里，因此不会被误判为可达来源。
  e_visual_observability:
    - 人类可以直接看到：有些 maroon 虽然步数更近，但卡在三面贴墙的死胡同；另一些更远，却位于
      较宽或半开阔的主通道里。输出若优先搬走后者，说明排序标准不是普通 BFS。
    - 由于目标块仍按 row-major 填入，观察者只需比较“哪些 maroon 消失了”，就能对照迷宫几何推断
      新的距离规则，不需要追踪额外颜色编码。
  f_visual_cleanliness:
    - 场景仍只有五种颜色，且各自语义稳定：orange 地板、magenta 墙、green/red 目标、maroon token。
    - 迷宫是单层平面结构，没有叠加标记；加宽只表现为局部 corridor 变宽，不会让对象边界混乱。
    - distractor 口袋数量受限，避免远处出现过多散点破坏读图。
  g_inferability_and_new_assumptions:
    - main() 新增假设：墙色固定可见且路径代价由 4 邻域墙数量决定。generator 始终只使用 magenta=6
      作为墙，并构造让这种“贴墙程度”可肉眼比较的通道。
    - main() 新增假设：至少有一个可通行入口紧邻目标块。generator 固定从四个基准入口之一开门。
    - main() 新增假设：存在至少一个 k，使“按墙代价选出的来源集合”和“按步数选出的来源集合”不同。
      generator 在采样时显式检查这一点，不满足就重造迷宫。
"""


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9

CARDINAL_GATES = [(3, 5), (5, 3), (5, 7), (7, 5)]
MAIN_LEN_RANGE = (11, 16)
BRANCH_COUNT_RANGE = (2, 4)
BRANCH_LEN_RANGE = (2, 4)
WIDEN_COUNT_RANGE = (4, 7)
POCKET_COUNT_RANGE = (0, 4)
K_RANGE = (2, 9)
MIN_REACHABLE_FLOOR = 12
GEN_RETRIES = 500


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= int(r) < GRID_H and 0 <= int(c) < GRID_W


def _dest_cells(center_r: int = CENTER_R, center_c: int = CENTER_C) -> list[tuple[int, int]]:
    return [(r, c) for r in range(center_r - 1, center_r + 2) for c in range(center_c - 1, center_c + 2)]


def _dest_adjacent_ring(dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    ring = set()
    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if _in_bounds(nr, nc) and (nr, nc) not in dest_set:
                ring.add((nr, nc))
    return ring


def _wall_exposure(grid: np.ndarray, r: int, c: int) -> int:
    return sum(1 for nr, nc in neighbors4(r, c) if _in_bounds(nr, nc) and int(grid[nr, nc]) == WALL_COLOR)


def _starts_from_dest(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in sorted(dest_set):
        for nr, nc in neighbors4(r, c):
            if not _in_bounds(nr, nc) or (nr, nc) in dest_set:
                continue
            if int(grid[nr, nc]) in (FLOOR_COLOR, MAROON) and (nr, nc) not in seen:
                seen.add((nr, nc))
                starts.append((nr, nc))
    return starts


def _step_distances(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> np.ndarray:
    starts = _starts_from_dest(grid, dest_set)
    assert starts, "maze must expose at least one passable entry beside the destination"
    dist = np.full(grid.shape, -1, dtype=int)
    q = list(starts)
    for r, c in starts:
        dist[r, c] = 0
    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        for nr, nc in neighbors4(r, c):
            if not _in_bounds(nr, nc) or (nr, nc) in dest_set:
                continue
            if int(grid[nr, nc]) not in (FLOOR_COLOR, MAROON):
                continue
            if int(dist[nr, nc]) != -1:
                continue
            dist[nr, nc] = int(dist[r, c] + 1)
            q.append((nr, nc))
    return dist


def _exposure_distances(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> np.ndarray:
    starts = _starts_from_dest(grid, dest_set)
    assert starts, "maze must expose at least one passable entry beside the destination"
    inf = 10**9
    dist = np.full(grid.shape, inf, dtype=int)
    heap: list[tuple[int, int, int]] = []
    for r, c in starts:
        cost = _wall_exposure(grid, r, c)
        if cost < int(dist[r, c]):
            dist[r, c] = cost
            heappush(heap, (cost, r, c))
    while heap:
        cost, r, c = heappop(heap)
        if cost != int(dist[r, c]):
            continue
        for nr, nc in neighbors4(r, c):
            if not _in_bounds(nr, nc) or (nr, nc) in dest_set:
                continue
            if int(grid[nr, nc]) not in (FLOOR_COLOR, MAROON):
                continue
            new_cost = int(cost + _wall_exposure(grid, nr, nc))
            if new_cost < int(dist[nr, nc]):
                dist[nr, nc] = new_cost
                heappush(heap, (new_cost, nr, nc))
    return dist


def _reachable_cells(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    step = _step_distances(grid, dest_set)
    return {(int(r), int(c)) for r, c in np.argwhere(step >= 0) if (int(r), int(c)) not in dest_set}


def _degree_in(cell: tuple[int, int], cells: set[tuple[int, int]]) -> int:
    r, c = cell
    return sum(1 for nr, nc in neighbors4(r, c) if (nr, nc) in cells)


def _is_boundary(cell: tuple[int, int]) -> bool:
    r, c = cell
    return r in (0, GRID_H - 1) or c in (0, GRID_W - 1)


def _grow_main_path(
    gate: tuple[int, int],
    dest_set: set[tuple[int, int]],
    forbidden_ring: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    target_len = random.randint(*MAIN_LEN_RANGE)
    min_len = max(MAIN_LEN_RANGE[0], target_len - 2)
    used = {gate}
    path = [gate]
    budget = 0

    def dfs() -> bool:
        nonlocal budget
        budget += 1
        if budget > 5000:
            return False
        cur = path[-1]
        if len(path) >= target_len and _is_boundary(cur):
            return True
        candidates = []
        for nr, nc in neighbors4(*cur):
            nxt = (nr, nc)
            if not _in_bounds(nr, nc) or nxt in used or nxt in dest_set:
                continue
            if nxt in forbidden_ring and nxt != gate:
                continue
            if _degree_in(nxt, used) != 1:
                continue
            outward = abs(nr - CENTER_R) + abs(nc - CENTER_C)
            future = 0
            for ar, ac in neighbors4(nr, nc):
                adj = (ar, ac)
                if not _in_bounds(ar, ac) or adj in used or adj in dest_set:
                    continue
                if adj in forbidden_ring and adj != gate:
                    continue
                future += 1
            score = future + outward + random.random()
            candidates.append((score, nxt))
        candidates.sort(reverse=True)
        for _score, nxt in candidates:
            used.add(nxt)
            path.append(nxt)
            if dfs():
                return True
            path.pop()
            used.remove(nxt)
        return len(path) >= min_len and _is_boundary(cur)

    if dfs() and len(path) >= min_len:
        return list(path)
    return None


def _grow_branch(
    floor_cells: set[tuple[int, int]],
    anchor: tuple[int, int],
    dest_set: set[tuple[int, int]],
    forbidden_ring: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    best: list[tuple[int, int]] = []
    target = random.randint(*BRANCH_LEN_RANGE)
    for _ in range(30):
        branch = [anchor]
        used = set(floor_cells)
        used.add(anchor)
        while len(branch) - 1 < target:
            cur = branch[-1]
            candidates = []
            for nr, nc in neighbors4(*cur):
                nxt = (nr, nc)
                if not _in_bounds(nr, nc) or nxt in used or nxt in dest_set:
                    continue
                if nxt in forbidden_ring:
                    continue
                if _degree_in(nxt, used) != 1:
                    continue
                score = abs(nr - CENTER_R) + abs(nc - CENTER_C) + random.random()
                candidates.append((score, nxt))
            if not candidates:
                break
            candidates.sort(reverse=True)
            nxt = candidates[0][1]
            branch.append(nxt)
            used.add(nxt)
        if len(branch) > len(best):
            best = branch[1:]
        if len(best) >= target:
            break
    return best


def _add_widenings(
    floor_cells: set[tuple[int, int]],
    main_path: list[tuple[int, int]],
    dest_set: set[tuple[int, int]],
    forbidden_ring: set[tuple[int, int]],
) -> None:
    candidates: list[tuple[int, int, int, int]] = []
    for idx in range(2, len(main_path) - 2):
        pr, pc = main_path[idx - 1]
        r, c = main_path[idx]
        nr, nc = main_path[idx + 1]
        if pr == r == nr:
            for dr in (-1, 1):
                candidates.append((idx, dr, 0, 1))
        elif pc == c == nc:
            for dc in (-1, 1):
                candidates.append((idx, 0, dc, 1))
    random.shuffle(candidates)
    target = min(len(candidates), random.randint(*WIDEN_COUNT_RANGE))
    added = 0
    for idx, dr, dc, _ in candidates:
        if added >= target:
            break
        base_cells = [main_path[idx]]
        pr, pc = main_path[idx]
        sr, sc = pr + dr, pc + dc
        side = (sr, sc)
        if not _in_bounds(sr, sc) or side in floor_cells or side in dest_set:
            continue
        if side in forbidden_ring:
            continue
        if _degree_in(side, floor_cells) > 2:
            continue
        floor_cells.add(side)
        added += 1
        if idx + 1 < len(main_path) - 1:
            nr, nc = main_path[idx + 1]
            extra = (nr + dr, nc + dc)
            if (
                _in_bounds(*extra)
                and extra not in floor_cells
                and extra not in dest_set
                and extra not in forbidden_ring
                and _degree_in(extra, floor_cells) <= 2
            ):
                floor_cells.add(extra)


def _add_isolated_pockets(
    floor_cells: set[tuple[int, int]],
    dest_set: set[tuple[int, int]],
    forbidden_ring: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    pockets: set[tuple[int, int]] = set()
    candidates = [(r, c) for r in range(GRID_H) for c in range(GRID_W)]
    random.shuffle(candidates)
    target = random.randint(*POCKET_COUNT_RANGE)
    for cell in candidates:
        if len(pockets) >= target:
            break
        r, c = cell
        if cell in floor_cells or cell in dest_set or cell in forbidden_ring:
            continue
        if any((nr, nc) in floor_cells for nr, nc in neighbors4(r, c)):
            continue
        if any((nr, nc) in pockets for nr, nc in neighbors4(r, c)):
            continue
        pockets.add(cell)
    return pockets


def _build_base_grid() -> tuple[np.ndarray, list[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    dest_cells = _dest_cells()
    dest_set = set(dest_cells)
    forbidden_ring = _dest_adjacent_ring(dest_set)
    gate = random.choice(CARDINAL_GATES)

    main_path = _grow_main_path(gate, dest_set, forbidden_ring)
    if main_path is None:
        return None

    floor_cells = set(main_path)
    branch_slots = main_path[2:-2]
    random.shuffle(branch_slots)
    branches_added = 0
    target_branches = random.randint(*BRANCH_COUNT_RANGE)
    for anchor in branch_slots:
        if branches_added >= target_branches:
            break
        branch = _grow_branch(floor_cells, anchor, dest_set, forbidden_ring)
        if len(branch) < BRANCH_LEN_RANGE[0]:
            continue
        floor_cells.update(branch)
        branches_added += 1
    if branches_added == 0:
        return None

    _add_widenings(floor_cells, main_path, dest_set, forbidden_ring)
    pockets = _add_isolated_pockets(floor_cells, dest_set, forbidden_ring)

    if len(floor_cells) < MIN_REACHABLE_FLOOR:
        return None

    grid = np.full((GRID_H, GRID_W), WALL_COLOR, dtype=int)
    for r, c in floor_cells | pockets:
        grid[r, c] = FLOOR_COLOR
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER
    return grid, main_path, floor_cells, pockets


def _sorted_reachable_by_metric(
    grid: np.ndarray,
    floor_cells: set[tuple[int, int]],
    dest_set: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    step = _step_distances(grid, dest_set)
    expo = _exposure_distances(grid, dest_set)
    reach = [cell for cell in floor_cells if int(step[cell]) >= 0]
    step_order = sorted(reach, key=lambda rc: (int(step[rc]), int(rc[0]), int(rc[1])))
    expo_order = sorted(reach, key=lambda rc: (int(expo[rc]), int(rc[0]), int(rc[1])))
    return step_order, expo_order


def _has_strict_selected_reversal(
    step: np.ndarray,
    expo: np.ndarray,
    step_order: list[tuple[int, int]],
    expo_order: list[tuple[int, int]],
    k: int,
) -> bool:
    step_top = set(step_order[:k])
    expo_top = set(expo_order[:k])
    if step_top == expo_top:
        return False
    chosen_farther = list(expo_top - step_top)
    rejected_nearer = list(step_top - expo_top)
    for ar, ac in chosen_farther:
        for br, bc in rejected_nearer:
            if int(step[ar, ac]) > int(step[br, bc]) and int(expo[ar, ac]) < int(expo[br, bc]):
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the reachable maroon pixels with the smallest wall-contact path cost into the centered 3x3 target block.

    Input structure:
    - A fixed green 3x3 destination surrounds the unique red center.
    - Orange cells are walkable floor, magenta cells are maze walls, and maroon cells sit on floor cells inside the maze.

    Rule:
    - Start from every floor/maroon cell directly adjacent to the destination block and traverse only floor or maroon cells.
    - For each route, count distance as the sum over the path of how many magenta wall neighbors each visited path cell has.
    - Keep only reachable maroon pixels, rank them by this accumulated wall-neighbor cost, and place the closest ones into the destination in row-major order.
    - Any maroon that moves is erased from its source cell and replaced by orange floor.

    Key insight: In the maze, the winning maroons are not always the fewest steps away; wider passages can beat shorter but tighter dead ends because distance is driven by repeated contact with nearby walls.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    assert (h, w) == (GRID_H, GRID_W), "grid must be 11x11"

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center"
    cr, cc = map(int, centers[0])

    dest_cells = _dest_cells(cr, cc)
    dest_cells.sort()
    dest_set = set(dest_cells)
    starts = _starts_from_dest(grid, dest_set)
    assert starts, "destination block must have at least one passable maze entry"

    expo = _exposure_distances(grid, dest_set)
    inf = 10**9
    maroons = [(int(expo[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    maroons = [item for item in maroons if item[0] < inf]
    maroons.sort()

    out = grid.copy()
    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}
    for r, c in chosen:
        out[r, c] = FLOOR_COLOR
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON
    return out


def _gen_attempt() -> np.ndarray | None:
    built = _build_base_grid()
    if built is None:
        return None
    grid, _main_path, floor_cells, pockets = built
    dest_set = set(_dest_cells())

    step = _step_distances(grid, dest_set)
    expo = _exposure_distances(grid, dest_set)
    reach = [cell for cell in floor_cells if int(step[cell]) >= 0]
    step_order = sorted(reach, key=lambda rc: (int(step[rc]), int(rc[0]), int(rc[1])))
    expo_order = sorted(reach, key=lambda rc: (int(expo[rc]), int(rc[0]), int(rc[1])))
    if len(expo_order) < K_RANGE[0]:
        return None

    max_k = min(K_RANGE[1], len(expo_order))
    diff_ks = [
        k
        for k in range(K_RANGE[0], max_k + 1)
        if _has_strict_selected_reversal(step, expo, step_order, expo_order, k)
    ]
    if not diff_ks:
        return None

    k = random.choice(diff_ks)
    source_cells = list(expo_order[:k])
    unreachable_floor = list(pockets)
    min_total = 4
    min_distractors = max(0, min_total - k)
    if min_distractors > len(unreachable_floor):
        return None
    max_distractors = min(4, len(unreachable_floor))
    if max_distractors < min_distractors:
        return None
    num_distractors = random.randint(min_distractors, max_distractors)
    distractor_cells = random.sample(unreachable_floor, num_distractors)

    inp = grid.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    out = grid.copy()
    for r, c in _dest_cells()[:k]:
        out[r, c] = MAROON
    for r, c in distractor_cells:
        out[r, c] = MAROON

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
