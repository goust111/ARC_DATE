# Combo variant of dd6b8c4b (index 23)
# combo_spec: D3/0+D8/3
# Name: 反向距离排序+棋盘格交替填充
#
# Design analysis
# a) 具体化与行为追踪:
#    - 画面骨架不变: 固定 11x11，中心 (5,5) 仍是 3x3 绿色目标块的红色锚点，外面是橙色地板、可选洋红墙和分散栗色 token。
#    - 原题流程是: 从目标块外侧可通行单元做 flood-fill，只保留可达栗色 token；按 (距离升序, 行, 列) 取前 k 个；再把它们按 row-major 连续填进 3x3 目标块前 k 格。
#    - 本融合版把两个环节同时改掉:
#      1. 候选 token 改为按 (距离降序, 行, 列) 排序，最远的先被搬走。
#      2. 目标块不再把前 k 个 row-major 位置全染成栗色，而只使用 row-major 奇数位
#         (1,3,5,7,9)，也就是从左上角开始的 3x3 棋盘格 5 个槽位；偶数位保持原色。
#    - 小例子: 若可达栗色点距离是 [1,2,4,5,6,7]，本变体容量只有 5 个棋盘格槽位。
#      原题会搬走距离 1,2,4,5,6 并把目标块左上前 5 格连续染栗色；
#      本题会搬走距离 7,6,5,4,2，并把目标块的奇数位染栗色，偶数位保留绿色/红色。
#      差异同时体现在“哪几个源点消失”和“目标块内部呈棋盘格而非实心前缀”。
# b) 复杂度校准:
#    - 若可达栗色数 <= 5，则反向排序与正向排序都可能只是在“全搬”，D3/0 退化不明显。
#      因此生成器强制总可达 source 数严格大于棋盘格容量，且 source 跨越至少两个 BFS 距离层。
#    - 若只做反向排序而不约束目标块，视觉差异主要出现在源点是否被清空；若只做棋盘格填充，
#      目标块会变花但“远近取谁”不明显。这里通过“远层恰好提供 5 个会被搬走的 token，近层额外保留 1..4 个 token”
#      来同时暴露两条规律。
#    - 棋盘格槽位固定为 5 个，不引入更多相位/旋转自由度，保证 2-4 个样例就能稳定看出规律。
# c) 隐式假设审查:
#    - 目标块定位、floor 推断、墙阻断下的 flood-fill 可达性与原题一致，原先的 BFS 语义仍成立。
#    - 原题隐含“目标块容量=9，且 row-major 前缀就是填充顺序”；融合后这条前提失效，
#      因此 solver 必须显式提取棋盘格槽位列表，而不能继续直接使用 `dest_cells[:k]`。
#    - 原题的 `maroons.sort()` 默认是最近优先；这里必须改成距离降序，否则 D3/0 不生效。
#    - `k` 不再由 `len(dest_cells)` 限制，而由 `len(fill_slots)` 限制；生成器必须同步保证样本设计围绕 5 槽位容量。
#    - 0-based row-major 偶数下标正好对应 1-based 奇数位，也是 3x3 的标准棋盘格相位。
#      solver 因此隐含假设目标块确实是完整 3x3；原任务 spec 固定中心在 (5,5)，生成器会保证这一点。
# d) 放置合法性:
#    - 所有 source 和 distractor 都只放在 floor 上，从不覆盖墙或目标块，也不会越界。
#    - 墙结构仍沿用原题的“目标块外环 + 外扩支臂/框架”，reachable 与 unreachable 由几何阻隔自然形成。
#    - 生成器先在仅 floor 的距离图上选近层/远层候选，再把这些 floor 位置改成 maroon，
#      因而 solver 的 `passable={floor, 9}` 与生成时的可达性是一致的。
# e) 视觉可观察性:
#    - 人类解题者能直接看到目标块出现固定相位的栗-绿-栗 / 绿-栗-绿 / 栗-绿-栗 棋盘格前缀，
#      这使 D8/3 十分显眼。
#    - 同时，输入里会保留若干“更近”的栗色点，而更远的一组被清空，解题者可从 input/output 对照中看出
#      不是“近的先搬”，而是“远的先搬”。
#    - 生成器拒绝所有 `nearest_checker_output == farthest_checker_output` 的候选，避免 D3/0 在样本中失活。
# f) 可视化整洁度:
#    - 只有一个目标块、一个墙色、一个 floor 色、一个 token 色，视觉语义延续原题，不引入额外装饰。
#    - 棋盘格变化只发生在固定 3x3 目标块内部，远近差异只体现在少量 source 是否保留，画面不会杂乱。
# g) 变化的可推断性:
#    - 新规则不引入新的颜色种类，只重排已有 maroon / green / red 的出现位置。
#    - 棋盘格相位固定、距离规则单调反转，都是少量样例可归纳的结构线索，不依赖颜色数值大小等不可视因素。
# 9) main() 的新隐式假设:
#    - main() 假设目标容器仍是完整 3x3，因此 row-major 奇数位才等价于棋盘格槽位；generate_input() 固定中心为 (5,5)。
#    - main() 假设样本若想体现 D3/0，至少要有 1 个近层幸存 source；generate_input() 用分层采样和显式退化检查保证这一点。
#    - main() 把“保留原绿色”实现为“非棋盘格槽位保留原目标块内容”；在生成样本里棋盘格 5 槽位会被全部填满，
#      所以红色中心总会被覆盖，这也让棋盘格相位更稳定、更容易被观察到。

import random

import numpy as np

from arc_utils import neighbors4


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9

WALL_PROB = 0.75
RING_GAP_RANGE = (1, 4)
WALL_STYLES = ["arms", "frame", "mixed"]

ARM_EXTEND_PROB = 0.5
ARM_LENGTH_RANGE = (1, 5)
FRAME_DIST_RANGE = (3, 4)
FRAME_CONNECT_PROB = 0.4
FRAME_GAP_REMOVE_PROB = 0.3
MIXED_EXTEND_PROB = 0.6

CHECKER_CAPACITY = 5
MOVED_COUNT = CHECKER_CAPACITY
DISTRACTOR_MAX = 10
TOTAL_MAROON_RANGE = (6, 16)
MIN_REACHABLE = 10
GEN_RETRIES = 500
INF = 10**9


def _destination_cells(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _checker_fill_slots(dest_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [cell for idx, cell in enumerate(dest_cells) if idx % 2 == 0]


def _infer_floor(grid: np.ndarray, dest_cells: list[tuple[int, int]], dest_set: set[tuple[int, int]]) -> int:
    h, w = grid.shape
    ring_colors: list[int] = []
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set:
                ring_colors.append(int(grid[rr, cc2]))

    candidates = [x for x in ring_colors if x not in (RED_CENTER, DEST_COLOR, MAROON)]
    assert candidates, "ring around destination must contain non-destination colors to infer the floor"
    vals, cnts = np.unique(candidates, return_counts=True)
    dominant_non_dest = int(vals[cnts.argmax()])
    other = [int(x) for x in np.unique(grid) if int(x) not in (RED_CENTER, DEST_COLOR, MAROON, dominant_non_dest)]
    floor = other[0] if len(other) == 1 else (max(other, key=lambda c: int((grid == c).sum())) if other else None)
    return dominant_non_dest if floor is None else floor


def _distance_map(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
    dest_set: set[tuple[int, int]],
    passable: set[int],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
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
    queue = list(starts)
    for r, c in starts:
        dist[r, c] = 0

    qi = 0
    while qi < len(queue):
        r, c = queue[qi]
        qi += 1
        d = int(dist[r, c])
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set:
                if int(grid[rr, cc2]) in passable and dist[rr, cc2] == INF:
                    dist[rr, cc2] = d + 1
                    queue.append((rr, cc2))

    return dist, starts


def _analyze_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int, list[tuple[int, int, int]]]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must contain exactly one red center"
    cr, cc = map(int, centers[0])

    dest_cells = _destination_cells(cr, cc, h, w)
    dest_set = set(dest_cells)
    fill_slots = _checker_fill_slots(dest_cells)
    floor = _infer_floor(grid, dest_cells, dest_set)
    dist, starts = _distance_map(grid, dest_cells, dest_set, {floor, MAROON})
    assert starts, "there must be a passable cell adjacent to the destination block"

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    maroons = [t for t in maroons if t[0] < INF]
    return dest_cells, fill_slots, floor, maroons


def _solve_with_modes(grid: np.ndarray, farthest_first: bool, checker_fill: bool) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    out = grid.copy()
    dest_cells, fill_slots, floor, maroons = _analyze_grid(grid)

    if farthest_first:
        maroons.sort(key=lambda t: (-t[0], t[1], t[2]))
    else:
        maroons.sort()

    target_slots = fill_slots if checker_fill else dest_cells
    k = min(len(target_slots), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}

    for r, c in chosen:
        out[r, c] = floor

    for r, c in target_slots[:k]:
        out[r, c] = MAROON

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the farthest reachable maroon pixels into a checkerboard subset of the 3x3 block around the lone red center.

    Input structure:
    - The board is an orange floor with optional magenta walls and scattered maroon source pixels.
    - A single red cell anchors a fixed green 3x3 destination block in the middle of the grid.

    Rule:
    - Flood-fill from passable cells touching the destination, walking only through floor or maroon cells, and keep only reachable maroon pixels.
    - Rank those candidates from farthest to nearest by BFS distance, breaking ties by row then column.
    - Move the selected maroons into the destination's row-major odd positions only, which forms a checkerboard pattern; the even positions keep their original green/red contents.

    Key insight:
    - The walls still determine reachability, but the visible output now combines two independent cues: source removal prefers the outermost reachable maroons, and the target block only accepts them on alternating checkerboard cells.
    """

    return _solve_with_modes(grid, farthest_first=True, checker_fill=True)


def _reachable_floor_and_dist(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], np.ndarray]:
    dest_cells = sorted(dest_set)
    dist, starts = _distance_map(grid, dest_cells, dest_set, {FLOOR_COLOR})
    if not starts:
        return set(), dist
    reachable = {(r, c) for r in range(GRID_H) for c in range(GRID_W) if dist[r, c] < INF}
    return reachable, dist


def _outward_dirs(r: int, c: int) -> list[tuple[int, int]]:
    dirs: list[tuple[int, int]] = []
    if r < CENTER_R:
        dirs.append((-1, 0))
    if r > CENTER_R:
        dirs.append((1, 0))
    if c < CENTER_C:
        dirs.append((0, -1))
    if c > CENTER_C:
        dirs.append((0, 1))
    return dirs


def _extend_arm(wall_cells: set[tuple[int, int]], dest_set: set[tuple[int, int]], r: int, c: int, max_len: int) -> None:
    dirs = _outward_dirs(r, c)
    if not dirs:
        return
    dr, dc = random.choice(dirs)
    length = random.randint(1, max_len)
    nr, nc = r, c
    for _ in range(length):
        nr, nc = nr + dr, nc + dc
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
            wall_cells.add((nr, nc))
        else:
            break


def _build_walls(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> None:
    ring = set()
    for dr, dc in dest_set:
        for nr, nc in neighbors4(dr, dc):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                ring.add((nr, nc))
    ring = sorted(ring)

    num_gaps = random.randint(*RING_GAP_RANGE)
    gap_cells = set(random.sample(ring, num_gaps))
    wall_ring = [cell for cell in ring if cell not in gap_cells]
    wall_cells = set(wall_ring)

    style = random.choice(WALL_STYLES)
    if style == "arms":
        for r, c in wall_ring:
            if random.random() < ARM_EXTEND_PROB:
                _extend_arm(wall_cells, dest_set, r, c, ARM_LENGTH_RANGE[1])
    elif style == "frame":
        dist2 = random.randint(*FRAME_DIST_RANGE)
        for r in range(CENTER_R - dist2, CENTER_R + dist2 + 1):
            for c in range(CENTER_C - dist2, CENTER_C + dist2 + 1):
                if 0 <= r < GRID_H and 0 <= c < GRID_W and (r, c) not in dest_set:
                    if r in (CENTER_R - dist2, CENTER_R + dist2) or c in (CENTER_C - dist2, CENTER_C + dist2):
                        wall_cells.add((r, c))
        for r, c in wall_ring:
            dirs = _outward_dirs(r, c)
            if dirs and random.random() < FRAME_CONNECT_PROB:
                dr, dc = random.choice(dirs)
                nr, nc = r + dr, c + dc
                while 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                    wall_cells.add((nr, nc))
                    nr, nc = nr + dr, nc + dc
        outer_frame = [cell for cell in wall_cells if cell not in set(wall_ring)]
        if outer_frame:
            n_remove = random.randint(1, max(1, len(outer_frame) // 4))
            for rc in random.sample(outer_frame, min(n_remove, len(outer_frame))):
                if random.random() < FRAME_GAP_REMOVE_PROB:
                    wall_cells.discard(rc)
    else:
        for r, c in wall_ring:
            if random.random() < MIXED_EXTEND_PROB:
                max_len = max(1, GRID_H - abs(r - CENTER_R) - 2)
                _extend_arm(wall_cells, dest_set, r, c, max_len)

    for r, c in wall_cells:
        grid[r, c] = WALL_COLOR


def _choose_reachable_sources(reachable_floor: set[tuple[int, int]], dist: np.ndarray):
    by_distance: dict[int, list[tuple[int, int]]] = {}
    for r, c in reachable_floor:
        d = int(dist[r, c])
        by_distance.setdefault(d, []).append((r, c))

    layers = sorted(by_distance)
    if len(layers) < 2:
        return None

    splits = []
    for idx in range(len(layers) - 1):
        near_pool = [cell for d in layers[: idx + 1] for cell in by_distance[d]]
        far_pool = [cell for d in layers[idx + 1 :] for cell in by_distance[d]]
        if len(near_pool) >= 1 and len(far_pool) >= MOVED_COUNT:
            splits.append((near_pool, far_pool))
    if not splits:
        return None

    near_pool, far_pool = random.choice(splits)
    moved_count = MOVED_COUNT
    max_leftover = min(len(near_pool), 4, TOTAL_MAROON_RANGE[1] - moved_count)
    if max_leftover < 1:
        return None
    leftover = random.randint(1, max_leftover)

    near_cells = random.sample(near_pool, leftover)
    far_cells = random.sample(far_pool, moved_count)
    source_cells = near_cells + far_cells

    ordered = sorted((int(dist[r, c]), r, c) for r, c in source_cells)
    nearest = {(r, c) for _, r, c in ordered[:moved_count]}
    farthest = {(r, c) for _, r, c in sorted(ordered, key=lambda t: (-t[0], t[1], t[2]))[:moved_count]}
    if nearest == farthest:
        return None

    return {
        "source_cells": source_cells,
        "chosen_far": farthest,
        "moved_count": moved_count,
    }


def _expected_output(
    base_grid: np.ndarray,
    fill_slots: list[tuple[int, int]],
    source_cells: list[tuple[int, int]],
    chosen_far: set[tuple[int, int]],
    distractor_cells: list[tuple[int, int]],
) -> np.ndarray:
    out = base_grid.copy()
    survivors = set(source_cells) - set(chosen_far)
    for r, c in survivors:
        out[r, c] = MAROON
    for r, c in distractor_cells:
        out[r, c] = MAROON
    for r, c in fill_slots[: len(chosen_far)]:
        out[r, c] = MAROON
    return out


def _variant_metrics(inp: np.ndarray) -> tuple[int, int]:
    desc_checker = _solve_with_modes(inp, farthest_first=True, checker_fill=True)
    asc_checker = _solve_with_modes(inp, farthest_first=False, checker_fill=True)
    desc_contiguous = _solve_with_modes(inp, farthest_first=True, checker_fill=False)
    sorting_gap = int(np.count_nonzero(desc_checker != asc_checker))
    checker_gap = int(np.count_nonzero(desc_checker != desc_contiguous))
    return sorting_gap, checker_gap


def _gen_attempt():
    dest_cells = [(r, c) for r in range(CENTER_R - 1, CENTER_R + 2) for c in range(CENTER_C - 1, CENTER_C + 2)]
    dest_set = set(dest_cells)
    fill_slots = _checker_fill_slots(sorted(dest_cells))

    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    if random.random() < WALL_PROB:
        _build_walls(grid, dest_set)

    reachable_floor, dist = _reachable_floor_and_dist(grid, dest_set)
    if len(reachable_floor) < MIN_REACHABLE:
        return None

    pick = _choose_reachable_sources(reachable_floor, dist)
    if pick is None:
        return None

    source_cells = pick["source_cells"]
    chosen_far = pick["chosen_far"]
    moved_count = int(pick["moved_count"])

    all_floor = {(r, c) for r in range(GRID_H) for c in range(GRID_W) if grid[r, c] == FLOOR_COLOR and (r, c) not in dest_set}
    unreachable_floor = list(all_floor - reachable_floor)

    base_total = len(source_cells)
    max_distractors = min(DISTRACTOR_MAX, len(unreachable_floor), TOTAL_MAROON_RANGE[1] - base_total)
    min_distractors = max(0, TOTAL_MAROON_RANGE[0] - base_total)
    if min_distractors > max_distractors:
        return None
    num_distractors = random.randint(min_distractors, max_distractors)
    distractor_cells = random.sample(unreachable_floor, num_distractors)

    inp = grid.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    expected = _expected_output(grid, fill_slots, source_cells, chosen_far, distractor_cells)

    try:
        result = main(inp)
        if not np.array_equal(result, expected):
            return None
        if moved_count > len(fill_slots):
            return None

        sorting_gap, checker_gap = _variant_metrics(inp)
        if sorting_gap <= 0 or checker_gap <= 0:
            return None
        return inp
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
