# Combo variant of dd6b8c4b (index 4)
# combo_spec: D3/0+D4/0
# Name: 反向距离排序+反向行优先填充
#
# Design analysis
# a) 具体化与行为追踪:
#    - 画面骨架保持原题: 固定 11x11, 中心 (5,5) 的 3x3 目标块, 红色中心(2),
#      其余目标格绿色(3), 外部是橙色地板(7), 可能有洋红墙(6), 以及栗色 token(9)。
#    - solver 仍先从目标块外侧相邻的可通行单元做 flood-fill, 只把能通过 floor/maroon
#      走到的栗色 token 当作候选。
#    - 原题是 “最近优先 + 左上到右下 row-major 填充”。
#      本组合改为 “最远优先 + 右下到左上的 reverse row-major 填充”。
#    - 小追踪:
#      若可达 token 距离依次为 [1,2,4,5], 且只需搬 2 个:
#      原题会搬走距离 1 和 2, 填到 [(4,4),(4,5)];
#      本组合会搬走距离 5 和 4, 填到 [(6,6),(6,5)]。
#      差异体现在两端: 被清空的是远端 token, 目标块增长的是右下角而非左上角。
# b) 复杂度校准:
#    - 这个组合有一个结构性张力: 想让 D3 可见, 必须让可达 token 数 > 9, 这样最近/最远
#      选中的来源集合才会不同; 但想让 D4 可见, 又必须部分填充 (k < 9), 否则 3x3 全满时
#      右下起填和左上起填在最终图像里不可区分。
#    - 由于本题 token 同色且 solver 固定 k=min(9, reachable_count), 单个样本无法同时把
#      两种差异都独立暴露出来。这不是实现错误, 而是原任务表示法的限制。
#    - 因此生成器采用两类样本族:
#      1. `full_far`: 让 reachable_count > 9, 强化 D3, 输出为满 3x3;
#      2. `partial_reverse`: 让 reachable_count = k < 9, 强化 D4, 输出为右下角部分填充。
#      批量样本中两类都会出现, 从而组合规则不会退化成单一维度。
# c) 隐式假设审查:
#    - 原 solver 里与目标块定位、地板推断、BFS 可达性有关的假设在组合后仍成立。
#    - 需要真正改动的只有两个编码点:
#      1. 候选排序从 `maroons.sort()` 改成按 `(-dist,row,col)`。
#      2. 目标写入顺序从 `dest_cells[:k]` 改成 `sorted(dest_cells, reverse=True)[:k]`。
#    - 多维叠加后没有额外冲突, 因为“选哪些来源”和“按什么目标路径写入”仍是两个串联阶段。
# d) 放置合法性:
#    - 生成器只在 floor 上放 token, 不覆盖墙和目标块, 坐标都由显式 bounds 检查保护。
#    - full_far 模式从不同距离层抽样: 9 个远层 token 会被搬走, 1-4 个近层 token 留作幸存者;
#      partial_reverse 模式只放 2..8 个可达 token, 它们都会被搬走。
#    - distractor 仍只放在 unreachable floor, 不与可达来源或墙重叠。
# e) 视觉可观察性:
#    - 对人类而言, partial_reverse 样本清楚展示“目标块从右下角开始长出来”。
#    - full_far 样本清楚展示“被搬走的是远处 token, 近处 token 留在原地”。
#    - 两类样本都只依赖空间位置与连通性, 不依赖颜色编号大小等不可视因素。
# f) 可视化整洁度:
#    - 仍然只有一个中心目标块、一套墙和若干散点 token, 不引入额外标记层。
#    - 通过限制 total maroon 数量与墙样式, 保持版面稀疏, 避免目标块附近过度拥挤。
# g) 变化的可推断性:
#    - 没有新增颜色语义。所有变化仍是 maroon 在 floor 与目标块之间移动。
#    - 规则差异来自“哪些 source 被搬走”与“目标块从哪一角开始填”这两种结构线索。
# h) 融合行为量化指标:
#    - 生成器内部把 accepted 样本分成两类:
#      `full_far` 要求 combo 输出不同于 original / D4-only;
#      `partial_reverse` 要求 combo 输出不同于 original / D3-only。
#    - 这样可统计一个批次里两类样本各占多少, 用来验证组合没有退化成单维版本。
# i) main() 的新隐式假设:
#    - solver 本身没有新增严格输入前提; 即便某样本只能显露一个维度, 组合规则仍是合法定义。
#    - generator 额外保证:
#      1. 至少存在一条从目标块外侧进入的可通行路径;
#      2. full_far 模式下可达 token 跨至少两个 BFS 距离层且总数 > 9;
#      3. partial_reverse 模式下可达 token 数在 2..8, 使反向填充可见。

from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import neighbors4


def _destination_cells(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _reverse_row_major_fill_order(dest_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(dest_cells, reverse=True)


def _infer_floor(grid: np.ndarray, dest_cells: list[tuple[int, int]], dest_set: set[tuple[int, int]]) -> int:
    ring_colors: list[int] = []
    h, w = grid.shape
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set:
                ring_colors.append(int(grid[rr, cc2]))

    cand_ring = [x for x in ring_colors if x not in (2, 3, 9)]
    assert cand_ring, "ring around destination must contain passable cells to infer floor color"
    vals, cnts = np.unique(cand_ring, return_counts=True)
    ring_blocker = int(vals[cnts.argmax()])
    other = [int(x) for x in np.unique(grid) if int(x) not in (2, 3, 9, ring_blocker)]
    floor = other[0] if len(other) == 1 else (max(other, key=lambda c: int((grid == c).sum())) if other else None)
    return ring_blocker if floor is None else floor


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
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set:
                if int(grid[rr, cc2]) in passable and (rr, cc2) not in seen:
                    seen.add((rr, cc2))
                    starts.append((rr, cc2))

    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q = list(starts)
    for r, c in starts:
        dist[r, c] = 0

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = dist[r, c]
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set:
                if int(grid[rr, cc2]) in passable and dist[rr, cc2] == inf:
                    dist[rr, cc2] = d + 1
                    q.append((rr, cc2))
    return dist, starts


def _reachable_analysis(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int, list[tuple[int, int, int]]]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape

    centers = np.argwhere(grid == 2)
    assert len(centers) == 1, "grid must have exactly one red (2) center pixel"
    cr, cc = map(int, centers[0])

    dest_cells = _destination_cells(cr, cc, h, w)
    fill_order = _reverse_row_major_fill_order(dest_cells)
    dest_set = set(dest_cells)
    floor = _infer_floor(grid, dest_cells, dest_set)
    passable = {floor, 9}
    dist, starts = _distance_map(grid, dest_cells, dest_set, passable)
    assert starts, "there must be at least one passable cell adjacent to the destination block to begin BFS"

    inf = 10**9
    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == 9)]
    maroons = [t for t in maroons if t[0] < inf]
    return dest_cells, fill_order, floor, maroons


def _solve_variant(grid: np.ndarray, *, descending_distance: bool, reverse_fill: bool) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    out = grid.copy()
    dest_cells, fill_order, floor, maroons = _reachable_analysis(grid)

    if descending_distance:
        maroons.sort(key=lambda t: (-t[0], t[1], t[2]))
    else:
        maroons.sort()

    write_order = fill_order if reverse_fill else dest_cells
    k = min(len(write_order), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}

    for r, c in chosen:
        out[r, c] = floor

    for r, c in write_order[:k]:
        out[r, c] = 9

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the outermost reachable maroon pixels into the central 3x3 block, filling it from bottom-right backward.

    Input structure:
    - The board is an 11x11 orange floor with a fixed green destination block centered on one red cell.
    - Magenta walls may carve corridors around that block, and maroon tokens are scattered on floor cells outside it.

    Rule:
    - Flood-fill from passable cells touching the destination, moving only through floor or maroon cells, to find reachable maroon candidates.
    - Order those candidates by BFS distance from far to near, keeping row/column tie-breaks within the same distance layer.
    - Remove the chosen sources and place the moved maroons into the destination in reverse row-major order, starting at the lower-right corner.

    Key insight:
    - The same wall-gated reachability test from the original task still decides which maroons are eligible, but both downstream directions are inverted: selection prefers the farthest candidates and writing prefers the bottom-right end of the block.
    """

    return _solve_variant(grid, descending_distance=True, reverse_fill=True)


# --------------- generator (c) ---------------

# ---- CFG: tunable generation parameters ----
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

PARTIAL_FILL_RANGE = (2, 8)
VISIBLE_FILL_COUNT = 9
DISTRACTOR_MAX = 10
TOTAL_MAROON_RANGE = (4, 16)
MIN_REACHABLE = 2
MIN_REACHABLE_FOR_FULL_FAR = 10
GEN_RETRIES = 500

MODE_FULL_FAR = "full_far"
MODE_PARTIAL_REVERSE = "partial_reverse"


def _dest_cells() -> list[tuple[int, int]]:
    return _destination_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)


def _bfs_reachable(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], np.ndarray]:
    dest_cells = sorted(dest_set)
    dist, starts = _distance_map(grid, dest_cells, dest_set, {FLOOR_COLOR})
    if not starts:
        return set(), dist
    reachable = {(r, c) for r in range(GRID_H) for c in range(GRID_W) if dist[r, c] < 10**9}
    return reachable, dist


def _outward_dirs(r: int, c: int) -> list[tuple[int, int]]:
    dirs = []
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


def _choose_full_far_sources(
    reachable_floor: set[tuple[int, int]],
    dist: np.ndarray,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    by_distance: dict[int, list[tuple[int, int]]] = {}
    for r, c in reachable_floor:
        d = int(dist[r, c])
        by_distance.setdefault(d, []).append((r, c))

    layers = sorted(by_distance)
    if len(layers) < 2:
        return None

    splits: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
    for idx in range(len(layers) - 1):
        near_pool = [cell for d in layers[: idx + 1] for cell in by_distance[d]]
        far_pool = [cell for d in layers[idx + 1 :] for cell in by_distance[d]]
        if len(near_pool) >= 1 and len(far_pool) >= VISIBLE_FILL_COUNT:
            splits.append((near_pool, far_pool))
    if not splits:
        return None

    near_pool, far_pool = random.choice(splits)
    max_leftover = min(4, len(near_pool), TOTAL_MAROON_RANGE[1] - VISIBLE_FILL_COUNT)
    if max_leftover < 1:
        return None

    leftover = random.randint(1, max_leftover)
    near_cells = random.sample(near_pool, leftover)
    far_cells = random.sample(far_pool, VISIBLE_FILL_COUNT)
    source_cells = near_cells + far_cells

    ordered = sorted((int(dist[r, c]), r, c) for r, c in source_cells)
    farthest = {(r, c) for _, r, c in sorted(ordered, key=lambda t: (-t[0], t[1], t[2]))[:VISIBLE_FILL_COUNT]}
    nearest = {(r, c) for _, r, c in ordered[:VISIBLE_FILL_COUNT]}
    if nearest == farthest:
        return None
    return source_cells, farthest


def _choose_partial_sources(reachable_floor: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    max_k = min(PARTIAL_FILL_RANGE[1], len(reachable_floor))
    if max_k < PARTIAL_FILL_RANGE[0]:
        return None
    k = random.randint(PARTIAL_FILL_RANGE[0], max_k)
    return random.sample(list(reachable_floor), k)


def _expected_output(
    base_grid: np.ndarray,
    fill_order: list[tuple[int, int]],
    source_cells: list[tuple[int, int]],
    chosen_sources: set[tuple[int, int]],
    distractor_cells: list[tuple[int, int]],
) -> np.ndarray:
    out = base_grid.copy()
    survivors = set(source_cells) - chosen_sources
    for r, c in survivors:
        out[r, c] = MAROON
    for r, c in distractor_cells:
        out[r, c] = MAROON
    for r, c in fill_order[: len(chosen_sources)]:
        out[r, c] = MAROON
    return out


def _sample_distractors(
    unreachable_floor: list[tuple[int, int]],
    source_count: int,
) -> list[tuple[int, int]] | None:
    max_distractors = min(DISTRACTOR_MAX, len(unreachable_floor), TOTAL_MAROON_RANGE[1] - source_count)
    min_distractors = max(0, TOTAL_MAROON_RANGE[0] - source_count)
    if min_distractors > max_distractors:
        return None
    count = random.randint(min_distractors, max_distractors)
    return random.sample(unreachable_floor, count)


def _validate_mode(inp: np.ndarray, expected: np.ndarray, mode: str) -> bool:
    combo = main(inp)
    if not np.array_equal(combo, expected):
        return False

    original = _solve_variant(inp, descending_distance=False, reverse_fill=False)
    d3_only = _solve_variant(inp, descending_distance=True, reverse_fill=False)
    d4_only = _solve_variant(inp, descending_distance=False, reverse_fill=True)

    if np.array_equal(combo, original):
        return False

    if mode == MODE_FULL_FAR:
        return np.array_equal(combo, d3_only) and not np.array_equal(combo, d4_only)
    if mode == MODE_PARTIAL_REVERSE:
        return np.array_equal(combo, d4_only) and not np.array_equal(combo, d3_only)
    return False


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _dest_cells()
    fill_order = _reverse_row_major_fill_order(dest_cells)
    dest_set = set(dest_cells)

    base_grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        base_grid[r, c] = DEST_COLOR
    base_grid[CENTER_R, CENTER_C] = RED_CENTER

    if random.random() < WALL_PROB:
        _build_walls(base_grid, dest_set)

    reachable_floor, dist = _bfs_reachable(base_grid, dest_set)
    if len(reachable_floor) < MIN_REACHABLE:
        return None

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if base_grid[r, c] == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = list(all_floor - reachable_floor)

    candidates: list[tuple[str, list[tuple[int, int]], set[tuple[int, int]]]] = []

    partial_sources = _choose_partial_sources(reachable_floor)
    if partial_sources is not None:
        candidates.append((MODE_PARTIAL_REVERSE, partial_sources, set(partial_sources)))

    if len(reachable_floor) >= MIN_REACHABLE_FOR_FULL_FAR:
        full_far = _choose_full_far_sources(reachable_floor, dist)
        if full_far is not None:
            source_cells, chosen_far = full_far
            candidates.append((MODE_FULL_FAR, source_cells, chosen_far))

    if not candidates:
        return None

    mode, source_cells, chosen_sources = random.choice(candidates)
    distractor_cells = _sample_distractors(unreachable_floor, len(source_cells))
    if distractor_cells is None:
        return None

    inp = base_grid.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    expected = _expected_output(base_grid, fill_order, source_cells, chosen_sources, distractor_cells)
    return inp if _validate_mode(inp, expected, mode) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
