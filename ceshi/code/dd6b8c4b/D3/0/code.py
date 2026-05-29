# Variant of dd6b8c4b
# Dimension: D3 — 收集对象的排序规则 (path)
# Change: sorting_by_distance——收集到的栗色像素按照到绿色方块的距离（flood-fill步数）从近到远排序... → 按距离从远到近排序
#
# Design analysis
# a) 具体化与行为追踪:
#    - 画面仍然是固定 11x11。中心 (5,5) 的红点和其周围 3x3 绿色目标块不变，橙色地板、洋红墙、栗色 token 的语义不变。
#    - solver 仍然先从目标块外侧的可通行单元做 flood-fill，只把能从目标块走到的栗色 token 视为候选。
#    - 原题在候选上按 (距离升序, 行, 列) 选前 k 个；本变体只翻转距离方向，改成按 (-距离, 行, 列) 选前 k 个。tie-break 保持行优先，避免引入额外维度变化。
#    - 小例子: 若可达栗色点距离分别是 [1, 2, 4, 5]，且 k=2，原题搬走距离 1 和 2；本题搬走距离 5 和 4。目标块里最终都只会出现 2 个栗色格，所以真正可见的差异是: 原题会清空近处 token，本题会清空远处 token，近处 token 留在原地。
# b) 复杂度校准:
#    - 仅反转排序方向太容易退化，因为如果可达栗色点总数正好等于 k，或它们都在同一距离层，升序和降序会给出同一组被搬走的 token。
#    - 由于 solver 本身会搬走 `min(9, 可达栗色点数)` 个 token，若想让远优先与近优先产生不同输出，就必须让可达栗色点数严格大于 9。
#    - 因此生成器显式构造 “近层保留 + 远层搬运” 的样本: 远层至少 9 个、近层额外保留 1-4 个，且源点横跨至少两个不同 BFS 距离层。
# c) 隐式假设审查:
#    - 目标块定位、地板推断、可通行颜色、BFS 可达性，这些都不依赖排序方向，原写法仍成立。
#    - `maroons.sort()` 依赖的是“最近优先”；这是唯一必须替换的 solver 关键行，改为按距离降序排序。
#    - `chosen = maroons[:k]` 在新排序下仍合法，因为 k 的意义不变，只是候选序列顺序改变。
#    - 输出阶段仍然是“清空被选中的源点 + 按 row-major 填充目标块前 k 格”，无需改动。
# d) 放置合法性:
#    - 生成器只在 floor 上放置栗色点，不覆盖墙和目标块，不越界。
#    - 墙结构仍沿用原题的环形/外扩策略；reachable 与 unreachable 由墙阻隔自然形成。
#    - 生成器在放置前先计算 floor-only 距离图，再从严格更近/更远的两组 floor 中抽样，保证 solver 的 passable={floor,9} 与生成时的可达性一致。
# e) 视觉可观察性:
#    - ARC 解题者能从样例中看到: 目标块填满同样数量的栗色格，但被清空的是“远处那批”而不是“近处那批”。
#    - 为了让这种差异稳定出现，生成器拒绝所有“升序/降序输出完全一致”的候选布局。
# f) 可视化整洁度:
#    - 仍然保持单个 3x3 目标块、单色墙、单色 token，没有叠加标记。
#    - 额外约束只影响哪些 floor 位置放 token，不增加新的视觉元素。
# g) 颜色变化可推断性:
#    - 本变体不引入新的颜色映射，颜色语义完全沿用原题，所以不存在额外的颜色归纳负担。
# 9) main() 的新隐式假设:
#    - main() 现在隐含假设“若希望观察到变体行为，候选栗色点最好跨多个距离层”；这不是 solver 正确性的必要条件，但确实是样例可辨识性的必要条件。
#    - generate_input() 通过显式检查 `nearest_k != farthest_k` 来保证这一点。

import numpy as np

from arc_utils import neighbors4
import random
from collections import deque


def _destination_cells(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


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
            if 0 <= rr < h and 0 <= cc2 < w and int(grid[rr, cc2]) in passable:
                if (rr, cc2) not in dest_set and (rr, cc2) not in seen:
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


def _reachable_maroons(grid: np.ndarray) -> tuple[list[tuple[int, int]], int, list[tuple[int, int, int]]]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape

    centers = np.argwhere(grid == 2)
    assert len(centers) == 1, "grid must have exactly one red (2) center pixel"
    cr, cc = map(int, centers[0])

    dest_cells = _destination_cells(cr, cc, h, w)
    dest_set = set(dest_cells)
    floor = _infer_floor(grid, dest_cells, dest_set)
    passable = {floor, 9}
    dist, starts = _distance_map(grid, dest_cells, dest_set, passable)
    assert starts, "there must be at least one passable cell adjacent to the destination block to begin BFS"

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == 9)]
    inf = 10**9
    maroons = [t for t in maroons if t[0] < inf]
    return dest_cells, floor, maroons


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the farthest reachable maroon pixels into the 3x3 block centered on the lone red cell.

    Input structure:
    - The board is an orange floor with optional magenta walls and scattered maroon source pixels.
    - A single red cell sits at the middle of a green 3x3 destination block.

    Rule:
    - Flood-fill outward from cells touching the destination, walking only through floor or maroon cells.
    - Only reachable maroon pixels are eligible, and they are ranked by distance from far to near before filling destination cells in row-major order.
    - Every chosen source pixel is erased back to the floor color after it is moved.

    Key insight:
    - The walls still decide which maroon pixels belong to the same reachable region, but unlike the original task the destination prefers the outermost reachable tokens first.
    """

    grid = np.asarray(grid, dtype=int)
    out = grid.copy()
    dest_cells, floor, maroons = _reachable_maroons(grid)

    maroons.sort(key=lambda t: (-t[0], t[1], t[2]))  # farthest first; ties stay row-major

    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}

    for r, c in chosen:
        out[r, c] = floor

    for r, c in dest_cells[:k]:
        out[r, c] = 9

    return out


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

FILL_K_RANGE = (2, 9)
DISTRACTOR_MAX = 10
TOTAL_MAROON_RANGE = (4, 16)
MIN_REACHABLE = 10
GEN_RETRIES = 400
VISIBLE_FILL_COUNT = FILL_K_RANGE[1]


def _bfs_reachable(grid, dest_set):
    """BFS from floor cells adjacent to dest through floor cells. Returns set of reachable (r,c)."""
    dest_cells = sorted(dest_set)
    dist, starts = _distance_map(grid, dest_cells, dest_set, {FLOOR_COLOR})
    if not starts:
        return set(), dist
    reachable = {(r, c) for r in range(GRID_H) for c in range(GRID_W) if dist[r, c] < 10**9}
    return reachable, dist


def _outward_dirs(r, c):
    """Return directions pointing away from center for a ring cell."""
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


def _extend_arm(wall_cells, dest_set, r, c, max_len):
    """Extend a wall arm outward from (r,c) in a random outward direction, up to max_len cells."""
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


def _build_walls(grid, dest_set):
    """Build wall barrier around dest with corridor openings, extending outward."""
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


def _choose_reachable_sources(reachable_floor, dist):
    by_distance: dict[int, list[tuple[int, int]]] = {}
    for r, c in reachable_floor:
        d = int(dist[r, c])
        by_distance.setdefault(d, []).append((r, c))

    layers = sorted(by_distance)
    if len(layers) < 2:
        return None

    splits = []
    for idx in range(len(layers) - 1):
        cutoff = layers[idx]
        near_pool = [cell for d in layers[: idx + 1] for cell in by_distance[d]]
        far_pool = [cell for d in layers[idx + 1 :] for cell in by_distance[d]]
        if len(far_pool) >= VISIBLE_FILL_COUNT:
            splits.append((cutoff, near_pool, far_pool))
    if not splits:
        return None

    cutoff, near_pool, far_pool = random.choice(splits)
    moved_count = VISIBLE_FILL_COUNT
    if len(far_pool) < moved_count:
        return None

    min_leftover = 1
    max_leftover = min(len(near_pool), TOTAL_MAROON_RANGE[1] - moved_count, 4)
    if min_leftover > max_leftover:
        return None

    leftover = random.randint(min_leftover, max_leftover)

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
        "cutoff": cutoff,
        "leftover": leftover,
    }


def _expected_output(base_grid, dest_cells, source_cells, chosen_far, distractor_cells):
    out = base_grid.copy()
    survivors = set(source_cells) - set(chosen_far)
    for r, c in survivors:
        out[r, c] = MAROON
    for r, c in distractor_cells:
        out[r, c] = MAROON
    fill_count = len(chosen_far)
    for r, c in dest_cells[:fill_count]:
        out[r, c] = MAROON
    return out


def _gen_attempt():
    """Build a candidate input grid; returns the grid or None on failure."""
    dest_cells = [(r, c) for r in range(CENTER_R - 1, CENTER_R + 2) for c in range(CENTER_C - 1, CENTER_C + 2)]
    dest_set = set(dest_cells)

    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    if random.random() < WALL_PROB:
        _build_walls(grid, dest_set)

    reachable_floor, dist = _bfs_reachable(grid, dest_set)
    if len(reachable_floor) < MIN_REACHABLE:
        return None

    pick = _choose_reachable_sources(reachable_floor, dist)
    if pick is None:
        return None

    source_cells = pick["source_cells"]
    chosen_far = pick["chosen_far"]

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

    out = _expected_output(grid, dest_cells, source_cells, chosen_far, distractor_cells)

    try:
        result = main(inp)
        if not np.array_equal(result, out):
            return None

        _, _, maroons = _reachable_maroons(inp)
        ordered_asc = sorted(maroons)
        ordered_desc = sorted(maroons, key=lambda t: (-t[0], t[1], t[2]))
        nearest = {(r, c) for _, r, c in ordered_asc[:VISIBLE_FILL_COUNT]}
        farthest = {(r, c) for _, r, c in ordered_desc[:VISIBLE_FILL_COUNT]}
        if nearest == farthest:
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
