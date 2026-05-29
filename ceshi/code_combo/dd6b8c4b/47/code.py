# Combo variant of dd6b8c4b (index 47)
# combo_spec: D3/2+D4/3
# Name: 距离对应位置填充+欧几里得距离排序
#
# Design analysis
# ---------------
# a) concretization_and_trace:
#    - 原题骨架保留: 11x11 画布中央固定一个 3x3 目标块, 中心红(2), 其余目标格绿(3); 背景地板橙(7),
#      墙是洋红(6), 栗色 token(9) 散落在目标块外。
#    - 求解第一步仍是“可达性过滤”: 只保留能从目标块外缘经由 4 邻接地板/栗色像素 flood-fill 到达的栗色 token。
#    - D3/2 改的是“源 token 排序”: 原题按 BFS 步数排, 现在改成按 token 到红中心的欧几里得距离排。
#      例如两个都可达的 token A=(3,5), B=(5,1), 若 A 的直线距离更小但绕墙更远, 原题会先取 B,
#      融合后会先取 A。
#    - D4/3 改的是“目标块填充位置排序”: 原题按 row-major 把前 k 个 token 填到
#      [(4,4),(4,5),(4,6),...], 现在改成按目标格到中心的距离从近到远填:
#      [(5,5),(4,5),(5,4),(5,6),(6,5),(4,4),(4,6),(6,4),(6,6)]。
#    - 小例子:
#      * 若可达 token 只有 4 个, 原题会把目标块左上 4 格染成 9; 本变体会把中心十字的 4 格染成 9。
#      * 若可达 token 有 10 个, 两边都会只搬 9 个; 此时目标块都被填满, D4/3 在该样例里不可见,
#        但 D3/2 会改变“哪 9 个源 token 被搬走”。
#
# b) complexity_calibration:
#    - 这个任务有一个硬约束: token 全同色且目标块只有 9 格。因此当可达 token 数 <= 8 时, 所有可达
#      token 都会被搬走, D3/2 不可见; 当可达 token 数 >= 9 时, 目标块会被填满, D4/3 不可见。
#    - 所以不能强行要求“每一张样例同时显性展示两项变化”, 那会退化成不可观察的实现细节。
#    - 生成器因此只保留两个可观察子模式:
#      1. target_visible: 可达 token 数为 2-8, 让 D4/3 的中心向外填充肉眼可见。
#      2. source_visible: 可达 token 数固定为 10, 让 D3/2 决定哪 9 个源 token 被搬走。
#    - 这两个模式都服从同一个 solver, 只是分别把组合中的两条控制链显性化。
#
# c) implicit_assumption_review:
#    - 原 solver 的 BFS 仍然有用, 但它只能做 reachability gate, 不能再直接当排序键。
#    - 原来的 `maroons.sort()` 默认是 `(flood_distance,row,col)`; 融合后必须改成
#      `(euclidean_distance_to_center,row,col)`。
#    - 原来的 `dest_cells[:k]` 隐含 row-major 目标顺序; 融合后必须改成“目标格到中心距离”的顺序。
#    - 原 solver 里“k = min(9, reachable_maroon_count)”仍成立; 新 generator 明确围绕这个容量上限造样例。
#    - 新 solver 额外依赖固定调色板 2/3/6/7/9; generator 显式保证这一点, 避免再做脆弱的颜色推断。
#
# d) placement_legality:
#    - 目标块始终是中心 3x3, 不越界。
#    - 墙只放在目标块外; token 只放在地板上; 不与墙或目标块重叠。
#    - source_visible 模式里 10 个可达 token 用构造法选出, 保证它们都在同一可达区域内且彼此不同。
#    - distractor token 只放在不可达地板区, 不影响 reachability 判定。
#
# e) visual_observability:
#    - D4/3 的线索完全可见: 部分填充时输出会优先占据红中心和其十字邻格, 而不是左上角开始铺。
#    - D3/2 的线索也完全可见: 满填样例里目标块都变成 9, 但被擦除的源 token 不是“沿通路最近的 9 个”,
#      而是“离红中心直线距离最近的 9 个”。
#    - 排序只使用几何距离与行列 tie-break, 不依赖颜色编号或其它人类不可见因素。
#
# f) cleanliness:
#    - 控制元素只有目标块、墙、地板和少量 token, 不叠加额外标记。
#    - target_visible 模式优先使用 2-8 个可达 token, 输出中心团块很规整; source_visible 模式固定 10 个,
#      只在源区出现“被搬走/保留”的差异, 画面仍然清晰。
#
# g) inference_of_change:
#    - 人类可从少量样例归纳出两个稳定现象: “部分填充时先填中心附近” 与 “满填时不是按通路步数挑源 token”。
#    - generator 会拒绝:
#      * target_visible 中与旧 row-major 输出完全相同的样例;
#      * source_visible 中与旧 BFS 排序选源完全相同的样例。

from __future__ import annotations

from collections import deque
import random

import numpy as np


GRID_H = 11
GRID_W = 11
CENTER_R = 5
CENTER_C = 5

FLOOR = 7
WALL = 6
DEST = 3
RED = 2
MAROON = 9

WALL_PROB = 0.8
RING_GAP_RANGE = (1, 4)
WALL_STYLES = ("arms", "frame", "mixed")
ARM_EXTEND_PROB = 0.5
ARM_LENGTH_RANGE = (1, 5)
FRAME_DIST_RANGE = (3, 4)
FRAME_CONNECT_PROB = 0.4
FRAME_GAP_REMOVE_PROB = 0.3
MIXED_EXTEND_PROB = 0.6

TARGET_VISIBLE_RANGE = (2, 8)
SOURCE_VISIBLE_COUNT = 10
DISTRACTOR_MAX = 8
TOTAL_MAROON_RANGE = (4, 16)
MIN_REACHABLE = 10
GENERATE_MAX_RETRIES = 800


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _dest_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    cr, cc = center
    return [(r, c) for r in range(cr - 1, cr + 2) for c in range(cc - 1, cc + 2)]


def _sqdist(cell: tuple[int, int], center: tuple[int, int]) -> int:
    return (int(cell[0]) - int(center[0])) ** 2 + (int(cell[1]) - int(center[1])) ** 2


def _combo_target_order(center: tuple[int, int]) -> list[tuple[int, int]]:
    return sorted(_dest_cells(center), key=lambda rc: (_sqdist(rc, center), rc[0], rc[1]))


def _row_major_target_order(center: tuple[int, int]) -> list[tuple[int, int]]:
    return sorted(_dest_cells(center))


def _bfs_distance(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> np.ndarray:
    h, w = grid.shape
    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q: deque[tuple[int, int]] = deque()
    seen: set[tuple[int, int]] = set()
    for r, c in dest_set:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set:
                if int(grid[nr, nc]) in (FLOOR, MAROON) and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    dist[nr, nc] = 0
                    q.append((nr, nc))
    assert q, "there must be a passable cell adjacent to the destination block"

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] == inf:
                if int(grid[nr, nc]) in (FLOOR, MAROON) and (nr, nc) not in dest_set:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
    return dist


def _reachable_maroons(
    grid: np.ndarray, center: tuple[int, int]
) -> tuple[list[tuple[int, int]], np.ndarray]:
    dest_set = set(_dest_cells(center))
    dist = _bfs_distance(grid, dest_set)
    inf = 10**9
    maroons = []
    for r, c in map(tuple, np.argwhere(grid == MAROON)):
        rr, cc = int(r), int(c)
        if int(dist[rr, cc]) < inf:
            maroons.append((rr, cc))
    return maroons, dist


def _sort_sources_combo(
    cells: list[tuple[int, int]], center: tuple[int, int]
) -> list[tuple[int, int]]:
    return sorted(cells, key=lambda rc: (_sqdist(rc, center), rc[0], rc[1]))


def _sort_sources_legacy(
    cells: list[tuple[int, int]], dist: np.ndarray
) -> list[tuple[int, int]]:
    return sorted(cells, key=lambda rc: (int(dist[rc[0], rc[1]]), rc[0], rc[1]))


def _solve_with_orders(
    grid: np.ndarray, *, source_mode: str, target_mode: str
) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    out = grid.copy()

    centers = np.argwhere(grid == RED)
    assert len(centers) == 1, "grid must contain exactly one red center"
    center = (int(centers[0, 0]), int(centers[0, 1]))

    maroons, dist = _reachable_maroons(grid, center)
    if source_mode == "combo":
        source_order = _sort_sources_combo(maroons, center)
    elif source_mode == "legacy":
        source_order = _sort_sources_legacy(maroons, dist)
    else:
        raise ValueError(source_mode)

    if target_mode == "combo":
        target_order = _combo_target_order(center)
    elif target_mode == "legacy":
        target_order = _row_major_target_order(center)
    else:
        raise ValueError(target_mode)

    k = min(len(target_order), len(source_order))
    chosen = source_order[:k]
    for r, c in chosen:
        out[r, c] = FLOOR
    for r, c in target_order[:k]:
        out[r, c] = MAROON
    return out


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_orders(grid, source_mode="legacy", target_mode="legacy")


def _sample_metrics(grid: np.ndarray) -> dict[str, int]:
    centers = np.argwhere(grid == RED)
    center = (int(centers[0, 0]), int(centers[0, 1]))
    maroons, dist = _reachable_maroons(grid, center)
    combo_sources = _sort_sources_combo(maroons, center)
    legacy_sources = _sort_sources_legacy(maroons, dist)
    k = min(9, len(maroons))
    combo_pick = set(combo_sources[:k])
    legacy_pick = set(legacy_sources[:k])
    combo_targets = set(_combo_target_order(center)[:k])
    legacy_targets = set(_row_major_target_order(center)[:k])
    combo_out = _solve_with_orders(grid, source_mode="combo", target_mode="combo")
    legacy_out = _legacy_main(grid)
    source_diff = len(combo_pick ^ legacy_pick)
    target_diff = len(combo_targets ^ legacy_targets)
    out_diff = int(np.sum(combo_out != legacy_out))
    return {
        "reachable": len(maroons),
        "moved": k,
        "source_symdiff": source_diff,
        "target_symdiff": target_diff,
        "output_diff_vs_legacy": out_diff,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull reachable maroon pixels into the 3x3 block around the unique red center,
    but rank both sources and target slots by distance instead of scan order.

    Input structure:
    - The board is an 11x11 orange floor with optional magenta walls and scattered maroon tokens.
    - A fixed green 3x3 target block surrounds one red center pixel; tokens never start inside that block.

    Rule:
    - First flood-fill outward from floor cells touching the target block, moving only through floor or maroon cells,
      and keep only maroon tokens that are reachable through that passable region.
    - Sort the reachable maroon tokens by Euclidean distance to the red center, nearest first
      (ties break top-to-bottom, then left-to-right).
    - Sort the 3x3 target positions by their own distance to the same center, nearest first,
      and place the nearest reachable tokens into the nearest target slots; vacated source cells become floor.

    Key insight:
    - Walls still matter, because they decide which maroon tokens are even eligible, but once eligibility is fixed,
      the move order is no longer “shortest corridor first” or “row-major fill”. The whole transformation is driven by
      two visible distance orderings measured against the same red-centered square.
    """

    return _solve_with_orders(grid, source_mode="combo", target_mode="combo")


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


def _extend_arm(
    wall_cells: set[tuple[int, int]],
    dest_set: set[tuple[int, int]],
    r: int,
    c: int,
    max_len: int,
) -> None:
    dirs = _outward_dirs(r, c)
    if not dirs:
        return
    dr, dc = random.choice(dirs)
    length = random.randint(1, max_len)
    nr, nc = r, c
    for _ in range(length):
        nr += dr
        nc += dc
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
            wall_cells.add((nr, nc))
        else:
            break


def _build_walls(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> None:
    ring = set()
    for r, c in dest_set:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                ring.add((nr, nc))
    ring_list = sorted(ring)
    gap_count = random.randint(*RING_GAP_RANGE)
    gap_cells = set(random.sample(ring_list, gap_count))
    wall_ring = [cell for cell in ring_list if cell not in gap_cells]
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
                    nr += dr
                    nc += dc
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
        grid[r, c] = WALL


def _blank_layout() -> tuple[np.ndarray, list[tuple[int, int]], set[tuple[int, int]]]:
    center = (CENTER_R, CENTER_C)
    dest_cells = _dest_cells(center)
    dest_set = set(dest_cells)
    grid = np.full((GRID_H, GRID_W), FLOOR, dtype=int)
    for r, c in dest_cells:
        grid[r, c] = DEST
    grid[CENTER_R, CENTER_C] = RED
    if random.random() < WALL_PROB:
        _build_walls(grid, dest_set)
    return grid, dest_cells, dest_set


def _reachable_floor(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], np.ndarray]:
    dist = _bfs_distance(grid, dest_set)
    inf = 10**9
    cells = []
    for r in range(GRID_H):
        for c in range(GRID_W):
            if (r, c) not in dest_set and int(grid[r, c]) == FLOOR and int(dist[r, c]) < inf:
                cells.append((r, c))
    return cells, dist


def _find_source_visible_pack(
    reachable_floor: list[tuple[int, int]],
    center: tuple[int, int],
    dist: np.ndarray,
) -> list[tuple[int, int]] | None:
    legacy_order = _sort_sources_legacy(reachable_floor, dist)
    combo_order = _sort_sources_combo(reachable_floor, center)
    legacy_rank = {cell: i for i, cell in enumerate(legacy_order)}
    combo_rank = {cell: i for i, cell in enumerate(combo_order)}

    candidates: list[list[tuple[int, int]]] = []
    for a in reachable_floor:
        for b in reachable_floor:
            if a == b:
                continue
            if not (combo_rank[a] < combo_rank[b] and legacy_rank[b] < legacy_rank[a]):
                continue
            core = [
                cell
                for cell in reachable_floor
                if combo_rank[cell] < combo_rank[a] and legacy_rank[cell] < legacy_rank[b]
            ]
            if len(core) < 8:
                continue
            core.sort(key=lambda cell: (max(combo_rank[cell], legacy_rank[cell]), combo_rank[cell], legacy_rank[cell], cell[0], cell[1]))
            pack = core[:8] + [a, b]
            if len(set(pack)) == SOURCE_VISIBLE_COUNT:
                candidates.append(pack)
    if not candidates:
        return None
    return random.choice(candidates)


def _choose_target_visible_sources(
    reachable_floor: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    n = random.randint(TARGET_VISIBLE_RANGE[0], min(TARGET_VISIBLE_RANGE[1], len(reachable_floor)))
    return random.sample(reachable_floor, n)


def _choose_distractors(
    unreachable_floor: list[tuple[int, int]],
    n_sources: int,
) -> list[tuple[int, int]]:
    min_total, max_total = TOTAL_MAROON_RANGE
    min_needed = max(0, min_total - n_sources)
    max_allowed = min(DISTRACTOR_MAX, len(unreachable_floor), max_total - n_sources)
    if max_allowed < min_needed:
        return []
    n = random.randint(min_needed, max_allowed)
    return random.sample(unreachable_floor, n) if n else []


def _mode_ok(inp: np.ndarray, mode: str) -> bool:
    metrics = _sample_metrics(inp)
    if mode == "target_visible":
        return (
            TARGET_VISIBLE_RANGE[0] <= metrics["reachable"] <= TARGET_VISIBLE_RANGE[1]
            and metrics["target_symdiff"] > 0
            and metrics["output_diff_vs_legacy"] > 0
        )
    if mode == "source_visible":
        return (
            metrics["reachable"] == SOURCE_VISIBLE_COUNT
            and metrics["moved"] == 9
            and metrics["source_symdiff"] > 0
            and metrics["output_diff_vs_legacy"] > 0
        )
    raise ValueError(mode)


def _gen_attempt() -> np.ndarray | None:
    grid, _dest_cells_list, dest_set = _blank_layout()
    reachable_floor, dist = _reachable_floor(grid, dest_set)
    if len(reachable_floor) < MIN_REACHABLE:
        return None

    all_floor = [
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if int(grid[r, c]) == FLOOR and (r, c) not in dest_set
    ]
    unreachable_floor = [cell for cell in all_floor if cell not in set(reachable_floor)]

    center = (CENTER_R, CENTER_C)
    mode = random.choice(("target_visible", "source_visible"))

    if mode == "target_visible":
        if len(reachable_floor) < TARGET_VISIBLE_RANGE[0]:
            return None
        source_cells = _choose_target_visible_sources(reachable_floor)
    else:
        if len(reachable_floor) < SOURCE_VISIBLE_COUNT:
            return None
        source_cells = _find_source_visible_pack(reachable_floor, center, dist)
        if source_cells is None:
            return None

    distractor_cells = _choose_distractors(unreachable_floor, len(source_cells))
    if len(source_cells) + len(distractor_cells) < TOTAL_MAROON_RANGE[0]:
        return None

    inp = grid.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    try:
        out = main(inp)
    except Exception:
        return None

    if not _mode_ok(inp, mode):
        return None
    if not np.array_equal(out, _solve_with_orders(inp, source_mode="combo", target_mode="combo")):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError(f"failed to generate a valid combo sample after {GENERATE_MAX_RETRIES} attempts")
