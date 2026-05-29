# Combo variant of dd6b8c4b (index 43)
# combo_spec: D5/3+D9/0
# Name: 框架上的标记像素+规则矩形框架
#
# 设计分析（先于实现）:
# a) 具体化与行为追踪
#    - 原题: 3x3 绿色目标块周围有不规则品红墙，9 是散落在地板上的像素；从目标块邻接地板做 BFS，
#      找到可达 9，按距离排序后搬进 3x3 区域。
#    - 融合后: 9 不再散落，而是嵌在规则品红矩形框架的边上；矩形固定内缩 1 格，棋盘四角再补 4 个品红隔断，
#      于是框外只剩四条彼此隔离的 1 像素宽橙色条带。框架哪一边开口，就只能到达哪一边的外侧条带，因此只有对应边上的标记可被“收集”。
#    - 例子: 若上边和右边各有开口，而下边/左边无开口，则 BFS 可从中心区域穿过上/右开口到达顶部条带和右侧条带，
#      因而仅上边/右边的 9 参与排序；左边/下边的 9 留在框架上不动。差异点在于“9 的可达性”从散点地板连通性，
#      改成了“框架外侧条带是否被开口接通”。
# b) 复杂度校准
#    - 矩形框架保持唯一且规则，避免再引入第二层复杂墙结构。
#    - 开口每边 0-2 个，边上标记每边 1-4 个，离散档位少且重复充分。
#    - 角点一律保留为纯品红墙，不放标记，避免一个角点同时属于两条边导致可达性歧义。
# c) 隐式假设审查
#    - 原 solver 假设 9 在地板上且可通行；这里不成立，故 BFS 只能走橙色地板，9 仅作为“附着在框架上的候选标记”。
#    - 原 solver 从目标块周围推断不规则墙/地板；这里改为依赖固定 11x11 几何中的 inset-1 矩形框和四个角隔断，
#      generator 保证这一新前提成立。
#    - 原 solver 直接以 9 所在格的距离排序；这里必须改为看 9 的“框外侧邻接地板”是否可达，并以触及该外侧邻格的最短距离排序。
# d) 放置合法性
#    - 目标块固定在中心 3x3，矩形框固定在 bbox=(1,1)-(9,9)，不会越界。
#    - 开口仅取非角边格；标记仅放在未开口的非角边格；三类对象互不重叠。
# e) 视觉可观察性
#    - 内缩 1 格加四角隔断是关键：矩形外侧四条橙色条带互相隔离，人眼能直接看出“哪条边开口，就连通哪条边的外条带”。
#    - 关闭边上的标记在输出中保留，打开边上的部分标记搬入中心，输入输出对比能直接暴露规则。
# f) 可视化整洁度
#    - 元素只分为中心目标块、矩形框、开口和边标记四类，全部轴对齐，无重叠涂抹。
# g) 变化的可推断性
#    - 颜色角色仍少且稳定: 橙=地板，品红=框架，绿/红=目标块，栗色=待搬运标记。
#      没有依赖不可视数值；排序依据来自几何距离与开口位置，能从少量样例中归纳。

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


GRID_H = GRID_W = 11
CENTER_R = CENTER_C = 5
FRAME_R0 = FRAME_C0 = 1
FRAME_R1 = FRAME_C1 = 9

FLOOR_COLOR = 7
WALL_COLOR = 6
DEST_COLOR = 3
RED_CENTER = 2
MARKER_COLOR = 9

SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400


def _dest_cells(h: int, w: int, cr: int, cc: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _side_cells(r0: int, c0: int, r1: int, c1: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "top": [(r0, c) for c in range(c0 + 1, c1)],
        "bottom": [(r1, c) for c in range(c0 + 1, c1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
    }


def _find_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    assert grid.shape == (GRID_H, GRID_W), "generator and solver both assume 11x11 grids"
    r0, c0, r1, c1 = FRAME_R0, FRAME_C0, FRAME_R1, FRAME_C1
    assert grid[r0, c0] == WALL_COLOR and grid[r0, c1] == WALL_COLOR
    assert grid[r1, c0] == WALL_COLOR and grid[r1, c1] == WALL_COLOR
    return r0, c0, r1, c1


def _distance_from_dest(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> np.ndarray:
    h, w = grid.shape
    inf = h * w + 10
    dist = np.full((h, w), inf, dtype=int)
    q = deque()

    for r, c in dest_set:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dest_set and grid[rr, cc] == FLOOR_COLOR:
                if dist[rr, cc] == inf:
                    dist[rr, cc] = 0
                    q.append((rr, cc))

    assert q, "destination block must touch at least one floor cell"

    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dest_set and grid[rr, cc] == FLOOR_COLOR:
                if dist[rr, cc] == inf:
                    dist[rr, cc] = dist[r, c] + 1
                    q.append((rr, cc))

    return dist


def _reachable_markers(
    grid: np.ndarray,
    dist: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = frame_bbox
    inf = int(dist.max())
    records: list[tuple[int, int, int]] = []

    for r, c in np.argwhere(grid == MARKER_COLOR):
        r, c = int(r), int(c)
        if not (r in (r0, r1) or c in (c0, c1)):
            continue

        best = None
        for rr, cc in neighbors4(r, c):
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            is_outside = rr < r0 or rr > r1 or cc < c0 or cc > c1
            if is_outside and grid[rr, cc] == FLOOR_COLOR and dist[rr, cc] < inf:
                cand = int(dist[rr, cc]) + 1
                best = cand if best is None else min(best, cand)

        if best is not None:
            records.append((int(best), r, c))

    records.sort()
    return records


def _transform(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center"
    cr, cc = map(int, centers[0])

    dest_cells = _dest_cells(h, w, cr, cc)
    dest_set = set(dest_cells)
    frame_bbox = _find_frame_bbox(grid)
    dist = _distance_from_dest(grid, dest_set)
    reachable = _reachable_markers(grid, dist, frame_bbox)

    k = min(len(dest_cells), len(reachable))
    chosen = {(r, c) for _d, r, c in reachable[:k]}

    for r, c in chosen:
        out[r, c] = WALL_COLOR
    for r, c in dest_cells[:k]:
        out[r, c] = MARKER_COLOR

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the reachable frame-edge markers into the centered 3x3 destination block.

    Input structure:
    - Orange floor fills the board, with a unique red center inside a fixed green 3x3 target block.
    - A magenta rectangular frame surrounds that block; some non-corner frame cells are replaced by maroon markers, and some frame cells are removed as 1-pixel openings.

    Rule:
    - Flood-fill starts from floor cells adjacent to the green block and travels only through orange floor.
    - A maroon frame marker is collectible only if the flood can touch it from the outside of the rectangle through one of the frame openings.
    - Collect reachable markers in increasing shortest-path distance to that outside touch point, break ties by row then column, erase those markers back to magenta, and write them into the 3x3 block in row-major order.

    Key insight:
    - Because the rectangle is inset by exactly one cell and the four board corners are plugged with magenta blockers, the exterior orange strips stay 4-disconnected, so openings control reachability by frame side rather than by arbitrary global proximity.
    """

    return _transform(grid)


def _base_grid() -> np.ndarray:
    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in _dest_cells(GRID_H, GRID_W, CENTER_R, CENTER_C):
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    grid[FRAME_R0, FRAME_C0 : FRAME_C1 + 1] = WALL_COLOR
    grid[FRAME_R1, FRAME_C0 : FRAME_C1 + 1] = WALL_COLOR
    grid[FRAME_R0 : FRAME_R1 + 1, FRAME_C0] = WALL_COLOR
    grid[FRAME_R0 : FRAME_R1 + 1, FRAME_C1] = WALL_COLOR
    grid[0, 0] = WALL_COLOR
    grid[0, GRID_W - 1] = WALL_COLOR
    grid[GRID_H - 1, 0] = WALL_COLOR
    grid[GRID_H - 1, GRID_W - 1] = WALL_COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    grid = _base_grid()
    side_cells = _side_cells(FRAME_R0, FRAME_C0, FRAME_R1, FRAME_C1)

    n_open_sides = random.randint(1, 3)
    open_sides = set(random.sample(list(SIDES), n_open_sides))

    openings: set[tuple[int, int]] = set()
    for side in SIDES:
        count = random.randint(1, 2) if side in open_sides else 0
        if count:
            openings.update(random.sample(side_cells[side], count))

    marker_counts = {}
    for side in SIDES:
        max_count = len(side_cells[side]) - sum(1 for cell in side_cells[side] if cell in openings)
        if max_count <= 0:
            return None
        marker_counts[side] = random.randint(1, min(4, max_count))

    reachable_count = sum(marker_counts[side] for side in open_sides)
    total_markers = sum(marker_counts.values())
    if not (2 <= reachable_count <= 8):
        return None
    if not (4 <= total_markers <= 16):
        return None
    if total_markers == reachable_count:
        return None

    for r, c in openings:
        grid[r, c] = FLOOR_COLOR

    for side in SIDES:
        choices = [cell for cell in side_cells[side] if cell not in openings]
        markers = random.sample(choices, marker_counts[side])
        for r, c in markers:
            grid[r, c] = MARKER_COLOR

    try:
        out = main(grid)
    except Exception:
        return None

    if not np.any(out == MARKER_COLOR):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
