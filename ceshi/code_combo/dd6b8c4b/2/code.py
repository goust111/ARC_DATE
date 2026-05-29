# Combo variant of dd6b8c4b (index 2)
# combo_spec: D1/0+D9/0
# Name: 直线射线可达性+规则矩形框架

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4

# Design analysis
# ===============
# combo_variant_spec:
#   summary:
#     - 保留原题的中心 3x3 目标块、栗色源点、行优先回填。
#     - 把“可达性”从 BFS 连通改成 12 条水平/垂直直线射线。
#     - 把围墙形状从不规则障碍改成 5x5 规则矩形框架；四条边各有 0-2 个明确的 1 像素开口。
#   concrete_behavior_and_trace:
#     - 目标块固定在中心 rows 4..6, cols 4..6。框架是包围它的 5x5 边框 rows 3..7, cols 3..7。
#     - 射线起点不是任意地板，而是 12 个“边中点位”：
#       top=(3,4..6), bottom=(7,4..6), left=(4..6,3), right=(4..6,7)。
#     - 若 (3,5) 是开口、(2,5) 是地板、(1,5) 是栗色，则向上的射线会命中 (1,5)，它可移动。
#     - 若同一开口下把栗色放到 (2,4)，旧 BFS 可走 (3,5)->(2,5)->(2,4)，原题会吸走它；
#       现在射线只检查 column 5，所以它保持不动。这就是融合后的关键差异。
#     - 若 (3,4) 位置被框架墙封死，即使更外侧的 (1,4) 有栗色，它也既不可见也不可连通；
#       这体现“哪些开口允许哪条射线射出”的框架维度。
#   complexity_calibration:
#     - 每边最多 2 个开口，总开口数最多 8，所以当前变体的可见候选上限是 8。
#       这比原题略低，但仍足够覆盖 2-8 个被搬运源，并保留排序/截断行为。
#     - 为避免样例退化成“只有几条直线上的点”，生成器强制加入至少 1 个 BFS 可达但不在任何视线里的栗色诱饵；
#       人类因此能看到“框架开口 + 直线视野”共同决定结果，而不是单纯最近点搬运。
#     - 同时额外加入 0-3 个被封口挡住的同线栗色，帮助观察“有同线但无开口时不会移动”。
#   implicit_assumptions_review:
#     - 原 main 依赖 `passable={floor,9}` 和 BFS 距离矩阵；融合后这两点都失效，必须改为逐射线扫描。
#     - 原 main 会从目标块周围颜色推断 floor/wall；本变体固定沿用原题配色，
#       因为规则矩形框架与开口位置本身就是重要线索，没必要再引入颜色推断的不确定性。
#     - 原 generator 用 reachable_floor 采样源，会制造很多“连得通但不一定看得到”的点；
#       这在本变体里必须分成三类显式构造：
#       1) visible ray sources
#       2) BFS-only decoys
#       3) blocked-by-frame same-line distractors
#     - 射线一旦遇到栗色就停止，因此一条射线上最多只有最靠内的那颗会参与排序；
#       生成器据此避免把 BFS-only 诱饵放到开口射线的净空通道上。
#   placement_legality:
#     - 所有对象都在 11x11 内；目标块与框架位置固定，因此不会越界。
#     - 开口只从 12 个非角框架格中挑选，角点始终保持墙色，保证“规则矩形框架”的视觉完整性。
#     - visible 源只放在各自开口向外的同一直线上；BFS-only 诱饵只放在框外、且不位于任何开口射线的格子上；
#       blocked-line 诱饵只放在封死射线外侧的同线格子上。三类位置互不重叠。
#   observability_and_cleanliness:
#     - 输入里稳定可见三层结构：中心绿块、外层洋红矩形框架、外部栗色散点。
#     - 输出里只有“视线内首个栗色”会进入目标块，其他近邻栗色保留，可从 2-3 个样例中肉眼归纳。
#     - 不引入额外颜色或隐式编号，排序仍然是可见距离 + 行列顺序，完全可视。
#   generator_guarantees_for_new_main:
#     - 恰好一个红色中心，且中心 3x3 完整存在。
#     - 至少 2 个可见栗色，至多 8 个；至少 1 个 BFS-only 诱饵。
#     - 生成后显式验证：
#       1) 新 solver 与构造目标一致。
#       2) 旧 BFS 规则选中的源集合与新规则不同。
#       3) blocked / BFS-only 诱饵都不会误入新规则的可见集合。

GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9

MIN_VISIBLE = 2
MAX_VISIBLE = 8
TOTAL_MAROON_RANGE = (4, 16)
MAX_BFS_ONLY = 6
MAX_BLOCKED_LINE = 3
GEN_RETRIES = 500


def _dest_cells(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _ray_specs(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int, int, int]]:
    specs: list[tuple[int, int, int, int]] = []
    for c in range(cc - 1, cc + 2):
        if 0 <= cr - 2 < h:
            specs.append((cr - 2, c, -1, 0))
        if 0 <= cr + 2 < h:
            specs.append((cr + 2, c, 1, 0))
    for r in range(cr - 1, cr + 2):
        if 0 <= cc - 2 < w:
            specs.append((r, cc - 2, 0, -1))
        if 0 <= cc + 2 < w:
            specs.append((r, cc + 2, 0, 1))
    return specs


def _ray_lane(spec: tuple[int, int, int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c, dr, dc = spec
    lane: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        lane.append((r, c))
        r += dr
        c += dc
    return lane


def _side_groups(cr: int, cc: int, h: int, w: int) -> dict[str, list[tuple[int, int, int, int]]]:
    return {
        "top": [(cr - 2, c, -1, 0) for c in range(cc - 1, cc + 2) if 0 <= cr - 2 < h],
        "bottom": [(cr + 2, c, 1, 0) for c in range(cc - 1, cc + 2) if 0 <= cr + 2 < h],
        "left": [(r, cc - 2, 0, -1) for r in range(cr - 1, cr + 2) if 0 <= cc - 2 < w],
        "right": [(r, cc + 2, 0, 1) for r in range(cr - 1, cr + 2) if 0 <= cc + 2 < w],
    }


def _frame_cells(cr: int, cc: int) -> set[tuple[int, int]]:
    r0, r1 = cr - 2, cr + 2
    c0, c1 = cc - 2, cc + 2
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _ray_candidates(grid: np.ndarray, cr: int, cc: int) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for sr, sc, dr, dc in _ray_specs(cr, cc, h, w):
        r, c = sr, sc
        dist = 0
        while 0 <= r < h and 0 <= c < w:
            val = int(grid[r, c])
            if val == FLOOR_COLOR:
                r += dr
                c += dc
                dist += 1
                continue
            if val == MAROON:
                candidates.append((dist, r, c))
            break
    candidates.sort()
    return candidates


def _old_bfs_ranked_maroons(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    queue: deque[tuple[int, int]] = deque()

    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set:
                if int(grid[nr, nc]) in (FLOOR_COLOR, MAROON) and dist[nr, nc] == inf:
                    dist[nr, nc] = 0
                    queue.append((nr, nc))

    while queue:
        r, c = queue.popleft()
        d = int(dist[r, c])
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set:
                if int(grid[nr, nc]) in (FLOOR_COLOR, MAROON) and dist[nr, nc] == inf:
                    dist[nr, nc] = d + 1
                    queue.append((nr, nc))

    ranked = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON) if int(dist[r, c]) < inf]
    ranked.sort()
    return ranked


def _reachable_floor_cells(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = grid.shape
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()

    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set:
                if int(grid[nr, nc]) == FLOOR_COLOR and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and (nr, nc) not in visited:
                if int(grid[nr, nc]) == FLOOR_COLOR:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return visited


def _choose_open_specs() -> list[tuple[int, int, int, int]]:
    groups = _side_groups(CENTER_R, CENTER_C, GRID_H, GRID_W)
    while True:
        open_specs: list[tuple[int, int, int, int]] = []
        for side_specs in groups.values():
            count = random.randint(0, 2)
            if count > 0:
                open_specs.extend(random.sample(side_specs, count))
        if MIN_VISIBLE <= len(open_specs) <= MAX_VISIBLE:
            return open_specs


def _place_frame(base: np.ndarray, open_specs: set[tuple[int, int, int, int]]) -> None:
    all_specs = set(_ray_specs(CENTER_R, CENTER_C, GRID_H, GRID_W))
    for r, c in _frame_cells(CENTER_R, CENTER_C):
        base[r, c] = WALL_COLOR
    for spec in open_specs:
        sr, sc, _, _ = spec
        base[sr, sc] = FLOOR_COLOR
    for sr, sc, _, _ in all_specs - open_specs:
        base[sr, sc] = WALL_COLOR


def _visible_source_cells(open_specs: list[tuple[int, int, int, int]]) -> dict[tuple[int, int, int, int], tuple[int, int]]:
    source_cells: dict[tuple[int, int, int, int], tuple[int, int]] = {}
    for spec in open_specs:
        lane = _ray_lane(spec, GRID_H, GRID_W)
        assert len(lane) >= 2
        # Keep at least one floor cell beyond the opening so the ray behavior is legible.
        idx = random.randint(1, len(lane) - 1)
        source_cells[spec] = lane[idx]
    return source_cells


def _blocked_same_line_cells(closed_specs: list[tuple[int, int, int, int]], occupied: set[tuple[int, int]]) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for spec in closed_specs:
        lane = _ray_lane(spec, GRID_H, GRID_W)
        for cell in lane[1:]:
            if cell not in occupied:
                candidates.append(cell)
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for cell in candidates:
        if cell in occupied:
            continue
        chosen.append(cell)
        occupied.add(cell)
        if len(chosen) >= MAX_BLOCKED_LINE:
            break
    return chosen


def _bfs_only_candidates(
    base: np.ndarray,
    dest_set: set[tuple[int, int]],
    open_specs: list[tuple[int, int, int, int]],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    reachable = _reachable_floor_cells(base, dest_set)
    frame_cells = _frame_cells(CENTER_R, CENTER_C)
    open_lane_cells: set[tuple[int, int]] = set()
    for spec in open_specs:
        open_lane_cells.update(_ray_lane(spec, GRID_H, GRID_W))

    candidates: list[tuple[int, int]] = []
    for cell in reachable:
        if cell in occupied:
            continue
        if cell in open_lane_cells:
            continue
        if cell in frame_cells:
            continue
        candidates.append(cell)

    # Prefer cells close to at least one opening so the "turning after exiting" pattern is visually obvious.
    def score(cell: tuple[int, int]) -> tuple[int, int, int]:
        r, c = cell
        best = min(abs(r - sr) + abs(c - sc) for sr, sc, _, _ in open_specs)
        return (best, r, c)

    candidates.sort(key=score)
    return candidates


def _build_expected(grid: np.ndarray, dest_cells: list[tuple[int, int]], chosen_sources: set[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for r, c in chosen_sources:
        out[r, c] = FLOOR_COLOR
    for r, c in dest_cells[: len(chosen_sources)]:
        out[r, c] = MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the closest maroon pixels that are directly visible through the frame openings into the central 3×3 block.

    Input structure:
    - An orange field contains a fixed green 3×3 target block with one red center and a magenta 5×5 rectangular frame around it.
    - The frame is mostly solid, but some non-corner side cells are single-cell openings; maroon pixels are scattered outside the frame.

    Rule:
    - From each boundary cell of the 3×3 block, fire one outward horizontal or vertical ray. A ray can travel only through orange cells and stops at the first non-orange cell.
    - That first blocking cell contributes only if it is maroon, so a frame opening exposes at most one candidate on that line while a wall cell blocks the ray immediately.
    - Sort all visible candidates by ray distance, then row and column; move the earliest ones into the destination cells in row-major order and erase only the selected sources.

    Key insight:
    - The puzzle is about line of sight through explicit rectangular-frame openings, not about general corridor connectivity. A maroon pixel may still be reachable after exiting an opening, but if it requires a turn or sits behind another blocker on the same ray, it stays where it is.
    """

    grid = np.asarray(grid, dtype=int)
    out = grid.copy()
    h, w = grid.shape

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red (2) center pixel"
    cr, cc = map(int, centers[0])

    dest_cells = _dest_cells(cr, cc, h, w)
    visible = _ray_candidates(grid, cr, cc)

    k = min(len(dest_cells), len(visible))
    chosen = {(r, c) for _, r, c in visible[:k]}

    for r, c in chosen:
        out[r, c] = FLOOR_COLOR
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON

    return out


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _dest_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)
    dest_set = set(dest_cells)

    base = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        base[r, c] = DEST_COLOR
    base[CENTER_R, CENTER_C] = RED_CENTER

    open_specs = _choose_open_specs()
    open_set = set(open_specs)
    all_specs = _ray_specs(CENTER_R, CENTER_C, GRID_H, GRID_W)
    closed_specs = [spec for spec in all_specs if spec not in open_set]

    _place_frame(base, open_set)

    visible_map = _visible_source_cells(open_specs)
    occupied = set(dest_cells) | _frame_cells(CENTER_R, CENTER_C)
    for cell in visible_map.values():
        if cell in occupied:
            return None
        occupied.add(cell)
        base[cell] = MAROON

    blocked_cells = _blocked_same_line_cells(closed_specs, occupied)
    for r, c in blocked_cells:
        base[r, c] = MAROON

    bfs_only_pool = _bfs_only_candidates(base, dest_set, open_specs, occupied)
    if not bfs_only_pool:
        return None

    max_bfs_only = min(MAX_BFS_ONLY, len(bfs_only_pool), TOTAL_MAROON_RANGE[1] - len(visible_map) - len(blocked_cells))
    if max_bfs_only <= 0:
        return None

    min_total, _max_total = TOTAL_MAROON_RANGE
    need_bfs_only = max(1, min_total - len(visible_map) - len(blocked_cells))
    low = max(1, need_bfs_only)
    if low > max_bfs_only:
        return None
    bfs_only_count = random.randint(low, max_bfs_only)
    bfs_only_cells = bfs_only_pool[:bfs_only_count]
    for r, c in bfs_only_cells:
        base[r, c] = MAROON
        occupied.add((r, c))

    visible = _ray_candidates(base, CENTER_R, CENTER_C)
    visible_sources = {(r, c) for _, r, c in visible}
    expected_visible = set(visible_map.values())
    if visible_sources != expected_visible:
        return None

    ranked_old = _old_bfs_ranked_maroons(base, dest_set)
    old_chosen = {(r, c) for _, r, c in ranked_old[: min(len(dest_cells), len(ranked_old))]}
    new_chosen = {(r, c) for _, r, c in visible[: min(len(dest_cells), len(visible))]}
    if old_chosen == new_chosen:
        return None
    if not any(cell in old_chosen for cell in bfs_only_cells):
        return None

    out = _build_expected(base, dest_cells, new_chosen)

    try:
        result = main(base)
    except Exception:
        return None
    if not np.array_equal(result, out):
        return None
    return base


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
