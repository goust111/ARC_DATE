# Variant of dd6b8c4b
# Dimension: D1 — 可达性判定方式 (path)
# Change: flood_fill——从绿色方块出发，通过橙色背景像素进行4连通flood-fill扩散，判断栗色像素是否可达... → 直线射线可达性

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4

# Design analysis
# ===============
# variant_spec:
#   core_change:
#     old_path:
#       从 3x3 目标块外侧的可通行格开始做 4 连通 BFS，所有通过橙色地板连到的栗色都算候选。
#     new_path:
#       不再允许拐弯。目标块四边各自向外发射水平/垂直射线；射线只能穿过橙色地板，
#       一旦遇到第一个非地板像素就停止，且只有这个第一个阻挡像素是栗色时，它才是候选。
#   concrete_behavior:
#     grid:
#       仍然是 11x11，中心固定为绿色 3x3 块，中央唯一红点。
#       外侧一圈大多是洋红墙，但会留出若干开口，像“观察窗”一样决定哪些射线能射出去。
#     visible_effect:
#       直线正前方的栗色会被吸入目标块；只要需要绕开墙或先直走再拐弯才能到达，
#       哪怕旧 BFS 规则下是连通的，现在也不会移动。
#   trace_example:
#     old_logic:
#       顶边中央开口在 (3,5)，栗色在 (2,4)。BFS 可走 (3,5)->(2,5)->(2,4)，因此旧题会移动它。
#     new_logic:
#       同一布局下，向上的射线只检查列 5；它会看到 (2,5),(1,5)...，永远看不到拐到 (2,4) 的栗色，
#       所以该栗色留在原位。只有同列上的第一个栗色才会被移动。
#     delta:
#       差异发生在“候选筛选”环节，排序与回填机制保持不变。
#   complexity_calibration:
#     - 如果仍然把墙开口限制在 1-4 个，则每条射线最多拿到 1 个栗色，候选上限太低，
#       会把原题的 2-9 甚至 >9 候选排序空间压缩掉。
#     - 因此生成器允许 3-12 条开射线；这样既能保留“有墙、有限视野”的外观，
#       也能继续生成 2-9 个会被填入目标块的案例，并偶尔出现 >9 个可见栗色以保留截断排序。
#   implicit_assumptions_review:
#     - 原 solver 用 BFS 距离矩阵 `dist` 假设“可达性=连通性”；变体中这个前提失效，必须改成逐射线扫描。
#     - 原 solver 把 `passable={floor,9}`，意味着 maroon 也可作为通道；变体中 maroon 反而应该终止视线，
#       否则一条射线会穿透前景，退回到“同一直线的整串都可达”的弱化版本。
#     - 原 generator 通过 `reachable_floor` 采样源像素；变体中如果继续这样放置，会大量生成“连通但不可见”的栗色，
#       导致 solver 与 generator 语义脱节。必须显式区分 ray-visible 源 与 BFS-only 诱饵。
#   placement_legality:
#     - 目标块固定在中心，所有栗色都放在块外。
#     - 射线源位置是目标块外侧一圈的 12 个格子；生成器只在其中一部分留空，其余设墙。
#     - 可见源像素每条射线至多 1 个，且放在该射线的直线上；BFS-only 诱饵只放在“旧 BFS 可到但当前射线看不到”的地板上。
#   observability:
#     - 样例会稳定包含两类栗色：直线正前方会移动的，和虽然连得上但不在视线里的不会移动的。
#       这使人类能从少量样例中看出“视野而非连通性”。
#     - 颜色、排序、目标位置都保持原题风格，解题者只需要重新识别“路径定义”这一维变化。
#   cleanliness:
#     - 仍然只使用 5 种颜色，结构层次清晰：中心目标块、外圈墙、外部地板、栗色点。
#     - 不叠加额外标记，不引入不可见属性排序。
#   solver_assumptions_and_generator_guarantees:
#     - solver 假设恰好一个红色中心，且周围 3x3 都在网格内；生成器固定满足。
#     - solver 假设颜色固定为 floor=7, wall=6, dest=3, center=2, maroon=9；生成器固定满足。
#     - solver 假设被选中的可见源数量可为 0-12；generator 至少放 2 个，且经验证输出与 solver 一致。

GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9

OPEN_RAY_COUNT_RANGE = (3, 12)
VISIBLE_TOTAL_RANGE = (2, 12)
TOTAL_MAROON_RANGE = (4, 16)
BFS_ONLY_RANGE = (1, 6)
OUTER_WALL_PROB = 0.10
OUTER_WALL_MAX = 8
GEN_RETRIES = 400


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


def _old_bfs_reachable(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = grid.shape
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set:
                if int(grid[nr, nc]) in (FLOOR_COLOR, MAROON) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and (nr, nc) not in visited:
                if int(grid[nr, nc]) in (FLOOR_COLOR, MAROON):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return visited


def _ray_clear_floor_cells(grid: np.ndarray, cr: int, cc: int) -> set[tuple[int, int]]:
    h, w = grid.shape
    clear: set[tuple[int, int]] = set()
    for sr, sc, dr, dc in _ray_specs(cr, cc, h, w):
        r, c = sr, sc
        while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == FLOOR_COLOR:
            clear.add((r, c))
            r += dr
            c += dc
    return clear


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the closest directly visible maroon pixels into the 3×3 block around the unique red center.

    Input structure:
    - Orange floor surrounds a fixed green 3×3 block with one red center, plus optional magenta wall cells and scattered maroon pixels.
    - The only important access routes are the horizontal/vertical sight lines that continue straight outward from the block's four sides.

    Rule:
    - From each side cell of the 3×3 block, shoot one outward ray. A ray may cross orange floor cells only; it stops at the first non-floor cell.
    - That first blocking cell counts as a candidate only when it is maroon. Rank all such candidates by ray distance, then by row and column.
    - Move the closest candidates into the destination block in row-major order, erasing only the maroon sources that were selected.

    Key insight:
    - Reachability is no longer about turning through corridors. A maroon pixel can be topologically connected through the floor yet still remain untouched if it is not the first thing seen on a straight ray from the green block.
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


def _add_outer_walls(grid: np.ndarray, open_specs: list[tuple[int, int, int, int]], source_cells: set[tuple[int, int]]) -> None:
    protected = set(source_cells)
    for spec in open_specs:
        protected.update(_ray_lane(spec, GRID_H, GRID_W))

    candidates: list[tuple[int, int]] = []
    for r in range(GRID_H):
        for c in range(GRID_W):
            if grid[r, c] != FLOOR_COLOR:
                continue
            if (r, c) in protected:
                continue
            if 2 <= r <= 8 and 2 <= c <= 8:
                continue
            candidates.append((r, c))

    random.shuffle(candidates)
    added = 0
    for r, c in candidates:
        if added >= OUTER_WALL_MAX:
            break
        if random.random() >= OUTER_WALL_PROB:
            continue
        grid[r, c] = WALL_COLOR
        added += 1


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _dest_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)
    dest_set = set(dest_cells)

    base = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        base[r, c] = DEST_COLOR
    base[CENTER_R, CENTER_C] = RED_CENTER

    specs = _ray_specs(CENTER_R, CENTER_C, GRID_H, GRID_W)
    max_visible = min(VISIBLE_TOTAL_RANGE[1], len(specs))
    visible_total = random.randint(VISIBLE_TOTAL_RANGE[0], max_visible)
    open_count = random.randint(max(OPEN_RAY_COUNT_RANGE[0], visible_total), OPEN_RAY_COUNT_RANGE[1])

    open_specs = random.sample(specs, open_count)
    open_set = set(open_specs)

    for spec in specs:
        sr, sc, _, _ = spec
        if spec not in open_set:
            base[sr, sc] = WALL_COLOR

    source_specs = random.sample(open_specs, visible_total)
    source_cells: set[tuple[int, int]] = set()
    for spec in source_specs:
        lane = _ray_lane(spec, GRID_H, GRID_W)
        assert len(lane) >= 2
        dist = random.randint(1, len(lane) - 1)
        r, c = lane[dist]
        source_cells.add((r, c))
        base[r, c] = MAROON

    _add_outer_walls(base, open_specs, source_cells)

    visible_now = _ray_candidates(base, CENTER_R, CENTER_C)
    if len(visible_now) != visible_total:
        return None
    if {(r, c) for _, r, c in visible_now} != source_cells:
        return None

    clear_cells = _ray_clear_floor_cells(base, CENTER_R, CENTER_C)
    bfs_reachable = _old_bfs_reachable(base, dest_set)

    bfs_only_candidates = [
        (r, c)
        for (r, c) in bfs_reachable
        if base[r, c] == FLOOR_COLOR and (r, c) not in clear_cells and (r, c) not in source_cells
    ]
    if not bfs_only_candidates:
        return None

    max_extra = min(
        BFS_ONLY_RANGE[1],
        len(bfs_only_candidates),
        TOTAL_MAROON_RANGE[1] - visible_total,
    )
    min_extra = min(BFS_ONLY_RANGE[0], max_extra)
    if visible_total + max_extra < TOTAL_MAROON_RANGE[0]:
        return None
    if max_extra <= 0:
        return None

    need_floor = max(0, TOTAL_MAROON_RANGE[0] - visible_total)
    low = max(min_extra, need_floor)
    if low > max_extra:
        return None
    bfs_only_count = random.randint(low, max_extra)
    bfs_only_cells = random.sample(bfs_only_candidates, bfs_only_count)

    inp = base.copy()
    for r, c in bfs_only_cells:
        inp[r, c] = MAROON

    # Ensure these added maroons stay off the direct rays.
    visible_after = _ray_candidates(inp, CENTER_R, CENTER_C)
    if {(r, c) for _, r, c in visible_after} != source_cells:
        return None

    ranked = sorted(visible_after)
    k = min(len(dest_cells), len(ranked))
    chosen_sources = {(r, c) for _, r, c in ranked[:k]}

    out = inp.copy()
    for r, c in chosen_sources:
        out[r, c] = FLOOR_COLOR
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON

    try:
        result = main(inp)
        if np.array_equal(result, out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
