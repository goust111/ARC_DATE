# Combo variant of dd6b8c4b (index 42)
# combo_spec: D1/0+D14/1
# Name: 方块位于角落+直线射线可达性

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4

# Design analysis
# ===============
# combo_variant:
#   concrete_behavior:
#     - 保留原题的 11x11 画布、3x3 绿色目标块、唯一红色中心、橙地板 / 品红墙 / 栗色 token。
#     - D1/0 把“可达”改成“视线首阻挡”：从 3x3 方块四边外侧的 12 个位置分别向外发射水平/竖直射线，
#       射线只穿过橙地板，遇到第一个非地板像素就停止；只有该首个阻挡像素是栗色时，它才会被吸入目标块。
#     - D14/1 不改变求解动作本身，但强制绿色方块在一个更大的品红矩形框内偏向某个角落：
#       两个相邻方向是近侧，离墙只有 0-2 格；另外两个方向是远侧，离墙有 3-4 格。
#       因而 12 条射线长度高度不对称，输入看起来像“角落里的观察口”而不是居中的房间。
#   trace_example:
#     - 例：若方块偏在框内左上角，gap=(top=1,left=0,bottom=4,right=3)。
#       向上射线最多只能看见 1 格后撞墙，向左射线立即撞墙；向下 / 向右则能沿长走廊继续扫描。
#     - 原题 flood-fill 时，室内所有拐弯可达的栗色都会参与排序；例如位于下方长走廊旁边一格的栗色，
#       可以通过“先下再转弯”连到目标块，因此旧题会移动它。
#     - 融合后同一栗色虽然仍在同一房间内，但不在任何首阻挡直线视野上，所以保留在原位。
#       真正被移动的只是一组“每条视线首先看见的栗色”。
#   complexity_calibration:
#     - 如果把方块完全贴死在角落，两侧射线会直接失效，最多只剩 6 个候选，容易让“距离排序/截断到 9”失去作用。
#     - 因此生成器采用“靠角但不总是贴墙”的 near-gap 方案：近侧 gap 为 0-2，远侧 gap 为 3-4。
#       这样仍能看出明显角落偏置，同时不少样例保留 10-12 条可用射线，排序和 `>9` 截断仍会真实发生。
#     - 另外，生成器稳定加入至少一枚“旧 BFS 可达、但新射线不可见”的室内栗色；并在部分射线上放置被前景遮挡的深层栗色。
#       这保证 D1 的变化不是装饰，而是会改变输出。
#   implicit_assumptions_review:
#     - 原 solver 的核心假设“可达性=4 连通连通性”失效，必须改成逐射线扫描首个非地板像素。
#     - 原 solver 允许栗色充当 BFS 通道；在本题里栗色恰恰必须终止视线，否则会退化成“同一射线整串都可见”。
#     - 原 generator 默认目标块固定在中心附近且墙只围成对称外圈；D14/1 打破这一点，所以 solver 不能依赖绝对坐标，
#       只能通过唯一红心恢复 3x3 目标块。
#     - 新 solver 还隐含假设：颜色固定、恰有一个红心、目标块完整在网格内、墙不会覆盖目标块。
#       下面的 generator 固定使用这些颜色并显式构造合法框与目标块来保证它们。
#   placement_legality:
#     - 框始终是轴对齐矩形边框；目标块严格放在框内；墙、目标块、栗色互不重叠。
#     - 近侧 / 远侧 gap 直接决定框尺寸：`height = top_gap + bottom_gap + 5`，
#       `width = left_gap + right_gap + 5`，因此不会出现越界或墙压到目标块的情况。
#     - 所有“可见源”都放在某条射线的室内地板段上；所有“BFS-only 诱饵”都放在同一房间内但避开全部射线格；
#       因此它们一定旧题可达、但不会意外变成新题的首阻挡候选。
#   observability_and_cleanliness:
#     - 人类可直接看到两个稳定线索：一是绿色块明显偏在框的一角，四侧空隙长短悬殊；二是有些栗色明明和目标块在同一房间里，
#       输出却不移动，只有正对某条短/长视线的栗色会被搬进 3x3 块。
#     - 颜色总数仍只有 5 种，图层只有“框 / 地板 / 目标块 / 栗色”四层，没有隐藏数值排序或不可见标记。
#       框内诱饵与射线候选通过几何位置分离，视觉上不混杂。
#   anti_degeneracy_checks:
#     - 生成器拒绝 `main(inp)` 与旧 BFS 版 `_old_main(inp)` 完全相同的样本，避免退化成原题。
#     - 生成器还记录两个关键控制量是否生效：
#       1. `bfs_only_count > 0`，证明存在“连得上但看不见”的栗色。
#       2. `corner_bias_strength = min(far_gap - near_gap)`，证明方块确实偏在角落而非居中。

GRID_H, GRID_W = 11, 11
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9

CORNERS = ("tl", "tr", "bl", "br")
GEN_RETRIES = 500
TOTAL_MAROON_RANGE = (4, 16)


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


def _ray_side(spec: tuple[int, int, int, int]) -> str:
    _, _, dr, dc = spec
    if dr == -1:
        return "up"
    if dr == 1:
        return "down"
    if dc == -1:
        return "left"
    return "right"


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


def _bfs_ranked_maroons(grid: np.ndarray, cr: int, cc: int) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    dest_cells = _dest_cells(cr, cc, h, w)
    dest_set = set(dest_cells)
    passable = {FLOOR_COLOR, MAROON}

    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    dist = np.full((h, w), -1, dtype=int)

    for r, c in dest_cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and int(grid[nr, nc]) in passable:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    dist[nr, nc] = 0
                    queue.append((nr, nc))

    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and (nr, nc) not in visited:
                if int(grid[nr, nc]) in passable:
                    visited.add((nr, nc))
                    dist[nr, nc] = dist[r, c] + 1
                    queue.append((nr, nc))

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON) if dist[r, c] >= 0]
    maroons.sort()
    return maroons


def _old_main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    centers = np.argwhere(out == RED_CENTER)
    assert len(centers) == 1
    cr, cc = map(int, centers[0])
    dest_cells = _dest_cells(cr, cc, *out.shape)
    ranked = _bfs_ranked_maroons(out, cr, cc)
    k = min(len(dest_cells), len(ranked))
    chosen = {(r, c) for _, r, c in ranked[:k]}
    for r, c in chosen:
        out[r, c] = FLOOR_COLOR
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON
    return out


def _open_floor_run(grid: np.ndarray, spec: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    h, w = grid.shape
    r, c, dr, dc = spec
    lane: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == FLOOR_COLOR:
        lane.append((r, c))
        r += dr
        c += dc
    return lane


def _sample_gap_profile(corner: str) -> dict[str, int]:
    if random.random() < 0.30:
        near_a = random.randint(0, 1)
        near_b = random.randint(1, 2)
    else:
        near_a = random.randint(1, 2)
        near_b = random.randint(1, 2)

    far_a = random.randint(max(near_a + 2, 3), 4)
    far_b = random.randint(max(near_b + 2, 3), 4)

    if corner == "tl":
        return {"up": near_a, "left": near_b, "down": far_a, "right": far_b}
    if corner == "tr":
        return {"up": near_a, "right": near_b, "down": far_a, "left": far_b}
    if corner == "bl":
        return {"down": near_a, "left": near_b, "up": far_a, "right": far_b}
    return {"down": near_a, "right": near_b, "up": far_a, "left": far_b}


def _draw_frame_and_dest(
    grid: np.ndarray, gaps: dict[str, int], top: int, left: int
) -> tuple[tuple[int, int, int, int], int, int]:
    total_h = gaps["up"] + gaps["down"] + 5
    total_w = gaps["left"] + gaps["right"] + 5
    bottom = top + total_h - 1
    right = left + total_w - 1

    grid[top : bottom + 1, left] = WALL_COLOR
    grid[top : bottom + 1, right] = WALL_COLOR
    grid[top, left : right + 1] = WALL_COLOR
    grid[bottom, left : right + 1] = WALL_COLOR

    dest_r0 = top + 1 + gaps["up"]
    dest_c0 = left + 1 + gaps["left"]
    for r in range(dest_r0, dest_r0 + 3):
        for c in range(dest_c0, dest_c0 + 3):
            grid[r, c] = DEST_COLOR
    cr, cc = dest_r0 + 1, dest_c0 + 1
    grid[cr, cc] = RED_CENTER

    return (top, left, bottom, right), cr, cc


def _corner_bias_strength(gaps: dict[str, int], corner: str) -> int:
    if corner == "tl":
        return min(gaps["down"] - gaps["up"], gaps["right"] - gaps["left"])
    if corner == "tr":
        return min(gaps["down"] - gaps["up"], gaps["left"] - gaps["right"])
    if corner == "bl":
        return min(gaps["up"] - gaps["down"], gaps["right"] - gaps["left"])
    return min(gaps["up"] - gaps["down"], gaps["left"] - gaps["right"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull the closest straight-line visible maroon pixels into the 3×3 block around the unique red center.

    Input structure:
    - An orange floor contains one green 3×3 destination block with a red center, enclosed by a magenta rectangular frame.
    - The frame is asymmetric: the green block sits near one frame corner, so some outward directions are short and cramped while the opposite directions open into longer corridors.

    Rule:
    - From each of the 12 side-adjacent positions around the 3×3 block, fire one horizontal or vertical ray outward.
    - A ray crosses only orange floor and stops at its first non-floor cell; that cell is a candidate only if it is maroon.
    - Rank all candidates by ray distance, then by row and column, and move the closest ones into the destination block in row-major order while erasing only the selected source pixels.

    Key insight:
    - Being in the same room is no longer enough. The magenta frame puts the block in a corner-biased viewing position, so many maroon pixels are flood-fill reachable but still ignored because no straight ray sees them first.
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


# --------------- generator (combo D1/0 + D14/1) ---------------


def _gen_attempt() -> np.ndarray | None:
    base = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)

    corner = random.choice(CORNERS)
    gaps = _sample_gap_profile(corner)
    total_h = gaps["up"] + gaps["down"] + 5
    total_w = gaps["left"] + gaps["right"] + 5
    if total_h > GRID_H or total_w > GRID_W:
        return None

    frame_top = random.randint(0, GRID_H - total_h)
    frame_left = random.randint(0, GRID_W - total_w)
    frame, cr, cc = _draw_frame_and_dest(base, gaps, frame_top, frame_left)
    dest_set = set(_dest_cells(cr, cc, GRID_H, GRID_W))

    if _corner_bias_strength(gaps, corner) < 2:
        return None

    specs = _ray_specs(cr, cc, GRID_H, GRID_W)
    lane_map = {spec: _open_floor_run(base, spec) for spec in specs}
    available_specs = [spec for spec in specs if lane_map[spec]]
    if len(available_specs) < 4:
        return None

    far_sides = {
        "tl": {"down", "right"},
        "tr": {"down", "left"},
        "bl": {"up", "right"},
        "br": {"up", "left"},
    }[corner]
    far_specs = [spec for spec in available_specs if _ray_side(spec) in far_sides]
    near_specs = [spec for spec in available_specs if _ray_side(spec) not in far_sides]
    if len(far_specs) < 4:
        return None

    if len(available_specs) >= 10 and random.random() < 0.65:
        visible_total = random.randint(10, len(available_specs))
    else:
        visible_total = random.randint(4, min(9, len(available_specs)))

    far_take_low = max(2, visible_total - len(near_specs))
    far_take_high = min(len(far_specs), visible_total)
    if far_take_low > far_take_high:
        return None
    far_take = random.randint(far_take_low, far_take_high)
    near_take = visible_total - far_take

    selected_specs = random.sample(far_specs, far_take)
    if near_take > 0:
        selected_specs += random.sample(near_specs, near_take)

    inp = base.copy()
    visible_cells: set[tuple[int, int]] = set()
    visible_index: dict[tuple[int, int, int, int], int] = {}

    for spec in selected_specs:
        lane = lane_map[spec]
        idx = random.randint(0, len(lane) - 1)
        cell = lane[idx]
        visible_cells.add(cell)
        visible_index[spec] = idx
        inp[cell] = MAROON

    shadow_sources: list[tuple[tuple[int, int, int, int], list[tuple[int, int]]]] = []
    for spec in selected_specs:
        lane = lane_map[spec]
        later = lane[visible_index[spec] + 1 :]
        if later:
            shadow_sources.append((spec, later))

    random.shuffle(shadow_sources)
    shadow_cap = min(2, len(shadow_sources), TOTAL_MAROON_RANGE[1] - len(visible_cells))
    shadow_count = random.randint(0, shadow_cap) if shadow_cap > 0 else 0
    for spec, later in shadow_sources[:shadow_count]:
        cell = random.choice(later)
        inp[cell] = MAROON

    lane_union: set[tuple[int, int]] = set()
    for lane in lane_map.values():
        lane_union.update(lane)

    room_cells = {
        (r, c)
        for r in range(frame[0] + 1, frame[2])
        for c in range(frame[1] + 1, frame[3])
        if (r, c) not in dest_set
    }
    bfs_only_candidates = [cell for cell in room_cells if cell not in lane_union and int(inp[cell]) == FLOOR_COLOR]
    if not bfs_only_candidates:
        return None

    room_spare = TOTAL_MAROON_RANGE[1] - int((inp == MAROON).sum())
    bfs_only_count = random.randint(1, min(4, len(bfs_only_candidates), max(1, room_spare)))
    for r, c in random.sample(bfs_only_candidates, bfs_only_count):
        inp[r, c] = MAROON

    outside_floor = [
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if int(inp[r, c]) == FLOOR_COLOR and not (frame[0] <= r <= frame[2] and frame[1] <= c <= frame[3])
    ]
    outside_cap = min(2, len(outside_floor), TOTAL_MAROON_RANGE[1] - int((inp == MAROON).sum()))
    outside_count = random.randint(0, outside_cap) if outside_cap > 0 else 0
    for r, c in random.sample(outside_floor, outside_count):
        inp[r, c] = MAROON

    total_maroons = int((inp == MAROON).sum())
    if total_maroons < TOTAL_MAROON_RANGE[0] or total_maroons > TOTAL_MAROON_RANGE[1]:
        return None

    visible_ranked = _ray_candidates(inp, cr, cc)
    if len(visible_ranked) != len(visible_cells):
        return None
    if {(r, c) for _, r, c in visible_ranked} != visible_cells:
        return None

    bfs_ranked = _bfs_ranked_maroons(inp, cr, cc)
    bfs_only_maroons = {(r, c) for _, r, c in bfs_ranked} - visible_cells
    if not bfs_only_maroons:
        return None

    new_out = main(inp)
    old_out = _old_main(inp)
    if np.array_equal(new_out, old_out):
        return None

    expected = inp.copy()
    dest_cells = _dest_cells(cr, cc, GRID_H, GRID_W)
    k = min(len(dest_cells), len(visible_ranked))
    chosen = {(r, c) for _, r, c in visible_ranked[:k]}
    for r, c in chosen:
        expected[r, c] = FLOOR_COLOR
    for r, c in dest_cells[:k]:
        expected[r, c] = MAROON

    if not np.array_equal(new_out, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
