# Variant of 8f3a5a89
# Dimension: D14 — 涂色颜色规则 (trajectory)
# Change: 统一涂橙色 (color 7)，所有可达边界像素使用同一种颜色... → 按相邻墙壁颜色决定涂色

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# Design analysis
# - requested_reference: 16de56c4
# - reference_solver_status: 当前工作区里没有 `16de56c4` 的 solver 文件，只找到题目数据
#   `ARC-data/ARC-AGI-2/data/evaluation/16de56c4.json`，因此按“找不到就跳过”处理。
# design_analysis:
#   concretization:
#   - 保留原题的几何骨架：左下角单像素 marker、边界接触的结构墙、marker 一侧的可达区域、
#     以及在可达区域内沿边界涂出 1 像素厚的内边框。
#   - D14 只改“边框涂什么颜色”。输入里现在允许两种结构墙颜色：
#     蓝墙 `1` 和红墙 `2`；输出边框对应映射为：蓝墙旁涂橙 `7`，红墙旁涂绿 `3`。
#   - 为了让人类能看出规律，生成器强制两种结构墙都出现在每个样例里，并且两种填色都真的出
#     现在输出边框中；否则样例会退化成原题的“单色边框”。
#   behavior_trace:
#   - 原逻辑：某个可达边界像素只要满足“在网格边缘 / 8 邻接结构墙 / 4 邻接不可达区”之一，
#     就统一改成 `7`。
#   - 新逻辑：仍先找出同一批边界像素，但不再无条件写 `7`；而是查询离它最近的结构墙像素
#     属于哪种墙色，再通过 `1->7, 2->3` 的映射决定输出颜色。
#   - 例子：若某个边界像素位于蓝色竖墙内侧，原题输出 `7`，变体仍输出 `7`；若位于红色矩形
#     内侧，原题也会输出 `7`，但变体必须改成 `3`。这就是唯一核心差异。
#   complexity_calibration:
#   - 只使用两种墙色和两种对应填色，颜色自由度足够看出“配对”，又不会多到需要从 2-3 个
#     样例里猜测复杂查表。
#   - 保持原题的装饰墙机制，但把可见装饰墙限制在边框深处，避免装饰墙紧贴边界导致“为什么
#     这个边界不是跟它变色”的视觉噪声。
#   implicit_assumptions_in_original_main:
#   - `wall_col = most frequent non-bg non-marker color` 依赖“墙只有一种颜色”。变体下若仍这样做，
#     另一种墙色会被错误当成装饰或忽略，故必须改成“所有非背景、非 marker 的颜色都参与
#     结构墙检测”。
#   - `fill_col = 7` 依赖“所有边界颜色恒定”。变体下必须改成逐像素决策，否则红墙内侧仍会被
#     错涂成橙色。
#   - 原题对边界像素的判定与上色耦合很弱，改动应只落在“上色”步骤；可达性、结构墙判别、
#     装饰墙清除这三部分可以完全复用。
#   placement_legality:
#   - 结构墙仍由触边的 4 连通形状组成，但不同组件之间增加 8 邻域间隔，避免不同颜色的墙直
#     接贴在一起而让某些边界像素同时“挨着”两种墙色。
#   - 生成器还会拒绝任何“最近结构墙颜色出现平局”的边界布局；solver 保留一个确定性平局规
#     则以防外部输入，但训练样例不会依赖这个模糊分支。
#   visual_observability:
#   - 每个样例都要求两种结构墙颜色都产生非零边界段，所以解题者能直接看见“蓝墙那边是橙，
#     红墙那边是绿”，而不是跨样例拼凑。
#   neatness:
#   - 可见装饰墙远离输出边框；隐藏装饰墙不接触边界；结构墙之间不相邻。这样颜色归因只落在
#     真正的结构墙上，画面层次仍和原题一致。
#   color_inference:
#   - 颜色映射是固定的、低基数的，并且总是伴随几何线索出现：边框颜色跟最近结构墙颜色绑定，
#     而不是按不可见编号排序或随机变化。
#   new_main_assumption_audit:
#   - `main()` 新增的关键假设是：每个需要涂色的边界像素都能归属到唯一最近的结构墙颜色。
#     为了让规则可学习，`generate_input()` 明确筛掉最近颜色平局的案例。
#   - `main()` 仍允许输入里存在两种墙色的装饰块，因为第二次 flood 会把非结构墙像素视为可穿
#     过；生成器通过放置约束保证这些装饰不会破坏结构墙识别。


BG = 8
WALL_BLUE = 1
WALL_RED = 2
WALL_COLORS = (WALL_BLUE, WALL_RED)
MARKER = 6
FILL_BY_WALL = {
    WALL_BLUE: 7,  # orange
    WALL_RED: 3,   # green
}
FILL_COLORS = tuple(sorted(set(FILL_BY_WALL.values())))

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

NUM_STRUCTS_RANGE = (2, 4)
STRUCT_SHAPE_RETRIES = 12
RECT_MAX_DIM_DIV = 3
TRI_MAX_SIZE = 6
TRI_MIN_DIM_DIV = 3

REACH_RATIO_RANGE = (0.38, 0.87)

NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

MIN_FILL_PIXELS_PER_COLOR = 3
GEN_ATTEMPTS = 1200


def _flood(start: tuple[int, int], passable, h: int, w: int) -> set[tuple[int, int]]:
    q = deque([start])
    vis = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and passable(nr, nc):
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


def _touches_border(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = _dilate4(mask)
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _boundary_mask(reach: np.ndarray, structural: np.ndarray) -> np.ndarray:
    candidate = reach.copy()
    edge = np.zeros_like(candidate)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    near_struct = _dilate8(structural > 0)
    near_unreach = _dilate4(~reach)
    return candidate & (edge | near_struct | near_unreach)


def _nearest_wall_color(
    r: int,
    c: int,
    wall_pixels: list[tuple[int, int, int]],
    structural: np.ndarray,
) -> tuple[int, bool]:
    adjacent_colors = set()
    for nr, nc in neighbors8(r, c):
        if 0 <= nr < structural.shape[0] and 0 <= nc < structural.shape[1]:
            col = int(structural[nr, nc])
            if col:
                adjacent_colors.add(col)
    if len(adjacent_colors) == 1:
        return next(iter(adjacent_colors)), False
    if len(adjacent_colors) > 1:
        return min(adjacent_colors), True

    best_dist = None
    best_colors: set[int] = set()
    for wr, wc, wall_col in wall_pixels:
        dist = max(abs(r - wr), abs(c - wc))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_colors = {wall_col}
        elif dist == best_dist:
            best_colors.add(wall_col)

    assert best_dist is not None, "no structural wall pixels found"
    return min(best_colors), len(best_colors) > 1


def _compute_boundary_fill(structural: np.ndarray, reach: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    boundary = _boundary_mask(reach, structural)
    fills = np.zeros(structural.shape, dtype=int)
    wall_pixels = [(int(r), int(c), int(structural[r, c])) for r, c in np.argwhere(structural > 0)]
    ambiguous = 0
    for r, c in np.argwhere(boundary):
        wall_col, is_ambiguous = _nearest_wall_color(int(r), int(c), wall_pixels, structural)
        fills[r, c] = FILL_BY_WALL[wall_col]
        ambiguous += int(is_ambiguous)
    return boundary, fills, ambiguous


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the marker-side region and color its inner rim from the nearest structural wall.

    Input structure:
    - The most common color is background, and one unique corner pixel is the start marker.
    - Barrier pieces use a wall palette rather than a single wall color: blue and red may both appear.
    - Border-touching wall components are true structure; interior wall-colored fragments are decorations.

    Rule:
    - Flood from the marker corner while treating every wall-colored cell as blocked to locate the marker side.
    - Re-identify only border-touching, reach-adjacent wall components as structural, then erase unreachable wall-colored decorations.
    - Paint each reachable boundary background cell with the fill color paired to its nearest structural wall:
      blue walls produce orange, and red walls produce green.

    Key insight: The region geometry is solved exactly as in the base task; the new reasoning step is to inherit
    boundary color from structural wall color instead of using one global paint color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    counts = Counter(int(v) for v in g.ravel())
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    marker = None
    for r, c in corners:
        v = int(g[r, c])
        if v != bg and counts.get(v, 0) == 1:
            marker = (r, c, v)
            break
    assert marker is not None, "no unique single-pixel corner marker found"

    mr, mc, marker_col = marker
    wall_cols = [c for c in counts.keys() if c not in (bg, marker_col)]
    wall_col_set = set(wall_cols)
    assert wall_cols, "no wall-colored cells found"
    assert all(c in FILL_BY_WALL for c in wall_cols), "unexpected wall color in input"

    reach0 = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col),
        h,
        w,
    )

    structural = np.zeros((h, w), dtype=int)
    for wall_col in wall_cols:
        for comp_arr in components_mask(g == wall_col, connectivity=4):
            comp = [(int(r), int(c)) for r, c in comp_arr]
            if not _touches_border(comp, h, w):
                continue
            if any(any((nr, nc) in reach0 for nr, nc in neighbors4(r, c)) for r, c in comp):
                for r, c in comp:
                    structural[r, c] = wall_col

    reach = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) in wall_col_set and structural[r, c] == 0),
        h,
        w,
    )

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) in wall_col_set and structural[r, c] == 0 and (r, c) not in reach:
                out[r, c] = bg

    reach_mask = np.zeros((h, w), dtype=bool)
    for r, c in reach:
        reach_mask[r, c] = True

    boundary, fills, _ = _compute_boundary_fill(structural, reach_mask)
    for r, c in np.argwhere(boundary):
        if int(out[r, c]) == bg:
            out[r, c] = fills[r, c]

    return out


def _shape_vertical_line(h, w, rng):
    c = rng.randint(2, w - 2)
    mode = rng.choice(["top", "bottom", "full"])
    if mode == "full":
        r0, r1 = 0, h - 1
    elif mode == "top":
        r0, r1 = 0, rng.randint(max(1, h // 3), h - 2)
    else:
        r0, r1 = rng.randint(1, h - max(2, h // 3)), h - 1
    return [(r, c) for r in range(r0, r1 + 1)]


def _shape_horizontal_line(h, w, rng):
    r = rng.randint(1, h - 2)
    if rng.choice(["left", "right"]) == "left":
        c0, c1 = 0, rng.randint(max(1, w // 3), w - 2)
    else:
        c0, c1 = rng.randint(1, w - max(2, w // 3)), w - 1
    return [(r, c) for c in range(c0, c1 + 1)]


def _shape_rectangle(h, w, rng):
    rh = rng.randint(2, max(2, h // RECT_MAX_DIM_DIV))
    rw = rng.randint(2, max(2, w // RECT_MAX_DIM_DIV))
    anchor = rng.choice(["top", "right", "tr", "top_mid", "right_mid", "bottom_right"])
    if anchor == "top":
        r0, c0 = 0, rng.randint(1, max(1, w - rw - 1))
    elif anchor == "right":
        r0, c0 = rng.randint(1, max(1, h - rh - 1)), w - rw
    elif anchor == "tr":
        r0, c0 = 0, w - rw
    elif anchor == "top_mid":
        r0, c0 = 0, rng.randint(2, max(2, w - rw - 2))
    elif anchor == "right_mid":
        r0, c0 = rng.randint(2, max(2, h - rh - 2)), w - rw
    else:
        r0, c0 = h - rh, w - rw
    return [(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)]


def _shape_left_diamond(h, w, rng):
    max_size = min(TRI_MAX_SIZE, h // TRI_MIN_DIM_DIV, w // TRI_MIN_DIM_DIV)
    if max_size < 3:
        return None
    size = rng.randint(3, max_size)
    r_center = rng.randint(size, h - size - 1)
    top = r_center - (size - 1)
    cells = []
    for i in range(size):
        cells.extend((top + i, c) for c in range(i + 1))
    for i in range(size - 1):
        cells.extend((top + size + i, c) for c in range(size - 1 - i))
    return cells


def _shape_top_triangle(h, w, rng):
    max_size = min(TRI_MAX_SIZE, h // TRI_MIN_DIM_DIV, w // TRI_MIN_DIM_DIV)
    if max_size < 2:
        return None
    size = rng.randint(2, max_size)
    c_start = rng.randint(1, max(1, w - 2 * size))
    cells = []
    for i in range(size):
        cells.extend((i, c) for c in range(c_start, c_start + size - i))
    return cells


def _shape_right_triangle(h, w, rng):
    max_size = min(TRI_MAX_SIZE, h // 2, w // 2)
    if max_size < 2:
        return None
    size = rng.randint(2, max_size)
    r_start = rng.randint(0, max(0, h - size - 1))
    cells = []
    for i in range(size):
        length = i + 1
        cells.extend((r_start + i, c) for c in range(w - length, w))
    return cells


SHAPE_FACTORIES = [
    _shape_vertical_line,
    _shape_horizontal_line,
    _shape_rectangle,
    _shape_left_diamond,
    _shape_top_triangle,
    _shape_right_triangle,
]


def _flood_mask(start, blocked, h, w):
    reach = np.zeros((h, w), dtype=bool)
    reach[start] = True
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reach[nr, nc] and not blocked[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def _cells_have_clearance(occupied: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if occupied[r, c]:
            return False
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                if occupied[nr, nc]:
                    return False
    return True


def _place_structural_walls(h, w, marker_rc, rng):
    num = rng.randint(*NUM_STRUCTS_RANGE)
    planned_colors = [WALL_BLUE, WALL_RED]
    planned_colors.extend(rng.choice(WALL_COLORS) for _ in range(num - 2))
    rng.shuffle(planned_colors)

    structural = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for wall_col in planned_colors:
        placed = False
        for _ in range(STRUCT_SHAPE_RETRIES):
            cells = rng.choice(SHAPE_FACTORIES)(h, w, rng)
            if not cells:
                continue
            if marker_rc in cells or not _cells_have_clearance(occupied, cells):
                continue
            for r, c in cells:
                structural[r, c] = wall_col
                occupied[r, c] = True
            placed = True
            break
        if not placed:
            return None

    if not all((structural == wall_col).any() for wall_col in WALL_COLORS):
        return None
    return structural


def _validate_structural(structural, reach, h, w):
    reach_nbr = _dilate4(reach)
    for wall_col in WALL_COLORS:
        for comp_arr in components_mask(structural == wall_col, connectivity=4):
            comp = [(int(r), int(c)) for r, c in comp_arr]
            if not _touches_border(comp, h, w):
                return False
            if not any(reach_nbr[r, c] for r, c in comp):
                return False
    return True


def _paint_output(structural, reach, marker_rc):
    out = np.full(structural.shape, BG, dtype=int)
    out[structural > 0] = structural[structural > 0]
    out[marker_rc] = MARKER

    boundary, fills, ambiguous = _compute_boundary_fill(structural, reach)
    paintable = boundary & (out == BG)
    out[paintable] = fills[paintable]
    return out, ambiguous


def _place_colored_decor(allowed_mask, count, rng):
    h, w = allowed_mask.shape
    placed = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    if count == 0:
        return placed
    for _ in range(count):
        area = rng.randint(*DECOR_AREA_RANGE)
        shapes = [(1, area), (area, 1)]
        if area >= 3:
            shapes.append((2, 2))
        rng.shuffle(shapes)
        for ph, pw in shapes:
            if ph > h or pw > w:
                continue
            patch = np.ones((ph, pw), dtype=int)
            canvas = np.where(allowed_mask & ~occupied, 0, 9)
            try:
                r0, c0 = random_free_location(
                    canvas,
                    patch,
                    transparent=0,
                    border=DECOR_BORDER,
                    padding=DECOR_PADDING,
                    padding_connectivity=8,
                )
            except ValueError:
                continue
            wall_col = rng.choice(WALL_COLORS)
            placed[r0:r0 + ph, c0:c0 + pw] = wall_col
            occupied[r0:r0 + ph, c0:c0 + pw] = True
            break
        else:
            return None
    return placed


def _gen_attempt():
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    structural = _place_structural_walls(h, w, marker_rc, rng)
    if structural is None or structural[marker_rc] != 0:
        return None

    reach = _flood_mask(marker_rc, structural > 0, h, w)
    reach_frac = reach.mean()
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None
    if not _validate_structural(structural, reach, h, w):
        return None

    out, ambiguous = _paint_output(structural, reach, marker_rc)
    if ambiguous:
        return None

    fill_mask = np.isin(out, FILL_COLORS)
    if not fill_mask.any() or len(components_mask(fill_mask, connectivity=4)) != 1:
        return None

    fill_counts = {fill_col: int(np.sum(out == fill_col)) for fill_col in FILL_COLORS}
    if any(fill_counts[fill_col] < MIN_FILL_PIXELS_PER_COLOR for fill_col in FILL_COLORS):
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = reach & (out == BG)
    visible_allowed &= ~_dilate8(fill_mask)
    visible_allowed &= ~_dilate8(structural > 0)

    hidden_allowed = (~reach) & (structural == 0)
    hidden_allowed[0, :] = hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = hidden_allowed[:, -1] = False

    visible_decor = _place_colored_decor(visible_allowed, n_visible, rng)
    if visible_decor is None:
        visible_decor = np.zeros((h, w), dtype=int)
    hidden_decor = _place_colored_decor(hidden_allowed, n_hidden, rng)
    if hidden_decor is None:
        hidden_decor = np.zeros((h, w), dtype=int)

    out_final = out.copy()
    out_final[visible_decor > 0] = visible_decor[visible_decor > 0]

    inp = out_final.copy()
    inp[np.isin(inp, FILL_COLORS)] = BG
    inp[hidden_decor > 0] = hidden_decor[hidden_decor > 0]

    if not np.array_equal(main(inp), out_final):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D14 variant")
