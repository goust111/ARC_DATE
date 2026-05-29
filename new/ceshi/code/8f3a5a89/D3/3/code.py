# Variant of 8f3a5a89
# Dimension: D3 — 源对象（标记像素）位置 (src)
# Change: 单个品红像素固定在底左角 (H-1, 0)，始终为BL角... → 标记在网格内部（非边缘）

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# Design analysis
# ----------------
# yaml:
#   variant_goal:
#     keep:
#       - 颜色体系不变: 背景8, 墙1, 标记6, 输出描边7
#       - 输出语义不变: 从标记所在的可达区域中, 找结构墙, 擦掉结构墙另一侧的装饰墙,
#         再在可达区域内侧刷一圈橙色边
#       - 装饰墙仍然可能保留(在可达区内)或被擦除(在可达区外)
#     change:
#       - 品红标记不再固定在左下角
#       - 可达区域不再依赖网格边界闭合, 而是由内部蓝墙完整围住
#   a_concretization_and_trace:
#     input_picture:
#       - 一个完全位于网格内部的“房间”形背景区域, 中间某处有1个品红像素
#       - 房间外沿有1格厚的蓝色闭合墙, 墙不会碰到网格边界
#       - 墙外是大片背景色, 其中可混入一些蓝色装饰墙; 房间内部深处也可放少量蓝色装饰墙
#     output_picture:
#       - 房间内侧沿结构墙刷一圈橙色
#       - 房间内的装饰墙保留
#       - 房间外的装饰墙擦成背景
#     trace_example:
#       - 原题: 标记在BL角, 先从角点洪填, 与边界连通的蓝墙组件被当成结构墙
#       - 变体: 标记在内部, 若还沿用“接触边界的墙才算结构墙”, 则闭合房间墙会被误判为装饰
#       - 新逻辑: 先从标记洪填得到房间内部 reach0; 再从网格边界洪填得到外部 outside0;
#         同时邻接 reach0 和 outside0 的蓝墙组件才是结构墙
#   b_complexity_calibration:
#     - 只改 D3 核心, 不引入新颜色或多级规则
#     - 但墙形从“边缘挂靠片段”改成“内部闭环”, 视觉差异明显
#     - 房间形状允许矩形加1-3个矩形凸出, 保持少量自由度且样例间有重复结构支撑
#   c_implicit_assumption_audit_main:
#     - 原 `for corner in corners` 假设标记一定在角上; 变体下必须改成全局搜唯一单像素颜色
#     - 原 `touches_border` 识别结构墙; 变体下闭合内墙不碰边, 该条件必错
#     - 原画橙色时对 grid edge 直接上色; 变体下可达区不依赖边界, 这条应移除
#     - 原第二次洪填允许穿过非结构墙, 这条仍成立, 因为内部装饰墙仍需视为可穿过
#   d_placement_legality:
#     - 房间区域始终保持在距网格边界至少2格的位置, 这样由其外扩1格得到的结构墙不会碰边
#     - 结构墙由房间区域的4邻外边界直接生成, 天然与房间不重叠且形成闭环
#     - 装饰墙只放在许可掩码内, 并与结构墙保持至少1格间距, 避免粘连导致识别歧义
#     - 标记只放在房间深处的背景格, 不放在边缘或未来会变橙的位置
#   e_visual_observability:
#     - 人类可直接看到: 品红点不在角上, 蓝墙圈在内部闭合, 橙色只沿内部房间边界出现
#     - 若生成器退化回靠边结构, 肉眼会立刻变回原题风格, 因此将“墙不碰边”和“标记不在边缘”
#       作为强约束
#   f_visual_cleanliness:
#     - 可见装饰墙限制在房间深处, 不挤占橙色环
#     - 隐藏装饰墙限制在房间外且不贴结构墙, 避免蓝墙层层叠在一起难辨结构
#   g_color_inference:
#     - 仍只有4种颜色, 且颜色角色稳定: 蓝=墙, 品红=起点, 橙=输出边框, 青=背景
#   h_variant_metric_for_postcheck:
#     - 指标1: 标记是否严格位于非边缘位置
#     - 指标2: 结构墙是否完全不接触网格边界
#     - 指标3: 同一输入交给原 solver 与新 solver, 输出是否发生分歧
#       (若长期一致, 说明没有真正触发 D3 行为)
#   i_new_main_assumptions_and_generator_guarantees:
#     - `main()` 新假设1: 唯一单像素非背景色就是标记; generator 保证只有1个品红像素
#     - 新假设2: 至少存在一种非背景且非标记的主色作为墙色; generator 只使用蓝墙
#     - 新假设3: 结构墙组件同时贴着内部 reach0 与边界外部 outside0; generator 保证房间无洞、
#       结构墙是1格厚闭环、装饰墙不与结构墙粘连
#     - 新假设4: 标记位于结构墙包围的区域内; generator 只从房间深处选标记


def _flood_many(starts: list[tuple[int, int]], passable, h: int, w: int) -> set[tuple[int, int]]:
    q = deque()
    vis: set[tuple[int, int]] = set()
    for r, c in starts:
        if 0 <= r < h and 0 <= c < w and (r, c) not in vis and passable(r, c):
            vis.add((r, c))
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and passable(nr, nc):
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


def _flood(start: tuple[int, int], passable, h: int, w: int) -> set[tuple[int, int]]:
    return _flood_many([start], passable, h, w)


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


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _border_starts(h: int, w: int) -> list[tuple[int, int]]:
    starts = []
    for r in range(h):
        starts.append((r, 0))
        starts.append((r, w - 1))
    for c in range(w):
        starts.append((0, c))
        starts.append((h - 1, c))
    return starts


def _mask_from_coords(coords: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _compute_fill_mask(structural: np.ndarray, reach: np.ndarray) -> np.ndarray:
    candidate = reach.copy()
    near_struct = _dilate8(structural)
    near_unreach = _dilate4(~reach)
    return candidate & (near_struct | near_unreach)


def _room_walls(region: np.ndarray) -> np.ndarray:
    return _dilate8(region) & ~region


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint an inner border around the wall-enclosed region containing an interior marker.

    Input structure:
    - bg is the dominant color; blue cells are walls and magenta is a unique single-pixel start marker.
    - The marker sits strictly inside the grid, inside a wall-enclosed room rather than on a corner.
    - Some blue components form the room boundary; other blue bits are decorations either inside the room or outside it.

    Rule:
    - Flood-fill from the interior marker across background to recover the room's reachable area.
    - Flood-fill from the grid border across background to recover the exterior; blue components touching both sides are the structural walls.
    - Erase non-structural blue decorations on the exterior side, and paint orange on reachable background cells that lie on the room's inner boundary.

    Key insight: In this variant, a true wall component is detected by separating the marker-reachable interior from the border-reachable exterior, not by touching the grid edge.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    counts = Counter(int(v) for v in g.ravel())
    marker_cells = [
        (r, c, int(g[r, c]))
        for r in range(h)
        for c in range(w)
        if int(g[r, c]) != bg and counts.get(int(g[r, c]), 0) == 1
    ]
    assert len(marker_cells) == 1, "expected exactly one unique single-pixel marker"
    mr, mc, marker_col = marker_cells[0]
    assert 0 < mr < h - 1 and 0 < mc < w - 1, "marker must be interior in this variant"

    other_cols = [col for col in counts if col not in (bg, marker_col)]
    assert other_cols, "no wall color found"
    wall_col = max(other_cols, key=lambda col: counts[col])

    passable_bg = lambda r, c: int(g[r, c]) in (bg, marker_col)
    reach0 = _flood((mr, mc), passable_bg, h, w)
    outside0 = _flood_many(_border_starts(h, w), passable_bg, h, w)

    walls: set[tuple[int, int]] = set()
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = [(int(r), int(c)) for r, c in comp_arr]
        touches_inside = any(
            0 <= nr < h and 0 <= nc < w and (nr, nc) in reach0
            for r, c in comp
            for nr, nc in neighbors4(r, c)
        )
        touches_outside = any(
            0 <= nr < h and 0 <= nc < w and (nr, nc) in outside0
            for r, c in comp
            for nr, nc in neighbors4(r, c)
        )
        if touches_inside and touches_outside:
            walls |= set(comp)

    reach = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and (r, c) not in walls),
        h,
        w,
    )

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = bg

    reach_mask = _mask_from_coords(reach, h, w)
    structural_mask = _mask_from_coords(walls, h, w)
    fill_mask = _compute_fill_mask(structural_mask, reach_mask)
    out[fill_mask & (out == bg)] = 7
    return out


BG = 8
WALL = 1
MARKER = 6
FILL = 7

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)
REACH_RATIO_RANGE = (0.12, 0.45)
NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1
GEN_ATTEMPTS = 400


def _flood_mask(start: tuple[int, int], blocked: np.ndarray, h: int, w: int) -> np.ndarray:
    reach = np.zeros((h, w), dtype=bool)
    if blocked[start]:
        return reach
    reach[start] = True
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not reach[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def _flood_mask_many(starts: list[tuple[int, int]], blocked: np.ndarray, h: int, w: int) -> np.ndarray:
    reach = np.zeros((h, w), dtype=bool)
    q = deque()
    for r, c in starts:
        if 0 <= r < h and 0 <= c < w and not blocked[r, c] and not reach[r, c]:
            reach[r, c] = True
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not reach[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def _room_mask(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    region = np.zeros((h, w), dtype=bool)

    max_rh = min(8, h - 4)
    max_rw = min(8, w - 4)
    if max_rh < 4 or max_rw < 4:
        return None

    rh = rng.randint(4, max_rh)
    rw = rng.randint(4, max_rw)
    r0 = rng.randint(2, h - rh - 2)
    c0 = rng.randint(2, w - rw - 2)
    region[r0 : r0 + rh, c0 : c0 + rw] = True

    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    for side in sides[: rng.randint(0, 3)]:
        if side == "top":
            max_depth = r0 - 2
            if max_depth < 1:
                continue
            ah = rng.randint(1, min(3, max_depth))
            aw = rng.randint(2, rw)
            ac0 = rng.randint(c0, c0 + rw - aw)
            region[r0 - ah : r0, ac0 : ac0 + aw] = True
        elif side == "bottom":
            max_depth = h - (r0 + rh) - 2
            if max_depth < 1:
                continue
            ah = rng.randint(1, min(3, max_depth))
            aw = rng.randint(2, rw)
            ac0 = rng.randint(c0, c0 + rw - aw)
            region[r0 + rh : r0 + rh + ah, ac0 : ac0 + aw] = True
        elif side == "left":
            max_depth = c0 - 2
            if max_depth < 1:
                continue
            aw = rng.randint(1, min(3, max_depth))
            ah = rng.randint(2, rh)
            ar0 = rng.randint(r0, r0 + rh - ah)
            region[ar0 : ar0 + ah, c0 - aw : c0] = True
        else:
            max_depth = w - (c0 + rw) - 2
            if max_depth < 1:
                continue
            aw = rng.randint(1, min(3, max_depth))
            ah = rng.randint(2, rh)
            ar0 = rng.randint(r0, r0 + rh - ah)
            region[ar0 : ar0 + ah, c0 + rw : c0 + rw + aw] = True

    if len(components_mask(region, connectivity=4)) != 1:
        return None
    return region


def _place_decor(allowed_mask: np.ndarray, count: int, rng: random.Random) -> np.ndarray | None:
    h, w = allowed_mask.shape
    placed = np.zeros((h, w), dtype=bool)
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
            canvas = np.where(allowed_mask & ~placed, 0, 9)
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
            placed[r0 : r0 + ph, c0 : c0 + pw] = True
            break
        else:
            return None
    return placed


def _paint_output(structural: np.ndarray, region: np.ndarray, marker_rc: tuple[int, int]) -> np.ndarray:
    out = np.full(region.shape, BG, dtype=int)
    out[structural] = WALL
    out[marker_rc] = MARKER
    fill_mask = _compute_fill_mask(structural, region)
    out[fill_mask & (out == BG)] = FILL
    return out


def _validate_structural(structural: np.ndarray, region: np.ndarray) -> bool:
    h, w = structural.shape
    if _touches_border(structural):
        return False
    if len(components_mask(structural, connectivity=4)) != 1:
        return False
    outside = _flood_mask_many(_border_starts(h, w), structural, h, w)
    return bool((outside & region).sum() == 0 and outside[0, 0])


def _gen_attempt() -> np.ndarray | None:
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)

    region = _room_mask(h, w, rng)
    if region is None:
        return None

    structural = _room_walls(region)
    if not _validate_structural(structural, region):
        return None

    reach_frac = float(region.mean())
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    fill_mask = _compute_fill_mask(structural, region)
    core = region & ~fill_mask
    if core.sum() < 2:
        return None

    marker_choices = np.argwhere(core)
    mr, mc = map(int, marker_choices[rng.randrange(len(marker_choices))])
    marker_rc = (mr, mc)
    if mr in (0, h - 1) or mc in (0, w - 1):
        return None

    out = _paint_output(structural, region, marker_rc)
    if np.count_nonzero(out == FILL) < 8:
        return None
    if np.count_nonzero(out == BG) < 6:
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = core.copy()
    visible_allowed[marker_rc] = False
    visible_allowed &= ~_dilate8(structural)

    outside_bg = ~(region | structural)
    hidden_allowed = outside_bg & ~_dilate8(structural)
    hidden_allowed[0, :] = False
    hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = False
    hidden_allowed[:, -1] = False

    visible_placed = _place_decor(visible_allowed, n_visible, rng)
    if visible_placed is None:
        return None
    hidden_placed = _place_decor(hidden_allowed, n_hidden, rng)
    if hidden_placed is None:
        return None

    out_final = out.copy()
    out_final[visible_placed] = WALL

    inp = out_final.copy()
    inp[inp == FILL] = BG
    inp[hidden_placed] = WALL

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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D3 variant")
