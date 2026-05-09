# Combo variant of 136b0064 (index 1)
# combo_spec: D1/2+D2/1
# Name: 面积排序与反转映射双重推理
#
# Design analysis
# ===============
# combo_variant:
#   preserved_structure:
#     - 输入仍是 Hx15，H = 4*N - 1，N in 2..5；第 7 列是整高分隔线。
#     - 左侧仍有两列 3x3 字形槽位，右侧仍只有一个灰色起点。
#     - 四种字形模板与颜色绑定保持不变：area 4/5/6/7 仍对应固定形状与颜色。
#   fused_rule:
#     - D1/2: 不再按列主序读取字形，而是按像素面积升序处理：先所有 area 4，再 area 5，再 area 6，再 area 7。
#     - D2/1: 面积到折线段的映射反转为：
#         area 4 -> 向左 4 格
#         area 5 -> 向右 3 格
#         area 6 -> 向左 2 格
#         area 7 -> 向下 2 格
#     - 面积同时控制“什么时候执行”和“执行什么动作”，形成双重推理链。
#   behavior_trace:
#     - 例：N=2，左侧四个槽位按列主序摆放的面积是 [7, 5, 4, 6]，灰色起点移回左画布后在列 4。
#     - 原题会直接按 [7, 5, 4, 6] 处理，且 area 4 仍是竖段，得到的折线次序受空间位置支配。
#     - 本融合变体先把四个字形按面积重排成 [4, 5, 6, 7]，再用反转后的映射执行：
#         area 4: 先下 1 再向左画满 4 格，终点从 (0,4) 到 (1,1)
#         area 5: 再下 1 后向右 3 格，终点到 (2,3)
#         area 6: 再下 1 后向左 2 格，终点到 (3,2)
#         area 7: 最后下 2 格，终点到 (5,2)
#     - 差异不只是一张查表变化：同一组字形因为先排序再解码，输出中的颜色先后与折线走向都会一起改变。
#   complexity_calibration:
#     - 仍只有 4 种离散面积档位，没有引入颜色重映射或隐藏状态。
#     - 生成器强制四种面积都出现，并要求输入槽位中的列主序面积序列不是升序，避免样例退化成“看起来还是按位置读”。
#     - 同时过滤掉“只改顺序但轨迹没变”或“只改映射但轨迹没变”的情况，保证两个维度都被样例激活。
#   implicit_assumption_review:
#     - `find_full_height_constant_col` 仍依赖唯一整高非背景列；融合规则没有破坏这点。
#     - 右侧只有一个灰点、去掉分隔线后平移回左画布，这一几何前提仍成立。
#     - `find_components(..., monochromatic=True)` 仍依赖每个字形是单色 3x3、8 连通组件；生成器继续严格按模板放置。
#     - 原 solver 的 `glyphs.sort(key=(c0,r0))` 隐含“空间位置就是指令顺序”；这里必须改为按 area 排序。
#     - 原 solver 的 `if area == 4` 特判隐含“只有 area 4 是竖线”；融合后竖线变成 area 7，因此必须统一改成查表。
#     - 本实现对同面积 tie 使用 `(c0, r0)` 仅作稳定排序；因为同面积字形颜色和动作完全相同，不会重新引入位置语义。
#   placement_legality:
#     - 生成器继续使用 output-first/path-first 思路，但实际可控变量变成四种面积的计数，而不是任意顺序。
#     - 先枚举一个在新映射下能完全留在 7 列左画布内的升序面积多重集，再把这些面积随机打乱写回槽位。
#     - 每一步都检查终点行列不越界；3x3 槽位固定且互不重叠，因此输入对象不会相撞。
#   observability_and_cleanliness:
#     - 输出仍是一条单调向下的折线，每段开始前必先下移一格；不同颜色段最多在端点相接，不会产生大片覆盖噪声。
#     - 由于面积 4/5/6 都是水平段而 area 7 是末尾竖段，观察者能从多个样例中看出“最小面积先走且最长、最大面积最后且最短”的逆向关系。
#   fusion_metric:
#     - 用两个二值指标审查样例是否退化：
#         order_effect = combo_trace != same_mapping_but_column_major_trace
#         map_effect = combo_trace != same_order_but_original_mapping_trace
#     - 仅保留 `order_effect` 和 `map_effect` 都为真的输入，确保融合行为不是名义上的。

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper (only used by this solver): find a column where all values are identical and != bg.
    Returns (col_idx, color) or (None, None).
    """
    H, W = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(W)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


AREA_TO_STEP = {
    4: {"dr": 0, "dc": -1, "length": 4},  # left 4
    5: {"dr": 0, "dc": 1, "length": 3},   # right 3
    6: {"dr": 0, "dc": -1, "length": 2},  # left 2
    7: {"dr": 1, "dc": 0, "length": 2},   # down 2
}

ORIGINAL_AREA_TO_STEP = {
    4: {"dr": 1, "dc": 0, "length": 2},   # down 2
    5: {"dr": 0, "dc": 1, "length": 3},   # right 3
    6: {"dr": 0, "dc": -1, "length": 4},  # left 4
    7: {"dr": 0, "dc": -1, "length": 2},  # left 2
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the divided glyph board into a colored polyline, using area-ranked execution.

    Input structure:
    - A full-height divider splits the grid into a left glyph panel and a right start panel.
    - The left panel contains monochrome 3x3 glyph components; the right panel contains one gray start cell.

    Rule:
    - Remove the divider, translate the gray start back into left-canvas coordinates, and collect all 3x3 glyphs.
    - Process glyphs by increasing pixel area rather than by position, then convert each area with the inverted map:
      area 4 → left 4, area 5 → right 3, area 6 → left 2, area 7 → down 2.
    - For every glyph, move down one row, paint that entry cell with the glyph color, then extend the rest of the segment.

    Key insight:
    - Pixel area now carries two coupled meanings at once: it decides both the priority of a glyph in the instruction stream
      and the motion it emits, with the smallest shapes producing the earliest and longest segments.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.full((H, int(div_c)), bg, dtype=int)

    gray = np.argwhere(g == 5)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[int(r_cur), int(c_cur)] = 5

    left = g[:, : int(div_c)]
    ignore = {int(bg), 5}
    if div_color is not None:
        ignore.add(int(div_color))

    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in ignore:
            continue
        b = comp["bbox"]
        if bbox_h(b) == 3 and bbox_w(b) == 3:
            r0, c0, _r1, _c1 = map(int, b)
            glyphs.append({"color": color, "r0": int(r0), "c0": int(c0), "area": int(comp["area"])})

    glyphs.sort(key=lambda d: (int(d["area"]), int(d["c0"]), int(d["r0"])))

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        assert area in AREA_TO_STEP, f"unexpected glyph area: {area}"
        step = AREA_TO_STEP[area]

        r_cur = int(r_cur + 1)
        assert r_cur < H, f"segment exits bottom: r_cur={r_cur}, H={H}"
        out[r_cur, int(c_cur)] = color

        dr = int(step["dr"])
        dc = int(step["dc"])
        seg_len = int(step["length"])
        end_r = int(r_cur + dr * (seg_len - 1))
        end_c = int(c_cur + dc * (seg_len - 1))
        assert 0 <= end_r < H, f"segment exits bottom: end_r={end_r}, H={H}"
        assert 0 <= end_c < int(div_c), (
            f"segment exits canvas: c_cur={c_cur}, dc={dc}, seg_len={seg_len}, div_c={div_c}"
        )

        for _ in range(seg_len - 1):
            r_cur = int(r_cur + dr)
            c_cur = int(c_cur + dc)
            out[r_cur, c_cur] = color

    return out


# --------------- generator ---------------

N_MIN, N_MAX = 2, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0

_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_AREAS = (4, 5, 6, 7)

MIN_INVERSION_PAIRS = 2
MAX_PERM_ATTEMPTS = 80
MAX_ATTEMPTS = 1500

_FEASIBLE_SORTED_CACHE: dict[tuple[int, int], list[tuple[int, ...]]] = {}


def _total_delta(area: int, mapping: dict[int, dict[str, int]]) -> tuple[int, int]:
    step = mapping[area]
    dr = int(step["dr"])
    dc = int(step["dc"])
    seg_len = int(step["length"])
    return int(1 + dr * (seg_len - 1)), int(dc * (seg_len - 1))


def _trace_signature(areas: list[int] | tuple[int, ...], start_col: int, mapping: dict[int, dict[str, int]]) -> tuple[tuple[int, int, int], ...]:
    r, c = GRAY_START_ROW, int(start_col)
    trace: list[tuple[int, int, int]] = [(r, c, GRAY)]
    for area in areas:
        step = mapping[int(area)]
        color = _AREA_TO_COLOR[int(area)]
        r += 1
        trace.append((r, c, color))
        for _ in range(int(step["length"]) - 1):
            r += int(step["dr"])
            c += int(step["dc"])
            trace.append((r, c, color))
    return tuple(trace)


def _path_is_valid(areas: list[int] | tuple[int, ...], start_col: int, H: int, mapping: dict[int, dict[str, int]]) -> bool:
    r, c = GRAY_START_ROW, int(start_col)
    if not (0 <= c < LEFT_W):
        return False
    for area in areas:
        dr_total, dc_total = _total_delta(int(area), mapping)
        r += int(dr_total)
        c += int(dc_total)
        if not (0 <= c < LEFT_W):
            return False
        if not (0 <= r < H):
            return False
    return True


def _positive_compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(int(total),)]
    out: list[tuple[int, ...]] = []
    for first in range(1, int(total) - parts + 2):
        for rest in _positive_compositions(int(total) - int(first), parts - 1):
            out.append((int(first), *rest))
    return out


def _feasible_sorted_areas(N: int, start_col: int) -> list[tuple[int, ...]]:
    key = (int(N), int(start_col))
    if key in _FEASIBLE_SORTED_CACHE:
        return _FEASIBLE_SORTED_CACHE[key]

    total = 2 * int(N)
    H = 4 * int(N) - 1
    feasible: list[tuple[int, ...]] = []
    for counts in _positive_compositions(total, len(_AREAS)):
        areas: list[int] = []
        for area, count in zip(_AREAS, counts):
            areas.extend([int(area)] * int(count))
        area_tuple = tuple(areas)
        if _path_is_valid(area_tuple, start_col, H, AREA_TO_STEP):
            feasible.append(area_tuple)

    _FEASIBLE_SORTED_CACHE[key] = feasible
    return feasible


def _count_inversion_pairs(seq: list[int] | tuple[int, ...]) -> int:
    vals = list(map(int, seq))
    inv = 0
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if vals[i] > vals[j]:
                inv += 1
    return int(inv)


def _sample_slot_areas(sorted_areas: tuple[int, ...], start_col: int) -> list[int] | None:
    base = list(map(int, sorted_areas))
    combo_trace = _trace_signature(base, start_col, AREA_TO_STEP)

    for _ in range(MAX_PERM_ATTEMPTS):
        slot_areas = base.copy()
        random.shuffle(slot_areas)
        if slot_areas == base:
            continue
        if slot_areas[0] == 4 and slot_areas[-1] == 7:
            continue
        if _count_inversion_pairs(slot_areas) < MIN_INVERSION_PAIRS:
            continue

        d2_only_trace = _trace_signature(slot_areas, start_col, AREA_TO_STEP)
        d1_only_trace = _trace_signature(base, start_col, ORIGINAL_AREA_TO_STEP)
        if combo_trace == d2_only_trace:
            continue
        if combo_trace == d1_only_trace:
            continue
        return slot_areas
    return None


def _render_grid(N: int, start_col: int, slot_areas: list[int]) -> np.ndarray:
    H = 4 * int(N) - 1
    grid = np.zeros((H, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + int(start_col)] = GRAY

    for i, area in enumerate(slot_areas):
        tile_col = TILE_COLS[i // int(N)]
        r0 = TILE_ROW_STRIDE * (i % int(N))
        grid[r0:r0 + 3, tile_col:tile_col + 3] = _TEMPLATES[int(area)] * _AREA_TO_COLOR[int(area)]
    return grid


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(N_MIN, N_MAX)
    H = 4 * int(N) - 1
    start_col = random.randint(0, LEFT_W - 1)

    feasible = _feasible_sorted_areas(N, start_col)
    if not feasible:
        return None
    sorted_areas = random.choice(feasible)
    assert len(set(sorted_areas)) == len(_AREAS)
    assert _path_is_valid(sorted_areas, start_col, H, AREA_TO_STEP)

    slot_areas = _sample_slot_areas(sorted_areas, start_col)
    if slot_areas is None:
        return None
    return _render_grid(N, start_col, slot_areas)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            inp = _gen_attempt()
            if inp is None:
                continue
            out = main(inp)
            if out.shape != (inp.shape[0], LEFT_W):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
