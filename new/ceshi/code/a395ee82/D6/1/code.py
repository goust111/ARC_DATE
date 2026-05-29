# Variant of a395ee82
# Dimension: D6 — 晶格间距与缩放关系 (meta)
# Change: 晶格间距gunit=2(固定)，stamp间距=晶格偏移×模板尺寸(h,w)——晶格结构与stamp阵列之间存在尺寸缩放... → 各方向间距不等

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the D6 variant
# ----------------------------------
# design_analysis:
#   concretization:
#     - 输入仍然由一个大号单色模板 T 和一组单例像素组成；单例中唯一的 T 色像素仍是原点，其余单例仍为 O 色。
#     - 变化只发生在单例晶格的采样间距：竖直方向按 ROW_GUNIT 个像素取样，水平方向按 COL_GUNIT 个像素取样，且两者不相等。
#     - 输出仍是在模板 bbox 坐标系里按晶格位移复制模板；区别在于必须把输入里“行方向间距”和“列方向间距”分别还原成晶格单位。
#     - 小例子：若输入单例相对原点的像素偏移是 (-2, 0), (0, 3), (2, 3)，原题会错误地用 gcd=1 或 2/3 的单一单位解释；本变体应解读为
#       晶格位移 (-1, 0), (0, 1), (1, 1)，随后在输出中分别平移 (-h, 0), (0, w), (h, w)。
#   complexity_calibration:
#     - 只引入一个新的自由度对：ROW_GUNIT=2、COL_GUNIT=3。它足够显眼，不会让任务退化为原题，也不会引入过多分支。
#     - 模板、颜色数、单例数量、晶格连通性都保持原样，避免对象识别难度无端上升。
#   implicit_assumption_review:
#     - 原代码 `gcd_vals = [abs(x) for dr, dc in deltas for x in (dr, dc) if x != 0]` 默认行列共享一个单位；这里必须拆成两个轴分别求 gcd。
#     - 原代码 `deltas = [(dr // gunit, dc // gunit, ...)]` 默认两个方向都能被同一个单位整除；变体中需要 `(dr // row_unit, dc // col_unit)`。
#     - 原生成器里所有 `GUNIT` 参与的边界、bbox、冲突检测都隐含“横纵相等”；现在这些式子都要按行列分别替换。
#   placement_legality:
#     - 模板 stamp 的输出放置逻辑不变，仍需保证所有 stamp 在 22x22 内且互不重叠。
#     - 单例 cluster 的输入放置改为使用 `(ROW_GUNIT, COL_GUNIT)` 计算 bbox 与具体坐标，仍保持与模板 bbox 至少隔 1 个背景格。
#   visual_observability:
#     - 人眼能直接看到单例点阵的行间距和列间距不同，而输出里的模板阵列仍按模板高/宽为步长排列，因此“分别推断两个轴的缩放”是可见的。
#     - 晶格 polyomino 仍跨越 3-4 个单位行列，并且原点严格位于内部，所以每个样例里通常都能看到多个竖直步长和多个水平步长。
#   cleanliness:
#     - 只保留三种颜色和两类对象；单例与模板 bbox 保持分离，避免输入杂乱。
#     - 输出仍是规则的模板阵列，不引入新的叠加层次。
#   color_inference:
#     - 颜色规则完全沿用原题：T 表示模板本色，O 表示原点副本和重叠色。颜色不承担排序或数值编码作用。
#   new_solver_assumptions:
#     - `main()` 现在要求单例布局在行轴和列轴上都出现至少一个非零偏移；生成器通过“spanning + strictly interior origin”保证这一点。
#     - `main()` 仍假设最大连通块就是模板，生成器继续保证单例 cluster 与模板分离且单例互不 4-连通。


def _axis_unit(offsets: list[int]) -> int:
    vals = [abs(int(x)) for x in offsets if int(x) != 0]
    assert vals
    return max(1, int(math.gcd(*vals)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a template array from a singleton-marked rectangular lattice.

    Input structure:
    - The background is uniform. One large monochrome component is the template, colored T.
    - Elsewhere there is a set of isolated pixels arranged on a coarse lattice.
    - Exactly one singleton uses color T and marks the lattice origin; all remaining singletons use color O.
    - The lattice is anisotropic: row spacing and column spacing are both constant, but they are not equal.

    Rule:
    - Read each singleton's displacement from the T-colored origin.
    - Infer the row lattice unit from row offsets alone, and the column lattice unit from column offsets alone.
    - Convert each singleton displacement into lattice steps, then stamp the template at offsets scaled by the template bbox height and width.
    - Paint the origin copy in color O, and also repaint any overlapped pixels in color O.

    Key insight:
    - The singleton layout is not measured with one shared pixel unit anymore. The solver must decode vertical and horizontal spacing separately before translating the template copies.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps

    template = max(comps, key=lambda d: int(d["area"]))
    templ_coords = template["coords"]
    templ_col = int(g[int(templ_coords[0, 0]), int(templ_coords[0, 1])])
    r0, c0, r1, c1 = map(int, template["bbox"])
    h = int(bbox_h((r0, c0, r1, c1)))
    w = int(bbox_w((r0, c0, r1, c1)))
    templ_rel = [(int(r) - r0, int(c) - c0) for r, c in templ_coords]

    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if comp is template:
            continue
        if int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        singletons.append((rr, cc, int(g[rr, cc])))

    origin = [(r, c) for (r, c, col) in singletons if int(col) == int(templ_col)]
    assert len(origin) == 1
    orr, orc = origin[0]

    other_cols = [int(col) for (_r, _c, col) in singletons if int(col) != int(templ_col)]
    assert other_cols
    overlap_col = int(Counter(other_cols).most_common(1)[0][0])

    raw_deltas = [(int(r - orr), int(c - orc), int(col)) for (r, c, col) in singletons]
    row_unit = _axis_unit([dr for dr, _dc, _col in raw_deltas])
    col_unit = _axis_unit([dc for _dr, dc, _col in raw_deltas])
    deltas = [(int(dr // row_unit), int(dc // col_unit), int(col)) for dr, dc, col in raw_deltas]

    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)

    for dr, dc, col in deltas:
        off_r = int(dr * h)
        off_c = int(dc * w)
        is_origin = int(col) == int(templ_col)
        for rr, cc in templ_rel:
            R = int(r0 + off_r + rr)
            C = int(c0 + off_c + cc)
            if 0 <= R < H and 0 <= C < W:
                total[R, C] += 1
                if is_origin:
                    center[R, C] += 1

    out = np.full((H, W), int(bg), dtype=int)
    out[total > 0] = int(templ_col)
    out[(total == 1) & (center > 0)] = int(overlap_col)
    out[total >= 2] = int(overlap_col)
    return out


# --------------- generator (D6) ---------------

GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
ROW_GUNIT = 2
COL_GUNIT = 3

SPANNING_RETRIES = 40
TEMPLATE_RETRIES = 60
SINGLETON_RETRIES = 50
GENERATE_RETRIES = 300


def _random_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    start = (random.randint(0, rows - 1), random.randint(0, cols - 1))
    cells = {start}
    frontier = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            frontier.append((nr, nc))
    while len(cells) < n and frontier:
        cell = random.choice(frontier)
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in cells:
                frontier.append((nr, nc))
    return list(cells) if len(cells) == n else None


def _spanning_polyomino(n: int, rows: int, cols: int) -> list[tuple[int, int]] | None:
    for _ in range(SPANNING_RETRIES):
        cells = _random_polyomino(n, rows, cols)
        if cells is None:
            continue
        rs = [r for r, _c in cells]
        cs = [c for _r, c in cells]
        if min(rs) == 0 and max(rs) == rows - 1 and min(cs) == 0 and max(cs) == cols - 1:
            return cells
    return None


def _gen_template(h: int, w: int) -> list[tuple[int, int]] | None:
    total = h * w
    lo = max(TEMPLATE_AREA_RANGE[0], int(np.ceil(total * TEMPLATE_FILL_RANGE[0])))
    hi = min(TEMPLATE_AREA_RANGE[1], int(np.floor(total * TEMPLATE_FILL_RANGE[1])))
    if lo > hi:
        return None
    for _ in range(TEMPLATE_RETRIES):
        area = random.randint(lo, hi)
        cells = _spanning_polyomino(area, h, w)
        if cells is not None:
            return cells
    return None


def _gen_lattice(n: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    ext_r = random.randint(*LATTICE_EXTENT_RANGE)
    ext_c = random.randint(*LATTICE_EXTENT_RANGE)
    cells = _spanning_polyomino(n, ext_r, ext_c)
    if cells is None:
        return None
    interior = [(r, c) for r, c in cells if 0 < r < ext_r - 1 and 0 < c < ext_c - 1]
    if not interior:
        return None
    return cells, random.choice(interior)


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return min_dr, max_dr, min_dc, max_dc


def _expected_output(
    bg: int,
    T: int,
    O: int,
    templ_cells: list[tuple[int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    deltas: list[tuple[int, int]],
) -> np.ndarray:
    total = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    center = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for dr, dc in deltas:
        is_origin = dr == 0 and dc == 0
        for tr, tc in templ_cells:
            R = r0 + dr * h + tr
            C = c0 + dc * w + tc
            total[R, C] += 1
            if is_origin:
                center[R, C] += 1
    out = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    out[total > 0] = T
    out[(total == 1) & (center > 0)] = O
    out[total >= 2] = O
    return out


def _place_singletons(
    deltas: list[tuple[int, int]],
    min_dr: int,
    max_dr: int,
    min_dc: int,
    max_dc: int,
    templ_cells: list[tuple[int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    bg: int,
    T: int,
    O: int,
    occupied: set[tuple[int, int]],
) -> np.ndarray | None:
    orr_lo, orr_hi = -min_dr * ROW_GUNIT, GRID_SIZE - 1 - max_dr * ROW_GUNIT
    orc_lo, orc_hi = -min_dc * COL_GUNIT, GRID_SIZE - 1 - max_dc * COL_GUNIT
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = r0, r0 + h - 1
    tc0, tc1 = c0, c0 + w - 1
    expected = _expected_output(bg, T, O, templ_cells, r0, c0, h, w, deltas)

    for _ in range(SINGLETON_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)

        sr0 = orr + min_dr * ROW_GUNIT
        sr1 = orr + max_dr * ROW_GUNIT
        sc0 = orc + min_dc * COL_GUNIT
        sc1 = orc + max_dc * COL_GUNIT
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        if any((orr + dr * ROW_GUNIT, orc + dc * COL_GUNIT) in occupied for dr, dc in deltas):
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        for tr, tc in templ_cells:
            grid[r0 + tr, c0 + tc] = T
        for dr, dc in deltas:
            rr = orr + dr * ROW_GUNIT
            cc = orc + dc * COL_GUNIT
            grid[rr, cc] = T if (dr == 0 and dc == 0) else O

        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    assert ROW_GUNIT != COL_GUNIT
    bg, T, O = random.sample(range(10), 3)

    h = random.choice(TEMPLATE_H_CHOICES)
    w = TEMPLATE_W
    templ_cells = _gen_template(h, w)
    if templ_cells is None:
        return None

    n_sing = random.randint(*SINGLETON_COUNT_RANGE)
    lat = _gen_lattice(n_sing)
    if lat is None:
        return None
    lattice_cells, origin = lat
    deltas = [(r - origin[0], c - origin[1]) for r, c in lattice_cells]
    min_dr, max_dr, min_dc, max_dc = _delta_bounds(deltas)

    r0_lo, r0_hi = -min_dr * h, GRID_SIZE - max_dr * h - h
    c0_lo, c0_hi = -min_dc * w, GRID_SIZE - max_dc * w - w
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    r0 = random.randint(r0_lo, r0_hi)
    c0 = random.randint(c0_lo, c0_hi)

    occupied: set[tuple[int, int]] = set()
    for dr, dc in deltas:
        for tr, tc in templ_cells:
            pt = (r0 + dr * h + tr, c0 + dc * w + tc)
            if pt in occupied:
                return None
            occupied.add(pt)

    return _place_singletons(
        deltas, min_dr, max_dr, min_dc, max_dc,
        templ_cells, r0, c0, h, w, bg, T, O, occupied,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
