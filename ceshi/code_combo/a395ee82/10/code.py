# Combo variant of a395ee82 (index 10)
# combo_spec: D2/2+D7/1
# Name: 不均匀八连通晶格

from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis for the combo variant
# -------------------------------------
# fused_rule:
#   - 输入仍然分成两个对象群: 一个较大的单色模板 T, 以及 5-6 个孤立单例像素.
#   - 单例像素的颜色语义不变: 恰有 1 个是 T 色原点, 其余都为 O 色.
#   - 变化 1 来自 D2/2: 单例像素不再遵循单一公因子步长. 排序后的相邻可见行/列间距只允许
#     两档, gap=2 表示 1 个 stamp 步长, gap=3 表示 2 个 stamp 步长.
#   - 变化 2 来自 D7/1: 单例像素在“排序后的离散行列等级”上形成 8 连通但非 4 连通的粗晶格,
#     因而会出现纯对角邻接.
#
# behavior_trace:
#   - 具体小例子: 可见源行坐标为 [4, 6, 9], 列坐标为 [3, 6, 8].
#     行 gap=[2,3] 解码成步长等级 [0,1,3], 列 gap=[3,2] 解码成 [0,2,3].
#   - 若原点位于可见等级 (1,1), 一个 rank 邻居从 (1,1) 斜向到 (0,2) 只是在拓扑上
#     做了 (-1,+1) 的 8 邻接, 但真正 stamp 平移却是 (-1,+1) 行列等级经过局部 gap 解码后得到
#     (-1,+1) 或 (-1,+2) 之类的不均匀位移.
#   - 原题会对 raw delta 取全局 gcd, 再把归一化后的 (dr,dc) 同时用于拓扑理解与 stamp 平移.
#     这里这两个角色被拆开:
#       1. 拓扑检查使用排序等级 rank 坐标, 从而正确识别 8 连通和对角连接.
#       2. 输出平移使用 gap 解码后的 step 坐标, 从而体现不均匀间距.
#
# complexity_calibration:
#   - gap 类别只保留 {2,3} 两档, 人类可以从重复出现的短/长间距直接观察到编码规律,
#     不需要猜连续参数.
#   - 生成器强制 lattice 为 8 连通但非 4 连通, 并要求至少一个轴同时出现两种 gap,
#     这样输入不会退化成“只是普通规则加一点随机摆放”.
#   - 输出仍然只用 bg/T/O 三色, 原点副本和重叠仍统一染成 O, 避免再引入额外颜色映射负担.
#
# implicit_assumption_audit:
#   - 原 solver 的 gcd 归一化依赖“所有 singleton 偏移都来自同一个像素步长”, 在本变体中失效.
#     因此 main() 改为先按可见行/列排序, 再逐段解码 gap.
#   - 原 solver 默认粗晶格等价于归一化后的整数偏移; 这里不再成立.
#     main() 改为维护两套坐标:
#       rank 坐标: 只反映离散晶格拓扑, 用来断言 8 连通 / 非 4 连通 / 原点在内部.
#       step 坐标: 用来决定模板副本的真实平移量.
#   - 新 main() 还假设: 所有非模板组件都是面积 1 的 singleton; 单例数在 5-6 之间;
#     排序后的相邻行列 gap 只能取 2 或 3; 至少整体上两种 gap 都出现.
#   - generator 通过枚举 8 连通候选、限制 gap 模式、以及回验 main(grid)==expected
#     来保证这些假设恒成立.
#
# placement_legality:
#   - 模板仍为 3-4 x 3 的单色 4 连通形状, 所有副本必须完整落在 22x22 内.
#   - 生成器按 step 坐标先放输出副本, 显式拒绝任何像素级重叠.
#   - singleton 簇按 raw 像素偏移放置, 与模板 bbox 至少隔 1 个背景格, 且 singleton 不能落在任何 stamp 像素上.
#
# observability_and_cleanliness:
#   - 人眼可见的线索有两层: 输入里的 singleton 出现短/长不均匀间距, 输出里的模板副本出现斜向阵列.
#   - 拓扑变化通过“有纯对角连接的 lattice”保证显著, 间距变化通过“同一轴上两种 gap 都重复出现”
#     保证可归纳, 不依赖颜色编号等不可视信息.
#   - 所有对象仍保持模板区与 singleton 区分离, 画面整洁, 不会因为多层覆盖而难以读图.
#
# new_main_assumptions_review:
#   - `_decode_axis(...)` 假设排序后的唯一坐标就是可见晶格等级, generator 通过 spanning lattice
#     保证每个中间等级都至少被某个 singleton 占据.
#   - `_is_connected(rank_cells, 8)` 假设 8 连通是在 rank 坐标上定义, 这正是本融合规格下的粗晶格语义.
#   - stamp 循环假设 step 偏移与模板 bbox 尺度相乘后仍完全在界内, generator 先按这些 step 偏移做合法性筛选.


GAP_TO_STEP = {2: 1, 3: 2}
SOURCE_GAP_CHOICES = tuple(sorted(GAP_TO_STEP))
REQUIRED_GAP_SET = set(SOURCE_GAP_CHOICES)

GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
MAX_AXIS_STEP_SPAN = 4

SINGLETON_RETRIES = 80
GENERATE_RETRIES = 400

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]


def _dirs(connectivity: int) -> list[tuple[int, int]]:
    return DIR4 if int(connectivity) == 4 else DIR8


def _is_connected(
    cells: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    connectivity: int,
) -> bool:
    pts = {tuple(map(int, cell)) for cell in cells}
    if not pts:
        return False
    start = next(iter(pts))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in _dirs(connectivity):
            nxt = (int(r + dr), int(c + dc))
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(pts)


def _touches_all_edges(
    cells: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    rows: int,
    cols: int,
) -> bool:
    rs = [int(r) for r, _c in cells]
    cs = [int(c) for _r, c in cells]
    return min(rs) == 0 and max(rs) == rows - 1 and min(cs) == 0 and max(cs) == cols - 1


def _interior_cells(
    cells: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    rows: int,
    cols: int,
) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in cells if 0 < int(r) < rows - 1 and 0 < int(c) < cols - 1]


def _diagonal_contacts(
    cells: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> int:
    pts = {tuple(map(int, cell)) for cell in cells}
    total = 0
    for r, c in pts:
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nbr = (int(r + dr), int(c + dc))
            if nbr in pts and (r, c) < nbr:
                total += 1
    return total


def _decode_axis(values: list[int]) -> tuple[dict[int, int], dict[int, int], list[int]]:
    """Return rank map, decoded step map, and the visible adjacent gaps for one axis."""
    vals = sorted({int(v) for v in values})
    assert len(vals) >= 2

    rank_map = {int(v): idx for idx, v in enumerate(vals)}
    step_map = {int(vals[0]): 0}
    gaps: list[int] = []
    acc = 0
    for prev, cur in zip(vals, vals[1:]):
        gap = int(cur - prev)
        assert gap in GAP_TO_STEP
        gaps.append(int(gap))
        acc += int(GAP_TO_STEP[gap])
        step_map[int(cur)] = int(acc)
    return rank_map, step_map, gaps


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    min_dr = min(int(dr) for dr, _dc in deltas)
    max_dr = max(int(dr) for dr, _dc in deltas)
    min_dc = min(int(dc) for _dr, dc in deltas)
    max_dc = max(int(dc) for _dr, dc in deltas)
    return int(min_dr), int(max_dr), int(min_dc), int(max_dc)


def _render_expected_output(
    bg: int,
    H: int,
    W: int,
    templ_rel: list[tuple[int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    step_deltas: list[tuple[int, int]],
    templ_col: int,
    overlap_col: int,
) -> np.ndarray:
    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)

    for dr, dc in step_deltas:
        off_r = int(dr * h)
        off_c = int(dc * w)
        is_origin = dr == 0 and dc == 0
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


def _enumerate_template_candidates() -> dict[int, list[list[tuple[int, int]]]]:
    out: dict[int, list[list[tuple[int, int]]]] = {h: [] for h in TEMPLATE_H_CHOICES}
    for h in TEMPLATE_H_CHOICES:
        total = h * TEMPLATE_W
        lo = max(TEMPLATE_AREA_RANGE[0], int(np.ceil(total * TEMPLATE_FILL_RANGE[0])))
        hi = min(TEMPLATE_AREA_RANGE[1], int(np.floor(total * TEMPLATE_FILL_RANGE[1])))
        coords = [(r, c) for r in range(h) for c in range(TEMPLATE_W)]
        for area in range(lo, hi + 1):
            for cells in itertools.combinations(coords, area):
                if not _touches_all_edges(cells, h, TEMPLATE_W):
                    continue
                if not _is_connected(cells, connectivity=4):
                    continue
                out[h].append([(int(r), int(c)) for r, c in cells])
    return out


def _enumerate_lattice_candidates() -> dict[int, list[dict[str, object]]]:
    out: dict[int, list[dict[str, object]]] = {
        n: [] for n in range(SINGLETON_COUNT_RANGE[0], SINGLETON_COUNT_RANGE[1] + 1)
    }
    for rows in range(LATTICE_EXTENT_RANGE[0], LATTICE_EXTENT_RANGE[1] + 1):
        for cols in range(LATTICE_EXTENT_RANGE[0], LATTICE_EXTENT_RANGE[1] + 1):
            coords = [(r, c) for r in range(rows) for c in range(cols)]
            for n in range(SINGLETON_COUNT_RANGE[0], SINGLETON_COUNT_RANGE[1] + 1):
                for cells in itertools.combinations(coords, n):
                    if not _touches_all_edges(cells, rows, cols):
                        continue
                    if not _is_connected(cells, connectivity=8):
                        continue
                    if _is_connected(cells, connectivity=4):
                        continue
                    interiors = _interior_cells(cells, rows, cols)
                    if not interiors:
                        continue
                    diag_contacts = _diagonal_contacts(cells)
                    if diag_contacts == 0:
                        continue
                    out[n].append(
                        {
                            "rows": int(rows),
                            "cols": int(cols),
                            "cells": [(int(r), int(c)) for r, c in cells],
                            "interiors": interiors,
                            "diag_contacts": int(diag_contacts),
                        }
                    )
    return out


def _enumerate_gap_layouts() -> dict[tuple[int, int], list[dict[str, object]]]:
    out: dict[tuple[int, int], list[dict[str, object]]] = {}
    for rows in range(LATTICE_EXTENT_RANGE[0], LATTICE_EXTENT_RANGE[1] + 1):
        for cols in range(LATTICE_EXTENT_RANGE[0], LATTICE_EXTENT_RANGE[1] + 1):
            key = (int(rows), int(cols))
            out[key] = []
            row_gap_choices = list(itertools.product(SOURCE_GAP_CHOICES, repeat=rows - 1))
            col_gap_choices = list(itertools.product(SOURCE_GAP_CHOICES, repeat=cols - 1))
            for row_gaps in row_gap_choices:
                row_steps = sum(int(GAP_TO_STEP[g]) for g in row_gaps)
                if row_steps > MAX_AXIS_STEP_SPAN:
                    continue
                for col_gaps in col_gap_choices:
                    col_steps = sum(int(GAP_TO_STEP[g]) for g in col_gaps)
                    if col_steps > MAX_AXIS_STEP_SPAN:
                        continue
                    all_gaps = set(int(g) for g in row_gaps + col_gaps)
                    if all_gaps != REQUIRED_GAP_SET:
                        continue
                    if len(set(row_gaps)) == 1 and len(set(col_gaps)) == 1:
                        continue

                    row_src = [0]
                    row_dec = [0]
                    for gap in row_gaps:
                        row_src.append(int(row_src[-1] + gap))
                        row_dec.append(int(row_dec[-1] + GAP_TO_STEP[int(gap)]))

                    col_src = [0]
                    col_dec = [0]
                    for gap in col_gaps:
                        col_src.append(int(col_src[-1] + gap))
                        col_dec.append(int(col_dec[-1] + GAP_TO_STEP[int(gap)]))

                    out[key].append(
                        {
                            "row_gaps": [int(g) for g in row_gaps],
                            "col_gaps": [int(g) for g in col_gaps],
                            "row_src": row_src,
                            "col_src": col_src,
                            "row_steps": row_dec,
                            "col_steps": col_dec,
                        }
                    )
    return out


TEMPLATE_CANDIDATES = _enumerate_template_candidates()
LATTICE_CANDIDATES = _enumerate_lattice_candidates()
GAP_LAYOUTS = _enumerate_gap_layouts()
assert all(TEMPLATE_CANDIDATES[h] for h in TEMPLATE_H_CHOICES)
assert all(LATTICE_CANDIDATES[n] for n in range(SINGLETON_COUNT_RANGE[0], SINGLETON_COUNT_RANGE[1] + 1))
assert all(GAP_LAYOUTS.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp translated copies of one template from an irregular 8-neighbor singleton scaffold.

    Input structure:
    - The background is uniform, and one large monochrome T-colored component is the template sprite.
    - Away from it sit 5-6 isolated singleton pixels: exactly one singleton is also T and marks the scaffold origin, while the rest are O.
    - The singleton scaffold is not evenly spaced: sorted neighboring rows/cols differ by 2 or 3 pixels, and the singleton ranks form an 8-connected coarse shape with diagonal-only contacts allowed.

    Rule:
    - Read singleton rows and cols as two warped axes. Adjacent visible gaps decode locally as 2->1 stamp step and 3->2 stamp steps.
    - Use rank coordinates only to recover the coarse lattice topology and locate the interior origin; use decoded step coordinates to compute template translations in units of template height and width.
    - Stamp the template at every decoded offset, recolor the origin copy to O, and also recolor any overlapped cells to O.

    Key insight:
    - The same singleton scaffold carries two kinds of information at once: its rank ordering says which points are diagonal neighbors in the hidden 8-connected lattice, while its short-vs-long visible gaps say how far apart the stamped copies must be.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps

    template = max(comps, key=lambda d: int(d["area"]))
    assert int(template["area"]) > 1
    templ_coords = template["coords"]
    templ_col = int(g[int(templ_coords[0, 0]), int(templ_coords[0, 1])])
    r0, c0, r1, c1 = map(int, template["bbox"])
    h = int(bbox_h((r0, c0, r1, c1)))
    w = int(bbox_w((r0, c0, r1, c1)))
    templ_rel = [(int(r) - int(r0), int(c) - int(c0)) for r, c in templ_coords]

    singletons: list[tuple[int, int, int]] = []
    for comp in comps:
        if comp is template:
            continue
        assert int(comp["area"]) == 1
        rr, cc = map(int, comp["coords"][0])
        singletons.append((int(rr), int(cc), int(g[int(rr), int(cc)])))

    assert SINGLETON_COUNT_RANGE[0] <= len(singletons) <= SINGLETON_COUNT_RANGE[1]

    origin = [(r, c) for r, c, col in singletons if int(col) == int(templ_col)]
    assert len(origin) == 1
    orr, orc = origin[0]

    other_cols = [int(col) for _r, _c, col in singletons if int(col) != int(templ_col)]
    assert other_cols
    overlap_col = int(Counter(other_cols).most_common(1)[0][0])

    row_rank, row_step, row_gaps = _decode_axis([r for r, _c, _col in singletons])
    col_rank, col_step, col_gaps = _decode_axis([c for _r, c, _col in singletons])
    assert set(row_gaps + col_gaps) == REQUIRED_GAP_SET

    origin_rank_r = int(row_rank[int(orr)])
    origin_rank_c = int(col_rank[int(orc)])
    origin_step_r = int(row_step[int(orr)])
    origin_step_c = int(col_step[int(orc)])

    rank_cells: list[tuple[int, int]] = []
    step_deltas: list[tuple[int, int, int]] = []
    for r, c, col in singletons:
        rank_cells.append(
            (
                int(row_rank[int(r)] - origin_rank_r),
                int(col_rank[int(c)] - origin_rank_c),
            )
        )
        step_deltas.append(
            (
                int(row_step[int(r)] - origin_step_r),
                int(col_step[int(c)] - origin_step_c),
                int(col),
            )
        )

    assert (0, 0) in rank_cells
    assert _is_connected(rank_cells, connectivity=8)
    assert not _is_connected(rank_cells, connectivity=4)
    assert _diagonal_contacts(rank_cells) > 0

    rs = [int(dr) for dr, _dc in rank_cells]
    cs = [int(dc) for _dr, dc in rank_cells]
    assert min(rs) < 0 < max(rs)
    assert min(cs) < 0 < max(cs)

    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)

    for dr, dc, col in step_deltas:
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


def _gen_template(h: int, w: int) -> list[tuple[int, int]] | None:
    if int(w) != TEMPLATE_W:
        return None
    choices = TEMPLATE_CANDIDATES.get(int(h), [])
    if not choices:
        return None
    return list(random.choice(choices))


def _gen_lattice(n: int) -> tuple[dict[str, object], tuple[int, int]] | None:
    choices = LATTICE_CANDIDATES.get(int(n), [])
    if not choices:
        return None
    spec = random.choice(choices)
    origin = random.choice(spec["interiors"])
    return spec, (int(origin[0]), int(origin[1]))


def _choose_gap_layout(rows: int, cols: int) -> dict[str, object] | None:
    choices = GAP_LAYOUTS.get((int(rows), int(cols)), [])
    if not choices:
        return None
    return random.choice(choices)


def _place_singletons(
    raw_deltas: list[tuple[int, int]],
    raw_bounds: tuple[int, int, int, int],
    templ_cells: list[tuple[int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    bg: int,
    templ_col: int,
    overlap_col: int,
    occupied: set[tuple[int, int]],
    expected: np.ndarray,
) -> np.ndarray | None:
    min_dr, max_dr, min_dc, max_dc = raw_bounds
    orr_lo, orr_hi = -min_dr, GRID_SIZE - 1 - max_dr
    orc_lo, orc_hi = -min_dc, GRID_SIZE - 1 - max_dc
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = int(r0), int(r0 + h - 1)
    tc0, tc1 = int(c0), int(c0 + w - 1)

    for _ in range(SINGLETON_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)

        sr0 = int(orr + min_dr)
        sr1 = int(orr + max_dr)
        sc0 = int(orc + min_dc)
        sc1 = int(orc + max_dc)
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        src_points = [(int(orr + dr), int(orc + dc)) for dr, dc in raw_deltas]
        if any(pt in occupied for pt in src_points):
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
        for tr, tc in templ_cells:
            grid[int(r0 + tr), int(c0 + tc)] = int(templ_col)
        for rr, cc in src_points:
            grid[int(rr), int(cc)] = int(overlap_col)
        grid[int(orr), int(orc)] = int(templ_col)

        try:
            produced = main(grid)
        except Exception:
            continue
        if np.array_equal(produced, expected):
            return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    bg, templ_col, overlap_col = random.sample(range(10), 3)

    h = random.choice(TEMPLATE_H_CHOICES)
    w = TEMPLATE_W
    templ_cells = _gen_template(h, w)
    if templ_cells is None:
        return None

    n_sing = random.randint(*SINGLETON_COUNT_RANGE)
    lattice = _gen_lattice(n_sing)
    if lattice is None:
        return None
    lattice_spec, origin = lattice
    lattice_cells = list(lattice_spec["cells"])
    rows = int(lattice_spec["rows"])
    cols = int(lattice_spec["cols"])

    gap_layout = _choose_gap_layout(rows, cols)
    if gap_layout is None:
        return None
    row_src = list(gap_layout["row_src"])
    col_src = list(gap_layout["col_src"])
    row_steps = list(gap_layout["row_steps"])
    col_steps = list(gap_layout["col_steps"])

    raw_deltas: list[tuple[int, int]] = []
    step_deltas: list[tuple[int, int]] = []
    for lr, lc in lattice_cells:
        raw_deltas.append(
            (
                int(row_src[int(lr)] - row_src[int(origin[0])]),
                int(col_src[int(lc)] - col_src[int(origin[1])]),
            )
        )
        step_deltas.append(
            (
                int(row_steps[int(lr)] - row_steps[int(origin[0])]),
                int(col_steps[int(lc)] - col_steps[int(origin[1])]),
            )
        )

    min_sdr, max_sdr, min_sdc, max_sdc = _delta_bounds(step_deltas)
    r0_lo, r0_hi = -min_sdr * h, GRID_SIZE - max_sdr * h - h
    c0_lo, c0_hi = -min_sdc * w, GRID_SIZE - max_sdc * w - w
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None

    r0 = random.randint(r0_lo, r0_hi)
    c0 = random.randint(c0_lo, c0_hi)

    occupied: set[tuple[int, int]] = set()
    for dr, dc in step_deltas:
        for tr, tc in templ_cells:
            pt = (int(r0 + dr * h + tr), int(c0 + dc * w + tc))
            if pt in occupied:
                return None
            occupied.add(pt)

    expected = _render_expected_output(
        bg=bg,
        H=GRID_SIZE,
        W=GRID_SIZE,
        templ_rel=templ_cells,
        r0=r0,
        c0=c0,
        h=h,
        w=w,
        step_deltas=step_deltas,
        templ_col=templ_col,
        overlap_col=overlap_col,
    )

    return _place_singletons(
        raw_deltas=raw_deltas,
        raw_bounds=_delta_bounds(raw_deltas),
        templ_cells=templ_cells,
        r0=r0,
        c0=c0,
        h=h,
        w=w,
        bg=bg,
        templ_col=templ_col,
        overlap_col=overlap_col,
        occupied=occupied,
        expected=expected,
    )


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
