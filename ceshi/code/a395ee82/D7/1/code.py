# Variant of a395ee82
# Dimension: D7 — 晶格拓扑结构 (meta)
# Change: 晶格为4-连通多联骨牌，5-6个单元，范围3-4x3-4，原点严格位于内部(非边界)... → 8-连通晶格

from __future__ import annotations

import itertools
import math
import random
from collections import Counter, deque

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# Design analysis
# variant_spec:
#   concretization:
#     - 模板对象不变: 输入里仍有一个较大的单色 T 色模板, 其余是 5-6 个间隔固定的单例像素.
#     - 变化只发生在“粗晶格”的拓扑: 单例像素在原网格里仍彼此相隔 gunit=2, 但把它们除以步长后,
#       原题要求得到 4-连通格点集; 现在改成 8-连通, 因而允许只靠对角接触的格点.
#     - 具体可视例子: 原题常见格点可以是十字 {(0,1),(1,0),(1,1),(1,2),(2,1)}, 输出副本呈上下左右.
#       变体可以是 X 形 {(0,0),(0,2),(1,1),(2,0),(2,2)}, 以中心为原点后得到
#       {(-1,-1),(-1,1),(0,0),(1,-1),(1,1)}, 输出里会出现明确的对角 stamp 阵列.
#   behavior_trace:
#     - 原逻辑在读取 singletons 后, 只是做 gcd 归一化并按 (dr*h, dc*w) 平移模板; 这一步本来就兼容对角偏移.
#     - 因此变体真正需要改的是“什么样的 singleton 布局算合法输入”: solver 要显式接受 8-连通粗晶格,
#       generator 要实际生成包含对角连接的样例, 否则会退化成原任务分布.
#   complexity_calibration:
#     - 若只把“允许 8-连通”写进说明而生成器仍大多产出 4-连通形状, 人眼几乎看不出差异.
#     - 为避免退化, 生成器强制 lattice 在粗网格上 8-连通但非 4-连通, 至少存在一处纯对角连接.
#     - 其余维度保持稳定: 颜色仍只用 bg/T/O 三色, 模板仍为 4-连通单色精灵, stamp 规则不改.
#   implicit_assumptions_review:
#     - find_components(..., connectivity=4) 仍然成立: 单例像素在原网格里间距固定为 2, 即使粗晶格对角相邻,
#       在像素层面也不会连成组件, 所以每个单例依旧是面积 1 的 4-连通组件.
#     - gcd 推断步长仍然成立: 对角布局只会带来同时非零的 (dr, dc), 不会破坏公共步长.
#     - 原 solver 没有检查 lattice 拓扑; 现在补上“粗晶格 8-连通 + 原点位于 bbox 内部”的断言,
#       使 main() 与变体分布显式对齐.
#   placement_legality:
#     - 模板平移单位仍是 (h, w), 所以对角副本会按一个模板高和一个模板宽同时偏移.
#     - 生成器继续保证所有 stamp 完全在 22x22 内, 且不同副本不重叠; 这样输出整洁, O 色主要只标识原点副本.
#     - singleton 簇与模板 bbox 仍要求至少隔 1 个背景像素, 并且 singletons 不能落在任何 stamp 像素上.
#   observability:
#     - 通过“8-连通但非 4-连通”的 lattice, 训练样例会稳定出现斜向排布的模板副本, 肉眼可见.
#     - 不使用颜色 id 做排序或分支; 颜色语义仍由结构决定: T 是普通模板色, O 是原点副本/重叠色.
#   cleanliness:
#     - 保留原题的单模板 + 单例晶格二分布局, 避免多层叠加.
#     - 不引入额外标记、额外颜色或额外对象, 让唯一新现象就是“对角连接导致的对角 stamp”.
#   new_main_assumptions:
#     - 非模板组件必须全是单例.
#     - singleton 数量仍在 5-6.
#     - 归一化后的粗晶格必须 8-连通, 且原点在其 bbox 严格内部.
#     - generator 通过候选池筛选保证这些条件恒成立.


# --------------- generator (D7 variant) ---------------

GRID_SIZE = 22
TEMPLATE_H_CHOICES = [3, 4]
TEMPLATE_W = 3
TEMPLATE_AREA_RANGE = (6, 10)
TEMPLATE_FILL_RANGE = (0.65, 0.85)
SINGLETON_COUNT_RANGE = (5, 6)
LATTICE_EXTENT_RANGE = (3, 4)
GUNIT = 2

SINGLETON_RETRIES = 80
GENERATE_RETRIES = 300

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]


def _dirs(connectivity: int) -> list[tuple[int, int]]:
    return DIR4 if int(connectivity) == 4 else DIR8


def _is_connected(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...], connectivity: int) -> bool:
    pts = {tuple(map(int, cell)) for cell in cells}
    if not pts:
        return False
    q = deque([next(iter(pts))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for dr, dc in _dirs(connectivity):
            nxt = (r + dr, c + dc)
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(pts)


def _touches_all_edges(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...], rows: int, cols: int) -> bool:
    rs = [int(r) for r, _c in cells]
    cs = [int(c) for _r, c in cells]
    return min(rs) == 0 and max(rs) == rows - 1 and min(cs) == 0 and max(cs) == cols - 1


def _interior_cells(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...], rows: int, cols: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in cells if 0 < int(r) < rows - 1 and 0 < int(c) < cols - 1]


def _diagonal_contacts(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> int:
    pts = {tuple(map(int, cell)) for cell in cells}
    total = 0
    for r, c in pts:
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nbr = (r + dr, c + dc)
            if nbr in pts and (r, c) < nbr:
                total += 1
    return total


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
                    diag = _diagonal_contacts(cells)
                    if diag == 0:
                        continue
                    out[n].append(
                        {
                            "rows": rows,
                            "cols": cols,
                            "cells": [(int(r), int(c)) for r, c in cells],
                            "interiors": interiors,
                            "diag_contacts": diag,
                        }
                    )
    return out


TEMPLATE_CANDIDATES = _enumerate_template_candidates()
LATTICE_CANDIDATES = _enumerate_lattice_candidates()
assert all(TEMPLATE_CANDIDATES[h] for h in TEMPLATE_H_CHOICES)
assert all(LATTICE_CANDIDATES[n] for n in range(SINGLETON_COUNT_RANGE[0], SINGLETON_COUNT_RANGE[1] + 1))


def _render_expected_output(
    bg: int,
    H: int,
    W: int,
    templ_rel: list[tuple[int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    deltas: list[tuple[int, int]],
    templ_col: int,
    overlap_col: int,
) -> np.ndarray:
    total = np.zeros((H, W), dtype=int)
    center = np.zeros((H, W), dtype=int)

    for dr, dc in deltas:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp translated copies of a monochrome template from an 8-neighbor singleton lattice.

    Input structure:
    - The background is uniform. One large monochrome component in color T is the template sprite.
    - Elsewhere are 5-6 isolated singleton pixels on a fixed-step coarse lattice; exactly one singleton is also color T and marks the lattice origin, while the rest are color O.
    - After normalizing by the common singleton spacing, those lattice points form an 8-connected shape whose origin sits strictly inside its bounding box, so diagonal branches are allowed.

    Rule:
    - Measure every singleton offset from the unique T-colored origin and divide by the common lattice step.
    - Use each normalized offset as a template translation in whole template-box units: row offset scales by template height and column offset scales by template width.
    - Paint all translated copies with T, then recolor the origin copy to O.
    - Any cell covered by multiple copies is also recolored to O.

    Key insight:
    - The singleton colored T is both the anchor of the coarse lattice and the indicator of which translated copy must switch from T to O; the only new topology is that the lattice may connect diagonally.
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

    deltas = [(int(r - orr), int(c - orc), int(col)) for r, c, col in singletons]
    gcd_vals = [abs(int(x)) for dr, dc, _col in deltas for x in (dr, dc) if int(x) != 0]
    assert gcd_vals
    gunit = max(1, int(math.gcd(*gcd_vals)))
    deltas = [(int(dr // gunit), int(dc // gunit), int(col)) for dr, dc, col in deltas]

    coarse_cells = [(int(dr), int(dc)) for dr, dc, _col in deltas]
    assert (0, 0) in coarse_cells
    assert _is_connected(coarse_cells, connectivity=8)
    rs = [dr for dr, _dc in coarse_cells]
    cs = [dc for _dr, dc in coarse_cells]
    assert min(rs) < 0 < max(rs)
    assert min(cs) < 0 < max(cs)

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


def _gen_template(h: int, w: int) -> list[tuple[int, int]] | None:
    if w != TEMPLATE_W:
        return None
    choices = TEMPLATE_CANDIDATES.get(int(h), [])
    if not choices:
        return None
    return list(random.choice(choices))


def _gen_lattice(n: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    choices = LATTICE_CANDIDATES.get(int(n), [])
    if not choices:
        return None
    spec = random.choice(choices)
    cells = list(spec["cells"])
    origin = random.choice(spec["interiors"])
    return cells, origin


def _delta_bounds(deltas: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    min_dr = min(dr for dr, _dc in deltas)
    max_dr = max(dr for dr, _dc in deltas)
    min_dc = min(dc for _dr, dc in deltas)
    max_dc = max(dc for _dr, dc in deltas)
    return min_dr, max_dr, min_dc, max_dc


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
    templ_col: int,
    overlap_col: int,
    occupied: set[tuple[int, int]],
    expected: np.ndarray,
) -> np.ndarray | None:
    orr_lo, orr_hi = -min_dr * GUNIT, GRID_SIZE - 1 - max_dr * GUNIT
    orc_lo, orc_hi = -min_dc * GUNIT, GRID_SIZE - 1 - max_dc * GUNIT
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    tr0, tr1 = r0, r0 + h - 1
    tc0, tc1 = c0, c0 + w - 1

    for _ in range(SINGLETON_RETRIES):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)

        sr0 = orr + min_dr * GUNIT
        sr1 = orr + max_dr * GUNIT
        sc0 = orc + min_dc * GUNIT
        sc1 = orc + max_dc * GUNIT
        if not (sr1 + 1 < tr0 or tr1 + 1 < sr0 or sc1 + 1 < tc0 or tc1 + 1 < sc0):
            continue

        if any((orr + dr * GUNIT, orc + dc * GUNIT) in occupied for dr, dc in deltas):
            continue

        grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        for tr, tc in templ_cells:
            grid[r0 + tr, c0 + tc] = templ_col
        for dr, dc in deltas:
            grid[orr + dr * GUNIT, orc + dc * GUNIT] = templ_col if (dr == 0 and dc == 0) else overlap_col

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
    lattice_cells, origin = lattice
    deltas = [(int(r - origin[0]), int(c - origin[1])) for r, c in lattice_cells]
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
        deltas=deltas,
        templ_col=templ_col,
        overlap_col=overlap_col,
    )

    return _place_singletons(
        deltas=deltas,
        min_dr=min_dr,
        max_dr=max_dr,
        min_dc=min_dc,
        max_dc=max_dc,
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
